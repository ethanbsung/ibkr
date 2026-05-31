#!/usr/bin/env python3
"""
Paper-trading engine — capital-agnostic, multi-strategy, reallocation-safe.

Design (per requirements):
  • Each strategy is tracked in RETURN / NAV space (growth of $1), NOT dollars.
    Crypto returns are scale-invariant, so a strategy's track record is identical
    regardless of how much capital it's allocated. This makes the per-strategy
    data immune to reallocation — changing capital never rewrites performance.
  • Capital lives in a separate config (paper/portfolio.json). The portfolio
    layer (see run_paper.py) combines strategy NAVs by allocation weight into a
    dollar portfolio. Reallocate or add a strategy by editing the config only.

A Strategy answers one question via compute(): "today's prices and target
weights?" The engine holds weights, marks P&L from price changes, applies the
no-trade band and turnover cost, and compounds a NAV. Idempotent per bar.

Ledgers per strategy in paper/ledgers/<name>/:
  state.json     {date, weights:{sym:w}, prices:{sym:px}, nav}   (resume point)
  daily.csv      date, ret, nav, gross_exposure, turnover, cost, n_positions
  positions.csv  date, symbol, price, target_w, held_w
  trades.csv     date, symbol, d_weight, price, cost
"""

import csv
import json
import math
import os
from dataclasses import dataclass

import pandas as pd


# ── Strategy contract ────────────────────────────────────────────────────────

@dataclass
class Strategy:
    """
    Base class — capital-agnostic. Subclasses implement compute() and set costs.
    `band` is the no-trade buffer as a fraction of capital (0.004 = 0.4%).
    fee/slip are per side of weight traded (turnover). For a two-leg strategy
    (e.g. carry) set them to the COMBINED both-leg cost.
    """
    name: str
    fee_per_side: float = 0.0035
    slip_per_side: float = 0.0005
    band: float = 0.004

    def compute(self, asof=None):
        """Return (bar_date, prices:{sym:px}, target_weights:{sym:w}).
        Weights are fractions of the strategy's (notional) capital; |sum| ≤ 1
        keeps it unlevered. Prices may be a synthetic index (e.g. carry)."""
        raise NotImplementedError


# ── Per-strategy persistence (return/NAV space) ──────────────────────────────

class Book:
    DAILY_COLS = ['date', 'ret', 'nav', 'gross_exposure', 'turnover', 'cost', 'n_positions']
    POS_COLS   = ['date', 'symbol', 'price', 'target_w', 'held_w']
    TRADE_COLS = ['date', 'symbol', 'd_weight', 'price', 'cost']

    def __init__(self, name, root):
        self.dir = os.path.join(root, name)
        os.makedirs(self.dir, exist_ok=True)
        self.state_path = os.path.join(self.dir, 'state.json')
        self.daily_path = os.path.join(self.dir, 'daily.csv')
        self.pos_path   = os.path.join(self.dir, 'positions.csv')
        self.trade_path = os.path.join(self.dir, 'trades.csv')

    def load_state(self):
        if not os.path.exists(self.state_path):
            return {'date': None, 'weights': {}, 'prices': {}, 'nav': 1.0}
        with open(self.state_path) as fh:
            return json.load(fh)

    def save_state(self, date, weights, prices, nav):
        with open(self.state_path, 'w') as fh:
            json.dump({'date': str(date),
                       'weights': {k: v for k, v in weights.items() if abs(v) > 1e-9},
                       'prices': prices, 'nav': nav}, fh, indent=2)

    def _append(self, path, cols, rows):
        new = not os.path.exists(path)
        with open(path, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            if new:
                w.writeheader()
            for r in rows:
                w.writerow(r)

    def append_daily(self, row):     self._append(self.daily_path, self.DAILY_COLS, [row])
    def append_positions(self, rows): self._append(self.pos_path, self.POS_COLS, rows)
    def append_trades(self, rows):   self._append(self.trade_path, self.TRADE_COLS, rows)

    def daily_frame(self):
        if not os.path.exists(self.daily_path):
            return pd.DataFrame()
        df = pd.read_csv(self.daily_path, parse_dates=['date']).set_index('date')
        return df[~df.index.duplicated(keep='last')].sort_index()


# ── Execution ────────────────────────────────────────────────────────────────

def _band_target(tgt_w, held_w, band):
    """Buffered target: keep current weight if within the band; else move to edge."""
    if abs(tgt_w - held_w) <= band:
        return held_w
    return tgt_w - math.copysign(band, tgt_w - held_w)


class PaperEngine:
    def __init__(self, ledger_root):
        self.root = ledger_root
        os.makedirs(ledger_root, exist_ok=True)

    def run_strategy(self, strat, asof=None, verbose=True):
        """Execute one strategy for its latest (or `asof`) bar. Idempotent."""
        bar_date, prices, targets = strat.compute(asof)
        book = Book(strat.name, self.root)
        st = book.load_state()
        if st['date'] is not None and str(bar_date) <= st['date']:
            if verbose:
                print(f"  [{strat.name}] already processed through {st['date']} — skip")
            return book

        prev_w, prev_p, nav = st['weights'], st['prices'], st['nav']
        cost_side = strat.fee_per_side + strat.slip_per_side

        # 1) P&L from holding yesterday's weights over this bar (instrument returns).
        pnl = 0.0
        for s, w in prev_w.items():
            if s in prices and s in prev_p and prev_p[s]:
                pnl += w * (prices[s] / prev_p[s] - 1.0)

        # 2) Rebalance to target weights through the no-trade band.
        new_w, pos_rows, trade_rows, turnover = {}, [], [], 0.0
        for s in sorted(set(targets) | set(prev_w)):
            if s not in prices:                      # can't trade what we can't price
                new_w[s] = prev_w.get(s, 0.0)
                continue
            tgt, held = targets.get(s, 0.0), prev_w.get(s, 0.0)
            w = _band_target(tgt, held, strat.band)
            dw = w - held
            if abs(dw) > 1e-9:
                turnover += abs(dw)
                trade_rows.append({'date': bar_date, 'symbol': s, 'd_weight': round(dw, 6),
                                   'price': prices[s], 'cost': round(abs(dw) * cost_side, 6)})
            if abs(w) > 1e-9:
                new_w[s] = w
            pos_rows.append({'date': bar_date, 'symbol': s, 'price': prices[s],
                             'target_w': round(tgt, 6), 'held_w': round(w, 6)})

        cost = turnover * cost_side
        ret = pnl - cost
        nav *= (1 + ret)

        book.append_daily({'date': bar_date, 'ret': round(ret, 8), 'nav': round(nav, 8),
                           'gross_exposure': round(sum(abs(v) for v in new_w.values()), 4),
                           'turnover': round(turnover, 4), 'cost': round(cost, 8),
                           'n_positions': len(new_w)})
        book.append_positions(pos_rows)
        if trade_rows:
            book.append_trades(trade_rows)
        book.save_state(bar_date, new_w, prices, nav)

        if verbose:
            print(f"  [{strat.name}] bar {bar_date}: ret {ret*100:+.3f}%  nav {nav:.4f}  "
                  f"gross {sum(abs(v) for v in new_w.values())*100:.0f}%  "
                  f"{len(trade_rows)} trades")
        return book

    def run(self, strategies, asof=None, verbose=True):
        return {s.name: self.run_strategy(s, asof, verbose) for s in strategies}
