#!/usr/bin/env python3
"""
Strategy plugins for the paper engine (capital-agnostic — they emit weights).

CryptoTrendStrategy — validated daily EWMAC ensemble, long/flat, 10 majors.
  Source: Coinbase public daily candles. Reuses crypto/crypto_trend.py signal.

CryptoCarryStrategy — funding-rate carry, long spot / short perp, quality coins.
  Source: Coinbase public perp product data (funding_rate + index_price + perp
  price — no auth). Presents a per-coin synthetic CARRY INDEX (compounds funding
  + basis) as the engine's "price", so the generic engine marks its P&L. Holds a
  coin only when its smoothed funding is positive (don't short when you'd pay).
  Maintains the index in its own aux state so it's continuous across daily runs.

Add a strategy: subclass engine.Strategy, implement compute(), add to run_paper.
"""

import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Strategy

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'research', 'crypto'))
import crypto_trend as ct   # noqa: E402

_LEDGERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ledgers')


def _get_json(url, timeout=30):
    req = urllib.request.Request(url, headers={'User-Agent': 'paper-trader/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


# ── Trend ────────────────────────────────────────────────────────────────────

COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/{pid}/candles?granularity=86400"
TREND_UNIVERSE = {
    'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'SOL': 'SOL-USD', 'BNB': 'BNB-USD',
    'XRP': 'XRP-USD', 'ADA': 'ADA-USD', 'AVAX': 'AVAX-USD', 'LINK': 'LINK-USD',
    'DOGE': 'DOGE-USD', 'LTC': 'LTC-USD',
}


def _fetch_daily_closes(pid):
    rows = _get_json(COINBASE_CANDLES.format(pid=pid))   # [[t,lo,hi,op,cl,vol], …] newest first
    today = datetime.now(timezone.utc).date()
    data = {}
    for t, lo, hi, op, cl, vol in rows:
        d = datetime.fromtimestamp(t, tz=timezone.utc).date()
        if d < today:                                    # drop in-progress current day
            data[pd.Timestamp(d)] = cl
    return pd.Series(data).sort_index()


class CryptoTrendStrategy(Strategy):
    def __init__(self, universe=None, fee_per_side=ct.FEE_MAKER,
                 slip_per_side=ct.SLIP_SIDE, band=0.004):
        super().__init__(name='crypto_trend', fee_per_side=fee_per_side,
                         slip_per_side=slip_per_side, band=band)
        self.universe = universe or TREND_UNIVERSE

    def compute(self, asof=None):
        closes = {}
        for coin, pid in self.universe.items():
            try:
                s = _fetch_daily_closes(pid)
            except Exception as e:
                print(f"    WARN trend {coin} ({pid}): {e}")
                continue
            if asof is not None:
                s = s[s.index <= pd.Timestamp(asof)]
            if len(s) > 130:
                closes[coin] = s
        if not closes:
            raise RuntimeError("crypto_trend: no price data")
        data = {coin: pd.DataFrame({'close': s, 'ret': s.pct_change()})
                for coin, s in closes.items()}
        C, R, V = ct.build_panel(data, '1d')
        F = ct.forecast_panel(C, V, None)
        W = ct.weights(F, V, '1d', 'longflat')
        bar_date = C.index[-1].date()
        tw = {s: float(W[s].iloc[-1]) for s in W.columns if pd.notna(W[s].iloc[-1])}
        px = {s: float(C[s].iloc[-1]) for s in C.columns if pd.notna(C[s].iloc[-1])}
        return bar_date, px, tw


# ── Funding carry ────────────────────────────────────────────────────────────

COINBASE_PERP = "https://api.coinbase.com/api/v3/brokerage/market/products/{pid}"
COINBASE_INTX_FUNDING = ("https://api.international.coinbase.com/api/v1/instruments/"
                         "{sym}/funding?result_limit=300")
# Quality coins only (reliable funding). Coinbase perp product ids.
CARRY_UNIVERSE = {
    'BTC': 'BTC-PERP-INTX', 'ETH': 'ETH-PERP-INTX', 'XRP': 'XRP-PERP-INTX',
    'ADA': 'ADA-PERP-INTX', 'LINK': 'LINK-PERP-INTX', 'DOGE': 'DOGE-PERP-INTX',
    'LTC': 'LTC-PERP-INTX',
}
CARRY_FUNDING_SMOOTH = 0.1     # EWMA alpha for the daily-funding signal


def _fetch_spot_perp(pid):
    """Current spot (index) and perp price snapshot, for the basis leg."""
    d = _get_json(COINBASE_PERP.format(pid=pid))
    fpd = d['future_product_details']
    return {'perp': float(d['price']), 'spot': float(fpd['index_price'])}


def _fetch_funding_since(sym, since_iso):
    """
    REALIZED hourly funding from Coinbase INTX (public). Returns
    (sum_rate_since, latest_event_time, recent_rates) where sum_rate_since is the
    total funding settled with event_time > since_iso (None on first run → 0).
    Newest-first list. result_limit=300 buffers ~12 days against missed runs.
    """
    d = _get_json(COINBASE_INTX_FUNDING.format(sym=sym))
    results = d.get('results', [])
    total, latest, recent = 0.0, since_iso, []
    for r in results:
        et = r['event_time']
        rate = float(r['funding_rate'])
        recent.append(rate)
        if since_iso is not None and et > since_iso:
            total += rate
        if latest is None or et > latest:
            latest = et
    return total, latest, recent


class CryptoCarryStrategy(Strategy):
    """Long spot / short perp funding carry. Weight 1/n among +funding coins.
    Accrues REALIZED hourly funding (summed per daily run), not an estimate."""

    def __init__(self, universe=None, fee_per_side=0.0039, slip_per_side=0.0006, band=0.01):
        # cost = both legs: spot(0.35%+0.03%) + perp(0.04%+0.03%) ≈ 0.45%/toggle.
        super().__init__(name='crypto_carry', fee_per_side=fee_per_side,
                         slip_per_side=slip_per_side, band=band)
        self.universe = universe or CARRY_UNIVERSE
        self.aux_path = os.path.join(_LEDGERS, 'crypto_carry', 'carry_aux.json')

    def _load_aux(self):
        if os.path.exists(self.aux_path):
            with open(self.aux_path) as fh:
                return json.load(fh)
        return {'date': None, 'coins': {}}

    def _save_aux(self, aux):
        os.makedirs(os.path.dirname(self.aux_path), exist_ok=True)
        with open(self.aux_path, 'w') as fh:
            json.dump(aux, fh, indent=2)

    def compute(self, asof=None):
        today = (pd.Timestamp(asof).date() if asof else datetime.now(timezone.utc).date())
        aux = self._load_aux()
        prev_date = pd.Timestamp(aux['date']).date() if aux['date'] else None
        days = max((today - prev_date).days, 1) if prev_date else 1

        coins = aux['coins']
        prices = {}
        for coin, pid in self.universe.items():
            intx_sym = pid.replace('-INTX', '')          # 'BTC-PERP-INTX' → 'BTC-PERP'
            c = coins.get(coin)
            try:
                q = _fetch_spot_perp(pid)
                funding_since, latest_ft, recent = _fetch_funding_since(
                    intx_sym, c['last_ft'] if c else None)
            except Exception as e:
                print(f"    WARN carry {coin} ({pid}): {e}")
                continue
            if c is None:                                # first sight: seed, no return
                seed_daily = sum(recent[:24]) if recent else 0.0   # ~1 day of realized funding
                coins[coin] = {'index': 1.0, 'spot': q['spot'], 'perp': q['perp'],
                               'sf': seed_daily, 'last_ft': latest_ft}
            else:
                basis = (q['spot'] / c['spot'] - 1) - (q['perp'] / c['perp'] - 1)
                carry_ret = funding_since + basis        # REALIZED funding collected + basis P&L
                c['index'] *= (1 + carry_ret)
                c['spot'], c['perp'], c['last_ft'] = q['spot'], q['perp'], latest_ft
                a = CARRY_FUNDING_SMOOTH
                c['sf'] = a * (funding_since / days) + (1 - a) * c['sf']   # daily-funding EWMA
            prices[coin] = coins[coin]['index']

        # Signal: hold coins with positive smoothed funding, equal weight (gross ≤ 1).
        active = [c for c in prices if coins[c]['sf'] > 0]
        w = 1.0 / len(active) if active else 0.0
        target_w = {c: (w if c in active else 0.0) for c in prices}

        aux['date'] = str(today)
        aux['coins'] = coins
        self._save_aux(aux)
        return today, prices, target_w
