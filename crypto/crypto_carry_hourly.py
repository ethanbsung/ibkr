#!/usr/bin/env python3
"""
Funding carry at HOURLY resolution — the realistic-risk recompute.

The daily-close backtest gave Sharpe 5 because daily closes smooth over the
position's real intraday mark-to-market swings. This re-marks the cash-and-carry
EVERY HOUR (capturing intraday basis vol), collects funding at the 8h
settlements, and uses the perp's intraday high to test whether the short-perp
leg would have been LIQUIDATED at various leverage levels.

Expect: vol up, Sharpe down to something believable, max drawdown deeper, and a
clear read on what leverage survives the deleveraging events.

Data: hourly spot (*_1h_binance.csv) + hourly perp (*_1h_perp_binance.csv,
OHLC) + 8h funding.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crypto_carry_backtest import QUALITY, SIG_SPAN

BPY_H = 365 * 24


def load_hourly(coins):
    spot, perp_c, perp_h, fund = {}, {}, {}, {}
    for coin in coins:
        c = coin.lower()
        sp = f"Data/{c}usdt_1h_binance.csv"
        pp = f"Data/{c}usdt_1h_perp_binance.csv"
        fp = f"Data/{c}usdt_funding_binance.csv"
        if not all(os.path.exists(x) for x in (sp, pp, fp)):
            continue
        s = pd.read_csv(sp, parse_dates=['time']).set_index('time')['close']
        p = pd.read_csv(pp, parse_dates=['time']).set_index('time')[['high', 'close']]
        f = pd.read_csv(fp, parse_dates=['time']).set_index('time')['rate']
        spot[coin] = s; perp_c[coin] = p['close']; perp_h[coin] = p['high']; fund[coin] = f
    idx = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in spot.values()])))
    Sp = pd.DataFrame({c: spot[c].reindex(idx) for c in spot})
    Pc = pd.DataFrame({c: perp_c[c].reindex(idx) for c in spot})
    Ph = pd.DataFrame({c: perp_h[c].reindex(idx) for c in spot})
    Fu = pd.DataFrame({c: fund[c].reindex(idx) for c in spot})      # NaN except 8h marks
    return Sp, Pc, Ph, Fu


def main():
    Sp, Pc, Ph, Fu = load_hourly(QUALITY)
    coins = list(Sp.columns)
    print(f"Hourly carry recompute — {coins}\n")

    sret = Sp.pct_change()
    pret = Pc.pct_change()

    # Hold signal: daily-smoothed funding > 0, lagged, broadcast to hourly.
    fund_daily = Fu.groupby(Fu.index.normalize()).sum()
    sig_daily = (fund_daily.ewm(span=SIG_SPAN, min_periods=5).mean().shift(1) > 0).astype(float)
    sig = sig_daily.reindex(Sp.index.normalize()).set_axis(Sp.index)
    w = sig.div(sig.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # Hourly P&L: basis MTM every hour + funding collected at 8h settlements.
    basis_h = (w * (sret - pret)).sum(axis=1)
    funding_h = (w * Fu.fillna(0.0)).sum(axis=1)        # Fu non-zero only at settlements
    total_h = (basis_h + funding_h).dropna()

    def stats(r, label):
        r = r.dropna()
        eq = (1 + r).cumprod()
        ann = r.mean() * BPY_H * 100
        vol = r.std() * np.sqrt(BPY_H) * 100
        sh = r.mean() / r.std() * np.sqrt(BPY_H) if r.std() > 0 else np.nan
        dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
        print(f"  {label:<26} ann {ann:>+6.1f}%  vol {vol:>5.1f}%  Sharpe {sh:>5.2f}  maxDD {dd:>6.1f}%")

    print("=" * 78)
    print("  HOURLY-RESOLUTION CARRY  (gross, basis marked every hour)")
    print("=" * 78)
    stats(funding_h, "funding only")
    stats(basis_h, "basis only (hourly MTM)")
    stats(total_h, "total")
    print("  Compare daily-close: total gross Sharpe was 8.3, vol 1.8%.")

    # ── Liquidation test on the short-perp leg ───────────────────────────────
    # Short perp loses when perp rises. Per hour, adverse move = perp_high/perp_prev_close − 1.
    print("\n" + "=" * 78)
    print("  LIQUIDATION TEST — short-perp leg, intraday adverse (upward) excursion")
    print("=" * 78)
    adverse = (Ph / Pc.shift(1) - 1)                    # per-coin hourly up-spike vs prior close
    adverse = adverse.where(w.shift(1) > 0)             # only while positioned
    worst = adverse.max()
    print(f"  {'coin':<7}{'worst 1h up-spike%':>20}{'when':>14}")
    for c in coins:
        a = adverse[c].dropna()
        if a.empty:
            continue
        print(f"  {c:<7}{a.max()*100:>19.1f}%  {a.idxmax():%Y-%m-%d}")
    print(f"\n  Leverage survivability (a short at Lx liquidates on a +{'{:.0f}'.format(0)}…/L move):")
    print(f"  {'leverage':<10}{'liq threshold':>15}{'# coin-hours liquidated':>26}")
    for L in [2, 3, 5, 10]:
        thr = 1.0 / L
        liq = (adverse > thr).sum().sum()
        print(f"  {L}x{'':<8}{'+'+format(thr*100,'.0f')+'%':>15}{int(liq):>26}")
    print("\n  Even one liquidation = the carry's catastrophic tail (lose the perp margin).")


if __name__ == '__main__':
    main()
