#!/usr/bin/env python3
"""
Answer three sharp questions about the carry backtest with hard numbers:
  1. How can Sharpe be ~4.5 with that equity curve? (Sharpe vs Calmar vs underwater)
  2. Do the ~10% returns reconcile with the curve? (CAGR check)
  3. How is it 'orthogonal' to trend if both like bulls / suffer bears?
     (daily corr vs REGIME corr)
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crypto_carry_backtest import load_panel, backtest, trend_daily, QUALITY, COST_TOGGLE


def max_underwater_days(eq):
    peak = eq.cummax()
    underwater = eq < peak * (1 - 1e-9)
    best = cur = 0
    for u in underwater:
        cur = cur + 1 if u else 0
        best = max(best, cur)
    return best


def main():
    Fund, Sret, Pret = load_panel(QUALITY)
    carry, _, _, _ = backtest(Fund, Sret, Pret, COST_TOGGLE)
    trend = trend_daily()
    df = pd.DataFrame({'carry': carry, 'trend': trend}).dropna()
    c = df['carry']

    eq = (1 + c).cumprod()
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    maxdd = ((eq - eq.cummax()) / eq.cummax()).min()
    sharpe = c.mean() / c.std() * np.sqrt(365)

    print("=" * 70)
    print("  Q1 & Q2 — carry risk/return reconciled")
    print("=" * 70)
    print(f"  total return {(eq.iloc[-1]-1)*100:>6.1f}%   over {yrs:.1f} yr   "
          f"→ CAGR {cagr*100:.1f}%   (matches the curve $1→${eq.iloc[-1]:.2f})")
    print(f"  daily vol {c.std()*100:.3f}%  → ann vol {c.std()*np.sqrt(365)*100:.1f}%")
    print(f"  SHARPE {sharpe:.2f}   = high only because daily vol is tiny")
    print(f"  CALMAR (CAGR/|maxDD|) {cagr/abs(maxdd):.2f}   ← the honest risk-adjusted number")
    print(f"  max drawdown {maxdd*100:.1f}%   longest underwater: "
          f"{max_underwater_days(eq)} days ({max_underwater_days(eq)/365:.1f} yr)")
    # autocorrelation / variance ratio: drawdowns are slow & persistent
    ac1 = c.autocorr(1)
    vr20 = c.rolling(20).sum().var() / (20 * c.var())
    print(f"  lag-1 autocorr {ac1:+.3f}   variance-ratio(20d) {vr20:.2f}  "
          f"(>1 ⇒ persistent moves ⇒ true risk > sqrt-time, Sharpe overstated)")

    print("\n" + "=" * 70)
    print("  Q3 — 'orthogonal'? daily corr vs REGIME corr")
    print("=" * 70)
    print(f"  overall daily corr(carry, trend) = {df['carry'].corr(df['trend']):+.3f}")

    # Regime split: BTC above/below its 200-day MA
    btc = pd.read_csv("Data/btcusdt_1d_binance.csv", parse_dates=['time']).set_index('time')['close']
    btc.index = btc.index.tz_localize(None).normalize() if btc.index.tz is not None else btc.index.normalize()
    bull = (btc > btc.rolling(200).mean()).reindex(df.index).ffill().fillna(False)

    for name, mask in [('BULL (BTC>200dMA)', bull), ('BEAR (BTC<200dMA)', ~bull)]:
        sub = df[mask]
        if len(sub) < 30:
            continue
        car = sub['carry'].mean() * 365 * 100
        tre = sub['trend'].mean() * 365 * 100
        rho = sub['carry'].corr(sub['trend'])
        print(f"  {name:<20} carry {car:>+6.1f}%/yr   trend {tre:>+6.1f}%/yr   "
              f"corr {rho:+.2f}   ({len(sub)} days)")
    print("\n  → daily corr ~0 (different daily drivers: funding vs price), BUT both")
    print("    earn in bulls & bleed in bears = shared REGIME exposure. They will")
    print("    tend to draw down TOGETHER in a crypto bear. 'Uncorrelated' day-to-day")
    print("    ≠ uncorrelated in the tail.")


if __name__ == '__main__':
    main()
