#!/usr/bin/env python3
"""
Combined crypto portfolio: daily TREND (long/flat) + funding CARRY.

Blends the two validated return streams at a capital split to show the combined
equity curve, Sharpe/Calmar, and — the honest part — how much (or little) the
drawdown improves, given they share crypto bull/bear regime exposure.

Usage:  python3 crypto/portfolio_backtest.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crypto_carry_backtest import load_panel, backtest as carry_bt, trend_daily, QUALITY, COST_TOGGLE


def stats(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    maxdd = ((eq - eq.cummax()) / eq.cummax()).min()
    sh = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else np.nan
    return dict(cagr=cagr*100, vol=r.std()*np.sqrt(365)*100, sharpe=sh,
                maxdd=maxdd*100, calmar=cagr/abs(maxdd) if maxdd < 0 else np.nan)


def main():
    trend = trend_daily()
    Fund, Sret, Pret = load_panel(QUALITY)
    carry, _, _, _ = carry_bt(Fund, Sret, Pret, COST_TOGGLE)
    df = pd.DataFrame({'trend': trend, 'carry': carry}).dropna()
    print(f"Overlap {df.index[0].date()}→{df.index[-1].date()}  ({len(df)} days)  "
          f"daily corr {df['trend'].corr(df['carry']):+.3f}\n")

    print("=" * 84)
    print(f"  {'allocation':<18}{'Sharpe':>8}{'Calmar':>8}{'CAGR%':>8}{'Vol%':>7}{'MaxDD%':>9}")
    print("=" * 84)
    allocs = [('100% trend', 1.0, 0.0), ('85/15 trend/carry', 0.85, 0.15),
              ('70/30', 0.70, 0.30), ('50/50', 0.50, 0.50), ('100% carry', 0.0, 1.0)]
    curves = {}
    for label, wt, wc in allocs:
        blend = wt * df['trend'] + wc * df['carry']
        s = stats(blend)
        curves[label] = (1 + blend).cumprod()
        print(f"  {label:<18}{s['sharpe']:>8.2f}{s['calmar']:>8.2f}{s['cagr']:>8.1f}"
              f"{s['vol']:>7.1f}{s['maxdd']:>9.1f}")
    print("=" * 84)

    # Honest diversification check: combined maxDD vs weighted-average maxDD
    dd_t, dd_c = stats(df['trend'])['maxdd'], stats(df['carry'])['maxdd']
    dd_blend = stats(0.7*df['trend'] + 0.3*df['carry'])['maxdd']
    print(f"\n  Diversification reality (70/30): standalone maxDD trend {dd_t:.1f}% / "
          f"carry {dd_c:.1f}%")
    print(f"  weighted-avg would be {0.7*dd_t+0.3*dd_c:.1f}%; actual blend {dd_blend:.1f}%")
    print(f"  → modest DD improvement: they draw down together in bears (regime-correlated).")

    # ── Plot 70/30 combined vs standalones ───────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                   gridspec_kw={'height_ratios': [3, 1]})
    for label, color, lw in [('100% trend', 'steelblue', 1.0),
                             ('100% carry', 'seagreen', 1.0),
                             ('70/30', 'black', 1.8)]:
        eq = curves[label]
        ax1.plot(eq.index, eq.values, color=color, lw=lw, label=label)
    ax1.set_yscale('log'); ax1.set_ylabel('growth of $1 (log)')
    ax1.set_title('Crypto portfolio: trend + funding carry (net of realistic cost)')
    ax1.legend(); ax1.grid(True, alpha=0.3, which='both')
    blend = 0.7*df['trend'] + 0.3*df['carry']
    eqb = (1 + blend).cumprod()
    dd = (eqb - eqb.cummax()) / eqb.cummax() * 100
    ax2.fill_between(dd.index, dd, 0, color='crimson', alpha=0.4)
    ax2.set_ylabel('70/30 drawdown (%)'); ax2.set_xlabel('date'); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/crypto_portfolio_equity.png', dpi=150)
    print("\n  Chart saved to results/crypto_portfolio_equity.png")


if __name__ == '__main__':
    main()
