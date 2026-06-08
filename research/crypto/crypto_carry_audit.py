#!/usr/bin/env python3
"""
Funding-carry backtest AUDIT — hunt for the bias inflating Sharpe to 5.

Checks, in order of suspicion:
  1. Return decomposition: funding vs basis (spot−perp). Basis should be ~0 mean
     and just add vol; if it's contributing return, that's a one-time/biased gain.
  2. Is the basis term a one-time decay? (cumulative path front/back-loaded)
  3. Data sanity: spot/perp alignment, basis distribution, absurd (spot−perp) days
     that signal misaligned or mis-scaled series.
  4. Vol realism: daily-close vol vs a higher bar — is the low vol an artifact?
  5. Cost realism: add slippage on both legs, re-price.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crypto_carry_backtest import load_panel, QUALITY, SPOT_FEE, PERP_FEE, SIG_SPAN


def ann_sharpe(r):
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else np.nan


def main():
    Fund, Sret, Pret = load_panel(QUALITY)
    coins = list(Fund.columns)
    print(f"Quality universe: {coins}\n")

    # ── Signal & weights (same as backtest) ──────────────────────────────────
    sig = (Fund.ewm(span=SIG_SPAN, min_periods=5).mean().shift(1) > 0).astype(float)
    sig = sig.where(Fund.notna().shift(1), 0.0)
    w = sig.div(sig.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    funding_pnl = (w * Fund.fillna(0.0)).sum(axis=1)
    basis_pnl   = (w * (Sret - Pret).fillna(0.0)).sum(axis=1)
    total = funding_pnl + basis_pnl

    print("=" * 74)
    print("  1) RETURN DECOMPOSITION  (portfolio, gross)")
    print("=" * 74)
    for label, s in [('funding only', funding_pnl), ('basis only (spot-perp)', basis_pnl),
                     ('total', total)]:
        eq = (1 + s).cumprod()
        print(f"  {label:<24} annRet {s.mean()*365*100:>+6.2f}%   "
              f"vol {s.std()*np.sqrt(365)*100:>5.2f}%   Sharpe {ann_sharpe(s):>6.2f}   "
              f"cum {(eq.iloc[-1]-1)*100:>+7.1f}%")

    # ── 2) Is basis a one-time decay? split-half contribution ────────────────
    half = len(basis_pnl) // 2
    print("\n" + "=" * 74)
    print("  2) BASIS P&L over time (one-time decay would be front-loaded)")
    print("=" * 74)
    print(f"  first-half cum basis: {basis_pnl.iloc[:half].sum()*100:+.1f}%   "
          f"second-half: {basis_pnl.iloc[half:].sum()*100:+.1f}%")
    print("  by year:  " + "  ".join(
        f"{y}:{basis_pnl[basis_pnl.index.year==y].sum()*100:+.1f}%"
        for y in range(2020, 2027)))

    # ── 3) Data sanity: raw basis per coin, alignment, outliers ──────────────
    print("\n" + "=" * 74)
    print("  3) DATA SANITY — raw (spot_ret − perp_ret) per coin")
    print("=" * 74)
    print(f"  {'coin':<7}{'mean ann%':>11}{'std daily%':>12}{'max|day|%':>11}{'#>2%':>7}")
    diff = Sret - Pret
    for c in coins:
        d = diff[c].dropna()
        print(f"  {c:<7}{d.mean()*365*100:>+11.2f}{d.std()*100:>12.3f}"
              f"{d.abs().max()*100:>11.2f}{(d.abs()>0.02).sum():>7}")
    print("  (mean ann should be ~0; a big positive = basis-decay bias or misalignment)")

    # ── 4) Vol realism: funding-only Sharpe is the crux ──────────────────────
    print("\n" + "=" * 74)
    print("  4) WHY SHARPE IS HIGH")
    print("=" * 74)
    print(f"  funding-only daily vol: {funding_pnl.std()*100:.3f}%  → ann {funding_pnl.std()*np.sqrt(365)*100:.2f}%")
    print(f"  This is tiny because funding is a steady drip. Daily-close basis vol")
    print(f"  ({basis_pnl.std()*np.sqrt(365)*100:.1f}%) is the only risk captured — intraday")
    print(f"  basis blowouts / liquidation are INVISIBLE at daily resolution.")

    # ── 5) Cost realism: add slippage on both legs ───────────────────────────
    print("\n" + "=" * 74)
    print("  5) COST with slippage added (both legs)")
    print("=" * 74)
    turnover = (w - w.shift(1)).abs().sum(axis=1)
    for slip in [0.0, 0.0005, 0.0010, 0.0020]:
        cps = SPOT_FEE + PERP_FEE + 2 * slip      # slip on each leg
        net = total - turnover * cps
        print(f"  slippage {slip*100:.2f}%/leg → cost {cps*100:.2f}%/toggle  "
              f"Sharpe {ann_sharpe(net):>5.2f}  annRet {net.mean()*365*100:>+5.1f}%")


if __name__ == '__main__':
    main()
