#!/usr/bin/env python3
"""
Trend strategy evaluation — is it worth ADDING?
─────────────────────────────────────────────────
Two questions the raw Sharpe can't answer:

  1. Does trend-timing beat just holding crypto?  → vs buy-&-hold BTC and an
     equal-weight basket over the same window (quantify the drawdown value-add).
  2. Is it a diversifier to the IBS futures book?  → correlation of daily returns,
     and what a blended portfolio's Sharpe looks like across allocations.

Reuses the validated long/flat EWMAC ensemble from crypto_trend and the real IBS
backtest from aggregate_port. Returns are aligned on business days (IBS trades
weekdays; crypto's weekend moves fold into the Monday bar).

Usage:  python3 crypto/crypto_trend_eval.py
"""

import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))   # repo root, for `portfolio` package
import crypto_trend as ct
from crypto_mr_backtest import load_clean, FNAME_RE
import glob

BPY = 365          # crypto / calendar annualisation for the daily series
BPY_B = 252        # business-day annualisation once aligned to weekdays


def to_dateindex(s):
    """Normalise any (possibly tz-aware) datetime-indexed series to naive dates."""
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s = s.copy(); s.index = idx.normalize()
    return s[~s.index.duplicated(keep='last')]


def ann_sharpe(r, bpy):
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(bpy) if not r.empty and r.std() > 0 else np.nan


def maxdd(r):
    eq = (1 + r.fillna(0)).cumprod()
    return ((eq - eq.cummax()) / eq.cummax()).min() * 100


# ── Crypto trend (long/flat ensemble) + benchmarks ───────────────────────────

def crypto_returns():
    data = {}
    for p in sorted(glob.glob("Data/*_1d_binance.csv")):
        m = FNAME_RE.search(os.path.basename(p))
        if m:
            data[m.group(1).upper()] = load_clean(p, '1d', False)
    C, R, V = ct.build_panel(data, '1d')
    F = ct.forecast_panel(C, V, None)
    net, _, _ = ct.backtest(F, R, V, '1d', 'longflat')
    trend = to_dateindex(net)
    btc_col = 'BTCUSDT' if 'BTCUSDT' in C.columns else C.columns[0]
    btc = to_dateindex(C[btc_col].pct_change())
    ew = to_dateindex(R.mean(axis=1))        # equal-weight daily-rebalanced basket
    return trend, btc, ew


# ── IBS futures portfolio returns (real backtest) ────────────────────────────

def ibs_returns():
    import portfolio.aggregate_port as ap
    idata = ap.load_all_data()
    ind = ap.precompute_ibs(idata)
    state = ap.run_backtest(idata, ind)
    cal = pd.date_range(start=ap.START_DATE, end=ap.END_DATE, freq='B')
    series = []
    for name, _ in ap.STRATEGIES:
        curve = pd.DataFrame(state[name]['equity_curve'], columns=['Time', 'Equity'])
        curve = curve.set_index('Time')
        curve = curve[~curve.index.duplicated(keep='last')].sort_index()
        series.append(curve['Equity'].reindex(cal, method='ffill'))
    combined = sum(series).dropna()
    return to_dateindex(combined.pct_change())


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Building crypto trend + benchmarks...")
    trend, btc, ew = crypto_returns()
    print("Running IBS futures backtest...")
    ibs = ibs_returns()

    # ── 1) Trend vs buy-and-hold (crypto window, calendar-daily) ─────────────
    span = trend.dropna().index
    btc_w, ew_w = btc.reindex(span), ew.reindex(span)
    print("\n" + "=" * 64)
    print("  1) TREND vs BUY-AND-HOLD  (crypto window, daily)")
    print("=" * 64)
    print(f"     {'strategy':<22}{'Sharpe':>8}{'AnnRet%':>9}{'MaxDD%':>9}")
    for label, r in [('Trend long/flat', trend), ('Buy&hold BTC', btc_w),
                     ('Equal-weight basket', ew_w)]:
        eq = (1 + r.fillna(0)).cumprod()
        yrs = (r.dropna().index[-1] - r.dropna().index[0]).days / 365.25
        ann = (eq.iloc[-1] ** (1 / yrs) - 1) * 100
        print(f"     {label:<22}{ann_sharpe(r, BPY):>8.2f}{ann:>9.1f}{maxdd(r):>9.1f}")
    print("\n     → Trend's job is drawdown control, not out-returning a bull.")

    # ── 2) Correlation to IBS + blended portfolio ────────────────────────────
    df = pd.DataFrame({'IBS': ibs, 'CryptoTrend': trend}).dropna()
    rho = df['IBS'].corr(df['CryptoTrend'])
    s_ibs = ann_sharpe(df['IBS'], BPY_B)
    s_cry = ann_sharpe(df['CryptoTrend'], BPY_B)

    print("\n" + "=" * 64)
    print("  2) DIVERSIFICATION vs IBS FUTURES  (business-day overlap)")
    print("=" * 64)
    print(f"     overlap: {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}  ({len(df)} days)")
    print(f"     correlation(IBS, CryptoTrend) = {rho:+.3f}")
    print(f"     standalone Sharpe — IBS {s_ibs:.2f}   CryptoTrend {s_cry:.2f}")

    # Blend vol-normalised streams across crypto allocations.
    ibs_n = df['IBS'] / df['IBS'].std()
    cry_n = df['CryptoTrend'] / df['CryptoTrend'].std()
    print(f"\n     {'crypto alloc':<14}{'blend Sharpe':>13}")
    best = (0, -9)
    for x in [0.0, 0.1, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        blend = (1 - x) * ibs_n + x * cry_n
        s = ann_sharpe(blend, BPY_B)
        flag = ''
        if s > best[1]:
            best = (x, s)
        print(f"     {x*100:>5.0f}%{'':<8}{s:>13.2f}")
    print(f"\n     → best blend ≈ {best[0]*100:.0f}% crypto risk, Sharpe {best[1]:.2f} "
          f"(vs IBS-only {ann_sharpe(ibs_n, BPY_B):.2f})")
    print("     (vol-normalised streams; illustrates the diversification lift, "
          "not a position-sizing recommendation.)")


if __name__ == '__main__':
    main()
