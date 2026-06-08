#!/usr/bin/env python3
"""
Funding-rate carry — STAGE 1 characterization
────────────────────────────────────────────────
Perp funding is persistently positive (retail is structurally long) ⇒ a
market-neutral cash-and-carry (long spot / short perp) collects it. Before
building the full basis model (which needs perp prices), answer the cheap,
decisive questions from funding data alone:

  1. How big is the carry, and is it PERSISTENT across regimes or just a
     2020-21 bull artifact?  (annualised funding by calendar year)
  2. How often is funding negative (when a short PAYS)?
  3. Does a simple "collect only when funding is positive" filter help?
  4. Is the carry return stream uncorrelated to the daily trend book?

NOTE: the Sharpe of pure funding collection is flattered — it omits basis P&L
and tail/liquidation risk of the two-leg trade. Stage 1 is about MAGNITUDE and
REGIME, not a tradeable Sharpe. That comes in stage 2 with perp prices.

Usage:  python3 crypto/crypto_carry.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crypto_trend as ct
from crypto_mr_backtest import FNAME_RE


def load_funding():
    funding = {}
    for p in sorted(glob.glob("Data/*_funding_binance.csv")):
        coin = os.path.basename(p).split('_')[0].upper().replace('USDT', '')
        df = pd.read_csv(p, parse_dates=['time']).set_index('time')
        funding[coin] = df
    return funding


def ann_factor(interval_hours):
    return 24 / interval_hours * 365


def main():
    funding = load_funding()
    print(f"Loaded funding for {len(funding)} coins\n")

    # ── 1) Annualised funding by calendar year (the regime question) ─────────
    coins = list(funding.keys())
    years = range(2020, 2027)
    print("=" * 92)
    print("  ANNUALISED FUNDING BY YEAR  (%, what a short collects; negative = short pays)")
    print("=" * 92)
    print(f"  {'coin':<7}" + "".join(f"{y:>8}" for y in years) + f"{'  all':>9}")
    table = {}
    for c in coins:
        d = funding[c]
        af = ann_factor(d['interval_hours'].median())
        row = ""
        for y in years:
            yr = d[d.index.year == y]['rate']
            row += f"{yr.mean()*af*100:>8.1f}" if len(yr) else f"{'—':>8}"
        allmean = d['rate'].mean() * af * 100
        table[c] = d
        print(f"  {c:<7}{row}{allmean:>+9.1f}")
    print("=" * 92)

    # ── 2) Build aligned 8h funding panel + simple strategies ────────────────
    idx = sorted(set().union(*[set(d.index) for d in funding.values()]))
    F = pd.DataFrame({c: funding[c]['rate'].reindex(idx) for c in coins}).sort_index()

    # "Always short all": equal-weight, collect funding every period.
    always = F.mean(axis=1)
    # "Short only when positive": per coin, short next period only if last funding > 0.
    sig = (F.shift(1) > 0).astype(float)
    cond = (F * sig).sum(axis=1) / sig.sum(axis=1).replace(0, np.nan)
    cond = cond.fillna(0.0)

    pos_share = (F > 0).mean().mean() * 100
    print(f"\n  Funding positive {pos_share:.0f}% of the time (across coins/periods)")

    def carry_stats(stream, label):
        s = stream.dropna()
        # periods/year from median spacing
        ppd = pd.Series(s.index).diff().dt.total_seconds().median()
        per_yr = 365 * 86400 / ppd if ppd else 1095
        ann = s.mean() * per_yr * 100
        vol = s.std() * np.sqrt(per_yr) * 100
        shp = s.mean() / s.std() * np.sqrt(per_yr) if s.std() > 0 else np.nan
        print(f"  {label:<26} ann {ann:>+6.1f}%   vol {vol:>5.1f}%   "
              f"'Sharpe' {shp:>5.1f} (gross-of-basis, overstated)")
        return s

    print("\n  Gross carry (funding only, equal-weight portfolio):")
    s_always = carry_stats(always, "always short all")
    s_cond   = carry_stats(cond, "short only when +funding")

    # ── 3) By-year gross carry of the always-short portfolio ─────────────────
    print("\n  Always-short portfolio, annualised carry by year:")
    per_yr = 1095
    row = "   "
    for y in years:
        yr = always[always.index.year == y]
        row += f"{y}:{yr.mean()*per_yr*100:>5.1f}%  " if len(yr) else ""
    print(row)

    # ── 4) Correlation to the daily trend book ───────────────────────────────
    carry_daily = (s_cond.fillna(0)).resample('D').sum()      # funding summed per day
    carry_daily.index = carry_daily.index.normalize()
    data = {}
    for p in sorted(glob.glob("Data/*_1d_binance.csv")):
        m = FNAME_RE.search(os.path.basename(p))
        if m:
            data[m.group(1).upper()] = ct.load_clean(p, '1d', False)
    C, R, V = ct.build_panel(data, '1d')
    Fc = ct.forecast_panel(C, V, None)
    trend, _, _ = ct.backtest(Fc, R, V, '1d', 'longflat', band=0.004)
    trend.index = pd.DatetimeIndex(trend.index).tz_localize(None).normalize()
    df = pd.DataFrame({'carry': carry_daily, 'trend': trend}).dropna()
    df = df[(df['carry'] != 0) | (df['trend'] != 0)]
    print(f"\n  Correlation carry↔trend (daily): {df['carry'].corr(df['trend']):+.3f}  "
          f"(n={len(df)})")
    print("=" * 92)
    print("  Key question: is the by-year carry consistently positive, or bull-only?")


if __name__ == '__main__':
    main()
