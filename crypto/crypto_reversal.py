#!/usr/bin/env python3
"""
Liquidation / overreaction reversal — EVENT STUDY (stage 1)
─────────────────────────────────────────────────────────────
Question: after an extreme short-horizon move (liquidation cascade / panic),
does price OVERSHOOT and partially revert? And is it a real edge or just the
bad-print round-trips we flagged in the data audit?

Method: define a shock as a 1h return beyond Z σ of the coin's *recent* vol
(vol lagged so the shock doesn't inflate its own denominator). For every shock,
measure the forward cumulative return over H = 1..48 hours, pooled across the 10
majors. Compare down-shocks (expect bounce → positive fwd) and up-shocks (expect
fade → negative fwd) against the unconditional baseline. Spikes are EXCLUDED so a
real overreaction edge is separated from data errors.

No cost, no sizing — this only answers "is the signal there?" before we build a
tradeable backtest.

Usage:
  python3 crypto/crypto_reversal.py
  python3 crypto/crypto_reversal.py --vol-span 168 --interval 1h
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crypto_mr_backtest import load_clean, FNAME_RE

HORIZONS = [1, 2, 4, 8, 12, 24, 48]      # forward windows, in bars
Z_LEVELS = [2.5, 3.0, 4.0, 5.0]          # shock thresholds (σ)


def forward_returns(ret, horizons):
    """
    For each bar t, cumulative return over (t+1 .. t+H), per horizon.
    Built from clean returns (gaps already NaN → treated as 0 contribution).
    Returns dict[H] -> Series aligned to t.
    """
    logr = np.log1p(ret.fillna(0.0))
    out = {}
    for H in horizons:
        # sum of next H log-returns, shifted so it's indexed at the entry bar t
        fwd_log = logr.shift(-1).rolling(H).sum().shift(-(H - 1))
        out[H] = np.expm1(fwd_log)
    return out


def event_study(data, vol_span, interval):
    # Pool every coin's bars together.
    rows_down, rows_up, baseline = {H: [] for H in HORIZONS}, {H: [] for H in HORIZONS}, {H: [] for H in HORIZONS}

    for sym, df in data.items():
        ret = df['ret']
        vol = ret.ewm(span=vol_span, min_periods=vol_span).std().shift(1)  # lagged baseline vol
        z = ret / vol
        fwd = forward_returns(ret, HORIZONS)

        valid = z.notna()
        down = valid & (z < 0)            # filled per-threshold below
        for H in HORIZONS:
            f = fwd[H]
            ok = f.notna() & z.notna()
            baseline[H].append(f[ok])     # unconditional forward returns
            # store (z, fwd) pairs for thresholding
            rows_down[H].append(pd.DataFrame({'z': z[ok], 'f': f[ok]}))

    base = {H: pd.concat(baseline[H]) for H in HORIZONS}
    paired = {H: pd.concat(rows_down[H]) for H in HORIZONS}
    return base, paired


def summarize(base, paired):
    bpy_note = ""
    print("\nBaseline (unconditional) mean forward return, all bars:")
    print("  " + "  ".join(f"{H}h:{base[H].mean()*100:+.2f}%" for H in HORIZONS))

    for Z in Z_LEVELS:
        print("\n" + "=" * 86)
        print(f"  SHOCK THRESHOLD |z| > {Z}σ")
        print("=" * 86)
        for side, sign, label in [('DOWN-shock (fade crash → expect bounce)', -1, 'bounce'),
                                  ('UP-shock (fade rally → expect drop)', +1, 'drop')]:
            print(f"\n  {side}")
            print(f"  {'horizon':<9}{'n':>7}{'mean fwd':>11}{'median':>10}"
                  f"{'%pos':>8}{'t-stat':>9}{'vs base':>10}")
            for H in HORIZONS:
                d = paired[H]
                sel = d[d['z'] < -Z] if sign < 0 else d[d['z'] > Z]
                if len(sel) < 20:
                    print(f"  {H:<9}{len(sel):>7}{'—':>11}")
                    continue
                f = sel['f']
                t = f.mean() / (f.std() / np.sqrt(len(f))) if f.std() > 0 else np.nan
                edge = (f.mean() - base[H].mean()) * 100
                print(f"  {H:<9}{len(sel):>7}{f.mean()*100:>+10.2f}%{f.median()*100:>+9.2f}%"
                      f"{(f > 0).mean()*100:>7.1f}%{t:>9.2f}{edge:>+9.2f}%")
    print("\n" + "=" * 86)
    print("  Reading: for the DOWN-shock bounce edge to be real, mean-fwd should be")
    print("  positive, t-stat ≳ 2-3, and 'vs base' clearly positive (> unconditional).")


def main():
    ap = argparse.ArgumentParser(description='Liquidation/overreaction reversal event study')
    ap.add_argument('--interval', default='1h')
    ap.add_argument('--vol-span', type=int, default=168, help='EWMA vol span in bars (default 1 week)')
    args = ap.parse_args()

    paths = sorted(glob.glob(f"Data/*_{args.interval}_binance.csv"))
    data = {}
    for p in paths:
        m = FNAME_RE.search(os.path.basename(p))
        if m:
            data[m.group(1).upper()] = load_clean(p, args.interval, exclude_spikes=True)
    print(f"Loaded {len(data)} coins [{args.interval}], spikes EXCLUDED, "
          f"vol span {args.vol_span} bars")

    base, paired = event_study(data, args.vol_span, args.interval)
    summarize(base, paired)


if __name__ == '__main__':
    main()
