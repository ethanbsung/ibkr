#!/usr/bin/env python3
"""
Crypto data quality check
──────────────────────────
Audits the Binance CSVs produced by Data/binance_data_getter.py BEFORE any
backtest. Garbage-in-garbage-out is fatal for intraday mean-reversion: a single
bad print or a gap treated as a sequential bar manufactures fake reversion.

For every Data/*_binance.csv it reports:
  • coverage     — actual vs expected bars, % present
  • gaps         — count, total missing bars, the largest ones (with timestamps)
  • OHLC sanity  — high<max(o,c), low>min(o,c), high<low
  • flat / stale — zero-range bars and runs of identical closes (frozen feed)
  • zero volume  — bars with no trading
  • extreme moves— largest |returns| (eyeball: real event vs data error)
  • spike prints — round-trip spikes (big move immediately reversed) = likely bad data
  • gap-adjacent — # of returns that span a gap (must be masked in MR studies)

Outputs a console report, a summary table, and results/crypto_data_quality.png
(coverage timeline + return distributions).

Usage:
  python3 portfolio/crypto_data_quality.py
  python3 portfolio/crypto_data_quality.py --glob "Data/btc*_binance.csv"
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

INTERVAL_SECONDS = {'1m': 60, '5m': 300, '15m': 900, '30m': 1800,
                    '1h': 3600, '4h': 14400, '1d': 86400}

# Per-interval "extreme single-bar return" thresholds (fraction). Above this we
# list the move so you can judge real-event vs bad-print.
EXTREME_RET = {'1h': 0.15, '4h': 0.25, '1d': 0.40}

FNAME_RE = re.compile(r'([a-z0-9]+)_([0-9]+[mhd])_binance\.csv$')


def parse_name(path):
    m = FNAME_RE.search(os.path.basename(path))
    if not m:
        return None, None
    return m.group(1).upper(), m.group(2)


def load(path):
    df = pd.read_csv(path, parse_dates=['time']).set_index('time').sort_index()
    return df


def audit(df, interval):
    step = INTERVAL_SECONDS[interval]
    out = {}

    # ── Coverage & gaps ──────────────────────────────────────────────────────
    span_secs = (df.index[-1] - df.index[0]).total_seconds()
    expected = int(round(span_secs / step)) + 1
    out['bars'] = len(df)
    out['expected'] = expected
    out['coverage_pct'] = 100.0 * len(df) / expected if expected else np.nan

    deltas = df.index.to_series().diff().dt.total_seconds()
    gap_mask = deltas > step * 1.5
    gaps = deltas[gap_mask]
    out['n_gaps'] = int(len(gaps))
    out['missing_bars'] = int(((gaps / step) - 1).round().sum()) if len(gaps) else 0
    # largest gaps (end timestamp, duration in bars)
    out['top_gaps'] = sorted(
        [(ts, int(round(d / step)) - 1) for ts, d in gaps.items()],
        key=lambda x: -x[1])[:5]
    out['gap_adjacent_returns'] = int(gap_mask.sum())  # returns spanning a gap

    # ── OHLC sanity ──────────────────────────────────────────────────────────
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    ohlc_bad = (h < pd.concat([o, c], axis=1).max(axis=1)) | \
               (l > pd.concat([o, c], axis=1).min(axis=1)) | (h < l)
    out['ohlc_violations'] = int(ohlc_bad.sum())

    # ── Flat / stale / zero-volume ───────────────────────────────────────────
    out['zero_range'] = int(((h - l) == 0).sum())
    out['zero_volume'] = int((df['volume'] == 0).sum()) if 'volume' in df else 0
    # longest run of identical consecutive closes (frozen feed)
    same = (c.diff() == 0)
    run, best = 0, 0
    for v in same.to_numpy():
        run = run + 1 if v else 0
        best = max(best, run)
    out['max_flat_run'] = int(best)

    # ── Returns: extremes & round-trip spikes ────────────────────────────────
    ret = c.pct_change()
    # mask returns that span a gap (they're not real single-bar returns)
    ret_clean = ret.mask(gap_mask)
    thr = EXTREME_RET.get(interval, 0.40)
    extreme = ret_clean[ret_clean.abs() > thr]
    out['n_extreme'] = int(len(extreme))
    out['top_moves'] = sorted(
        [(ts, r) for ts, r in extreme.items()],
        key=lambda x: -abs(x[1]))[:5]

    # round-trip spike: big move immediately reversed by >=60% next bar
    r = ret_clean.to_numpy()
    spikes = 0
    for i in range(len(r) - 1):
        if np.isfinite(r[i]) and np.isfinite(r[i + 1]) and abs(r[i]) > thr:
            if np.sign(r[i]) != np.sign(r[i + 1]) and abs(r[i + 1]) > 0.6 * abs(r[i]):
                spikes += 1
    out['spike_prints'] = spikes

    out['daily_vol_pct'] = ret_clean.std() * 100
    out['_ret'] = ret_clean.dropna()
    return out


def fmt_pct(x):
    return f"{x:.2f}%" if np.isfinite(x) else "n/a"


def print_report(sym, interval, a):
    print(f"\n{'─'*70}")
    print(f"  {sym}  [{interval}]")
    print(f"{'─'*70}")
    print(f"  bars: {a['bars']:,} / {a['expected']:,} expected"
          f"   coverage: {fmt_pct(a['coverage_pct'])}")
    print(f"  gaps: {a['n_gaps']}  (~{a['missing_bars']} missing bars,"
          f" {a['gap_adjacent_returns']} gap-spanning returns to mask)")
    if a['top_gaps']:
        biggest = ", ".join(f"{ts:%Y-%m-%d}:{n}bars" for ts, n in a['top_gaps'][:3])
        print(f"        largest → {biggest}")

    flags = []
    if a['ohlc_violations']: flags.append(f"OHLC violations: {a['ohlc_violations']}")
    if a['zero_range']:      flags.append(f"zero-range bars: {a['zero_range']}")
    if a['zero_volume']:     flags.append(f"zero-volume bars: {a['zero_volume']}")
    if a['max_flat_run'] > 3: flags.append(f"longest flat run: {a['max_flat_run']}")
    if a['spike_prints']:    flags.append(f"round-trip spikes: {a['spike_prints']}")
    print(f"  integrity: {'  |  '.join(flags) if flags else 'clean'}")

    print(f"  per-bar vol: {fmt_pct(a['daily_vol_pct'])}"
          f"   extreme moves (>{int(EXTREME_RET.get(interval,0.4)*100)}%): {a['n_extreme']}")
    for ts, r in a['top_moves'][:5]:
        print(f"        {ts:%Y-%m-%d %H:%M}  {r*100:+.1f}%")


def plot(results, out='results/crypto_data_quality.png'):
    syms = [(s, i) for (s, i) in results if i == '1h'] or list(results.keys())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                   gridspec_kw={'height_ratios': [2, 1]})

    # Coverage timeline: one row per series, green span = data, red tick = gap.
    labels = []
    for y, (key) in enumerate(syms):
        a = results[key]
        rt = a['_ret']
        if rt.empty:
            continue
        start, end = rt.index[0], rt.index[-1]
        ax1.hlines(y, start, end, color='seagreen', lw=6, alpha=0.7)
        for ts, n in a['top_gaps']:
            ax1.plot(ts, y, marker='|', color='crimson', ms=14, mew=2)
        labels.append(f"{key[0]} {key[1]}")
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels)
    ax1.set_title('Data coverage timeline  (green = data, red | = largest gaps)')
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.grid(True, axis='x', alpha=0.3)

    # Return distributions (1h series), log y to expose fat tails.
    for key in syms:
        rt = results[key]['_ret']
        if not rt.empty:
            ax2.hist(rt.values, bins=200, histtype='step', alpha=0.6, label=key[0])
    ax2.set_yscale('log')
    ax2.set_title('Per-bar return distributions (log count — inspect tails)')
    ax2.set_xlabel('return'); ax2.legend(ncol=5, fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"\n  Chart saved to {out}")


def main():
    ap = argparse.ArgumentParser(description='Crypto data quality audit')
    ap.add_argument('--glob', default='Data/*_binance.csv')
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"No files match {args.glob}")
        return

    results = {}
    for p in paths:
        sym, interval = parse_name(p)
        if sym is None or interval not in INTERVAL_SECONDS:
            continue
        a = audit(load(p), interval)
        results[(sym, interval)] = a
        print_report(sym, interval, a)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print(f"  {'SYMBOL':<8}{'ITV':<5}{'BARS':>9}{'COVER':>8}{'GAPS':>6}"
          f"{'OHLC':>6}{'SPIKE':>7}{'FLAT':>6}{'VOL%':>8}  RANGE")
    print("=" * 92)
    for (sym, interval), a in results.items():
        rt = a['_ret']
        rng = f"{rt.index[0]:%Y-%m}→{rt.index[-1]:%Y-%m}" if not rt.empty else "n/a"
        print(f"  {sym:<8}{interval:<5}{a['bars']:>9,}{a['coverage_pct']:>7.1f}%"
              f"{a['n_gaps']:>6}{a['ohlc_violations']:>6}{a['spike_prints']:>7}"
              f"{a['max_flat_run']:>6}{a['daily_vol_pct']:>7.1f}%  {rng}")
    print("=" * 92)

    plot(results)


if __name__ == '__main__':
    main()
