#!/usr/bin/env python3
"""
Crypto intraday mean-reversion research harness
─────────────────────────────────────────────────
First-pass backtest for short-horizon MR on the Binance majors. Clean data-prep
is baked in by construction (per the data-quality audit):

  • gap-spanning returns are masked     (no fake reversion across exchange outages)
  • zero-range bars are masked          (IBS-style signals divide by their range)
  • round-trip spike bars optionally     (--exclude-spikes: don't let the strategy
    excluded                              profit from bad prints snapping back)

Every instrument is vol-normalized (DOGE runs ~10×/day BTC's vol — equal notional
would be a DOGE bet). Positions are continuous (crypto trades fractional), so the
forecast actually expresses itself. Cost = 0.2% round-trip on turnover; results
are shown GROSS and NET because fee drag is what kills intraday MR.

Two starter signals (compare with --signal):
  zscore : fade deviation of price from its EWMA, in units of price vol (Bollinger-ish)
  ibs    : crypto-adapted Internal Bar Strength — fade (close-low)/(high-low)

No-lookahead: signal uses info through close of bar t; it earns bar t+1's return
(positions are shift(1)'d). An out-of-sample holdout (last 30%) is reported
separately — trust that number, not the in-sample one.

Usage:
  python3 portfolio/crypto_mr_backtest.py                      # zscore grid, 1h
  python3 portfolio/crypto_mr_backtest.py --signal ibs
  python3 portfolio/crypto_mr_backtest.py --exclude-spikes
  python3 portfolio/crypto_mr_backtest.py --interval 1d
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────
COST_ROUND_TRIP = 0.002          # 0.2% of notional, round trip
COST_ONE_WAY    = COST_ROUND_TRIP / 2
RISK_TARGET     = 0.20           # portfolio annualised vol target
VOL_SPAN        = 72             # bars, EWMA vol lookback (~3 days at 1h)
ENTRY_Z         = 2.0            # z at which the signal saturates to full size
SPIKE_THR       = {'1h': 0.15, '4h': 0.25, '1d': 0.40}

INTERVAL_SECONDS  = {'1m': 60, '5m': 300, '15m': 900, '30m': 1800,
                     '1h': 3600, '4h': 14400, '1d': 86400}
BARS_PER_YEAR_MAP = {'1m': 365*24*60, '5m': 365*24*12, '15m': 365*24*4,
                     '30m': 365*24*2, '1h': 365*24, '4h': 365*6, '1d': 365}

FNAME_RE = re.compile(r'([a-z0-9]+)_([0-9]+[mhd])_binance\.csv$')


# ── Clean data prep ──────────────────────────────────────────────────────────

def load_clean(path, interval, exclude_spikes=False):
    """
    Load one series and return a DataFrame with:
      open, high, low, close, ret (cleaned), valid (bool)
    'ret' has gap-spanning, zero-range, and (optionally) spike bars set to NaN so
    no downstream calculation can harvest them.
    """
    step = INTERVAL_SECONDS[interval]
    df = pd.read_csv(path, parse_dates=['time']).set_index('time').sort_index()

    ret = df['close'].pct_change()

    # 1) mask returns that span a gap (not a real single-bar move)
    dt = df.index.to_series().diff().dt.total_seconds()
    gap = dt > step * 1.5
    ret = ret.mask(gap)

    # 2) mask zero-range bars (degenerate for range-based signals)
    zero_range = (df['high'] - df['low']) == 0
    ret = ret.mask(zero_range)

    # 3) optionally mask round-trip spike bars (likely bad prints)
    thr = SPIKE_THR.get(interval, 0.40)
    if exclude_spikes:
        nxt = ret.shift(-1)
        spike = ((ret.abs() > thr) & (np.sign(ret) != np.sign(nxt)) &
                 (nxt.abs() > 0.6 * ret.abs())).fillna(False).astype(bool)
        ret = ret.mask(spike)
        ret = ret.mask(spike.shift(1, fill_value=False))   # and its reversal bar

    df['ret'] = ret
    df['valid'] = ret.notna()
    return df


def per_bar_vol(ret, span=VOL_SPAN):
    return ret.ewm(span=span, min_periods=span).std()


# ── Signals (return a forecast in [-1, 1]; +1 = max long) ────────────────────

def signal_zscore(df, lookback):
    """Fade price's deviation from its EWMA, measured in price-vol units."""
    close = df['close']
    ma = close.ewm(span=lookback, min_periods=lookback).mean()
    sd = close.rolling(lookback, min_periods=lookback).std()
    z = (close - ma) / sd
    fc = (-z / ENTRY_Z).clip(-1, 1)        # high price → short, low price → long
    return fc.where(df['valid'])


def signal_ibs(df, lookback=None):
    """Crypto-adapted IBS: fade (close-low)/(high-low). lookback unused."""
    rng = df['high'] - df['low']
    ibs = ((df['close'] - df['low']) / rng).where(rng > 0)
    fc = ((0.5 - ibs) * 2).clip(-1, 1)     # ibs low → long, ibs high → short
    return fc.where(df['valid'])


SIGNALS = {'zscore': signal_zscore, 'ibs': signal_ibs}


# ── Backtest (weight-space, vectorised, turnover-costed) ─────────────────────

def backtest(data, signal_fn, lookback, interval):
    """
    Equal-risk portfolio over the union of timestamps. Each instrument gets a
    target weight = forecast × (RISK_TARGET / n) / ann_vol, rebalanced every bar.
    Returns (net_equity, gross_equity, stats_dict).
    """
    bpy = BARS_PER_YEAR_MAP[interval]
    idx = sorted(set().union(*[set(df.index) for df in data.values()]))
    idx = pd.DatetimeIndex(idx)
    n = len(data)

    weights, rets = {}, {}
    for sym, df in data.items():
        d = df.reindex(idx)
        fc = signal_fn(d, lookback)
        vol = per_bar_vol(d['ret']).reindex(idx)
        ann_vol = vol * np.sqrt(bpy)
        w = (fc * (RISK_TARGET / n) / ann_vol)
        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-2, 2)  # leverage cap/instr
        weights[sym] = w
        rets[sym] = d['ret'].fillna(0.0)

    W = pd.DataFrame(weights).fillna(0.0)
    R = pd.DataFrame(rets).fillna(0.0)

    pnl_gross = (W.shift(1) * R).sum(axis=1)
    turnover = (W - W.shift(1)).abs().sum(axis=1)
    cost = turnover * COST_ONE_WAY
    pnl_net = pnl_gross - cost

    gross_eq = (1 + pnl_gross).cumprod()
    net_eq = (1 + pnl_net).cumprod()

    stats = {
        'ann_turnover': turnover.mean() * bpy,
        'avg_gross_leverage': W.abs().sum(axis=1).mean(),
        'pnl_net': pnl_net,
    }
    return net_eq, gross_eq, stats


# ── Metrics ──────────────────────────────────────────────────────────────────

def metrics(equity, interval):
    bpy = BARS_PER_YEAR_MAP[interval]
    rets = equity.pct_change().dropna()
    if rets.empty or equity.iloc[-1] <= 0:
        return {}
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    ann = (equity.iloc[-1] ** (1 / years) - 1) * 100 if years > 0 else np.nan
    std = rets.std()
    sharpe = rets.mean() / std * np.sqrt(bpy) if std > 0 else np.nan
    peak = equity.cummax()
    maxdd = ((equity - peak) / peak).min() * 100
    return {'ann_ret': ann, 'ann_vol': std * np.sqrt(bpy) * 100,
            'sharpe': sharpe, 'maxdd': maxdd}


def _sharpe(equity, interval):
    bpy = BARS_PER_YEAR_MAP[interval]
    r = equity.pct_change().dropna()
    return r.mean() / r.std() * np.sqrt(bpy) if not r.empty and r.std() > 0 else float('nan')


def run_diagnostic(data, signal_fn, interval, lookbacks):
    """
    Disentangle direction (MR vs momentum) from cost (gross vs net) so we know
    whether to flip the signal, slow it down, or abandon time-series MR. Prints
    each row as it finishes — never silent.
    """
    print("\n" + "=" * 78)
    print(f"  {'LOOKBACK':<9}{'MR_gross':>10}{'MR_net':>9}{'MOM_gross':>11}"
          f"{'MOM_net':>9}{'turn/yr':>9}")
    print("=" * 78)
    flip = lambda df, lb, _o=signal_fn: -_o(df, lb)
    for lb in lookbacks:
        import sys
        net,  gross,  st = backtest(data, signal_fn, lb, interval)
        mnet, mgross, _  = backtest(data, flip,      lb, interval)
        print(f"  {lb:<9}{_sharpe(gross, interval):>10.2f}{_sharpe(net, interval):>9.2f}"
              f"{_sharpe(mgross, interval):>11.2f}{_sharpe(mnet, interval):>9.2f}"
              f"{st['ann_turnover']:>9.0f}", flush=True)
    print("=" * 78)
    print("  MR = fade deviation, MOM = follow it. gross ignores cost; net = 0.2% RT.")
    print("  Positive MOM_gross ⇒ crypto trends at this horizon. Net<<gross ⇒ fee-fragile.")


def split_holdout(data, frac=0.30):
    """Split each series at the global (1-frac) timestamp → (train, test) dicts."""
    idx = sorted(set().union(*[set(df.index) for df in data.values()]))
    cut = idx[int(len(idx) * (1 - frac))]
    train = {s: df[df.index < cut] for s, df in data.items()}
    test  = {s: df[df.index >= cut] for s, df in data.items()}
    return train, test, cut


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Crypto intraday MR backtest')
    ap.add_argument('--signal', choices=SIGNALS, default='zscore')
    ap.add_argument('--interval', default='1h')
    ap.add_argument('--lookbacks', nargs='+', type=int, default=[12, 24, 48, 96])
    ap.add_argument('--exclude-spikes', action='store_true')
    ap.add_argument('--diagnostic', action='store_true',
                    help='MR vs momentum, gross vs net, across lookbacks')
    ap.add_argument('--glob', default=None)
    args = ap.parse_args()

    patt = args.glob or f"Data/*_{args.interval}_binance.csv"
    paths = sorted(glob.glob(patt))
    if not paths:
        print(f"No files match {patt}")
        return

    data = {}
    for p in paths:
        m = FNAME_RE.search(os.path.basename(p))
        if m:
            data[m.group(1).upper()] = load_clean(p, args.interval, args.exclude_spikes)
    print(f"Loaded {len(data)} instruments [{args.interval}]  signal={args.signal}"
          f"  spikes={'excluded' if args.exclude_spikes else 'kept'}")

    signal_fn = SIGNALS[args.signal]
    lookbacks = args.lookbacks if args.signal == 'zscore' else [0]

    if args.diagnostic:
        run_diagnostic(data, signal_fn, args.interval, lookbacks)
        return

    train, test, cut = split_holdout(data)
    print(f"OOS holdout starts {cut:%Y-%m-%d} (last 30%)\n")

    print("=" * 96)
    print(f"  {'LOOKBACK':<9}{'│':<2}{'FULL  Sharpe':>13}{'AnnRet%':>9}{'Vol%':>7}{'MaxDD%':>8}"
          f"{'  │':<3}{'OOS Sharpe':>11}{'AnnRet%':>9}{'MaxDD%':>8}{'  │':<3}{'Turn/yr':>8}")
    print("=" * 96)

    best = None
    for lb in lookbacks:
        ne_full, _, st_full = backtest(data, signal_fn, lb, args.interval)
        ne_oos,  _, _        = backtest(test, signal_fn, lb, args.interval)
        mf, mo = metrics(ne_full, args.interval), metrics(ne_oos, args.interval)
        if not mf or not mo:
            continue
        print(f"  {lb:<9}{'│':<2}{mf['sharpe']:>13.2f}{mf['ann_ret']:>9.1f}{mf['ann_vol']:>7.1f}"
              f"{mf['maxdd']:>8.1f}{'  │':<3}{mo['sharpe']:>11.2f}{mo['ann_ret']:>9.1f}"
              f"{mo['maxdd']:>8.1f}{'  │':<3}{st_full['ann_turnover']:>8.0f}")
        if best is None or mo['sharpe'] > best[1]:
            best = (lb, mo['sharpe'], ne_full)
    print("=" * 96)
    print("  Net of 0.2% round-trip cost. Trust the OOS columns. High Turn/yr = fee-fragile.")

    if best:
        lb, _, eq = best
        plt.figure(figsize=(14, 6))
        plt.plot(eq.index, eq.values, lw=1.2, color='steelblue')
        plt.axvline(cut, color='crimson', ls='--', lw=1, label='OOS start')
        plt.yscale('log')
        plt.title(f"Crypto MR  |  {args.signal} (lookback {lb})  |  {args.interval}  |  "
                  f"net of {COST_ROUND_TRIP*100:.1f}% RT  |  growth of $1 (log)")
        plt.ylabel('equity (×)'); plt.legend(); plt.grid(True, alpha=0.3)
        os.makedirs('results', exist_ok=True)
        out = f"results/crypto_mr_{args.signal}_{args.interval}.png"
        plt.tight_layout(); plt.savefig(out, dpi=150)
        print(f"\n  Chart saved to {out}")


if __name__ == '__main__':
    main()
