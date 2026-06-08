#!/usr/bin/env python3
"""
Cross-sectional crypto mean-reversion / momentum
──────────────────────────────────────────────────
Time-series MR is the wrong side at hourly bars (these coins trend). This tests
the OTHER form of MR: cross-sectional. Each bar, rank the coins by recent
vol-adjusted return, demean across the basket (→ dollar-neutral), and bet:

  MR  : long the laggards, short the leaders   (short-term reversal)
  MOM : long the leaders, short the laggards   (cross-sectional momentum)

Demeaning removes the whole market's beta, so this is immune to "everything
trended in the 2017-21 bull" — it harvests coins moving relative to the basket.
Being long/short, gross exposure is fixed at 1.0 so Sharpe reflects the pure
edge (leverage/vol-targeting is a separate later step).

Clean data-prep (gap/zero-range/spike masking) and vol estimation are reused
from crypto_mr_backtest. Cost = 0.2% round-trip on turnover; gross vs net shown.
No-lookahead: weights from info through bar t earn bar t+1's return.

Usage:
  python3 portfolio/crypto_xsec_mr.py                 # diagnostic grid, 1h
  python3 portfolio/crypto_xsec_mr.py --interval 1d
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Robust import of the sibling module regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crypto.crypto_mr_backtest import (load_clean, per_bar_vol, FNAME_RE,
                                BARS_PER_YEAR_MAP, COST_ONE_WAY, COST_ROUND_TRIP)

VOL_SPAN = 72


# ── Build aligned panel ──────────────────────────────────────────────────────

def build_panel(data, interval):
    """Aligned (close, ret, vol) DataFrames — columns = coins, index = union time."""
    idx = sorted(set().union(*[set(df.index) for df in data.values()]))
    idx = pd.DatetimeIndex(idx)
    R, V = {}, {}
    for sym, df in data.items():
        d = df.reindex(idx)
        R[sym] = d['ret']
        V[sym] = per_bar_vol(d['ret'], VOL_SPAN)
    return pd.DataFrame(R), pd.DataFrame(V)


# ── Cross-sectional weights ──────────────────────────────────────────────────

def xsec_weights(R, V, lookback, direction):
    """
    direction = -1 → mean-reversion (fade), +1 → momentum (follow).
    Score = recent vol-adjusted cumulative return; demeaned across coins each
    bar; normalised to gross exposure 1.0 (dollar-neutral by construction).
    """
    score = R.fillna(0.0).rolling(lookback, min_periods=lookback).sum()
    z = score / (V * np.sqrt(lookback))      # vol-adjust so DOGE doesn't dominate
    z = z.where(R.notna())                    # only rank coins live at t

    demeaned = z.sub(z.mean(axis=1), axis=0)  # remove market beta
    w = (direction * demeaned).where(z.notna(), 0.0)

    gross = w.abs().sum(axis=1)
    w = w.div(gross.replace(0, np.nan), axis=0).fillna(0.0)   # gross = 1.0
    return w


def compute(R, V, lookback, direction):
    """Return (weights, gross_pnl_series, turnover_series) — no cost applied."""
    w = xsec_weights(R, V, lookback, direction)
    ret = R.fillna(0.0)
    pnl_gross = (w.shift(1) * ret).sum(axis=1)
    turnover = (w - w.shift(1)).abs().sum(axis=1)
    return w, pnl_gross, turnover


def run(R, V, lookback, direction, interval, cost_one_way=COST_ONE_WAY):
    w, pnl_gross, turnover = compute(R, V, lookback, direction)
    pnl_net = pnl_gross - turnover * cost_one_way
    return ((1 + pnl_net).cumprod(), (1 + pnl_gross).cumprod(),
            turnover.mean() * BARS_PER_YEAR_MAP[interval])


def sharpe(equity, interval):
    r = equity.pct_change().dropna()
    bpy = BARS_PER_YEAR_MAP[interval]
    return r.mean() / r.std() * np.sqrt(bpy) if not r.empty and r.std() > 0 else float('nan')


def pnl_metrics(pnl, interval):
    """Annualised stats from a per-bar return series."""
    bpy = BARS_PER_YEAR_MAP[interval]
    r = pnl.dropna()
    if r.empty or r.std() == 0:
        return {'sharpe': float('nan'), 'ann_ret': float('nan'),
                'ann_vol': float('nan'), 'maxdd': float('nan')}
    eq = (1 + r).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.25
    ann_ret = (eq.iloc[-1] ** (1 / years) - 1) * 100 if years > 0 else float('nan')
    maxdd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    return {'sharpe': r.mean() / r.std() * np.sqrt(bpy),
            'ann_ret': ann_ret, 'ann_vol': r.std() * np.sqrt(bpy) * 100, 'maxdd': maxdd}


def validate_momentum(R, V, lb, interval):
    """Battery of robustness checks for the cross-sectional momentum signal."""
    bpy = BARS_PER_YEAR_MAP[interval]
    cut = R.index[int(len(R) * 0.70)]
    w, pnl_gross, turnover = compute(R, V, lb, +1)
    pnl_net = pnl_gross - turnover * COST_ONE_WAY

    print(f"\n{'='*70}\n  CROSS-SECTIONAL MOMENTUM — validation  (lookback {lb}, {interval})\n{'='*70}")

    # 1) In-sample vs OOS
    is_m = pnl_metrics(pnl_net[pnl_net.index < cut], interval)
    oos_m = pnl_metrics(pnl_net[pnl_net.index >= cut], interval)
    full_m = pnl_metrics(pnl_net, interval)
    print(f"\n  1) NET performance (after 0.2% RT)")
    print(f"     {'period':<14}{'Sharpe':>8}{'AnnRet%':>9}{'Vol%':>7}{'MaxDD%':>8}")
    for label, m in [('full', full_m), (f'in-samp', is_m), ('OOS (last30%)', oos_m)]:
        print(f"     {label:<14}{m['sharpe']:>8.2f}{m['ann_ret']:>9.1f}"
              f"{m['ann_vol']:>7.1f}{m['maxdd']:>8.1f}")

    # 2) Per-calendar-year net Sharpe (regime stability)
    print(f"\n  2) NET Sharpe by calendar year (does it survive outside the bull?)")
    by_year = pnl_net.groupby(pnl_net.index.year)
    row = "     "
    for yr, r in by_year:
        s = r.mean() / r.std() * np.sqrt(bpy) if r.std() > 0 else float('nan')
        row += f"{yr}:{s:>5.1f}  "
    print(row)

    # 3) Cost sensitivity
    print(f"\n  3) Cost sensitivity (full-period net Sharpe)")
    print("     " + "  ".join(f"{c*100:.1f}%RT:{sharpe((1+(pnl_gross-turnover*c/2)).cumprod(), interval):>5.2f}"
                              for c in [0.0, 0.001, 0.002, 0.003, 0.005]))

    # 4) Spike robustness handled by caller (re-run with --exclude-spikes)
    # 5) Long vs short leg (is it just long beta?)
    ret = R.fillna(0.0)
    long_pnl  = (w.clip(lower=0).shift(1) * ret).sum(axis=1)
    short_pnl = (w.clip(upper=0).shift(1) * ret).sum(axis=1)
    lm, sm = pnl_metrics(long_pnl, interval), pnl_metrics(short_pnl, interval)
    print(f"\n  4) Leg decomposition (gross, no cost)")
    print(f"     long  leg  Sharpe {lm['sharpe']:>5.2f}   AnnRet {lm['ann_ret']:>6.1f}%")
    print(f"     short leg  Sharpe {sm['sharpe']:>5.2f}   AnnRet {sm['ann_ret']:>6.1f}%")
    print(f"\n     turnover {turnover.mean()*bpy:.0f}/yr   "
          f"avg #live coins {R.notna().sum(axis=1).mean():.1f}")

    # plot
    plt.figure(figsize=(14, 6))
    plt.plot(pnl_net.index, (1 + pnl_net).cumprod(), lw=1.3, color='seagreen', label='net')
    plt.plot(pnl_gross.index, (1 + pnl_gross).cumprod(), lw=0.8, color='gray',
             alpha=0.6, label='gross')
    plt.axvline(cut, color='crimson', ls='--', lw=1, label='OOS start')
    plt.yscale('log'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.title(f"X-sectional momentum  lb={lb} {interval}  net {COST_ROUND_TRIP*100:.1f}%RT")
    plt.ylabel('growth of $1 (log)')
    os.makedirs('results', exist_ok=True)
    out = f"results/crypto_xsec_mom_validate_{interval}.png"
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"\n  Chart saved to {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Cross-sectional crypto MR / momentum')
    ap.add_argument('--interval', default='1h')
    ap.add_argument('--lookbacks', nargs='+', type=int, default=[1, 3, 6, 12, 24, 48])
    ap.add_argument('--exclude-spikes', action='store_true')
    ap.add_argument('--validate', action='store_true',
                    help='Run robustness battery on momentum at --lb')
    ap.add_argument('--lb', type=int, default=10, help='lookback for --validate')
    args = ap.parse_args()

    paths = sorted(glob.glob(f"Data/*_{args.interval}_binance.csv"))
    if not paths:
        print(f"No files for interval {args.interval}")
        return

    data = {}
    for p in paths:
        m = FNAME_RE.search(os.path.basename(p))
        if m:
            data[m.group(1).upper()] = load_clean(p, args.interval, args.exclude_spikes)
    print(f"Loaded {len(data)} coins [{args.interval}]  "
          f"spikes={'excluded' if args.exclude_spikes else 'kept'}")

    R, V = build_panel(data, args.interval)

    if args.validate:
        validate_momentum(R, V, args.lb, args.interval)
        return

    # Out-of-sample holdout (last 30%).
    cut = R.index[int(len(R) * 0.70)]
    print(f"OOS holdout starts {cut:%Y-%m-%d}\n")

    print("=" * 86)
    print(f"  {'LOOKBACK':<9}{'MR_gross':>10}{'MR_net':>9}{'MR_OOSnet':>11}"
          f"{'MOM_gross':>11}{'MOM_net':>9}{'turn/yr':>9}")
    print("=" * 86)

    best = None
    for lb in args.lookbacks:
        mr_net, mr_gross, turn = run(R, V, lb, -1, args.interval)
        mo_net, mo_gross, _    = run(R, V, lb, +1, args.interval)
        # OOS for MR
        Ro, Vo = R[R.index >= cut], V[V.index >= cut]
        mr_oos, _, _ = run(Ro, Vo, lb, -1, args.interval)
        s_oos = sharpe(mr_oos, args.interval)
        print(f"  {lb:<9}{sharpe(mr_gross, args.interval):>10.2f}"
              f"{sharpe(mr_net, args.interval):>9.2f}{s_oos:>11.2f}"
              f"{sharpe(mo_gross, args.interval):>11.2f}{sharpe(mo_net, args.interval):>9.2f}"
              f"{turn:>9.0f}", flush=True)
        if best is None or s_oos > best[1]:
            best = (lb, s_oos, mr_net)
    print("=" * 86)
    print("  Dollar-neutral, gross=1.0. MR=long laggards/short leaders. Net = 0.2% RT.")
    print("  Trust MR_OOSnet. Positive there = a real cross-sectional reversal edge.")

    if best:
        lb, _, eq = best
        plt.figure(figsize=(14, 6))
        plt.plot(eq.index, eq.values, lw=1.2, color='seagreen')
        plt.axvline(cut, color='crimson', ls='--', lw=1, label='OOS start')
        plt.title(f"Cross-sectional MR  |  lookback {lb}  |  {args.interval}  |  "
                  f"net {COST_ROUND_TRIP*100:.1f}% RT  |  growth of $1")
        plt.ylabel('equity (×)'); plt.legend(); plt.grid(True, alpha=0.3)
        os.makedirs('results', exist_ok=True)
        out = f"results/crypto_xsec_mr_{args.interval}.png"
        plt.tight_layout(); plt.savefig(out, dpi=150)
        print(f"\n  Chart saved to {out}")


if __name__ == '__main__':
    main()
