#!/usr/bin/env python3
"""
Crypto time-series trend following (EWMAC)
────────────────────────────────────────────
Every prior test converged here: trend, long-biased, daily+ horizon is the only
repeatable cost-survivable signal in the majors. This builds the Carver-style
continuous EWMAC forecast per coin and compares three position modes off the
SAME forecast (so the only difference is how we treat shorts, which bled in
every earlier test):

  longshort  : full long and short            (classic EWMAC)
  longbias   : full long, shorts capped small (SHORT_CAP × size)
  longflat   : long when trend up, else flat

Continuous vol-targeted positions (crypto fractional sizing lets the forecast
express itself — the thing futures granularity prevented). Realistic Coinbase
SPOT cost: 0.40%/side default (0.35% maker fee + 0.05% slippage, Intro tier);
gross vs net; OOS = last 30%; per-year Sharpe for regime honesty.
Data prep / vol reused from crypto_mr_backtest.

Usage:
  python3 portfolio/crypto_trend.py                       # ensemble, all 3 modes
  python3 portfolio/crypto_trend.py --speeds 16 64        # single EWMAC pair
  python3 portfolio/crypto_trend.py --exclude-spikes
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crypto_mr_backtest import load_clean, per_bar_vol, FNAME_RE, BARS_PER_YEAR_MAP

RISK_TARGET   = 0.20
VOL_SPAN      = 32
FORECAST_TARGET = 10.0
FORECAST_CAP    = 20.0
SHORT_CAP       = 0.30        # longbias mode: shorts sized at 30% of full
# Carver daily forecast scalars + 3-speed ensemble FDM.
SCALARS = {(8, 32): 5.95, (16, 64): 4.10, (32, 128): 2.79, (64, 256): 1.91}
FDM = 1.25

CAPITAL = 5_000              # starting capital ($); spot is scale-invariant in %

# ── Cost model: Coinbase SPOT at $5k / Intro tier (<$10k monthly volume) ──────
# Per-side cost = exchange fee + slippage, charged on traded notional (turnover).
# Spot chosen over nano futures: at $5k the nano contract notional (~$750 BTC)
# exceeds the vol-targeted positions, so futures can't size the strategy at all.
FEE_MAKER = 0.0035          # Coinbase Advanced maker, Intro-2 tier
FEE_TAKER = 0.0075          # taker (market orders), same tier
SLIP_SIDE = 0.0005          # ~5bps spread/slippage on liquid majors at $5k size
COST_PER_SIDE  = FEE_MAKER + SLIP_SIDE      # default: limit/maker orders = 0.40%/side
COST_ONE_WAY   = COST_PER_SIDE              # turnover is charged per side
COST_ROUND_TRIP = COST_PER_SIDE * 2


def build_panel(data, interval):
    idx = pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in data.values()])))
    C, R, V = {}, {}, {}
    for sym, df in data.items():
        d = df.reindex(idx)
        C[sym] = d['close']
        R[sym] = d['ret']
        V[sym] = per_bar_vol(d['ret'], VOL_SPAN)
    return pd.DataFrame(C), pd.DataFrame(R), pd.DataFrame(V)


def ewmac_forecast_one(close, vol, fast, slow, scalar):
    raw = (close.ewm(span=fast, min_periods=fast).mean()
           - close.ewm(span=slow, min_periods=slow).mean())
    norm = raw / (close * vol)                      # vol/price normalised
    return (norm * scalar).clip(-FORECAST_CAP, FORECAST_CAP)


def forecast_panel(C, V, speeds):
    """Per-coin forecast: single speed, or FDM-scaled ensemble if speeds is None."""
    out = {}
    for sym in C.columns:
        close, vol = C[sym], V[sym]
        if speeds is not None:
            f, s = speeds
            fc = ewmac_forecast_one(close, vol, f, s, SCALARS.get((f, s), 1.0))
        else:
            parts = [ewmac_forecast_one(close, vol, f, s, sc) for (f, s), sc in SCALARS.items()
                     if s <= 128]                    # 3-speed ensemble (8/16/32 fast)
            fc = (pd.concat(parts, axis=1).mean(axis=1) * FDM).clip(-FORECAST_CAP, FORECAST_CAP)
        out[sym] = fc.where(C[sym].notna())
    return pd.DataFrame(out)


def weights(F, V, interval, mode):
    """Vol-targeted weights from forecast F. n = # live coins each bar."""
    bpy = BARS_PER_YEAR_MAP[interval]
    ann_vol = V * np.sqrt(bpy)
    n = F.notna().sum(axis=1).replace(0, np.nan)
    # per-coin weight = (forecast/10) × (risk_target / n_live_coins) / ann_vol
    w = (F / FORECAST_TARGET).div(n, axis=0).mul(RISK_TARGET) / ann_vol
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if mode == 'longflat':
        w = w.clip(lower=0)
    elif mode == 'longbias':
        w = w.where(w >= 0, w * SHORT_CAP)
    w = w.clip(-2, 2)                                # per-coin leverage cap
    return w


def apply_buffer(W, band):
    """
    Carver buffering: hold the current position while the target stays within
    `band` (fraction of capital) of it; when the target drifts further, trade
    only to the NEAR EDGE of the band, not all the way to target. Suppresses
    daily dust trades. Sequential in time (held_t depends on held_{t-1}).
    """
    if band <= 0:
        return W
    tgt = np.nan_to_num(W.to_numpy(), nan=0.0)
    held = np.empty_like(tgt)
    prev = np.zeros(tgt.shape[1])
    for t in range(tgt.shape[0]):
        diff = tgt[t] - prev
        over = np.abs(diff) > band
        prev = np.where(over, tgt[t] - np.sign(diff) * band, prev)
        held[t] = prev
    return pd.DataFrame(held, index=W.index, columns=W.columns)


def backtest(F, R, V, interval, mode, cost=COST_ONE_WAY, band=0.0):
    w = weights(F, V, interval, mode)
    w = apply_buffer(w, band)                 # actual held position after buffering
    ret = R.fillna(0.0)
    pnl_gross = (w.shift(1) * ret).sum(axis=1)
    turnover = (w - w.shift(1)).abs().sum(axis=1)
    pnl_net = pnl_gross - turnover * cost
    return pnl_net, pnl_gross, turnover.mean() * BARS_PER_YEAR_MAP[interval]


def metrics(pnl, interval):
    bpy = BARS_PER_YEAR_MAP[interval]
    r = pnl.dropna()
    if r.empty or r.std() == 0:
        return dict(sharpe=np.nan, ann_ret=np.nan, ann_vol=np.nan, maxdd=np.nan)
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    return dict(sharpe=r.mean() / r.std() * np.sqrt(bpy),
                ann_ret=(eq.iloc[-1] ** (1 / yrs) - 1) * 100 if yrs > 0 else np.nan,
                ann_vol=r.std() * np.sqrt(bpy) * 100,
                maxdd=((eq - eq.cummax()) / eq.cummax()).min() * 100)


def attribution(C, R, V, F, interval, mode):
    """
    Decompose the strategy's return:
      • per instrument — each coin's additive contribution to portfolio return,
        its standalone Sharpe, share of total PnL, time-in-market, avg weight
      • per EWMAC speed — each speed run standalone through the same pipeline
    Per-coin contributions are arithmetic (sum of daily contributions) so they
    add up to the portfolio total. Per-speed rows are standalone Sharpes and do
    NOT sum (the long/flat clip + FDM are non-linear) — read them as 'how much
    does each speed pull on its own'.
    """
    bpy = BARS_PER_YEAR_MAP[interval]
    w = weights(F, V, interval, mode)
    ret = R.fillna(0.0)
    contrib = w.shift(1) * ret                              # per-coin daily PnL
    turn = (w - w.shift(1)).abs()
    net_i = contrib - turn * COST_ONE_WAY                   # per-coin net of cost
    total = net_i.sum().sum()

    print("\n" + "=" * 78)
    print(f"  RETURN ATTRIBUTION  (mode={mode})")
    print("=" * 78)
    print(f"\n  By instrument  (contribution adds to portfolio total)")
    print(f"  {'coin':<10}{'Contrib%':>10}{'Share%':>8}{'Sharpe':>8}"
          f"{'TimeInMkt%':>12}{'AvgWt':>8}")
    rows = []
    for c in net_i.columns:
        s = net_i[c]
        rows.append((c, s.sum() * 100, 100 * s.sum() / total if total else np.nan,
                     s.mean() / s.std() * np.sqrt(bpy) if s.std() > 0 else np.nan,
                     (w[c] != 0).mean() * 100, w[c].abs().mean()))
    for c, contr, share, shrp, tim, awt in sorted(rows, key=lambda x: -x[1]):
        print(f"  {c:<10}{contr:>10.1f}{share:>8.1f}{shrp:>8.2f}{tim:>12.1f}{awt:>8.3f}")

    print(f"\n  By EWMAC speed  (standalone long/flat — do NOT sum; FDM/clip non-linear)")
    print(f"  {'speed':<12}{'Sharpe':>8}{'AnnRet%':>9}{'Vol%':>7}{'MaxDD%':>8}{'turn/yr':>9}")
    for (f, s) in [(k) for k in SCALARS if k[1] <= 128]:
        Fs = forecast_panel(C, V, (f, s))
        net, _, turn = backtest(Fs, R, V, interval, mode)
        m = metrics(net, interval)
        print(f"  ({f},{s}){'':<{7-len(str(f))-len(str(s))}}{m['sharpe']:>8.2f}"
              f"{m['ann_ret']:>9.1f}{m['ann_vol']:>7.1f}{m['maxdd']:>8.1f}{turn:>9.0f}")
    net, _, turn = backtest(F, R, V, interval, mode)
    m = metrics(net, interval)
    print(f"  {'ensemble':<12}{m['sharpe']:>8.2f}{m['ann_ret']:>9.1f}{m['ann_vol']:>7.1f}"
          f"{m['maxdd']:>8.1f}{turn:>9.0f}")
    print(f"\n  → faster speeds dominate on crypto; ensemble trades a little in-sample"
          f"\n    Sharpe for robustness (not betting the book on one speed surviving OOS).")


def main():
    ap = argparse.ArgumentParser(description='Crypto EWMAC trend following')
    ap.add_argument('--interval', default='1d')
    ap.add_argument('--speeds', nargs=2, type=int, default=None,
                    help='single EWMAC pair, e.g. --speeds 16 64; omit for ensemble')
    ap.add_argument('--exclude-spikes', action='store_true')
    ap.add_argument('--attribution', action='store_true',
                    help='per-instrument and per-speed return decomposition')
    ap.add_argument('--attr-mode', default='longflat',
                    choices=['longshort', 'longbias', 'longflat'],
                    help='position mode for attribution (default longflat)')
    args = ap.parse_args()

    paths = sorted(glob.glob(f"Data/*_{args.interval}_binance.csv"))
    data = {}
    for p in paths:
        m = FNAME_RE.search(os.path.basename(p))
        if m:
            data[m.group(1).upper()] = load_clean(p, args.interval, args.exclude_spikes)
    speeds = tuple(args.speeds) if args.speeds else None
    print(f"Loaded {len(data)} coins [{args.interval}]  "
          f"signal={'ensemble' if speeds is None else f'EWMAC{speeds}'}  "
          f"spikes={'excluded' if args.exclude_spikes else 'kept'}")

    C, R, V = build_panel(data, args.interval)
    F = forecast_panel(C, V, speeds)
    cut = R.index[int(len(R) * 0.70)]
    print(f"OOS holdout starts {cut:%Y-%m-%d}\n")

    print(f"Cost: {COST_PER_SIDE*100:.2f}%/side spot "
          f"({FEE_MAKER*100:.2f}% maker fee + {SLIP_SIDE*100:.2f}% slippage)  |  "
          f"capital ${CAPITAL:,}\n")

    print("=" * 92)
    print(f"  {'MODE':<11}{'FULL Sharpe':>12}{'AnnRet%':>9}{'Vol%':>7}{'MaxDD%':>8}"
          f"{'  │':<3}{'OOS Sharpe':>11}{'AnnRet%':>9}{'MaxDD%':>8}{'  │':<3}{'turn/yr':>8}")
    print("=" * 92)
    curves = {}
    for mode in ['longshort', 'longbias', 'longflat']:
        net, gross, turn = backtest(F, R, V, args.interval, mode)
        mf = metrics(net, args.interval)
        mo = metrics(net[net.index >= cut], args.interval)
        curves[mode] = (1 + net).cumprod()
        print(f"  {mode:<11}{mf['sharpe']:>12.2f}{mf['ann_ret']:>9.1f}{mf['ann_vol']:>7.1f}"
              f"{mf['maxdd']:>8.1f}{'  │':<3}{mo['sharpe']:>11.2f}{mo['ann_ret']:>9.1f}"
              f"{mo['maxdd']:>8.1f}{'  │':<3}{turn:>8.0f}", flush=True)
    print("=" * 92)

    # ── Cost sensitivity (longflat) — does the realistic fee break it? ────────
    print(f"\n  Cost sensitivity (longflat, full-period net):")
    print(f"  {'per-side':<12}{'scenario':<26}{'Sharpe':>8}{'AnnRet%':>9}"
          f"{'$fees/yr':>10}")
    scenarios = [(0.0010, 'old backtest (0.2% RT)'),
                 (FEE_MAKER + SLIP_SIDE, 'maker + slip (default)'),
                 (FEE_TAKER + SLIP_SIDE, 'taker + slip (market orders)')]
    for cps, label in scenarios:
        net, _, turn = backtest(F, R, V, args.interval, 'longflat', cost=cps)
        m = metrics(net, args.interval)
        print(f"  {cps*100:>6.2f}%{'':<5}{label:<26}{m['sharpe']:>8.2f}"
              f"{m['ann_ret']:>9.1f}{turn*cps*CAPITAL:>10.0f}")
    print("=" * 92)

    # ── No-trade band sweep (longflat, default maker cost) ────────────────────
    print(f"\n  No-trade band sweep (longflat, {COST_PER_SIDE*100:.2f}%/side):")
    print(f"  {'band ($/$5k)':<14}{'turn/yr':>9}{'$fees/yr':>10}{'FullSharpe':>12}"
          f"{'OOS Sharpe':>12}{'MaxDD%':>9}")
    for band in [0.0, 0.002, 0.004, 0.006, 0.010]:
        net, _, turn = backtest(F, R, V, args.interval, 'longflat', band=band)
        mf = metrics(net, args.interval)
        mo = metrics(net[net.index >= cut], args.interval)
        tag = f"{band*100:.1f}% (${band*CAPITAL:.0f})"
        print(f"  {tag:<14}{turn:>9.1f}{turn*COST_PER_SIDE*CAPITAL:>10.0f}"
              f"{mf['sharpe']:>12.2f}{mo['sharpe']:>12.2f}{mf['maxdd']:>9.1f}")
    print("=" * 92)
    print("  band = don't trade a coin until its target drifts > band from current.")

    # per-year net Sharpe for longbias (regime honesty)
    net, _, _ = backtest(F, R, V, args.interval, 'longbias')
    bpy = BARS_PER_YEAR_MAP[args.interval]
    yr = "  longbias net Sharpe/yr: "
    for y, r in net.groupby(net.index.year):
        yr += f"{y}:{(r.mean()/r.std()*np.sqrt(bpy) if r.std()>0 else float('nan')):>5.1f} "
    print(yr)
    print(f"\n  Net of {COST_PER_SIDE*100:.2f}%/side spot cost. Trust OOS. "
          f"longflat avoids the short drag seen in earlier tests.")

    if args.attribution:
        attribution(C, R, V, F, args.interval, args.attr_mode)

    plt.figure(figsize=(14, 6))
    for mode, eq in curves.items():
        plt.plot(eq.index, eq.values, lw=1.2, label=mode)
    plt.axvline(cut, color='crimson', ls='--', lw=1, label='OOS start')
    plt.yscale('log'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.title(f"Crypto EWMAC trend  |  {'ensemble' if speeds is None else speeds}  |  "
              f"{args.interval}  |  net {COST_ROUND_TRIP*100:.1f}%RT  |  growth of $1 (log)")
    plt.ylabel('equity (×)')
    os.makedirs('results', exist_ok=True)
    out = f"results/crypto_trend_{args.interval}.png"
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"\n  Chart saved to {out}")


if __name__ == '__main__':
    main()
