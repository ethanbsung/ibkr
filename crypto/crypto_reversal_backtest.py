#!/usr/bin/env python3
"""
Liquidation / overreaction reversal — TRADEABLE BACKTEST (stage 2)
────────────────────────────────────────────────────────────────────
The event study proved a one-directional edge: BUY extreme down-shocks (≥Nσ 1h
crash), hold ~H hours, capture the bounce. Up-shocks don't fade. This sizes and
costs it as a real strategy on Coinbase crypto FUTURES (flat $0.10/side ⇒ ~0%
commission; cost is spread/slippage, modelled per-side and swept).

Event-driven, weight-space accounting. A coin can hold one position at a time;
during market-wide crashes many fire at once, so total gross is capped and new
entries scaled down. Reports net Sharpe across a cost sweep and — the key
question — correlation of daily returns to the daily trend book.

Long-only (buy panic). 8 Coinbase-futures coins (BNB, ADA have no futures).

Usage:
  python3 crypto/crypto_reversal_backtest.py
  python3 crypto/crypto_reversal_backtest.py --z 5 --hold 4
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crypto_mr_backtest import load_clean, FNAME_RE, BARS_PER_YEAR_MAP
import crypto_trend as ct

FUTURES_COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'AVAX', 'LINK', 'DOGE', 'LTC']

VOL_SPAN      = 168     # 1 week of hours; lagged baseline for the z-score
TREND_SPAN    = 336     # 2 weeks of hours; slow EMA for the uptrend filter
TARGET_RISK   = 0.15    # per-trade: weight = TARGET_RISK / coin_ann_vol
W_CAP         = 0.50    # max weight per single trade
MAX_GROSS     = 1.50    # cap on total simultaneous exposure (crash concurrency)


def build_panel(coins, interval):
    data = {}
    for p in sorted(glob.glob(f"Data/*_{interval}_binance.csv")):
        m = FNAME_RE.search(os.path.basename(p))
        coin = m.group(1).upper().replace('USDT', '') if m else None
        if coin in coins:
            data[coin] = load_clean(p, interval, exclude_spikes=True)
    idx = pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in data.values()])))
    C = pd.DataFrame({s: d['close'].reindex(idx) for s, d in data.items()})
    R = pd.DataFrame({s: d['ret'].reindex(idx) for s, d in data.items()})

    vol = R.ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std()
    Z_raw = R / vol.shift(1)                                    # total-move z-score
    ann_vol = vol * np.sqrt(BARS_PER_YEAR_MAP[interval])

    # Idiosyncratic z: demarket by cross-sectional median, then z-score the residual.
    # A market-wide crash → residual ≈ 0 → no trigger (kills the steamroller trades).
    mkt = R.median(axis=1)
    resid = R.sub(mkt, axis=0)
    rvol = resid.ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std()
    Z_idio = resid / rvol.shift(1)

    # Uptrend filter: price above its slow EMA (buy dips in uptrends only).
    uptrend = C > C.ewm(span=TREND_SPAN, min_periods=TREND_SPAN).mean()

    return C, R, Z_raw, Z_idio, ann_vol, uptrend


def backtest(R, Z, ann_vol, z_entry, hold_h, cost_side, interval, uptrend=None):
    """Event-driven long-only reversal. Returns an hourly net return Series.
    Entry: Z < -z_entry (and uptrend[t,c] if a trend filter mask is given)."""
    coins = list(R.columns)
    Rv = R.fillna(0.0).to_numpy()
    Zv = Z.to_numpy()
    AV = ann_vol.to_numpy()
    UT = uptrend.to_numpy() if uptrend is not None else None
    T, n = Rv.shape

    open_w = np.zeros(n)        # current weight per coin (0 = flat)
    open_until = np.full(n, -1) # last bar index this position accrues
    out = np.zeros(T)

    for t in range(T):
        # 1) accrue PnL from positions open during bar t
        pnl = float(np.dot(open_w, Rv[t]))
        cost = 0.0

        # 2) close positions whose hold ends at t (they accrued ret[t] already)
        closing = (open_until == t) & (open_w > 0)
        cost += float(open_w[closing].sum()) * cost_side
        open_w[closing] = 0.0
        open_until[closing] = -1

        # 3) entries: down-shock & flat & vol known
        gross = float(open_w.sum())
        for c in range(n):
            if open_w[c] > 0:
                continue
            z = Zv[t, c]; av = AV[t, c]
            if not np.isfinite(z) or z >= -z_entry or not np.isfinite(av) or av <= 0:
                continue
            if UT is not None and not UT[t, c]:     # trend filter: uptrend only
                continue
            w = min(W_CAP, TARGET_RISK / av)
            w = min(w, MAX_GROSS - gross)        # respect concurrency cap
            if w <= 1e-6:
                continue
            open_w[c] = w
            open_until[c] = t + hold_h           # accrues bars t+1 .. t+hold_h
            gross += w
            cost += w * cost_side                # entry cost

        out[t] = pnl - cost

    return pd.Series(out, index=R.index)


def metrics(ret, interval):
    bpy = BARS_PER_YEAR_MAP[interval]
    r = ret.dropna()
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    active = (r != 0).mean()
    return {
        'sharpe': r.mean() / r.std() * np.sqrt(bpy) if r.std() > 0 else np.nan,
        'ann_ret': (eq.iloc[-1] ** (1 / yrs) - 1) * 100 if yrs > 0 else np.nan,
        'ann_vol': r.std() * np.sqrt(bpy) * 100,
        'maxdd': ((eq - eq.cummax()) / eq.cummax()).min() * 100,
        'time_in_mkt': active * 100,
    }


def trend_daily_returns():
    """Daily long/flat ensemble net returns, to correlate against."""
    data = {}
    for p in sorted(glob.glob("Data/*_1d_binance.csv")):
        m = FNAME_RE.search(os.path.basename(p))
        if m:
            data[m.group(1).upper()] = ct.load_clean(p, '1d', False)
    C, Rd, V = ct.build_panel(data, '1d')
    F = ct.forecast_panel(C, V, None)
    net, _, _ = ct.backtest(F, Rd, V, '1d', 'longflat', band=0.004)
    s = net.copy(); s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s


def main():
    ap = argparse.ArgumentParser(description='Liquidation reversal tradeable backtest')
    ap.add_argument('--interval', default='1h')
    ap.add_argument('--z', type=float, default=4.0, help='shock threshold (sigma)')
    ap.add_argument('--hold', type=int, default=4, help='holding period (bars)')
    args = ap.parse_args()

    C, R, Z_raw, Z_idio, ann_vol, uptrend = build_panel(FUTURES_COINS, args.interval)
    cost = 0.0015     # realistic per-side (≈0.02% commission + ~0.13% spread/slippage)
    print(f"Loaded {R.shape[1]} futures coins [{args.interval}]  entry z<-{args.z}σ  "
          f"hold {args.hold}h  cost {cost*100:.2f}%/side  (spikes excluded)\n")

    trend = trend_daily_returns()

    variants = [
        ('none (baseline)',        Z_raw,  None),
        ('idiosyncratic',          Z_idio, None),
        ('trend filter',           Z_raw,  uptrend),
        ('idio + trend',           Z_idio, uptrend),
    ]
    print("=" * 92)
    print(f"  {'filter':<20}{'Sharpe':>8}{'AnnRet%':>9}{'Vol%':>7}{'MaxDD%':>8}"
          f"{'%inMkt':>8}{'corr→trend':>12}")
    print("=" * 92)
    for label, Z, ut in variants:
        ret = backtest(R, Z, ann_vol, args.z, args.hold, cost, args.interval, uptrend=ut)
        m = metrics(ret, args.interval)
        rev_daily = (1 + ret).resample('D').prod() - 1
        rev_daily.index = rev_daily.index.normalize()
        d = pd.DataFrame({'r': rev_daily, 't': trend}).dropna()
        d = d[(d['r'] != 0) | (d['t'] != 0)]
        rho = d['r'].corr(d['t'])
        print(f"  {label:<20}{m['sharpe']:>8.2f}{m['ann_ret']:>9.1f}{m['ann_vol']:>7.1f}"
              f"{m['maxdd']:>8.1f}{m['time_in_mkt']:>7.1f}%{rho:>+12.3f}")
    print("=" * 92)
    print("  Idiosyncratic demarkets the shock (kills systemic-crash trades);")
    print("  trend filter buys dips only in uptrends. Looking for +Sharpe, shallow DD.")


if __name__ == '__main__':
    main()
