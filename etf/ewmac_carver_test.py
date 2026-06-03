#!/usr/bin/env python3
"""
ewmac_carver_test.py — run the ibkr_fut "source of truth" EWMAC engine on the
ETF universe, apples-to-apples against the live ETF implementation
(etf/ewmac_backtest.py).

Reuses ibkr_fut.foundations + ibkr_fut.ewmac_signals UNCHANGED. The only
adaptations are the things that genuinely differ for cash ETFs vs futures:
    multiplier   = 1          (1 share, price is the notional per unit)
    fx           = 1          (USD instruments)
    rolls/year   = 0          (no contract rolls)
    prices       = yfinance total-return close
    returns      = simple pct_change (no back-adjustment)
    commission   = 0          (Alpaca commission-free; cost = spread only)

This lets us see what Carver's published-scalar, correlation-handcraft,
constant-buffer engine produces on the exact same ETF data the live system
trades — isolating the effect of the methodology differences documented in the
comparison (fitted vs fixed scalars, equal-weight vs handcraft, etc.).

Usage:
    python3 etf/ewmac_carver_test.py
    python3 etf/ewmac_carver_test.py --vol-target 0.25 --start 2010-01-01
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibkr_fut.foundations import (
    ANNUAL_DAYS,
    blended_vol,
    pct_returns,
    sigma_p_from_pct,
    sr_cost_per_trade,
    handcraft_weights,
    idm_from_corr,
    performance_stats,
)
from ibkr_fut.ewmac_signals import combined_forecast, eligible_speeds

DATA_DIR      = "Data/etf"
UNIVERSE_FILE = "Data/etf/etf_universe_greedy.json"
SPREADS_FILE  = "Data/etf/etf_spreads.json"

CAPITAL          = 100_000
TARGET_RISK      = 0.20
BUFFER_FRAC      = 0.10
COMMISSION       = 0.0     # Alpaca commission-free
IDM_CAP          = 2.5
MIN_HISTORY_DAYS = 512
SPREAD_DEFAULT_BPS = 2.0


# ── Data ───────────────────────────────────────────────────────────────────────
def load_close(ticker: str, start: str) -> pd.Series:
    path = os.path.join(DATA_DIR, f"{ticker.lower()}_1d_yf.csv")
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    s = pd.read_csv(path, parse_dates=["time"]).set_index("time")["close"]
    s = s[s.index >= start]
    # Normalise tz / time component so the multi-instrument index aligns cleanly.
    s.index = s.index.normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_spreads() -> dict[str, float]:
    if not os.path.exists(SPREADS_FILE):
        return {}
    return json.load(open(SPREADS_FILE)).get("half_spread_bps", {})


# ── Single-instrument backtest (mirrors ibkr_fut.backtest_ewmac.run_single) ─────
def run_single_etf(ticker: str, prices: pd.Series, half_spread_bps: float,
                   capital: float, target_risk: float) -> dict:
    """
    EWMAC backtest for one ETF. mult=1, fx=1, no rolls. Returns are P&L/capital
    so std_dev reflects the realised risk target. Position is in (fractional)
    shares; cost is the round-trip half-spread on traded notional.
    """
    if len(prices) < MIN_HISTORY_DAYS:
        return {}

    ret   = pct_returns(prices)
    sigma = blended_vol(ret)
    sp    = sigma_p_from_pct(prices, sigma)        # daily risk in price points

    med_price = float(prices[prices > 0].median())
    med_sigma = float(sigma.dropna().median())
    if med_sigma <= 0 or med_price <= 0 or np.isnan(med_price):
        return {}

    # Spread in price points at the median price (constant, like SpreadCost).
    spread = med_price * half_spread_bps / 1e4

    # Per-speed cost filter. ETFs have no rolls, so rolls_per_year = 0.
    c_trade = sr_cost_per_trade(spread, 1.0, med_price, med_sigma, COMMISSION)
    active  = eligible_speeds(c_trade, rolls_per_year=0)
    if not active:
        return {}

    sigma_a = sigma.reindex(prices.index, method="ffill")
    sp_a    = sp.reindex(prices.index, method="ffill")
    forecast_s = combined_forecast(prices, sp_a, active)

    current_pos = 0.0
    total_cost  = 0.0
    abs_changes = 0.0
    abs_pos_sum = 0.0
    pnl_list, dates = [], []

    prices_arr   = prices.values
    sigma_arr    = sigma_a.values
    forecast_arr = forecast_s.values
    idx          = prices.index

    for i in range(len(idx)):
        price   = float(prices_arr[i])
        sig_pct = float(sigma_arr[i])

        if np.isnan(price) or price <= 0 or np.isnan(sig_pct) or sig_pct <= 0:
            pnl_list.append(0.0); dates.append(idx[i])
            abs_pos_sum += abs(current_pos)
            continue

        # Step 1: P&L from yesterday's position (shares × price change)
        daily_pnl = 0.0
        if i > 0:
            prev = float(prices_arr[i - 1])
            if not np.isnan(prev):
                daily_pnl = current_pos * (price - prev)

        # Step 2: target position (shares). mult=1, fx=1.
        forecast = float(forecast_arr[i])
        N_target = (forecast * capital * IDM_local * target_risk
                    / (10.0 * price * sig_pct))

        # Step 3: buffer band = 0.10 × average (forecast=10) position — constant.
        B     = (BUFFER_FRAC * capital * IDM_local * target_risk
                 / (price * sig_pct))
        lower = N_target - B
        upper = N_target + B

        if   current_pos < lower: new_pos = lower
        elif current_pos > upper: new_pos = upper
        else:                     new_pos = current_pos

        trade_size = abs(new_pos - current_pos)
        trade_cost = 0.0
        if trade_size > 0:
            trade_cost  = trade_size * (2.0 * spread + COMMISSION)
            total_cost += trade_cost
            abs_changes += trade_size
            current_pos  = new_pos

        abs_pos_sum += abs(current_pos)
        pnl_list.append(daily_pnl - trade_cost)
        dates.append(idx[i])

    pnl_s         = pd.Series(pnl_list, index=dates, name=ticker)
    daily_returns = pnl_s / capital
    equity_curve  = capital * (1.0 + daily_returns).cumprod()

    n_days    = len(dates)
    years     = n_days / ANNUAL_DAYS
    avg_abs_N = abs_pos_sum / n_days if n_days else 1.0
    turnover  = (abs_changes / 2.0) / avg_abs_N / years if years and avg_abs_N else 0.0
    costs_pct = (total_cost / capital / years) * 100 if years else 0.0

    stats = performance_stats(equity_curve, daily_returns, costs_pct=costs_pct,
                              turnover=turnover)
    return dict(daily_returns=daily_returns, stats=stats, costs_total=total_cost,
                ticker=ticker)


# IDM_local is set to 1.0 for the standalone pass; the portfolio IDM is applied
# in the dynamic-reweighting combine below (matches run_portfolio).
IDM_local = 1.0


# ── Portfolio (mirrors ibkr_fut.backtest_ewmac.run_portfolio combine) ───────────
def run_portfolio_etf(vol_target: float, start: str) -> None:
    u = json.load(open(UNIVERSE_FILE))
    tickers       = u["selected"]
    asset_classes = u.get("asset_classes", {})
    spreads       = load_spreads()

    print(f"Loading {len(tickers)} ETFs from {start}…")
    px = {tk: load_close(tk, start) for tk in tickers}
    px = {tk: s for tk, s in px.items() if len(s) >= MIN_HISTORY_DAYS}
    print(f"  {len(px)} ETFs with ≥{MIN_HISTORY_DAYS}d history")

    # Standalone pass (weight=1, IDM=1) at TARGET_RISK / vol_target.
    results: dict[str, dict] = {}
    for tk, s in px.items():
        r = run_single_etf(tk, s, spreads.get(tk, SPREAD_DEFAULT_BPS),
                            CAPITAL, vol_target)
        if r:
            results[tk] = r
    print(f"  {len(results)} ETFs ran successfully")

    ret_df = pd.DataFrame({tk: r["daily_returns"] for tk, r in results.items()}).sort_index()
    instr  = list(ret_df.columns)

    # Source-of-truth weighting: correlation-cluster handcraft + IDM = 1/sqrt(w'Cw).
    corr    = ret_df.corr(min_periods=252).fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    weights = handcraft_weights(instr, corr)
    idm_full = min(idm_from_corr(weights, corr), IDM_CAP)
    print(f"  IDM (handcraft, capped {IDM_CAP}): {idm_full:.3f}")

    # Dynamic live-universe reweighting (the run_portfolio fix), IDM capped.
    W = np.array([weights.get(i, 0.0) for i in instr])
    C = corr.loc[instr, instr].values.copy()
    np.fill_diagonal(C, 1.0)
    live = ret_df.notna().values
    R    = ret_df.fillna(0.0).values
    port = np.zeros(len(ret_df))
    cache: dict[tuple, tuple | None] = {}
    for t in range(len(ret_df)):
        key = tuple(np.flatnonzero(live[t]))
        if not key:
            continue
        c = cache.get(key, 0)
        if c == 0:
            idxs = np.asarray(key); w_a = W[idxs]; s = float(w_a.sum())
            if s <= 0:
                cache[key] = None; continue
            w_n = w_a / s
            Csub = C[np.ix_(idxs, idxs)]
            var  = float(w_n @ Csub @ w_n)
            idm_t = min(1.0 / np.sqrt(var) if var > 0 else 1.0, IDM_CAP)
            c = cache[key] = (idxs, w_n, idm_t)
        if c is None:
            continue
        idxs, w_n, idm_t = c
        port[t] = idm_t * float(w_n @ R[t, idxs])

    port_ret = pd.Series(port, index=ret_df.index)
    port_eq  = CAPITAL * (1.0 + port_ret).cumprod()

    years     = len(port_eq) / ANNUAL_DAYS
    tot_cost  = sum(r["costs_total"] * weights.get(tk, 0.0) * idm_full
                    for tk, r in results.items())
    costs_pct = (tot_cost / CAPITAL / years) * 100 if years else 0.0
    avg_turn  = float(np.mean([r["stats"]["turnover"] for r in results.values()]))

    stats = performance_stats(port_eq, port_ret, costs_pct=costs_pct, turnover=avg_turn)

    print("\n" + "=" * 78)
    print(f"  SOURCE-OF-TRUTH EWMAC ENGINE ON ETF DATA  (vol target {vol_target*100:.0f}%)")
    print("=" * 78)
    print(f"  Period: {port_eq.index[0].date()} → {port_eq.index[-1].date()}  ({years:.1f}y)")
    for k in ("mean_annual_return_pct", "std_dev_pct", "sharpe_ratio",
              "max_drawdown_pct", "avg_drawdown_pct", "costs_pct", "turnover",
              "skew", "lower_tail", "upper_tail"):
        print(f"    {k:<24} {stats[k]}")

    # Per-asset-class median standalone stats
    by_class: dict[str, list] = {}
    for tk, r in results.items():
        by_class.setdefault(asset_classes.get(tk, "OTHER"), []).append(r["stats"])
    print("\n  Standalone median SR by asset class:")
    for ac in sorted(by_class):
        srs = [s["sharpe_ratio"] for s in by_class[ac]]
        print(f"    {ac:<12} n={len(srs):<3} SR={np.median(srs):+.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol-target", type=float, default=TARGET_RISK)
    ap.add_argument("--start", default="2010-01-01")
    args = ap.parse_args()
    run_portfolio_etf(args.vol_target, args.start)
