#!/usr/bin/env python3
"""
ewmac_carver.py — ETF EWMAC trend on the shared `ibkr_fut` source-of-truth engine.

Forecast, volatility, cost-eligibility, handcraft weights and IDM all come from
`ibkr_fut.foundations` + `ibkr_fut.ewmac_signals` via the shared `InstrumentSpec`
and `instrument_signals()` — the exact same code the futures backtest uses. There
is no ETF-private copy of the signal math (the old `etf/ewmac_backtest.py` had its
own fitted scalars, equal-weight handcraft and shrinking buffer; this replaces it).

ETFs differ from futures in ways the engine must respect, so the *portfolio*
simulation here is a position-matrix sim (not the futures standalone-combine),
because the ETF constraints are non-linear in the actual share position:

  mult = 1, fx = 1, rolls = 0          cash instruments, no contract rolls
  cost  = |Δnotional| · half_spread_bps/1e4    (one-way half-spread; commission 0, Alpaca)
  vol target 25%                       (user choice; Alpaca account)
  gross leverage cap 1.9x              Reg-T 2x limit — a real constraint for ETFs
  long-only clip                       non-shortable tickers (etf_shortability.json)
  whole-share shorts                   shorts round to whole shares; sub-1-share → 0
  buffer band = 0.10 × average position   Carver-correct constant band (not 10% of optimal)

Cash/money-market ETFs (price vol < 3%) are screened OUT of the universe upstream
(see Data/etf/etf_universe_greedy.json hard_excluded) — sized to a vol target they
take absurd leverage on a near-riskless drift.

Usage:
    python3 etf/ewmac_carver.py
    python3 etf/ewmac_carver.py --vol-target 0.25 --start 2010-01-01 --capital 100000
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
    handcraft_weights,
    idm_from_corr,
    performance_stats,
)
from ibkr_fut.backtest_ewmac import InstrumentSpec, instrument_signals, simulate_instrument

DATA_DIR          = "Data/etf"
UNIVERSE_FILE     = "Data/etf/etf_universe_greedy.json"
SPREADS_FILE      = "Data/etf/etf_spreads.json"
SHORTABILITY_FILE = "Data/etf/etf_shortability.json"

CAPITAL            = 100_000
VOL_TARGET         = 0.25
BUFFER_FRAC        = 0.10
IDM_CAP            = 2.5
GROSS_LEVERAGE_CAP = 1.9     # Reg-T 2x with safety buffer; ETFs can't lever like futures
MIN_HISTORY_DAYS   = 512
SPREAD_DEFAULT_BPS = 2.0


# ── Data ───────────────────────────────────────────────────────────────────────
def load_close(ticker: str, start: str) -> pd.Series:
    path = os.path.join(DATA_DIR, f"{ticker.lower()}_1d_yf.csv")
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    s = pd.read_csv(path, parse_dates=["time"]).set_index("time")["close"]
    s = s[s.index >= start]
    s.index = s.index.normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_spreads() -> dict[str, float]:
    if not os.path.exists(SPREADS_FILE):
        return {}
    return json.load(open(SPREADS_FILE)).get("half_spread_bps", {})


def load_nonshortable() -> set[str]:
    if not os.path.exists(SHORTABILITY_FILE):
        return set()
    syms = json.load(open(SHORTABILITY_FILE)).get("symbols", {})
    return {tk for tk, d in syms.items() if not d.get("shortable", True)}


def build_specs(tickers, start, spreads, nonshortable) -> dict[str, InstrumentSpec]:
    specs = {}
    for tk in tickers:
        s = load_close(tk, start)
        if len(s) < MIN_HISTORY_DAYS:
            continue
        med_price = float(s[s > 0].median())
        bps = spreads.get(tk, SPREAD_DEFAULT_BPS)
        specs[tk] = InstrumentSpec(
            name=tk, prices=s, raw_price=s, fx=pd.Series(1.0, index=s.index),
            mult=1.0, spread=med_price * bps / 1e4, rolls=0, commission=0.0,
            long_only=(tk in nonshortable),
        )
    return specs


# ── Portfolio simulation (position-matrix, ETF constraints) ─────────────────────
def run(vol_target: float, start: str, capital: float, verbose: bool = True,
        universe_file: str = UNIVERSE_FILE) -> dict:
    u = json.load(open(universe_file))
    tickers       = u["selected"]
    asset_classes = u.get("asset_classes", {})
    spreads       = load_spreads()
    nonshortable  = load_nonshortable()

    print(f"Loading {len(tickers)} ETFs from {start}…")
    specs = build_specs(tickers, start, spreads, nonshortable)

    # Shared signal builder — identical forecast/vol/eligibility as the futures engine.
    sigs = {tk: instrument_signals(sp) for tk, sp in specs.items()}
    sigs = {tk: s for tk, s in sigs.items() if s is not None}
    instr = list(sigs.keys())
    print(f"  {len(instr)} ETFs with signals "
          f"({len([t for t in instr if specs[t].long_only])} long-only)")

    # Align everything to the union index.
    px_df  = pd.DataFrame({tk: specs[tk].prices for tk in instr}).sort_index()
    fc_df  = pd.DataFrame({tk: sigs[tk]["forecast"] for tk in instr}).reindex(px_df.index)
    vol_df = pd.DataFrame({tk: sigs[tk]["sigma"]    for tk in instr}).reindex(px_df.index)

    # Weights: correlation-cluster handcraft + IDM (shared functions).
    returns = px_df.pct_change()
    corr = returns.corr(min_periods=252).fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    weights = handcraft_weights(instr, corr)
    print(f"  IDM (handcraft, cap {IDM_CAP}): "
          f"{min(idm_from_corr(weights, corr), IDM_CAP):.3f}")

    # numpy views for the daily loop
    px   = px_df.values
    fc   = fc_df.values
    vol  = vol_df.values
    bps  = np.array([spreads.get(tk, SPREAD_DEFAULT_BPS) for tk in instr]) / 1e4
    lo   = np.array([specs[tk].long_only for tk in instr])
    W    = np.array([weights.get(tk, 0.0) for tk in instr])
    C    = corr.loc[instr, instr].values.copy(); np.fill_diagonal(C, 1.0)
    T, n = px.shape

    pos      = np.zeros(n)        # held shares
    px_prev  = np.full(n, np.nan) # last marked price per instrument
    pnl_list = []
    total_cost = 0.0
    abs_turn   = 0.0
    capped_days = 0
    idm_cache: dict[tuple, tuple] = {}

    for t in range(T):
        live = (np.isfinite(px[t]) & np.isfinite(fc[t]) & np.isfinite(vol[t])
                & (px[t] > 0) & (vol[t] > 0))
        idxs = np.flatnonzero(live)
        if idxs.size == 0:
            pnl_list.append(0.0)
            continue

        # Step 1: P&L from yesterday's positions (only instruments marked yesterday).
        markable = idxs[np.isfinite(px_prev[idxs])]
        pnl_t = float(np.sum(pos[markable] * (px[t, markable] - px_prev[markable])))

        # Dynamic live-universe weights + IDM (cap), cached per distinct live set.
        key = tuple(idxs)
        c = idm_cache.get(key)
        if c is None:
            w_a = W[idxs]; s = float(w_a.sum())
            w_n = w_a / s if s > 0 else np.full(idxs.size, 1.0 / idxs.size)
            Csub = C[np.ix_(idxs, idxs)]
            var  = float(w_n @ Csub @ w_n)
            idm_t = min(1.0 / np.sqrt(var) if var > 0 else 1.0, IDM_CAP)
            c = idm_cache[key] = (w_n, idm_t)
        w_n, idm_t = c

        # Step 2: target shares. base = forecast=10 ("average") position → constant buffer band.
        p = px[t, idxs]; v = vol[t, idxs]; f = fc[t, idxs]
        base   = capital * idm_t * w_n * vol_target / (v * p)   # shares at |forecast|=10
        target = base * (f / 10.0)
        lo_i   = lo[idxs]
        target = np.where(lo_i & (target < 0), 0.0, target)

        # Gross leverage cap (Reg-T): scale targets (and the buffer band) if over.
        gross = float(np.sum(np.abs(target) * p))
        if gross > GROSS_LEVERAGE_CAP * capital and gross > 0:
            scale = GROSS_LEVERAGE_CAP * capital / gross
            target *= scale; base *= scale
            capped_days += 1

        # Step 3: buffer to constant band, whole-share shorts, cost.
        B   = BUFFER_FRAC * np.abs(base)
        cur = pos[idxs]
        new = np.where(cur < target - B, target - B,
              np.where(cur > target + B, target + B, cur))
        new = np.where(lo_i & (new < 0), 0.0, new)
        # Whole-share shorts: shorts round to whole shares (sub-1-share short → 0); longs fractional.
        short = new < 0
        new = np.where(short, np.round(new), new)

        delta = np.abs(new - cur)
        cost_t = float(np.sum(delta * p * bps[idxs]))
        total_cost += cost_t
        abs_turn   += float(np.sum(delta * p))   # traded notional

        pos[idxs]     = new
        px_prev[idxs] = p
        pnl_list.append(pnl_t - cost_t)

    pnl_s = pd.Series(pnl_list, index=px_df.index)
    daily_ret = pnl_s / capital
    equity = capital * (1.0 + daily_ret).cumprod()

    years     = T / ANNUAL_DAYS
    costs_pct = (total_cost / capital / years) * 100 if years else 0.0
    turnover  = (abs_turn / capital / years) if years else 0.0   # × capital / yr
    stats = performance_stats(equity, daily_ret, costs_pct=costs_pct, turnover=turnover)

    if verbose:
        print("\n" + "=" * 78)
        print(f"  ETF EWMAC — SHARED ibkr_fut ENGINE   (vol target {vol_target*100:.0f}%, "
              f"gross cap {GROSS_LEVERAGE_CAP}x)")
        print("=" * 78)
        print(f"  Capital: ${capital:,.0f}   Period: {equity.index[0].date()} → "
              f"{equity.index[-1].date()}  ({years:.1f}y)")
        print(f"  Leverage cap binds: {capped_days}/{T} days "
              f"({capped_days/T*100:.1f}%)")
        for k in ("mean_annual_return_pct", "std_dev_pct", "sharpe_ratio",
                  "max_drawdown_pct", "avg_drawdown_pct", "costs_pct", "turnover",
                  "skew", "lower_tail", "upper_tail"):
            print(f"    {k:<24} {stats[k]}")

        # Standalone median SR by asset class (uses the shared simulate_instrument).
        by_class: dict[str, list] = {}
        for tk in instr:
            r = simulate_instrument(specs[tk], capital=capital, target_risk=vol_target,
                                    round_positions=False)
            if r:
                by_class.setdefault(asset_classes.get(tk, "OTHER"), []).append(
                    r["stats"]["sharpe_ratio"])
        print("\n  Standalone median SR by asset class:")
        for ac in sorted(by_class):
            print(f"    {ac:<12} n={len(by_class[ac]):<3} SR={np.median(by_class[ac]):+.2f}")

    return dict(equity=equity, daily_returns=daily_ret, stats=stats,
                capped_days=capped_days, n_days=T)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol-target", type=float, default=VOL_TARGET)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--capital", type=float, default=CAPITAL)
    ap.add_argument("--universe", default=UNIVERSE_FILE,
                    help="universe json (default greedy; pass jumbo for trade-everything)")
    args = ap.parse_args()
    run(args.vol_target, args.start, args.capital, universe_file=args.universe)
