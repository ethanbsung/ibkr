#!/usr/bin/env python3
"""
EWMAC trend-following with volatility regime adjustment.

Extends ewmac_backtest.py. Per-instrument, per-bar:

  V_i,t  = blended_vol_i,t / rolling_mean(blended_vol_i, window=2560 bars)
  Q_i,t  = expanding quantile of V_i,t   (0=lowest ever seen, 1=highest ever)
  M_i,t  = EWMA(span=10)[2 - 1.5 * Q_i,t]     range [0.5, 2.0]

  Q=0   → M=2.0  (low vol regime  — scale positions up)
  Q=2/3 → M=1.0  (neutral)
  Q=1   → M=0.5  (high vol regime — scale positions down)

Each raw EWMAC forecast is multiplied by M before the forecast scalar is applied.
Forecast scalars are estimated on the unmodified raw series (same as the base).
FDM, weights, IDM, sizing formula, and position buffer are unchanged.

Usage:
  python3 etf/ewmac_vol_adj.py
  python3 etf/ewmac_vol_adj.py --no-base          # skip re-running the base
  python3 etf/ewmac_vol_adj.py --vol-target 0.20
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ewmac_backtest import (
    EWMAC_VARIANTS, FORECAST_CAP, FORECAST_TARGET, BDAYS, WARMUP_BARS,
    GROSS_LEVERAGE_CAP, VOL_YEARS_LONG, FDM,
    blended_vol, normalised_ewmac, estimate_scalar,
    compute_handcraft_weights, compute_idm, apply_buffer,
    load_prices, performance_metrics, yearly_returns,
    run_backtest,
    UNIVERSE_FILE, RESULTS_DIR,
)

VOL_WINDOW    = VOL_YEARS_LONG * BDAYS   # 2560 bars = 10-yr rolling mean window
MULT_SPAN     = 10                        # EWMA span for smoothing M
Q_MIN_PERIODS = 100                       # min bars before Q is valid


# ── Vol regime functions ───────────────────────────────────────────────────────

def vol_relative(vol: pd.Series) -> pd.Series:
    """V_i,t = vol_i,t / 10-yr rolling mean of vol. min_periods=BDAYS to avoid
    dividing by a single-bar mean early in the series."""
    mean = vol.rolling(window=VOL_WINDOW, min_periods=BDAYS).mean()
    return vol / mean


def expanding_quantile(V: pd.Series) -> pd.Series:
    """Q_i,t = fraction of historical V values at or below V_i,t.
    Strictly expanding — no look-ahead."""
    return V.expanding(min_periods=Q_MIN_PERIODS).apply(
        lambda x: float((x <= x[-1]).mean()), raw=True
    )


def vol_multiplier(Q: pd.Series) -> pd.Series:
    """M_i,t = EWMA(span=10)[2 - 1.5 * Q_i,t].
    Q=0 → M→2, Q=2/3 → M→1, Q=1 → M→0.5."""
    return (2.0 - 1.5 * Q).ewm(span=MULT_SPAN, min_periods=1).mean()


# ── Vol-adjusted forecast builder ─────────────────────────────────────────────

def build_vol_adj_forecasts(
    prices: pd.DataFrame,
    vols: pd.DataFrame,
    scalars: dict,
    multipliers: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """
    Identical to ewmac_backtest.build_forecasts() except each raw EWMAC is
    multiplied by M before the scalar is applied.

    Returns:
      per_speed_fc : {(fast,slow): DataFrame(dates, tickers)}
      combined_fc  : DataFrame(dates, tickers)
    """
    per_speed_fc = {}
    for fast, slow in EWMAC_VARIANTS:
        speed_df = pd.DataFrame(index=prices.index)
        for tk in prices.columns:
            raw    = normalised_ewmac(prices[tk], fast, slow, vols[tk])
            M      = multipliers[tk]
            scalar = scalars.get((fast, slow), 1.0)
            speed_df[tk] = (raw * M * scalar).clip(-FORECAST_CAP, FORECAST_CAP)
        per_speed_fc[(fast, slow)] = speed_df

    combined_fc = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for tk in prices.columns:
        avg = pd.concat([per_speed_fc[fs][tk] for fs in EWMAC_VARIANTS], axis=1).mean(axis=1)
        combined_fc[tk] = (avg * FDM).clip(-FORECAST_CAP, FORECAST_CAP)
        combined_fc[tk].iloc[:WARMUP_BARS] = np.nan

    return per_speed_fc, combined_fc


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_vol_adj_backtest(
    tickers: list[str],
    capital: float,
    vol_target: float,
    start: str,
    asset_classes: dict,
    whole_share_shorts: bool = False,
) -> dict:

    print(f"\nLoading prices for {len(tickers)} ETFs from {start}…")
    prices  = load_prices(tickers, start)
    tickers = list(prices.columns)
    print(f"  {len(tickers)} ETFs loaded, {len(prices)} trading days")

    returns = prices.pct_change()
    vols    = pd.DataFrame({tk: blended_vol(returns[tk]) for tk in tickers})

    # ── Vol regime multipliers ────────────────────────────────────────────────
    print("  Computing vol regime multipliers (expanding quantile — slow)…")
    multipliers = pd.DataFrame(index=prices.index, columns=tickers, dtype=float)
    for i, tk in enumerate(tickers):
        V = vol_relative(vols[tk])
        Q = expanding_quantile(V)
        multipliers[tk] = vol_multiplier(Q)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(tickers)} done…")

    m_flat = multipliers.stack()
    print(f"  M  mean={m_flat.mean():.3f}  std={m_flat.std():.3f}"
          f"  p10={m_flat.quantile(0.1):.3f}  p90={m_flat.quantile(0.9):.3f}")

    # ── Forecast scalars — estimated on unmodified raw (same as base) ─────────
    print("  Estimating forecast scalars…")
    scalars = {}
    for fast, slow in EWMAC_VARIANTS:
        raw_all = pd.concat([
            normalised_ewmac(prices[tk], fast, slow, vols[tk]) for tk in tickers
        ]).dropna()
        scalars[(fast, slow)] = estimate_scalar(raw_all)
        print(f"    EWMAC({fast:3},{slow:4})  scalar={scalars[(fast,slow)]:.2f}")

    # ── Vol-adjusted forecasts ────────────────────────────────────────────────
    print("  Building vol-adjusted forecasts…")
    per_speed_fc, combined_fc = build_vol_adj_forecasts(
        prices, vols, scalars, multipliers)

    # ── Weights + IDM ─────────────────────────────────────────────────────────
    weights = compute_handcraft_weights(tickers, asset_classes)
    idm     = compute_idm(weights, returns)
    print(f"  IDM: {idm:.3f}")

    group_totals: dict[str, float] = {}
    for tk, w in weights.items():
        g = asset_classes.get(tk, "OTHER")
        group_totals[g] = group_totals.get(g, 0) + w
    print("  Asset-class group weights:")
    for g, total in sorted(group_totals.items(), key=lambda x: -x[1]):
        print(f"    {g:<20} {total*100:.1f}%")

    # ── Position sizing ───────────────────────────────────────────────────────
    print("  Sizing positions…")
    raw_pos = pd.DataFrame(index=prices.index, columns=tickers, dtype=float)
    for tk in tickers:
        w = weights.get(tk, 0.0)
        raw_pos[tk] = (capital * idm * w * vol_target
                       * (combined_fc[tk] / FORECAST_TARGET)
                       / vols[tk].replace(0, np.nan))

    if GROSS_LEVERAGE_CAP is not None:
        gross       = raw_pos.abs().sum(axis=1)
        scale_cap   = (capital * GROSS_LEVERAGE_CAP / gross).clip(upper=1.0)
        raw_pos     = raw_pos.multiply(scale_cap, axis=0)
        days_capped = (scale_cap < 1.0).sum()
        avg_sc      = scale_cap[scale_cap < 1.0].mean() if days_capped > 0 else 1.0
        print(f"  Leverage cap {GROSS_LEVERAGE_CAP}x: binds {days_capped}/{len(scale_cap)} days"
              f"  ({days_capped/len(scale_cap)*100:.1f}%),  avg scale={avg_sc:.1%}")

    if whole_share_shorts:
        shares     = raw_pos / prices
        short_mask = raw_pos < 0
        raw_pos    = raw_pos.where(~short_mask, shares.round() * prices)

    print("  Applying position buffer…")
    buffered_pos = pd.DataFrame(index=prices.index, columns=tickers, dtype=float)
    for tk in tickers:
        buffered_pos[tk] = apply_buffer(raw_pos[tk])

    # ── P&L ───────────────────────────────────────────────────────────────────
    pos_shifted   = buffered_pos.shift(1)
    pnl_matrix    = pos_shifted * returns
    portfolio_pnl = pnl_matrix.sum(axis=1)
    portfolio_ret = portfolio_pnl / capital
    equity        = (1 + portfolio_ret).cumprod() * capital

    # ── Per-instrument attribution ─────────────────────────────────────────────
    inst_metrics: dict[str, dict] = {}
    total_pnl = portfolio_pnl.sum()
    for tk in tickers:
        inst_pnl    = pos_shifted[tk] * returns[tk]
        inst_equity = (1 + (inst_pnl / capital).fillna(0)).cumprod() * capital
        m = performance_metrics(inst_equity.dropna())
        m["weight"]           = weights.get(tk, 0.0)
        m["contribution_pct"] = inst_pnl.sum() / total_pnl * 100 if total_pnl != 0 else 0
        m["asset_class"]      = asset_classes.get(tk, "OTHER")
        inst_metrics[tk] = m

    # ── Per-speed attribution ─────────────────────────────────────────────────
    speed_metrics: dict[tuple, dict] = {}
    for fast, slow in EWMAC_VARIANTS:
        speed_pnl = pd.Series(0.0, index=prices.index)
        for tk in tickers:
            fc_s = per_speed_fc[(fast, slow)][tk].copy()
            fc_s.iloc[:WARMUP_BARS] = np.nan
            w = weights.get(tk, 0.0)
            pos_s = (capital * idm * w * vol_target
                     * (fc_s / FORECAST_TARGET)
                     / vols[tk].replace(0, np.nan))
            speed_pnl += pos_s.shift(1).fillna(0) * returns[tk].fillna(0)
        speed_eq = (1 + (speed_pnl / capital)).cumprod() * capital
        speed_metrics[(fast, slow)] = performance_metrics(speed_eq)

    return dict(
        equity=equity, portfolio_ret=portfolio_ret, pnl_matrix=pnl_matrix,
        inst_metrics=inst_metrics, speed_metrics=speed_metrics,
        weights=weights, idm=idm, scalars=scalars,
        prices=prices, vols=vols, combined_fc=combined_fc,
        multipliers=multipliers, group_totals=group_totals,
    )


# ── Comparison reporting ───────────────────────────────────────────────────────

def print_comparison(base_res: dict, adj_res: dict) -> None:
    bm = performance_metrics(base_res["equity"])
    am = performance_metrics(adj_res["equity"])

    print("\n" + "=" * 65)
    print("  BASE EWMAC  vs  EWMAC + VOL REGIME ADJUSTMENT")
    print("=" * 65)
    print(f"  {'Metric':<22} {'Base':>10} {'Vol Adj':>10} {'Diff':>10}")
    print("  " + "-" * 60)

    def row(label, b, a, pct=False):
        diff = a - b
        sfx  = "%" if pct else ""
        fmt  = ".2f" if pct else ".3f"
        sign = "+" if diff >= 0 else ""
        print(f"  {label:<22} {b:>9{fmt}}{sfx}  {a:>9{fmt}}{sfx}  {sign}{diff:{fmt}}{sfx}")

    row("CAGR",          bm["cagr"]*100,        am["cagr"]*100,        pct=True)
    row("Annual vol",    bm["ann_vol"]*100,      am["ann_vol"]*100,     pct=True)
    row("Sharpe",        bm["sharpe"],           am["sharpe"])
    row("Max drawdown",  bm["max_dd"]*100,       am["max_dd"]*100,      pct=True)
    row("Calmar",        bm["calmar"],           am["calmar"])
    row("% months +ve",  bm["pct_pos_months"],  am["pct_pos_months"],  pct=True)

    mult = adj_res["multipliers"].stack()
    print(f"\n  Vol multiplier  mean={mult.mean():.3f}  std={mult.std():.3f}"
          f"  p10={mult.quantile(0.1):.3f}  p90={mult.quantile(0.9):.3f}")

    base_yr = yearly_returns(base_res["equity"])
    adj_yr  = yearly_returns(adj_res["equity"])
    years   = sorted(set(base_yr.index) | set(adj_yr.index))
    print("\n  Year-by-year returns:")
    print(f"  {'Year':<6} {'Base':>8} {'Vol Adj':>9} {'Diff':>8}")
    for yr in years:
        b = base_yr.get(yr, float("nan"))
        a = adj_yr.get(yr,  float("nan"))
        d = a - b if not (np.isnan(a) or np.isnan(b)) else float("nan")
        print(f"  {yr:<6} "
              f"{b*100:>+7.1f}%  "
              f"{a*100:>+8.1f}%  "
              f"{d*100:>+7.1f}%" if not np.isnan(d) else
              f"  {yr:<6} {b*100:>+7.1f}%  {'n/a':>8}  {'n/a':>8}")


def plot_comparison(base_res: dict, adj_res: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    base_eq = base_res["equity"]
    adj_eq  = adj_res["equity"]
    bm      = performance_metrics(base_eq)
    am      = performance_metrics(adj_eq)

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # 1. Equity curves
    ax = axes[0]
    nb = base_eq / base_eq.iloc[0]
    na = adj_eq  / adj_eq.iloc[0]
    ax.plot(nb.index, nb, color="steelblue", lw=1.2,
            label=f"Base EWMAC   SR={bm['sharpe']:.2f}  CAGR={bm['cagr']*100:.1f}%  "
                  f"MaxDD={bm['max_dd']*100:.1f}%")
    ax.plot(na.index, na, color="darkorange", lw=1.2, alpha=0.85,
            label=f"Vol Adj EWMAC  SR={am['sharpe']:.2f}  CAGR={am['cagr']*100:.1f}%  "
                  f"MaxDD={am['max_dd']*100:.1f}%")
    ax.set_title("EWMAC vs EWMAC + Volatility Regime Adjustment  (normalised to 1.0)")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Drawdowns
    ax = axes[1]
    dd_b = (base_eq - base_eq.cummax()) / base_eq.cummax() * 100
    dd_a = (adj_eq  - adj_eq.cummax())  / adj_eq.cummax()  * 100
    ax.fill_between(dd_b.index, dd_b, 0, alpha=0.35, color="steelblue", label="Base")
    ax.fill_between(dd_a.index, dd_a, 0, alpha=0.35, color="darkorange", label="Vol Adj")
    ax.set_title("Drawdowns (%)")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Portfolio-average vol multiplier
    ax = axes[2]
    mean_M = adj_res["multipliers"].mean(axis=1)
    ax.plot(mean_M.index, mean_M, color="purple", lw=0.9, alpha=0.8,
            label="Mean M (cross-instrument avg)")
    ax.axhline(1.0, color="black", lw=0.8, linestyle="--", alpha=0.5, label="M=1 neutral")
    ax.fill_between(mean_M.index, mean_M, 1.0,
                    where=(mean_M >= 1.0), alpha=0.2, color="seagreen",
                    label="M>1  low vol regime")
    ax.fill_between(mean_M.index, mean_M, 1.0,
                    where=(mean_M < 1.0), alpha=0.2, color="crimson",
                    label="M<1  high vol regime")
    ax.set_title("Portfolio-average Vol Multiplier M")
    ax.set_ylabel("M")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "etf_ewmac_vol_adj.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Chart saved to {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="EWMAC + vol regime adjustment backtest")
    ap.add_argument("--vol-target",         type=float, default=0.25)
    ap.add_argument("--capital",            type=float, default=5_000)
    ap.add_argument("--start",              default="2008-01-01")
    ap.add_argument("--no-base",            action="store_true",
                    help="Skip re-running the base backtest (saves time)")
    ap.add_argument("--no-plot",            action="store_true")
    ap.add_argument("--whole-share-shorts", action="store_true")
    args = ap.parse_args()

    with open(UNIVERSE_FILE) as f:
        u = json.load(f)
    tickers       = u["selected"]
    asset_classes = u.get("asset_classes", {})
    print(f"Universe: {len(tickers)} instruments")

    if not args.no_base:
        print("\n── BASE EWMAC ──────────────────────────────────────────────────────")
        base_res = run_backtest(tickers, args.capital, args.vol_target, args.start,
                                asset_classes, whole_share_shorts=args.whole_share_shorts)
    else:
        base_res = None

    print("\n── EWMAC + VOL REGIME ADJUSTMENT ──────────────────────────────────")
    adj_res = run_vol_adj_backtest(tickers, args.capital, args.vol_target, args.start,
                                   asset_classes, whole_share_shorts=args.whole_share_shorts)

    if base_res is not None:
        print_comparison(base_res, adj_res)
        if not args.no_plot:
            plot_comparison(base_res, adj_res)
    else:
        from ewmac_backtest import print_report, plot_results
        print_report(adj_res, args.vol_target, args.capital)
        if not args.no_plot:
            plot_results(adj_res, args.vol_target)


if __name__ == "__main__":
    main()
