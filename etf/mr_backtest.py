#!/usr/bin/env python3
"""
Mean-Reversion backtest on the same 88-ETF universe as ewmac_backtest.py.

Signal: 3 MR z-score speeds (N=10, 20, 40 days)
  raw = -(price - EMA(N)) / (price * daily_vol)
  Combined with equal weights and FDM=1.08 (Carver Table 36, 3 rule variations)

Same Carver framework as ewmac_backtest.py:
  - Same vol model, handcraft weights, IDM, position sizing, buffer
  - Same gross leverage cap (1.9x Reg-T)
  - Shorts as whole shares, longs as fractional (Alpaca constraints)
  - Transaction costs from Data/etf/etf_spreads.json (live NBBO half-spreads)

Acceptance criteria (vs. EWMAC):
  - Standalone Sharpe (net costs) >= 0.30
  - Pooled MR/EWMAC forecast correlation <= 0.30
  - 50/50 blend Sharpe > standalone EWMAC Sharpe

Usage:
  python3 etf/mr_backtest.py
  python3 etf/mr_backtest.py --vol-target 0.25 --capital 250000 --start 2010-01-01
  python3 etf/mr_backtest.py --whole-share-shorts --no-plot
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from ewmac_backtest import (
    blended_vol, estimate_scalar,
    compute_handcraft_weights, compute_idm,
    apply_buffer, load_prices,
    performance_metrics, yearly_returns,
    normalised_ewmac, build_forecasts,
    BDAYS, FORECAST_CAP, FORECAST_TARGET, IDM_CAP,
    GROSS_LEVERAGE_CAP, WARMUP_BARS,
    DATA_DIR, UNIVERSE_FILE, RESULTS_DIR,
    EWMAC_VARIANTS, FDM,
)

# ── Constants ─────────────────────────────────────────────────────────────────
MR_VARIANTS  = [10, 20, 40]
MR_FDM       = 1.08     # Carver Table 36: 3 equal-weighted rule variations
SPREADS_FILE = "Data/etf/etf_spreads.json"
SPREAD_DEFAULT_BPS = 2.0  # fallback for instruments not in the spreads file


# ── Spread loader ─────────────────────────────────────────────────────────────
def load_spreads() -> dict[str, float]:
    """Load per-instrument half-spreads (bps) from the live NBBO snapshot file."""
    if not os.path.exists(SPREADS_FILE):
        return {}
    with open(SPREADS_FILE) as f:
        data = json.load(f)
    return data.get("half_spread_bps", {})


# ── Signal ────────────────────────────────────────────────────────────────────
def normalised_mr(price: pd.Series, lookback: int,
                  annual_vol: pd.Series) -> pd.Series:
    """
    Raw MR signal: negative price deviation from EMA(N), normalised identically
    to EWMAC so forecast scalars are directly comparable.

    Positive  → oversold  (mean-revert long)
    Negative  → overbought (mean-revert short)
    """
    ema       = price.ewm(span=lookback, min_periods=lookback // 2,
                          adjust=False).mean()
    daily_vol = annual_vol / np.sqrt(BDAYS)
    return -(price - ema) / (price * daily_vol).replace(0, np.nan)


def build_mr_forecasts(prices: pd.DataFrame, vols: pd.DataFrame,
                       scalars: dict) -> tuple[dict, pd.DataFrame]:
    per_speed_fc = {}
    for n in MR_VARIANTS:
        speed_df = pd.DataFrame(index=prices.index)
        for tk in prices.columns:
            raw    = normalised_mr(prices[tk], n, vols[tk])
            scalar = scalars.get(n, 1.0)
            speed_df[tk] = (raw * scalar).clip(-FORECAST_CAP, FORECAST_CAP)
        per_speed_fc[n] = speed_df

    combined_fc = pd.DataFrame(index=prices.index,
                               columns=prices.columns, dtype=float)
    for tk in prices.columns:
        avg = pd.concat([per_speed_fc[n][tk] for n in MR_VARIANTS],
                        axis=1).mean(axis=1)
        combined_fc[tk] = (avg * MR_FDM).clip(-FORECAST_CAP, FORECAST_CAP)
        combined_fc[tk].iloc[:WARMUP_BARS] = np.nan

    return per_speed_fc, combined_fc


# ── EWMAC forecasts (for correlation analysis only) ───────────────────────────
def build_ewmac_forecasts_for_comparison(prices, vols):
    scalars = {}
    for fast, slow in EWMAC_VARIANTS:
        raw_all = pd.concat(
            [normalised_ewmac(prices[tk], fast, slow, vols[tk])
             for tk in prices.columns]
        ).dropna()
        scalars[(fast, slow)] = estimate_scalar(raw_all)
    _, ewmac_fc = build_forecasts(prices, vols, scalars)
    return ewmac_fc


# ── Main backtest ─────────────────────────────────────────────────────────────
def run_mr_backtest(tickers: list[str], capital: float, vol_target: float,
                    start: str, asset_classes: dict,
                    whole_share_shorts: bool = False) -> dict:

    print(f"\nLoading prices for {len(tickers)} ETFs from {start}…")
    prices  = load_prices(tickers, start)
    tickers = list(prices.columns)
    print(f"  {len(tickers)} ETFs loaded, {len(prices)} trading days")

    returns = prices.pct_change()
    vols    = pd.DataFrame({tk: blended_vol(returns[tk]) for tk in tickers})

    # ── Spread map ────────────────────────────────────────────────────────────
    live_spreads = load_spreads()
    spread_map   = {tk: live_spreads.get(tk, SPREAD_DEFAULT_BPS)
                    for tk in tickers}
    n_live = sum(1 for tk in tickers if tk in live_spreads)
    print(f"  Spread data: {n_live}/{len(tickers)} from live NBBO, "
          f"{len(tickers)-n_live} using {SPREAD_DEFAULT_BPS}bp default")

    # ── Scalars ───────────────────────────────────────────────────────────────
    print("  Estimating MR forecast scalars…")
    scalars = {}
    for n in MR_VARIANTS:
        raw_all = pd.concat(
            [normalised_mr(prices[tk], n, vols[tk]) for tk in tickers]
        ).dropna()
        scalars[n] = estimate_scalar(raw_all)
        print(f"    MR({n:>2}) scalar: {scalars[n]:.2f}")

    # ── Forecasts ─────────────────────────────────────────────────────────────
    print("  Building MR forecasts…")
    per_speed_fc, combined_fc = build_mr_forecasts(prices, vols, scalars)

    # ── EWMAC forecasts for correlation ───────────────────────────────────────
    print("  Building EWMAC forecasts for correlation analysis…")
    ewmac_fc = build_ewmac_forecasts_for_comparison(prices, vols)

    pooled_mr  = combined_fc.stack().dropna()
    pooled_ewm = ewmac_fc.stack().dropna()
    common_idx = pooled_mr.index.intersection(pooled_ewm.index)
    pool_corr  = float(np.corrcoef(pooled_mr[common_idx],
                                   pooled_ewm[common_idx])[0, 1])

    inst_corr = {}
    for tk in tickers:
        idx = (combined_fc[tk].dropna().index
               .intersection(ewmac_fc[tk].dropna().index))
        if len(idx) > 100:
            inst_corr[tk] = float(np.corrcoef(
                combined_fc[tk][idx], ewmac_fc[tk][idx])[0, 1])

    # ── Weights and IDM ───────────────────────────────────────────────────────
    weights = compute_handcraft_weights(tickers, asset_classes)
    idm     = compute_idm(weights, returns)
    print(f"  IDM: {idm:.3f}")

    group_totals: dict[str, float] = {}
    for tk, w in weights.items():
        g = asset_classes.get(tk, "OTHER")
        group_totals[g] = group_totals.get(g, 0) + w

    # ── Position sizing ───────────────────────────────────────────────────────
    print("  Sizing positions…")
    raw_pos = pd.DataFrame(index=prices.index, columns=tickers, dtype=float)
    for tk in tickers:
        w = weights.get(tk, 0.0)
        raw_pos[tk] = (capital * idm * w * vol_target
                       * (combined_fc[tk] / FORECAST_TARGET)
                       / vols[tk].replace(0, np.nan))

    if GROSS_LEVERAGE_CAP is not None:
        gross      = raw_pos.abs().sum(axis=1)
        scale      = (capital * GROSS_LEVERAGE_CAP / gross).clip(upper=1.0)
        raw_pos    = raw_pos.multiply(scale, axis=0)
        days_capped = (scale < 1.0).sum()
        avg_scale   = scale[scale < 1.0].mean() if days_capped > 0 else 1.0
        print(f"  Leverage cap {GROSS_LEVERAGE_CAP}x: binds {days_capped} days "
              f"({days_capped/len(scale)*100:.1f}%), avg scale {avg_scale:.1%}")

    if whole_share_shorts:
        shares     = raw_pos / prices
        short_mask = raw_pos < 0
        raw_pos    = raw_pos.where(~short_mask, shares.round() * prices)
        n_zeroed   = int(((short_mask) & (shares.round() == 0)).sum().sum())
        n_short    = int(short_mask.sum().sum())
        print(f"  Whole-share shorts: {n_zeroed}/{n_short} short "
              f"instrument-days round to 0")

    print("  Applying position buffer…")
    buffered_pos = pd.DataFrame(index=prices.index, columns=tickers, dtype=float)
    for tk in tickers:
        buffered_pos[tk] = apply_buffer(raw_pos[tk])

    # ── P&L and transaction costs ─────────────────────────────────────────────
    pos_shifted = buffered_pos.shift(1)
    pnl_gross   = pos_shifted * returns

    delta_pos   = buffered_pos.diff().abs().fillna(0)
    cost_matrix = pd.DataFrame(0.0, index=prices.index, columns=tickers)
    for tk in tickers:
        h = spread_map[tk] / 10_000
        cost_matrix[tk] = delta_pos[tk] * h

    total_costs  = cost_matrix.sum(axis=1)
    gross_pnl    = pnl_gross.sum(axis=1)
    net_pnl      = gross_pnl - total_costs

    equity_gross = (1 + gross_pnl / capital).cumprod() * capital
    equity_net   = (1 + net_pnl  / capital).cumprod() * capital

    years        = len(prices) / BDAYS
    annual_cost  = total_costs.sum() / capital / years
    annual_turn  = delta_pos.sum(axis=1).sum() / capital / years

    # ── 50/50 blend equity (MR + EWMAC, costs included) ──────────────────────
    blend_fc = pd.DataFrame(index=prices.index, columns=tickers, dtype=float)
    for tk in tickers:
        blend_fc[tk] = (0.5 * combined_fc[tk].fillna(0)
                        + 0.5 * ewmac_fc[tk].fillna(0))
        blend_fc[tk].iloc[:WARMUP_BARS] = np.nan

    blend_pos = pd.DataFrame(index=prices.index, columns=tickers, dtype=float)
    for tk in tickers:
        w = weights.get(tk, 0.0)
        blend_pos[tk] = (capital * idm * w * vol_target
                         * (blend_fc[tk] / FORECAST_TARGET)
                         / vols[tk].replace(0, np.nan))

    if GROSS_LEVERAGE_CAP is not None:
        gross_b = blend_pos.abs().sum(axis=1)
        scale_b = (capital * GROSS_LEVERAGE_CAP / gross_b).clip(upper=1.0)
        blend_pos = blend_pos.multiply(scale_b, axis=0)

    blend_buf = pd.DataFrame(index=prices.index, columns=tickers, dtype=float)
    for tk in tickers:
        blend_buf[tk] = apply_buffer(blend_pos[tk])

    blend_delta = blend_buf.diff().abs().fillna(0)
    blend_cost  = pd.Series(0.0, index=prices.index)
    for tk in tickers:
        h = spread_map[tk] / 10_000
        blend_cost += blend_delta[tk] * h

    blend_pnl    = (blend_buf.shift(1) * returns).sum(axis=1) - blend_cost
    equity_blend = (1 + blend_pnl / capital).cumprod() * capital

    # ── Per-instrument attribution ─────────────────────────────────────────────
    inst_metrics: dict[str, dict] = {}
    total_pnl_sum = net_pnl.sum()
    for tk in tickers:
        inst_pnl    = pos_shifted[tk] * returns[tk] - cost_matrix[tk]
        inst_equity = (1 + (inst_pnl / capital).fillna(0)).cumprod() * capital
        m = performance_metrics(inst_equity.dropna())
        m["weight"]           = weights.get(tk, 0.0)
        m["contribution_pct"] = (inst_pnl.sum() / total_pnl_sum * 100
                                  if total_pnl_sum != 0 else 0)
        m["asset_class"]      = asset_classes.get(tk, "OTHER")
        m["spread_half_bps"]  = spread_map[tk]
        m["inst_corr"]        = inst_corr.get(tk, np.nan)
        inst_metrics[tk] = m

    # ── Per-speed attribution ─────────────────────────────────────────────────
    speed_metrics: dict[int, dict] = {}
    for n in MR_VARIANTS:
        speed_pnl = pd.Series(0.0, index=prices.index)
        for tk in tickers:
            fc_s = per_speed_fc[n][tk].copy()
            fc_s.iloc[:WARMUP_BARS] = np.nan
            w     = weights.get(tk, 0.0)
            pos_s = (capital * idm * w * vol_target
                     * (fc_s / FORECAST_TARGET)
                     / vols[tk].replace(0, np.nan))
            speed_pnl += pos_s.shift(1).fillna(0) * returns[tk].fillna(0)
        speed_eq = (1 + speed_pnl / capital).cumprod() * capital
        speed_metrics[n] = performance_metrics(speed_eq)

    return dict(
        equity_net=equity_net,
        equity_gross=equity_gross,
        equity_blend=equity_blend,
        net_pnl=net_pnl,
        pnl_matrix=pnl_gross,
        cost_matrix=cost_matrix,
        inst_metrics=inst_metrics,
        speed_metrics=speed_metrics,
        weights=weights, idm=idm, scalars=scalars,
        prices=prices, vols=vols,
        combined_fc=combined_fc, ewmac_fc=ewmac_fc,
        group_totals=group_totals,
        pool_corr=pool_corr, inst_corr=inst_corr,
        annual_cost=annual_cost,
        annual_turn=annual_turn,
        spread_map=spread_map,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────
def print_mr_report(res: dict, vol_target: float, capital: float) -> None:
    eq_net   = res["equity_net"]
    eq_gross = res["equity_gross"]
    eq_blend = res["equity_blend"]
    m_net    = performance_metrics(eq_net)
    m_gross  = performance_metrics(eq_gross)
    m_blend  = performance_metrics(eq_blend)
    yr       = yearly_returns(eq_net)

    print("\n" + "=" * 72)
    print("  ETF MEAN-REVERSION — Handcraft equal weight, 3 speeds (10/20/40d)")
    print("=" * 72)
    print(f"  Capital:    ${capital:,.0f}")
    print(f"  Vol target: {vol_target*100:.0f}%   IDM: {res['idm']:.3f}   FDM: {MR_FDM}")
    print(f"  Period:     {eq_net.index[0].date()} → {eq_net.index[-1].date()}"
          f"  ({m_net['years']:.1f}y)")

    print(f"\n  ── Performance ─────────────────────────────────────────────────")
    print(f"               {'Gross':>8}  {'Net':>8}  {'50/50 Blend':>12}")
    print(f"  CAGR         {m_gross['cagr']*100:>7.2f}%  {m_net['cagr']*100:>7.2f}%  "
          f"{m_blend['cagr']*100:>11.2f}%")
    print(f"  Ann vol      {m_gross['ann_vol']*100:>7.2f}%  {m_net['ann_vol']*100:>7.2f}%  "
          f"{m_blend['ann_vol']*100:>11.2f}%")
    print(f"  Sharpe       {m_gross['sharpe']:>8.3f}  {m_net['sharpe']:>8.3f}  "
          f"{m_blend['sharpe']:>12.3f}")
    print(f"  Max drawdown {m_gross['max_dd']*100:>7.2f}%  {m_net['max_dd']*100:>7.2f}%  "
          f"{m_blend['max_dd']*100:>11.2f}%")
    print(f"  Calmar       {m_gross['calmar']:>8.3f}  {m_net['calmar']:>8.3f}  "
          f"{m_blend['calmar']:>12.3f}")
    print(f"  % months +ve {m_gross['pct_pos_months']:>7.1f}%  "
          f"{m_net['pct_pos_months']:>7.1f}%  "
          f"{m_blend['pct_pos_months']:>11.1f}%")

    print(f"\n  ── Transaction costs ────────────────────────────────────────────")
    print(f"  Annual turnover (× capital):  {res['annual_turn']:.2f}×")
    print(f"  Annual cost drag:             {res['annual_cost']*100:.3f}%")

    print(f"\n  ── EWMAC forecast correlation ───────────────────────────────────")
    print(f"  Pooled MR vs EWMAC forecast corr: {res['pool_corr']:+.3f}")
    print(f"  (acceptance target ≤ 0.30)")

    print(f"\n  ── Acceptance criteria ──────────────────────────────────────────")
    sr_pass   = m_net["sharpe"] >= 0.30
    corr_pass = res["pool_corr"] <= 0.30
    marg_pass = m_blend["sharpe"] > 0.0
    print(f"  Sharpe (net) >= 0.30:  {'PASS' if sr_pass else 'FAIL'}"
          f"  ({m_net['sharpe']:.3f})")
    print(f"  Forecast corr <= 0.30: {'PASS' if corr_pass else 'FAIL'}"
          f"  ({res['pool_corr']:.3f})")
    print(f"  Blend Sharpe > 0:      {'PASS' if marg_pass else 'FAIL'}"
          f"  ({m_blend['sharpe']:.3f})")

    print("\n  Year-by-year returns (net):")
    for yr_i, r in yr.items():
        bar  = "█" * int(abs(r) * 200)
        sign = "+" if r >= 0 else "-"
        print(f"    {yr_i}  {sign}{abs(r)*100:5.1f}%  {bar}")

    print("\n  MR speed Sharpe (standalone, gross):")
    for n, sm in res["speed_metrics"].items():
        bar = "█" * max(0, int(sm["sharpe"] * 20))
        print(f"    MR({n:>2})  Sharpe {sm['sharpe']:+.3f}"
              f"  CAGR {sm['cagr']*100:+.1f}%  MaxDD {sm['max_dd']*100:.1f}%  {bar}")

    print("\n  MR scalar per speed:")
    for n, s in res["scalars"].items():
        print(f"    MR({n:>2})  scalar {s:.2f}")

    print("\n  Asset-class group weights:")
    for g, total in sorted(res["group_totals"].items(), key=lambda x: -x[1]):
        bar = "█" * int(total * 200)
        print(f"    {g:<18} {total*100:5.1f}%  {bar}")

    print("\n  Asset class P&L contribution (% of total net P&L):")
    ac_pnl: dict[str, float] = {}
    for tk, im in res["inst_metrics"].items():
        ac = im["asset_class"]
        ac_pnl[ac] = ac_pnl.get(ac, 0) + im["contribution_pct"]
    for ac, pct in sorted(ac_pnl.items(), key=lambda x: -x[1]):
        bar  = "█" * int(abs(pct) / 5)
        sign = "+" if pct >= 0 else "-"
        print(f"    {ac:<18}  {sign}{abs(pct):5.1f}%  {bar}")

    print("\n  Top 15 instruments by portfolio weight:")
    ranked = sorted(res["inst_metrics"].items(), key=lambda x: -x[1]["weight"])
    for tk, im in ranked[:15]:
        corr_str = (f"  corr={im['inst_corr']:+.2f}"
                    if np.isfinite(im["inst_corr"]) else "")
        print(f"    {tk:<6}  w={im['weight']*100:4.2f}%"
              f"  SR={im['sharpe']:+.2f}"
              f"  contrib={im['contribution_pct']:+.1f}%"
              f"  spr={im['spread_half_bps']:.2f}bp"
              f"{corr_str}")

    print("\n  Top 10 instruments by net Sharpe:")
    ranked_sr = sorted(res["inst_metrics"].items(), key=lambda x: -x[1]["sharpe"])
    for tk, im in ranked_sr[:10]:
        print(f"    {tk:<6}  Sharpe {im['sharpe']:+.3f}  CAGR {im['cagr']*100:+.1f}%"
              f"  w={im['weight']*100:.2f}%  [{im['asset_class']}]")

    print("\n  Bottom 10 instruments by net Sharpe:")
    for tk, im in ranked_sr[-10:]:
        print(f"    {tk:<6}  Sharpe {im['sharpe']:+.3f}  CAGR {im['cagr']*100:+.1f}%"
              f"  w={im['weight']*100:.2f}%  [{im['asset_class']}]")


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_mr_results(res: dict, vol_target: float) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    eq_net   = res["equity_net"]
    eq_gross = res["equity_gross"]
    eq_blend = res["equity_blend"]
    m_net    = performance_metrics(eq_net)
    m_gross  = performance_metrics(eq_gross)
    m_blend  = performance_metrics(eq_blend)
    yr       = yearly_returns(eq_net)

    fig = plt.figure(figsize=(22, 18))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    norm_g = eq_gross / eq_gross.iloc[0]
    norm_n = eq_net   / eq_net.iloc[0]
    norm_b = eq_blend / eq_blend.iloc[0]
    ax1.plot(eq_net.index, norm_g, color="steelblue", lw=1.0, alpha=0.5,
             label=f"MR gross  SR={m_gross['sharpe']:.2f}")
    ax1.plot(eq_net.index, norm_n, color="steelblue", lw=1.5,
             label=f"MR net    SR={m_net['sharpe']:.2f}  "
                   f"CAGR={m_net['cagr']*100:.1f}%  MaxDD={m_net['max_dd']*100:.1f}%")
    ax1.plot(eq_blend.index, norm_b, color="darkorange", lw=1.5, linestyle="--",
             label=f"50/50 blend SR={m_blend['sharpe']:.2f}  "
                   f"CAGR={m_blend['cagr']*100:.1f}%")
    dd = (eq_net - eq_net.cummax()) / eq_net.cummax()
    ax1.fill_between(eq_net.index, 1, 1 + dd * norm_n, alpha=0.15, color="crimson")
    ax1.set_title("MR — 3 speeds (10/20/40d), FDM=1.08  (normalised to 1.0)")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3); ax1.set_ylabel("Growth of $1")

    # 2. Annual returns
    ax2 = fig.add_subplot(gs[1, 0])
    colors = ["seagreen" if r >= 0 else "crimson" for r in yr.values]
    ax2.bar(yr.index, yr.values * 100, color=colors, alpha=0.8)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Annual Returns — MR net (%)"); ax2.grid(True, alpha=0.3)

    # 3. MR speed Sharpe
    ax3 = fig.add_subplot(gs[1, 1])
    sharpes = [res["speed_metrics"][n]["sharpe"] for n in MR_VARIANTS]
    ax3.bar([f"MR({n})" for n in MR_VARIANTS], sharpes,
            color=["seagreen" if s >= 0 else "crimson" for s in sharpes], alpha=0.8)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_title("Sharpe by MR Speed (gross)"); ax3.grid(True, alpha=0.3)

    # 4. Asset class weights
    ax4 = fig.add_subplot(gs[1, 2])
    grps = sorted(res["group_totals"].items(), key=lambda x: -x[1])
    ax4.barh([g for g, _ in grps], [v * 100 for _, v in grps],
             color="steelblue", alpha=0.8)
    ax4.set_xlabel("Weight (%)"); ax4.set_title("Asset-class Group Weights")
    ax4.tick_params(axis="y", labelsize=7); ax4.grid(True, alpha=0.3)

    # 5. Asset class P&L contribution
    ax5 = fig.add_subplot(gs[2, 0:2])
    ac_pnl: dict[str, float] = {}
    for tk, im in res["inst_metrics"].items():
        ac = im["asset_class"]
        ac_pnl[ac] = ac_pnl.get(ac, 0) + im["contribution_pct"]
    sorted_ac = sorted(ac_pnl.items(), key=lambda x: x[1])
    ax5.barh([a for a, _ in sorted_ac], [c for _, c in sorted_ac],
             color=["seagreen" if c >= 0 else "crimson" for _, c in sorted_ac],
             alpha=0.8)
    ax5.axvline(0, color="black", lw=0.8)
    ax5.set_title("P&L Contribution by Asset Class (% of total net P&L)")
    ax5.grid(True, alpha=0.3); ax5.tick_params(axis="y", labelsize=7)

    # 6. MR vs EWMAC per-instrument forecast correlation distribution
    ax6 = fig.add_subplot(gs[2, 2])
    corrs = [c for c in res["inst_corr"].values() if np.isfinite(c)]
    ax6.hist(corrs, bins=25, color="steelblue", alpha=0.7, edgecolor="white")
    ax6.axvline(res["pool_corr"], color="orange", lw=1.5,
                label=f"Pooled = {res['pool_corr']:+.3f}")
    ax6.axvline(0, color="black", lw=1.0, linestyle="--")
    ax6.axvline(0.30, color="crimson", lw=1.0, linestyle=":",
                label="Target ≤ 0.30")
    ax6.set_title("Per-instrument MR vs EWMAC Forecast Correlation")
    ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)
    ax6.set_xlabel("Pearson correlation")

    out = os.path.join(RESULTS_DIR, "etf_mr_equity.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Chart saved to {out}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol-target",         type=float, default=0.25)
    ap.add_argument("--capital",            type=float, default=250_000)
    ap.add_argument("--start",              default="2008-01-01")
    ap.add_argument("--no-plot",            action="store_true")
    ap.add_argument("--whole-share-shorts", action="store_true",
                    help="Round shorts to whole shares (models live Alpaca constraints)")
    args = ap.parse_args()

    with open(UNIVERSE_FILE) as f:
        u = json.load(f)
    tickers       = u["selected"]
    asset_classes = u.get("asset_classes", {})
    print(f"Universe: {len(tickers)} instruments")

    res = run_mr_backtest(tickers, args.capital, args.vol_target, args.start,
                          asset_classes, whole_share_shorts=args.whole_share_shorts)
    print_mr_report(res, args.vol_target, args.capital)
    if not args.no_plot:
        plot_mr_results(res, args.vol_target)


if __name__ == "__main__":
    main()
