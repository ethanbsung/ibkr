#!/usr/bin/env python3
"""
EWMAC trend-following backtest on the 101-ETF curated universe.

Full Carver framework with two-level instrument weighting:

  TIER LEVEL (5 tiers, asymmetric allocation):
    US Equity      20%  — sectors have genuine independent signals
    Intl Equity    15%  — country ETFs are correlated equity beta, down-weighted
    Fixed Income   25%  — best per-instrument Sharpe, genuinely diversifying
    Commodities    25%  — most genuinely independent instruments
    Alternatives   15%  — FX is clean; vol/crypto/thematic are noisy

  WITHIN-TIER LEVEL (Sharpe shrinkage):
    Estimate standalone Sharpe for each instrument.
    Shrink 75% toward equal weight within tier — heavy shrinkage because
    Sharpe estimates are noisy on short samples.
    Floor within-tier weight at 10% of equal weight so no instrument is
    fully excluded, just reduced.

  IDM: computed from actual correlation matrix and weight vector.
  Cap 2.5.

  Signal: 6 EWMAC speeds (2,8)(4,16)(8,32)(16,64)(32,128)(64,256)
  Vol:    blended — 70% EWMA(32) + 30% 10-yr rolling, 5% floor
  Forecast scaling: scalars estimated empirically, cap ±20, FDM from
                    inter-forecast correlation matrix.
  Buffer: 10% position buffer to reduce whipsaw turnover.

Usage:
  python3 etf/ewmac_backtest.py
  python3 etf/ewmac_backtest.py --vol-target 0.25 --start 2010-01-01 --capital 5000
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Constants ─────────────────────────────────────────────────────────────────
EWMAC_VARIANTS   = [(2,8),(4,16),(8,32),(16,64),(32,128),(64,256)]
VOL_SPAN_SHORT   = 32
VOL_WEIGHT_SHORT = 0.70
VOL_YEARS_LONG   = 10
VOL_WEIGHT_LONG  = 0.30
VOL_FLOOR        = 0.05
FORECAST_CAP     = 20.0
FORECAST_TARGET  = 10.0
BDAYS            = 256
WARMUP_BARS      = 270
IDM_CAP          = 2.5
GROSS_LEVERAGE_CAP = 1.9   # Reg-T 2x limit with small safety buffer; set None to disable

DATA_DIR       = "Data/etf"
UNIVERSE_FILE  = "Data/etf/etf_universe_greedy.json"
RESULTS_DIR    = "results"


# ── IDM lookup (Carver Table 4-4, used only for reporting) ────────────────────
def idm_from_count(n: int) -> float:
    table = [(1,1.0),(2,1.20),(3,1.48),(4,1.56),(5,1.70),
             (6,1.90),(7,2.10),(14,2.20),(24,2.30),(29,2.40)]
    for threshold, val in reversed(table):
        if n >= threshold:
            return val
    return IDM_CAP


# ── Volatility ────────────────────────────────────────────────────────────────
def blended_vol(returns: pd.Series) -> pd.Series:
    short_vol = returns.ewm(span=VOL_SPAN_SHORT, min_periods=2).std() * np.sqrt(BDAYS)
    long_window = VOL_YEARS_LONG * BDAYS
    long_vol = short_vol.rolling(window=long_window, min_periods=int(BDAYS * 0.5)).mean()
    long_vol = long_vol.fillna(short_vol)
    blended = VOL_WEIGHT_SHORT * short_vol + VOL_WEIGHT_LONG * long_vol
    return blended.clip(lower=VOL_FLOOR)


# ── Forecast ──────────────────────────────────────────────────────────────────
def normalised_ewmac(price: pd.Series, fast: int, slow: int,
                     annual_vol: pd.Series) -> pd.Series:
    f = price.ewm(span=fast, min_periods=fast, adjust=False).mean()
    s = price.ewm(span=slow, min_periods=slow, adjust=False).mean()
    daily_vol = annual_vol / np.sqrt(BDAYS)
    return (f - s) / (price * daily_vol).replace(0, np.nan)


def estimate_scalar(raw_series: pd.Series) -> float:
    avg_abs = raw_series.dropna().abs().mean()
    return FORECAST_TARGET / avg_abs if avg_abs > 0 else 1.0


def compute_fdm(forecast_df: pd.DataFrame) -> float:
    clean = forecast_df.dropna(how="all")
    if clean.shape[1] < 2 or len(clean) < 100:
        return 1.0
    C = clean.corr()
    n = len(C)
    w = np.ones(n) / n
    port_var = float(w @ C.values @ w)
    return min(1.0 / np.sqrt(max(port_var, 1e-6)), FORECAST_CAP / FORECAST_TARGET)


def build_forecasts(prices: pd.DataFrame, vols: pd.DataFrame,
                    scalars: dict) -> tuple[dict, pd.DataFrame]:
    """
    Returns:
      per_speed_fc: {(fast,slow): DataFrame of shape (dates, tickers)}
      combined_fc:  DataFrame of shape (dates, tickers) — FDM-weighted combination
    """
    per_speed_fc = {}
    for fast, slow in EWMAC_VARIANTS:
        speed_df = pd.DataFrame(index=prices.index)
        for tk in prices.columns:
            norm = normalised_ewmac(prices[tk], fast, slow, vols[tk])
            scalar = scalars.get((fast, slow), 1.0)
            speed_df[tk] = (norm * scalar).clip(-FORECAST_CAP, FORECAST_CAP)
        per_speed_fc[(fast, slow)] = speed_df

    # Combine across speeds with FDM per instrument
    combined_fc = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for tk in prices.columns:
        fc_for_fdm = pd.DataFrame({fs: per_speed_fc[fs][tk] for fs in EWMAC_VARIANTS})
        fdm = compute_fdm(fc_for_fdm)
        combined_fc[tk] = (fc_for_fdm.mean(axis=1) * fdm).clip(-FORECAST_CAP, FORECAST_CAP)
        combined_fc[tk].iloc[:WARMUP_BARS] = np.nan

    return per_speed_fc, combined_fc


# ── Handcraft weights ─────────────────────────────────────────────────────────
def compute_handcraft_weights(tickers: list[str], asset_classes: dict) -> dict[str, float]:
    """
    Carver handcrafting: equal weight within each asset-class group,
    equal weight across groups. No historical performance used.
    """
    groups: dict[str, list[str]] = {}
    for tk in tickers:
        g = asset_classes.get(tk, "OTHER")
        groups.setdefault(g, []).append(tk)

    n_groups = len(groups)
    weights: dict[str, float] = {}
    for g, members in groups.items():
        for tk in members:
            weights[tk] = (1.0 / n_groups) * (1.0 / len(members))

    total = sum(weights.values())
    return {tk: w / total for tk, w in weights.items()}


def compute_idm(weights: dict[str, float], returns: pd.DataFrame) -> float:
    """
    IDM = 1 / sqrt(w^T Σ w)
    where Σ is the full-sample pairwise return correlation matrix.
    """
    tickers = [tk for tk in weights if tk in returns.columns]
    w = np.array([weights[tk] for tk in tickers])
    w /= w.sum()   # renormalise in case some tickers are missing

    corr = returns[tickers].corr(min_periods=252).fillna(0.0).values
    np.fill_diagonal(corr, 1.0)

    port_var = float(w @ corr @ w)
    return min(1.0 / np.sqrt(max(port_var, 1e-6)), IDM_CAP)


def size_targets(capital: float, tickers, fc_row, vol_row,
                 weights: dict, idm: float, vol_target: float,
                 gross_cap=GROSS_LEVERAGE_CAP):
    """
    Single-bar position sizing — the SINGLE source of truth shared by the live
    strategy (etf/live/strategy_ewmac.py) and the backtest, so the two cannot
    drift on the formula or the gross-leverage cap.

    target_usd = capital · idm · weight · vol_target · (forecast/FORECAST_TARGET) / annual_vol

    fc_row / vol_row are per-ticker Series (or dict-like) for one date.
    Returns ({ticker: target_usd}, scale) where `scale` is the gross-cap factor
    applied (1.0 if the cap didn't bind).  The backtest's vectorized sizing in
    run_backtest mirrors this expression bar-by-bar.
    """
    targets = {}
    for tk in tickers:
        fc   = fc_row.get(tk, np.nan)
        avol = vol_row.get(tk, np.nan)
        if not np.isfinite(fc) or not np.isfinite(avol) or avol <= 0:
            targets[tk] = 0.0
            continue
        w = weights.get(tk, 0.0)
        targets[tk] = float(capital * idm * w * vol_target * (fc / FORECAST_TARGET) / avol)

    gross = sum(abs(v) for v in targets.values())
    scale = 1.0
    if gross_cap is not None and gross > gross_cap * capital and gross > 0:
        scale   = gross_cap * capital / gross
        targets = {tk: v * scale for tk, v in targets.items()}
    return targets, scale


# ── Position buffer ───────────────────────────────────────────────────────────
def apply_buffer(optimal: pd.Series, buffer_fraction: float = 0.10) -> pd.Series:
    opt_arr = optimal.values.astype(float)
    pos_arr = np.empty_like(opt_arr)
    pos_arr[:] = np.nan
    for i in range(len(opt_arr)):
        opt = opt_arr[i]
        if np.isnan(opt):
            pos_arr[i] = np.nan
            continue
        prev = pos_arr[i - 1] if i > 0 else np.nan
        if np.isnan(prev):
            pos_arr[i] = opt
            continue
        bw = abs(opt) * buffer_fraction
        lo, hi = opt - bw, opt + bw
        if   prev < lo: pos_arr[i] = lo
        elif prev > hi: pos_arr[i] = hi
        else:           pos_arr[i] = prev
    return pd.Series(pos_arr, index=optimal.index)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_prices(tickers: list[str], start: str) -> pd.DataFrame:
    frames = {}
    for tk in tickers:
        path = os.path.join(DATA_DIR, f"{tk.lower()}_1d_yf.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["time"]).set_index("time")["close"]
        df = df[df.index >= start]
        if len(df) > WARMUP_BARS:
            frames[tk] = df
    return pd.DataFrame(frames).sort_index()


# ── Metrics ───────────────────────────────────────────────────────────────────
def performance_metrics(equity: pd.Series) -> dict:
    ret   = equity.pct_change().dropna()
    years = len(ret) / BDAYS
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr  = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    avol  = ret.std() * np.sqrt(BDAYS)
    sharpe = cagr / avol if avol > 0 else 0
    dd    = (equity - equity.cummax()) / equity.cummax()
    maxdd = dd.min()
    cal   = cagr / abs(maxdd) if maxdd != 0 else 0
    monthly = equity.resample("ME").last().pct_change().dropna()
    pct_pos = (monthly > 0).mean() * 100
    return dict(cagr=cagr, ann_vol=avol, sharpe=sharpe,
                max_dd=maxdd, calmar=cal, pct_pos_months=pct_pos, years=years)


def yearly_returns(equity: pd.Series) -> pd.Series:
    ann = equity.resample("YE").last().pct_change().dropna()
    ann.index = ann.index.year
    return ann


# ── Main backtest ──────────────────────────────────────────────────────────────
def run_backtest(tickers: list[str], capital: float, vol_target: float,
                 start: str, asset_classes: dict,
                 whole_share_shorts: bool = False) -> dict:

    print(f"\nLoading prices for {len(tickers)} ETFs from {start}…")
    prices = load_prices(tickers, start)
    tickers = list(prices.columns)
    print(f"  {len(tickers)} ETFs loaded, {len(prices)} trading days")

    returns = prices.pct_change()
    vols    = pd.DataFrame({tk: blended_vol(returns[tk]) for tk in tickers})

    # ── Scalars ──────────────────────────────────────────────────────────────
    print("  Estimating forecast scalars…")
    scalars = {}
    for fast, slow in EWMAC_VARIANTS:
        raw_all = pd.concat([normalised_ewmac(prices[tk], fast, slow, vols[tk])
                              for tk in tickers]).dropna()
        scalars[(fast, slow)] = estimate_scalar(raw_all)
        print(f"    EWMAC({fast},{slow}) scalar: {scalars[(fast,slow)]:.2f}")

    # ── Forecasts ─────────────────────────────────────────────────────────────
    print("  Building forecasts…")
    per_speed_fc, combined_fc = build_forecasts(prices, vols, scalars)

    # ── Handcraft weights ─────────────────────────────────────────────────────
    weights = compute_handcraft_weights(tickers, asset_classes)
    idm     = compute_idm(weights, returns)
    print(f"  IDM (correlation-adjusted): {idm:.3f}")

    # Print group summary
    group_totals: dict[str, float] = {}
    for tk, w in weights.items():
        g = asset_classes.get(tk, "OTHER")
        group_totals[g] = group_totals.get(g, 0) + w
    print("  Asset-class group weights (handcraft equal weight):")
    for g, total in sorted(group_totals.items(), key=lambda x: -x[1]):
        print(f"    {g:<20} {total*100:.1f}%")

    # ── Position sizing ───────────────────────────────────────────────────────
    print("  Sizing positions…")
    raw_pos = pd.DataFrame(index=prices.index, columns=tickers, dtype=float)
    for tk in tickers:
        w   = weights.get(tk, 0.0)
        raw_pos[tk] = (capital * idm * w * vol_target
                       * (combined_fc[tk] / FORECAST_TARGET)
                       / vols[tk].replace(0, np.nan))

    # Apply gross leverage cap (same constraint as live Reg-T Alpaca account).
    # If gross exposure > cap × capital on any day, scale all positions down
    # proportionally — preserving relative weights but cutting overall size.
    if GROSS_LEVERAGE_CAP is not None:
        gross      = raw_pos.abs().sum(axis=1)
        scale      = (capital * GROSS_LEVERAGE_CAP / gross).clip(upper=1.0)
        raw_pos    = raw_pos.multiply(scale, axis=0)
        days_capped = (scale < 1.0).sum()
        avg_scale   = scale[scale < 1.0].mean() if days_capped > 0 else 1.0
        print(f"  Leverage cap {GROSS_LEVERAGE_CAP}x: binds on {days_capped} of "
              f"{len(scale)} days ({days_capped/len(scale)*100:.1f}%), "
              f"avg scale when capped: {avg_scale:.1%}")

    # Live realism: Alpaca rejects fractional short sales, so the live system
    # places shorts as WHOLE-share qty orders — sub-1-share shorts round to 0
    # (the book is realised long-biased).  Longs stay fractional (notional).
    # Model that here so backtest expectations match the live book.
    if whole_share_shorts:
        shares     = raw_pos / prices
        short_mask = raw_pos < 0
        raw_pos    = raw_pos.where(~short_mask, shares.round() * prices)
        n_short_bars = int(short_mask.sum().sum())
        n_zeroed     = int(((short_mask) & (shares.round() == 0)).sum().sum())
        print(f"  Whole-share shorts: {n_zeroed}/{n_short_bars} short "
              f"instrument-days round to 0 (long-biased realisation)")

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
    total_pnl_sum = portfolio_pnl.sum()
    for tk in tickers:
        inst_pnl    = pos_shifted[tk] * returns[tk]
        inst_equity = (1 + (inst_pnl / capital).fillna(0)).cumprod() * capital
        m = performance_metrics(inst_equity.dropna())
        m["weight"]     = weights.get(tk, 0.0)
        m["contribution_pct"] = (inst_pnl.sum() / total_pnl_sum * 100
                                  if total_pnl_sum != 0 else 0)
        m["asset_class"] = asset_classes.get(tk, "OTHER")
        inst_metrics[tk] = m

    # ── Per-speed attribution ─────────────────────────────────────────────────
    speed_metrics: dict[tuple, dict] = {}
    for fast, slow in EWMAC_VARIANTS:
        speed_pnl = pd.Series(0.0, index=prices.index)
        for tk in tickers:
            fc_s = per_speed_fc[(fast, slow)][tk].copy()
            fc_s.iloc[:WARMUP_BARS] = np.nan
            w   = weights.get(tk, 0.0)
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
        group_totals=group_totals,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────
def print_report(res: dict, vol_target: float, capital: float) -> None:
    eq = res["equity"]
    m  = performance_metrics(eq)
    yr = yearly_returns(eq)

    print("\n" + "=" * 72)
    print("  ETF EWMAC TREND — Handcraft equal weight (87 instruments)")
    print("=" * 72)
    print(f"  Capital:       ${capital:,.0f}")
    print(f"  Vol target:    {vol_target*100:.0f}%   IDM: {res['idm']:.3f}")
    print(f"  Period:        {eq.index[0].date()} → {eq.index[-1].date()}"
          f"  ({m['years']:.1f}y)")
    print(f"  CAGR:          {m['cagr']*100:.2f}%")
    print(f"  Annual vol:    {m['ann_vol']*100:.2f}%")
    print(f"  Sharpe:        {m['sharpe']:.3f}")
    print(f"  Max drawdown:  {m['max_dd']*100:.2f}%")
    print(f"  Calmar:        {m['calmar']:.3f}")
    print(f"  % months +ve:  {m['pct_pos_months']:.1f}%")

    print("\n  Year-by-year returns:")
    for yr_i, r in yr.items():
        bar  = "█" * int(abs(r) * 200)
        sign = "+" if r >= 0 else "-"
        print(f"    {yr_i}  {sign}{abs(r)*100:5.1f}%  {bar}")

    print("\n  Asset-class group weights (handcraft equal weight):")
    for g, total in sorted(res["group_totals"].items(), key=lambda x: -x[1]):
        bar = "█" * int(total * 200)
        print(f"    {g:<20} {total*100:5.1f}%  {bar}")

    print("\n  EWMAC speed Sharpe (standalone):")
    for (f, s), sm in res["speed_metrics"].items():
        bar = "█" * max(0, int(sm["sharpe"] * 20))
        print(f"    EWMAC({f:3},{s:4})  Sharpe {sm['sharpe']:+.3f}"
              f"  CAGR {sm['cagr']*100:+.1f}%  MaxDD {sm['max_dd']*100:.1f}%  {bar}")

    print("\n  Asset class P&L contribution (% of total P&L):")
    ac_pnl: dict[str, float] = {}
    for tk, im in res["inst_metrics"].items():
        ac = im["asset_class"]
        ac_pnl[ac] = ac_pnl.get(ac, 0) + im["contribution_pct"]
    for ac, pct in sorted(ac_pnl.items(), key=lambda x: -x[1]):
        bar  = "█" * int(abs(pct) / 5)
        sign = "+" if pct >= 0 else "-"
        print(f"    {ac:<18}  {sign}{abs(pct):5.1f}%  {bar}")

    print("\n  Top 15 instruments by portfolio weight:")
    ranked_w = sorted(res["inst_metrics"].items(), key=lambda x: -x[1]["weight"])
    for tk, im in ranked_w[:15]:
        print(f"    {tk:<6}  w={im['weight']*100:4.2f}%"
              f"  SR_port={im['sharpe']:+.2f}"
              f"  contrib={im['contribution_pct']:+.1f}%"
              f"  [{im['asset_class']}]")

    print("\n  Top 10 instruments by portfolio Sharpe:")
    ranked_sr = sorted(res["inst_metrics"].items(), key=lambda x: -x[1]["sharpe"])
    for tk, im in ranked_sr[:10]:
        print(f"    {tk:<6}  Sharpe {im['sharpe']:+.3f}  "
              f"CAGR {im['cagr']*100:+.1f}%  w={im['weight']*100:.2f}%"
              f"  [{im['asset_class']}]")

    print("\n  Bottom 10 instruments by portfolio Sharpe:")
    for tk, im in ranked_sr[-10:]:
        print(f"    {tk:<6}  Sharpe {im['sharpe']:+.3f}  "
              f"CAGR {im['cagr']*100:+.1f}%  w={im['weight']*100:.2f}%"
              f"  [{im['asset_class']}]")


def plot_results(res: dict, vol_target: float) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    eq = res["equity"]
    m  = performance_metrics(eq)
    yr = yearly_returns(eq)

    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    norm = eq / eq.iloc[0]
    ax1.plot(eq.index, norm, color="steelblue", lw=1.2,
             label=f"EWMAC  Sharpe {m['sharpe']:.2f}  CAGR {m['cagr']*100:.1f}%  MaxDD {m['max_dd']*100:.1f}%")
    dd = (eq - eq.cummax()) / eq.cummax()
    ax1.fill_between(eq.index, 1, 1 + dd * norm, alpha=0.18, color="crimson")
    ax1.set_title("ETF EWMAC — Handcraft equal weight, 87 instruments  (normalised to 1.0)")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3); ax1.set_ylabel("Growth of $1")

    # 2. Annual returns
    ax2 = fig.add_subplot(gs[1, 0])
    colors = ["seagreen" if r >= 0 else "crimson" for r in yr.values]
    ax2.bar(yr.index, yr.values * 100, color=colors, alpha=0.8)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Annual Returns (%)"); ax2.grid(True, alpha=0.3)

    # 3. EWMAC speed Sharpe
    ax3 = fig.add_subplot(gs[1, 1])
    labels  = [f"({f},{s})" for f, s in EWMAC_VARIANTS]
    sharpes = [res["speed_metrics"][(f, s)]["sharpe"] for f, s in EWMAC_VARIANTS]
    ax3.bar(labels, sharpes, color=["seagreen" if s >= 0 else "crimson" for s in sharpes], alpha=0.8)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_title("Sharpe by EWMAC Speed"); ax3.grid(True, alpha=0.3)

    # 4. Group weights (bar)
    ax4 = fig.add_subplot(gs[1, 2])
    grps = sorted(res["group_totals"].items(), key=lambda x: -x[1])
    ax4.barh([g for g, _ in grps], [v * 100 for _, v in grps],
             color="steelblue", alpha=0.8)
    ax4.set_xlabel("Weight (%)")
    ax4.set_title("Asset-class Group Weights")
    ax4.tick_params(axis="y", labelsize=7)
    ax4.grid(True, alpha=0.3)

    # 5. Asset class contribution
    ax5 = fig.add_subplot(gs[2, 0:2])
    ac_pnl: dict[str, float] = {}
    for tk, im in res["inst_metrics"].items():
        ac = im["asset_class"]
        ac_pnl[ac] = ac_pnl.get(ac, 0) + im["contribution_pct"]
    sorted_ac = sorted(ac_pnl.items(), key=lambda x: x[1])
    ax5.barh([a for a, _ in sorted_ac], [c for _, c in sorted_ac],
             color=["seagreen" if c >= 0 else "crimson" for _, c in sorted_ac], alpha=0.8)
    ax5.axvline(0, color="black", lw=0.8)
    ax5.set_title("P&L Contribution by Asset Class (%)"); ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis="y", labelsize=7)

    # 6. Instrument Sharpe distribution
    ax6 = fig.add_subplot(gs[2, 2])
    all_sr = [im["sharpe"] for im in res["inst_metrics"].values()]
    ax6.hist(all_sr, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
    ax6.axvline(np.mean(all_sr), color="orange", lw=1.5,
                label=f"Mean {np.mean(all_sr):.2f}")
    ax6.axvline(0, color="crimson", lw=1, linestyle="--")
    ax6.set_title("Per-instrument Sharpe Distribution")
    ax6.legend(); ax6.grid(True, alpha=0.3)

    out = os.path.join(RESULTS_DIR, "etf_ewmac_equity.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Chart saved to {out}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol-target", type=float, default=0.25)
    ap.add_argument("--capital",    type=float, default=5_000)
    ap.add_argument("--start",      default="2008-01-01")
    ap.add_argument("--no-plot",    action="store_true")
    ap.add_argument("--whole-share-shorts", action="store_true",
                    help="Round shorts to whole shares (models the live Alpaca "
                         "long-bias: sub-1-share shorts drop to 0)")
    args = ap.parse_args()

    with open(UNIVERSE_FILE) as f:
        u = json.load(f)
    tickers       = u["selected"]
    asset_classes = u.get("asset_classes", {})
    print(f"Universe: {len(tickers)} instruments")

    res = run_backtest(tickers, args.capital, args.vol_target, args.start,
                       asset_classes, whole_share_shorts=args.whole_share_shorts)
    print_report(res, args.vol_target, args.capital)
    if not args.no_plot:
        plot_results(res, args.vol_target)


if __name__ == "__main__":
    main()
