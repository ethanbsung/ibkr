"""
backtest_dynamic.py — EWMAC backtest driven by Carver's dynamic portfolio
optimisation (AFTS dynamic-optimisation section; equations in ibkr_fut/calcs.txt
lines 217-332).

Unlike backtest_ewmac.py (which sizes each instrument independently with a per-
instrument buffer band), this runs ONE joint daily optimisation across the whole
universe: each day it computes the ideal unrounded position for every live
instrument, then picks the integer position vector that minimises the portfolio
tracking error, penalising trading costs and buffering in tracking-error space.
This is what makes a small account hold a sensible, diversified integer subset of
the ~99-instrument Jumbo instead of naively rounding each instrument to zero.

Reuses the validated Jumbo signal/sizing pipeline from backtest_ewmac.py
(_pst_spec, instrument_signals) and the optimisation primitives in dynamic_opt.py.
"""

import sys
import os
from dataclasses import replace as dc_replace
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibkr_fut.foundations import (
    ANNUAL_DAYS,
    compute_corr_matrix,
    handcraft_weights,
    performance_stats,
)
from ibkr_fut.backtest_ewmac import (
    pst,
    _pst_spec,
    instrument_signals,
    TARGET_RISK,
    IDM_CAP,
)
from ibkr_fut.dynamic_opt import CovarianceEstimator, optimise_positions

# ── Parameters ───────────────────────────────────────────────────────────────────

CAPITAL_SWEEP = [100_000, 250_000, 1_000_000, 50_000_000]
RESULTS_DIR   = "ibkr_fut/results"


# ── Instrument selection filter ───────────────────────────────────────────────────

def get_eligible_set(universe: dict[str, str]) -> set[str]:
    """
    Apply Carver's instrument selection filters (cost, too-safe, history) and
    return the set of eligible instrument names.  Volume filters are skipped
    when no cached volume data exists (instruments are assumed to pass).

    Returns a set of instrument names that pass all applicable filters.
    """
    from ibkr_fut.instrument_selection import build_selection_table
    from ibkr_fut.volume_collector import load_cache

    volume_cache = load_cache()
    tbl = build_selection_table(pst, universe, volume_cache)
    eligible = set(tbl.index[tbl["eligible"]].tolist())

    # Report what was filtered out
    ineligible = tbl[~tbl["eligible"]]
    if len(ineligible) > 0:
        print(f"  Instrument selection filter: {len(eligible)}/{len(tbl)} pass")
        by_reason: dict[str, list[str]] = {}
        for instr, row in ineligible.iterrows():
            key = row["reason"] or "unknown"
            # Simplify reason to the first failure type for grouping
            if "sr_cost" in key:
                tag = "expensive (SR cost)"
            elif "vol=" in key and "%" in key:
                tag = "too safe (low vol)"
            elif "history" in key:
                tag = "insufficient history"
            elif "no price data" in key:
                tag = "no price data"
            else:
                tag = key
            by_reason.setdefault(tag, []).append(instr)
        for reason, instrs in sorted(by_reason.items()):
            print(f"    {reason}: {sorted(instrs)}")
    else:
        print(f"  Instrument selection filter: all {len(eligible)} instruments pass")

    return eligible


def _build_universe(
    instruments: list[str],
    tradable_set: set | None = None,
    lookback_days: int | None = None,
    signal_fn=None,
    return_components: bool = False,
):
    """
    Build aligned numpy panels for every instrument with a valid signal.  This is the ONE
    universe builder shared by EWMAC trend (default), basic carry, and combined carry+trend
    — they differ only in `signal_fn`; vol, correlation, handcraft weights, IDM, and the
    covariance estimator are strategy-independent.  Returns a dict of arrays (T x n) plus
    per-instrument scalars and the covariance estimator, or None if nothing is tradable.

    signal_fn(spec, mp) -> {"sigma","forecast",...} | None : per-instrument signal builder.
    Default is the EWMAC trend `instrument_signals` (which ignores `mp`).  Pass
    `carry_instrument_signals` / `carry_trend_instrument_signals` for the other strategies;
    those need the `mp` (multiple_prices) term structure, which is fetched here and passed
    in (None if the instrument has no term-structure file — the carry builders return None).

    tradable_set: optional set of instrument names we are allowed to trade. The full
    `instruments` list forms the optimisation universe (drives covariance + target
    weights); names outside tradable_set are held at min=max=current so the optimiser
    transfers their risk onto correlated tradable markets (calcs.txt lines 326-330).
    None => every instrument with valid signals is tradable (original behaviour).

    lookback_days: if set, trim all price series to at most this many calendar days
    before today before building the panels.  Use ~4000 for live runs to bound memory
    while fully covering the blended-vol long window (a 2520-TRADING-day rolling average
    ≈ 3650 calendar days, so 4000 calendar days covers it with margin).  The EWMA
    covariance (25wk/32d) and handcraft correlation (full history, read straight from
    pst) don't depend on this window.  None => full history (backtest behaviour).

    return_components: also return (specs, signals) so callers (e.g. the proportion sweep)
    can recompute just the forecast panel without rebuilding the covariance model.
    """
    if signal_fn is None:
        signal_fn = lambda spec, mp: instrument_signals(spec)   # EWMAC trend default

    cutoff_date: pd.Timestamp | None = None
    if lookback_days is not None:
        cutoff_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=lookback_days)

    specs, signals = {}, {}
    skipped: list[tuple[str, str]] = []
    for instr in instruments:
        spec = _pst_spec(instr)
        if spec is None:
            skipped.append((instr, "no spec (no data/metadata or short history)"))
            continue
        # Term structure for carry signals; None for instruments lacking the file (trend
        # ignores it, carry returns None). _pst_spec already reads this file, so cached.
        try:
            mp = pst.multiple_prices(instr)
        except (FileNotFoundError, KeyError):
            mp = None
        if cutoff_date is not None:
            spec = dc_replace(
                spec,
                prices=spec.prices[spec.prices.index >= cutoff_date],
                raw_price=spec.raw_price[spec.raw_price.index >= cutoff_date],
            )
            if mp is not None:
                mp = mp[mp.index >= cutoff_date]
        sig = signal_fn(spec, mp)
        if sig is None:
            skipped.append((instr, "no eligible signal / invalid vol"))
            continue
        specs[instr] = spec
        signals[instr] = sig

    names = list(specs.keys())
    if not names:
        return None
    print(f"  {len(names)} instruments with valid signals"
          + (f" (lookback {lookback_days}d)" if lookback_days else ""))
    if skipped:
        print(f"  {len(skipped)} skipped: "
              + ", ".join(f"{i} ({r})" for i, r in skipped))

    # Common daily index = union of all instrument price indices.
    idx = None
    for instr in names:
        ix = specs[instr].prices.index
        idx = ix if idx is None else idx.union(ix)
    idx = idx.sort_values()
    T, n = len(idx), len(names)

    price    = np.full((T, n), np.nan)   # back-adjusted (P&L driver)
    raw      = np.full((T, n), np.nan)   # raw contract price (sizing denominator)
    fx       = np.full((T, n), np.nan)
    sigma    = np.full((T, n), np.nan)   # blended annualised vol (sizing)
    forecast = np.full((T, n), np.nan)
    mult     = np.zeros(n)
    spread   = np.zeros(n)
    commission = np.zeros(n)

    for j, instr in enumerate(names):
        s = specs[instr]
        price[:, j]    = s.prices.reindex(idx).values
        raw[:, j]      = s.raw_price.reindex(idx).ffill().values
        fx[:, j]       = s.fx.reindex(idx, method="ffill").values
        sigma[:, j]    = signals[instr]["sigma"].reindex(idx).values
        forecast[:, j] = signals[instr]["forecast"].reindex(idx).values
        mult[j]        = s.mult
        spread[j]      = s.spread
        commission[j]  = s.commission

    # Static handcraft weights + correlation (for the unrounded-N weight_i and IDM).
    print("  computing correlation matrix + handcraft weights...")
    corr_matrix = compute_corr_matrix(names, pst)
    weights     = handcraft_weights(names, corr_matrix)
    W = np.array([weights.get(i, 0.0) for i in names])
    C = corr_matrix.reindex(index=names, columns=names).values.copy()
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)

    # Time-varying covariance estimator (weekly EWMA corr + daily EWMA vol).
    print("  building covariance estimator...")
    est = CovarianceEstimator(
        {i: specs[i].prices for i in names},
        {i: specs[i].raw_price for i in names},
    )

    if tradable_set is None:
        tradable = np.ones(n, dtype=bool)
    else:
        tradable = np.array([i in tradable_set for i in names], dtype=bool)
        print(f"  {int(tradable.sum())}/{n} instruments tradable "
              f"(rest held at min=max=current)")

    uni = dict(
        names=names, idx=idx, price=price, raw=raw, fx=fx, sigma=sigma,
        forecast=forecast, mult=mult, spread=spread, commission=commission,
        W=W, C=C, est=est, tradable=tradable,
    )
    if return_components:
        return uni, specs, signals
    return uni


NAIVE_BUFFER_FRAC = 0.10   # per-instrument buffer band for the naive baseline  [calcs line 113]


def _simulate(uni: dict, capital: float, target_risk: float,
              use_costs: bool, use_buffering: bool, mode: str = "dynamic",
              rolling_capital: bool = True) -> dict:
    """
    Run the joint daily loop for one capital level.
    mode="dynamic": Carver dynamic optimisation (greedy + cost + buffer).
    mode="naive":   independent per-instrument rounding with a buffer band — the
                    baseline dynamic optimisation is meant to beat at small capital.
    rolling_capital: if True, position sizing uses the compounded equity (P&L
                     reinvested) rather than fixed starting capital. This better
                     reflects real account growth but makes the backtest path-
                     dependent and non-linear.
    """
    names, idx = uni["names"], uni["idx"]
    price, raw, fx = uni["price"], uni["raw"], uni["fx"]
    sigma, forecast = uni["sigma"], uni["forecast"]
    mult, spread, commission = uni["mult"], uni["spread"], uni["commission"]
    W, C, est = uni["W"], uni["C"], uni["est"]
    tradable = uni["tradable"]
    T, n = len(idx), len(names)
    idx_values = idx.values

    pos = np.zeros(n)               # full-universe integer positions held
    pnl_list = np.zeros(T)
    total_cost = 0.0
    abs_changes = 0.0               # one-way contracts traded (turnover numerator)
    abs_pos_sum = 0.0
    n_held_sum = 0.0
    te_sum, te_count = 0.0, 0       # tracking error of held vs ideal (diagnostic)
    gross_lev = np.zeros(T)         # realised gross leverage of held book (diagnostic)
    idm_cache: dict[tuple, tuple] = {}

    equity_running = float(capital)  # tracks compounded equity when rolling_capital=True

    for t in range(T):
        p, r, f = price[t], raw[t], fx[t]
        s, fc = sigma[t], forecast[t]
        valid = (
            ~np.isnan(p) & ~np.isnan(r) & (r > 0)
            & ~np.isnan(s) & (s > 0) & ~np.isnan(f) & ~np.isnan(fc)
        )

        # Step 1: P&L from yesterday's positions over today's back-adjusted change.
        daily_pnl = 0.0
        if t > 0:
            m = valid & ~np.isnan(price[t - 1])
            if m.any():
                dp = p[m] - price[t - 1][m]
                daily_pnl = float(np.sum(pos[m] * dp * mult[m] * f[m]))

        # Rolling capital: use yesterday's compounded equity for today's sizing.
        # Update after P&L is known so sizing always uses start-of-day equity.
        sizing_capital = equity_running if rolling_capital else capital
        if rolling_capital and t > 0:
            equity_running = max(equity_running + daily_pnl, 1.0)  # floor at $1

        live_idx = np.flatnonzero(valid)
        if live_idx.size == 0:
            pnl_list[t] = daily_pnl
            abs_pos_sum += np.abs(pos).sum()
            continue

        # Step 2: live-universe renormalised weights + IDM (cached per liveness pattern).
        key = tuple(live_idx.tolist())
        cached = idm_cache.get(key)
        if cached is None:
            w_live = W[live_idx]
            ssum = float(w_live.sum())
            w_n = w_live / ssum if ssum > 0 else np.full(live_idx.size, 1.0 / live_idx.size)
            Csub = C[np.ix_(live_idx, live_idx)]
            var = float(w_n @ Csub @ w_n)
            idm_t = min(1.0 / np.sqrt(var), IDM_CAP) if var > 0 else 1.0
            idm_cache[key] = (w_n, idm_t)
        else:
            w_n, idm_t = cached

        ml, rl, fl, sl, fcl = mult[live_idx], r[live_idx], f[live_idx], s[live_idx], fc[live_idx]

        # Step 3: ideal unrounded positions  [calcs line 214]
        N_unrounded = (fcl * sizing_capital * idm_t * w_n * target_risk) / (10.0 * ml * rl * fl * sl)
        weight_per_contract = ml * rl * fl / sizing_capital                   # [calcs line 226]
        # One-way cost per contract: half-spread crossing (SpreadCost) + half the
        # round-trip commission. (Was 2*spread + full commission = 2x the true cost.)
        cost_per_contract = (spread[live_idx] * ml + commission[live_idx] / 2.0) * fl

        cov = est.covariance_by_index(pd.Timestamp(idx_values[t]), live_idx)
        prev_live = pos[live_idx]

        # Step 4: integer positions, by mode.
        if mode == "naive":
            # Per-instrument rounding with a buffer band (no joint optimisation).
            N_avg = (sizing_capital * idm_t * w_n * target_risk) / (ml * rl * fl * sl)
            B = NAIVE_BUFFER_FRAC * np.abs(N_avg)
            lower = np.round(N_unrounded - B)
            upper = np.round(N_unrounded + B)
            N_star = np.where(prev_live < lower, lower,
                              np.where(prev_live > upper, upper, prev_live))
        else:
            N_star = optimise_positions(
                covariance=cov,
                weight_per_contract=weight_per_contract,
                optimal_unrounded_positions=N_unrounded,
                previous_positions=prev_live,
                cost_per_contract=cost_per_contract,
                capital=sizing_capital,
                target_risk=target_risk,
                use_costs=use_costs,
                use_buffering=use_buffering,
                tradable=tradable[live_idx],
            )

        # Step 5: trade costs on the change, then carry positions forward.
        trades = np.abs(N_star - prev_live)
        trade_cost = float(np.sum(trades * cost_per_contract))
        total_cost += trade_cost
        abs_changes += float(trades.sum())
        pos[live_idx] = N_star

        # Diagnostics: tracking error of held vs ideal (annualised std).
        e = (N_star - N_unrounded) * weight_per_contract
        te = float(np.sqrt(max(e @ cov @ e, 0.0)))
        te_sum += te
        te_count += 1
        n_held_sum += int(np.count_nonzero(N_star))

        # Realised gross leverage = total notional / capital (calcs.txt line 33).
        gross_lev[t] = float(np.sum(np.abs(N_star) * ml * rl * fl) / capital)

        abs_pos_sum += np.abs(pos).sum()
        pnl_list[t] = daily_pnl - trade_cost

    pnl_s = pd.Series(pnl_list, index=idx)

    if rolling_capital:
        # Reconstruct equity by compounding from starting capital.
        equity = pd.Series(index=idx, dtype=float)
        eq = float(capital)
        for t2, pnl in enumerate(pnl_list):
            eq = max(eq + pnl, 1.0)
            equity.iloc[t2] = eq
        daily_returns = equity.pct_change().fillna(0.0)
        # Annualise costs as % of average equity over the run.
        avg_equity = float(equity.mean())
        costs_pct = (total_cost / avg_equity / (T / ANNUAL_DAYS)) * 100 if T else 0.0
    else:
        daily_returns = pnl_s / capital
        equity = capital * (1.0 + daily_returns).cumprod()
        costs_pct = (total_cost / capital / (T / ANNUAL_DAYS)) * 100 if T else 0.0

    years = T / ANNUAL_DAYS
    avg_abs_N = abs_pos_sum / T if T else 1.0
    turnover = (abs_changes / 2.0) / avg_abs_N / years if years and avg_abs_N else 0.0

    stats = performance_stats(equity, daily_returns, costs_pct=costs_pct, turnover=turnover)
    stats["avg_instruments_held"] = round(n_held_sum / te_count, 1) if te_count else 0.0
    stats["avg_tracking_error_pct"] = round(100 * te_sum / te_count, 2) if te_count else 0.0
    gl = gross_lev[gross_lev > 0]
    stats["gross_lev_mean"] = round(float(gl.mean()), 1) if gl.size else 0.0
    stats["gross_lev_p95"]  = round(float(np.percentile(gl, 95)), 1) if gl.size else 0.0
    stats["gross_lev_max"]  = round(float(gl.max()), 1) if gl.size else 0.0

    return {"equity": equity, "daily_returns": daily_returns, "stats": stats}


def position_history(
    uni: dict,
    capital: float,
    target_risk: float = TARGET_RISK,
    lookback_years: int = 10,
) -> pd.DataFrame:
    """
    Run the dynamic-opt simulation and record the integer position held for every
    instrument on every day in the last `lookback_years` years.

    Returns a DataFrame (dates × instruments) of integer positions, restricted to
    dates where at least one position is held.  Instruments never held are dropped.

    The full universe history is still simulated from the beginning so that the
    optimiser's carry-forward state is correct at the start of the window; only
    the output is trimmed to the lookback window.
    """
    names, idx = uni["names"], uni["idx"]
    price, raw, fx = uni["price"], uni["raw"], uni["fx"]
    sigma, forecast = uni["sigma"], uni["forecast"]
    mult, spread, commission = uni["mult"], uni["spread"], uni["commission"]
    W, C, est = uni["W"], uni["C"], uni["est"]
    tradable = uni["tradable"]
    T, n = len(idx), len(names)
    idx_values = idx.values

    window_start = pd.Timestamp.today().normalize() - pd.DateOffset(years=lookback_years)

    pos       = np.zeros(n)
    pos_log   = np.full((T, n), 0, dtype=np.int32)
    idm_cache: dict[tuple, tuple] = {}

    for t in range(T):
        p, r, f = price[t], raw[t], fx[t]
        s, fc = sigma[t], forecast[t]
        valid = (
            ~np.isnan(p) & ~np.isnan(r) & (r > 0)
            & ~np.isnan(s) & (s > 0) & ~np.isnan(f) & ~np.isnan(fc)
        )
        live_idx = np.flatnonzero(valid)
        if live_idx.size == 0:
            pos_log[t] = pos.astype(np.int32)
            continue

        key = tuple(live_idx.tolist())
        cached = idm_cache.get(key)
        if cached is None:
            w_live = W[live_idx]
            ssum   = float(w_live.sum())
            w_n    = w_live / ssum if ssum > 0 else np.full(live_idx.size, 1.0 / live_idx.size)
            Csub   = C[np.ix_(live_idx, live_idx)]
            var    = float(w_n @ Csub @ w_n)
            idm_t  = min(1.0 / np.sqrt(var), IDM_CAP) if var > 0 else 1.0
            idm_cache[key] = (w_n, idm_t)
        else:
            w_n, idm_t = cached

        ml_l = mult[live_idx]; rl = r[live_idx]; fl = f[live_idx]
        sl = s[live_idx]; fcl = fc[live_idx]

        N_unrounded      = (fcl * capital * idm_t * w_n * target_risk) / (10.0 * ml_l * rl * fl * sl)
        weight_per_contract = ml_l * rl * fl / capital
        cost_per_contract   = (spread[live_idx] * ml_l + commission[live_idx] / 2.0) * fl
        cov = est.covariance_by_index(pd.Timestamp(idx_values[t]), live_idx)

        N_star = optimise_positions(
            covariance=cov,
            weight_per_contract=weight_per_contract,
            optimal_unrounded_positions=N_unrounded,
            previous_positions=pos[live_idx],
            cost_per_contract=cost_per_contract,
            capital=capital,
            target_risk=target_risk,
            use_costs=True,
            use_buffering=True,
            tradable=tradable[live_idx],
        )

        pos[live_idx] = N_star
        pos_log[t]    = pos.astype(np.int32)

    df = pd.DataFrame(pos_log, index=idx, columns=names)

    # Trim to lookback window
    df = df[df.index >= window_start]

    # Keep only instruments that were ever held (non-zero)
    df = df.loc[:, (df != 0).any()]

    # Keep only days where at least one position is held
    df = df[(df != 0).any(axis=1)]

    return df


def summarise_position_history(df: pd.DataFrame, universe: dict[str, str]) -> None:
    """
    Print a human-readable summary of the position history DataFrame:
      - Per-instrument: % of days held, avg position when held, direction breakdown
      - Monthly heatmap of number of instruments held
    """
    instruments = df.columns.tolist()
    total_days  = len(df)

    print(f"\n{'=' * 80}")
    print(f"POSITION HISTORY SUMMARY  |  {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Total trading days in window: {total_days}")
    print(f"{'=' * 80}")

    # Per-instrument breakdown
    rows = []
    for instr in instruments:
        s       = df[instr]
        held    = s[s != 0]
        pct     = 100.0 * len(held) / total_days
        avg_pos = held.mean() if len(held) else 0.0
        n_long  = (held > 0).sum()
        n_short = (held < 0).sum()
        rows.append({
            "Instrument":  instr,
            "Class":       universe.get(instr, ""),
            "Days held":   len(held),
            "% days":      round(pct, 1),
            "Avg pos":     round(avg_pos, 2),
            "Long days":   n_long,
            "Short days":  n_short,
            "Max long":    int(held.max()) if len(held) else 0,
            "Max short":   int(held.min()) if len(held) else 0,
        })

    rows.sort(key=lambda x: -x["Days held"])
    print(f"\n{'Instrument':<20} {'Class':<8} {'Days':>5} {'%Days':>6} "
          f"{'AvgPos':>7} {'LongD':>6} {'ShortD':>7} {'MaxL':>5} {'MaxS':>5}")
    print("-" * 78)
    for r in rows:
        print(f"  {r['Instrument']:<18} {r['Class']:<8} {r['Days held']:>5} {r['% days']:>6.1f} "
              f"{r['Avg pos']:>7.2f} {r['Long days']:>6} {r['Short days']:>7} "
              f"{r['Max long']:>5} {r['Max short']:>5}")

    # Quarterly count of instruments held
    print(f"\nQuarterly avg # instruments held:")
    quarterly = df.astype(bool).sum(axis=1).resample("QE").mean().round(1)
    for dt, val in quarterly.items():
        print(f"  {dt.strftime('%Y Q%q')}: {val:.1f}")


def run_dynamic(
    instruments: list[str] | dict,
    capitals: list[float] = CAPITAL_SWEEP,
    target_risk: float = TARGET_RISK,
    use_costs: bool = True,
    use_buffering: bool = True,
    label: str = "(Jumbo)",
    tradable_set: set | None = None,
    apply_filter: bool = True,
    rolling_capital: bool = True,
) -> dict:
    """Build the universe once, then run the dynamic-opt backtest at each capital level.

    apply_filter: if True, run instrument_selection filters and restrict
                  trading to the eligible set (ineligible instruments are kept
                  in the universe for covariance/weight purposes but held at 0).
    """
    universe_dict = instruments if isinstance(instruments, dict) else {i: "" for i in instruments}
    names = list(universe_dict.keys())
    print(f"Loading {len(names)} instruments {label} for dynamic optimisation...")
    if apply_filter:
        eligible = get_eligible_set(universe_dict)
        tradable_set = eligible if tradable_set is None else tradable_set & eligible

    uni = _build_universe(names, tradable_set)
    if uni is None:
        print("No tradable instruments.")
        return {}

    rows, curves = [], {}
    for cap in capitals:
        print(f"\nSimulating capital = ${cap:,.0f} ...")
        res = _simulate(uni, cap, target_risk, use_costs, use_buffering,
                        rolling_capital=rolling_capital)
        st = res["stats"]
        curves[cap] = res["equity"]
        rows.append({
            "Capital":      f"${cap:,.0f}",
            "Mean Ret %":   st["mean_annual_return_pct"],
            "Std Dev %":    st["std_dev_pct"],
            "SR":           st["sharpe_ratio"],
            "Costs %":      st["costs_pct"],
            "Turnover":     st["turnover"],
            "Max DD %":     st["max_drawdown_pct"],
            "Skew":         st["skew"],
            "Avg #Held":    st["avg_instruments_held"],
            "Lev mean":     st["gross_lev_mean"],
            "Lev p95":      st["gross_lev_p95"],
            "Lev max":      st["gross_lev_max"],
        })

    print("\n" + "=" * 110)
    print(f"DYNAMIC OPTIMISATION — EWMAC {label} | costs={use_costs} buffering={use_buffering}")
    print("=" * 110)
    print(tabulate(rows, headers="keys", floatfmt=".2f", tablefmt="simple"))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    for cap, eq in curves.items():
        ax.plot(eq.index, eq.values, linewidth=0.8, label=f"${cap:,.0f}")
    ax.set_yscale("log")
    ax.set_title(f"Dynamic-Optimisation EWMAC {label} — rolling equity by capital level")
    ax.set_ylabel("Portfolio equity ($, log scale)")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{RESULTS_DIR}/dynamic_sweep.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\nEquity curves saved to {out}")

    return {"rows": rows, "curves": curves, "universe": uni}


def run_comparison(
    instruments: list[str] | dict,
    capitals: list[float] = (100_000, 250_000, 1_000_000),
    target_risk: float = TARGET_RISK,
    label: str = "(Carver Jumbo)",
    tradable_set: set | None = None,
    apply_filter: bool = True,
) -> dict:
    """Dynamic optimisation vs naive per-instrument rounding at matched capital levels."""
    universe_dict = instruments if isinstance(instruments, dict) else {i: "" for i in instruments}
    names = list(universe_dict.keys())
    print(f"Loading {len(names)} instruments {label} for dynamic-vs-naive comparison...")

    if apply_filter:
        eligible = get_eligible_set(universe_dict)
        tradable_set = eligible if tradable_set is None else tradable_set & eligible

    uni = _build_universe(names, tradable_set)
    if uni is None:
        print("No tradable instruments.")
        return {}

    rows = []
    for cap in capitals:
        for mode in ("naive", "dynamic"):
            print(f"\nSimulating {mode} @ ${cap:,.0f} ...")
            res = _simulate(uni, cap, target_risk, use_costs=True,
                            use_buffering=(mode == "dynamic"), mode=mode)
            st = res["stats"]
            rows.append({
                "Capital":    f"${cap:,.0f}",
                "Mode":       mode,
                "Mean Ret %": st["mean_annual_return_pct"],
                "Std Dev %":  st["std_dev_pct"],
                "SR":         st["sharpe_ratio"],
                "Costs %":    st["costs_pct"],
                "Turnover":   st["turnover"],
                "Max DD %":   st["max_drawdown_pct"],
                "Avg #Held":  st["avg_instruments_held"],
                "Avg TE %":   st["avg_tracking_error_pct"],
            })

    print("\n" + "=" * 110)
    print(f"DYNAMIC vs NAIVE — EWMAC {label}")
    print("=" * 110)
    print(tabulate(rows, headers="keys", floatfmt=".2f", tablefmt="simple"))
    return {"rows": rows, "universe": uni}


def run_snapshot(
    instruments: dict[str, str],
    capital: float,
    target_risk: float = TARGET_RISK,
    apply_filter: bool = True,
    use_costs: bool = False,
    use_buffering: bool = False,
) -> None:
    """
    Print today's ideal and integer positions for a given capital level.
    Shows what the portfolio looks like RIGHT NOW given current prices and signals.
    """
    from ibkr_fut.instrument_selection import build_selection_table
    from ibkr_fut.volume_collector import load_cache

    universe_dict = instruments
    names = list(universe_dict.keys())

    eligible_set = None
    if apply_filter:
        eligible_set = get_eligible_set(universe_dict)

    uni = _build_universe(names)
    if uni is None:
        print("No instruments with valid signals.")
        return

    # Use only the last row of data (today's snapshot)
    idx = uni["idx"]
    T   = len(idx)
    t   = T - 1  # last day

    p   = uni["price"][t]
    r   = uni["raw"][t]
    f   = uni["fx"][t]
    s   = uni["sigma"][t]
    fc  = uni["forecast"][t]
    ml  = uni["mult"]
    sp  = uni["spread"]
    com = uni["commission"]
    W   = uni["W"]
    C   = uni["C"]
    est = uni["est"]
    nms = uni["names"]

    valid = (
        ~np.isnan(p) & ~np.isnan(r) & (r > 0)
        & ~np.isnan(s) & (s > 0) & ~np.isnan(f) & ~np.isnan(fc)
    )
    live_idx = np.flatnonzero(valid)

    w_live = W[live_idx]
    ssum   = float(w_live.sum())
    w_n    = w_live / ssum if ssum > 0 else np.full(live_idx.size, 1.0 / live_idx.size)
    Csub   = C[np.ix_(live_idx, live_idx)]
    var    = float(w_n @ Csub @ w_n)
    idm_t  = min(1.0 / np.sqrt(var), IDM_CAP) if var > 0 else 1.0

    ml_l  = ml[live_idx]
    rl    = r[live_idx]
    fl    = f[live_idx]
    sl    = s[live_idx]
    fcl   = fc[live_idx]

    N_ideal = (fcl * capital * idm_t * w_n * target_risk) / (10.0 * ml_l * rl * fl * sl)
    wpc     = ml_l * rl * fl / capital
    cpc     = (sp[live_idx] * ml_l + com[live_idx] / 2.0) * fl

    cov    = est.covariance_by_index(pd.Timestamp(idx[-1]), live_idx)
    tradable_mask = np.ones(live_idx.size, dtype=bool)
    if eligible_set is not None:
        tradable_mask = np.array([nms[i] in eligible_set for i in live_idx], dtype=bool)

    N_int = optimise_positions(
        covariance=cov,
        weight_per_contract=wpc,
        optimal_unrounded_positions=N_ideal,
        previous_positions=np.zeros(live_idx.size),
        cost_per_contract=cpc,
        capital=capital,
        target_risk=target_risk,
        use_costs=use_costs,
        use_buffering=use_buffering,
        tradable=tradable_mask,
    )

    # Build result table
    filter_note = " [with selection filter]" if apply_filter else ""
    print(f"\n{'=' * 90}")
    print(f"SNAPSHOT — {idx[-1].date()}  |  capital=${capital:,.0f}  |  IDM={idm_t:.3f}{filter_note}")
    print(f"{'=' * 90}")

    rows = []
    sel_tbl = build_selection_table(pst, universe_dict, load_cache()) if apply_filter else None

    for k, j in enumerate(live_idx):
        instr = nms[j]
        held  = int(N_int[k])
        ideal = float(N_ideal[k])
        notional = abs(held) * ml[j] * r[j] * f[j]
        pct_cap  = notional / capital * 100

        eligible_flag = ""
        fail_reason   = ""
        if sel_tbl is not None and instr in sel_tbl.index:
            row = sel_tbl.loc[instr]
            if not row["eligible"]:
                eligible_flag = " [FILTERED]"
                fail_reason   = row["reason"]

        if held != 0 or abs(ideal) >= 0.15:
            rows.append({
                "Instrument": instr,
                "Class":      universe_dict.get(instr, ""),
                "Forecast":   round(float(fcl[k]), 1),
                "vol_pct":    round(float(sl[k]) * 100, 1),
                "Weight":     round(float(w_n[k]) * 100, 2),
                "N_ideal":    round(ideal, 2),
                "N_int":      held,
                "Notional":   f"${notional:,.0f}",
                "pct_cap":    f"{pct_cap:.0f}%",
                "Status":     ("HELD" if held != 0 else "—") + eligible_flag,
                "Filter":     fail_reason,
            })

    held_rows    = [r for r in rows if r["N_int"] != 0]
    skipped_rows = [r for r in rows if r["N_int"] == 0]

    print(f"\nHeld ({len(held_rows)} instruments):")
    print(f"  {'Instrument':<18} {'Class':<8} {'Fcst':>6} {'Vol%':>5} {'Wgt%':>6} "
          f"{'N_ideal':>8} {'N_int':>6} {'Notional':>12} {'PctCap':>6}  Status")
    print("  " + "-" * 90)
    for r in sorted(held_rows, key=lambda x: abs(x["N_ideal"]), reverse=True):
        print(f"  {r['Instrument']:<18} {r['Class']:<8} {r['Forecast']:>6.1f} {r['vol_pct']:>5.1f} "
              f"{r['Weight']:>6.2f} {r['N_ideal']:>8.2f} {r['N_int']:>6} "
              f"{r['Notional']:>12} {r['pct_cap']:>6}  {r['Status']}")

    if skipped_rows:
        print(f"\nNear-miss (|N_ideal|>=0.15 but N_int=0):")
        print(f"  {'Instrument':<18} {'Class':<8} {'Fcst':>6} {'Vol%':>5} {'N_ideal':>8}  Status/Filter")
        print("  " + "-" * 72)
        for r in sorted(skipped_rows, key=lambda x: abs(x["N_ideal"]), reverse=True):
            detail = r["Filter"] if r["Filter"] else ""
            print(f"  {r['Instrument']:<18} {r['Class']:<8} {r['Forecast']:>6.1f} "
                  f"{r['vol_pct']:>5.1f} {r['N_ideal']:>8.2f}  {r['Status']}  {detail}")

    total_notional = sum(
        abs(int(r["N_int"])) * uni["mult"][uni["names"].index(r["Instrument"])]
        * uni["raw"][t][uni["names"].index(r["Instrument"])]
        * uni["fx"][t][uni["names"].index(r["Instrument"])]
        for r in held_rows
    )
    print(f"\n  Total gross notional: ${total_notional:,.0f}  |  "
          f"Gross leverage: {total_notional/capital:.2f}x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dynamic-optimisation EWMAC backtest")
    parser.add_argument("--jumbo", action="store_true", help="Use Carver's Jumbo universe")
    parser.add_argument("--universe", action="store_true",
                        help="Use full UNIVERSE (104 instruments) instead of Jumbo")
    parser.add_argument("--capital", type=float, default=None,
                        help="Run a single capital level instead of the sweep")
    parser.add_argument("--compare", action="store_true",
                        help="Compare dynamic vs naive per-instrument rounding")
    parser.add_argument("--filter", action="store_true",
                        help="Apply instrument selection filters (cost/vol/history)")
    parser.add_argument("--snapshot", action="store_true",
                        help="Print today's positions for --capital (default $100k)")
    parser.add_argument("--history", action="store_true",
                        help="Print position history over the last 10 years at --capital")
    parser.add_argument("--rolling", action="store_true",
                        help="Use rolling (compounded) capital for position sizing")
    parser.add_argument("--costs", action="store_true",
                        help="Enable cost penalty + buffering in snapshot (matches live system)")
    parser.add_argument("--no-costs", action="store_true", help="Disable the cost penalty")
    parser.add_argument("--no-buffering", action="store_true", help="Disable buffering")
    args = parser.parse_args()

    from ibkr_fut.jumbo import JUMBO
    from ibkr_fut.instrument_universe import UNIVERSE

    instruments = UNIVERSE if args.universe else JUMBO
    label = "(UNIVERSE)" if args.universe else "(Carver Jumbo)"

    if args.snapshot:
        cap = args.capital or 100_000
        run_snapshot(instruments, cap, apply_filter=args.filter,
                     use_costs=args.costs, use_buffering=args.costs)
    elif args.history:
        cap = args.capital or 100_000
        print(f"Loading {len(instruments)} instruments {label} ...")
        uni = _build_universe(list(instruments.keys()) if isinstance(instruments, dict) else list(instruments))
        if uni:
            if args.filter:
                eligible = get_eligible_set(instruments if isinstance(instruments, dict) else {i: "" for i in instruments})
                uni["tradable"] = np.array([n in eligible for n in uni["names"]], dtype=bool)
            print(f"Running position history at ${cap:,.0f} ...")
            df = position_history(uni, cap)
            summarise_position_history(df, instruments if isinstance(instruments, dict) else {})
    elif args.compare:
        caps = [args.capital] if args.capital else [100_000, 250_000, 1_000_000]
        run_comparison(instruments, capitals=caps, label=label,
                       apply_filter=args.filter)
    else:
        caps = [args.capital] if args.capital else CAPITAL_SWEEP
        run_dynamic(
            instruments,
            capitals=caps,
            use_costs=not args.no_costs,
            use_buffering=not args.no_buffering,
            label=label,
            apply_filter=args.filter,
            rolling_capital=args.rolling,
        )
