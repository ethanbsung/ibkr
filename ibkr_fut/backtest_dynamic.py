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
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibkr_fut.foundations import (
    PST_CUTOFF,
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
    COMMISSION,
)
from ibkr_fut.dynamic_opt import CovarianceEstimator, optimise_positions

# ── Parameters ───────────────────────────────────────────────────────────────────

CAPITAL_SWEEP = [100_000, 250_000, 1_000_000, 50_000_000]
RESULTS_DIR   = "ibkr_fut/results"


def _build_universe(instruments: list[str], tradable_set: set | None = None) -> dict | None:
    """
    Build aligned numpy panels for every instrument with valid signals.
    Returns a dict of arrays (T x n) plus per-instrument scalars and the
    covariance estimator, or None if nothing is tradable.

    tradable_set: optional set of instrument names we are allowed to trade. The full
    `instruments` list forms the optimisation universe (drives covariance + target
    weights); names outside tradable_set are held at min=max=current so the optimiser
    transfers their risk onto correlated tradable markets (calcs.txt lines 326-330).
    None => every instrument with valid signals is tradable (original behaviour).
    """
    specs, signals = {}, {}
    for instr in instruments:
        spec = _pst_spec(instr)
        if spec is None:
            continue
        sig = instrument_signals(spec)
        if sig is None:
            continue
        specs[instr] = spec
        signals[instr] = sig

    names = list(specs.keys())
    if not names:
        return None
    print(f"  {len(names)} instruments with valid signals")

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

    return dict(
        names=names, idx=idx, price=price, raw=raw, fx=fx, sigma=sigma,
        forecast=forecast, mult=mult, spread=spread, commission=commission,
        W=W, C=C, est=est, tradable=tradable,
    )


NAIVE_BUFFER_FRAC = 0.10   # per-instrument buffer band for the naive baseline  [calcs line 113]


def _simulate(uni: dict, capital: float, target_risk: float,
              use_costs: bool, use_buffering: bool, mode: str = "dynamic") -> dict:
    """
    Run the joint daily loop for one capital level.
    mode="dynamic": Carver dynamic optimisation (greedy + cost + buffer).
    mode="naive":   independent per-instrument rounding with a buffer band — the
                    baseline dynamic optimisation is meant to beat at small capital.
    """
    names, idx = uni["names"], uni["idx"]
    price, raw, fx = uni["price"], uni["raw"], uni["fx"]
    sigma, forecast = uni["sigma"], uni["forecast"]
    mult, spread, commission = uni["mult"], uni["spread"], uni["commission"]
    W, C, est = uni["W"], uni["C"], uni["est"]
    T, n = len(idx), len(names)
    idx_values = idx.values

    pos = np.zeros(n)               # full-universe integer positions held
    pnl_list = np.zeros(T)
    total_cost = 0.0
    abs_changes = 0.0               # one-way contracts traded (turnover numerator)
    abs_pos_sum = 0.0
    n_held_sum = 0.0
    te_sum, te_count = 0.0, 0       # tracking error of held vs ideal (diagnostic)
    idm_cache: dict[tuple, tuple] = {}

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
        N_unrounded = (fcl * capital * idm_t * w_n * target_risk) / (10.0 * ml * rl * fl * sl)
        weight_per_contract = ml * rl * fl / capital                      # [calcs line 226]
        # One-way cost per contract: half-spread crossing (SpreadCost) + half the
        # round-trip commission. (Was 2*spread + full commission = 2x the true cost.)
        cost_per_contract = (spread[live_idx] * ml + commission[live_idx] / 2.0) * fl

        cov = est.covariance_by_index(pd.Timestamp(idx_values[t]), live_idx)
        prev_live = pos[live_idx]

        # Step 4: integer positions, by mode.
        if mode == "naive":
            # Per-instrument rounding with a buffer band (no joint optimisation).
            N_avg = (capital * idm_t * w_n * target_risk) / (ml * rl * fl * sl)
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
                capital=capital,
                target_risk=target_risk,
                use_costs=use_costs,
                use_buffering=use_buffering,
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

        abs_pos_sum += np.abs(pos).sum()
        pnl_list[t] = daily_pnl - trade_cost

    pnl_s = pd.Series(pnl_list, index=idx)
    daily_returns = pnl_s / capital
    equity = capital * (1.0 + daily_returns).cumprod()

    years = T / ANNUAL_DAYS
    avg_abs_N = abs_pos_sum / T if T else 1.0
    turnover = (abs_changes / 2.0) / avg_abs_N / years if years and avg_abs_N else 0.0
    costs_pct = (total_cost / capital / years) * 100 if years else 0.0

    stats = performance_stats(equity, daily_returns, costs_pct=costs_pct, turnover=turnover)
    stats["avg_instruments_held"] = round(n_held_sum / te_count, 1) if te_count else 0.0
    stats["avg_tracking_error_pct"] = round(100 * te_sum / te_count, 2) if te_count else 0.0

    return {"equity": equity, "daily_returns": daily_returns, "stats": stats}


def run_dynamic(
    instruments: list[str] | dict,
    capitals: list[float] = CAPITAL_SWEEP,
    target_risk: float = TARGET_RISK,
    use_costs: bool = True,
    use_buffering: bool = True,
    label: str = "(Jumbo)",
) -> dict:
    """Build the universe once, then run the dynamic-opt backtest at each capital level."""
    names = list(instruments.keys()) if isinstance(instruments, dict) else list(instruments)
    print(f"Loading {len(names)} instruments {label} for dynamic optimisation...")
    uni = _build_universe(names)
    if uni is None:
        print("No tradable instruments.")
        return {}

    rows, curves = [], {}
    for cap in capitals:
        print(f"\nSimulating capital = ${cap:,.0f} ...")
        res = _simulate(uni, cap, target_risk, use_costs, use_buffering)
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
            "Avg TE %":     st["avg_tracking_error_pct"],
        })

    print("\n" + "=" * 110)
    print(f"DYNAMIC OPTIMISATION — EWMAC {label} | costs={use_costs} buffering={use_buffering}")
    print("=" * 110)
    print(tabulate(rows, headers="keys", floatfmt=".2f", tablefmt="simple"))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    for cap, eq in curves.items():
        ax.plot(eq.index, eq.values / eq.iloc[0], linewidth=0.8, label=f"${cap:,.0f}")
    ax.set_yscale("log")
    ax.set_title(f"Dynamic-Optimisation EWMAC {label} — growth of $1 by capital level")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_xlabel("Date")
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
) -> dict:
    """Dynamic optimisation vs naive per-instrument rounding at matched capital levels."""
    names = list(instruments.keys()) if isinstance(instruments, dict) else list(instruments)
    print(f"Loading {len(names)} instruments {label} for dynamic-vs-naive comparison...")
    uni = _build_universe(names)
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dynamic-optimisation EWMAC backtest")
    parser.add_argument("--jumbo", action="store_true", help="Use Carver's Jumbo universe")
    parser.add_argument("--capital", type=float, default=None,
                        help="Run a single capital level instead of the sweep")
    parser.add_argument("--compare", action="store_true",
                        help="Compare dynamic vs naive per-instrument rounding")
    parser.add_argument("--no-costs", action="store_true", help="Disable the cost penalty")
    parser.add_argument("--no-buffering", action="store_true", help="Disable buffering")
    args = parser.parse_args()

    from ibkr_fut.jumbo import JUMBO
    caps = [args.capital] if args.capital else CAPITAL_SWEEP
    if args.compare:
        run_comparison(JUMBO, capitals=caps if args.capital else (100_000, 250_000, 1_000_000))
    else:
        run_dynamic(
            JUMBO,
            capitals=caps,
            use_costs=not args.no_costs,
            use_buffering=not args.no_buffering,
            label="(Carver Jumbo)",
        )
