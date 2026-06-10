"""
backtest_carry_dynamic.py — Basic-carry backtest driven by Carver's dynamic portfolio
optimisation (AFTS "Strategy 10: Basic Carry").

Identical machinery to backtest_dynamic.py (EWMAC) — same vol scaling, handcraft weights,
IDM, and the joint integer optimiser in dynamic_opt.py — but the per-instrument forecast
is the carry forecast from carry_signals.py instead of EWMAC.

Defaults are chosen to mirror the live system (ibkr_fut/system_state.md) as closely as a
backtest can:
  • universe        = instrument_universe.UNIVERSE (not just the Jumbo)
  • tradable set    = instrument-selection filter applied (cost/vol/history), exactly as
                      the live build_tradable_set; ineligible names stay in the risk model
                      but are held at their current position
  • costs + buffering = always on (the live optimiser runs with both)
  • target risk     = TARGET_RISK (0.20)

Position sizing uses fixed capital (like the EWMAC backtest_dynamic reported runs), so
the risk target, realised vol, and gross-leverage diagnostics are clean and directly
comparable to the trend backtest.  (Live reads capital from IBKR daily — effectively
compounding — but that only rescales positions; it doesn't change SR/vol/skew.)

The signal-agnostic engine (`_simulate`), position history, and the selection filter are
imported unchanged from backtest_dynamic.py; only the universe *builder* is carry-specific.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

from dataclasses import replace as dc_replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibkr_fut.foundations import compute_corr_matrix, handcraft_weights
from ibkr_fut.backtest_ewmac import pst, _pst_spec, TARGET_RISK, IDM_CAP
from ibkr_fut.dynamic_opt import CovarianceEstimator, optimise_positions
from ibkr_fut.backtest_dynamic import (
    _simulate,
    position_history,
    summarise_position_history,
    get_eligible_set,
    RESULTS_DIR,
)
from ibkr_fut.carry_signals import carry_instrument_signals
from ibkr_fut.instrument_universe import UNIVERSE

DEFAULT_CAPITAL = 100_000


# ── Universe builder (carry forecast) ──────────────────────────────────────────────

def _build_carry_universe(
    instruments: list[str],
    tradable_set: set | None = None,
    lookback_days: int | None = None,
) -> dict | None:
    """
    Carry analogue of backtest_dynamic._build_universe: build aligned numpy panels for
    every instrument with a valid carry signal, plus per-instrument scalars and the
    covariance estimator.  Returns the same dict shape `_simulate` consumes, or None.

    Differs from the EWMAC builder in one line — the forecast comes from
    carry_instrument_signals (needs pst.multiple_prices) instead of instrument_signals.
    Handcraft weights / correlation / covariance / IDM are strategy-independent and built
    identically.

    lookback_days: if set, trim all price series to at most this many calendar days
    before today before building the panels (mirrors backtest_dynamic._build_universe;
    use ~3000 for a future live carry run to avoid OOM).  None => full history.
    """
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
        try:
            mp = pst.multiple_prices(instr)
        except (FileNotFoundError, KeyError):
            skipped.append((instr, "no carry data (multiple_prices missing)"))
            continue
        if cutoff_date is not None:
            spec = dc_replace(
                spec,
                prices=spec.prices[spec.prices.index >= cutoff_date],
                raw_price=spec.raw_price[spec.raw_price.index >= cutoff_date],
            )
            mp = mp[mp.index >= cutoff_date]
        sig = carry_instrument_signals(spec, mp)
        if sig is None:
            skipped.append((instr, "no eligible carry spans / invalid vol"))
            continue
        specs[instr] = spec
        signals[instr] = sig

    names = list(specs.keys())
    if not names:
        return None
    print(f"  {len(names)} instruments with valid carry signals"
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
    forecast = np.full((T, n), np.nan)   # combined capped carry forecast
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


# ── Runners ─────────────────────────────────────────────────────────────────────

def run_carry(
    instruments: dict[str, str],
    capital: float = DEFAULT_CAPITAL,
    target_risk: float = TARGET_RISK,
) -> dict:
    """
    Build the carry universe once (filter applied) and run the dynamic-opt backtest at
    `capital` with live-matching settings: costs + buffering + rolling capital all on.
    """
    names = list(instruments.keys())
    print(f"Loading {len(names)} instruments for carry dynamic optimisation...")

    eligible = get_eligible_set(instruments)
    uni = _build_carry_universe(names, tradable_set=eligible)
    if uni is None:
        print("No tradable instruments.")
        return {}

    print(f"\nSimulating capital = ${capital:,.0f} ...")
    res = _simulate(uni, capital, target_risk, use_costs=True, use_buffering=True,
                    rolling_capital=False)
    st = res["stats"]

    rows = [{
        "Capital":    f"${capital:,.0f}",
        "Mean Ret %": st["mean_annual_return_pct"],
        "Std Dev %":  st["std_dev_pct"],
        "SR":         st["sharpe_ratio"],
        "Costs %":    st["costs_pct"],
        "Turnover":   st["turnover"],
        "Avg DD %":   st["avg_drawdown_pct"],
        "Max DD %":   st["max_drawdown_pct"],
        "Skew":       st["skew"],
        "Avg #Held":  st["avg_instruments_held"],
        "Lev mean":   st["gross_lev_mean"],
        "Lev p95":    st["gross_lev_p95"],
        "Lev max":    st["gross_lev_max"],
    }]

    print("\n" + "=" * 110)
    print("DYNAMIC OPTIMISATION — BASIC CARRY (UNIVERSE) | costs=True buffering=True | fixed capital")
    print("=" * 110)
    print(tabulate(rows, headers="keys", floatfmt=".2f", tablefmt="simple"))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    eq = res["equity"]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(eq.index, eq.values, linewidth=0.8, color="darkgreen", label=f"${capital:,.0f}")
    ax.set_yscale("log")
    ax.set_title(f"Dynamic-Optimisation Basic Carry (UNIVERSE) — rolling equity @ ${capital:,.0f}")
    ax.set_ylabel("Portfolio equity ($, log scale)")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{RESULTS_DIR}/carry_dynamic_sweep.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\nEquity curve saved to {out}")

    return {"rows": rows, "equity": eq, "universe": uni}


def run_carry_snapshot(
    instruments: dict[str, str],
    capital: float = DEFAULT_CAPITAL,
    target_risk: float = TARGET_RISK,
) -> None:
    """
    Print today's ideal and integer carry positions for `capital`, using the same
    live-matching settings (filter + costs + buffering).  Shows what the carry book looks
    like right now given current prices, vols, and term structure.
    """
    names = list(instruments.keys())
    eligible = get_eligible_set(instruments)
    uni = _build_carry_universe(names)   # full universe for the risk model
    if uni is None:
        print("No instruments with valid carry signals.")
        return

    idx = uni["idx"]
    t   = len(idx) - 1
    p, r, f = uni["price"][t], uni["raw"][t], uni["fx"][t]
    s, fc   = uni["sigma"][t], uni["forecast"][t]
    ml, sp, com = uni["mult"], uni["spread"], uni["commission"]
    W, C, est, nms = uni["W"], uni["C"], uni["est"], uni["names"]

    valid = (~np.isnan(p) & ~np.isnan(r) & (r > 0)
             & ~np.isnan(s) & (s > 0) & ~np.isnan(f) & ~np.isnan(fc))
    live_idx = np.flatnonzero(valid)

    w_live = W[live_idx]
    ssum   = float(w_live.sum())
    w_n    = w_live / ssum if ssum > 0 else np.full(live_idx.size, 1.0 / live_idx.size)
    Csub   = C[np.ix_(live_idx, live_idx)]
    var    = float(w_n @ Csub @ w_n)
    idm_t  = min(1.0 / np.sqrt(var), IDM_CAP) if var > 0 else 1.0

    ml_l, rl, fl = ml[live_idx], r[live_idx], f[live_idx]
    sl, fcl      = s[live_idx], fc[live_idx]

    N_ideal = (fcl * capital * idm_t * w_n * target_risk) / (10.0 * ml_l * rl * fl * sl)
    wpc     = ml_l * rl * fl / capital
    cpc     = (sp[live_idx] * ml_l + com[live_idx] / 2.0) * fl
    cov     = est.covariance_by_index(pd.Timestamp(idx[-1]), live_idx)
    tradable_mask = np.array([nms[i] in eligible for i in live_idx], dtype=bool)

    N_int = optimise_positions(
        covariance=cov, weight_per_contract=wpc,
        optimal_unrounded_positions=N_ideal,
        previous_positions=np.zeros(live_idx.size),
        cost_per_contract=cpc, capital=capital, target_risk=target_risk,
        use_costs=True, use_buffering=True, tradable=tradable_mask,
    )

    print(f"\n{'=' * 92}")
    print(f"CARRY SNAPSHOT — {idx[-1].date()}  |  capital=${capital:,.0f}  |  IDM={idm_t:.3f}  [filtered]")
    print(f"{'=' * 92}")

    rows = []
    for k, j in enumerate(live_idx):
        instr = nms[j]
        held  = int(N_int[k])
        ideal = float(N_ideal[k])
        if held == 0 and abs(ideal) < 0.15:
            continue
        notional = abs(held) * ml[j] * r[j] * f[j]
        rows.append({
            "instr": instr, "class": instruments.get(instr, ""),
            "carry_fc": round(float(fcl[k]), 1), "vol%": round(float(sl[k]) * 100, 1),
            "wgt%": round(float(w_n[k]) * 100, 2), "N_ideal": round(ideal, 2),
            "N_int": held, "notional": notional, "pct_cap": notional / capital * 100,
            "held": held != 0,
        })

    held_rows = sorted([r for r in rows if r["held"]], key=lambda x: -abs(x["N_ideal"]))
    near      = sorted([r for r in rows if not r["held"]], key=lambda x: -abs(x["N_ideal"]))

    print(f"\nHeld ({len(held_rows)} instruments):")
    print(f"  {'Instrument':<16} {'Class':<8} {'Carry':>6} {'Vol%':>5} {'Wgt%':>6} "
          f"{'N_ideal':>8} {'N_int':>6} {'Notional':>13} {'PctCap':>7}")
    print("  " + "-" * 86)
    gross = 0.0
    for r in held_rows:
        gross += r["notional"]
        print(f"  {r['instr']:<16} {r['class']:<8} {r['carry_fc']:>6.1f} {r['vol%']:>5.1f} "
              f"{r['wgt%']:>6.2f} {r['N_ideal']:>8.2f} {r['N_int']:>6} "
              f"${r['notional']:>11,.0f} {r['pct_cap']:>6.0f}%")

    if near:
        print(f"\nNear-miss (|N_ideal|>=0.15 but N_int=0):")
        for r in near:
            print(f"  {r['instr']:<16} {r['class']:<8} carry={r['carry_fc']:>6.1f} "
                  f"N_ideal={r['N_ideal']:>6.2f}")

    print(f"\n  Total gross notional: ${gross:,.0f}  |  Gross leverage: {gross / capital:.2f}x")


def run_carry_history(instruments: dict[str, str], capital: float = DEFAULT_CAPITAL) -> None:
    """Position history over the last 10 years at `capital` (filter applied)."""
    names = list(instruments.keys())
    eligible = get_eligible_set(instruments)
    uni = _build_carry_universe(names, tradable_set=eligible)
    if uni is None:
        print("No tradable instruments.")
        return
    print(f"Running carry position history at ${capital:,.0f} ...")
    df = position_history(uni, capital)
    summarise_position_history(df, instruments)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dynamic-optimisation Basic Carry backtest")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL,
                        help=f"Capital level (default ${DEFAULT_CAPITAL:,.0f})")
    parser.add_argument("--snapshot", action="store_true",
                        help="Print today's carry positions instead of the backtest")
    parser.add_argument("--history", action="store_true",
                        help="Print 10-year carry position history instead of the backtest")
    args = parser.parse_args()

    if args.snapshot:
        run_carry_snapshot(UNIVERSE, args.capital)
    elif args.history:
        run_carry_history(UNIVERSE, args.capital)
    else:
        run_carry(UNIVERSE, args.capital)
