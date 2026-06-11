"""
backtest_carry_trend_dynamic.py — Combined carry + trend backtest driven by Carver's
dynamic portfolio optimisation (AFTS "Strategy 11: Combined carry and trend").

Identical machinery to backtest_dynamic.py / backtest_carry_dynamic.py — same vol scaling,
handcraft weights, IDM, and the joint integer optimiser in dynamic_opt.py — but the
per-instrument forecast is the 60/40 trend+carry blend from carry_trend_signals.py.

Default mode is a 3-WAY COMPARISON (trend-only / carry-only / combined) over the SAME
universe at one capital, reproducing Carver's Table 55: is combining worth it?  An
optional --sweep reproduces Tables 49/50 (SR/DD/skew as the trend share goes 0%→100%).

To keep the three strategies apples-to-apples (as Carver does on one Jumbo), every
strategy is built on the identical instrument set: a name is included only if it has both
adjusted prices AND multiple_prices (carry needs the term structure).  Settings mirror
live: instrument-selection filter on, costs + buffering on, fixed capital, τ = 0.20.
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
from ibkr_fut.foundations import sigma_p_from_pct
from ibkr_fut.backtest_ewmac import pst, instrument_signals, TARGET_RISK, IDM_CAP
from ibkr_fut.dynamic_opt import optimise_positions
from ibkr_fut.backtest_dynamic import (
    _build_universe,
    _simulate,
    position_history,
    summarise_position_history,
    get_eligible_set,
    RESULTS_DIR,
)
from ibkr_fut.carry_signals import carry_instrument_signals
from ibkr_fut.carry_trend_signals import (
    carry_trend_instrument_signals,
    combined_carry_trend_forecast,
    TREND_WEIGHT,
)
from ibkr_fut.instrument_universe import UNIVERSE

DEFAULT_CAPITAL = 250_000


# Per-instrument signal builders, all (spec, mp) -> {"sigma","forecast",...} | None, so
# the shared backtest_dynamic._build_universe can drive any of the three strategies.
def _trend_signal(spec, mp):
    return instrument_signals(spec)            # mp ignored (trend needs no term structure)


def _combined_signal(spec, mp):
    return carry_trend_instrument_signals(spec, mp, TREND_WEIGHT)


# carry uses carry_instrument_signals(spec, mp) directly (already the right signature).
STRATEGIES = [
    ("Trend (EWMAC)", _trend_signal),
    ("Carry",         carry_instrument_signals),
    (f"Combined {TREND_WEIGHT:.0%}/{1-TREND_WEIGHT:.0%}", _combined_signal),
]


def _forecast_panel(uni: dict, specs: dict, signal_fn) -> np.ndarray:
    """
    Compute a (T x n) forecast panel for `signal_fn` over an ALREADY-BUILT universe,
    reusing its shared sigma/correlation/covariance/handcraft model.  A column is NaN
    where the strategy has no signal for that instrument (e.g. trend on a carry-only
    name) — it then stays in the risk model but is never sized.

    This is how the 3-way comparison holds the universe FIXED (the same combined universe
    the live system trades) and varies only the forecast, so it cleanly isolates what carry
    adds — and builds the expensive covariance model once instead of three times.
    """
    idx = uni["idx"]
    panel = np.full((len(idx), len(uni["names"])), np.nan)
    for j, instr in enumerate(uni["names"]):
        try:
            mp = pst.multiple_prices(instr)
        except (FileNotFoundError, KeyError):
            mp = None
        sig = signal_fn(specs[instr], mp)
        if sig is not None:
            panel[:, j] = sig["forecast"].reindex(idx).values
    return panel


# ── 3-way comparison (default) ──────────────────────────────────────────────────────

def run_comparison(
    instruments: dict[str, str],
    capital: float = DEFAULT_CAPITAL,
    target_risk: float = TARGET_RISK,
) -> dict:
    """
    Run trend-only, carry-only, and combined through the dynamic optimiser at `capital`
    (filter + costs + buffering on, fixed capital), holding the UNIVERSE and risk model
    FIXED and varying only the forecast.  This matches deployment: the live system trades
    one combined universe with one risk model and never runs a standalone trend/carry
    system, so the right question is "on that universe, what does each forecast give, and
    what does blending carry into trend add?" — not "how would a trend-only system on its
    own universe do?".  Holding the universe fixed also cleanly isolates the forecast
    effect and lets the expensive covariance model be built ONCE.  Prints a Table-55-style
    comparison and saves all three equity curves.
    """
    names = list(instruments.keys())
    eligible = get_eligible_set(instruments)

    # Build the combined universe once — exactly the universe live trades (valid set = the
    # union of trend- and carry-eligible names) — then evaluate each forecast on it.
    print("\nBuilding combined (live) universe...")
    built = _build_universe(names, tradable_set=eligible,
                            signal_fn=_combined_signal, return_components=True)
    if built is None:
        print("No tradable instruments.")
        return {}
    uni, specs, _ = built

    rows, equities = [], {}
    for label, fn in STRATEGIES:
        print(f"Simulating {label} @ ${capital:,.0f} ...")
        uni["forecast"] = _forecast_panel(uni, specs, fn)
        res = _simulate(uni, capital, target_risk, use_costs=True, use_buffering=True,
                        rolling_capital=False)
        st = res["stats"]
        rows.append({
            "Strategy":   label,
            "Mean Ret %": st["mean_annual_return_pct"],
            "Std Dev %":  st["std_dev_pct"],
            "SR":         st["sharpe_ratio"],
            "Costs %":    st["costs_pct"],
            "Turnover":   st["turnover"],
            "Avg DD %":   st["avg_drawdown_pct"],
            "Max DD %":   st["max_drawdown_pct"],
            "Skew":       st["skew"],
            "Avg #Held":  st["avg_instruments_held"],
        })
        equities[label] = res["equity"]

    print("\n" + "=" * 118)
    print(f"STRATEGY 11 — TREND vs CARRY vs COMBINED (UNIVERSE) | costs+buffering | "
          f"fixed ${capital:,.0f} | net of costs")
    print("=" * 118)
    print(tabulate(rows, headers="keys", floatfmt=".2f", tablefmt="simple"))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {"Trend (EWMAC)": "steelblue", "Carry": "darkorange"}
    for label, eq in equities.items():
        ax.plot(eq.index, eq.values, linewidth=0.9,
                color=colors.get(label, "darkgreen"), label=label)
    ax.set_yscale("log")
    ax.set_title(f"Strategy 11 — Trend vs Carry vs Combined (UNIVERSE) @ ${capital:,.0f}")
    ax.set_ylabel("Portfolio equity ($, log scale)")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{RESULTS_DIR}/carry_trend_comparison.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\nEquity curves saved to {out}")

    return {"rows": rows, "equities": equities}


# ── Proportion sweep (--sweep) ─────────────────────────────────────────────────────

SWEEP_WEIGHTS = [0.0, 0.25, 0.40, 0.50, 0.60, 0.75, 1.0]


def _sweep_prep(uni: dict, specs: dict, signals: dict) -> dict:
    """
    Per-instrument forecast inputs that do NOT depend on the trend weight (sigma_p, the
    term structure, the active rule lists), computed once so the weight loop only re-blends
    instead of re-fetching multiple_prices and re-deriving sigma_p on every weight.
    """
    prep = {}
    for instr in uni["names"]:
        spec, sig = specs[instr], signals[instr]
        sigma_a = sig["sigma"]
        sp_a    = sigma_p_from_pct(spec.raw_price, sigma_a)
        try:
            mp = pst.multiple_prices(instr)
        except (FileNotFoundError, KeyError):
            mp = None
        prep[instr] = (spec.prices, spec.raw_price, sp_a, sigma_a, mp,
                       sig["active_trend"], sig["active_carry"])
    return prep


def _blend_forecast_panel(uni: dict, prep: dict, trend_weight: float):
    """Rebuild only uni['forecast'] for a new trend weight (covariance/weights unchanged)."""
    idx = uni["idx"]
    for j, instr in enumerate(uni["names"]):
        prices, raw_price, sp_a, sigma_a, mp, active_trend, active_carry = prep[instr]
        fc = combined_carry_trend_forecast(
            prices, raw_price, sp_a, sigma_a, mp, active_trend, active_carry, trend_weight,
        )
        uni["forecast"][:, j] = fc.reindex(idx).values


def run_sweep(
    instruments: dict[str, str],
    capital: float = DEFAULT_CAPITAL,
    target_risk: float = TARGET_RISK,
) -> dict:
    """
    Sweep the trend share 0%→100% (Carver Tables 49/50).  Builds the combined universe
    ONCE, then for each weight recomputes just the forecast panel and re-simulates, so the
    covariance model is built a single time.
    """
    names = list(instruments.keys())
    eligible = get_eligible_set(instruments)

    print("Building combined universe (once)...")
    built = _build_universe(names, tradable_set=eligible,
                            signal_fn=_combined_signal, return_components=True)
    if built is None:
        print("No tradable instruments.")
        return {}
    uni, specs, signals = built
    prep = _sweep_prep(uni, specs, signals)

    rows = []
    for w in SWEEP_WEIGHTS:
        _blend_forecast_panel(uni, prep, w)
        res = _simulate(uni, capital, target_risk, use_costs=True, use_buffering=True,
                        rolling_capital=False)
        st = res["stats"]
        rows.append({
            "Trend %":    f"{w:.0%}",
            "Carry %":    f"{1-w:.0%}",
            "Mean Ret %": st["mean_annual_return_pct"],
            "Std Dev %":  st["std_dev_pct"],
            "SR":         st["sharpe_ratio"],
            "Avg DD %":   st["avg_drawdown_pct"],
            "Max DD %":   st["max_drawdown_pct"],
            "Skew":       st["skew"],
            "Turnover":   st["turnover"],
        })
        print(f"  trend={w:.0%}: SR={st['sharpe_ratio']:.3f} "
              f"maxDD={st['max_drawdown_pct']:.1f}% skew={st['skew']:.2f}")

    print("\n" + "=" * 104)
    print(f"STRATEGY 11 — TREND/CARRY PROPORTION SWEEP (UNIVERSE) | "
          f"fixed ${capital:,.0f} | costs+buffering")
    print("=" * 104)
    print(tabulate(rows, headers="keys", floatfmt=".2f", tablefmt="simple"))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    sr = [r["SR"] for r in rows]
    dd = [abs(r["Max DD %"]) for r in rows]
    xs = [w * 100 for w in SWEEP_WEIGHTS]
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(xs, sr, "o-", color="navy", label="Sharpe ratio")
    ax1.set_xlabel("Trend share (%)  —  carry share = 100 − trend")
    ax1.set_ylabel("Sharpe ratio", color="navy")
    ax2 = ax1.twinx()
    ax2.plot(xs, dd, "s--", color="firebrick", label="Max drawdown")
    ax2.set_ylabel("Max drawdown (%)", color="firebrick")
    ax1.set_title(f"Strategy 11 — SR & Max DD vs trend share (UNIVERSE) @ ${capital:,.0f}")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{RESULTS_DIR}/carry_trend_sweep.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\nSweep plot saved to {out}")

    return {"rows": rows}


# ── Snapshot / history ─────────────────────────────────────────────────────────────

def run_snapshot(
    instruments: dict[str, str],
    capital: float = DEFAULT_CAPITAL,
    target_risk: float = TARGET_RISK,
) -> None:
    """Today's combined integer book from flat (filter + costs + buffering)."""
    names = list(instruments.keys())
    eligible = get_eligible_set(instruments)
    uni = _build_universe(names, signal_fn=_combined_signal)  # full universe for risk model
    if uni is None:
        print("No instruments with valid combined signals.")
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
    print(f"COMBINED CARRY+TREND SNAPSHOT — {idx[-1].date()}  |  capital=${capital:,.0f}  "
          f"|  IDM={idm_t:.3f}  |  trend={TREND_WEIGHT:.0%}  [filtered]")
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
            "fc": round(float(fcl[k]), 1), "vol%": round(float(sl[k]) * 100, 1),
            "wgt%": round(float(w_n[k]) * 100, 2), "N_ideal": round(ideal, 2),
            "N_int": held, "notional": notional, "pct_cap": notional / capital * 100,
            "held": held != 0,
        })

    held_rows = sorted([r for r in rows if r["held"]], key=lambda x: -abs(x["N_ideal"]))
    near      = sorted([r for r in rows if not r["held"]], key=lambda x: -abs(x["N_ideal"]))

    print(f"\nHeld ({len(held_rows)} instruments):")
    print(f"  {'Instrument':<16} {'Class':<8} {'Fcst':>6} {'Vol%':>5} {'Wgt%':>6} "
          f"{'N_ideal':>8} {'N_int':>6} {'Notional':>13} {'PctCap':>7}")
    print("  " + "-" * 86)
    gross = 0.0
    for r in held_rows:
        gross += r["notional"]
        print(f"  {r['instr']:<16} {r['class']:<8} {r['fc']:>6.1f} {r['vol%']:>5.1f} "
              f"{r['wgt%']:>6.2f} {r['N_ideal']:>8.2f} {r['N_int']:>6} "
              f"${r['notional']:>11,.0f} {r['pct_cap']:>6.0f}%")

    if near:
        print(f"\nNear-miss (|N_ideal|>=0.15 but N_int=0):")
        for r in near:
            print(f"  {r['instr']:<16} {r['class']:<8} fc={r['fc']:>6.1f} "
                  f"N_ideal={r['N_ideal']:>6.2f}")

    print(f"\n  Total gross notional: ${gross:,.0f}  |  Gross leverage: {gross / capital:.2f}x")


def run_history(instruments: dict[str, str], capital: float = DEFAULT_CAPITAL) -> None:
    """Combined position history over the last 10 years at `capital` (filter applied)."""
    names = list(instruments.keys())
    eligible = get_eligible_set(instruments)
    uni = _build_universe(names, tradable_set=eligible, signal_fn=_combined_signal)
    if uni is None:
        print("No tradable instruments.")
        return
    print(f"Running combined position history at ${capital:,.0f} ...")
    df = position_history(uni, capital)
    summarise_position_history(df, instruments)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Dynamic-optimisation Combined Carry+Trend backtest (Strategy 11)")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL,
                        help=f"Capital level (default ${DEFAULT_CAPITAL:,.0f})")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep trend/carry proportion 0%%→100%% (Carver Tables 49/50)")
    parser.add_argument("--snapshot", action="store_true",
                        help="Print today's combined positions instead of the backtest")
    parser.add_argument("--history", action="store_true",
                        help="Print 10-year combined position history instead of the backtest")
    args = parser.parse_args()

    if args.snapshot:
        run_snapshot(UNIVERSE, args.capital)
    elif args.history:
        run_history(UNIVERSE, args.capital)
    elif args.sweep:
        run_sweep(UNIVERSE, args.capital)
    else:
        run_comparison(UNIVERSE, args.capital)
