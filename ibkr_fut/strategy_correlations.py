"""
strategy_correlations.py — Pairwise return correlations across all trading strategies.

Builds the combined universe ONCE (same as the live system), simulates each strategy
over it, and reports:
  • per-strategy performance stats
  • pairwise correlation matrix (printed + saved as CSV)
  • heatmap PNG

To add a new strategy: append one (label, signal_fn) entry to STRATEGIES below.
signal_fn must match the (spec, mp) -> {"sigma","forecast",...} | None contract.
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
from ibkr_fut.backtest_ewmac import pst, instrument_signals, TARGET_RISK
from ibkr_fut.carry_signals import carry_instrument_signals
from ibkr_fut.carry_trend_signals import carry_trend_instrument_signals, TREND_WEIGHT
from ibkr_fut.backtest_dynamic import _build_universe, _simulate, get_eligible_set, RESULTS_DIR
from ibkr_fut.backtest_carry_trend_dynamic import _forecast_panel, _combined_signal
from ibkr_fut.instrument_universe import UNIVERSE

DEFAULT_CAPITAL = 250_000


def _trend_signal(spec, mp):
    return instrument_signals(spec)


# ── Strategy registry ─────────────────────────────────────────────────────────
# Add new strategies here. Each entry is (label, signal_fn).
# signal_fn(spec, mp) -> {"sigma", "forecast", ...} | None
STRATEGIES = [
    ("Trend (EWMAC)",                            _trend_signal),
    ("Carry",                                    carry_instrument_signals),
    (f"Combined {TREND_WEIGHT:.0%}/{1-TREND_WEIGHT:.0%}", _combined_signal),
]
# ─────────────────────────────────────────────────────────────────────────────


def run_correlations(
    instruments: dict[str, str],
    capital: float = DEFAULT_CAPITAL,
    target_risk: float = TARGET_RISK,
) -> pd.DataFrame:
    """
    Build the combined (live) universe once, simulate each strategy over it, and
    compute pairwise correlations of daily returns.  Holding the universe fixed
    isolates the forecast effect, exactly as the live system operates.
    """
    names = list(instruments.keys())
    eligible = get_eligible_set(instruments)

    print("\nBuilding combined universe (once)...")
    built = _build_universe(
        names, tradable_set=eligible,
        signal_fn=_combined_signal, return_components=True,
    )
    if built is None:
        print("No tradable instruments.")
        return pd.DataFrame()
    uni, specs, _ = built

    daily_returns: dict[str, pd.Series] = {}
    stats_rows = []

    for label, fn in STRATEGIES:
        print(f"Simulating {label} @ ${capital:,.0f} ...")
        uni["forecast"] = _forecast_panel(uni, specs, fn)
        res = _simulate(uni, capital, target_risk, use_costs=True, use_buffering=True,
                        rolling_capital=False)
        daily_returns[label] = res["equity"].pct_change().dropna()
        st = res["stats"]
        stats_rows.append({
            "Strategy": label,
            "SR":       st["sharpe_ratio"],
            "Ret %":    st["mean_annual_return_pct"],
            "Vol %":    st["std_dev_pct"],
            "MaxDD %":  st["max_drawdown_pct"],
            "Skew":     st["skew"],
        })

    # Align all return series to the common trading-day index
    ret_df = pd.DataFrame(daily_returns).dropna()
    corr = ret_df.corr()
    labels = corr.columns.tolist()

    # ── Stats table ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"STRATEGY STATS  |  fixed ${capital:,.0f}  |  costs + buffering  |  net of costs")
    print("=" * 80)
    print(tabulate(stats_rows, headers="keys", floatfmt=".2f", tablefmt="simple"))

    # ── Correlation matrix ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PAIRWISE RETURN CORRELATIONS  (daily, net-of-cost equity curves)")
    print("=" * 80)
    corr_rows = [{"Strategy": s, **{o: round(corr.loc[s, o], 3) for o in labels}}
                 for s in labels]
    print(tabulate(corr_rows, headers="keys", floatfmt=".3f", tablefmt="simple"))

    # ── Heatmap ───────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.8), max(5, n * 1.4)))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="black" if abs(val) < 0.7 else "white")
    ax.set_title(f"Strategy return correlations  |  ${capital:,.0f}  |  costs+buffering",
                 fontsize=11, pad=12)
    plt.tight_layout()
    heatmap_out = f"{RESULTS_DIR}/strategy_correlations.png"
    plt.savefig(heatmap_out, dpi=120)
    plt.close()
    print(f"\nHeatmap  → {heatmap_out}")

    csv_out = f"{RESULTS_DIR}/strategy_correlations.csv"
    corr.to_csv(csv_out)
    print(f"CSV      → {csv_out}")

    return corr


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Pairwise strategy return correlations + heatmap")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL,
                        help=f"Capital level (default ${DEFAULT_CAPITAL:,.0f})")
    args = parser.parse_args()
    run_correlations(UNIVERSE, args.capital)
