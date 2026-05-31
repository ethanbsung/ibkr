#!/usr/bin/env python3
"""
Greedy instrument selection for the ETF universe.

Implements Carver's static portfolio optimisation algorithm from:
  "Advanced Futures Trading Strategies" (2023), and
  https://qoppac.blogspot.com/2021/06/static-optimisation-of-best-set-of.html

Algorithm
---------
1. Compute per-instrument net SR:
     SR_net_i = notional_SR (0.50) - SR_cost_i - size_penalty_i
   where:
     SR_cost_i  = annual_expense_ratio / vol_target  (ETF ongoing cost)
     size_penalty = 0  (fractional shares — no minimum-position constraint)

2. Start with the instrument that has the highest SR_net.

3. Greedily add the instrument that maximises portfolio SR:
     SR_port = w^T μ / sqrt(w^T Σ w)
   where:
     μ     = vector of SR_net per instrument
     Σ     = pairwise correlation matrix of instrument returns
     w     = handcraft weights (equal within asset-class group,
             equal across groups)

4. Stop when the best possible addition yields
     SR_new < 0.9 × max(SR seen so far)

References
----------
pysystemtrade source:
  systems/provided/static_small_system_optimise/optimise_small_system.py
  sysquant/optimisation/full_handcrafting.py
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
NOTIONAL_SR   = 0.50   # Carver: uniform pre-cost SR assumed for all instruments
VOL_TARGET    = 0.25   # portfolio vol target (25%)
IDM_CAP       = 2.50   # Carver's IDM cap (Table 4-4 in Systematic Trading)
STOP_RATIO    = 0.90   # stop when new_SR < 0.90 × best_SR (from pysystemtrade source)
CORR_START    = "2016-01-01"
MIN_OVERLAP   = 500    # minimum overlapping daily bars for pairwise correlation
CORR_FILL     = 0.30   # assumed correlation when no overlapping history

DATA_DIR       = "Data/etf"
UNIVERSE_FILE  = "Data/etf/etf_universe_curated.json"
EXPENSE_FILE   = "Data/etf/expense_ratios.json"
OUTPUT_FILE    = "Data/etf/etf_universe_greedy.json"
RESULTS_DIR    = "results"

# Individual stocks that ended up in the ETF universe — excluded from selection
INDIVIDUAL_STOCKS = {"AMT", "O", "PLD"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_returns(tickers: list[str], start: str) -> pd.DataFrame:
    frames = {}
    for tk in tickers:
        path = os.path.join(DATA_DIR, f"{tk.lower()}_1d_yf.csv")
        if not os.path.exists(path):
            continue
        s = pd.read_csv(path, parse_dates=["time"]).set_index("time")["close"]
        s = s[s.index >= start].pct_change()
        if s.dropna().__len__() > 100:
            frames[tk] = s
    return pd.DataFrame(frames).dropna(how="all")


# ── Handcraft weights ─────────────────────────────────────────────────────────
# Canonical copy lives in ewmac_backtest — import it so universe SELECTION and
# position SIZING can never drift to different weighting schemes.
from ewmac_backtest import compute_handcraft_weights as handcraft_weights


# ── Portfolio SR ──────────────────────────────────────────────────────────────

def portfolio_sr(
    instruments: list[str],
    asset_classes: dict,
    corr: pd.DataFrame,
    sr_net: dict[str, float],
) -> float:
    """
    SR_port = w^T μ / sqrt(w^T Σ w)

    where
      μ_i = sr_net[i]  (net Sharpe per instrument)
      Σ   = correlation matrix (fills to CORR_FILL where history is absent)
      w   = handcraft weights

    This is the exact formula from pysystemtrade neg_SR().
    IDM = 1/sqrt(w^T Σ w) is implicitly capped at IDM_CAP.
    """
    n = len(instruments)
    if n == 0:
        return 0.0

    w_dict = handcraft_weights(instruments, asset_classes)
    w  = np.array([w_dict[tk]    for tk in instruments])
    mu = np.array([sr_net[tk]    for tk in instruments])

    # Correlation sub-matrix — fill missing pairs with CORR_FILL
    Sigma = np.full((n, n), CORR_FILL)
    np.fill_diagonal(Sigma, 1.0)
    for i, ti in enumerate(instruments):
        for j, tj in enumerate(instruments):
            if i == j:
                continue
            if ti in corr.index and tj in corr.columns:
                v = corr.at[ti, tj]
                if not np.isnan(v):
                    Sigma[i, j] = float(v)

    port_var = float(w @ Sigma @ w)
    idm      = min(1.0 / np.sqrt(max(port_var, 1e-8)), IDM_CAP)

    # w^T μ / sqrt(w^T Σ w)  ≡  IDM × (w^T μ)
    return float(idm * (w @ mu))


# ── Greedy selection ──────────────────────────────────────────────────────────

def greedy_select(
    candidates: list[str],
    asset_classes: dict,
    corr: pd.DataFrame,
    sr_net: dict[str, float],
) -> tuple[list[str], list[float]]:
    """
    Greedy instrument selection following Carver's static optimisation.

    Returns (selected_instruments, portfolio_sr_at_each_step).

    Stopping rule (exact from optimise_small_system.py line 59-61):
      if new_SR < max_SR * 0.9:
          break
    """
    # Sort by standalone SR_net so the first pick is deterministic
    ranked = sorted(candidates, key=lambda tk: sr_net[tk], reverse=True)

    # Seed with the best single instrument
    portfolio  = [ranked[0]]
    remaining  = ranked[1:]
    current_sr = portfolio_sr(portfolio, asset_classes, corr, sr_net)
    max_sr     = current_sr
    sr_history = [current_sr]

    print(f"\n  Seed:  {portfolio[0]:<6}  SR_net={sr_net[portfolio[0]]:.4f}"
          f"  Port SR={current_sr:.4f}")

    step = 1
    while remaining:
        best_candidate = None
        best_new_sr    = -np.inf

        for candidate in remaining:
            test_sr = portfolio_sr(
                portfolio + [candidate], asset_classes, corr, sr_net)
            if test_sr > best_new_sr:
                best_new_sr    = test_sr
                best_candidate = candidate

        if best_candidate is None:
            break

        # Carver stopping criterion
        if best_new_sr < STOP_RATIO * max_sr:
            g = asset_classes.get(best_candidate, "?")
            print(f"\n  STOP  best addition {best_candidate} ({g})"
                  f"  gives SR={best_new_sr:.4f}"
                  f" < {STOP_RATIO}×{max_sr:.4f}={STOP_RATIO*max_sr:.4f}")
            break

        step += 1
        portfolio.append(best_candidate)
        remaining.remove(best_candidate)
        current_sr = best_new_sr
        max_sr     = max(max_sr, current_sr)
        sr_history.append(current_sr)

        g      = asset_classes.get(best_candidate, "?")
        er_pct = (NOTIONAL_SR - sr_net[best_candidate]) * VOL_TARGET * 100
        print(f"  +{len(portfolio):3}  {best_candidate:<6}  [{g:<20}]"
              f"  SR_net={sr_net[best_candidate]:.4f}  ER≈{er_pct:.2f}%"
              f"  → Port SR={current_sr:.4f}  (max={max_sr:.4f})")

    return portfolio, sr_history


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Greedy ETF instrument selection")
    ap.add_argument("--vol-target",    type=float, default=VOL_TARGET,
                    help="Portfolio vol target (default 0.25)")
    ap.add_argument("--notional-sr",   type=float, default=NOTIONAL_SR,
                    help="Assumed pre-cost SR for all instruments (default 0.5)")
    ap.add_argument("--stop-ratio",    type=float, default=STOP_RATIO,
                    help="Stop when new_SR < ratio × max_SR (default 0.90)")
    ap.add_argument("--no-plot",       action="store_true")
    args = ap.parse_args()

    # ── Load universe ─────────────────────────────────────────────────────────
    with open(UNIVERSE_FILE) as f:
        u = json.load(f)
    raw_tickers   = u["selected"]
    asset_classes = u.get("asset_classes", {})

    tickers = [tk for tk in raw_tickers if tk not in INDIVIDUAL_STOCKS]
    dropped_stocks = INDIVIDUAL_STOCKS & set(raw_tickers)
    print(f"Universe: {len(raw_tickers)} instruments in {UNIVERSE_FILE}")
    if dropped_stocks:
        print(f"Dropped individual stocks (no ER): {sorted(dropped_stocks)}")
    print(f"Candidate ETFs: {len(tickers)}")

    # ── Load expense ratios ────────────────────────────────────────────────────
    with open(EXPENSE_FILE) as f:
        er_data = json.load(f)
    er_pct: dict = er_data["expense_ratios_pct"]  # values in percent, e.g. 0.0945

    # ── Compute SR_net ────────────────────────────────────────────────────────
    # SR_net_i = notional_SR - SR_cost_i - size_penalty_i
    # For ETFs with fractional shares: size_penalty = 0
    # SR_cost_i = (ER_pct / 100) / vol_target
    sr_cost: dict[str, float] = {}
    sr_net:  dict[str, float] = {}
    for tk in tickers:
        er = er_pct.get(tk)
        if er is None:
            er = 0.50          # conservative default for missing ER
        sr_cost[tk] = (er / 100.0) / args.vol_target
        sr_net[tk]  = args.notional_sr - sr_cost[tk]

    print(f"\nNotional pre-cost SR : {args.notional_sr}")
    print(f"Vol target           : {args.vol_target*100:.0f}%")
    print(f"\nTop 10 most expensive (highest SR_cost):")
    for tk, sc in sorted(sr_cost.items(), key=lambda x: -x[1])[:10]:
        print(f"  {tk:<6}  ER={er_pct.get(tk,0):.4f}%"
              f"  SR_cost={sc:.4f}  SR_net={sr_net[tk]:.4f}")

    # ── Correlations ──────────────────────────────────────────────────────────
    print(f"\nLoading daily returns from {CORR_START}…")
    returns  = load_returns(tickers, CORR_START)
    ok       = [tk for tk in tickers if tk in returns.columns]
    missing  = sorted(set(tickers) - set(ok))
    if missing:
        print(f"  No data: {missing}")
    print(f"  {len(ok)} / {len(tickers)} tickers have sufficient history")

    print("  Computing pairwise correlation matrix…")
    corr = returns[ok].corr(method="pearson", min_periods=MIN_OVERLAP)

    sr_net_ok = {tk: sr_net[tk] for tk in ok}

    # ── Greedy selection ──────────────────────────────────────────────────────
    print(f"\nRunning greedy selection  (stop ratio={args.stop_ratio})…")
    portfolio, sr_history = greedy_select(
        ok, asset_classes, corr, sr_net_ok)

    final_sr = sr_history[-1] if sr_history else 0.0

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  Selected {len(portfolio)} instruments  |"
          f"  Final portfolio SR = {final_sr:.4f}")
    print(f"{'='*68}")

    groups: dict[str, list[str]] = {}
    for tk in portfolio:
        g = asset_classes.get(tk, "OTHER")
        groups.setdefault(g, []).append(tk)

    print("\n  Selected instruments by asset class:")
    for g in sorted(groups):
        print(f"    {g:<20}  {', '.join(sorted(groups[g]))}")

    not_selected = sorted(set(ok) - set(portfolio))
    print(f"\n  Not selected ({len(not_selected)}):")
    for tk in not_selected:
        g  = asset_classes.get(tk, "OTHER")
        er = er_pct.get(tk, 0)
        print(f"    {tk:<6}  [{g:<20}]  ER={er:.2f}%"
              f"  SR_net={sr_net_ok[tk]:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "selected": sorted(portfolio),
        "n_selected": len(portfolio),
        "final_portfolio_sr": round(final_sr, 4),
        "algorithm": {
            "method":          "greedy_static_optimisation",
            "reference":       "Carver, Advanced Futures Trading Strategies (2023)",
            "notional_sr":     args.notional_sr,
            "vol_target":      args.vol_target,
            "stop_ratio":      args.stop_ratio,
            "corr_start":      CORR_START,
            "min_overlap_bars":MIN_OVERLAP,
            "corr_fill":       CORR_FILL,
            "idm_cap":         IDM_CAP,
            "size_penalty":    "zero (fractional ETF shares)",
        },
        "per_instrument": {
            tk: {
                "sr_net":           round(sr_net_ok[tk], 5),
                "sr_cost":          round(sr_cost[tk],   5),
                "expense_ratio_pct":er_pct.get(tk),
                "asset_class":      asset_classes.get(tk, "OTHER"),
            }
            for tk in portfolio
        },
        "asset_classes": {tk: asset_classes.get(tk, "OTHER") for tk in portfolio},
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {OUTPUT_FILE}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not args.no_plot:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(13, 5))
        xs = list(range(1, len(sr_history) + 1))
        ax.plot(xs, sr_history, marker="o", ms=4, lw=1.5, color="steelblue",
                label="Portfolio SR after each addition")
        ax.axhline(final_sr,              color="orange",  lw=1.2, linestyle="--",
                   label=f"Final  SR = {final_sr:.3f}")
        ax.axhline(max(sr_history) * STOP_RATIO, color="crimson", lw=1.2,
                   linestyle=":", label=f"Stop threshold ({STOP_RATIO}× peak)")
        ax.set_xlabel("Number of instruments in portfolio")
        ax.set_ylabel("Estimated portfolio Sharpe ratio")
        ax.set_title(
            f"Greedy ETF instrument selection\n"
            f"Notional SR={args.notional_sr}  Vol target={args.vol_target*100:.0f}%  "
            f"Stop={args.stop_ratio}  → {len(portfolio)} instruments selected")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        out_path = os.path.join(RESULTS_DIR, "etf_greedy_selection.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Chart → {out_path}")


if __name__ == "__main__":
    main()
