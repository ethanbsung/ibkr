"""
instrument_selection.py — Periodic instrument eligibility filter.

Implements Carver's four bad-market filters and outputs a tradable universe
for use in backtest_dynamic.py and live_dynamic.py.

Run periodically (weekly/monthly) to refresh the eligible set:
    python instrument_selection.py              # full report for UNIVERSE
    python instrument_selection.py --save       # also write eligible set to JSON

Filters applied (Carver AFTS / pysystemtrade duplicate_remove_markets.py):
  1. SR cost per trade > MAX_SR_COST (0.01)    → too expensive
  2. Daily volume contracts < MIN_VOLUME_CONTRACTS (100)  → not enough liquidity
  3. Daily volume risk < MIN_VOLUME_RISK ($1.5M/day)      → not enough liquidity
  4. Annual vol% < MIN_ANN_VOL_PCT (5.0%)      → too safe to size meaningfully

The min annual vol threshold is derived from:
  min_vol = max_forecast_scalar × IDM × inst_weight × risk_target / max_leverage
          = 2.0 × 2.5 × 0.04 × 0.25 / 1.0 = 5.0%

Volume data comes from volume_collector.py's cache (Data/volume_cache.csv).
Price/vol/cost data comes from PST price data via PSTLoader.

Output written to Data/eligible_instruments.json when --save is passed.
"""

import os
import sys
import json
import argparse
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibkr_fut.pst_loader import PSTLoader
from ibkr_fut.instrument_universe import UNIVERSE
from ibkr_fut.foundations import (
    blended_vol,
    pct_returns_backadjusted,
    sr_cost_per_trade,
    PST_CUTOFF,
)
from ibkr_fut.volume_collector import load_cache, compute_volume_risk

# ── Filter thresholds ─────────────────────────────────────────────────────────

MAX_SR_COST            = 0.01    # SR units per trade (Carver hard limit)
MIN_VOLUME_CONTRACTS   = 100     # avg daily contracts (20-day lookback)
MIN_VOLUME_RISK        = 1.5     # $M/day annualised risk volume
MIN_ANN_VOL_PCT        = 5.0     # annual % vol minimum (too-safe filter)
# min_vol = 2.0 × 2.5 × 0.04 × 0.25 / 1.0 = 5.0%
MIN_HISTORY_DAYS       = 512     # minimum price history (same as backtest)

_REPO        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ELIGIBLE_PATH = os.path.join(_REPO, "Data", "eligible_instruments.json")


# ── Per-instrument stats ───────────────────────────────────────────────────────

def _instrument_stats(pst: PSTLoader, instrument: str) -> dict:
    """
    Compute cost, vol, and history length for one instrument.
    Returns a dict with keys: sr_cost, ann_vol_pct, history_days, ok (bool).
    Returns None if price data is unavailable.
    """
    try:
        info  = pst.instrument_info(instrument)
        mult  = float(info["Pointsize"])
        spread = float(info.get("SpreadCost", np.nan))
        # Commission: PerBlock (flat) or Percentage * price * mult
        per_block  = float(info.get("PerBlock", 0.0) or 0.0)
        commission = per_block * 2  # round-trip

        raw = pst.multiple_prices(instrument)["PRICE"]
        raw = raw[raw.index <= PST_CUTOFF].dropna()
        adj = pst.adjusted_prices(instrument)
        adj = adj[adj.index <= PST_CUTOFF]

        if raw.empty or adj.empty:
            return None

        ret  = pct_returns_backadjusted(adj, raw)
        vol  = blended_vol(ret).dropna()

        if vol.empty:
            return None

        last_price = float(raw.iloc[-1])
        last_vol   = float(vol.iloc[-1])   # annualised fraction
        history    = len(ret.dropna())

        if pd.isna(spread) or last_price <= 0 or last_vol <= 0:
            return None

        cost = sr_cost_per_trade(spread, mult, last_price, last_vol, commission)

        return {
            "sr_cost":       cost,
            "ann_vol_pct":   last_vol * 100.0,
            "history_days":  history,
        }
    except Exception:
        return None


# ── Selection logic ───────────────────────────────────────────────────────────

def build_selection_table(
    pst: PSTLoader,
    universe: dict[str, str] | None = None,
    volume_cache: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the full selection table for all instruments in the universe.

    Returns a DataFrame indexed by instrument with columns:
      asset_class, sr_cost, ann_vol_pct, history_days, avg_contracts,
      volume_risk_mday, fail_cost, fail_vol_count, fail_vol_risk,
      fail_too_safe, fail_history, eligible
    """
    universe = universe or UNIVERSE
    volume_cache = volume_cache if volume_cache is not None else load_cache()
    vol_map = dict(zip(volume_cache["instrument"], volume_cache["avg_contracts"]))

    rows = []
    for instr, cls in universe.items():
        stats = _instrument_stats(pst, instr)
        avg_c = vol_map.get(instr, np.nan)
        risk  = compute_volume_risk(pst, instr, avg_c)

        if stats is None:
            rows.append({
                "instrument":    instr,
                "asset_class":   cls,
                "sr_cost":       np.nan,
                "ann_vol_pct":   np.nan,
                "history_days":  0,
                "avg_contracts": avg_c,
                "volume_risk_mday": risk,
                "fail_cost":     False,
                "fail_vol_count": False,
                "fail_vol_risk": False,
                "fail_too_safe": False,
                "fail_history":  True,
                "eligible":      False,
                "reason":        "no price data",
            })
            continue

        fail_cost      = stats["sr_cost"]     > MAX_SR_COST
        fail_vol_count = (not np.isnan(avg_c)) and (avg_c < MIN_VOLUME_CONTRACTS)
        fail_vol_risk  = (not np.isnan(risk))  and (risk  < MIN_VOLUME_RISK)
        fail_too_safe  = stats["ann_vol_pct"] < MIN_ANN_VOL_PCT
        fail_history   = stats["history_days"] < MIN_HISTORY_DAYS

        reasons = []
        if fail_cost:      reasons.append(f"sr_cost={stats['sr_cost']:.4f}")
        if fail_vol_count: reasons.append(f"contracts={avg_c:.0f}")
        if fail_vol_risk:  reasons.append(f"vol_risk={risk:.2f}M")
        if fail_too_safe:  reasons.append(f"vol={stats['ann_vol_pct']:.1f}%")
        if fail_history:   reasons.append(f"history={stats['history_days']}d")

        eligible = not any([fail_cost, fail_vol_count, fail_vol_risk,
                            fail_too_safe, fail_history])
        # Volume unknowns (NaN) do not disqualify — they are "unverified"
        if np.isnan(avg_c):
            fail_vol_count = False
        if np.isnan(risk):
            fail_vol_risk = False
        eligible = not any([fail_cost, fail_vol_count, fail_vol_risk,
                            fail_too_safe, fail_history])

        rows.append({
            "instrument":       instr,
            "asset_class":      cls,
            "sr_cost":          stats["sr_cost"],
            "ann_vol_pct":      stats["ann_vol_pct"],
            "history_days":     stats["history_days"],
            "avg_contracts":    avg_c,
            "volume_risk_mday": risk,
            "fail_cost":        fail_cost,
            "fail_vol_count":   fail_vol_count,
            "fail_vol_risk":    fail_vol_risk,
            "fail_too_safe":    fail_too_safe,
            "fail_history":     fail_history,
            "eligible":         eligible,
            "reason":           "; ".join(reasons) if reasons else "",
        })

    return pd.DataFrame(rows).set_index("instrument")


def eligible_instruments(
    pst: PSTLoader,
    universe: dict[str, str] | None = None,
    volume_cache: pd.DataFrame | None = None,
) -> list[str]:
    """Return sorted list of instruments that pass all filters."""
    tbl = build_selection_table(pst, universe, volume_cache)
    return sorted(tbl.index[tbl["eligible"]].tolist())


# ── Report ─────────────────────────────────────────────────────────────────────

def print_selection_report(
    pst: PSTLoader,
    universe: dict[str, str] | None = None,
    volume_cache: pd.DataFrame | None = None,
) -> None:
    universe = universe or UNIVERSE
    tbl = build_selection_table(pst, universe, volume_cache)

    eligible  = tbl[tbl["eligible"]]
    ineligible = tbl[~tbl["eligible"]]

    print(f"\n{'=' * 90}")
    print(f"INSTRUMENT SELECTION REPORT  |  {date.today()}  |  universe={len(tbl)}")
    print(f"{'=' * 90}")
    print(f"\nThresholds:")
    print(f"  SR cost per trade    <= {MAX_SR_COST}")
    print(f"  Daily volume (con)   >= {MIN_VOLUME_CONTRACTS}")
    print(f"  Daily volume ($M)    >= {MIN_VOLUME_RISK}")
    print(f"  Annual vol           >= {MIN_ANN_VOL_PCT}%")
    print(f"  Min history          >= {MIN_HISTORY_DAYS} days")
    print()

    # Summary by asset class
    print(f"{'Asset class':<12} {'Total':>6} {'Eligible':>9} {'Ineligible':>11}")
    print("-" * 42)
    for cls in sorted(tbl["asset_class"].unique()):
        sub = tbl[tbl["asset_class"] == cls]
        n_ok = sub["eligible"].sum()
        print(f"  {cls:<10} {len(sub):>6} {n_ok:>9} {len(sub)-n_ok:>11}")
    print(f"  {'TOTAL':<10} {len(tbl):>6} {len(eligible):>9} {len(ineligible):>11}")
    print()

    # Eligible instruments
    print(f"ELIGIBLE ({len(eligible)}):")
    print(f"{'Instrument':<20} {'Class':<8} {'SR cost':>8} {'Vol%':>6} {'History':>8} {'AvgCon':>8} {'Risk$M':>7}")
    print("-" * 70)
    for instr, row in eligible.sort_values("asset_class").iterrows():
        avg_c = f"{row['avg_contracts']:8.0f}" if not np.isnan(row['avg_contracts']) else "       ?"
        risk  = f"{row['volume_risk_mday']:7.2f}" if not np.isnan(row['volume_risk_mday']) else "      ?"
        print(f"  {instr:<18} {row['asset_class']:<8} {row['sr_cost']:8.4f} "
              f"{row['ann_vol_pct']:6.1f} {row['history_days']:8} {avg_c} {risk}")
    print()

    # Ineligible instruments with reasons
    if len(ineligible) > 0:
        print(f"INELIGIBLE ({len(ineligible)}):")
        print(f"{'Instrument':<20} {'Class':<8} {'Reason'}")
        print("-" * 70)
        for instr, row in ineligible.sort_values("asset_class").iterrows():
            print(f"  {instr:<18} {row['asset_class']:<8} {row['reason']}")
    print()

    # Volume unknowns
    unknown_vol = tbl[tbl["avg_contracts"].isna() & tbl["eligible"]]
    if len(unknown_vol) > 0:
        print(f"NOTE: {len(unknown_vol)} eligible instruments have unverified volume "
              f"(run volume_collector.py to fetch):")
        for instr in unknown_vol.index:
            print(f"  {instr}")
        print()


# ── Save eligible set ─────────────────────────────────────────────────────────

def save_eligible(instruments: list[str]) -> None:
    data = {
        "date":        date.today().isoformat(),
        "instruments": instruments,
    }
    with open(ELIGIBLE_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Eligible instruments saved to {ELIGIBLE_PATH}")


def load_eligible() -> list[str] | None:
    """Load previously saved eligible instrument list, or None if not found."""
    if not os.path.exists(ELIGIBLE_PATH):
        return None
    with open(ELIGIBLE_PATH) as f:
        data = json.load(f)
    return data.get("instruments")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instrument selection filter report")
    parser.add_argument("--save", action="store_true",
                        help="Save eligible instrument list to Data/eligible_instruments.json")
    parser.add_argument("--jumbo-only", action="store_true",
                        help="Filter only the JUMBO 99 instruments (not full UNIVERSE)")
    args = parser.parse_args()

    from ibkr_fut.jumbo import JUMBO
    universe = JUMBO if args.jumbo_only else UNIVERSE

    pst    = PSTLoader()
    volume = load_cache()
    print_selection_report(pst, universe, volume)

    if args.save:
        elig = eligible_instruments(pst, universe, volume)
        save_eligible(elig)
        print(f"\n{len(elig)} eligible instruments saved.")
