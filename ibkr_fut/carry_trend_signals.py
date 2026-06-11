"""
carry_trend_signals.py — Combined carry + trend forecast engine
(Carver AFTS "Strategy 11: Combined carry and trend").

Trend and carry are two largely-uncorrelated return sources (divergent vs convergent).
Strategy 11 blends them into ONE forecast that feeds the identical sizing equation
N = forecast·capital·IDM·w·τ / (10·mult·price·fx·σ) and the same dynamic optimiser — only
the forecast generation differs from the standalone trend/carry engines.

Combination (book screenshots in ibkr_fut/carry_trend/), per instrument:
  1. Build each *individual* rule's capped forecast (avg≈10, ±20), PRE-FDM:
       • trend: each cost-eligible EWMAC speed via ewmac_signals (raw→scaled→capped)
       • carry: each cost-eligible carry span via carry_signals' building blocks
  2. Forecast weights (Carver Table 51, top-down, weights sum to 1):
       w_trend_each = TREND_WEIGHT / n_trend ,  w_carry_each = (1-TREND_WEIGHT) / n_carry
     with TREND_WEIGHT = 0.50 (50% divergent/trend, 50% convergent/carry — the split the
     live system trades; Carver's book default is 60/40).
     If one style has zero active rules the other style gets 100% (1/n each).
  3. Weighted sum of all per-rule capped forecasts.
  4. Multiply by a SINGLE forecast diversification multiplier keyed by the *total* active
     rule count (Carver Table 52) — NOT the per-style FDMs in ewmac/carry_signals.
  5. Cap the combined forecast to ±20.

Reuses ewmac_signals + carry_signals building blocks; neither is modified.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibkr_fut.foundations import (
    blended_vol,
    pct_returns_backadjusted,
    sigma_p_from_pct,
    sr_cost_per_trade,
)
from ibkr_fut.ewmac_signals import (
    EWMAC_SPEEDS,
    FORECAST_CAP,            # ±20, shared across both styles
    eligible_speeds,
    raw_forecast,
    scaled_forecast,
    capped_forecast,
)
from ibkr_fut.carry_signals import (
    carry_eligible_spans,
    carry_span_forecasts,   # shared per-span carry building block (incl. stale fail-safe)
)

# ── Constants ────────────────────────────────────────────────────────────────────

# Allocation between the two styles. Trend is divergent, carry is convergent. This single
# number drives every per-rule weight and is the one source of truth shared by the live
# system and the backtest. Set to 0.50 (50/50) — the split the live system trades; Carver's
# book default is 60/40 (convergent 40% / divergent 60%).
TREND_WEIGHT = 0.50

# Combined forecast diversification multiplier keyed by the TOTAL number of active
# trading rules across both styles (Carver Table 52 — "for any strategy").  Our maximum
# is 5 EWMAC speeds + 4 carry spans = 9 rules — all of which are exact keys below, so in
# practice combined_fdm is a direct lookup.  (The 3→4 jump 1.03→1.23 is in the book as
# printed.)  Keys 10–13 are retained as a small margin if the rule set is ever expanded.
COMBINED_FDM: dict[int, float] = {
    1: 1.00, 2: 1.02, 3: 1.03, 4: 1.23, 5: 1.25, 6: 1.27, 7: 1.29,
    8: 1.32, 9: 1.34, 10: 1.35, 11: 1.36, 12: 1.38, 13: 1.39,
}


def combined_fdm(n_rules: int) -> float:
    """FDM for a given total active-rule count (Carver Table 52).  Clamped to the table:
    n≤0 → 1.0, n above the table → the largest tabulated value."""
    if n_rules <= 0:
        return 1.0
    if n_rules in COMBINED_FDM:
        return COMBINED_FDM[n_rules]
    return COMBINED_FDM[max(COMBINED_FDM)]   # n beyond the table (cannot occur with 9 rules)


# ── Per-rule capped forecasts (pre-FDM) ─────────────────────────────────────────────

def _trend_capped_forecasts(
    prices: pd.Series,
    sigma_p: pd.Series,
    active_fast: list[int],
) -> dict[int, pd.Series]:
    """Capped EWMAC forecast for each active speed (reuses ewmac_signals atoms)."""
    speed_map = dict(EWMAC_SPEEDS)
    out: dict[int, pd.Series] = {}
    for fast in active_fast:
        rf = raw_forecast(prices, fast, speed_map[fast], sigma_p)
        out[fast] = capped_forecast(scaled_forecast(rf, fast))
    return out


# ── Combined forecast ───────────────────────────────────────────────────────────────

def combined_carry_trend_forecast(
    prices: pd.Series,
    raw_price: pd.Series,
    sigma_p: pd.Series,
    sigma_pct: pd.Series,
    mp: pd.DataFrame,
    active_trend: list[int],
    active_carry: list[int],
    trend_weight: float = TREND_WEIGHT,
) -> pd.Series:
    """
    Steps 2–5: blend the per-rule capped forecasts 60/40 (trend/carry), apply the combined
    FDM (by total active-rule count), and cap to ±20.  Aligned to prices.index.

    Robust to a stale/missing carry leg: carry_span_forecasts returns NaN where the term
    structure is stale, and the trend forecast is defined wherever price/vol exist.  We
    therefore blend PER TIMESTAMP — using both styles where both have signal, and falling
    back to whichever style is present (at its own 100% weight + own-count FDM) where the
    other is NaN — instead of letting a NaN carry leg wipe a valid trend forecast.  When
    carry is present (the normal case) this is identical to the static 60/40 blend.
    """
    idx = prices.index
    n_t, n_c = len(active_trend), len(active_carry)
    if n_t == 0 and n_c == 0:
        return pd.Series(0.0, index=idx)

    trend_fcs = _trend_capped_forecasts(prices, sigma_p, active_trend)
    carry_fcs = carry_span_forecasts(mp, raw_price, sigma_pct, active_carry)

    # Equal-weight mean within each style; an all-NaN row → NaN (= no signal this bar).
    mt = (pd.DataFrame(trend_fcs).mean(axis=1).reindex(idx)
          if trend_fcs else pd.Series(np.nan, index=idx))
    mc = (pd.DataFrame(carry_fcs).mean(axis=1).reindex(idx)
          if carry_fcs else pd.Series(np.nan, index=idx))

    fdm_both = combined_fdm(n_t + n_c)
    fdm_t    = combined_fdm(n_t)
    fdm_c    = combined_fdm(n_c)

    both   = mt.notna() & mc.notna()
    t_only = mt.notna() & mc.isna()
    c_only = mt.isna()  & mc.notna()

    combined = pd.Series(np.nan, index=idx)
    combined[both]   = (trend_weight * mt + (1.0 - trend_weight) * mc)[both] * fdm_both
    combined[t_only] = mt[t_only] * fdm_t
    combined[c_only] = mc[c_only] * fdm_c
    return combined.clip(-FORECAST_CAP, FORECAST_CAP)


def carry_trend_instrument_signals(
    spec,
    mp: pd.DataFrame,
    trend_weight: float = TREND_WEIGHT,
) -> dict | None:
    """
    Combined-forecast analogue of backtest_ewmac.instrument_signals /
    carry_signals.carry_instrument_signals.  Returns {"sigma", "forecast",
    "active_trend", "active_carry"} — the same sigma/forecast shape the dynamic-opt sim
    loop consumes — or None if the stats are invalid or NEITHER style has an eligible rule.

    spec : InstrumentSpec (from backtest_ewmac._pst_spec).
    mp   : pst.multiple_prices(instr) — PRICE/CARRY/*_CONTRACT columns, or None if the
           instrument has no term-structure file (then the combined forecast is trend-only).
    """
    ret   = pct_returns_backadjusted(spec.prices, spec.raw_price)
    sigma = blended_vol(ret)
    sp    = sigma_p_from_pct(spec.raw_price, sigma)   # points vol off the raw price (trend)

    med_price = float(spec.raw_price[spec.raw_price > 0].median())
    med_sigma = float(sigma.dropna().median())
    if med_sigma <= 0 or med_price <= 0 or np.isnan(med_price):
        return None

    # Same cost gate as each standalone engine, off median price/vol (time-invariant).
    c_trade = sr_cost_per_trade(spec.spread, spec.mult, med_price, med_sigma, spec.commission)
    active_trend = eligible_speeds(c_trade, spec.rolls)
    # No term-structure file => no carry rules; the blend degrades to trend-only.
    active_carry = [] if mp is None else carry_eligible_spans(c_trade, spec.rolls)
    if not active_trend and not active_carry:
        return None

    sigma_a = sigma.reindex(spec.prices.index, method="ffill")
    sp_a    = sp.reindex(spec.prices.index, method="ffill")
    forecast_s = combined_carry_trend_forecast(
        spec.prices, spec.raw_price, sp_a, sigma_a, mp,
        active_trend, active_carry, trend_weight,
    )
    return {
        "sigma": sigma_a, "forecast": forecast_s,
        "active_trend": active_trend, "active_carry": active_carry,
    }


# ── Validation ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ibkr_fut.pst_loader import PSTLoader
    from ibkr_fut.backtest_ewmac import _pst_spec, instrument_signals
    from ibkr_fut.carry_signals import carry_instrument_signals

    pst = PSTLoader()
    print(f"TREND_WEIGHT={TREND_WEIGHT}  (trend {TREND_WEIGHT:.0%} / carry {1-TREND_WEIGHT:.0%})")
    print(f"FDM by rule count: " + ", ".join(f"{n}:{combined_fdm(n)}" for n in range(1, 10)))
    print(f"\n{'Instrument':<14} {'trend':<14} {'carry':<14} {'last_c':>7} {'avg|c|':>7} "
          f"{'min':>6} {'max':>6}  {'vs trend/carry last':>22}")
    print("-" * 108)
    checks = ["SP500_micro", "CRUDE_W_mini", "GOLD_micro", "US10", "BUND", "GAS_US_mini"]
    for instr in checks:
        spec = _pst_spec(instr)
        if spec is None:
            print(f"{instr:<14} no spec")
            continue
        mp  = pst.multiple_prices(instr)
        sig = carry_trend_instrument_signals(spec, mp)
        if sig is None:
            print(f"{instr:<14} no eligible rules")
            continue
        c = sig["forecast"].dropna()
        # standalone last forecasts for comparison
        ts = instrument_signals(spec)
        cs = carry_instrument_signals(spec, mp)
        t_last = float(ts["forecast"].dropna().iloc[-1]) if ts else float("nan")
        k_last = float(cs["forecast"].dropna().iloc[-1]) if cs else float("nan")
        print(f"{instr:<14} {str(sig['active_trend']):<14} {str(sig['active_carry']):<14} "
              f"{c.iloc[-1]:>7.2f} {c.abs().mean():>7.2f} {c.min():>6.1f} {c.max():>6.1f}  "
              f"trend={t_last:>6.1f} carry={k_last:>6.1f}")
