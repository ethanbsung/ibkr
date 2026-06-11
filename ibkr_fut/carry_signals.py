"""
carry_signals.py — Carry forecast engine (Carver AFTS "Strategy 10: Basic Carry").

The carry forecast is a drop-in replacement for the EWMAC combined forecast: it feeds the
exact same sizing equation N = forecast·capital·IDM·w·τ / (10·mult·price·fx·σ) and the
same dynamic optimiser.  Only the forecast generation differs.

Pipeline per instrument (book screenshots in ibkr_fut/carry/):
  1. Raw carry = price of nearer contract − price of further contract (sign handled by the
     expiry difference, so it works whether the CARRY contract is nearer or further).
  2. Annualise: divide by the expiry gap in years.
  3. Risk-adjust: divide by annualised vol in price points (σ% · price = σ_p · 16).
  4. Smooth with an EWMA over several spans (5/20/60/120 business days).
  5. Scale (×30), cap to ±20.
  6. Equal-weight the spans, apply the FDM, cap the combined forecast to ±20.

Mirrors the structure of ibkr_fut/ewmac_signals.py and reuses the core math in
ibkr_fut/foundations.py.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibkr_fut.foundations import (
    blended_vol,
    pct_returns_backadjusted,
    sr_cost_per_trade,
    annual_sr_cost,
)
from ibkr_fut.ewmac_signals import FORECAST_CAP, COST_LIMIT_SR   # shared with EWMAC

# ── Constants ────────────────────────────────────────────────────────────────────

# Carry smoothing spans (business days): a week / month / quarter / half-year.  The
# book uses these four as the carry "trading rule variations" (analogous to the EWMAC
# speeds), then equal-weights and applies an FDM.
CARRY_SPANS: list[int] = [5, 20, 60, 120]

# Single forecast scalar applied to every smoothed span so the scaled forecast has an
# average absolute value of ~10 (book value, constant across instruments).
CARRY_FORECAST_SCALAR = 30.0

# Max trading days to forward-fill a stale carry term structure before the forecast
# goes NaN (and the instrument drops out of the tradable set). Carry IS the spread, so
# a frozen multiple_prices file (e.g. a broken roll calendar) must fail safe rather than
# keep trading months-old carry — see live FRESH_DAYS=5.
CARRY_FFILL_LIMIT = 5

# Forecast diversification multiplier, keyed by the sorted tuple of active spans
# (book Table 45).  All four spans → 1.04.
CARRY_FDM: dict[tuple[int, ...], float] = {
    (5, 20, 60, 120): 1.04,
    (20, 60, 120):    1.03,
    (60, 120):        1.02,
    (120,):           1.00,
}

# Annual turnover (round-trips/yr) per carry span, used by the cost-eligibility gate.
# From book Table 40 (Carry20≈3.12, Carry120≈1.22); Carry5/Carry60 approximated on the
# same monotone profile.  Carry turnover is low, so given the universe pre-filter
# (SR cost/trade ≤ 0.01) every span passes — the gate is included for faithfulness with
# Carver's trading plan but is effectively non-binding.
CARRY_SPAN_TURNOVER: dict[int, float] = {
    5:   6.0,
    20:  3.12,
    60:  1.8,
    120: 1.22,
}
# COST_LIMIT_SR is imported from ewmac_signals (single source of truth, shared by both
# styles so the combined-strategy cost gate stays consistent across trend and carry).


# ── Raw carry ──────────────────────────────────────────────────────────────────────

def _contract_months(contract: pd.Series) -> pd.Series:
    """
    Convert a pysystemtrade contract code (float YYYYMMDD with DD=00, e.g. 20260600.0)
    to an absolute month count (year·12 + month) for differencing expiries.
    """
    ym   = (contract / 100).round()        # YYYYMM
    year = (ym // 100).astype("float64")
    mon  = (ym % 100).astype("float64")
    return year * 12 + mon


def annualised_raw_carry(mp: pd.DataFrame) -> pd.Series:
    """
    Annualised raw carry in price points.  [book: "Annualised raw carry"]

        raw_carry          = PRICE − CARRY
        expiry_gap_years   = (carry_month − price_month) / 12
        annualised_raw     = raw_carry / expiry_gap_years

    Dividing by the *signed* expiry gap makes the sign correct regardless of whether the
    CARRY contract is nearer or further than the PRICE (held) contract:
      • CARRY further  (gap > 0): backwardation ⇒ PRICE>CARRY ⇒ positive carry.
      • CARRY nearer   (gap < 0): the two negatives cancel, same interpretation.
    Rows where the two contracts coincide (gap = 0) are NaN.
    """
    gap_years = (_contract_months(mp["CARRY_CONTRACT"])
                 - _contract_months(mp["PRICE_CONTRACT"])) / 12.0
    gap_years = gap_years.replace(0.0, np.nan)
    return (mp["PRICE"] - mp["CARRY"]) / gap_years


# ── Forecast ─────────────────────────────────────────────────────────────────────

def carry_eligible_spans(cost_per_trade: float, rolls_per_year: int) -> list[int]:
    """
    Return the carry spans whose annual SR cost falls below COST_LIMIT_SR.  Mirrors
    ewmac_signals.eligible_speeds — slower spans trade less and so survive higher costs.
    """
    eligible = []
    for span in CARRY_SPANS:
        cost = annual_sr_cost(cost_per_trade, rolls_per_year, CARRY_SPAN_TURNOVER[span])
        if cost < COST_LIMIT_SR:
            eligible.append(span)
    return eligible


def carry_span_forecasts(
    mp: pd.DataFrame,
    raw_price: pd.Series,
    sigma_pct: pd.Series,
    active_spans: list[int],
) -> dict[int, pd.Series]:
    """
    Per-span capped carry forecast (each scaled so avg |value| ≈ 10, capped ±20), aligned
    to raw_price.index.  Shared building block: carry_forecast (Strategy 10) equal-weights
    these + applies the per-style FDM; carry_trend_signals (Strategy 11) reuses them at the
    per-rule level for the combined 60/40 blend — so the carry math lives in one place.

    raw_price : front/PRICE contract price (always positive) — risk-adjust denominator.
    sigma_pct : annualised vol as a fraction (blended_vol), same series used for sizing.

    Stale-data fail-safe: annualised carry is forward-filled at most CARRY_FFILL_LIMIT
    trading days; PAST that horizon the carry is stale and the span forecast is forced to
    NaN, so the instrument drops out of the tradable set rather than trading months-old
    carry.  This masking is essential — a bare ewm().mean() carries its last value forward
    through NaNs indefinitely, which would silently defeat the ffill bound (see live
    FRESH_DAYS).  Rows before the first carry observation are NaN for the same reason.

    Returns {} for an empty span list (so callers with no carry rule never touch mp).
    """
    if not active_spans:
        return {}
    idx = raw_price.index
    arc = annualised_raw_carry(mp).reindex(idx).ffill(limit=CARRY_FFILL_LIMIT)
    stale = arc.isna()   # NaN after the ffill horizon (or before any carry observation)

    # Risk-adjusted carry (dimensionless, ~SR units): annualised carry in price points
    # divided by annualised vol in price points (= σ% · price = σ_p · 16).
    denom = (raw_price * sigma_pct).replace(0.0, np.nan)
    carry_raw = arc / denom

    out: dict[int, pd.Series] = {}
    for span in active_spans:
        smoothed = carry_raw.ewm(span=span, min_periods=1, adjust=False).mean()
        capped   = (smoothed * CARRY_FORECAST_SCALAR).clip(-FORECAST_CAP, FORECAST_CAP)
        out[span] = capped.where(~stale)   # force NaN where carry is stale/absent
    return out


def carry_forecast(
    mp: pd.DataFrame,
    raw_price: pd.Series,
    sigma_pct: pd.Series,
    active_spans: list[int],
) -> pd.Series:
    """
    Combined, capped carry forecast aligned to raw_price.index (Carver Strategy 10):
    equal-weight the per-span forecasts, apply the FDM, cap to ±20.  NaN wherever the
    carry is stale (see carry_span_forecasts), which fails the instrument out safely.
    """
    if not active_spans:
        return pd.Series(0.0, index=raw_price.index)

    per_span = carry_span_forecasts(mp, raw_price, sigma_pct, active_spans)
    w = 1.0 / len(active_spans)
    raw_combined = sum(w * fc for fc in per_span.values())

    fdm = CARRY_FDM.get(tuple(sorted(active_spans)), 1.0)
    return (raw_combined * fdm).clip(-FORECAST_CAP, FORECAST_CAP)


def carry_instrument_signals(spec, mp: pd.DataFrame) -> dict | None:
    """
    Carry analogue of backtest_ewmac.instrument_signals.  Returns the vol and combined
    carry-forecast series for one instrument plus its active spans, or None if the stats
    are invalid or no span is cost-eligible.  Same return shape so the dynamic-opt sim
    loop consumes it unchanged.

    spec : InstrumentSpec (from backtest_ewmac._pst_spec) — prices/raw_price/fx/mult/etc.
    mp   : pst.multiple_prices(instr) DataFrame with PRICE/CARRY/*_CONTRACT columns, or
           None if the instrument has no term-structure file (carry can't be computed).
    """
    if mp is None:
        return None
    ret   = pct_returns_backadjusted(spec.prices, spec.raw_price)
    sigma = blended_vol(ret)

    med_price = float(spec.raw_price[spec.raw_price > 0].median())
    med_sigma = float(sigma.dropna().median())
    if med_sigma <= 0 or med_price <= 0 or np.isnan(med_price):
        return None

    # SR cost per trade at median price/vol → which carry spans are cheap enough to run.
    c_trade = sr_cost_per_trade(spec.spread, spec.mult, med_price, med_sigma, spec.commission)
    active  = carry_eligible_spans(c_trade, spec.rolls)
    if not active:
        return None

    sigma_a    = sigma.reindex(spec.prices.index, method="ffill")
    forecast_s = carry_forecast(mp, spec.raw_price, sigma_a, active)
    return {"sigma": sigma_a, "forecast": forecast_s, "active": active}


# ── Validation ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ibkr_fut.pst_loader import PSTLoader
    from ibkr_fut.backtest_ewmac import _pst_spec

    pst = PSTLoader()
    print(f"Carry spans={CARRY_SPANS}  scalar={CARRY_FORECAST_SCALAR}  "
          f"FDM={CARRY_FDM[tuple(CARRY_SPANS)]}\n")
    print(f"{'Instrument':<14} {'spans':<18} {'last_fc':>8} {'avg|fc|':>8} "
          f"{'min':>7} {'max':>7}  note")
    print("-" * 78)
    checks = [
        ("SP500_micro",  "equity (expect contango / negative)"),
        ("CRUDE_W_mini", "energy (expect backwardation / positive)"),
        ("BUND",         "bond"),
        ("GAS_US_mini",  "energy (seasonal)"),
        ("GOLD_micro",   "metal"),
        ("US10",         "bond"),
    ]
    for instr, note in checks:
        spec = _pst_spec(instr)
        if spec is None:
            print(f"{instr:<14} no spec")
            continue
        mp  = pst.multiple_prices(instr)
        sig = carry_instrument_signals(spec, mp)
        if sig is None:
            print(f"{instr:<14} no eligible spans")
            continue
        f = sig["forecast"].dropna()
        print(f"{instr:<14} {str(sig['active']):<18} {f.iloc[-1]:>8.2f} "
              f"{f.abs().mean():>8.2f} {f.min():>7.1f} {f.max():>7.1f}  {note}")
