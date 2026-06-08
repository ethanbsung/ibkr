"""
ewmac_signals.py — EWMAC forecast engine.
Implements forecast generation for all 6 EWMAC speed variants and their combination.
All equations reference ibkr_fut/calcs.txt (Carver AFTS chapters 7-9).
"""

import sys
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibkr_fut.pst_loader import PSTLoader
from ibkr_fut.foundations import (
    PST_CUTOFF,
    blended_vol,
    pct_returns,
    sigma_p_from_pct,
    annual_sr_cost,
    sr_cost_per_trade,
)

# ── Constants ────────────────────────────────────────────────────────────────────

FORECAST_SCALARS: dict[int, float] = {
    2:  12.1,
    4:  8.53,
    8:  5.95,
    16: 4.10,
    32: 2.79,
    64: 1.91,
}

# (fast_span, slow_span) pairs for each EWMAC speed  [calcs line 122-128]
# EWMAC2 dropped: across the Jumbo it loses money even GROSS of its ~7%/yr cost
# drag (2-vs-8-day crossovers whipsaw), and removing it lifts portfolio SR
# 0.847→0.902 (full) / 0.407→0.439 (2010+) while cutting costs/turnover ~⅓.
# See ibkr_fut/exp_speeds.py and Carver AFTS (he starts at EWMAC4).
EWMAC_SPEEDS: list[tuple[int, int]] = [
    (4,  16),
    (8,  32),
    (16, 64),
    (32, 128),
    (64, 256),
]

# FDM lookup: key = sorted tuple of active fast spans  [calcs lines 193-199]
FDM_TABLE: dict[tuple[int, ...], float] = {
    (2, 4, 8, 16, 32, 64): 1.26,
    (4, 8, 16, 32, 64):    1.19,
    (8, 16, 32, 64):       1.13,
    (16, 32, 64):          1.08,
    (32, 64):              1.03,
    (64,):                 1.00,
}

# Annual turnover (round-trips/yr) per EWMAC speed, used by the cost-eligibility
# gate. These are the REALIZED standalone buffered turnovers measured from this
# engine (ibkr_fut/exp_speeds.py) and match Carver's published per-rule figures
# (~48,24,12,7,4,3 vs his 50,25,12,6.6,3.4,1.8). The old values (8,4,2.3,1.5,0.9,
# 0.5) understated turnover ~6×, so the gate saw ~⅙ the true cost and almost never
# excluded fast rules. (2 retained for reference; not in the active speed set.)
_SPEED_TURNOVER: dict[int, float] = {
    2:  48.0,
    4:  24.0,
    8:  12.0,
    16: 7.0,
    32: 4.0,
    64: 3.0,
}

COST_LIMIT_SR = 0.15   # max annual SR cost to include a trading rule  [calcs line 171]
FORECAST_CAP  = 20.0   # [calcs line 106]


# ── Core signal functions ────────────────────────────────────────────────────────

def ewma(prices: pd.Series, span: int) -> pd.Series:
    """
    Exponentially weighted moving average of price.  [calcs line 45]
    adjust=False gives Carver's recursive form EWMA_t = λ·P_t + (1-λ)·EWMA_{t-1}
    (matches pysystemtrade), rather than pandas' default re-weighted average.
    """
    return prices.ewm(span=span, min_periods=1, adjust=False).mean()


def raw_forecast(
    prices: pd.Series,
    fast_span: int,
    slow_span: int,
    sigma_p: pd.Series,
) -> pd.Series:
    """
    Raw forecast = (EWMA_fast - EWMA_slow) / sigma_p.  [calcs line 99 / 140]
    sigma_p: daily risk in price points.
    """
    return (ewma(prices, fast_span) - ewma(prices, slow_span)) / sigma_p


def scaled_forecast(raw: pd.Series, fast_span: int) -> pd.Series:
    """
    Scaled forecast = raw * forecast_scalar.  [calcs line 100 / 142]
    Scalar is speed-specific from Carver's table.
    """
    return raw * FORECAST_SCALARS[fast_span]


def capped_forecast(scaled: pd.Series) -> pd.Series:
    """Cap forecast to [-20, +20].  [calcs line 106 / 143]"""
    return scaled.clip(-FORECAST_CAP, FORECAST_CAP)


def eligible_speeds(
    cost_per_trade: float,
    rolls_per_year: int,
) -> list[int]:
    """
    Return fast spans whose annual SR cost falls below COST_LIMIT_SR.  [calcs lines 173-176]
    Turnover estimates come from Carver's table for each EWMAC speed.
    """
    eligible = []
    for fast, _ in EWMAC_SPEEDS:
        turnover = _SPEED_TURNOVER[fast]
        cost = annual_sr_cost(cost_per_trade, rolls_per_year, turnover)
        if cost < COST_LIMIT_SR:
            eligible.append(fast)
    return eligible


def combined_forecast(
    prices: pd.Series,
    sigma_p: pd.Series,
    active_fast_spans: list[int],
) -> pd.Series:
    """
    Combined forecast from multiple EWMAC speeds.  [calcs lines 205-211]

    Steps:
    1. Compute capped forecast for each active speed.
    2. Equal-weight average across speeds.
    3. Multiply by FDM from lookup table.
    4. Cap final combined forecast at ±20.
    """
    if not active_fast_spans:
        return pd.Series(0.0, index=prices.index)

    # Build speed→slow_span map for easy lookup
    speed_map = dict(EWMAC_SPEEDS)
    w = 1.0 / len(active_fast_spans)

    raw_combined = pd.Series(0.0, index=prices.index)
    for fast in active_fast_spans:
        slow = speed_map[fast]
        rf = raw_forecast(prices, fast, slow, sigma_p)
        sf = scaled_forecast(rf, fast)
        cf = capped_forecast(sf)
        raw_combined = raw_combined + w * cf

    fdm_key = tuple(sorted(active_fast_spans))
    fdm = FDM_TABLE.get(fdm_key, 1.0)

    scaled_combined = raw_combined * fdm                      # [calcs line 208]
    return scaled_combined.clip(-FORECAST_CAP, FORECAST_CAP) # [calcs line 211]


# ── Validation ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    pst   = PSTLoader()
    INSTR = "SP500_micro"
    info  = pst.instrument_info(INSTR)
    mult  = info["Pointsize"]

    prices = pst.adjusted_prices(INSTR)
    prices = prices[prices.index <= PST_CUTOFF]

    ret     = pct_returns(prices)
    sigma   = blended_vol(ret)
    sp      = sigma_p_from_pct(prices, sigma)

    last_price = float(prices.iloc[-1])
    last_sigma = float(sigma.iloc[-1])
    spread     = float(info.get("SpreadCost", 0.0))
    rolls      = 4  # quarterly

    c_per_trade = sr_cost_per_trade(spread, mult, last_price, last_sigma)
    active = eligible_speeds(c_per_trade, rolls)

    print(f"\n=== {INSTR} EWMAC forecasts | data through {PST_CUTOFF.date()} ===")
    print(f"\nEligible EWMAC speeds (cost < {COST_LIMIT_SR} SR/yr):")
    print(f"  {'Speed':<10} {'Turnover':>10} {'Ann SR cost':>14} {'Eligible':>10}")
    speed_map = dict(EWMAC_SPEEDS)
    for fast, _ in EWMAC_SPEEDS:
        to   = _SPEED_TURNOVER[fast]
        cost = annual_sr_cost(c_per_trade, rolls, to)
        elig = "YES" if fast in active else "NO"
        print(f"  EWMAC{fast:<5} {to:>10.1f} {cost:>14.5f} {elig:>10}")

    print(f"\nFDM for eligible speeds {active}: {FDM_TABLE.get(tuple(sorted(active)), 1.0)}")

    comb = combined_forecast(prices, sp, active)
    print(f"\nCombined forecast — last 20 rows:")
    print(comb.tail(20).round(2).to_string())
    print(f"\nForecast stats: min={comb.min():.2f}, max={comb.max():.2f}, "
          f"mean={comb.mean():.2f}, std={comb.std():.2f}")

    # Plot combined forecast and individual speed forecasts
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    plot_start = "2010-01-01"

    prices_plot = prices[prices.index >= plot_start]
    axes[0].plot(prices_plot.index, prices_plot.values, linewidth=0.8)
    axes[0].set_title(f"{INSTR} — Adjusted Price")
    axes[0].set_ylabel("Price (adjusted)")

    for fast in active:
        slow = speed_map[fast]
        rf = raw_forecast(prices, fast, slow, sp)
        sf = scaled_forecast(rf, fast)
        cf = capped_forecast(sf)
        axes[1].plot(cf[cf.index >= plot_start].index,
                     cf[cf.index >= plot_start].values,
                     label=f"EWMAC{fast}", linewidth=0.7, alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[1].axhline(20, color="red", linewidth=0.5, linestyle=":")
    axes[1].axhline(-20, color="red", linewidth=0.5, linestyle=":")
    axes[1].set_title("Individual Capped Forecasts")
    axes[1].set_ylabel("Forecast")
    axes[1].legend(fontsize=7, ncol=3)

    comb_plot = comb[comb.index >= plot_start]
    axes[2].plot(comb_plot.index, comb_plot.values, linewidth=0.8, color="navy")
    axes[2].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[2].axhline(20, color="red", linewidth=0.5, linestyle=":")
    axes[2].axhline(-20, color="red", linewidth=0.5, linestyle=":")
    axes[2].set_title(f"Combined Forecast (FDM={FDM_TABLE.get(tuple(sorted(active)), 1.0)})")
    axes[2].set_ylabel("Forecast")

    plt.tight_layout()
    out = "ibkr_fut/results/ewmac_forecast_validation.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\nForecast plot saved to {out}")
