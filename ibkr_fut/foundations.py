"""
foundations.py — Core math primitives for systematic futures strategies.
All equations reference ibkr_fut/calcs.txt (Carver AFTS chapters 1-9).
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy as sch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibkr_fut.pst_loader import PSTLoader

PST_CUTOFF  = pd.Timestamp.today().normalize()   # use all data through the present (PST static dump + IBKR-appended)
ANNUAL_DAYS = 256   # trading days per year            [calcs line 21]
VOL_SCALAR  = 16    # sqrt(256) — daily→annual scaling  [calcs line 27]


# ── Returns ─────────────────────────────────────────────────────────────────────

def pct_returns(prices: pd.Series) -> pd.Series:
    """Daily % returns R%_t = (P_t - P_{t-1}) / P_{t-1}.  [calcs line 10]"""
    return prices.pct_change().dropna()


def pct_returns_backadjusted(
    adjusted_prices: pd.Series,
    contract_prices: pd.Series,
) -> pd.Series:
    """
    Daily % returns from a Panama back-adjusted price series.  [calcs line 10]

    The price *change* comes from the back-adjusted series (P_t - P_{t-1}); it
    carries the roll gaps and is the true point P&L of holding the position.
    It is divided by the *current raw contract price*, NEVER the back-adjusted
    level.  Additive back-adjustment can drive the adjusted level to zero or
    negative (e.g. HANG, GILT, OMX, US10), which makes pct_change() on the
    adjusted series explode; the raw contract price is always positive, so it is
    the correct denominator for returns, volatility and position sizing.
    """
    contract = contract_prices.reindex(adjusted_prices.index).ffill()
    return (adjusted_prices.diff() / contract.shift(1)).dropna()


# ── Volatility ───────────────────────────────────────────────────────────────────

# Volatility warmup: an EWMA std built from 1-2 observations is ~0, and position
# size divides by vol, so a near-zero warmup vol produces astronomically large
# positions (e.g. GOLD: 3M contracts on day 2 at sigma=0.04%). Require this many
# returns before the estimate is usable; earlier values are left NaN and the
# backtest holds no position until the estimate is reliable. This is the
# principled replacement for the old absolute SIGMA_PCT_FLOOR, which both failed
# to stop the warmup blow-up and distorted genuinely low-vol instruments.
VOL_WARMUP_DAYS = 10


def ewma_vol(returns: pd.Series, span: int = 32) -> pd.Series:
    """
    EWMA annualised % volatility.  [calcs lines 46-47]
    lambda = 2/(span+1), span=32 per Carver AFTS. adjust=False gives the
    recursive EWMA form (matches Carver/pysystemtrade) rather than pandas'
    default re-weighted average.
    Returns annualised vol as a fraction (e.g. 0.16 = 16%). The first
    VOL_WARMUP_DAYS values are NaN — the estimate is not yet reliable, so
    positions are not sized off a meaningless near-zero warmup vol.
    """
    daily_vol = returns.ewm(span=span, min_periods=VOL_WARMUP_DAYS, adjust=False).std()
    return daily_vol * VOL_SCALAR


def blended_vol(
    returns: pd.Series,
    short_span: int = 32,
    long_window: int = 2520,
) -> pd.Series:
    """
    Blended vol estimate: 0.7 * short-run EWMA + 0.3 * ten-year rolling avg.  [calcs line 48]
    long_window=2520 ≈ 10 years of trading days.
    Returns annualised vol as a fraction (e.g. 0.16 = 16%).
    """
    short    = ewma_vol(returns, span=short_span)
    long_avg = short.rolling(window=long_window, min_periods=1).mean()
    return 0.7 * short + 0.3 * long_avg


# ── Position sizing ──────────────────────────────────────────────────────────────

def sigma_p_from_pct(prices: pd.Series, sigma_pct: pd.Series) -> pd.Series:
    """
    Daily risk in price points = (Price * sigma_%) / 16.  [calcs line 27]
    sigma_pct: annualised vol as a fraction (e.g. 0.16 for 16%).
    """
    return (prices * sigma_pct) / VOL_SCALAR


def position_size_N(
    capital: float,
    target_risk: float,
    multiplier: float,
    prices: pd.Series,
    sigma_pct: pd.Series,
    fx: float | pd.Series = 1.0,
) -> pd.Series:
    """
    Baseline position size (no forecast, no IDM).
    N = Capital * target_risk / (Multiplier * Price * FX * sigma_%).  [calcs line 26]
    """
    return (capital * target_risk) / (multiplier * prices * fx * sigma_pct)


# ── Costs ────────────────────────────────────────────────────────────────────────

def sr_cost_per_trade(
    spread_cost_points: float,
    multiplier: float,
    price: float,
    sigma_pct: float,
    commission: float = 0.0,
) -> float:
    """
    Risk-adjusted cost per round-trip trade in SR units.  [calcs lines 52-54]
    spread_cost_points: half-spread per one-way trade in price points (SpreadCost column).
    commission: round-trip commission in base currency per contract.
    """
    spread_currency = 2 * spread_cost_points * multiplier   # round-trip spread
    total_currency  = spread_currency + commission
    notional        = price * multiplier
    return total_currency / (notional * sigma_pct)


def annual_sr_cost(
    cost_per_trade: float,
    rolls_per_year: int,
    turnover: float,
) -> float:
    """
    Annual risk-adjusted cost in SR units.  [calcs line 57]
    turnover: round-trip signal trades per year (excluding rolls).
    """
    return (rolls_per_year * 2 + turnover) * cost_per_trade


# ── Performance statistics ───────────────────────────────────────────────────────

def tail_ratios(returns: pd.Series) -> dict:
    """
    Relative fat-tail ratios vs a normal distribution.  [calcs lines 14-17]
    lower_tail > 1 = fatter left tail than normal.
    upper_tail > 1 = fatter right tail than normal.
    """
    r    = returns.dropna()
    p01  = float(r.quantile(0.01))
    p30  = float(r.quantile(0.30))
    p70  = float(r.quantile(0.70))
    p99  = float(r.quantile(0.99))

    # For return distributions both p01 and p30 are typically negative;
    # their ratio is positive. A normal distribution gives ≈ 4.43.
    lower_ratio = (p01 / p30) if p30 != 0 else np.nan
    upper_ratio = (p99 / p70) if p70 != 0 else np.nan

    return {
        "lower_tail": lower_ratio / 4.43,
        "upper_tail": upper_ratio / 4.43,
    }


def _drawdown_series(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return (equity - running_max) / running_max


def performance_stats(
    equity_curve: pd.Series,
    daily_returns: pd.Series | None = None,
    costs_pct: float = 0.0,
    turnover: float = 0.0,
) -> dict:
    """
    All 10 output metrics matching Carver's per-asset-class tables.
    costs_pct: annual costs as % of capital (computed from trade log, passed in).
    turnover: average annual round-trips per instrument (from trade log).
    """
    if daily_returns is None:
        daily_returns = equity_curve.pct_change().dropna()

    r  = daily_returns.dropna()
    dd = _drawdown_series(equity_curve)

    mean_daily = float(r.mean())
    std_daily  = float(r.std())
    sr = (mean_daily * ANNUAL_DAYS) / (std_daily * VOL_SCALAR) if std_daily > 0 else np.nan

    tails = tail_ratios(r)

    return {
        "mean_annual_return_pct": round(mean_daily * ANNUAL_DAYS * 100, 2),
        "costs_pct":              round(costs_pct, 3),
        "avg_drawdown_pct":       round(float(dd[dd < 0].mean()) * 100 if (dd < 0).any() else 0.0, 2),
        "max_drawdown_pct":       round(float(dd.min()) * 100, 2),
        "std_dev_pct":            round(std_daily * VOL_SCALAR * 100, 2),
        "sharpe_ratio":           round(sr, 3),
        "turnover":               round(turnover, 1),
        "skew":                   round(float(r.skew()), 3),
        "lower_tail":             round(tails["lower_tail"], 2),
        "upper_tail":             round(tails["upper_tail"], 2),
    }


# ── Instrument weighting ─────────────────────────────────────────────────────────

def compute_corr_matrix(
    instruments: list[str],
    pst: PSTLoader,
    min_periods: int = 52,
) -> pd.DataFrame:
    """
    Pairwise return correlation matrix for all instruments up to PST_CUTOFF.
    Uses all overlapping history; NaN pairs (insufficient overlap) are filled with 0.
    """
    returns_dict: dict[str, pd.Series] = {}
    for instr in instruments:
        try:
            prices = pst.adjusted_prices(instr)
            prices = prices[prices.index <= PST_CUTOFF]
            contract = pst.multiple_prices(instr)["PRICE"]
            contract = contract[contract.index <= PST_CUTOFF]
            ret = pct_returns_backadjusted(prices, contract)
            if len(ret) >= min_periods:
                returns_dict[instr] = ret
        except Exception:
            continue

    if not returns_dict:
        return pd.DataFrame()

    corr = pd.DataFrame(returns_dict).corr(min_periods=min_periods)
    corr = corr.fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    return corr


def handcraft_weights(
    instruments: list[str],
    corr_matrix: pd.DataFrame,
) -> dict[str, float]:
    """
    Recursive correlation-based hierarchical clustering to allocate weights.  [calcs lines 65-68]
    Adapts Carver's handcraftPortfolio from carver_handcrafting.py.

    At each level, instruments are split into 2 correlation clusters;
    50% of the current level weight goes to each cluster, then recurses.

    corr_matrix: pd.DataFrame returned by compute_corr_matrix().
    Returns {instrument: weight} summing to 1.0.
    """
    valid = [i for i in instruments if i in corr_matrix.index]
    if not valid:
        return {}
    weights = _handcraft_recursive(valid, corr_matrix)
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()} if total > 0 else weights


def _handcraft_recursive(instruments: list[str], corr: pd.DataFrame) -> dict[str, float]:
    n = len(instruments)
    if n <= 2:
        return {instr: 1.0 / n for instr in instruments}

    clusters = _cluster_into_2(corr.loc[instruments, instruments].values, instruments)
    weights: dict[str, float] = {}
    cluster_share = 1.0 / len(clusters)
    for cluster in clusters:
        for instr, w in _handcraft_recursive(cluster, corr).items():
            weights[instr] = w * cluster_share
    return weights


def _cluster_into_2(corr_np: np.ndarray, names: list[str]) -> list[list[str]]:
    """Split instruments into 2 correlation-based clusters via complete linkage."""
    try:
        d = sch.distance.pdist(corr_np)
        L = sch.linkage(d, method="complete")
        cutoff = L[len(corr_np) - 2][2] - 1e-9
        ind = list(sch.fcluster(L, cutoff, "distance"))
        buckets: dict[int, list[str]] = {}
        for name, cid in zip(names, ind):
            buckets.setdefault(cid, []).append(name)
        result = list(buckets.values())
        if len(result) < 2:
            raise ValueError
        return result
    except Exception:
        mid = len(names) // 2
        return [names[:mid], names[mid:]]


def idm_from_corr(weights: dict[str, float], corr_matrix: pd.DataFrame) -> float:
    """
    IDM = 1 / sqrt(w' C w)  using the actual instrument correlation matrix.
    More accurate than the count-based lookup table.
    """
    instruments = [i for i in weights if i in corr_matrix.index]
    if not instruments:
        return 1.0
    w = np.array([weights[i] for i in instruments])
    C = corr_matrix.loc[instruments, instruments].values
    np.fill_diagonal(C, 1.0)
    variance = float(w @ C @ w)
    return 1.0 / np.sqrt(variance) if variance > 0 else 1.0


def idm_from_count(n: int) -> float:
    """Lookup-table approximation for IDM.  [calcs line 69] Use idm_from_corr() when possible."""
    if n <= 1:  return 1.00
    if n == 2:  return 1.20
    if n == 3:  return 1.48
    if n == 4:  return 1.56
    if n == 5:  return 1.70
    if n == 6:  return 1.90
    if n == 7:  return 2.10
    if n <= 14: return 2.20
    if n <= 24: return 2.30
    if n <= 29: return 2.40
    return 2.50


# ── Validation ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pst   = PSTLoader()
    INSTR = "SP500_micro"
    info  = pst.instrument_info(INSTR)
    mult  = info["Pointsize"]

    prices = pst.adjusted_prices(INSTR)
    prices = prices[prices.index <= PST_CUTOFF]

    ret   = pct_returns(prices)
    sigma = blended_vol(ret)
    sp    = sigma_p_from_pct(prices, sigma)
    N     = position_size_N(50_000_000, 0.20, mult, prices, sigma)

    last_price = float(prices.iloc[-1])
    last_sigma = float(sigma.iloc[-1])
    last_sp    = float(sp.iloc[-1])
    last_N     = float(N.iloc[-1])

    print(f"\n=== {INSTR} | data through {PST_CUTOFF.date()} ===")
    print(f"{'Last price:':<32} {last_price:.2f}")
    print(f"{'Blended vol (annual %):':<32} {last_sigma * 100:.2f}%")
    print(f"{'sigma_p (daily price pts):':<32} {last_sp:.2f}")
    print(f"{'N at $50M / 20% risk:':<32} {last_N:.1f} contracts")

    print(f"\nBlended vol last 5 rows (annual %):")
    print((sigma.tail(5) * 100).round(2).to_string())

    spread = float(info.get("SpreadCost", 0.0))
    rolls  = 4   # quarterly rolls
    c = sr_cost_per_trade(spread, mult, last_price, last_sigma)
    print(f"\nSpread cost (half-spread): {spread} pts | SR cost per trade: {c:.6f}")
    print(f"{'Turnover':<12} {'Annual SR cost':>16}")
    for to in [0.5, 1.5, 4.0, 8.0]:
        ann = annual_sr_cost(c, rolls, to)
        print(f"  {to:<10.1f} {ann:>16.5f}")

    print(f"\nIDM for 10 instruments: {idm_from_count(10)}")
    print(f"IDM for 30 instruments: {idm_from_count(30)}")
    print(f"IDM for 100 instruments: {idm_from_count(100)}")
