"""
dynamic_opt.py — Carver's dynamic portfolio optimisation (AFTS).

Every day we hold the set of *integer* futures positions that most closely
matches the ideal unrounded portfolio, by minimising the standard deviation of
the tracking-error portfolio. A cost penalty discourages expensive trades and
tracking-error buffering suppresses needless turnover.

All equations reference ibkr_fut/calcs.txt (Carver AFTS, dynamic optimisation
section, lines 217-332). Carver's own reference code in rob_port/ (dynamic_basic.py,
dynamic_costs_buffering.py, carver_corr_est.py, carv_obj_func.py) was used only as
a correctness reference; this is a clean numpy reimplementation.

Phase 1: covariance estimation (CovarianceEstimator).
Phase 2: greedy integer optimiser (no costs).
Phase 3: cost penalty.
Phase 4: tracking-error buffering.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibkr_fut.foundations import ewma_vol, pct_returns_backadjusted

# ── Constants (from the book / Carver reference code) ─────────────────────────────
COST_MULTIPLIER   = 50.0    # cost penalty weight in the objective       [calcs line 272]
BUFFER_FRACTION   = 0.05    # tracking-error buffer = 0.05 * target_risk [calcs line 303]
CORR_SPAN_WEEKS   = 25      # EWMA span for weekly-return correlations    (carver_corr_est)
VOL_SPAN_DAYS     = 32      # EWMA span for annualised vol (diagonal)     [calcs line 236]
CORR_SHRINKAGE    = 0.5     # shrink sample correlation toward identity   (carver_corr_est)
CORR_MIN_PERIODS  = 3       # min weekly obs before a correlation is used


# ══════════════════════════════════════════════════════════════════════════════════
# Phase 1 — Covariance estimation
# ══════════════════════════════════════════════════════════════════════════════════

def shrink_correlation(corr: np.ndarray, shrinkage: float = CORR_SHRINKAGE) -> np.ndarray:
    """
    Shrink a sample correlation matrix toward the identity (zero off-diagonal prior).
    shrunk = shrinkage * I + (1 - shrinkage) * corr.   (carver_corr_est shrink_to_offdiag)

    With shrinkage=0.5 the off-diagonals are halved and the diagonal stays 1. This
    guarantees a positive-definite matrix when the sample correlation is PSD
    (eigenvalues become 0.5 + 0.5*lambda >= 0.5), which keeps the tracking-error
    quadratic form well behaved.
    """
    n = corr.shape[0]
    return shrinkage * np.eye(n) + (1.0 - shrinkage) * corr


def covariance_from_corr_and_vol(corr: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """
    Build a covariance matrix from a correlation matrix and a vector of annualised
    standard deviations:  Sigma = diag(vol) . corr . diag(vol).   [calcs line 232]
    """
    D = np.diag(vol)
    return D @ corr @ D


def weekly_pct_returns(
    adjusted_prices: pd.Series,
    contract_prices: pd.Series,
) -> pd.Series:
    """
    Weekly percentage returns for correlation estimation.   [calcs line 235-236]

    Resample to weekly (Friday) closes, then take the back-adjusted price *change*
    over the divided by the prior week's raw contract price — the same logic as the
    daily pct_returns_backadjusted, sampled weekly. A six-month-ish EWMA on these is
    the correlation look-back Carver uses.
    """
    contract = contract_prices.reindex(adjusted_prices.index).ffill()
    adj_w = adjusted_prices.resample("W-FRI").last()
    con_w = contract.resample("W-FRI").last()
    return (adj_w.diff() / con_w.shift(1)).rename(adjusted_prices.name)


class CovarianceEstimator:
    """
    Precomputes time-varying correlation (weekly EWMA) and volatility (daily EWMA)
    for a universe of instruments, and serves a shrunk covariance matrix for any
    (date, live-subset) on demand.

    adjusted_prices_dict / contract_prices_dict: {instrument: pd.Series}.
    """

    def __init__(
        self,
        adjusted_prices_dict: dict[str, pd.Series],
        contract_prices_dict: dict[str, pd.Series],
        corr_span_weeks: int = CORR_SPAN_WEEKS,
        vol_span_days: int = VOL_SPAN_DAYS,
        shrinkage: float = CORR_SHRINKAGE,
    ):
        self.instruments = list(adjusted_prices_dict.keys())
        self._col_pos = {name: i for i, name in enumerate(self.instruments)}
        self.shrinkage = shrinkage

        # Daily annualised vol per instrument (fraction), aligned into one frame.
        vol_cols = {}
        weekly_cols = {}
        for instr in self.instruments:
            adj = adjusted_prices_dict[instr]
            con = contract_prices_dict[instr]
            ret = pct_returns_backadjusted(adj, con)
            vol_cols[instr] = ewma_vol(ret, span=vol_span_days)
            weekly_cols[instr] = weekly_pct_returns(adj, con)

        self.vol_df = pd.DataFrame(vol_cols).sort_index().reindex(columns=self.instruments)
        self._vol_index = self.vol_df.index
        self._vol_values = self.vol_df.values   # (Tv, n), positional lookup

        weekly_df = pd.DataFrame(weekly_cols).sort_index().reindex(columns=self.instruments)
        # Pairwise EWMA correlation panel: MultiIndex (date, instrument) -> columns.
        self._corr_panel = weekly_df.ewm(
            span=corr_span_weeks, min_periods=CORR_MIN_PERIODS, ignore_na=True
        ).corr(pairwise=True)
        self._weekly_dates = weekly_df.index
        self._full_corr_cache: dict[pd.Timestamp, np.ndarray] = {}

    def _full_corr(self, weekly_date: pd.Timestamp) -> np.ndarray:
        """Full (n x n) sample correlation aligned to self.instruments, NaN->0, diag=1."""
        m = self._full_corr_cache.get(weekly_date)
        if m is None:
            df = self._corr_panel.loc[weekly_date].reindex(
                index=self.instruments, columns=self.instruments
            )
            m = np.nan_to_num(df.values, nan=0.0)
            np.fill_diagonal(m, 1.0)
            self._full_corr_cache[weekly_date] = m
        return m

    def covariance_by_index(self, date: pd.Timestamp, col_idx: np.ndarray) -> np.ndarray:
        """
        Fast path: shrunk annualised covariance for the columns `col_idx` (positions
        into self.instruments) as of `date`. Used by the backtest loop.
        """
        pos = self._weekly_dates.searchsorted(date, side="right") - 1
        if pos < 0:
            corr = np.eye(len(col_idx))
        else:
            full = self._full_corr(self._weekly_dates[pos])
            corr = full[np.ix_(col_idx, col_idx)]
        corr = shrink_correlation(corr, self.shrinkage)

        vpos = self._vol_index.searchsorted(date, side="right") - 1
        if vpos < 0:
            vol = np.zeros(len(col_idx))
        else:
            vol = np.nan_to_num(self._vol_values[vpos][col_idx], nan=0.0)

        return covariance_from_corr_and_vol(corr, vol)

    def covariance(self, date: pd.Timestamp, instruments: list[str]) -> np.ndarray:
        """
        Shrunk annualised covariance matrix for `instruments` as of `date`.
        Correlation uses the most recent weekly estimate on/before `date`; vol uses
        the most recent daily estimate on/before `date`.
        """
        col_idx = np.array([self._col_pos[i] for i in instruments])
        return self.covariance_by_index(date, col_idx)


# ══════════════════════════════════════════════════════════════════════════════════
# Phase 2-3 — Objective and greedy optimiser
# ══════════════════════════════════════════════════════════════════════════════════

def tracking_error_std(
    weights: np.ndarray,
    target_weights: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """
    Standard deviation of the tracking-error portfolio: sqrt(e' Sigma e),
    where e = weights - target_weights.   [calcs line 232]
    """
    e = weights - target_weights
    var = float(e @ covariance @ e)
    return np.sqrt(var) if var > 0 else 0.0


def cost_penalty(
    weights: np.ndarray,
    previous_weights: np.ndarray,
    cost_in_weight: np.ndarray,
) -> float:
    """
    Cost penalty added to the tracking error: 50 * sum_i |trade_i| * cost_in_weight_i,
    with trades measured in portfolio-weight terms.   [calcs lines 271-272]
    """
    trades = np.abs(weights - previous_weights)
    return COST_MULTIPLIER * float(np.sum(trades * cost_in_weight))


def _objective(
    weights: np.ndarray,
    target_weights: np.ndarray,
    previous_weights: np.ndarray,
    covariance: np.ndarray,
    cost_in_weight: np.ndarray,
) -> float:
    return (
        tracking_error_std(weights, target_weights, covariance)
        + cost_penalty(weights, previous_weights, cost_in_weight)
    )


def greedy_optimise_weights(
    covariance: np.ndarray,
    weight_per_contract: np.ndarray,
    target_weights: np.ndarray,
    previous_weights: np.ndarray | None = None,
    cost_in_weight: np.ndarray | None = None,
    locked: np.ndarray | None = None,
) -> np.ndarray:
    """
    Greedy integer optimiser (calcs.txt lines 238-256).

    Start from zero weights. Each pass, for every instrument try moving its weight
    by one contract *in the direction of its optimal sign* and keep the single best
    one-contract move; accept it if it lowers the objective; repeat until no move
    improves. Direction-locking means positions never flip against the forecast and
    no hedged positions are taken.

    target_weights:  optimal unrounded portfolio weights w_i = N_i * weight_per_contract_i.
    Returns optimised weights w*_i (whole multiples of weight_per_contract_i).

    locked:  optional boolean mask of instruments held at min=max=current position
    (calcs.txt line 330) — used for the wider optimisation universe (instruments we
    optimise over but cannot trade). A locked instrument starts at its *previous*
    weight and is never moved, so its tracking-error contribution e_i = prev_i -
    target_i is fixed and the optimiser instead loads the desired risk onto its
    correlated, tradable neighbours through the off-diagonal covariance.

    The objective is evaluated incrementally for speed. The tracking-error variance
    V = e'Sigma e (e = w - target) updates in O(1) for a one-contract move at i:
        V_new = V + 2*d*g[i] + d^2 * Sigma_ii,  with g = Sigma e,
    and the cost penalty changes only in component i. After a move is accepted, the
    gradient g is updated by d*Sigma[:, i]. This makes each pass O(n) instead of O(n^2).
    """
    cov = np.asarray(covariance, dtype=float)
    target = np.asarray(target_weights, dtype=float)
    n = len(target)
    if previous_weights is None:
        previous_weights = np.zeros(n)
    prev = np.asarray(previous_weights, dtype=float)
    if cost_in_weight is None:
        cost_in_weight = np.zeros(n)
    cost = np.asarray(cost_in_weight, dtype=float)
    if locked is None:
        locked = np.zeros(n, dtype=bool)
    else:
        locked = np.asarray(locked, dtype=bool)

    direction = np.sign(target)
    step = np.asarray(weight_per_contract, dtype=float) * direction   # signed weight step
    diag = np.diag(cov)

    w = np.zeros(n)
    w[locked] = prev[locked]          # hold non-tradable instruments at current position
    e = w - target
    g = cov @ e                       # gradient Sigma e
    V = float(e @ g)
    cur_abs = np.abs(w - prev)        # per-instrument |trade| in weight terms
    cur_cost = COST_MULTIPLIER * float(np.sum(cur_abs * cost))
    cur_obj = (np.sqrt(V) if V > 0 else 0.0) + cur_cost

    immovable = (step == 0.0) | locked
    cost_term = COST_MULTIPLIER * cost

    while True:
        # Vectorised one-contract evaluation across all instruments at once.
        V_new = V + 2.0 * step * g + step * step * diag           # (n,)
        track_new = np.sqrt(np.clip(V_new, 0.0, None))
        new_abs = np.abs(w + step - prev)
        cost_new = cur_cost + cost_term * (new_abs - cur_abs)
        obj = track_new + cost_new
        obj[immovable] = np.inf

        i = int(np.argmin(obj))
        if obj[i] >= cur_obj - 1e-15:
            break

        d = step[i]
        w[i] += d
        e[i] += d
        g = g + d * cov[:, i]
        V = float(V_new[i])
        cur_obj = float(obj[i])
        cur_cost = float(cost_new[i])
        cur_abs[i] = new_abs[i]

    return w


# ══════════════════════════════════════════════════════════════════════════════════
# Phase 4 — Tracking-error buffering
# ══════════════════════════════════════════════════════════════════════════════════

def apply_buffering(
    optimised_weights: np.ndarray,
    previous_weights: np.ndarray,
    covariance: np.ndarray,
    weight_per_contract: np.ndarray,
    target_risk: float,
    buffer_fraction: float = BUFFER_FRACTION,
) -> np.ndarray:
    """
    Tracking-error buffering (calcs.txt lines 290-323).

    Measure T = sqrt(e_p' Sigma e_p), the tracking error of the *current* portfolio
    against the optimised one (variance only, no cost term). Trade only the fraction
    a = max((T - B)/T, 0) of the way toward the optimum, where B = buffer_fraction *
    target_risk. If T <= B no trade is needed. Trades are rounded to whole contracts.

    Returns the final post-buffer weights.
    """
    e_p = optimised_weights - previous_weights
    var = float(e_p @ covariance @ e_p)
    T = np.sqrt(var) if var > 0 else 0.0

    B = buffer_fraction * target_risk
    if T <= B or T <= 0:
        return previous_weights.copy()

    adj_factor = (T - B) / T

    prev_contracts = previous_weights / weight_per_contract
    opt_contracts = optimised_weights / weight_per_contract
    trades = np.round(adj_factor * (opt_contracts - prev_contracts))
    new_contracts = prev_contracts + trades
    return new_contracts * weight_per_contract


# ══════════════════════════════════════════════════════════════════════════════════
# Top-level: optimise integer positions for a single period
# ══════════════════════════════════════════════════════════════════════════════════

def optimise_positions(
    covariance: np.ndarray,
    weight_per_contract: np.ndarray,
    optimal_unrounded_positions: np.ndarray,
    previous_positions: np.ndarray | None = None,
    cost_per_contract: np.ndarray | None = None,
    capital: float | None = None,
    target_risk: float = 0.20,
    use_costs: bool = True,
    use_buffering: bool = True,
    tradable: np.ndarray | None = None,
) -> np.ndarray:
    """
    Full single-period dynamic optimisation: greedy + (optional) cost penalty +
    (optional) buffering. Returns the integer position vector N*.

    weight_per_contract_i = multiplier_i * price_i * fx_i / capital.   [calcs line 226]
    cost_per_contract_i is the cash cost of trading one contract (base currency);
    converted to weight terms as (C_i / capital) / weight_per_contract_i.  [calcs line 268]

    tradable:  optional boolean mask (True = may be traded). Instruments with
    tradable=False are part of the optimisation universe — they shape the target and
    the covariance — but are held at min=max=current position (calcs.txt line 330):
    Carver optimises over ~150 futures while only trading ~100 (calcs.txt lines 326-328).
    """
    n = len(optimal_unrounded_positions)
    wpc = np.asarray(weight_per_contract, dtype=float)
    target_weights = np.asarray(optimal_unrounded_positions, dtype=float) * wpc

    if previous_positions is None:
        previous_positions = np.zeros(n)
    prev_positions = np.asarray(previous_positions, dtype=float)
    previous_weights = prev_positions * wpc

    if use_costs and cost_per_contract is not None and capital:
        cost_in_weight = (np.asarray(cost_per_contract, dtype=float) / capital) / wpc
    else:
        cost_in_weight = np.zeros(n)

    locked = None if tradable is None else ~np.asarray(tradable, dtype=bool)

    opt_weights = greedy_optimise_weights(
        covariance, wpc, target_weights, previous_weights, cost_in_weight, locked
    )

    if use_buffering:
        final_weights = apply_buffering(
            opt_weights, previous_weights, covariance, wpc, target_risk
        )
    else:
        final_weights = opt_weights

    result = np.round(final_weights / wpc)
    if locked is not None:
        result[locked] = prev_positions[locked]   # enforce min=max=current exactly
    return result
