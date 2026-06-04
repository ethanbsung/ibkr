"""
dynamic_opt_test.py — unit tests for the dynamic optimisation primitives
(Phases 1-4). Run:  python -m pytest ibkr_fut/dynamic_opt_test.py -v
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibkr_fut.dynamic_opt import (
    shrink_correlation,
    covariance_from_corr_and_vol,
    weekly_pct_returns,
    CovarianceEstimator,
    tracking_error_std,
    cost_penalty,
    _objective,
    greedy_optimise_weights,
    apply_buffering,
    optimise_positions,
)


# ── Phase 1: covariance estimation ────────────────────────────────────────────────

def test_shrink_correlation_halves_offdiag_keeps_diag():
    corr = np.array([[1.0, 0.8], [0.8, 1.0]])
    shrunk = shrink_correlation(corr, 0.5)
    assert np.allclose(np.diag(shrunk), 1.0)
    assert np.isclose(shrunk[0, 1], 0.4)


def test_shrink_correlation_is_psd():
    # A genuine (PSD) correlation matrix stays PSD after shrinkage, with a margin.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5))
    corr = np.corrcoef(X, rowvar=False)
    shrunk = shrink_correlation(corr, 0.5)
    eigvals = np.linalg.eigvalsh(shrunk)
    assert eigvals.min() > 0.0


def test_covariance_from_corr_and_vol():
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    vol = np.array([0.2, 0.3])
    cov = covariance_from_corr_and_vol(corr, vol)
    # Diagonal is variance; off-diagonal is rho * s_i * s_j.
    assert np.isclose(cov[0, 0], 0.2 ** 2)
    assert np.isclose(cov[1, 1], 0.3 ** 2)
    assert np.isclose(cov[0, 1], 0.5 * 0.2 * 0.3)


def test_weekly_pct_returns_basic():
    idx = pd.bdate_range("2020-01-01", periods=20)
    adj = pd.Series(np.linspace(100, 110, 20), index=idx, name="X")
    con = pd.Series(100.0, index=idx, name="X")
    wk = weekly_pct_returns(adj, con)
    # Weekly change of back-adjusted price over a flat $100 contract price.
    assert wk.dropna().abs().sum() > 0
    assert wk.index.freqstr.startswith("W")


def test_covariance_estimator_diagonal_matches_vol():
    idx = pd.bdate_range("2015-01-01", periods=600)
    rng = np.random.default_rng(1)
    adj = {}
    con = {}
    for name, sd in [("A", 0.01), ("B", 0.02)]:
        rets = rng.standard_normal(600) * sd
        adj[name] = pd.Series(100 + np.cumsum(rets) * 100, index=idx, name=name)
        con[name] = pd.Series(100.0, index=idx, name=name)
    est = CovarianceEstimator(adj, con)
    cov = est.covariance(idx[-1], ["A", "B"])
    assert cov.shape == (2, 2)
    # Diagonal should be positive and B noticeably more volatile than A.
    assert cov[0, 0] > 0 and cov[1, 1] > 0
    assert cov[1, 1] > cov[0, 0]
    # PSD.
    assert np.linalg.eigvalsh(cov).min() >= -1e-12


# ── Phase 2: greedy optimiser (no costs) ──────────────────────────────────────────

def test_greedy_diagonal_recovers_nearest_integer():
    # Uncorrelated assets: tracking error is separable, so the greedy picks each
    # instrument's nearest integer position in the optimal direction.
    wpc = np.array([0.04, 0.05])
    target_positions = np.array([2.4, -1.6])           # ideal unrounded contracts
    target_weights = target_positions * wpc
    cov = np.diag([0.20 ** 2, 0.25 ** 2])
    w = greedy_optimise_weights(cov, wpc, target_weights)
    positions = np.round(w / wpc)
    assert np.allclose(positions, [2.0, -2.0])


def test_greedy_zero_target_zero_positions():
    wpc = np.array([0.04, 0.05, 0.03])
    target_weights = np.zeros(3)
    cov = np.eye(3) * 0.04
    w = greedy_optimise_weights(cov, wpc, target_weights)
    assert np.allclose(w, 0.0)


def test_greedy_single_asset_rounds_in_direction():
    wpc = np.array([0.04])
    # 2.4 contracts -> rounds down to 2 (3 would overshoot the target).
    w = greedy_optimise_weights(np.array([[0.04]]), wpc, np.array([2.4 * 0.04]))
    assert np.isclose(np.round(w / wpc)[0], 2.0)
    # -2.6 contracts -> rounds to -3.
    w = greedy_optimise_weights(np.array([[0.04]]), wpc, np.array([-2.6 * 0.04]))
    assert np.isclose(np.round(w / wpc)[0], -3.0)


def test_greedy_result_is_locally_optimal():
    # The incremental greedy must return a weight vector where no single one-contract
    # move (in the optimal direction) reduces the FULL objective recomputed naively.
    rng = np.random.default_rng(7)
    for _ in range(15):
        n = rng.integers(3, 8)
        wpc = rng.uniform(0.02, 0.08, n)
        target_positions = rng.uniform(-4, 4, n)
        target_weights = target_positions * wpc
        # Random PSD covariance.
        A = rng.standard_normal((n, n))
        corr = np.corrcoef(A @ A.T)
        vol = rng.uniform(0.1, 0.3, n)
        cov = covariance_from_corr_and_vol(corr, vol)
        prev = np.round(rng.uniform(-2, 2, n)) * wpc
        cost_in_weight = rng.uniform(0, 0.01, n)

        w = greedy_optimise_weights(cov, wpc, target_weights, prev, cost_in_weight)
        base = _objective(w, target_weights, prev, cov, cost_in_weight)
        direction = np.sign(target_weights)
        for i in range(n):
            if direction[i] == 0:
                continue
            cand = w.copy()
            cand[i] += wpc[i] * direction[i]
            assert _objective(cand, target_weights, prev, cov, cost_in_weight) >= base - 1e-12


# ── Phase 3: cost penalty ─────────────────────────────────────────────────────────

def test_cost_penalty_value():
    weights = np.array([0.10, -0.05])
    prev = np.array([0.0, 0.0])
    cost_in_weight = np.array([0.01, 0.02])
    # 50 * (|0.10|*0.01 + |0.05|*0.02) = 50 * (0.001 + 0.001) = 0.1
    assert np.isclose(cost_penalty(weights, prev, cost_in_weight), 0.1)


def test_cost_zero_reduces_to_phase2():
    wpc = np.array([0.04, 0.05])
    target_weights = np.array([2.4, -1.6]) * wpc
    cov = np.diag([0.20 ** 2, 0.25 ** 2])
    base = greedy_optimise_weights(cov, wpc, target_weights)
    with_zero_cost = greedy_optimise_weights(
        cov, wpc, target_weights,
        previous_weights=np.zeros(2), cost_in_weight=np.zeros(2),
    )
    assert np.allclose(base, with_zero_cost)


def test_higher_cost_reduces_trading():
    # As the cost penalty rises, total traded weight (from a flat book) is
    # monotonically non-increasing.
    wpc = np.array([0.04, 0.05, 0.03])
    target_positions = np.array([3.3, -2.6, 1.4])
    cov = np.diag([0.20 ** 2, 0.25 ** 2, 0.15 ** 2])
    prev = np.zeros(3)
    traded = []
    for scale in [0.0, 0.001, 0.01, 0.05, 0.2]:
        cost_in_weight = np.array([scale, scale, scale])
        w = greedy_optimise_weights(cov, wpc, target_positions * wpc, prev, cost_in_weight)
        traded.append(np.abs(w - prev).sum())
    diffs = np.diff(traded)
    assert (diffs <= 1e-12).all(), traded


# ── Phase 4: tracking-error buffering ─────────────────────────────────────────────

def test_buffering_book_example():
    # Reproduce the worked example: optimum [2,6,0] contracts from a flat book,
    # weight-per-contract 0.04 each, tracking error T=2.35%, buffer B=1% (=0.05*0.20).
    # a = (2.35 - 1)/2.35 = 0.574 -> trades round(0.574*[2,6,0]) = [1,3,0].
    wpc = np.array([0.04, 0.04, 0.04])
    opt_positions = np.array([2.0, 6.0, 0.0])
    opt_weights = opt_positions * wpc
    prev_weights = np.zeros(3)

    # Choose a diagonal covariance so that T = sqrt(e' Sigma e) = 0.0235 exactly.
    e = opt_weights - prev_weights
    target_T = 0.01 / (1.0 - 0.574)            # ~0.023474  -> a == 0.574
    v = target_T ** 2 / float(e @ e)           # equal diagonal variance
    cov = np.diag([v, v, v])

    final = apply_buffering(opt_weights, prev_weights, cov, wpc, target_risk=0.20)
    positions = np.round(final / wpc)
    assert np.allclose(positions, [1.0, 3.0, 0.0]), positions


def test_buffering_no_trade_within_buffer():
    # Tiny tracking error (below the 1% buffer) -> hold previous positions.
    wpc = np.array([0.04, 0.04])
    prev_weights = np.array([1.0, 1.0]) * wpc
    opt_weights = np.array([1.0, 1.0]) * wpc + np.array([0.0001, 0.0001])
    cov = np.diag([0.04, 0.04])
    final = apply_buffering(opt_weights, prev_weights, cov, wpc, target_risk=0.20)
    assert np.allclose(final, prev_weights)


# ── Top-level integration ─────────────────────────────────────────────────────────

def test_optimise_positions_end_to_end():
    wpc = np.array([0.04, 0.05])
    cov = np.diag([0.20 ** 2, 0.25 ** 2])
    N = optimise_positions(
        covariance=cov,
        weight_per_contract=wpc,
        optimal_unrounded_positions=np.array([2.4, -1.6]),
        previous_positions=np.zeros(2),
        cost_per_contract=np.array([0.0, 0.0]),
        capital=1_000_000,
        target_risk=0.20,
        use_costs=True,
        use_buffering=False,
    )
    assert np.allclose(N, [2.0, -2.0])
    assert N.dtype == float and np.allclose(N, np.round(N))


# ── min=max=current / wider optimisation universe (calcs.txt 326-330) ──────────────

def _corr_cov(vol, rho):
    v = np.asarray(vol, float)
    r = np.array(rho, float)
    return np.diag(v) @ r @ np.diag(v)


def test_locked_instrument_held_at_current():
    # A non-tradable instrument must keep its current position exactly, never trade.
    cov = _corr_cov([0.20, 0.20], [[1.0, 0.0], [0.0, 1.0]])
    wpc = np.array([0.01, 0.01])
    N = optimise_positions(
        covariance=cov, weight_per_contract=wpc,
        optimal_unrounded_positions=np.array([4.0, 4.0]),
        previous_positions=np.array([2.0, 0.0]),
        use_costs=False, use_buffering=False,
        tradable=np.array([False, True]),
    )
    assert N[0] == 2.0          # locked at its current 2 contracts
    assert N[1] == 4.0          # tradable one optimises freely


def test_locked_risk_transfers_to_correlated_tradable():
    # Carver's point (calcs 326-328): optimise over an instrument you can't trade and
    # its risk loads onto a correlated tradable neighbour.
    cov = _corr_cov([0.20, 0.20], [[1.0, 0.95], [0.95, 1.0]])
    wpc = np.array([0.01, 0.01])
    target = np.array([3.0, 3.0])
    both = optimise_positions(cov, wpc, target, previous_positions=np.zeros(2),
                              use_costs=False, use_buffering=False)
    locked = optimise_positions(cov, wpc, target, previous_positions=np.zeros(2),
                                use_costs=False, use_buffering=False,
                                tradable=np.array([False, True]))
    assert locked[0] == 0.0
    assert locked[1] > both[1]   # neighbour takes extra contracts to cover the locked one
