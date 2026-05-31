#!/usr/bin/env python3
"""
Performance metrics from a strategy's return/NAV ledger, plus portfolio
combination by capital allocation.

Everything derives from the per-bar `ret` the engine logs, so a strategy's
metrics are capital-invariant and adding a strategy needs no new code.
"""

import numpy as np
import pandas as pd

DEFAULT_ANN = 365   # 24/7 daily calendar


def return_metrics(ret, ann=DEFAULT_ANN):
    """Risk/return metrics from a daily return Series."""
    r = ret.dropna()
    if len(r) < 2 or r.std() == 0:
        return {k: np.nan for k in
                ('cagr', 'ann_vol', 'sharpe', 'calmar', 'max_dd', 'cur_dd')}
    nav = (1 + r).cumprod()
    yrs = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    dd = (nav - nav.cummax()) / nav.cummax()
    maxdd = dd.min()
    return {
        'cagr':    cagr,
        'ann_vol': r.std() * np.sqrt(ann),
        'sharpe':  r.mean() / r.std() * np.sqrt(ann),
        'calmar':  cagr / abs(maxdd) if maxdd < 0 else np.nan,
        'max_dd':  maxdd,
        'cur_dd':  dd.iloc[-1],
    }


def cost_metrics(daily):
    return {
        'cum_cost':       daily.get('cost', pd.Series(dtype=float)).sum(),
        'cum_turnover':   daily.get('turnover', pd.Series(dtype=float)).sum(),
    }


def summarize(daily, ann=DEFAULT_ANN):
    """Full metric dict for one strategy ledger (return/NAV space)."""
    if daily.empty:
        return {}
    m = return_metrics(daily['ret'], ann)
    m.update(cost_metrics(daily))
    m['nav'] = daily['nav'].iloc[-1]
    m['total_return'] = daily['nav'].iloc[-1] - 1
    m['days'] = len(daily)
    return m


def combine_portfolio(returns_by_strategy, allocations):
    """
    Capital-weighted blend of strategy daily returns into one portfolio return
    stream. `allocations` = {name: weight}; weights summing < 1 leave the rest in
    cash (0 return). Reallocation = pass different weights — strategy data is
    untouched. Returns (portfolio_return_series, aligned_returns_df).
    """
    if not returns_by_strategy:
        return pd.Series(dtype=float), pd.DataFrame()
    aligned = pd.DataFrame(returns_by_strategy).sort_index().fillna(0.0)
    w = pd.Series(allocations).reindex(aligned.columns).fillna(0.0)
    port = aligned.mul(w, axis=1).sum(axis=1)
    return port, aligned
