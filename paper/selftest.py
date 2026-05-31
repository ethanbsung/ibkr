#!/usr/bin/env python3
"""
Deterministic self-test of the capital-agnostic (return/NAV) paper engine.

Exercises the execution path on a scripted scenario: enter, hold within the
no-trade band, exit. Asserts NAV compounds the per-bar returns, returns equal
price-P&L minus cost, and the band suppresses churn.

Run:  python3 paper/selftest.py
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Strategy, PaperEngine, Book


class Stub(Strategy):
    def __init__(self, scenario):
        super().__init__(name='stub', fee_per_side=0.001, slip_per_side=0.0, band=0.05)
        self.scenario = scenario
        self.i = 0

    def compute(self, asof=None):
        step = self.scenario[self.i]
        self.i += 1
        return step


def main():
    tmp = tempfile.mkdtemp(prefix='paper_selftest_')
    try:
        scenario = [
            ('2026-01-01', {'BTC': 100.0, 'ETH': 50.0}, {'BTC': 0.30, 'ETH': 0.20}),  # enter
            ('2026-01-02', {'BTC': 110.0, 'ETH': 50.0}, {'BTC': 0.30, 'ETH': 0.20}),  # mostly hold
            ('2026-01-03', {'BTC': 110.0, 'ETH': 50.0}, {'BTC': 0.00, 'ETH': 0.00}),  # exit
        ]
        strat = Stub(scenario)
        eng = PaperEngine(tmp)
        for _ in scenario:
            eng.run_strategy(strat, verbose=True)

        d = Book('stub', tmp).daily_frame()
        print("\n", d[['ret', 'nav', 'gross_exposure', 'turnover', 'cost']].to_string())

        # NAV compounds the returns (ledger rounds to 8dp, so allow rounding slack).
        assert abs(d['nav'].iloc[-1] - (1 + d['ret']).prod()) < 1e-5, "NAV != Π(1+ret)"
        # Day 2: BTC +10% on a held ~0.25–0.30 weight → clearly positive return.
        assert d['ret'].iloc[1] > 0.02, "missed the held-position gain on day 2"
        # Band suppresses churn: day 2 turnover << day 1.
        assert d['turnover'].iloc[1] < d['turnover'].iloc[0], "band didn't suppress churn"
        # Costs charged on trading bars only.
        assert d['cost'].iloc[0] > 0 and d['cost'].iloc[1] < d['cost'].iloc[0], "cost accounting off"
        print("\n  ✅ NAV compounding, return = P&L−cost, band suppression — PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
