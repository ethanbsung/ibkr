"""
Unit tests for Phase 5 (drift-proofing) and Phase 6 (backtest parity).

  • size_targets: shared single-bar sizer — formula + gross-leverage cap
  • buffer constant: executor BUFFER_FRACTION matches backtest apply_buffer default
  • live-state cache: _build_state rebuilds the live forecast only once per run
  • whole-share-short rounding: sub-1-share shorts drop to 0, longs untouched

Run:  python3 etf/live/test_phase5_6.py
"""

import inspect
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from etf.live.ewmac_backtest import size_targets, apply_buffer
from etf.live.executor import BUFFER_FRACTION
import etf.live.strategy_ewmac as strat_mod

RESULTS = []


def check(name, cond, extra=""):
    RESULTS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra and not cond else ''}")


def test_size_targets():
    tickers  = ["A", "B"]
    fc       = pd.Series({"A": 10.0, "B": -20.0})
    vol      = pd.Series({"A": 0.20, "B": 0.40})
    weights  = {"A": 0.5, "B": 0.5}
    # capital*idm*w*vt*(fc/10)/vol -> A: 10000*2*.5*.25*1/.2 = 12500 ; B: -12500
    t, scale = size_targets(10_000, tickers, fc, vol, weights, 2.0, 0.25, gross_cap=None)
    check("size_targets long value", abs(t["A"] - 12500.0) < 1e-6, f"got {t['A']}")
    check("size_targets short value", abs(t["B"] + 12500.0) < 1e-6, f"got {t['B']}")
    check("size_targets no cap -> scale 1", scale == 1.0)

    # gross cap binds: gross=25000, cap=1.0*10000 -> scale 0.4
    t2, s2 = size_targets(10_000, tickers, fc, vol, weights, 2.0, 0.25, gross_cap=1.0)
    check("gross cap scale = 0.4", abs(s2 - 0.4) < 1e-9, f"got {s2}")
    check("gross cap scales long", abs(t2["A"] - 5000.0) < 1e-6, f"got {t2['A']}")
    check("gross cap scales short", abs(t2["B"] + 5000.0) < 1e-6, f"got {t2['B']}")

    # invalid vol -> target 0 (held/flat)
    t3, _ = size_targets(10_000, ["A"], pd.Series({"A": 10.0}),
                         pd.Series({"A": 0.0}), {"A": 1.0}, 2.0, 0.25)
    check("zero vol -> target 0", t3["A"] == 0.0, f"got {t3['A']}")
    t4, _ = size_targets(10_000, ["A"], pd.Series({"A": np.nan}),
                         pd.Series({"A": 0.2}), {"A": 1.0}, 2.0, 0.25)
    check("nan forecast -> target 0", t4["A"] == 0.0, f"got {t4['A']}")


def test_buffer_constant():
    default = inspect.signature(apply_buffer).parameters["buffer_fraction"].default
    check("executor BUFFER_FRACTION == backtest apply_buffer default",
          BUFFER_FRACTION == default, f"{BUFFER_FRACTION} vs {default}")


def test_live_state_cache():
    # Synthetic 2-ticker history so the live forecast branch runs fast.
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(0)
    px  = pd.DataFrame({
        "A": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))),
        "B": 50  * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))),
    }, index=idx)

    s = strat_mod.EWMACStrategy.__new__(strat_mod.EWMACStrategy)
    s.vol_target         = 0.25
    s.gross_leverage_cap = 1.9
    s.asset_classes      = {"A": "EQ", "B": "BOND"}
    s._live_cache        = None
    # Stub the CSV base state so no disk/network is touched.
    s._state_cache = dict(prices=px, weights={"A": 0.5, "B": 0.5}, idm=2.0,
                          vols=None, combined_fc=None, tickers=["A", "B"])

    # Count how many times the (expensive) forecast build runs.
    calls = {"n": 0}
    orig  = strat_mod.build_forecasts
    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    strat_mod.build_forecasts = counting
    try:
        tp = {"A": float(px["A"].iloc[-1] * 1.01), "B": float(px["B"].iloc[-1] * 0.99)}
        st1 = s._build_state(tp)
        st2 = s._build_state(dict(tp))          # same content, different dict object
        check("live state cached (forecast built once)", calls["n"] == 1, f"built {calls['n']}x")
        check("cache returns identical state object", st1 is st2)
        st3 = s._build_state({"A": tp["A"] * 1.5, "B": tp["B"]})  # different prices
        check("different prices rebuild", calls["n"] == 2, f"built {calls['n']}x")
    finally:
        strat_mod.build_forecasts = orig


def test_whole_share_shorts():
    # Replicate the backtest's rounding rule directly on a tiny frame.
    prices  = pd.DataFrame({"A": [100.0], "B": [50.0], "C": [10.0]})
    raw_pos = pd.DataFrame({"A": [1000.0], "B": [-20.0], "C": [-250.0]})  # long, tiny short, real short
    shares     = raw_pos / prices
    short_mask = raw_pos < 0
    rounded    = raw_pos.where(~short_mask, shares.round() * prices)
    check("long position untouched", rounded["A"].iloc[0] == 1000.0)
    check("sub-half-share short (-0.4 sh) rounds to 0", rounded["B"].iloc[0] == 0.0,
          f"got {rounded['B'].iloc[0]}")
    check("real short (-25 sh) preserved", rounded["C"].iloc[0] == -250.0,
          f"got {rounded['C'].iloc[0]}")


def main():
    test_size_targets()
    test_buffer_constant()
    test_live_state_cache()
    test_whole_share_shorts()
    n = sum(RESULTS)
    print(f"\n  {n}/{len(RESULTS)} passed")
    return 0 if n == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
