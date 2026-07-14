"""
Unit tests for the volume-based roll logic in pst_updater.py (BUG-4 fix).

These mirror pysystemtrade's check_if_forward_liquid / get_smoothed_volume rule
(pst-group/pysystemtrade) and the roll-calendar lockstep write. DATA-INDEPENDENT:
synthetic volume series + a temp roll calendar, no IB and no PST CSV warehouse.

Run:  source venv/bin/activate && python3 -m pytest ibkr_fut/test_volume_roll.py -q
"""

import os
import tempfile
from datetime import date, timedelta

import pandas as pd
import pytest

from ibkr_fut import pst_updater as pu


ASOF = date(2026, 6, 17)


def _vol(vals, end=ASOF):
    idx = [pd.Timestamp(end - timedelta(days=len(vals) - 1 - i)) for i in range(len(vals))]
    return pd.Series(vals, index=idx, dtype=float)


# ── forward_is_liquid: Carver's exact thresholds (1.0 / 0.01 / 100) ─────────────

def test_forward_liquid_high_relative_volume():
    # rel just over 1.0 → liquid regardless of absolute
    assert pu.forward_is_liquid(100, 101) is True
    assert pu.forward_is_liquid(100, 99) is False


def test_forward_liquid_relative_and_absolute_floor():
    # rel 0.5 (> 0.01) AND abs 150 (> 100) → liquid
    assert pu.forward_is_liquid(300, 150) is True
    # rel 0.5 but abs 50 (< 100) → not liquid
    assert pu.forward_is_liquid(100, 50) is False


def test_forward_liquid_dead_priced_contract_rolls():
    # priced contract gone no-trade (smoothed ~0) → ratio explodes → roll.
    # This is the BRE dead-window trigger.
    assert pu.forward_is_liquid(0, 50) is True


def test_forward_liquid_weak_rule_gated_by_calendar():
    # OBS-22: the weak rule (rel > 1%, abs > 100) must not initiate a roll far
    # from the calendar roll date — CORN rolled Dec26→Dec27 in June off 1.6%
    # relative volume.
    assert pu.forward_is_liquid(172766, 2826, near_roll=False) is False
    assert pu.forward_is_liquid(172766, 2826, near_roll=True) is True


def test_forward_liquid_strong_rule_ignores_calendar():
    # Forward more liquid than front → the market has moved on; roll even if
    # the calendar date is far (dying-front escape hatch).
    assert pu.forward_is_liquid(100, 101, near_roll=False) is True
    assert pu.forward_is_liquid(0, 50, near_roll=False) is True


# ── smoothed_volume: EWMA span=3, ignore >14d old, 0.0 when none recent ─────────

def test_smoothed_volume_recent_flat():
    assert round(pu.smoothed_volume(_vol([100] * 5), ASOF)) == 100


def test_smoothed_volume_ignores_stale():
    stale = _vol([500] * 3, end=ASOF - timedelta(days=30))
    assert pu.smoothed_volume(stale, ASOF) == 0.0


def test_smoothed_volume_empty_and_none():
    assert pu.smoothed_volume(pd.Series(dtype=float), ASOF) == 0.0
    assert pu.smoothed_volume(None, ASOF) == 0.0


def test_smoothed_volume_weights_recent_higher():
    # rising volume → EWMA last value above the simple mean
    rising = _vol([10, 20, 40, 80])
    assert pu.smoothed_volume(rising, ASOF) > 37.5  # simple mean


# ── advance_roll_calendar_to: lockstep with an early roll ───────────────────────

@pytest.fixture
def temp_pst(monkeypatch):
    tmp = tempfile.mkdtemp()
    os.makedirs(f"{tmp}/roll_calendars_csv", exist_ok=True)
    monkeypatch.setattr(pu, "PST_BASE", tmp)
    return tmp


def _write_cal(tmp, rows):
    df = pd.DataFrame(rows).set_index("DATE_TIME")
    df.to_csv(f"{tmp}/roll_calendars_csv/BRE.csv")


_ROLL_CFG = pd.Series({
    "HoldRollCycle": "FGHJKMNQUVXZ", "PricedRollCycle": "FGHJKMNQUVXZ",
    "CarryOffset": 1, "ExpiryOffset": 17, "RollOffsetDays": -5,
})


def test_advance_roll_calendar_lockstep(temp_pst):
    _write_cal(temp_pst, [
        {"DATE_TIME": pd.Timestamp("2026-05-25 20:00"), "current_contract": 20260500,
         "next_contract": 20260600, "carry_contract": 20260700},
        {"DATE_TIME": pd.Timestamp("2026-06-25 20:00"), "current_contract": 20260600,
         "next_contract": 20260700, "carry_contract": 20260800},
        {"DATE_TIME": pd.Timestamp("2026-07-25 20:00"), "current_contract": 20260700,
         "next_contract": 20260800, "carry_contract": 20260900},
    ])

    # early roll out of 202606 → 202607 on 2026-06-10 (before the 6/25 calendar roll)
    pu.advance_roll_calendar_to("BRE", "202606", "202607", date(2026, 6, 10), _ROLL_CFG)
    pu.extend_roll_calendar("BRE", _ROLL_CFG)

    out = pd.read_csv(f"{temp_pst}/roll_calendars_csv/BRE.csv",
                      parse_dates=["DATE_TIME"]).sort_values("DATE_TIME")

    # the closing row for the old contract is dated the early-roll date at 20:00
    roll_row = out[out.DATE_TIME == pd.Timestamp("2026-06-10 20:00")]
    assert len(roll_row) == 1
    assert int(roll_row.iloc[0].current_contract) // 100 == 202606
    assert int(roll_row.iloc[0].next_contract) // 100 == 202607

    # get_roll_info semantics: as of 2026-06-12 the held contract is the rolled-into one
    fut = out[pd.to_datetime(out.DATE_TIME).dt.normalize() >= pd.Timestamp("2026-06-12")]
    assert int(fut.iloc[0].current_contract) // 100 == 202607

    # the old future rows (6/25, 7/25) on/after the roll were dropped & rebuilt
    assert not (out.DATE_TIME == pd.Timestamp("2026-06-25 20:00")).any()


def test_advance_roll_calendar_no_file_is_safe(temp_pst):
    # No calendar and no multiple_prices to bootstrap → no crash, just a warning.
    pu.advance_roll_calendar_to("NOPE", "202606", "202607", date(2026, 6, 10), _ROLL_CFG)
    assert not os.path.exists(f"{temp_pst}/roll_calendars_csv/NOPE.csv")
