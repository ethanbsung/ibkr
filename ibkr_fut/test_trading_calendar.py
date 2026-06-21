"""Tests for ibkr_fut.trading_calendar — the daemon's calendar-aware staleness gate.

Pins the exact 2026-06 incident behaviour that motivated the helper: a snapshot
built off Friday's close must read as FRESH when evaluated Sunday evening (the
previous gate compared to date.today() and false-alarmed), while a genuinely
lagging snapshot reads as STALE.
"""

from datetime import date, datetime

import pandas as pd
import pytest
import pytz

from ibkr_fut.trading_calendar import (
    last_completed_session,
    sessions_behind,
    is_snapshot_fresh,
)

ET = pytz.timezone("America/New_York")


def _et(y, m, d, hh, mm=0):
    return pd.Timestamp(ET.localize(datetime(y, m, d, hh, mm)))


# ── last_completed_session ───────────────────────────────────────────────────

@pytest.mark.parametrize("now, expected", [
    # Sunday 6pm: the new week's Globex session has opened but closes Monday,
    # so the freshest *settled* session is the previous Friday. This is the
    # case the user explicitly asked to confirm.
    (_et(2026, 6, 21, 18, 0), date(2026, 6, 19)),
    # Saturday: still Friday.
    (_et(2026, 6, 20, 18, 0), date(2026, 6, 19)),
    # Friday evening (past the settlement grace): Friday itself.
    (_et(2026, 6, 19, 18, 30), date(2026, 6, 19)),
    # Monday evening (realistic ~18:30 compute time): Monday.
    (_et(2026, 6, 22, 18, 30), date(2026, 6, 22)),
    # Tuesday evening: Tuesday.
    (_et(2026, 6, 23, 18, 30), date(2026, 6, 23)),
])
def test_last_completed_session(now, expected):
    assert last_completed_session(now) == expected


def test_settlement_grace_holds_back_at_exact_close():
    """At exactly 18:00 ET the just-closed session isn't yet 'settled' (grace),
    so expected stays at the prior session until pst_updater would realistically
    have the data."""
    assert last_completed_session(_et(2026, 6, 22, 18, 0)) == date(2026, 6, 19)
    # ...and a half-hour later Monday has settled.
    assert last_completed_session(_et(2026, 6, 22, 18, 30)) == date(2026, 6, 22)


# ── sessions_behind / is_snapshot_fresh ──────────────────────────────────────

def test_friday_snapshot_is_fresh_on_sunday():
    """The core fix: Fri-close snapshot consumed Sunday evening must be FRESH."""
    sun = _et(2026, 6, 21, 18, 0)
    assert sessions_behind(date(2026, 6, 19), sun) == 0
    assert is_snapshot_fresh(date(2026, 6, 19), sun)


def test_genuinely_stale_snapshot_flagged():
    """Thursday-dated snapshot on Sunday = pst_updater missed Friday → STALE(1)."""
    sun = _et(2026, 6, 21, 18, 0)
    assert sessions_behind(date(2026, 6, 18), sun) == 1
    assert not is_snapshot_fresh(date(2026, 6, 18), sun)


def test_real_incident_snapshot_is_stale():
    """The actual 06-17 snapshot evaluated Saturday 06-20 lagged 2 sessions
    (06-18, 06-19)."""
    sat = _et(2026, 6, 20, 18, 0)
    assert sessions_behind(date(2026, 6, 17), sat) == 2


def test_data_ahead_of_expectation_is_fresh():
    """Snapshot newer than the last completed session (shouldn't happen, but
    must never be treated as stale)."""
    fri = _et(2026, 6, 19, 18, 30)
    assert sessions_behind(date(2026, 6, 22), fri) == 0


def test_juneteenth_is_a_session_not_skipped():
    """CME trades a (shortened) session on Juneteenth 2026-06-19, so the gate
    counts it — a Wed-06-17 snapshot on Mon-06-22 lags Thu, Fri(holiday), Mon."""
    mon = _et(2026, 6, 22, 18, 30)
    # Sessions after 06-17 up to 06-22: 06-18, 06-19, 06-22 → 3.
    assert sessions_behind(date(2026, 6, 17), mon) == 3
