"""
trading_calendar.py

CME Globex trading-calendar helpers, shared by the live daemon's staleness gate
and anything else that needs to know "what is the most recent session whose
settled data pst_updater should have."

Why this exists
---------------
The daemon used to flag a snapshot as failed whenever ``pst_date != date.today()``.
That is wrong around weekends and holidays:

  * A CME Globex *session* is named by its CLOSE date and closes at 18:00 ET.
    The session labelled ``2026-06-22`` runs Sun 18:00 ET → Mon 18:00 ET.
  * The live compute runs at 18:00 ET. pst_updater (run just before it) pulls
    the just-settled close, so the snapshot it builds carries the date of the
    LAST COMPLETED session — which on a Sunday-evening run is the *previous
    Friday*, not "today".
  * CME also trades shortened sessions on some US holidays (e.g. Juneteenth),
    so "is the market closed today" is the wrong question entirely.

The correct staleness test is therefore: is the snapshot's data date EQUAL to
the most recent session whose settlement data should exist by now? If it lags
behind that, pst_updater / the Gateway genuinely failed.

Calendar choice
---------------
``pandas_market_calendars`` ``CMES`` calendar (CME Globex; the ``us_futures``
alias resolves to the same holiday rules). It encodes CME holiday closures and
shortened sessions, including Juneteenth.
"""

import functools
from datetime import date, datetime, timedelta

import pandas as pd

try:
    import pandas_market_calendars as mcal
except ImportError as e:   # pragma: no cover - dependency is pinned in requirements_live.txt
    raise ImportError(
        "pandas_market_calendars is required for the trading-calendar staleness "
        "gate. Install it: pip install pandas_market_calendars"
    ) from e

# CME settlement posts a few minutes after the 18:00 ET close; don't treat a
# session as "should have data" until we're safely past that. So at exactly
# 18:00 ET on a weekday the *expected* date is still the prior session until
# settlement is realistically available. Keep this comfortably below the gap
# between the 18:00 close and the ~18:00 compute run so Friday's own close
# counts as expected on Friday evening.
_SETTLE_GRACE = timedelta(minutes=20)

_CAL_NAME = "CMES"
_ET = "America/New_York"


@functools.lru_cache(maxsize=1)
def _calendar():
    return mcal.get_calendar(_CAL_NAME)


def _now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=_ET)


def last_completed_session(now_et: pd.Timestamp | None = None) -> date:
    """Date of the most recent CME session whose settled data should exist now.

    A session "should have data" once we are ``_SETTLE_GRACE`` past its 18:00 ET
    close. This is the date pst_updater is expected to have written and therefore
    the date a fresh snapshot's PRICE data should carry.

    Examples (all at 18:00 ET, EDT):
      Sun 06-21 → 2026-06-19 (Fri)   # Sun session opens but closes Mon
      Mon 06-22 → 2026-06-22 (Mon)   # Mon session just settled (past grace)
      Sat 06-20 → 2026-06-19 (Fri)
    """
    if now_et is None:
        now_et = _now_et()
    elif now_et.tzinfo is None:
        now_et = now_et.tz_localize(_ET)

    cutoff = now_et - _SETTLE_GRACE
    cal = _calendar()
    # 12-day lookback comfortably spans any holiday + weekend gap.
    sched = cal.schedule(
        start_date=(cutoff - pd.Timedelta(days=12)).date().isoformat(),
        end_date=cutoff.date().isoformat(),
        tz=_ET,
    )
    closed = sched[sched["market_close"] <= cutoff]
    if closed.empty:
        # Degenerate (shouldn't happen with a 12-day window); fall back to the
        # most recent scheduled session regardless of close time.
        return sched.index[-1].date()
    return closed.index[-1].date()


def sessions_behind(data_date: date, now_et: pd.Timestamp | None = None) -> int:
    """How many completed sessions the given data date lags the expected one.

    0  → fresh (data_date == last completed session)
    >0 → stale by that many sessions (genuine pst_updater/Gateway failure)
    <0 → data is *ahead* of expectation (shouldn't happen; treat as fresh)
    """
    if now_et is None:
        now_et = _now_et()
    elif now_et.tzinfo is None:
        now_et = now_et.tz_localize(_ET)

    expected = last_completed_session(now_et)
    if data_date >= expected:
        return 0
    cal = _calendar()
    sched = cal.schedule(
        start_date=data_date.isoformat(),
        end_date=expected.isoformat(),
        tz=_ET,
    )
    # Count sessions strictly after data_date up to and including expected.
    later = [d.date() for d in sched.index if d.date() > data_date]
    return len(later)


def is_snapshot_fresh(data_date: date, now_et: pd.Timestamp | None = None) -> bool:
    """True when the snapshot's data date is the most recent completed session."""
    return sessions_behind(data_date, now_et) == 0


if __name__ == "__main__":
    # Quick manual check: print the expected session for a few moments.
    import pytz
    et = pytz.timezone(_ET)
    for d in ["2026-06-19", "2026-06-20", "2026-06-21", "2026-06-22", "2026-06-23"]:
        ts = pd.Timestamp(et.localize(datetime.fromisoformat(d + "T18:00:00")))
        print(f"{d} 18:00 ET → last completed session = {last_completed_session(ts)}")
