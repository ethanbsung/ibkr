#!/usr/bin/env python3
"""
ibkr_fut/watchdog.py — daemon heartbeat watchdog.

Run from cron every ~15 min. The execution daemon (live_dynamic.py --mode
daemon) touches ibkr_fut/daemon_heartbeat.txt once per cycle (~10 min,
including the no-snapshot / reconnect-failure / stale-PST wait states). The
daemon runs 24/7 once started, so a heartbeat older than STALE_AFTER_SECS
(two cycles plus slack) is anomalous at any hour — except before the very
first deployment, where the missing-file case is handled the same way.

Decision logic:
  - heartbeat fresh                      → exit 0 silently (no log/Discord spam)
  - heartbeat stale/missing + halt file  → Discord alert, DO NOT restart.
    The daemon may have been deliberately killed by the kill switch /
    daily-loss circuit breaker; never resurrect a daemon past it.
  - heartbeat stale/missing, no halt     → Discord alert + restart via
    run_execution.sh (which owns PID-file cleanup, log rotation, and its own
    halt-file check — daemon-start logic is not duplicated here).

Alert suppression: Discord alerts are rate-limited via a marker file
(watchdog_last_alert, mtime-based) to one per ALERT_SUPPRESS_SECS, but the
restart is still attempted on every run.

Exit code: 0 if fresh or restart attempted, 1 if down-with-halt-file or the
restart itself failed.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ibkr_fut.risk_check import _send_discord

HERE             = Path(__file__).parent
HEARTBEAT_PATH   = HERE / "daemon_heartbeat.txt"
HALT_FILE        = HERE / "risk_halt.txt"
ALERT_MARKER     = HERE / "watchdog_last_alert"
RUN_EXECUTION_SH = HERE / "run_execution.sh"

STALE_AFTER_SECS    = 25 * 60      # two daemon cycles (600 s) plus slack
ALERT_SUPPRESS_SECS = 2 * 3600     # at most one Discord alert per 2 h


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def heartbeat_age_secs(now: float | None = None) -> float | None:
    """Seconds since the daemon last touched the heartbeat; None if missing."""
    if now is None:
        now = time.time()
    try:
        return now - HEARTBEAT_PATH.stat().st_mtime
    except FileNotFoundError:
        return None


def should_alert(now: float | None = None) -> bool:
    """True unless we already alerted within ALERT_SUPPRESS_SECS."""
    if now is None:
        now = time.time()
    try:
        return now - ALERT_MARKER.stat().st_mtime > ALERT_SUPPRESS_SECS
    except FileNotFoundError:
        return True


def mark_alerted() -> None:
    try:
        ALERT_MARKER.touch()
    except Exception as e:
        print(f"[{_now_str()}] [WATCHDOG] WARNING: could not write alert marker: {e}")


def restart_daemon() -> bool:
    """Restart the daemon via run_execution.sh; returns True on rc==0."""
    try:
        res = subprocess.run([str(RUN_EXECUTION_SH)], capture_output=True, text=True)
    except Exception as e:
        print(f"[{_now_str()}] [WATCHDOG] restart failed to launch: {e}")
        return False
    out = (res.stdout or "") + (res.stderr or "")
    if out.strip():
        print(out.rstrip())
    return res.returncode == 0


def check_and_act(now: float | None = None) -> str:
    """
    Core decision logic. Returns the action taken:
      'fresh' | 'halt_no_restart' | 'restarted' | 'restart_failed'
    """
    if now is None:
        now = time.time()

    age = heartbeat_age_secs(now)
    if age is not None and age < STALE_AFTER_SECS:
        return "fresh"   # silent — cron runs often

    desc = "MISSING" if age is None else f"STALE ({age / 60:.0f} min old)"

    # Never resurrect a daemon past the kill switch / circuit breaker.
    if HALT_FILE.exists():
        print(f"[{_now_str()}] [WATCHDOG] heartbeat {desc} and halt file present "
              f"— NOT restarting (kill switch respected)")
        if should_alert(now):
            _send_discord(
                f"[WATCHDOG] Daemon heartbeat {desc} and risk_halt.txt is PRESENT.\n"
                f"NOT restarting — daemon was likely halted deliberately "
                f"(kill switch / daily-loss circuit breaker).\n"
                f"Investigate, then remove ibkr_fut/risk_halt.txt and rerun "
                f"run_execution.sh to resume."
            )
            mark_alerted()
        return "halt_no_restart"

    print(f"[{_now_str()}] [WATCHDOG] heartbeat {desc} — restarting daemon "
          f"via run_execution.sh")
    if should_alert(now):
        _send_discord(
            f"[WATCHDOG] Daemon heartbeat {desc} — restarting via "
            f"run_execution.sh.\nCheck ibkr_fut/daemon_cron.log for why it died."
        )
        mark_alerted()

    ok = restart_daemon()
    if ok:
        print(f"[{_now_str()}] [WATCHDOG] restart issued OK")
        return "restarted"
    print(f"[{_now_str()}] [WATCHDOG] restart FAILED (see run_execution output above)")
    return "restart_failed"


def main() -> int:
    action = check_and_act()
    return 0 if action in ("fresh", "restarted") else 1


if __name__ == "__main__":
    sys.exit(main())
