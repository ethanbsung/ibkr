#!/usr/bin/env python3
"""
scripts/daily_report.py

Daily monitoring report for the trading system. Sends a Telegram message covering:
  - Cron health (did each pipeline run in the expected window?)
  - PST data health (staleness, carry, roll calendars)
  - Futures positions (targets vs live IBKR, with CSV fallback)
  - P&L summary across all strategies (last 5 trading days)
  - Daemon log summary (cycles, orders placed, errors)

Schedule: 7 AM ET (11:00 UTC) weekdays.
Delivery: Discord webhook — set DISCORD_WEBHOOK_URL in .env.
"""

import importlib.util
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# ── Paths & config ────────────────────────────────────────────────────────────

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

# Load .env (KEY=VALUE lines) into os.environ without overwriting existing vars
_env_path = REPO / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            if not os.environ.get(_k.strip()):
                os.environ[_k.strip()] = _v.strip().strip('"').strip("'")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

UTC = timezone.utc

# name → cron descriptor. Health is judged against the job's *expected* last run
# (schedule-aware) rather than a flat "hours ago", so weekend/holiday gaps on
# weekday-only jobs no longer false-flag as failures (a Fri-only-last-run job is
# healthy all weekend). Fields:
#   log       — log path relative to REPO
#   days      — set of weekdays the job runs (0=Mon … 6=Sun); None = every day
#   hour_utc  — the UTC hour it's scheduled at (EDT offsets, matching crontab_vps.txt)
#   grace_h   — hours after the scheduled run before a missing update counts as ✗
#   trading   — if True, "days" are filtered to CME trading sessions (skips CME
#               holidays via trading_calendar), so a holiday closure isn't flagged
#
# The "daemon" is a long-running process (continuous mtime), judged by its own
# 24h cycle-count check below, so it carries days=None and is special-cased.
CRONS = {
    #                log path                        days              hour  grace trading
    "pst_updater":  ("pst_updater.log",              {0,1,2,3,4},      22,   6,    True),
    "compute":      ("ibkr_fut/dynamic_cron.log",    {0,1,2,3,4},      22,   6,    True),
    "daemon":       ("ibkr_fut/daemon_cron.log",     None,             None, 24,   False),
    "etf_daily":    ("paper/etf_daily.log",          {0,1,2,3,4},      19,   6,    True),
    "crypto_paper": ("paper/paper_cron.log",         None,             10,   16,   False),
}

LEDGERS = {
    "ibkr_dynamic": REPO / "paper" / "ledgers" / "ibkr_dynamic",
    "etf_ewmac":    REPO / "paper" / "ledgers" / "etf_ewmac",
    "crypto_trend": REPO / "paper" / "ledgers" / "crypto_trend",
    "crypto_carry": REPO / "paper" / "ledgers" / "crypto_carry",
}

IB_HOST, IB_PORT, IB_CLIENT = "127.0.0.1", 4002, 20

# IB Gateway session-log dir (holds ibgateway.<YYYYMMDD>.<HHMMSS>.ibgzenc, one file
# per Gateway start — the cleanest "when did it restart" signal, UTC on this host).
# The settings-dir name is opaque/host-specific, so glob for the one containing ibg.xml.
GATEWAY_JTS_DIR = Path.home() / "Jts"
# A Gateway restart is "in-window" (risk to live execution) if it lands while any
# traded session is open. The daemon trades ~22:00 UTC (US reopen) through ~07:00
# UTC (Eurex). A restart outside that is on the maintenance window and benign.
GATEWAY_WINDOW_START_H, GATEWAY_WINDOW_END_H = 22, 7   # UTC; wraps midnight


# ── Timestamp parsing ─────────────────────────────────────────────────────────

_TS_PATTERNS = [
    (re.compile(r"\[?(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z\]?"),  "%Y-%m-%dT%H:%M:%S"),
    (re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}"),   "%Y-%m-%d %H:%M:%S"),
    (re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC"),      "%Y-%m-%d %H:%M:%S"),
]


def _parse_timestamps(log_path: Path) -> list[datetime]:
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(errors="replace").splitlines()[-500:]
    except Exception:
        return []
    stamps = []
    for line in lines:
        for pat, fmt in _TS_PATTERNS:
            m = pat.search(line)
            if m:
                try:
                    stamps.append(datetime.strptime(m.group(1), fmt).replace(tzinfo=UTC))
                except ValueError:
                    pass
                break
    return stamps


# ── Section: cron health ──────────────────────────────────────────────────────

def _log_mtime_utc(log_path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(log_path.stat().st_mtime, tz=UTC)
    except Exception:
        return None


def _is_cme_session(d, cal) -> bool:
    """True if date d is a CME trading session (not a weekend/holiday)."""
    if cal is None:
        # No calendar available → fall back to weekday check (Mon–Fri).
        return d.weekday() < 5
    try:
        sched = cal.schedule(start_date=d.isoformat(), end_date=d.isoformat())
        return not sched.empty
    except Exception:
        return d.weekday() < 5


def _expected_last_run(days, hour_utc, trading, now, cal) -> datetime | None:
    """Most recent datetime this job was scheduled to run, at/before `now`.

    Walks back up to 10 days from today, returning the scheduled run
    (`hour_utc:00` UTC) on the most recent eligible day:
      - day-of-week must be in `days` (None = every day), and
      - if `trading`, the day must be a CME session (skips holidays).
    Returns None if `hour_utc` is None (caller handles differently).
    """
    if hour_utc is None:
        return None
    for back in range(0, 11):
        d = (now - timedelta(days=back)).date()
        run_at = datetime(d.year, d.month, d.day, hour_utc, 0, tzinfo=UTC)
        if run_at > now:
            continue  # today's run hasn't happened yet; keep walking back
        if days is not None and d.weekday() not in days:
            continue
        if trading and not _is_cme_session(d, cal):
            continue
        return run_at
    return None


def section_cron_health() -> tuple[list[str], list[str]]:
    lines = ["CRON HEALTH"]
    warnings = []
    now = datetime.now(UTC)

    # CME calendar for holiday-aware "trading day" filtering; degrade gracefully
    # to a plain weekday check if the dependency isn't importable.
    try:
        from ibkr_fut.trading_calendar import _calendar
        cal = _calendar()
    except Exception:
        cal = None

    for name, (rel, days, hour_utc, grace_h, trading) in CRONS.items():
        log_path = REPO / rel
        if not log_path.exists():
            lines.append(f"  {name:<15} ✗  log not found ({rel})")
            warnings.append(f"{name}: log not found")
            continue

        mtime = _log_mtime_utc(log_path)
        if mtime is None:
            lines.append(f"  {name:<15} ✗  (cannot stat log)")
            warnings.append(f"{name}: cannot stat log")
            continue

        age_h = (now - mtime).total_seconds() / 3600

        if name == "daemon":
            # Long-running process: judge by its own 24h cycle-count, not a
            # scheduled run time. Show cycle count from parsed timestamps.
            try:
                content = log_path.read_text(errors="replace").splitlines()[-500:]
                cutoff  = now - timedelta(hours=24)
                n_cycles = sum(
                    1 for ln in content
                    if "Cycle done" in ln and (
                        lambda m: m and datetime.strptime(
                            m.group(1), "%Y-%m-%dT%H:%M:%S"
                        ).replace(tzinfo=UTC) >= cutoff
                    )(_TS_PATTERNS[0][0].search(ln))
                )
            except Exception:
                n_cycles = "?"

            if age_h > grace_h:
                lines.append(f"  {name:<15} ✗  last modified {mtime.strftime('%Y-%m-%d %H:%M')} UTC  ({age_h:.0f}h ago)")
                warnings.append(f"{name}: no activity in {age_h:.0f}h")
            else:
                lines.append(f"  {name:<15} ✓  {mtime.strftime('%Y-%m-%d %H:%M')} UTC  ({n_cycles} cycles in 24h)")
            continue

        # Schedule-aware freshness: healthy if the log was touched at/after the
        # job's most recent *expected* run (minus grace). A weekday-only job that
        # last ran Friday is healthy all weekend; a genuinely missed run is ✗.
        expected = _expected_last_run(days, hour_utc, trading, now, cal)
        if expected is None:
            # No schedule resolvable (e.g. no eligible day in 10d) → fall back to
            # a generous flat-age check so we never crash the section.
            if age_h > grace_h + 24:
                lines.append(f"  {name:<15} ✗  {mtime.strftime('%Y-%m-%d %H:%M')} UTC  ({age_h:.0f}h ago)")
                warnings.append(f"{name}: last run {age_h:.0f}h ago")
            else:
                lines.append(f"  {name:<15} ✓  {mtime.strftime('%Y-%m-%d %H:%M')} UTC")
            continue

        deadline = expected + timedelta(hours=grace_h)
        if mtime < expected and now > deadline:
            # The most recent scheduled run should have updated the log by now,
            # but the log predates it → a real miss.
            lines.append(
                f"  {name:<15} ✗  {mtime.strftime('%Y-%m-%d %H:%M')} UTC  "
                f"(missed {expected.strftime('%a %m-%d %H:%M')} UTC run)"
            )
            warnings.append(
                f"{name}: missed expected run at {expected.strftime('%Y-%m-%d %H:%M')} UTC"
            )
        else:
            lines.append(f"  {name:<15} ✓  {mtime.strftime('%Y-%m-%d %H:%M')} UTC")

    return lines, warnings


# ── Section: PST data health ──────────────────────────────────────────────────

def _load_dhc():
    spec = importlib.util.spec_from_file_location(
        "data_health_check", HERE / "data_health_check.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def section_data_health() -> tuple[list[str], list[str]]:
    lines = ["DATA HEALTH"]
    warnings = []
    try:
        dhc   = _load_dhc()
        today = pd.Timestamp.today().normalize()
        pw    = dhc.check_prices(today, verbose=False)
        cw    = dhc.check_carry(verbose=False)
        rw    = dhc.check_roll_calendars(today, verbose=False)
        fw    = dhc.check_fx(today, verbose=False)

        n_inst  = len(dhc.UNIVERSE)
        n_stale = sum(1 for w in pw if "STALE" in w)
        n_miss  = sum(1 for w in pw if "MISSING" in w)
        lines.append(f"  Prices: {n_inst - n_stale - n_miss}/{n_inst} current  "
                     f"({n_stale} stale  {n_miss} missing)")

        all_w = pw + cw + rw + fw
        for w in all_w[:10]:
            lines.append(f"  ⚠  {w.strip()}")
            warnings.append(w.strip())
        if len(all_w) > 10:
            lines.append(f"  ... and {len(all_w) - 10} more")
        if not all_w:
            lines.append("  All checks clean")
    except Exception as e:
        lines.append(f"  ✗  check failed: {e}")
        warnings.append(f"data health check error: {e}")

    return lines, warnings


# ── Shared live IBKR fetch ────────────────────────────────────────────────────

# Cache the single live IB read so positions + P&L sections share one connection
# and one source of truth. _SENTINEL distinguishes "not yet attempted" from
# "attempted and failed" (which legitimately returns None).
_IB_LIVE_CACHE = "__unset__"


def _fetch_ib_live():
    """
    One live IBKR read for the whole report: ({equity}, held_by_instr, unknown).

    Returns (equity_or_None, held|{}, unknown|[]). On any failure (gateway down,
    deps missing) returns (None, {}, []) so callers fall back gracefully. Cached
    module-wide — connects once (clientId IB_CLIENT), reuses the canonical
    symbol→instrument mapping from live_dynamic so held is in the PST-instrument
    namespace (matching targets), not raw IB symbols.
    """
    global _IB_LIVE_CACHE
    if _IB_LIVE_CACHE != "__unset__":
        return _IB_LIVE_CACHE

    result = (None, {}, [])
    try:
        from ib_insync import IB
        from ibkr_fut.live_dynamic import (get_positions_by_instr, load_ib_config,
                                           get_equity)
        ib = IB()
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT, timeout=8, readonly=True)
        try:
            equity = get_equity(ib)
            ibcfg = load_ib_config()
            by_instr, unknown = get_positions_by_instr(ib, ibcfg)
            held = {instr: sum(months.values()) for instr, months in by_instr.items()}
            result = (equity, held, unknown)
        finally:
            ib.disconnect()
    except Exception:
        result = (None, {}, [])

    _IB_LIVE_CACHE = result
    return result


# ── Section: futures positions ────────────────────────────────────────────────

def section_futures_positions() -> tuple[list[str], list[str]]:
    lines = ["FUTURES POSITIONS"]
    warnings = []

    snap_path = REPO / "ibkr_fut" / "targets_snapshot.json"
    if not snap_path.exists():
        lines.append("  ✗  targets_snapshot.json not found")
        warnings.append("targets_snapshot.json missing")
        return lines, warnings

    with open(snap_path) as fh:
        snap = json.load(fh)
    targets   = snap.get("targets", {})
    snap_date = snap.get("date", "unknown")
    status    = snap.get("diag", {}).get("_meta", {}).get("status", {})

    # Live IBKR first (gateway up via IBC) — held is in the PST-instrument
    # namespace (matching targets) via the shared canonical mapping. The CSV
    # fallback stores IB symbols, so translate them to instruments too.
    _, held, unknown = _fetch_ib_live()
    source = "live" if held or unknown else "cached"
    for sym, exch, qty in unknown:
        warnings.append(f"unmapped IB position {sym} {exch} {qty:+d}")

    if source == "cached":
        held = {}
        pos_csv = LEDGERS["ibkr_dynamic"] / "positions.csv"
        if pos_csv.exists():
            df = pd.read_csv(pos_csv)
            if not df.empty:
                # positions.csv keys by IB symbol — map back to instrument so it
                # compares against targets in the same namespace.
                sym_to_instr = {}
                try:
                    from ibkr_fut.live_dynamic import ib_symbol_to_instr, load_ib_config
                    sym_to_instr = ib_symbol_to_instr(load_ib_config())
                except Exception:
                    warnings.append("position source degraded — names may not match targets")
                latest = df[df["date"] == df["date"].max()]
                for _, row in latest.iterrows():
                    if int(row["qty"]) != 0:
                        key = sym_to_instr.get(row["symbol"]) or row["symbol"]
                        held[key] = held.get(key, 0) + int(row["qty"])

    all_syms = sorted(set(list(targets) + list(held)))

    lines.append(f"  Snapshot: {snap_date}  [source: {source}]")
    if not all_syms:
        lines.append("  (flat — no targets, no positions)")
        return lines, warnings

    # One instrument per line: target, held, and a status-aware flag.
    #   active      — freely optimised; t != h is a genuine mismatch (order pending).
    #   reduce_only — held but not tradable; may only unwind toward 0. If it's still
    #                 moving toward 0 that's expected (UNWINDING), not a mismatch.
    #   frozen      — held, not tradable, no signal today; pinned at current.
    lines.append(f"  {'Instr':<14} {'tgt':>5} {'held':>5}  status")
    for s in all_syms:
        t  = targets.get(s, 0)
        h  = held.get(s, 0)
        st = status.get(s, "active" if s in targets else "untracked")
        match = (t == h)

        if match:
            note = "ok" if st == "active" else st.replace("_", "-")
        elif st == "reduce_only":
            # Unwinding toward zero is the intended behaviour, not an error — but
            # surface it so it's visible while the daemon works the order down.
            note = "← UNWINDING (reduce-only)"
            warnings.append(f"{s} unwinding: target={t:+d} held={h:+d} (reduce-only)")
        elif st == "frozen":
            note = "← stranded (frozen, no signal)"
            warnings.append(f"{s} stranded: held={h:+d} but not tradable (frozen)")
        else:
            note = "← MISMATCH"
            warnings.append(f"position mismatch {s}: target={t:+d} held={h:+d}")

        lines.append(f"  {s:<14} {t:>+5d} {h:>+5d}  {note}")

    return lines, warnings


# ── Section: P&L summary ──────────────────────────────────────────────────────

def _read_daily(ledger_dir: Path) -> pd.DataFrame:
    fp = ledger_dir / "daily.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date", keep="last")


def _detect_daemon_mode() -> str:
    """
    Ground-truth LIVE/DRY-RUN/UNKNOWN from what the daemon actually logged in the
    last 24 h. The most recent "Cycle done" line wins: a plain cycle ("N placed …
    deferred") is LIVE, a "(DRY-RUN)" cycle is DRY-RUN. UNKNOWN if no cycle in
    range (e.g. daemon down) — callers should not label the account in that case.
    """
    log_path = REPO / "ibkr_fut" / "daemon_cron.log"
    if not log_path.exists():
        return "UNKNOWN"
    try:
        content = log_path.read_text(errors="replace").splitlines()
    except Exception:
        return "UNKNOWN"

    cutoff = datetime.now(UTC) - timedelta(hours=24)
    ts_re  = re.compile(r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z\]")
    dry_re = re.compile(r"Cycle done \(DRY-RUN\):")
    live_re = re.compile(r"Cycle done(?! \(DRY-RUN\)).*placed")

    mode = "UNKNOWN"
    for line in content:
        m = ts_re.search(line)
        if m:
            try:
                if datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC) < cutoff:
                    continue
            except ValueError:
                pass
        if dry_re.search(line):
            mode = "DRY-RUN"
        elif live_re.search(line):
            mode = "LIVE"
    return mode


def section_pnl_summary() -> tuple[list[str], list[str]]:
    lines = ["P&L SUMMARY (last 5 trading days)"]
    warnings = []
    lines.append(f"  {'Strategy':<18} {'Today':>10}  {'5d cumul':>10}  {'NAV / equity':>14}")
    lines.append("  " + "-" * 60)

    for name, ledger_dir in LEDGERS.items():
        df = _read_daily(ledger_dir)
        if df.empty:
            lines.append(f"  {name:<18} (no history)")
            continue

        try:
            last5   = df.tail(5)
            today_r = float(last5["ret"].iloc[-1])
            cum_r   = float((1 + last5["ret"].astype(float)).prod() - 1)
            sgn_t   = "+" if today_r >= 0 else ""
            sgn_c   = "+" if cum_r >= 0 else ""
            today_s = f"{sgn_t}{today_r*100:.2f}%"
            cum_s   = f"{sgn_c}{cum_r*100:.2f}%"

            # NAV / equity label
            if name == "ibkr_dynamic":
                # Prefer LIVE NetLiquidation (can't go stale if log_daily is
                # missed); fall back to state.json last_equity with a [cached]
                # marker so a stale fallback is visible.
                live_eq, _, _ = _fetch_ib_live()
                if live_eq is not None:
                    eq, eq_src = live_eq, "live"
                else:
                    state_p = ledger_dir / "state.json"
                    eq = json.load(open(state_p)).get("last_equity") if state_p.exists() else None
                    eq_src = "cached"
                nav_s = f"${eq:>10,.0f} [{eq_src}]" if eq else ""

                # Flag a stale ledger: live equity far from daily.csv's last row
                # means log_daily hasn't flushed.
                if live_eq is not None and not df.empty and "equity" in df.columns:
                    last_logged = float(df["equity"].iloc[-1])
                    if last_logged and abs(live_eq - last_logged) / last_logged > 0.02:
                        warnings.append(
                            f"ibkr_dynamic equity drift: live ${live_eq:,.0f} vs "
                            f"ledger ${last_logged:,.0f} — log_daily may not be flushing")

                mode = _detect_daemon_mode()
                suffix = "" if mode == "LIVE" else ("  (dry-run)" if mode == "DRY-RUN" else "")
                if "daily_pnl_usd" in df.columns:
                    pnl_usd = float(last5["daily_pnl_usd"].iloc[-1])
                    sgn_p   = "+" if pnl_usd >= 0 else "-"
                    today_s = f"{sgn_p}${abs(pnl_usd):,.0f}"
            else:
                nav_val = float(df["nav"].iloc[-1]) if "nav" in df.columns else None
                nav_s   = f"NAV {nav_val:.4f}" if nav_val is not None else ""
                suffix  = ""

            lines.append(f"  {name:<18} {today_s:>10}  {cum_s:>10}  {nav_s}{suffix}")

        except Exception as e:
            lines.append(f"  {name:<18} (error: {e})")

    return lines, warnings


# ── Section: daemon log ───────────────────────────────────────────────────────

def section_daemon_summary() -> tuple[list[str], list[str]]:
    lines = ["DAEMON LOG (last 24 h)"]
    warnings = []

    log_path = REPO / "ibkr_fut" / "daemon_cron.log"
    if not log_path.exists():
        lines.append("  (log not found)")
        return lines, warnings

    try:
        content = log_path.read_text(errors="replace").splitlines()
    except Exception as e:
        lines.append(f"  (error reading log: {e})")
        return lines, warnings

    now    = datetime.now(UTC)
    cutoff = now - timedelta(hours=24)

    cycle_re = re.compile(
        r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z\] Cycle done.*?(\d+) placed.*?(\d+) deferred"
    )
    dry_re = re.compile(
        r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z\] Cycle done \(DRY-RUN\): (\d+) checked"
    )
    ts_re    = re.compile(r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z\]")
    # ib_insync / logging-module lines stamp "YYYY-MM-DD HH:MM:SS,mmm" (UTC host).
    # Without parsing this too, every ERROR the library ever logged bypasses the
    # 24h cutoff and is re-reported forever.
    py_ts_re = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}")
    # Exclude routine noise: reconnects, mkt data sub errors, brief gateway outage
    # Error 354 = no mkt data subscription; Error 300 = cascade cancel after 354
    _BENIGN = ("Peer closed connection", "Disconnecting", "Connection lost",
               "Not connected", "ConnectionRefusedError", "TimeoutError",
               "Make sure API port", "could not connect to IBKR after",
               "Error 354,", "Error 300,")
    error_re = re.compile(r"\b(ERROR|CRITICAL)\b")

    cycles = placed = deferred = errors = 0
    error_lines: list[str] = []

    last_ts = None   # untimestamped lines (tracebacks, wrapped output) inherit this
    for line in content:
        # Gate by timestamp: daemon's own [ISO Z] stamps or logging-module stamps
        ts = None
        m = ts_re.search(line)
        if m:
            fmt = "%Y-%m-%dT%H:%M:%S"
        else:
            m = py_ts_re.search(line)
            fmt = "%Y-%m-%d %H:%M:%S"
        if m:
            try:
                ts = datetime.strptime(m.group(1), fmt).replace(tzinfo=UTC)
            except ValueError:
                pass
        if ts is not None:
            last_ts = ts
        if last_ts is not None and last_ts < cutoff:
            continue

        if dry_re.search(line):
            cycles += 1
            continue

        m = cycle_re.search(line)
        if m:
            cycles  += 1
            placed  += int(m.group(2))
            deferred += int(m.group(3))
            continue

        if error_re.search(line) and not any(b in line for b in _BENIGN):
            errors += 1
            error_lines.append(line.strip())

    mode = _detect_daemon_mode()   # most-recent-cycle wins (shared with P&L tag)
    lines.append(
        f"  {cycles} cycles ({mode})  ·  {placed} placed  ·  {deferred} deferred  ·  {errors} errors"
    )
    if error_lines:
        for el in reversed(error_lines):   # most recent first
            lines.append(f"    {el}")
        warnings.append(f"{errors} daemon error(s) in last 24h")

    return lines, warnings


# ── Section: gateway health (restart / uptime SLO) ────────────────────────────

_GW_LOG_RE = re.compile(r"ibgateway\.(\d{8})\.(\d{6})\.ibgzenc$")


def _gateway_settings_dir() -> Path | None:
    """The active TWS settings dir (the one holding ibg.xml)."""
    try:
        for d in sorted(GATEWAY_JTS_DIR.glob("*/")):
            if (d / "ibg.xml").exists():
                return d
    except Exception:
        pass
    return None


def _in_trading_window(ts: datetime) -> bool:
    h = ts.hour
    if GATEWAY_WINDOW_START_H <= GATEWAY_WINDOW_END_H:
        return GATEWAY_WINDOW_START_H <= h < GATEWAY_WINDOW_END_H
    # window wraps midnight (e.g. 22 → 07)
    return h >= GATEWAY_WINDOW_START_H or h < GATEWAY_WINDOW_END_H


def section_gateway_health() -> tuple[list[str], list[str]]:
    """Lightweight uptime SLO: how many times the Gateway restarted in the last
    24h, and whether any restart landed inside the trading window. After the
    2026-07-01 timezone fix the daily restart should be a single event at 03:00
    ET (07:00 UTC) — off-window — so an in-window restart is now the alarm."""
    lines = ["GATEWAY HEALTH"]
    warnings = []
    now = datetime.now(UTC)
    cutoff = now - timedelta(hours=24)

    sdir = _gateway_settings_dir()
    if sdir is None:
        lines.append("  ?  settings dir not found (skipping restart audit)")
        return lines, warnings

    restarts = []
    try:
        for f in sdir.glob("ibgateway.*.ibgzenc"):
            m = _GW_LOG_RE.search(f.name)
            if not m:
                continue
            try:
                ts = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
            except ValueError:
                continue
            if ts >= cutoff:
                restarts.append(ts)
    except Exception as e:
        lines.append(f"  ?  could not read session logs ({e})")
        return lines, warnings

    restarts.sort()
    in_window = [t for t in restarts if _in_trading_window(t)]

    if not restarts:
        lines.append("  ✓  no Gateway restarts in last 24h")
    else:
        times = ", ".join(t.strftime("%m-%d %H:%M") for t in restarts)
        tag = "✓" if not in_window else "✗"
        lines.append(f"  {tag}  {len(restarts)} restart(s) in 24h: {times} UTC")
        if in_window:
            iw = ", ".join(t.strftime("%m-%d %H:%M") for t in in_window)
            lines.append(f"     ⚠ {len(in_window)} landed in the trading window ({iw} UTC)")
            warnings.append(
                f"gateway: {len(in_window)} in-window restart(s) — {iw} UTC "
                f"(expected a single off-window restart at 07:00 UTC)"
            )

    return lines, warnings


# ── Discord ───────────────────────────────────────────────────────────────────

def send_discord(text: str) -> None:
    # Discord has a 2000-char message limit; split into chunks if needed
    chunks = [text[i:i+1990] for i in range(0, len(text), 1990)]
    for chunk in chunks:
        data = json.dumps({"content": f"```\n{chunk}\n```"}).encode()
        req  = urllib.request.Request(
            DISCORD_WEBHOOK_URL, data=data,
            headers={"Content-Type": "application/json",
                     "User-Agent": "DiscordBot (private, 1.0)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status not in (200, 204):
                raise RuntimeError(f"Discord webhook returned {resp.status}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    day_str  = datetime.now(UTC).strftime("%a %Y-%m-%d")
    sections = [
        section_cron_health,
        section_gateway_health,
        section_data_health,
        section_futures_positions,
        section_pnl_summary,
        section_daemon_summary,
    ]

    all_lines: list[str] = [f"=== Trading System Report  {day_str} ==="]
    all_warnings: list[str] = []

    for fn in sections:
        try:
            sec_lines, sec_warns = fn()
        except Exception as e:
            sec_lines = [fn.__name__.replace("section_", "").upper(), f"  ✗ section crashed: {e}"]
            sec_warns = [f"{fn.__name__} crashed: {e}"]
        all_lines.append("")
        all_lines.extend(sec_lines)
        all_warnings.extend(sec_warns)

    status = "⚠" if all_warnings else "✓"
    header = f"[Trading] {status} {day_str}"
    if all_warnings:
        header += f"  ({len(all_warnings)} warning{'s' if len(all_warnings) != 1 else ''})"
    body = header + "\n\n" + "\n".join(all_lines[1:])  # skip duplicate date line

    print(body)

    if not DISCORD_WEBHOOK_URL:
        print("[report] DISCORD_WEBHOOK_URL not set — skipped")
        return

    try:
        send_discord(body)
        print("[report] Discord message sent")
    except Exception as e:
        fallback = HERE / "daily_report_fallback.txt"
        fallback.write_text(body + f"\n\nSend error: {e}\n")
        print(f"[report] Discord failed ({e}) — written to {fallback}")
        sys.exit(1)


if __name__ == "__main__":
    main()
