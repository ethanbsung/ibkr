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

# name → (log path relative to REPO, max age in hours before flagging stale)
CRONS = {
    "pst_updater":  ("pst_updater.log",               26),
    "compute":      ("ibkr_fut/dynamic_cron.log",      26),
    "daemon":       ("ibkr_fut/daemon_cron.log",        24),
    "etf_daily":    ("paper/etf_daily.log",             26),
    "crypto_paper": ("paper/paper_cron.log",            26),
}

LEDGERS = {
    "ibkr_dynamic": REPO / "paper" / "ledgers" / "ibkr_dynamic",
    "etf_ewmac":    REPO / "paper" / "ledgers" / "etf_ewmac",
    "crypto_trend": REPO / "paper" / "ledgers" / "crypto_trend",
    "crypto_carry": REPO / "paper" / "ledgers" / "crypto_carry",
}

IB_HOST, IB_PORT, IB_CLIENT = "127.0.0.1", 4002, 20


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


def section_cron_health() -> tuple[list[str], list[str]]:
    lines = ["CRON HEALTH"]
    warnings = []
    now = datetime.now(UTC)

    for name, (rel, max_h) in CRONS.items():
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
            # Also show cycle count from parsed timestamps (Cycle done lines)
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

            if age_h > max_h:
                lines.append(f"  {name:<15} ✗  last modified {mtime.strftime('%Y-%m-%d %H:%M')} UTC  ({age_h:.0f}h ago)")
                warnings.append(f"{name}: no activity in {age_h:.0f}h")
            else:
                lines.append(f"  {name:<15} ✓  {mtime.strftime('%Y-%m-%d %H:%M')} UTC  ({n_cycles} cycles in 24h)")
        else:
            if age_h > max_h:
                lines.append(f"  {name:<15} ✗  {mtime.strftime('%Y-%m-%d %H:%M')} UTC  ({age_h:.0f}h ago)")
                warnings.append(f"{name}: last run {age_h:.0f}h ago (>{max_h}h)")
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

    # Try live IBKR first (gateway always up via IBC)
    held   = {}
    source = "cached"
    try:
        from ib_insync import IB
        ib = IB()
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT, timeout=8, readonly=True)
        for pos in ib.positions():
            c = pos.contract
            if c.secType == "FUT":
                held[c.symbol] = int(pos.position)
        ib.disconnect()
        source = "live"
    except Exception:
        pos_csv = LEDGERS["ibkr_dynamic"] / "positions.csv"
        if pos_csv.exists():
            df = pd.read_csv(pos_csv)
            if not df.empty:
                latest = df[df["date"] == df["date"].max()]
                for _, row in latest.iterrows():
                    if int(row["qty"]) != 0:
                        held[row["symbol"]] = int(row["qty"])

    all_syms = sorted(set(list(targets) + list(held)))
    tgt_str  = "  ".join(f"{s}={targets.get(s, 0):+d}" for s in all_syms) if targets else "(none)"

    held_parts = []
    for s in all_syms:
        t = targets.get(s, 0)
        h = held.get(s, 0)
        flag = "  ← MISMATCH" if t != h else ""
        held_parts.append(f"{s}={h:+d}{flag}")
        if t != h:
            warnings.append(f"position mismatch {s}: target={t:+d} held={h:+d}")

    lines.append(f"  Snapshot: {snap_date}  [source: {source}]")
    lines.append(f"  Target:   {tgt_str}")
    lines.append(f"  Held:     {'  '.join(held_parts) or '(flat)'}")

    return lines, warnings


# ── Section: P&L summary ──────────────────────────────────────────────────────

def _read_daily(ledger_dir: Path) -> pd.DataFrame:
    fp = ledger_dir / "daily.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date", keep="last")


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
                state_p = ledger_dir / "state.json"
                eq = None
                if state_p.exists():
                    eq = json.load(open(state_p)).get("last_equity")
                nav_s = f"${eq:>10,.0f}" if eq else ""
                suffix = "  (dry-run)"
                if "daily_pnl_usd" in df.columns:
                    pnl_usd = float(last5["daily_pnl_usd"].iloc[-1])
                    sgn_p   = "+" if pnl_usd >= 0 else ""
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
    # Exclude routine noise: reconnects, mkt data sub errors, brief gateway outage
    # Error 354 = no mkt data subscription; Error 300 = cascade cancel after 354
    _BENIGN = ("Peer closed connection", "Disconnecting", "Connection lost",
               "Not connected", "ConnectionRefusedError", "TimeoutError",
               "Make sure API port", "could not connect to IBKR after",
               "Error 354,", "Error 300,")
    error_re = re.compile(r"\b(ERROR|CRITICAL)\b")

    cycles = placed = deferred = errors = 0
    is_dry = False
    error_lines: list[str] = []

    for line in content:
        # Gate by timestamp if present
        m = ts_re.search(line)
        if m:
            try:
                ts = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
                if ts < cutoff:
                    continue
            except ValueError:
                pass

        if dry_re.search(line):
            cycles += 1
            is_dry  = True
            continue

        m = cycle_re.search(line)
        if m:
            cycles  += 1
            placed  += int(m.group(2))
            deferred += int(m.group(3))
            continue

        if error_re.search(line) and not any(b in line for b in _BENIGN):
            errors += 1
            error_lines.append(line.strip()[:120])

    mode = "DRY-RUN" if is_dry else "LIVE"
    lines.append(
        f"  {cycles} cycles ({mode})  ·  {placed} placed  ·  {deferred} deferred  ·  {errors} errors"
    )
    if error_lines:
        for el in error_lines[:3]:
            lines.append(f"    {el}")
        if len(error_lines) > 3:
            lines.append(f"    ... and {len(error_lines) - 3} more")
        warnings.append(f"{errors} daemon error(s) in last 24h")

    return lines, warnings


# ── Discord ───────────────────────────────────────────────────────────────────

def send_discord(text: str) -> None:
    # Discord has a 2000-char message limit; split into chunks if needed
    chunks = [text[i:i+1990] for i in range(0, len(text), 1990)]
    for chunk in chunks:
        data = json.dumps({"content": f"```\n{chunk}\n```"}).encode()
        req  = urllib.request.Request(
            DISCORD_WEBHOOK_URL, data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status not in (200, 204):
                raise RuntimeError(f"Discord webhook returned {resp.status}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    day_str  = datetime.now(UTC).strftime("%a %Y-%m-%d")
    sections = [
        section_cron_health,
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
