#!/bin/bash
# EWMAC dynamic-optimisation daemon execution (passive-aggressive algo, market-hours aware).
# Starts live_dynamic.py --mode daemon as a background process.  The daemon cycles every
# DAEMON_SLEEP_SECS (~10 min), placing orders when each exchange opens:
#   ~6:05 PM ET  — CME/CBOT/NYMEX (reopens after maintenance halt)
#   ~8–9 PM ET   — Asian markets (TSE, OSE, KSE, SGX)
#   ~2–3 AM ET   — Eurex / LIFFE (BUND, BOBL, DAX, EUROSTX, CAC, etc.)
# Deferred instruments are retried on the next cycle; the daemon automatically reloads
# targets_snapshot.json when the compute phase writes a new one (next day 6 PM).
#
# VPS cron (America/New_York TZ, UTC host — EDT offset):
#   5 22 * * 0,1-4 /home/ethan/ibkr/ibkr_fut/run_execution.sh >> /home/ethan/ibkr/ibkr_fut/execution_cron.log 2>&1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"
source "$REPO/venv/bin/activate"

echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') run_execution start (repo=$REPO) ====="

PIDFILE=/tmp/live_dynamic_daemon.pid
DAEMON_LOG="$REPO/ibkr_fut/daemon_cron.log"

# ── Stop existing daemon before starting a fresh one ──────────────────────────
if [ -f "$PIDFILE" ]; then
    OLD_PID=$(cat "$PIDFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M:%S'): stopping previous daemon (PID $OLD_PID)"
        kill "$OLD_PID"
        for i in $(seq 1 15); do
            kill -0 "$OLD_PID" 2>/dev/null || break
            sleep 1
        done
        # Force-kill if still alive after 15 s (e.g. stuck in blocking IB call)
        if kill -0 "$OLD_PID" 2>/dev/null; then
            echo "$(date '+%Y-%m-%d %H:%M:%S'): force-killing daemon (PID $OLD_PID)"
            kill -9 "$OLD_PID" 2>/dev/null || true
            sleep 1
        fi
    fi
    rm -f "$PIDFILE"
fi

# ── Rotate daemon log if it exceeds 50 MB ─────────────────────────────────────
if [ -f "$DAEMON_LOG" ] && [ "$(stat -c%s "$DAEMON_LOG" 2>/dev/null || echo 0)" -gt 52428800 ]; then
    mv "$DAEMON_LOG" "${DAEMON_LOG}.1"
    echo "$(date '+%Y-%m-%d %H:%M:%S'): rotated daemon log → ${DAEMON_LOG}.1"
fi

# ── Start daemon in background (nohup ensures it survives shell exit) ─────────
nohup env PYTHONUNBUFFERED=1 python3 ibkr_fut/live_dynamic.py --mode daemon >> "$DAEMON_LOG" 2>&1 &  # add --execute when market data active
DAEMON_PID=$!
echo $DAEMON_PID > "$PIDFILE"

echo "$(date '+%Y-%m-%d %H:%M:%S'): daemon started (PID $DAEMON_PID) → $DAEMON_LOG"
echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') run_execution done ====="
