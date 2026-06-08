#!/bin/bash
# EWMAC dynamic-optimisation order execution (passive-aggressive algo).
# Runs at 6:05 PM ET, 5 minutes after run_dynamic.sh writes the targets snapshot.
# Reads targets_snapshot.json written by run_dynamic.sh and places limit orders.
#
# VPS cron (America/New_York TZ, UTC host — EDT offset):
#   5 22 * * 1-5 /home/ethan/ibkr/ibkr_fut/run_execution.sh >> /home/ethan/ibkr/ibkr_fut/execution_cron.log 2>&1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"
source "$REPO/venv/bin/activate"

echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') run_execution start (repo=$REPO) ====="

python3 ibkr_fut/live_dynamic.py --mode execute  # add --execute when Level 1 data is active

echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') run_execution done ====="
