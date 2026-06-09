#!/bin/bash
# Daily monitoring report — runs at 7 AM ET (11:00 UTC) weekdays.
# VPS cron: 0 11 * * 1-5 /home/ethan/ibkr/scripts/run_report.sh >> /home/ethan/ibkr/scripts/report_cron.log 2>&1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"
source "$REPO/venv/bin/activate"

echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') run_report start ====="
python3 scripts/daily_report.py
echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') run_report done ====="
