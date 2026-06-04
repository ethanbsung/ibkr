#!/bin/bash
# EWMAC dynamic-optimisation daily run (data refresh + execute).
# Portable: derives the repo root from this script's location, so it works on any
# host (laptop or VPS) with no path edits. Point cron at this wrapper; pass
# --execute to actually place orders, e.g. (VPS, /home/ethan, UTC host pinned to ET):
#   CRON_TZ=America/New_York
#   0 18 * * 1-5 /home/ethan/ibkr/ibkr_fut/run_dynamic.sh --execute >> /home/ethan/ibkr/ibkr_fut/dynamic_cron.log 2>&1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"
source "$REPO/venv/bin/activate"

echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') run_dynamic start (repo=$REPO) ====="

# 1. Refresh PST futures + FX data from IBKR so EWMAC signals use today's close.
#    Update the full Jumbo universe (non-US contracts may fail on a paper data
#    subscription — pst_updater skips those gracefully).
JUMBO_INSTRUMENTS=$(python3 -c "from ibkr_fut.jumbo import JUMBO; print(' '.join(JUMBO))")
python3 pst_updater.py $JUMBO_INSTRUMENTS --fx || echo "WARN: pst_updater nonzero; proceeding with existing data"

# 2. Run the EWMAC dynamic-optimisation executor (args passed through, e.g. --execute).
python3 ibkr_fut/live_dynamic.py "$@"
