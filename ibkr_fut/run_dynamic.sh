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

# 1. Refresh PST futures + FX data from IBKR so trend+carry signals use today's close.
#    --carry refreshes the multiple_prices term structure (CARRY leg) too; the combined
#    forecast sizes off carry, and live_dynamic ages the CARRY column (require_carry_fresh),
#    so a stale term structure would fail instruments out without this.
UNIVERSE_INSTRUMENTS=$(python3 -c "from ibkr_fut.instrument_universe import UNIVERSE; print(' '.join(UNIVERSE))") \
  || { echo "ERROR: failed to load UNIVERSE — aborting compute (no snapshot written)"; exit 1; }
python3 ibkr_fut/pst_updater.py $UNIVERSE_INSTRUMENTS --fx --carry || echo "WARN: pst_updater nonzero; proceeding with existing data"

# 2. Refresh volume cache weekly (stale after 7 days; used by instrument_selection filter).
python3 ibkr_fut/volume_collector.py --max-age 7 || echo "WARN: volume_collector nonzero; proceeding with existing cache"

# 3. Execution-path preflight: contract qualification, market hours, live market
#    data (open markets only). Failures alert via Discord but do NOT block
#    compute — broken instruments fail safe (skipped) at order time.
python3 ibkr_fut/preflight_check.py || echo "WARN: preflight reported failures (alerted); proceeding"

# 4. Run the combined carry+trend dynamic-optimisation executor — compute phase only.
#    Order execution is handled by the daemon (run_execution.sh), which picks up
#    the new snapshot automatically.
python3 ibkr_fut/live_dynamic.py --mode compute
