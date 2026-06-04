#!/bin/bash
# Daily paper-trading run. Cron-safe. Portable: repo root derived from script location.
# Crypto daily bars close at 00:00 UTC; running any time after that picks up the
# latest completed bar. Idempotent, so an extra run is harmless.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"
"$REPO/venv/bin/python3" paper/run_paper.py "$@" >> "$REPO/paper/paper_cron.log" 2>&1
