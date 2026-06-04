#!/bin/bash
# ETF shortability refresh. Portable: repo root derived from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO"
"$REPO/venv/bin/python3" etf/check_shortability.py
