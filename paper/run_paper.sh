#!/bin/bash
# Daily paper-trading run. Cron-safe: handles cwd + venv.
# Crypto daily bars close at 00:00 UTC; running any time after that picks up the
# latest completed bar. Idempotent, so an extra run is harmless.
cd /home/ethanbsung/ibkr
/home/ethanbsung/ibkr/venv/bin/python3 paper/run_paper.py "$@" >> /home/ethanbsung/ibkr/paper/paper_cron.log 2>&1
