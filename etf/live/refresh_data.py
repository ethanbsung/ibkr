#!/usr/bin/env python3
"""
Nightly ETF data refresh — keeps Data/etf/<tk>_1d_yf.csv current.

Runs AFTER the close, as a SEPARATE cron from the 3:58 PM execution job, so
yfinance history downloads never sit on the trade path.  (The execution job uses
yfinance *current* prices only — see etf/live/prices.py — and reads the CSVs that
this job keeps up to date.)

Does a FULL-history re-download with auto_adjust=True and OVERWRITES each file,
so the entire series stays on one consistent split/dividend adjustment basis.

Why full overwrite, not append:
  yfinance auto_adjust rescales ALL historical prices whenever a new dividend is
  paid.  Appending only new rows leaves the old rows on the pre-dividend basis,
  injecting a permanent discontinuity (a spurious 1-day return) at every dividend
  — bad for a bond/T-bill-heavy universe that distributes monthly.  This job is
  off the trade path, so the extra download time is fine.

Cron (America/New_York), after the close:
    30 18 * * 1-5  cd /home/ethanbsung/ibkr && source venv/bin/activate && \
                   python3 etf/live/refresh_data.py >> paper/etf_refresh.log 2>&1

Usage:
    python3 etf/live/refresh_data.py                 # refresh the live universe
    python3 etf/live/refresh_data.py --tickers SPY QQQ
"""

import argparse
import json
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from etf.live.etf_data_getter import download_ticker, OUTPUT_DIR

UNIVERSE_FILE = "Data/etf/etf_universe_greedy.json"
MIN_ROWS      = 50    # reject obviously-bad downloads


def refresh(tickers: list[str]) -> tuple[list[str], list[str]]:
    """Full-history re-download + overwrite for each ticker."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ok, failed = [], []
    for tk in tickers:
        df = download_ticker(tk)
        if df is None or len(df) < MIN_ROWS:
            failed.append(tk)
            print(f"  FAIL  {tk:<8}  (no/insufficient data)")
            time.sleep(0.3)
            continue
        path = os.path.join(OUTPUT_DIR, f"{tk.lower()}_1d_yf.csv")
        df.to_csv(path)   # overwrite — whole series on one adjustment basis
        ok.append(tk)
        print(f"  OK    {tk:<8}  {len(df)} rows  → {path}")
        time.sleep(0.2)   # be polite to yfinance
    return ok, failed


def main():
    ap = argparse.ArgumentParser(description="Nightly ETF data refresh (full history, overwrite)")
    ap.add_argument("--tickers", nargs="*", help="Override: specific tickers to refresh")
    args = ap.parse_args()

    if args.tickers:
        tickers = args.tickers
    else:
        with open(UNIVERSE_FILE) as f:
            tickers = json.load(f)["selected"]

    print(f"Nightly refresh: {len(tickers)} tickers (full history, overwrite)\n")
    ok, failed = refresh(tickers)
    print(f"\n  Refreshed {len(ok)}/{len(tickers)}"
          + (f", {len(failed)} failed: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
