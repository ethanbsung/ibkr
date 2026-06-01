#!/usr/bin/env python3
"""
One-time script: query Alpaca for shortability of every symbol in the ETF universe.
Saves Data/etf/etf_shortability.json. Re-run whenever the universe changes.

Usage:
  python3 etf/check_shortability.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from etf.live._env import load_dotenv

UNIVERSE_FILE      = "Data/etf/etf_universe_greedy.json"
SHORTABILITY_FILE  = "Data/etf/etf_shortability.json"
REQUEST_SPACING    = 0.35   # seconds between asset queries to respect rate limits


def main():
    load_dotenv()
    api_key    = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not api_secret:
        print("ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
        sys.exit(1)

    from alpaca.trading.client import TradingClient
    trading = TradingClient(api_key, api_secret, paper=True)

    with open(UNIVERSE_FILE) as f:
        u = json.load(f)
    tickers = u["selected"]
    asset_classes = u.get("asset_classes", {})

    print(f"Querying Alpaca asset info for {len(tickers)} symbols…")
    results = {}
    non_shortable = []

    for i, ticker in enumerate(sorted(tickers)):
        try:
            asset = trading.get_asset(ticker)
            shortable      = bool(asset.shortable)
            easy_to_borrow = bool(asset.easy_to_borrow)
            marginable     = bool(asset.marginable)
            results[ticker] = {
                "shortable":       shortable,
                "easy_to_borrow":  easy_to_borrow,
                "marginable":      marginable,
                "asset_class":     asset_classes.get(ticker, "UNKNOWN"),
            }
            flag = "" if shortable else "  ← NOT SHORTABLE"
            print(f"  {ticker:<6}  shortable={shortable}  etb={easy_to_borrow}{flag}")
            if not shortable:
                non_shortable.append(ticker)
        except Exception as e:
            print(f"  {ticker:<6}  ERROR: {e}")
            results[ticker] = {
                "shortable": True,   # assume shortable on error (conservative — don't silently block)
                "easy_to_borrow": True,
                "marginable": True,
                "asset_class": asset_classes.get(ticker, "UNKNOWN"),
                "error": str(e),
            }
        if i < len(tickers) - 1:
            time.sleep(REQUEST_SPACING)

    out = {
        "meta": {"universe_file": UNIVERSE_FILE, "n_symbols": len(tickers)},
        "symbols": results,
    }
    with open(SHORTABILITY_FILE, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nNon-shortable ({len(non_shortable)}): {sorted(non_shortable)}")
    print(f"Saved → {SHORTABILITY_FILE}")


if __name__ == "__main__":
    main()
