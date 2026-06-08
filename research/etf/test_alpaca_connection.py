#!/usr/bin/env python3
"""
Alpaca API connection test.
Run on VPS with: source venv/bin/activate && python3 etf/test_alpaca_connection.py

Requires env vars: ALPACA_API_KEY, ALPACA_SECRET_KEY
"""

import os
import sys
from pathlib import Path

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"

def load_dotenv():
    # Walk up from this file's location to find a .env
    for parent in Path(__file__).resolve().parents:
        env_file = parent / ".env"
        if env_file.exists():
            loaded = []
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key not in os.environ:  # don't override real env vars
                        os.environ[key] = val
                        loaded.append(key)
            print(f"{INFO} Loaded {len(loaded)} var(s) from {env_file}")
            return
    print(f"{INFO} No .env file found in any parent directory")


def check_env_vars():
    print("\n--- 1. Environment Variables ---")
    load_dotenv()
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

    if not api_key:
        print(f"{FAIL} ALPACA_API_KEY is not set.")
        print("       Fix: export ALPACA_API_KEY=your_key  (or add it to your .env and source it)")
        return None, None
    else:
        print(f"{PASS} ALPACA_API_KEY found ({api_key[:4]}...{api_key[-4:]})")

    if not secret_key:
        print(f"{FAIL} ALPACA_SECRET_KEY is not set.")
        print("       Fix: export ALPACA_SECRET_KEY=your_secret  (or add it to your .env and source it)")
        return None, None
    else:
        print(f"{PASS} ALPACA_SECRET_KEY found ({secret_key[:4]}...{secret_key[-4:]})")

    return api_key, secret_key


def check_import():
    print("\n--- 2. Package Import ---")
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        print(f"{PASS} alpaca-py packages imported successfully")
        return True
    except ImportError as e:
        print(f"{FAIL} Failed to import alpaca-py: {e}")
        print("       Fix: pip install alpaca-py")
        return False


def check_trading_client(api_key, secret_key):
    print("\n--- 3. Trading Client (Paper) ---")
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(api_key, secret_key, paper=True)
        account = client.get_account()
        print(f"{PASS} Connected to Alpaca paper trading account")
        print(f"{INFO} Account ID:     {account.id}")
        print(f"{INFO} Status:         {account.status}")
        print(f"{INFO} Equity:         ${float(account.equity):,.2f}")
        print(f"{INFO} Cash:           ${float(account.cash):,.2f}")
        print(f"{INFO} Buying power:   ${float(account.buying_power):,.2f}")
        print(f"{INFO} PDT flag:       {account.pattern_day_trader}")
        print(f"{INFO} Trading blocked:{account.trading_blocked}")
        if account.trading_blocked:
            print(f"       WARNING: Trading is blocked on this account. Check Alpaca dashboard.")
        return True
    except Exception as e:
        err = str(e)
        print(f"{FAIL} Trading client failed: {err}")
        if "403" in err or "forbidden" in err.lower():
            print("       Likely cause: API key does not have trading permissions, or wrong key type (live vs paper).")
        elif "401" in err or "unauthorized" in err.lower():
            print("       Likely cause: Invalid or expired API key/secret. Regenerate at alpaca.markets.")
        elif "connection" in err.lower() or "timeout" in err.lower():
            print("       Likely cause: Network issue or Alpaca API is down. Check https://status.alpaca.markets")
        else:
            print("       Check your key/secret and account status at alpaca.markets.")
        return False


def check_data_client(api_key, secret_key):
    print("\n--- 4. Market Data Client ---")
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        client = StockHistoricalDataClient(api_key, secret_key)
        req = StockLatestQuoteRequest(symbol_or_symbols=["SPY"])
        quotes = client.get_stock_latest_quote(req)
        spy = quotes["SPY"]
        print(f"{PASS} Market data client connected")
        print(f"{INFO} SPY latest ask:  ${float(spy.ask_price):,.2f}")
        print(f"{INFO} SPY latest bid:  ${float(spy.bid_price):,.2f}")
        return True
    except Exception as e:
        err = str(e)
        print(f"{FAIL} Data client failed: {err}")
        if "403" in err or "forbidden" in err.lower():
            print("       Likely cause: Your Alpaca plan may not include market data. Check subscription at alpaca.markets.")
        elif "401" in err or "unauthorized" in err.lower():
            print("       Likely cause: Invalid API key/secret for data endpoint.")
        elif "connection" in err.lower() or "timeout" in err.lower():
            print("       Likely cause: Network issue. Check connectivity or https://status.alpaca.markets")
        return False


def check_positions(api_key, secret_key):
    print("\n--- 5. Current Positions ---")
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(api_key, secret_key, paper=True)
        positions = client.get_all_positions()
        if not positions:
            print(f"{INFO} No open positions")
        else:
            print(f"{PASS} {len(positions)} open position(s):")
            for p in positions[:10]:
                print(f"{INFO}   {p.symbol:6s}  qty={p.qty:>10s}  market_value=${float(p.market_value):>10,.2f}")
            if len(positions) > 10:
                print(f"{INFO}   ... and {len(positions) - 10} more")
        return True
    except Exception as e:
        print(f"{FAIL} Could not fetch positions: {e}")
        return False


def main():
    print("=" * 50)
    print("Alpaca API Connection Test")
    print("=" * 50)

    api_key, secret_key = check_env_vars()
    if not api_key:
        print("\nCannot continue without API keys. Exiting.")
        sys.exit(1)

    if not check_import():
        print("\nCannot continue without alpaca-py. Exiting.")
        sys.exit(1)

    trading_ok = check_trading_client(api_key, secret_key)
    data_ok    = check_data_client(api_key, secret_key)
    positions_ok = check_positions(api_key, secret_key) if trading_ok else False

    print("\n--- Summary ---")
    print(f"  Trading client:  {'OK' if trading_ok else 'FAILED'}")
    print(f"  Data client:     {'OK' if data_ok else 'FAILED'}")
    print(f"  Positions fetch: {'OK' if positions_ok else 'FAILED' if trading_ok else 'SKIPPED'}")

    if trading_ok and data_ok:
        print("\nAll checks passed. Alpaca API is working correctly.")
        sys.exit(0)
    else:
        print("\nOne or more checks failed. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
