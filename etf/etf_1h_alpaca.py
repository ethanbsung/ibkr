#!/usr/bin/env python3
"""
Download 1h ETF bars from Alpaca for all ETFs that passed the quality check.
Alpaca's 1h data starts 2016-01-01.

Reads credentials from .env (ALPACA_API_KEY / ALPACA_SECRET_KEY).
Reads the ETF universe from Data/etf/etf_universe.json (produced by
etf_data_quality.py). Falls back to a hardcoded list if that file is absent.

Saves each ETF to Data/etf/<TICKER>_1h_alpaca.csv with columns:
  time, open, high, low, close, volume, vwap, n_trades

Usage:
  python3 etf/etf_1h_alpaca.py
  python3 etf/etf_1h_alpaca.py --tickers SPY QQQ TLT
  python3 etf/etf_1h_alpaca.py --overwrite
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd

# ── Load .env if present ──────────────────────────────────────────────────────
def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_dotenv()

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

API_KEY    = os.environ.get("ALPACA_API_KEY", "")
API_SECRET = os.environ.get("ALPACA_SECRET_KEY", "")

# Alpaca 1h history starts here
ALPACA_START = datetime(2016, 1, 1, tzinfo=timezone.utc)
OUTPUT_DIR   = "Data/etf"
UNIVERSE_FILE = "Data/etf/etf_universe.json"

# Fallback universe if JSON not found (from last quality run)
FALLBACK_TICKERS = [
    "ACWI","AGG","AMLP","AMT","ARKK","BIL","BITO","BND","BNDX","BOTZ","BWX",
    "CALF","COPX","COWZ","DBA","DBC","DGRO","ECH","EDV","EEM","EFA","EIDO",
    "EMB","EWA","EWC","EWG","EWH","EWI","EWJ","EWL","EWP","EWQ","EWT","EWU",
    "EWW","EWY","EWZ","EZA","FALN","FLOT","FXE","FXI","FXY","GDX","GDXJ",
    "GLD","GLDM","GSG","HDV","HYD","HYG","IAGG","IAU","IBB","ICLN","IEF",
    "IEFA","IEI","IEMG","IGIB","IGLB","IGSB","INDA","ITA","IUSG","IUSV",
    "IVV","IWM","IYR","IYT","JNK","KBE","KRE","LIT","LQD","MCHI","MDY",
    "MTUM","MUB","NOBL","O","OIH","PDBC","PLD","PPLT","QQQ","QUAL","REET",
    "SGOV","SHY","SILJ","SLV","SMH","SPY","STIP","SVXY","TAN","TIP","TLT",
    "UNG","USHY","USMV","USO","UUP","UVXY","VCIT","VCLT","VCSH","VEA","VGIT",
    "VGLT","VGSH","VIG","VLUE","VNQ","VNQI","VSS","VTEB","VTI","VTIP","VTV",
    "VUG","VWO","VWOB","VXUS","VXX","XBI","XHB","XLB","XLC","XLE","XLF",
    "XLI","XLK","XLP","XLRE","XLU","XLV","XLY","XOP","XRT",
]


def load_universe() -> list[str]:
    if os.path.exists(UNIVERSE_FILE):
        with open(UNIVERSE_FILE) as f:
            data = json.load(f)
        return data["passed"]
    print(f"  Warning: {UNIVERSE_FILE} not found, using fallback list")
    return FALLBACK_TICKERS


def _to_df(bars_obj, ticker: str) -> pd.DataFrame:
    """Convert alpaca-py bars object to a clean DataFrame."""
    df = bars_obj.df
    if df.empty:
        return df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(ticker, level="symbol")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("UTC").tz_localize(None)
    df.index.name = "time"
    df = df.rename(columns={"trade_count": "n_trades"})
    keep = [c for c in ["open","high","low","close","volume","vwap","n_trades"] if c in df.columns]
    return df[keep]


def fetch_bars(client: StockHistoricalDataClient, ticker: str) -> pd.DataFrame | None:
    """Fetch all 1h bars from ALPACA_START to now via year-chunk pagination."""
    from datetime import timedelta

    end = datetime.now(tz=timezone.utc)
    # Split into ~6-month windows to stay under the 10k-bar-per-request limit.
    # SPY has ~16 bars/day * 126 trading days ≈ 2,000 bars per 6-month window.
    window = timedelta(days=180)

    chunk_start = ALPACA_START
    frames = []

    while chunk_start < end:
        chunk_end = min(chunk_start + window, end)
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame(1, TimeFrameUnit.Hour),
            start=chunk_start,
            end=chunk_end,
            adjustment="all",
            feed="sip",
            limit=10000,
        )
        try:
            bars = client.get_stock_bars(req)
            df_chunk = _to_df(bars, ticker)
            if not df_chunk.empty:
                frames.append(df_chunk)
        except Exception as e:
            err = str(e)
            if "recent SIP data" in err or "subscription" in err.lower():
                # Free plan can't query the most recent ~2 weeks of SIP data; skip silently.
                pass
            else:
                print(f"\n  WARN {ticker} {chunk_start.date()}–{chunk_end.date()}: {e}")
        chunk_start = chunk_end

    if not frames:
        return None

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def main():
    ap = argparse.ArgumentParser(description="Download 1h ETF data from Alpaca")
    ap.add_argument("--tickers", nargs="*", help="Override: specific tickers to download")
    ap.add_argument("--overwrite", action="store_true", help="Redownload even if file exists")
    args = ap.parse_args()

    if not API_KEY or not API_SECRET:
        print("ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env or environment")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = StockHistoricalDataClient(API_KEY, API_SECRET)

    tickers = args.tickers if args.tickers else load_universe()
    print(f"Downloading 1h bars for {len(tickers)} ETFs from Alpaca (2016-01-01 → now)")
    print(f"Output: {OUTPUT_DIR}/<TICKER>_1h_alpaca.csv\n")

    ok, skip, fail = [], [], []

    for ticker in tickers:
        out_path = os.path.join(OUTPUT_DIR, f"{ticker.lower()}_1h_alpaca.csv")

        if not args.overwrite and os.path.exists(out_path):
            skip.append(ticker)
            print(f"  SKIP  {ticker:<8}  (exists)")
            continue

        print(f"  DL    {ticker:<8}", end="", flush=True)
        df = fetch_bars(client, ticker)

        if df is None or len(df) < 50:
            fail.append(ticker)
            print(f"  → FAIL")
            time.sleep(0.3)
            continue

        df.to_csv(out_path)
        ok.append(ticker)
        print(f"  → {len(df):,} bars  ({df.index[0].date()} – {df.index[-1].date()})")
        time.sleep(0.15)

    print(f"\n{'='*60}")
    print(f"  Downloaded: {len(ok)}")
    print(f"  Skipped:    {len(skip)}")
    print(f"  Failed:     {len(fail)}")
    if fail:
        print(f"  Failed: {', '.join(fail)}")
    print(f"\nData saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
