#!/usr/bin/env python3
"""
ETF daily data downloader using yfinance.
Downloads adjusted OHLCV data for a diversified ETF universe covering
equities (US + international), fixed income, commodities, real estate, and FX.
Uses period="max" to get the full history back to each ETF's inception.

Saves each ETF to Data/etf/<TICKER>_1d_yf.csv with columns:
  time, open, high, low, close, volume

Usage:
  python3 etf/etf_data_getter.py
  python3 etf/etf_data_getter.py --tickers SPY QQQ TLT
  python3 etf/etf_data_getter.py --overwrite
"""

import argparse
import os
import time

import pandas as pd
import yfinance as yf

# ── ETF Universe ─────────────────────────────────────────────────────────────
ETF_UNIVERSE = {
    # ── US Broad Equity ───────────────────────────────────────────────────────
    "US_EQUITY_BROAD": [
        ("SPY",  "S&P 500"),
        ("QQQ",  "Nasdaq 100"),
        ("IWM",  "Russell 2000 Small Cap"),
        ("MDY",  "S&P 400 Mid Cap"),
        ("VTI",  "US Total Market"),
        ("IVV",  "S&P 500 iShares"),
        ("VTV",  "US Value"),
        ("VUG",  "US Growth"),
        ("IUSG", "US Large Growth"),
        ("IUSV", "US Large Value"),
    ],

    # ── US Equity Sectors ─────────────────────────────────────────────────────
    "US_EQUITY_SECTORS": [
        ("XLK",  "Technology"),
        ("XLF",  "Financials"),
        ("XLV",  "Health Care"),
        ("XLE",  "Energy"),
        ("XLI",  "Industrials"),
        ("XLU",  "Utilities"),
        ("XLP",  "Consumer Staples"),
        ("XLY",  "Consumer Discretionary"),
        ("XLB",  "Materials"),
        ("XLRE", "Real Estate"),
        ("XLC",  "Communication Services"),
        ("XBI",  "Biotech"),
        ("SMH",  "Semiconductors"),
        ("XHB",  "Homebuilders"),
        ("XRT",  "Retail"),
        ("KRE",  "Regional Banks"),
        ("KBE",  "Banking"),
        ("IBB",  "Biotech iShares"),
        ("IYT",  "Transportation"),
        ("ITA",  "Aerospace & Defense"),
    ],

    # ── US Equity Factors ─────────────────────────────────────────────────────
    "US_EQUITY_FACTORS": [
        ("MTUM", "Momentum Factor"),
        ("QUAL", "Quality Factor"),
        ("VLUE", "Value Factor"),
        ("USMV", "Min Volatility"),
        ("DGRO", "Dividend Growth"),
        ("VIG",  "Dividend Appreciation"),
        ("HDV",  "High Dividend"),
        ("NOBL", "Dividend Aristocrats"),
        ("COWZ", "Cash Flow Yield"),
        ("CALF", "Small Cap Cash Flow"),
    ],

    # ── International Equity – Broad ──────────────────────────────────────────
    "INTL_EQUITY_BROAD": [
        ("EFA",  "MSCI EAFE Developed ex-US"),
        ("EEM",  "MSCI Emerging Markets"),
        ("VEA",  "Vanguard Developed ex-US"),
        ("VWO",  "Vanguard Emerging Markets"),
        ("IEFA", "iShares Core MSCI EAFE"),
        ("IEMG", "iShares Core MSCI EM"),
        ("ACWI", "MSCI All World"),
        ("VXUS", "Vanguard Total Intl"),
        ("VSS",  "Vanguard FTSE All-World ex-US Small Cap"),
        ("DWX",  "SPDR Intl Dividend"),
    ],

    # ── International Equity – Country ────────────────────────────────────────
    "INTL_EQUITY_COUNTRY": [
        ("EWJ",  "Japan"),
        ("EWG",  "Germany"),
        ("EWU",  "United Kingdom"),
        ("EWY",  "South Korea"),
        ("EWZ",  "Brazil"),
        ("FXI",  "China Large Cap"),
        ("MCHI", "MSCI China"),
        ("INDA", "MSCI India"),
        ("EWT",  "Taiwan"),
        ("EWA",  "Australia"),
        ("EWC",  "Canada"),
        ("EWH",  "Hong Kong"),
        ("EWQ",  "France"),
        ("EWI",  "Italy"),
        ("EWP",  "Spain"),
        ("EWL",  "Switzerland"),
        ("EWD",  "Sweden"),
        ("EWN",  "Netherlands"),
        ("EWS",  "Singapore"),
        ("THD",  "Thailand"),
        ("EPHE", "Philippines"),
        ("EWM",  "Malaysia"),
        ("EIDO", "Indonesia"),
        ("EWW",  "Mexico"),
        ("ECH",  "Chile"),
        ("GXG",  "Colombia"),
        ("ARGT", "Argentina"),
        ("EZA",  "South Africa"),
    ],

    # ── Fixed Income – US Treasuries ──────────────────────────────────────────
    "FIXED_INCOME_TREASURIES": [
        ("TLT",  "20+ Year Treasury"),
        ("IEF",  "7-10 Year Treasury"),
        ("IEI",  "3-7 Year Treasury"),
        ("SHY",  "1-3 Year Treasury"),
        ("BIL",  "1-3 Month T-Bill"),
        ("SGOV", "0-3 Month T-Bill"),
        ("VGLT", "Vanguard Long-Term Treasury"),
        ("VGIT", "Vanguard Intermediate Treasury"),
        ("VGSH", "Vanguard Short-Term Treasury"),
        ("ZROZ", "Zero Coupon Long Treasury"),
        ("EDV",  "Extended Duration Treasury"),
    ],

    # ── Fixed Income – Inflation Protected ───────────────────────────────────
    "FIXED_INCOME_TIPS": [
        ("TIP",  "TIPS Broad"),
        ("STIP", "0-5 Year TIPS"),
        ("VTIP", "Vanguard Short-Term TIPS"),
        ("LTPZ", "Long-Term TIPS"),
        ("RINF", "ProShares Inflation Expectations"),
    ],

    # ── Fixed Income – Corporate ──────────────────────────────────────────────
    "FIXED_INCOME_CORPORATE": [
        ("LQD",  "Investment Grade Corporate"),
        ("HYG",  "High Yield Corporate"),
        ("JNK",  "SPDR High Yield"),
        ("VCSH", "Vanguard Short-Term Corporate"),
        ("VCIT", "Vanguard Intermediate Corporate"),
        ("VCLT", "Vanguard Long-Term Corporate"),
        ("IGIB", "iShares Intermediate Corporate"),
        ("IGSB", "iShares Short-Term Corporate"),
        ("IGLB", "iShares Long-Term Corporate"),
        ("USHY", "iShares Broad USD High Yield"),
        ("FLOT", "Floating Rate"),
        ("FALN", "Fallen Angels High Yield"),
    ],

    # ── Fixed Income – International / Other ─────────────────────────────────
    "FIXED_INCOME_OTHER": [
        ("AGG",  "US Aggregate Bond"),
        ("BND",  "Vanguard Total Bond"),
        ("BNDX", "Vanguard Intl Bond"),
        ("EMB",  "EM USD Bond"),
        ("VWOB", "Vanguard EM Bond"),
        ("MUB",  "National Municipal"),
        ("HYD",  "High Yield Municipal"),
        ("VTEB", "Vanguard Tax-Exempt"),
        ("BWX",  "SPDR Global ex-US Bond"),
        ("IAGG", "iShares Core Intl Aggregate"),
    ],

    # ── Commodities – Metals ──────────────────────────────────────────────────
    "COMMODITIES_METALS": [
        ("GLD",  "Gold SPDR"),
        ("IAU",  "Gold iShares"),
        ("GLDM", "Gold MiniShares"),
        ("SLV",  "Silver iShares"),
        ("PPLT", "Platinum"),
        ("PALL", "Palladium"),
        ("DBP",  "Precious Metals"),
        ("DBB",  "Base Metals"),
        ("COPX", "Copper Miners"),
        ("SILJ", "Silver Junior Miners"),
        ("GDX",  "Gold Miners"),
        ("GDXJ", "Junior Gold Miners"),
    ],

    # ── Commodities – Energy ──────────────────────────────────────────────────
    "COMMODITIES_ENERGY": [
        ("USO",  "Crude Oil"),
        ("UNG",  "Natural Gas"),
        ("DBE",  "Energy Commodity"),
        ("XOP",  "Oil & Gas Exploration"),
        ("OIH",  "Oil Services"),
        ("AMLP", "Alerian MLP"),
    ],

    # ── Commodities – Agriculture ─────────────────────────────────────────────
    "COMMODITIES_AGRICULTURE": [
        ("CORN", "Corn"),
        ("WEAT", "Wheat"),
        ("SOYB", "Soybeans"),
        ("DBA",  "Agriculture Diversified"),
        ("CANE", "Sugar"),
        ("JO",   "Coffee"),
        ("NIB",  "Cocoa"),
        ("BAL",  "Cotton"),
        ("RJA",  "Agriculture Rogers"),
    ],

    # ── Commodities – Broad ───────────────────────────────────────────────────
    "COMMODITIES_BROAD": [
        ("DBC",  "DB Commodity Index"),
        ("PDBC", "Optimum Yield Commodity"),
        ("GSG",  "S&P GSCI Commodity"),
        ("COMT", "iShares Commodity"),
    ],

    # ── Real Estate ───────────────────────────────────────────────────────────
    "REAL_ESTATE": [
        ("VNQ",  "US REITs Vanguard"),
        ("IYR",  "US Real Estate iShares"),
        ("REET", "Global REIT"),
        ("VNQI", "Intl REITs ex-US"),
        ("REM",  "Mortgage REITs"),
        ("O",    "Realty Income (Net Lease)"),
        ("PLD",  "Prologis (Industrial REIT)"),
        ("AMT",  "American Tower (Cell Tower REIT)"),
    ],

    # ── FX / Currency ETFs ────────────────────────────────────────────────────
    "CURRENCIES": [
        ("FXE",  "Euro"),
        ("FXY",  "Japanese Yen"),
        ("FXB",  "British Pound"),
        ("FXA",  "Australian Dollar"),
        ("FXC",  "Canadian Dollar"),
        ("FXF",  "Swiss Franc"),
        ("FXS",  "Swedish Krona"),
        ("CEW",  "Emerging Market Currency"),
        ("UUP",  "US Dollar Bullish"),
        ("UDN",  "US Dollar Bearish"),
        ("DBV",  "G10 Currency Harvest"),
    ],

    # ── Volatility / VIX ─────────────────────────────────────────────────────
    "VOLATILITY": [
        ("VXX",  "VIX Short-Term Futures"),
        ("UVXY", "2x VIX Short-Term"),
        ("SVXY", "Short VIX"),
    ],

    # ── Thematic / Alternative ────────────────────────────────────────────────
    "THEMATIC": [
        ("ARKK", "ARK Innovation"),
        ("BITO", "Bitcoin Futures ETF"),
        ("IBIT", "iShares Bitcoin Trust"),
        ("FBTC", "Fidelity Bitcoin ETF"),
        ("ETHA", "iShares Ethereum Trust"),
        ("FETH", "Fidelity Ethereum ETF"),
        ("BOTZ", "Global Robotics & AI"),
        ("HACK", "Cybersecurity"),
        ("LIT",  "Lithium & Battery"),
        ("ICLN", "Clean Energy"),
        ("TAN",  "Solar"),
    ],
}

# Flatten to list of (ticker, description, asset_class)
ALL_ETFS = [
    (ticker, desc, asset_class)
    for asset_class, tickers in ETF_UNIVERSE.items()
    for ticker, desc in tickers
]

OUTPUT_DIR = "Data/etf"


def download_ticker(ticker: str) -> pd.DataFrame | None:
    """Download full history of adjusted daily OHLCV for a single ticker."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="max", auto_adjust=True, actions=False)
        if df.empty:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df.index.name = "time"
        df.columns = ["open", "high", "low", "close", "volume"]
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]
        return df
    except Exception as e:
        print(f"  ERROR {ticker}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description="Download ETF daily data from yfinance (full history)")
    ap.add_argument("--tickers", nargs="*", help="Override: specific tickers to download")
    ap.add_argument("--overwrite", action="store_true", help="Redownload even if file exists")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.tickers:
        target = [(t, t, "CUSTOM") for t in args.tickers]
    else:
        target = ALL_ETFS

    print(f"Downloading {len(target)} ETFs → {OUTPUT_DIR}/  (full history from inception)")
    print()

    ok, skip, fail = [], [], []

    for ticker, desc, asset_class in target:
        out_path = os.path.join(OUTPUT_DIR, f"{ticker.lower()}_1d_yf.csv")

        if not args.overwrite and os.path.exists(out_path):
            skip.append(ticker)
            print(f"  SKIP  {ticker:<8}  (exists)")
            continue

        print(f"  DL    {ticker:<8}  {desc}", end="", flush=True)
        df = download_ticker(ticker)

        if df is None or len(df) < 50:
            fail.append(ticker)
            print(f"  → FAIL (no/insufficient data)")
            time.sleep(0.3)
            continue

        df.to_csv(out_path)
        ok.append(ticker)
        print(f"  → {len(df):,} bars  ({df.index[0].date()} – {df.index[-1].date()})")
        time.sleep(0.2)

    print(f"\n{'='*60}")
    print(f"  Downloaded: {len(ok)}")
    print(f"  Skipped:    {len(skip)}")
    print(f"  Failed:     {len(fail)}")
    if fail:
        print(f"  Failed tickers: {', '.join(fail)}")
    print(f"\nData saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
