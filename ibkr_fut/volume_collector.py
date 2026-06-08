"""
volume_collector.py — Fetch 20-day average daily volume from IB for a list of
futures instruments and cache results to CSV.

Volume is used by instrument_selection.py to apply Carver's two liquidity filters:
  1. Daily volume in contracts >= MIN_VOLUME_CONTRACTS (100)
  2. Daily volume in risk >= MIN_VOLUME_RISK ($1.5M/day)

Volume risk ($M/day) = avg_daily_contracts × mult × price × annual_vol / 16 / 1e6

IB Gateway must be running (port 4002 paper, 4001 live) to fetch data.
Results are stored in Data/volume_cache.csv and refreshed when stale.

Usage:
    python volume_collector.py                    # update all UNIVERSE instruments
    python volume_collector.py SP500_micro BUND   # specific instruments
    python volume_collector.py --max-age 7        # refresh if older than 7 days
"""

import os
import sys
import time
import logging
import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd
from ib_insync import IB, Future, util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibkr_fut.pst_loader import PSTLoader
from ibkr_fut.instrument_universe import UNIVERSE
from ibkr_fut.foundations import blended_vol, pct_returns_backadjusted, PST_CUTOFF

# ── Config ────────────────────────────────────────────────────────────────────

IB_HOST   = "127.0.0.1"
IB_PORT   = 4002    # paper; live = 4001
IB_CLIENT = 15      # distinct client ID from pst_updater (10) and live (1-9)

LOOKBACK_DAYS = 20  # calendar days of volume history to average
MAX_AGE_DAYS  = 7   # re-fetch if cached entry older than this

_REPO       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH  = os.path.join(_REPO, "Data", "volume_cache.csv")
IB_CFG_PATH = os.path.join(_REPO, "Data", "pst", "ib_config", "ib_config_futures.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ── Cache I/O ─────────────────────────────────────────────────────────────────

def load_cache() -> pd.DataFrame:
    """
    Load cached volume data.
    Columns: instrument, avg_contracts, fetch_date
    """
    if not os.path.exists(CACHE_PATH):
        return pd.DataFrame(columns=["instrument", "avg_contracts", "fetch_date"])
    df = pd.read_csv(CACHE_PATH, parse_dates=["fetch_date"])
    return df


def save_cache(df: pd.DataFrame) -> None:
    df = df.sort_values("instrument").reset_index(drop=True)
    df.to_csv(CACHE_PATH, index=False)


def stale_instruments(
    instruments: list[str], cache: pd.DataFrame, max_age_days: int = MAX_AGE_DAYS
) -> list[str]:
    """Return instruments missing from cache or with fetch_date older than max_age_days."""
    cutoff = pd.Timestamp(date.today() - timedelta(days=max_age_days))
    if cache.empty:
        return list(instruments)
    fresh = set(
        cache.loc[cache["fetch_date"] >= cutoff, "instrument"].tolist()
    )
    return [i for i in instruments if i not in fresh]


# ── IB fetch ──────────────────────────────────────────────────────────────────

def _ib_params(ib_cfg: pd.DataFrame, instrument: str) -> dict | None:
    if instrument not in ib_cfg.index:
        return None
    row = ib_cfg.loc[instrument]
    mult = row.get("IBMultiplier", "")
    raw_curr = row.get("IBCurrency", "")
    currency = "" if (pd.isna(raw_curr) or str(raw_curr).strip().upper() == "NA") else str(raw_curr).strip()
    raw_tc = row.get("IBTradingClass", "")
    trading_class = "" if pd.isna(raw_tc) else str(raw_tc).strip()
    return {
        "symbol":        str(row["IBSymbol"]),
        "exchange":      str(row["IBExchange"]),
        "currency":      currency,
        "multiplier":    "" if pd.isna(mult) else (
            str(int(float(mult))) if float(mult) == int(float(mult)) else str(float(mult))
        ),
        "trading_class": trading_class,
    }


def fetch_volume(
    ib: IB,
    ib_cfg: pd.DataFrame,
    instrument: str,
    lookback_days: int = LOOKBACK_DAYS,
) -> float:
    """
    Fetch average daily volume (contracts) for the front contract of an instrument.
    Returns NaN on failure.
    """
    params = _ib_params(ib_cfg, instrument)
    if params is None:
        log.warning(f"  {instrument}: not in IB config")
        return np.nan

    today = date.today()
    start = today - timedelta(days=lookback_days + 10)  # pad for non-trading days
    duration = f"{lookback_days + 10} D"

    # Resolve the front contract via reqContractDetails (sorted by expiry),
    # then fetch historical data using the qualified conId.  IB Gateway rejects
    # historical data requests for futures without an explicit expiry or conId.
    underspec = Future(
        symbol=params["symbol"],
        exchange=params["exchange"],
        currency=params["currency"] or "",
        multiplier=params["multiplier"] or "",
        tradingClass=params["trading_class"] or "",
    )
    try:
        details = ib.reqContractDetails(underspec)
    except Exception as e:
        log.warning(f"  {instrument}: reqContractDetails error — {e}")
        return np.nan

    # Filter to monthly/quarterly contracts — skip daily and weekly expirations.
    # Daily contracts have localSymbol ending in ' D' (e.g. "FMEA 20260604 D").
    # Weekly contracts expire within 7 days; skip those too.
    today_str = date.today().strftime("%Y%m%d")
    cutoff_str = (date.today() + timedelta(days=7)).strftime("%Y%m%d")

    def _is_monthly(d) -> bool:
        exp = d.contract.lastTradeDateOrContractMonth[:8]
        if exp < today_str:
            return False
        if (d.contract.localSymbol or "").strip().endswith(" D"):
            return False
        if exp < cutoff_str:
            return False
        return True

    live = [d for d in details if _is_monthly(d)]
    if not live:
        # Fallback: any non-expired, non-daily contract
        live = [d for d in details
                if d.contract.lastTradeDateOrContractMonth[:8] >= today_str
                and not (d.contract.localSymbol or "").strip().endswith(" D")]
    if not live:
        log.warning(f"  {instrument}: no active contracts found")
        return np.nan

    live.sort(key=lambda d: d.contract.lastTradeDateOrContractMonth)
    # Use conId to avoid IB re-evaluating the underspecified contract on hist data requests
    front = live[0].contract
    from ib_insync import Contract as _Contract
    contract = _Contract(conId=front.conId, exchange=front.exchange)

    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
            keepUpToDate=False,
        )
    except Exception as e:
        log.warning(f"  {instrument}: IB error — {e}")
        return np.nan

    time.sleep(0.4)

    if not bars:
        log.warning(f"  {instrument}: no bars returned")
        return np.nan

    df = util.df(bars)
    if df is None or df.empty or "volume" not in df.columns:
        log.warning(f"  {instrument}: no volume in response")
        return np.nan

    vol_series = df["volume"].replace(0, np.nan).dropna()
    if vol_series.empty:
        return np.nan

    # Use last LOOKBACK_DAYS trading days
    avg = float(vol_series.tail(lookback_days).mean())
    log.info(f"  {instrument}: avg {avg:.0f} contracts/day ({len(vol_series)} bars)")
    return avg


# ── Risk volume computation ────────────────────────────────────────────────────

def compute_volume_risk(
    pst: PSTLoader,
    instrument: str,
    avg_contracts: float,
) -> float:
    """
    Volume risk in $M/day = avg_contracts × mult × price × annual_vol / 16 / 1e6

    Uses last available price and blended vol from PST data.
    Returns NaN if price data unavailable.
    """
    if np.isnan(avg_contracts):
        return np.nan
    try:
        info = pst.instrument_info(instrument)
        mult = float(info["Pointsize"])
        raw = pst.multiple_prices(instrument)["PRICE"]
        raw = raw[raw.index <= PST_CUTOFF]
        adj = pst.adjusted_prices(instrument)
        adj = adj[adj.index <= PST_CUTOFF]
        ret = pct_returns_backadjusted(adj, raw)
        vol = blended_vol(ret).dropna()
        if vol.empty or raw.empty:
            return np.nan
        last_price = float(raw.dropna().iloc[-1])
        last_vol   = float(vol.iloc[-1])   # annualised fraction
        # daily σ in price units = price × annual_vol / 16
        daily_risk_per_contract = mult * last_price * last_vol / 16.0
        return avg_contracts * daily_risk_per_contract / 1e6
    except Exception:
        return np.nan


# ── Main update logic ─────────────────────────────────────────────────────────

def update_volume(
    instruments: list[str] | None = None,
    max_age_days: int = MAX_AGE_DAYS,
    port: int = IB_PORT,
) -> pd.DataFrame:
    """
    Connect to IB, fetch volume for stale/missing instruments, update cache.
    Returns the full updated cache DataFrame.
    """
    instruments = instruments or list(UNIVERSE.keys())
    cache = load_cache()
    to_fetch = stale_instruments(instruments, cache, max_age_days)

    if not to_fetch:
        log.info("All instruments up to date in volume cache.")
        return cache

    log.info(f"Fetching volume for {len(to_fetch)} instruments ...")
    ib_cfg = pd.read_csv(IB_CFG_PATH, index_col="Instrument")

    ib = IB()
    ib.connect(IB_HOST, port, clientId=IB_CLIENT)
    log.info("Connected to IB.")

    results = []
    for i, instr in enumerate(to_fetch):
        log.info(f"[{i+1}/{len(to_fetch)}] {instr}")
        avg_vol = fetch_volume(ib, ib_cfg, instr)
        results.append({
            "instrument":   instr,
            "avg_contracts": avg_vol,
            "fetch_date":   pd.Timestamp(date.today()),
        })

    ib.disconnect()
    log.info("Disconnected from IB.")

    new_rows = pd.DataFrame(results)

    # Merge: keep existing rows for non-fetched instruments, overwrite fetched ones
    cache = cache[~cache["instrument"].isin(to_fetch)]
    cache = pd.concat([cache, new_rows], ignore_index=True)
    save_cache(cache)
    log.info(f"Volume cache updated: {CACHE_PATH}")
    return cache


# ── Report ────────────────────────────────────────────────────────────────────

def volume_report(pst: PSTLoader, cache: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Return a DataFrame with volume stats and pass/fail for each UNIVERSE instrument.

    Columns: instrument, asset_class, avg_contracts, volume_risk_mday,
             pass_contracts, pass_risk, pass_volume
    """
    from ibkr_fut.instrument_selection import MIN_VOLUME_CONTRACTS, MIN_VOLUME_RISK

    cache = cache if cache is not None else load_cache()
    vol_map = dict(zip(cache["instrument"], cache["avg_contracts"]))

    rows = []
    for instr, cls in UNIVERSE.items():
        avg_c = vol_map.get(instr, np.nan)
        risk  = compute_volume_risk(pst, instr, avg_c)
        rows.append({
            "instrument":       instr,
            "asset_class":      cls,
            "avg_contracts":    avg_c,
            "volume_risk_mday": risk,
            "pass_contracts":   bool(avg_c >= MIN_VOLUME_CONTRACTS) if not np.isnan(avg_c) else None,
            "pass_risk":        bool(risk  >= MIN_VOLUME_RISK)       if not np.isnan(risk)  else None,
        })

    df = pd.DataFrame(rows).set_index("instrument")
    df["pass_volume"] = df["pass_contracts"] & df["pass_risk"]
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch IB volume for futures instruments")
    parser.add_argument("instruments", nargs="*", help="Specific instruments (default: all UNIVERSE)")
    parser.add_argument("--max-age", type=int, default=MAX_AGE_DAYS,
                        help=f"Refresh entries older than N days (default {MAX_AGE_DAYS})")
    parser.add_argument("--port", type=int, default=IB_PORT,
                        help=f"IB Gateway port (default {IB_PORT})")
    parser.add_argument("--report", action="store_true",
                        help="Print volume report from cache (no IB connection needed)")
    args = parser.parse_args()

    if args.report:
        pst = PSTLoader()
        cache = load_cache()
        report = volume_report(pst, cache)
        print(f"\n{'Instrument':<20} {'Class':<8} {'AvgCon':>8} {'Risk$M':>8} {'OkCon':>6} {'OkRisk':>7}")
        print("-" * 65)
        for instr, row in report.iterrows():
            avg_c = f"{row['avg_contracts']:8.0f}" if not np.isnan(row['avg_contracts']) else "     n/a"
            risk  = f"{row['volume_risk_mday']:8.2f}" if not np.isnan(row['volume_risk_mday']) else "     n/a"
            ok_c  = "YES" if row["pass_contracts"] else ("?" if row["pass_contracts"] is None else "NO")
            ok_r  = "YES" if row["pass_risk"]      else ("?" if row["pass_risk"] is None else "NO")
            print(f"{instr:<20} {row['asset_class']:<8} {avg_c} {risk} {ok_c:>6} {ok_r:>7}")
    else:
        target = args.instruments if args.instruments else None
        update_volume(instruments=target, max_age_days=args.max_age, port=args.port)
