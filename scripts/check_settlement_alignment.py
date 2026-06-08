"""
check_settlement_alignment.py

Compares PST multiple_prices_csv PRICE column against IBKR reqHistoricalData
(whatToShow=TRADES, barSize=1 day) for all universe instruments over the past 3 weeks.

Prints a side-by-side table showing PST price vs IBKR close, and flags
rows where the discrepancy exceeds 0.1%.

Usage:
    python scripts/check_settlement_alignment.py
"""

import os, sys, time
import pandas as pd
from ib_insync import IB, Future, util

util.logToConsole("ERROR")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibkr_fut.instrument_universe import UNIVERSE

REPO     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PST_BASE = os.path.join(REPO, "Data", "pst", "futures", "multiple_prices_csv")
IB_CFG   = os.path.join(REPO, "Data", "pst", "ib_config", "ib_config_futures.csv")

IB_HOST   = "127.0.0.1"
IB_PORT   = 4002
IB_CLIENT = 77

INSTRUMENTS   = list(UNIVERSE.keys())
LOOKBACK_DAYS = 21
THRESHOLD_PCT = 0.10

# Instruments whose PST data is stale / discontinued — skip entirely
SKIP_INSTRUMENTS = {
    "BB3M",       # BSBY discontinued, data ends 2024-03-28
    "MSCIWORLD",  # data ends 2025-12-11, no active IBKR security def
    "MSCIASIA",   # data ends 2024-12-11, no active IBKR security def
}

# Per-instrument overrides for IBKR contract spec fields missing/wrong in ib_config.
# Verified by probing qualifyContracts against the live IB Gateway.
CONTRACT_OVERRIDES = {
    # EUREX equity indices — missing currency/tradingClass in ib_config
    "DAX":      {"tradingClass": "FDAX",  "currency": "EUR"},
    "EUROSTX":  {"tradingClass": "FESX",  "currency": "EUR"},
    "SMI":      {"tradingClass": "FSMI",  "currency": "CHF"},
    # AEX on FTA — no tradingClass in ib_config; currency must be EUR
    "AEX":      {"currency": "EUR"},
    # CAC40 — ib_config has MONEP which is correct for CAC futures; currency must be explicit
    "CAC":      {"exchange": "MONEP", "currency": "EUR"},
    # KSE KOSPI — currency missing from ib_config
    "KOSPI":    {"currency": "KRW"},
    # SILVER — specify tradingClass to pick standard 5000oz SI vs micro SIL
    "SILVER":   {"tradingClass": "SI"},
}


def contract_month(pst_contract: float) -> str:
    """Convert PST contract code (e.g. 20260600.0) → IBKR YYYYMM (e.g. 202606)."""
    return str(int(pst_contract))[:6]


def fetch_ibkr_closes(ib: IB, inst: str, ib_symbol: str, exchange: str,
                      currency: str, trading_class: str, hist_data_type: str,
                      contract_month_str: str):
    """
    Fetch daily settlement bars for a specific futures contract.
    Returns (pd.Series date→close, error_string_or_None).
    Applies CONTRACT_OVERRIDES, then falls back to includeExpired=True if needed.
    """
    overrides = CONTRACT_OVERRIDES.get(inst, {})
    exchange      = overrides.get("exchange",      exchange)
    currency      = overrides.get("currency",      currency)
    trading_class = overrides.get("tradingClass",  trading_class)

    def _build_contract(include_expired: bool) -> Future:
        c = Future(
            symbol=ib_symbol,
            lastTradeDateOrContractMonth=contract_month_str,
            exchange=exchange,
            currency=currency      if (currency      and pd.notna(currency))      else "",
            tradingClass=trading_class if (trading_class and pd.notna(trading_class)) else "",
        )
        c.includeExpired = include_expired
        return c

    # First try without includeExpired
    contract = _build_contract(False)
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        # Retry with includeExpired=True (handles expired front months)
        contract = _build_contract(True)
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            return pd.Series(dtype=float), "qualify failed (tried includeExpired)"

    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=f"{LOOKBACK_DAYS} D",
            barSizeSetting="1 day",
            whatToShow=hist_data_type,
            useRTH=False,   # match pst_updater which uses useRTH=False
            formatDate=1,
            keepUpToDate=False,
        )
    except Exception as e:
        return pd.Series(dtype=float), f"hist request failed: {e}"

    if not bars:
        return pd.Series(dtype=float), "no bars returned"

    df = util.df(bars)[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.set_index("date")["close"], None


def main():
    cfg = pd.read_csv(IB_CFG).set_index("Instrument")

    ib = IB()
    print(f"Connecting to IB Gateway {IB_HOST}:{IB_PORT} ...")
    ib.connect(IB_HOST, IB_PORT, IB_CLIENT)
    print("Connected.\n")

    results  = []
    skipped  = []
    errored  = []

    for inst in INSTRUMENTS:
        if inst in SKIP_INSTRUMENTS:
            skipped.append(inst)
            continue

        pst_path = os.path.join(PST_BASE, f"{inst}.csv")
        if not os.path.exists(pst_path):
            errored.append((inst, "PST file missing"))
            continue
        if inst not in cfg.index:
            errored.append((inst, "not in ib_config"))
            continue

        row          = cfg.loc[inst]
        ib_symbol    = row["IBSymbol"]
        exchange     = row["IBExchange"]
        currency     = row["IBCurrency"]     if pd.notna(row["IBCurrency"])     else ""
        trading_class= row["IBTradingClass"] if pd.notna(row["IBTradingClass"]) else ""
        raw_hd       = row["IBHistDataType"]
        hist_dtype   = "TRADES" if (pd.isna(raw_hd) or str(raw_hd).strip() == "") else str(raw_hd).strip()

        pst_df     = pd.read_csv(pst_path, index_col=0, parse_dates=True)
        pst_recent = pst_df.tail(LOOKBACK_DAYS).copy()
        if pst_recent.empty or "PRICE" not in pst_recent.columns:
            errored.append((inst, "no PRICE column"))
            continue

        # Use the current front contract (last row) so we always compare the live
        # contract — avoids picking up an expired contract via mode()
        last_contract = pst_recent["PRICE_CONTRACT"].dropna()
        if last_contract.empty:
            errored.append((inst, "no PRICE_CONTRACT"))
            continue
        pst_contract = last_contract.iloc[-1]
        cm_str       = contract_month(pst_contract)

        ibkr_closes, err = fetch_ibkr_closes(
            ib, inst, ib_symbol, exchange, currency, trading_class, hist_dtype, cm_str
        )
        time.sleep(0.4)

        if err:
            errored.append((inst, f"{ib_symbol} {cm_str}: {err}"))
            continue

        mask       = pst_recent["PRICE_CONTRACT"] == pst_contract
        pst_series = pst_recent[mask]["PRICE"].dropna()
        pst_series.index = pst_series.index.normalize()

        aligned = pd.DataFrame({"pst": pst_series, "ibkr": ibkr_closes}).dropna()
        if aligned.empty:
            errored.append((inst, "no overlapping dates"))
            continue

        aligned["diff_pct"] = (aligned["pst"] - aligned["ibkr"]).abs() / aligned["ibkr"] * 100
        aligned["flag"]     = aligned["diff_pct"] > THRESHOLD_PCT

        for date_idx, row_a in aligned.iterrows():
            results.append({
                "instrument": inst,
                "ib_symbol":  f"{ib_symbol} {cm_str}",
                "date":       date_idx.date(),
                "pst_price":  row_a["pst"],
                "ibkr_price": row_a["ibkr"],
                "diff_pct":   row_a["diff_pct"],
                "flag":       row_a["flag"],
            })

        flagged = aligned["flag"].sum()
        print(f"  {inst:<20} ({ib_symbol} {cm_str})  {len(aligned)} rows, {flagged} flagged")

    ib.disconnect()

    # ── Errors ────────────────────────────────────────────────────────────────
    if errored:
        print("\nCould not verify:")
        for inst, reason in errored:
            print(f"  {inst}: {reason}")

    if skipped:
        print(f"\nSkipped (stale/discontinued): {', '.join(skipped)}")

    if not results:
        print("\nNo results to display.")
        return

    res_df = pd.DataFrame(results).sort_values(["instrument", "date"])

    print("\n" + "=" * 90)
    print(f"{'Instrument':<20} {'Symbol':<14} {'Date':<12} {'PST':>10} {'IBKR':>10} {'Diff%':>7}  Flag")
    print("=" * 90)
    for _, r in res_df.iterrows():
        flag_str = " *** MISMATCH" if r["flag"] else ""
        print(f"{r['instrument']:<20} {r['ib_symbol']:<14} {str(r['date']):<12} "
              f"{r['pst_price']:>10.4f} {r['ibkr_price']:>10.4f} {r['diff_pct']:>6.2f}%{flag_str}")

    flagged_df = res_df[res_df["flag"]]
    print("=" * 90)
    print(f"\nSummary: {len(res_df)} rows compared across {res_df['instrument'].nunique()} instruments")
    print(f"         {len(flagged_df)} rows with discrepancy > {THRESHOLD_PCT}%")
    if not flagged_df.empty:
        print("\nMismatched rows:")
        for _, r in flagged_df.iterrows():
            print(f"  {r['instrument']} {r['date']}: PST={r['pst_price']:.4f}  "
                  f"IBKR={r['ibkr_price']:.4f}  diff={r['diff_pct']:.3f}%")


if __name__ == "__main__":
    main()
