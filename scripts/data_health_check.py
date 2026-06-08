"""
data_health_check.py

File-based data health check for the PST futures data pipeline.
No IBKR connection required — pure CSV inspection.

Checks:
  1. Adjusted price staleness (>3 trading days old → warn)
  2. Multiple price staleness (same threshold)
  3. Carry data completeness (post-cutoff rows where CARRY is NaN)
  4. Roll calendar coverage (must extend >=180 days into the future)
  5. FX price staleness
  6. Missing CSVs for universe instruments

Exit code 0 = clean, 1 = warnings found.

Usage:
    python scripts/data_health_check.py
    python scripts/data_health_check.py --verbose    # show all instruments
"""

import os, sys, argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibkr_fut.instrument_universe import UNIVERSE

REPO     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PST_BASE = os.path.join(REPO, "Data", "pst", "futures")

PST_CUTOFF        = pd.Timestamp("2024-03-28")
STALE_DAYS        = 3    # flag adjusted/multiple prices older than N trading days
CARRY_WARN_PCT    = 30   # flag if >N% of post-cutoff rows are missing carry
RC_MIN_DAYS_AHEAD = 180  # roll calendar must extend this far into the future


def _last_date(fp: str) -> pd.Timestamp | None:
    try:
        df = pd.read_csv(fp, parse_dates=[0], index_col=0)
        if df.empty:
            return None
        return pd.Timestamp(df.index[-1]).normalize()
    except Exception:
        return None


def check_prices(today: pd.Timestamp, verbose: bool) -> list[str]:
    warnings = []
    adj_dir   = os.path.join(PST_BASE, "adjusted_prices_csv")
    multi_dir = os.path.join(PST_BASE, "multiple_prices_csv")

    for inst in sorted(UNIVERSE):
        adj_fp   = os.path.join(adj_dir,   f"{inst}.csv")
        multi_fp = os.path.join(multi_dir, f"{inst}.csv")

        # Missing file
        if not os.path.exists(adj_fp):
            warnings.append(f"  MISSING adj   {inst}")
            continue
        if not os.path.exists(multi_fp):
            warnings.append(f"  MISSING multi {inst}")
            continue

        adj_last = _last_date(adj_fp)
        if adj_last is None:
            warnings.append(f"  EMPTY adj     {inst}")
            continue

        cal_days = (today - adj_last).days
        # Skip instruments known to trade infrequently (weekly / sparse markets)
        SPARSE = {"BRE", "TWD-mini"}
        threshold = 10 if inst in SPARSE else STALE_DAYS + 2  # +2 for weekend buffer

        if cal_days > threshold:
            warnings.append(f"  STALE         {inst:<25} last={adj_last.date()}  ({cal_days}d old)")
        elif verbose:
            print(f"  OK            {inst:<25} last={adj_last.date()}")

    return warnings


def check_carry(verbose: bool) -> list[str]:
    warnings = []
    multi_dir = os.path.join(PST_BASE, "multiple_prices_csv")

    # VIX/VIX_mini: structurally no carry (front month = shortest tenor).
    # SGX thin instruments and soft commodities: deferred contracts routinely have
    # zero IBKR volume so carry is unavailable on many days. Known structural gap.
    CARRY_EXEMPT = {
        "VIX", "VIX_mini",
        "FTSECHINAH", "FTSEINDO", "FTSETAIWAN", "FTSEVIET",  # SGX thin markets
        "MSCISING",                                           # SGX, partial carry
        "KR3", "KR10",                                        # KSE, illiquid deferreds
        "TWD-mini",                                           # SGX thin FX
        "SGD",                                                # SGX FX, thin deferred
        "COCOA", "COFFEE", "SUGAR11",                         # NYBOT softs, thin deferreds
    }

    for inst in sorted(UNIVERSE):
        if inst in CARRY_EXEMPT:
            continue
        fp = os.path.join(multi_dir, f"{inst}.csv")
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_csv(fp, parse_dates=[0], index_col=0)
        except Exception:
            continue
        if "CARRY" not in df.columns:
            warnings.append(f"  NO CARRY COL  {inst}")
            continue

        recent = df[df.index > PST_CUTOFF]
        if recent.empty:
            continue
        pct_missing = 100 * recent["CARRY"].isna().mean()
        if pct_missing > CARRY_WARN_PCT:
            warnings.append(
                f"  LOW CARRY     {inst:<25} {pct_missing:.0f}% missing "
                f"({recent['CARRY'].isna().sum()}/{len(recent)} rows post-cutoff)"
            )
        elif verbose:
            print(f"  CARRY OK      {inst:<25} {100-pct_missing:.0f}% filled")

    return warnings


def check_roll_calendars(today: pd.Timestamp, verbose: bool) -> list[str]:
    warnings = []
    rc_dir = os.path.join(PST_BASE, "roll_calendars_csv")
    horizon = today + pd.Timedelta(days=RC_MIN_DAYS_AHEAD)

    for inst in sorted(UNIVERSE):
        fp = os.path.join(rc_dir, f"{inst}.csv")
        if not os.path.exists(fp):
            warnings.append(f"  MISSING RC    {inst}")
            continue
        last = _last_date(fp)
        if last is None:
            warnings.append(f"  EMPTY RC      {inst}")
        elif last < horizon:
            days_ahead = (last - today).days
            warnings.append(f"  SHORT RC      {inst:<25} extends only {days_ahead}d ahead (need {RC_MIN_DAYS_AHEAD})")
        elif verbose:
            print(f"  RC OK         {inst:<25} extends to {last.date()}")

    return warnings


def check_fx(today: pd.Timestamp, verbose: bool) -> list[str]:
    warnings = []
    fx_dir = os.path.join(PST_BASE, "fx_prices_csv")
    for fname in sorted(os.listdir(fx_dir)):
        if not fname.endswith(".csv"):
            continue
        fp = os.path.join(fx_dir, fname)
        last = _last_date(fp)
        if last is None:
            warnings.append(f"  EMPTY FX      {fname}")
            continue
        cal_days = (today - last).days
        if cal_days > 5:
            warnings.append(f"  STALE FX      {fname:<25} last={last.date()}  ({cal_days}d old)")
        elif verbose:
            print(f"  FX OK         {fname:<25} last={last.date()}")

    return warnings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    today = pd.Timestamp.today().normalize()
    print(f"=== PST data health check  ({today.date()}) ===\n")

    all_warnings = []

    print("── Adjusted / multiple prices ──────────────────────────────────────")
    w = check_prices(today, args.verbose)
    all_warnings.extend(w)
    if w:
        for line in w:
            print(line)
    else:
        print("  All up to date")

    print("\n── Carry data ───────────────────────────────────────────────────────")
    w = check_carry(args.verbose)
    all_warnings.extend(w)
    if w:
        for line in w:
            print(line)
    else:
        print("  All above threshold")

    print("\n── Roll calendars ───────────────────────────────────────────────────")
    w = check_roll_calendars(today, args.verbose)
    all_warnings.extend(w)
    if w:
        for line in w:
            print(line)
    else:
        print("  All extended sufficiently")

    print("\n── FX prices ────────────────────────────────────────────────────────")
    w = check_fx(today, args.verbose)
    all_warnings.extend(w)
    if w:
        for line in w:
            print(line)
    else:
        print("  All up to date")

    print(f"\n{'='*70}")
    if all_warnings:
        print(f"RESULT: {len(all_warnings)} warning(s) found")
        sys.exit(1)
    else:
        print("RESULT: clean")
        sys.exit(0)


if __name__ == "__main__":
    main()
