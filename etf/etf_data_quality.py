#!/usr/bin/env python3
"""
ETF daily data quality checker.
Audits Data/etf/*_1d_yf.csv files and produces a pass/fail report.

Checks per ETF:
  • coverage       — actual vs expected trading days, % present
  • history length — years of data available
  • liquidity      — average dollar volume (vol * close); flag if < $10M/day
  • OHLC sanity    — high < max(o,c), low > min(o,c), high < low
  • flat/stale     — longest run of identical closes
  • zero volume    — bars with no trading
  • extreme moves  — daily returns > 25% (flag for data error vs real event)
  • spike prints   — round-trip spikes (bad data)
  • bid-ask proxy  — daily range / close as spread proxy (high = illiquid)

Outputs console report + summary table + results/etf_data_quality.png

Usage:
  python3 etf/etf_data_quality.py
  python3 etf/etf_data_quality.py --min-years 5 --min-adv 10
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

EXTREME_RET_DAILY = 0.25   # 25% single-day move triggers flag
MIN_YEARS_DEFAULT = 3
MIN_ADV_DEFAULT   = 10e6   # $10M average daily dollar volume

OUTPUT_DIR = "results"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"]).set_index("time").sort_index()
    return df


def trading_days_expected(start, end) -> int:
    """Approximate expected trading days (252/year)."""
    return int(round((end - start).days * 252 / 365))


def audit(df: pd.DataFrame, ticker: str) -> dict:
    out = {"ticker": ticker}

    c = df["close"]
    o, h, l = df["open"], df["high"], df["low"]
    vol = df.get("volume", pd.Series(0, index=df.index))

    # ── History length ───────────────────────────────────────────────────────
    out["start"] = df.index[0]
    out["end"]   = df.index[-1]
    out["years"] = (df.index[-1] - df.index[0]).days / 365.25

    # ── Coverage ─────────────────────────────────────────────────────────────
    out["bars"]     = len(df)
    expected        = trading_days_expected(df.index[0], df.index[-1])
    out["expected"] = expected
    out["coverage_pct"] = 100.0 * len(df) / expected if expected else np.nan

    # Gaps: consecutive trading days more than 4 calendar days apart
    deltas = df.index.to_series().diff().dt.days
    gap_mask = deltas > 7  # > 1 week = likely gap, not just weekend/holiday
    gaps = deltas[gap_mask]
    out["n_gaps"] = int(len(gaps))

    # ── Liquidity ────────────────────────────────────────────────────────────
    dollar_vol = vol * c
    out["avg_dollar_vol"] = dollar_vol.mean()
    out["median_dollar_vol"] = dollar_vol.median()
    out["min_price"] = float(c.min())
    out["max_price"] = float(c.max())
    out["last_price"] = float(c.iloc[-1])

    # ── OHLC sanity ──────────────────────────────────────────────────────────
    # 0.05% tolerance absorbs floating-point rounding in yfinance adjusted prices.
    tol = c * 0.0005
    ohlc_bad = (h + tol < pd.concat([o, c], axis=1).max(axis=1)) | \
               (l - tol > pd.concat([o, c], axis=1).min(axis=1)) | (h + tol < l)
    out["ohlc_violations"] = int(ohlc_bad.sum())

    # ── Flat / stale / zero-volume ───────────────────────────────────────────
    out["zero_volume"] = int((vol == 0).sum())
    same = (c.diff() == 0)
    run, best = 0, 0
    for v in same.to_numpy():
        run = run + 1 if v else 0
        best = max(best, run)
    out["max_flat_run"] = int(best)

    # ── Returns ──────────────────────────────────────────────────────────────
    ret = c.pct_change()
    out["daily_vol_pct"] = float(ret.std() * 100)
    out["annual_vol_pct"] = float(ret.std() * np.sqrt(252) * 100)

    extreme = ret[ret.abs() > EXTREME_RET_DAILY]
    out["n_extreme"] = int(len(extreme))
    out["top_moves"] = sorted(
        [(ts, r) for ts, r in extreme.items()],
        key=lambda x: -abs(x[1]))[:5]

    # Round-trip spikes
    r = ret.to_numpy()
    spikes = 0
    for i in range(len(r) - 1):
        if np.isfinite(r[i]) and np.isfinite(r[i + 1]) and abs(r[i]) > EXTREME_RET_DAILY:
            if np.sign(r[i]) != np.sign(r[i + 1]) and abs(r[i + 1]) > 0.6 * abs(r[i]):
                spikes += 1
    out["spike_prints"] = spikes

    # ── Spread proxy (daily range as % of price) ─────────────────────────────
    out["avg_range_pct"] = float(((h - l) / c).mean() * 100)

    # ── _ret for plotting ────────────────────────────────────────────────────
    out["_ret"] = ret.dropna()
    out["_close"] = c

    return out


def pass_fail(a: dict, min_years: float, min_adv: float) -> list[str]:
    """Return list of failure reasons, empty if passes."""
    fails = []
    if a["years"] < min_years:
        fails.append(f"short history ({a['years']:.1f}y < {min_years}y)")
    if a["avg_dollar_vol"] < min_adv:
        fails.append(f"illiquid (ADV ${a['avg_dollar_vol']/1e6:.1f}M < ${min_adv/1e6:.0f}M)")
    if a["coverage_pct"] < 85:
        fails.append(f"low coverage ({a['coverage_pct']:.1f}%)")
    if a["spike_prints"] > 5:
        fails.append(f"spike prints ({a['spike_prints']})")
    if a["ohlc_violations"] > 20:
        fails.append(f"OHLC errors ({a['ohlc_violations']})")
    # Only flag stale data if the ETF has meaningful volatility; near-cash
    # instruments (T-bills) legitimately have many identical closes.
    if a["max_flat_run"] > 10 and a["annual_vol_pct"] > 1.0:
        fails.append(f"stale data (flat run {a['max_flat_run']})")
    return fails


def print_report(a: dict, fails: list[str]) -> None:
    status = "PASS" if not fails else "FAIL"
    print(f"\n{'─'*70}")
    print(f"  {a['ticker']:<8}  [{status}]  {a['start'].date()} – {a['end'].date()}"
          f"  ({a['years']:.1f}y)")
    print(f"  bars: {a['bars']:,} / {a['expected']:,}  coverage: {a['coverage_pct']:.1f}%"
          f"  gaps: {a['n_gaps']}")
    print(f"  ADV: ${a['avg_dollar_vol']/1e6:.1f}M   last price: ${a['last_price']:.2f}"
          f"   annual vol: {a['annual_vol_pct']:.1f}%")
    print(f"  range proxy: {a['avg_range_pct']:.2f}%   OHLC errors: {a['ohlc_violations']}"
          f"   spikes: {a['spike_prints']}   flat run: {a['max_flat_run']}")
    if fails:
        print(f"  FAILURES: {' | '.join(fails)}")
    if a["n_extreme"]:
        print(f"  extreme moves (>{int(EXTREME_RET_DAILY*100)}%): {a['n_extreme']}")
        for ts, r in a["top_moves"][:3]:
            print(f"        {ts.date()}  {r*100:+.1f}%")


def plot(results: dict, out_path: str) -> None:
    tickers = list(results.keys())
    n = len(tickers)
    if n == 0:
        return

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # ── Panel 1: Coverage timeline ───────────────────────────────────────────
    ax1 = axes[0]
    for y, tk in enumerate(tickers):
        a = results[tk]
        ax1.hlines(y, a["start"], a["end"], color="seagreen", lw=3, alpha=0.6)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(tickers, fontsize=6)
    ax1.set_title("Data coverage timeline (green = data)")
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.grid(True, axis="x", alpha=0.3)

    # ── Panel 2: Average dollar volume (log scale) ────────────────────────────
    ax2 = axes[1]
    advs = [results[tk]["avg_dollar_vol"] / 1e6 for tk in tickers]
    colors = ["crimson" if adv < MIN_ADV_DEFAULT / 1e6 else "steelblue" for adv in advs]
    ax2.barh(range(n), advs, color=colors, alpha=0.7)
    ax2.set_xscale("log")
    ax2.axvline(MIN_ADV_DEFAULT / 1e6, color="crimson", linestyle="--", alpha=0.5,
                label=f"${MIN_ADV_DEFAULT/1e6:.0f}M threshold")
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(tickers, fontsize=6)
    ax2.set_xlabel("Avg Daily Dollar Volume ($M)")
    ax2.set_title("Liquidity: Average Daily Dollar Volume (red = below $10M threshold)")
    ax2.legend()
    ax2.grid(True, axis="x", alpha=0.3)

    # ── Panel 3: Annual volatility ────────────────────────────────────────────
    ax3 = axes[2]
    vols = [results[tk]["annual_vol_pct"] for tk in tickers]
    ax3.barh(range(n), vols, color="darkorange", alpha=0.7)
    ax3.set_yticks(range(n))
    ax3.set_yticklabels(tickers, fontsize=6)
    ax3.set_xlabel("Annual Volatility (%)")
    ax3.set_title("Annualised Volatility")
    ax3.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\n  Chart saved to {out_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="ETF data quality audit")
    ap.add_argument("--glob", default="Data/etf/*_1d_yf.csv")
    ap.add_argument("--min-years", type=float, default=MIN_YEARS_DEFAULT,
                    help="Min years of history to pass (default: 3)")
    ap.add_argument("--min-adv", type=float, default=MIN_ADV_DEFAULT / 1e6,
                    help="Min avg daily dollar volume in $M (default: 10)")
    ap.add_argument("--out", default="results/etf_data_quality.png")
    args = ap.parse_args()

    min_adv = args.min_adv * 1e6

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"No files match {args.glob}")
        return

    results = {}
    for p in paths:
        ticker = os.path.basename(p).replace("_1d_yf.csv", "").upper()
        df = load(p)
        a = audit(df, ticker)
        results[ticker] = a
        fails = pass_fail(a, args.min_years, min_adv)
        print_report(a, fails)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print(f"  {'TICKER':<8}{'YEARS':>7}{'BARS':>8}{'COVER%':>8}{'ADV($M)':>10}"
          f"{'ANVOL%':>8}{'RANGE%':>8}{'GAPS':>6}{'SPIKES':>8}  STATUS")
    print("=" * 110)

    passed, failed = [], []
    for ticker, a in results.items():
        fails = pass_fail(a, args.min_years, min_adv)
        status = "PASS" if not fails else f"FAIL: {fails[0]}"
        adv = a["avg_dollar_vol"] / 1e6
        print(f"  {ticker:<8}{a['years']:>7.1f}{a['bars']:>8,}{a['coverage_pct']:>7.1f}%"
              f"{adv:>9.1f}M{a['annual_vol_pct']:>7.1f}%{a['avg_range_pct']:>7.2f}%"
              f"{a['n_gaps']:>6}{a['spike_prints']:>8}  {status}")
        if not fails:
            passed.append(ticker)
        else:
            failed.append(ticker)

    print("=" * 110)
    print(f"\n  PASSED: {len(passed)}")
    print(f"  FAILED: {len(failed)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")

    print(f"\n  PASSED tickers (usable for backtesting):")
    print(f"  {', '.join(passed)}")

    # Save passing ticker list for other scripts to consume
    universe_path = "Data/etf/etf_universe.json"
    universe = {
        "passed": passed,
        "failed": failed,
        "min_years": args.min_years,
        "min_adv_m": args.min_adv,
        "n_passed": len(passed),
        "n_failed": len(failed),
    }
    with open(universe_path, "w") as f:
        json.dump(universe, f, indent=2)
    print(f"\n  Universe saved to {universe_path}")

    plot(results, args.out)


if __name__ == "__main__":
    main()
