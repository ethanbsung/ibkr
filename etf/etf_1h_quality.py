#!/usr/bin/env python3
"""
1h ETF data quality check for Alpaca-sourced bars.

Checks:
  • Date range and total bars
  • RTH vs extended-hours bar counts (timestamps in UTC)
  • Coverage during RTH (expected ~7 bars/day: 14:00–20:00 UTC)
  • Gap analysis — consecutive RTH bars > 1h apart
  • OHLC sanity (with 0.05% tolerance for float rounding)
  • Zero-volume bars
  • Return distribution (RTH closes only)
  • Extreme moves & round-trip spike prints

Usage:
  python3 etf/etf_1h_quality.py
  python3 etf/etf_1h_quality.py --tickers SPY QQQ
  python3 etf/etf_1h_quality.py --glob "Data/etf/*_1h_alpaca.csv"
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Regular trading hours in UTC: 13:30–20:00 (9:30am–4pm ET, no DST adjustment)
# We use 14:00–20:00 inclusive as a conservative RTH window (first full bar opens at 14:00 UTC)
RTH_START_UTC = 14   # 10:00 AM ET (conservative — catches 9:30 open bar)
RTH_END_UTC   = 20   # 4:00 PM ET close bar

EXTREME_RET   = 0.05   # 5% single-hour move triggers flag for 1h data
OUTPUT_DIR    = "results"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"]).set_index("time").sort_index()
    return df


def rth_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: True for bars that fall within RTH window (UTC hours)."""
    h = df.index.hour
    return (h >= RTH_START_UTC) & (h < RTH_END_UTC)


def audit_1h(df: pd.DataFrame, ticker: str) -> dict:
    out = {"ticker": ticker}
    out["start"] = df.index[0]
    out["end"]   = df.index[-1]
    out["years"] = (df.index[-1] - df.index[0]).days / 365.25
    out["total_bars"] = len(df)

    # ── RTH / extended split ──────────────────────────────────────────────────
    rth = df[rth_mask(df)]
    out["rth_bars"] = len(rth)
    out["ext_bars"] = len(df) - len(rth)

    # ── RTH coverage ─────────────────────────────────────────────────────────
    # Expected: ~7 RTH bars/trading day, ~252 days/year
    expected_rth = int(out["years"] * 252 * (RTH_END_UTC - RTH_START_UTC))
    out["expected_rth"] = expected_rth
    out["rth_coverage_pct"] = 100.0 * len(rth) / expected_rth if expected_rth else np.nan

    # ── Gap analysis (RTH only, same trading session) ────────────────────────
    # Convert to US/Eastern to group by calendar date, then find gaps within a day.
    rth_et = rth.copy()
    rth_et.index = rth.index.tz_localize("UTC").tz_convert("US/Eastern")
    rth_date = rth_et.index.date
    same_day = pd.Series(rth_date, index=rth.index) == pd.Series(
        pd.Series(rth_date).shift(1).values, index=rth.index)
    deltas_all = rth.index.to_series().diff().dt.total_seconds() / 3600
    # Intraday gap: consecutive bars are same calendar date (ET) but > 1.5h apart
    gap_mask = same_day & (deltas_all > 1.5)
    gaps = deltas_all[gap_mask]
    out["intraday_gaps"] = int(len(gaps))
    out["missing_rth_bars"] = int((gaps - 1).round().sum()) if len(gaps) else 0
    out["top_gaps"] = sorted(
        [(ts, float(d)) for ts, d in gaps.items()],
        key=lambda x: -x[1])[:5]

    # ── OHLC sanity ───────────────────────────────────────────────────────────
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    tol = c * 0.0005
    ohlc_bad = (h + tol < pd.concat([o, c], axis=1).max(axis=1)) | \
               (l - tol > pd.concat([o, c], axis=1).min(axis=1)) | (h + tol < l)
    out["ohlc_violations"] = int(ohlc_bad.sum())

    # ── Zero volume ───────────────────────────────────────────────────────────
    vol = df.get("volume", pd.Series(0, index=df.index))
    out["zero_vol_total"] = int((vol == 0).sum())
    out["zero_vol_rth"]   = int((vol[rth_mask(df)] == 0).sum())

    # ── Returns (RTH only, skip first bar of each session) ────────────────────
    rth_close = rth["close"]
    ret = rth_close.pct_change()
    # Mask returns that span overnight (consecutive RTH bars > 1.5h apart after overnight)
    rth_deltas = rth_close.index.to_series().diff().dt.total_seconds() / 3600
    overnight_ret_mask = rth_deltas > 5   # first bar of session
    ret_clean = ret.mask(overnight_ret_mask)

    out["hourly_vol_pct"]  = float(ret_clean.std() * 100)
    out["annual_vol_pct"]  = float(ret_clean.std() * np.sqrt(252 * 7) * 100)

    extreme = ret_clean[ret_clean.abs() > EXTREME_RET]
    out["n_extreme"] = int(len(extreme))
    out["top_moves"] = sorted(
        [(ts, float(r)) for ts, r in extreme.items()],
        key=lambda x: -abs(x[1]))[:5]

    # Round-trip spikes (potential bad prints)
    r = ret_clean.to_numpy()
    spikes = 0
    for i in range(len(r) - 1):
        if np.isfinite(r[i]) and np.isfinite(r[i+1]) and abs(r[i]) > EXTREME_RET:
            if np.sign(r[i]) != np.sign(r[i+1]) and abs(r[i+1]) > 0.6 * abs(r[i]):
                spikes += 1
    out["spike_prints"] = spikes

    out["_ret_clean"] = ret_clean.dropna()
    out["_rth_close"] = rth_close

    return out


def print_report(a: dict) -> None:
    print(f"\n{'─'*70}")
    print(f"  {a['ticker']:<8}  {a['start'].date()} – {a['end'].date()}  ({a['years']:.1f}y)")
    print(f"  total bars: {a['total_bars']:,}  |  RTH: {a['rth_bars']:,}  ext-hours: {a['ext_bars']:,}")
    print(f"  RTH coverage: {a['rth_coverage_pct']:.1f}%  "
          f"({a['rth_bars']:,} / {a['expected_rth']:,} expected)")
    print(f"  intraday gaps: {a['intraday_gaps']}  (~{a['missing_rth_bars']} missing bars)")
    if a["top_gaps"]:
        top = ", ".join(f"{ts:%Y-%m-%d %H:%M}:{d:.1f}h" for ts, d in a["top_gaps"][:3])
        print(f"    largest: {top}")
    print(f"  OHLC violations: {a['ohlc_violations']}  |  "
          f"zero-vol total: {a['zero_vol_total']} (RTH: {a['zero_vol_rth']})")
    print(f"  hourly vol: {a['hourly_vol_pct']:.3f}%  |  "
          f"implied annual: {a['annual_vol_pct']:.1f}%")
    print(f"  extreme moves (>{int(EXTREME_RET*100)}%/hr): {a['n_extreme']}  "
          f"spike prints: {a['spike_prints']}")
    if a["top_moves"]:
        for ts, r in a["top_moves"][:5]:
            print(f"    {ts:%Y-%m-%d %H:%M UTC}  {r*100:+.2f}%")


def plot(results: dict, out_path: str) -> None:
    tickers = list(results.keys())
    n = len(tickers)
    if n == 0:
        return

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(7 * ncols, 5 * nrows * 2))
    axes = np.array(axes).reshape(nrows * 2, ncols)

    for idx, ticker in enumerate(tickers):
        a = results[ticker]
        row_base = (idx // ncols) * 2
        col = idx % ncols

        # Panel A: hourly close price (RTH)
        ax1 = axes[row_base, col]
        rth_close = a["_rth_close"]
        ax1.plot(rth_close.index, rth_close.values, lw=0.6, color="steelblue")
        ax1.set_title(f"{ticker}  RTH close price")
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax1.tick_params(axis="x", labelsize=7)
        ax1.grid(True, alpha=0.3)

        # Panel B: return distribution
        ax2 = axes[row_base + 1, col]
        ret = a["_ret_clean"].dropna()
        ax2.hist(ret.values, bins=300, color="darkorange", alpha=0.7)
        ax2.set_yscale("log")
        ax2.set_title(f"{ticker}  hourly return dist (log y)")
        ax2.set_xlabel("return"); ax2.grid(True, alpha=0.3)
        stats = f"σ={a['hourly_vol_pct']:.3f}%/hr  ann={a['annual_vol_pct']:.0f}%"
        ax2.text(0.02, 0.95, stats, transform=ax2.transAxes,
                 fontsize=8, va="top")

    # hide unused subplots
    for idx in range(n, nrows * ncols):
        for row_off in range(2):
            axes[(idx // ncols) * 2 + row_off, idx % ncols].set_visible(False)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\n  Chart saved to {out_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="1h ETF data quality check")
    ap.add_argument("--tickers", nargs="*", help="Specific tickers (e.g. SPY QQQ)")
    ap.add_argument("--glob", default=None, help="Glob pattern override")
    ap.add_argument("--out", default="results/etf_1h_quality.png")
    args = ap.parse_args()

    if args.glob:
        paths = sorted(glob.glob(args.glob))
    elif args.tickers:
        paths = [f"Data/etf/{t.lower()}_1h_alpaca.csv" for t in args.tickers]
    else:
        paths = sorted(glob.glob("Data/etf/*_1h_alpaca.csv"))

    if not paths:
        print("No files found.")
        return

    results = {}
    for p in paths:
        if not os.path.exists(p):
            print(f"  Missing: {p}")
            continue
        ticker = os.path.basename(p).replace("_1h_alpaca.csv", "").upper()
        df = load(p)
        a = audit_1h(df, ticker)
        results[ticker] = a
        print_report(a)

    plot(results, args.out)


if __name__ == "__main__":
    main()
