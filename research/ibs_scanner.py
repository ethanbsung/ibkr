#!/usr/bin/env python3
"""
IBS Multi-Instrument Scanner
Runs IBS mean-reversion across every instrument with a daily data file
and ranks them by Sharpe ratio to find where the edge is strongest.

Returns are expressed in volatility-normalized units (daily change / rolling ATR)
so results are scale-invariant and comparable across instruments regardless of
price unit (cents vs dollars vs index points) or back-adjustment level.

Commission is expressed as a fixed fraction of ATR per round trip (0.10 ATR),
approximating realistic futures transaction costs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from io import StringIO

warnings.filterwarnings("ignore")

# ── Strategy parameters ────────────────────────────────────────────────────────
IBS_ENTRY   = 0.10   # enter long at close when IBS < this
IBS_EXIT    = 0.90   # exit long at close when IBS > this
START_DATE  = "2000-01-01"
END_DATE    = "2025-03-28"
DATA_DIR    = "Data"
MIN_TRADES  = 20     # skip instruments with too few trades
VOL_WINDOW  = 20     # rolling days for ATR / volatility estimate
COMM_ATR_RT = 0.10   # round-trip commission in units of ATR  (conservative)

# ── Instrument categories ──────────────────────────────────────────────────────
AGRICULTURAL = {"he", "gf", "le", "zc", "zs", "zw", "ke", "zo", "zl", "zm", "zr", "corn_mini"}
CURRENT_PORT = {"mes", "mnq", "mym", "mgc"}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_instruments():
    """Build instrument map from instruments.csv filtered to available data files."""
    path = os.path.join(DATA_DIR, "instruments.csv")
    df = pd.read_csv(path)
    instruments = {}
    for _, row in df.iterrows():
        sym = str(row["Symbol"]).strip().lower()
        file_path = os.path.join(DATA_DIR, f"{sym}_daily_data.csv")
        if os.path.exists(file_path):
            instruments[sym] = {
                "name": str(row["Name"]).strip(),
                "file": file_path,
            }
    return instruments


def load_data(file_path):
    """Load Barchart-format daily CSV, stripping Barchart footer and empty rows."""
    lines = []
    with open(file_path, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith('"'):
                continue
            lines.append(stripped)

    if len(lines) < 2:
        return pd.DataFrame()

    df = pd.read_csv(StringIO("\n".join(lines)), parse_dates=["Time"])
    required = {"High", "Low", "Last"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df.sort_values("Time", inplace=True)
    df = df[(df["Time"] >= START_DATE) & (df["Time"] <= END_DATE)].copy()
    df = df.dropna(subset=["High", "Low", "Last"]).reset_index(drop=True)
    return df


# ── Backtest engine ────────────────────────────────────────────────────────────

def run_ibs(data):
    """
    Single-contract IBS backtest using volatility-normalised daily returns.

    Returns are expressed in ATR units so results are comparable across instruments
    regardless of price scale, quoting convention, or back-adjustment level.

    Commission = COMM_ATR_RT ATR units (round trip) deducted per trade.
    Daily return when in position = daily_price_change / rolling_ATR.
    Daily return when flat = 0.
    """
    if len(data) < VOL_WINDOW + 10:
        return None, []

    df = data.copy()

    # IBS
    bar_range = df["High"] - df["Low"]
    df["IBS"] = np.where(bar_range > 0, (df["Last"] - df["Low"]) / bar_range, 0.5)

    # Daily price change (correct with additive back-adjustment)
    df["daily_chg"] = df["Last"].diff()

    # Rolling ATR as volatility normaliser
    df["atr"] = df["daily_chg"].abs().rolling(VOL_WINDOW).mean()
    df.dropna(subset=["atr"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    in_pos     = False
    entry_px   = 0.0
    entry_atr  = 1.0
    entry_time = None
    trades     = []
    daily_rets = []   # vol-normalised daily returns

    for _, row in df.iterrows():
        px  = row["Last"]
        ibs = row["IBS"]
        t   = row["Time"]
        chg = row["daily_chg"] if not pd.isna(row["daily_chg"]) else 0.0
        atr = row["atr"] if row["atr"] > 0 else 1.0

        if pd.isna(ibs) or pd.isna(px):
            daily_rets.append(0.0)
            continue

        if in_pos:
            daily_rets.append(chg / atr)   # vol-normalised return while holding
            if ibs > IBS_EXIT:
                raw_ret = (px - entry_px) / entry_atr   # ATR-units for trade P&L
                net_ret = raw_ret - COMM_ATR_RT
                trades.append({
                    "entry_time":  entry_time,
                    "exit_time":   t,
                    "entry_px":    entry_px,
                    "exit_px":     px,
                    "pnl_atr":     net_ret,
                    "win":         net_ret > 0,
                })
                in_pos = False
        else:
            daily_rets.append(0.0)
            if ibs < IBS_ENTRY:
                in_pos     = True
                entry_px   = px
                entry_atr  = atr
                entry_time = t
                # deduct entry commission from last return slot
                daily_rets[-1] = -COMM_ATR_RT / 2

    # Force-close open position at end
    if in_pos and len(df):
        last = df.iloc[-1]
        raw_ret = (last["Last"] - entry_px) / entry_atr
        net_ret = raw_ret - COMM_ATR_RT / 2   # half RT (exit only)
        trades.append({
            "entry_time":  entry_time,
            "exit_time":   last["Time"],
            "entry_px":    entry_px,
            "exit_px":     last["Last"],
            "pnl_atr":     net_ret,
            "win":         net_ret > 0,
        })

    rets = pd.Series(daily_rets, index=df["Time"].values, dtype=float)
    return rets, trades


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(rets, trades, data):
    """Derive performance statistics from vol-normalised return series."""
    if rets is None or len(trades) < MIN_TRADES:
        return None

    t0, t1 = rets.index[0], rets.index[-1]
    years   = (pd.Timestamp(t1) - pd.Timestamp(t0)).days / 365.25
    if years < 1.0:
        return None

    std = rets.std()
    sharpe = rets.mean() / std * np.sqrt(252) if std > 0 else 0.0

    # Equity curve in ATR-units cumulative sum (proxy for performance)
    eq   = rets.cumsum()
    peak = eq.cummax()
    dd   = eq - peak
    max_dd_atr = dd.min()   # in ATR units (negative)

    wins     = [t for t in trades if t["win"]]
    losses   = [t for t in trades if not t["win"]]
    wr       = len(wins) / len(trades) * 100
    gw       = sum(t["pnl_atr"] for t in wins)
    gl       = abs(sum(t["pnl_atr"] for t in losses))
    pf       = gw / gl if gl > 0 else float("nan")

    avg_hold = np.mean(
        [(pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])).days
         for t in trades]
    ) if trades else 0

    return {
        "sharpe":        sharpe,
        "max_dd_atr":    max_dd_atr,
        "win_rate":      wr,
        "profit_factor": pf,
        "n_trades":      len(trades),
        "trades_yr":     len(trades) / years,
        "avg_hold_days": avg_hold,
        "years":         years,
        "eq_series":     eq,   # for plotting
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    instruments = load_instruments()
    print(f"Found {len(instruments)} instruments with daily data files.\n")

    results     = {}
    eq_series   = {}

    for sym, spec in sorted(instruments.items()):
        data = load_data(spec["file"])
        if data.empty:
            continue

        rets, trades = run_ibs(data)
        metrics      = compute_metrics(rets, trades, data)

        if metrics:
            results[sym]   = {**metrics, "name": spec["name"]}
            eq_series[sym] = metrics.pop("eq_series")

    if not results:
        print("No results — check DATA_DIR.")
        return

    # ── Summary table ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results).T.sort_values("sharpe", ascending=False)

    hdr = (
        f"\n{'Rank':<5}{'Symbol':<12}{'Name':<30}"
        f"{'Sharpe':>8}{'MaxDD(σ)':>10}"
        f"{'WinRate%':>10}{'PF':>7}{'Trades':>8}{'Tr/Yr':>7}{'Hold(d)':>9}"
    )
    sep = "─" * len(hdr)
    print(sep)
    print(f"IBS STRATEGY — ALL INSTRUMENTS  (vol-normalised, comm={COMM_ATR_RT:.2f}×ATR RT)")
    print(sep)
    print(hdr)
    print(sep)

    for rank, (sym, row) in enumerate(df.iterrows(), 1):
        tag = ""
        if sym in AGRICULTURAL:
            tag = " [AG]"
        elif sym in CURRENT_PORT:
            tag = " [PORT]"
        pf_str = f"{row['profit_factor']:.2f}" if not np.isnan(row["profit_factor"]) else "  NaN"
        print(
            f"{rank:<5}{sym.upper():<12}{(row['name'] + tag):<30}"
            f"{row['sharpe']:>8.3f}{row['max_dd_atr']:>10.1f}"
            f"{row['win_rate']:>10.1f}{pf_str:>7}{int(row['n_trades']):>8}"
            f"{row['trades_yr']:>7.1f}{row['avg_hold_days']:>9.1f}"
        )

    # ── Agricultural vs current portfolio sub-table ────────────────────────────
    ag_df   = df[df.index.isin(AGRICULTURAL)]
    port_df = df[df.index.isin(CURRENT_PORT)]

    def mini_table(sub, label):
        print(f"\n  {label}")
        print(f"  {'Symbol':<10}{'Sharpe':>8}{'WinRate%':>10}{'PF':>7}{'Tr/Yr':>7}{'Hold(d)':>9}")
        for sym, row in sub.iterrows():
            pf_str = f"{row['profit_factor']:.2f}" if not np.isnan(row["profit_factor"]) else "  NaN"
            print(f"  {sym.upper():<10}{row['sharpe']:>8.3f}{row['win_rate']:>10.1f}"
                  f"{pf_str:>7}{row['trades_yr']:>7.1f}{row['avg_hold_days']:>9.1f}")

    print(f"\n{'─'*60}")
    print("AGRICULTURAL vs CURRENT PORTFOLIO")
    print(f"{'─'*60}")
    mini_table(ag_df,   "Agricultural futures  [AG]")
    mini_table(port_df, "Current portfolio     [PORT]")

    # ── Equity curves — top 20 by Sharpe ──────────────────────────────────────
    top20 = df.head(20).index.tolist()
    fig, axes = plt.subplots(4, 5, figsize=(20, 14), sharex=False)
    axes = axes.flatten()

    for ax, sym in zip(axes, top20):
        eq = eq_series[sym]
        color = "steelblue"
        if sym in AGRICULTURAL:
            color = "darkorange"
        elif sym in CURRENT_PORT:
            color = "forestgreen"

        ax.plot(pd.to_datetime(eq.index), eq.values, color=color, linewidth=0.9)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_title(
            f"{sym.upper()}  Sh={df.loc[sym,'sharpe']:.2f}  WR={df.loc[sym,'win_rate']:.0f}%",
            fontsize=8, fontweight="bold"
        )
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(top20):]:
        ax.set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="darkorange",  label="Agricultural [AG]"),
        Patch(facecolor="forestgreen", label="Current portfolio [PORT]"),
        Patch(facecolor="steelblue",   label="Other"),
    ]
    fig.legend(handles=legend_elements, loc="lower right", fontsize=9)
    fig.suptitle(
        f"IBS Strategy — Top 20 by Sharpe  "
        f"(IBS<{IBS_ENTRY} enter, IBS>{IBS_EXIT} exit | returns in ATR units | comm={COMM_ATR_RT}×ATR RT)",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("results/ibs_scanner.png", dpi=150)
    plt.show()
    print("\nChart saved to results/ibs_scanner.png")


if __name__ == "__main__":
    main()
