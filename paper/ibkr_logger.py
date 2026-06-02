#!/usr/bin/env python3
"""
IBKR live portfolio ledger — IBS on MES / MNQ / MGC.

Ledger files in paper/ledgers/ibkr_port/:
  daily.csv      — daily account snapshot + rolling performance
  trades.csv     — per-fill: signal px, fill px, slippage, commission
  positions.csv  — end-of-run positions snapshot
  state.json     — {initial_equity, last_equity, last_date}

Integration in live_signals.py:
    from paper.ibkr_logger import IBKRLedger
    ledger = IBKRLedger()
    ledger.log_fill(...)      # call after each confirmed fill
    ledger.log_daily(ib)      # call once at end of the 3:58 PM run

Standalone report:
    python3 paper/ibkr_logger.py
"""

import csv
import json
import os
import sys
from datetime import datetime, date as date_t

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

HERE         = os.path.dirname(os.path.abspath(__file__))
LEDGER_ROOT  = os.path.join(HERE, "ledgers", "ibkr_port")
STATE_PATH   = os.path.join(LEDGER_ROOT, "state.json")
DAILY_PATH   = os.path.join(LEDGER_ROOT, "daily.csv")
TRADES_PATH  = os.path.join(LEDGER_ROOT, "trades.csv")
POSITIONS_PATH = os.path.join(LEDGER_ROOT, "positions.csv")

DAILY_COLS = [
    "date", "equity", "daily_pnl_usd", "ret",
    "nav", "n_trades", "commission", "slippage_usd",
]
TRADE_COLS = [
    "timestamp", "date", "symbol", "contract", "action", "qty",
    "ibs", "signal_price", "fill_price", "fill_value",
    "slippage_usd", "slippage_bps", "commission",
]
POSITION_COLS = [
    "date", "symbol", "contract", "qty",
    "avg_cost", "last_price", "market_value", "unrealized_pnl",
]

IB_HOST, IB_PORT, CLIENT_ID = "127.0.0.1", 4002, 11


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_state():
    if not os.path.exists(STATE_PATH):
        return {"initial_equity": None, "last_equity": None, "last_date": None}
    with open(STATE_PATH) as fh:
        return json.load(fh)

def _save_state(state):
    os.makedirs(LEDGER_ROOT, exist_ok=True)
    with open(STATE_PATH, "w") as fh:
        json.dump(state, fh, indent=2)

def _append_csv(path, cols, rows):
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if is_new:
            w.writeheader()
        for r in rows:
            w.writerow(r)

def _read_csv(path, cols):
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path)

def _return_metrics(rets):
    r = rets.dropna()
    if len(r) < 5:
        return {}
    nav  = (1 + r).cumprod()
    yrs  = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    dd   = (nav - nav.cummax()) / nav.cummax()
    vol  = r.std() * np.sqrt(252)
    return {
        "ann_ret":  cagr,
        "ann_vol":  vol,
        "sharpe":   r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0,
        "calmar":   cagr / abs(dd.min()) if dd.min() < 0 else 0,
        "max_dd":   dd.min(),
        "cur_dd":   dd.iloc[-1],
    }


# ── Public API ────────────────────────────────────────────────────────────────

class IBKRLedger:
    def __init__(self):
        os.makedirs(LEDGER_ROOT, exist_ok=True)
        self._fills_today = []   # accumulate intraday fills before log_daily

    def log_fill(
        self, *, symbol, contract, action, qty,
        ibs, signal_price, fill_price, commission,
    ):
        """
        Record one confirmed fill. Call immediately after ib.sleep() post-order.

        symbol       — 'MES' / 'MNQ' / 'MGC'
        contract     — contract month YYYYMM
        action       — 'BUY' or 'SELL'
        qty          — number of contracts filled
        ibs          — IBS value that triggered this trade (0–1)
        signal_price — bar close price used for sizing / IBS calc
        fill_price   — actual execution price (trade.orderStatus.avgFillPrice)
        commission   — commission paid (sum from fill.commissionReport, 0 if unknown)
        """
        multipliers = {"MES": 5, "MNQ": 2, "MGC": 10}
        mult        = multipliers.get(symbol, 1)
        fill_value  = fill_price * qty * mult
        sign        = 1 if action == "BUY" else -1
        slip_usd    = sign * (fill_price - signal_price) * qty * mult
        slip_bps    = (sign * (fill_price - signal_price) / signal_price * 10_000
                       if signal_price else 0)
        row = {
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date":         date_t.today().isoformat(),
            "symbol":       symbol,
            "contract":     contract,
            "action":       action,
            "qty":          qty,
            "ibs":          round(ibs, 4),
            "signal_price": round(signal_price, 4),
            "fill_price":   round(fill_price, 4),
            "fill_value":   round(fill_value, 2),
            "slippage_usd": round(slip_usd, 2),
            "slippage_bps": round(slip_bps, 2),
            "commission":   round(commission, 4),
        }
        _append_csv(TRADES_PATH, TRADE_COLS, [row])
        self._fills_today.append(row)

    def log_daily(self, ib):
        """
        Snapshot account equity + positions from a live IB connection.
        Idempotent — skips if today already has a daily entry.
        Call once per trading day, at the end of the last execution run (3:58 PM ET).
        """
        today = date_t.today().isoformat()

        # Idempotency check
        if os.path.exists(DAILY_PATH):
            tail = pd.read_csv(DAILY_PATH).tail(1)
            if not tail.empty and str(tail["date"].iloc[-1]) >= today:
                return

        # Read account values from IBKR — use broker-reported figures exactly
        equity    = None
        daily_pnl = None
        for v in ib.accountValues():
            if v.currency != "USD":
                continue
            if v.tag == "NetLiquidation":
                equity = float(v.value)
            elif v.tag == "DailyPnL":
                daily_pnl = float(v.value)
        if equity is None:
            return

        state       = _load_state()
        prev_equity = state.get("last_equity") or equity

        # DailyPnL from IBKR is exact (includes margin interest, dividends, etc.).
        # Fall back to equity diff only if the tag is unavailable (some accounts).
        if daily_pnl is None:
            daily_pnl = equity - prev_equity
        ret = daily_pnl / prev_equity if prev_equity else 0.0

        # Inception baseline: set once as the flat-account cash value (NLV minus
        # any unrealized P&L already sitting in open positions). This ensures
        # total return counts unrealized P&L from existing positions from day one,
        # not just P&L that accrues after logging begins.
        init_equity = state.get("initial_equity")
        if init_equity is None:
            open_unrealized = sum(
                item.unrealizedPNL for item in ib.portfolio()
                if item.contract.secType == "FUT"
            )
            init_equity = equity - open_unrealized

        # NAV: cumulative growth from inception
        if os.path.exists(DAILY_PATH) and os.path.getsize(DAILY_PATH) > 0:
            prev_nav = pd.read_csv(DAILY_PATH)["nav"].iloc[-1]
        else:
            prev_nav = 1.0
        nav = prev_nav * (1 + ret)

        # Aggregate today's fills
        n_trades   = len(self._fills_today)
        commission = sum(f["commission"]   for f in self._fills_today)
        slippage   = sum(f["slippage_usd"] for f in self._fills_today)

        _append_csv(DAILY_PATH, DAILY_COLS, [{
            "date":          today,
            "equity":        round(equity, 2),
            "daily_pnl_usd": round(daily_pnl, 2),
            "ret":           round(ret, 8),
            "nav":           round(nav, 6),
            "n_trades":      n_trades,
            "commission":    round(commission, 4),
            "slippage_usd":  round(slippage, 2),
        }])

        # Positions snapshot
        pos_rows = []
        for item in ib.portfolio():
            c = item.contract
            if c.secType != "FUT":
                continue
            pos_rows.append({
                "date":           today,
                "symbol":         c.symbol,
                "contract":       (c.lastTradeDateOrContractMonth or "")[:6],
                "qty":            int(item.position),
                "avg_cost":       round(item.averageCost, 4),
                "last_price":     round(item.marketPrice, 4),
                "market_value":   round(item.marketValue, 2),
                "unrealized_pnl": round(item.unrealizedPNL, 2),
            })
        if pos_rows:
            _append_csv(POSITIONS_PATH, POSITION_COLS, pos_rows)

        # Update state
        _save_state({
            "initial_equity": init_equity,
            "last_equity":    equity,
            "last_date":      today,
        })

        # Refresh the report file so it's always current
        report()


# ── Report ────────────────────────────────────────────────────────────────────

def _write_report(path, lines):
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def report():
    W    = 78
    REPORT_PATH = os.path.join(LEDGER_ROOT, "report.txt")
    os.makedirs(LEDGER_ROOT, exist_ok=True)

    _lines = []
    def _emit(s=""):
        print(s)
        _lines.append(s)

    def _hdr(title):
        _emit(f"\n  {title}")
        _emit("  " + "─" * (W - 2))

    def _pct(x, sign=True):
        if x != x: return "    n/a"
        s = "+" if sign and x >= 0 else ""
        return f"{s}{x*100:.2f}%"

    def _usd(x, sign=True):
        if x != x: return "       n/a"
        s = "+" if sign and x >= 0 else ""
        return f"{s}${x:,.2f}"

    _emit()
    _emit("=" * W)
    _emit(f"  IBKR LIVE PORTFOLIO   IBS / MES · MNQ · MGC")
    _emit(f"  as of {datetime.now().strftime('%Y-%m-%d  %H:%M')}")
    _emit("=" * W)

    # ── Daily history ─────────────────────────────────────────────────────────
    daily = _read_csv(DAILY_PATH, DAILY_COLS)
    if daily.empty:
        _emit("\n  No history yet — run live_signals.py --execute to start logging.\n")
        _emit("=" * W)
        _write_report(REPORT_PATH, _lines)
        return

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").drop_duplicates("date", keep="last")
    rets  = daily.set_index("date")["ret"].astype(float)

    state       = _load_state()
    init_equity = state.get("initial_equity") or daily["equity"].iloc[0]
    cur_equity  = float(daily["equity"].iloc[-1])
    cum_pnl_usd = cur_equity - init_equity
    cum_ret     = cum_pnl_usd / init_equity
    n_days      = len(daily)
    pos_df      = _read_csv(POSITIONS_PATH, POSITION_COLS)
    open_pos_pnl = None
    if not pos_df.empty:
        today_pos = pos_df[pos_df["date"] == pos_df["date"].max()]
        active_pos = today_pos[today_pos["qty"].astype(int) != 0]
        open_pos_pnl = float(active_pos["unrealized_pnl"].sum()) if not active_pos.empty else 0.0

    # ── Account  (equity + daily P&L exact from IBKR; total return computed) ──
    _hdr("Account  [equity & daily P&L = IBKR broker values]")
    daily_pnl = float(daily["daily_pnl_usd"].iloc[-1])
    daily_ret  = float(daily["ret"].iloc[-1])
    _emit(f"  {'Net Liquidation':<20} {_usd(cur_equity, sign=False):<18}"
          f"   Inception: {_usd(init_equity, sign=False)}")
    _emit(f"  {'Daily P&L':<20} {_usd(daily_pnl):<18}  ({_pct(daily_ret)})")
    if open_pos_pnl is not None:
        _emit(f"  {'Open Position P&L':<20} {_usd(open_pos_pnl):<18}"
              f"  [from current open contracts]")
    _emit(f"  {'Total Return':<20} {_usd(cum_pnl_usd):<18}  ({_pct(cum_ret)})   {n_days} days")

    # ── Performance  (computed from equity series) ────────────────────────────
    _hdr("Performance  [computed from daily equity history]")
    m = _return_metrics(rets)
    if m:
        _emit(f"  {'Ann. Return':<18} {_pct(m['ann_ret']):<12}"
              f"   {'Ann. Vol':<14} {_pct(m['ann_vol'], sign=False)}")
        _emit(f"  {'Sharpe':<18} {m['sharpe']:>+.2f}          "
              f"   {'Calmar':<14} {m['calmar']:>+.2f}")
        _emit(f"  {'Max Drawdown':<18} {_pct(m['max_dd']):<12}"
              f"   {'Current DD':<14} {_pct(m['cur_dd'])}")
    else:
        _emit(f"  (need ≥5 days of history for annualized stats)")

    # ── Costs  (from fill records) ────────────────────────────────────────────
    _hdr("Costs  (cumulative since inception)  [from fill records]")
    cum_comm = daily["commission"].sum()
    cum_slip = daily["slippage_usd"].sum()
    cum_cost = cum_comm + cum_slip
    n_total  = int(daily["n_trades"].sum())
    _emit(f"  {'Commission':<20} {_usd(cum_comm, sign=False):<18}"
          f"   Total Trades   {n_total}")
    _emit(f"  {'Slippage vs signal':<20} {_usd(cum_slip):<18}")
    _emit(f"  {'Total Cost':<20} {_usd(cum_cost):<18}"
          f"   Avg/Trade      {_usd(cum_cost / n_total, sign=False) if n_total else '  n/a'}")

    # ── Positions  (exact from IBKR portfolio snapshot) ───────────────────────
    if not pos_df.empty:
        _hdr("Current Positions  [exact from IBKR — last snapshot]")
        today_pos = pos_df[pos_df["date"] == pos_df["date"].max()]
        active    = today_pos[today_pos["qty"].astype(int) != 0]
        if active.empty:
            _emit("  flat — no open positions")
        else:
            _emit(f"  {'Symbol':<8} {'Contract':<10} {'Qty':>4}  "
                  f"{'Avg Cost':>10}  {'Last Px':>9}  {'Mkt Value':>12}  {'Unreal P&L':>12}")
            for _, r in active.iterrows():
                qty   = int(r["qty"])
                upnl  = float(r["unrealized_pnl"])
                sign  = "+" if upnl >= 0 else ""
                _emit(f"  {r['symbol']:<8} {str(r['contract']):<10} {qty:>+4}  "
                      f"{float(r['avg_cost']):>10,.2f}  {float(r['last_price']):>9,.2f}  "
                      f"${float(r['market_value']):>10,.0f}  {sign}${upnl:>9,.0f}")

    # ── Recent trades  (fill price & commission exact from IBKR) ─────────────
    trades_df = _read_csv(TRADES_PATH, TRADE_COLS)
    if not trades_df.empty:
        _hdr("Recent Trades  (last 10)  [fill px & commission = IBKR exact]")
        _emit(f"  {'Date':<12} {'Sym':<5} {'Act':<5} {'Qty':>3}  "
              f"{'IBS':>5}  {'Signal Px':>10}  {'Fill Px':>10}  "
              f"{'Slip $':>8}  {'Slip bps':>8}  {'Comm':>6}")
        for _, r in trades_df.tail(10).iloc[::-1].iterrows():
            slip = float(r["slippage_usd"])
            slip_sign = "+" if slip >= 0 else ""
            _emit(f"  {str(r['date']):<12} {r['symbol']:<5} {r['action']:<5} {int(r['qty']):>3}  "
                  f"{float(r['ibs']):>5.3f}  {float(r['signal_price']):>10,.2f}  "
                  f"{float(r['fill_price']):>10,.2f}  "
                  f"{slip_sign}${abs(slip):>6.2f}  {float(r['slippage_bps']):>+7.1f}  "
                  f"${float(r['commission']):>5.2f}")

    # ── Monthly returns  (computed from equity series) ────────────────────────
    _hdr("Monthly Returns  [computed]")
    monthly = rets.resample("ME").apply(lambda r: (1 + r).prod() - 1)
    if monthly.empty:
        _emit("  (not enough history)")
    else:
        line, n = "  ", 0
        for month, ret_m in monthly.items():
            label = month.strftime("%b %Y")
            line += f"  {label:<10} {_pct(ret_m):<10}"
            n += 1
            if n % 3 == 0:
                _emit(line); line = "  "
        if n % 3 != 0:
            _emit(line)

    _emit("\n" + "=" * W + "\n")
    _write_report(REPORT_PATH, _lines)


# ── Standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--live" in sys.argv:
        # Connect to IBKR and take a fresh snapshot, then report
        sys.path.insert(0, os.path.dirname(HERE))
        from ib_insync import IB
        ib = IB()
        try:
            ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, timeout=10)
        except Exception as e:
            print(f"Cannot connect to IBKR: {e}")
            sys.exit(1)
        ledger = IBKRLedger()
        ledger.log_daily(ib)
        ib.disconnect()
    report()
