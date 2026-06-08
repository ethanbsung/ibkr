"""
SQLite ledger — source of truth for positions, P&L, fills, and execution quality.

Alpaca tracks cash flows but does not account for:
  - Expense ratio drag (accrued daily from held positions)
  - Implementation shortfall vs signal price
  - Spread cost captured on each trade

This ledger does.

Tables
------
signals      — target position at signal generation time
fills        — every executed order with bid/ask and slippage metrics
positions    — end-of-day position snapshot per instrument
daily_pnl    — portfolio-level daily summary
"""

import os
import sqlite3
from datetime import date, datetime
from typing import Optional


DB_DEFAULT = "Data/live/trading.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    date          TEXT    NOT NULL,
    strategy      TEXT    NOT NULL,
    ticker        TEXT    NOT NULL,
    forecast      REAL,
    annual_vol    REAL,
    weight        REAL,
    idm           REAL,
    target_usd    REAL,
    signal_price  REAL,
    asset_class   TEXT
);

CREATE TABLE IF NOT EXISTS fills (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp                TEXT    NOT NULL,
    date                     TEXT    NOT NULL,
    strategy                 TEXT    NOT NULL,
    ticker                   TEXT    NOT NULL,
    side                     TEXT    NOT NULL,
    target_shares            REAL,
    filled_shares            REAL,
    fill_price               REAL,
    fill_value               REAL,
    signal_price             REAL,
    bid                      REAL,
    ask                      REAL,
    mid                      REAL,
    spread_bps               REAL,
    slippage_vs_signal_bps   REAL,
    slippage_vs_mid_bps      REAL,
    order_id                 TEXT,
    order_status             TEXT,
    is_dry_run               INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS positions (
    date            TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    shares          REAL,
    close_price     REAL,
    market_value    REAL,
    cost_basis      REAL,
    unrealized_pnl  REAL,
    daily_pnl       REAL,
    PRIMARY KEY (date, strategy, ticker)
);

CREATE TABLE IF NOT EXISTS runs (
    date          TEXT NOT NULL,
    strategy      TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    mode          TEXT NOT NULL,   -- 'live' or 'dry_run'
    n_orders      INTEGER,
    status        TEXT,            -- 'completed' / 'aborted'
    PRIMARY KEY (date, strategy, mode)
);

CREATE TABLE IF NOT EXISTS daily_pnl (
    date                  TEXT NOT NULL,
    strategy              TEXT NOT NULL,
    gross_pnl             REAL,
    spread_cost           REAL,
    slippage_cost         REAL,
    margin_debit_interest REAL DEFAULT 0,
    htb_cost              REAL DEFAULT 0,
    portfolio_value       REAL,
    alpaca_equity         REAL,
    n_trades              INTEGER,
    total_volume_usd      REAL,
    turnover_pct          REAL,
    avg_slippage_bps      REAL,
    realized_vol          REAL,
    PRIMARY KEY (date, strategy)
);
"""


class Ledger:
    def __init__(self, db_path: str = DB_DEFAULT):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after the initial schema (idempotent)."""
        adds = {
            "daily_pnl": {
                "nav_return": "REAL",   # authoritative daily return from equity
                "equity_pnl": "REAL",   # authoritative daily P&L ($) from equity
            },
        }
        with self._conn() as conn:
            for table, cols in adds.items():
                have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
                for col, typ in cols.items():
                    if col not in have:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")

    # ── Run idempotency ───────────────────────────────────────────────────────

    def has_run(self, strategy: str, trade_date: str, mode: str = "live") -> bool:
        """True if a completed run of this mode already exists for the date."""
        with self._conn() as conn:
            row = conn.execute("""
                SELECT status FROM runs
                WHERE date=? AND strategy=? AND mode=?
            """, (trade_date, strategy, mode)).fetchone()
        return bool(row and row["status"] == "completed")

    def mark_run(self, strategy: str, trade_date: str, mode: str,
                 n_orders: int, status: str = "completed") -> None:
        """Record that a run happened, for duplicate-run protection."""
        ts = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO runs
                  (date, strategy, timestamp, mode, n_orders, status)
                VALUES (?,?,?,?,?,?)
            """, (trade_date, strategy, ts, mode, n_orders, status))

    # ── Signal recording ──────────────────────────────────────────────────────

    def record_signals(self, strategy: str, metadata: dict[str, dict],
                       prices: dict[str, float], trade_date: Optional[str] = None) -> None:
        """
        Log the per-instrument signal metadata generated today.
        prices: {ticker: closing_price} at signal generation time.
        """
        ts   = datetime.utcnow().isoformat()
        dt   = trade_date or date.today().isoformat()
        rows = []
        for tk, m in metadata.items():
            rows.append((
                ts, dt, strategy, tk,
                m.get("forecast"),
                m.get("annual_vol"),
                m.get("weight"),
                m.get("idm"),
                m.get("target_usd"),
                prices.get(tk),
                m.get("asset_class"),
            ))
        with self._conn() as conn:
            conn.executemany("""
                INSERT INTO signals
                  (timestamp, date, strategy, ticker, forecast, annual_vol,
                   weight, idm, target_usd, signal_price, asset_class)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, rows)

    # ── Fill recording ────────────────────────────────────────────────────────

    def record_fill(self, fill: dict) -> None:
        """
        Record a single fill.  fill dict keys:
          strategy, ticker, side, target_shares, filled_shares,
          fill_price, signal_price, bid, ask, order_id, order_status, is_dry_run
        """
        mid        = (fill["bid"] + fill["ask"]) / 2 if fill["bid"] and fill["ask"] else None
        spread_bps = ((fill["ask"] - fill["bid"]) / mid * 10_000
                      if mid and mid > 0 else None)
        side_sign  = 1 if fill["side"] == "buy" else -1

        fp = fill.get("fill_price")   # None for canceled/rejected orders
        slippage_vs_signal = (
            (fp - fill["signal_price"]) / fill["signal_price"] * 10_000 * side_sign
            if fp is not None and fill.get("signal_price") else None
        )
        slippage_vs_mid = (
            (fp - mid) / mid * 10_000 * side_sign
            if fp is not None and mid and mid > 0 else None
        )

        ts = datetime.utcnow().isoformat()
        dt = fill.get("date", date.today().isoformat())

        with self._conn() as conn:
            conn.execute("""
                INSERT INTO fills
                  (timestamp, date, strategy, ticker, side,
                   target_shares, filled_shares, fill_price, fill_value,
                   signal_price, bid, ask, mid, spread_bps,
                   slippage_vs_signal_bps, slippage_vs_mid_bps,
                   order_id, order_status, is_dry_run)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                ts, dt, fill["strategy"], fill["ticker"], fill["side"],
                fill.get("target_shares"), fill.get("filled_shares"),
                fill.get("fill_price"), fill.get("fill_value"),
                fill.get("signal_price"), fill.get("bid"), fill.get("ask"),
                mid, spread_bps,
                slippage_vs_signal, slippage_vs_mid,
                fill.get("order_id"), fill.get("order_status"),
                int(fill.get("is_dry_run", False)),
            ))

    # ── Position snapshot ─────────────────────────────────────────────────────

    def snapshot_positions(self, strategy: str,
                           positions_detail: dict[str, dict],
                           close_prices: dict[str, float],
                           trade_date: Optional[str] = None) -> None:
        """
        Record EOD position snapshot.

        positions_detail: {ticker: {shares, avg_entry_price, market_value}} from
        Alpaca (see Executor.get_positions_detail).  avg_entry_price is Alpaca's
        authoritative cost basis, so unrealized P&L is correct even after adds /
        partial closes.

        Stored per position:
          cost_basis     = avg_entry_price * shares       (signed $ cost)
          unrealized_pnl = mkt_val - cost_basis           = (px - avg)*shares
          daily_pnl      = prev_shares * (px - prev_px)    (price-only MTM diag;
                           isolates price moves from today's trades — the
                           authoritative portfolio P&L is equity-based in
                           record_daily_pnl, this column is diagnostic only)
        """
        dt = trade_date or date.today().isoformat()

        # Fetch yesterday's positions for the price-change diagnostic
        yesterday = self._get_last_positions(strategy, before_date=dt)

        rows = []
        for tk, info in positions_detail.items():
            shares = info.get("shares", 0.0)
            if shares == 0.0:
                continue
            px = close_prices.get(tk, 0.0)
            if px == 0:
                # No live price for a held name: fall back to Alpaca market value
                # so the position is still recorded (don't silently drop it).
                mv = info.get("market_value", 0.0)
                px = (mv / shares) if shares else 0.0
            if px == 0:
                continue

            avg_entry = info.get("avg_entry_price", 0.0) or px
            mkt_val   = shares * px
            cost_basis = avg_entry * shares
            unrealized = mkt_val - cost_basis

            prev      = yesterday.get(tk, {})
            prev_px   = prev.get("close_price")
            prev_sh   = prev.get("shares", 0.0)
            daily_pnl = prev_sh * (px - prev_px) if prev and prev_px else 0.0

            rows.append((
                dt, strategy, tk,
                shares, px, mkt_val,
                cost_basis, unrealized, daily_pnl,
            ))

        with self._conn() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO positions
                  (date, strategy, ticker, shares, close_price,
                   market_value, cost_basis, unrealized_pnl, daily_pnl)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, rows)

    # ── Daily P&L summary ─────────────────────────────────────────────────────

    def record_daily_pnl(self, strategy: str, alpaca_equity: Optional[float],
                         fills_today: list[dict],
                         trade_date: Optional[str] = None,
                         margin_debit_interest: float = 0.0,
                         htb_cost: float = 0.0) -> None:
        """
        Compute and store the daily portfolio-level P&L summary.

        P&L accounting:
          nav_return / equity_pnl = AUTHORITATIVE — derived from the change in
                          Alpaca account equity (reflects actual fills, spread,
                          and slippage already baked in).  This is the source of
                          truth for returns, realized vol, and the paper/ book.
          gross_pnl   = SUM of position-level price-only MTM — DIAGNOSTIC only;
                          does not include intraday trade P&L or missing names.
          alpaca_equity = end-of-day NAV from Alpaca.
          spread_cost, avg_slippage_bps = execution diagnostics (informational).
          margin_debit_interest, htb_cost = currently 0; populated if relevant.
        """
        dt = trade_date or date.today().isoformat()

        with self._conn() as conn:
            row = conn.execute("""
                SELECT SUM(daily_pnl)    AS gross_pnl,
                       SUM(market_value) AS port_value
                FROM positions
                WHERE date=? AND strategy=?
            """, (dt, strategy)).fetchone()

            gross_pnl  = row["gross_pnl"]  or 0.0
            port_value = row["port_value"] or 0.0

        # Spread and slippage recorded for diagnostics, not deducted from P&L
        # (Alpaca paper fills already reflect execution at market bid/ask)
        spread_cost = sum(
            abs(f.get("fill_value", 0)) * (f.get("spread_bps", 0) or 0) / 2 / 10_000
            for f in fills_today
        )

        n_trades     = len(fills_today)
        total_vol    = sum(abs(f.get("fill_value", 0)) for f in fills_today)
        turnover_pct = (total_vol / port_value * 100) if port_value > 0 else 0.0

        slippages = [f.get("slippage_vs_signal_bps") for f in fills_today
                     if f.get("slippage_vs_signal_bps") is not None]
        avg_slip  = sum(slippages) / len(slippages) if slippages else None

        # ── Authoritative return/P&L from the change in Alpaca equity ─────────
        prev_equity = self._last_equity(strategy, before_date=dt)
        if alpaca_equity and prev_equity and prev_equity > 0:
            nav_return = alpaca_equity / prev_equity - 1.0
            equity_pnl = alpaca_equity - prev_equity
        else:
            nav_return = 0.0          # first day (or no prior equity): flat
            equity_pnl = 0.0

        # Realized vol: annualized std of the authoritative daily returns.
        with self._conn() as conn:
            hist = conn.execute("""
                SELECT nav_return FROM daily_pnl
                WHERE strategy=? AND date < ? AND nav_return IS NOT NULL
                ORDER BY date DESC LIMIT 19
            """, (strategy, dt)).fetchall()
        rets = [nav_return] + [r["nav_return"] for r in hist]
        import numpy as np
        realized_vol = float(np.std(rets) * (252 ** 0.5)) if len(rets) >= 5 else None

        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO daily_pnl
                  (date, strategy, gross_pnl, spread_cost, slippage_cost,
                   margin_debit_interest, htb_cost,
                   portfolio_value, alpaca_equity, n_trades, total_volume_usd,
                   turnover_pct, avg_slippage_bps, realized_vol,
                   nav_return, equity_pnl)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                dt, strategy, gross_pnl, spread_cost, None,
                margin_debit_interest, htb_cost,
                port_value, alpaca_equity, n_trades, total_vol,
                turnover_pct, avg_slip, realized_vol,
                nav_return, equity_pnl,
            ))

    # ── Reporting ─────────────────────────────────────────────────────────────

    def report(self, strategy: str = "ewmac", n_days: int = 30) -> str:
        """Return a text summary of recent performance."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM daily_pnl
                WHERE strategy=?
                ORDER BY date DESC LIMIT ?
            """, (strategy, n_days)).fetchall()

            pos = conn.execute("""
                SELECT p.ticker, p.shares, p.close_price, p.market_value, p.daily_pnl
                FROM positions p
                WHERE p.strategy=?
                  AND p.date = (SELECT MAX(date) FROM positions WHERE strategy=?)
                ORDER BY p.market_value DESC
            """, (strategy, strategy)).fetchall()

            fills = conn.execute("""
                SELECT ticker, side, filled_shares, fill_price,
                       spread_bps, slippage_vs_signal_bps, slippage_vs_mid_bps,
                       is_dry_run, date
                FROM fills
                WHERE strategy=?
                ORDER BY timestamp DESC LIMIT 30
            """, (strategy,)).fetchall()

        lines = [
            f"\n{'='*68}",
            f"  LIVE LEDGER  —  {strategy.upper()}  (last {n_days} days)",
            f"{'='*68}",
        ]

        if rows:
            equity_pnl = sum((r["equity_pnl"] if "equity_pnl" in r.keys() else 0) or 0
                             for r in rows)
            gross = sum(r["gross_pnl"]   or 0 for r in rows)
            spr   = sum(r["spread_cost"] or 0 for r in rows)
            lines += [
                f"  P&L summary (last {len(rows)} days):",
                f"    Equity P&L          ${equity_pnl:>+10,.2f}  [authoritative]",
                f"    MTM P&L (diag)      ${gross:>+10,.2f}",
                f"    Spread cost (diag)  ${spr:>10,.2f}",
            ]
            rv = rows[0]["realized_vol"]
            if rv is not None:
                lines.append(f"    Realized vol        {rv*100:>9.1f}%  (annualized, from equity)")
            slips = [r["avg_slippage_bps"] for r in rows if r["avg_slippage_bps"]]
            if slips:
                lines.append(f"    Avg slippage  {sum(slips)/len(slips):+.1f} bps vs signal")
            if rows[0]["alpaca_equity"]:
                lines.append(f"    Alpaca equity (NAV)  ${rows[0]['alpaca_equity']:,.2f}  "
                              f"[authoritative]")

        if pos:
            lines += ["\n  Current positions (top 15 by value):"]
            for r in list(pos)[:15]:
                lines.append(
                    f"    {r['ticker']:<6}  {r['shares']:>10.4f} sh"
                    f"  @ ${r['close_price']:>8.2f}"
                    f"  = ${r['market_value']:>10,.2f}"
                    f"  daily P&L ${r['daily_pnl']:>+8,.2f}"
                )

        if fills:
            lines += ["\n  Recent fills:"]
            for r in list(fills)[:10]:
                dry = " [DRY]" if r["is_dry_run"] else ""
                slip = f"  slip {r['slippage_vs_signal_bps']:+.1f} bps" if r["slippage_vs_signal_bps"] is not None else ""
                spr  = f"  spread {r['spread_bps']:.1f} bps" if r["spread_bps"] is not None else ""
                lines.append(
                    f"    {r['date']}  {r['ticker']:<6}  {r['side']:<4}"
                    f"  {r['filled_shares']:>9.4f} sh @ ${r['fill_price']:>8.2f}"
                    f"{slip}{spr}{dry}"
                )

        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _last_equity(self, strategy: str, before_date: str) -> Optional[float]:
        """Most recent Alpaca equity recorded before `before_date`."""
        with self._conn() as conn:
            row = conn.execute("""
                SELECT alpaca_equity FROM daily_pnl
                WHERE strategy=? AND date < ? AND alpaca_equity IS NOT NULL
                ORDER BY date DESC LIMIT 1
            """, (strategy, before_date)).fetchone()
        return row["alpaca_equity"] if row else None

    def export_paper_book(self, strategy: str, paper_root: str,
                          book_name: Optional[str] = None) -> str:
        """
        Project the SQLite daily history into a paper/-format return/NAV book at
        paper_root/<book_name>/ so paper/run_paper.py can fold this strategy into
        the combined-portfolio report.  Rebuilt from scratch each call (the book
        is a pure projection of the authoritative nav_return series — idempotent,
        no duplicate rows).  Add the book to paper/portfolio.json to allocate it.
        """
        import csv
        import json as _json

        book    = book_name or f"etf_{strategy}"
        out_dir = os.path.join(paper_root, book)
        os.makedirs(out_dir, exist_ok=True)

        with self._conn() as conn:
            daily = conn.execute("""
                SELECT date, nav_return, total_volume_usd, spread_cost, alpaca_equity
                FROM daily_pnl WHERE strategy=? ORDER BY date ASC
            """, (strategy,)).fetchall()
            posagg = conn.execute("""
                SELECT date, SUM(ABS(market_value)) AS gross_mv, COUNT(*) AS n_pos
                FROM positions WHERE strategy=? GROUP BY date
            """, (strategy,)).fetchall()
        gross_by = {r["date"]: (r["gross_mv"] or 0.0, r["n_pos"] or 0) for r in posagg}

        rows, nav = [], 1.0
        for r in daily:
            ret = r["nav_return"] or 0.0
            nav *= (1.0 + ret)
            eq  = r["alpaca_equity"] or 0.0
            gmv, npos = gross_by.get(r["date"], (0.0, 0))
            rows.append({
                "date":           r["date"],
                "ret":            ret,
                "nav":            nav,
                "gross_exposure": (gmv / eq) if eq else 0.0,
                "turnover":       ((r["total_volume_usd"] or 0.0) / eq) if eq else 0.0,
                "cost":           ((r["spread_cost"] or 0.0) / eq) if eq else 0.0,
                "n_positions":    npos,
            })

        cols = ["date", "ret", "nav", "gross_exposure", "turnover", "cost", "n_positions"]
        with open(os.path.join(out_dir, "daily.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

        # state.json — resume point + current weights/prices (matches paper Book)
        weights, prices = {}, {}
        last_eq = daily[-1]["alpaca_equity"] if daily else 0.0
        with self._conn() as conn:
            latest = conn.execute("""
                SELECT ticker, market_value, close_price FROM positions
                WHERE strategy=? AND date=(SELECT MAX(date) FROM positions WHERE strategy=?)
            """, (strategy, strategy)).fetchall()
        for r in latest:
            if last_eq:
                weights[r["ticker"]] = (r["market_value"] or 0.0) / last_eq
            prices[r["ticker"]] = r["close_price"]
        state = {
            "date":    rows[-1]["date"] if rows else None,
            "weights": weights,
            "prices":  prices,
            "nav":     rows[-1]["nav"] if rows else 1.0,
        }
        with open(os.path.join(out_dir, "state.json"), "w") as fh:
            _json.dump(state, fh, indent=2)
        return out_dir

    def _get_last_positions(self, strategy: str,
                             before_date: str) -> dict[str, dict]:
        with self._conn() as conn:
            last_date = conn.execute("""
                SELECT MAX(date) FROM positions
                WHERE strategy=? AND date < ?
            """, (strategy, before_date)).fetchone()[0]
            if not last_date:
                return {}
            rows = conn.execute("""
                SELECT ticker, shares, close_price, market_value, cost_basis
                FROM positions WHERE strategy=? AND date=?
            """, (strategy, last_date)).fetchall()
        return {r["ticker"]: dict(r) for r in rows}
