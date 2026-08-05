"""
Unit tests for Phase 3 ledger accounting fixes.

Verifies (without Alpaca):
  • cost_basis / unrealized_pnl are correct after a position ADD  (bug #9)
  • daily nav_return / equity_pnl come from the Alpaca equity change (bug #3)
  • realized_vol is computed from the nav_return series                (bug #8)
  • export_paper_book writes a book readable by paper.engine.Book

Run:  python3 etf/live/test_ledger_phase3.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "paper"))

from etf.live.ledger import Ledger

RESULTS = []


def check(name, cond, extra=""):
    RESULTS.append(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra and not cond else ''}")


def detail(shares, avg, mv):
    return {"shares": shares, "avg_entry_price": avg, "market_value": mv}


def get_pos(L, strat, dt, tk):
    with L._conn() as c:
        r = c.execute("SELECT * FROM positions WHERE strategy=? AND date=? AND ticker=?",
                      (strat, dt, tk)).fetchone()
    return dict(r) if r else None


def get_daily(L, strat, dt):
    with L._conn() as c:
        r = c.execute("SELECT * FROM daily_pnl WHERE strategy=? AND date=?",
                      (strat, dt)).fetchone()
    return dict(r) if r else None


def main():
    db = os.path.join(tempfile.mkdtemp(), "t.db")
    L  = Ledger(db)
    S  = "ewmac"

    # ── Day 1: open 10 sh @ entry 100, price 100, equity 5000 ─────────────────
    L.snapshot_positions(S, {"AAA": detail(10, 100.0, 1000.0)}, {"AAA": 100.0}, "2026-01-01")
    L.record_daily_pnl(S, 5000.0, [], "2026-01-01")
    d1 = get_daily(L, S, "2026-01-01")
    check("day1 nav_return is 0 (first day, no prior equity)", d1["nav_return"] == 0.0)
    check("day1 equity_pnl is 0", d1["equity_pnl"] == 0.0)

    # ── Day 2: ADD to 20 sh, new avg entry 110, price 120, equity 5100 ────────
    # cost_basis = 110*20 = 2200 ; unrealized = 2400 - 2200 = 200 = (120-110)*20
    # The OLD bug used a carried per-share 100 -> unrealized 400 (wrong).
    L.snapshot_positions(S, {"AAA": detail(20, 110.0, 2400.0)}, {"AAA": 120.0}, "2026-01-02")
    L.record_daily_pnl(S, 5100.0, [], "2026-01-02")
    p2 = get_pos(L, S, "2026-01-02", "AAA")
    check("day2 cost_basis = avg_entry*shares (2200)", abs(p2["cost_basis"] - 2200.0) < 1e-6,
          f"got {p2['cost_basis']}")
    check("day2 unrealized_pnl correct after add (200)", abs(p2["unrealized_pnl"] - 200.0) < 1e-6,
          f"got {p2['unrealized_pnl']}")

    d2 = get_daily(L, S, "2026-01-02")
    check("day2 nav_return = 5100/5000-1 = 0.02", abs(d2["nav_return"] - 0.02) < 1e-9,
          f"got {d2['nav_return']}")
    check("day2 equity_pnl = +100", abs(d2["equity_pnl"] - 100.0) < 1e-6,
          f"got {d2['equity_pnl']}")

    # ── Days 3-6: feed a known equity path, check realized_vol from nav_return ─
    path = [("2026-01-03", 5202.0), ("2026-01-04", 5150.0),
            ("2026-01-05", 5250.0), ("2026-01-06", 5300.0)]
    for dt, eq in path:
        L.snapshot_positions(S, {"AAA": detail(20, 110.0, 2400.0)}, {"AAA": 120.0}, dt)
        L.record_daily_pnl(S, eq, [], dt)

    d6 = get_daily(L, S, "2026-01-06")
    check("realized_vol computed once >=5 returns exist", d6["realized_vol"] is not None,
          f"got {d6['realized_vol']}")

    # Independent recomputation of realized_vol from the stored nav_returns
    import numpy as np
    with L._conn() as c:
        navs = [r["nav_return"] for r in c.execute(
            "SELECT nav_return FROM daily_pnl WHERE strategy=? AND date<=? "
            "ORDER BY date DESC LIMIT 20", (S, "2026-01-06")).fetchall()]
    expected_rv = float(np.std(navs) * (252 ** 0.5))
    check("realized_vol == annualized std of nav_return series",
          abs(d6["realized_vol"] - expected_rv) < 1e-9,
          f"got {d6['realized_vol']} vs {expected_rv}")

    # ── Paper book export, readable by paper.engine.Book ──────────────────────
    proot = tempfile.mkdtemp()
    L.export_paper_book(S, proot, book_name="etf_ewmac")
    from engine import Book   # paper/engine.py (added to sys.path above)
    bf = Book("etf_ewmac", proot).daily_frame()
    check("paper book non-empty & parses", not bf.empty and "ret" in bf.columns)
    check("paper book has all 6 days", len(bf) == 6, f"got {len(bf)}")
    # ret column must equal stored nav_return for the last day
    check("paper book ret matches nav_return", abs(bf["ret"].iloc[-1] - d6["nav_return"]) < 1e-12)
    # nav is the cumulative product of (1+ret)
    expected_nav = float(np.prod([1 + r for r in bf["ret"].tolist()]))
    check("paper book nav = cumprod(1+ret)", abs(bf["nav"].iloc[-1] - expected_nav) < 1e-9,
          f"got {bf['nav'].iloc[-1]} vs {expected_nav}")

    n_pass = sum(RESULTS)
    print(f"\n  {n_pass}/{len(RESULTS)} passed")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
