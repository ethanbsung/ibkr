"""
Unit tests for Executor order-routing logic (Phase 1).

Tests the pure sizing/routing decisions — long vs short vs flip, whole-share
short rounding, the position buffer, and the notional ceiling — WITHOUT touching
Alpaca.  The order primitives (_order_notional / _order_qty) are stubbed to
record what *would* be sent.

Run:  python3 etf/live/test_executor_routing.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from etf.live.executor import Executor, MIN_TRADE_USD


def make_executor(max_order_notional=None):
    """Build an Executor without __init__ (no API keys / network)."""
    ex = Executor.__new__(Executor)
    ex.dry_run = True
    ex.max_order_notional = max_order_notional
    ex._orders = []

    def rec_notional(ticker, side, notional, price, target_usd, strategy):
        notional = round(notional, 2)
        if notional < MIN_TRADE_USD:
            return []
        ex._orders.append(("notional", ticker, side, round(notional, 2)))
        return [{"_": "n"}]

    def rec_qty(ticker, side, qty, price, target_usd, strategy):
        qty = int(qty)
        if qty <= 0:
            return []
        ex._orders.append(("qty", ticker, side, qty))
        return [{"_": "q"}]

    ex._order_notional = rec_notional
    ex._order_qty      = rec_qty
    return ex


def routed(cur, target_usd, price, max_notional=None):
    ex = make_executor(max_notional)
    ex._rebalance_one("XYZ", cur, target_usd, price, "ewmac")
    return ex._orders


def closed(cur, price):
    ex = make_executor()
    ex._close_one("XYZ", cur, price, "ewmac")
    return ex._orders


def check(name, got, expected):
    ok = got == expected
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"        expected {expected}")
        print(f"        got      {got}")
    return ok


def main():
    results = []

    # ── Long side (fractional notional) ───────────────────────────────────────
    results.append(check("flat -> long",
        routed(0, 1000, 100), [("notional", "XYZ", "buy", 1000.0)]))
    results.append(check("increase long",
        routed(5, 1000, 100), [("notional", "XYZ", "buy", 500.0)]))
    results.append(check("decrease long",
        routed(10, 300, 100), [("notional", "XYZ", "sell", 700.0)]))
    results.append(check("close long via target 0",
        routed(10, 0, 100), [("notional", "XYZ", "sell", 1000.0)]))
    results.append(check("buffer skips tiny long adjustment",
        routed(10, 1050, 100), []))

    # ── Short side (whole-share qty, round-to-zero) ───────────────────────────
    results.append(check("flat -> short (>=1 share)",
        routed(0, -1000, 100), [("qty", "XYZ", "sell", 10)]))
    results.append(check("short target rounds to zero from flat -> nothing",
        routed(0, -30, 100), []))
    results.append(check("short rounds to zero but holding long -> close long",
        routed(2, -30, 100), [("notional", "XYZ", "sell", 200.0)]))
    results.append(check("increase short",
        routed(-5, -1000, 100), [("qty", "XYZ", "sell", 5)]))
    results.append(check("reduce short (partial cover)",
        routed(-10, -300, 100), [("qty", "XYZ", "buy", 7)]))
    results.append(check("buffer skips tiny short adjustment",
        routed(-10, -1050, 100), []))

    # ── Flips (split into two orders, never cross zero) ───────────────────────
    results.append(check("short -> long flip",
        routed(-5, 1000, 100),
        [("qty", "XYZ", "buy", 5), ("notional", "XYZ", "buy", 1000.0)]))
    results.append(check("long -> short flip",
        routed(8, -1000, 100),
        [("notional", "XYZ", "sell", 800.0), ("qty", "XYZ", "sell", 10)]))

    # ── Notional ceiling ──────────────────────────────────────────────────────
    results.append(check("over notional ceiling -> nothing",
        routed(0, 1000, 100, max_notional=500), []))
    results.append(check("under notional ceiling -> trades",
        routed(0, 400, 100, max_notional=500), [("notional", "XYZ", "buy", 400.0)]))

    # ── Explicit close ────────────────────────────────────────────────────────
    results.append(check("close a long (fractional notional)",
        closed(7.5, 100), [("notional", "XYZ", "sell", 750.0)]))
    results.append(check("close a short (whole-share buy)",
        closed(-7, 100), [("qty", "XYZ", "buy", 7)]))

    n_pass = sum(results)
    print(f"\n  {n_pass}/{len(results)} passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
