#!/usr/bin/env python3
"""
Carver's passive-aggressive limit order execution algorithm for IBKR futures.

Reference: https://qoppac.blogspot.com/2014/10/the-worlds-simplest-execution-algorithim.html

Phase 1 — Passive (0 to PASSIVE_TIME_OUT seconds):
    Place a limit at the offside price (bid when buying, ask when selling).
    Goal: get filled without crossing the spread.

Phase 2 — Aggressive (after PASSIVE_TIME_OUT or an early trigger):
    Chase the inside spread (ask when buying, bid when selling), updating
    the limit price as the market moves.

Order is cancelled after TOTAL_TIME_OUT seconds if still unfilled.
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from ib_insync import IB, Contract, Order, Ticker


# ── Carver's original parameters ──────────────────────────────────────────────
PASSIVE_TIME_OUT    = 300   # seconds before switching to aggressive
TOTAL_TIME_OUT      = 600   # seconds before cancelling entirely
CANCEL_WAIT_TIME    = 60    # seconds to wait for cancel confirmation
IMBALANCE_THRESHOLD = 5.0   # bid/ask size ratio that triggers aggressive switch
MESSAGING_FREQUENCY = 30    # seconds between progress log lines


@dataclass
class FillResult:
    filled_qty:    int
    avg_price:     float
    status:        str    # Filled | PartiallyFilled | Unfilled | Cancelled | Skipped
    was_aggressive: bool
    commission:    float
    order_id:      int


def pre_trade_checks(
    ib:             IB,
    contract:       Contract,
    pst_close_ibkr: Optional[float],
    sigma:          float,
    order_qty:      int,
) -> Tuple[bool, str, Optional[Ticker]]:
    """
    Subscribe to market data and run pre-trade checks.

    Returns (ok, reason, ticker).
      ok=True  → checks passed; ticker is live and ready for algo_exec.
      ok=False → skip this order; market data already cancelled, ticker is None.

    Checks:
      1. Valid bid and ask arrive within 10 seconds.
      2. Live mid is within 3 daily SDs of the PST close used to size the position.
         (skipped when pst_close_ibkr or sigma are unavailable)
    """
    ticker = ib.reqMktData(contract, "", False, False)
    ib.sleep(10)

    bid, ask = ticker.bid, ticker.ask

    if not (_valid(bid) and _valid(ask) and bid < ask):
        ib.cancelMktData(contract)
        return False, "no valid bid/ask after 10 s", None

    if _valid(pst_close_ibkr) and sigma > 0:
        mid        = (bid + ask) / 2.0
        threshold  = 3.0 * sigma / math.sqrt(256)           # 3 daily SDs
        divergence = abs(mid - pst_close_ibkr) / pst_close_ibkr
        if divergence > threshold:
            ib.cancelMktData(contract)
            return (False,
                    f"price diverged {divergence:.2%} vs threshold {threshold:.2%}",
                    None)

    return True, "", ticker


def algo_exec(
    ib:       IB,
    contract: Contract,
    action:   str,       # "BUY" or "SELL"
    qty:      int,
    ticker:   Ticker,
) -> FillResult:
    """
    Passive-aggressive limit order execution.

    The Ticker must already be subscribed (via reqMktData in pre_trade_checks).
    The caller is responsible for calling ib.cancelMktData(contract) after this returns.
    """
    bid, ask = ticker.bid, ticker.ask

    # Offside price: the passive side we try to be filled on (no spread crossing).
    offside_price = bid if action == "BUY" else ask
    # Reference price for adverse-price detection (the other side of the spread).
    ref_price = ask if action == "BUY" else bid

    if not _valid(offside_price):
        return FillResult(0, 0.0, "Unfilled", False, 0.0, 0)

    order = Order(
        orderType     = "LMT",
        action        = action,
        totalQuantity = qty,
        lmtPrice      = offside_price,
        tif           = "DAY",
    )
    trade = ib.placeOrder(contract, order)
    print(f"      PASSIVE  {action} {qty}  lmt={offside_price:.4f}")

    start_time    = time.time()
    last_msg_time = start_time
    is_aggressive = False
    switch_reason = ""

    while True:
        ib.sleep(1)

        if trade.isDone():
            break

        elapsed = time.time() - start_time

        if time.time() - last_msg_time >= MESSAGING_FREQUENCY:
            phase  = "AGGRESSIVE" if is_aggressive else "PASSIVE"
            filled = int(trade.orderStatus.filled or 0)
            print(f"      [{phase}] {action} {qty}  elapsed={elapsed:.0f}s  "
                  f"lmt={order.lmtPrice:.4f}  filled={filled}")
            last_msg_time = time.time()

        if elapsed > TOTAL_TIME_OUT:
            ib.cancelOrder(order)
            ib.sleep(CANCEL_WAIT_TIME)
            break

        if not is_aggressive:
            if elapsed > PASSIVE_TIME_OUT:
                is_aggressive = True
                switch_reason = "passive timeout"
            elif _adverse_price(action, ref_price, ticker):
                is_aggressive = True
                switch_reason = "adverse price movement"
            elif _adverse_size(action, ticker):
                is_aggressive = True
                switch_reason = "adverse size/imbalance"

            if is_aggressive:
                print(f"      SWITCH → AGGRESSIVE ({switch_reason})")

        if is_aggressive:
            new_price = ticker.ask if action == "BUY" else ticker.bid
            if _valid(new_price) and new_price != order.lmtPrice:
                order.lmtPrice = new_price
                ib.placeOrder(contract, order)   # modify in place (same orderId)

    # ── Build result ──────────────────────────────────────────────────────────
    raw_status = trade.orderStatus.status or ""
    filled_qty = int(trade.orderStatus.filled or 0)
    avg_price  = float(trade.orderStatus.avgFillPrice or 0.0)
    commission = sum(
        f.commissionReport.commission
        for f in trade.fills
        if not math.isnan(f.commissionReport.commission)
    )

    if not raw_status:
        if filled_qty == qty:
            raw_status = "Filled"
        elif filled_qty > 0:
            raw_status = "PartiallyFilled"
        else:
            raw_status = "Unfilled"

    return FillResult(
        filled_qty    = filled_qty,
        avg_price     = avg_price,
        status        = raw_status,
        was_aggressive = is_aggressive,
        commission    = commission,
        order_id      = trade.order.orderId,
    )


# ── Private helpers ────────────────────────────────────────────────────────────

def _valid(price) -> bool:
    """True if price is a finite positive number."""
    try:
        return price is not None and not math.isnan(float(price)) and float(price) > 0
    except (TypeError, ValueError):
        return False


def _sz(v) -> float:
    """Safe size extraction (returns 0 for None/nan)."""
    try:
        f = float(v)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def _adverse_price(action: str, ref_price: float, ticker: Ticker) -> bool:
    """True if the market has moved against our passive limit since order placement.

    When buying, we placed a limit at the bid. If the ask rises above the initial
    ask (ref_price), the market has moved up and our passive bid won't attract sellers.
    When selling, symmetric logic applies.
    """
    if action == "BUY":
        return _valid(ticker.ask) and ticker.ask > ref_price
    else:
        return _valid(ticker.bid) and ticker.bid < ref_price


def _adverse_size(action: str, ticker: Ticker) -> bool:
    """True if the order book is heavily one-sided against our fill direction.

    When buying passively at the bid: if there are 5x more buyers than sellers,
    price pressure is upward and our passive bid is unlikely to attract a fill.
    When selling, symmetric logic applies.
    """
    bid_sz = _sz(ticker.bidSize)
    ask_sz = _sz(ticker.askSize)
    if action == "BUY":
        return ask_sz > 0 and bid_sz > IMBALANCE_THRESHOLD * ask_sz
    else:
        return bid_sz > 0 and ask_sz > IMBALANCE_THRESHOLD * bid_sz
