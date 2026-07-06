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
from datetime import datetime
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

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
    status:        str    # Filled | PartiallyFilled | Unfilled | Cancelled | Rejected | Skipped
    was_aggressive: bool
    commission:    float
    order_id:      int
    reject_reason: str = ""   # IB rejection text when the broker refused the order (BUG-13)


# IB error codes that accompany our OWN cancels rather than a broker rejection —
# must not be classified as "Rejected". 202 = "Order Canceled" (sent for every
# cancelOrder ack), 161 = cancel attempted in an uncancellable state.
_CANCEL_ERROR_CODES = {161, 202}


def _reject_reason(trade) -> str:
    """
    IB rejection text from the trade log, '' if none.

    Rejections (margin violation, missing permission, size/price limits) do NOT
    arrive as a distinct order status — they surface as TradeLogEntry rows
    carrying an errorCode, with the status left 'Inactive' or moved to
    'Cancelled' depending on the reject type (BUG-13). Reading the log covers
    both shapes; our own cancels (code 202) are excluded so a timed-out limit
    isn't misreported as rejected.
    """
    for entry in reversed(getattr(trade, "log", []) or []):
        code = getattr(entry, "errorCode", 0) or 0
        if code and code not in _CANCEL_ERROR_CODES:
            return f"IB error {code}: {getattr(entry, 'message', '')}"
    return ""


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
    heartbeat=None,      # optional callable, invoked on each progress tick (BUG-12)
) -> FillResult:
    """
    Passive-aggressive limit order execution.

    The Ticker must already be subscribed (via reqMktData in pre_trade_checks).
    The caller is responsible for calling ib.cancelMktData(contract) after this returns.

    heartbeat: optional zero-arg callable invoked on the MESSAGING_FREQUENCY
    progress tick, so a long-running order (up to TOTAL_TIME_OUT) keeps the
    daemon's liveness signal fresh — otherwise the watchdog mistakes a slow
    rebalance cycle for a dead daemon and kill -9s it mid-order (BUG-12).
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

        if (trade.orderStatus.status or "") == "Inactive":
            # Rejected orders land 'Inactive', which is NOT in ib_insync's
            # DoneStates — without this break the aggressive chase would spin
            # for the full TOTAL_TIME_OUT re-submitting a rejected order
            # (BUG-13). The reject reason is read from trade.log below.
            break

        elapsed = time.time() - start_time

        if time.time() - last_msg_time >= MESSAGING_FREQUENCY:
            phase  = "AGGRESSIVE" if is_aggressive else "PASSIVE"
            filled = int(trade.orderStatus.filled or 0)
            print(f"      [{phase}] {action} {qty}  elapsed={elapsed:.0f}s  "
                  f"lmt={order.lmtPrice:.4f}  filled={filled}")
            last_msg_time = time.time()
            if heartbeat:
                heartbeat()

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

    # Rejection normalisation (BUG-13): an unfilled order whose trade log carries
    # a broker error is a REJECTION, not a mere timeout — the caller must alert,
    # not silently retry next cycle. A partial fill keeps its fill status but
    # still carries the reject text for the remainder.
    reject = _reject_reason(trade)
    if reject and filled_qty == 0:
        raw_status = "Rejected"

    return FillResult(
        filled_qty    = filled_qty,
        avg_price     = avg_price,
        status        = raw_status,
        was_aggressive = is_aggressive,
        commission    = commission,
        order_id      = trade.order.orderId,
        reject_reason = reject,
    )


# ── Market-hours check ────────────────────────────────────────────────────────

_IB_TZ_MAP = {
    "US/Eastern":                  "America/New_York",
    "EST":                         "America/New_York",
    "EST (Eastern Standard Time)": "America/New_York",
    "CST (Central Standard Time)": "America/Chicago",
    "US/Central":                  "America/Chicago",
    "MET":                         "Europe/Berlin",
    "MET (Middle Europe Time)":    "Europe/Berlin",
    "GB-Eire":                     "Europe/London",
    "JST":                         "Asia/Tokyo",
    "JST (Japan Standard Time)":   "Asia/Tokyo",
    "Japan":                       "Asia/Tokyo",
    "Hongkong":                    "Asia/Hong_Kong",
    "Australia/NSW":               "Australia/Sydney",
    "Singapore":                   "Asia/Singapore",
    "Korea":                       "Asia/Seoul",
    "":                            "UTC",
}


def is_contract_okay_to_trade(ib: IB, contract: Contract) -> bool:
    """Return True if 'now' falls within one of the contract's IB trading windows.

    Parses the tradingHours string from IB contract details, e.g.:
      "20260608:1700-20260609:1600;20260609:CLOSED;20260610:1700-20260611:1600"
    Times are in the exchange's local timezone (given by timeZoneId).
    """
    try:
        details = ib.reqContractDetails(contract)
    except Exception:
        return False
    if not details:
        return False

    cd = details[0]
    tz_id = cd.timeZoneId or ""
    iana = _IB_TZ_MAP.get(tz_id, "UTC")
    try:
        tz = ZoneInfo(iana)
    except Exception:
        tz = ZoneInfo("UTC")

    now = datetime.now(tz)
    for seg in (cd.tradingHours or "").split(";"):
        seg = seg.strip()
        if not seg or "CLOSED" in seg:
            continue
        try:
            s, e = seg.split("-", 1)
            start = datetime.strptime(s.strip(), "%Y%m%d:%H%M").replace(tzinfo=tz)
            end   = datetime.strptime(e.strip(), "%Y%m%d:%H%M").replace(tzinfo=tz)
            if start <= now <= end:
                return True
        except Exception:
            continue
    return False


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
