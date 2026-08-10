#!/usr/bin/env python3
"""
scripts/unwind_qm_spread.py

One-off remediation for the BUG-28 QM runaway roll.

The daemon's delivery-month cache was poisoned during a gateway drop on
2026-08-08, mis-filing QM's +1 lot under a phantom month. Every subsequent
cycle recomputed a larger roll (1→2→4→8→16→32→64) until IB's margin check
started rejecting. The account was left holding a large phantom calendar
spread against a 1-lot target:

    QMV6 (conId 455805553, delivery 202610)   +64
    QMX6 (conId 455805546, delivery 202611)   -63

This script unwinds that spread down to the target position. It is DELIBERATELY
narrow: it only touches QM, it verifies the live book matches what it expects
before sending anything, and it works in staged chunks so a partial run leaves a
smaller — never a flipped — position.

DRY-RUN BY DEFAULT. Nothing is sent without --execute.

    # inspect the plan
    python3 scripts/unwind_qm_spread.py

    # send it, in chunks of 8, targeting a net +1 in the front month
    python3 scripts/unwind_qm_spread.py --execute

    # smaller chunks / different target
    python3 scripts/unwind_qm_spread.py --execute --chunk 4 --target 1

Halt the daemon first (touch ibkr_fut/risk_halt.txt) so it does not trade
against this script.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ib_insync import IB, Future, MarketOrder  # noqa: E402

# The two legs of the phantom spread, by conId (immutable, unambiguous).
FRONT_CONID = 455805553   # QMV6, delivery 202610
BACK_CONID  = 455805546   # QMX6, delivery 202611

HALT_FILE = Path(__file__).resolve().parent.parent / "ibkr_fut" / "risk_halt.txt"

# Refuse to run if the live book is wildly different from what we diagnosed —
# the operator should re-check rather than let this script improvise.
EXPECTED_FRONT = 64
EXPECTED_BACK  = -63
TOLERANCE      = 10       # allow drift if some legs filled/expired since diagnosis


def fetch_qm(ib):
    """Return {conId: position} for the two QM legs, from a live re-request."""
    positions = ib.reqPositions()
    ib.sleep(1)
    book = {}
    for p in positions:
        c = p.contract
        if c.symbol == "QM" and c.conId in (FRONT_CONID, BACK_CONID):
            book[c.conId] = int(p.position)
    return book


def qualify_leg(ib, con_id):
    c = Future(conId=con_id, exchange="NYMEX")
    quals = ib.qualifyContracts(c)
    if not quals:
        raise SystemExit(f"could not qualify conId {con_id}")
    return quals[0]


def plan_legs(front, back, target):
    """Orders needed to bring the spread to `target` net in the front month.

    Returned as [(conId, action, qty, why)]. The back leg is closed to flat and
    the front leg is reduced to `target` — the position the optimiser actually
    wants. Both are pure reductions toward zero; neither can flip a sign.
    """
    legs = []
    # ORDER MATTERS. The two legs offset each other for margin purposes, so
    # removing the SHORT leg first would strip that offset and leave a naked +64
    # long — spiking initial margin exactly when AvailableFunds is the binding
    # constraint (the same wall that rejected the daemon's rolls). Reduce the
    # oversized LONG leg first: every chunk frees margin and makes the rest safer.
    delta = target - front
    if delta != 0:
        legs.append((FRONT_CONID, "BUY" if delta > 0 else "SELL", abs(delta),
                     f"reduce front leg to target ({front:+d} → {target:+d})"))
    if back != 0:
        # Then close the short far leg (an artifact of the roll loop).
        legs.append((BACK_CONID, "BUY" if back < 0 else "SELL", abs(back),
                     f"close phantom back leg ({back:+d} → 0)"))
    return legs


def send_chunked(ib, contract, action, qty, chunk, execute, pause):
    """Send `qty` as a sequence of market orders no larger than `chunk`.

    Chunking keeps each order's margin impact small — the whole reason the
    daemon's 64-lot spread was rejected — and lets a partial run stop cleanly.
    """
    sent = filled = 0
    while sent < qty:
        this = min(chunk, qty - sent)
        if not execute:
            print(f"      [DRY-RUN] {action} {this} {contract.localSymbol}")
            sent += this
            continue
        order = MarketOrder(action, this, tif="DAY")
        trade = ib.placeOrder(contract, order)
        deadline = time.time() + 60
        while time.time() < deadline and not trade.isDone():
            ib.sleep(1)
        got = int(trade.filled())
        status = trade.orderStatus.status
        print(f"      {action} {this} {contract.localSymbol} → {status} "
              f"filled {got} @ {trade.orderStatus.avgFillPrice or 0:.4f}")
        if status not in ("Filled",) and got == 0:
            print(f"      STOP: order did not fill ({status}). "
                  f"Not sending further chunks.")
            break
        sent += this
        filled += got
        ib.sleep(pause)
    return filled


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true",
                    help="actually send orders (default: dry run)")
    ap.add_argument("--target", type=int, default=1,
                    help="desired net front-month position (default: 1)")
    ap.add_argument("--chunk", type=int, default=8,
                    help="max contracts per order (default: 8)")
    ap.add_argument("--pause", type=float, default=3.0,
                    help="seconds between chunks (default: 3)")
    ap.add_argument("--port", type=int, default=4002)
    ap.add_argument("--client-id", type=int, default=90)
    ap.add_argument("--force", action="store_true",
                    help="proceed even if the live book differs from the "
                         "diagnosed state")
    args = ap.parse_args()

    if args.execute and not HALT_FILE.exists():
        print(f"REFUSING: {HALT_FILE} does not exist.\n"
              f"Halt the daemon first so it does not trade against this script:\n"
              f"    touch {HALT_FILE}")
        return 1

    ib = IB()
    ib.connect("127.0.0.1", args.port, clientId=args.client_id, timeout=20)
    try:
        book = fetch_qm(ib)
        front = book.get(FRONT_CONID, 0)
        back  = book.get(BACK_CONID, 0)

        print(f"\nLive QM book")
        print(f"  QMV6 (202610, conId {FRONT_CONID}): {front:+d}")
        print(f"  QMX6 (202611, conId {BACK_CONID}): {back:+d}")
        print(f"  net: {front + back:+d}   target front-month: {args.target:+d}")

        if front == args.target and back == 0:
            print("\nAlready at target — nothing to do.")
            return 0

        drift = (abs(front - EXPECTED_FRONT) > TOLERANCE
                 or abs(back - EXPECTED_BACK) > TOLERANCE)
        if drift and not args.force:
            print(f"\nREFUSING: live book differs from the diagnosed state "
                  f"(expected ~{EXPECTED_FRONT:+d}/{EXPECTED_BACK:+d}). "
                  f"Re-check the position, then re-run with --force if this is "
                  f"still what you want.")
            return 1

        legs = plan_legs(front, back, args.target)
        print(f"\nPlan ({'EXECUTE' if args.execute else 'DRY RUN'}), "
              f"chunks of {args.chunk}:")
        for con_id, action, qty, why in legs:
            print(f"  • {action} {qty} conId={con_id}  — {why}")

        for con_id, action, qty, why in legs:
            contract = qualify_leg(ib, con_id)
            print(f"\n  {why}")
            send_chunked(ib, contract, action, qty, args.chunk,
                         args.execute, args.pause)

        if args.execute:
            ib.sleep(3)
            after = fetch_qm(ib)
            print(f"\nFinal QM book")
            print(f"  QMV6: {after.get(FRONT_CONID, 0):+d}")
            print(f"  QMX6: {after.get(BACK_CONID, 0):+d}")
            print(f"\nRemove {HALT_FILE} once you've verified this, then restart "
                  f"the daemon.")
        else:
            print(f"\nDry run only — nothing sent. Re-run with --execute "
                  f"(after halting the daemon) to apply.")
        return 0
    finally:
        ib.disconnect()


if __name__ == "__main__":
    sys.exit(main())
