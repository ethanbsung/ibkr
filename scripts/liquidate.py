#!/usr/bin/env python3
"""
liquidate.py — close all open futures positions via market orders.

Dry-run by default. Pass --execute to actually submit orders.

Usage:
    python3 scripts/liquidate.py             # preview only
    python3 scripts/liquidate.py --execute   # submit orders
"""

import argparse
import time
from ib_insync import IB, Future, Order

IB_HOST    = "127.0.0.1"
IB_PORT    = 4002
CLIENT_ID  = 99


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Submit orders (default is dry-run)")
    args = parser.parse_args()

    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
    print(f"Connected to IB Gateway (port {IB_PORT})")
    time.sleep(1)

    positions = ib.positions()
    if not positions:
        print("No open positions.")
        ib.disconnect()
        return

    print(f"\n{'DRY RUN' if not args.execute else 'EXECUTING'} — {len(positions)} position(s):\n")

    for pos in positions:
        contract = pos.contract
        ib.qualifyContracts(contract)  # fills in exchange and other missing fields
        qty = pos.position  # positive = long, negative = short
        side = "SELL" if qty > 0 else "BUY"
        close_qty = abs(int(qty))

        print(f"  {side} {close_qty} {contract.symbol} {contract.lastTradeDateOrContractMonth} "
              f"exchange={contract.exchange} (currently {qty:+.0f})  avgCost={pos.avgCost:.4f}")

        if args.execute:
            order = Order(
                action=side,
                totalQuantity=close_qty,
                orderType="MKT",
                tif="DAY",
            )
            trade = ib.placeOrder(contract, order)
            print(f"    → submitted orderId={trade.order.orderId}")

    if args.execute:
        print("\nWaiting 5s for fills...")
        time.sleep(5)
        ib.reqAllOpenOrders()
        time.sleep(1)
        open_orders = ib.openOrders()
        if open_orders:
            print(f"Still {len(open_orders)} order(s) open (market may be closed — orders are DAY).")
        else:
            print("All orders filled or no open orders remaining.")

    ib.disconnect()
    print("Disconnected.")


if __name__ == "__main__":
    main()
