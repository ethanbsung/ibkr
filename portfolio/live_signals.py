#!/usr/bin/env python3
"""
IBS Daily Signal Checker + Auto-Executor
100% IBS mean-reversion on MES / MNQ / MGC.

Dry-run (default) — print signals, place nothing:
  python3 portfolio/live_signals.py
  python3 portfolio/live_signals.py --only GC
  python3 portfolio/live_signals.py --only ES NQ

Live execution — place MOC orders:
  python3 portfolio/live_signals.py --execute
  python3 portfolio/live_signals.py --only GC --execute

Cron schedule (all times ET — adjust cron TZ or convert to UTC):
  15 13 * * 1-5  .../live_signals.py --only GC  --execute   # 1:15 PM MGC MOC
  45 15 * * 1-5  .../live_signals.py --only ES NQ --execute  # 3:45 PM MES/MNQ MOC
"""

import argparse
import logging
import calendar
import os
from datetime import datetime, timedelta
from ib_insync import IB, Future, Order

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Parameters ─────────────────────────────────────────────────────────────────
IBS_ENTRY     = 0.10
IBS_EXIT      = 0.90
IBS_PER_INSTR = 1.0 / 3   # equal weight, 3 instruments
VOL_SCALAR    = 2.0        # position size multiplier; 1.0 ≈ 5.4% ann vol, 2.0 ≈ 10.8% ann vol

IB_HOST   = '127.0.0.1'
IB_PORT   = 4002
CLIENT_ID = 4

# ── Contract specifications ────────────────────────────────────────────────────
CONTRACT_SPECS = {
    'ES': {
        'ibkr_symbol': 'MES', 'multiplier': 5,  'exchange': 'CME',
        'name': 'Micro S&P 500', 'moc_cutoff': '3:45 PM ET',
    },
    'NQ': {
        'ibkr_symbol': 'MNQ', 'multiplier': 2,  'exchange': 'CME',
        'name': 'Micro Nasdaq',  'moc_cutoff': '3:45 PM ET',
    },
    'GC': {
        'ibkr_symbol': 'MGC', 'multiplier': 10, 'exchange': 'COMEX',
        'name': 'Micro Gold',   'moc_cutoff': '1:15 PM ET',
    },
}


# ── Contract month helpers ─────────────────────────────────────────────────────

def _third_friday(year, month):
    first = datetime(year, month, 1)
    days_to_fri = (4 - first.weekday()) % 7
    return first + timedelta(days=days_to_fri + 14)

def _third_to_last_biz(year, month):
    last = datetime(year, month, calendar.monthrange(year, month)[1])
    count, d = 0, last
    while True:
        if d.weekday() < 5:
            count += 1
            if count == 3:
                return d
        d -= timedelta(days=1)

def quarterly_contract_month(roll_days=7):
    today  = datetime.now()
    cutoff = today + timedelta(days=roll_days)
    for year in range(today.year, today.year + 2):
        for month in (3, 6, 9, 12):
            if _third_friday(year, month) >= cutoff:
                return f"{year}{month:02d}"
    raise RuntimeError("Cannot determine quarterly contract month")

def gold_contract_month(roll_days=5):
    today  = datetime.now()
    cutoff = today + timedelta(days=roll_days)
    for year in range(today.year, today.year + 2):
        for month in (2, 4, 6, 8, 10, 12):
            if _third_to_last_biz(year, month) >= cutoff:
                return f"{year}{month:02d}"
    raise RuntimeError("Cannot determine gold contract month")


# ── IBKR helpers ───────────────────────────────────────────────────────────────

def get_equity(ib):
    for v in ib.accountValues():
        if v.tag == 'NetLiquidation' and v.currency == 'USD':
            return float(v.value)
    return None

def get_positions(ib):
    held       = {sym: 0 for sym in CONTRACT_SPECS}
    symbol_map = {spec['ibkr_symbol']: sym for sym, spec in CONTRACT_SPECS.items()}
    for pos in ib.positions():
        if pos.contract.symbol in symbol_map:
            held[symbol_map[pos.contract.symbol]] = int(pos.position)
    return held

def get_daily_bar(ib, contract):
    bars = ib.reqHistoricalData(
        contract, endDateTime='', durationStr='5 D',
        barSizeSetting='1 day', whatToShow='TRADES',
        useRTH=False, formatDate=1, keepUpToDate=False,
    )
    return bars[-1] if bars else None

def get_pending_ibkr_symbols(ib):
    """Return set of IBKR symbols (e.g. 'MES') that already have an open order."""
    return {trade.contract.symbol for trade in ib.openTrades()}


# ── Position sizing ────────────────────────────────────────────────────────────

def ibs_size(equity, price, multiplier):
    contract_value = price * multiplier
    if contract_value <= 0:
        return 0
    return round(equity * IBS_PER_INSTR * VOL_SCALAR / contract_value)


# ── Order placement ────────────────────────────────────────────────────────────

def place_moc(ib, contract, action, quantity):
    """
    Place a Market-on-Close order.
    Returns the Trade object (live-updated by ib_insync).
    """
    order = Order(
        orderType='MOC',
        action=action,           # 'BUY' or 'SELL'
        totalQuantity=quantity,
        outsideRth=False,        # execute at official close only
    )
    return ib.placeOrder(contract, order)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='IBS Signal Checker + Executor')
    parser.add_argument('--only', nargs='+', metavar='SYM',
                        help='Check only these symbols, e.g. --only GC')
    parser.add_argument('--execute', action='store_true',
                        help='Place MOC orders (omit for dry-run)')
    args = parser.parse_args()

    valid_syms = set(CONTRACT_SPECS)
    if args.only:
        active_syms = {s.upper() for s in args.only}
        unknown = active_syms - valid_syms
        if unknown:
            print(f"ERROR: Unknown symbols: {unknown}.  Valid: {valid_syms}")
            return
    else:
        active_syms = valid_syms

    ib = IB()
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
    except Exception as e:
        print(f"ERROR: Could not connect to IBKR — {e}")
        print("Make sure TWS or IB Gateway is running.")
        return

    q_month = quarterly_contract_month()
    g_month = gold_contract_month()
    contract_months = {'ES': q_month, 'NQ': q_month, 'GC': g_month}

    contracts = {}
    for sym in active_syms:
        spec  = CONTRACT_SPECS[sym]
        raw   = Future(symbol=spec['ibkr_symbol'],
                       lastTradeDateOrContractMonth=contract_months[sym],
                       exchange=spec['exchange'], currency='USD')
        quals = ib.qualifyContracts(raw)
        if not quals:
            print(f"WARNING: Could not qualify {sym} — skipping.")
            continue
        contracts[sym] = quals[0]

    if not contracts:
        print("ERROR: No contracts qualified.")
        ib.disconnect()
        return

    equity          = get_equity(ib)
    if equity is None:
        print("WARNING: Could not read account equity. Using $250,000.")
        equity = 250_000.0

    ibkr_positions  = get_positions(ib)
    pending_symbols = get_pending_ibkr_symbols(ib)
    today           = datetime.now().strftime('%Y-%m-%d')

    bars = {sym: get_daily_bar(ib, contract) for sym, contract in contracts.items()}

    # ── Header ────────────────────────────────────────────────────────────────
    mode    = "EXECUTE" if args.execute else "DRY-RUN"
    checking = ' / '.join(CONTRACT_SPECS[s]['ibkr_symbol'] for s in CONTRACT_SPECS if s in active_syms)
    print()
    print("=" * 76)
    print(f"  IBS  [{mode}]  |  {today}  |  checking: {checking}")
    print(f"  Equity: ${equity:,.2f}   per instrument: ${equity * IBS_PER_INSTR:,.0f}")
    print(f"  MOC cutoffs: MES/MNQ → 3:45 PM ET   MGC → 1:15 PM ET")
    print("=" * 76)

    placed_orders = []
    skipped       = []

    for sym, contract in contracts.items():
        spec       = CONTRACT_SPECS[sym]
        mul        = spec['multiplier']
        ibkr_sym   = spec['ibkr_symbol']
        net_pos    = ibkr_positions.get(sym, 0)
        bar        = bars.get(sym)

        print(f"\n  {'─' * 72}")
        print(f"  {sym}  {spec['name']}  ({ibkr_sym} {contract_months[sym]})"
              f"  |  position: {net_pos:+d}")

        if bar is None:
            print(f"  IBS: no bar data available")
            continue

        h, l, c = bar.high, bar.low, bar.close
        rng      = h - l
        ibs      = (c - l) / rng if rng > 0 else 0.5
        tgt      = ibs_size(equity, c, mul)

        print(f"  H {h:.2f}  L {l:.2f}  C {c:.2f}  |  IBS: {ibs:.3f}  |  target: {tgt} contract(s)")

        action = None
        qty    = 0

        if net_pos == 0 and ibs < IBS_ENTRY and tgt > 0:
            action, qty = 'BUY', tgt
            print(f"  *** ENTRY  (IBS {ibs:.3f} < {IBS_ENTRY})")
            print(f"  ACTION: BUY {tgt} {ibkr_sym}  — MOC, cutoff {spec['moc_cutoff']}")

        elif net_pos > 0 and ibs > IBS_EXIT:
            action, qty = 'SELL', net_pos
            print(f"  *** EXIT   (IBS {ibs:.3f} > {IBS_EXIT})")
            print(f"  ACTION: SELL {net_pos} {ibkr_sym}  — MOC, cutoff {spec['moc_cutoff']}")

        elif net_pos > 0:
            print(f"  HOLD long ({net_pos} contract(s))")
        else:
            print(f"  flat — no signal")

        if action is None:
            continue   # nothing to do for this instrument

        if not args.execute:
            continue   # dry-run: printed signal above, stop here

        # ── Live execution ────────────────────────────────────────────────────
        if ibkr_sym in pending_symbols:
            print(f"  SKIPPED — open order already exists for {ibkr_sym}")
            skipped.append(ibkr_sym)
            continue

        trade = place_moc(ib, contract, action, qty)
        ib.sleep(2)   # let TWS process and push status back

        status = trade.orderStatus.status
        print(f"  ORDER PLACED → status: {status}  "
              f"(orderId {trade.order.orderId})")
        placed_orders.append(
            (f"{action} {qty:>2} {ibkr_sym} MOC", status, trade.order.orderId)
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 76)
    if args.execute:
        if placed_orders:
            print("  ORDERS PLACED:")
            for desc, status, oid in placed_orders:
                print(f"    {desc:<30}  status: {status:<20}  id: {oid}")
        if skipped:
            print(f"  SKIPPED (already pending): {', '.join(skipped)}")
        if not placed_orders and not skipped:
            print("  NO ORDERS PLACED — no signals triggered")
    else:
        print("  DRY-RUN — no orders placed. Re-run with --execute to submit.")
    print("=" * 76)
    print()

    ib.disconnect()


if __name__ == '__main__':
    main()
