#!/usr/bin/env python3
"""
IBS Daily Signal Checker + Auto-Executor
100% IBS mean-reversion on MES / MNQ / MGC.

Signals are generated from the in-progress daily bar fetched via IBKR's
reqHistoricalData (endDateTime='') — no real-time subscription required.
Orders are plain DAY market orders placed just before each session close.

Dry-run (default) — print signals, place nothing:
  python3 portfolio/live_signals.py
  python3 portfolio/live_signals.py --only GC
  python3 portfolio/live_signals.py --only ES NQ

Live execution — place market orders:
  python3 portfolio/live_signals.py --execute
  python3 portfolio/live_signals.py --only GC --execute

Cron schedule (all times ET — adjust cron TZ or convert to UTC):
  28 13 * * 1-5  .../live_signals.py --only GC  --execute   # 1:28 PM — 2 min before MGC close
  58 15 * * 1-5  .../live_signals.py --only ES NQ --execute  # 3:58 PM — 2 min before MES/MNQ close
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
        'name': 'Micro S&P 500', 'exec_time': '3:58 PM ET',
    },
    'NQ': {
        'ibkr_symbol': 'MNQ', 'multiplier': 2,  'exchange': 'CME',
        'name': 'Micro Nasdaq',  'exec_time': '3:58 PM ET',
    },
    'GC': {
        'ibkr_symbol': 'MGC', 'multiplier': 10, 'exchange': 'COMEX',
        'name': 'Micro Gold',   'exec_time': '1:28 PM ET',
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

def get_positions_with_contract(ib):
    """Return {sym: {contract_month_YYYYMM: qty}} for all IBS instruments."""
    held       = {sym: {} for sym in CONTRACT_SPECS}
    symbol_map = {spec['ibkr_symbol']: sym for sym, spec in CONTRACT_SPECS.items()}
    for pos in ib.positions():
        sym_key = symbol_map.get(pos.contract.symbol)
        if sym_key is None:
            continue
        qty = int(pos.position)
        if qty == 0:
            continue
        # lastTradeDateOrContractMonth can be YYYYMM or YYYYMMDD; take first 6 chars
        month = (pos.contract.lastTradeDateOrContractMonth or '')[:6]
        held[sym_key][month] = held[sym_key].get(month, 0) + qty
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

def place_mkt(ib, contract, action, quantity):
    """
    Place a DAY market order for immediate execution.
    Returns the Trade object (live-updated by ib_insync).
    """
    order = Order(
        orderType='MKT',
        action=action,       # 'BUY' or 'SELL'
        totalQuantity=quantity,
        tif='DAY',
    )
    return ib.placeOrder(contract, order)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='IBS Signal Checker + Executor')
    parser.add_argument('--only', nargs='+', metavar='SYM',
                        help='Check only these symbols, e.g. --only GC')
    parser.add_argument('--execute', action='store_true',
                        help='Place market orders (omit for dry-run)')
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

    positions_by_month = get_positions_with_contract(ib)
    pending_symbols    = get_pending_ibkr_symbols(ib)
    today           = datetime.now().strftime('%Y-%m-%d')

    bars = {sym: get_daily_bar(ib, contract) for sym, contract in contracts.items()}

    # ── Header ────────────────────────────────────────────────────────────────
    mode    = "EXECUTE" if args.execute else "DRY-RUN"
    checking = ' / '.join(CONTRACT_SPECS[s]['ibkr_symbol'] for s in CONTRACT_SPECS if s in active_syms)
    print()
    print("=" * 76)
    print(f"  IBS  [{mode}]  |  {today}  |  checking: {checking}")
    print(f"  Equity: ${equity:,.2f}   per instrument: ${equity * IBS_PER_INSTR:,.0f}")
    print(f"  Execution: MES/MNQ → MKT @ 3:58 PM ET   MGC → MKT @ 1:28 PM ET")
    print("=" * 76)

    placed_orders = []
    skipped       = []

    for sym, contract in contracts.items():
        spec         = CONTRACT_SPECS[sym]
        mul          = spec['multiplier']
        ibkr_sym     = spec['ibkr_symbol']
        target_month = contract_months[sym]

        # Position breakdown by contract month
        pos_by_month = positions_by_month.get(sym, {})
        target_qty   = pos_by_month.get(target_month, 0)
        old_months   = {m: q for m, q in pos_by_month.items()
                        if m != target_month and q != 0}
        old_total    = sum(old_months.values())
        net_pos      = target_qty + old_total   # total economic exposure

        print(f"\n  {'─' * 72}")
        print(f"  {sym}  {spec['name']}  ({ibkr_sym} {target_month})"
              f"  |  position: {net_pos:+d}")
        if old_months:
            detail = '  '.join(f"{m}: {q:+d}" for m, q in sorted(old_months.items()))
            print(f"  ⚠ ROLL NEEDED — other months: {detail}")

        bar = bars.get(sym)
        if bar is None:
            print(f"  IBS: no bar data available")
            continue

        h, l, c = bar.high, bar.low, bar.close
        rng      = h - l
        ibs      = (c - l) / rng if rng > 0 else 0.5
        tgt      = ibs_size(equity, c, mul)

        print(f"  H {h:.2f}  L {l:.2f}  C {c:.2f}  |  IBS: {ibs:.3f}  |  target: {tgt} contract(s)")

        # IBS signal → desired qty in target month after all operations
        if net_pos == 0 and ibs < IBS_ENTRY and tgt > 0:
            desired_qty = tgt
            print(f"  *** ENTRY  (IBS {ibs:.3f} < {IBS_ENTRY})")
        elif net_pos > 0 and ibs > IBS_EXIT:
            desired_qty = 0
            print(f"  *** EXIT   (IBS {ibs:.3f} > {IBS_EXIT})")
        elif net_pos > 0 and net_pos != tgt:
            desired_qty = tgt
            delta_label = tgt - net_pos
            print(f"  *** REBALANCE  {net_pos} → {tgt} contract(s)"
                  f"  ({'BUY' if delta_label > 0 else 'SELL'} {abs(delta_label)})")
        elif net_pos > 0:
            desired_qty = net_pos   # hold size; still rolls if old months present
            print(f"  HOLD long ({net_pos} contract(s)) — correctly sized")
        else:
            desired_qty = 0
            if not old_months:
                print(f"  flat — no signal")

        # Build order list
        # Phase A: roll-close — sell/buy old-month positions to zero
        roll_closes = [('SELL' if q > 0 else 'BUY', abs(q), m)
                       for m, q in old_months.items()]
        # Phase B: adjust target-month to desired_qty
        target_delta = desired_qty - target_qty
        target_order = None
        if target_delta > 0:
            target_order = ('BUY', target_delta)
            lbl = 'ROLL OPEN' if old_months and desired_qty == net_pos else 'MKT'
            print(f"  ACTION: BUY {target_delta} {ibkr_sym} {target_month}"
                  f"  [{lbl}]  @ {spec['exec_time']}")
        elif target_delta < 0:
            target_order = ('SELL', -target_delta)
            print(f"  ACTION: SELL {-target_delta} {ibkr_sym} {target_month}"
                  f"  [MKT]  @ {spec['exec_time']}")
        for roll_act, roll_qty, old_month in roll_closes:
            print(f"  ACTION: {roll_act} {roll_qty} {ibkr_sym} {old_month}"
                  f"  [ROLL CLOSE]  MKT")

        if not roll_closes and target_order is None:
            continue   # nothing to do

        if not args.execute:
            continue   # dry-run: printed actions above, stop here

        # ── Live execution ────────────────────────────────────────────────────
        if ibkr_sym in pending_symbols:
            print(f"  SKIPPED — open order already exists for {ibkr_sym}")
            skipped.append(ibkr_sym)
            continue

        # Execute roll closes first (old months → zero)
        for roll_act, roll_qty, old_month in roll_closes:
            raw = Future(symbol=ibkr_sym,
                         lastTradeDateOrContractMonth=old_month,
                         exchange=spec['exchange'], currency='USD')
            old_quals = ib.qualifyContracts(raw)
            if not old_quals:
                print(f"  WARNING: could not qualify {ibkr_sym} {old_month}"
                      f" — skipping roll close")
                continue
            trade = place_mkt(ib, old_quals[0], roll_act, roll_qty)
            ib.sleep(2)
            status = trade.orderStatus.status
            print(f"  ROLL CLOSE {ibkr_sym} {old_month}"
                  f"  {roll_act} {roll_qty}  → {status}"
                  f"  (id {trade.order.orderId})")
            placed_orders.append(
                (f"ROLL {roll_act} {roll_qty:>2} {ibkr_sym} {old_month} MKT",
                 status, trade.order.orderId)
            )

        # Execute target-month order (entry / exit / rebalance / roll-open)
        if target_order:
            action, qty = target_order
            trade = place_mkt(ib, contract, action, qty)
            ib.sleep(2)
            status = trade.orderStatus.status
            print(f"  ORDER PLACED → status: {status}"
                  f"  (orderId {trade.order.orderId})")
            placed_orders.append(
                (f"{action} {qty:>2} {ibkr_sym} {target_month} MKT",
                 status, trade.order.orderId)
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
