#!/usr/bin/env python3
"""
IBS + EWMAC Daily Signal Checker
75% IBS mean-reversion  |  25% EWMAC(64,256) trend-following

Run schedule
  1:15 PM ET  — check MGC  (MOC cutoff before 1:30 PM ET COMEX close)
  3:45 PM ET  — check MES and MNQ  (MOC cutoff before 4:00 / 4:15 PM ET)
  Either run  — EWMAC section always shown; signal changes ~1x/year per instrument

Capital split
  75% IBS   → 25% per instrument (3 instruments, equal weight, long-only)
  25% EWMAC → 8.3% per instrument, vol-targeted, long or short

IBS execution:   MOC order at the cutoff time shown per instrument
EWMAC execution: MOC same day a crossover is detected; position held for months
EWMAC state:     Saved to ewmac_state.json — tracks direction + contracts per
                 instrument so crossovers are detected between runs.

Position accounting note
  IBKR shows a single net position per instrument.  This script infers the IBS
  portion as:  net_pos − ewmac_state_contracts.  If you manually adjust a
  position, update ewmac_state.json accordingly so the inference stays correct.
"""

import json
import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import calendar
from ib_insync import IB, Future, util

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Parameters ─────────────────────────────────────────────────────────────────
IBS_ENTRY   = 0.10
IBS_EXIT    = 0.90
IBS_ALLOC   = 0.75      # 75% of equity to IBS sub-portfolio
EWMAC_ALLOC = 0.25      # 25% of equity to EWMAC sub-portfolio
N_INSTR     = 3
IBS_PER_INSTR   = IBS_ALLOC  / N_INSTR   # 0.2500 per instrument
EWMAC_PER_INSTR = EWMAC_ALLOC / N_INSTR  # 0.0833 per instrument

RISK_TARGET = 0.20
FAST_SPAN   = 64
SLOW_SPAN   = 256
VOL_SPAN    = 32
VOL_FLOOR   = 0.05

IB_HOST   = '127.0.0.1'
IB_PORT   = 7497
CLIENT_ID = 4

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ewmac_state.json')

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
        ibkr_sym = pos.contract.symbol
        if ibkr_sym in symbol_map:
            held[symbol_map[ibkr_sym]] = int(pos.position)
    return held

def get_daily_bar(ib, contract):
    bars = ib.reqHistoricalData(
        contract, endDateTime='', durationStr='5 D',
        barSizeSetting='1 day', whatToShow='TRADES',
        useRTH=False, formatDate=1, keepUpToDate=False,
    )
    return bars[-1] if bars else None

def get_long_history(ib, contract):
    """Fetch 2 years of daily bars for EMA(256) computation."""
    return ib.reqHistoricalData(
        contract, endDateTime='', durationStr='2 Y',
        barSizeSetting='1 day', whatToShow='TRADES',
        useRTH=False, formatDate=1, keepUpToDate=False,
    )


# ── Signal computation ─────────────────────────────────────────────────────────

def compute_ewmac(bars):
    """
    Returns (direction, ann_vol) where direction is +1 (long) or -1 (short).
    Returns (None, None) if fewer than SLOW_SPAN bars are available.
    """
    if len(bars) < SLOW_SPAN:
        return None, None
    closes   = pd.Series([b.close for b in bars], dtype=float)
    fast_ema = closes.ewm(span=FAST_SPAN, min_periods=FAST_SPAN).mean()
    slow_ema = closes.ewm(span=SLOW_SPAN, min_periods=SLOW_SPAN).mean()
    if pd.isna(fast_ema.iloc[-1]) or pd.isna(slow_ema.iloc[-1]):
        return None, None
    direction = 1 if fast_ema.iloc[-1] > slow_ema.iloc[-1] else -1
    raw_vol   = closes.pct_change().ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std().iloc[-1]
    ann_vol   = max(VOL_FLOOR, raw_vol * np.sqrt(256))
    return direction, ann_vol


# ── Position sizing ────────────────────────────────────────────────────────────

def ibs_size(equity, price, multiplier):
    return max(1, round(equity * IBS_PER_INSTR / (price * multiplier)))

def ewmac_size(equity, price, multiplier, ann_vol):
    target = (equity * EWMAC_PER_INSTR * RISK_TARGET) / (price * multiplier * ann_vol)
    return max(1, round(target))


# ── EWMAC state persistence ────────────────────────────────────────────────────

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {sym: {'direction': None, 'entry_date': None, 'contracts': 0}
            for sym in CONTRACT_SPECS}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
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
    for sym, spec in CONTRACT_SPECS.items():
        raw       = Future(symbol=spec['ibkr_symbol'],
                           lastTradeDateOrContractMonth=contract_months[sym],
                           exchange=spec['exchange'], currency='USD')
        qualified = ib.qualifyContracts(raw)
        if not qualified:
            print(f"WARNING: Could not qualify {sym} — skipping.")
            continue
        contracts[sym] = qualified[0]

    if not contracts:
        print("ERROR: No contracts qualified.")
        ib.disconnect()
        return

    equity = get_equity(ib)
    if equity is None:
        print("WARNING: Could not read account equity. Using $50,000.")
        equity = 50_000.0

    ibkr_positions = get_positions(ib)
    ewmac_state    = load_state()
    state_changed  = False
    today          = datetime.now().strftime('%Y-%m-%d')

    # Fetch all data upfront
    bars_today   = {}
    bars_history = {}
    for sym, contract in contracts.items():
        bar = get_daily_bar(ib, contract)
        if bar:
            bars_today[sym] = bar
        history = get_long_history(ib, contract)
        if history:
            bars_history[sym] = history

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print("=" * 76)
    print(f"  IBS + EWMAC  |  {today}")
    print(f"  Equity: ${equity:,.2f}   "
          f"IBS alloc: ${equity * IBS_ALLOC:,.0f}   "
          f"EWMAC alloc: ${equity * EWMAC_ALLOC:,.0f}")
    print(f"  MOC cutoffs: MES/MNQ 3:45 PM ET  |  MGC 1:15 PM ET")
    print("=" * 76)

    ibs_orders   = []
    ewmac_orders = []

    for sym, contract in contracts.items():
        spec    = CONTRACT_SPECS[sym]
        mul     = spec['multiplier']
        net_pos = ibkr_positions.get(sym, 0)

        # Infer IBS-held contracts from net position minus known EWMAC state
        st          = ewmac_state.get(sym, {})
        prev_dir    = st.get('direction')
        ewmac_held  = st.get('contracts', 0) * (prev_dir or 0)
        ibs_held    = net_pos - ewmac_held

        print(f"\n  {'─'*72}")
        print(f"  {sym}  {spec['name']}  ({spec['ibkr_symbol']} {contract_months[sym]})"
              f"  |  net pos: {net_pos:+d}  (IBS: {ibs_held:+d}  EWMAC: {ewmac_held:+d})")

        # ── IBS signal ────────────────────────────────────────────────────────
        bar = bars_today.get(sym)
        if bar is not None:
            h, l, c = bar.high, bar.low, bar.close
            rng      = h - l
            ibs      = (c - l) / rng if rng > 0 else 0.5
            ibs_tgt  = ibs_size(equity, c, mul)

            print(f"  IBS   H {h:.2f}  L {l:.2f}  C {c:.2f}"
                  f"  |  IBS: {ibs:.3f}  |  target size: {ibs_tgt} contract(s)")

            if ibs_held == 0 and ibs < IBS_ENTRY:
                print(f"  *** IBS ENTRY  (IBS {ibs:.3f} < {IBS_ENTRY})")
                print(f"  ACTION: BUY {ibs_tgt} {spec['ibkr_symbol']}"
                      f"  — MOC, cutoff {spec['moc_cutoff']}")
                ibs_orders.append(
                    (f"BUY  {ibs_tgt:>2}  {spec['ibkr_symbol']}  IBS entry",
                     spec['moc_cutoff']))
            elif ibs_held > 0 and ibs > IBS_EXIT:
                print(f"  *** IBS EXIT   (IBS {ibs:.3f} > {IBS_EXIT})")
                print(f"  ACTION: SELL {ibs_held} {spec['ibkr_symbol']}"
                      f"  — MOC, cutoff {spec['moc_cutoff']}")
                ibs_orders.append(
                    (f"SELL {ibs_held:>2}  {spec['ibkr_symbol']}  IBS exit",
                     spec['moc_cutoff']))
            elif ibs_held > 0:
                print(f"  IBS: HOLD long ({ibs_held} contract(s))")
            else:
                print(f"  IBS: no signal — flat")
        else:
            print(f"  IBS: no bar data available")

        # ── EWMAC signal ──────────────────────────────────────────────────────
        history = bars_history.get(sym)
        if history:
            direction, ann_vol = compute_ewmac(history)
            if direction is not None:
                price     = bars_today[sym].close if sym in bars_today else history[-1].close
                ew_tgt    = ewmac_size(equity, price, mul, ann_vol)
                dir_str   = "LONG" if direction == 1 else "SHORT"

                print(f"  EWMAC EMA({FAST_SPAN},{SLOW_SPAN}): {dir_str}"
                      f"  |  ann_vol: {ann_vol:.1%}  |  target: {ew_tgt} contract(s)")

                if prev_dir is None:
                    # First run — initialize and enter
                    action = "BUY" if direction == 1 else "SELL"
                    print(f"  *** EWMAC INIT — no prior state, entering {dir_str}")
                    print(f"  ACTION: {action} {ew_tgt} {spec['ibkr_symbol']}"
                          f"  — MOC today ({dir_str} entry)")
                    ewmac_orders.append(
                        (f"{action} {ew_tgt:>2}  {spec['ibkr_symbol']}  EWMAC init {dir_str}",
                         "today MOC"))
                    ewmac_state[sym] = {
                        'direction':  direction,
                        'entry_date': today,
                        'contracts':  ew_tgt,
                    }
                    state_changed = True

                elif direction != prev_dir:
                    # Crossover — flip position
                    prev_str  = "LONG" if prev_dir == 1 else "SHORT"
                    old_net   = st.get('contracts', 1) * prev_dir
                    new_net   = ew_tgt * direction
                    trade_qty = abs(new_net - old_net)
                    action    = "BUY" if direction == 1 else "SELL"
                    print(f"  *** EWMAC CROSSOVER  {prev_str} → {dir_str}")
                    print(f"  ACTION: {action} {trade_qty} {spec['ibkr_symbol']}"
                          f"  — MOC today (flip to {dir_str})")
                    ewmac_orders.append(
                        (f"{action} {trade_qty:>2}  {spec['ibkr_symbol']}  EWMAC flip → {dir_str}",
                         "today MOC"))
                    ewmac_state[sym] = {
                        'direction':  direction,
                        'entry_date': today,
                        'contracts':  ew_tgt,
                    }
                    state_changed = True

                else:
                    entry = st.get('entry_date', '?')
                    held  = st.get('contracts', '?')
                    print(f"  EWMAC: HOLD {dir_str}  ({held} contract(s) since {entry})")
            else:
                print(f"  EWMAC: insufficient history ({len(history)} bars, need {SLOW_SPAN})")
        else:
            print(f"  EWMAC: no history data")

    # ── Persist updated state ─────────────────────────────────────────────────
    if state_changed:
        save_state(ewmac_state)
        print(f"\n  [ewmac_state.json updated]")

    # ── Order summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 76)
    all_orders = [(o, t, 'IBS')   for o, t in ibs_orders] + \
                 [(o, t, 'EWMAC') for o, t in ewmac_orders]
    if all_orders:
        print("  ORDERS TO PLACE:")
        for order, cutoff, strat in all_orders:
            print(f"    [{strat:<5}]  {order:<40}  cutoff: {cutoff}")
        print()
        print("  IBS orders:   MOC — fills at official settlement price")
        print("  EWMAC orders: MOC same day — position held for months until next crossover")
    else:
        print("  NO ORDERS TODAY — hold all positions")
    print("=" * 76)
    print()

    ib.disconnect()


if __name__ == '__main__':
    main()
