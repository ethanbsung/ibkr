#!/usr/bin/env python
import pandas as pd
import numpy as np
from ib_insync import *
import logging
import sys
import json
import os
import time
from datetime import datetime, timedelta
import pytz

# -------------------------------
# Configuration Parameters
# -------------------------------
IB_HOST = '127.0.0.1'
IB_PORT = 4002
CLIENT_ID = 1

# -------------------------------
# Dynamic Allocation Parameters (matching aggregate_port.py exactly)
# -------------------------------
# Note: Commissions are automatically handled by IBKR in live trading

# Target percentage allocations (must sum to 100%)
allocation_percentages = {
    'IBS_ES': 0.10,    # 10% to ES IBS strategy
    'IBS_YM': 0.10,    # 10% to YM IBS strategy  
    'IBS_GC': 0.10,    # 10% to GC IBS strategy
    'IBS_NQ': 0.10,    # 10% to NQ IBS strategy
    'IBS_ZQ': 0.10,    # 10% to ZQ IBS strategy
    'Williams': 0.50   # 50% to Williams strategy
}

# Rebalancing parameters
rebalance_threshold = 0.05  # Rebalance when allocation drifts >5% from target
rebalance_frequency_days = 30  # Also rebalance monthly regardless

# Contract Specifications and Multipliers (matching aggregate_port.py exactly)
contract_specs = {
    'ES': {'multiplier': 5, 'contract_month': '202506', 'exchange': 'CME'},      # MES multiplier
    'YM': {'multiplier': 0.50, 'contract_month': '202506', 'exchange': 'CBOT'},   # MYM multiplier  
    'GC': {'multiplier': 10, 'contract_month': '202506', 'exchange': 'COMEX'},     # MGC multiplier
    'NQ': {'multiplier': 2, 'contract_month': '202506', 'exchange': 'CME'},      # MNQ multiplier
    'ZQ': {'multiplier': 4167, 'contract_month': '202506', 'exchange': 'CBOT'}    # ZQ multiplier
}

# IBS entry/exit thresholds (common for all IBS instruments)
ibs_entry_threshold = 0.1       # Enter when IBS < 0.1
ibs_exit_threshold  = 0.9       # Exit when IBS > 0.9

# Williams %R strategy parameters (applied to ES only)
williams_period = 2             # 2-day lookback
wr_buy_threshold  = -90
wr_sell_threshold = -30

# -------------------------------
# Dynamic Position Sizing Functions (matching aggregate_port.py exactly)
# -------------------------------
def calculate_position_size(current_equity, target_allocation_pct, price, multiplier, min_contracts=2):
    """
    Calculate number of contracts based on current equity and target allocation.
    
    Args:
        current_equity: Current account equity
        target_allocation_pct: Target percentage allocation (0.0 to 1.0)
        price: Current price of the instrument
        multiplier: Contract multiplier
        min_contracts: Minimum number of contracts (default 2)
    
    Returns:
        Number of contracts to trade
    """
    target_dollar_amount = current_equity * target_allocation_pct
    contract_value = price * multiplier
    
    if contract_value <= 0:
        return min_contracts
    
    calculated_contracts = target_dollar_amount / contract_value
    
    # Round to nearest integer, minimum specified contracts
    contracts = max(min_contracts, round(calculated_contracts))
    
    return int(contracts)

def check_rebalancing_needed(current_allocations, target_allocations, threshold=0.05):
    """
    Check if rebalancing is needed based on allocation drift.
    
    Args:
        current_allocations: Dict of current allocations by strategy
        target_allocations: Dict of target allocations by strategy  
        threshold: Drift threshold (default 5%)
    
    Returns:
        Boolean indicating if rebalancing is needed
    """
    for strategy in target_allocations:
        current_pct = current_allocations.get(strategy, 0)
        target_pct = target_allocations[strategy]
        drift = abs(current_pct - target_pct)
        
        if drift > threshold:
            return True
    
    return False

# -------------------------------
# State Management
# -------------------------------
STATE_FILE = 'portfolio_state.json'

def load_portfolio_state():
    """Load portfolio state from file"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    
    # Default state - current_equity will be set from IBKR account
    return {
        'positions': {strategy: {'in_position': False, 'position': None} for strategy in allocation_percentages},
        'last_rebalance_date': None,
        'current_equity': None  # Will be populated from IBKR account
    }

def save_portfolio_state(state):
    """Save portfolio state to file"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

# -------------------------------
# Helper Functions
# -------------------------------
def format_end_datetime(dt, tz):
    """Format the end datetime in UTC using the format yyyymmdd-HH:MM:SS."""
    dt = dt.replace(hour=23, minute=59, second=59, microsecond=0)
    dt_utc = dt.astimezone(pytz.UTC)
    return dt_utc.strftime("%Y%m%d-%H:%M:%S")

def get_daily_bar(ib, contract, end_datetime):
    """Request the most recent completed daily bar for the contract."""
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end_datetime,
        durationStr='2 D',  # Get 2 days to ensure we have yesterday's data
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    return bars if bars else []



def get_account_equity(ib):
    """Get current account equity from IBKR"""
    try:
        account_values = ib.accountValues()
        for value in account_values:
            if value.tag == 'NetLiquidation' and value.currency == 'USD':
                return float(value.value)
    except:
        pass
    return None

def compute_ibs(bar):
    """Calculate IBS with safety check for zero range (matching aggregate_port.py exactly)"""
    range_val = bar.high - bar.low
    if range_val == 0:
        return 0.5  # Neutral IBS when no range
    else:
        return (bar.close - bar.low) / range_val

def compute_williams_r(bars):
    """
    Compute Williams %R using bars with safety check (matching aggregate_port.py exactly)
    Formula: -100 * (highestHigh - current_close) / (highestHigh - lowestLow)
    """
    if len(bars) < williams_period:
        return -50  # Neutral value
    
    # Use last 2 bars
    recent_bars = bars[-williams_period:]
    highest_high = max(bar.high for bar in recent_bars)
    lowest_low = min(bar.low for bar in recent_bars)
    current_close = recent_bars[-1].close
    
    range_val = highest_high - lowest_low
    if range_val == 0:
        return -50  # Neutral Williams %R when no range
    else:
        return -100 * (highest_high - current_close) / range_val

def qualify_contracts(ib):
    """Qualify all trading contracts"""
    contracts = {}
    
    # Create futures contracts
    mes_contract = Future(symbol='MES', lastTradeDateOrContractMonth=contract_specs['ES']['contract_month'], 
                         exchange=contract_specs['ES']['exchange'], currency='USD')
    mym_contract = Future(symbol='MYM', lastTradeDateOrContractMonth=contract_specs['YM']['contract_month'], 
                         exchange=contract_specs['YM']['exchange'], currency='USD')
    mgc_contract = Future(symbol='MGC', lastTradeDateOrContractMonth=contract_specs['GC']['contract_month'], 
                         exchange=contract_specs['GC']['exchange'], currency='USD')
    mnq_contract = Future(symbol='MNQ', lastTradeDateOrContractMonth=contract_specs['NQ']['contract_month'], 
                         exchange=contract_specs['NQ']['exchange'], currency='USD')
    zq_contract = Future(symbol='ZQ', lastTradeDateOrContractMonth=contract_specs['ZQ']['contract_month'], 
                        exchange=contract_specs['ZQ']['exchange'], currency='USD')
    
    # Qualify all contracts
    try:
        qualified = ib.qualifyContracts(mes_contract, mym_contract, mgc_contract, mnq_contract, zq_contract)
        if len(qualified) != 5:
            raise ValueError(f"Only {len(qualified)} out of 5 contracts qualified")
        
        contracts['ES'] = qualified[0]
        contracts['YM'] = qualified[1] 
        contracts['GC'] = qualified[2]
        contracts['NQ'] = qualified[3]
        contracts['ZQ'] = qualified[4]
        
        return contracts
    except Exception as e:
        logging.error(f"Contract qualification error: {e}")
        return None

def wait_until_close(target_hour=17, target_minute=0, timezone='US/Eastern', lead_seconds=10):
    """Wait until the specified time minus lead_seconds."""
    tz = pytz.timezone(timezone)
    while True:
        now = datetime.now(tz)
        close_today = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        target_time = close_today - timedelta(seconds=lead_seconds)
        
        if now >= target_time:
            return
        else:
            sleep_time = (target_time - now).total_seconds()
            if sleep_time < 0:
                sleep_time = 30  # fallback
            time.sleep(sleep_time)

# -------------------------------
# Main Trading Logic (matching aggregate_port.py exactly)
# -------------------------------
def run_daily_signals(ib):
    logger = logging.getLogger()
    
    # Load current portfolio state
    state = load_portfolio_state()
    
    # Qualify all contracts
    contracts = qualify_contracts(ib)
    if contracts is None:
        logger.error("Failed to qualify contracts")
        return
    
    logger.info("All contracts qualified successfully")
    
    # Get current account equity
    current_equity = get_account_equity(ib)
    if current_equity is None:
        if state['current_equity'] is not None:
            logger.warning("Could not get account equity from IBKR, using stored value")
            current_equity = state['current_equity']
        else:
            logger.error("Could not get account equity from IBKR and no stored value available")
            return
    else:
        state['current_equity'] = current_equity
        logger.info(f"Current account equity: ${current_equity:,.2f}")
    
    tz = pytz.timezone('US/Eastern')
    current_dt = datetime.now(tz)
    end_datetime_str = format_end_datetime(current_dt, tz)
    
    # Process each IBS strategy (matching aggregate_port.py logic exactly)
    for symbol in ['ES', 'YM', 'GC', 'NQ', 'ZQ']:
        strategy_key = f'IBS_{symbol}'
        contract = contracts[symbol]
        multiplier = contract_specs[symbol]['multiplier']
        
        logger.info(f"\nProcessing {strategy_key}...")
        
        # Get recent bars for IBS calculation
        bars = get_daily_bar(ib, contract, end_datetime_str)
        if not bars:
            logger.warning(f"No bars available for {symbol}")
            continue
        
        # Use most recent bar
        current_bar = bars[-1]
        current_price = current_bar.close
        
        # Calculate IBS with safety check (matching aggregate_port.py exactly)
        ibs = compute_ibs(current_bar)
        
        logger.info(f"{symbol} - Price: {current_price}, High: {current_bar.high}, Low: {current_bar.low}, IBS: {ibs:.3f}")
        
        # Calculate position size based on current equity and allocation
        target_contracts = calculate_position_size(
            current_equity, 
            allocation_percentages[strategy_key], 
            current_price, 
            multiplier
        )
        
        logger.info(f"{symbol} - Target contracts: {target_contracts}")
        
        # Get current position state
        strategy_state = state['positions'][strategy_key]
        
        # Execute IBS trading logic (matching aggregate_port.py exactly)
        if strategy_state['in_position']:
            if ibs > ibs_exit_threshold:
                # Exit position
                logger.info(f"{symbol} - IBS exit signal (IBS: {ibs:.3f} > {ibs_exit_threshold})")
                current_contracts = strategy_state['position']['contracts']
                
                order = Order(action='SELL', totalQuantity=current_contracts, orderType='MOC', tif='DAY')
                trade = ib.placeOrder(contract, order)
                
                logger.info(f"Placed IBS MOC SELL order for {current_contracts} {symbol} contracts")
                
                # Update state
                strategy_state['in_position'] = False
                strategy_state['position'] = None
        else:
            if ibs < ibs_entry_threshold:
                # Enter position
                logger.info(f"{symbol} - IBS entry signal (IBS: {ibs:.3f} < {ibs_entry_threshold})")
                
                order = Order(action='BUY', totalQuantity=target_contracts, orderType='MOC', tif='DAY')
                trade = ib.placeOrder(contract, order)
                
                logger.info(f"Placed IBS MOC BUY order for {target_contracts} {symbol} contracts")
                
                # Update state
                strategy_state['in_position'] = True
                strategy_state['position'] = {
                    'entry_price': current_price,
                    'entry_time': current_dt.isoformat(),
                    'contracts': target_contracts
                }
    
    # Process Williams %R strategy (ES only, matching aggregate_port.py logic exactly)
    logger.info(f"\nProcessing Williams strategy...")
    
    es_contract = contracts['ES']
    es_bars = get_daily_bar(ib, es_contract, end_datetime_str)
    
    if len(es_bars) >= williams_period:
        current_bar = es_bars[-1]
        current_price = current_bar.close
        
        # Calculate Williams %R with safety check (matching aggregate_port.py exactly)
        williams_r = compute_williams_r(es_bars)
        
        logger.info(f"ES Williams - Price: {current_price}, Williams %R: {williams_r:.2f}")
        
        # Calculate position size
        target_contracts = calculate_position_size(
            current_equity, 
            allocation_percentages['Williams'], 
            current_price, 
            contract_specs['ES']['multiplier']
        )
        
        logger.info(f"Williams - Target contracts: {target_contracts}")
        
        # Get current position state
        williams_state = state['positions']['Williams']
        
        # Execute Williams trading logic (matching aggregate_port.py exactly)
        if williams_state['in_position']:
            # Check exit conditions: current_price > yesterday's high OR williams_r > sell_threshold
            exit_signal = False
            if len(es_bars) >= 2:
                yesterdays_high = es_bars[-2].high
                if current_price > yesterdays_high:
                    logger.info(f"Williams exit signal - Price {current_price} > Yesterday's high {yesterdays_high}")
                    exit_signal = True
            
            if williams_r > wr_sell_threshold:
                logger.info(f"Williams exit signal - Williams %R {williams_r:.2f} > {wr_sell_threshold}")
                exit_signal = True
            
            if exit_signal:
                current_contracts = williams_state['position']['contracts']
                
                order = Order(action='SELL', totalQuantity=current_contracts, orderType='MOC', tif='DAY')
                trade = ib.placeOrder(es_contract, order)
                
                logger.info(f"Placed Williams MOC SELL order for {current_contracts} ES contracts")
                
                # Update state
                williams_state['in_position'] = False
                williams_state['position'] = None
        else:
            if williams_r < wr_buy_threshold:
                # Enter position
                logger.info(f"Williams entry signal - Williams %R {williams_r:.2f} < {wr_buy_threshold}")
                
                order = Order(action='BUY', totalQuantity=target_contracts, orderType='MOC', tif='DAY')
                trade = ib.placeOrder(es_contract, order)
                
                logger.info(f"Placed Williams MOC BUY order for {target_contracts} ES contracts")
                
                # Update state
                williams_state['in_position'] = True
                williams_state['position'] = {
                    'entry_price': current_price,
                    'entry_time': current_dt.isoformat(),
                    'contracts': target_contracts
                }
    else:
        logger.warning("Insufficient Williams data - need at least 2 bars")
    
    # Save updated state
    save_portfolio_state(state)
    
    # Log current positions
    logger.info(f"\n=== CURRENT PORTFOLIO STATE ===")
    logger.info(f"Account Equity: ${current_equity:,.2f}")
    for strategy, strategy_state in state['positions'].items():
        if strategy_state['in_position']:
            pos = strategy_state['position']
            logger.info(f"{strategy}: IN POSITION - {pos['contracts']} contracts @ {pos['entry_price']}")
        else:
            logger.info(f"{strategy}: NO POSITION")

def print_portfolio_summary(ib):
    """Print portfolio summary and recent OHLC data"""
    logger = logging.getLogger()
    
    # Load current state
    state = load_portfolio_state()
    
    logger.info("=== PORTFOLIO SUMMARY ===")
    logger.info("Dynamic Percentage-Based Allocation:")
    for strategy, pct in allocation_percentages.items():
        logger.info(f"  • {strategy}: {pct*100:.0f}%")
    
    # Get current equity from IBKR if not available in state
    current_equity = state['current_equity']
    if current_equity is None:
        current_equity = get_account_equity(ib)
        if current_equity is not None:
            state['current_equity'] = current_equity
            save_portfolio_state(state)
    
    if current_equity is not None:
        logger.info(f"Current Equity: ${current_equity:,.2f}")
    else:
        logger.info("Current Equity: Unable to retrieve from IBKR")
    
    # Qualify contracts and get recent data
    contracts = qualify_contracts(ib)
    if contracts:
        tz = pytz.timezone('US/Eastern')
        end_datetime_str = format_end_datetime(datetime.now(tz), tz)
        
        logger.info("\nRecent OHLC data:")
        for symbol, contract in contracts.items():
            bars = get_daily_bar(ib, contract, end_datetime_str)
            if bars:
                recent_bar = bars[-1]
                logger.info(f"{symbol}: Date: {recent_bar.date}, O: {recent_bar.open}, H: {recent_bar.high}, L: {recent_bar.low}, C: {recent_bar.close}")

def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    
    ib = IB()
    
    # Connect to IBKR
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
        logger.info("Connected to IBKR.")
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return

    # Print portfolio summary on startup
    print_portfolio_summary(ib)

    # Wait until 5 seconds before market close (5 PM Eastern)
    logger.info("Waiting until 5 seconds before market close...")
    wait_until_close(target_hour=17, target_minute=0, timezone='US/Eastern', lead_seconds=5)

    # Run daily signals with exact logic from aggregate_port.py
    logger.info("Running daily signal generation...")
    run_daily_signals(ib)

    # Disconnect
    ib.disconnect()
    logger.info("Finished running daily signals. Disconnected from IBKR.")

if __name__ == '__main__':
    main()