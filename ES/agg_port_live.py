#!/usr/bin/env python
from ib_insync import *
import logging
import sys
import json
import os
import time
from datetime import datetime, timedelta
import pytz
from collections import defaultdict

# -------------------------------
# Configuration Parameters
# -------------------------------
IB_HOST = '127.0.0.1'
IB_PORT = 4002
CLIENT_ID = 1

# -------------------------------
# Dynamic Allocation Parameters (matching aggregate_port.py exactly)
# -------------------------------
# Risk multiplier for larger position sizes (matching enhanced backtest)
risk_multiplier = 3.0              # 3x larger positions for higher risk/reward
# Target percentage allocations (must sum to 100%)
# 50/50 split between IBS and Williams strategies, equal weighting within each
allocation_percentages = {
    'IBS_ES': 0.125,      # 12.5% to ES IBS strategy
    'IBS_YM': 0.125,      # 12.5% to YM IBS strategy  
    'IBS_GC': 0.125,      # 12.5% to GC IBS strategy
    'IBS_NQ': 0.125,      # 12.5% to NQ IBS strategy
    'Williams_ES': 0.125, # 12.5% to ES Williams strategy
    'Williams_YM': 0.125, # 12.5% to YM Williams strategy
    'Williams_GC': 0.125, # 12.5% to GC Williams strategy
    'Williams_NQ': 0.125  # 12.5% to NQ Williams strategy
}

# Note: Position sizing automatically maintains target allocations as equity changes

# Contract Specifications and Multipliers (matching aggregate_port.py exactly)
contract_specs = {
    'ES': {'multiplier': 5, 'contract_month': '202506', 'exchange': 'CME'},      # MES multiplier
    'YM': {'multiplier': 0.50, 'contract_month': '202506', 'exchange': 'CBOT'},   # MYM multiplier  
    'GC': {'multiplier': 10, 'contract_month': '202508', 'exchange': 'COMEX'},     # MGC multiplier - moved to August to avoid delivery window
    'NQ': {'multiplier': 2, 'contract_month': '202506', 'exchange': 'CME'}      # MNQ multiplier
}

# IBS entry/exit thresholds
ibs_entry_threshold = 0.1       # Enter when IBS < 0.1
ibs_exit_threshold  = 0.9       # Exit when IBS > 0.9

# Williams %R strategy parameters
williams_period = 2             # 2-day lookback
wr_buy_threshold  = -90
wr_sell_threshold = -30

# -------------------------------
# Dynamic Position Sizing Functions (matching aggregate_port.py exactly)
# -------------------------------
def calculate_position_size(current_equity, target_allocation_pct, price, multiplier, min_contracts=1):
    """
    Calculate number of contracts based on current equity and target allocation with enhanced risk.
    
    Args:
        current_equity: Current account equity
        target_allocation_pct: Target percentage allocation (0.0 to 1.0)
        price: Current price of the instrument
        multiplier: Contract multiplier
        min_contracts: Minimum number of contracts (default 1)
    
    Returns:
        Number of contracts to trade
    """
    target_dollar_amount = current_equity * target_allocation_pct * risk_multiplier
    contract_value = price * multiplier
    
    if contract_value <= 0:
        return min_contracts
    
    calculated_contracts = target_dollar_amount / contract_value
    
    # Round to nearest integer, minimum specified contracts
    contracts = max(min_contracts, round(calculated_contracts))
    
    return int(contracts)



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
        durationStr='5 D',
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=False,
        formatDate=1
    )
    return bars if bars else []

def get_williams_bars(ib, contract, end_datetime):
    """Request sufficient daily bars for Williams %R calculation (minimum 2 bars needed)."""
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end_datetime,
        durationStr='1 W',
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=False,
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

def get_positions_from_ibkr(ib, contracts):
    """Get current positions from IBKR API and map them to our strategy names"""
    logger = logging.getLogger()
    
    try:
        # Get all positions from IBKR
        positions = ib.positions()
        
        # Get current market prices from daily bars (same reliable source as signals)
        tz = pytz.timezone('US/Eastern')
        current_dt = datetime.now(tz)
        end_datetime_str = format_end_datetime(current_dt, tz)
        
        current_prices = {}
        for symbol, contract in contracts.items():
            try:
                bars = get_daily_bar(ib, contract, end_datetime_str)
                if bars and len(bars) > 0:
                    current_prices[symbol] = bars[-1].close
                    logger.info(f"Current {symbol} price from daily bar: {current_prices[symbol]}")
            except Exception as e:
                logger.warning(f"Could not get current price for {symbol}: {e}")
        
        # Create a mapping from contract to strategy
        contract_to_strategy = {}
        for symbol, contract in contracts.items():
            # All contracts are used for both IBS and Williams strategies
            contract_to_strategy[contract.conId] = [f'IBS_{symbol}', f'Williams_{symbol}']
        
        # Initialize position tracking
        ibkr_positions = {strategy: {'in_position': False, 'position': None} for strategy in allocation_percentages}

        # Load saved state once for position mapping
        saved_state = load_portfolio_state()

        # Process IBKR positions
        for position in positions:
            if position.contract.conId in contract_to_strategy:
                strategies = contract_to_strategy[position.contract.conId]
                
                if position.position != 0:  # Non-zero position
                    # Get the current market price for entry_price estimation
                    symbol = position.contract.symbol
                    if symbol == 'MES':
                        symbol = 'ES'
                    elif symbol == 'MYM':
                        symbol = 'YM'
                    elif symbol == 'MGC':
                        symbol = 'GC'
                    elif symbol == 'MNQ':
                        symbol = 'NQ'
                    
                    # Use current price from daily bars we already retrieved
                    if symbol in current_prices:
                        entry_price = current_prices[symbol]
                        logger.info(f"Using current market price {entry_price} for {symbol} position")
                    else:
                        # Fallback to reasonable estimates if daily bar data not available
                        logger.warning(f"No current price data for {symbol}, using default estimate")
                        default_prices = {'ES': 6000, 'GC': 3300, 'YM': 42000, 'NQ': 21000}
                        entry_price = default_prices.get(symbol, 1000)
                    
                    position_info = {
                        'entry_price': entry_price,
                        'entry_time': datetime.now().isoformat(),  # We don't know actual entry time
                        'contracts': int(abs(position.position))  # Use absolute value for long positions
                    }
                    
                    # For all instruments, determine which strategy has the position using loaded state
                    # Check which strategies expect to have positions for this instrument
                    ibs_strategy = f'IBS_{symbol}'
                    williams_strategy = f'Williams_{symbol}'
                    
                    ibs_expecting = saved_state['positions'][ibs_strategy]['in_position']
                    williams_expecting = saved_state['positions'][williams_strategy]['in_position']
                    
                    if ibs_expecting and not williams_expecting:
                        # IBS strategy should have the position
                        ibkr_positions[ibs_strategy] = {
                            'in_position': True,
                            'position': position_info
                        }
                        logger.info(f"Found {ibs_strategy} position: {position_info['contracts']} contracts")
                    elif williams_expecting and not ibs_expecting:
                        # Williams strategy should have the position
                        ibkr_positions[williams_strategy] = {
                            'in_position': True,
                            'position': position_info
                        }
                        logger.info(f"Found {williams_strategy} position: {position_info['contracts']} contracts")
                    elif ibs_expecting and williams_expecting:
                        # Both expect positions - distribute based on saved position sizes
                        # This would be handled by the router state, but for now assign to IBS by default
                        logger.warning(f"Both {ibs_strategy} and {williams_strategy} expect positions but only one exists. Assigning to {ibs_strategy} by default.")
                        ibkr_positions[ibs_strategy] = {
                            'in_position': True,
                            'position': position_info
                        }
                        logger.info(f"Found {ibs_strategy} position: {position_info['contracts']} contracts")
                    else:
                        # Neither expects a position, default to IBS strategy
                        logger.warning(f"Found unexpected {symbol} position, assigning to {ibs_strategy} by default.")
                        ibkr_positions[ibs_strategy] = {
                            'in_position': True,
                            'position': position_info
                        }
                        logger.info(f"Found {ibs_strategy} position: {position_info['contracts']} contracts")
        
        return ibkr_positions
        
    except Exception as e:
        logger.error(f"Error getting positions from IBKR: {e}")
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
    
    # Qualify all contracts
    try:
        qualified = ib.qualifyContracts(mes_contract, mym_contract, mgc_contract, mnq_contract)
        if len(qualified) != 4:
            raise ValueError(f"Only {len(qualified)} out of 4 contracts qualified")
        
        contracts['ES'] = qualified[0]
        contracts['YM'] = qualified[1] 
        contracts['GC'] = qualified[2]
        contracts['NQ'] = qualified[3]
        
        return contracts
    except Exception as e:
        logging.error(f"Contract qualification error: {e}")
        return None

def place_order_with_fallback(ib, contract, action, quantity, symbol, dry_run=False):
    """
    Place an order with fallback from MOC to limit order if MOC is not supported.
    Returns the trade object if successful, None if failed or in dry run mode.
    """
    logger = logging.getLogger()
    
    # Check if we're outside the trading window (4:55-5:05 PM Eastern) - use dry run mode
    if not is_trading_window() or dry_run:
        logger.info(f"DRY RUN: Would place MOC {action} order for {quantity} {symbol} contracts (outside trading window)")
        # Return None for dry run mode - do not update position state
        return None
    
    # For other instruments, try MOC first
    try:
        order = Order(action=action, totalQuantity=quantity, orderType='MOC', tif='DAY')
        trade = ib.placeOrder(contract, order)
        logger.info(f"Placed MOC {action} order for {quantity} {symbol} contracts")
        
        # Wait a moment to see if there are any immediate errors
        ib.sleep(1)
        
        # Check if order was rejected
        if trade.orderStatus.status == 'Cancelled':
            for log_entry in trade.log:
                if '387' in log_entry.message or 'Unsupported order type' in log_entry.message:
                    logger.warning(f"MOC order rejected for {symbol}, falling back to limit order")
                    return place_limit_order(ib, contract, action, quantity, symbol)
                elif '201' in log_entry.message or 'physical delivery' in log_entry.message.lower():
                    logger.error(f"Order rejected for {symbol}: Contract in delivery window or near expiration")
                    return None
        
        return trade
    except Exception as e:
        logger.error(f"Order placement failed for {symbol}: {e}")
        return None

def place_limit_order(ib, contract, action, quantity, symbol):
    """Place a limit order near current market price"""
    logger = logging.getLogger()
    
    # Check if we're outside the trading window (4:55-5:05 PM Eastern) - use dry run mode
    if not is_trading_window():
        logger.info(f"DRY RUN: Would place LMT {action} order for {quantity} {symbol} contracts (outside trading window)")
        # Return None for dry run mode - do not update position state
        return None
    
    # Get current price
    ticker = ib.reqMktData(contract, '', False, False)
    ib.sleep(2)  # Wait for price data
    
    current_price = None
    if ticker.last and ticker.last > 0:
        current_price = ticker.last
    elif ticker.close and ticker.close > 0:
        current_price = ticker.close
    elif ticker.bid and ticker.ask:
        current_price = (ticker.bid + ticker.ask) / 2
    
    # Cancel market data subscription
    ib.cancelMktData(ticker)
    
    if current_price:
        # Set limit price slightly favorable (0.1% buffer)
        if action == 'BUY':
            limit_price = current_price * 1.001  # Slightly above current price
        else:  # SELL
            limit_price = current_price * 0.999  # Slightly below current price
        
        # Define minimum tick sizes for each symbol
        tick_sizes = {
            'ES': 0.25,      # MES / ES trades in 0.25-point increments
            'NQ': 0.25,      # MNQ / NQ trades in 0.25-point increments
            'YM': 1.0,       # MYM trades in whole-point increments (1-point = $0.50)
            'GC': 0.1        # MGC trades in 0.1-point increments
        }

        tick_size = tick_sizes.get(symbol, 0.25)  # Default to 0.25 if unknown
        limit_price = round(limit_price / tick_size) * tick_size
        
        try:
            order = Order(action=action, totalQuantity=quantity, orderType='LMT', 
                         lmtPrice=limit_price, tif='DAY')
            trade = ib.placeOrder(contract, order)
            logger.info(f"Placed LMT {action} order for {quantity} {symbol} contracts at {limit_price}")
            
            # Wait a moment to check for immediate rejection
            ib.sleep(1)
            
            # Check if order was rejected due to delivery window or other issues
            if trade.orderStatus.status == 'Cancelled':
                for log_entry in trade.log:
                    if '201' in log_entry.message or 'physical delivery' in log_entry.message.lower():
                        logger.error(f"Limit order rejected for {symbol}: Contract in delivery window or near expiration")
                        return None
            
            return trade
        except Exception as e:
            logger.error(f"Limit order placement failed for {symbol}: {e}")
            return None
    else:
        logger.error(f"Could not get current price for {symbol} limit order")
        return None

def is_market_hours(timezone='US/Eastern'):
    """Check if futures markets are currently open (6 PM Sunday - 5 PM Friday Eastern)"""
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    
    # Market hours: Sunday 6 PM - Friday 5 PM Eastern (futures markets)
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    hour = now.hour
    minute = now.minute
    
    # Saturday - market closed all day
    if weekday == 5:  # Saturday
        return False
    
    # Sunday before 6 PM - market closed
    if weekday == 6 and hour < 18:  # Sunday
        return False
    
    # Monday-Thursday after 5 PM - market closed until 6 PM
    if weekday in [0, 1, 2, 3] and hour >= 17:  # Monday-Thursday
        if hour > 18 or (hour == 18 and minute >= 0):  # After 6 PM, market reopens
            return True
        else:  # Between 5 PM and 6 PM, market closed
            return False
    
    # Friday after 5 PM - market closed for weekend
    if weekday == 4 and hour >= 17:  # Friday
        return False
    
    # Sunday after 6 PM - market open
    if weekday == 6 and hour >= 18:
        return True
    
    # Monday-Friday during regular hours (before 5 PM) - market open
    if weekday in [0, 1, 2, 3, 4] and hour < 17:
        return True
    
    return False

def is_trading_window(timezone='US/Eastern'):
    """Check if we're in the specific trading window (4:55-5:05 PM Eastern) for placing EOD orders"""
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    hour = now.hour
    minute = now.minute
    
    # Only trade Monday-Friday
    if weekday not in [0, 1, 2, 3, 4]:  # Monday-Friday
        return False
    
    # Only trade between 4:55 PM and 5:05 PM Eastern
    if hour == 16 and minute >= 55:  # 4:55-4:59 PM
        return True
    elif hour == 17 and minute <= 5:  # 5:00-5:05 PM
        return True
    
    return False

def get_next_market_close(timezone='US/Eastern'):
    """Get the next 5 PM Eastern market close on a trading day (Monday-Friday)"""
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    
    # Start with today's 5 PM
    target_time = now.replace(hour=17, minute=0, second=0, microsecond=0)
    
    # If it's already past 5 PM today, move to tomorrow
    if now >= target_time:
        target_time += timedelta(days=1)
    
    # Keep advancing until we hit a weekday (Monday=0, Sunday=6)
    while target_time.weekday() > 4:  # Saturday=5, Sunday=6
        target_time += timedelta(days=1)
    
    return target_time


# ============================================================
# Symbol‑level position router (used where multiple strategies
# trade the same contract – e.g. IBS_ES and Williams on MES)
# ============================================================
ROUTER_STATE_FILE = 'router_state.json'


class PositionRouter:
    """
    Maintains virtual lots per strategy for a single contract and nets those
    into one real IBKR order each time `sync()` is called.
    """
    def __init__(self, ib, contract, symbol='ES'):
        self.ib = ib
        self.contract = contract
        self.symbol = symbol
        self.strategy_lots = self._load_state()

    # ---------- persistence ----------
    def _load_state(self):
        if os.path.exists(ROUTER_STATE_FILE):
            try:
                with open(ROUTER_STATE_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        # default – no virtual positions yet
        return defaultdict(int)

    def _save_state(self):
        with open(ROUTER_STATE_FILE, 'w') as f:
            json.dump(self.strategy_lots, f, indent=2)

    # ---------- live position ----------
    def _live_position(self):
        for p in self.ib.positions():
            if p.contract.conId == self.contract.conId:
                return int(p.position)
        return 0

    # ---------- public ----------
    def sync(self, desired_dict):
        """
        desired_dict: {'IBS_ES': +2, 'Williams': 0, ...}
        Places ONE order to move from current net to desired net.
        """
        live_total = self._live_position()
        desired_total = sum(desired_dict.values())
        net_change = desired_total - live_total

        # no trade needed
        if net_change == 0:
            self.strategy_lots.update(desired_dict)
            self._save_state()
            return

        action = 'BUY' if net_change > 0 else 'SELL'
        qty = abs(net_change)

        # use existing helper with MOC→LMT fallback
        trade = place_order_with_fallback(
            self.ib,
            self.contract,
            action,
            qty,
            self.symbol
        )

        # optimistic virtual‑lot update (we rely on nightly reconciliation)
        if trade is not None:
            self.strategy_lots.update(desired_dict)
            self._save_state()

def wait_until_next_close(timezone='US/Eastern', lead_seconds=10):
    """Wait until the next 5 PM Eastern market close minus lead_seconds."""
    tz = pytz.timezone(timezone)
    logger = logging.getLogger()
    
    target_time = get_next_market_close(timezone) - timedelta(seconds=lead_seconds)
    now = datetime.now(tz)
    
    if now >= target_time:
        logger.info("Already at or past target time, proceeding immediately")
        return
    
    wait_seconds = (target_time - now).total_seconds()
    logger.info(f"Waiting until {target_time.strftime('%Y-%m-%d %H:%M:%S %Z')} ({wait_seconds/3600:.1f} hours)")
    
    # Show periodic updates for long waits
    while True:
        now = datetime.now(tz)
        if now >= target_time:
            break
        
        remaining_seconds = (target_time - now).total_seconds()
        
        # Sleep in chunks, showing progress for long waits
        if remaining_seconds > 3600:  # More than 1 hour
            logger.info(f"Waiting {remaining_seconds/3600:.1f} hours until market close...")
            time.sleep(min(1800, remaining_seconds))  # Sleep up to 30 minutes at a time
        elif remaining_seconds > 300:  # More than 5 minutes
            logger.info(f"Waiting {remaining_seconds/60:.1f} minutes until market close...")
            time.sleep(min(60, remaining_seconds))  # Sleep up to 1 minute at a time
        else:
            time.sleep(remaining_seconds)
            break

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

    # ---------- routers for all instruments (each shared by IBS & Williams) ----------
    routers = {}
    desired_positions = {}  # will hold target lots per instrument per strategy
    
    for symbol in ['ES', 'YM', 'GC', 'NQ']:
        routers[symbol] = PositionRouter(ib, contracts[symbol], symbol)
        desired_positions[symbol] = {}  # will hold IBS and Williams target lots for this symbol
    
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
    
    # Synchronize positions with IBKR
    logger.info("Synchronizing positions with IBKR...")
    ibkr_positions = get_positions_from_ibkr(ib, contracts)
    if ibkr_positions is not None:
        # Compare with local state and update
        position_changes = []
        for strategy in state['positions']:
            local_pos = state['positions'][strategy]
            ibkr_pos = ibkr_positions[strategy]
            
            # Check for differences
            if local_pos['in_position'] != ibkr_pos['in_position']:
                position_changes.append(f"{strategy}: Local={local_pos['in_position']}, IBKR={ibkr_pos['in_position']}")
                state['positions'][strategy] = ibkr_pos
            elif local_pos['in_position'] and ibkr_pos['in_position']:
                # Both have positions, check contract count
                local_contracts = local_pos['position']['contracts']
                ibkr_contracts = ibkr_pos['position']['contracts']
                if local_contracts != ibkr_contracts:
                    position_changes.append(f"{strategy}: Local={local_contracts} contracts, IBKR={ibkr_contracts} contracts")
                    state['positions'][strategy] = ibkr_pos
        
        if position_changes:
            logger.warning("Position differences found between local state and IBKR:")
            for change in position_changes:
                logger.warning(f"  {change}")
            logger.info("Local state updated to match IBKR positions")
        else:
            logger.info("Local positions match IBKR positions")
    else:
        logger.warning("Could not retrieve positions from IBKR, using local state")
    
    tz = pytz.timezone('US/Eastern')
    current_dt = datetime.now(tz)
    end_datetime_str = format_end_datetime(current_dt, tz)
    
    # Process each IBS strategy (router-only approach for all instruments)
    for symbol in ['ES', 'YM', 'GC', 'NQ']:
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

        # Enhanced logging for position sizing
        allocation_pct = allocation_percentages[strategy_key]
        target_dollar = current_equity * allocation_pct * risk_multiplier
        logger.info(f"{symbol} - Allocation: {allocation_pct*100:.0f}% * {risk_multiplier}x = ${target_dollar:,.0f} target")
        logger.info(f"{symbol} - Target contracts: {target_contracts}")

        # Get current position state
        strategy_state = state['positions'][strategy_key]

        # Execute IBS trading logic using router approach
        if strategy_state['in_position']:
            desired_qty = 0 if ibs > ibs_exit_threshold else strategy_state['position']['contracts']
            if ibs > ibs_exit_threshold:
                logger.info(f"{symbol} - IBS exit signal (IBS: {ibs:.3f} > {ibs_exit_threshold})")
            else:
                logger.info(f"{symbol} - Holding position (IBS: {ibs:.3f}, exit threshold: {ibs_exit_threshold})")
        else:
            desired_qty = target_contracts if ibs < ibs_entry_threshold else 0
            if ibs < ibs_entry_threshold:
                logger.info(f"{symbol} - IBS entry signal (IBS: {ibs:.3f} < {ibs_entry_threshold})")
            else:
                logger.info(f"{symbol} - No entry signal (IBS: {ibs:.3f}, entry threshold: {ibs_entry_threshold})")

        desired_positions[symbol][strategy_key] = desired_qty
    
    # Process Williams %R strategy for all instruments
    for symbol in ['ES', 'YM', 'GC', 'NQ']:
        strategy_key = f'Williams_{symbol}'
        contract = contracts[symbol]
        multiplier = contract_specs[symbol]['multiplier']
        
        logger.info(f"\nProcessing {strategy_key}...")

        # Get bars for Williams %R calculation
        bars = get_williams_bars(ib, contract, end_datetime_str)

        logger.info(f"{strategy_key} - Retrieved {len(bars)} bars, need {williams_period}")
        if len(bars) > 0:
            logger.info(f"{strategy_key} - Recent bars: {[f'{bar.date}' for bar in bars[-min(3, len(bars)):]]}")

        if len(bars) >= williams_period:
            current_bar = bars[-1]
            current_price = current_bar.close

            # Calculate Williams %R with safety check (matching aggregate_port.py exactly)
            williams_r = compute_williams_r(bars)

            logger.info(f"{symbol} Williams - Price: {current_price}, Williams %R: {williams_r:.2f}")

            # Calculate position size
            target_contracts = calculate_position_size(
                current_equity,
                allocation_percentages[strategy_key],
                current_price,
                multiplier
            )

            # Enhanced logging for position sizing
            allocation_pct = allocation_percentages[strategy_key]
            target_dollar = current_equity * allocation_pct * risk_multiplier
            logger.info(f"{strategy_key} - Allocation: {allocation_pct*100:.0f}% * {risk_multiplier}x = ${target_dollar:,.0f} target")
            logger.info(f"{strategy_key} - Target contracts: {target_contracts}")

            # Get current position state
            williams_state = state['positions'][strategy_key]

            # Execute Williams trading logic using router approach
            if williams_state['in_position']:
                exit_signal = False
                if len(bars) >= 2 and current_price > bars[-2].high:
                    exit_signal = True
                    logger.info(f"{symbol} Williams - Exit signal: price above previous high")
                if williams_r > wr_sell_threshold:
                    exit_signal = True
                    logger.info(f"{symbol} Williams - Exit signal: Williams %R > {wr_sell_threshold}")
                
                desired_qty = 0 if exit_signal else williams_state['position']['contracts']
                if not exit_signal:
                    logger.info(f"{symbol} Williams - Holding position (Williams %R: {williams_r:.2f})")
            else:
                desired_qty = target_contracts if williams_r < wr_buy_threshold else 0
                if williams_r < wr_buy_threshold:
                    logger.info(f"{symbol} Williams - Entry signal (Williams %R: {williams_r:.2f} < {wr_buy_threshold})")
                else:
                    logger.info(f"{symbol} Williams - No entry signal (Williams %R: {williams_r:.2f}, threshold: {wr_buy_threshold})")

            desired_positions[symbol][strategy_key] = desired_qty
        else:
            logger.warning(f"Insufficient Williams data for {symbol} - need at least {williams_period} bars")
            desired_positions[symbol][strategy_key] = 0

    # ---------- execute net orders for all instruments ----------
    for symbol in ['ES', 'YM', 'GC', 'NQ']:
        routers[symbol].sync(desired_positions[symbol])

        # reflect router outcome in local state (simplified – optimistic)
        current_price = 0  # Will be set from the last processed bar for this symbol
        
        # Get current price for state updates
        bars = get_daily_bar(ib, contracts[symbol], end_datetime_str)
        if bars:
            current_price = bars[-1].close
        
        for strat, lots in routers[symbol].strategy_lots.items():
            st = state['positions'][strat]
            if lots == 0:
                st['in_position'] = False
                st['position'] = None
            else:
                st['in_position'] = True
                if st['position'] is None:
                    st['position'] = {
                        'entry_price': current_price,
                        'entry_time': current_dt.isoformat(),
                        'contracts': lots
                    }
                else:
                    st['position']['contracts'] = lots
    
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
    logger.info("50/50 IBS/Williams Split with Equal Instrument Weighting:")
    
    # Group by strategy type for cleaner display
    ibs_strategies = {k: v for k, v in allocation_percentages.items() if k.startswith('IBS_')}
    williams_strategies = {k: v for k, v in allocation_percentages.items() if k.startswith('Williams_')}
    
    logger.info("  IBS Strategies (50% total):")
    for strategy, pct in sorted(ibs_strategies.items()):
        logger.info(f"    • {strategy}: {pct*100:.1f}%")
    
    logger.info("  Williams Strategies (50% total):")
    for strategy, pct in sorted(williams_strategies.items()):
        logger.info(f"    • {strategy}: {pct*100:.1f}%")
        
    logger.info(f"  • Risk Multiplier: {risk_multiplier}x (LARGER POSITION SIZES)")
    logger.info("  • Enhanced risk/reward with larger position sizes")
    
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
        # Show current positions from IBKR
        logger.info("\nCurrent Positions (from IBKR):")
        ibkr_positions = get_positions_from_ibkr(ib, contracts)
        if ibkr_positions:
            any_positions = False
            for strategy, pos_state in ibkr_positions.items():
                if pos_state['in_position']:
                    pos = pos_state['position']
                    logger.info(f"  {strategy}: {pos['contracts']} contracts @ ${pos['entry_price']:.2f}")
                    any_positions = True
            if not any_positions:
                logger.info("  No positions found")
        else:
            logger.warning("  Could not retrieve positions from IBKR")
        
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
    
    # Check current time and next market close
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    next_close = get_next_market_close()
    
    logger.info(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"Next market close: {next_close.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    if is_trading_window():
        logger.info("LIVE TRADING MODE - Within trading window (4:55-5:05 PM Eastern)")
    else:
        logger.info("Will execute trades 10 seconds before next market close")
    
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

    # Always wait until 10 seconds before next market close (5 PM Eastern)
    logger.info("Waiting until 10 seconds before next market close...")
    wait_until_next_close(timezone='US/Eastern', lead_seconds=10)

    # Run daily signals with exact logic from aggregate_port.py
    logger.info("Running daily signal generation...")
    run_daily_signals(ib)

    # Disconnect
    ib.disconnect()
    logger.info("Finished running daily signals. Disconnected from IBKR.")

if __name__ == '__main__':
    main()
