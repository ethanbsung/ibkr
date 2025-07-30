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
IB_HOST = '171.27.16.1'
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

# -------------------------------
# Contract Rollover Configuration
# -------------------------------
# Days before expiration to trigger rollover
ROLLOVER_DAYS_BEFORE_EXPIRATION = 5

# Contract month progression for each instrument
# Format: {'current_month': 'next_month', ...}
CONTRACT_ROLLOVER_CHAINS = {
    'ES': {  # E-mini S&P 500 - Quarterly (Mar, Jun, Sep, Dec)
        '202506': '202509',  # Aug -> Sep (special case)
        '202509': '202512',  # Sep -> Dec
        '202512': '202603',  # Dec -> Mar
        '202603': '202606',  # Mar -> Jun
        '202606': '202609',  # Jun -> Sep
        '202609': '202612',  # Sep -> Dec (future)
    },
    'YM': {  # E-mini Dow - Quarterly (Mar, Jun, Sep, Dec)
        '202506': '202509',  # Aug -> Sep (special case)
        '202509': '202512',  # Sep -> Dec
        '202512': '202603',  # Dec -> Mar
        '202603': '202606',  # Mar -> Jun
        '202606': '202609',  # Jun -> Sep
        '202609': '202612',  # Sep -> Dec (future)
    },
    'GC': {  # Gold - Monthly except Jan, Mar, May, Jul, Oct, Dec are most liquid
        '202510': '202512',  # Oct -> Dec
        '202512': '202602',  # Dec -> Feb
        '202602': '202604',  # Feb -> Apr
        '202604': '202606',  # Apr -> Jun
        '202606': '202608',  # Jun -> Aug
        '202608': '202610',  # Aug -> Oct
    },
    'NQ': {  # E-mini NASDAQ - Quarterly (Mar, Jun, Sep, Dec)
        '202506': '202509',  # Aug -> Sep (special case)
        '202509': '202512',  # Sep -> Dec
        '202512': '202603',  # Dec -> Mar
        '202603': '202606',  # Mar -> Jun
        '202606': '202609',  # Jun -> Sep
        '202609': '202612',  # Sep -> Dec (future)
    }
}

# ⚠️  CRITICAL: EXPIRATION DATES MUST BE VERIFIED WITH OFFICIAL SOURCES
# 
# These expiration dates are placeholders and MUST be updated with official 
# dates from CME Group and COMEX before use in live trading.
# 
# Official sources:
# - CME Group: https://www.cmegroup.com/trading/equity-index/files/Equity_Expiration_Calendar_2025.pdf
# - COMEX Gold: https://www.cmegroup.com/trading/metals/precious/gold.html
#
# ⚠️  DO NOT USE THESE PLACEHOLDER DATES FOR ACTUAL TRADING ⚠️
CONTRACT_EXPIRATION_DATES = {
    'ES': {
        # PLACEHOLDER - VERIFY WITH CME GROUP OFFICIAL CALENDAR
        '202509': datetime(2025, 9, 19),  # September 19, 2025 (verified)
        '202512': None,  # Must verify December expiration
        '202603': None,  # Must verify March 2026 expiration
        '202606': None,  # Must verify June 2026 expiration
    },
    'YM': {
        # PLACEHOLDER - VERIFY WITH CME GROUP OFFICIAL CALENDAR
        '202509': datetime(2025, 9, 19),  # September 19, 2025 (same as ES)
        '202512': None,  # Must verify December expiration
        '202603': None,  # Must verify March 2026 expiration
        '202606': None,  # Must verify June 2026 expiration
    },
    'GC': {
        # PLACEHOLDER - VERIFY WITH COMEX OFFICIAL CALENDAR
        '202510': datetime(2025, 10, 29),  # October 29, 2025 (estimated)
        '202512': None,  # Must verify December expiration
        '202602': None,  # Must verify February 2026 expiration
        '202604': None,  # Must verify April 2026 expiration
        '202606': None,  # Must verify June 2026 expiration
        '202608': None,  # Must verify August 2026 expiration
    },
    'NQ': {
        # PLACEHOLDER - VERIFY WITH CME GROUP OFFICIAL CALENDAR
        '202509': datetime(2025, 9, 19),  # September 19, 2025 (same as ES/YM)
        '202512': None,  # Must verify December expiration
        '202603': None,  # Must verify March 2026 expiration
        '202606': None,  # Must verify June 2026 expiration
    }
}

# Contract Specifications and Multipliers (matching aggregate_port.py exactly)
contract_specs = {
    'ES': {'multiplier': 5, 'contract_month': '202509', 'exchange': 'CME'},      # MES multiplier
    'YM': {'multiplier': 0.50, 'contract_month': '202509', 'exchange': 'CBOT'},   # MYM multiplier  
    'GC': {'multiplier': 10, 'contract_month': '202510', 'exchange': 'COMEX'},     # MGC multiplier - moved to October to avoid delivery window
    'NQ': {'multiplier': 2, 'contract_month': '202509', 'exchange': 'CME'}      # MNQ multiplier
}

# IBS entry/exit thresholds
ibs_entry_threshold = 0.1       # Enter when IBS < 0.1
ibs_exit_threshold  = 0.9       # Exit when IBS > 0.9

# Williams %R strategy parameters
williams_period = 2             # 2-day lookback
wr_buy_threshold  = -90
wr_sell_threshold = -30

# -------------------------------
# Rollover Management Functions
# -------------------------------
def get_contract_expiration_date(symbol, contract_month):
    """Get the expiration date for a specific contract"""
    logger = logging.getLogger()
    try:
        expiration_date = CONTRACT_EXPIRATION_DATES[symbol][contract_month]
        if expiration_date is None:
            logger.error(f"⚠️  CRITICAL: Expiration date for {symbol} {contract_month} not verified!")
            logger.error(f"   Must check official CME/COMEX calendar before trading")
            logger.error(f"   https://www.cmegroup.com/trading/equity-index/files/Equity_Expiration_Calendar_2025.pdf")
            return None
        return expiration_date
    except KeyError:
        logger.warning(f"No expiration date configured for {symbol} {contract_month}")
        return None

def get_days_to_expiration(symbol, contract_month):
    """Calculate days until contract expiration"""
    expiration_date = get_contract_expiration_date(symbol, contract_month)
    if expiration_date is None:
        return None
    
    today = datetime.now().date()
    expiration_date = expiration_date.date()
    
    days_to_expiration = (expiration_date - today).days
    return days_to_expiration

def get_next_contract_month(symbol, current_month):
    """Get the next contract month for rollover"""
    try:
        return CONTRACT_ROLLOVER_CHAINS[symbol][current_month]
    except KeyError:
        logger = logging.getLogger()
        logger.warning(f"No rollover chain found for {symbol} {current_month}")
        return None

def should_rollover_contract(symbol, contract_month):
    """Check if a contract should be rolled over"""
    days_to_expiration = get_days_to_expiration(symbol, contract_month)
    
    if days_to_expiration is None:
        return False
    
    return days_to_expiration <= ROLLOVER_DAYS_BEFORE_EXPIRATION

def update_contract_specs_for_rollover(symbol, new_contract_month):
    """Update the global contract_specs with new contract month"""
    if symbol in contract_specs:
        old_month = contract_specs[symbol]['contract_month']
        contract_specs[symbol]['contract_month'] = new_contract_month
        
        logger = logging.getLogger()
        logger.info(f"Updated {symbol} contract from {old_month} to {new_contract_month}")
        return True
    return False

def check_and_update_rollover_chains():
    """Automatically update rollover chains for future dates"""
    logger = logging.getLogger()
    
    for symbol in CONTRACT_ROLLOVER_CHAINS:
        current_year = datetime.now().year
        
        # Generate additional rollover mappings for next year if needed
        if symbol in ['ES', 'YM', 'NQ']:  # Quarterly contracts
            quarters = ['03', '06', '09', '12']
            for year in [current_year, current_year + 1]:
                for i, quarter in enumerate(quarters):
                    current_contract = f"{year}{quarter}"
                    next_quarter_idx = (i + 1) % len(quarters)
                    next_year = year if next_quarter_idx > i else year + 1
                    next_contract = f"{next_year}{quarters[next_quarter_idx]}"
                    
                    if current_contract not in CONTRACT_ROLLOVER_CHAINS[symbol]:
                        CONTRACT_ROLLOVER_CHAINS[symbol][current_contract] = next_contract
        
        elif symbol == 'GC':  # Monthly contracts (simplified to bimonthly for liquidity)
            months = ['02', '04', '06', '08', '10', '12']
            for year in [current_year, current_year + 1]:
                for i, month in enumerate(months):
                    current_contract = f"{year}{month}"
                    next_month_idx = (i + 1) % len(months)
                    next_year = year if next_month_idx > i else year + 1
                    next_contract = f"{next_year}{months[next_month_idx]}"
                    
                    if current_contract not in CONTRACT_ROLLOVER_CHAINS[symbol]:
                        CONTRACT_ROLLOVER_CHAINS[symbol][current_contract] = next_contract

def execute_contract_rollover(ib, symbol, old_contract, new_contract, router):
    """Execute the actual contract rollover"""
    logger = logging.getLogger()
    
    old_contract_month = old_contract.lastTradeDateOrContractMonth
    new_contract_month = new_contract.lastTradeDateOrContractMonth
    
    try:
        logger.info(f"=== EXECUTING ROLLOVER FOR {symbol} ===")
        logger.info(f"Rolling from {old_contract.symbol} {old_contract_month}")
        logger.info(f"Rolling to {new_contract.symbol} {new_contract_month}")
        
        # Get current virtual positions for this symbol from router
        symbol_strategies = [s for s in allocation_percentages.keys() if s.endswith(f'_{symbol}')]
        
        total_virtual_position = 0
        strategy_positions = {}
        
        for strategy in symbol_strategies:
            virtual_lots = router.strategy_lots.get(strategy, 0)
            strategy_positions[strategy] = virtual_lots
            total_virtual_position += virtual_lots
            
        logger.info(f"Current virtual positions for {symbol}: {strategy_positions}")
        logger.info(f"Total virtual position to roll: {total_virtual_position}")
        
        if total_virtual_position == 0:
            logger.info(f"No positions to roll for {symbol}")
            # Update contract specs even with no positions
            update_contract_specs_for_rollover(symbol, new_contract_month)
            router.contract = new_contract
            return True
        
        # Step 1: Close all positions in the old contract
        logger.info(f"Step 1: Closing positions in old contract")
        close_positions = {strategy: 0 for strategy in symbol_strategies}
        router.sync(close_positions)
        ib.sleep(1)  # Reduced from 2 seconds
        
        # Step 2: Update contract specifications
        logger.info(f"Step 2: Updating contract specifications")
        update_contract_specs_for_rollover(symbol, new_contract_month)
        
        # Step 3: Update router contract reference
        logger.info(f"Step 3: Updating router contract reference")
        router.contract = new_contract
        
        # Step 4: Re-establish positions in new contract
        logger.info(f"Step 4: Re-establishing positions in new contract")
        router.sync(strategy_positions)
        ib.sleep(1)  # Reduced from 2 seconds
        
        logger.info(f"=== ROLLOVER COMPLETED FOR {symbol} ===")
        return True
        
    except Exception as e:
        logger.error(f"Error during rollover execution for {symbol}: {e}")
        return False

def check_rollover_requirements(contracts):
    """Check all contracts for rollover requirements and warn if needed"""
    logger = logging.getLogger()
    
    rollover_warnings = []
    
    for symbol, contract in contracts.items():
        current_month = contract_specs[symbol]['contract_month']
        days_to_expiration = get_days_to_expiration(symbol, current_month)
        
        if days_to_expiration is not None:
            if should_rollover_contract(symbol, current_month):
                next_month = get_next_contract_month(symbol, current_month)
                rollover_warnings.append({
                    'symbol': symbol,
                    'current_month': current_month,
                    'next_month': next_month,
                    'days_to_expiration': days_to_expiration
                })
                logger.warning(f"ROLLOVER NEEDED: {symbol} contract {current_month} expires in {days_to_expiration} days -> Roll to {next_month}")
            else:
                logger.info(f"{symbol} contract {current_month}: {days_to_expiration} days to expiration")
    
    return rollover_warnings

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
                state = json.load(f)

            # -----------------------------------------------------------
            # Robustness patch: ensure all expected strategies are present
            # -----------------------------------------------------------
            # Older versions of the state file may pre-date new strategies
            # added to `allocation_percentages`, which can lead to
            # KeyErrors when the live code tries to access them.  Here we
            # reconcile the on-disk state with the current list of
            # strategies, defaulting to no position where necessary.
            if 'positions' not in state or not isinstance(state['positions'], dict):
                state['positions'] = {}

            for strategy in allocation_percentages:
                state['positions'].setdefault(strategy, {
                    'in_position': False,
                    'position': None
                })

            # Maintain required top-level keys
            state.setdefault('last_rebalance_date', None)
            state.setdefault('current_equity', None)

            return state
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

def get_current_price(ib, contract, symbol):
    """Get current market price using real-time market data"""
    logger = logging.getLogger()
    
    try:
        # Request real-time market data
        ticker = ib.reqMktData(contract, '', False, False)
        ib.sleep(2)  # Wait for price data
        
        current_price = None
        if ticker.last and ticker.last > 0:
            current_price = ticker.last
            price_source = "last"
        elif ticker.close and ticker.close > 0:
            current_price = ticker.close
            price_source = "close"
        elif ticker.bid and ticker.ask:
            current_price = (ticker.bid + ticker.ask) / 2
            price_source = "mid"
        
        # Cancel market data subscription using the contract (not ticker)
        ib.cancelMktData(contract)
        
        if current_price:
            logger.info(f"Current {symbol} price: {current_price} (source: {price_source})")
            return current_price
        else:
            logger.warning(f"Could not get current price for {symbol}")
            return None
            
    except Exception as e:
        logger.warning(f"Error getting current price for {symbol}: {e}")
        return None

def compute_live_ibs(bar, current_price):
    """Calculate IBS using current live price instead of historical close"""
    range_val = bar.high - bar.low
    
    # If we have a live price, extend the range if needed
    if current_price > bar.high:
        logger = logging.getLogger()
        logger.info(f"Live price {current_price} above daily high {bar.high} - extending range")
        range_val = current_price - bar.low
    elif current_price < bar.low:
        logger = logging.getLogger()
        logger.info(f"Live price {current_price} below daily low {bar.low} - extending range")
        range_val = bar.high - current_price
    
    if range_val == 0:
        return 0.5  # Neutral IBS when no range
    else:
        return (current_price - bar.low) / range_val

def compute_live_williams_r(bars, current_price):
    """
    Compute Williams %R using live current price with safety check
    Formula: -100 * (highestHigh - current_close) / (highestHigh - lowestLow)
    """
    if len(bars) < williams_period:
        return -50  # Neutral value
    
    # Use last 2 bars for range calculation
    recent_bars = bars[-williams_period:]
    highest_high = max(bar.high for bar in recent_bars)
    lowest_low = min(bar.low for bar in recent_bars)
    
    # Extend range if current price is outside historical range
    if current_price > highest_high:
        logger = logging.getLogger()
        logger.info(f"Live price {current_price} above period high {highest_high} - extending range")
        highest_high = current_price
    elif current_price < lowest_low:
        logger = logging.getLogger()
        logger.info(f"Live price {current_price} below period low {lowest_low} - extending range")
        lowest_low = current_price
    
    range_val = highest_high - lowest_low
    if range_val == 0:
        return -50  # Neutral Williams %R when no range
    else:
        return -100 * (highest_high - current_price) / range_val

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

def get_positions_from_ibkr_optimized(ib, contracts, market_data):
    """Get current positions from IBKR API using cached market data for better performance"""
    logger = logging.getLogger()
    
    try:
        # Get all positions from IBKR
        positions = ib.positions()
        
        # Use cached price data instead of making new API calls
        current_prices = {}
        for symbol in contracts.keys():
            if market_data[symbol]['current_price']:
                current_prices[symbol] = market_data[symbol]['current_price']
                logger.info(f"Using cached {symbol} price: {current_prices[symbol]}")
        
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
                    
                    # Use cached price data
                    if symbol in current_prices:
                        entry_price = current_prices[symbol]
                        logger.info(f"Using cached market price {entry_price} for {symbol} position")
                    else:
                        # Fallback to reasonable estimates if cached data not available
                        logger.warning(f"No cached price data for {symbol}, using default estimate")
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

# Old function removed - replaced by get_positions_from_ibkr_optimized() for better performance

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
    Place a market order for immediate execution during trading hours.
    Returns the trade object if successful, None if failed or in dry run mode.
    """
    logger = logging.getLogger()
    
    # Check if we're outside the trading window (4:58-5:05 PM Eastern) - use dry run mode
    if not is_trading_window() or dry_run:
        logger.info(f"DRY RUN: Would place MKT {action} order for {quantity} {symbol} contracts (outside trading window)")
        # Return None for dry run mode - do not update position state
        return None
    
    # Use market orders for immediate execution before close
    try:
        order = Order(action=action, totalQuantity=quantity, orderType='MKT', tif='DAY')
        trade = ib.placeOrder(contract, order)
        logger.info(f"Placed MKT {action} order for {quantity} {symbol} contracts")
        
        # Brief wait to see if there are any immediate errors
        ib.sleep(0.5)  # Reduced from 1 second
        
        # Check if order was rejected
        if trade.orderStatus.status == 'Cancelled':
            for log_entry in trade.log:
                if '201' in log_entry.message or 'physical delivery' in log_entry.message.lower():
                    logger.error(f"Order rejected for {symbol}: Contract in delivery window or near expiration")
                    return None
                elif 'exchange is closed' in log_entry.message.lower():
                    logger.error(f"Order rejected for {symbol}: Exchange is closed")
                    return None
        
        return trade
    except Exception as e:
        logger.error(f"Order placement failed for {symbol}: {e}")
        return None
    

def is_trading_window(timezone='US/Eastern'):
    """Check if we're in the specific trading window (4:58-5:05 PM Eastern) for placing EOD orders"""
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    hour = now.hour
    minute = now.minute
    
    # Only trade Monday-Friday
    if weekday not in [0, 1, 2, 3, 4]:  # Monday-Friday
        return False
    
    # Only trade between 4:58 PM and 5:05 PM Eastern (extended window for execution)
    if hour == 16 and minute >= 58:  # 4:58-4:59 PM
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
    def __init__(self, ib, contract, symbol='ES', portfolio_state=None):
        self.ib = ib
        self.contract = contract
        self.symbol = symbol
        self.strategy_lots = self._load_state()
        
        # Initialize router state from portfolio state if provided
        if portfolio_state:
            self._sync_with_portfolio_state(portfolio_state)

    def _sync_with_portfolio_state(self, portfolio_state):
        """Initialize router state from current portfolio positions"""
        logger = logging.getLogger()
        
        # Find strategies that trade this symbol
        symbol_strategies = [s for s in portfolio_state['positions'].keys() 
                           if s.endswith(f'_{self.symbol}')]
        
        for strategy in symbol_strategies:
            strategy_state = portfolio_state['positions'][strategy]
            if strategy_state['in_position'] and strategy_state['position']:
                current_contracts = strategy_state['position']['contracts']
                self.strategy_lots[strategy] = current_contracts
                logger.info(f"Router {self.symbol}: Initialized {strategy} with {current_contracts} contracts")
            else:
                self.strategy_lots[strategy] = 0
        
        # Save the initialized state
        self._save_state()

    # ---------- persistence ----------
    def _load_state(self):
        if os.path.exists(ROUTER_STATE_FILE):
            try:
                with open(ROUTER_STATE_FILE, 'r') as f:
                    all_router_state = json.load(f)
                    # Return only this symbol's state
                    return defaultdict(int, all_router_state.get(self.symbol, {}))
            except Exception:
                pass
        # default – no virtual positions yet
        return defaultdict(int)

    def _save_state(self):
        # Load existing state for all symbols
        all_state = {}
        if os.path.exists(ROUTER_STATE_FILE):
            try:
                with open(ROUTER_STATE_FILE, 'r') as f:
                    all_state = json.load(f)
            except Exception:
                pass
        
        # Update this symbol's state
        all_state[self.symbol] = dict(self.strategy_lots)
        
        # Save back to file
        with open(ROUTER_STATE_FILE, 'w') as f:
            json.dump(all_state, f, indent=2)

    # ---------- live position ----------
    def _live_position(self):
        for p in self.ib.positions():
            if p.contract.conId == self.contract.conId:
                return int(p.position)
        return 0

    # ---------- public ----------
    def sync(self, desired_dict):
        """
        desired_dict: {'IBS_ES': +2, 'Williams_ES': 0, ...}
        Places ONE order to move from current net to desired net.
        """
        logger = logging.getLogger()
        
        live_total = self._live_position()
        desired_total = sum(desired_dict.values())
        net_change = desired_total - live_total
        
        # Log the calculation details
        logger.info(f"Router {self.symbol}: Current virtual lots: {dict(self.strategy_lots)}")
        logger.info(f"Router {self.symbol}: Desired lots: {desired_dict}")
        logger.info(f"Router {self.symbol}: Live IBKR position: {live_total}")
        logger.info(f"Router {self.symbol}: Desired total: {desired_total}")
        logger.info(f"Router {self.symbol}: Net change needed: {net_change}")

        # no trade needed
        if net_change == 0:
            logger.info(f"Router {self.symbol}: No trade needed")
            self.strategy_lots.update(desired_dict)
            self._save_state()
            return

        action = 'BUY' if net_change > 0 else 'SELL'
        qty = abs(net_change)
        
        logger.info(f"Router {self.symbol}: Placing {action} order for {qty} contracts")

        trade = place_order_with_fallback(
            self.ib,
            self.contract,
            action,
            qty,
            self.symbol
        )

        # optimistic virtual‑lot update (we rely on nightly reconciliation)
        if trade is not None:
            logger.info(f"Router {self.symbol}: Order placed successfully, updating virtual lots")
            self.strategy_lots.update(desired_dict)
            self._save_state()
        else:
            logger.warning(f"Router {self.symbol}: Order failed, keeping existing virtual lots")

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
    logger.info(f"Waiting until {target_time.strftime('%Y-%m-%d %H:%M:%S %Z')} ({wait_seconds/60:.1f} minutes before market close - optimized for faster execution)")
    
    # Show periodic updates for long waits
    while True:
        now = datetime.now(tz)
        if now >= target_time:
            break
        
        remaining_seconds = (target_time - now).total_seconds()
        
        # Sleep in chunks, showing progress for long waits
        if remaining_seconds > 3600:  # More than 1 hour
            logger.info(f"Waiting {remaining_seconds/3600:.1f} hours until execution time...")
            time.sleep(min(1800, remaining_seconds))  # Sleep up to 30 minutes at a time
        elif remaining_seconds > 300:  # More than 5 minutes
            logger.info(f"Waiting {remaining_seconds/60:.1f} minutes until execution time...")
            time.sleep(min(60, remaining_seconds))  # Sleep up to 1 minute at a time
        else:
            time.sleep(remaining_seconds)
            break

# -------------------------------
# Optimized Market Data Functions
# -------------------------------
def fetch_all_market_data(ib, contracts, end_datetime_str):
    """
    Fetch all required market data for all contracts in parallel to minimize execution time.
    Returns cached data structure with bars and current prices.
    """
    logger = logging.getLogger()
    logger.info("Fetching all market data in parallel...")
    
    # Pre-fetch all historical data first (can be done in parallel by IBKR)
    market_data = {}
    
    # Request all daily bars simultaneously
    daily_bar_requests = {}
    williams_bar_requests = {}
    
    for symbol, contract in contracts.items():
        # Daily bars for IBS
        daily_bar_requests[symbol] = ib.reqHistoricalData(
            contract,
            endDateTime=end_datetime_str,
            durationStr='5 D',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1,
            keepUpToDate=False
        )
        
        # Williams bars
        williams_bar_requests[symbol] = ib.reqHistoricalData(
            contract,
            endDateTime=end_datetime_str,
            durationStr='1 W',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1,
            keepUpToDate=False
        )
    
    # Wait for all historical data requests to complete
    ib.sleep(1)  # Reduced from multiple 2-second waits
    
    # Get current prices if in trading window
    current_prices = {}
    if is_trading_window():
        logger.info("Requesting live prices for all contracts...")
        # Request all market data subscriptions simultaneously
        tickers = {}
        for symbol, contract in contracts.items():
            tickers[symbol] = ib.reqMktData(contract, '', False, False)
        
        # Single wait for all price data
        ib.sleep(1.5)  # Reduced from 2 seconds per contract
        
        # Extract prices and cancel subscriptions
        for symbol, ticker in tickers.items():
            current_price = None
            if ticker.last and ticker.last > 0:
                current_price = ticker.last
                price_source = "last"
            elif ticker.close and ticker.close > 0:
                current_price = ticker.close
                price_source = "close"
            elif ticker.bid and ticker.ask:
                current_price = (ticker.bid + ticker.ask) / 2
                price_source = "mid"
            
            if current_price:
                current_prices[symbol] = (current_price, price_source)
                logger.info(f"Live {symbol} price: {current_price} (source: {price_source})")
            
            # Cancel market data subscription
            ib.cancelMktData(contracts[symbol])
    
    # Organize all data
    for symbol in contracts.keys():
        daily_bars = daily_bar_requests[symbol] if daily_bar_requests[symbol] else []
        williams_bars = williams_bar_requests[symbol] if williams_bar_requests[symbol] else []
        
        # Get current price (live or historical fallback)
        current_price = None
        is_live = False
        price_source = "historical"
        
        if symbol in current_prices:
            current_price, price_source = current_prices[symbol]
            is_live = True
        elif daily_bars:
            current_price = daily_bars[-1].close
            price_source = "historical"
        
        market_data[symbol] = {
            'daily_bars': daily_bars,
            'williams_bars': williams_bars,
            'current_price': current_price,
            'is_live': is_live,
            'price_source': price_source
        }
        
        logger.info(f"{symbol} data ready - Price: {current_price} ({'LIVE' if is_live else 'HISTORICAL'})")
    
    logger.info(f"All market data fetched in parallel - ready for strategy processing")
    return market_data

def get_cached_bar_data(market_data, symbol):
    """Get cached daily bar data for IBS calculation"""
    data = market_data[symbol]
    daily_bars = data['daily_bars']
    
    if not daily_bars:
        return None, None, False
    
    current_bar = daily_bars[-1]
    current_price = data['current_price']
    is_live = data['is_live']
    
    return current_bar, current_price, is_live

def get_cached_williams_data(market_data, symbol):
    """Get cached Williams bar data"""
    data = market_data[symbol]
    williams_bars = data['williams_bars']
    
    if not williams_bars or len(williams_bars) < williams_period:
        return None, None, False
    
    current_price = data['current_price']
    is_live = data['is_live']
    
    return williams_bars, current_price, is_live

# -------------------------------
# Main Trading Logic (matching aggregate_port.py exactly)
# -------------------------------
def run_daily_signals(ib):
    logger = logging.getLogger()
    
    # Load current portfolio state
    state = load_portfolio_state()
    
    # Check and update rollover chains for future contracts
    check_and_update_rollover_chains()
    
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
        routers[symbol] = PositionRouter(ib, contracts[symbol], symbol, state)
        desired_positions[symbol] = {}  # will hold IBS and Williams target lots for this symbol
    
    # ========== ROLLOVER CHECK ==========
    # Check if any contracts need to be rolled (warning only - manual rollover required)
    logger.info("=== CHECKING ROLLOVER REQUIREMENTS ===")
    rollover_warnings = check_rollover_requirements(contracts)
    
    if rollover_warnings:
        logger.warning("⚠️  MANUAL ROLLOVER REQUIRED FOR THE FOLLOWING CONTRACTS:")
        for warning in rollover_warnings:
            logger.warning(f"  {warning['symbol']}: {warning['current_month']} -> {warning['next_month']} ({warning['days_to_expiration']} days)")
        logger.warning("Use --force-rollover [SYMBOL] to manually execute rollover")
    else:
        logger.info("✅ No contract rollovers needed")
    # ====================================
    
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
    
    # Synchronize positions with IBKR using cached price data
    logger.info("Synchronizing positions with IBKR...")
    ibkr_positions = get_positions_from_ibkr_optimized(ib, contracts, market_data)
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
    
    # ========== OPTIMIZED: FETCH ALL MARKET DATA ONCE ==========
    # Fetch all market data in parallel to minimize execution time
    market_data = fetch_all_market_data(ib, contracts, end_datetime_str)
    # ===========================================================
    
    # Process each IBS strategy (router-only approach for all instruments)
    for symbol in ['ES', 'YM', 'GC', 'NQ']:
        strategy_key = f'IBS_{symbol}'
        contract = contracts[symbol]
        multiplier = contract_specs[symbol]['multiplier']

        logger.info(f"\nProcessing {strategy_key}...")

        # Get current bar data from cached market data
        current_bar, current_price, is_live = get_cached_bar_data(market_data, symbol)
        if not current_bar or not current_price:
            logger.warning(f"No price data available for {symbol}")
            continue

        # Calculate IBS with current price (live or historical)
        if is_live:
            ibs = compute_live_ibs(current_bar, current_price)
        else:
            ibs = compute_ibs(current_bar)

        logger.info(f"{symbol} - Price: {current_price}, High: {current_bar.high}, Low: {current_bar.low}, IBS: {ibs:.3f} ({'LIVE' if is_live else 'HISTORICAL'})")

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
            if ibs > ibs_exit_threshold:
                desired_qty = 0
                logger.info(f"{symbol} - IBS exit signal (IBS: {ibs:.3f} > {ibs_exit_threshold})")
            else:
                desired_qty = target_contracts  # Use target contracts for rebalancing
                current_contracts = strategy_state['position']['contracts']
                if desired_qty != current_contracts:
                    logger.info(f"{symbol} - IBS rebalancing: {current_contracts} → {desired_qty} contracts (IBS: {ibs:.3f})")
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

        # Get Williams bar data from cached market data
        bars, current_price, is_live = get_cached_williams_data(market_data, symbol)

        if bars:
            logger.info(f"{strategy_key} - Retrieved {len(bars)} bars, need {williams_period}")
            logger.info(f"{strategy_key} - Recent bars: {[f'{bar.date}' for bar in bars[-min(3, len(bars)):]]}")

        if bars and current_price:
            # Calculate Williams %R with current price (live or historical)
            if is_live:
                williams_r = compute_live_williams_r(bars, current_price)
            else:
                williams_r = compute_williams_r(bars)

            logger.info(f"{symbol} Williams - Price: {current_price}, Williams %R: {williams_r:.2f} ({'LIVE' if is_live else 'HISTORICAL'})")

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
                    logger.info(f"{symbol} Williams - Exit signal: price above previous high ({current_price} > {bars[-2].high})")
                if williams_r > wr_sell_threshold:
                    exit_signal = True
                    logger.info(f"{symbol} Williams - Exit signal: Williams %R > {wr_sell_threshold} ({williams_r:.2f})")
                
                if exit_signal:
                    desired_qty = 0
                else:
                    desired_qty = target_contracts  # Use target contracts for rebalancing
                    current_contracts = williams_state['position']['contracts']
                    if desired_qty != current_contracts:
                        logger.info(f"{symbol} Williams - Rebalancing: {current_contracts} → {desired_qty} contracts (Williams %R: {williams_r:.2f})")
                    else:
                        logger.info(f"{symbol} Williams - Holding position (Williams %R: {williams_r:.2f})")
            else:
                desired_qty = target_contracts if williams_r < wr_buy_threshold else 0
                if williams_r < wr_buy_threshold:
                    logger.info(f"{symbol} Williams - Entry signal (Williams %R: {williams_r:.2f} < {wr_buy_threshold})")
                else:
                    logger.info(f"{symbol} Williams - No entry signal (Williams %R: {williams_r:.2f}, threshold: {wr_buy_threshold})")

            desired_positions[symbol][strategy_key] = desired_qty
        else:
            logger.warning(f"Insufficient Williams data for {symbol}")
            desired_positions[symbol][strategy_key] = 0

    # ---------- execute net orders for all instruments ----------
    for symbol in ['ES', 'YM', 'GC', 'NQ']:
        routers[symbol].sync(desired_positions[symbol])

        # reflect router outcome in local state (simplified – optimistic)
        # Use cached price data instead of making new API calls
        current_price = market_data[symbol]['current_price']
        
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

def print_rollover_status():
    """Print contract rollover status for all instruments"""
    logger = logging.getLogger()
    
    logger.info("=== CONTRACT ROLLOVER STATUS ===")
    
    for symbol in ['ES', 'YM', 'GC', 'NQ']:
        current_month = contract_specs[symbol]['contract_month']
        days_to_expiration = get_days_to_expiration(symbol, current_month)
        next_month = get_next_contract_month(symbol, current_month)
        
        if days_to_expiration is not None:
            if days_to_expiration <= ROLLOVER_DAYS_BEFORE_EXPIRATION:
                status = f"⚠️  ROLLOVER NEEDED"
            elif days_to_expiration <= 10:
                status = f"⚡ APPROACHING EXPIRATION"
            else:
                status = f"✅ OK"
                
            logger.info(f"{symbol} {current_month}: {days_to_expiration} days to expiration - {status}")
            if next_month:
                logger.info(f"  → Next contract: {next_month}")
        else:
            logger.warning(f"{symbol} {current_month}: Expiration date unknown")

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
    
    # Display rollover status
    print_rollover_status()
    
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
        try:
            # Note: This would need market_data for optimization, but here we fall back to simple display
            ibkr_positions = ib.positions()
            any_positions = False
            for position in ibkr_positions:
                if position.position != 0:
                    symbol = position.contract.symbol
                    logger.info(f"  {symbol}: {position.position} contracts @ ${position.avgCost:.2f}")
                    any_positions = True
            if not any_positions:
                logger.info("  No positions found")
        except Exception as e:
            logger.warning(f"  Could not retrieve positions from IBKR: {e}")
        
        tz = pytz.timezone('US/Eastern')
        end_datetime_str = format_end_datetime(datetime.now(tz), tz)
        
        logger.info("\nRecent OHLC data:")
        for symbol, contract in contracts.items():
            bars = get_daily_bar(ib, contract, end_datetime_str)
            if bars:
                recent_bar = bars[-1]
                logger.info(f"{symbol}: Date: {recent_bar.date}, O: {recent_bar.open}, H: {recent_bar.high}, L: {recent_bar.low}, C: {recent_bar.close}")

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Live Portfolio Trading with Rollover Management')
    parser.add_argument('--check-rollover', action='store_true', 
                       help='Check rollover status without running trading logic')
    parser.add_argument('--force-rollover', type=str, 
                       help='Force rollover for specific symbol (ES, YM, GC, NQ)')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    
    # Handle rollover check mode
    if args.check_rollover:
        logger.info("=== ROLLOVER STATUS CHECK MODE ===")
        print_rollover_status()
        return
    
    # Check current time and next market close
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    next_close = get_next_market_close()
    
    logger.info(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"Next market close: {next_close.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    if is_trading_window():
        logger.info("LIVE TRADING MODE - Within trading window (4:58-5:05 PM Eastern)")
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

    # Handle force rollover mode
    if args.force_rollover:
        symbol = args.force_rollover.upper()
        if symbol not in ['ES', 'YM', 'GC', 'NQ']:
            logger.error(f"Invalid symbol for force rollover: {symbol}")
            ib.disconnect()
            return
            
        logger.info(f"=== FORCING ROLLOVER FOR {symbol} ===")
        
        # Qualify contracts and set up routers
        contracts = qualify_contracts(ib)
        if contracts is None:
            logger.error("Failed to qualify contracts")
            ib.disconnect()
            return
            
        state = load_portfolio_state()
        routers = {}
        for sym in ['ES', 'YM', 'GC', 'NQ']:
            routers[sym] = PositionRouter(ib, contracts[sym], sym, state)
        
        # Force rollover for specified symbol
        current_month = contract_specs[symbol]['contract_month']
        next_month = get_next_contract_month(symbol, current_month)
        
        if next_month is None:
            logger.error(f"Cannot determine next contract month for {symbol}")
            ib.disconnect()
            return
            
        # Create new contract and execute rollover
        try:
            if symbol == 'ES':
                new_contract = Future(symbol='MES', lastTradeDateOrContractMonth=next_month,
                                    exchange=contract_specs[symbol]['exchange'], currency='USD')
            elif symbol == 'YM':
                new_contract = Future(symbol='MYM', lastTradeDateOrContractMonth=next_month,
                                    exchange=contract_specs[symbol]['exchange'], currency='USD')
            elif symbol == 'GC':
                new_contract = Future(symbol='MGC', lastTradeDateOrContractMonth=next_month,
                                    exchange=contract_specs[symbol]['exchange'], currency='USD')
            elif symbol == 'NQ':
                new_contract = Future(symbol='MNQ', lastTradeDateOrContractMonth=next_month,
                                    exchange=contract_specs[symbol]['exchange'], currency='USD')
            
            qualified = ib.qualifyContracts(new_contract)
            if not qualified:
                logger.error(f"Failed to qualify new contract for {symbol} {next_month}")
                ib.disconnect()
                return
                
            new_contract = qualified[0]
            success = execute_contract_rollover(ib, symbol, contracts[symbol], new_contract, 
                                              routers[symbol])
            
            if success:
                logger.info(f"Force rollover completed successfully for {symbol}")
            else:
                logger.error(f"Force rollover failed for {symbol}")
                
        except Exception as e:
            logger.error(f"Error during force rollover: {e}")
        
        ib.disconnect()
        return

    # Print portfolio summary on startup
    print_portfolio_summary(ib)

    # Always wait until 20 seconds before next market close (5 PM Eastern) for faster execution
    logger.info("Waiting until 20 seconds before next market close...")
    wait_until_next_close(timezone='US/Eastern', lead_seconds=10)

    # Run daily signals with exact logic from aggregate_port.py
    logger.info("Running daily signal generation...")
    run_daily_signals(ib)

    # Save daily account snapshot for GitHub Actions
    logger.info("Saving daily account snapshot...")
    try:
        import sys
        import os
        # Add parent directory to path to import account_summary
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        
        from account_summary import save_daily_snapshot
        if save_daily_snapshot(ib):
            logger.info("Daily snapshot saved successfully")
        else:
            logger.warning("Failed to save daily snapshot")
    except Exception as e:
        logger.error(f"Error saving daily snapshot: {e}")

    # Disconnect
    ib.disconnect()
    logger.info("Finished running daily signals. Disconnected from IBKR.")

if __name__ == '__main__':
    main()
