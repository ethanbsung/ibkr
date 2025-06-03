import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
from ib_insync import IB, Future, util
import time as tm

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# -------------------------------
# IBKR Connection Parameters
# -------------------------------
IB_HOST = '127.0.0.1'
IB_PORT = 4002  # Paper trading port (use 7496 for live)
CLIENT_ID = 3   # Unique client ID

# -------------------------------
# Parameters & User Settings
# -------------------------------
initial_capital = 30000.0         # total capital ($30,000)
commission_per_order = 1.24       # commission per order (per contract)

# Date ranges for combined backtest
# Original backtest period (local data)
original_start_date = '2000-01-01'
original_end_date = '2025-03-12'

# IBKR data period (continuation)
ibkr_start_date = '2025-03-12'  # Start from March 12, 2025
ibkr_end_date = datetime.now().strftime('%Y-%m-%d')  # To present

# -------------------------------
# Dynamic Percentage-Based Allocation Settings
# -------------------------------
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

# Validate allocations sum to 100%
total_allocation = sum(allocation_percentages.values())
if abs(total_allocation - 1.0) > 0.001:
    raise ValueError(f"Allocations must sum to 100%, current sum: {total_allocation*100:.1f}%")

logger.info("Dynamic Percentage-Based Allocation Settings:")
for strategy, pct in allocation_percentages.items():
    logger.info(f"  {strategy}: {pct*100:.1f}%")

# Contract Specifications and Multipliers
contract_specs = {
    'ES': {'multiplier': 5, 'contract_month': '202506'},      # MES multiplier
    'YM': {'multiplier': 0.50, 'contract_month': '202506'},   # MYM multiplier  
    'GC': {'multiplier': 10, 'contract_month': '202506'},     # MGC multiplier
    'NQ': {'multiplier': 2, 'contract_month': '202506'},      # MNQ multiplier
    'ZQ': {'multiplier': 4167, 'contract_month': '202506'}    # ZQ multiplier
}

# Extract individual multipliers for backward compatibility
multiplier_es = contract_specs['ES']['multiplier']
multiplier_ym = contract_specs['YM']['multiplier'] 
multiplier_gc = contract_specs['GC']['multiplier']
multiplier_nq = contract_specs['NQ']['multiplier']
multiplier_zq = contract_specs['ZQ']['multiplier']

# IBS entry/exit thresholds (common for all IBS instruments)
ibs_entry_threshold = 0.1       # Enter when IBS < 0.1
ibs_exit_threshold  = 0.9       # Exit when IBS > 0.9

# Williams %R strategy parameters (applied to ES only)
williams_period = 2             # 2-day lookback
wr_buy_threshold  = -90
wr_sell_threshold = -30
williams_contracts = 1          # Williams trades ES with 1 contract (multiplier_es)

# -------------------------------
# Dynamic Position Sizing Functions
# -------------------------------
def calculate_position_size(current_equity, target_allocation_pct, price, multiplier, min_contracts=2):
    """
    Calculate number of contracts based on current equity and target allocation.
    
    Args:
        current_equity: Current account equity
        target_allocation_pct: Target percentage allocation (0.0 to 1.0)
        price: Current price of the instrument
        multiplier: Contract multiplier
        min_contracts: Minimum number of contracts (default 1)
    
    Returns:
        Number of contracts to trade
    """
    target_dollar_amount = current_equity * target_allocation_pct
    contract_value = price * multiplier
    
    if contract_value <= 0:
        return min_contracts
    
    calculated_contracts = target_dollar_amount / contract_value
    
    # Round to nearest integer, minimum 1 contract
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
# IBKR Data Fetching Functions
# -------------------------------
def fetch_daily_data(ib, contract, start_date_str, end_date_str):
    """
    Fetch daily OHLC data from IBKR for a given contract and date range.
    """
    try:
        # Calculate duration
        start_dt = pd.to_datetime(start_date_str)
        end_dt = pd.to_datetime(end_date_str)
        duration_days = (end_dt - start_dt).days + 1
        duration_str = f"{duration_days} D"
        
        # Format end date for IBKR API
        end_datetime_str = end_dt.strftime("%Y%m%d 23:59:59")
        
        logger.info(f"Fetching daily data for {contract.localSymbol} from {start_date_str} to {end_date_str}")
        
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_datetime_str,
            durationStr=duration_str,
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,  # Regular trading hours only
            formatDate=1
        )
        
        if not bars:
            logger.warning(f"No data returned for {contract.localSymbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = util.df(bars)
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'date': 'Time',
            'open': 'Open', 
            'high': 'High',
            'low': 'Low',
            'close': 'Last'
        })
        
        # Ensure Time column is datetime
        df['Time'] = pd.to_datetime(df['Time'])
        
        # Filter to exact date range
        df = df[(df['Time'] >= start_date_str) & (df['Time'] <= end_date_str)].reset_index(drop=True)
        
        logger.info(f"Retrieved {len(df)} daily bars for {contract.localSymbol}")
        tm.sleep(1)  # Rate limiting
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {contract.localSymbol}: {e}")
        return pd.DataFrame()

# -------------------------------
# PART 1: Original Backtest with Local Data
# -------------------------------
logger.info("="*60)
logger.info("PART 1: Running original backtest with local data...")
logger.info("="*60)

# Load original local data files
def load_local_data(file_path, start_date, end_date):
    """Load and filter local CSV data"""
    try:
        data = pd.read_csv(file_path, parse_dates=['Time'])
        data.sort_values('Time', inplace=True)
        data = data[(data['Time'] >= start_date) & (data['Time'] <= end_date)].reset_index(drop=True)
        return data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

# Local data file paths
local_files = {
    'ES': "Data/mes_daily_data.csv",
    'YM': "Data/mym_daily_data.csv", 
    'GC': "Data/mgc_daily_data.csv",
    'NQ': "Data/mnq_daily_data.csv",
    'ZQ': "Data/zq_daily_data.csv"
}

# Load all local datasets
logger.info("Loading local data files...")
local_data = {}
for symbol, file_path in local_files.items():
    local_data[symbol] = load_local_data(file_path, original_start_date, original_end_date)
    if not local_data[symbol].empty:
        logger.info(f"Loaded {len(local_data[symbol])} bars for {symbol}")
    else:
        logger.warning(f"No data loaded for {symbol}")

# Check if we have ES data (required)
if local_data['ES'].empty:
    logger.error("No local ES data - cannot proceed")
    exit(1)

# Run original backtest for each strategy with dynamic allocation
def run_original_backtest():
    """Run the original backtest with local data using dynamic allocation"""
    
    # Track total portfolio equity and individual strategy allocations
    total_equity = initial_capital
    strategy_values = {}
    last_rebalance_day = 0
    
    # Initialize strategy capital allocations
    for strategy in allocation_percentages:
        strategy_values[strategy] = {
            'capital': total_equity * allocation_percentages[strategy],
            'in_position': False,
            'position': None,
            'equity_curve': []
        }
    
    # Get longest data series for iteration
    max_rows = max(len(local_data[symbol]) for symbol in ['ES', 'YM', 'GC', 'NQ', 'ZQ'] if not local_data[symbol].empty)
    
    # Main backtest loop
    for day_idx in range(max_rows):
        current_date = None
        daily_total_equity = 0
        
        # Process each IBS strategy
        for symbol in ['ES', 'YM', 'GC', 'NQ', 'ZQ']:
            strategy_key = f'IBS_{symbol}'
            
            if local_data[symbol].empty or day_idx >= len(local_data[symbol]):
                # No data available - just track capital
                strategy_values[strategy_key]['equity_curve'].append((current_date, strategy_values[strategy_key]['capital']))
                daily_total_equity += strategy_values[strategy_key]['capital']
                continue
            
            row = local_data[symbol].iloc[day_idx]
            current_date = row['Time']
            current_price = row['Last']
            
            # Calculate IBS
            ibs = (row['Last'] - row['Low']) / (row['High'] - row['Low'])
            
            multiplier = contract_specs[symbol]['multiplier']
            strategy_data = strategy_values[strategy_key]
            
            # Calculate current position size based on allocation
            current_contracts = calculate_position_size(
                total_equity, 
                allocation_percentages[strategy_key], 
                current_price, 
                multiplier
            )
            
            # Execute trading logic
            if strategy_data['in_position']:
                if ibs > ibs_exit_threshold:
                    # Exit position
                    exit_price = current_price
                    profit = (exit_price - strategy_data['position']['entry_price']) * multiplier * strategy_data['position']['contracts'] - commission_per_order * strategy_data['position']['contracts']
                    strategy_data['capital'] += profit
                    strategy_data['in_position'] = False
                    strategy_data['position'] = None
            else:
                if ibs < ibs_entry_threshold:
                    # Enter position
                    entry_price = current_price
                    strategy_data['in_position'] = True
                    strategy_data['capital'] -= commission_per_order * current_contracts
                    strategy_data['position'] = {
                        'entry_price': entry_price, 
                        'entry_time': current_date,
                        'contracts': current_contracts
                    }
            
            # Calculate current equity (including unrealized P&L)
            if strategy_data['in_position']:
                unrealized = (current_price - strategy_data['position']['entry_price']) * multiplier * strategy_data['position']['contracts']
                equity = strategy_data['capital'] + unrealized
            else:
                equity = strategy_data['capital']
                
            strategy_data['equity_curve'].append((current_date, equity))
            daily_total_equity += equity
        
        # Process Williams strategy (ES only)
        if not local_data['ES'].empty and day_idx >= 1 and day_idx < len(local_data['ES']):
            # Need at least 2 days for Williams %R calculation
            es_data = local_data['ES'].iloc[max(0, day_idx-williams_period+1):day_idx+1]
            
            if len(es_data) >= williams_period:
                current_row = es_data.iloc[-1]
                current_date = current_row['Time']
                current_price = current_row['Last']
                
                # Calculate Williams %R
                highest_high = es_data['High'].max()
                lowest_low = es_data['Low'].min()
                williams_r = -100 * (highest_high - current_price) / (highest_high - lowest_low)
                
                strategy_data = strategy_values['Williams']
                
                # Calculate current position size
                current_contracts = calculate_position_size(
                    total_equity, 
                    allocation_percentages['Williams'], 
                    current_price, 
                    multiplier_es
                )
                
                # Execute Williams trading logic
                if strategy_data['in_position']:
                    if day_idx > 0:
                        yesterdays_high = local_data['ES'].iloc[day_idx-1]['High']
                        if (current_price > yesterdays_high) or (williams_r > wr_sell_threshold):
                            # Exit position
                            exit_price = current_price
                            profit = (exit_price - strategy_data['position']['entry_price']) * multiplier_es * strategy_data['position']['contracts'] - commission_per_order * strategy_data['position']['contracts']
                            strategy_data['capital'] += profit
                            strategy_data['in_position'] = False
                            strategy_data['position'] = None
                else:
                    if williams_r < wr_buy_threshold:
                        # Enter position
                        entry_price = current_price
                        strategy_data['in_position'] = True
                        strategy_data['capital'] -= commission_per_order * current_contracts
                        strategy_data['position'] = {
                            'entry_price': entry_price, 
                            'entry_time': current_date,
                            'contracts': current_contracts
                        }
                
                # Calculate current equity
                if strategy_data['in_position']:
                    unrealized = (current_price - strategy_data['position']['entry_price']) * multiplier_es * strategy_data['position']['contracts']
                    equity = strategy_data['capital'] + unrealized
                else:
                    equity = strategy_data['capital']
                    
                strategy_data['equity_curve'].append((current_date, equity))
                daily_total_equity += equity
        
        # Update total equity for next day's position sizing
        if current_date is not None:
            total_equity = daily_total_equity
            
            # Check for rebalancing (every 30 days)
            if day_idx - last_rebalance_day >= rebalance_frequency_days:
                logger.info(f"Rebalancing on {current_date} (Day {day_idx})")
                last_rebalance_day = day_idx
    
    # Close any remaining positions and prepare results
    results = {}
    for strategy_key in allocation_percentages:
        strategy_data = strategy_values[strategy_key]
        
        # Close final positions if they exist
        if strategy_data['in_position'] and strategy_data['equity_curve']:
            if strategy_key.startswith('IBS_'):
                symbol = strategy_key.split('_')[1]
                if not local_data[symbol].empty:
                    final_row = local_data[symbol].iloc[-1]
                    final_price = final_row['Last']
                    multiplier = contract_specs[symbol]['multiplier']
                    
                    profit = (final_price - strategy_data['position']['entry_price']) * multiplier * strategy_data['position']['contracts'] - commission_per_order * strategy_data['position']['contracts']
                    strategy_data['capital'] += profit
                    strategy_data['equity_curve'][-1] = (final_row['Time'], strategy_data['capital'])
            
            elif strategy_key == 'Williams' and not local_data['ES'].empty:
                final_row = local_data['ES'].iloc[-1]
                final_price = final_row['Last']
                
                profit = (final_price - strategy_data['position']['entry_price']) * multiplier_es * strategy_data['position']['contracts'] - commission_per_order * strategy_data['position']['contracts']
                strategy_data['capital'] += profit
                strategy_data['equity_curve'][-1] = (final_row['Time'], strategy_data['capital'])
        
        results[strategy_key] = {
            'equity_curve': strategy_data['equity_curve'],
            'final_capital': strategy_data['capital']
        }
    
    return results
    


# Run original backtest
original_results = run_original_backtest()

# Extract final capital values for each strategy
final_capitals = {}
for strategy in ['IBS_ES', 'IBS_YM', 'IBS_GC', 'IBS_NQ', 'IBS_ZQ', 'Williams']:
    final_capitals[strategy] = original_results[strategy]['final_capital']

logger.info("Original backtest completed!")
logger.info("Final capital values:")
for strategy, capital in final_capitals.items():
    logger.info(f"  {strategy}: ${capital:,.2f}")

# -------------------------------
# PART 2: Continue with IBKR Data
# -------------------------------
logger.info("="*60)
logger.info("PART 2: Continuing with IBKR data...")
logger.info("="*60)

# Connect to IBKR
logger.info("Connecting to IBKR...")
ib = IB()
try:
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
    logger.info("Connected to IBKR successfully")
except Exception as e:
    logger.error(f"Failed to connect to IBKR: {e}")
    exit(1)

# -------------------------------
# Define IBKR Contracts
# -------------------------------
try:
    # Define futures contracts for each instrument
    es_contract = Future(symbol='MES', lastTradeDateOrContractMonth=contract_specs['ES']['contract_month'], exchange='CME', currency='USD')
    ym_contract = Future(symbol='MYM', lastTradeDateOrContractMonth=contract_specs['YM']['contract_month'], exchange='CBOT', currency='USD') 
    gc_contract = Future(symbol='MGC', lastTradeDateOrContractMonth=contract_specs['GC']['contract_month'], exchange='COMEX', currency='USD')
    nq_contract = Future(symbol='MNQ', lastTradeDateOrContractMonth=contract_specs['NQ']['contract_month'], exchange='CME', currency='USD')
    zq_contract = Future(symbol='ZQ', lastTradeDateOrContractMonth=contract_specs['ZQ']['contract_month'], exchange='CBOT', currency='USD')
    
    # Qualify all contracts
    contracts_to_qualify = [es_contract, ym_contract, gc_contract, nq_contract, zq_contract]
    qualified_contracts = ib.qualifyContracts(*contracts_to_qualify)
    
    if len(qualified_contracts) != 5:
        logger.error(f"Only {len(qualified_contracts)} out of 5 contracts qualified")
        ib.disconnect()
        exit(1)
    
    es_contract, ym_contract, gc_contract, nq_contract, zq_contract = qualified_contracts
    logger.info("All contracts qualified successfully")
    
except Exception as e:
    logger.error(f"Error defining/qualifying contracts: {e}")
    ib.disconnect()
    exit(1)

# -------------------------------
# Fetch Historical Data
# -------------------------------
logger.info("Fetching historical data for all instruments...")

# Fetch data for ES (used for benchmark, IBS ES, and Williams strategies)
data_es = fetch_daily_data(ib, es_contract, ibkr_start_date, ibkr_end_date)
if data_es.empty:
    logger.error("No IBKR data for ES - cannot proceed")
    ib.disconnect()
    exit(1)

# Calculate benchmark performance (for IBKR period only)
benchmark_initial_close = data_es['Last'].iloc[0]
benchmark_final_close = data_es['Last'].iloc[-1]
benchmark_return = ((benchmark_final_close / benchmark_initial_close) - 1) * 100

benchmark_years = (data_es['Time'].iloc[-1] - data_es['Time'].iloc[0]).days / 365.25
benchmark_annualized_return = ((benchmark_final_close / benchmark_initial_close) ** (1 / benchmark_years) - 1) * 100

# Fetch data for other IBS instruments
data_ym = fetch_daily_data(ib, ym_contract, ibkr_start_date, ibkr_end_date)
data_gc = fetch_daily_data(ib, gc_contract, ibkr_start_date, ibkr_end_date)
data_nq = fetch_daily_data(ib, nq_contract, ibkr_start_date, ibkr_end_date)
data_zq = fetch_daily_data(ib, zq_contract, ibkr_start_date, ibkr_end_date)

# Check if we have data for all instruments
datasets = [
    (data_es, "ES"), (data_ym, "YM"), (data_gc, "GC"), 
    (data_nq, "NQ"), (data_zq, "ZQ")
]

for data, name in datasets:
    if data.empty:
        logger.warning(f"No data available for {name}")

# Disconnect from IBKR as we have all the data we need
ib.disconnect()
logger.info("Disconnected from IBKR")

# -------------------------------
# Prepare IBS Data for Each Instrument
# -------------------------------
logger.info("Calculating IBS for all instruments...")

# IBS for ES
data_ibs_es = data_es.copy()
data_ibs_es['IBS'] = (data_ibs_es['Last'] - data_ibs_es['Low']) / (data_ibs_es['High'] - data_ibs_es['Low'])

# IBS for YM
data_ibs_ym = data_ym.copy()
if not data_ibs_ym.empty:
    data_ibs_ym['IBS'] = (data_ibs_ym['Last'] - data_ibs_ym['Low']) / (data_ibs_ym['High'] - data_ibs_ym['Low'])

# IBS for GC
data_ibs_gc = data_gc.copy()
if not data_ibs_gc.empty:
    data_ibs_gc['IBS'] = (data_ibs_gc['Last'] - data_ibs_gc['Low']) / (data_ibs_gc['High'] - data_ibs_gc['Low'])

# IBS for NQ
data_ibs_nq = data_nq.copy()
if not data_ibs_nq.empty:
    data_ibs_nq['IBS'] = (data_ibs_nq['Last'] - data_ibs_nq['Low']) / (data_ibs_nq['High'] - data_ibs_nq['Low'])

# IBS for ZQ
data_ibs_zq = data_zq.copy()
if not data_ibs_zq.empty:
    data_ibs_zq['IBS'] = (data_ibs_zq['Last'] - data_ibs_zq['Low']) / (data_ibs_zq['High'] - data_ibs_zq['Low'])

# -------------------------------
# Prepare Williams Data (ES only)
# -------------------------------
data_williams = data_es.copy()
data_williams['HighestHigh'] = data_williams['High'].rolling(window=williams_period).max()
data_williams['LowestLow'] = data_williams['Low'].rolling(window=williams_period).min()
data_williams.dropna(subset=['HighestHigh', 'LowestLow'], inplace=True)
data_williams.reset_index(drop=True, inplace=True)
data_williams['WilliamsR'] = -100 * (data_williams['HighestHigh'] - data_williams['Last']) / (data_williams['HighestHigh'] - data_williams['LowestLow'])

# -------------------------------
# IBKR Backtest with Dynamic Allocation
# -------------------------------
logger.info("Running IBKR backtest with dynamic allocation...")

# Initialize IBKR strategy tracking
ibkr_data_map = {
    'IBS_ES': data_es,
    'IBS_YM': data_ym, 
    'IBS_GC': data_gc,
    'IBS_NQ': data_nq,
    'IBS_ZQ': data_zq
}

symbol_map = {
    'IBS_ES': 'ES',
    'IBS_YM': 'YM', 
    'IBS_GC': 'GC',
    'IBS_NQ': 'NQ',
    'IBS_ZQ': 'ZQ'
}

# Start with ending capitals from original backtest
ibkr_strategy_values = {}
for strategy in allocation_percentages:
    ibkr_strategy_values[strategy] = {
        'capital': final_capitals[strategy],
        'in_position': False,
        'position': None,
        'equity_curve': []
    }

# Calculate total starting equity for IBKR period
total_ibkr_equity = sum(final_capitals.values())

# Get maximum rows for IBKR data
max_ibkr_rows = len(data_es) if not data_es.empty else 0

# Main IBKR backtest loop
for day_idx in range(max_ibkr_rows):
    current_date = None
    daily_total_equity = 0
    
    # Process IBS strategies
    for strategy_key in ['IBS_ES', 'IBS_YM', 'IBS_GC', 'IBS_NQ', 'IBS_ZQ']:
        symbol = symbol_map[strategy_key]
        data = ibkr_data_map[strategy_key]
        
        if data.empty or day_idx >= len(data):
            # No data - just track capital
            ibkr_strategy_values[strategy_key]['equity_curve'].append((current_date, ibkr_strategy_values[strategy_key]['capital']))
            daily_total_equity += ibkr_strategy_values[strategy_key]['capital']
            continue
        
        row = data.iloc[day_idx]
        current_date = row['Time']
        current_price = row['Last']
        
        # Calculate IBS
        ibs = (row['Last'] - row['Low']) / (row['High'] - row['Low'])
        
        multiplier = contract_specs[symbol]['multiplier']
        strategy_data = ibkr_strategy_values[strategy_key]
        
        # Calculate current position size based on total equity
        current_contracts = calculate_position_size(
            total_ibkr_equity, 
            allocation_percentages[strategy_key], 
            current_price, 
            multiplier
        )
        
        # Execute IBS trading logic
        if strategy_data['in_position']:
            if ibs > ibs_exit_threshold:
                # Exit position
                exit_price = current_price
                profit = (exit_price - strategy_data['position']['entry_price']) * multiplier * strategy_data['position']['contracts'] - commission_per_order * strategy_data['position']['contracts']
                strategy_data['capital'] += profit
                strategy_data['in_position'] = False
                strategy_data['position'] = None
        else:
            if ibs < ibs_entry_threshold:
                # Enter position
                entry_price = current_price
                strategy_data['in_position'] = True
                strategy_data['capital'] -= commission_per_order * current_contracts
                strategy_data['position'] = {
                    'entry_price': entry_price, 
                    'entry_time': current_date,
                    'contracts': current_contracts
                }
        
        # Calculate current equity
        if strategy_data['in_position']:
            unrealized = (current_price - strategy_data['position']['entry_price']) * multiplier * strategy_data['position']['contracts']
            equity = strategy_data['capital'] + unrealized
        else:
            equity = strategy_data['capital']
            
        strategy_data['equity_curve'].append((current_date, equity))
        daily_total_equity += equity
    
    # Process Williams strategy (uses ES data)
    if not data_es.empty and day_idx >= 1:
        # Need at least 2 days for Williams %R calculation
        es_window = data_es.iloc[max(0, day_idx-williams_period+1):day_idx+1]
        
        if len(es_window) >= williams_period:
            current_row = es_window.iloc[-1]
            current_date = current_row['Time']
            current_price = current_row['Last']
            
            # Calculate Williams %R
            highest_high = es_window['High'].max()
            lowest_low = es_window['Low'].min()
            williams_r = -100 * (highest_high - current_price) / (highest_high - lowest_low)
            
            strategy_data = ibkr_strategy_values['Williams']
            
            # Calculate current position size
            current_contracts = calculate_position_size(
                total_ibkr_equity, 
                allocation_percentages['Williams'], 
                current_price, 
                multiplier_es
            )
            
            # Execute Williams trading logic
            if strategy_data['in_position']:
                if day_idx > 0:
                    yesterdays_high = data_es.iloc[day_idx-1]['High']
                    if (current_price > yesterdays_high) or (williams_r > wr_sell_threshold):
                        # Exit position
                        exit_price = current_price
                        profit = (exit_price - strategy_data['position']['entry_price']) * multiplier_es * strategy_data['position']['contracts'] - commission_per_order * strategy_data['position']['contracts']
                        strategy_data['capital'] += profit
                        strategy_data['in_position'] = False
                        strategy_data['position'] = None
            else:
                if williams_r < wr_buy_threshold:
                    # Enter position
                    entry_price = current_price
                    strategy_data['in_position'] = True
                    strategy_data['capital'] -= commission_per_order * current_contracts
                    strategy_data['position'] = {
                        'entry_price': entry_price, 
                        'entry_time': current_date,
                        'contracts': current_contracts
                    }
            
            # Calculate current equity
            if strategy_data['in_position']:
                unrealized = (current_price - strategy_data['position']['entry_price']) * multiplier_es * strategy_data['position']['contracts']
                equity = strategy_data['capital'] + unrealized
            else:
                equity = strategy_data['capital']
                
            strategy_data['equity_curve'].append((current_date, equity))
            daily_total_equity += equity
    
    # Update total equity for next day
    if current_date is not None:
        total_ibkr_equity = daily_total_equity

# Close any remaining positions
for strategy_key in allocation_percentages:
    strategy_data = ibkr_strategy_values[strategy_key]
    
    if strategy_data['in_position'] and strategy_data['equity_curve']:
        if strategy_key.startswith('IBS_'):
            symbol = symbol_map[strategy_key]
            data = ibkr_data_map[strategy_key]
            if not data.empty:
                final_row = data.iloc[-1]
                final_price = final_row['Last']
                multiplier = contract_specs[symbol]['multiplier']
                
                profit = (final_price - strategy_data['position']['entry_price']) * multiplier * strategy_data['position']['contracts'] - commission_per_order * strategy_data['position']['contracts']
                strategy_data['capital'] += profit
                strategy_data['equity_curve'][-1] = (final_row['Time'], strategy_data['capital'])
        
        elif strategy_key == 'Williams' and not data_es.empty:
            final_row = data_es.iloc[-1]
            final_price = final_row['Last']
            
            profit = (final_price - strategy_data['position']['entry_price']) * multiplier_es * strategy_data['position']['contracts'] - commission_per_order * strategy_data['position']['contracts']
            strategy_data['capital'] += profit
            strategy_data['equity_curve'][-1] = (final_row['Time'], strategy_data['capital'])

# Convert to DataFrames for compatibility with existing code
equity_df_es = pd.DataFrame(ibkr_strategy_values['IBS_ES']['equity_curve'], columns=['Time', 'Equity'])
equity_df_es.set_index('Time', inplace=True)

equity_df_ym = pd.DataFrame(ibkr_strategy_values['IBS_YM']['equity_curve'], columns=['Time', 'Equity'])
equity_df_ym.set_index('Time', inplace=True)

equity_df_gc = pd.DataFrame(ibkr_strategy_values['IBS_GC']['equity_curve'], columns=['Time', 'Equity'])
equity_df_gc.set_index('Time', inplace=True)

equity_df_nq = pd.DataFrame(ibkr_strategy_values['IBS_NQ']['equity_curve'], columns=['Time', 'Equity'])
equity_df_nq.set_index('Time', inplace=True)

equity_df_zq = pd.DataFrame(ibkr_strategy_values['IBS_ZQ']['equity_curve'], columns=['Time', 'Equity'])
equity_df_zq.set_index('Time', inplace=True)

equity_df_w = pd.DataFrame(ibkr_strategy_values['Williams']['equity_curve'], columns=['Time', 'Equity'])
equity_df_w.set_index('Time', inplace=True)

# -------------------------------
# Combine Original and IBKR Equity Curves
# -------------------------------
logger.info("Combining original and IBKR equity curves...")

# Convert original results to DataFrames
original_equity_dfs = {}
for strategy in ['IBS_ES', 'IBS_YM', 'IBS_GC', 'IBS_NQ', 'IBS_ZQ', 'Williams']:
    equity_curve = original_results[strategy]['equity_curve']
    df = pd.DataFrame(equity_curve, columns=['Time', 'Equity'])
    df.set_index('Time', inplace=True)
    original_equity_dfs[strategy] = df

# IBKR equity DataFrames (already created above)
ibkr_equity_dfs = {
    'IBS_ES': equity_df_es,
    'IBS_YM': equity_df_ym, 
    'IBS_GC': equity_df_gc,
    'IBS_NQ': equity_df_nq,
    'IBS_ZQ': equity_df_zq,
    'Williams': equity_df_w
}

# Combine original and IBKR periods for each strategy
combined_equity_dfs = {}
for strategy in ['IBS_ES', 'IBS_YM', 'IBS_GC', 'IBS_NQ', 'IBS_ZQ', 'Williams']:
    original_df = original_equity_dfs[strategy]
    ibkr_df = ibkr_equity_dfs[strategy]
    
    # Combine the two periods
    combined_df = pd.concat([original_df, ibkr_df])
    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]  # Remove duplicate dates
    combined_df.sort_index(inplace=True)
    
    combined_equity_dfs[strategy] = combined_df

# Create overall date range
full_start_date = original_start_date
full_end_date = ibkr_end_date
common_dates = pd.date_range(start=full_start_date, end=full_end_date, freq='D')

# Reindex all combined DataFrames to common date range
for strategy in combined_equity_dfs:
    combined_equity_dfs[strategy] = combined_equity_dfs[strategy].reindex(common_dates, method='ffill')

# Calculate combined portfolio equity
combined_IBS = (combined_equity_dfs['IBS_ES']['Equity'] + 
                combined_equity_dfs['IBS_YM']['Equity'] +
                combined_equity_dfs['IBS_GC']['Equity'] + 
                combined_equity_dfs['IBS_NQ']['Equity'] +
                combined_equity_dfs['IBS_ZQ']['Equity'])

combined_equity = combined_IBS + combined_equity_dfs['Williams']['Equity']
combined_equity_df = pd.DataFrame({'Equity': combined_equity}, index=common_dates)

# -------------------------------
# Calculate Performance Metrics for Each Period
# -------------------------------

def calculate_performance_metrics(equity_df, start_date_str, end_date_str, initial_capital):
    """Calculate performance metrics for a given equity curve and period"""
    if equity_df.empty:
        return {}
    
    start_balance = equity_df['Equity'].iloc[0]
    final_balance = equity_df['Equity'].iloc[-1]
    total_return_pct = ((final_balance / initial_capital) - 1) * 100
    
    start_dt = pd.to_datetime(start_date_str)
    end_dt = pd.to_datetime(end_date_str)
    years = (end_dt - start_dt).days / 365.25
    annualized_return_pct = ((final_balance / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else np.nan
    
    returns = equity_df['Equity'].pct_change().dropna()
    volatility_annual_pct = returns.std() * np.sqrt(252) * 100 if len(returns) > 1 else np.nan
    
    peak = equity_df['Equity'].cummax()
    drawdown = (equity_df['Equity'] - peak) / peak
    max_drawdown_pct = drawdown.min() * 100
    max_drawdown_dollar = (peak - equity_df['Equity']).max()
    
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else np.nan
    downside_returns = returns[returns < 0]
    sortino = (returns.mean() / downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 and downside_returns.std() != 0 else np.nan
    calmar = (annualized_return_pct / abs(max_drawdown_pct)) if max_drawdown_pct != 0 else np.nan
    
    return {
        'start_balance': start_balance,
        'final_balance': final_balance,
        'total_return_pct': total_return_pct,
        'annualized_return_pct': annualized_return_pct,
        'volatility_annual_pct': volatility_annual_pct,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_drawdown_pct': max_drawdown_pct,
        'max_drawdown_dollar': max_drawdown_dollar,
        'years': years
    }

# Calculate metrics for original period
logger.info("Calculating performance metrics for original period...")
original_end_dt = pd.to_datetime(original_end_date)
original_period_equity = combined_equity_df[combined_equity_df.index <= original_end_dt].copy()
original_metrics = calculate_performance_metrics(original_period_equity, original_start_date, original_end_date, initial_capital)

# Calculate metrics for IBKR period
logger.info("Calculating performance metrics for IBKR period...")
ibkr_start_dt = pd.to_datetime(ibkr_start_date)
ibkr_period_equity = combined_equity_df[combined_equity_df.index >= ibkr_start_dt].copy()
# Use the ending capital from original period as the "initial" capital for IBKR period calculation
ibkr_initial_capital = original_metrics['final_balance']
ibkr_metrics = calculate_performance_metrics(ibkr_period_equity, ibkr_start_date, ibkr_end_date, ibkr_initial_capital)

# Calculate metrics for combined period
logger.info("Calculating performance metrics for combined period...")
combined_metrics = calculate_performance_metrics(combined_equity_df, full_start_date, full_end_date, initial_capital)

# -------------------------------
# Results Output
# -------------------------------
def format_metrics(metrics, period_name):
    """Format metrics into a results dictionary"""
    return {
        "Period": period_name,
        "Start Balance": f"${metrics['start_balance']:,.2f}",
        "Final Balance": f"${metrics['final_balance']:,.2f}",
        "Total Return": f"{metrics['total_return_pct']:.2f}%",
        "Annualized Return": f"{metrics['annualized_return_pct']:.2f}%" if not np.isnan(metrics['annualized_return_pct']) else "NaN",
        "Volatility (Annual)": f"{metrics['volatility_annual_pct']:.2f}%" if not np.isnan(metrics['volatility_annual_pct']) else "NaN",
        "Sharpe Ratio": f"{metrics['sharpe']:.2f}" if not np.isnan(metrics['sharpe']) else "NaN",
        "Sortino Ratio": f"{metrics['sortino']:.2f}" if not np.isnan(metrics['sortino']) else "NaN",
        "Calmar Ratio": f"{metrics['calmar']:.2f}" if not np.isnan(metrics['calmar']) else "NaN",
        "Max Drawdown (%)": f"{metrics['max_drawdown_pct']:.2f}%",
        "Max Drawdown ($)": f"${metrics['max_drawdown_dollar']:.2f}",
        "Duration (Years)": f"{metrics['years']:.2f}"
    }

# Format results for each period
original_results = format_metrics(original_metrics, f"{original_start_date} to {original_end_date}")
ibkr_results = format_metrics(ibkr_metrics, f"{ibkr_start_date} to {ibkr_end_date}")
combined_results = format_metrics(combined_metrics, f"{full_start_date} to {full_end_date}")

print("\n" + "="*80)
print("DYNAMIC ALLOCATION AGGREGATE PORTFOLIO PERFORMANCE")
print("="*80)

print(f"\n📊 ALLOCATION STRATEGY OVERVIEW")
print("-" * 60)
print("Using Dynamic Percentage-Based Capital Allocation:")
for strategy, pct in allocation_percentages.items():
    print(f"  • {strategy}: {pct*100:.0f}%")
print(f"  • Rebalancing Threshold: {rebalance_threshold*100:.0f}% drift")
print(f"  • Rebalancing Frequency: Every {rebalance_frequency_days} days")
print("\nKey Benefits:")
print("  • Position sizes scale with account equity")
print("  • Maintains consistent risk profile as capital grows")
print("  • No fixed dollar amounts - pure percentage allocation")
print("  • Enables compound growth across all strategies")

# Print Original Period Performance
print(f"\n🔵 ORIGINAL PERIOD PERFORMANCE (Historical Data)")
print("-" * 60)
for key, value in original_results.items():
    print(f"{key:25}: {value:>15}")

# Print IBKR Period Performance
print(f"\n🔴 IBKR PERIOD PERFORMANCE (Live Data)")
print("-" * 60)
for key, value in ibkr_results.items():
    print(f"{key:25}: {value:>15}")

# Print Combined Period Performance
print(f"\n📊 COMBINED PERIOD PERFORMANCE")
print("-" * 60)
for key, value in combined_results.items():
    print(f"{key:25}: {value:>15}")

print(f"\n" + "="*80)
print("DATA QUALITY REPORT")
print("="*80)
print(f"Original Period ({original_start_date} to {original_end_date}):")
for symbol in ['ES', 'YM', 'GC', 'NQ', 'ZQ']:
    bars_count = len(local_data[symbol]) if not local_data[symbol].empty else 0
    print(f"  {symbol} bars: {bars_count}")

print(f"\nIBKR Period ({ibkr_start_date} to {ibkr_end_date}):")
print(f"  ES bars retrieved: {len(data_es)}")
print(f"  YM bars retrieved: {len(data_ym) if not data_ym.empty else 0}")
print(f"  GC bars retrieved: {len(data_gc) if not data_gc.empty else 0}")
print(f"  NQ bars retrieved: {len(data_nq) if not data_nq.empty else 0}")
print(f"  ZQ bars retrieved: {len(data_zq) if not data_zq.empty else 0}")

# Performance Comparison
print(f"\n" + "="*80)
print("PERFORMANCE COMPARISON")
print("="*80)
print(f"{'Metric':<25} {'Original':<15} {'IBKR':<15} {'Combined':<15}")
print("-" * 75)

comparison_metrics = [
    ('Total Return', 'Total Return'),
    ('Annualized Return', 'Annualized Return'), 
    ('Volatility (Annual)', 'Volatility (Annual)'),
    ('Sharpe Ratio', 'Sharpe Ratio'),
    ('Max Drawdown (%)', 'Max Drawdown (%)')
]

for display_name, key in comparison_metrics:
    original_val = original_results[key].replace('%', '').replace('$', '').replace(',', '')
    ibkr_val = ibkr_results[key].replace('%', '').replace('$', '').replace(',', '')
    combined_val = combined_results[key].replace('%', '').replace('$', '').replace(',', '')
    print(f"{display_name:<25} {original_results[key]:<15} {ibkr_results[key]:<15} {combined_results[key]:<15}")

# -------------------------------
# Plot the Combined Equity Curve with Different Colors
# -------------------------------
plt.figure(figsize=(16, 8))

# Split data into original and IBKR periods for different colors
original_end = pd.to_datetime(original_end_date)
ibkr_start = pd.to_datetime(ibkr_start_date)

# Original period data
original_mask = combined_equity_df.index <= original_end
original_equity = combined_equity_df[original_mask]

# IBKR period data  
ibkr_mask = combined_equity_df.index >= ibkr_start
ibkr_equity = combined_equity_df[ibkr_mask]

# Plot original period in blue
plt.plot(original_equity.index, original_equity['Equity'], 
         label='Historical Data (Local CSV)', linewidth=2, color='steelblue')

# Plot IBKR period in red
plt.plot(ibkr_equity.index, ibkr_equity['Equity'], 
         label='Live Data (Interactive Brokers)', linewidth=2, color='red')

# Add vertical line at transition
plt.axvline(x=original_end, color='gray', linestyle='--', alpha=0.7, 
           label=f'Data Transition ({original_end_date})')

plt.title(f'Dynamic Allocation Portfolio ({full_start_date} to {full_end_date})\nPercentage-Based Allocation: IBS (10% each: ES, YM, GC, NQ, ZQ) + Williams (50%)')
plt.xlabel('Date')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Add text box with combined metrics
combined_textstr = f'COMBINED:\nTotal Return: {combined_metrics["total_return_pct"]:.1f}%\nAnnualized: {combined_metrics["annualized_return_pct"]:.1f}%\nSharpe: {combined_metrics["sharpe"]:.2f}\nMax DD: {combined_metrics["max_drawdown_pct"]:.1f}%'
props_combined = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, combined_textstr, transform=plt.gca().transAxes, fontsize=9,
         verticalalignment='top', bbox=props_combined)

# Add text box for original period metrics
original_textstr = f'HISTORICAL:\nReturn: {original_metrics["total_return_pct"]:.1f}%\nAnnualized: {original_metrics["annualized_return_pct"]:.1f}%\nSharpe: {original_metrics["sharpe"]:.2f}'
props_original = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
plt.text(0.02, 0.78, original_textstr, transform=plt.gca().transAxes, fontsize=9,
         verticalalignment='top', bbox=props_original)

# Add text box for IBKR period metrics
if not np.isnan(ibkr_metrics["total_return_pct"]):
    ibkr_textstr = f'LIVE DATA:\nReturn: {ibkr_metrics["total_return_pct"]:.1f}%\nAnnualized: {ibkr_metrics["annualized_return_pct"]:.1f}%\nSharpe: {ibkr_metrics["sharpe"]:.2f}'
    props_ibkr = dict(boxstyle='round', facecolor='lightcoral', alpha=0.8)
    plt.text(0.02, 0.58, ibkr_textstr, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=props_ibkr)

# Add separate text box showing transition point capital
transition_capital = combined_equity_df.loc[original_end, 'Equity']
transition_textstr = f'Capital at Transition:\n${transition_capital:,.0f}'
props_transition = dict(boxstyle='round', facecolor='lightcyan', alpha=0.8)
plt.text(0.02, 0.38, transition_textstr, transform=plt.gca().transAxes, fontsize=9,
         verticalalignment='top', bbox=props_transition)

plt.show()

logger.info("Backtest completed successfully!") 