import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# -------------------------------
# Parameters & User Settings
# -------------------------------
initial_capital = 30000.0         # total capital ($30,000)
commission_per_order = 1.24       # commission per order (per contract)

# Date range for all strategies
start_date = '2000-01-01'
end_date   = '2025-03-12'

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
    'ES': {'multiplier': 5, 'file': "Data/mes_daily_data.csv"},      # MES multiplier
    'YM': {'multiplier': 0.50, 'file': "Data/mym_daily_data.csv"},   # MYM multiplier  
    'GC': {'multiplier': 10, 'file': "Data/mgc_daily_data.csv"},     # MGC multiplier
    'NQ': {'multiplier': 2, 'file': "Data/mnq_daily_data.csv"},      # MNQ multiplier
    'ZQ': {'multiplier': 4167, 'file': "Data/zq_daily_data.csv"}     # ZQ multiplier
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
# Data Preparation & Benchmark (ES data for benchmark and Williams)
# -------------------------------
# Load ES data (used for benchmark, IBS ES, and Williams strategies)
data_es = pd.read_csv(contract_specs['ES']['file'], parse_dates=['Time'])
data_es.sort_values('Time', inplace=True)
data_es = data_es[(data_es['Time'] >= start_date) & (data_es['Time'] <= end_date)].reset_index(drop=True)

benchmark_initial_close = data_es['Last'].iloc[0]
benchmark_final_close   = data_es['Last'].iloc[-1]
benchmark_return = ((benchmark_final_close / benchmark_initial_close) - 1) * 100

benchmark_years = (data_es['Time'].iloc[-1] - data_es['Time'].iloc[0]).days / 365.25
benchmark_annualized_return = ((benchmark_final_close / benchmark_initial_close) ** (1 / benchmark_years) - 1) * 100

# -------------------------------
# Load Data for All Instruments
# -------------------------------
def load_instrument_data(symbol, start_date, end_date):
    """Load and filter instrument data"""
    try:
        file_path = contract_specs[symbol]['file']
        data = pd.read_csv(file_path, parse_dates=['Time'])
        data.sort_values('Time', inplace=True)
        data = data[(data['Time'] >= start_date) & (data['Time'] <= end_date)].reset_index(drop=True)
        return data
    except Exception as e:
        logger.error(f"Error loading {symbol} data: {e}")
        return pd.DataFrame()

# Load all instrument data
logger.info("Loading instrument data...")
instrument_data = {}
for symbol in ['ES', 'YM', 'GC', 'NQ', 'ZQ']:
    instrument_data[symbol] = load_instrument_data(symbol, start_date, end_date)
    if not instrument_data[symbol].empty:
        logger.info(f"Loaded {len(instrument_data[symbol])} bars for {symbol}")
    else:
        logger.warning(f"No data loaded for {symbol}")

# Use ES data for benchmark calculations
data_es = instrument_data['ES']

# -------------------------------
# Dynamic Allocation Backtest Engine
# -------------------------------
def run_dynamic_allocation_backtest():
    """Run unified backtest with dynamic percentage-based allocation"""
    
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
    max_rows = max(len(instrument_data[symbol]) for symbol in ['ES', 'YM', 'GC', 'NQ', 'ZQ'] if not instrument_data[symbol].empty)
    
    logger.info(f"Running dynamic allocation backtest for {max_rows} days...")
    
    # Main backtest loop
    for day_idx in range(max_rows):
        current_date = None
        daily_total_equity = 0
        
        # Process each IBS strategy
        for symbol in ['ES', 'YM', 'GC', 'NQ', 'ZQ']:
            strategy_key = f'IBS_{symbol}'
            
            if instrument_data[symbol].empty or day_idx >= len(instrument_data[symbol]):
                # No data available - just track capital
                if current_date is None and not instrument_data['ES'].empty and day_idx < len(instrument_data['ES']):
                    current_date = instrument_data['ES'].iloc[day_idx]['Time']
                strategy_values[strategy_key]['equity_curve'].append((current_date, strategy_values[strategy_key]['capital']))
                daily_total_equity += strategy_values[strategy_key]['capital']
                continue
            
            row = instrument_data[symbol].iloc[day_idx]
            current_date = row['Time']
            current_price = row['Last']
            
            # Calculate IBS with safety check for zero range
            range_val = row['High'] - row['Low']
            if range_val == 0:
                ibs = 0.5  # Neutral IBS when no range
            else:
                ibs = (row['Last'] - row['Low']) / range_val
            
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
        if not instrument_data['ES'].empty and day_idx >= 1 and day_idx < len(instrument_data['ES']):
            # Need at least 2 days for Williams %R calculation
            es_data = instrument_data['ES'].iloc[max(0, day_idx-williams_period+1):day_idx+1]
            
            if len(es_data) >= williams_period:
                current_row = es_data.iloc[-1]
                current_date = current_row['Time']
                current_price = current_row['Last']
                
                # Calculate Williams %R with safety check
                highest_high = es_data['High'].max()
                lowest_low = es_data['Low'].min()
                range_val = highest_high - lowest_low
                if range_val == 0:
                    williams_r = -50  # Neutral Williams %R when no range
                else:
                    williams_r = -100 * (highest_high - current_price) / range_val
                
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
                        yesterdays_high = instrument_data['ES'].iloc[day_idx-1]['High']
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
                if not instrument_data[symbol].empty:
                    final_row = instrument_data[symbol].iloc[-1]
                    final_price = final_row['Last']
                    multiplier = contract_specs[symbol]['multiplier']
                    
                    profit = (final_price - strategy_data['position']['entry_price']) * multiplier * strategy_data['position']['contracts'] - commission_per_order * strategy_data['position']['contracts']
                    strategy_data['capital'] += profit
                    strategy_data['equity_curve'][-1] = (final_row['Time'], strategy_data['capital'])
            
            elif strategy_key == 'Williams' and not instrument_data['ES'].empty:
                final_row = instrument_data['ES'].iloc[-1]
                final_price = final_row['Last']
                
                profit = (final_price - strategy_data['position']['entry_price']) * multiplier_es * strategy_data['position']['contracts'] - commission_per_order * strategy_data['position']['contracts']
                strategy_data['capital'] += profit
                strategy_data['equity_curve'][-1] = (final_row['Time'], strategy_data['capital'])
        
        results[strategy_key] = {
            'equity_curve': strategy_data['equity_curve'],
            'final_capital': strategy_data['capital']
        }
    
    return results

# Run the dynamic allocation backtest
logger.info("Starting dynamic allocation backtest...")
backtest_results = run_dynamic_allocation_backtest()

# -------------------------------
# Convert Results to DataFrames for Compatibility
# -------------------------------
def create_clean_equity_df(equity_curve):
    """Create DataFrame from equity curve and handle duplicates"""
    df = pd.DataFrame(equity_curve, columns=['Time', 'Equity'])
    df.set_index('Time', inplace=True)
    # Remove duplicate timestamps, keeping the last value
    df = df[~df.index.duplicated(keep='last')]
    # Sort by datetime index to ensure monotonic ordering
    df.sort_index(inplace=True)
    return df

# Convert backtest results to equity DataFrames
equity_df_es = create_clean_equity_df(backtest_results['IBS_ES']['equity_curve'])
equity_df_ym = create_clean_equity_df(backtest_results['IBS_YM']['equity_curve'])
equity_df_gc = create_clean_equity_df(backtest_results['IBS_GC']['equity_curve'])
equity_df_nq = create_clean_equity_df(backtest_results['IBS_NQ']['equity_curve'])
equity_df_zq = create_clean_equity_df(backtest_results['IBS_ZQ']['equity_curve'])
equity_df_w = create_clean_equity_df(backtest_results['Williams']['equity_curve'])

# -------------------------------
# Aggregate Performance: Combine Equity Curves
# -------------------------------
# Reindex all equity DataFrames to a common daily date range.
common_dates = pd.date_range(start=start_date, end=end_date, freq='D')
equity_df_es = equity_df_es.reindex(common_dates, method='ffill')
equity_df_ym = equity_df_ym.reindex(common_dates, method='ffill')
equity_df_gc = equity_df_gc.reindex(common_dates, method='ffill')
equity_df_nq = equity_df_nq.reindex(common_dates, method='ffill')
equity_df_zq = equity_df_zq.reindex(common_dates, method='ffill')
equity_df_w = equity_df_w.reindex(common_dates, method='ffill')

# Combined IBS equity is the sum of ES, YM, GC, NQ, and ZQ IBS strategies.
combined_IBS = (equity_df_es['Equity'] + equity_df_ym['Equity'] +
                equity_df_gc['Equity'] + equity_df_nq['Equity'] +
                equity_df_zq['Equity'])
# Overall combined equity is IBS + Williams
combined_equity = combined_IBS + equity_df_w['Equity']
combined_equity_df = pd.DataFrame({'Equity': combined_equity}, index=common_dates)

# -------------------------------
# Calculate Aggregate Performance Metrics
# -------------------------------
initial_capital_combined = initial_capital
final_account_balance = combined_equity_df['Equity'].iloc[-1]
total_return_percentage = ((final_account_balance / initial_capital_combined) - 1) * 100

years = (combined_equity_df.index[-1] - combined_equity_df.index[0]).days / 365.25
annualized_return_percentage = ((final_account_balance / initial_capital_combined) ** (1 / years) - 1) * 100 if years > 0 else np.nan

combined_equity_df['returns'] = combined_equity_df['Equity'].pct_change()
volatility_annual = combined_equity_df['returns'].std() * np.sqrt(252) * 100

combined_equity_df['EquityPeak'] = combined_equity_df['Equity'].cummax()
combined_equity_df['Drawdown'] = (combined_equity_df['Equity'] - combined_equity_df['EquityPeak']) / combined_equity_df['EquityPeak']
max_drawdown_percentage = combined_equity_df['Drawdown'].min() * 100

combined_equity_df['DrawdownAmount'] = combined_equity_df['EquityPeak'] - combined_equity_df['Equity']
max_drawdown_dollar = combined_equity_df['DrawdownAmount'].max()

sharpe_ratio = (combined_equity_df['returns'].mean() / combined_equity_df['returns'].std() * np.sqrt(252)
                if combined_equity_df['returns'].std() != 0 else np.nan)
downside_std = combined_equity_df[combined_equity_df['returns'] < 0]['returns'].std()
sortino_ratio = (combined_equity_df['returns'].mean() / downside_std * np.sqrt(252)
                 if downside_std != 0 else np.nan)
calmar_ratio = (annualized_return_percentage / abs(max_drawdown_percentage)
                if max_drawdown_percentage != 0 else np.nan)

results = {
    "Start Date": start_date,
    "End Date": end_date,
    "Final Account Balance": f"${final_account_balance:,.2f}",
    "Total Return": f"{total_return_percentage:.2f}%",
    "Benchmark Total Return": f"{benchmark_return:.2f}%",
    "Benchmark Annualized Return": f"{benchmark_annualized_return:.2f}%",
    "Annualized Return": f"{annualized_return_percentage:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "NaN",
    "Sortino Ratio": f"{sortino_ratio:.2f}" if not np.isnan(sortino_ratio) else "NaN",
    "Calmar Ratio": f"{calmar_ratio:.2f}" if not np.isnan(calmar_ratio) else "NaN",
    "Max Drawdown (%)": f"{max_drawdown_percentage:.2f}%",
    "Max Drawdown ($)": f"${max_drawdown_dollar:.2f}"
}

print("\n" + "="*80)
print("DYNAMIC ALLOCATION PORTFOLIO PERFORMANCE")
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

print(f"\n📈 PERFORMANCE SUMMARY")
print("-" * 60)
for key, value in results.items():
    print(f"{key:30}: {value:>15}")

print(f"\n💰 FINAL CAPITAL BY STRATEGY")
print("-" * 60)
for strategy in allocation_percentages:
    final_capital = backtest_results[strategy]['final_capital']
    initial_allocation = initial_capital * allocation_percentages[strategy]
    growth_factor = final_capital / initial_allocation
    print(f"{strategy:15}: ${final_capital:>10,.0f} ({growth_factor:>6.2f}x growth)")

total_final = sum(backtest_results[strategy]['final_capital'] for strategy in allocation_percentages)
total_growth = total_final / initial_capital
print(f"{'TOTAL':15}: ${total_final:>10,.0f} ({total_growth:>6.2f}x growth)")

# -------------------------------
# Plot the Combined Equity Curve
# -------------------------------
plt.figure(figsize=(14, 7))
plt.plot(combined_equity_df.index, combined_equity_df['Equity'], label='Dynamic Allocation Portfolio', color='steelblue', linewidth=2)
plt.title('Dynamic Allocation Portfolio Performance\nPercentage-Based Allocation: IBS (10% each: ES, YM, GC, NQ, ZQ) + Williams (50%)')
plt.xlabel('Time')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()