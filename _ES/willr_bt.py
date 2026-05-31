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

# Input file path for ES futures daily data
data_file = "Data/mes_daily_data.csv"  # File should include: Time, High, Low, Last, Volume (if available)

# Backtest parameters
initial_capital = 30000.0         # starting account balance in dollars
commission_per_order = 1.24       # commission per order (per contract)
multiplier = 5                    # each point move is worth $5 per contract

# Dynamic position sizing parameters (matching aggregate_port.py)
risk_multiplier = 3.0             # 3x larger positions for higher risk/reward
target_allocation_pct = 1.0       # 100% allocation for Williams strategy
min_contracts = 1                 # minimum number of contracts

# Custom start and end date (format: 'YYYY-MM-DD')
start_date = '2000-01-01'
end_date   = '2020-01-01'

# Williams %R parameters
williams_period = 2  # 2-day lookback
buy_threshold = -90
sell_threshold = -30

# -------------------------------
# Dynamic Position Sizing Function
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
# Data Preparation
# -------------------------------

# Read daily ES futures data
data = pd.read_csv(data_file, parse_dates=['Time'])
data.sort_values('Time', inplace=True)

# Filter data based on custom date range
data = data[(data['Time'] >= start_date) & (data['Time'] <= end_date)].reset_index(drop=True)

# -------------------------------
# Williams %R Calculation
# -------------------------------
# Formula: Williams %R = -100 * (HighestHigh(n) - Close) / (HighestHigh(n) - LowestLow(n))

# Rolling Highest High and Lowest Low for 'williams_period' days
data['HighestHigh'] = data['High'].rolling(window=williams_period).max()
data['LowestLow'] = data['Low'].rolling(window=williams_period).min()

# Shift out initial NaNs
data.dropna(subset=['HighestHigh', 'LowestLow'], inplace=True)
data.reset_index(drop=True, inplace=True)

# Calculate Williams %R
data['WilliamsR'] = -100 * (data['HighestHigh'] - data['Last']) / (data['HighestHigh'] - data['LowestLow'])

# -------------------------------
# Backtest Simulation
# -------------------------------

capital = initial_capital  # realized account equity
in_position = False        # flag if a trade is active
position = None            # dictionary to hold trade details
trade_results = []         # list to record completed trades
equity_curve = []          # list of (Time, mark-to-market Equity)

# For benchmark: Buy and Hold ES (enter at first available close)
initial_close = data['Last'].iloc[0]
benchmark_equity = (data.set_index('Time')['Last'] / initial_close) * initial_capital
benchmark_equity = benchmark_equity.reindex(data['Time'], method='ffill').fillna(method='ffill')

for i in range(len(data)):
    row = data.iloc[i]
    current_time = row['Time']
    current_price = row['Last']
    current_wr = row['WilliamsR']
    
    # If already in a position, check for exit condition
    if in_position:
        # We need to check:
        # 1) today's close > yesterday's high, OR
        # 2) Williams %R > -30
        # Also make sure i > 0 for "yesterday's high"
        if i > 0:
            yesterdays_high = data['High'].iloc[i-1]
            if (current_price > yesterdays_high) or (current_wr > sell_threshold):
                # Exit
                exit_price = current_price
                exit_contracts = position['contracts']  # Use stored contract count
                profit = (exit_price - position['entry_price']) * multiplier * exit_contracts
                # Subtract commission on exit
                profit -= commission_per_order * exit_contracts

                trade_results.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': profit,
                    'contracts': exit_contracts
                })
                logger.info(f"SELL signal at {current_time} | Exit Price: {exit_price:.2f} | Contracts: {exit_contracts} | Profit: {profit:.2f}")
                capital += profit
                in_position = False
                position = None
    else:
        # Buy condition: Williams %R < -90
        if current_wr < buy_threshold:
            entry_price = current_price
            
            # Calculate dynamic position size based on current equity
            num_contracts = calculate_position_size(
                capital, 
                target_allocation_pct, 
                current_price, 
                multiplier, 
                min_contracts
            )
            
            in_position = True
            # Subtract commission on entry
            capital -= commission_per_order * num_contracts
            position = {
                'entry_price': entry_price,
                'entry_time': current_time,
                'contracts': num_contracts
            }
            
            # Enhanced logging for position sizing
            target_dollar = capital * target_allocation_pct * risk_multiplier
            logger.info(f"BUY signal at {current_time} | Entry Price: {entry_price:.2f}")
            logger.info(f"Dynamic sizing: {target_allocation_pct*100:.0f}% * {risk_multiplier}x = ${target_dollar:,.0f} target")
            logger.info(f"Contracts: {num_contracts}")
    
    # Mark-to-market equity calculation
    if in_position:
        # For a long position, unrealized PnL = (current price - entry price) * multiplier * number of contracts.
        unrealized = (current_price - position['entry_price']) * multiplier * position['contracts']
        equity = capital + unrealized
    else:
        equity = capital
    equity_curve.append((current_time, equity))

# Close any open position at the end of the backtest period
if in_position:
    row = data.iloc[-1]
    current_time = row['Time']
    current_price = row['Last']
    exit_price = current_price
    final_contracts = position['contracts']  # Use stored contract count
    profit = (exit_price - position['entry_price']) * multiplier * final_contracts
    # Subtract commission on exit
    profit -= commission_per_order * final_contracts

    trade_results.append({
        'entry_time': position['entry_time'],
        'exit_time': current_time,
        'entry_price': position['entry_price'],
        'exit_price': exit_price,
        'profit': profit,
        'contracts': final_contracts
    })
    logger.info(f"Closing open position at end {current_time} | Exit Price: {exit_price:.2f} | Contracts: {final_contracts} | Profit: {profit:.2f}")
    capital += profit
    equity = capital
    in_position = False
    position = None
    equity_curve[-1] = (current_time, equity)

# Convert equity curve to DataFrame
equity_df = pd.DataFrame(equity_curve, columns=['Time', 'Equity'])
equity_df.set_index('Time', inplace=True)

# -------------------------------
# Performance Metrics Calculation
# -------------------------------
final_account_balance = capital
total_return_percentage = ((final_account_balance / initial_capital) - 1) * 100

start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)
years = (end_dt - start_dt).days / 365.25
annualized_return_percentage = ((final_account_balance / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else np.nan

equity_df['returns'] = equity_df['Equity'].pct_change()
# Using ~252 trading days per year
volatility_annual = equity_df['returns'].std() * np.sqrt(252) * 100

total_trades = len(trade_results)
winning_trades = [t for t in trade_results if t['profit'] > 0]
losing_trades = [t for t in trade_results if t['profit'] <= 0]
win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
profit_factor = (sum(t['profit'] for t in winning_trades) / abs(sum(t['profit'] for t in losing_trades))
                 if losing_trades and sum(t['profit'] for t in losing_trades) != 0 else np.nan)

avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else np.nan
avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else np.nan

sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252)
                if equity_df['returns'].std() != 0 else np.nan)
downside_std = equity_df[equity_df['returns'] < 0]['returns'].std()
sortino_ratio = (equity_df['returns'].mean() / downside_std * np.sqrt(252)
                 if downside_std != 0 else np.nan)

equity_df['EquityPeak'] = equity_df['Equity'].cummax()
equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['EquityPeak']) / equity_df['EquityPeak']
max_drawdown_percentage = equity_df['Drawdown'].min() * 100
equity_df['DrawdownDollar'] = equity_df['EquityPeak'] - equity_df['Equity']
max_drawdown_dollar = equity_df['DrawdownDollar'].max()
average_drawdown_dollar = equity_df.loc[equity_df['DrawdownDollar'] > 0, 'DrawdownDollar'].mean()
average_drawdown_percentage = equity_df['Drawdown'].mean() * 100
calmar_ratio = (annualized_return_percentage / abs(max_drawdown_percentage)
                if max_drawdown_percentage != 0 else np.nan)

results = {
    "Start Date": start_date,
    "End Date": end_date,
    "Risk Multiplier": f"{risk_multiplier}x",
    "Target Allocation": f"{target_allocation_pct*100:.0f}%",
    "Final Account Balance": f"${final_account_balance:,.2f}",
    "Total Return": f"{total_return_percentage:.2f}%",
    "Annualized Return": f"{annualized_return_percentage:.2f}%",
    "Benchmark Return": f"{((data['Last'].iloc[-1]/initial_close)-1)*100:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Total Trades": total_trades,
    "Winning Trades": len(winning_trades),
    "Losing Trades": len(losing_trades),
    "Win Rate": f"{win_rate:.2f}%",
    "Profit Factor": f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "NaN",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "NaN",
    "Sortino Ratio": f"{sortino_ratio:.2f}" if not np.isnan(sortino_ratio) else "NaN",
    "Calmar Ratio": f"{calmar_ratio:.2f}" if not np.isnan(calmar_ratio) else "NaN",
    "Max Drawdown (%)": f"{max_drawdown_percentage:.2f}%",
    "Average Drawdown (%)": f"{average_drawdown_percentage:.2f}%",
    "Max Drawdown ($)": f"${max_drawdown_dollar:,.2f}",
    "Average Drawdown ($)": f"${average_drawdown_dollar:,.2f}",
    "Average Win ($)": f"${avg_win:,.2f}",
    "Average Loss ($)": f"${avg_loss:,.2f}",
}

print("\n" + "="*80)
print("ENHANCED RISK WILLIAMS %R STRATEGY PERFORMANCE")
print("="*80)

print(f"\n📊 STRATEGY OVERVIEW")
print("-" * 60)
print(f"Williams %R Strategy with Enhanced Risk:")
print(f"  • Risk Multiplier: {risk_multiplier}x (LARGER POSITION SIZES)")
print(f"  • Target Allocation: {target_allocation_pct*100:.0f}%")
print(f"  • Dynamic position sizing scales with account equity")
print(f"  • Enhanced risk/reward with larger position sizes")

print(f"\n📈 PERFORMANCE SUMMARY")
print("-" * 60)
for key, value in results.items():
    print(f"{key:30}: {value:>15}")

# -------------------------------
# Plot Equity Curve
# -------------------------------
plt.figure(figsize=(14, 8))
plt.plot(equity_df.index, equity_df['Equity'], label='Enhanced Risk Williams %R Strategy', color='steelblue', linewidth=2)
# Optionally plot benchmark
# plt.plot(benchmark_equity.index, benchmark_equity, label='Benchmark (Buy & Hold)', alpha=0.7)
plt.title(f'Enhanced Risk Williams %R Strategy Performance ({risk_multiplier}x Risk Multiplier)\nDynamic Position Sizing with {target_allocation_pct*100:.0f}% Allocation')
plt.xlabel('Time')
plt.ylabel('Account Balance ($)')

# Format y-axis to show dollar amounts clearly
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Add horizontal grid lines at key dollar amounts
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()