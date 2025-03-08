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
data_file = "Data/es_daily_data.csv"  # CSV file with columns: Time, High, Low, Last, Volume (if available)

# Common backtest parameters for each strategy
initial_capital = 10000.0         # starting account balance per strategy (aggregate will be 2x this)
commission_per_order = 1.24       # commission per order (per contract)
num_contracts = 2                 # number of contracts to trade
multiplier = 5                    # each point move is worth $5 per contract

# Common date range for both strategies
start_date = '2000-01-01'
end_date   = '2024-12-31'

# Strategy 1 (IBS) parameters
ibs_entry_threshold = 0.1       # Enter when IBS < 0.1
ibs_exit_threshold  = 0.9       # Exit when IBS > 0.9

# Strategy 2 (Williams %R) parameters
williams_period = 2  # 2-day lookback
wr_buy_threshold  = -90
wr_sell_threshold = -30

# -------------------------------
# Data Preparation
# -------------------------------
# Read and sort the data, then filter by the common date range.
data = pd.read_csv(data_file, parse_dates=['Time'])
data.sort_values('Time', inplace=True)
data = data[(data['Time'] >= start_date) & (data['Time'] <= end_date)].reset_index(drop=True)

# For benchmark calculation, capture the initial and final close prices.
benchmark_initial_close = data['Last'].iloc[0]
benchmark_final_close   = data['Last'].iloc[-1]
benchmark_return = ((benchmark_final_close / benchmark_initial_close) - 1) * 100

# Create separate copies for each strategy.
data1 = data.copy()  # for IBS strategy
data2 = data.copy()  # for Williams %R strategy

# Strategy 1: Calculate IBS = (Last - Low) / (High - Low)
data1['IBS'] = (data1['Last'] - data1['Low']) / (data1['High'] - data1['Low'])

# Strategy 2: Calculate Williams %R
data2['HighestHigh'] = data2['High'].rolling(window=williams_period).max()
data2['LowestLow'] = data2['Low'].rolling(window=williams_period).min()
data2.dropna(subset=['HighestHigh', 'LowestLow'], inplace=True)
data2.reset_index(drop=True, inplace=True)
data2['WilliamsR'] = -100 * (data2['HighestHigh'] - data2['Last']) / (data2['HighestHigh'] - data2['LowestLow'])

# -------------------------------
# Backtest Simulation for Strategy 1 (IBS)
# -------------------------------
capital1 = initial_capital
in_position1 = False
position1 = None
trade_results1 = []
equity_curve1 = []

for i, row in data1.iterrows():
    current_time = row['Time']
    current_price = row['Last']
    
    # If in a position, check for exit condition: IBS > ibs_exit_threshold.
    if in_position1:
        if row['IBS'] > ibs_exit_threshold:
            exit_price = current_price
            profit = (exit_price - position1['entry_price']) * multiplier * num_contracts - commission_per_order * num_contracts
            trade_results1.append({
                'entry_time': position1['entry_time'],
                'exit_time': current_time,
                'entry_price': position1['entry_price'],
                'exit_price': exit_price,
                'profit': profit,
                'contracts': num_contracts
            })
            logger.info(f"Strategy 1 SELL at {current_time} | Exit Price: {exit_price:.2f} | Profit: {profit:.2f}")
            capital1 += profit
            in_position1 = False
            position1 = None
    else:
        # Entry condition: IBS < ibs_entry_threshold.
        if row['IBS'] < ibs_entry_threshold:
            entry_price = current_price
            in_position1 = True
            capital1 -= commission_per_order * num_contracts  # deduct entry commission
            position1 = {
                'entry_price': entry_price,
                'entry_time': current_time,
                'contracts': num_contracts
            }
            logger.info(f"Strategy 1 BUY at {current_time} | Entry Price: {entry_price:.2f}")

    # Mark-to-market equity calculation.
    if in_position1:
        unrealized = (current_price - position1['entry_price']) * multiplier * num_contracts
        equity = capital1 + unrealized
    else:
        equity = capital1
    equity_curve1.append((current_time, equity))

# Close any open position at the end.
if in_position1:
    row = data1.iloc[-1]
    current_time = row['Time']
    current_price = row['Last']
    exit_price = current_price
    profit = (exit_price - position1['entry_price']) * multiplier * num_contracts - commission_per_order * num_contracts
    trade_results1.append({
        'entry_time': position1['entry_time'],
        'exit_time': current_time,
        'entry_price': position1['entry_price'],
        'exit_price': exit_price,
        'profit': profit,
        'contracts': num_contracts
    })
    logger.info(f"Strategy 1 Closing at {current_time} | Exit Price: {exit_price:.2f} | Profit: {profit:.2f}")
    capital1 += profit
    equity_curve1[-1] = (current_time, capital1)
    in_position1 = False
    position1 = None

equity_df1 = pd.DataFrame(equity_curve1, columns=['Time', 'Equity'])
equity_df1.set_index('Time', inplace=True)

# -------------------------------
# Backtest Simulation for Strategy 2 (Williams %R)
# -------------------------------
capital2 = initial_capital
in_position2 = False
position2 = None
trade_results2 = []
equity_curve2 = []

for i in range(len(data2)):
    row = data2.iloc[i]
    current_time = row['Time']
    current_price = row['Last']
    current_wr = row['WilliamsR']
    
    # If in a position, check exit condition:
    # either today's close > yesterday's high OR Williams %R > wr_sell_threshold.
    if in_position2:
        if i > 0:
            yesterdays_high = data2['High'].iloc[i-1]
            if (current_price > yesterdays_high) or (current_wr > wr_sell_threshold):
                exit_price = current_price
                profit = (exit_price - position2['entry_price']) * multiplier * num_contracts - commission_per_order * num_contracts
                trade_results2.append({
                    'entry_time': position2['entry_time'],
                    'exit_time': current_time,
                    'entry_price': position2['entry_price'],
                    'exit_price': exit_price,
                    'profit': profit,
                    'contracts': num_contracts
                })
                logger.info(f"Strategy 2 SELL at {current_time} | Exit Price: {exit_price:.2f} | Profit: {profit:.2f}")
                capital2 += profit
                in_position2 = False
                position2 = None
    else:
        # Entry condition: Williams %R < wr_buy_threshold.
        if current_wr < wr_buy_threshold:
            entry_price = current_price
            in_position2 = True
            capital2 -= commission_per_order * num_contracts  # deduct entry commission
            position2 = {
                'entry_price': entry_price,
                'entry_time': current_time,
                'contracts': num_contracts
            }
            logger.info(f"Strategy 2 BUY at {current_time} | Entry Price: {entry_price:.2f}")
    
    # Mark-to-market equity calculation.
    if in_position2:
        unrealized = (current_price - position2['entry_price']) * multiplier * num_contracts
        equity = capital2 + unrealized
    else:
        equity = capital2
    equity_curve2.append((current_time, equity))

# Close any open position at the end.
if in_position2:
    row = data2.iloc[-1]
    current_time = row['Time']
    current_price = row['Last']
    exit_price = current_price
    profit = (exit_price - position2['entry_price']) * multiplier * num_contracts - commission_per_order * num_contracts
    trade_results2.append({
        'entry_time': position2['entry_time'],
        'exit_time': current_time,
        'entry_price': position2['entry_price'],
        'exit_price': exit_price,
        'profit': profit,
        'contracts': num_contracts
    })
    logger.info(f"Strategy 2 Closing at {current_time} | Exit Price: {exit_price:.2f} | Profit: {profit:.2f}")
    capital2 += profit
    equity_curve2[-1] = (current_time, capital2)
    in_position2 = False
    position2 = None

equity_df2 = pd.DataFrame(equity_curve2, columns=['Time', 'Equity'])
equity_df2.set_index('Time', inplace=True)

# -------------------------------
# Aggregate Performance: Combine the Two Equity Curves
# -------------------------------
# Reindex both DataFrames to a common daily date range.
common_dates = pd.date_range(start=start_date, end=end_date, freq='D')
equity_df1 = equity_df1.reindex(common_dates, method='ffill')
equity_df2 = equity_df2.reindex(common_dates, method='ffill')

# The combined equity is the sum of the two strategies (each starting with $10k, so aggregate = $20k).
combined_equity = equity_df1['Equity'] + equity_df2['Equity']
combined_equity_df = pd.DataFrame({'Equity': combined_equity}, index=common_dates)

# -------------------------------
# Calculate Aggregate Performance Metrics
# -------------------------------
initial_capital_combined = 2 * initial_capital  # $20,000 total initial capital
final_account_balance = combined_equity_df['Equity'].iloc[-1]
total_return_percentage = ((final_account_balance / initial_capital_combined) - 1) * 100

years = (combined_equity_df.index[-1] - combined_equity_df.index[0]).days / 365.25
annualized_return_percentage = ((final_account_balance / initial_capital_combined) ** (1 / years) - 1) * 100 if years > 0 else np.nan

combined_equity_df['returns'] = combined_equity_df['Equity'].pct_change()
volatility_annual = combined_equity_df['returns'].std() * np.sqrt(252) * 100

combined_equity_df['EquityPeak'] = combined_equity_df['Equity'].cummax()
combined_equity_df['Drawdown'] = (combined_equity_df['Equity'] - combined_equity_df['EquityPeak']) / combined_equity_df['EquityPeak']
max_drawdown_percentage = combined_equity_df['Drawdown'].min() * 100

sharpe_ratio = (combined_equity_df['returns'].mean() / combined_equity_df['returns'].std() * np.sqrt(252)
                if combined_equity_df['returns'].std() != 0 else np.nan)
downside_std = combined_equity_df[combined_equity_df['returns'] < 0]['returns'].std()
sortino_ratio = (combined_equity_df['returns'].mean() / downside_std * np.sqrt(252)
                 if downside_std != 0 else np.nan)
calmar_ratio = (annualized_return_percentage / abs(max_drawdown_percentage)
                if max_drawdown_percentage != 0 else np.nan)

# Combine trade information from both strategies.
all_trades = trade_results1 + trade_results2
total_trades = len(all_trades)
winning_trades = [t for t in all_trades if t['profit'] > 0]
losing_trades  = [t for t in all_trades if t['profit'] <= 0]
profit_factor = (sum(t['profit'] for t in winning_trades) / abs(sum(t['profit'] for t in losing_trades))
                 if losing_trades and sum(t['profit'] for t in losing_trades) != 0 else np.nan)

results = {
    "Start Date": start_date,
    "End Date": end_date,
    "Final Account Balance": f"${final_account_balance:,.2f}",
    "Total Return": f"{total_return_percentage:.2f}%",
    "Benchmark Return": f"{benchmark_return:.2f}%",
    "Annualized Return": f"{annualized_return_percentage:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Total Trades": total_trades,
    "Profit Factor": f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "NaN",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "NaN",
    "Sortino Ratio": f"{sortino_ratio:.2f}" if not np.isnan(sortino_ratio) else "NaN",
    "Calmar Ratio": f"{calmar_ratio:.2f}" if not np.isnan(calmar_ratio) else "NaN",
    "Max Drawdown (%)": f"{max_drawdown_percentage:.2f}%",
}

print("\nAggregate Performance Summary:")
for key, value in results.items():
    print(f"{key:30}: {value:>15}")

# -------------------------------
# Plot the Combined Equity Curve
# -------------------------------
plt.figure(figsize=(14, 7))
plt.plot(combined_equity_df.index, combined_equity_df['Equity'], label='Combined Strategy Equity')
plt.title('Aggregate Equity Curve of Combined Strategies')
plt.xlabel('Time')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()