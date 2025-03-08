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
data_file = "Data/es_daily_data.csv"  # File should include: Time, High, Low, Last, Volume (if available)

# Backtest parameters
initial_capital = 10000.0         # starting account balance in dollars
commission_per_order = 1.24       # commission per order (per contract)
num_contracts = 1                 # number of contracts to trade
multiplier = 5                    # each point move is worth $5 per contract

# Custom start and end date (format: 'YYYY-MM-DD')
start_date = '2000-01-01'
end_date   = '2024-12-31'

# CCI parameters
cci_period = 2         # N-day lookback for CCI
cci_buy_threshold = -10
cci_sell_threshold = 10

# -------------------------------
# Data Preparation
# -------------------------------

# Read daily ES futures data
data = pd.read_csv(data_file, parse_dates=['Time'])
data.sort_values('Time', inplace=True)

# Filter data based on custom date range
data = data[(data['Time'] >= start_date) & (data['Time'] <= end_date)].reset_index(drop=True)

# -------------------------------
# CCI Calculation
# -------------------------------
# Typical Price (TP) = (High + Low + Close) / 3
data['TP'] = (data['High'] + data['Low'] + data['Last']) / 3.0

# Rolling mean of TP
data['TP_SMA'] = data['TP'].rolling(window=cci_period).mean()

# Mean deviation
def mean_deviation(series):
    return np.mean(np.abs(series - series.mean()))

# We apply a rolling apply to get mean deviation of TP
data['MeanDev'] = data['TP'].rolling(window=cci_period).apply(mean_deviation, raw=False)

# CCI = (TP - TP_SMA) / (0.015 * MeanDev)
# We shift enough so we have valid data
data['CCI'] = (data['TP'] - data['TP_SMA']) / (0.015 * data['MeanDev'])

# Drop rows that have NaN in CCI
data.dropna(subset=['CCI'], inplace=True)
data.reset_index(drop=True, inplace=True)

# -------------------------------
# Strategy Backtest
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
    current_cci = row['CCI']
    
    # If already in a position, check for exit condition
    if in_position:
        # Sell condition: CCI crosses above the sell threshold
        # We'll check the previous CCI to see if we "crossed" from below
        prev_cci = data['CCI'].iloc[i-1] if i > 0 else np.nan
        
        if (prev_cci <= cci_sell_threshold) and (current_cci > cci_sell_threshold):
            exit_price = current_price  # exit at the close price
            # Calculate profit based on the multiplier and number of contracts.
            profit = (exit_price - position['entry_price']) * multiplier * num_contracts
            # Subtract commission on exit
            profit -= commission_per_order * num_contracts
            
            trade_results.append({
                'entry_time': position['entry_time'],
                'exit_time': current_time,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'profit': profit,
                'contracts': num_contracts
            })
            logger.info(f"SELL signal at {current_time} | Exit Price: {exit_price:.2f} | Profit: {profit:.2f}")
            capital += profit  # update realized capital
            in_position = False
            position = None
    else:
        # Buy condition: CCI crosses below the buy threshold
        # We'll check the previous CCI to see if we "crossed" from above
        if i > 0:
            prev_cci = data['CCI'].iloc[i-1]
            if (prev_cci >= cci_buy_threshold) and (current_cci < cci_buy_threshold):
                entry_price = current_price  # enter at the close price
                in_position = True
                # Deduct commission on entry
                capital -= commission_per_order * num_contracts
                position = {
                    'entry_price': entry_price,
                    'entry_time': current_time,
                    'contracts': num_contracts
                }
                logger.info(f"BUY signal at {current_time} | Entry Price: {entry_price:.2f} | Contracts: {num_contracts}")
    
    # Mark-to-market equity calculation
    if in_position:
        # For a long position, unrealized PnL = (current price - entry price) * multiplier * number of contracts.
        unrealized = (current_price - position['entry_price']) * multiplier * num_contracts
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
    profit = (exit_price - position['entry_price']) * multiplier * num_contracts
    # Subtract commission on exit
    profit -= commission_per_order * num_contracts
    trade_results.append({
        'entry_time': position['entry_time'],
        'exit_time': current_time,
        'entry_price': position['entry_price'],
        'exit_price': exit_price,
        'profit': profit,
        'contracts': num_contracts
    })
    logger.info(f"Closing open position at end {current_time} | Exit Price: {exit_price:.2f} | Profit: {profit:.2f}")
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

# Annualization
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

print("\nPerformance Summary:")
for key, value in results.items():
    print(f"{key:30}: {value:>15}")

# -------------------------------
# Plot Equity Curve
# -------------------------------
plt.figure(figsize=(14, 7))
plt.plot(equity_df.index, equity_df['Equity'], label='Strategy Equity')
# Optionally plot benchmark
# plt.plot(benchmark_equity.index, benchmark_equity, label='Benchmark (Buy & Hold)', alpha=0.7)
plt.title('Equity Curve: CCI Strategy')
plt.xlabel('Time')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()