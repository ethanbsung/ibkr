import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import math

# Parameters
SMA_PERIOD = 25
THRESHOLD = 10  # Points deviation to trigger trades (adjust as needed)
INITIAL_BALANCE = 5000
CONTRACT_SIZE = 1  # Number of MES contracts
STOP_LOSS = 5  # Points
TAKE_PROFIT = 10  # Points
MES_TICK_VALUE = 5  # $ per tick

# Load ES Futures Data
# Replace 'es_5m_data.csv' with your actual file path
# The CSV file should have 'date' and 'close' columns
es_data = pd.read_csv('Data/es_5m_data.csv', parse_dates=['date'])

# Ensure es_data is sorted
es_data.sort_values('date', inplace=True)

# Calculate 25-period SMA on ES Close Price
es_data['SMA_25'] = es_data['close'].rolling(window=SMA_PERIOD).mean()

# Calculate Deviation from SMA
es_data['Deviation'] = es_data['close'] - es_data['SMA_25']

# Initialize Backtest Variables
balance = INITIAL_BALANCE
position = 0  # 1 for long, -1 for short, 0 for no position
entry_price = 0
balance_history = []
positions_history = []
equity_history = []
trade_results = []
winning_trades = []
losing_trades = []
exposure_time = 0  # Number of periods in position
total_periods = len(es_data)
current_drawdown = 0
max_drawdown = 0
equity_peak = INITIAL_BALANCE
drawdowns = []
drawdown_durations = []
drawdown_start = None

# Iterate over the es_data to generate signals and execute trades
for idx, row in es_data.iterrows():
    if np.isnan(row['SMA_25']):
        # Not enough data to compute SMA
        balance_history.append(balance)
        equity_history.append(balance)
        positions_history.append(position)
        continue

    if position == 0:
        # Check for entry signals
        if row['Deviation'] <= -THRESHOLD:
            # Enter Long
            position = 1
            entry_price = row['close']
            stop_loss_price = entry_price - STOP_LOSS
            take_profit_price = entry_price + TAKE_PROFIT
            trade = {
                'Entry_Time': row['date'],
                'Entry_Price': entry_price,
                'Position': 'Long',
                'Stop_Loss': stop_loss_price,
                'Take_Profit': take_profit_price,
                'Exit_Time': None,
                'Exit_Price': None,
                'Profit': None,
                'Reason': None
            }
            trade_results.append(trade)
            exposure_time += 1
            print(f"Long Entry at {row['date']} | Price: {entry_price}")
        elif row['Deviation'] >= THRESHOLD:
            # Enter Short
            position = -1
            entry_price = row['close']
            stop_loss_price = entry_price + STOP_LOSS
            take_profit_price = entry_price - TAKE_PROFIT
            trade = {
                'Entry_Time': row['date'],
                'Entry_Price': entry_price,
                'Position': 'Short',
                'Stop_Loss': stop_loss_price,
                'Take_Profit': take_profit_price,
                'Exit_Time': None,
                'Exit_Price': None,
                'Profit': None,
                'Reason': None
            }
            trade_results.append(trade)
            exposure_time += 1
            print(f"Short Entry at {row['date']} | Price: {entry_price}")
    else:
        # Update exposure time
        exposure_time += 1

        # Check for exit signals based on mean reversion
        exit_reason = None
        exit_price = None
        profit = 0

        if position == 1:
            # Long Position
            # Check for mean reversion exit
            if row['Deviation'] >= 0:
                exit_price = row['close']
                exit_reason = 'Mean Reversion'
            # Check for Take Profit
            elif row['close'] >= take_profit_price:
                exit_price = take_profit_price
                exit_reason = 'Take Profit'
            # Check for Stop Loss
            elif row['close'] <= stop_loss_price:
                exit_price = stop_loss_price
                exit_reason = 'Stop Loss'
        elif position == -1:
            # Short Position
            # Check for mean reversion exit
            if row['Deviation'] <= 0:
                exit_price = row['close']
                exit_reason = 'Mean Reversion'
            # Check for Take Profit
            elif row['close'] <= take_profit_price:
                exit_price = take_profit_price
                exit_reason = 'Take Profit'
            # Check for Stop Loss
            elif row['close'] >= stop_loss_price:
                exit_price = stop_loss_price
                exit_reason = 'Stop Loss'

        if exit_reason:
            # Exit the position
            if position == 1:
                profit = (exit_price - entry_price) * CONTRACT_SIZE * MES_TICK_VALUE
            elif position == -1:
                profit = (entry_price - exit_price) * CONTRACT_SIZE * MES_TICK_VALUE
            balance += profit
            trade_results[-1].update({
                'Exit_Time': row['date'],
                'Exit_Price': exit_price,
                'Profit': profit,
                'Reason': exit_reason
            })
            if profit > 0:
                winning_trades.append(trade_results[-1])
            else:
                losing_trades.append(trade_results[-1])
            print(f"{trade_results[-1]['Position']} Exit at {row['date']} | "
                  f"Price: {exit_price} | Profit: {profit:.2f} | Reason: {exit_reason}")
            position = 0
            entry_price = 0

    # Update equity
    equity = balance
    if position != 0:
        # Mark to market
        if position == 1:
            unrealized_profit = (row['close'] - entry_price) * CONTRACT_SIZE * MES_TICK_VALUE
        elif position == -1:
            unrealized_profit = (entry_price - row['close']) * CONTRACT_SIZE * MES_TICK_VALUE
        equity = balance + unrealized_profit
    equity_history.append(equity)
    balance_history.append(balance if position == 0 else balance + unrealized_profit)
    positions_history.append(position)

    # Update Equity Peak and Drawdown
    if equity > equity_peak:
        equity_peak = equity
        if current_drawdown != 0:
            drawdowns.append(current_drawdown)
            if drawdown_start:
                drawdown_duration = (row['date'] - drawdown_start).days
                drawdown_durations.append(drawdown_duration)
                drawdown_start = None
            current_drawdown = 0
    else:
        drawdown = equity_peak - equity
        if drawdown > current_drawdown:
            current_drawdown = drawdown
            if drawdown_start is None:
                drawdown_start = row['date']
        if drawdown > max_drawdown:
            max_drawdown = drawdown

# Final Equity Peak Check
if current_drawdown != 0:
    drawdowns.append(current_drawdown)
    if drawdown_start:
        drawdown_duration = (es_data['date'].iloc[-1] - drawdown_start).days
        drawdown_durations.append(drawdown_duration)

# Add balance and position to es_data
es_data['Balance'] = balance_history
es_data['Equity'] = equity_history
es_data['Position'] = positions_history

# Calculate Returns
es_data['Returns'] = es_data['Equity'].pct_change().fillna(0)

# Performance Metrics Calculations
start_date = es_data['date'].min().strftime("%Y-%m-%d")
end_date = es_data['date'].max().strftime("%Y-%m-%d")
exposure_time_percentage = (exposure_time / total_periods) * 100
final_balance = balance
equity_peak_final = equity_peak
total_return = (final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100

# Annualized Return
total_days = (es_data['date'].iloc[-1] - es_data['date'].iloc[0]).days
annualized_return = ((final_balance / INITIAL_BALANCE) ** (365 / total_days) - 1) * 100 if total_days > 0 else 0

# Benchmark Return (Assuming no benchmark, set to 0 or calculate based on S&P 500)
benchmark_return = 0  # Placeholder

# Volatility (Annual)
volatility_annual = es_data['Returns'].std() * np.sqrt(252)

# Win Rate and Profit Factor
total_trades = len(trade_results)
winning_trades_count = len(winning_trades)
losing_trades_count = len(losing_trades)
win_rate = (winning_trades_count / total_trades * 100) if total_trades > 0 else 0
sum_profits = sum([trade['Profit'] for trade in winning_trades])
sum_losses = abs(sum([trade['Profit'] for trade in losing_trades]))
profit_factor = (sum_profits / sum_losses) if sum_losses != 0 else math.inf

# Sharpe Ratio
risk_free_rate = 0.0  # Assuming risk-free rate is 0
sharpe_ratio = (es_data['Returns'].mean() - risk_free_rate) / es_data['Returns'].std() * np.sqrt(252) if es_data['Returns'].std() != 0 else 0

# Sortino Ratio
downside_returns = es_data['Returns'][es_data['Returns'] < 0]
sortino_ratio = (es_data['Returns'].mean() - risk_free_rate) / downside_returns.std() * np.sqrt(252) if downside_returns.std() != 0 else 0

# Calmar Ratio
calmar_ratio = (total_return / (max_drawdown * MES_TICK_VALUE)) if max_drawdown != 0 else math.inf

# Max Drawdown and Average Drawdown
max_drawdown_percentage = (max_drawdown / equity_peak_final) * 100 if equity_peak_final != 0 else 0
average_drawdown = (np.mean(drawdowns) / equity_peak_final) * 100 if drawdowns else 0

# Drawdown Durations
max_drawdown_duration_days = max(drawdown_durations) if drawdown_durations else 0
average_drawdown_duration_days = (np.mean(drawdown_durations) if drawdown_durations else 0)

# Prepare Results
results = {
    "Start Date": start_date,
    "End Date": end_date,
    "Exposure Time": f"{exposure_time_percentage:.2f}%",
    "Final Account Balance": f"${final_balance:,.2f}",
    "Equity Peak": f"${equity_peak_final:,.2f}",
    "Total Return": f"{total_return:.2f}%",
    "Annualized Return": f"{annualized_return:.2f}%",
    "Benchmark Return": f"{benchmark_return:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Total Trades": total_trades,
    "Winning Trades": winning_trades_count,
    "Losing Trades": losing_trades_count,
    "Win Rate": f"{win_rate:.2f}%",
    "Profit Factor": f"{profit_factor:.2f}",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}",
    "Sortino Ratio": f"{sortino_ratio:.2f}",
    "Calmar Ratio": f"{calmar_ratio:.2f}",
    "Max Drawdown": f"{max_drawdown_percentage:.2f}%",
    "Average Drawdown": f"{average_drawdown:.2f}%",
    "Max Drawdown Duration": f"{max_drawdown_duration_days:.2f} days",
    "Average Drawdown Duration": f"{average_drawdown_duration_days:.2f} days",
}

# Print Performance Summary
print("\nPerformance Summary:")
for key, value in results.items():
    print(f"{key:25}: {value:>15}")
