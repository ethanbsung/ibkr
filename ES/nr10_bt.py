import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import pandas_ta as ta  # used for ATR on daily timeframe

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# -------------------------------
# Parameters & User Settings
# -------------------------------

# Input file paths (change these to switch timeframes)
data_file = "Data/es_1h_data.csv"      # Hourly timeframe file
daily_file = "Data/es_daily_data.csv"   # Daily file for computing the 200-day MA and daily ATR

# Backtest parameters
initial_capital = 10000.0           # starting account balance in dollars
lookback_period = 10                # lookback period for breakout calculation (can be adjusted)
volume_lookback = 10                # average volume period for volume filter
volume_multiplier = 1.5             # breakout bar volume must be >1.5x average volume
multiplier = 5                      # $5 per point (ES futures)
commission_per_order = 1.24         # commission per order (entry or exit)

# Daily ATR settings for stop loss and take profit
stop_atr_multiplier = 2            # stop loss = 1 ATR (daily)
tp_atr_multiplier = 3               # take profit = 2 ATR (daily)
atr_period = 14                     # ATR period for daily timeframe

# Custom start and end date (format: 'YYYY-MM-DD')
start_date = '2008-01-01'
end_date   = '2024-12-31'

# -------------------------------
# Data Preparation
# -------------------------------

# Read hourly timeframe data
data = pd.read_csv(data_file, parse_dates=['Time'])
data.sort_values('Time', inplace=True)
# Filter data based on custom date range
data = data[(data['Time'] >= start_date) & (data['Time'] <= end_date)].reset_index(drop=True)

# Read daily data for the 200-day moving average and ATR.
data_daily = pd.read_csv(daily_file, parse_dates=['Time'])
data_daily.sort_values('Time', inplace=True)
data_daily['MA200'] = data_daily['Last'].rolling(window=200, min_periods=1).mean()

# Calculate daily ATR using pandas-ta on the daily timeframe.
data_daily['ATR'] = ta.atr(high=data_daily['High'], low=data_daily['Low'], close=data_daily['Last'], length=atr_period)

# Prepare date column for merging
data_daily['Date'] = data_daily['Time'].dt.date
data['Date'] = data['Time'].dt.date

# Merge daily indicators (MA200 and daily ATR) onto the hourly data (matching by date)
data = pd.merge(data, data_daily[['Date', 'MA200', 'ATR']], on='Date', how='left')
data.drop(columns=['Date'], inplace=True)

# Calculate breakout levels and volume average on the hourly data using the chosen lookback period.
data['prev_high'] = data['High'].shift(1).rolling(window=lookback_period, min_periods=lookback_period).max()
data['prev_low'] = data['Low'].shift(1).rolling(window=lookback_period, min_periods=lookback_period).min()
data['avg_volume'] = data['Volume'].shift(1).rolling(window=volume_lookback, min_periods=volume_lookback).mean()

# -------------------------------
# Backtest Simulation
# -------------------------------
capital = initial_capital  # realized account equity
in_position = False        # flag if a trade is active
position = None            # dictionary to hold trade details
trade_results = []         # list to record completed trades
exposure_bars = 0          # count of bars when a trade is active
equity_curve = []          # list of (Time, mark-to-market Equity)

for i, row in data.iterrows():
    current_time = row['Time']
    current_price = row['Last']
    
    # Skip bars if daily ATR or breakout levels are not available.
    if pd.isna(row['ATR']) or pd.isna(row['prev_high']) or pd.isna(row['prev_low']):
        equity_curve.append((current_time, capital))
        continue

    # If a trade is active, update trailing stop each hour and check exit conditions.
    if in_position:
        exposure_bars += 1
        
        if position['direction'] == 'long':
            # Update maximum favorable price and adjust trailing stop accordingly.
            if row['High'] > position['max_price']:
                position['max_price'] = row['High']
                position['trailing_stop'] = position['max_price'] - stop_atr_multiplier * row['ATR']
            
            # Check take profit first
            if row['High'] >= position['take_profit']:
                exit_price = position['take_profit']
                trade_profit = (exit_price - position['entry_price']) * multiplier - commission_per_order
                trade_results.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'direction': 'long',
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': trade_profit
                })
                logger.info(f"Long TP hit at {current_time} | Exit Price: {exit_price:.2f} | Profit: {trade_profit:.2f}")
                capital += trade_profit
                in_position = False
                position = None
            # Then check stop loss (trailing stop)
            elif row['Low'] <= position['trailing_stop']:
                exit_price = position['trailing_stop']
                trade_profit = (exit_price - position['entry_price']) * multiplier - commission_per_order
                trade_results.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'direction': 'long',
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': trade_profit
                })
                logger.info(f"Long SL hit at {current_time} | Exit Price: {exit_price:.2f} | Profit: {trade_profit:.2f}")
                capital += trade_profit
                in_position = False
                position = None

        elif position['direction'] == 'short':
            # Update minimum favorable price and adjust trailing stop accordingly.
            if row['Low'] < position['min_price']:
                position['min_price'] = row['Low']
                position['trailing_stop'] = position['min_price'] + stop_atr_multiplier * row['ATR']
            
            # Check take profit first
            if row['Low'] <= position['take_profit']:
                exit_price = position['take_profit']
                trade_profit = (position['entry_price'] - exit_price) * multiplier - commission_per_order
                trade_results.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'direction': 'short',
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': trade_profit
                })
                logger.info(f"Short TP hit at {current_time} | Exit Price: {exit_price:.2f} | Profit: {trade_profit:.2f}")
                capital += trade_profit
                in_position = False
                position = None
            # Then check stop loss (trailing stop)
            elif row['High'] >= position['trailing_stop']:
                exit_price = position['trailing_stop']
                trade_profit = (position['entry_price'] - exit_price) * multiplier - commission_per_order
                trade_results.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'direction': 'short',
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': trade_profit
                })
                logger.info(f"Short SL hit at {current_time} | Exit Price: {exit_price:.2f} | Profit: {trade_profit:.2f}")
                capital += trade_profit
                in_position = False
                position = None

    # Check for entry signals only at the close of a candle and if not already in a position.
    if not in_position and i >= lookback_period:
        if row['Volume'] > volume_multiplier * row['avg_volume']:
            # Long entry: if the close breaks above the previous high and is above the MA200.
            if (row['Last'] > row['prev_high']) and (row['Last'] > row['MA200']):
                entry_price = row['Last']
                trailing_stop = entry_price - stop_atr_multiplier * row['ATR']
                take_profit = entry_price + tp_atr_multiplier * row['ATR']
                in_position = True
                capital -= commission_per_order  # commission on entry
                # Initialize 'max_price' to track the best price reached.
                position = {
                    'direction': 'long',
                    'entry_price': entry_price,
                    'entry_time': current_time,
                    'max_price': entry_price,       # New field for trailing stop calculation
                    'trailing_stop': trailing_stop,
                    'take_profit': take_profit
                }
                logger.info(f"Entering LONG at {current_time} | Entry Price: {entry_price:.2f} | Initial SL: {trailing_stop:.2f} | TP: {take_profit:.2f}")
            # Short entry: if the close breaks below the previous low and is below the MA200.
            elif (row['Last'] < row['prev_low']) and (row['Last'] < row['MA200']):
                entry_price = row['Last']
                trailing_stop = entry_price + stop_atr_multiplier * row['ATR']
                take_profit = entry_price - tp_atr_multiplier * row['ATR']
                in_position = True
                capital -= commission_per_order  # commission on entry
                # Initialize 'min_price' to track the best price reached.
                position = {
                    'direction': 'short',
                    'entry_price': entry_price,
                    'entry_time': current_time,
                    'min_price': entry_price,       # New field for trailing stop calculation
                    'trailing_stop': trailing_stop,
                    'take_profit': take_profit
                }
                logger.info(f"Entering SHORT at {current_time} | Entry Price: {entry_price:.2f} | Initial SL: {trailing_stop:.2f} | TP: {take_profit:.2f}")
    
    # Mark-to-market equity calculation.
    if in_position:
        if position['direction'] == 'long':
            unrealized = (current_price - position['entry_price']) * multiplier
        else:
            unrealized = (position['entry_price'] - current_price) * multiplier
        equity = capital + unrealized
    else:
        equity = capital
    equity_curve.append((current_time, equity))

# Close any open position at the end of the backtest period.
if in_position:
    row = data.iloc[-1]
    current_time = row['Time']
    current_price = row['Last']
    if position['direction'] == 'long':
        exit_price = current_price
        trade_profit = (exit_price - position['entry_price']) * multiplier - commission_per_order
    else:
        exit_price = current_price
        trade_profit = (position['entry_price'] - exit_price) * multiplier - commission_per_order
    trade_results.append({
        'entry_time': position['entry_time'],
        'exit_time': current_time,
        'direction': position['direction'],
        'entry_price': position['entry_price'],
        'exit_price': exit_price,
        'profit': trade_profit
    })
    logger.info(f"Closing open {position['direction'].upper()} at end {current_time} | Exit Price: {exit_price:.2f} | Profit: {trade_profit:.2f}")
    capital += trade_profit
    equity = capital
    in_position = False
    position = None
    equity_curve[-1] = (current_time, equity)

# Convert equity curve to DataFrame.
equity_df = pd.DataFrame(equity_curve, columns=['Time', 'Equity'])
equity_df.set_index('Time', inplace=True)

# -------------------------------
# Performance Metrics Calculation
# -------------------------------
total_bars = len(data)
exposure_time_percentage = (exposure_bars / total_bars) * 100
final_account_balance = capital
total_return_percentage = ((final_account_balance / initial_capital) - 1) * 100

start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)
years = (end_dt - start_dt).days / 365.25
annualized_return_percentage = ((final_account_balance / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else np.nan

initial_close = data['Last'].iloc[0]
benchmark_equity = (data.set_index('Time')['Last'] / initial_close) * initial_capital
benchmark_equity = benchmark_equity.reindex(equity_df.index, method='ffill').fillna(method='ffill')

equity_df['returns'] = equity_df['Equity'].pct_change()
volatility_annual = equity_df['returns'].std() * np.sqrt(1512) * 100

total_trades = len(trade_results)
winning_trades = [t for t in trade_results if t['profit'] > 0]
losing_trades = [t for t in trade_results if t['profit'] <= 0]
win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
profit_factor = (sum(t['profit'] for t in winning_trades) / abs(sum(t['profit'] for t in losing_trades))
                 if losing_trades and sum(t['profit'] for t in losing_trades) != 0 else np.nan)

avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else np.nan
avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else np.nan

sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(1512)
                if equity_df['returns'].std() != 0 else np.nan)
downside_std = equity_df[equity_df['returns'] < 0]['returns'].std()
sortino_ratio = (equity_df['returns'].mean() / downside_std * np.sqrt(1512)
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

drawdown_durations = []
current_duration = 0
for eq, peak in zip(equity_df['Equity'], equity_df['EquityPeak']):
    if eq < peak:
        current_duration += 1
    else:
        if current_duration > 0:
            drawdown_durations.append(current_duration)
        current_duration = 0
if current_duration > 0:
    drawdown_durations.append(current_duration)
if drawdown_durations:
    max_drawdown_duration_days = (max(drawdown_durations) * 4) / 24
    average_drawdown_duration_days = (np.mean(drawdown_durations) * 4) / 24
else:
    max_drawdown_duration_days = 0
    average_drawdown_duration_days = 0

results = {
    "Start Date": start_date,
    "End Date": end_date,
    "Exposure Time": f"{exposure_time_percentage:.2f}%",
    "Final Account Balance": f"${final_account_balance:,.2f}",
    "Equity Peak": f"${equity_df['Equity'].cummax().iloc[-1]:,.2f}",
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
    "Max Drawdown Duration": f"{max_drawdown_duration_days:.2f} days",
    "Average Drawdown Duration": f"{average_drawdown_duration_days:.2f} days",
    "Average Win ($)": f"${avg_win:,.2f}",
    "Average Loss ($)": f"${avg_loss:,.2f}",
}

print("\nPerformance Summary:")
for key, value in results.items():
    print(f"{key:30}: {value:>15}")

plt.figure(figsize=(14, 7))
plt.plot(equity_df.index, equity_df['Equity'], label='Strategy Equity')
plt.plot(benchmark_equity.index, benchmark_equity, label='Benchmark Equity (Buy & Hold)', alpha=0.7)
plt.title('Equity Curve: Strategy vs Benchmark')
plt.xlabel('Time')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()