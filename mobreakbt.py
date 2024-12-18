import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ----------------------- User Parameters -----------------------

# Paths to your CSV data files
daily_data_file = 'es_daily_data.csv'       # Daily ES futures data
intraday_data_file = 'es_5m_data.csv'     # 5-minute ES futures data

# Contract and PnL Parameters
multiplier = 5            # Trading 5 MES contracts
point_value_mes = 5.0     # $5 per point per MES contract

# Initial account balance
initial_cash = 5000.0    # Starting with $100,000

# Custom Backtest Date Range (Set to None to include all data)
backtest_start_date = '2022-12-18'  # Format: 'YYYY-MM-DD' or None
backtest_end_date = '2023-12-31'    # Format: 'YYYY-MM-DD' or None

# Moving Average Parameters
ma_period = 50            # 50-day moving average for trend identification

# Volume Confirmation Parameters
volume_ma_period = 20     # 20-day moving average for volume
volume_multiplier = 1.5   # Current volume must be > 1.5 * average volume

# Trading Costs
commission_per_side = 0.62  # $0.62 per side
slippage_per_side = 0.25    # $0.25 per side
total_cost_per_trade = (commission_per_side + slippage_per_side) * 2  # $1.74 per trade

# Stop Loss Parameters
stop_loss_points = 10  # Stop loss set 10 points away from entry price

# ----------------------- End of User Parameters -----------------------

# Function to parse and validate dates
def parse_date(date_str):
    if date_str is None:
        return None
    try:
        return pd.to_datetime(date_str)
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Use 'YYYY-MM-DD'.")

# Parse start and end dates
start_date = parse_date(backtest_start_date)
end_date = parse_date(backtest_end_date)

# Load daily data
daily_df = pd.read_csv(daily_data_file, parse_dates=['date'])
daily_df.set_index('date', inplace=True)
daily_df = daily_df.sort_index()

# Ensure the DataFrame's index is timezone-naive (since daily data typically doesn't include timezone)
if daily_df.index.tz is not None:
    daily_df.index = daily_df.index.tz_convert(None)

# Ensure the data is clean and properly typed
required_daily_columns = ['open','high','low','close','volume','average','barCount','contract']
for col in required_daily_columns:
    if col not in daily_df.columns:
        raise ValueError(f"Missing column '{col}' in daily data.")
daily_df[required_daily_columns] = daily_df[required_daily_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values in required columns
daily_df.dropna(subset=required_daily_columns, inplace=True)

# Apply date filters to daily data
if start_date:
    daily_df = daily_df[daily_df.index >= start_date]
if end_date:
    daily_df = daily_df[daily_df.index <= end_date]

# Check if daily data is available after filtering
if daily_df.empty:
    raise ValueError("No daily data available for the specified date range.")

# Calculate Moving Averages for Trend
daily_df['MA'] = daily_df['close'].rolling(window=ma_period).mean()

# Drop rows where moving average is not available
daily_df.dropna(subset=['MA'], inplace=True)

# Load intraday (5-minute) data
intraday_df = pd.read_csv(intraday_data_file, parse_dates=['date'])
intraday_df.set_index('date', inplace=True)
intraday_df = intraday_df.sort_index()

# Ensure the DataFrame's index is timezone-naive
if intraday_df.index.tz is not None:
    intraday_df.index = intraday_df.index.tz_convert(None)

# Ensure the data is clean and properly typed
required_intraday_columns = ['open','high','low','close','volume','average','barCount','contract']
for col in required_intraday_columns:
    if col not in intraday_df.columns:
        raise ValueError(f"Missing column '{col}' in intraday data.")
intraday_df[required_intraday_columns] = intraday_df[required_intraday_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values in required columns
intraday_df.dropna(subset=required_intraday_columns, inplace=True)

# Apply date filters to intraday data
if start_date:
    intraday_df = intraday_df[intraday_df.index >= start_date]
if end_date:
    intraday_df = intraday_df[intraday_df.index <= end_date]

# Check if intraday data is available after filtering
if intraday_df.empty:
    raise ValueError("No intraday data available for the specified date range.")

# Calculate Moving Averages for Volume Confirmation
intraday_df['Volume_MA'] = intraday_df['volume'].rolling(window=volume_ma_period).mean()

# Drop rows where Volume_MA is not available
intraday_df.dropna(subset=['Volume_MA'], inplace=True)

# Initialize variables for backtest
trades = []              # List to store trade details
cash = initial_cash      # Starting cash
equity = initial_cash    # Current equity
equity_curve = []        # To store equity over time
total_trading_days = 0
trading_days_with_trades = 0

# Iterate over each trading day starting from the second day
for i in range(1, len(daily_df)):
    current_day = daily_df.index[i]
    prev_day = daily_df.index[i - 1]
    total_trading_days += 1
    
    # Determine the prevailing trend
    current_close = daily_df.loc[current_day, 'close']
    current_ma = daily_df.loc[current_day, 'MA']
    
    if current_close > current_ma:
        trend = 'Uptrend'
    elif current_close < current_ma:
        trend = 'Downtrend'
    else:
        trend = 'Sideways'
    
    # Get previous day's high and low
    prev_high = daily_df.loc[prev_day, 'high']
    prev_low = daily_df.loc[prev_day, 'low']
    
    # Filter intraday data for the current day
    # Assuming 'date' in intraday data includes both date and time
    intraday_day = intraday_df.loc[current_day.strftime('%Y-%m-%d')]
    
    # Skip if no intraday data for the current day
    if intraday_day.empty:
        equity_curve.append({'date': current_day, 'equity': equity})
        continue
    
    # Initialize trade variables for the day
    direction = 0  # 1 for long, -1 for short
    entry_price = None
    entry_time = None
    exit_price = None
    exit_time = None
    exit_reason = None  # 'Take Profit', 'Stop Loss', 'End of Day'
    
    # Iterate over each 5-minute bar to detect breakout and manage exit conditions
    for timestamp, row in intraday_day.iterrows():
        if direction == 0:
            # Volume confirmation
            volume_confirmed = row['volume'] > (volume_multiplier * row['Volume_MA'])
            
            if trend == 'Uptrend':
                # Long breakout condition
                if row['high'] > prev_high and volume_confirmed:
                    direction = 1
                    entry_price = prev_high  # Enter at breakout price
                    entry_time = timestamp
                    trading_days_with_trades += 1
                    # Define Take Profit and Stop Loss
                    take_profit = daily_df.loc[prev_day, 'high']
                    stop_loss = entry_price - stop_loss_points
            elif trend == 'Downtrend':
                # Short breakout condition
                if row['low'] < prev_low and volume_confirmed:
                    direction = -1
                    entry_price = prev_low  # Enter at breakout price
                    entry_time = timestamp
                    trading_days_with_trades += 1
                    # Define Take Profit and Stop Loss
                    take_profit = daily_df.loc[prev_day, 'low']
                    stop_loss = entry_price + stop_loss_points
            else:
                # No clear trend; do not enter any trades
                continue
        else:
            # Trade is open; check for exit conditions
            if direction == 1:
                # Long Position
                if row['high'] >= take_profit:
                    exit_price = take_profit
                    exit_time = timestamp
                    exit_reason = 'Take Profit'
                elif row['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_time = timestamp
                    exit_reason = 'Stop Loss'
            elif direction == -1:
                # Short Position
                if row['low'] <= take_profit:
                    exit_price = take_profit
                    exit_time = timestamp
                    exit_reason = 'Take Profit'
                elif row['high'] >= stop_loss:
                    exit_price = stop_loss
                    exit_time = timestamp
                    exit_reason = 'Stop Loss'
            
            # If exit condition met, close the trade
            if exit_price is not None:
                # Calculate PnL
                pnl = (exit_price - entry_price) * direction * point_value_mes * multiplier
                
                # Subtract transaction costs (entry and exit)
                pnl -= total_cost_per_trade
                
                # Update cash and equity
                cash += pnl
                equity = cash  # Positions are closed
                
                # Record trade
                trades.append({
                    'date': current_day,
                    'direction': 'Long' if direction == 1 else 'Short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_mes': pnl,
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'exit_reason': exit_reason
                })
                
                # Reset trade variables
                direction = 0
                entry_price = None
                entry_time = None
                exit_price = None
                exit_time = None
                exit_reason = None
                break  # Exit after trade is closed
    
    # After iterating through intraday bars, check if trade is still open and needs to be closed at EOD
    if direction != 0 and entry_price is not None:
        # Exit at day's final close
        exit_price = daily_df.loc[current_day, 'close']
        exit_time = current_day
        exit_reason = 'End of Day'
        
        # Calculate PnL
        pnl = (exit_price - entry_price) * direction * point_value_mes * multiplier
        
        # Subtract transaction costs (entry and exit)
        pnl -= total_cost_per_trade
        
        # Update cash and equity
        cash += pnl
        equity = cash  # Positions are closed
        
        # Record trade
        trades.append({
            'date': current_day,
            'direction': 'Long' if direction == 1 else 'Short',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_mes': pnl,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'exit_reason': exit_reason
        })
    
    # Record equity for the day
    equity_curve.append({'date': current_day, 'equity': equity})

# Create DataFrame for trades
trades_df = pd.DataFrame(trades)

# Create Equity Curve DataFrame
equity_df = pd.DataFrame(equity_curve)
equity_df.set_index('date', inplace=True)

# Calculate Daily Returns
equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)

# Calculate Cumulative Returns
equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod()

# Performance Metrics Calculation

# Start and End Dates
actual_start_date = equity_df.index.min()
actual_end_date = equity_df.index.max()

# Exposure Time: Percentage of days with trades
exposure_time_percentage = (trading_days_with_trades / total_trading_days) * 100 if total_trading_days > 0 else 0.0

# Final Account Balance
final_account_balance = equity_df['equity'].iloc[-1]

# Equity Peak
equity_df['equity_peak'] = equity_df['equity'].cummax()

# Drawdown
equity_df['drawdown'] = (equity_df['equity'] - equity_df['equity_peak']) / equity_df['equity_peak']

# Max Drawdown
max_drawdown = equity_df['drawdown'].min() * 100 if not equity_df['drawdown'].empty else 0.0

# Average Drawdown
average_drawdown = equity_df['drawdown'].mean() * 100 if not equity_df['drawdown'].empty else 0.0

# Calculate Drawdown Durations
drawdown = equity_df['drawdown']
is_drawdown = drawdown < 0
drawdown_shift = is_drawdown.shift(1, fill_value=False)
drawdown_start = (~drawdown_shift) & is_drawdown
drawdown_end = drawdown_shift & (~is_drawdown)

drawdown_starts = equity_df.index[drawdown_start].tolist()
drawdown_ends = equity_df.index[drawdown_end].tolist()

# If a drawdown is ongoing at the end, add the last date as end
if len(drawdown_starts) > len(drawdown_ends):
    drawdown_ends.append(equity_df.index[-1])

# Calculate durations
drawdown_durations = []
for start, end in zip(drawdown_starts, drawdown_ends):
    duration = (end - start).days
    drawdown_durations.append(duration)

if drawdown_durations:
    max_drawdown_duration_days = max(drawdown_durations)
    average_drawdown_duration_days = sum(drawdown_durations) / len(drawdown_durations)
else:
    max_drawdown_duration_days = 0
    average_drawdown_duration_days = 0

# Equity Peak
equity_peak = equity_df['equity_peak'].max()

# Total Return
total_return = (final_account_balance - initial_cash) / initial_cash * 100  # As percentage

# Annualized Return
num_years = (actual_end_date - actual_start_date).days / 365.25
annualized_return = ((final_account_balance / initial_cash) ** (1 / num_years) - 1) * 100 if num_years > 0 else 0.0

# Volatility (Annual)
volatility = equity_df['returns'].std() * np.sqrt(252) * 100  # As percentage

# Total Trades
total_trades = len(trades_df)

# Winning Trades
winning_trades = trades_df[trades_df['pnl_mes'] > 0]
num_winning_trades = len(winning_trades)

# Losing Trades
losing_trades = trades_df[trades_df['pnl_mes'] < 0]
num_losing_trades = len(losing_trades)

# Win Rate
win_rate = (num_winning_trades / total_trades * 100) if total_trades > 0 else 0.0

# Profit Factor
gross_profit = trades_df['pnl_mes'][trades_df['pnl_mes'] > 0].sum()
gross_loss = -trades_df['pnl_mes'][trades_df['pnl_mes'] < 0].sum()

profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

# Sharpe Ratio (Assuming risk-free rate = 0)
sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std()) * np.sqrt(252) if equity_df['returns'].std() != 0 else np.nan

# Sortino Ratio (Assuming risk-free rate = 0)
negative_returns = equity_df['returns'][equity_df['returns'] < 0]
sortino_ratio = (equity_df['returns'].mean() / negative_returns.std()) * np.sqrt(252) if negative_returns.std() != 0 else np.nan

# Calmar Ratio
calmar_ratio = (annualized_return / abs(max_drawdown)) if max_drawdown != 0 else np.inf

# Benchmark Return (Set to 0 or customize as needed)
benchmark_return = 0.0  # Placeholder

# Compile Results
results = {
    "Start Date": actual_start_date.strftime("%Y-%m-%d"),
    "End Date": actual_end_date.strftime("%Y-%m-%d"),
    "Exposure Time": f"{exposure_time_percentage:.2f}%",
    "Final Account Balance": f"${final_account_balance:,.2f}",
    "Equity Peak": f"${equity_peak:,.2f}",
    "Total Return": f"{total_return:.2f}%",
    "Annualized Return": f"{annualized_return:.2f}%",
    "Benchmark Return": f"{benchmark_return:.2f}%",
    "Volatility (Annual)": f"{volatility:.2f}%",
    "Total Trades": total_trades,
    "Winning Trades": num_winning_trades,
    "Losing Trades": num_losing_trades,
    "Win Rate": f"{win_rate:.2f}%",
    "Profit Factor": f"{profit_factor:.2f}",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "nan",
    "Sortino Ratio": f"{sortino_ratio:.2f}" if not np.isnan(sortino_ratio) else "nan",
    "Calmar Ratio": f"{calmar_ratio:.2f}" if not np.isinf(calmar_ratio) else "inf",
    "Max Drawdown": f"{max_drawdown:.2f}%",
    "Average Drawdown": f"{average_drawdown:.2f}%",
    "Max Drawdown Duration": f"{max_drawdown_duration_days:.2f} days",
    "Average Drawdown Duration": f"{average_drawdown_duration_days:.2f} days",
}

# Print Performance Summary
print("\nPerformance Summary:")
print("--------------------")
for key, value in results.items():
    print(f"{key:25}: {value:>15}")

# Optional: Plot Equity Curve and Drawdowns
plt.figure(figsize=(14, 7))

# Plot Equity Curve
plt.subplot(2, 1, 1)
plt.plot(equity_df.index, equity_df['equity'], label='Equity Curve')
plt.plot(equity_df.index, equity_df['equity_peak'], label='Equity Peak', linestyle='--')
plt.title('Equity Curve')
plt.xlabel('Date')
plt.ylabel('Equity ($)')
plt.legend()

# Plot Drawdowns
plt.subplot(2, 1, 2)
plt.fill_between(equity_df.index, equity_df['drawdown'] * 100, color='red', alpha=0.3)
plt.title('Drawdowns')
plt.xlabel('Date')
plt.ylabel('Drawdown (%)')

plt.tight_layout()
plt.show()