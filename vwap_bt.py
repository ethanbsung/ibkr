import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# ----------------------------
# Load and Prepare the Data
# ----------------------------

# Load MES 5-minute data (using ES futures data)
df = pd.read_csv("es_5m_data.csv", parse_dates=['date'], index_col='date')

# Ensure required columns exist
required_columns = {'open', 'high', 'low', 'close', 'volume'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Data must contain {required_columns} columns.")

# Sort the DataFrame by datetime index
df.sort_index(inplace=True)

# ----------------------------
# Select Backtest Date Range
# ----------------------------

# Define the backtest period
from_date = '2022-10-11'  # YYYY-MM-DD
to_date = '2024-12-11'    # YYYY-MM-DD

# Filter the DataFrame for the specified date range
df = df.loc[from_date:to_date]

# ----------------------------
# Calculate Indicators
# ----------------------------

# Correct VWAP Calculation: Reset VWAP at the start of each trading day

# Extract date component for grouping
df['date_only'] = df.index.date

# Calculate cumulative volume and cumulative (price * volume) per day
df['cum_volume'] = df.groupby('date_only')['volume'].cumsum()
df['cum_pv'] = df.groupby('date_only').apply(lambda x: (x['close'] * x['volume']).cumsum()).reset_index(level=0, drop=True)

# Calculate VWAP per day
df['VWAP'] = df['cum_pv'] / df['cum_volume']

# Drop rows with NaN values resulting from cumulative calculations
df.dropna(inplace=True)

# Drop helper columns to clean up the DataFrame
df.drop(['cum_volume', 'cum_pv', 'date_only'], axis=1, inplace=True)

# ----------------------------
# Initialize Backtest Parameters
# ----------------------------

initial_balance = 5000  # Starting capital
cash = initial_balance
multiplier = 5          # MES futures multiplier
position_size = 1       # Number of MES contracts per trade
commission = 1.24       # Round-trip commission per trade
slippage = 0.25         # Simulated slippage in points

# Buy and Sell Thresholds (in points)
buy_threshold = 10          # Buy when price is 15 points below VWAP
take_profit = 15            # Take Profit: 10 points above entry price
stop_loss = 3               # Stop Loss: 4 points below entry price
vwap_sell_threshold = 10    # Sell when price is 15 points above VWAP

# Tracking variables
position = 0                # Current position: 0 = flat, 1 = long
entry_price = 0.0
tp_price = 0.0
sl_price = 0.0
trades = []                 # List to store trade details
equity = []                 # Equity over time
dates = []                  # Corresponding dates for equity

# For exposure time calculation
total_bars = len(df)
bars_in_position = 0

# For drawdown calculation
equity_peak = initial_balance
drawdowns = []
current_drawdown = 0.0
max_drawdown = 0.0
drawdown_durations = []
current_drawdown_duration = 0
max_drawdown_duration = 0
average_drawdown = 0.0
average_drawdown_duration_days = 0.0

# Correct bars per trading day for 5-minute intervals during extended hours
bars_per_trading_day = 276  # 1380 minutes / 5-minute bars

# ----------------------------
# Backtest Loop with Enhanced Logging
# ----------------------------

for i in range(len(df)):
    current_time = df.index[i]
    current_price = df['close'].iloc[i]
    current_vwap = df['VWAP'].iloc[i]
    
    # Update exposure time
    if position > 0:
        bars_in_position += 1
    
    # Strategy Logic
    if position == 0:
        # Buy Signal: Price is a certain amount below VWAP
        if current_price < (current_vwap - buy_threshold):
            entry_price = current_price + slippage
            tp_price = entry_price + take_profit      # Target exit price above entry
            sl_price = entry_price - stop_loss        # Stop Loss below entry
            position = position_size
            cash -= entry_price * multiplier          # MES Futures multiplier is $5 per point
            trades.append({
                'Action': 'BUY',
                'Date': current_time,
                'Price': entry_price,
                'Target Sell Price': tp_price,
                'Stop Loss Price': sl_price,
                'PnL': 0.0
            })
            print(f"BUY at {current_time} | Price: {entry_price:.2f} | TP: {tp_price:.2f} | SL: {sl_price:.2f}")
    elif position > 0:
        # Check for Take Profit, Stop Loss, or VWAP Sell Threshold
        sell_triggered = False
        pnl = 0.0
        actual_exit_price = 0.0
        sell_reason = ""
        
        # 1. Take Profit (TP) Condition
        if current_price >= tp_price:
            actual_exit_price = tp_price - slippage
            cash += actual_exit_price * multiplier
            pnl = (actual_exit_price - entry_price) * multiplier * position_size - commission
            sell_triggered = True
            sell_reason = 'TP'
        
        # 2. VWAP Sell Threshold Condition
        elif current_price >= (current_vwap + vwap_sell_threshold):
            actual_exit_price = (current_vwap + vwap_sell_threshold) - slippage
            cash += actual_exit_price * multiplier
            pnl = (actual_exit_price - entry_price) * multiplier * position_size - commission
            sell_triggered = True
            sell_reason = 'VWAP Sell Threshold'
        
        # 3. Stop Loss (SL) Condition
        elif current_price <= sl_price:
            actual_exit_price = sl_price + slippage
            cash += actual_exit_price * multiplier
            pnl = (actual_exit_price - entry_price) * multiplier * position_size - commission
            sell_triggered = True
            sell_reason = 'SL'
        
        # Execute Sell if any condition is met
        if sell_triggered:
            trades.append({
                'Action': 'SELL',
                'Date': current_time,
                'Price': actual_exit_price,
                'PnL': pnl
            })
            position = 0
            print(f"SELL ({sell_reason}) at {current_time} | Price: {actual_exit_price:.2f} | PnL: {pnl:.2f}")
    
    # Calculate current equity
    current_equity = cash + (current_price * multiplier * position if position > 0 else 0)
    equity.append(current_equity)
    dates.append(current_time)
    
    # Update equity peak
    if current_equity > equity_peak:
        equity_peak = current_equity
    
    # Calculate drawdown
    current_drawdown = (equity_peak - current_equity) / equity_peak * 100
    drawdowns.append(current_drawdown)
    
    # Track drawdown durations
    if current_drawdown > 0:
        current_drawdown_duration += 1
    else:
        if current_drawdown_duration > 0:
            drawdown_durations.append(current_drawdown_duration)
            print(f"Drawdown of {current_drawdown_duration} bars ended on {current_time}")
            current_drawdown_duration = 0
    
    # Update max_drawdown
    if current_drawdown > max_drawdown:
        max_drawdown = current_drawdown

# Handle last drawdown duration
if current_drawdown_duration > 0:
    drawdown_durations.append(current_drawdown_duration)
    print(f"Final drawdown of {current_drawdown_duration} bars ended on {current_time}")

# ----------------------------
# Calculate Drawdown Durations in Days
# ----------------------------

# Calculate average and max drawdown durations in days
if drawdown_durations:
    average_drawdown_duration_days = np.mean(drawdown_durations) / bars_per_trading_day
    max_drawdown_duration_days = max(drawdown_durations) / bars_per_trading_day
else:
    average_drawdown_duration_days = 0.0
    max_drawdown_duration_days = 0.0

average_drawdown = np.mean(drawdowns)

# ----------------------------
# Final Portfolio Value
# ----------------------------

final_balance = cash + (df['close'].iloc[-1] * multiplier * position if position > 0 else 0)

# Calculate Total Return
total_return = (final_balance - initial_balance) / initial_balance * 100

# Calculate Annualized Return
start_date = df.index.min()
end_date = df.index.max()
duration = end_date - start_date
years = duration.days / 365.25
annualized_return = ((final_balance / initial_balance) ** (1 / years) - 1) * 100 if years > 0 else 0.0

# Calculate Benchmark Return (Buy and Hold)
benchmark_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100

# Calculate Returns for Volatility and Sharpe Ratio
equity_series = pd.Series(equity, index=dates)
returns = equity_series.pct_change().dropna()

# Annualized Volatility
# Number of 5-min bars in a trading year: 252 days * 276 bars/day = 69,552 bars
volatility_annual = returns.std() * np.sqrt(252 * bars_per_trading_day)

# Calculate Trade Metrics
trade_results = []
winning_trades = []
losing_trades = []
profit_factor = 0.0

# Ensure trades are in BUY-SELL pairs
for i in range(0, len(trades), 2):
    if i + 1 < len(trades):
        buy = trades[i]
        sell = trades[i + 1]
        pnl = sell['PnL']
        trade_results.append(pnl)
        if pnl > 0:
            winning_trades.append(pnl)
        else:
            losing_trades.append(pnl)

if losing_trades:
    profit_factor = abs(sum(winning_trades)) / abs(sum(losing_trades))
else:
    profit_factor = np.nan  # Avoid division by zero

# Win Rate
win_rate = (len(winning_trades) / len(trade_results) * 100) if trade_results else 0.0

# Sharpe Ratio (Assuming risk-free rate = 0)
sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * bars_per_trading_day) if returns.std() != 0 else np.nan

# Sortino Ratio (Assuming risk-free rate = 0)
downside_returns = returns[returns < 0]
sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252 * bars_per_trading_day) if downside_returns.std() != 0 else np.nan

# Calmar Ratio
calmar_ratio = (annualized_return / max_drawdown) if max_drawdown != 0 else np.nan

# Exposure Time
exposure_time_percentage = (bars_in_position / total_bars) * 100

# ----------------------------
# Display Trades
# ----------------------------

trades_df = pd.DataFrame(trades)
# Calculate PnL only for SELL actions
trades_df['PnL'] = trades_df.apply(lambda row: row['PnL'] if row['Action'] == 'SELL' else np.nan, axis=1)
print("\nTrades:")
print(trades_df.to_string(index=False))

# ----------------------------
# Performance Summary
# ----------------------------

print("\nPerformance Summary:")
results = {
    "Start Date": start_date.strftime("%Y-%m-%d"),
    "End Date": end_date.strftime("%Y-%m-%d"),
    "Exposure Time": f"{exposure_time_percentage:.2f}%",
    "Final Account Balance": f"${final_balance:,.2f}",
    "Equity Peak": f"${equity_peak:,.2f}",
    "Total Return": f"{total_return:.2f}%",
    "Annualized Return": f"{annualized_return:.2f}%",
    "Benchmark Return": f"{benchmark_return:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Total Trades": len(trade_results),
    "Winning Trades": len(winning_trades),
    "Losing Trades": len(losing_trades),
    "Win Rate": f"{win_rate:.2f}%",
    "Profit Factor": f"{profit_factor:.2f}",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}",
    "Sortino Ratio": f"{sortino_ratio:.2f}",
    "Calmar Ratio": f"{calmar_ratio:.2f}",
    "Max Drawdown": f"{max_drawdown:.2f}%",
    "Average Drawdown": f"{average_drawdown:.2f}%",
    "Max Drawdown Duration": f"{max_drawdown_duration_days:.2f} days",
    "Average Drawdown Duration": f"{average_drawdown_duration_days:.2f} days",
}

for key, value in results.items():
    print(f"{key}: {value}")

# ----------------------------
# Plot Equity Curve with VWAP and Thresholds
# ----------------------------

plt.figure(figsize=(14, 7))
plt.plot(equity_series, label='Equity Curve', color='blue')

plt.title("Equity Curve - VWAP Mean Reversion Strategy with SL, TP, and VWAP Sell Threshold (MES Futures)")
plt.xlabel("Date")
plt.ylabel("Equity ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()