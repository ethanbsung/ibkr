import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import pytz
from datetime import time

# ----------------------------
# Helper Function for RTH
# ----------------------------
EASTERN = pytz.timezone('US/Eastern')
RTH_START = time(9, 30)
RTH_END = time(16, 0)

def is_rth(timestamp):
    """
    Check if the given timestamp (UTC) is within RTH (09:30-16:00 ET Monday-Friday).
    """
    if timestamp is None or pd.isna(timestamp):
        return False
    ts_eastern = timestamp.tz_convert(EASTERN)
    # Check weekday (Mon-Fri) and time range
    return (ts_eastern.weekday() < 5) and (RTH_START <= ts_eastern.time() < RTH_END)

# ----------------------------
# Load and Prepare the Data
# ----------------------------
df = pd.read_csv("es_1m_data.csv", parse_dates=['date'], index_col='date')

required_columns = {'open', 'high', 'low', 'close', 'volume'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Data must contain {required_columns} columns.")

df.sort_index(inplace=True)

# Ensure the index is timezone-aware (assuming UTC)
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')

# ----------------------------
# Select Backtest Date Range
# ----------------------------
from_date = '2023-10-11'
to_date = '2024-12-11'
df = df.loc[from_date:to_date]

# ----------------------------
# Calculate VWAP
# ----------------------------
df['date_only'] = df.index.date
df['cum_volume'] = df.groupby('date_only')['volume'].cumsum()
df['cum_pv'] = df.groupby('date_only').apply(lambda x: (x['close'] * x['volume']).cumsum()).reset_index(level=0, drop=True)
df['VWAP'] = df['cum_pv'] / df['cum_volume']
df.dropna(inplace=True)
df.drop(['cum_volume', 'cum_pv', 'date_only'], axis=1, inplace=True)

# ----------------------------
# Initialize Backtest Parameters
# ----------------------------
initial_balance = 5000
cash = initial_balance
multiplier = 5
position_size = 1
commission = 1.24
slippage = 0.25

# Strategy thresholds
buy_threshold = 20       # Buy if price is 10 points below VWAP
take_profit = 10         # Take Profit when 15 points above entry
stop_loss = 5            # Stop Loss 3 points below entry

# Removed 'vwap_sell_threshold' as per requirements

# Initialize tracking variables
position = 0
entry_price = 0.0
tp_price = 0.0
sl_price = 0.0
trades = []
equity = []
dates = []

total_bars = len(df)
bars_in_position = 0

# Initialize drawdown tracking variables
equity_peak = initial_balance
drawdowns = []
drawdown_durations = []          # Ensure this list is initialized
current_drawdown_duration = 0

bars_per_trading_day = 276  # Approx bars per trading day (5-min bars including extended hours)

# ----------------------------
# Backtest Loop
# ----------------------------
for i in range(len(df)):
    current_time = df.index[i]
    current_price = df['close'].iloc[i]
    current_vwap = df['VWAP'].iloc[i]
    
    # Update exposure time
    if position > 0:
        bars_in_position += 1
    
    # Strategy Logic: Long-Only
    if position == 0:
        # Check if in RTH before placing a new trade
        if is_rth(current_time):
            # Long Entry Condition: Low of the bar dips below (VWAP - buy_threshold)
            if df['low'].iloc[i] < (current_vwap - buy_threshold):
                entry_price = current_price + slippage
                tp_price = entry_price + take_profit
                sl_price = entry_price - stop_loss
                position = position_size
                cash -= entry_price * multiplier
                trades.append({
                    'Action': 'BUY',
                    'Date': current_time,
                    'Price': entry_price,
                    'Target Sell Price': tp_price,
                    'Stop Loss Price': sl_price,
                    'PnL': 0.0
                })
                print(f"BUY at {current_time} | Price: {entry_price:.2f} | TP: {tp_price:.2f} | SL: {sl_price:.2f}")
    else:
        # Position is Long
        sell_triggered = False
        pnl = 0.0
        actual_exit_price = 0.0
        sell_reason = ""
        
        # Exit conditions (can trigger any time)
        # 1. Take Profit
        if current_price >= tp_price:
            actual_exit_price = tp_price - slippage
            cash += actual_exit_price * multiplier
            pnl = (actual_exit_price - entry_price) * multiplier * position_size - commission
            sell_triggered = True
            sell_reason = 'TP'
        
        # 2. Stop Loss
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

# Handle last drawdown
if current_drawdown_duration > 0:
    drawdown_durations.append(current_drawdown_duration)
    print(f"Final drawdown of {current_drawdown_duration} bars ended on {current_time}")

# ----------------------------
# Calculate Drawdown Metrics
# -------------------------- --
if len(drawdowns) > 0:
    max_drawdown = np.max(drawdowns)  # Maximum drawdown percentage
    average_drawdown = np.mean(drawdowns)  # Average drawdown percentage
else:
    max_drawdown = 0.0
    average_drawdown = 0.0

if drawdown_durations:
    average_drawdown_duration_days = np.mean(drawdown_durations) / bars_per_trading_day
    max_drawdown_duration_days = max(drawdown_durations) / bars_per_trading_day
else:
    average_drawdown_duration_days = 0.0
    max_drawdown_duration_days = 0.0

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

# Calculate Benchmark Return
benchmark_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100

# Calculate Returns for Volatility and Sharpe Ratio
equity_series = pd.Series(equity, index=dates)
returns = equity_series.pct_change().dropna()

# Annualized Volatility
volatility_annual = returns.std() * np.sqrt(252 * bars_per_trading_day)

# Extract trades and PnL
trade_results = []
winning_trades = []
losing_trades = []
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
    profit_factor = np.nan

win_rate = (len(winning_trades) / len(trade_results) * 100) if trade_results else 0.0

# Sharpe and Sortino Ratios
sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * bars_per_trading_day) if returns.std() != 0 else np.nan
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
# Plot Equity Curve
# ----------------------------

plt.figure(figsize=(14, 7))
plt.plot(equity_series, label='Equity Curve', color='blue')
plt.title("Equity Curve - Long-Only VWAP Mean Reversion with SL and TP (MES Futures)")
plt.xlabel("Date")
plt.ylabel("Equity ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
