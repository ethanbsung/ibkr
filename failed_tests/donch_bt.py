import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------
DONCHIAN_PERIOD = 20  # Lookback period for the Donchian Channels
INITIAL_CAPITAL = 5000  # Starting capital in USD
CONTRACT_SIZE = 5  # ES futures contract size
STOP_MULTIPLIER = 1  # Multiplier for ATR-based stop-loss
TAKE_PROFIT_MULTIPLIER = 3  # Multiplier for ATR-based take-profit

# Transaction Costs
# SLIPPAGE = 0.1  # Removed as per user request
COMMISSION = 1.24  # USD per trade

# Custom Backtest Period (Optional)
START_DATE = '2000-01-01'  # Set to None to include all data
END_DATE = '2020-01-01'    # Set to None to include all data

# ----------------------------
# Load Data
# ----------------------------
# Adjust the file path as needed
data = pd.read_csv('Data/mes_daily_data.csv', parse_dates=['Time'])

# Set 'Time' as the index
data.set_index('Time', inplace=True)

# Ensure the data is sorted by time
data.sort_index(inplace=True)

# Select only the necessary columns
data = data[['Open', 'High', 'Low', 'Last']].copy()

# Rename columns to lowercase to maintain consistency
data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Last': 'close'}, inplace=True)

# Ensure columns are numeric
data['close'] = pd.to_numeric(data['close'], errors='coerce')
data['high'] = pd.to_numeric(data['high'], errors='coerce')
data['low'] = pd.to_numeric(data['low'], errors='coerce')

# Drop rows with NaN in critical columns
data.dropna(subset=['close', 'high', 'low'], inplace=True)

# Apply Custom Start and End Dates
if START_DATE:
    data = data[data.index >= pd.to_datetime(START_DATE)]
if END_DATE:
    data = data[data.index <= pd.to_datetime(END_DATE)]

# ----------------------------
# Calculate Indicators
# ----------------------------

# Calculate Donchian Channels based on previous DONCHIAN_PERIOD periods (excluding current)
data['Upper'] = data['high'].shift(1).rolling(window=DONCHIAN_PERIOD).max()
data['Lower'] = data['low'].shift(1).rolling(window=DONCHIAN_PERIOD).min()
data['Middle'] = (data['Upper'] + data['Lower']) / 2

# Calculate ATR for stop-loss
data['H-L'] = data['high'] - data['low']
data['H-PC'] = abs(data['high'] - data['close'].shift(1))
data['L-PC'] = abs(data['low'] - data['close'].shift(1))
data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
data['ATR'] = data['TR'].rolling(window=DONCHIAN_PERIOD).mean()
# Drop intermediate ATR calculation columns
data.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, inplace=True)

# ----------------------------
# Generate Signals
# ----------------------------
data['Long'] = (data['close'] > data['Upper']).astype(int)
data['Short'] = (data['close'] < data['Lower']).astype(int)

# Debugging: Check number of signals
print("Number of Long signals:", data['Long'].sum())
print("Number of Short signals:", data['Short'].sum())

# Debugging: Sample data with channels and signals
print("\nSample Data with Donchian Channels and Signals:")
print(data[['close', 'Upper', 'Lower', 'Long', 'Short']].dropna().head(25))

# ----------------------------
# Define Positions
# ----------------------------
data['Position'] = 0
data.loc[data['Long'] == 1, 'Position'] = 1   # Long
data.loc[data['Short'] == 1, 'Position'] = -1 # Short
data['Position'] = data['Position'].ffill().fillna(0)  # Carry forward positions

# ----------------------------
# Strategy Entry and Exit Signals
# ----------------------------
data['Signal'] = data['Position'].diff()

# ----------------------------
# Initialize Variables for Trade Tracking
# ----------------------------
trades = []
current_trade = None

# ----------------------------
# Iterate Over the DataFrame to Track Trades
# ----------------------------
for idx, row in data.iterrows():
    # Check for active trade and manage stop-loss and take-profit
    if current_trade is not None and not np.isnan(row['ATR']):
        if current_trade['Position'] == 'Long':
            # Calculate stop-loss and take-profit prices
            stop_price = current_trade['Entry Price'] - STOP_MULTIPLIER * row['ATR']
            take_profit_price = current_trade['Entry Price'] + TAKE_PROFIT_MULTIPLIER * row['ATR']
            
            # Check if stop-loss or take-profit is hit
            if row['low'] <= stop_price:
                exit_price = stop_price
                exit_reason = 'Stop Loss'
            elif row['high'] >= take_profit_price:
                exit_price = take_profit_price
                exit_reason = 'Take Profit'
            else:
                exit_price = None
                exit_reason = None
            
            if exit_price is not None:
                # Exit the trade
                current_trade['Exit Time'] = idx
                current_trade['Exit Price'] = exit_price
                if exit_reason == 'Stop Loss':
                    current_trade['PnL'] = (exit_price - current_trade['Entry Price']) * CONTRACT_SIZE - COMMISSION
                else:  # Take Profit
                    current_trade['PnL'] = (exit_price - current_trade['Entry Price']) * CONTRACT_SIZE - COMMISSION
                current_trade['Duration'] = (current_trade['Exit Time'] - current_trade['Entry Time']).total_seconds() / 3600
                current_trade['Exit Reason'] = exit_reason
                trades.append(current_trade)
                current_trade = None
                
        elif current_trade['Position'] == 'Short':
            # Calculate stop-loss and take-profit prices
            stop_price = current_trade['Entry Price'] + STOP_MULTIPLIER * row['ATR']
            take_profit_price = current_trade['Entry Price'] - TAKE_PROFIT_MULTIPLIER * row['ATR']
            
            # Check if stop-loss or take-profit is hit
            if row['high'] >= stop_price:
                exit_price = stop_price
                exit_reason = 'Stop Loss'
            elif row['low'] <= take_profit_price:
                exit_price = take_profit_price
                exit_reason = 'Take Profit'
            else:
                exit_price = None
                exit_reason = None
            
            if exit_price is not None:
                # Exit the trade
                current_trade['Exit Time'] = idx
                current_trade['Exit Price'] = exit_price
                if exit_reason == 'Stop Loss':
                    current_trade['PnL'] = (current_trade['Entry Price'] - exit_price) * CONTRACT_SIZE - COMMISSION
                else:  # Take Profit
                    current_trade['PnL'] = (current_trade['Entry Price'] - exit_price) * CONTRACT_SIZE - COMMISSION
                current_trade['Duration'] = (current_trade['Exit Time'] - current_trade['Entry Time']).total_seconds() / 3600
                current_trade['Exit Reason'] = exit_reason
                trades.append(current_trade)
                current_trade = None

    # Existing signal handling
    if row['Signal'] == 1:  # Enter Long
        if current_trade is None:
            current_trade = {
                'Entry Time': idx,
                'Entry Price': row['close'],
                'Position': 'Long',
                'Exit Time': None,
                'Exit Price': None,
                'PnL': None,
                'Duration': None,
                'Exit Reason': None
            }
    elif row['Signal'] == -1:  # Enter Short
        if current_trade is None:
            current_trade = {
                'Entry Time': idx,
                'Entry Price': row['close'],
                'Position': 'Short',
                'Exit Time': None,
                'Exit Price': None,
                'PnL': None,
                'Duration': None,
                'Exit Reason': None
            }
    elif row['Position'] == 0 and current_trade is not None:
        # Exit the current trade at current close price
        current_trade['Exit Time'] = idx
        current_trade['Exit Price'] = row['close']
        if current_trade['Position'] == 'Long':
            current_trade['PnL'] = (row['close'] - current_trade['Entry Price']) * CONTRACT_SIZE - COMMISSION
        else:  # Short
            current_trade['PnL'] = (current_trade['Entry Price'] - row['close']) * CONTRACT_SIZE - COMMISSION
        current_trade['Duration'] = (current_trade['Exit Time'] - current_trade['Entry Time']).total_seconds() / 3600  # Duration in hours
        current_trade['Exit Reason'] = 'Exit Signal'
        trades.append(current_trade)
        current_trade = None

# If a trade is open at the end, close it
if current_trade is not None:
    current_trade['Exit Time'] = data.index[-1]
    current_trade['Exit Price'] = data['close'].iloc[-1]
    if current_trade['Position'] == 'Long':
        current_trade['PnL'] = (current_trade['Exit Price'] - current_trade['Entry Price']) * CONTRACT_SIZE - COMMISSION
    else:  # Short
        current_trade['PnL'] = (current_trade['Entry Price'] - current_trade['Exit Price']) * CONTRACT_SIZE - COMMISSION
    current_trade['Duration'] = (current_trade['Exit Time'] - current_trade['Entry Time']).total_seconds() / 3600  # Duration in hours
    current_trade['Exit Reason'] = 'End of Data'
    trades.append(current_trade)

# ----------------------------
# Convert Trades to DataFrame
# ----------------------------
trade_results = pd.DataFrame(trades)

# Debugging: Check the contents of trade_results
print("\nTrade Results Columns:", trade_results.columns.tolist())
print("Trade Results Head:")
print(trade_results.head())

# ----------------------------
# Calculate Strategy Returns
# ----------------------------
# Initialize Capital Series
data['Capital'] = INITIAL_CAPITAL
current_capital = INITIAL_CAPITAL

# Create a Series to hold capital over time
capital_series = pd.Series(index=data.index, dtype='float64')
capital_series.iloc[0] = INITIAL_CAPITAL

# Iterate over trades to update capital
for trade in trades:
    exit_time = trade['Exit Time']
    pnl = trade['PnL']
    current_capital += pnl
    capital_series.loc[exit_time] = current_capital

# Forward fill the capital_series to account for periods without trades
capital_series = capital_series.fillna(method='ffill')

# Handle NaNs in 'Daily Return' before pct_change
data['Capital'] = capital_series
data['Capital'].fillna(method='ffill', inplace=True)
data['Daily Return'] = data['Capital'].resample('D').last().pct_change(fill_method=None)

# Calculate Equity Peak
data['Equity Peak'] = data['Capital'].cummax()

# Drawdown
data['Drawdown'] = data['Equity Peak'] - data['Capital']
data['Drawdown %'] = (data['Drawdown'] / data['Equity Peak']) * 100

# Max Drawdown
max_drawdown = data['Drawdown %'].max()

# Average Drawdown
average_drawdown = data['Drawdown %'].mean()

# Max Drawdown Duration
drawdown = data['Drawdown %']
is_drawdown = drawdown > 0
drawdown_groups = (is_drawdown != is_drawdown.shift()).cumsum()
drawdown_durations = data.groupby(drawdown_groups)['Drawdown %'].apply(
    lambda x: (x.index[-1] - x.index[0]).total_seconds() / 86400 if x.all() else 0
)
max_drawdown_duration = drawdown_durations.max()  # In days

# Average Drawdown Duration
average_drawdown_duration = drawdown_durations[drawdown_durations > 0].mean()  # In days

# Total Return
total_return = (data['Capital'].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

# Annualized Return
num_years = (data.index[-1] - data.index[0]).days / 365.25
annualized_return = ((data['Capital'].iloc[-1] / INITIAL_CAPITAL) ** (1 / num_years) - 1) * 100

# Volatility (Annual)
volatility = data['Daily Return'].std() * np.sqrt(252)

# Sharpe Ratio
risk_free_rate = 0.0  # Assuming risk-free rate is 0
sharpe_ratio = (data['Daily Return'].mean() - risk_free_rate) / data['Daily Return'].std() * np.sqrt(252) if data['Daily Return'].std() != 0 else np.nan

# Sortino Ratio
negative_returns = data['Daily Return'][data['Daily Return'] < 0]
sortino_ratio = (data['Daily Return'].mean() - risk_free_rate) / negative_returns.std() * np.sqrt(252) if negative_returns.std() != 0 else np.nan

# Calmar Ratio
calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else np.nan

# Trade Statistics
total_trades = len(trade_results)
if total_trades > 0:
    winning_trades = trade_results[trade_results['PnL'] > 0]
    losing_trades = trade_results[trade_results['PnL'] <= 0]
    win_rate = (len(winning_trades) / total_trades) * 100
    profit_factor = (winning_trades['PnL'].sum() / abs(losing_trades['PnL'].sum())) if losing_trades['PnL'].sum() != 0 else np.nan
else:
    winning_trades = pd.DataFrame()
    losing_trades = pd.DataFrame()
    win_rate = 0
    profit_factor = np.nan

# Exposure Time (percentage of time in market)
exposure_time = (data['Position'].abs().sum() / len(data)) * 100

# Equity Peak
equity_peak = data['Equity Peak'].iloc[-1]

# Final Capital
final_capital = data['Capital'].iloc[-1]

# Benchmark Return (assuming buy and hold)
benchmark_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100

# ----------------------------
# Results Summary
# ----------------------------
print("\nPerformance Summary:")
results = {
    "Start Date": data.index.min().strftime("%Y-%m-%d"),
    "End Date": data.index.max().strftime("%Y-%m-%d"),
    "Exposure Time": f"{exposure_time:.2f}%",
    "Final Account Balance": f"${final_capital:,.2f}",
    "Equity Peak": f"${equity_peak:,.2f}",
    "Total Return": f"{total_return:.2f}%",
    "Annualized Return": f"{annualized_return:.2f}%",
    "Benchmark Return": f"{benchmark_return:.2f}%",
    "Volatility (Annual)": f"{volatility:.2f}%",
    "Total Trades": total_trades,
    "Winning Trades": len(winning_trades),
    "Losing Trades": len(losing_trades),
    "Win Rate": f"{win_rate:.2f}%",
    "Profit Factor": f"{profit_factor:.2f}",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}",
    "Sortino Ratio": f"{sortino_ratio:.2f}",
    "Calmar Ratio": f"{calmar_ratio:.2f}",
    "Max Drawdown": f"{max_drawdown:.2f}%",
    "Average Drawdown": f"{average_drawdown:.2f}%",
    "Max Drawdown Duration": f"{max_drawdown_duration:.2f} days",
    "Average Drawdown Duration": f"{average_drawdown_duration:.2f} days",
}

for key, value in results.items():
    print(f"{key:25}: {value:>15}")

# ----------------------------
# Analyze Exit Reasons
# ----------------------------
print("\nExit Reasons:")
print(trade_results['Exit Reason'].value_counts())

# ----------------------------
# Plot the Strategy
# ----------------------------
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['close'], label='ES Price', alpha=0.5)
plt.plot(data.index, data['Upper'], label='Upper Donchian', linestyle='--', color='green')
plt.plot(data.index, data['Lower'], label='Lower Donchian', linestyle='--', color='red')
plt.fill_between(data.index, data['Upper'], data['Lower'], color='gray', alpha=0.1)

# Highlight Long and Short signals
long_signals = data[data['Long'] == 1]
short_signals = data[data['Short'] == 1]

plt.scatter(long_signals.index, long_signals['close'], marker='^', color='g', label='Buy Signal', alpha=1)
plt.scatter(short_signals.index, short_signals['close'], marker='v', color='r', label='Sell Signal', alpha=1)

plt.legend()
plt.title('Donchian Channel Strategy with Trade Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid()
plt.show()