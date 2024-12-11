import pandas as pd
import numpy as np

# Load CSV data
csv_file = 'es_30m_data.csv'  # Replace with your actual file path

# Read the CSV without 'date_parser' and handle date parsing separately
try:
    df = pd.read_csv(
        csv_file,
        dtype={
            'open': float,
            'high': float,
            'low': float,
            'close': float,
            'volume': float,
            'average': float,
            'barCount': int,
            'contract': str
        }
    )
except FileNotFoundError:
    print(f"File not found: {csv_file}")
    exit(1)
except pd.errors.EmptyDataError:
    print("No data: The CSV file is empty.")
    exit(1)
except Exception as e:
    print(f"An error occurred while reading the CSV: {e}")
    exit(1)

# Parse 'date' column with timezone information
try:
    df['date'] = pd.to_datetime(df['date'], utc=True)  # Automatically handles timezone
except ValueError as ve:
    print(f"Date parsing error: {ve}")
    exit(1)

# Ensure DataFrame is sorted by date
df.sort_values('date', inplace=True)

# Set 'date' as the index
df.set_index('date', inplace=True)

# Display the first and last few rows to verify
print("DataFrame Head:")
print(df.head())

print("\nDataFrame Tail:")
print(df.tail())

# Strategy Parameters
bollinger_period = 15
bollinger_stddev = 2
stop_loss_points = 5
take_profit_points = 15
commission_per_side = 0.47
total_commission = commission_per_side * 2
initial_cash = 5000

# Calculate Bollinger Bands
df['ma'] = df['close'].rolling(window=bollinger_period).mean()
df['std'] = df['close'].rolling(window=bollinger_period).std()
df['upper_band'] = df['ma'] + (bollinger_stddev * df['std'])
df['lower_band'] = df['ma'] - (bollinger_stddev * df['std'])

# Drop initial rows with NaN values due to rolling calculations
df.dropna(inplace=True)

# Initialize variables
position_size = 0
entry_price = None
position_type = None  
cash = initial_cash
trade_results = []
balance_series = [initial_cash]
exposure_bars = 0

# For Drawdown Duration Calculations
in_drawdown = False
drawdown_start = None
drawdown_durations = []

# Backtesting loop
for i in range(len(df)):
    current_price = df['close'].iloc[i]
    high_price = df['high'].iloc[i]
    low_price = df['low'].iloc[i]

    # Count exposure when position is active
    if position_size != 0:
        exposure_bars += 1

    if position_size == 0:
        # No open position, check for entry signals
        if current_price < df['lower_band'].iloc[i]:
            # Enter Long
            position_size = 1
            entry_price = current_price
            position_type = 'long'
            stop_loss_price = entry_price - stop_loss_points
            take_profit_price = entry_price + take_profit_points
            # print(f"Entered Long Position at {entry_price:.2f}")

        elif current_price > df['upper_band'].iloc[i]:
            # Enter Short
            position_size = 1
            entry_price = current_price
            position_type = 'short'
            stop_loss_price = entry_price + stop_loss_points
            take_profit_price = entry_price - take_profit_points
            # print(f"Entered Short Position at {entry_price:.2f}")

    else:
        # Position is open, check if the limit orders are triggered
        if position_type == 'long':
            # For a long position, check if stop or take profit triggered
            if low_price <= stop_loss_price:
                # Stopped out at stop_loss_price
                pnl = ((stop_loss_price - entry_price) * 5) - total_commission  # ES multiplier is 50
                cash += pnl
                trade_results.append(pnl)
                balance_series.append(cash)
                # print(f"STOPPED OUT LONG at {stop_loss_price:.2f} | Loss: {pnl:.2f}")
                # Reset position
                position_size = 0
                position_type = None
                entry_price = None
                stop_loss_price = None
                take_profit_price = None

            elif high_price >= take_profit_price:
                # Took profit at take_profit_price
                pnl = ((take_profit_price - entry_price) * 5) - total_commission
                cash += pnl
                trade_results.append(pnl)
                balance_series.append(cash)
                # print(f"EXITED LONG at {take_profit_price:.2f} | Profit: {pnl:.2f}")
                # Reset position
                position_size = 0
                position_type = None
                entry_price = None
                stop_loss_price = None
                take_profit_price = None

        elif position_type == 'short':
            # For a short position, check if stop or take profit triggered
            if high_price >= stop_loss_price:
                # Stopped out short at stop_loss_price
                pnl = ((entry_price - stop_loss_price) * 5) - total_commission
                cash += pnl
                trade_results.append(pnl)
                balance_series.append(cash)
                # print(f"STOPPED OUT SHORT at {stop_loss_price:.2f} | Loss: {pnl:.2f}")
                # Reset position
                position_size = 0
                position_type = None
                entry_price = None
                stop_loss_price = None
                take_profit_price = None

            elif low_price <= take_profit_price:
                # Took profit short at take_profit_price
                pnl = ((entry_price - take_profit_price) * 5) - total_commission
                cash += pnl
                trade_results.append(pnl)
                balance_series.append(cash)
                # print(f"EXITED SHORT at {take_profit_price:.2f} | Profit: {pnl:.2f}")
                # Reset position
                position_size = 0
                position_type = None
                entry_price = None
                stop_loss_price = None
                take_profit_price = None

    # Drawdown Duration Tracking
    current_balance = cash
    running_max = max(balance_series)  # Current running maximum

    if current_balance < running_max:
        if not in_drawdown:
            in_drawdown = True
            drawdown_start = df.index[i]
    else:
        if in_drawdown:
            in_drawdown = False
            drawdown_end = df.index[i]
            duration = (drawdown_end - drawdown_start).days + (drawdown_end - drawdown_start).seconds / 86400
            drawdown_durations.append(duration)

# Handle if still in drawdown at the end of the data
if in_drawdown:
    drawdown_end = df.index[-1]
    duration = (drawdown_end - drawdown_start).days + (drawdown_end - drawdown_start).seconds / 86400
    drawdown_durations.append(duration)

# Portfolio Summary Calculations
balance_series = pd.Series(balance_series, index=df.index[:len(balance_series)])
daily_returns = balance_series.pct_change().dropna()

# Performance Metrics
total_return_percentage = ((cash - initial_cash) / initial_cash) * 100
trading_days = max((df.index.max() - df.index.min()).days, 1)
annualized_return_percentage = ((cash / initial_cash) ** (252 / trading_days)) - 1
benchmark_return = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
equity_peak = balance_series.max()
volatility_annual = daily_returns.std() * np.sqrt(252) * 100
sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0

# Sortino Ratio Calculation
# Calculate downside returns
downside_returns = daily_returns.copy()
downside_returns[downside_returns > 0] = 0
downside_std = downside_returns.std() * np.sqrt(252)
sortino_ratio = (daily_returns.mean() / downside_std) if downside_std != 0 else 0

# Drawdown Calculations
running_max_series = balance_series.cummax()
drawdowns = (balance_series - running_max_series) / running_max_series
max_drawdown = drawdowns.min() * 100
average_drawdown = drawdowns[drawdowns < 0].mean() * 100

# Exposure Time
exposure_time_percentage = (exposure_bars / len(df)) * 100

# Profit Factor
winning_trades = [pnl for pnl in trade_results if pnl > 0]
losing_trades = [pnl for pnl in trade_results if pnl <= 0]
profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf')

# Calmar Ratio Calculation
calmar_ratio = (total_return_percentage / abs(max_drawdown)) if max_drawdown != 0 else float('inf')

# Drawdown Duration Calculations
if drawdown_durations:
    max_drawdown_duration_days = max(drawdown_durations)
    average_drawdown_duration_days = np.mean(drawdown_durations)
else:
    max_drawdown_duration_days = 0
    average_drawdown_duration_days = 0

# Results Summary
print("\nPerformance Summary:")
results = {
    "Start Date": df.index.min().strftime("%Y-%m-%d"),
    "End Date": df.index.max().strftime("%Y-%m-%d"),
    "Exposure Time": f"{exposure_time_percentage:.2f}%",
    "Final Account Balance": f"${cash:,.2f}",
    "Equity Peak": f"${equity_peak:,.2f}",
    "Total Return": f"{total_return_percentage:.2f}%",
    "Annualized Return": f"{annualized_return_percentage * 100:.2f}%",
    "Benchmark Return": f"{benchmark_return:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Total Trades": len(trade_results),
    "Winning Trades": len(winning_trades),
    "Losing Trades": len(losing_trades),
    "Win Rate": f"{(len(winning_trades)/len(trade_results)*100) if trade_results else 0:.2f}%",
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
    print(f"{key:25}: {value:>15}")