import pandas as pd
import numpy as np

# Function to load data
def load_data(csv_file):
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
            },
            parse_dates=['date'],
            date_format="%Y-%m-%d %H:%M:%S%z"
        )
        df.sort_values('date', inplace=True)
        df.set_index('date', inplace=True)
        return df
    except FileNotFoundError:
        print(f"File not found: {csv_file}")
        exit(1)
    except pd.errors.EmptyDataError:
        print("No data: The CSV file is empty.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        exit(1)

# Load datasets
csv_file_1m = 'es_1m_data.csv'
csv_file_5m = 'es_5m_data.csv'
csv_file_30m = 'es_30m_data.csv'

df_1m = load_data(csv_file_1m)
df_5m = load_data(csv_file_5m)
df_30m = load_data(csv_file_30m)

# Check for duplicates in the index of each DataFrame and remove them
for df_name, df in [('df_1m', df_1m), ('df_5m', df_5m), ('df_30m', df_30m)]:
    if df.index.duplicated().any():
        print(f"Duplicate indices found in {df_name}. Removing duplicates.")
        # Option 1: Remove duplicates, keeping the first occurrence
        df = df[~df.index.duplicated(keep='first')]
        # Option 2: Aggregate duplicates (e.g., take mean)
        # df = df.groupby(df.index).mean()
        # Reassign the cleaned DataFrame back to its original variable
        globals()[df_name] = df
    else:
        print(f"No duplicate indices in {df_name}.")

# Adjust the date range dynamically and normalize to UTC
start_time = pd.to_datetime(df_30m.index.min(), utc=True)
end_time = pd.to_datetime(df_30m.index.max(), utc=True)

# Ensure the 1-minute DataFrame index is in UTC
df_1m.index = pd.to_datetime(df_1m.index, utc=True)

# Slice the 1-minute DataFrame using consistent UTC timezones
df_1m = df_1m.loc[start_time:end_time]

print(f"Start Time (UTC): {start_time}")
print(f"End Time (UTC): {end_time}")
print(f"1-Minute Data Range: {df_1m.index.min()} to {df_1m.index.max()}")

# Calculate Bollinger Bands on 30m data
bollinger_period = 15
bollinger_stddev = 2

df_30m['ma'] = df_30m['close'].rolling(window=bollinger_period).mean()
df_30m['std'] = df_30m['close'].rolling(window=bollinger_period).std()
df_30m['upper_band'] = df_30m['ma'] + (bollinger_stddev * df_30m['std'])
df_30m['lower_band'] = df_30m['ma'] - (bollinger_stddev * df_30m['std'])

df_30m.dropna(inplace=True)

# Ensure df_30m has no duplicate indices after cleaning
if df_30m.index.duplicated().any():
    print("Duplicate indices found in df_30m after initial cleaning. Removing duplicates.")
    df_30m = df_30m[~df_30m.index.duplicated(keep='first')]
else:
    print("No duplicate indices in df_30m after initial cleaning.")

# Initialize backtest variables
position_size = 0
entry_price = None
position_type = None  
cash = 5000
trade_results = []
balance_series = [5000]  # Keep as a list
exposure_bars = 0

# For Drawdown Duration Calculations
in_drawdown = False
drawdown_start = None
drawdown_durations = []

# Define which high-frequency data to use
df_high_freq = df_1m  # Change to df_5m if preferred

def evaluate_exit(entry_time, position_type, entry_price, stop_loss, take_profit, df_high_freq):
    """
    Determines whether the stop-loss or take-profit is hit first using higher-frequency data.
    """
    bar_end_time = entry_time + pd.Timedelta(minutes=30)
    df_period = df_high_freq.loc[entry_time:bar_end_time]

    # Handle case where bar_end_time might not exist in df_high_freq
    if bar_end_time not in df_period.index:
        print(f"Bar end time {bar_end_time} not found in high-frequency data.")
        # Exit at the last available close price within the period
        if not df_period.empty:
            exit_price = df_period['close'].iloc[-1]
            exit_time = df_period.index[-1]
        else:
            # If df_period is empty, exit at entry price
            exit_price = entry_price
            exit_time = entry_time
        hit_take_profit = None
        return exit_price, exit_time, hit_take_profit

    # Iterate through each higher-frequency bar within the 30m period
    for timestamp, row in df_period.iterrows():
        high = row['high']
        low = row['low']

        if position_type == 'long':
            if high >= take_profit and low <= stop_loss:
                # Determine which was hit first based on open price
                if row['open'] <= stop_loss:
                    return stop_loss, timestamp, False
                else:
                    return take_profit, timestamp, True
            elif high >= take_profit:
                return take_profit, timestamp, True
            elif low <= stop_loss:
                return stop_loss, timestamp, False

        elif position_type == 'short':
            if low <= take_profit and high >= stop_loss:
                if row['open'] >= stop_loss:
                    return stop_loss, timestamp, False
                else:
                    return take_profit, timestamp, True
            elif low <= take_profit:
                return take_profit, timestamp, True
            elif high >= stop_loss:
                return stop_loss, timestamp, False

    # If neither condition is met within the bar, exit at bar close
    exit_price_series = df_high_freq.loc[bar_end_time]['close']
    if isinstance(exit_price_series, pd.Series):
        # Handle multiple close prices by taking the first one
        exit_price = exit_price_series.iloc[0]
    else:
        exit_price = exit_price_series  # float

    exit_time = bar_end_time
    hit_take_profit = None
    return exit_price, exit_time, hit_take_profit

# Backtesting loop
for i in range(len(df_30m)):
    current_bar = df_30m.iloc[i]
    current_time = df_30m.index[i]
    current_price = current_bar['close']
    high_price = current_bar['high']
    low_price = current_bar['low']

    # Count exposure when position is active
    if position_size != 0:
        exposure_bars += 1

    if position_size == 0:
        # No open position, check for entry signals based on 30m bar
        if current_price < current_bar['lower_band']:
            # Enter Long
            position_size = 1
            entry_price = current_price
            position_type = 'long'
            stop_loss_price = entry_price - 5  # Adjust as per your strategy
            take_profit_price = entry_price + 15  # Adjust as per your strategy
            entry_time = current_time
            # print(f"Entered Long at {entry_price} on {entry_time}")

        elif current_price > current_bar['upper_band']:
            # Enter Short
            position_size = 1
            entry_price = current_price
            position_type = 'short'
            stop_loss_price = entry_price + 5  # Adjust as per your strategy
            take_profit_price = entry_price - 15  # Adjust as per your strategy
            entry_time = current_time
            # print(f"Entered Short at {entry_price} on {entry_time}")

    else:
        # Position is open, use higher-frequency data to evaluate exit
        exit_price, exit_time, hit_take_profit = evaluate_exit(
            entry_time,
            position_type,
            entry_price,
            stop_loss_price,
            take_profit_price,
            df_high_freq
        )

        if exit_price is not None and exit_time is not None:
            # Calculate P&L based on the exit condition
            if hit_take_profit is True:
                if position_type == 'long':
                    pnl = ((exit_price - entry_price) * 50) - (0.47 * 2)
                elif position_type == 'short':
                    pnl = ((entry_price - exit_price) * 50) - (0.47 * 2)
                trade_results.append(pnl)
                cash += pnl
                balance_series.append(cash)  # Append to list

            elif hit_take_profit is False:
                if position_type == 'long':
                    pnl = ((exit_price - entry_price) * 50) - (0.47 * 2)
                elif position_type == 'short':
                    pnl = ((entry_price - exit_price) * 50) - (0.47 * 2)
                trade_results.append(pnl)
                cash += pnl
                balance_series.append(cash)  # Append to list

            else:
                # Neither take profit nor stop loss was hit; exit at bar close
                if position_type == 'long':
                    pnl = ((exit_price - entry_price) * 50) - (0.47 * 2)
                elif position_type == 'short':
                    pnl = ((entry_price - exit_price) * 50) - (0.47 * 2)
                trade_results.append(pnl)
                cash += pnl
                balance_series.append(cash)  # Append to list

            # Reset position
            position_size = 0
            position_type = None
            entry_price = None
            stop_loss_price = None
            take_profit_price = None
            entry_time = None

    # **Do Not Convert `balance_series` to a Series Inside the Loop**
    # Remove or comment out the following line:
    # balance_series = pd.Series(balance_series, index=df_30m.index[:len(balance_series)])

    # Drawdown Duration Tracking (unchanged)
    # We'll handle this after converting to a Series

# After the Backtesting Loop

# Convert balance_series to a Pandas Series
balance_series = pd.Series(balance_series, index=df_30m.index[:len(balance_series)])

# Drawdown Duration Tracking
for i in range(len(balance_series)):
    current_balance = balance_series.iloc[i]
    running_max = balance_series.iloc[:i+1].max()

    if current_balance < running_max:
        if not in_drawdown:
            in_drawdown = True
            drawdown_start = balance_series.index[i]
    else:
        if in_drawdown:
            in_drawdown = False
            drawdown_end = balance_series.index[i]
            duration = (drawdown_end - drawdown_start).total_seconds() / 86400  # Duration in days
            drawdown_durations.append(duration)

# Handle if still in drawdown at the end of the data
if in_drawdown:
    drawdown_end = balance_series.index[-1]
    duration = (drawdown_end - drawdown_start).total_seconds() / 86400
    drawdown_durations.append(duration)

# Portfolio Summary Calculations
daily_returns = balance_series.resample('D').last().pct_change().dropna()

# Performance Metrics
total_return_percentage = ((cash - 5000) / 5000) * 100
trading_days = max((df_30m.index.max() - df_30m.index.min()).days, 1)
annualized_return_percentage = ((cash / 5000) ** (252 / trading_days)) - 1
benchmark_return = ((df_30m['close'].iloc[-1] - df_30m['close'].iloc[0]) / df_30m['close'].iloc[0]) * 100
equity_peak = balance_series.max()
volatility_annual = daily_returns.std() * np.sqrt(252) * 100
sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0

# Sortino Ratio Calculation
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
exposure_time_percentage = (exposure_bars / len(df_30m)) * 100

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
    "Start Date": df_30m.index.min().strftime("%Y-%m-%d"),
    "End Date": df_30m.index.max().strftime("%Y-%m-%d"),
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