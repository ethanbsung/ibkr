import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

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

# Option 1: Custom Backtest Period (Replace These Dates)
custom_start_date = "2022-09-25"
custom_end_date = "2024-12-11"

# Option 2: Use Full Available Data (if custom dates are not set)
if custom_start_date and custom_end_date:
    start_time = pd.to_datetime(custom_start_date, utc=True)
    end_time = pd.to_datetime(custom_end_date, utc=True)
else:
    start_time = pd.to_datetime(df_30m.index.min(), utc=True)
    end_time = pd.to_datetime(df_30m.index.max(), utc=True)

# Ensure the 1-minute DataFrame index is in UTC
df_1m.index = pd.to_datetime(df_1m.index, utc=True)

# Slice the 1-minute DataFrame using the chosen backtest period
df_1m = df_1m.loc[start_time:end_time]
df_30m = df_30m.loc[start_time:end_time]

print(f"Backtesting from {start_time} to {end_time}")

# Calculate Bollinger Bands on 30m data
bollinger_period = 15
bollinger_stddev = 2

df_30m['ma'] = df_30m['close'].rolling(window=bollinger_period).mean()
df_30m['std'] = df_30m['close'].rolling(window=bollinger_period).std()
df_30m['upper_band'] = df_30m['ma'] + (bollinger_stddev * df_30m['std'])
df_30m['lower_band'] = df_30m['ma'] - (bollinger_stddev * df_30m['std'])

df_30m.dropna(inplace=True)

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

def evaluate_exit(position_type, entry_price, stop_loss, take_profit, df_high_freq, entry_time):
    """
    Determines whether the stop-loss or take-profit is hit using higher-frequency data.
    Returns exit_price, exit_time, and hit_take_profit flag.
    """
    df_period = df_high_freq.loc[entry_time:]
    
    # Iterate through each higher-frequency bar after entry_time
    for timestamp, row in df_period.iterrows():
        high = row['high']
        low = row['low']

        if position_type == 'long':
            if high >= take_profit and low <= stop_loss:
                # Determine which was hit first
                # Assuming the open of the bar is the first price, check sequence
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

    # If neither condition is met, wait for the next bar
    return None, None, None

# Backtesting loop
for i in range(len(df_30m)):
    current_bar = df_30m.iloc[i]
    current_time = df_30m.index[i]
    current_price = current_bar['close']

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
            take_profit_price = entry_price + 10  # Adjust as per your strategy
            entry_time = current_time
            #print(f"Entered LONG at {entry_price} on {entry_time} UTC")

        elif current_price > current_bar['upper_band']:
            # Enter Short
            position_size = 1
            entry_price = current_price
            position_type = 'short'
            stop_loss_price = entry_price + 5  # Adjust as per your strategy
            take_profit_price = entry_price - 10  # Adjust as per your strategy
            entry_time = current_time
            #print(f"Entered SHORT at {entry_price} on {entry_time} UTC")

    else:
        # Position is open, check high-frequency data until exit
        exit_price, exit_time, hit_take_profit = evaluate_exit(
            position_type,
            entry_price,
            stop_loss_price,
            take_profit_price,
            df_high_freq,
            entry_time
        )

        if exit_price is not None and exit_time is not None:
            # Calculate P&L based on the exit condition
            if position_type == 'long':
                pnl = ((exit_price - entry_price) * 5) - (0.47 * 2)  # Example: 5 contracts, $0.47 spread cost
            elif position_type == 'short':
                pnl = ((entry_price - exit_price) * 5) - (0.47 * 2)
            
            trade_results.append(pnl)
            cash += pnl
            balance_series.append(cash)  # Append to list

            # Print trade exit details
            if hit_take_profit:
                exit_type = "TAKE PROFIT"
            else:
                exit_type = "STOP LOSS"

            #print(f"Exited {position_type.upper()} at {exit_price} on {exit_time} UTC via {exit_type} for P&L: ${pnl:.2f}")

            # Reset position variables
            position_size = 0
            position_type = None
            entry_price = None
            stop_loss_price = None
            take_profit_price = None
            entry_time = None

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

# Fix the FutureWarning by specifying fill_method=None
daily_returns = balance_series.resample('D').last().pct_change(fill_method=None).dropna()

# Define Sortino Ratio Calculation Function
def calculate_sortino_ratio(daily_returns, target_return=0):
    """
    Calculate the annualized Sortino Ratio.
    """
    if daily_returns.empty:
        return np.nan
    
    # Calculate excess returns
    excess_returns = daily_returns - target_return
    
    # Calculate downside returns (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]

    # Handle cases where there are no downside returns
    if downside_returns.empty or downside_returns.std() == 0:
        return np.inf  # No downside risk means infinite Sortino Ratio

    # Annualize downside standard deviation
    downside_std = downside_returns.std() * np.sqrt(252)

    # Annualize mean excess return
    annualized_mean_excess_return = daily_returns.mean() * 252

    # Return Sortino Ratio
    return annualized_mean_excess_return / downside_std

# Performance Metrics
total_return_percentage = ((cash - 5000) / 5000) * 100
trading_days = max((df_30m.index.max() - df_30m.index.min()).days, 1)
annualized_return_percentage = ((cash / 5000) ** (252 / trading_days)) - 1
benchmark_return = ((df_30m['close'].iloc[-1] - df_30m['close'].iloc[0]) / df_30m['close'].iloc[0]) * 100
equity_peak = balance_series.max()

volatility_annual = daily_returns.std() * np.sqrt(252) * 100
sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
sortino_ratio = calculate_sortino_ratio(daily_returns)

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

# Calculate daily PnL
daily_pnl = balance_series.resample('D').last().diff().dropna()




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

'''
# Plot Equity Curve
plt.figure(figsize=(12, 6))
plt.plot(balance_series, label='Equity Curve', color='b')
plt.title("Equity Curve")
plt.xlabel("Date")
plt.ylabel("Account Balance ($)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left')
plt.show()
'''