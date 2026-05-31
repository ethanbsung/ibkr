import pandas as pd
import numpy as np
from itertools import product

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

csv_file_1m = 'es_1m_data.csv'
csv_file_5m = 'es_5m_data.csv'
csv_file_30m = 'es_30m_data.csv'

df_1m = load_data(csv_file_1m)
df_5m = load_data(csv_file_5m)
df_30m = load_data(csv_file_30m)

# Backtest period
custom_start_date = "2022-10-01"
custom_end_date = "2023-10-01"

if custom_start_date and custom_end_date:
    start_time = pd.to_datetime(custom_start_date, utc=True)
    end_time = pd.to_datetime(custom_end_date, utc=True)
else:
    start_time = pd.to_datetime(df_30m.index.min(), utc=True)
    end_time = pd.to_datetime(df_30m.index.max(), utc=True)

df_1m.index = pd.to_datetime(df_1m.index, utc=True)
df_1m = df_1m.loc[start_time:end_time]
df_30m = df_30m.loc[start_time:end_time]

print(f"Backtesting from {start_time} to {end_time}")

def run_backtest(df_1m, df_30m, lookback, stop_loss_points, take_profit_points):
    # Copy data to avoid mutation
    df_30m = df_30m.copy()
    df_1m = df_1m.copy()
    
    # Calculate Bollinger Bands
    bollinger_stddev = 2
    df_30m['ma'] = df_30m['close'].rolling(window=lookback).mean()
    df_30m['std'] = df_30m['close'].rolling(window=lookback).std()
    df_30m['upper_band'] = df_30m['ma'] + (bollinger_stddev * df_30m['std'])
    df_30m['lower_band'] = df_30m['ma'] - (bollinger_stddev * df_30m['std'])
    df_30m.dropna(inplace=True)

    # Initialize backtest variables
    position_size = 0
    entry_price = None
    position_type = None  
    cash = 5000
    trade_results = []
    balance_series = [5000]
    exposure_bars = 0

    # For Drawdown Duration Calculations
    in_drawdown = False
    drawdown_start = None
    drawdown_durations = []

    df_high_freq = df_1m  # Using 1-minute data here, adjust if needed

    def evaluate_exit(position_type, entry_price, stop_loss, take_profit, df_high_freq, entry_time):
        df_period = df_high_freq.loc[entry_time:]
        for timestamp, row in df_period.iterrows():
            high = row['high']
            low = row['low']

            if position_type == 'long':
                if high >= take_profit and low <= stop_loss:
                    # Check which got hit first
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
            # No open position, check for entry signals
            if current_price < current_bar['lower_band']:
                # Enter Long
                position_size = 1
                entry_price = current_price
                position_type = 'long'
                stop_loss_price = entry_price - stop_loss_points
                take_profit_price = entry_price + take_profit_points
                entry_time = current_time

            elif current_price > current_bar['upper_band']:
                # Enter Short
                position_size = 1
                entry_price = current_price
                position_type = 'short'
                stop_loss_price = entry_price + stop_loss_points
                take_profit_price = entry_price - take_profit_points
                entry_time = current_time

        else:
            # Position is open, check exit conditions in high-frequency data
            exit_price, exit_time, hit_take_profit = evaluate_exit(
                position_type,
                entry_price,
                stop_loss_price,
                take_profit_price,
                df_high_freq,
                entry_time
            )

            if exit_price is not None and exit_time is not None:
                # Calculate P&L
                if position_type == 'long':
                    pnl = ((exit_price - entry_price) * 5) - (0.47 * 2)
                else:
                    pnl = ((entry_price - exit_price) * 5) - (0.47 * 2)

                trade_results.append(pnl)
                cash += pnl
                balance_series.append(cash)

                # Reset position
                position_size = 0
                position_type = None
                entry_price = None
                stop_loss_price = None
                take_profit_price = None
                entry_time = None

    # After the Backtesting Loop
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
                duration = (drawdown_end - drawdown_start).total_seconds() / 86400
                drawdown_durations.append(duration)

    if in_drawdown:
        drawdown_end = balance_series.index[-1]
        duration = (drawdown_end - drawdown_start).total_seconds() / 86400
        drawdown_durations.append(duration)

    # Performance Calculations
    daily_returns = balance_series.resample('D').last().pct_change(fill_method=None).dropna()

    total_return_percentage = ((cash - 5000) / 5000) * 100
    trading_days = max((df_30m.index.max() - df_30m.index.min()).days, 1)
    sharpe_ratio = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() != 0 else 0
    winning_trades = [pnl for pnl in trade_results if pnl > 0]
    losing_trades = [pnl for pnl in trade_results if pnl <= 0]
    profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf')
    win_rate = (len(winning_trades)/len(trade_results)*100) if trade_results else 0

    # Return all metrics as a dict
    return {
        'lookback_period': lookback,
        'stop_loss_points': stop_loss_points,
        'take_profit_points': take_profit_points,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'final_balance': cash,
        'return_percentage': total_return_percentage,
        'total_trades': len(trade_results),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': win_rate
    }

# Define parameter ranges
lookback_periods = [10, 15, 20]
stop_losses = [5, 10, 15]
take_profits = [10, 15, 20]

# Store results
results_list = []

total_iterations = len(lookback_periods) * len(stop_losses) * len(take_profits)
current_iteration = 0

for lookback, sl, tp in product(lookback_periods, stop_losses, take_profits):
    current_iteration += 1
    print(f"Running optimization {current_iteration}/{total_iterations} -> (Lookback: {lookback}, SL: {sl}, TP: {tp})")
    metrics = run_backtest(df_1m, df_30m, lookback, sl, tp)
    results_list.append(metrics)

# Convert results to a DataFrame
results_df = pd.DataFrame(results_list)

# Find best by chosen metric, for example Sharpe Ratio
best_params = results_df.loc[results_df['sharpe_ratio'].idxmax()]

print("\nBest Parameters Found:")
for key in best_params.index:
    print(f"{key}: {best_params[key]}")

# If you prefer another metric, you can change the idxmax() column.
# For example, if you prefer profit factor:
# best_params = results_df.loc[results_df['profit_factor'].idxmax()]