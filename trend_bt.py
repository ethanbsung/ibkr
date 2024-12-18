import pandas as pd
import numpy as np

# Load historical data from CSV
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

# Load CSV data
csv_file = 'es_4h_data.csv'  # Replace with your CSV file path
df = load_data(csv_file)

# Define moving average parameters
fast_period = 20
slow_period = 50
take_profit_points = 20
stop_loss_points = 15
commission_per_side = 0.62
total_commission = commission_per_side * 2

# Calculate moving averages
df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
df['slow_ma'] = df['close'].rolling(window=slow_period).mean()

# Initialize variables
position_size = 0
entry_price = None
position_type = None
initial_cash = 5000
cash = initial_cash
trade_results = []
exposure_bars = 0
drawdown_durations = []
in_drawdown = False
drawdown_start = None

# Backtesting loop
for i in range(max(fast_period, slow_period), len(df)):
    current_price = df['close'].iloc[i]
    high_price = df['high'].iloc[i]
    low_price = df['low'].iloc[i]

    if position_size == 0:
        if df['fast_ma'].iloc[i] > df['slow_ma'].iloc[i] and df['fast_ma'].iloc[i - 1] <= df['slow_ma'].iloc[i - 1]:
            position_size = 1
            entry_price = current_price
            position_type = 'long'

        elif df['fast_ma'].iloc[i] < df['slow_ma'].iloc[i] and df['fast_ma'].iloc[i - 1] >= df['slow_ma'].iloc[i - 1]:
            position_size = 1
            entry_price = current_price
            position_type = 'short'

    elif position_type == 'long':
        if (high_price - entry_price) >= take_profit_points:
            exit_price = entry_price + take_profit_points
            pnl = (exit_price - entry_price) * position_size * 5 - total_commission
            cash += pnl
            trade_results.append(pnl)
            position_size = 0
        elif (entry_price - low_price) >= stop_loss_points:
            exit_price = entry_price - stop_loss_points
            pnl = (exit_price - entry_price) * position_size * 5 - total_commission
            cash += pnl
            trade_results.append(pnl)
            position_size = 0

    elif position_type == 'short':
        if (entry_price - low_price) >= take_profit_points:
            exit_price = entry_price - take_profit_points
            pnl = (entry_price - exit_price) * position_size * 5 - total_commission
            cash += pnl
            trade_results.append(pnl)
            position_size = 0
        elif (high_price - entry_price) >= stop_loss_points:
            exit_price = entry_price + stop_loss_points
            pnl = (entry_price - exit_price) * position_size * 5 - total_commission
            cash += pnl
            trade_results.append(pnl)
            position_size = 0

# Calculate balance and performance metrics
balance_series = pd.Series(initial_cash, index=df.index, dtype='float64')
trade_indices = df.index[max(fast_period, slow_period):len(trade_results) + max(fast_period, slow_period)]
balance_series[trade_indices] = initial_cash + np.cumsum(trade_results).astype('float64')

df['balance'] = balance_series.ffill()
df['drawdown'] = df['balance'] / df['balance'].cummax() - 1
max_drawdown = df['drawdown'].min() * 100

# Calculate returns
daily_returns = balance_series.resample('D').last().pct_change(fill_method=None).dropna()
returns = df['balance'].pct_change().dropna()

# Calculate Sharpe and Sortino ratios
sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else float('nan')

def calculate_sortino_ratio(daily_returns, target_return=0):
    excess_returns = daily_returns - target_return
    downside_returns = excess_returns[excess_returns < 0]
    if downside_returns.empty or downside_returns.std() == 0:
        return np.inf
    downside_std = downside_returns.std() * np.sqrt(252)
    annualized_mean_excess_return = daily_returns.mean() * 252
    return annualized_mean_excess_return / downside_std

sortino_ratio = calculate_sortino_ratio(daily_returns)

# Final Metrics
total_return_percentage = ((cash - initial_cash) / initial_cash) * 100
trading_days = max((df.index.max() - df.index.min()).days, 1)
annualized_return_percentage = ((cash / initial_cash) ** (252 / trading_days)) - 1
benchmark_return = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
equity_peak = balance_series.max()
volatility_annual = daily_returns.std() * np.sqrt(252) * 100

winning_trades = [pnl for pnl in trade_results if pnl > 0]
losing_trades = [pnl for pnl in trade_results if pnl <= 0]
profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf')

calmar_ratio = (total_return_percentage / abs(max_drawdown)) if max_drawdown != 0 else float('inf')

# Print Results
results = {
    "Start Date": df.index.min().strftime("%Y-%m-%d"),
    "End Date": df.index.max().strftime("%Y-%m-%d"),
    "Exposure Time": f"{(exposure_bars / len(df)) * 100:.2f}%",
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
}

print("\nPerformance Summary:")
for key, value in results.items():
    print(f"{key:25}: {value:>15}")