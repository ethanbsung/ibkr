import pandas as pd
import numpy as np
from itertools import product

# Optimization function
def optimize_strategy(df, fast_range, slow_range, metric="sharpe_ratio"):
    best_result = {
        "fast_period": None,
        "slow_period": None,
        "best_metric": -np.inf,
        "results": None
    }

    # Grid search over all combinations
    for fast_period, slow_period in product(fast_range, slow_range):
        if fast_period >= slow_period:
            continue  # Ensure fast MA < slow MA

        # Calculate moving averages
        df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=slow_period).mean()

        # Run backtest
        cash, trade_results, balance_series = run_backtest(df, fast_period, slow_period)

        # Calculate metrics
        daily_returns = balance_series.resample('D').last().pct_change(fill_method=None).dropna()
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
        sortino_ratio = calculate_sortino_ratio(daily_returns)
        total_return = ((cash - initial_cash) / initial_cash) * 100

        # Choose metric for optimization
        metric_value = {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "total_return": total_return
        }.get(metric, sharpe_ratio)

        # Update best result if improved
        if metric_value > best_result["best_metric"]:
            best_result.update({
                "fast_period": fast_period,
                "slow_period": slow_period,
                "best_metric": metric_value,
                "results": {
                    "Sharpe Ratio": sharpe_ratio,
                    "Sortino Ratio": sortino_ratio,
                    "Total Return (%)": total_return,
                    "Final Account Balance": f"${cash:,.2f}"
                }
            })

    return best_result

# Backtest Function
def run_backtest(df, fast_period, slow_period):
    initial_cash = 10000
    cash = initial_cash
    position_size = 0
    entry_price = None
    position_type = None
    trade_results = []

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

    # Create balance series
    balance_series = pd.Series(initial_cash, index=df.index, dtype='float64')
    trade_indices = df.index[max(fast_period, slow_period):len(trade_results) + max(fast_period, slow_period)]
    balance_series[trade_indices] = initial_cash + np.cumsum(trade_results).astype('float64')

    return cash, trade_results, balance_series

# Sortino Ratio Function
def calculate_sortino_ratio(daily_returns, target_return=0):
    excess_returns = daily_returns - target_return
    downside_returns = excess_returns[excess_returns < 0]
    if downside_returns.empty or downside_returns.std() == 0:
        return np.inf
    downside_std = downside_returns.std() * np.sqrt(252)
    annualized_mean_excess_return = daily_returns.mean() * 252
    return annualized_mean_excess_return / downside_std

# Run Optimization
fast_range = range(5, 51, 5)   # Fast MA range (e.g., 5 to 50)
slow_range = range(10, 101, 10)  # Slow MA range (e.g., 10 to 100)

best_settings = optimize_strategy(df, fast_range, slow_range, metric="sharpe_ratio")

# Print Best Results
print("\nBest Settings Found:")
print(f"Fast MA: {best_settings['fast_period']}")
print(f"Slow MA: {best_settings['slow_period']}")
print(f"Best Metric (Sharpe Ratio): {best_settings['best_metric']:.2f}")
print("\nPerformance Summary:")
for key, value in best_settings["results"].items():
    print(f"{key:25}: {value}")