import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from itertools import product
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def run_backtest(df, buy_threshold, take_profit, stop_loss):
    """
    Executes the VWAP mean reversion backtest with both long and short positions.

    Parameters:
    - df (DataFrame): The input data containing price and volume information.
    - buy_threshold (float): Threshold below VWAP to trigger a long entry.
    - take_profit (float): Points above/below entry price to take profit.
    - stop_loss (float): Points below/above entry price to stop loss.

    Returns:
    - results (dict): Performance metrics for the given parameter set.
    - trades (list): List of executed trades with details.
    - equity_series (Series): Equity curve over time.
    """
    # Initialize Backtest Parameters
    initial_balance = 5000  # Starting capital
    cash = initial_balance
    multiplier = 5          # MES futures multiplier ($5 per point)
    position_size = 1       # Number of MES contracts per trade
    commission = 1.24       # Round-trip commission per trade
    slippage = 0.25         # Simulated slippage in points

    # Sell Threshold uses the same buy_threshold
    vwap_sell_threshold = buy_threshold  # Sell threshold equals buy_threshold

    # Tracking variables
    position = 0                # Current position: 0 = flat, +1 = long, -1 = short
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

    # Correct bars per trading day for 5-minute intervals during extended hours
    bars_per_trading_day = 276  # 1380 minutes / 5-minute bars

    # Backtest Loop
    for i in range(len(df)):
        current_time = df.index[i]
        current_price = df['close'].iloc[i]
        current_vwap = df['VWAP'].iloc[i]

        # Update exposure time
        if position != 0:
            bars_in_position += 1

        # Strategy Logic
        if position == 0:
            # Buy Signal: Price is a certain amount below VWAP
            if current_price < (current_vwap - buy_threshold):
                entry_price = current_price + slippage
                tp_price = entry_price + take_profit      # Target exit price above entry
                sl_price = entry_price - stop_loss        # Stop Loss below entry
                position = 1  # Long position
                cash -= entry_price * multiplier          # Deduct the cost
                trades.append({
                    'Action': 'BUY',
                    'Date': current_time,
                    'Price': entry_price,
                    'Target Price': tp_price,
                    'Stop Loss': sl_price,
                    'PnL': 0.0
                })
                # Uncomment the next line to see trade logs during optimization
                # print(f"BUY at {current_time} | Price: {entry_price:.2f} | TP: {tp_price:.2f} | SL: {sl_price:.2f}")
            # Short Signal: Price is a certain amount above VWAP
            elif current_price > (current_vwap + buy_threshold):
                entry_price = current_price - slippage
                tp_price = entry_price - take_profit      # Target exit price below entry
                sl_price = entry_price + stop_loss        # Stop Loss above entry
                position = -1  # Short position
                cash += entry_price * multiplier          # Add the proceeds from shorting
                trades.append({
                    'Action': 'SHORT',
                    'Date': current_time,
                    'Price': entry_price,
                    'Target Price': tp_price,
                    'Stop Loss': sl_price,
                    'PnL': 0.0
                })
                # Uncomment the next line to see trade logs during optimization
                # print(f"SHORT at {current_time} | Price: {entry_price:.2f} | TP: {tp_price:.2f} | SL: {sl_price:.2f}")
        elif position == 1:
            # Long Position: Check for exit conditions
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
                position = 0  # Reset to flat
                # Uncomment the next line to see trade logs during optimization
                # print(f"SELL ({sell_reason}) at {current_time} | Price: {actual_exit_price:.2f} | PnL: {pnl:.2f}")
        elif position == -1:
            # Short Position: Check for exit conditions
            cover_triggered = False
            pnl = 0.0
            actual_exit_price = 0.0
            cover_reason = ""

            # 1. Take Profit (TP) Condition
            if current_price <= tp_price:
                actual_exit_price = tp_price + slippage
                cash -= actual_exit_price * multiplier
                pnl = (entry_price - actual_exit_price) * multiplier * position_size - commission
                cover_triggered = True
                cover_reason = 'TP'

            # 2. VWAP Buy Threshold Condition
            elif current_price <= (current_vwap - vwap_sell_threshold):
                actual_exit_price = (current_vwap - vwap_sell_threshold) + slippage
                cash -= actual_exit_price * multiplier
                pnl = (entry_price - actual_exit_price) * multiplier * position_size - commission
                cover_triggered = True
                cover_reason = 'VWAP Buy Threshold'

            # 3. Stop Loss (SL) Condition
            elif current_price >= sl_price:
                actual_exit_price = sl_price - slippage
                cash -= actual_exit_price * multiplier
                pnl = (entry_price - actual_exit_price) * multiplier * position_size - commission
                cover_triggered = True
                cover_reason = 'SL'

            # Execute Cover if any condition is met
            if cover_triggered:
                trades.append({
                    'Action': 'COVER',
                    'Date': current_time,
                    'Price': actual_exit_price,
                    'PnL': pnl
                })
                position = 0  # Reset to flat
                # Uncomment the next line to see trade logs during optimization
                # print(f"COVER ({cover_reason}) at {current_time} | Price: {actual_exit_price:.2f} | PnL: {pnl:.2f}")

        # Calculate current equity
        if position == 1:
            current_equity = cash + (current_price * multiplier * position_size)
        elif position == -1:
            current_equity = cash - (current_price * multiplier * position_size)
        else:
            current_equity = cash

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
                # Uncomment the next line to see drawdown logs during optimization
                # print(f"Drawdown of {current_drawdown_duration} bars ended on {current_time}")
                current_drawdown_duration = 0

        # Update max_drawdown
        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown

    # Handle last drawdown duration
    if current_drawdown_duration > 0:
        drawdown_durations.append(current_drawdown_duration)
        # Uncomment the next line to see final drawdown logs during optimization
        # print(f"Final drawdown of {current_drawdown_duration} bars ended on {current_time}")

    # Calculate Drawdown Durations in Days
    if drawdown_durations:
        average_drawdown_duration_days = np.mean(drawdown_durations) / bars_per_trading_day
        max_drawdown_duration_days = max(drawdown_durations) / bars_per_trading_day
    else:
        average_drawdown_duration_days = 0.0
        max_drawdown_duration_days = 0.0

    average_drawdown = np.mean(drawdowns)

    # Final Portfolio Value
    if position == 1:
        final_balance = cash + (df['close'].iloc[-1] * multiplier * position_size)
    elif position == -1:
        final_balance = cash - (df['close'].iloc[-1] * multiplier * position_size)
    else:
        final_balance = cash

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

    # Ensure trades are in BUY-SELL and SHORT-COVER pairs
    for i in range(0, len(trades), 2):
        if i + 1 < len(trades):
            entry = trades[i]
            exit = trades[i + 1]
            if entry['Action'] == 'BUY' and exit['Action'] == 'SELL':
                pnl = exit['PnL']
            elif entry['Action'] == 'SHORT' and exit['Action'] == 'COVER':
                pnl = exit['PnL']
            else:
                continue  # Skip incomplete or mismatched pairs
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

    # Compile Results
    results = {
        "buy_threshold": buy_threshold,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "Total Return (%)": total_return,
        "Annualized Return (%)": annualized_return,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
        "Max Drawdown (%)": max_drawdown,
        "Win Rate (%)": win_rate,
        "Profit Factor": profit_factor,
        "Total Trades": len(trade_results),
        "Winning Trades": len(winning_trades),
        "Losing Trades": len(losing_trades),
        "Exposure Time (%)": exposure_time_percentage
    }

    return results, trades, equity_series

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
from_date = '2022-09-25'  # YYYY-MM-DD
to_date = '2023-12-11'    # YYYY-MM-DD

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
# Define Optimization Parameter Grid
# ----------------------------

# Define parameter ranges for optimization
buy_thresholds = [10, 15, 20]      # Example: 10, 15, 20 points below/above VWAP
take_profits = [5, 10, 15]         # Example: 5, 10, 15 points above/below entry
stop_losses = [3, 4, 5]            # Example: 3, 4, 5 points below/above entry

# ----------------------------
# Implement Grid Search for Optimization
# ----------------------------

# Initialize a list to store all optimization results
optimization_results = []

# Generate all possible combinations of parameters
parameter_combinations = list(product(buy_thresholds, take_profits, stop_losses))

# Total number of combinations
total_combinations = len(parameter_combinations)
print(f"Total parameter combinations to test: {total_combinations}")

# Iterate over each combination and run the backtest
for idx, (buy_threshold, take_profit, stop_loss) in enumerate(parameter_combinations, 1):
    print(f"\nRunning backtest {idx}/{total_combinations} with parameters:")
    print(f"Buy Threshold: {buy_threshold}, Take Profit: {take_profit}, Stop Loss: {stop_loss}")

    # Execute backtest
    results, trades, equity_series = run_backtest(df, buy_threshold, take_profit, stop_loss)

    # Append the results
    optimization_results.append(results)

# ----------------------------
# Analyze Optimization Results
# ----------------------------

# Convert optimization results to a DataFrame
optimization_df = pd.DataFrame(optimization_results)

# Display all results
print("\nOptimization Results:")
print(optimization_df)

# Identify the best parameter set based on Sharpe Ratio
best_sharpe = optimization_df.loc[optimization_df['Sharpe Ratio'].idxmax()]
print("\nBest Parameter Set based on Sharpe Ratio:")
print(best_sharpe)

# Alternatively, identify the best parameter set based on Total Return
best_return = optimization_df.loc[optimization_df['Total Return (%)'].idxmax()]
print("\nBest Parameter Set based on Total Return:")
print(best_return)

# ----------------------------
# Visualize Optimization Results
# ----------------------------

# Scatter plot of Buy Threshold vs. Take Profit colored by Sharpe Ratio
plt.figure(figsize=(12, 6))
scatter = plt.scatter(optimization_df['buy_threshold'], optimization_df['take_profit'],
                      c=optimization_df['Sharpe Ratio'], cmap='viridis', s=100, alpha=0.7)
plt.colorbar(scatter, label='Sharpe Ratio')
plt.xlabel('Buy Threshold (points below/above VWAP)')
plt.ylabel('Take Profit (points above/below entry)')
plt.title('Buy Threshold vs. Take Profit colored by Sharpe Ratio')
plt.grid(True)
plt.show()

# Scatter plot of Buy Threshold vs. Stop Loss colored by Total Return
plt.figure(figsize=(12, 6))
scatter = plt.scatter(optimization_df['buy_threshold'], optimization_df['stop_loss'],
                      c=optimization_df['Total Return (%)'], cmap='plasma', s=100, alpha=0.7)
plt.colorbar(scatter, label='Total Return (%)')
plt.xlabel('Buy Threshold (points below/above VWAP)')
plt.ylabel('Stop Loss (points below/above entry)')
plt.title('Buy Threshold vs. Stop Loss colored by Total Return')
plt.grid(True)
plt.show()

# Heatmap for Buy Threshold and Take Profit with Sharpe Ratio
pivot_sharpe = optimization_df.pivot(index='buy_threshold', columns='take_profit', values='Sharpe Ratio')
plt.figure(figsize=(10, 8))
plt.title('Heatmap of Sharpe Ratio')
sns.heatmap(pivot_sharpe, annot=True, fmt=".2f", cmap='YlGnBu')
plt.xlabel('Take Profit (points above/below entry)')
plt.ylabel('Buy Threshold (points below/above VWAP)')
plt.show()

# Heatmap for Buy Threshold and Stop Loss with Total Return
pivot_return = optimization_df.pivot(index='buy_threshold', columns='stop_loss', values='Total Return (%)')
plt.figure(figsize=(10, 8))
plt.title('Heatmap of Total Return (%)')
sns.heatmap(pivot_return, annot=True, fmt=".2f", cmap='YlOrRd')
plt.xlabel('Stop Loss (points below/above entry)')
plt.ylabel('Buy Threshold (points below/above VWAP)')
plt.show()