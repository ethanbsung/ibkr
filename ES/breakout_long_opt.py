import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import timedelta, time
from itertools import product

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)  # Change to DEBUG for more detailed logs
logger = logging.getLogger(__name__)

# --- Function to Load Data ---
def load_data(csv_file):
    try:
        # Read the CSV without 'infer_datetime_format' to avoid FutureWarning
        df = pd.read_csv(
            csv_file,
            parse_dates=['Time'],
            dayfirst=False,  # Adjust based on your date format (True if day comes first)
            na_values=['', 'NA', 'NaN']  # Handle missing values
        )
        
        # Check if 'Time' column exists
        if 'Time' not in df.columns:
            logger.error("The CSV file does not contain a 'Time' column.")
            exit(1)
        
        # Verify 'Time' column data type
        if not np.issubdtype(df['Time'].dtype, np.datetime64):
            logger.error("The 'Time' column was not parsed as datetime. Please check the date format.")
            exit(1)
        
        # Handle timezone information if present
        if df['Time'].dt.tz is not None:
            df['Time'] = df['Time'].dt.tz_convert(None)
        
        # Sort by 'Time' and set it as the index
        df.sort_values('Time', inplace=True)
        df.set_index('Time', inplace=True)
        
        # Drop the 'Symbol' column as it's redundant for backtesting (unless needed)
        if 'Symbol' in df.columns:
            df.drop(columns=['Symbol'], inplace=True)
        
        # Rename 'Last' to 'Close' for clarity
        if 'Last' in df.columns:
            df.rename(columns={'Last': 'Close'}, inplace=True)
        
        # Optional: Drop unnecessary columns
        unnecessary_cols = ['Change', '%Chg', 'Open Int']
        df.drop(columns=[col for col in unnecessary_cols if col in df.columns], inplace=True)
        
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {csv_file}")
        exit(1)
    except pd.errors.EmptyDataError:
        logger.error("The CSV file is empty.")
        exit(1)
    except pd.errors.ParserError as e:
        logger.error(f"Parser error while reading the CSV: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading the CSV: {e}")
        exit(1)

# --- Backtest Function ---
def run_backtest(df_30m, rolling_window, stop_loss_points, take_profit_points, 
                ES_MULTIPLIER=5, POSITION_SIZE=1, COMMISSION=1.24, INITIAL_CASH=5000):
    """
    Runs the backtest for given parameters.
    
    Parameters:
    - df_30m: Resampled 30-minute DataFrame.
    - rolling_window: Number of bars for rolling high.
    - stop_loss_points: Stop loss in points.
    - take_profit_points: Take profit in points.
    - ES_MULTIPLIER: Multiplier for ES points.
    - POSITION_SIZE: Number of contracts.
    - COMMISSION: Commission per trade.
    - INITIAL_CASH: Starting cash.
    
    Returns:
    - results: Dictionary of performance metrics.
    - balance_df: DataFrame of equity over time.
    """
    # Copy the DataFrame to avoid modifying the original
    df = df_30m.copy()
    
    # Compute Rolling High
    df['Rolling_High'] = df['High'].shift(1).rolling(window=rolling_window, min_periods=rolling_window).max()
    
    # Remove rows where rolling high is NaN
    df.dropna(subset=['Rolling_High'], inplace=True)
    
    # Initialize variables
    cash = INITIAL_CASH
    trade_results = []
    balance_series = [INITIAL_CASH]
    balance_dates = [df.index.min()]
    position = None
    active_bars = 0
    total_bars = len(df)
    
    # Define regular trading hours
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    # Backtesting Loop
    for idx, (current_time, row) in enumerate(df.iterrows()):
        if position is None:
            # Entry Condition: Breakout above the rolling high
            # Ensure that the current time is within regular trading hours (09:30 to 16:00)
            if (current_time.time() >= market_open) and (current_time.time() < market_close):
                breakout_level = row['Rolling_High']
                # Check if the current bar's high breaks the breakout level
                if row['High'] > breakout_level:
                    entry_price = breakout_level  # Assume entry at the breakout level
                    stop_loss_price = entry_price - stop_loss_points
                    take_profit_price = entry_price + take_profit_points

                    # Enter the trade
                    position = {
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss_price,
                        'take_profit': take_profit_price
                    }
                    active_bars += 1  # Increase exposure
                    # Uncomment the following lines for detailed logs
                    # logger.info(f"[ENTRY] Long entered at {entry_price} on {current_time}")
                    # logger.debug(f"Breakout Detected! Current High: {row['High']} > Rolling High: {breakout_level}")
        else:
            # Manage existing position within the current 30-min bar
            # Check if stop loss or take profit is hit
            current_high = row['High']
            current_low = row['Low']

            exit_time = current_time
            exit_price = row['Close']  # Default exit price

            # Flags to check if exit occurred
            exited = False

            # Check if stop loss is hit
            if current_low <= position['stop_loss']:
                exit_price = position['stop_loss']
                pnl = (exit_price - position['entry_price']) * POSITION_SIZE * ES_MULTIPLIER - COMMISSION
                cash += pnl  # Add PnL to cash
                trade_results.append(pnl)
                balance_series.append(cash)
                balance_dates.append(exit_time)
                # Uncomment the following lines for detailed logs
                # logger.info(f"[STOP LOSS] Exit at {exit_price} on {exit_time}, PnL: ${pnl:,.2f}")
                # logger.debug(f"Stop Loss Hit: Exit Price: {exit_price}, PnL: {pnl}")
                position = None
                exited = True

            # Check if take profit is hit
            elif current_high >= position['take_profit']:
                exit_price = position['take_profit']
                pnl = (exit_price - position['entry_price']) * POSITION_SIZE * ES_MULTIPLIER - COMMISSION
                cash += pnl  # Add PnL to cash
                trade_results.append(pnl)
                balance_series.append(cash)
                balance_dates.append(exit_time)
                # Uncomment the following lines for detailed logs
                # logger.info(f"[TAKE PROFIT] Exit at {exit_price} on {exit_time}, PnL: ${pnl:,.2f}")
                # logger.debug(f"Take Profit Hit: Exit Price: {exit_price}, PnL: {pnl}")
                position = None
                exited = True

            # No exit triggered; position remains open until SL or TP is hit
            # No EOD exit

        # Update balance for the current bar if no trade occurred
        if not position:
            if len(balance_series) == len(balance_dates):
                balance_series.append(cash)
                balance_dates.append(current_time)

    # --- Calculate Exposure Time ---
    exposure_time_percentage = (active_bars / total_bars) * 100 if total_bars > 0 else 0

    # --- Create Balance DataFrame ---
    balance_df = pd.DataFrame({
        'Datetime': balance_dates,
        'Equity': balance_series
    }).set_index('Datetime').sort_index()

    # --- Calculate Equity Peak ---
    equity_peak = balance_df['Equity'].max()

    # --- Calculate Maximum Drawdown ---
    rolling_max = balance_df['Equity'].cummax()
    drawdown = (balance_df['Equity'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100  # As percentage

    # --- Calculate Max Drawdown Duration ---
    # Identify drawdown periods
    drawdown_periods = drawdown[drawdown < 0]
    if not drawdown_periods.empty:
        end_dates = drawdown_periods.index.to_series().diff().ne(timedelta(minutes=30)).cumsum()
        drawdown_groups = drawdown_periods.groupby(end_dates)
        drawdown_durations = drawdown_groups.size()
        max_drawdown_duration_days = drawdown_durations.max() * 0.0208333  # 30 minutes = 0.0208333 days
        average_drawdown_duration_days = drawdown_durations.mean() * 0.0208333
    else:
        max_drawdown_duration_days = 0
        average_drawdown_duration_days = 0

    # --- Calculate Average Drawdown ---
    average_drawdown = drawdown.min() * 100  # Using min drawdown as average for simplicity
    # For a more accurate average, consider using the mean of all drawdowns

    # --- Calculate Profit Factor ---
    gross_profit = sum([pnl for pnl in trade_results if pnl > 0])
    gross_loss = abs(sum([pnl for pnl in trade_results if pnl < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan

    # --- Identify Winning and Losing Trades ---
    winning_trades = [pnl for pnl in trade_results if pnl > 0]
    losing_trades = [pnl for pnl in trade_results if pnl < 0]

    # --- Calculate Sortino Ratio ---
    # Assuming a minimal acceptable return (MAR) of 0
    mar = 0
    strategy_returns = np.array(trade_results) / INITIAL_CASH
    downside_returns = np.where(strategy_returns < mar, strategy_returns - mar, 0)
    expected_return = np.mean(strategy_returns) if len(strategy_returns) > 0 else 0
    downside_deviation = np.std(downside_returns) if np.std(downside_returns) > 0 else np.nan
    sortino_ratio = (expected_return - mar) / downside_deviation * np.sqrt(252) if downside_deviation != 0 else np.nan

    # --- Calculate Calmar Ratio ---
    days = (df.index.max() - df.index.min()).days if (df.index.max() - df.index.min()).days > 0 else 1
    annualized_return_percentage = ((cash / INITIAL_CASH) ** (365.0 / days) - 1) * 100
    calmar_ratio = annualized_return_percentage / abs(max_drawdown) if max_drawdown != 0 else np.nan

    # --- Calculate Benchmark Return ---
    # Assuming benchmark is buy-and-hold based on the close price
    initial_close = df.iloc[0]['Close']
    final_close = df.iloc[-1]['Close']
    benchmark_return = ((final_close - initial_close) / initial_close) * 100

    # --- Calculate Sharpe Ratio ---
    # Assuming risk-free rate is 0 for simplicity
    returns = balance_df['Equity'].pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else np.nan

    # --- Performance Metrics ---
    results = {
        "Rolling Window": rolling_window,
        "Stop Loss Points": stop_loss_points,
        "Take Profit Points": take_profit_points,
        "Exposure Time (%)": f"{exposure_time_percentage:.2f}",
        "Final Account Balance ($)": f"{cash:,.2f}",
        "Equity Peak ($)": f"{equity_peak:,.2f}",
        "Total Return (%)": f"{((cash - INITIAL_CASH) / INITIAL_CASH) * 100:.2f}",
        "Annualized Return (%)": f"{annualized_return_percentage:.2f}",
        "Benchmark Return (%)": f"{benchmark_return:.2f}",
        "Volatility (Annual %)": f"{balance_df['Equity'].pct_change().std() * np.sqrt(252) * 100:.2f}",
        "Total Trades": len(trade_results),
        "Winning Trades": len(winning_trades),
        "Losing Trades": len(losing_trades),
        "Win Rate (%)": f"{(len(winning_trades)/len(trade_results)*100) if trade_results else 0:.2f}",
        "Profit Factor": f"{profit_factor:.2f}",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Sortino Ratio": f"{sortino_ratio:.2f}",
        "Calmar Ratio": f"{calmar_ratio:.2f}",
        "Max Drawdown (%)": f"{max_drawdown:.2f}",
        "Average Drawdown (%)": f"{average_drawdown:.2f}",
        "Max Drawdown Duration (days)": f"{max_drawdown_duration_days:.2f}",
        "Average Drawdown Duration (days)": f"{average_drawdown_duration_days:.2f}",
    }

    return results, balance_df

# --- Main Optimization Loop ---
def optimize_strategy(df_30m):
    # Define parameter grids
    rolling_window_options = range(10, 31, 5)  # Example: 15 to 30
    stop_loss_options = range(3, 11, 1)        # Example: 3 to 10
    take_profit_options = range(3, 31, 1)     # Example: 3 to 30

    # Create all combinations
    parameter_combinations = list(product(rolling_window_options, stop_loss_options, take_profit_options))

    # Initialize list to store results
    optimization_results = []

    # Iterate over each combination
    for params in parameter_combinations:
        rolling_window, stop_loss, take_profit = params
        print(f"\nRunning Backtest with Rolling Window: {rolling_window}, Stop Loss: {stop_loss}, Take Profit: {take_profit}")

        # Run backtest and capture both results and balance_df
        results, _ = run_backtest(
            df_30m=df_30m,
            rolling_window=rolling_window,
            stop_loss_points=stop_loss,
            take_profit_points=take_profit,
            ES_MULTIPLIER=5,
            POSITION_SIZE=1,
            COMMISSION=1.24,
            INITIAL_CASH=5000
        )

        # Append results
        optimization_results.append(results)

        # Print results
        print("Performance Metrics:")
        for key, value in results.items():
            if key in ["Rolling Window", "Stop Loss Points", "Take Profit Points", "Total Trades", "Winning Trades", "Losing Trades"]:
                print(f"  {key}: {value}")
            elif key == "Sharpe Ratio":
                print(f"  {key}: {value}")
            else:
                # To keep the output concise, only print selected metrics
                if key in ["Exposure Time (%)", "Final Account Balance ($)", "Equity Peak ($)",
                           "Total Return (%)", "Annualized Return (%)", "Benchmark Return (%)",
                           "Volatility (Annual %)", "Win Rate (%)", "Profit Factor",
                           "Sortino Ratio", "Calmar Ratio", "Max Drawdown (%)",
                           "Average Drawdown (%)", "Max Drawdown Duration (days)",
                           "Average Drawdown Duration (days)"]:
                    print(f"  {key}: {value}")

    # Convert results to DataFrame for further analysis
    results_df = pd.DataFrame(optimization_results)

    # Sort the DataFrame by Sharpe Ratio in descending order
    results_df['Sharpe Ratio Float'] = pd.to_numeric(results_df['Sharpe Ratio'], errors='coerce')
    sorted_results = results_df.sort_values(by='Sharpe Ratio Float', ascending=False).dropna(subset=['Sharpe Ratio Float'])

    return sorted_results

# --- Execute Optimization ---
if __name__ == "__main__":
    # --- Configuration Parameters ---
    INTRADAY_DATA_FILE = 'es_1m_data.csv'    # Path to 1-minute CSV
    INITIAL_CASH = 5000                      # Define INITIAL_CASH globally

    # --- Load Intraday Dataset ---
    df_intraday = load_data(INTRADAY_DATA_FILE)

    # --- Verify Full Data Range ---
    print("\nFull Data Range:")
    print(df_intraday.index.min(), "to", df_intraday.index.max())

    # --- Define Backtest Period ---
    # You can customize these dates as needed
    custom_start_date = "2021-01-01"
    custom_end_date = "2024-12-23"
    start_time = pd.to_datetime(custom_start_date).tz_localize(None)
    end_time = pd.to_datetime(custom_end_date).tz_localize(None)

    # --- Filter Data by Backtest Period ---
    df_intraday_filtered = df_intraday.loc[start_time:end_time].copy()
    print("\nFiltered Data Range:")
    print(df_intraday_filtered.index.min(), "to", df_intraday_filtered.index.max())
    print("Number of Rows After Filtering:", len(df_intraday_filtered))

    # --- Resample to 30-minute bars ---
    df_30m = df_intraday_filtered.resample('30min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    print("\nResampled 30-Minute Data Range:")
    print(df_30m.index.min(), "to", df_30m.index.max())
    print("Number of Rows After Resampling:", len(df_30m))

    # --- Optimize Strategy ---
    optimization_results_df = optimize_strategy(df_30m)

    # --- Save Optimization Results to CSV (Optional) ---
    optimization_results_df.to_csv('optimization_results_by_sharpe_ratio.csv', index=False)
    print("\nOptimization complete. Results saved to 'optimization_results_by_sharpe_ratio.csv'.")

    # --- Save Top N Parameter Combinations to CSV ---
    top_n = 10  # Define how many top parameter sets you want to save
    top_parameters_df = optimization_results_df.head(top_n).drop(columns=['Sharpe Ratio Float'])
    top_parameters_df.to_csv('top_parameters_by_sharpe_ratio.csv', index=False)
    print(f"\nTop {top_n} parameter combinations saved to 'top_parameters_by_sharpe_ratio.csv'.")

    # --- Identify Best Strategy Based on Sharpe Ratio ---
    if not optimization_results_df.empty:
        best_strategy = optimization_results_df.iloc[0]
        print("\nBest Strategy Based on Sharpe Ratio:")
        for key, value in best_strategy.items():
            if key != "Sharpe Ratio Float":  # Exclude the helper column
                print(f"  {key}: {value}")

        # --- Plotting the Best Strategy's Equity Curve ---
        # Run backtest again to get balance_df for the best strategy
        best_params = {
            "rolling_window": best_strategy["Rolling Window"],
            "stop_loss_points": best_strategy["Stop Loss Points"],
            "take_profit_points": best_strategy["Take Profit Points"]
        }
        best_results, balance_df_best = run_backtest(
            df_30m=df_30m,
            rolling_window=best_params["rolling_window"],
            stop_loss_points=best_params["stop_loss_points"],
            take_profit_points=best_params["take_profit_points"],
            ES_MULTIPLIER=5,
            POSITION_SIZE=1,
            COMMISSION=1.24,
            INITIAL_CASH=INITIAL_CASH
        )

        # Create benchmark equity curve
        initial_close = df_30m.iloc[0]['Close']
        benchmark_equity = (df_30m['Close'] / initial_close) * INITIAL_CASH
        benchmark_equity = benchmark_equity.reindex(balance_df_best.index, method='ffill').fillna(method='ffill')

        # Ensure no NaNs in benchmark_equity
        num_benchmark_nans = benchmark_equity.isna().sum()
        if num_benchmark_nans > 0:
            logger.warning(f"Benchmark equity has {num_benchmark_nans} NaN values. Filling with forward fill.")
            benchmark_equity = benchmark_equity.fillna(method='ffill')

        # Create a DataFrame for plotting
        equity_plot_df = pd.DataFrame({
            'Strategy': balance_df_best['Equity'],
            'Benchmark': benchmark_equity
        })

        # Plotting
        plt.figure(figsize=(14, 7))
        plt.plot(equity_plot_df.index, equity_plot_df['Strategy'], label='Strategy Equity')
        plt.plot(equity_plot_df.index, equity_plot_df['Benchmark'], label='Benchmark Equity')
        plt.title('Equity Curve: Best Strategy vs Benchmark')
        plt.xlabel('Time')
        plt.ylabel('Account Balance ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No valid strategies found during optimization.")