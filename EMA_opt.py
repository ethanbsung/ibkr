import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pytz
import logging
import sys
import time
import os
import warnings
from itertools import product

# --- Suppress Specific FutureWarnings (Optional) ---
warnings.filterwarnings("ignore", category=FutureWarning, message=".*'H' is deprecated.*")

# --- Configuration Parameters ---
INITIAL_CASH = 5000          # Starting cash
POSITION_SIZE = 1            # Number of contracts per trade
CONTRACT_MULTIPLIER = 5      # Contract multiplier
COMMISSION = 1.24            # Commission per trade

# --- Data File ---
CSV_FILE_DAILY = 'es_4h_data.csv'

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs if needed
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# --- Helper Functions ---

def load_data(csv_file):
    """
    Loads daily CSV data into a pandas DataFrame with appropriate data types and datetime parsing.

    Parameters:
        csv_file (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded and indexed DataFrame.
    """
    try:
        # Define column names for daily data
        df = pd.read_csv(
            csv_file,
            dtype={
                'Symbol': str,
                'Open': float,
                'High': float,
                'Low': float,
                'Last': float,    # Assuming 'Last' represents 'Close'
                'Volume': float,
                'Open Int': float  # May contain NaN
            },
            parse_dates=['Time'],
            date_format="%Y-%m-%d %H:%M"
        )

        # Sort by Time to ensure chronological order
        df.sort_values('Time', inplace=True)
        df.set_index('Time', inplace=True)

        # Rename columns to match the script's expectations
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Last': 'close',
            'Volume': 'volume'
            # 'Open Int' is optional and ignored in calculations
        }, inplace=True)

        # Drop unnecessary columns if they exist
        columns_to_keep = ['open', 'high', 'low', 'close', 'volume']
        existing_columns = [col for col in columns_to_keep if col in df.columns]
        df = df[existing_columns]

        # **Localize to UTC directly**
        if df.index.tz is None:
            df = df.tz_localize('UTC')
            logger.debug(f"Localized naive datetime index to UTC for {csv_file}.")
        else:
            df = df.tz_convert('UTC')
            logger.debug(f"Converted timezone-aware datetime index to UTC for {csv_file}.")

        # Remove duplicate timestamps by keeping the last entry (most liquid contract)
        df = df[~df.index.duplicated(keep='last')]
        logger.debug(f"Removed duplicate timestamps for {csv_file}.")

        return df
    except FileNotFoundError:
        logger.error(f"File not found: {csv_file}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logger.error("No data: The CSV file is empty.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred while reading the CSV '{csv_file}': {e}")
        sys.exit(1)

def calculate_emas(df, short_period, long_period):
    """
    Calculates short and long EMAs for the given DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'close' prices.
        short_period (int): Period for short EMA.
        long_period (int): Period for long EMA.

    Returns:
        pd.DataFrame: DataFrame with additional EMA columns.
    """
    df[f'short_ema_{short_period}'] = df['close'].ewm(span=short_period, adjust=False).mean()
    df[f'long_ema_{long_period}'] = df['close'].ewm(span=long_period, adjust=False).mean()
    return df

def detect_crossovers(df, short_period, long_period):
    """
    Detects EMA crossovers in the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing EMAs.
        short_period (int): Period for short EMA.
        long_period (int): Period for long EMA.

    Returns:
        pd.DataFrame: DataFrame with 'crossover' column indicating crossover signals.
                      1 for bullish crossover, -1 for bearish crossover, 0 otherwise.
    """
    short_ema = f'short_ema_{short_period}'
    long_ema = f'long_ema_{long_period}'

    # Initialize 'crossover' column using .loc to avoid SettingWithCopyWarning
    df.loc[:, 'crossover'] = 0

    # Bullish crossover: Short EMA crosses above Long EMA
    bullish = (df[short_ema] > df[long_ema]) & (df[short_ema].shift(1) <= df[long_ema].shift(1))
    df.loc[bullish, 'crossover'] = 1

    # Bearish crossover: Short EMA crosses below Long EMA
    bearish = (df[short_ema] < df[long_ema]) & (df[short_ema].shift(1) >= df[long_ema].shift(1))
    df.loc[bearish, 'crossover'] = -1

    return df

def calculate_sortino_ratio(daily_returns, target_return=0):
    """
    Calculate the annualized Sortino Ratio.

    Parameters:
        daily_returns (pd.Series): Series of daily returns.
        target_return (float): Minimum acceptable return.

    Returns:
        float: Annualized Sortino Ratio.
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

def run_backtest(df, short_period, long_period):
    """
    Runs the backtest for given EMA periods.

    Parameters:
        df (pd.DataFrame): DataFrame containing market data.
        short_period (int): Short EMA period.
        long_period (int): Long EMA period.

    Returns:
        dict: Dictionary containing performance metrics.
    """
    # Calculate EMAs
    df = calculate_emas(df, short_period, long_period)

    # Detect Crossovers
    df = detect_crossovers(df, short_period, long_period)

    # Initialize Backtest Variables
    position_size = 0
    entry_price = None
    position_type = None  
    cash = INITIAL_CASH
    trade_results = []
    balance_series = [INITIAL_CASH]  # Initialize with starting cash
    balance_times = [df.index[0]]  # Initialize with first date
    exposure_days = 0

    # For Drawdown Duration Calculations
    in_drawdown = False
    drawdown_start = None
    drawdown_durations = []

    # Backtesting Loop
    for i in range(len(df)):
        current_bar = df.iloc[i]
        current_time = df.index[i]
        current_price = current_bar['close']

        # Count exposure when position is active
        if position_size != 0:
            exposure_days += 1

        if position_size == 0:
            # No open position, check for entry signals based on EMA crossover
            if current_bar['crossover'] == 1:
                # Bullish Crossover: Enter Long
                position_size = POSITION_SIZE
                entry_price = current_price
                position_type = 'long'
                entry_time = current_time
                # Log Entry
                # logger.info(f"Entered LONG at {entry_price} on {entry_time.strftime('%Y-%m-%d')} UTC")
        else:
            # Position is open, check for exit signal (bearish crossover)
            if current_bar['crossover'] == -1:
                # Bearish Crossover: Exit Long
                exit_price = current_price
                pnl = (exit_price - entry_price) * CONTRACT_MULTIPLIER * position_size - COMMISSION
                trade_results.append(pnl)
                cash += pnl
                balance_series.append(cash)
                balance_times.append(current_time)
                # Log Exit
                # logger.info(f"Exited LONG at {exit_price} on {current_time.strftime('%Y-%m-%d')} UTC for P&L: ${pnl:.2f}")

                # Reset position variables
                position_size = 0
                position_type = None
                entry_price = None
                entry_time = None

        # Append current cash to balance_series if no trade occurred
        if position_size == 0:
            balance_series.append(cash)
            balance_times.append(current_time)

    # Handle the case where the last position is still open at the end of the backtest
    if position_size != 0:
        exit_price = df.iloc[-1]['close']
        exit_time = df.index[-1]
        pnl = (exit_price - entry_price) * CONTRACT_MULTIPLIER * position_size - COMMISSION
        trade_results.append(pnl)
        cash += pnl
        balance_series.append(cash)
        balance_times.append(exit_time)
        # logger.info(f"Exited LONG at {exit_price} on {exit_time.strftime('%Y-%m-%d')} UTC via END OF DATA for P&L: ${pnl:.2f}")

    # Convert balance_series to a Pandas Series with appropriate index
    balance_series = pd.Series(balance_series, index=balance_times)

    # Calculate Benchmark Equity Curve
    initial_price = df['close'].iloc[0]  # First closing price in the backtest period
    benchmark_equity_curve = (df['close'] / initial_price) * INITIAL_CASH

    # Drawdown Duration Tracking (optional, can be added to metrics if needed)

    # Calculate Daily Returns
    daily_returns = balance_series.pct_change().dropna()

    # Calculate Performance Metrics
    total_return_percentage = ((cash - INITIAL_CASH) / INITIAL_CASH) * 100
    trading_days = max((df.index.max() - df.index.min()).days, 1)
    annualized_return_percentage = ((cash / INITIAL_CASH) ** (252 / trading_days)) - 1
    benchmark_return = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
    equity_peak = balance_series.max()

    volatility_annual = daily_returns.std() * np.sqrt(252) * 100
    risk_free_rate = 0  # Example: 0 for simplicity or use the current rate
    sharpe_ratio = ((daily_returns.mean() - risk_free_rate) / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    sortino_ratio = calculate_sortino_ratio(daily_returns)

    # Drawdown Calculations
    running_max_series = balance_series.cummax()
    drawdowns = (balance_series - running_max_series) / running_max_series
    max_drawdown = drawdowns.min() * 100
    average_drawdown = drawdowns[drawdowns < 0].mean() * 100

    # Exposure Time
    exposure_days_total = balance_series.diff().notnull().sum()  # Approximation
    exposure_time_percentage = (exposure_days / len(df)) * 100

    # Profit Factor
    winning_trades = [pnl for pnl in trade_results if pnl > 0]
    losing_trades = [pnl for pnl in trade_results if pnl <= 0]
    profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf')

    # Calmar Ratio Calculation
    calmar_ratio = (total_return_percentage / abs(max_drawdown)) if max_drawdown != 0 else float('inf')

    # Prepare Performance Metrics Dictionary
    metrics = {
        "Short EMA": short_period,
        "Long EMA": long_period,
        "Total Return (%)": total_return_percentage,
        "Annualized Return (%)": annualized_return_percentage * 100,
        "Benchmark Return (%)": benchmark_return,
        "Volatility (Annual %)": volatility_annual,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
        "Max Drawdown (%)": max_drawdown,
        "Average Drawdown (%)": average_drawdown,
        "Profit Factor": profit_factor,
        "Total Trades": len(trade_results),
        "Winning Trades": len(winning_trades),
        "Losing Trades": len(losing_trades),
        "Win Rate (%)": (len(winning_trades)/len(trade_results)*100) if trade_results else 0,
        "Exposure Time (%)": exposure_time_percentage
    }

    return {
        "metrics": metrics,
        "balance_series": balance_series,
        "benchmark_equity_curve": benchmark_equity_curve
    }

def optimize_ema(df, short_periods, long_periods):
    """
    Optimizes EMA periods by evaluating all combinations and selecting the best based on Sharpe Ratio.

    Parameters:
        df (pd.DataFrame): DataFrame containing market data.
        short_periods (list): List of short EMA periods to test.
        long_periods (list): List of long EMA periods to test.

    Returns:
        pd.DataFrame: DataFrame containing performance metrics for each EMA combination.
        tuple: Best (short_period, long_period) pair based on Sharpe Ratio.
    """
    results = []

    # Generate all valid combinations where short < long
    for short, long in product(short_periods, long_periods):
        if short >= long:
            continue  # Ensure short EMA is less than long EMA

        backtest_result = run_backtest(df.copy(), short, long)
        metrics = backtest_result["metrics"]
        metrics["Short EMA"] = short
        metrics["Long EMA"] = long
        results.append(metrics)

        logger.info(f"Tested Short EMA: {short}, Long EMA: {long} | Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Identify the best EMA pair based on Sharpe Ratio
    best_row = results_df.loc[results_df['Sharpe Ratio'].idxmax()]
    best_short = best_row['Short EMA']
    best_long = best_row['Long EMA']

    logger.info(f"Best EMA Pair -> Short EMA: {best_short}, Long EMA: {best_long} with Sharpe Ratio: {best_row['Sharpe Ratio']:.2f}")

    return results_df, (best_short, best_long)

# --- Main Execution ---

if __name__ == "__main__":
    # --- Load and Prepare Data ---
    logger.info("Loading daily dataset...")
    df_daily = load_data(CSV_FILE_DAILY)  # Load daily data
    logger.info("Daily dataset loaded successfully.")
    print(f"Daily Data Range: {df_daily.index.min()} to {df_daily.index.max()}")
    print(f"Daily Data Sample:\n{df_daily.head()}")

    # --- Define Backtest Period ---
    # Custom Backtest Period (Replace These Dates as needed)
    custom_start_date = "2014-09-25"
    custom_end_date = "2024-12-11"

    # Convert custom dates to UTC
    try:
        start_time = pd.to_datetime(custom_start_date).tz_localize('UTC')
        end_time = pd.to_datetime(custom_end_date).tz_localize('UTC')
    except Exception as e:
        logger.error(f"Error parsing custom dates: {e}")
        sys.exit(1)

    logger.info(f"Backtest Period: {start_time} to {end_time}")

    # --- Slice Dataframe to Backtest Period ---
    logger.info(f"Slicing daily data from {start_time} to {end_time}...")
    df_daily_bt = df_daily.loc[start_time:end_time]
    logger.info("Daily data sliced to backtest period.")
    print(f"Sliced Daily Data Range: {df_daily_bt.index.min()} to {df_daily_bt.index.max()}")
    print(f"Sliced Daily Data Sample:\n{df_daily_bt.head()}")

    # --- Define EMA Period Ranges for Optimization ---
    short_periods = range(10, 61, 5)   # 10, 15, ..., 60
    long_periods = range(20, 301, 10) # 100, 120, ..., 300

    # --- Run Optimization ---
    logger.info("Starting EMA optimization...")
    start_opt_time = time.time()
    results_df, best_pair = optimize_ema(df_daily_bt, short_periods, long_periods)
    end_opt_time = time.time()
    logger.info(f"EMA optimization completed in {(end_opt_time - start_opt_time)/60:.2f} minutes")

    # --- Display Top 5 Results Based on Sharpe Ratio ---
    top_n = 5
    top_results = results_df.sort_values(by='Sharpe Ratio', ascending=False).head(top_n)
    print(f"\nTop {top_n} EMA Combinations Based on Sharpe Ratio:")
    print(top_results[['Short EMA', 'Long EMA', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Total Return (%)']])

    # --- Plot Heatmap of Sharpe Ratios ---
    try:
        import seaborn as sns

        # Pivot the DataFrame for heatmap
        pivot_table = results_df.pivot(index='Short EMA', columns='Long EMA', values='Sharpe Ratio')

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title('Sharpe Ratio Heatmap for EMA Period Combinations')
        plt.xlabel('Long EMA Period')
        plt.ylabel('Short EMA Period')
        plt.tight_layout()
        plt.show()
    except ImportError:
        logger.warning("Seaborn is not installed. Skipping heatmap visualization.")

    # --- Run Backtest with Best EMA Pair and Plot Equity Curves ---
    best_short, best_long = best_pair
    logger.info(f"Running backtest with Best EMA Pair: Short EMA = {best_short}, Long EMA = {best_long}")
    backtest_best = run_backtest(df_daily_bt.copy(), best_short, best_long)
    balance_series_best = backtest_best["balance_series"]
    benchmark_equity_curve_best = backtest_best["benchmark_equity_curve"]

    # --- Plot Equity Curves ---
    plt.figure(figsize=(14, 7))

    # Plot strategy's equity curve
    plt.plot(balance_series_best.index, balance_series_best.values, label='Strategy Equity Curve', color='blue')

    # Plot scaled benchmark equity curve (Buy-and-Hold)
    plt.plot(df_daily_bt.index, benchmark_equity_curve_best, label='Buy-and-Hold Benchmark (ES Future)', color='orange')

    # Add titles and labels
    plt.title(f'Equity Curve Comparison: Strategy vs. Buy-and-Hold Benchmark (Best EMA Pair: {best_short}/{best_long})')
    plt.xlabel('Date')
    plt.ylabel('Account Balance ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Display the plot
    plt.show()

    # --- Save Optimization Results to CSV (Optional) ---
    results_df.to_csv('EMA_Optimization_Results.csv', index=False)
    logger.info("Optimization results saved to 'EMA_Optimization_Results.csv'")