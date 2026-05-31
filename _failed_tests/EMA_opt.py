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
CSV_FILE = 'Data/es_30m_data.csv'

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
    Loads 30-minute futures data into a pandas DataFrame with appropriate data types and datetime parsing.

    Parameters:
        csv_file (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded and indexed DataFrame.
    """
    try:
        logger.info(f"Reading CSV file: {csv_file}")
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
            parse_dates=['Time'],  # Parse 'Time' as datetime
            infer_datetime_format=True
            # Removed 'date_format' as it's not a valid parameter
        )

        # Rename columns to match the script's expectations
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Last': 'close',
            'Volume': 'volume'
            # 'Open Int' is optional and ignored in calculations
        }, inplace=True)

        # Verify that 'Time' is parsed correctly
        if df['Time'].isnull().any():
            logger.error("Some 'Time' entries could not be parsed as datetime.")
            sys.exit(1)

        # Sort by Time to ensure chronological order
        df.sort_values('Time', inplace=True)
        df.set_index('Time', inplace=True)

        # Verify that the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error(f"The 'Time' column was not parsed as datetime. Check the CSV format.")
            sys.exit(1)

        # Localize to UTC directly
        if df.index.tz is None:
            df = df.tz_localize('UTC')
            logger.debug(f"Localized naive datetime index to UTC for {csv_file}.")
        else:
            df = df.tz_convert('UTC')
            logger.debug(f"Converted timezone-aware datetime index to UTC for {csv_file}.")

        # Drop unnecessary columns if they exist
        columns_to_keep = ['open', 'high', 'low', 'close', 'volume']
        existing_columns = [col for col in columns_to_keep if col in df.columns]
        df = df[existing_columns]

        # Remove duplicate timestamps by keeping the last entry (most liquid contract)
        initial_length = len(df)
        df = df[~df.index.duplicated(keep='last')]
        duplicates_removed = initial_length - len(df)
        if duplicates_removed > 0:
            logger.debug(f"Removed {duplicates_removed} duplicate timestamps for {csv_file}.")

        logger.info(f"Successfully loaded {len(df)} rows from {csv_file}.")
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
    logger.debug(f"Calculated short EMA ({short_period}) and long EMA ({long_period}).")
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

    # Initialize 'crossover' column
    df['crossover'] = 0

    # Bullish crossover: Short EMA crosses above Long EMA
    bullish = (df[short_ema] > df[long_ema]) & (df[short_ema].shift(1) <= df[long_ema].shift(1))
    df.loc[bullish, 'crossover'] = 1

    # Bearish crossover: Short EMA crosses below Long EMA
    bearish = (df[short_ema] < df[long_ema]) & (df[short_ema].shift(1) >= df[long_ema].shift(1))
    df.loc[bearish, 'crossover'] = -1

    crossover_count = df['crossover'].abs().sum()
    logger.debug(f"Detected {crossover_count} crossover events.")
    return df

def calculate_rsi(df, period=14):
    """
    Calculates the Relative Strength Index (RSI) for the given DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'close' prices.
        period (int): Lookback period for RSI calculation.

    Returns:
        pd.DataFrame: DataFrame with an additional 'rsi' column.
    """
    delta = df['close'].diff()

    # Separate positive and negative changes
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the average gain and loss
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    # Calculate RS (Relative Strength) and RSI
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    return df

def calculate_atr(df, period=14):
    """
    Calculates the Average True Range (ATR) for the given DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' prices.
        period (int): Lookback period for ATR calculation.

    Returns:
        pd.DataFrame: DataFrame with an additional 'atr' column.
    """
    df['tr'] = np.maximum(df['high'] - df['low'], 
                          np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                     abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(window=period).mean()
    return df

def backtest_strategy(df, short_period, long_period, rsi_thresholds=(30, 70), atr_threshold=None):
    """
    Backtests the EMA crossover strategy with optional RSI and ATR filters.

    Parameters:
        df (pd.DataFrame): DataFrame containing price data and indicators.
        short_period (int): Period for short EMA.
        long_period (int): Period for long EMA.
        rsi_thresholds (tuple): (Oversold threshold, Overbought threshold).
        atr_threshold (float or None): ATR threshold to define balanced regimes. If None, no ATR filter is applied.

    Returns:
        dict: Performance metrics including Sharpe Ratio.
    """
    # Calculate EMAs
    df = calculate_emas(df, short_period, long_period)

    # Detect crossovers
    df = detect_crossovers(df, short_period, long_period)

    # Calculate RSI
    df = calculate_rsi(df, period=14)

    # Calculate ATR
    df = calculate_atr(df, period=14)

    # Drop rows with NaN values
    df.dropna(inplace=True)

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

    # Iterate through each bar in the DataFrame
    for i in range(len(df)):
        current_bar = df.iloc[i]
        current_time = df.index[i]
        current_price = current_bar['close']
        current_rsi = current_bar['rsi']
        current_atr = current_bar['atr']

        # Optional ATR Filter: Skip trading during balanced regimes
        if atr_threshold is not None:
            if current_atr < atr_threshold:
                # Balanced regime detected; skip trading
                balance_series.append(cash)
                balance_times.append(current_time)
                continue

        # Count exposure when position is active
        if position_size != 0:
            exposure_days += 1

        if position_size == 0:
            # No open position, check for entry signals based on EMA crossover and RSI
            if current_bar['crossover'] == 1 and current_rsi < rsi_thresholds[0]:
                # Bullish Crossover: Enter Long
                position_size = POSITION_SIZE
                entry_price = current_price
                position_type = 'long'
                entry_time = current_time
                logger.debug(f"Entered LONG at {entry_price} on {entry_time.strftime('%Y-%m-%d %H:%M')} UTC")
        else:
            # Position is open, check for exit signal based on EMA crossover and RSI
            if current_bar['crossover'] == -1 and current_rsi > rsi_thresholds[1]:
                # Bearish Crossover: Exit Long
                exit_price = current_price
                pnl = (exit_price - entry_price) * CONTRACT_MULTIPLIER * position_size - COMMISSION
                trade_results.append(pnl)
                cash += pnl
                balance_series.append(cash)
                balance_times.append(current_time)
                logger.debug(f"Exited LONG at {exit_price} on {current_time.strftime('%Y-%m-%d %H:%M')} UTC for P&L: ${pnl:.2f}")

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
        logger.debug("Position still open at the end of the backtest. Closing at last available price.")
        exit_price = df.iloc[-1]['close']
        exit_time = df.index[-1]
        pnl = (exit_price - entry_price) * CONTRACT_MULTIPLIER * position_size - COMMISSION
        trade_results.append(pnl)
        cash += pnl
        balance_series.append(cash)
        balance_times.append(exit_time)
        logger.debug(f"Exited LONG at {exit_price} on {exit_time.strftime('%Y-%m-%d %H:%M')} UTC via END OF DATA for P&L: ${pnl:.2f}")

    # Convert balance_series to a Pandas Series with appropriate index
    balance_series = pd.Series(balance_series, index=balance_times)

    # Calculate Benchmark Equity Curve (Buy-and-Hold)
    initial_price = df['close'].iloc[0]  # First closing price in the backtest period
    benchmark_equity_curve = (df['close'] / initial_price) * INITIAL_CASH

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

    # Drawdown Calculations
    running_max_series = balance_series.cummax()
    drawdowns = (balance_series - running_max_series) / running_max_series
    max_drawdown = drawdowns.min() * 100
    average_drawdown = drawdowns[drawdowns < 0].mean() * 100

    # Exposure Time
    exposure_time_percentage = (exposure_days / len(df)) * 100

    # Profit Factor
    winning_trades = [pnl for pnl in trade_results if pnl > 0]
    losing_trades = [pnl for pnl in trade_results if pnl <= 0]
    profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf')

    # Calmar Ratio Calculation
    calmar_ratio = (total_return_percentage / abs(max_drawdown)) if max_drawdown != 0 else float('inf')

    # Drawdown Duration Calculations
    in_drawdown = False
    drawdown_start = None
    drawdown_durations = []

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
                duration = (drawdown_end - drawdown_start).days
                drawdown_durations.append(duration)

    # Handle if still in drawdown at the end of the data
    if in_drawdown:
        drawdown_end = balance_series.index[-1]
        duration = (drawdown_end - drawdown_start).days
        drawdown_durations.append(duration)

    if drawdown_durations:
        max_drawdown_duration_days = max(drawdown_durations)
        average_drawdown_duration_days = np.mean(drawdown_durations)
    else:
        max_drawdown_duration_days = 0
        average_drawdown_duration_days = 0

    # Compile Performance Metrics
    performance = {
        "Final Account Balance": cash,
        "Equity Peak": equity_peak,
        "Total Return (%)": total_return_percentage,
        "Annualized Return (%)": annualized_return_percentage * 100,
        "Benchmark Return (%)": benchmark_return,
        "Volatility (Annual %)": volatility_annual,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown,
        "Average Drawdown (%)": average_drawdown,
        "Calmar Ratio": calmar_ratio,
        "Exposure Time (%)": exposure_time_percentage,
        "Profit Factor": profit_factor,
        "Total Trades": len(trade_results),
        "Winning Trades": len(winning_trades),
        "Losing Trades": len(losing_trades),
        "Win Rate (%)": (len(winning_trades)/len(trade_results)*100) if trade_results else 0,
        "Max Drawdown Duration (days)": max_drawdown_duration_days,
        "Average Drawdown Duration (days)": average_drawdown_duration_days
    }

    return performance, balance_series, benchmark_equity_curve

def optimize_parameters(in_sample_df, short_periods, long_periods, rsi_thresholds=(30, 70), atr_threshold=None):
    """
    Optimizes EMA periods to maximize Sharpe Ratio on in-sample data.

    Parameters:
        in_sample_df (pd.DataFrame): In-sample DataFrame.
        short_periods (list): List of short EMA periods to test.
        long_periods (list): List of long EMA periods to test.
        rsi_thresholds (tuple): (Oversold threshold, Overbought threshold).
        atr_threshold (float or None): ATR threshold to define balanced regimes.

    Returns:
        tuple: (Best parameters as a dict, DataFrame with all parameter performances)
    """
    results = []
    parameter_combinations = list(product(short_periods, long_periods))
    total_combinations = len(parameter_combinations)
    logger.info(f"Starting parameter optimization over {total_combinations} combinations...")

    for idx, (short, long) in enumerate(parameter_combinations, 1):
        if short >= long:
            # Ensure short period is less than long period
            continue
        logger.debug(f"Testing combination {idx}/{total_combinations}: SHORT_EMA={short}, LONG_EMA={long}")
        performance, _, _ = backtest_strategy(
            in_sample_df.copy(),
            short_period=short,
            long_period=long,
            rsi_thresholds=rsi_thresholds,
            atr_threshold=atr_threshold
        )
        results.append({
            'SHORT_EMA': short,
            'LONG_EMA': long,
            'Sharpe Ratio': performance['Sharpe Ratio'],
            'Total Return (%)': performance['Total Return (%)']
        })

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df['Sharpe Ratio'].idxmax()]
    best_parameters = {
        'SHORT_EMA': int(best_row['SHORT_EMA']),
        'LONG_EMA': int(best_row['LONG_EMA'])
    }

    logger.info(f"Optimization completed. Best Sharpe Ratio: {best_row['Sharpe Ratio']:.2f} with SHORT_EMA={best_parameters['SHORT_EMA']} and LONG_EMA={best_parameters['LONG_EMA']}.")

    return best_parameters, results_df

# --- Main Execution ---

if __name__ == "__main__":
    # Load the full dataset
    logger.info("Loading the full dataset...")
    full_df = load_data(CSV_FILE)

    # Define the optimization and validation periods
    # Assuming the dataset covers at least 10 years
    # In-Sample: Last 2 years
    # Out-of-Sample: Previous 8 years

    end_date = full_df.index.max()
    start_date = end_date - pd.DateOffset(years=10)  # Adjust as per your data range

    # Slice the full data to the last 10 years
    full_df = full_df.loc[start_date:end_date]

    # Define in-sample and out-of-sample periods
    in_sample_start = end_date - pd.DateOffset(years=2)
    in_sample_end = end_date

    out_of_sample_start = start_date
    out_of_sample_end = in_sample_start - pd.Timedelta(minutes=1)  # Ensure no overlap

    logger.info(f"In-Sample Period: {in_sample_start.strftime('%Y-%m-%d')} to {in_sample_end.strftime('%Y-%m-%d')}")
    logger.info(f"Out-of-Sample Period: {out_of_sample_start.strftime('%Y-%m-%d')} to {out_of_sample_end.strftime('%Y-%m-%d')}")

    in_sample_df = full_df.loc[in_sample_start:in_sample_end]
    out_of_sample_df = full_df.loc[out_of_sample_start:out_of_sample_end]

    logger.info(f"In-Sample Data Points: {len(in_sample_df)}")
    logger.info(f"Out-of-Sample Data Points: {len(out_of_sample_df)}")

    # Define parameter ranges for optimization
    short_ema_range = range(10, 31, 5)  # 10, 15, 20, 25, 30
    long_ema_range = range(40, 61, 5)   # 40, 45, 50, 55, 60

    # Optional ATR Filter: Define ATR threshold or set to None
    atr_threshold = None  # e.g., 5.0

    # Optimize parameters on in-sample data
    best_params, optimization_results = optimize_parameters(
        in_sample_df,
        short_periods=short_ema_range,
        long_periods=long_ema_range,
        rsi_thresholds=(30, 70),
        atr_threshold=atr_threshold
    )

    # Display Optimization Results
    print("\nOptimization Results:")
    print(optimization_results.sort_values(by='Sharpe Ratio', ascending=False).head(10))

    # Apply the best parameters to out-of-sample data
    logger.info(f"Applying best parameters to Out-of-Sample data: SHORT_EMA={best_params['SHORT_EMA']}, LONG_EMA={best_params['LONG_EMA']}")
    performance_out, balance_out, benchmark_out = backtest_strategy(
        out_of_sample_df.copy(),
        short_period=best_params['SHORT_EMA'],
        long_period=best_params['LONG_EMA'],
        rsi_thresholds=(30, 70),
        atr_threshold=atr_threshold
    )

    # Display Out-of-Sample Performance
    print("\nOut-of-Sample Performance Summary:")
    for key, value in performance_out.items():
        if isinstance(value, float):
            print(f"{key:35}: {value:>15.2f}")
        else:
            print(f"{key:35}: {value:>15}")

    # --- Plot Equity Curves for Out-of-Sample ---
    plt.figure(figsize=(14, 7))

    # Plot strategy's equity curve
    plt.plot(balance_out.index, balance_out.values, label='Strategy Equity Curve', color='blue')

    # Plot scaled benchmark equity curve (Buy-and-Hold)
    plt.plot(benchmark_out.index, benchmark_out.values, label='Buy-and-Hold Benchmark (ES Future)', color='orange')

    # Add titles and labels
    plt.title('Equity Curve Comparison: Strategy vs. Buy-and-Hold Benchmark (Out-of-Sample)')
    plt.xlabel('Date')
    plt.ylabel('Account Balance ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Display the plot
    plt.show()