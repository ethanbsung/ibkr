import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pytz
import logging
import sys
import gc
from ib_insync import *

# --- Configuration Parameters ---
IB_HOST = '127.0.0.1'        # IBKR Gateway/TWS host
IB_PORT = 7497               # IBKR Gateway/TWS paper trading port
CLIENT_ID = 2                # Unique client ID (ensure it's different from other scripts)
EXEC_SYMBOL = 'MES'           # Micro E-mini S&P 500 for execution
EXEC_EXPIRY = '202503'        # March 2025 (format: YYYYMM)
EXEC_EXCHANGE = 'CME'         # Exchange for MES
CURRENCY = 'USD'

INITIAL_CASH = 5000          # Starting cash
POSITION_SIZE = 1            # Number of MES contracts per trade
CONTRACT_MULTIPLIER = 5      # Contract multiplier for MES

BOLLINGER_PERIOD = 15
BOLLINGER_STDDEV = 2
STOP_LOSS_DISTANCE = 5        # Points away from entry
TAKE_PROFIT_DISTANCE = 10     # Points away from entry

# --- Setup Logging ---
logging.basicConfig(
    level=logging.WARNING,  # Set to WARNING to reduce logging verbosity during backtest
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# --- Helper Function to Filter RTH ---
def filter_rth(df):
    """
    Filters the DataFrame to include only Regular Trading Hours (09:30 - 16:00 ET) on weekdays.

    Parameters:
        df (pd.DataFrame): The input DataFrame with a timezone-aware datetime index.

    Returns:
        pd.DataFrame: The filtered DataFrame containing only RTH data.
    """
    # Define US/Eastern timezone
    eastern = pytz.timezone('US/Eastern')

    # Ensure the index is timezone-aware
    if df.index.tz is None:
        # Assume the data is in US/Eastern if no timezone is set
        df = df.tz_localize(eastern)
        logger.debug("Localized naive datetime index to US/Eastern.")
    else:
        # Convert to US/Eastern to standardize
        df = df.tz_convert(eastern)
        logger.debug("Converted timezone-aware datetime index to US/Eastern.")

    # Filter for weekdays (Monday=0 to Friday=4)
    df_eastern = df[df.index.weekday < 5]

    # Filter for RTH hours: 09:30 to 16:00
    df_rth = df_eastern.between_time('09:30', '16:00')

    # Convert back to UTC for consistency in further processing
    df_rth = df_rth.tz_convert('UTC')

    return df_rth

# --- Function to Create IBKR Contract ---
def create_ib_contract(symbol, exchange, currency, expiry):
    """
    Creates an IBKR contract for the specified symbol.

    Parameters:
        symbol (str): The ticker symbol (e.g., 'MES').
        exchange (str): The exchange (e.g., 'CME').
        currency (str): The currency (e.g., 'USD').
        expiry (str): The contract expiry in YYYYMM format (e.g., '202503').

    Returns:
        Contract: An IBKR Contract object.
    """
    contract = Future(symbol=symbol, exchange=exchange, currency=currency, lastTradeDateOrContractMonth=expiry)
    return contract

# --- Function to Fetch Historical Data ---
def fetch_historical_data(ib, contract, durationStr, barSizeSetting, whatToShow='TRADES'):
    """
    Fetches historical data for a given contract.

    Parameters:
        ib (IB): An instance of the IB connection.
        contract (Contract): The IBKR contract.
        durationStr (str): Duration string (e.g., '3 M').
        barSizeSetting (str): Bar size (e.g., '1 min', '30 mins').
        whatToShow (str): Data type to fetch (e.g., 'TRADES').

    Returns:
        pd.DataFrame: Historical data as a pandas DataFrame.
    """
    logger.info(f"Requesting historical data: {barSizeSetting}, Duration: {durationStr}")
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=durationStr,
            barSizeSetting=barSizeSetting,
            whatToShow=whatToShow,
            useRTH=False,  # Include extended hours
            formatDate=1,
            keepUpToDate=False
        )
        if not bars:
            logger.error(f"No data returned for {barSizeSetting} bars.")
            sys.exit(1)
        df = util.df(bars)
        df['datetime'] = pd.to_datetime(df['date']).dt.tz_localize('UTC')
        df.set_index('datetime', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.rename(columns={'close': 'close'}, inplace=True)  # Ensure consistent naming
        logger.info(f"Fetched {len(df)} bars for {barSizeSetting} data.")
        return df
    except Exception as e:
        logger.error(f"An error occurred while fetching historical data: {e}")
        sys.exit(1)

# --- Vectorized Exit Evaluation Function ---
def evaluate_exit_vectorized(position_type, entry_price, stop_loss, take_profit, df_high_freq, entry_time):
    """
    Determines the exit price and time using vectorized operations.

    Parameters:
        position_type (str): 'long' or 'short'
        entry_price (float): Entry price
        stop_loss (float): Stop-loss price
        take_profit (float): Take-profit price
        df_high_freq (pd.DataFrame): 1-minute high-frequency data
        entry_time (pd.Timestamp): Timestamp of entry

    Returns:
        tuple: (exit_price, exit_time, hit_take_profit) or (None, None, None)
    """
    # Slice the high-frequency data from entry_time onwards
    df_period = df_high_freq.loc[entry_time:]

    if df_period.empty:
        return None, None, None

    # Define a valid maximum timestamp within pandas' supported range
    max_valid_timestamp = pd.Timestamp('2262-04-11 23:47:16.854775807', tz='UTC')

    if position_type == 'long':
        # Identify bars where stop-loss or take-profit is hit
        hit_sl = df_period[df_period['low'] <= stop_loss]
        hit_tp = df_period[df_period['high'] >= take_profit]

        # Get the first occurrence of each
        first_sl = hit_sl.index.min() if not hit_sl.empty else max_valid_timestamp
        first_tp = hit_tp.index.min() if not hit_tp.empty else max_valid_timestamp

        # Determine which condition was hit first
        if first_sl < first_tp:
            return stop_loss, first_sl, False  # Stop-loss hit first
        elif first_tp < first_sl:
            return take_profit, first_tp, True   # Take-profit hit first
        elif first_sl == first_tp and first_sl != max_valid_timestamp:
            # Both conditions hit in the same bar; determine based on open price
            row = df_period.loc[first_sl]
            if row['open'] <= stop_loss:
                return stop_loss, first_sl, False
            else:
                return take_profit, first_sl, True

    elif position_type == 'short':
        # Identify bars where stop-loss or take-profit is hit
        hit_sl = df_period[df_period['high'] >= stop_loss]
        hit_tp = df_period[df_period['low'] <= take_profit]

        # Get the first occurrence of each
        first_sl = hit_sl.index.min() if not hit_sl.empty else max_valid_timestamp
        first_tp = hit_tp.index.min() if not hit_tp.empty else max_valid_timestamp

        # Determine which condition was hit first
        if first_sl < first_tp:
            return stop_loss, first_sl, False  # Stop-loss hit first
        elif first_tp < first_sl:
            return take_profit, first_tp, True   # Take-profit hit first
        elif first_sl == first_tp and first_sl != max_valid_timestamp:
            # Both conditions hit in the same bar; determine based on open price
            row = df_period.loc[first_sl]
            if row['open'] >= stop_loss:
                return stop_loss, first_sl, False
            else:
                return take_profit, first_sl, True

    return None, None, None

# --- Load Datasets from IBKR ---
def load_data_from_ibkr():
    """
    Connects to IBKR API, fetches historical data for 1-minute and 30-minute bars,
    and returns the corresponding DataFrames.

    Returns:
        tuple: (df_1m, df_30m_full)
    """
    # Initialize IBKR Connection
    logger.info("Initializing IBKR connection...")
    ib = IB()
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
        logger.info("Connected to IBKR successfully.")
    except Exception as e:
        logger.error(f"Failed to connect to IBKR: {e}")
        sys.exit(1)

    # Create IBKR Contract
    contract = create_ib_contract(EXEC_SYMBOL, EXEC_EXCHANGE, CURRENCY, EXEC_EXPIRY)

    # Fetch 1-Minute Data
    duration_1m = '3 M'    # Last 3 months
    bar_size_1m = '1 min'
    df_1m = fetch_historical_data(ib, contract, duration_1m, bar_size_1m)

    # Fetch 30-Minute Data
    duration_30m = '3 M'   # Last 3 months
    bar_size_30m = '30 mins'
    df_30m_full = fetch_historical_data(ib, contract, duration_30m, bar_size_30m)

    # Disconnect IBKR
    ib.disconnect()
    logger.info("Disconnected from IBKR.")

    return df_1m, df_30m_full

# --- Main Execution ---
def main():
    # --- Load Datasets ---
    logger.info("Loading datasets from IBKR API...")
    df_1m, df_30m_full = load_data_from_ibkr()
    logger.info("Datasets loaded successfully.")

    # --- Localize df_1m to US/Eastern and Convert to UTC ---
    eastern = pytz.timezone('US/Eastern')
    if df_1m.index.tz is None:
        df_1m = df_1m.tz_localize(eastern).tz_convert('UTC')
        logger.debug("Localized 1-Minute data to US/Eastern and converted to UTC.")
    else:
        df_1m = df_1m.tz_convert('UTC')
        logger.debug("Converted 1-Minute data to UTC.")

    # --- Localize df_30m_full to US/Eastern and Convert to UTC ---
    if df_30m_full.index.tz is None:
        df_30m_full = df_30m_full.tz_localize(eastern).tz_convert('UTC')
        logger.debug("Localized 30-Minute data to US/Eastern and converted to UTC.")
    else:
        df_30m_full = df_30m_full.tz_convert('UTC')
        logger.debug("Converted 30-Minute data to UTC.")

    # --- Verify Data Ranges ---
    logger.info(f"1-Minute Data Range: {df_1m.index.min()} to {df_1m.index.max()}")
    logger.info(f"30-Minute Data Range: {df_30m_full.index.min()} to {df_30m_full.index.max()}")

    # --- Define Backtest Period ---
    # Option 1: Custom Backtest Period (Replace These Dates as needed)
    custom_start_date = "2024-06-29"  # Adjust based on your data
    custom_end_date = "2024-12-24"    # Adjust based on your data

    # Option 2: Use Full Available Data (if custom dates are not set)
    if custom_start_date and custom_end_date:
        try:
            start_time = pd.to_datetime(custom_start_date).tz_localize('UTC')
            end_time = pd.to_datetime(custom_end_date).tz_localize('UTC')
        except Exception as e:
            logger.error(f"Error parsing custom dates: {e}")
            sys.exit(1)
    else:
        start_time = df_30m_full.index.min()
        end_time = df_30m_full.index.max()

    logger.info(f"Backtest Period: {start_time} to {end_time}")

    # --- Slice Dataframes to Backtest Period ---
    logger.info(f"Slicing data from {start_time} to {end_time}...")
    try:
        df_1m = df_1m.loc[start_time:end_time]
        logger.info(f"Sliced 1-Minute Data: {len(df_1m)} data points")
    except KeyError:
        logger.warning("No data found for the specified 1-minute backtest period.")
        df_1m = pd.DataFrame(columns=df_1m.columns)

    try:
        df_30m_full = df_30m_full.loc[start_time:end_time]
        logger.info(f"Sliced 30-Minute Full Data: {len(df_30m_full)} data points")
    except KeyError:
        logger.warning("No data found for the specified 30-minute backtest period.")
        df_30m_full = pd.DataFrame(columns=df_30m_full.columns)
    logger.info("Data sliced to backtest period.")

    # --- Check if df_30m_full is Empty After Slicing ---
    if df_30m_full.empty:
        logger.error("No 30-minute data available for the specified backtest period. Exiting.")
        sys.exit(1)

    # --- Calculate Bollinger Bands on Full 30m Data (Including Extended Hours) ---
    logger.info("Calculating Bollinger Bands on full 30-minute data (including extended hours)...")
    df_30m_full['ma'] = df_30m_full['close'].rolling(window=BOLLINGER_PERIOD, min_periods=BOLLINGER_PERIOD).mean()
    df_30m_full['std'] = df_30m_full['close'].rolling(window=BOLLINGER_PERIOD, min_periods=BOLLINGER_PERIOD).std()
    df_30m_full['upper_band'] = df_30m_full['ma'] + (BOLLINGER_STDDEV * df_30m_full['std'])
    df_30m_full['lower_band'] = df_30m_full['ma'] - (BOLLINGER_STDDEV * df_30m_full['std'])

    # Check for NaNs after rolling calculations
    num_ma_nans = df_30m_full['ma'].isna().sum()
    num_std_nans = df_30m_full['std'].isna().sum()
    logger.info(f"'ma' column has {num_ma_nans} NaN values after rolling calculation.")
    logger.info(f"'std' column has {num_std_nans} NaN values after rolling calculation.")

    # Drop rows with NaN in any of the Bollinger Bands columns
    df_30m_full.dropna(subset=['ma', 'std', 'upper_band', 'lower_band'], inplace=True)
    logger.info(f"After Bollinger Bands calculation, 30-Minute Full Data Points: {len(df_30m_full)}")

    # --- Check if df_30m_full is Empty After Bollinger Bands ---
    if df_30m_full.empty:
        logger.error("All 30-minute data points were removed after Bollinger Bands calculation. Exiting.")
        sys.exit(1)

    # --- Filter RTH Data Separately for Trade Execution ---
    logger.info("Applying RTH filter to 30-minute data for trade execution...")
    df_30m_rth = filter_rth(df_30m_full)
    logger.info(f"30-Minute RTH Data Points after Filtering: {len(df_30m_rth)}")

    if df_30m_rth.empty:
        logger.warning("No 30-minute RTH data points after filtering. Exiting backtest.")
        sys.exit(1)

    logger.info("RTH filtering applied to 30-minute data for trade execution.")

    # --- Confirm Data Points After Filtering ---
    print(f"\nBacktesting from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    print(f"1-Minute Data Points after Filtering: {len(df_1m)}")
    print(f"30-Minute Full Data Points after Slicing: {len(df_30m_full)}")
    print(f"30-Minute RTH Data Points after RTH Filtering: {len(df_30m_rth)}")

    # --- Initialize Backtest Variables ---
    position_size = 0
    entry_price = None
    position_type = None  
    cash = INITIAL_CASH
    trade_results = []
    balance_series = []  # Initialize as an empty list to store balance at each bar
    exposure_bars = 0

    # --- Ensure df_high_freq is sorted by index ---
    df_high_freq = df_1m.sort_index()

    # --- Convert 'close', 'open', 'high', 'low' to float32 for memory efficiency ---
    df_high_freq = df_high_freq.astype({
        'open': 'float32',
        'high': 'float32',
        'low': 'float32',
        'close': 'float32',
        'volume': 'float32',
        'average': 'float32',
        'barCount': 'int32'
    })

    # --- Backtesting Loop ---
    logger.info("Starting backtesting loop...")
    for i, current_time in enumerate(df_30m_rth.index):
        current_bar = df_30m_rth.iloc[i]
        current_price = current_bar['close']

        # Count exposure when position is active
        if position_size != 0:
            exposure_bars += 1

        if position_size == 0:
            # No open position, check for entry signals based on RTH 30m bar
            upper_band = df_30m_full.loc[current_time, 'upper_band']
            lower_band = df_30m_full.loc[current_time, 'lower_band']

            if current_price < lower_band:
                # Enter Long
                position_size = POSITION_SIZE
                entry_price = current_price
                position_type = 'long'
                stop_loss_price = entry_price - STOP_LOSS_DISTANCE
                take_profit_price = entry_price + TAKE_PROFIT_DISTANCE
                entry_time = current_time
                logger.debug(f"Entered LONG at {entry_price} on {entry_time} UTC")

            elif current_price > upper_band:
                # Enter Short
                position_size = POSITION_SIZE
                entry_price = current_price
                position_type = 'short'
                stop_loss_price = entry_price + STOP_LOSS_DISTANCE
                take_profit_price = entry_price - TAKE_PROFIT_DISTANCE
                entry_time = current_time
                logger.debug(f"Entered SHORT at {entry_price} on {entry_time} UTC")

        else:
            # Position is open, check for exit conditions using vectorized function
            exit_price, exit_time, hit_tp = evaluate_exit_vectorized(
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
                    pnl = ((exit_price - entry_price) * CONTRACT_MULTIPLIER * position_size) - (0.62 * 2)  # Example: Spread cost
                elif position_type == 'short':
                    pnl = ((entry_price - exit_price) * CONTRACT_MULTIPLIER * position_size) - (0.62 * 2)

                trade_results.append(pnl)
                cash += pnl

                # Log trade exit details at DEBUG level
                exit_type = "TAKE PROFIT" if hit_tp else "STOP LOSS"
                logger.debug(f"Exited {position_type.upper()} at {exit_price} on {exit_time} UTC via {exit_type} for P&L: ${pnl:.2f}")

                # Reset position variables
                position_size = 0
                position_type = None
                entry_price = None
                stop_loss_price = None
                take_profit_price = None
                entry_time = None

        # Append current cash to balance_series at each bar
        balance_series.append(cash)

        # Optional: Progress Indicator every 100,000 bars
        if (i + 1) % 100000 == 0:
            logger.info(f"Processed {i + 1} out of {len(df_30m_rth)} 30-minute bars.")

    # --- Free Memory ---
    del df_high_freq
    gc.collect()

    logger.info("Backtesting loop completed.")

    # --- Post-Backtest Calculations ---
    
    # Convert balance_series to a Pandas Series with appropriate index
    balance_series = pd.Series(balance_series, index=df_30m_rth.index)

    # --- Calculate Performance Metrics ---
    
    # Total Return
    total_return_percentage = ((cash - INITIAL_CASH) / INITIAL_CASH) * 100

    # Calculate trading days
    trading_days = max((df_30m_full.index.max() - df_30m_full.index.min()).days, 1)

    # Annualized Return
    annualized_return_percentage = ((cash / INITIAL_CASH) ** (252 / trading_days) - 1) * 100

    # Benchmark Return (Buy & Hold)
    benchmark_return = ((df_30m_full['close'].iloc[-1] - df_30m_full['close'].iloc[0]) / df_30m_full['close'].iloc[0]) * 100

    # Equity Peak
    equity_peak = balance_series.max()

    # Calculate Daily Returns
    daily_equity = balance_series.resample('D').ffill()
    daily_returns = daily_equity.pct_change().dropna()

    # Volatility (Annual)
    volatility_annual = daily_returns.std() * np.sqrt(252) * 100

    # Sharpe Ratio
    risk_free_rate = 0  # Example: 0 for simplicity or use the current rate
    sharpe_ratio = ((daily_returns.mean() - risk_free_rate) / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0

    # Sortino Ratio
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

    sortino_ratio = calculate_sortino_ratio(daily_returns)

    # Drawdown Calculations
    running_max_series = balance_series.cummax()
    drawdowns = (balance_series - running_max_series) / running_max_series
    max_drawdown = drawdowns.min() * 100
    average_drawdown = drawdowns[drawdowns < 0].mean() * 100 if not drawdowns[drawdowns < 0].empty else 0

    # Exposure Time
    exposure_time_percentage = (exposure_bars / len(df_30m_rth)) * 100 if len(df_30m_rth) > 0 else 0

    # Profit Factor
    winning_trades = [pnl for pnl in trade_results if pnl > 0]
    losing_trades = [pnl for pnl in trade_results if pnl <= 0]
    profit_factor = (sum(winning_trades) / abs(sum(losing_trades))) if losing_trades else float('inf')

    # Calmar Ratio
    calmar_ratio = (total_return_percentage / abs(max_drawdown)) if max_drawdown != 0 else float('inf')

    # Drawdown Duration Calculations
    drawdown_periods = drawdowns[drawdowns < 0]
    if not drawdown_periods.empty:
        # Identify contiguous segments of drawdown
        # Create a boolean series where True indicates a drawdown
        is_drawdown = drawdowns < 0

        # Identify start and end of each drawdown period
        drawdown_changes = is_drawdown.ne(is_drawdown.shift())
        drawdown_groups = drawdown_changes.cumsum()

        # Group by drawdown periods
        drawdown_groups = is_drawdown.groupby(drawdown_groups)

        # Calculate durations for each drawdown period
        drawdown_durations = []
        for name, group in drawdown_groups:
            if group.iloc[0]:  # Only consider groups where drawdown is True
                duration = (group.index[-1] - group.index[0]).total_seconds() / 86400  # Duration in days
                drawdown_durations.append(duration)

        if drawdown_durations:
            max_drawdown_duration_days = max(drawdown_durations)
            average_drawdown_duration_days = np.mean(drawdown_durations)
        else:
            max_drawdown_duration_days = 0
            average_drawdown_duration_days = 0
    else:
        max_drawdown_duration_days = 0
        average_drawdown_duration_days = 0

    # --- Results Summary ---
    print("\nPerformance Summary:")
    results = {
        "Start Date": df_30m_full.index.min().strftime("%Y-%m-%d"),
        "End Date": df_30m_full.index.max().strftime("%Y-%m-%d"),
        "Exposure Time": f"{exposure_time_percentage:.2f}%",
        "Final Account Balance": f"${cash:,.2f}",
        "Equity Peak": f"${equity_peak:,.2f}",
        "Total Return": f"{total_return_percentage:.2f}%",
        "Annualized Return": f"{annualized_return_percentage:.2f}%",
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

    # --- Plot Equity Curves ---
    if len(balance_series) < 2:
        logger.warning("Not enough data points to plot equity curves.")
    else:
        # Create benchmark equity curve
        initial_close = df_30m_full['close'].iloc[0]
        benchmark_equity = (df_30m_full['close'] / initial_close) * INITIAL_CASH

        # Align the benchmark to the strategy's balance_series
        benchmark_equity = benchmark_equity.reindex(balance_series.index, method='ffill')

        # Verify benchmark_equity alignment
        logger.info(f"Benchmark Equity Range: {benchmark_equity.index.min()} to {benchmark_equity.index.max()}")

        # Ensure no NaNs in benchmark_equity
        num_benchmark_nans = benchmark_equity.isna().sum()
        if num_benchmark_nans > 0:
            logger.warning(f"Benchmark equity has {num_benchmark_nans} NaN values. Filling with forward fill.")
            benchmark_equity = benchmark_equity.fillna(method='ffill')

        # Create a DataFrame for plotting
        equity_df = pd.DataFrame({
            'Strategy': balance_series,
            'Benchmark': benchmark_equity
        })

        # Plotting
        plt.figure(figsize=(14, 7))
        plt.plot(equity_df['Strategy'], label='Strategy Equity')
        plt.plot(equity_df['Benchmark'], label='Benchmark Equity')
        plt.title('Equity Curve: Strategy vs Benchmark')
        plt.xlabel('Time')
        plt.ylabel('Account Balance ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # --- Optional: Save Trade Results to CSV ---
    # Uncomment the following lines if you wish to save trade results for further analysis
    # trade_df = pd.DataFrame(trade_results, columns=['PnL'])
    # trade_df.to_csv('trade_results.csv', index=False)
    # logger.info("Trade results saved to 'trade_results.csv'.")

    # --- Optional: Save Balance Series to CSV ---
    # balance_series.to_csv('balance_series.csv')
    # logger.info("Balance series saved to 'balance_series.csv'.")

if __name__ == "__main__":
    main()