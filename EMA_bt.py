import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pytz
import logging
import sys
import time
import os

# --- Configuration Parameters ---
INITIAL_CASH = 5000          # Starting cash
POSITION_SIZE = 1            # Number of contracts per trade
CONTRACT_MULTIPLIER = 5      # Contract multiplier

SHORT_EMA = 19                # Short EMA period (19 minutes)
LONG_EMA = 240                # Long EMA period (4 hours)

STOP_LOSS_DISTANCE = 2        # Stop-loss in points
TAKE_PROFIT_DISTANCE = 7.5    # Take-profit in points

SPREAD_COST_PER_TRADE = 0.62 * 2  # Spread cost per trade (example value)

# --- Data Files ---
CSV_FILE_1M = 'es_1m_data.csv'
# CSV_FILE_5M = 'es_5m_data.csv'  # Removed as it's no longer needed
# CSV_FILE_1H = 'es_1h_data.csv'  # Removed as it's no longer needed

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
        # Assume the data is in UTC if no timezone is set
        df = df.tz_localize('UTC')
    else:
        # Convert to UTC to standardize
        df = df.tz_convert('UTC')
    
    # Convert index to US/Eastern timezone for filtering
    df_eastern = df.copy()
    df_eastern.index = df_eastern.index.tz_convert(eastern)
    
    # Filter for weekdays (Monday=0 to Friday=4)
    df_eastern = df_eastern[df_eastern.index.weekday < 5]
    
    # Filter for RTH hours: 09:30 to 16:00
    df_rth = df_eastern.between_time('09:30', '16:00')
    
    # Convert back to UTC for consistency in further processing
    df_rth.index = df_rth.index.tz_convert('UTC')
    
    return df_rth

def load_data(csv_file):
    """
    Loads CSV data into a pandas DataFrame with appropriate data types and datetime parsing.
    
    Parameters:
        csv_file (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded and indexed DataFrame.
    """
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
            date_format="%Y-%m-%d %H:%M:%S%z"  # Using 'date_format' instead of 'date_parser'
        )
        df.sort_values('date', inplace=True)
        df.set_index('date', inplace=True)
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {csv_file}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logger.error("No data: The CSV file is empty.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred while reading the CSV: {e}")
        sys.exit(1)

def calculate_emas(df, short_span, long_span):
    """
    Calculates short and long EMAs for the given DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'close' prices.
        short_span (int): Period for short EMA.
        long_span (int): Period for long EMA.
        
    Returns:
        pd.DataFrame: DataFrame with additional EMA columns.
    """
    df[f'short_ema_{short_span}'] = df['close'].ewm(span=short_span, adjust=False).mean()
    df[f'long_ema_{long_span}'] = df['close'].ewm(span=long_span, adjust=False).mean()
    return df

def detect_crossovers(df, short_span, long_span):
    """
    Detects EMA crossovers in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing EMAs.
        short_span (int): Period for short EMA.
        long_span (int): Period for long EMA.
        
    Returns:
        pd.DataFrame: DataFrame with 'crossover' column indicating crossover signals.
                      1 for bullish crossover, -1 for bearish crossover, 0 otherwise.
    """
    short_ema = f'short_ema_{short_span}'
    long_ema = f'long_ema_{long_span}'
    
    # Initialize 'crossover' column
    df['crossover'] = 0
    
    # Bullish crossover: Short EMA crosses above Long EMA
    bullish = (df[short_ema] > df[long_ema]) & (df[short_ema].shift(1) <= df[long_ema].shift(1))
    df.loc[bullish, 'crossover'] = 1
    
    # Bearish crossover: Short EMA crosses below Long EMA
    bearish = (df[short_ema] < df[long_ema]) & (df[short_ema].shift(1) >= df[long_ema].shift(1))
    df.loc[bearish, 'crossover'] = -1
    
    return df

def evaluate_exit(position_type, entry_price, stop_loss, take_profit, df_high_freq, entry_time):
    """
    Determines whether the stop-loss or take-profit is hit using high-frequency data.
    
    Parameters:
        position_type (str): 'long' or 'short'
        entry_price (float): Price at which the position was entered
        stop_loss (float): Stop-loss price
        take_profit (float): Take-profit price
        df_high_freq (pd.DataFrame): High-frequency DataFrame (1-minute)
        entry_time (pd.Timestamp): Timestamp when the position was entered
    
    Returns:
        tuple: (exit_price, exit_time, hit_take_profit)
    """
    # Slice the high-frequency data from entry_time onwards
    df_period = df_high_freq.loc[entry_time:]
    
    # Iterate through each high-frequency bar after entry_time
    for timestamp, row in df_period.iterrows():
        high = row['high']
        low = row['low']
        
        if position_type == 'long':
            if high >= take_profit and low <= stop_loss:
                # Determine which was hit first
                if row['open'] <= stop_loss:
                    return stop_loss, timestamp, False  # Stop loss hit first
                else:
                    return take_profit, timestamp, True   # Take profit hit first
            elif high >= take_profit:
                return take_profit, timestamp, True
            elif low <= stop_loss:
                return stop_loss, timestamp, False
        
        elif position_type == 'short':
            if low <= take_profit and high >= stop_loss:
                # Determine which was hit first
                if row['open'] >= stop_loss:
                    return stop_loss, timestamp, False  # Stop loss hit first
                else:
                    return take_profit, timestamp, True   # Take profit hit first
            elif low <= take_profit:
                return take_profit, timestamp, True
            elif high >= stop_loss:
                return stop_loss, timestamp, False
    
    # If neither condition is met, return None
    return None, None, None

# --- Load and Prepare Data ---

# --- Load Datasets ---
logger.info("Loading 1-minute dataset...")
df_1m = load_data(CSV_FILE_1M)
# df_5m = load_data(CSV_FILE_5M)  # Removed as it's no longer needed
# df_1h_full = load_data(CSV_FILE_1H)  # Removed as it's no longer needed
logger.info("1-minute dataset loaded successfully.")

# --- Define Backtest Period ---
# Option 1: Custom Backtest Period (Replace These Dates)
custom_start_date = "2022-09-25"
custom_end_date = "2024-12-11"

# Option 2: Use Full Available Data (if custom dates are not set)
if custom_start_date and custom_end_date:
    start_time = pd.to_datetime(custom_start_date, utc=True)
    end_time = pd.to_datetime(custom_end_date, utc=True)
else:
    start_time = pd.to_datetime(df_1m.index.min(), utc=True)
    end_time = pd.to_datetime(df_1m.index.max(), utc=True)

# --- Slice Dataframes to Backtest Period ---
logger.info(f"Slicing 1-minute data from {start_time} to {end_time}...")
df_1m = df_1m.loc[start_time:end_time]
logger.info("1-minute data sliced to backtest period.")

# --- Calculate EMAs on 1-Minute Data ---
logger.info("Calculating EMAs on 1-minute data...")
df_1m = calculate_emas(df_1m, SHORT_EMA, LONG_EMA)
df_1m.dropna(inplace=True)
logger.info("EMAs calculated on 1-minute data.")

# --- Detect Crossovers ---
logger.info("Detecting EMA crossovers...")
df_1m = detect_crossovers(df_1m, SHORT_EMA, LONG_EMA)
logger.info("EMA crossovers detected.")

# --- Filter RTH Data Separately for Trade Execution ---
logger.info("Applying RTH filter to 1-minute data for trade execution...")
df_1m_rth = filter_rth(df_1m)
logger.info("RTH filtering applied to 1-minute data for trade execution.")

# --- Confirm Data Points After Filtering ---
print(f"\nBacktesting Parameters:")
print(f"Backtesting Period      : {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
print(f"1-Minute Data Points    : {len(df_1m)}")
print(f"1-Minute RTH Data Points: {len(df_1m_rth)}")

# --- Initialize Backtest Variables ---
position_size = 0
entry_price = None
position_type = None  
cash = INITIAL_CASH
trade_results = []
balance_series = [INITIAL_CASH]  # Keep as a list
exposure_bars = 0

# For Drawdown Duration Calculations
in_drawdown = False
drawdown_start = None
drawdown_durations = []

# Define which high-frequency data to use
df_high_freq = df_1m_rth  # Use 1-minute data for both EMA and exit evaluations

# --- Backtesting Loop ---
logger.info("Starting backtesting loop...")
start_backtest_time = time.time()

for i in range(len(df_1m_rth)):
    current_bar = df_1m_rth.iloc[i]
    current_time = df_1m_rth.index[i]
    current_price = current_bar['close']
    
    # Count exposure when position is active
    if position_size != 0:
        exposure_bars += 1
    
    if position_size == 0:
        # No open position, check for entry signals based on EMA crossover
        if current_bar['crossover'] == 1:
            # Bullish Crossover: Enter Long
            position_size = POSITION_SIZE
            entry_price = current_price
            position_type = 'long'
            stop_loss_price = entry_price - STOP_LOSS_DISTANCE
            take_profit_price = entry_price + TAKE_PROFIT_DISTANCE
            entry_time = current_time
            logger.debug(f"Entered LONG at {entry_price} on {entry_time} UTC")
        
        elif current_bar['crossover'] == -1:
            # Bearish Crossover: Enter Short
            position_size = POSITION_SIZE
            entry_price = current_price
            position_type = 'short'
            stop_loss_price = entry_price + STOP_LOSS_DISTANCE
            take_profit_price = entry_price - TAKE_PROFIT_DISTANCE
            entry_time = current_time
            logger.debug(f"Entered SHORT at {entry_price} on {entry_time} UTC")
    
    else:
        # Position is open, check high-frequency data for exit
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
                pnl = ((exit_price - entry_price) * CONTRACT_MULTIPLIER * position_size) - SPREAD_COST_PER_TRADE
            elif position_type == 'short':
                pnl = ((entry_price - exit_price) * CONTRACT_MULTIPLIER * position_size) - SPREAD_COST_PER_TRADE
            
            trade_results.append(pnl)
            cash += pnl
            balance_series.append(cash)  # Update account balance
            
            # Print trade exit details
            exit_type = "TAKE PROFIT" if hit_take_profit else "STOP LOSS"
            logger.debug(f"Exited {position_type.upper()} at {exit_price} on {exit_time} UTC via {exit_type} for P&L: ${pnl:.2f}")
            
            # Reset position variables
            position_size = 0
            position_type = None
            entry_price = None
            stop_loss_price = None
            take_profit_price = None
            entry_time = None

end_backtest_time = time.time()
logger.info("Backtesting loop completed.")
logger.info(f"Backtesting duration: {(end_backtest_time - start_backtest_time)/60:.2f} minutes")

# --- Post-Backtest Calculations ---

# --- Convert balance_series to a Pandas Series with appropriate index ---
balance_series = pd.Series(balance_series, index=df_1m_rth.index[:len(balance_series)])

# --- Drawdown Duration Tracking ---
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

# --- Calculate Daily Returns ---
# Handle the FutureWarning by specifying fill_method explicitly
daily_returns = balance_series.resample('D').ffill().pct_change(fill_method=None).dropna()

# --- Define Sortino Ratio Calculation Function ---
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

# --- Performance Metrics ---

total_return_percentage = ((cash - INITIAL_CASH) / INITIAL_CASH) * 100
trading_days = max((df_1m_rth.index.max() - df_1m_rth.index.min()).days, 1)
annualized_return_percentage = ((cash / INITIAL_CASH) ** (252 / trading_days)) - 1
benchmark_return = ((df_1m_rth['close'].iloc[-1] - df_1m_rth['close'].iloc[0]) / df_1m_rth['close'].iloc[0]) * 100
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
exposure_time_percentage = (exposure_bars / len(df_1m_rth)) * 100

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

# --- Results Summary ---
print("\nPerformance Summary:")
results = {
    "Start Date": df_1m_rth.index.min().strftime("%Y-%m-%d"),
    "End Date": df_1m_rth.index.max().strftime("%Y-%m-%d"),
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
