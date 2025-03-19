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
import pandas_ta as ta  # New import for technical analysis

# --- Suppress Specific FutureWarnings (Optional) ---
warnings.filterwarnings("ignore", category=FutureWarning, message=".*'H' is deprecated.*")

# --- Configuration Parameters ---
INITIAL_CASH = 10000          # Starting cash
POSITION_SIZE = 1            # Number of contracts per trade
CONTRACT_MULTIPLIER = 5      # Contract multiplier
COMMISSION = 1.24            # Commission per trade

SHORT_EMA_PERIOD = 64        # Short EMA period (e.g., 64-period EMA)
LONG_EMA_PERIOD = 256        # Long EMA period (e.g., 256-period EMA)

# --- Data File ---
CSV_FILE_DAILY = 'Data/es_daily_data.csv'

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
    Loads hourly futures data into a pandas DataFrame with appropriate data types and datetime parsing.
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
            parse_dates=['Time'],
            infer_datetime_format=True
        )

        # Rename columns to match expectations
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Last': 'close',
            'Volume': 'volume'
        }, inplace=True)

        # Check datetime parsing
        if df['Time'].isnull().any():
            logger.error("Some 'Time' entries could not be parsed as datetime.")
            sys.exit(1)

        # Sort and set the index
        df.sort_values('Time', inplace=True)
        df.set_index('Time', inplace=True)

        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("The 'Time' column was not parsed as datetime. Check the CSV format.")
            sys.exit(1)

        # Localize to UTC if necessary
        if df.index.tz is None:
            df = df.tz_localize('UTC')
            logger.debug(f"Localized naive datetime index to UTC for {csv_file}.")
        else:
            df = df.tz_convert('UTC')
            logger.debug(f"Converted timezone-aware datetime index to UTC for {csv_file}.")

        # Keep only necessary columns
        columns_to_keep = ['open', 'high', 'low', 'close', 'volume']
        existing_columns = [col for col in columns_to_keep if col in df.columns]
        df = df[existing_columns]

        # Remove duplicate timestamps
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

def detect_crossovers(df, short_period, long_period):
    """
    Detects EMA crossovers in the DataFrame using pandas-ta generated EMA columns.
    Assumes the EMA columns are named 'EMA_{short_period}' and 'EMA_{long_period}'.
    """
    short_ema = f'EMA_{short_period}'
    long_ema = f'EMA_{long_period}'

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

# --- Load and Prepare Data ---

logger.info("Loading dataset...")
df_daily = load_data(CSV_FILE_DAILY)
logger.info("Dataset loaded successfully.")
print(f"Data Range: {df_daily.index.min()} to {df_daily.index.max()}")
print(f"Data Sample:\n{df_daily.head()}")

# --- Define Backtest Period ---
custom_start_date = "2000-01-01"
custom_end_date = "2024-12-11"

try:
    start_time = pd.to_datetime(custom_start_date).tz_localize('UTC')
    end_time = pd.to_datetime(custom_end_date).tz_localize('UTC')
except Exception as e:
    logger.error(f"Error parsing custom dates: {e}")
    sys.exit(1)

logger.info(f"Backtest Period: {start_time} to {end_time}")

logger.info(f"Slicing data from {start_time} to {end_time}...")
df_bt = df_daily.loc[start_time:end_time]
logger.info("Data sliced to backtest period.")
print(f"Sliced Data Range: {df_bt.index.min()} to {df_bt.index.max()}")
print(f"Sliced Data Sample:\n{df_bt.head()}")

# --- Calculate EMAs using pandas-ta ---
logger.info("Calculating EMAs using pandas-ta on backtest data...")
# This will append EMA columns named 'EMA_{length}'
df_bt = df_bt.copy()  # To avoid SettingWithCopyWarning
df_bt.ta.ema(length=SHORT_EMA_PERIOD, append=True)
df_bt.ta.ema(length=LONG_EMA_PERIOD, append=True)
# Drop rows with NaN values that may result from the EMA calculations
df_bt.dropna(inplace=True)
logger.info("EMAs calculated on backtest data.")
print(f"EMAs on Backtest Data Sample:\n{df_bt[[f'EMA_{SHORT_EMA_PERIOD}', f'EMA_{LONG_EMA_PERIOD}']].head()}")

# --- Detect Crossovers on EMAs ---
logger.info("Detecting EMA crossovers on backtest data...")
df_bt = detect_crossovers(df_bt, SHORT_EMA_PERIOD, LONG_EMA_PERIOD)
logger.info("EMA crossovers detected on backtest data.")
print(f"EMA Crossovers on Backtest Data Sample:\n{df_bt[[f'EMA_{SHORT_EMA_PERIOD}', f'EMA_{LONG_EMA_PERIOD}', 'crossover']].head()}")

print(f"\nBacktesting Parameters:")
print(f"Backtesting Period       : {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
print(f"Data Points              : {len(df_bt)}")

# --- Initialize Backtest Variables ---
position_size = 0
entry_price = None
position_type = None  
cash = INITIAL_CASH
trade_results = []
balance_series = [INITIAL_CASH]
balance_times = [df_bt.index[0]]
exposure_days = 0

in_drawdown = False
drawdown_start = None
drawdown_durations = []

# --- Backtesting Loop ---
logger.info("Starting backtesting loop...")
start_backtest_time = time.time()

for i in range(len(df_bt)):
    current_bar = df_bt.iloc[i]
    current_time = df_bt.index[i]
    current_price = current_bar['close']

    if position_size != 0:
        exposure_days += 1

    if position_size == 0:
        if current_bar['crossover'] == 1:
            position_size = POSITION_SIZE
            entry_price = current_price
            position_type = 'long'
            entry_time = current_time
            logger.info(f"Entered LONG at {entry_price} on {entry_time.strftime('%Y-%m-%d')} UTC")
    else:
        if current_bar['crossover'] == -1:
            exit_price = current_price
            pnl = (exit_price - entry_price) * CONTRACT_MULTIPLIER * position_size - COMMISSION
            trade_results.append(pnl)
            cash += pnl
            balance_series.append(cash)
            balance_times.append(current_time)
            logger.info(f"Exited LONG at {exit_price} on {current_time.strftime('%Y-%m-%d')} UTC for P&L: ${pnl:.2f}")

            position_size = 0
            position_type = None
            entry_price = None
            entry_time = None

    if position_size == 0:
        balance_series.append(cash)
        balance_times.append(current_time)

if position_size != 0:
    logger.info("Position still open at the end of the backtest. Closing at last available price.")
    exit_price = df_bt.iloc[-1]['close']
    exit_time = df_bt.index[-1]
    pnl = (exit_price - entry_price) * CONTRACT_MULTIPLIER * position_size - COMMISSION
    trade_results.append(pnl)
    cash += pnl
    balance_series.append(cash)
    balance_times.append(exit_time)
    logger.info(f"Exited LONG at {exit_price} on {exit_time.strftime('%Y-%m-%d')} UTC via END OF DATA for P&L: ${pnl:.2f}")

end_backtest_time = time.time()
logger.info("Backtesting loop completed.")
logger.info(f"Backtesting duration: {(end_backtest_time - start_backtest_time)/60:.2f} minutes")

# --- Post-Backtest Calculations ---
balance_series = pd.Series(balance_series, index=balance_times)
initial_price = df_bt['close'].iloc[0]
benchmark_equity_curve = (df_bt['close'] / initial_price) * INITIAL_CASH

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

if in_drawdown:
    drawdown_end = balance_series.index[-1]
    duration = (drawdown_end - drawdown_start).days
    drawdown_durations.append(duration)

daily_returns = balance_series.pct_change().dropna()

def calculate_sortino_ratio(daily_returns, target_return=0):
    if daily_returns.empty:
        return np.nan
    excess_returns = daily_returns - target_return
    downside_returns = excess_returns[excess_returns < 0]
    if downside_returns.empty or downside_returns.std() == 0:
        return np.inf
    downside_std = downside_returns.std() * np.sqrt(252)
    annualized_mean_excess_return = daily_returns.mean() * 252
    return annualized_mean_excess_return / downside_std

total_return_percentage = ((cash - INITIAL_CASH) / INITIAL_CASH) * 100
trading_days = max((df_bt.index.max() - df_bt.index.min()).days, 1)
annualized_return_percentage = ((cash / INITIAL_CASH) ** (252 / trading_days)) - 1
benchmark_return = ((df_bt['close'].iloc[-1] - df_bt['close'].iloc[0]) / df_bt['close'].iloc[0]) * 100
equity_peak = balance_series.max()

volatility_annual = daily_returns.std() * np.sqrt(252) * 100
risk_free_rate = 0  
sharpe_ratio = ((daily_returns.mean() - risk_free_rate) / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
sortino_ratio = calculate_sortino_ratio(daily_returns)

running_max_series = balance_series.cummax()
drawdowns = (balance_series - running_max_series) / running_max_series
max_drawdown = drawdowns.min() * 100
average_drawdown = drawdowns[drawdowns < 0].mean() * 100

exposure_time_percentage = (exposure_days / len(df_bt)) * 100

winning_trades = [pnl for pnl in trade_results if pnl > 0]
losing_trades = [pnl for pnl in trade_results if pnl <= 0]
profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf')
calmar_ratio = (total_return_percentage / abs(max_drawdown)) if max_drawdown != 0 else float('inf')

if drawdown_durations:
    max_drawdown_duration_days = max(drawdown_durations)
    average_drawdown_duration_days = np.mean(drawdown_durations)
else:
    max_drawdown_duration_days = 0
    average_drawdown_duration_days = 0

print("\nPerformance Summary:")
results = {
    "Start Date": df_bt.index.min().strftime("%Y-%m-%d"),
    "End Date": df_bt.index.max().strftime("%Y-%m-%d"),
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

# --- Plot Equity Curves ---
plt.figure(figsize=(14, 7))
plt.plot(balance_series.index, balance_series.values, label='Strategy Equity Curve', color='blue')
plt.plot(df_bt.index, benchmark_equity_curve, label='Buy-and-Hold Benchmark (ES Future)', color='orange')
plt.title('Equity Curve Comparison: Strategy vs. Buy-and-Hold Benchmark (ES Future)')
plt.xlabel('Date')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()