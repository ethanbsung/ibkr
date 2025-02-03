import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from datetime import datetime, time

# ----------------------------
# Configuration Parameters
# ----------------------------
DATA_PATH = 'Data/es_1m_data.csv'  # Path to your CSV data file

# Strategy Parameters
ATR_PERIOD = 14                  # Lookback period for ATR
THRESHOLD_MULTIPLIER = 1.0       # Price must be 1 ATR away from VWAP to enter
STOP_LOSS_MULTIPLIER = 0.5       # Stop loss distance in ATR (0.5 ATR)
INITIAL_CASH = 5000              # Starting account equity
MULTIPLIER = 5                   # Contract multiplier (e.g., $5 per point)
COMMISSION = 1.24                # Commission per trade leg (entry or exit)
CONTRACTS = 1                    # Number of contracts per trade

# Backtest Date Range (inclusive)
START_DATE = '2008-01-01'        # Adjust these dates if needed
END_DATE = '2024-12-31'

# Trading hours (local time assumed to be Eastern)
TRADING_START = time(9, 30)
TRADING_END = time(16, 0)

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# ----------------------------
# Helper Functions
# ----------------------------
def load_and_preprocess_data(file_path):
    """
    Loads the CSV file (expected columns: Symbol, Time, Open, High, Low, Last, Change, %Chg, Volume, Open Int),
    strips extra whitespace from column names, renames columns,
    converts Time to datetime (using an explicit format), sets it as the index,
    and filters by date and trading hours.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"Data file {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        sys.exit(1)
    
    # Remove extra whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Rename columns for consistency.
    df.rename(columns={
        'Time': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Last': 'close',
        'Volume': 'volume'
    }, inplace=True)
    
    # Convert 'timestamp' to datetime using an explicit format.
    try:
        # Adjust the format if needed. For example: "2008-08-01 10:43"
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M')
    except Exception as e:
        logger.error(f"Error converting timestamp: {e}")
        sys.exit(1)
    
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # Log a sample of the data to verify that timestamps look correct.
    logger.info("Sample data after timestamp conversion:")
    logger.info(df.head(5))
    
    # Filter by backtest date range.
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    logger.info(f"Filtering data between {start_dt.date()} and {end_dt.date()}")
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    if df.empty:
        logger.error("No data in the selected date range. Please choose a valid date range.")
        sys.exit(1)
    
    # Filter to Regular Trading Hours (RTH)
    df = df.between_time(TRADING_START.strftime("%H:%M"), TRADING_END.strftime("%H:%M"))
    if df.empty:
        logger.error("No data remains after filtering to regular trading hours.")
        sys.exit(1)
    
    # Drop rows with missing values and ensure volume is numeric.
    df.dropna(inplace=True)
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df.dropna(subset=['volume'], inplace=True)
    
    logger.info(f"Data loaded and preprocessed. Date range: {df.index.min()} to {df.index.max()}. Total bars: {len(df)}")
    return df

def calculate_vwap(df):
    """
    Computes the intraday VWAP for each bar.
    VWAP is calculated per trading day as the cumulative sum of (price*volume)
    divided by the cumulative sum of volume.
    """
    df['pv'] = df['close'] * df['volume']
    df['cum_pv'] = df.groupby(df.index.date)['pv'].cumsum()
    df['cum_volume'] = df.groupby(df.index.date)['volume'].cumsum()
    df['vwap'] = df['cum_pv'] / df['cum_volume']
    return df

def calculate_atr(df, period=ATR_PERIOD):
    """
    Calculates the Average True Range (ATR) over the specified period.
    """
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['prev_close']).abs()
    df['tr3'] = (df['low'] - df['prev_close']).abs()
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR'] = df['true_range'].rolling(window=period, min_periods=1).mean()
    return df

# ----------------------------
# Main Backtest Logic
# ----------------------------
def run_atr_mean_reversion():
    # Load and preprocess data
    df = load_and_preprocess_data(DATA_PATH)
    
    # Calculate VWAP and ATR
    df = calculate_vwap(df)
    df = calculate_atr(df, period=ATR_PERIOD)
    
    # Calculate threshold value (ATR multiplier) for entry signals.
    df['threshold'] = THRESHOLD_MULTIPLIER * df['ATR']
    
    # Generate trading signals:
    #   +1 for a long signal when close < (vwap - threshold)
    #   -1 for a short signal when close > (vwap + threshold)
    #   0 otherwise
    df['signal'] = 0
    df.loc[df['close'] < (df['vwap'] - df['threshold']), 'signal'] = 1   # long signal
    df.loc[df['close'] > (df['vwap'] + df['threshold']), 'signal'] = -1  # short signal

    # Backtest state variables
    trades = []           # To record trade details
    equity_timeline = []  # To record equity over time
    position = 0          # 0: flat, 1: long, -1: short
    entry_price = 0.0
    entry_time = None
    stop_loss_level = None  # Set when entering a trade

    cash = INITIAL_CASH
    in_trade_flags = []   # 1 if in a trade for that bar; 0 otherwise

    logger.info("Starting backtest loop...")
    # Iterate over each bar
    for timestamp, row in df.iterrows():
        price = row['close']
        sig = row['signal']
        current_vwap = row['vwap']
        current_ATR = row['ATR']

        # Record whether we are in a trade
        in_trade_flags.append(1 if position != 0 else 0)

        # If not in a position, look for an entry signal.
        if position == 0:
            if sig == 1:
                # Enter long position when price is 1 ATR below VWAP.
                entry_price = price
                entry_time = timestamp
                position = 1
                # Set stop loss for long: entry price minus 0.5 ATR.
                stop_loss_level = entry_price - (STOP_LOSS_MULTIPLIER * current_ATR)
                logger.info(f"Enter LONG at {timestamp} price: {price:.2f}, Stop Loss: {stop_loss_level:.2f}")
            elif sig == -1:
                # Enter short position when price is 1 ATR above VWAP.
                entry_price = price
                entry_time = timestamp
                position = -1
                # Set stop loss for short: entry price plus 0.5 ATR.
                stop_loss_level = entry_price + (STOP_LOSS_MULTIPLIER * current_ATR)
                logger.info(f"Enter SHORT at {timestamp} price: {price:.2f}, Stop Loss: {stop_loss_level:.2f}")
        else:
            # Already in a position; check for exit conditions.
            # First, check stop loss.
            if position == 1 and price <= stop_loss_level:
                # Long position stop loss hit.
                exit_price = price
                exit_time = timestamp
                pnl = (exit_price - entry_price) * MULTIPLIER * CONTRACTS - (COMMISSION * 2)
                cash += pnl
                trades.append({
                    'Entry Time': entry_time,
                    'Exit Time': exit_time,
                    'Position': 'Long (Stop Loss)',
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'PnL': pnl
                })
                logger.info(f"LONG Stop Loss hit at {timestamp} price: {price:.2f} | PnL: {pnl:.2f}")
                position = 0
                stop_loss_level = None
            elif position == -1 and price >= stop_loss_level:
                # Short position stop loss hit.
                exit_price = price
                exit_time = timestamp
                pnl = (entry_price - exit_price) * MULTIPLIER * CONTRACTS - (COMMISSION * 2)
                cash += pnl
                trades.append({
                    'Entry Time': entry_time,
                    'Exit Time': exit_time,
                    'Position': 'Short (Stop Loss)',
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'PnL': pnl
                })
                logger.info(f"SHORT Stop Loss hit at {timestamp} price: {price:.2f} | PnL: {pnl:.2f}")
                position = 0
                stop_loss_level = None
            else:
                # Check for take profit when price reverts back to VWAP.
                if position == 1 and price >= current_vwap:
                    exit_price = price
                    exit_time = timestamp
                    pnl = (exit_price - entry_price) * MULTIPLIER * CONTRACTS - (COMMISSION * 2)
                    cash += pnl
                    trades.append({
                        'Entry Time': entry_time,
                        'Exit Time': exit_time,
                        'Position': 'Long (Take Profit)',
                        'Entry Price': entry_price,
                        'Exit Price': exit_price,
                        'PnL': pnl
                    })
                    logger.info(f"LONG Take Profit exit at {timestamp} price: {price:.2f} | PnL: {pnl:.2f}")
                    position = 0
                    stop_loss_level = None
                elif position == -1 and price <= current_vwap:
                    exit_price = price
                    exit_time = timestamp
                    pnl = (entry_price - exit_price) * MULTIPLIER * CONTRACTS - (COMMISSION * 2)
                    cash += pnl
                    trades.append({
                        'Entry Time': entry_time,
                        'Exit Time': exit_time,
                        'Position': 'Short (Take Profit)',
                        'Entry Price': entry_price,
                        'Exit Price': exit_price,
                        'PnL': pnl
                    })
                    logger.info(f"SHORT Take Profit exit at {timestamp} price: {price:.2f} | PnL: {pnl:.2f}")
                    position = 0
                    stop_loss_level = None

        # Mark-to-market equity calculation:
        if position == 1:
            mtm = (price - entry_price) * MULTIPLIER * CONTRACTS
            equity_value = cash + mtm
        elif position == -1:
            mtm = (entry_price - price) * MULTIPLIER * CONTRACTS
            equity_value = cash + mtm
        else:
            equity_value = cash
        equity_timeline.append((timestamp, equity_value))

    # If still in a position at the end, force an exit.
    if position != 0:
        final_price = df.iloc[-1]['close']
        exit_time = df.index[-1]
        if position == 1:
            pnl = (final_price - entry_price) * MULTIPLIER * CONTRACTS - (COMMISSION * 2)
            trade_side = 'Long (Force Exit)'
        else:
            pnl = (entry_price - final_price) * MULTIPLIER * CONTRACTS - (COMMISSION * 2)
            trade_side = 'Short (Force Exit)'
        cash += pnl
        trades.append({
            'Entry Time': entry_time,
            'Exit Time': exit_time,
            'Position': trade_side,
            'Entry Price': entry_price,
            'Exit Price': final_price,
            'PnL': pnl
        })
        logger.info(f"Force exit {trade_side} at {exit_time} price: {final_price:.2f} | PnL: {pnl:.2f}")
        equity_timeline.append((exit_time, cash))
        in_trade_flags.append(0)
        position = 0

    # Convert equity timeline into a pandas Series.
    if len(equity_timeline) == 0:
        logger.error("No equity data was generated during the backtest.")
        sys.exit(1)
    balance_series = pd.Series(
        data=[eq for ts, eq in equity_timeline],
        index=[ts for ts, eq in equity_timeline]
    )

    # ----------------------------
    # Performance Metrics
    # ----------------------------
    total_bars = len(df)
    exposure_time_percentage = (sum(in_trade_flags) / total_bars) * 100 if total_bars else 0

    final_balance = cash
    equity_peak = balance_series.cummax().max()
    total_return_percentage = ((final_balance - INITIAL_CASH) / INITIAL_CASH) * 100

    if len(balance_series) > 0:
        delta_days = (balance_series.index[-1] - balance_series.index[0]).total_seconds() / (3600 * 24)
    else:
        delta_days = 0

    annualized_return_percentage = ((final_balance / INITIAL_CASH) ** (365 / delta_days) - 1) * 100 if delta_days > 0 else 0

    # Create a buy-and-hold benchmark using 30-minute resampled data.
    df_30m = df.resample('30min').last().dropna()
    if not df_30m.empty:
        benchmark_start = df_30m['close'].iloc[0]
        benchmark_end = df_30m['close'].iloc[-1]
        benchmark_return = ((benchmark_end / benchmark_start) - 1) * 100
    else:
        benchmark_return = 0

    # ----------------------------
    # Print Performance Summary
    # ----------------------------
    print("\nPerformance Summary:")
    results = {
        "Start Date": df.index.min().strftime("%Y-%m-%d"),
        "End Date": df.index.max().strftime("%Y-%m-%d"),
        "Exposure Time": f"{exposure_time_percentage:.2f}%",
        "Final Account Balance": f"${final_balance:,.2f}",
        "Equity Peak": f"${equity_peak:,.2f}",
        "Total Return": f"{total_return_percentage:.2f}%",
        "Annualized Return": f"{annualized_return_percentage:.2f}%",
        "Benchmark Return": f"{benchmark_return:.2f}%"
    }
    for key, value in results.items():
        print(f"{key:25}: {value:>15}")

    # ----------------------------
    # Print Trade Details
    # ----------------------------
    print("\nTrade Details:")
    for t in trades:
        print(f"{t['Position']:20} | Entry: {t['Entry Time']} @ {t['Entry Price']:.2f}  -->  Exit: {t['Exit Time']} @ {t['Exit Price']:.2f} | PnL: {t['PnL']:.2f}")

    # ----------------------------
    # Plot Equity Curves
    # ----------------------------
    if len(balance_series) < 2:
        logger.warning("Not enough data points to plot equity curves.")
    else:
        if not df_30m.empty:
            initial_close = df_30m['close'].iloc[0]
            benchmark_equity = (df_30m['close'] / initial_close) * INITIAL_CASH
            benchmark_equity = benchmark_equity.reindex(balance_series.index, method='ffill').fillna(method='ffill')
        else:
            benchmark_equity = pd.Series(index=balance_series.index, data=INITIAL_CASH)
            
        equity_df_plot = pd.DataFrame({
            'Strategy': balance_series,
            'Benchmark': benchmark_equity
        })

        plt.figure(figsize=(14, 7))
        plt.plot(equity_df_plot.index, equity_df_plot['Strategy'], label='Strategy Equity')
        plt.plot(equity_df_plot.index, equity_df_plot['Benchmark'], label='Benchmark Equity', linestyle='--')
        plt.title('Equity Curve: Strategy vs Benchmark')
        plt.xlabel('Time')
        plt.ylabel('Account Balance ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_atr_mean_reversion()