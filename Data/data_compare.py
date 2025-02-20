import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
import pytz
import logging
import sys
import time as tm  # to avoid conflict with datetime.time

# Import IBKR API via ib_insync
from ib_insync import IB, Future, util

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# --- Configuration Parameters ---
INITIAL_CAPITAL = 10000
TIMEFRAME = '1 min'

# Date ranges
# For IBKR data and CSV comparison, we use June 2024 to December 2024.
CSV_START_DATE = pd.Timestamp('2024-06-01', tz='UTC')
CSV_END_DATE   = pd.Timestamp('2024-12-31 23:59:59', tz='UTC')

# IBKR Connection Parameters
IB_HOST = '127.0.0.1'
IB_PORT = 4002
IB_CLIENT_ID = 1

# Define the ES futures contract. Adjust the localSymbol if needed.
ES_CONTRACT = Future(localSymbol='ESH5', exchange='CME', currency='USD')

# --- Historical Data Fetch Function (from your strategy example) ---
def fetch_ibkr_data(ib, contract, bar_size, start_time, end_time, useRTH=False):
    """
    Fetch historical data from IBKR using ib_insync in chunks.
    Iterates backward in time from end_time to start_time, requesting data in chunks.
    """
    if bar_size == '1 min':
        max_chunk = pd.Timedelta(days=7)
    elif bar_size == '30 mins':
        max_chunk = pd.Timedelta(days=365)
    elif bar_size == '5 secs':
        max_chunk = pd.Timedelta(days=1)
    else:
        max_chunk = pd.Timedelta(days=30)
    
    current_end = end_time
    all_bars = []
    while current_end > start_time:
        current_start = max(start_time, current_end - max_chunk)
        delta = current_end - current_start
        if delta < pd.Timedelta(days=1):
            duration_str = "1 D"
        else:
            duration_days = delta.days
            duration_str = f"{duration_days} D"
        
        end_dt_str = current_end.strftime("%Y%m%d %H:%M:%S")
        logger.info(f"Requesting {bar_size} data from {current_start} to {current_end} "
                    f"(Duration: {duration_str}) for contract {contract.localSymbol}")
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_dt_str,
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=useRTH,
                formatDate=1
            )
        except Exception as e:
            logger.error(f"Error requesting historical data: {e}")
            break

        if not bars:
            logger.warning("No bars returned for this chunk.")
            break

        df_chunk = util.df(bars)
        all_bars.append(df_chunk)
        # Update current_end to just before the earliest bar in this chunk
        earliest_bar_time = pd.to_datetime(df_chunk['date'].min())
        current_end = earliest_bar_time - pd.Timedelta(seconds=1)
        tm.sleep(1)  # Pause to respect IBKR pacing limits

    if all_bars:
        df_all = pd.concat(all_bars, ignore_index=True)
        df_all.sort_values('date', inplace=True)
        return df_all
    else:
        logger.error("No historical data fetched.")
        return pd.DataFrame()

def mark_rth(df):
    """
    Mark each bar as inside Regular Trading Hours (RTH: 09:30-16:00 ET on weekdays).
    """
    eastern = pytz.timezone('US/Eastern')
    if df.index.tz is None:
        df = df.tz_localize(eastern)
    else:
        df = df.tz_convert(eastern)
    df['is_rth'] = df.index.weekday < 5  # Weekdays only
    df['time'] = df.index.time
    df.loc[(df['time'] < time(9, 30)) | (df['time'] > time(16, 0)), 'is_rth'] = False
    df.drop(['time'], axis=1, inplace=True)
    df = df.tz_convert('UTC')  # Convert back to UTC
    return df

def compute_equity_curve(df, price_col='close', initial_capital=5000):
    """
    Compute a simple equity curve based on cumulative returns from the given price column.
    """
    df = df.copy()
    df['pct_change'] = df[price_col].pct_change().fillna(0)
    df['cum_return'] = (1 + df['pct_change']).cumprod()
    df['equity'] = initial_capital * df['cum_return']
    return df

# --- Main Execution ---
def main():
    # Use the same date range for IBKR data as for CSV comparison.
    ibkr_start = CSV_START_DATE
    ibkr_end = CSV_END_DATE

    # Connect to IBKR and fetch historical data.
    ib = IB()
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
    except Exception as e:
        logger.error(f"Failed to connect to IBKR: {e}")
        sys.exit(1)

    logger.info("Fetching IBKR historical data...")
    df_ibkr = fetch_ibkr_data(ib, ES_CONTRACT, TIMEFRAME, ibkr_start, ibkr_end, useRTH=False)
    ib.disconnect()

    if df_ibkr.empty:
        logger.error("No IBKR data fetched.")
        sys.exit(1)
    
    # Rename the date column and set it as the index.
    df_ibkr.rename(columns={'date': 'Time'}, inplace=True)
    df_ibkr.set_index('Time', inplace=True)
    df_ibkr.index = pd.to_datetime(df_ibkr.index)
    df_ibkr = mark_rth(df_ibkr)

    if 'close' not in df_ibkr.columns:
        logger.error("IBKR data does not have a 'close' column.")
        sys.exit(1)

    # Compute the equity curve for IBKR data.
    df_ibkr = compute_equity_curve(df_ibkr, price_col='close', initial_capital=INITIAL_CAPITAL)
    
    # Plot IBKR equity curve.
    plt.figure(figsize=(12, 6))
    plt.plot(df_ibkr.index, df_ibkr['equity'], label='IBKR Equity Curve')
    plt.title("Equity Curve from IBKR Data (ES Futures, 1-min Bars)")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    '''
    # --- Load CSV Data ---
    try:
        df_csv = pd.read_csv("Data/es_1m_data.csv", parse_dates=['Time'])
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        sys.exit(1)
    
    df_csv.set_index('Time', inplace=True)
    df_csv.index = pd.to_datetime(df_csv.index)
    # Filter CSV data to only include dates from June 2024 to December 2024.
    df_csv = df_csv[(df_csv.index >= CSV_START_DATE) & (df_csv.index <= CSV_END_DATE)]
    
    # Rename the 'Last' column to 'close' for consistency.
    if 'Last' in df_csv.columns:
        df_csv.rename(columns={'Last': 'close'}, inplace=True)
    else:
        logger.error("CSV file is missing the 'Last' column.")
        sys.exit(1)
    
    
    # Compute the equity curve for the CSV data.
    df_csv = compute_equity_curve(df_csv, price_col='close', initial_capital=INITIAL_CAPITAL)
    
    # Plot CSV equity curve.
    plt.figure(figsize=(12, 6))
    plt.plot(df_csv.index, df_csv['equity'], color='orange', label='CSV Equity Curve')
    plt.title("Equity Curve from CSV Data (ES Futures, 1-min Bars, Jun-Dec 2024)")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    '''

if __name__ == "__main__":
    main()