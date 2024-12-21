from ib_insync import IB, Future, util
import pandas as pd
import logging
from datetime import datetime, timedelta
import pytz
import time

# Configure terminal logging
logging.basicConfig(
    level=logging.INFO,  # Logs all events to the terminal
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Retry settings
MAX_RETRIES = 5       
RETRY_DELAY = 5       


# Create ES Futures Contract
def create_es_future_contract(expiration):
    return Future(
        symbol='ES',
        exchange='CME',
        currency='USD',
        lastTradeDateOrContractMonth=expiration,
        includeExpired=True
    )


# Check Market Hours
def is_market_open(timestamp):
    """
    Check if a timestamp falls within ES futures market hours.
    Market is closed between:
    - Daily Maintenance: 5:00 PM - 6:00 PM ET (10:00 PM - 11:00 PM UTC)
    - Weekends: Friday 5:00 PM ET to Sunday 6:00 PM ET
    """
    eastern = pytz.timezone('US/Eastern')
    dt_eastern = timestamp.astimezone(eastern)

    # Market is closed on weekends
    if dt_eastern.weekday() >= 5:  # Saturday/Sunday
        return False
    
    # Market is closed during the daily maintenance window
    if dt_eastern.time() >= datetime.strptime("17:00:00", "%H:%M:%S").time() and dt_eastern.time() < datetime.strptime("18:00:00", "%H:%M:%S").time():
        return False
    
    return True


# Fetch Historical Data with Correct Market Times
def fetch_minute_data_with_retries(ib, contract, start_date_str, end_date_str):
    all_data = []
    eastern = pytz.timezone('US/Eastern')
    utc = pytz.utc

    try:
        start_naive = datetime.strptime(start_date_str, '%Y%m%d %H:%M:%S')
        end_naive = datetime.strptime(end_date_str, '%Y%m%d %H:%M:%S')
        start_utc = eastern.localize(start_naive).astimezone(utc)
        end_utc = eastern.localize(end_naive).astimezone(utc)
    except ValueError as ve:
        logger.error(f"Date parsing error: {ve}")
        return pd.DataFrame()

    current_end = end_utc

    while current_end > start_utc:
        chunk_start = max(start_utc, current_end - timedelta(days=7))
        end_datetime_formatted = current_end.strftime('%Y%m%d-%H:%M:%S')

        if not is_market_open(current_end):
            current_end -= timedelta(minutes=1)
            continue

        retry_count = 0
        success = False
        
        while retry_count < MAX_RETRIES and not success:
            try:
                logger.info(f"Requesting data from {chunk_start.date()} to {current_end.date()}")
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=end_datetime_formatted,
                    durationStr='7 D',  # Request 7 days of data
                    barSizeSetting='1 min',
                    whatToShow='TRADES',
                    useRTH=False,
                    formatDate=1,
                    keepUpToDate=False
                )

                if bars:
                    df = util.df(bars)
                    df['contract'] = contract.lastTradeDateOrContractMonth
                    all_data.append(df)
                    logger.info(f"Fetched {len(df)} records from {chunk_start.date()} to {current_end.date()}")
                else:
                    logger.warning(f"No data returned from {chunk_start.date()} to {current_end.date()}")

                success = True
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Error fetching data: {e}")
                if retry_count < MAX_RETRIES:
                    delay = RETRY_DELAY * (2 ** (retry_count - 1))
                    logger.info(f"Retrying in {delay} seconds... (Attempt {retry_count})")
                    time.sleep(delay)
                else:
                    logger.error(f"Max retries reached. Failed to fetch data from {chunk_start.date()} to {current_end.date()}")

        current_end = chunk_start - timedelta(seconds=1)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date'], utc=True)
        combined_df = combined_df.sort_values(by=['date', 'contract']).reset_index(drop=True)
        return combined_df
    else:
        return pd.DataFrame()


# Main Execution
def main():
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=1)
        logger.info("Connected to IB")
    except Exception as e:
        logger.error(f"Failed to connect to IB: {e}")
        return

    # Define Futures Contract Dates
    contracts_with_dates = [
        (create_es_future_contract('202212'), '20220917 23:59:59', '20221216 23:59:59'),
        (create_es_future_contract('202303'), '20221216 23:59:59', '20230316 23:59:59'),
        (create_es_future_contract('202306'), '20230316 23:59:59', '20230616 23:59:59'),
        (create_es_future_contract('202309'), '20230616 23:59:59', '20230916 23:59:59'),
        (create_es_future_contract('202312'), '20230916 23:59:59', '20231216 23:59:59'),
        (create_es_future_contract('202403'), '20231216 23:59:59', '20240316 23:59:59'),
        (create_es_future_contract('202406'), '20240316 23:59:59', '20240616 23:59:59'),
        (create_es_future_contract('202409'), '20240616 23:59:59', '20240916 23:59:59'),
        (create_es_future_contract('202412'), '20240916 23:59:59', '20241216 23:59:59'),
        (create_es_future_contract('202503'), '20241216 23:59:59', '20250316 23:59:59')
    ]

    data_frames = []

    for contract, start_date, end_date in contracts_with_dates:
        ib.qualifyContracts(contract)
        df = fetch_minute_data_with_retries(
            ib, contract, start_date_str=start_date, end_date_str=end_date
        )
        if not df.empty:
            data_frames.append(df)

    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
        combined_df = combined_df.sort_values(by=['date', 'contract']).reset_index(drop=True)
        
        # Save and log the data
        combined_df.to_csv('combined_es_1min_futures_data.csv', index=False)
        logger.info("Data saved to 'combined_es_1min_futures_data.csv'")
    else:
        logger.warning("No data fetched for the specified contracts.")

    ib.disconnect()
    logger.info("Disconnected from IB")


if __name__ == "__main__":
    main()