from ib_insync import IB, Future, util
import pandas as pd
import logging
from datetime import datetime, timedelta
import pytz
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for retry logic
MAX_RETRIES = 5       # Maximum number of retries per request
RETRY_DELAY = 5       # Initial delay in seconds before retrying

def create_es_future_contract(expiration):
    """
    Create an ES future contract based on the expiration date.
    
    Parameters:
        expiration (str): Expiration date in 'YYYYMMDD' format.
    
    Returns:
        Future: An IB-insync Future contract object.
    """
    return Future(
        symbol='ES',
        exchange='CME',
        currency='USD',
        lastTradeDateOrContractMonth=expiration,
        includeExpired=True
    )

def fetch_minute_data_with_retries(ib, contract, start_date_str, end_date_str, bar_size='1 min', what_to_show='TRADES'):
    """
    Fetch 1-minute historical data for a contract from start_date to end_date in weekly chunks.
    Implements retry logic with exponential backoff.
    
    Parameters:
        ib (IB): An instance of the IB class.
        contract (Future): The IB-insync Future contract object.
        start_date_str (str): Start date as a string in 'YYYYMMDD HH:MM:SS' format.
        end_date_str (str): End date as a string in 'YYYYMMDD HH:MM:SS' format.
        bar_size (str): Size of each bar (default: '1 min').
        what_to_show (str): Type of data to retrieve (default: 'TRADES').
    
    Returns:
        pd.DataFrame: Combined DataFrame of historical data or empty DataFrame on failure.
    """
    all_data = []
    
    # Define time zones
    eastern = pytz.timezone('US/Eastern')
    utc = pytz.utc

    # Parse start_date and end_date strings to datetime objects
    try:
        start_naive = datetime.strptime(start_date_str, '%Y%m%d %H:%M:%S')
        end_naive = datetime.strptime(end_date_str, '%Y%m%d %H:%M:%S')
    except ValueError as ve:
        logger.error(f"Date parsing error: {ve}")
        return pd.DataFrame()
    
    # Localize to US/Eastern
    start_eastern = eastern.localize(start_naive)
    end_eastern = eastern.localize(end_naive)
    
    # Convert to UTC
    start_utc = start_eastern.astimezone(utc)
    end_utc = end_eastern.astimezone(utc)
    
    # Initialize current_end
    current_end = end_utc
    
    while current_end > start_utc:
        # Define the chunk_start as max(start, current_end - 7 days)
        chunk_start = max(start_utc, current_end - timedelta(days=7))
        
        # Format endDateTime as 'yyyymmdd-HH:MM:SS'
        end_datetime_formatted = current_end.strftime('%Y%m%d-%H:%M:%S')
        
        retry_count = 0
        success = False
        while retry_count < MAX_RETRIES and not success:
            try:
                logger.info(f"Requesting 1-minute data for {contract.lastTradeDateOrContractMonth} from {chunk_start} to {current_end}")
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=end_datetime_formatted,
                    durationStr='7 D',              # Duration for each chunk
                    barSizeSetting=bar_size,
                    whatToShow=what_to_show,
                    useRTH=False,
                    formatDate=1,                    # Date format as 'yyyymmdd-HH:MM:SS'
                    keepUpToDate=False
                )
                
                if bars:
                    df = util.df(bars)
                    df['contract'] = contract.lastTradeDateOrContractMonth
                    all_data.append(df)
                    logger.info(f"Fetched {len(df)} records from {chunk_start} to {current_end}")
                else:
                    logger.warning(f"No data returned for {contract.lastTradeDateOrContractMonth} from {chunk_start} to {current_end}")
                
                success = True  # Exit retry loop
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Error fetching data for {contract.lastTradeDateOrContractMonth} from {chunk_start} to {current_end}: {e}")
                if retry_count < MAX_RETRIES:
                    delay = RETRY_DELAY * (2 ** (retry_count - 1))  # Exponential backoff
                    logger.info(f"Retrying in {delay} seconds... (Attempt {retry_count})")
                    time.sleep(delay)
                else:
                    logger.error(f"Max retries reached. Failed to fetch data for {contract.lastTradeDateOrContractMonth} from {chunk_start} to {current_end}")
        
        # Move the window back by 7 days
        current_end = chunk_start - timedelta(seconds=1)  # Avoid overlapping
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        # Convert 'date' column to datetime with UTC
        try:
            combined_df['date'] = pd.to_datetime(combined_df['date'], utc=True)
        except ValueError as ve:
            logger.error(f"Date parsing error in combined DataFrame: {ve}")
            return pd.DataFrame()
        
        # Sort by date and contract
        combined_df = combined_df.sort_values(by=['date', 'contract']).reset_index(drop=True)
        return combined_df
    else:
        return pd.DataFrame()

def main():
    # Connect to IB
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=1)
        logger.info("Connected to IB")
    except Exception as e:
        logger.error(f"Failed to connect to IB: {e}")
        return
    
    # Define Correct Non-Overlapping Contracts with (contract, start_date, end_date)
    contracts_with_dates = [
        (create_es_future_contract('202212'), '20220917 23:59:59', '20221216 23:59:59'),  # Sep-Dec 2022
        (create_es_future_contract('202303'), '20221217 23:59:59', '20230316 23:59:59'),  # Dec 2022 - Mar 2023
        (create_es_future_contract('202306'), '20230317 23:59:59', '20230616 23:59:59'),  # Mar-Jun 2023
        (create_es_future_contract('202309'), '20230617 23:59:59', '20230916 23:59:59'),  # Jun-Sep 2023
        (create_es_future_contract('202312'), '20230917 23:59:59', '20231216 23:59:59'),  # Sep-Dec 2023
        (create_es_future_contract('202403'), '20231217 23:59:59', '20240316 23:59:59'),  # Dec 2023 - Mar 2024
        (create_es_future_contract('202406'), '20240317 23:59:59', '20240616 23:59:59'),  # Mar-Jun 2024
        (create_es_future_contract('202409'), '20240617 23:59:59', '20240916 23:59:59'),  # Jun-Sep 2024
        (create_es_future_contract('202412'), '20240917 23:59:59', '20241216 23:59:59'),  # Sep-Dec 2024
    ]
    
    # Fetch historical data for each contract
    data_frames = []
    for contract, start_date, end_date in contracts_with_dates:
        ib.qualifyContracts(contract)
        df = fetch_minute_data_with_retries(
            ib,
            contract,
            start_date_str=start_date,
            end_date_str=end_date,
            bar_size='1 min',
            what_to_show='TRADES'
        )
        if not df.empty:
            data_frames.append(df)
    
    # Combine DataFrames if data is available
    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
        
        # Ensure 'date' is already parsed to datetime with UTC
        # Sort by date and contract
        combined_df = combined_df.sort_values(by=['date', 'contract']).reset_index(drop=True)
        
        # Display the combined DataFrame
        print("Combined DataFrame Head:")
        print(combined_df.head())
        
        print("\nCombined DataFrame Tail:")
        print(combined_df.tail())
        
        # Save to CSV
        combined_df.to_csv('combined_es_1min_futures_data.csv', index=False)
        logger.info("Combined DataFrame saved to 'combined_es_1min_futures_data.csv'")
    else:
        logger.warning("No data fetched for the specified contracts.")
    
    # Disconnect from IB
    ib.disconnect()
    logger.info("Disconnected from IB")

if __name__ == "__main__":
    main()