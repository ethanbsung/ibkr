from ib_insync import IB, Future, util
import pandas as pd
import logging
from datetime import datetime
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_es_future_contract(expiration):
    """
    Create an ES future contract based on the expiration date.
    """
    return Future(
        symbol='ES',
        exchange='CME',
        currency='USD',
        lastTradeDateOrContractMonth=expiration,
        includeExpired=True
    )

def convert_to_utc(start_date_str, end_date_str):
    """
    Convert date strings from US/Eastern to UTC timezone and format them as 'yyyymmdd-HH:MM:SS'.
    """
    eastern = pytz.timezone('US/Eastern')
    utc = pytz.utc

    try:
        start_naive = datetime.strptime(start_date_str, '%Y%m%d %H:%M:%S')
        end_naive = datetime.strptime(end_date_str, '%Y%m%d %H:%M:%S')
    except ValueError as ve:
        logger.error(f"Date parsing error: {ve}")
        return None, None

    # Localize to US/Eastern and convert to UTC
    start_eastern = eastern.localize(start_naive)
    end_eastern = eastern.localize(end_naive)
    start_utc = start_eastern.astimezone(utc)
    end_utc = end_eastern.astimezone(utc)
    
    # Format as 'yyyymmdd-HH:MM:SS' to explicitly indicate UTC
    return start_utc.strftime('%Y%m%d-%H:%M:%S'), end_utc.strftime('%Y%m%d-%H:%M:%S')

def fetch_historical_data(ib, contract, start_date_str, end_date_str, bar_size='15 mins', what_to_show='TRADES'):
    """
    Fetch historical OHLCV data for a given contract.
    """
    start_date, end_date = convert_to_utc(start_date_str, end_date_str)
    if not start_date or not end_date:
        logger.error(f"Invalid date conversion for {contract.lastTradeDateOrContractMonth}")
        return pd.DataFrame()

    logger.info(f"Fetching data for {contract.lastTradeDateOrContractMonth} from {start_date} to {end_date}")
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_date,
            durationStr='3 M',
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=False,
            formatDate=1,
            keepUpToDate=False
        )
    except Exception as e:
        logger.error(f"Error fetching data for {contract.lastTradeDateOrContractMonth}: {e}")
        return pd.DataFrame()

    if not bars:
        logger.warning(f"No data returned for {contract.lastTradeDateOrContractMonth}")
        return pd.DataFrame()

    df = util.df(bars)
    df['contract'] = contract.lastTradeDateOrContractMonth

    logger.info(f"Fetched {len(df)} records for contract {contract.lastTradeDateOrContractMonth}")
    return df

def main():
    # Connect to IB
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=3)
        logger.info("Connected to IB")
    except Exception as e:
        logger.error(f"Failed to connect to IB: {e}")
        return

    # Define Correct Non-Overlapping Contracts
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
        (create_es_future_contract('202503'), '20241216 23:59:59', '20250316 23:59:59')   # Dec-Mar 2025
    ]
    
    # Fetch historical data for each contract
    data_frames = []
    for contract, start_date, end_date in contracts_with_dates:
        ib.qualifyContracts(contract)
        df = fetch_historical_data(ib, contract, start_date_str=start_date, end_date_str=end_date)
        if not df.empty:
            data_frames.append(df)

    # Combine DataFrames if data is available
    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date'], utc=True)
        combined_df = combined_df.sort_values(by=['date', 'contract']).reset_index(drop=True)

        # Display the combined DataFrame
        print(combined_df.head())
        print(combined_df.tail())

        # Save to CSV
        combined_df.to_csv('combined_es_futures_data.csv', index=False)
        logger.info("Combined DataFrame saved to 'combined_es_futures_data.csv'")
    else:
        logger.warning("No data fetched for the specified contracts.")
    
    # Disconnect from IB
    ib.disconnect()
    logger.info("Disconnected from IB")

if __name__ == "__main__":
    main()