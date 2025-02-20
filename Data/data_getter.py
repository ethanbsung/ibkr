from ib_insync import IB, Future, util
import pandas as pd
import logging
from datetime import datetime, timedelta
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
        exchange='CME',  # Use 'GLOBEX' for ES futures
        currency='USD',
        lastTradeDateOrContractMonth=expiration,
        includeExpired=True
    )

def fetch_historical_data_chunked(ib, contract, start_date_str, end_date_str, bar_size='1 min', what_to_show='TRADES'):
    """
    Fetch historical OHLCV data for a given contract in chunks.
    The start_date_str and end_date_str are in 'YYYYMMDD HH:MM:SS' format (US/Eastern time).
    Data is fetched in one-week chunks to avoid IB's duration limits for 1-minute bars.
    """
    # Convert input strings to timezone-aware datetime objects in US/Eastern, then to UTC.
    eastern = pytz.timezone('US/Eastern')
    utc = pytz.utc
    try:
        start_naive = datetime.strptime(start_date_str, '%Y%m%d %H:%M:%S')
        end_naive = datetime.strptime(end_date_str, '%Y%m%d %H:%M:%S')
    except ValueError as ve:
        logger.error(f"Date parsing error: {ve}")
        return pd.DataFrame()

    start_eastern = eastern.localize(start_naive)
    end_eastern = eastern.localize(end_naive)
    start_utc = start_eastern.astimezone(utc)
    end_utc = end_eastern.astimezone(utc)

    all_bars = []
    current_end = end_utc

    logger.info(f"Fetching data for contract {contract.lastTradeDateOrContractMonth} "
                f"from {start_utc.strftime('%Y%m%d-%H:%M:%S')} to {end_utc.strftime('%Y%m%d-%H:%M:%S')}")
    
    # Loop backward from the end date in one-week chunks.
    while current_end > start_utc:
        # IB API expects the endDateTime as a string in UTC format: YYYYMMDD-HH:MM:SS
        end_str = current_end.strftime('%Y%m%d-%H:%M:%S')
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_str,
                durationStr='1 W',  # Request one week of data
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=False,
                formatDate=1,
                keepUpToDate=False
            )
        except Exception as e:
            logger.error(f"Error fetching data chunk ending {end_str} for contract {contract.lastTradeDateOrContractMonth}: {e}")
            break

        if not bars:
            logger.warning(f"No data returned for chunk ending {end_str} for contract {contract.lastTradeDateOrContractMonth}")
            break

        df_chunk = util.df(bars)
        all_bars.append(df_chunk)
        logger.info(f"Fetched {len(df_chunk)} records for chunk ending {end_str} for contract {contract.lastTradeDateOrContractMonth}")

        # Update current_end to one minute before the earliest bar in this chunk.
        earliest_bar_time = df_chunk['date'].min()
        current_end = earliest_bar_time - timedelta(minutes=1)

        # If the updated current_end is before the desired start, exit the loop.
        if current_end < start_utc:
            break

    if all_bars:
        df_all = pd.concat(all_bars, ignore_index=True)
        df_all['contract'] = contract.lastTradeDateOrContractMonth
        logger.info(f"Total records fetched for contract {contract.lastTradeDateOrContractMonth}: {len(df_all)}")
        return df_all
    else:
        logger.warning(f"No data fetched for contract {contract.lastTradeDateOrContractMonth}")
        return pd.DataFrame()

def main():
    # Connect to IB
    ib = IB()
    try:
        ib.connect('127.0.0.1', 4002, clientId=3)
        logger.info("Connected to IB")
    except Exception as e:
        logger.error(f"Failed to connect to IB: {e}")
        return

    # Define non-overlapping contracts and their corresponding US/Eastern date ranges.
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
        (create_es_future_contract('202503'), '20241217 23:59:59', '20250210 15:59:59')   # Dec-Mar 2025
    ]
    
    # Fetch historical data for each contract.
    data_frames = []
    for contract, start_date, end_date in contracts_with_dates:
        # Qualify the contract so IB fills in the necessary details.
        try:
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                logger.error(f"Could not qualify contract: {contract}")
                continue
        except Exception as e:
            logger.error(f"Error qualifying contract {contract}: {e}")
            continue

        df = fetch_historical_data_chunked(ib, contract, start_date_str=start_date, end_date_str=end_date)
        if not df.empty:
            data_frames.append(df)

    # Combine DataFrames if data is available.
    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date'], utc=True)
        combined_df = combined_df.sort_values(by=['date', 'contract']).reset_index(drop=True)

        # Display the combined DataFrame.
        print("Head of combined DataFrame:")
        print(combined_df.head())
        print("\nTail of combined DataFrame:")
        print(combined_df.tail())

        # Save to CSV.
        csv_filename = 'ib_es_1m_data.csv'
        combined_df.to_csv(csv_filename, index=False)
        logger.info(f"Combined DataFrame saved to '{csv_filename}'")
    else:
        logger.warning("No data fetched for the specified contracts.")
    
    # Disconnect from IB.
    ib.disconnect()
    logger.info("Disconnected from IB")

if __name__ == "__main__":
    main()