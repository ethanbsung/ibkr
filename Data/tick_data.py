from ib_insync import *
import pandas as pd
from datetime import datetime, timedelta
import calendar
import time

def get_third_friday(year, month):
    """
    Calculate the date of the third Friday of the given month and year.
    
    Parameters:
        year (int): The year of the desired month.
        month (int): The month for which to find the third Friday.
    
    Returns:
        str: The third Friday in 'YYYYMMDD' format.
    """
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    month_calendar = c.monthdatescalendar(year, month)
    fridays = [day for week in month_calendar for day in week if day.weekday() == 4 and day.month == month]
    
    if len(fridays) >= 3:
        third_friday = fridays[2]
    else:
        # Fallback in case there are less than 3 Fridays, which shouldn't happen for any month
        third_friday = fridays[-1]
    
    return third_friday.strftime('%Y%m%d')

def get_contract_month(date):
    """
    Determine the contract expiration date (YYYYMMDD) based on the date's quarter.
    ES futures have quarterly expirations: March (03), June (06),
    September (09), December (12).
    
    Parameters:
        date (datetime): The date for which to determine the contract month.
    
    Returns:
        str: The expiration date in 'YYYYMMDD' format.
    """
    year = date.year
    month = date.month
    if month <= 3:
        exp_month = 3
    elif month <= 6:
        exp_month = 6
    elif month <= 9:
        exp_month = 9
    else:
        exp_month = 12
    return get_third_friday(year, exp_month)

def get_date_ranges(start_date, end_date, delta_months=3):
    """
    Generate a list of (start, end) date tuples with delta_months intervals.
    
    Parameters:
        start_date (datetime): The start date of the range.
        end_date (datetime): The end date of the range.
        delta_months (int): The number of months per interval.
    
    Returns:
        list of tuples: Each tuple contains (start_datetime, end_datetime).
    """
    date_ranges = []
    current_start = start_date
    while current_start < end_date:
        # Calculate the next period's start month
        year = current_start.year
        month = current_start.month + delta_months
        if month > 12:
            month -= 12
            year += 1
        try:
            next_start = datetime(year, month, current_start.day, 
                                  current_start.hour, current_start.minute, current_start.second)
        except ValueError:
            # Handle end of month issues by moving to the last day of the previous month
            next_start = datetime(year, month, 1) - timedelta(seconds=1)
        
        current_end = min(next_start - timedelta(seconds=1), end_date)
        date_ranges.append((current_start, current_end))
        current_start = next_start
    return date_ranges

def main():
    # Initialize IB connection
    ib = IB()
    try:
        ib.connect(host='127.0.0.1', port=7497, clientId=1)
        print("Connected to IBKR API.")
    except Exception as e:
        print(f"Failed to connect to IBKR API: {e}")
        return

    # Define the symbol and other parameters
    symbol = 'ES'
    exchange = 'CME'  # Correct exchange for ES futures
    currency = 'USD'

    # Define the date range (last 2 years from current date)
    # Assuming current date is 2025-01-22 as per system message
    end_date = datetime(2025, 1, 22)  # Current date
    start_date = end_date - timedelta(days=2*365)  # Approximate 2 years

    # Generate 3-month intervals within the date range
    date_ranges = get_date_ranges(start_date, end_date, delta_months=3)

    all_ticks = []  # List to accumulate all tick data

    for idx, (start, end) in enumerate(date_ranges):
        contract_month_str = get_contract_month(start)
        # Define the futures contract for the given expiration date
        contract = Future(
            symbol=symbol,
            lastTradeDateOrContractMonth=contract_month_str,  # 'YYYYMMDD' format
            exchange=exchange,
            currency=currency
        )
        try:
            qualified_contracts = ib.qualifyContracts(contract)
            if not qualified_contracts:
                print(f"[{idx+1}/{len(date_ranges)}] No qualified contract found for {contract_month_str}. Skipping.")
                continue
            qualified_contract = qualified_contracts[0]
            print(f"[{idx+1}/{len(date_ranges)}] Qualified contract: {qualified_contract}")
        except Exception as e:
            print(f"Error qualifying contract {contract_month_str}: {e}")
            continue  # Skip to the next date range

        # Define the time window for tick data in space-separated format
        # Example: '20230122 15:59:59'
        start_dt_str = start.strftime("%Y%m%d %H:%M:%S")
        end_dt_str = end.strftime("%Y%m%d %H:%M:%S")

        print(f"Requesting ticks for contract {contract_month_str} from {start_dt_str} to {end_dt_str}")

        # Initialize variables for pagination
        finished = False
        req_start = start
        req_end = end
        while not finished:
            try:
                ticks = ib.reqHistoricalTicks(
                    contract=qualified_contract,
                    startDateTime=req_start.strftime("%Y%m%d %H:%M:%S"),  # Space-separated format
                    endDateTime=req_end.strftime("%Y%m%d %H:%M:%S"),      # Space-separated format
                    numberOfTicks=1000,  # Maximum ticks per request
                    useRth=False,        # Set to False to include all trading hours
                    whatToShow='TRADES', # Data type
                    ignoreSize=True
                )

                if not ticks:
                    print("No more ticks retrieved.")
                    break

                # Append retrieved ticks
                for tick in ticks:
                    all_ticks.append({
                        'time': tick.time,
                        'price': tick.price,
                        'size': tick.size,
                        'exchange': tick.exchange,
                        'specialConditions': tick.specialConditions
                    })

                print(f"Retrieved {len(ticks)} ticks.")

                # Prepare for next pagination
                last_tick_time = ticks[-1].time
                if last_tick_time >= req_end:
                    finished = True
                else:
                    # Increment by one second to avoid duplicate ticks
                    req_start = last_tick_time + timedelta(seconds=1)

                # Respect API rate limits
                time.sleep(1)  # Adjust sleep time as needed

            except Exception as e:
                print(f"Error retrieving ticks: {e}")
                print("Retrying in 5 seconds...")
                time.sleep(5)  # Wait before retrying

    # Disconnect from IBKR
    ib.disconnect()
    print("Disconnected from IBKR API.")

    # Save the accumulated tick data to a CSV file
    if all_ticks:
        df = pd.DataFrame(all_ticks)
        # Convert 'time' column to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
        # Sort by time
        df.sort_values('time', inplace=True)
        # Save to CSV
        output_filename = 'es_tick_data_last_2_years.csv'
        df.to_csv(output_filename, index=False)
        print(f"Tick data saved to {output_filename}.")
    else:
        print("No tick data was retrieved.")

if __name__ == "__main__":
    main()