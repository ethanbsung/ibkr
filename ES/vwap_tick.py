from ib_insync import *
import csv
from datetime import datetime, timedelta
import time

# Function to format datetime for IB API
def format_datetime(dt):
    return dt.strftime('%Y%m%d %H:%M:%S')

# Connect to IBKR
ib = IB()
ib.connect(host='127.0.0.1', port=7497, clientId=1)

# Define the contract
contract = Future(
    symbol='ES', 
    lastTradeDateOrContractMonth='20250321', 
    exchange='CME', 
    currency='USD'
)
ib.qualifyContracts(contract)

# Define the CSV file name with a timestamp
csv_filename = f"ticks_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Open the CSV file for writing and write the header
with open(csv_filename, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Price', 'Size', 'Exchange'])

# Initialize variables for looping
startDateTime = ""  # Start from the earliest available data
endDateTime = "20250122 15:59:59"  # Your specified end date and time
batch_size = 1000  # Number of ticks per request
all_ticks_retrieved = False  # Flag to control the loop

print("Starting to retrieve all tick data...")

while not all_ticks_retrieved:
    try:
        # Request a batch of historical ticks
        ticks = ib.reqHistoricalTicks(
            contract=contract,
            startDateTime=startDateTime,
            endDateTime=endDateTime,
            numberOfTicks=batch_size,
            useRth=False,
            whatToShow='TRADES',  # Data type
            ignoreSize=True
        )
        
        if not ticks:
            print("No more ticks retrieved. Completed data collection.")
            break  # Exit the loop if no more ticks are returned

        print(f"Retrieved {len(ticks)} ticks.")

        # Open the CSV file in append mode
        with open(csv_filename, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            for tick in ticks:
                # Convert time to a readable format if it's a datetime object
                if isinstance(tick.time, datetime):
                    tick_time = tick.time.strftime('%Y-%m-%d %H:%M:%S.%f')
                else:
                    tick_time = str(tick.time)
                
                writer.writerow([tick_time, tick.price, tick.size, tick.exchange])

        # Update startDateTime to the last tick's time plus a small delta to avoid duplicates
        last_tick_time = ticks[-1].time
        if isinstance(last_tick_time, datetime):
            # Add one microsecond to avoid retrieving the same tick again
            new_start = last_tick_time + timedelta(microseconds=1)
            startDateTime = format_datetime(new_start)
        else:
            # If tick.time is not a datetime object, handle accordingly
            # This is unlikely, but added for completeness
            # Assume tick.time is in 'YYYYMMDD HH:MM:SS' format
            new_start = datetime.strptime(last_tick_time, '%Y%m%d %H:%M:%S') + timedelta(microseconds=1)
            startDateTime = format_datetime(new_start)
        
        print(f"Next request will start from: {startDateTime}")

        # Optional: Sleep briefly to respect API rate limits
        time.sleep(0.1)  # Adjust as necessary

    except Exception as e:
        print(f"Error retrieving historical ticks: {e}")
        print("Retrying in 5 seconds...")
        time.sleep(5)  # Wait before retrying

# Disconnect from IBKR
ib.disconnect()
print("Disconnected from IBKR.")
print(f"All tick data has been saved to {csv_filename}.")