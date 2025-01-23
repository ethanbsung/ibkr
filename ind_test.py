import time
import pandas as pd
import pandas_ta as ta
from ib_insync import IB, util
import numpy as np

# Connect to IBKR API
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # Update host/port/clientId as needed

# Define contract (example: ES futures)
from ib_insync import Future
contract = Future('ES', '202503', 'CME')  # Replace '202303' with the appropriate expiry

ib.qualifyContracts(contract)

# Function to fetch historical data and calculate indicators
def calculate_indicators():
    # Fetch 15-minute historical data for the past 2 hours
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='1 D',  # Adjust duration to cover enough 15-min intervals
        barSizeSetting='15 mins',
        whatToShow='TRADES',
        useRTH=False
    )
    
    # Convert data to a DataFrame
    df = util.df(bars)
    df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
    
    # Calculate RSI on 15-minute time frame
    df['RSI'] = ta.rsi(df.Close, length=16)  # You can adjust the length as needed
    
    # Print the latest RSI
    latest_data = df.iloc[-1]
    print(f"Time: {latest_data['date']}, RSI: {latest_data['RSI']:.2f}")
    return df

# Update every minute
try:
    while True:
        calculate_indicators()
        time.sleep(60)  # Wait for one minute before fetching new data
except KeyboardInterrupt:
    print("Stopped by user")
finally:
    ib.disconnect()