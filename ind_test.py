from ib_insync import *
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

# Connect to Interactive Brokers
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # Ensure TWS or IB Gateway is running

# Define the futures contract
contract = Future(symbol='ES', lastTradeDateOrContractMonth='202503', exchange='CME', currency='USD')
contract = ib.qualifyContracts(contract)[0]

# Subscribed data
data = []

# Define RSI function
def rsi(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI)."""
    delta = ohlc['close'].diff()

    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    _gain = up.ewm(com=(period - 1), min_periods=period).mean()
    _loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()

    RS = _gain / _loss
    return pd.Series(100 - (100 / (1 + RS)), name="RSI")

def vwap(ohlc: pd.DataFrame) -> pd.Series:
    """Calculate the Volume Weighted Average Price (VWAP) since market open."""
    # Convert index to US/Eastern time
    ohlc['date'] = ohlc.index  # Reset the index temporarily
    ohlc['date'] = ohlc['date'].dt.tz_convert('US/Eastern')  # Convert directly

    # Determine the start of the trading day (6 PM EST / 23:00 UTC)
    market_open = ohlc['date'].apply(
        lambda x: x.replace(hour=18, minute=0, second=0) if x.time() >= time(18, 0) else (x - timedelta(days=1)).replace(hour=18, minute=0, second=0)
    )
    
    # Filter data since the most recent market open
    ohlc = ohlc[ohlc['date'] >= market_open]

    # Calculate VWAP
    typical_price = (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3
    cumulative_vwap = (typical_price * ohlc['volume']).cumsum() / ohlc['volume'].cumsum()
    return pd.Series(cumulative_vwap, name="VWAP")

# Process live market data
def on_bar_update(bars, has_new_bar):
    global data
    # Append new bar data
    for bar in bars:
        data.append({
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })

    # Convert to DataFrame and aggregate to 15-minute bars
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Resample to 15-minute intervals
    ohlc = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Calculate RSI
    ohlc['RSI'] = rsi(ohlc)

    # Calculate VWAP since market open
    ohlc['VWAP'] = vwap(ohlc)

    # Display the latest RSI and VWAP values
    print(f"Latest RSI: {ohlc['RSI'].iloc[-1]:.2f}, Latest VWAP: {ohlc['VWAP'].iloc[-1]:.2f}")

# Subscribe to live market data
bars = ib.reqHistoricalData(
    contract=contract,
    endDateTime='',
    durationStr='1 D',
    barSizeSetting='5 secs',
    whatToShow='TRADES',
    useRTH=False,
    keepUpToDate=True
)
bars.updateEvent += on_bar_update

# Start the event loop
try:
    ib.run()
except KeyboardInterrupt:
    print("Exiting...")
    ib.disconnect()