from ib_insync import IB, Future
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

# Connect to Interactive Brokers
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=2)  # Ensure TWS or IB Gateway is running

# Define the ES futures contract.
# Adjust the lastTradeDateOrContractMonth as needed.
contract = Future(symbol='ES', lastTradeDateOrContractMonth='202503', exchange='CME', currency='USD')
contract = ib.qualifyContracts(contract)[0]

# Global container for bar data (each 5-second bar received)
data = []

# ----------------------------
# Define indicator functions
# ----------------------------

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
    # Work on a copy to avoid modifying the original DataFrame.
    df = ohlc.copy()
    # Create a column with the index as datetime and convert to US/Eastern.
    df['date'] = df.index.tz_convert('US/Eastern')
    # Determine market open time for each row.
    # Here we assume market open is at 18:00 Eastern (i.e. 23:00 UTC).
    market_open = df['date'].apply(
        lambda x: x.replace(hour=18, minute=0, second=0, microsecond=0) 
        if x.time() >= time(18, 0) 
        else (x - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
    )
    # Only use data since the most recent market open.
    df = df[df['date'] >= market_open]
    # Calculate typical price and cumulative VWAP.
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cumulative_vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return pd.Series(cumulative_vwap, name="VWAP")

# -------------------------------------------
# Callback function to process live 5-second bars
# -------------------------------------------

def on_bar_update(bars, has_new_bar):
    global data
    # Append each new bar (5-sec bar) to our data list.
    for bar in bars:
        data.append({
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })

    # Convert the collected data into a DataFrame.
    df = pd.DataFrame(data)
    if df.empty:
        return

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Resample to 15-minute bars.
    ohlc = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Calculate RSI and VWAP on the 15-minute bars.
    ohlc['RSI'] = rsi(ohlc)
    ohlc['VWAP'] = vwap(ohlc)

    # Get the latest 15-minute bar values.
    latest_price = ohlc['close'].iloc[-1]
    latest_rsi = ohlc['RSI'].iloc[-1]
    latest_vwap = ohlc['VWAP'].iloc[-1]

    # Display the latest values.
    print(f"Price: {latest_price:.2f}, Latest RSI: {latest_rsi:.2f}, Latest VWAP: {latest_vwap:.2f}")

# -------------------------------------------
# Subscribe to live 5-second bars
# -------------------------------------------

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

# -------------------------------------------
# Start the event loop to process data
# -------------------------------------------
try:
    print("Streaming live 5-second bars. Press Ctrl+C to exit.")
    ib.run()
except KeyboardInterrupt:
    print("Exiting...")
    ib.cancelHistoricalData(bars)
    ib.disconnect()