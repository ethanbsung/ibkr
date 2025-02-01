from ib_insync import *
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import pytz
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger()

# Connect to Interactive Brokers
ib = IB()
try:
    ib.connect('127.0.0.1', 7497, clientId=2)
except Exception as e:
    logger.error(f"Could not connect to IBKR: {e}")
    sys.exit(1)

# Define the ES futures contract.
contract = Future(symbol='ES', lastTradeDateOrContractMonth='202503', exchange='CME', currency='USD')
contract = ib.qualifyContracts(contract)[0]

# Global variables for continuous tick-level indicator updates.
rolling_prices = []   # For RSI calculation: store the last N tick prices.
cum_volume = 0.0      # For VWAP: cumulative volume since market open.
cum_price_volume = 0.0  # For VWAP: cumulative sum of price * volume.
session_start = None  # To keep track of the current session's start time.

# --- Indicator Function for RSI ---
def rsi(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI) on a price series provided as a DataFrame."""
    delta = ohlc['close'].diff()

    # Make copies for up and down moves.
    up, down = delta.copy(), delta.copy()
    # Only keep positive gains; set negative gains to 0.
    up[up < 0] = 0
    # Only keep negative losses; set positive losses to 0.
    down[down > 0] = 0

    # Calculate the exponential weighted moving averages.
    _gain = up.ewm(com=(period - 1), min_periods=period).mean()
    _loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()

    RS = _gain / _loss
    return pd.Series(100 - (100 / (1 + RS)), name="RSI")

# --- Function to update indicators directly from tick data ---
def update_indicators(tick):
    """
    Update RSI and VWAP using each incoming tick.
    This function maintains a rolling window for RSI and cumulative values for VWAP.
    """
    global rolling_prices, cum_volume, cum_price_volume, session_start

    # --- Determine session boundaries for VWAP reset ---
    # Assume tick.time is in UTC. Convert to Eastern to define the market open.
    eastern = pytz.timezone("US/Eastern")
    tick_time = tick.time  # tick.time is already a datetime with tz info (usually UTC)
    tick_time_eastern = tick_time.astimezone(eastern)

    # Define market open as 18:00 Eastern.
    if tick_time_eastern.time() >= time(18, 0):
        market_open_eastern = tick_time_eastern.replace(hour=18, minute=0, second=0, microsecond=0)
    else:
        market_open_eastern = (tick_time_eastern - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)

    # Convert market open back to UTC.
    market_open_utc = market_open_eastern.astimezone(pytz.UTC)

    # If session_start is not set or the tick is before the current session, reset the cumulative sums.
    if (session_start is None) or (tick_time < session_start):
        session_start = market_open_utc
        cum_volume = 0.0
        cum_price_volume = 0.0
        rolling_prices = []  # reset the RSI price window

    # --- Update rolling window for RSI ---
    rolling_prices.append(tick.price)
    # Optionally, limit the window size (e.g., last 200 ticks)
    if len(rolling_prices) > 200:
        rolling_prices = rolling_prices[-200:]

    # --- Update VWAP cumulative values ---
    cum_volume += tick.size
    cum_price_volume += tick.price * tick.size
    current_vwap = cum_price_volume / cum_volume if cum_volume > 0 else tick.price

    # --- Calculate RSI on the tick prices ---
    # We need at least (period + 1) data points; here period is 14.
    if len(rolling_prices) >= 15:
        prices_df = pd.DataFrame({'close': rolling_prices})
        current_rsi_series = rsi(prices_df, period=14)
        current_rsi = current_rsi_series.iloc[-1]
    else:
        current_rsi = np.nan  # Not enough data to compute RSI

    # Print (or otherwise use) the current indicators.
    print(f"Latest Price: {tick.price:.2f}, RSI: {current_rsi:.2f}, VWAP: {current_vwap:.2f}")

# --- Tick Data Event Handler ---
def onTickByTick(tick):
    update_indicators(tick)

# --- Preload Historical Data (optional) ---
# You might want to seed your rolling window and cumulative values using historical data.
# (This part remains similar to your previous code if desired.)
try:
    historicalBars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='1 D',
        barSizeSetting='5 secs',
        whatToShow='TRADES',
        useRTH=False,
        formatDate=1
    )
except Exception as e:
    logger.error(f"Error fetching historical data: {e}")
    historicalBars = []

if historicalBars:
    hist_df = util.df(historicalBars)
    # Convert the 'date' column to a timezone-aware datetime in UTC.
    hist_df['date'] = pd.to_datetime(hist_df['date']).dt.tz_convert('UTC')
    # Seed our rolling_prices and VWAP cumulative sums using historical close prices and volume.
    for _, row in hist_df.iterrows():
        rolling_prices.append(row['close'])
        cum_volume += row['volume']
        cum_price_volume += row['close'] * row['volume']
    # Print initial indicator values if possible.
    if len(rolling_prices) >= 15:
        prices_df = pd.DataFrame({'close': rolling_prices})
        init_rsi = rsi(prices_df, period=14).iloc[-1]
    else:
        init_rsi = np.nan
    init_vwap = cum_price_volume / cum_volume if cum_volume > 0 else np.nan
    print(f"Initial Price: {rolling_prices[-1]:.2f}, RSI: {init_rsi:.2f}, VWAP: {init_vwap:.2f}")
else:
    logger.warning("No historical data returned; starting with an empty dataset.")

# --- Subscribe to Tick-by-Tick Data ---
tickSub = ib.reqTickByTickData(contract, tickType="Last", numberOfTicks=0, ignoreSize=False)
tickSub.updateEvent += onTickByTick

print("Streaming tick data for ES futures. Press Ctrl+C to exit.")
try:
    ib.run()  # Process events continuously.
except KeyboardInterrupt:
    print("Exiting tick data stream...")
    ib.cancelTickByTickData(tickSub)
    ib.disconnect()