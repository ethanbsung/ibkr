import pandas as pd
import numpy as np
import datetime
from ib_insync import *
import logging
import sys
import pytz

# --- Configuration Parameters ---
IB_HOST = '127.0.0.1'        # IBKR Gateway/TWS host
IB_PORT = 7497               # IBKR Gateway/TWS paper trading port
CLIENT_ID = 1                # Unique client ID

DATA_SYMBOL = 'ES'           # E-mini S&P 500 for data
DATA_EXPIRY = '202503'       # March 2025
DATA_EXCHANGE = 'CME'        # Exchange for ES

EXEC_SYMBOL = 'MES'          # Micro E-mini S&P 500 for execution
EXEC_EXPIRY = '202503'       # March 2025
EXEC_EXCHANGE = 'CME'        # Exchange for MES
CURRENCY = 'USD'

INITIAL_CASH = 5000          # Starting cash
POSITION_SIZE = 1            # Number of MES contracts per trade

BOLLINGER_PERIOD = 15
BOLLINGER_STDDEV = 2
STOP_LOSS_DISTANCE = 5        # Points away from entry
TAKE_PROFIT_DISTANCE = 10     # Points away from entry

# RTH: 09:30 - 16:00 ET, Monday to Friday
RTH_START = datetime.time(9, 00)
RTH_END = datetime.time(15, 59)
EASTERN = pytz.timezone('US/Eastern')

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# --- Connect to IBKR ---
ib = IB()
logger.info("Connecting to IBKR...")
try:
    ib.connect(host=IB_HOST, port=IB_PORT, clientId=CLIENT_ID)
    logger.info("Connected to IBKR.")
except Exception as e:
    logger.error(f"Failed to connect to IBKR: {e}")
    sys.exit(1)

# --- Define Contracts ---
es_contract = Future(symbol=DATA_SYMBOL, lastTradeDateOrContractMonth=DATA_EXPIRY, exchange=DATA_EXCHANGE, currency=CURRENCY)
mes_contract = Future(symbol=EXEC_SYMBOL, lastTradeDateOrContractMonth=EXEC_EXPIRY, exchange=EXEC_EXCHANGE, currency=CURRENCY)

# Qualify Contracts
try:
    qualified_contracts = ib.qualifyContracts(es_contract, mes_contract)
    es_contract = qualified_contracts[0]
    mes_contract = qualified_contracts[1]
    logger.info(f"Qualified ES Contract: {es_contract}")
    logger.info(f"Qualified MES Contract: {mes_contract}")
except Exception as e:
    logger.error(f"Error qualifying contracts: {e}")
    ib.disconnect()
    sys.exit(1)

# --- Request Historical Data for ES (Full Data) ---
try:
    logger.info("Requesting historical ES data (including ETH)...")
    bars_30m_full = ib.reqHistoricalData(
        contract=es_contract,
        endDateTime='',
        durationStr='90 D',
        barSizeSetting='30 mins',
        whatToShow='TRADES',
        useRTH=False,                     
        formatDate=1,
        keepUpToDate=False
    )

    if bars_30m_full:
        df_30m_full = util.df(bars_30m_full)
        df_30m_full.set_index('date', inplace=True)
        df_30m_full.sort_index(inplace=True)
        # Ensure the index is timezone-aware (UTC)
        df_30m_full.index = pd.to_datetime(df_30m_full.index, utc=True)
        logger.info("Successfully retrieved full 30m historical data (including ETH).")
    else:
        logger.warning("No 30m historical data received.")
        df_30m_full = pd.DataFrame()

except Exception as e:
    logger.error(f"Error requesting historical data: {e}")
    ib.disconnect()
    sys.exit(1)

def calculate_bollinger_bands(df, period=15, stddev=2):
    if len(df) < period:
        return df
    df['ma'] = df['close'].rolling(window=period).mean()
    df['std'] = df['close'].rolling(window=period).std()
    df['upper_band'] = df['ma'] + (stddev * df['std'])
    df['lower_band'] = df['ma'] - (stddev * df['std'])
    return df

def filter_rth(df):
    if df.index.tz is None:
        df = df.tz_localize('UTC')
    else:
        df = df.tz_convert('UTC')

    df_eastern = df.copy()
    df_eastern.index = df_eastern.index.tz_convert(EASTERN)

    df_eastern = df_eastern[df_eastern.index.weekday < 5]
    df_rth = df_eastern.between_time(RTH_START, RTH_END)
    df_rth.index = df_rth.index.tz_convert('UTC')
    return df_rth

# Calculate Bollinger Bands on Full Data
df_30m_full = calculate_bollinger_bands(df_30m_full, BOLLINGER_PERIOD, BOLLINGER_STDDEV)

# Print out the DataFrame to see what's in it after historical load
logger.info("df_30m_full after initial Bollinger calculation:")
print(df_30m_full.tail(50))

df_30m_rth = filter_rth(df_30m_full)
logger.info("RTH filtering applied to 30-minute data for trade execution.")

cash = INITIAL_CASH
balance_series = [INITIAL_CASH]
position = None
pending_order = False
current_30min_start = None
current_30min_bars = []

def on_trade_filled(trade):
    global position, pending_order
    fill = trade.fills[-1]  # Get the latest fill
    logger.info(f"Trade Filled - Order ID {trade.order.orderId}: {trade.order.action} {fill.size} @ {fill.price}")
    if trade.isFilled():
        entry_price = fill.price
        action = trade.order.action.upper()
        position_type = 'LONG' if action == 'BUY' else 'SHORT'
        logger.info(f"Entered {position_type} position at {entry_price}")
        position = position_type
        pending_order = False

def on_order_status(trade):
    global position, pending_order
    logger.info(f"Trade Status Update - Order ID {trade.order.orderId}: {trade.orderStatus.status}")
    if trade.orderStatus.status in ('Cancelled', 'Inactive'):
        logger.info(f"Order ID {trade.order.orderId} has been {trade.orderStatus.status.lower()}.")
        if position is None:
            pending_order = False

def place_bracket_order(action, current_price):
    global pending_order
    if action.upper() not in ['BUY', 'SELL']:
        logger.error(f"Invalid action: {action}. Must be 'BUY' or 'SELL'.")
        return

    if action.upper() == 'BUY':
        take_profit_price = current_price + TAKE_PROFIT_DISTANCE
        stop_loss_price = current_price - STOP_LOSS_DISTANCE
    else:
        take_profit_price = current_price - TAKE_PROFIT_DISTANCE
        stop_loss_price = current_price + STOP_LOSS_DISTANCE

    try:
        # Create a standard bracket order with the parent as a limit order
        bracket = ib.bracketOrder(
            action=action.upper(),
            quantity=POSITION_SIZE,
            limitPrice=current_price,      # Parent is a limit order at current_price
            takeProfitPrice=take_profit_price,
            stopLossPrice=stop_loss_price
        )

        # Place the parent order and get the Trade object
        parent_trade = ib.placeOrder(mes_contract, bracket[0])
        logger.info(f"Placed Parent {bracket[0].orderType} Order ID {bracket[0].orderId} for {bracket[0].action} at {bracket[0].lmtPrice}")

        # Attach event handlers to the Trade object
        parent_trade.filledEvent += on_trade_filled
        parent_trade.statusEvent += on_order_status

        # Place Take-Profit and Stop-Loss Orders
        take_profit_trade = ib.placeOrder(mes_contract, bracket[1])
        logger.info(f"Placed Take-Profit {bracket[1].orderType} Order ID {bracket[1].orderId} for {bracket[1].action} at {bracket[1].lmtPrice}")

        stop_loss_trade = ib.placeOrder(mes_contract, bracket[2])
        logger.info(f"Placed Stop-Loss {bracket[2].orderType} Order ID {bracket[2].orderId} for {bracket[2].action} at {bracket[2].auxPrice}")

        pending_order = True
        logger.info("Bracket order placed successfully and event handlers attached.")

    except Exception as e:
        logger.error(f"Failed to place bracket order: {e}")
        pending_order = False


def is_rth(timestamp):
    if timestamp is None:
        return False
    ts_eastern = timestamp.astimezone(EASTERN)
    return ts_eastern.weekday() < 5 and RTH_START <= ts_eastern.time() < RTH_END

def execute_trade(action, current_price, current_time):
    global pending_order
    if pending_order:
        logger.info("There is already a pending order. Skipping new trade execution.")
        return

    if not is_rth(current_time):
        logger.info("Current time is not in RTH. Skipping trade entry.")
        return

    logger.info(f"Entry Signal: {action}")
    logger.info(f"Current Price: {current_price}")

    if current_time not in df_30m_full.index:
        logger.warning(f"No data row for {current_time} in df_30m_full. Skipping trade.")
        return

    row = df_30m_full.loc[current_time]
    upper_band = row.get('upper_band', np.nan)
    lower_band = row.get('lower_band', np.nan)

    if pd.isna(upper_band) or pd.isna(lower_band):
        logger.warning(f"No valid Bollinger Bands data (NaN) available for {current_time}. Skipping trade.")
        return

    logger.info(f"Lower Threshold (Bollinger Band): {lower_band}")
    logger.info(f"Upper Threshold (Bollinger Band): {upper_band}")

    place_bracket_order(action, current_price)

def on_realtime_bar(ticker, hasNewBar):
    global current_30min_start, current_30min_bars, df_30m_full, df_30m_rth, position, cash, pending_order

    try:
        if hasNewBar:
            if len(ticker) == 0:
                logger.warning("No bars received in RealTimeBarList.")
                return

            bar = ticker[-1]
            if bar.time.tzinfo is None:
                bar_time = pytz.UTC.localize(bar.time.replace(second=0, microsecond=0))
            else:
                bar_time = bar.time.replace(second=0, microsecond=0)

            minute = bar_time.minute
            candle_start_minute = (minute // 30) * 30
            candle_start_time = bar_time.replace(minute=candle_start_minute, second=0, microsecond=0)

            # If we detect a new 30-minute candle
            if current_30min_start != candle_start_time:
                # Finalize the previous candle when a new one starts
                if current_30min_start is not None and current_30min_bars:
                    # The row should already be updated continuously, but let's confirm final calculation:
                    open_30 = current_30min_bars[0]['open']
                    high_30 = max(b['high'] for b in current_30min_bars)
                    low_30 = min(b['low'] for b in current_30min_bars)
                    close_30 = current_30min_bars[-1]['close']
                    volume_30 = sum(b['volume'] for b in current_30min_bars)

                    # Update final values for the just-closed candle
                    df_30m_full.loc[current_30min_start, ['open', 'high', 'low', 'close', 'volume']] = [open_30, high_30, low_30, close_30, volume_30]

                    # Recalculate Bollinger Bands only if we have enough data
                    if len(df_30m_full) >= BOLLINGER_PERIOD:
                        df_30m_full = calculate_bollinger_bands(df_30m_full, BOLLINGER_PERIOD, BOLLINGER_STDDEV)

                    logger.info("df_30m_full after finalizing the old bar:")
                    print(df_30m_full.tail(10))

                    # Re-filter RTH data
                    df_30m_rth = filter_rth(df_30m_full)

                    if is_rth(current_30min_start):
                        current_price = close_30
                        current_time = current_30min_start
                        logger.info(f"\nNew 30-min bar closed at {current_time} UTC with close price: {current_price}")

                        if current_time in df_30m_full.index:
                            row = df_30m_full.loc[current_time]
                            upper_band = row.get('upper_band', np.nan)
                            lower_band = row.get('lower_band', np.nan)
                        else:
                            upper_band = np.nan
                            lower_band = np.nan

                        if not pd.isna(upper_band) and not pd.isna(lower_band):
                            logger.info(f"Bollinger Bands - Upper: {upper_band}, Lower: {lower_band}")
                            logger.info(f"Current Price: {current_price}")

                            if position is None and not pending_order:
                                if current_price < lower_band:
                                    # Enter Long during RTH
                                    execute_trade('BUY', current_price, current_time)
                                elif current_price > upper_band:
                                    # Enter Short during RTH
                                    execute_trade('SELL', current_price, current_time)
                                else:
                                    logger.info("No trading signal detected.")
                        else:
                            logger.warning(f"No Bollinger Bands data available for {current_time}. Skipping trade.")
                    else:
                        logger.info(f"New 30-min bar at {current_30min_start} UTC is outside RTH. No trade executed.")

                # Start a new 30-minute candle
                current_30min_start = candle_start_time
                current_30min_bars = []

            # Append the current 5-second bar data to the ongoing 30-min candle
            current_30min_bars.append({
                'open': bar.open_,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            })

            # Continuously update the current candle row in df_30m_full
            # This ensures that if we lose connection or something, we always have the latest partial data.
            if current_30min_start is not None:
                open_30 = current_30min_bars[0]['open']
                high_30 = max(b['high'] for b in current_30min_bars)
                low_30 = min(b['low'] for b in current_30min_bars)
                close_30 = current_30min_bars[-1]['close']
                volume_30 = sum(b['volume'] for b in current_30min_bars)

                # Update or create the row for the current candle in progress
                df_30m_full.loc[current_30min_start, ['open', 'high', 'low', 'close', 'volume']] = [open_30, high_30, low_30, close_30, volume_30]

    except Exception as e:
        logger.error(f"Error in on_realtime_bar handler: {e}")

try:
    logger.info("Requesting real-time 5-second bars (including ETH)...")
    ticker = ib.reqRealTimeBars(
        contract=es_contract,
        barSize=5,
        whatToShow='TRADES',
        useRTH=False,
        realTimeBarsOptions=[]
    )
    ticker.updateEvent += on_realtime_bar
    logger.info("Real-time bar handler assigned.")
except Exception as e:
    logger.error(f"Failed to subscribe to real-time bars: {e}")
    ib.disconnect()
    sys.exit(1)

logger.info("Starting event loop...")
try:
    ib.run()
except KeyboardInterrupt:
    logger.info("Interrupt received, shutting down...")
finally:
    ib.disconnect()
    logger.info("Disconnected from IBKR.")