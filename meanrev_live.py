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

PARENT_ORDER_TIMEOUT = 300    # 5 minutes timeout for parent order fill

# RTH: 09:30 - 16:00 ET, Monday to Friday
RTH_START = datetime.time(9, 30)
RTH_END = datetime.time(16, 0)
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
        durationStr='90 D',               # Increased duration to ensure sufficient data for Bollinger Bands
        barSizeSetting='30 mins',
        whatToShow='TRADES',
        useRTH=False,                     # Include ETH data
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

# --- Define Helper Functions ---

def calculate_bollinger_bands(df, period=15, stddev=2):
    """
    Calculate Bollinger Bands for the given DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'close' prices.
        period (int): Rolling window period for moving average.
        stddev (float): Number of standard deviations for bands.

    Returns:
        pd.DataFrame: DataFrame with Bollinger Bands added.
    """
    df['ma'] = df['close'].rolling(window=period).mean()
    df['std'] = df['close'].rolling(window=period).std()
    df['upper_band'] = df['ma'] + (stddev * df['std'])
    df['lower_band'] = df['ma'] - (stddev * df['std'])
    return df

def filter_rth(df):
    """
    Filters the DataFrame to include only Regular Trading Hours (09:30 - 16:00 ET) on weekdays.

    Parameters:
        df (pd.DataFrame): The input DataFrame with a timezone-aware datetime index.

    Returns:
        pd.DataFrame: The filtered DataFrame containing only RTH data.
    """
    # Ensure the index is timezone-aware
    if df.index.tz is None:
        # Assume the data is in UTC if no timezone is set
        df = df.tz_localize('UTC')
    else:
        # Convert to UTC to standardize
        df = df.tz_convert('UTC')

    # Convert index to US/Eastern timezone for filtering
    df_eastern = df.copy()
    df_eastern.index = df_eastern.index.tz_convert(EASTERN)

    # Filter for weekdays (Monday=0 to Friday=4)
    df_eastern = df_eastern[df_eastern.index.weekday < 5]

    # Filter for RTH hours: 09:30 to 16:00
    df_rth = df_eastern.between_time(RTH_START, RTH_END)

    # Convert back to UTC for consistency in further processing
    df_rth.index = df_rth.index.tz_convert('UTC')

    return df_rth

# --- Calculate Bollinger Bands on Full Data ---
df_30m_full = calculate_bollinger_bands(df_30m_full, BOLLINGER_PERIOD, BOLLINGER_STDDEV)
df_30m_full.dropna(inplace=True)
logger.info("Bollinger Bands calculated on full 30m data (including ETH).")

# --- Filter RTH Data for Trade Execution ---
df_30m_rth = filter_rth(df_30m_full)
logger.info("RTH filtering applied to 30-minute data for trade execution.")

# --- Initialize Variables ---
cash = INITIAL_CASH
balance_series = [INITIAL_CASH]
position = None  # No open position initially
pending_order = False  # To track if there's a pending order

scheduled_cancellations = {}
current_30min_start = None
current_30min_bars = []

# --- Timeout Cancellation Function ---
def schedule_order_cancellation(trade, parent_order, timeout=PARENT_ORDER_TIMEOUT):
    def cancel_order():
        if not trade.isFilled():
            logger.info(f"Parent Order ID {parent_order.orderId} not filled within {timeout} seconds. Cancelling order.")
            ib.cancelOrder(parent_order)
            scheduled_cancellations.pop(parent_order.orderId, None)

    scheduled = ib.callLater(timeout, cancel_order)
    scheduled_cancellations[parent_order.orderId] = scheduled
    logger.info(f"Scheduled cancellation for Parent Order ID {parent_order.orderId} in {timeout} seconds.")

def cancel_scheduled_cancellation(order_id):
    scheduled = scheduled_cancellations.get(order_id, None)
    if scheduled:
        ib.cancelScheduledCall(scheduled)
        scheduled_cancellations.pop(order_id, None)
        logger.info(f"Cancelled scheduled cancellation for Parent Order ID {order_id}.")

# --- Callbacks ---
def on_trade_filled(trade, fill):
    """
    Callback when a trade is filled.
    """
    logger.info(f"Trade Filled - Order ID {trade.order.orderId}: {trade.order.action} {fill.size} @ {fill.price}")
    if trade.isFilled():
        entry_price = fill.price
        action = trade.order.action.upper()
        position_type = 'LONG' if action == 'BUY' else 'SHORT'
        logger.info(f"Entered {position_type} position at {entry_price}")
        global position, pending_order
        position = position_type
        pending_order = False
        cancel_scheduled_cancellation(trade.order.orderId)

def on_order_status(trade, fill):
    """
    Callback when an order's status changes.
    """
    logger.info(f"Trade Status Update - Order ID {trade.order.orderId}: {trade.order.status}")
    if trade.order.status in ('Cancelled', 'Inactive'):
        logger.info(f"Order ID {trade.order.orderId} has been {trade.order.status.lower()}.")
        global position, pending_order
        position = None
        pending_order = False
        cancel_scheduled_cancellation(trade.order.orderId)

# --- Place Bracket Order with Market Parent and Limit/Stop Children ---
def place_bracket_order(action, current_price):
    """
    Places a bracket order:
    - Parent: MarketOrder (immediate execution during RTH)
    - Child 1: Take Profit (LimitOrder)
    - Child 2: Stop Loss (StopOrder)
    """
    if action.upper() not in ['BUY', 'SELL']:
        logger.error(f"Invalid action: {action}. Must be 'BUY' or 'SELL'.")
        return

    if action.upper() == 'BUY':
        take_profit_price = current_price + TAKE_PROFIT_DISTANCE
        stop_loss_price = current_price - STOP_LOSS_DISTANCE
    else:
        take_profit_price = current_price - TAKE_PROFIT_DISTANCE
        stop_loss_price = current_price + STOP_LOSS_DISTANCE

    # Parent (Market) Order
    parent = MarketOrder(action=action.upper(), totalQuantity=POSITION_SIZE)
    parent.transmit = False
    parent.outsideRth = True

    # Take Profit Order (Limit)
    take_profit_action = 'SELL' if action.upper() == 'BUY' else 'BUY'
    take_profit_order = LimitOrder(
        action=take_profit_action,
        totalQuantity=POSITION_SIZE,
        lmtPrice=take_profit_price,
        parentId=parent.orderId,
        transmit=False
    )
    take_profit_order.outsideRth = True

    # Stop Loss Order (Stop)
    stop_loss_action = 'SELL' if action.upper() == 'BUY' else 'BUY'
    stop_loss_order = StopOrder(
        action=stop_loss_action,
        totalQuantity=POSITION_SIZE,
        stopPrice=stop_loss_price,
        parentId=parent.orderId,
        transmit=True
    )
    stop_loss_order.outsideRth = True

    try:
        # Place Parent Order
        trade_parent = ib.placeOrder(mes_contract, parent)
        logger.info(f"Placed Parent {action.upper()} Market Order ID {parent.orderId}")

        # Schedule cancellation if not filled
        schedule_order_cancellation(trade_parent, parent)

        # Place child orders
        trade_take_profit = ib.placeOrder(mes_contract, take_profit_order)
        logger.info(f"Placed Take Profit Limit Order ID {take_profit_order.orderId} at {take_profit_price}")

        trade_stop_loss = ib.placeOrder(mes_contract, stop_loss_order)
        logger.info(f"Placed Stop Loss Stop Order ID {stop_loss_order.orderId} at {stop_loss_price}")

        # Attach event handlers
        trade_parent.filledEvent += on_trade_filled
        trade_parent.statusEvent += on_order_status
        trade_take_profit.filledEvent += on_trade_filled
        trade_take_profit.statusEvent += on_order_status
        trade_stop_loss.filledEvent += on_trade_filled
        trade_stop_loss.statusEvent += on_order_status

        global pending_order
        pending_order = True
        logger.info("Bracket order placed successfully.")

    except Exception as e:
        logger.error(f"Failed to place bracket order: {e}")
        pending_order = False

# --- Execute Trade During RTH ---
def is_rth(timestamp):
    """
    Check if the given timestamp (UTC) is within RTH (09:30-16:00 ET Monday-Friday).
    """
    if timestamp is None:
        return False
    # Convert to Eastern time
    ts_eastern = timestamp.astimezone(EASTERN)
    # Check weekday (Mon-Fri) and time
    if ts_eastern.weekday() < 5 and RTH_START <= ts_eastern.time() < RTH_END:
        return True
    return False

def execute_trade(action, current_price, current_time):
    """
    Execute a trade by placing a bracket order if conditions are met.
    """
    global pending_order
    if pending_order:
        logger.info("There is already a pending order. Skipping new trade execution.")
        return

    # Ensure we are in RTH for entry
    if not is_rth(current_time):
        logger.info("Current time is not in RTH. Skipping trade entry.")
        return

    logger.info(f"Entry Signal: {action}")
    logger.info(f"Current Price: {current_price}")
    # Retrieve Bollinger Bands from full data
    try:
        upper_band = df_30m_full.loc[current_time, 'upper_band']
        lower_band = df_30m_full.loc[current_time, 'lower_band']
    except KeyError:
        logger.warning(f"No Bollinger Bands data available for {current_time}. Skipping trade.")
        return

    logger.info(f"Lower Threshold (Bollinger Band): {lower_band}")
    logger.info(f"Upper Threshold (Bollinger Band): {upper_band}")

    place_bracket_order(action, current_price)

# --- Real-Time Bar Handler ---
def on_realtime_bar(ticker, hasNewBar):
    """
    Handler for incoming real-time bars.
    Logs each 5-second bar and constructs 30-minute bars from them to evaluate trade signals.
    """
    global current_30min_start, current_30min_bars, df_30m_full, df_30m_rth, position, cash, pending_order

    try:
        if hasNewBar:
            if len(ticker) == 0:
                logger.warning("No bars received in RealTimeBarList.")
                return

            bar = ticker[-1]
            # bar.time is already UTC-aware
            bar_time = bar.time.replace(second=0, microsecond=0)
            minute = bar_time.minute

            # Determine the start time of the current 30-minute candle
            candle_start_minute = (minute // 30) * 30
            candle_start_time = bar_time.replace(minute=candle_start_minute, second=0, microsecond=0)
            # candle_start_time should still be UTC-aware here

            if current_30min_start != candle_start_time:
                # New 30-minute candle has started
                if current_30min_start is not None and current_30min_bars:
                    # Aggregate the previous 30-minute candle
                    open_30 = current_30min_bars[0]['open']
                    high_30 = max(b['high'] for b in current_30min_bars)
                    low_30 = min(b['low'] for b in current_30min_bars)
                    close_30 = current_30min_bars[-1]['close']
                    volume_30 = sum(b['volume'] for b in current_30min_bars)

                    new_30_bar = {
                        'open': open_30,
                        'high': high_30,
                        'low': low_30,
                        'close': close_30,
                        'volume': volume_30
                    }

                    new_row = pd.DataFrame(new_30_bar, index=[current_30min_start])
                    df_30m_full = pd.concat([df_30m_full, new_row])
                    df_30m_full = calculate_bollinger_bands(df_30m_full, BOLLINGER_PERIOD, BOLLINGER_STDDEV)
                    df_30m_full.dropna(inplace=True)

                    # Re-filter RTH data
                    df_30m_rth = filter_rth(df_30m_full)

                    # Check if the new 30m bar is within RTH
                    if is_rth(current_30min_start):
                        current_price = close_30
                        current_time = current_30min_start
                        logger.info(f"\nNew 30-min bar closed at {current_time} UTC with close price: {current_price}")
                        try:
                            upper_band = df_30m_full.loc[current_time, 'upper_band']
                            lower_band = df_30m_full.loc[current_time, 'lower_band']
                        except KeyError:
                            logger.warning(f"No Bollinger Bands data available for {current_time}. Skipping trade.")
                            upper_band = None
                            lower_band = None

                        if upper_band is not None and lower_band is not None:
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
                        logger.info(f"New 30-min bar at {current_30min_start} UTC is outside RTH. No trade executed.")

                # Reset for new 30-minute candle
                current_30min_start = candle_start_time
                current_30min_bars = []

            # Append current 5-second bar to current 30-minute candle
            current_30min_bars.append({
                'open': bar.open_,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            })

    except Exception as e:
        logger.error(f"Error in on_realtime_bar handler: {e}")

# --- Subscribe to Real-Time Bars ---
try:
    logger.info("Requesting real-time 5-second bars (including ETH)...")
    ticker = ib.reqRealTimeBars(
        contract=es_contract,
        barSize=5,                       # 5-second bars
        whatToShow='TRADES',             # Trade data
        useRTH=False,                    # Include ETH data
        realTimeBarsOptions=[]
    )
    ticker.updateEvent += on_realtime_bar
    logger.info("Real-time bar handler assigned.")
except Exception as e:
    logger.error(f"Failed to subscribe to real-time bars: {e}")
    ib.disconnect()
    sys.exit(1)

# --- Start Event Loop ---
logger.info("Starting event loop...")
try:
    ib.run()
except KeyboardInterrupt:
    logger.info("Interrupt received, shutting down...")
finally:
    ib.disconnect()
    logger.info("Disconnected from IBKR.")