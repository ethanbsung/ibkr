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

# --- Request Historical Data for ES (Data Contract) ---
try:
    logger.info("Requesting historical ES data...")
    bars_30m = ib.reqHistoricalData(
        contract=es_contract,
        endDateTime='',
        durationStr='60 D',
        barSizeSetting='30 mins',
        whatToShow='TRADES',
        useRTH=False,
        formatDate=1,
        keepUpToDate=False
    )

    if bars_30m:
        df_30m = util.df(bars_30m)
        df_30m.set_index('date', inplace=True)
        df_30m.sort_index(inplace=True)
        logger.info("Successfully retrieved 30m historical data.")
    else:
        logger.warning("No 30m historical data received.")
        df_30m = pd.DataFrame()

except Exception as e:
    logger.error(f"Error requesting historical data: {e}")
    ib.disconnect()
    sys.exit(1)

# --- Calculate Bollinger Bands ---
def calculate_bollinger_bands(df, period=15, stddev=2):
    df['ma'] = df['close'].rolling(window=period).mean()
    df['std'] = df['close'].rolling(window=period).std()
    df['upper_band'] = df['ma'] + (stddev * df['std'])
    df['lower_band'] = df['ma'] - (stddev * df['std'])
    return df

df_30m = calculate_bollinger_bands(df_30m, BOLLINGER_PERIOD, BOLLINGER_STDDEV)
df_30m.dropna(inplace=True)
logger.info("Bollinger Bands calculated for 30m data.")

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
    
    outsideRth=True allows children to fill in extended hours.
    Once one child is filled, the other is automatically canceled (OCA group by bracket order).
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
    logger.info(f"Lower Threshold (Bollinger Band): {df_30m.iloc[-1]['lower_band']}")
    logger.info(f"Upper Threshold (Bollinger Band): {df_30m.iloc[-1]['upper_band']}")

    place_bracket_order(action, current_price)

# --- Real-Time Bar Handler ---
def on_realtime_bar(ticker, hasNewBar):
    global current_30min_start, current_30min_bars, df_30m, position, cash, pending_order

    try:
        if hasNewBar:
            if len(ticker) == 0:
                logger.warning("No bars received in RealTimeBarList.")
                return

            bar = ticker[-1]
            bar_time = bar.time.replace(second=0, microsecond=0)
            minute = bar_time.minute
            candle_start_minute = (minute // 30) * 30
            candle_start_time = bar_time.replace(minute=candle_start_minute, second=0, microsecond=0)

            if current_30min_start != candle_start_time:
                # New 30-min candle
                if current_30min_start is not None and current_30min_bars:
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
                    df_30m = pd.concat([df_30m, new_row])
                    df_30m = calculate_bollinger_bands(df_30m, BOLLINGER_PERIOD, BOLLINGER_STDDEV)
                    df_30m.dropna(inplace=True)

                    if not df_30m.empty:
                        current_price = close_30
                        current_time = current_30min_start
                        logger.info(f"\nNew 30-min bar closed at {current_time} UTC with close price: {current_price}")
                        logger.info(
                            f"Bollinger Bands - Upper: {df_30m.iloc[-1]['upper_band']}, "
                            f"Lower: {df_30m.iloc[-1]['lower_band']}"
                        )
                        logger.info(f"Current Price: {current_price}")
                        logger.info(f"Upper Threshold (Bollinger Band): {df_30m.iloc[-1]['upper_band']}")
                        logger.info(f"Lower Threshold (Bollinger Band): {df_30m.iloc[-1]['lower_band']}")

                        if position is None and not pending_order:
                            if current_price < df_30m.iloc[-1]['lower_band']:
                                # Enter Long during RTH
                                execute_trade('BUY', current_price, current_time)
                            elif current_price > df_30m.iloc[-1]['upper_band']:
                                # Enter Short during RTH
                                execute_trade('SELL', current_price, current_time)
                            else:
                                logger.info("No trading signal detected.")

                current_30min_start = candle_start_time
                current_30min_bars = []

            # Append current bar
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
ticker = ib.reqRealTimeBars(
    contract=es_contract,
    barSize=5,
    whatToShow='TRADES',
    useRTH=False,
    realTimeBarsOptions=[]
)
ticker.updateEvent += on_realtime_bar
logger.info("Real-time bar handler assigned.")

# --- Start Event Loop ---
logger.info("Starting event loop...")
try:
    ib.run()
except KeyboardInterrupt:
    logger.info("Interrupt received, shutting down...")
finally:
    ib.disconnect()
    logger.info("Disconnected from IBKR.")