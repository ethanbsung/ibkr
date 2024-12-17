import pandas as pd
import numpy as np
import datetime
from decimal import Decimal
from ib_insync import *
import logging

# --- Configuration Parameters ---
IB_HOST = '127.0.0.1'        # IBKR Gateway/TWS host
IB_PORT = 7497               # IBKR Gateway/TWS paper trading port
CLIENT_ID = 1                # Unique client ID
DATA_SYMBOL = 'ES'           # E-mini S&P 500 for data
DATA_EXPIRY = '202503'       # March 2025
DATA_EXCHANGE = 'CME'        # Exchange for ES

EXEC_SYMBOL = 'MES'           # Micro E-mini S&P 500 for execution
EXEC_EXPIRY = '202503'        # March 2025
EXEC_EXCHANGE = 'CME'         # Exchange for MES

CURRENCY = 'USD'

INITIAL_CASH = 5000          # Starting cash
POSITION_SIZE = 1            # Number of MES contracts per trade

BOLLINGER_PERIOD = 15
BOLLINGER_STDDEV = 2
STOP_LOSS_DISTANCE = 5        # Points away from entry
TAKE_PROFIT_DISTANCE = 10     # Points away from entry

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
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
    exit(1)

# --- Define Contracts ---
# ES Contract for Data
es_contract = Future(
    symbol=DATA_SYMBOL,
    lastTradeDateOrContractMonth=DATA_EXPIRY,
    exchange=DATA_EXCHANGE,
    currency=CURRENCY
)

# MES Contract for Execution
mes_contract = Future(
    symbol=EXEC_SYMBOL,
    lastTradeDateOrContractMonth=EXEC_EXPIRY,
    exchange=EXEC_EXCHANGE,
    currency=CURRENCY
)

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
    exit(1)

# --- Request Historical Data for ES (Data Contract) ---
try:
    logger.info("Requesting historical ES data...")

    # Request 30-minute bars
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
        logger.debug(df_30m.head())
    else:
        logger.warning("No 30m historical data received.")

except Exception as e:
    logger.error(f"Error requesting historical data: {e}")
    ib.disconnect()
    exit(1)

# --- Calculate Bollinger Bands for ES ---
def calculate_bollinger_bands(df, period=15, stddev=2):
    df['ma'] = df['close'].rolling(window=period).mean()
    df['std'] = df['close'].rolling(window=period).std()
    df['upper_band'] = df['ma'] + (stddev * df['std'])
    df['lower_band'] = df['ma'] - (stddev * df['std'])
    return df

df_30m = calculate_bollinger_bands(df_30m, BOLLINGER_PERIOD, BOLLINGER_STDDEV)
df_30m.dropna(inplace=True)
logger.info("Bollinger Bands calculated for 30m data.")
logger.debug(df_30m.tail())

# --- Initialize Variables ---
cash = INITIAL_CASH
balance_series = [INITIAL_CASH]
position = None  # No open position initially
pending_order = False  # To track if there's a pending order

# --- Define Bracket Order Function ---
def get_bracket_order(action, quantity, take_profit_price, stop_loss_price):
    """
    Create a bracket order using ib_insync's bracketOrder helper.
    """
    # Parent order: MarketOrder to ensure immediate execution
    parent_order = MarketOrder(action=action, totalQuantity=quantity)
    
    # Take Profit Order: Limit Order
    take_profit_action = 'SELL' if action.upper() == 'BUY' else 'BUY'
    take_profit_order = LimitOrder(
        action=take_profit_action,
        totalQuantity=quantity,
        lmtPrice=take_profit_price,
        parentId=parent_order.orderId
    )
    
    # Stop Loss Order: Stop Order
    stop_loss_action = 'SELL' if action.upper() == 'BUY' else 'BUY'
    stop_loss_order = StopOrder(
        action=stop_loss_action,
        totalQuantity=quantity,
        stopPrice=stop_loss_price,  # Corrected parameter name
        parentId=parent_order.orderId
    )
    
    return parent_order, take_profit_order, stop_loss_order

# --- Order Filled Callback ---
def on_trade_update(trade, fill):
    """
    Handles trade updates. If a parent order is filled, logs the entry.
    """
    global cash, position, pending_order
    if fill:
        logger.info(f"Trade Update - Order ID {trade.order.orderId}: {trade.order.action} {trade.order.totalQuantity} @ {fill.price}")
        entry_price = fill.price
        if trade.order.action.upper() == 'BUY':
            position = {
                'type': 'long',
                'entry_price': entry_price,
                'entry_time': datetime.datetime.utcnow()
            }
            logger.info(f"Entered LONG position at {entry_price}")
        elif trade.order.action.upper() == 'SELL':
            position = {
                'type': 'short',
                'entry_price': entry_price,
                'entry_time': datetime.datetime.utcnow()
            }
            logger.info(f"Entered SHORT position at {entry_price}")
        pending_order = False  # Reset pending order flag upon fill

# --- Trade Status Update Callback ---
def on_trade_status_update(trade, new_status):
    """
    Handles status updates for trades.
    """
    global position, pending_order
    logger.info(f"Trade ID {trade.order.orderId} status changed to {new_status.status}")
    if new_status.status in ('Filled', 'Cancelled', 'Inactive'):
        # Handle trade completion or cancellation if needed
        if new_status.status in ('Cancelled', 'Inactive'):
            logger.info(f"Trade ID {trade.order.orderId} has been {new_status.status.lower()}. Resetting position.")
            position = None
            pending_order = False  # Reset pending order flag

# --- Subscribe to Real-Time ES Bars ---
logger.info("Subscribing to real-time ES bars...")
ticker = ib.reqRealTimeBars(
    contract=es_contract,
    barSize=5,          # 5-second bars
    whatToShow='TRADES',
    useRTH=False,
    realTimeBarsOptions=[]
)
logger.info("Subscribed to real-time bars.")

# --- Initialize Real-Time Bar Aggregation ---
current_30min_start = None
current_30min_bars = []
thirty_min_bars = df_30m.copy()

# --- Define Trade Execution Function ---
def execute_trade(action, current_price, current_time):
    global pending_order
    if pending_order:
        logger.info("There is already a pending order. Skipping new trade execution.")
        return

    logger.info(f"Entry Signal: {action}")
    logger.info(f"Current Price: {current_price}")
    logger.info(f"Lower Threshold (Bollinger Band): {thirty_min_bars.iloc[-1]['lower_band']}")
    logger.info(f"Upper Threshold (Bollinger Band): {thirty_min_bars.iloc[-1]['upper_band']}")

    if action.upper() == 'BUY':
        take_profit_price = current_price + TAKE_PROFIT_DISTANCE
        stop_loss_price = current_price - STOP_LOSS_DISTANCE
    elif action.upper() == 'SELL':
        take_profit_price = current_price - TAKE_PROFIT_DISTANCE
        stop_loss_price = current_price + STOP_LOSS_DISTANCE
    else:
        logger.error(f"Unknown action: {action}")
        return

    # Create bracket order
    parent_order, take_profit_order, stop_loss_order = get_bracket_order(
        action=action,
        quantity=POSITION_SIZE,
        take_profit_price=take_profit_price,
        stop_loss_price=stop_loss_price
    )

    try:
        # Place parent order and capture the Trade object
        trade = ib.placeOrder(mes_contract, parent_order)
        logger.info(f"Placed parent {action} order with ID {parent_order.orderId}")

        # Attach event handlers to the Trade object to handle fills and status updates
        trade.filledEvent += lambda fill: on_trade_update(trade, fill)
        trade.statusEvent += lambda new_status: on_trade_status_update(trade, new_status)

        pending_order = True  # Set pending order flag

    except Exception as e:
        logger.error(f"Failed to place order: {e}")
        pending_order = False

# --- Define Real-Time Bar Handler ---
def on_realtime_bar(ticker, hasNewBar):
    global current_30min_start, current_30min_bars, thirty_min_bars, position, cash, pending_order

    try:
        if hasNewBar:
            if len(ticker) == 0:
                logger.warning("No bars received in RealTimeBarList.")
                return

            # The latest bar
            bar = ticker[-1]
            # Convert bar time to UTC and remove seconds and microseconds
            bar_time = bar.time.replace(second=0, microsecond=0)

            # Determine the start time of the current 30-minute candle
            minute = bar_time.minute
            candle_start_minute = (minute // 30) * 30
            candle_start_time = bar_time.replace(minute=candle_start_minute, second=0, microsecond=0)

            if current_30min_start != candle_start_time:
                # New 30-minute candle detected
                if current_30min_start is not None and current_30min_bars:
                    # Finalize the previous 30-minute candle
                    open_30 = current_30min_bars[0]['open']
                    high_30 = max(bar_data['high'] for bar_data in current_30min_bars)
                    low_30 = min(bar_data['low'] for bar_data in current_30min_bars)
                    close_30 = current_30min_bars[-1]['close']
                    volume_30 = sum(bar_data['volume'] for bar_data in current_30min_bars)

                    new_30_bar = {
                        'open': open_30,
                        'high': high_30,
                        'low': low_30,
                        'close': close_30,
                        'volume': volume_30
                    }

                    # Convert new_30_bar to a DataFrame row
                    new_row = pd.DataFrame(new_30_bar, index=[current_30min_start])

                    # Append to thirty_min_bars using pd.concat
                    thirty_min_bars = pd.concat([thirty_min_bars, new_row])

                    # Recalculate Bollinger Bands
                    thirty_min_bars = calculate_bollinger_bands(thirty_min_bars, BOLLINGER_PERIOD, BOLLINGER_STDDEV)
                    thirty_min_bars.dropna(inplace=True)

                    # Ensure that thirty_min_bars is not empty after dropping NaNs
                    if thirty_min_bars.empty:
                        logger.warning("thirty_min_bars is empty after recalculating Bollinger Bands. Skipping trade logic.")
                        return

                    current_price = close_30
                    current_time = current_30min_start

                    logger.info(f"\nNew 30-min bar closed at {current_time} UTC with close price: {current_price}")
                    logger.info(
                        f"Bollinger Bands - Upper: {thirty_min_bars.iloc[-1]['upper_band']}, "
                        f"Lower: {thirty_min_bars.iloc[-1]['lower_band']}"
                    )
                    logger.info(f"Current Price: {current_price}")
                    logger.info(f"Upper Threshold (Bollinger Band): {thirty_min_bars.iloc[-1]['upper_band']}")
                    logger.info(f"Lower Threshold (Bollinger Band): {thirty_min_bars.iloc[-1]['lower_band']}")

                    # **Log the last few entries for verification**
                    logger.debug("Latest 30-Minute Bars Data:")
                    logger.debug(thirty_min_bars.tail())

                    # --- Trading Logic ---
                    if position is None and not pending_order:
                        if current_price < thirty_min_bars.iloc[-1]['lower_band']:
                            # Enter Long
                            execute_trade('BUY', current_price, current_time)
                        elif current_price > thirty_min_bars.iloc[-1]['upper_band']:
                            # Enter Short
                            execute_trade('SELL', current_price, current_time)
                        else:
                            logger.info("No trading signal detected.")

                # Reset for the new 30-minute candle
                current_30min_start = candle_start_time
                current_30min_bars = []

            # Append current bar to the 30-minute candle
            current_30min_bars.append({
                'open': bar.open_,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            })

    except Exception as e:
        logger.error(f"Error in on_realtime_bar handler: {e}")

# --- Assign Real-Time Bar Handler ---
ticker.updateEvent += on_realtime_bar
logger.info("Real-time bar handler assigned.")

# --- Start the Event Loop ---
logger.info("Starting event loop...")
try:
    ib.run()
except KeyboardInterrupt:
    logger.info("Interrupt received, shutting down...")
finally:
    ib.disconnect()
    logger.info("Disconnected from IBKR.")