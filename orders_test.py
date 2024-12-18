import datetime
from ib_insync import *
import logging
import sys

# --- Configuration Parameters ---
IB_HOST = '127.0.0.1'        # IBKR Gateway/TWS host
IB_PORT = 7497               # IBKR Gateway/TWS paper trading port
CLIENT_ID = 1                # Unique client ID (ensure it's different from other scripts)
EXEC_SYMBOL = 'MES'           # Micro E-mini S&P 500 for execution
EXEC_EXPIRY = '202503'        # March 2025
EXEC_EXCHANGE = 'CME'         # Exchange for MES
CURRENCY = 'USD'
POSITION_SIZE = 1            # Number of MES contracts per trade
TAKE_PROFIT_DISTANCE = 10     # Points away from entry
STOP_LOSS_DISTANCE = 5        # Points away from entry

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
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
    sys.exit(1)

# --- Define Execution Contract ---
mes_contract = Future(
    symbol=EXEC_SYMBOL,
    lastTradeDateOrContractMonth=EXEC_EXPIRY,
    exchange=EXEC_EXCHANGE,
    currency=CURRENCY
)

# --- Qualify Contract ---
try:
    qualified_contracts = ib.qualifyContracts(mes_contract)
    mes_contract = qualified_contracts[0]
    logger.info(f"Qualified MES Contract: {mes_contract}")
except Exception as e:
    logger.error(f"Error qualifying MES contract: {e}")
    ib.disconnect()
    sys.exit(1)

# --- Define Bracket Order Function ---
def get_bracket_order(action, quantity, take_profit_price, stop_loss_price):
    """
    Create a bracket order consisting of a parent order and two child orders (take profit and stop loss).
    Sets `outsideRth=True` to allow execution during extended hours.
    """
    # Parent order: MarketOrder to ensure immediate execution
    parent_order = MarketOrder(action=action, totalQuantity=quantity)
    parent_order.transmit = False  # Do not transmit yet
    parent_order.outsideRth = True  # Allow execution outside RTH

    # Take Profit Order: Limit Order
    take_profit_action = 'SELL' if action.upper() == 'BUY' else 'BUY'
    take_profit_order = LimitOrder(
        action=take_profit_action,
        totalQuantity=quantity,
        lmtPrice=take_profit_price,
        parentId=parent_order.orderId,
        transmit=False  # Do not transmit yet
    )
    take_profit_order.outsideRth = True  # Allow execution outside RTH

    # Stop Loss Order: Stop Order
    stop_loss_action = 'SELL' if action.upper() == 'BUY' else 'BUY'
    stop_loss_order = StopOrder(
        action=stop_loss_action,
        totalQuantity=quantity,
        stopPrice=stop_loss_price,
        parentId=parent_order.orderId,
        transmit=True  # Transmit last to send all orders together
    )
    stop_loss_order.outsideRth = True  # Allow execution outside RTH

    return parent_order, take_profit_order, stop_loss_order

# --- Order Filled Callback ---
def on_trade_filled(trade, fill):
    """
    Callback function triggered when a trade is filled.
    """
    logger.info(f"Trade Filled - Order ID {trade.order.orderId}: {trade.order.action} {trade.order.totalQuantity} @ {fill.price}")
    if trade.isFilled():
        entry_price = fill.price
        action = trade.order.action.upper()
        position_type = 'LONG' if action == 'BUY' else 'SHORT'
        logger.info(f"Entered {position_type} position at {entry_price}")
        # Optionally, implement additional logic here

# --- Trade Status Update Callback ---
def on_order_status(trade, fill):
    """
    Callback function triggered when a trade's status is updated.
    """
    logger.info(f"Trade Status Update - Order ID {trade.order.orderId}: {trade.order.status}")
    if trade.order.status in ('Cancelled', 'Inactive'):
        logger.info(f"Order ID {trade.order.orderId} has been {trade.order.status.lower()}.")
        # Optionally, handle order cancellation or inactivity

# --- Place Bracket Order Function ---
def place_bracket_order(action, current_price):
    """
    Places a bracket order based on the action and current price.
    Sets orders to execute during extended hours by setting `outsideRth=True`.
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

    parent_order, take_profit_order, stop_loss_order = get_bracket_order(
        action=action.upper(),
        quantity=POSITION_SIZE,
        take_profit_price=take_profit_price,
        stop_loss_price=stop_loss_price
    )

    try:
        # Place parent order
        trade_parent = ib.placeOrder(mes_contract, parent_order)
        logger.info(f"Placed Parent {action.upper()} Order ID {parent_order.orderId}")

        # Place take profit order
        trade_take_profit = ib.placeOrder(mes_contract, take_profit_order)
        logger.info(f"Placed Take Profit Order ID {take_profit_order.orderId} at {take_profit_price}")

        # Place stop loss order
        trade_stop_loss = ib.placeOrder(mes_contract, stop_loss_order)
        logger.info(f"Placed Stop Loss Order ID {stop_loss_order.orderId} at {stop_loss_price}")

        # Attach event handlers to Trade objects
        trade_parent.filledEvent += on_trade_filled
        trade_parent.statusEvent += on_order_status

        trade_take_profit.filledEvent += on_trade_filled
        trade_take_profit.statusEvent += on_order_status

        trade_stop_loss.filledEvent += on_trade_filled
        trade_stop_loss.statusEvent += on_order_status

        logger.info("Bracket order placed successfully.")

    except Exception as e:
        logger.error(f"Failed to place bracket order: {e}")

# --- Main Execution Logic ---
def main():
    # Define a test action and current price
    # For testing purposes, you can change the action to 'BUY' or 'SELL'
    test_action = 'BUY'  # Change to 'SELL' to test selling
    test_current_price = 6122.50  # Replace with a realistic test price

    logger.info(f"Placing test bracket order: {test_action} at {test_current_price}")
    place_bracket_order(test_action, test_current_price)

    # Run the event loop to process callbacks
    try:
        ib.run()
    except KeyboardInterrupt:
        logger.info("Interrupt received, shutting down...")
    finally:
        ib.disconnect()
        logger.info("Disconnected from IBKR.")

if __name__ == "__main__":
    main()