import sys
import logging
from ib_insync import *

# --- Configuration Parameters ---
IB_HOST = '127.0.0.1'        # IBKR Gateway/TWS host
IB_PORT = 7497               # IBKR Gateway/TWS paper trading port
CLIENT_ID = 2                # Unique client ID

EXEC_SYMBOL = 'MES'          # Micro E-mini S&P 500 for execution
EXEC_EXPIRY = '202503'       # March 2025
EXEC_EXCHANGE = 'CME'        # Exchange for MES
CURRENCY = 'USD'

POSITION_SIZE = 1            # Number of MES contracts per trade
TAKE_PROFIT_DISTANCE = 10     # Points away from entry
STOP_LOSS_DISTANCE = 5        # Points away from entry

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

# --- Define and Qualify Contract ---
mes_contract = Future(
    symbol=EXEC_SYMBOL,
    lastTradeDateOrContractMonth=EXEC_EXPIRY,
    exchange=EXEC_EXCHANGE,
    currency=CURRENCY
)

try:
    qualified_contracts = ib.qualifyContracts(mes_contract)
    mes_contract = qualified_contracts[0]
    logger.info(f"Qualified MES Contract: {mes_contract}")
except Exception as e:
    logger.error(f"Error qualifying MES contract: {e}")
    ib.disconnect()
    sys.exit(1)

# --- Event Handlers ---
def on_trade_filled(trade, fill):
    logger.info(f"Trade Filled - Order ID {trade.order.orderId}: {trade.order.action} {fill.size} @ {fill.price}")
    if trade.isFilled():
        entry_price = fill.price
        action = trade.order.action.upper()
        position_type = 'LONG' if action == 'BUY' else 'SHORT'
        logger.info(f"Entered {position_type} position at {entry_price}")

def on_order_status(trade, fill):
    logger.info(f"Trade Status Update - Order ID {trade.order.orderId}: {trade.orderStatus.status}")
    if trade.orderStatus.status in ('Cancelled', 'Inactive'):
        logger.info(f"Order ID {trade.order.orderId} has been {trade.orderStatus.status.lower()}.")

# --- Place Bracket Order ---
def place_bracket_order(action, current_price):
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
        # Create a bracket order
        bracket = ib.bracketOrder(
            action=action.upper(),
            quantity=POSITION_SIZE,
            limitPrice=current_price,  # Parent order price; using market order instead
            takeProfitPrice=take_profit_price,
            stopLossPrice=stop_loss_price
        )

        # Modify parent order to be a Market Order
        bracket[0].orderType = 'MKT'
        bracket[0].transmit = False  # Transmit all together

        bracket[1].transmit = False
        bracket[2].transmit = True  # Transmit the last order to send all

        # Place all orders together
        for order in bracket:
            ib.placeOrder(mes_contract, order)
            logger.info(f"Placed {order.orderType} Order ID {order.orderId} for {order.action} at {order.lmtPrice if hasattr(order, 'lmtPrice') else order.auxPrice}")

        # Attach event handlers
        for order in bracket:
            trade = ib.trades()[-1]  # Get the most recently placed trade
            trade.filledEvent += on_trade_filled
            trade.statusEvent += on_order_status

        logger.info("Bracket order placed successfully.")

    except Exception as e:
        logger.error(f"Failed to place bracket order: {e}")

# --- Execute Test Order ---
if __name__ == "__main__":
    try:
        # Example parameters for testing
        test_action = 'BUY'  # or 'SELL'
        test_current_price = 5976  # Replace with a realistic test price

        logger.info(f"Submitting test {test_action} order at price {test_current_price}...")
        place_bracket_order(test_action, test_current_price)

        logger.info("Starting event loop to monitor order status...")
        ib.run()
    except KeyboardInterrupt:
        logger.info("Interrupt received, shutting down...")
    finally:
        ib.disconnect()
        logger.info("Disconnected from IBKR.")