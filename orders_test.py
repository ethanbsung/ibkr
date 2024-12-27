from ib_insync import *
import logging
import sys

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

# --- Define Contract ---
mes_contract = Future(symbol=EXEC_SYMBOL, lastTradeDateOrContractMonth=EXEC_EXPIRY, exchange=EXEC_EXCHANGE, currency=CURRENCY)

try:
    mes_contract = ib.qualifyContracts(mes_contract)[0]
    logger.info(f"Qualified MES Contract: {mes_contract}")
except Exception as e:
    logger.error(f"Error qualifying contract: {e}")
    ib.disconnect()
    sys.exit(1)

# --- Event Handlers ---
def on_trade_filled(trade):
    fill = trade.fills[-1]  # Get the latest fill
    logger.info(f"Trade Filled - Order ID {trade.order.orderId}: {trade.order.action} {fill.size} @ {fill.price}")

def on_order_status(trade):
    logger.info(f"Trade Status Update - Order ID {trade.order.orderId}: {trade.orderStatus.status}")

# --- Place Bracket Order ---
def place_bracket_order(action, current_price):
    if action.upper() not in ['BUY', 'SELL']:
        logger.error(f"Invalid action: {action}. Must be 'BUY' or 'SELL'.")
        return

    # Define take-profit and stop-loss prices
    if action.upper() == 'BUY':
        take_profit_price = current_price + TAKE_PROFIT_DISTANCE
        stop_loss_price = current_price - STOP_LOSS_DISTANCE
    else:
        take_profit_price = current_price - TAKE_PROFIT_DISTANCE
        stop_loss_price = current_price + STOP_LOSS_DISTANCE

    try:
        # Create the bracket order
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

        # Attach event handlers
        parent_trade.filledEvent += on_trade_filled
        parent_trade.statusEvent += on_order_status

        # Place the child orders (take-profit and stop-loss)
        take_profit_trade = ib.placeOrder(mes_contract, bracket[1])
        stop_loss_trade = ib.placeOrder(mes_contract, bracket[2])

        logger.info("Bracket order placed successfully.")

    except Exception as e:
        logger.error(f"Failed to place bracket order: {e}")

# --- Test Bracket Order Placement ---
current_price = 6090  # Example current price
action = 'SELL'         # Example action: 'BUY' or 'SELL'

logger.info(f"Testing bracket order placement: Action={action}, Current Price={current_price}")
place_bracket_order(action, current_price)

# --- Start Event Loop ---
logger.info("Starting event loop...")
try:
    ib.run()
except KeyboardInterrupt:
    logger.info("Interrupt received, shutting down...")
finally:
    ib.disconnect()
    logger.info("Disconnected from IBKR.")