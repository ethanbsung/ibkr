from ib_insync import IB, Future
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Step 2: Connect to IBKR
ib = IB()
try:
    ib.connect('127.0.0.1', 7497, clientId=3)
except Exception as e:
    logging.error(f"Could not connect to IBKR: {e}")
    sys.exit(1)

# Step 3: Define the ES Futures Contract
# Adjust the contract details as needed.
es_contract = Future(localSymbol='ESH5', exchange='CME', currency='USD')
# For standard E-mini S&P 500 futures, you might use:
# es_contract = Future('ES', 'GLOBEX', 'USD')

# Step 5: Subscribe to tick-by-tick trade data.
tickSub = ib.reqTickByTickData(es_contract, tickType="Last", numberOfTicks=0, ignoreSize=False)

# Step 6: Define and attach an event handler to process incoming ticks.
def onTickByTick(tick):
    # Process the tick: here we simply print the price and size.
    print(f"Tick received: Price={tick.price}, Size={tick.size}")
    # Insert your trading signal logic here.

tickSub.updateEvent += onTickByTick

# Step 7: Run the event loop to continuously receive tick data.
print("Streaming tick data for ES futures. Press Ctrl+C to stop.")
try:
    ib.run()  # This will block and process incoming events.
except KeyboardInterrupt:
    print("Stopping tick data stream.")

# Step 8: Cancel the subscription and disconnect when done.
ib.cancelTickByTickData(tickSub)
ib.disconnect()
print("Disconnected from IBKR.")