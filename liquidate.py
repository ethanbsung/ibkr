from ib_insync import *

# Connect to Interactive Brokers TWS or Gateway
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # Use port 7497 for paper trading (7496 for live accounts)

# Define the MES futures contract with correct specifications
mes_contract = Future(
    symbol='MES',              # Micro E-mini S&P 500 symbol
    exchange='CME',            # Correct exchange for Interactive Brokers
    currency='USD',            # USD currency
    lastTradeDateOrContractMonth='202409'  # Example expiry (September 2024). Update to the correct expiry for the front month
)

# Qualify the contract to ensure it is valid and tradable
ib.qualifyContracts(mes_contract)

# Retrieve and print the account balance
account_summary = ib.accountSummary()
for item in account_summary:
    if item.tag == 'TotalCashBalance' and item.currency == 'USD':
        print(f"Account Balance (USD): {item.value}")

# Retrieve and print active positions
positions = ib.positions()
if not positions:
    print("No active positions.")
else:
    print("Active Positions:")
    for position in positions:
        contract = position.contract
        
        # Check if the position is for the MES September contract
        if contract.symbol == 'MES' and contract.lastTradeDateOrContractMonth.startswith('202409'):
            position_size = position.position
            print(f"Symbol: {contract.symbol}, Expiry: {contract.lastTradeDateOrContractMonth}, Position: {position_size}, Average Cost: {position.avgCost}")

            # Ensure the contract has the correct exchange set
            contract.exchange = 'CME'  # Explicitly set the exchange for the contract

            # Determine the action to liquidate the position
            if position_size > 0:
                # If long, sell to close the position
                order = MarketOrder('SELL', position_size)
                print(f"Placing order to SELL {position_size} contracts to close long position.")
            elif position_size < 0:
                # If short, buy to close the position
                order = MarketOrder('BUY', abs(position_size))
                print(f"Placing order to BUY {abs(position_size)} contracts to close short position.")
            else:
                print("No position to liquidate.")
                continue

            # Execute the liquidation order
            trade = ib.placeOrder(contract, order)
            # Wait for the order to be filled
            ib.sleep(2)
            print(f"Order status for {contract.symbol} {contract.lastTradeDateOrContractMonth}: {trade.orderStatus.status}")
        else:
            print(f"No matching active positions for MES September contract.")

# Disconnect from TWS
ib.disconnect()