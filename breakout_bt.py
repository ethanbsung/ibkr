from ib_insync import *
import pandas as pd
import numpy as np

# Connect to Interactive Brokers TWS or Gateway
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # Use port 7497 for paper trading (7496 for live accounts)

# Define the MES futures contract with correct specifications
mes_contract = Future(
    symbol='MES',              # Micro E-mini S&P 500 symbol
    exchange='CME',            # Correct exchange for Interactive Brokers
    currency='USD',            # USD currency
    lastTradeDateOrContractMonth='202409'  # Example expiry (September 2024)
)

# Qualify the contract to ensure it is valid and tradable
ib.qualifyContracts(mes_contract)

# Strategy Parameters
lookback_period = 20  # Number of periods to look back to determine support and resistance
stop_loss_points = 10  # Stop loss in points
take_profit_points = 15  # Take profit in points
commission_per_side = 0.62  # Commission per contract per side

# Retrieve historical data for backtesting from 01/01/2024 to 09/01/2024 with 30-minute bars
bars = ib.reqHistoricalData(
    mes_contract,
    endDateTime='20240901 23:59:59',  # End date for historical data
    durationStr='8 M',  # Duration covering 8 months (January to September)
    barSizeSetting='5 mins',  # 30-minute bars
    whatToShow='TRADES',
    useRTH=False,  # Use all trading hours, including extended trading hours
    formatDate=1
)

# Convert historical data to pandas DataFrame
df = util.df(bars)

# Ensure the index is a DatetimeIndex
df.index = pd.to_datetime(df.index)

# Calculate support and resistance levels
df['rolling_high'] = df['high'].rolling(window=lookback_period).max()
df['rolling_low'] = df['low'].rolling(window=lookback_period).min()

# Initialize variables to track positions, PnL, and trades
position_size = 0
entry_price = None
position_type = None  # 'long' or 'short'
initial_cash = 10000  # Set initial cash to $10,000 for backtesting
cash = initial_cash
trade_results = []

# Backtesting loop
for i in range(lookback_period, len(df)):
    current_price = df['close'].iloc[i]
    current_high = df['rolling_high'].iloc[i]
    current_low = df['rolling_low'].iloc[i]

    if position_size == 0:
        # Check for a long entry signal
        if current_price > current_high:
            # Buy 1 contract (long position)
            position_size = 1
            entry_price = current_price
            position_type = 'long'
            print(f"Buy order: {position_size} contract at {current_price:.2f}")

        # Check for a short entry signal
        elif current_price < current_low:
            # Sell 1 contract (short position)
            position_size = 1
            entry_price = current_price
            position_type = 'short'
            print(f"Sell order (short): {position_size} contract at {current_price:.2f}")

    elif position_type == 'long':
        # Calculate the absolute price movement for long positions
        price_change = current_price - entry_price

        # Check for take profit or stop loss conditions
        if price_change >= take_profit_points:
            pnl = (price_change * position_size * 5) - (commission_per_side * 2)  # Profit or loss per contract
            cash += pnl  # Update cash with PnL only
            trade_results.append(pnl)  # Record trade result
            print(f"Profit target hit. Sell order (exit): {position_size} contract at {current_price:.2f} (Profit/Loss: {pnl:.2f})")
            position_size = 0  # Exit position
            entry_price = None
            position_type = None
        elif price_change <= -stop_loss_points:
            pnl = (price_change * position_size * 5) - (commission_per_side * 2)  # Profit or loss per contract
            cash += pnl  # Update cash with PnL only
            trade_results.append(pnl)  # Record trade result
            print(f"Stop loss hit. Sell order (exit): {position_size} contract at {current_price:.2f} (Profit/Loss: {pnl:.2f})")
            position_size = 0  # Exit position
            entry_price = None
            position_type = None

    elif position_type == 'short':
        # Calculate the absolute price movement for short positions
        price_change = entry_price - current_price

        # Check for take profit or stop loss conditions
        if price_change >= take_profit_points:
            pnl = (price_change * position_size * 5) - (commission_per_side * 2)  # Profit or loss per contract
            cash += pnl  # Update cash with PnL only
            trade_results.append(pnl)  # Record trade result
            print(f"Profit target hit. Buy to cover (exit): {position_size} contract at {current_price:.2f} (Profit/Loss: {pnl:.2f})")
            position_size = 0  # Exit position
            entry_price = None
            position_type = None
        elif price_change <= -stop_loss_points:
            pnl = (price_change * position_size * 5) - (commission_per_side * 2)  # Profit or loss per contract
            cash += pnl  # Update cash with PnL only
            trade_results.append(pnl)  # Record trade result
            print(f"Stop loss hit. Buy to cover (exit): {position_size} contract at {current_price:.2f} (Profit/Loss: {pnl:.2f})")
            position_size = 0  # Exit position
            entry_price = None
            position_type = None

# Set the final price for the last available price in the dataset
final_price = df['close'].iloc[-1]

# Check if there is an open position at the end of the backtest
if position_size != 0:
    if position_type == 'long':
        pnl = (final_price - entry_price) * position_size * 5 - (commission_per_side * 2)  # Calculate PnL for closing
        cash += pnl  # Update cash with final position value
        trade_results.append(pnl)  # Record trade result
        print(f"Final Sell order (exit): {position_size} contract at {final_price:.2f} (Profit/Loss: {pnl:.2f})")
    elif position_type == 'short':
        pnl = (entry_price - final_price) * position_size * 5 - (commission_per_side * 2)  # Calculate PnL for closing
        cash += pnl  # Update cash with final position value
        trade_results.append(pnl)  # Record trade result
        print(f"Final Buy to cover (exit): {position_size} contract at {final_price:.2f} (Profit/Loss: {pnl:.2f})")

# Print the last price at the end of the backtest
print(f"\nPrice at the end of the backtest: {final_price:.2f}")

# Calculate performance metrics
total_return = cash - initial_cash
winning_trades = [result for result in trade_results if result > 0]
losing_trades = [result for result in trade_results if result <= 0]

number_of_wins = len(winning_trades)
number_of_losses = len(losing_trades)
win_rate = number_of_wins / len(trade_results) * 100 if trade_results else 0
average_win = np.mean(winning_trades) if winning_trades else 0
average_loss = np.mean(losing_trades) if losing_trades else 0
profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf')

# Calculate Sharpe Ratio
returns = pd.Series(trade_results).pct_change().dropna()
sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else float('nan')

# Output results
print("\nTrade Analysis:")
print(f"Total Trades: {len(trade_results)}")
print(f"Winning Trades: {number_of_wins}")
print(f"Losing Trades: {number_of_losses}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Average Win: ${average_win:.2f}")
print(f"Average Loss: ${average_loss:.2f}")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

print("\nTotal Return: ${total_return:.2f}")
print(f"Final Account Balance: ${cash:.2f}")

# Disconnect from TWS
ib.disconnect()