from ib_insync import *
import pandas as pd
import numpy as np
import datetime

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

# Parameters for the strategy
fast_period = 20
slow_period = 50
take_profit_points = 25  # Take profit at 30 points gain
stop_loss_points = 10    # Stop loss at 10 points loss
commission_per_side = 0.62  # Commission per contract per side
total_commission = commission_per_side * 2  # Total commission per round trip (buy and sell)

# Retrieve historical data for backtesting from 01/01/2024 to 09/01/2024 with 30-minute bars
bars = ib.reqHistoricalData(
    mes_contract,
    endDateTime='20240901 23:59:59',  # End date for historical data
    durationStr='12 M',  # Duration covering 8 months (January to September)
    barSizeSetting='1 Hour',  # 1 Hour bars
    whatToShow='TRADES',
    useRTH=False,  # Use all trading hours, including extended trading hours
    formatDate=1
)

# Convert historical data to pandas DataFrame
df = util.df(bars)

# Ensure the index is a DatetimeIndex
df.index = pd.to_datetime(df.index)

# Calculate moving averages
df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
df['slow_ma'] = df['close'].rolling(window=slow_period).mean()

# Initialize variables to track positions, PnL, and trades
position_size = 0
entry_price = None
position_type = None  # 'long' or 'short'
initial_cash = 10000  # Set initial cash to $10,000 for backtesting
cash = initial_cash
trade_results = []

# Define final_price early to prevent NameError
final_price = df['close'].iloc[-1]  # Initialize with the last close price in case no trades are made

# Backtesting loop
for i in range(max(fast_period, slow_period), len(df)):
    current_price = df['close'].iloc[i]
    high_price = df['high'].iloc[i]  # Use high price of the bar
    low_price = df['low'].iloc[i]    # Use low price of the bar

    if position_size == 0:
        # Check for a long entry signal
        if df['fast_ma'].iloc[i] > df['slow_ma'].iloc[i] and df['fast_ma'].iloc[i - 1] <= df['slow_ma'].iloc[i - 1]:
            # Buy 1 contract (long position)
            position_size = 1
            entry_price = current_price
            position_type = 'long'
            print(f"Buy order: {position_size} contract at {current_price:.2f}")
            print()  # New line for separation
        
        # Check for a short entry signal
        elif df['fast_ma'].iloc[i] < df['slow_ma'].iloc[i] and df['fast_ma'].iloc[i - 1] >= df['slow_ma'].iloc[i - 1]:
            # Sell 1 contract (short position)
            position_size = 1
            entry_price = current_price
            position_type = 'short'
            print(f"Sell order (short): {position_size} contract at {current_price:.2f}")
            print()  # New line for separation

    elif position_type == 'long':
        # Calculate the price change for a long position
        price_change = high_price - entry_price

        # Check for take profit or stop loss conditions
        if price_change >= take_profit_points:
            exit_price = entry_price + take_profit_points  # Set exit price to take profit level
            pnl = (exit_price - entry_price) * position_size * 5 - total_commission  # Profit or loss per contract
            cash += pnl  # Update cash with PnL only
            trade_results.append(pnl)  # Record trade result
            print(f"Profit target hit. Sell order (exit): {position_size} contract at {exit_price:.2f} (Profit/Loss: {pnl:.2f})")
            print(f"Cash after selling: ${cash:.2f}")
            print()  # New line for separation
            position_size = 0  # Exit position
            entry_price = None
            position_type = None
        elif (entry_price - low_price) >= stop_loss_points:
            exit_price = entry_price - stop_loss_points  # Set exit price to stop loss level
            pnl = (exit_price - entry_price) * position_size * 5 - total_commission  # Profit or loss per contract
            cash += pnl  # Update cash with PnL only
            trade_results.append(pnl)  # Record trade result
            print(f"Stop loss hit. Sell order (exit): {position_size} contract at {exit_price:.2f} (Profit/Loss: {pnl:.2f})")
            print(f"Cash after stop loss hit: ${cash:.2f}")
            print()  # New line for separation
            position_size = 0  # Exit position
            entry_price = None
            position_type = None

    elif position_type == 'short':
        # Calculate the price change for a short position
        price_change = entry_price - low_price

        # Check for take profit or stop loss conditions
        if price_change >= take_profit_points:
            exit_price = entry_price - take_profit_points  # Set exit price to take profit level
            pnl = (entry_price - exit_price) * position_size * 5 - total_commission  # Profit or loss per contract
            cash += pnl  # Update cash with PnL only
            trade_results.append(pnl)  # Record trade result
            print(f"Profit target hit. Buy to cover (exit): {position_size} contract at {exit_price:.2f} (Profit/Loss: {pnl:.2f})")
            print(f"Cash after buying to cover: ${cash:.2f}")
            print()  # New line for separation
            position_size = 0  # Exit position
            entry_price = None
            position_type = None
        elif (high_price - entry_price) >= stop_loss_points:
            exit_price = entry_price + stop_loss_points  # Set exit price to stop loss level
            pnl = (entry_price - exit_price) * position_size * 5 - total_commission  # Profit or loss per contract
            cash += pnl  # Update cash with PnL only
            trade_results.append(pnl)  # Record trade result
            print(f"Stop loss hit. Buy to cover (exit): {position_size} contract at {exit_price:.2f} (Profit/Loss: {pnl:.2f})")
            print(f"Cash after stop loss hit: ${cash:.2f}")
            print()  # New line for separation
            position_size = 0  # Exit position
            entry_price = None
            position_type = None

# Check if there is an open position at the end of the backtest
if position_size != 0:
    final_price = df['close'].iloc[-1]  # Use the last available price to close the position
    if position_type == 'long':
        pnl = (final_price - entry_price) * position_size * 5 - total_commission  # Calculate PnL for closing
        cash += pnl  # Update cash with final position value
        trade_results.append(pnl)  # Record trade result
        print(f"Final Sell order (exit): {position_size} contract at {final_price:.2f} (Profit/Loss: {pnl:.2f})")
        print(f"Cash after final sell: ${cash:.2f}")
    elif position_type == 'short':
        pnl = (entry_price - final_price) * position_size * 5 - total_commission  # Calculate PnL for closing
        cash += pnl  # Update cash with final position value
        trade_results.append(pnl)  # Record trade result
        print(f"Final Buy to cover (exit): {position_size} contract at {final_price:.2f} (Profit/Loss: {pnl:.2f})")
        print(f"Cash after final buy to cover: ${cash:.2f}")
    print()  # New line for separation
    position_size = 0  # Ensure position is fully closed
    position_type = None

# Print the last price at the end of the backtest
print(f"\nPrice at the end of the backtest: {final_price:.2f}")

# Correct balance series update
balance_series = pd.Series(initial_cash, index=df.index, dtype='float64')
trade_indices = df.index[max(fast_period, slow_period):len(trade_results) + max(fast_period, slow_period)]
balance_series[trade_indices] = initial_cash + np.cumsum(trade_results).astype('float64')

# Calculate drawdown based on balance
df['balance'] = balance_series.ffill()
df['drawdown'] = df['balance'] / df['balance'].cummax() - 1
max_drawdown = df['drawdown'].min() * 100

# Calculate Sharpe Ratio
returns = df['balance'].pct_change().dropna()
sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else float('nan')

# Calculate total return
total_return = cash - initial_cash  # Correctly calculate total return as final balance - initial cash

# Output results
print("\nDetailed Trade Analysis:")
total_profit = sum(result for result in trade_results if result > 0)
total_loss = sum(result for result in trade_results if result <= 0)
print(f"Total Profits from Winning Trades: ${total_profit:.2f}")
print(f"Total Losses from Losing Trades: ${total_loss:.2f}")

# Recalculate Profit Factor to ensure correctness
profit_factor = total_profit / abs(total_loss) if total_loss != 0 else float('inf')
print(f"Profit Factor: {profit_factor:.2f}")

print("\nTrade Analysis:")
print(f"Total Trades: {len(trade_results)}")
print(f"Winning Trades: {len([result for result in trade_results if result > 0])}")
print(f"Losing Trades: {len([result for result in trade_results if result <= 0])}")
print(f"Win Rate: {len([result for result in trade_results if result > 0]) / len(trade_results) * 100:.2f}%" if trade_results else "Win Rate: N/A")
print(f"Average Win: {np.mean([result for result in trade_results if result > 0]):.2f}" if len([result for result in trade_results if result > 0]) > 0 else "Average Win: N/A")
print(f"Average Loss: {np.mean([result for result in trade_results if result <= 0]):.2f}" if len([result for result in trade_results if result <= 0]) > 0 else "Average Loss: N/A")

print("\nDrawdown Analysis:")
print(f"Max Drawdown: {max_drawdown:.2f}%")

print("\nSharpe Ratio Analysis:")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Print the total return
print(f"\nTotal Return: ${total_return:.2f}")

# Print the final account balance
final_balance = cash
print(f"Final Account Balance: ${final_balance:.2f}")

# Disconnect from TWS
ib.disconnect()