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
stop_loss_points = 10  # Stop loss in points
take_profit_points = 20  # Take profit in points
commission_per_side = 0.62  # Commission per contract per side
total_commission = commission_per_side * 2  # Total commission per round trip (buy and sell)

# Retrieve historical data for backtesting from 01/01/2024 to 09/01/2024 with 30-minute bars
bars = ib.reqHistoricalData(
    mes_contract,
    endDateTime='20240901 23:59:59',  # End date for historical data
    durationStr='12 M',  # Duration covering 8 months (January to September)
    barSizeSetting='30 mins',  # 30-minute bars
    whatToShow='TRADES',
    useRTH=False,  # Use all trading hours, including extended trading hours
    formatDate=1
)

# Convert historical data to pandas DataFrame
df = util.df(bars)
df.index = pd.to_datetime(df.index)

def backtest_bollinger(bollinger_period, bollinger_stddev):
    df['ma'] = df['close'].rolling(window=bollinger_period).mean()
    df['std'] = df['close'].rolling(window=bollinger_period).std()
    df['upper_band'] = df['ma'] + (bollinger_stddev * df['std'])
    df['lower_band'] = df['ma'] - (bollinger_stddev * df['std'])

    # Initialize variables to track positions, PnL, and trades
    position_size = 0
    entry_price = None
    position_type = None  # 'long' or 'short'
    initial_cash = 10000  # Set initial cash to $10,000 for backtesting
    cash = initial_cash
    trade_results = []

    # Backtesting loop
    for i in range(bollinger_period, len(df)):
        current_price = df['close'].iloc[i]
        high_price = df['high'].iloc[i]  # High price of the current bar
        low_price = df['low'].iloc[i]    # Low price of the current bar

        if position_size == 0:
            # Check for a long entry signal
            if current_price < df['lower_band'].iloc[i]:
                # Buy 1 contract (long position)
                position_size = 1
                entry_price = current_price
                position_type = 'long'

            # Check for a short entry signal
            elif current_price > df['upper_band'].iloc[i]:
                # Sell 1 contract (short position)
                position_size = 1
                entry_price = current_price
                position_type = 'short'

        elif position_type == 'long':
            # Calculate the maximum adverse price movement for a long position
            price_change_profit = high_price - entry_price
            price_change_loss = entry_price - low_price

            # Check for take profit or stop loss conditions
            if price_change_profit >= take_profit_points:
                pnl = (take_profit_points * position_size * 5) - total_commission  # Profit per contract
                cash += pnl  # Update cash with PnL only
                trade_results.append(pnl)  # Record trade result
                position_size = 0  # Exit position
                entry_price = None
                position_type = None
            elif price_change_loss >= stop_loss_points:
                pnl = (-stop_loss_points * position_size * 5) - total_commission  # Loss per contract
                cash += pnl  # Update cash with PnL only
                trade_results.append(pnl)  # Record trade result
                position_size = 0  # Exit position
                entry_price = None
                position_type = None

        elif position_type == 'short':
            # Calculate the maximum favorable price movement for a short position
            price_change_profit = entry_price - low_price
            price_change_loss = high_price - entry_price

            # Check for take profit or stop loss conditions
            if price_change_profit >= take_profit_points:
                pnl = (take_profit_points * position_size * 5) - total_commission  # Profit per contract
                cash += pnl  # Update cash with PnL only
                trade_results.append(pnl)  # Record trade result
                position_size = 0  # Exit position
                entry_price = None
                position_type = None
            elif price_change_loss >= stop_loss_points:
                pnl = (-stop_loss_points * position_size * 5) - total_commission  # Loss per contract
                cash += pnl  # Update cash with PnL only
                trade_results.append(pnl)  # Record trade result
                position_size = 0  # Exit position
                entry_price = None
                position_type = None

    # Calculate performance metrics
    total_return = cash - initial_cash
    winning_trades = [result for result in trade_results if result > 0]
    losing_trades = [result for result in trade_results if result <= 0]
    profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf')
    returns = pd.Series(trade_results).pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else float('nan')

    return total_return, sharpe_ratio, profit_factor

# Optimization loop
best_params = None
best_sharpe = -float('inf')
results = []

for bollinger_period in range(10, 31, 5):  # Example: periods 10, 15, 20, 25, 30
    for bollinger_stddev in np.arange(1.5, 3.5, 0.5):  # Example: stddevs 1.5, 2.0, 2.5, 3.0
        total_return, sharpe_ratio, profit_factor = backtest_bollinger(bollinger_period, bollinger_stddev)

        if sharpe_ratio > best_sharpe:
            best_sharpe = sharpe_ratio
            best_params = (bollinger_period, bollinger_stddev)

        # Save the results of this combination
        results.append({
            'period': bollinger_period,
            'stddev': bollinger_stddev,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'profit_factor': profit_factor
        })

        # Print results for each iteration
        print(f"Tested Period: {bollinger_period}, StdDev: {bollinger_stddev}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}, Total Return: ${total_return:.2f}, Profit Factor: {profit_factor:.2f}")
        print("-" * 60)

# Print best result
print("\nBest Parameters:")
print(f"Best Period: {best_params[0]}, Best StdDev: {best_params[1]}")
print(f"Best Sharpe Ratio: {best_sharpe:.2f}")