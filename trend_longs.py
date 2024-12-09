from ib_insync import *
import pandas as pd
import numpy as np

# Connect to Interactive Brokers TWS or Gateway
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define the MES futures contract
mes_contract = Future(
    symbol='MES',
    exchange='CME',
    currency='USD',
    lastTradeDateOrContractMonth='202409'
)

# Qualify the contract
ib.qualifyContracts(mes_contract)

# Strategy Parameters
short_moving_avg_period = 10  # Short period for the moving average
long_moving_avg_period = 30   # Long period for the moving average
stop_loss_points = 5  # Stop loss in points
take_profit_points = 10  # Take profit in points
commission_per_side = 0.62  # Commission per contract per side
initial_cash = 10000  # Starting cash balance
contract_multiplier = 5  # $5 per point for MES

# Retrieve 12 months of historical data (5-minute bars)
bars = ib.reqHistoricalData(
    mes_contract,
    endDateTime='20240909 23:59:59',  # End date for historical data
    durationStr='3 M',  # 12 months of data
    barSizeSetting='15 mins',  # 5-minute bars
    whatToShow='TRADES',
    useRTH=False,  # Use all trading hours
    formatDate=1,
    timeout = 120
)

# Convert historical data to pandas DataFrame
df = util.df(bars)
df.index = pd.to_datetime(df.index)

# Calculate the short and long moving averages
df['short_moving_avg'] = df['close'].rolling(window=short_moving_avg_period).mean()
df['long_moving_avg'] = df['close'].rolling(window=long_moving_avg_period).mean()

# Initialize variables
position_size = 0
entry_price = None
cash = initial_cash
trade_results = []
daily_returns = []
total_commissions = 0  # Track total commissions
max_balance = initial_cash  # Track the highest balance
max_drawdown = 0  # Track max drawdown

# Backtesting loop
for i in range(long_moving_avg_period, len(df)):
    current_price = df['close'].iloc[i]
    current_high = df['high'].iloc[i]
    current_low = df['low'].iloc[i]
    short_moving_avg = df['short_moving_avg'].iloc[i]
    long_moving_avg = df['long_moving_avg'].iloc[i]

    if position_size == 0:
        # Check for long entry signal when short MA crosses above long MA
        if short_moving_avg > long_moving_avg:
            position_size = 1  # Enter long position
            entry_price = current_price
            take_profit_level = entry_price + take_profit_points
            stop_loss_level = entry_price - stop_loss_points
            print(f"Long entry: {position_size} contract at {entry_price:.2f}")
            print(f"Take profit level: {take_profit_level:.2f}, Stop loss level: {stop_loss_level:.2f}")

    elif position_size > 0:
        # Check stop-loss hit
        if current_low <= stop_loss_level:
            pnl = (stop_loss_level - entry_price) * position_size * contract_multiplier - (commission_per_side * 2)
            total_commissions += commission_per_side * 2  # Add commissions for the trade
            cash += pnl
            trade_results.append(pnl)
            daily_returns.append(pnl / cash)
            print(f"Stop loss hit: {position_size} contract at {stop_loss_level:.2f} (PnL: {pnl:.2f})")
            position_size = 0
            entry_price = None

        # Check take-profit hit
        elif current_high >= take_profit_level:
            pnl = (take_profit_level - entry_price) * position_size * contract_multiplier - (commission_per_side * 2)
            total_commissions += commission_per_side * 2  # Add commissions for the trade
            cash += pnl
            trade_results.append(pnl)
            daily_returns.append(pnl / cash)
            print(f"Take profit hit: {position_size} contract at {take_profit_level:.2f} (PnL: {pnl:.2f})")
            position_size = 0
            entry_price = None

        # Exit if short MA crosses below long MA, if stop-loss or take-profit is not hit
        elif short_moving_avg < long_moving_avg:
            pnl = (current_price - entry_price) * position_size * contract_multiplier - (commission_per_side * 2)
            total_commissions += commission_per_side * 2  # Add commissions for the trade
            cash += pnl
            trade_results.append(pnl)
            daily_returns.append(pnl / cash)
            print(f"Exit on moving average cross: {position_size} contract at {current_price:.2f} (PnL: {pnl:.2f})")
            position_size = 0
            entry_price = None

    # Track max balance and drawdown
    if cash > max_balance:
        max_balance = cash  # Update max balance
    drawdown = (max_balance - cash) / max_balance  # Calculate drawdown
    if drawdown > max_drawdown:
        max_drawdown = drawdown  # Update max drawdown

# Final account balance and trade analysis
total_return = cash - initial_cash
print(f"\nFinal Account Balance: ${cash:.2f}")
print(f"Total Return: ${total_return:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Total Commissions: ${total_commissions:.2f}")

# Trade analysis
winning_trades = [result for result in trade_results if result > 0]
losing_trades = [result for result in trade_results if result <= 0]

number_of_wins = len(winning_trades)
number_of_losses = len(losing_trades)
win_rate = number_of_wins / len(trade_results) * 100 if trade_results else 0
average_win = np.mean(winning_trades) if winning_trades else 0
average_loss = np.mean(losing_trades) if losing_trades else 0
profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf')

# Sharpe Ratio Calculation
returns_series = pd.Series(daily_returns)
if returns_series.std() != 0:
    sharpe_ratio = (returns_series.mean() / returns_series.std()) * np.sqrt(252)
else:
    sharpe_ratio = float('nan')

# Output performance metrics
print(f"Total Trades: {len(trade_results)}")
print(f"Winning Trades: {number_of_wins}")
print(f"Losing Trades: {number_of_losses}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Average Win: ${average_win:.2f}")
print(f"Average Loss: ${average_loss:.2f}")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Disconnect from TWS
ib.disconnect()