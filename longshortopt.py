import itertools
from ib_insync import *
import pandas as pd
import numpy as np

# Connect to IBKR
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define the MES futures contract
mes_contract = Future(
    symbol='MES',
    exchange='CME',
    currency='USD',
    lastTradeDateOrContractMonth='202409'
)
ib.qualifyContracts(mes_contract)

# Retrieve historical data (6 months, 15-minute bars)
bars = ib.reqHistoricalData(
    mes_contract,
    endDateTime='20240909 23:59:59',
    durationStr='6 M',
    barSizeSetting='15 mins',
    whatToShow='TRADES',
    useRTH=False,
    formatDate=1
)
df = util.df(bars)
df.index = pd.to_datetime(df.index)

# Define strategy function to calculate Sharpe ratio
def strategy(stop_loss_points, take_profit_points):
    df['short_moving_avg'] = df['close'].rolling(window=20).mean()
    df['long_moving_avg'] = df['close'].rolling(window=50).mean()

    position_size = 0
    entry_price = None
    cash = 10000
    daily_returns = []
    commission_per_side = 0.62
    contract_multiplier = 5

    for i in range(50, len(df)):
        current_price = df['close'].iloc[i]
        short_moving_avg = df['short_moving_avg'].iloc[i]
        long_moving_avg = df['long_moving_avg'].iloc[i]

        if position_size == 0:
            if short_moving_avg > long_moving_avg:
                position_size = 1
                entry_price = current_price
                take_profit_level = entry_price + take_profit_points
                stop_loss_level = entry_price - stop_loss_points
            elif short_moving_avg < long_moving_avg:
                position_size = -1
                entry_price = current_price
                take_profit_level = entry_price - take_profit_points
                stop_loss_level = entry_price + stop_loss_points

        elif position_size == 1:
            if current_price <= stop_loss_level:
                pnl = (stop_loss_level - entry_price) * contract_multiplier - (commission_per_side * 2)
                cash += pnl
                daily_returns.append(pnl / cash)
                position_size = 0
            elif current_price >= take_profit_level:
                pnl = (take_profit_level - entry_price) * contract_multiplier - (commission_per_side * 2)
                cash += pnl
                daily_returns.append(pnl / cash)
                position_size = 0

        elif position_size == -1:
            if current_price >= stop_loss_level:
                pnl = (stop_loss_level - entry_price) * contract_multiplier - (commission_per_side * 2)
                cash += pnl
                daily_returns.append(pnl / cash)
                position_size = 0
            elif current_price <= take_profit_level:
                pnl = (take_profit_level - entry_price) * contract_multiplier - (commission_per_side * 2)
                cash += pnl
                daily_returns.append(pnl / cash)
                position_size = 0

    returns_series = pd.Series(daily_returns)
    if returns_series.std() != 0:
        sharpe_ratio = (returns_series.mean() / returns_series.std()) * np.sqrt(252)
    else:
        sharpe_ratio = float('nan')

    return sharpe_ratio

# Define stop loss and take profit ranges for optimization
stop_loss_range = range(4, 10, 2)  # Example: 4, 6, 8 points
take_profit_range = range(8, 20, 4)  # Example: 8, 12, 16 points

# Run optimization
best_sharpe = -np.inf
best_params = None

for stop_loss, take_profit in itertools.product(stop_loss_range, take_profit_range):
    sharpe = strategy(stop_loss, take_profit)
    print(f'Stop Loss: {stop_loss}, Take Profit: {take_profit}, Sharpe: {sharpe}')
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_params = (stop_loss, take_profit)

print(f'Best Parameters - Stop Loss: {best_params[0]}, Take Profit: {best_params[1]}, Best Sharpe: {best_sharpe:.2f}')

# Disconnect from TWS
ib.disconnect()