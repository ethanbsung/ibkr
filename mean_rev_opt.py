from ib_insync import *
import pandas as pd
import numpy as np
from itertools import product

# Connect to Interactive Brokers TWS or Gateway
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define Contracts
es_contract = Future(symbol='ES', exchange='CME', currency='USD', lastTradeDateOrContractMonth='202412')

# Qualify contracts
ib.qualifyContracts(es_contract)

# Retrieve Historical Data for ES
bars = ib.reqHistoricalData(
    es_contract,
    endDateTime='20241208 23:59:59',
    durationStr='12 M',
    barSizeSetting='30 mins',
    whatToShow='TRADES',
    useRTH=False,
    formatDate=1
)

# Convert historical data to DataFrame
df = util.df(bars)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Optimization Ranges
lookback_periods = range(10, 51, 5)
stop_losses = range(3, 15, 1)
take_profits = range(5, 25, 1)

# Initialize Best Config
best_config = None
best_sharpe_ratio = -float('inf')

# Backtesting and Optimization
for lookback, stop_loss, take_profit in product(lookback_periods, stop_losses, take_profits):
    # Calculate Bollinger Bands
    df['ma'] = df['close'].rolling(window=lookback).mean()
    df['std'] = df['close'].rolling(window=lookback).std()
    df['upper_band'] = df['ma'] + (2 * df['std'])
    df['lower_band'] = df['ma'] - (2 * df['std'])

    # Initialize Variables
    position_size = 0
    entry_price = None
    stop_loss_price = None
    take_profit_price = None
    position_type = None
    initial_cash = 5000
    cash = initial_cash
    trade_results = []
    exposure_bars = 0

    # Backtesting Loop
    for i in range(lookback, len(df)):
        current_price = df['close'].iloc[i]
        high_price = df['high'].iloc[i]
        low_price = df['low'].iloc[i]

        # Track Exposure
        if position_size != 0:
            exposure_bars += 1

        # Entry Logic
        if position_size == 0:
            if current_price < df['lower_band'].iloc[i]:
                position_size = 1
                entry_price = current_price
                position_type = 'long'
                stop_loss_price = entry_price - stop_loss
                take_profit_price = entry_price + take_profit

            elif current_price > df['upper_band'].iloc[i]:
                position_size = 1
                entry_price = current_price
                position_type = 'short'
                stop_loss_price = entry_price + stop_loss
                take_profit_price = entry_price - take_profit

        # Exit Logic
        elif position_type == 'long':
            if low_price <= stop_loss_price:
                pnl = ((stop_loss_price - entry_price) * 5) - 0.94
                cash += pnl
                trade_results.append(pnl)
                position_size = 0

            elif high_price >= take_profit_price:
                pnl = ((take_profit_price - entry_price) * 5) - 0.94
                cash += pnl
                trade_results.append(pnl)
                position_size = 0

        elif position_type == 'short':
            if high_price >= stop_loss_price:
                pnl = ((entry_price - stop_loss_price) * 5) - 0.94
                cash += pnl
                trade_results.append(pnl)
                position_size = 0

            elif low_price <= take_profit_price:
                pnl = ((entry_price - take_profit_price) * 5) - 0.94
                cash += pnl
                trade_results.append(pnl)
                position_size = 0

    # Calculate Metrics
    balance_series = pd.Series([initial_cash] + trade_results).cumsum()
    daily_returns = balance_series.pct_change().dropna()

    # Correct Sharpe Ratio Calculation
    sharpe_ratio = (
        daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        if daily_returns.std() != 0 else -float('inf')
    )

    total_return_percentage = ((cash - initial_cash) / initial_cash) * 100
    profit_factor = sum([t for t in trade_results if t > 0]) / abs(sum([t for t in trade_results if t <= 0])) if sum(
        [t for t in trade_results if t <= 0]) != 0 else float('inf')

    # Debug Output
    print(f"Testing Lookback={lookback}, Stop Loss={stop_loss}, Take Profit={take_profit}, Trades={len(trade_results)}, Sharpe={sharpe_ratio:.2f}")

    # Update Best Config
    if sharpe_ratio > best_sharpe_ratio:
        best_sharpe_ratio = sharpe_ratio
        best_config = {
            'lookback_period': lookback,
            'stop_loss_points': stop_loss,
            'take_profit_points': take_profit,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'final_balance': cash,
            'return_percentage': total_return_percentage,
            'total_trades': len(trade_results)
        }

# Print Best Results
if best_config:
    print("\nBest Strategy Configuration:")
    for key, value in best_config.items():
        print(f"{key:25}: {value:.2f}" if isinstance(value, float) else f"{key:25}: {value}")
else:
    print("No valid strategy configuration found.")

# Disconnect from TWS
ib.disconnect()