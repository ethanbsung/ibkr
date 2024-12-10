from ib_insync import *
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

# Connect to Interactive Brokers TWS or Gateway
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define the ES futures contract (E-mini S&P 500)
es_contract = Future(
    symbol='ES',
    exchange='CME',
    currency='USD',
    lastTradeDateOrContractMonth='202412'
)

# Qualify the contract to ensure it is valid and tradable
ib.qualifyContracts(es_contract)

# Retrieve Historical Data for ES
bars = ib.reqHistoricalData(
    es_contract,
    endDateTime='20240608 23:59:59',
    durationStr='6 M',
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
lookback_periods = range(10, 31, 5)          # Bollinger Bands lookback periods: 10, 15, 20, 25
stop_losses = range(2, 10, 1)                # Stop loss points: 2, 3, 4, 5
take_profits = range(5, 30, 1)             # Take profit points: 10, 15, 20, 25, 30

# Initialize Best Config
best_config = None
best_sharpe_ratio = -float('inf')
min_win_rate = 25.0  # Minimum win rate threshold

# Backtesting and Optimization
total_combinations = len(lookback_periods) * len(stop_losses) * len(take_profits)
print(f"Total combinations to test: {total_combinations}")

# Initialize a counter for progress tracking
current_combination = 0

for lookback, stop_loss, take_profit in product(
    lookback_periods, stop_losses, take_profits
):
    # Increment the counter
    current_combination += 1

    # Print progress every 100 combinations
    if current_combination % 100 == 0 or current_combination == total_combinations:
        print(f"Testing combination {current_combination} of {total_combinations}")

    # Reset DataFrame to avoid accumulation of calculated columns
    df_backtest = df.copy()

    # Calculate Bollinger Bands
    df_backtest['ma'] = df_backtest['close'].rolling(window=lookback).mean()
    df_backtest['std'] = df_backtest['close'].rolling(window=lookback).std()
    df_backtest['upper_band'] = df_backtest['ma'] + (2 * df_backtest['std'])
    df_backtest['lower_band'] = df_backtest['ma'] - (2 * df_backtest['std'])

    # Drop rows with NaN values due to rolling calculations
    df_backtest.dropna(inplace=True)

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
    for i in range(len(df_backtest)):
        current_price = df_backtest['close'].iloc[i]
        high_price = df_backtest['high'].iloc[i]
        low_price = df_backtest['low'].iloc[i]

        # Count exposure when position is active
        if position_size != 0:
            exposure_bars += 1

        if position_size == 0:
            # No open position, check for entry signals based on Bollinger Bands
            if current_price < df_backtest['lower_band'].iloc[i]:
                # Enter Long
                position_size = 1
                entry_price = current_price
                position_type = 'long'
                # Set stop loss and take profit prices
                stop_loss_price = entry_price - stop_loss
                take_profit_price = entry_price + take_profit
                # Debugging output
                # print(f"Entered Long Position at {entry_price:.2f}")
            
            elif current_price > df_backtest['upper_band'].iloc[i]:
                # Enter Short
                position_size = 1
                entry_price = current_price
                position_type = 'short'
                # Set stop loss and take profit prices
                stop_loss_price = entry_price + stop_loss
                take_profit_price = entry_price - take_profit
                # Debugging output
                # print(f"Entered Short Position at {entry_price:.2f}")

        else:
            # Position is open, check if the limit orders are triggered
            if position_type == 'long':
                # For a long position, check if stop or take profit triggered
                # Check stop loss first
                if low_price <= stop_loss_price:
                    # Stopped out at stop_loss_price
                    pnl = ((stop_loss_price - entry_price) * 5) - 0.94  # ES multiplier is 50
                    cash += pnl
                    trade_results.append(pnl)
                    position_size = 0
                    # Debugging output
                    # print(f"STOPPED OUT LONG at {stop_loss_price:.2f} | Loss: {pnl:.2f}")

                elif high_price >= take_profit_price:
                    # Took profit at take_profit_price
                    pnl = ((take_profit_price - entry_price) * 5) - 0.94
                    cash += pnl
                    trade_results.append(pnl)
                    position_size = 0
                    # Debugging output
                    # print(f"EXITED LONG at {take_profit_price:.2f} | Profit: {pnl:.2f}")

            elif position_type == 'short':
                # For a short position, check if stop or take profit triggered
                # Check stop loss first
                if high_price >= stop_loss_price:
                    # Stopped out short at stop_loss_price
                    pnl = ((entry_price - stop_loss_price) * 5) - 0.94
                    cash += pnl
                    trade_results.append(pnl)
                    position_size = 0
                    # Debugging output
                    # print(f"STOPPED OUT SHORT at {stop_loss_price:.2f} | Loss: {pnl:.2f}")

                elif low_price <= take_profit_price:
                    # Took profit short at take_profit_price
                    pnl = ((entry_price - take_profit_price) * 5) - 0.94
                    cash += pnl
                    trade_results.append(pnl)
                    position_size = 0
                    # Debugging output
                    # print(f"EXITED SHORT at {take_profit_price:.2f} | Profit: {pnl:.2f}")

    # Calculate Metrics
    if trade_results:
        balance_series = pd.Series([initial_cash] + trade_results).cumsum()
    else:
        balance_series = pd.Series([initial_cash])

    daily_returns = balance_series.pct_change().dropna()

    # Sharpe Ratio Calculation (Assuming 252 trading days)
    sharpe_ratio = (
        (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        if daily_returns.std() != 0 else -float('inf')
    )

    # Total Return
    total_return_percentage = ((cash - initial_cash) / initial_cash) * 100

    # Profit Factor
    winning_trades = [t for t in trade_results if t > 0]
    losing_trades = [t for t in trade_results if t <= 0]
    profit_factor = (sum(winning_trades) / abs(sum(losing_trades))) if sum(losing_trades) != 0 else float('inf')

    # Win Rate Calculation
    total_trades = len(trade_results)
    winning_trades_count = len(winning_trades)
    win_rate = (winning_trades_count / total_trades) * 100 if total_trades > 0 else 0.0

    # Update Best Config based on Sharpe Ratio and Win Rate >30%
    if win_rate >= min_win_rate and sharpe_ratio > best_sharpe_ratio:
        best_sharpe_ratio = sharpe_ratio
        best_config = {
            'lookback_period': lookback,
            'stop_loss_points': stop_loss,
            'take_profit_points': take_profit,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'final_balance': cash,
            'return_percentage': total_return_percentage,
            'total_trades': total_trades,
            'winning_trades': winning_trades_count,
            'losing_trades': len(losing_trades),
            'win_rate': win_rate
        }

# Print Best Results
if best_config:
    print("\nBest Strategy Configuration (Win Rate > 30%):")
    for key, value in best_config.items():
        if isinstance(value, float):
            print(f"{key:25}: {value:.2f}")
        else:
            print(f"{key:25}: {value}")
else:
    print("No valid strategy configuration found with a win rate over 30%.")

# Disconnect from TWS
ib.disconnect()