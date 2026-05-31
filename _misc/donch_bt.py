import pandas as pd
import numpy as np
from ib_insync import *
from datetime import datetime, time, timedelta

# ==============================
# Configuration Parameters
# ==============================

# Donchian Channel Parameters
DONCHIAN_PERIOD = 20  # Look-back period for Donchian Channels

# Strategy Parameters
STOP_LOSS = 10  # in ticks (1 tick = 0.25 for ES futures)
TAKE_PROFIT = 20  # in ticks

# Commissions
COMMISSION_PER_TRADE = 1.24  # USD

# Trading Hours (Regular Trading Hours for ES futures: 09:30 - 16:00 EST)
TRADING_START = time(9, 30)
TRADING_END = time(16, 0)

# ==============================
# Connect to IBKR API
# ==============================

def connect_ibkr():
    try:
        ib = IB()
        ib.connect('127.0.0.1', 7497, clientId=1)  # Ensure TWS/Gateway is running
        print("Connected to IBKR")
        return ib
    except Exception as e:
        print("Failed to connect to IBKR:", e)
        exit()

# ==============================
# Get Most Recent ES Futures Contract
# ==============================

def get_most_recent_es_contract(ib):
    es = Future(symbol='ES', exchange='CME', currency='USD')
    contracts = ib.reqContractDetails(es)
    if not contracts:
        print("No ES futures contracts found.")
        exit()
    # Sort contracts by expiry date
    contracts_sorted = sorted(contracts, key=lambda x: x.contract.lastTradeDateOrContractMonth, reverse=True)
    most_recent_contract = contracts_sorted[0].contract
    print(f"Using contract: {most_recent_contract.lastTradeDateOrContractMonth}")
    return most_recent_contract

# ==============================
# Fetch Historical Data
# ==============================

def fetch_historical_data(ib, contract, duration='6 M', bar_size='1 mins'):
    # Request historical data
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow='TRADES',
        useRTH=True,  # Regular Trading Hours
        formatDate=1
    )
    if not bars:
        print("No historical data retrieved.")
        exit()
    # Convert to DataFrame
    df = util.df(bars)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

# ==============================
# Implement Donchian Channels Strategy
# ==============================

def donchian_backtest(df, period, stop_loss_ticks, take_profit_ticks):
    # Calculate Donchian Channels
    df['donchian_high'] = df['high'].rolling(window=period).max().shift(1)
    df['donchian_low'] = df['low'].rolling(window=period).min().shift(1)

    # Initialize backtest variables
    position = 0  # 1 for long, -1 for short, 0 for no position
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    trades = []

    for current_time, row in df.iterrows():
        current_open = row['open']
        current_high = row['high']
        current_low = row['low']
        current_close = row['close']
        donchian_high = row['donchian_high']
        donchian_low = row['donchian_low']

        # Skip if Donchian Channels not available
        if np.isnan(donchian_high) or np.isnan(donchian_low):
            continue

        # Check if current time is within regular trading hours
        if not (TRADING_START <= current_time.time() <= TRADING_END):
            continue

        if position == 0:
            # Check for breakout to go long
            if current_open > donchian_high:
                position = 1
                entry_price = current_open
                stop_loss = entry_price - stop_loss_ticks * 0.25
                take_profit = entry_price + take_profit_ticks * 0.25
                trades.append({
                    'entry_time': current_time,
                    'entry_price': entry_price,
                    'type': 'Long'
                })
                continue
            # Check for breakout to go short
            elif current_open < donchian_low:
                position = -1
                entry_price = current_open
                stop_loss = entry_price + stop_loss_ticks * 0.25
                take_profit = entry_price - take_profit_ticks * 0.25
                trades.append({
                    'entry_time': current_time,
                    'entry_price': entry_price,
                    'type': 'Short'
                })
                continue
        elif position == 1:
            # Check for stop loss or take profit
            if current_low <= stop_loss or current_high >= take_profit:
                exit_price = stop_loss if current_low <= stop_loss else take_profit
                exit_time = current_time
                trades[-1].update({
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'profit': exit_price - entry_price - COMMISSION_PER_TRADE
                })
                position = 0
        elif position == -1:
            # Check for stop loss or take profit
            if current_high >= stop_loss or current_low <= take_profit:
                exit_price = stop_loss if current_high >= stop_loss else take_profit
                exit_time = current_time
                trades[-1].update({
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'profit': entry_price - exit_price - COMMISSION_PER_TRADE
                })
                position = 0

    # Close any open positions at the end
    if position != 0:
        exit_price = df.iloc[-1]['close']
        exit_time = df.index[-1]
        profit = (exit_price - entry_price) * position - COMMISSION_PER_TRADE
        trades[-1].update({
            'exit_time': exit_time,
            'exit_price': exit_price,
            'profit': profit
        })

    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)
    return trades_df

# ==============================
# Calculate Performance Metrics
# ==============================

def calculate_performance(trades_df):
    total_trades = len(trades_df)
    profitable_trades = trades_df[trades_df['profit'] > 0]
    win_rate = len(profitable_trades) / total_trades * 100 if total_trades > 0 else 0
    total_profit = trades_df['profit'].sum()
    avg_profit = trades_df['profit'].mean() if total_trades > 0 else 0
    max_drawdown = trades_df['profit'].cumsum().min()
    commission_total = total_trades * COMMISSION_PER_TRADE

    print("===== Backtest Performance =====")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {len(profitable_trades)}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Average Profit per Trade: ${avg_profit:.2f}")
    print(f"Total Commissions: ${commission_total:.2f}")
    print(f"Max Drawdown: ${max_drawdown:.2f}")

    return trades_df

# ==============================
# Main Execution
# ==============================

def main():
    ib = connect_ibkr()
    contract = get_most_recent_es_contract(ib)
    df = fetch_historical_data(ib, contract, duration='2 Y', bar_size='1 day')
    ib.disconnect()

    trades_df = donchian_backtest(df, DONCHIAN_PERIOD, STOP_LOSS, TAKE_PROFIT)
    performance = calculate_performance(trades_df)

    # Display Trades
    print("\n===== Trades =====")
    print(trades_df)

if __name__ == "__main__":
    main()