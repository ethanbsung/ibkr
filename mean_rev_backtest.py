from ib_insync import *
import pandas as pd
import numpy as np

# Connect to Interactive Brokers TWS or Gateway
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define the MES futures contract
mes_contract = Future(
    symbol='ES',
    exchange='CME',
    currency='USD',
    lastTradeDateOrContractMonth='202412'
)

# Define Both Contracts
es_contract = Future(
    symbol='ES', 
    exchange='CME', 
    currency='USD', 
    lastTradeDateOrContractMonth='202412')

# Qualify the contract to ensure it is valid and tradable
ib.qualifyContracts(mes_contract)

# Strategy Parameters
bollinger_period = 15
bollinger_stddev = 2
stop_loss_points = 3
take_profit_points = 23
commission_per_side = 0.47
total_commission = commission_per_side * 2
initial_cash = 5000

# Retrieve historical data
bars = ib.reqHistoricalData(
    es_contract,
    endDateTime='20241208 23:59:59',
    durationStr='12 M',
    barSizeSetting='30 mins',
    whatToShow='TRADES',
    useRTH=False,
    formatDate=1
)

df = util.df(bars)

# Ensure DataFrame index is set to datetime format
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Extract correct start and end dates
start_date = df.index.min()
end_date = df.index.max()

# Debugging output to verify date extraction
print(f"Start Date from DataFrame: {start_date}")
print(f"End Date from DataFrame: {end_date}")

# Calculate Bollinger Bands
df['ma'] = df['close'].rolling(window=bollinger_period).mean()
df['std'] = df['close'].rolling(window=bollinger_period).std()
df['upper_band'] = df['ma'] + (bollinger_stddev * df['std'])
df['lower_band'] = df['ma'] - (bollinger_stddev * df['std'])

# Initialize variables
position_size = 0
entry_price = None
position_type = None  
cash = initial_cash
trade_results = []
balance_series = [initial_cash]

exposure_bars = 0  # Count bars when in a position

# In a more live-like simulation, once we enter a position, 
# we will set limit orders for take profit and stop loss.
# For longs:
#   Stop loss limit order at (entry_price - stop_loss_points)
#   Take profit limit order at (entry_price + take_profit_points)
#
# For shorts:
#   Stop loss limit order at (entry_price + stop_loss_points)
#   Take profit limit order at (entry_price - take_profit_points)

stop_loss_price = None
take_profit_price = None

# Backtesting loop
for i in range(bollinger_period, len(df)):
    current_price = df['close'].iloc[i]
    high_price = df['high'].iloc[i]
    low_price = df['low'].iloc[i]

    # Count exposure when position is active
    if position_size != 0:
        exposure_bars += 1

    if position_size == 0:
        # No open position, check for entry signals
        if current_price < df['lower_band'].iloc[i]:
            # Enter Long
            position_size = 1
            entry_price = current_price
            position_type = 'long'
            # Set limit orders
            stop_loss_price = entry_price - stop_loss_points
            take_profit_price = entry_price + take_profit_points
            print(f"Entered Long Position at {entry_price:.2f}")

        elif current_price > df['upper_band'].iloc[i]:
            # Enter Short
            position_size = 1
            entry_price = current_price
            position_type = 'short'
            # Set limit orders
            stop_loss_price = entry_price + stop_loss_points
            take_profit_price = entry_price - take_profit_points
            print(f"Entered Short Position at {entry_price:.2f}")

    else:
        # Position is open, check if the limit orders are triggered
        if position_type == 'long':
            # For a long position, check if stop or take profit triggered
            # Check stop loss first, as stops are protective and usually trigger first if both occur same bar
            if low_price <= stop_loss_price:
                # Stopped out at stop_loss_price
                pnl = ((stop_loss_price - entry_price) * position_size * 5) - total_commission
                cash += pnl
                trade_results.append(pnl)
                balance_series.append(cash)
                print(f"STOPPED OUT LONG at {stop_loss_price:.2f} | Loss: {pnl:.2f}")
                position_size = 0
                position_type = None
                entry_price = None
                stop_loss_price = None
                take_profit_price = None

            elif high_price >= take_profit_price:
                # Took profit at take_profit_price
                pnl = ((take_profit_price - entry_price) * position_size * 5) - total_commission
                cash += pnl
                trade_results.append(pnl)
                balance_series.append(cash)
                print(f"EXITED LONG at {take_profit_price:.2f} | Profit: {pnl:.2f}")
                position_size = 0
                position_type = None
                entry_price = None
                stop_loss_price = None
                take_profit_price = None

        elif position_type == 'short':
            # For a short position, check if stop or take profit triggered
            # Check stop loss first
            if high_price >= stop_loss_price:
                # Stopped out short at stop_loss_price
                pnl = ((entry_price - stop_loss_price) * position_size * 5) - total_commission
                cash += pnl
                trade_results.append(pnl)
                balance_series.append(cash)
                print(f"STOPPED OUT SHORT at {stop_loss_price:.2f} | Loss: {pnl:.2f}")
                position_size = 0
                position_type = None
                entry_price = None
                stop_loss_price = None
                take_profit_price = None

            elif low_price <= take_profit_price:
                # Took profit short at take_profit_price
                pnl = ((entry_price - take_profit_price) * position_size * 5) - total_commission
                cash += pnl
                trade_results.append(pnl)
                balance_series.append(cash)
                print(f"EXITED SHORT at {take_profit_price:.2f} | Profit: {pnl:.2f}")
                position_size = 0
                position_type = None
                entry_price = None
                stop_loss_price = None
                take_profit_price = None

# Final Portfolio Calculations
balance_series = pd.Series(balance_series)
daily_returns = balance_series.pct_change().dropna()

# Performance Metrics
total_return_percentage = ((cash - initial_cash) / initial_cash) * 100
trading_days = max((end_date - start_date).days, 1)  # Prevent division by zero
annualized_return_percentage = ((cash / initial_cash) ** (252 / trading_days)) - 1
benchmark_return = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
equity_peak = balance_series.max()
volatility_annual = daily_returns.std() * np.sqrt(252) * 100
sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else 0
sortino_ratio = (
    daily_returns.mean() / daily_returns[daily_returns < 0].std() * np.sqrt(252)
    if daily_returns[daily_returns < 0].std() != 0 else 0
)

# Drawdown Calculations
running_max = balance_series.cummax()
drawdowns = (balance_series - running_max) / running_max
max_drawdown = drawdowns.min() * 100
average_drawdown = drawdowns[drawdowns < 0].mean() * 100

# Calculate Max and Average Drawdown Durations
drawdown_durations = (drawdowns < 0).astype(int).groupby((drawdowns >= 0).astype(int).cumsum()).sum()
max_drawdown_duration_days = drawdown_durations.max() / 48  # 48 bars per trading day
average_drawdown_duration_days = drawdown_durations.mean() / 48

# Exposure Time Calculation
exposure_time_percentage = (exposure_bars / len(df)) * 100

# Profit Factor Calculation
winning_trades = [pnl for pnl in trade_results if pnl > 0]
losing_trades = [pnl for pnl in trade_results if pnl <= 0]
if losing_trades:
    profit_factor = sum(winning_trades) / abs(sum(losing_trades))
else:
    profit_factor = float('inf') if winning_trades else 0

# Results Summary
print("\nPerformance Summary:")
results = {
    "Start Date": start_date.strftime("%Y-%m-%d"),
    "End Date": end_date.strftime("%Y-%m-%d"),
    "Exposure Time": f"{exposure_time_percentage:.2f}%",
    "Final Account Balance": f"${cash:,.2f}",
    "Equity Peak": f"${equity_peak:,.2f}",
    "Total Return": f"{total_return_percentage:.2f}%",
    "Annualized Return": f"{annualized_return_percentage * 100:.2f}%",
    "Benchmark Return": f"{benchmark_return:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Total Trades": len(trade_results),
    "Winning Trades": len(winning_trades),
    "Losing Trades": len(losing_trades),
    "Win Rate": f"{(len(winning_trades)/len(trade_results)*100) if trade_results else 0:.2f}%",
    "Profit Factor": f"{profit_factor:.2f}",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}",
    "Sortino Ratio": f"{sortino_ratio:.2f}",
    "Calmar Ratio": f"{total_return_percentage / abs(max_drawdown) if max_drawdown != 0 else 'Inf'}",
    "Max Drawdown": f"{max_drawdown:.2f}%",
    "Average Drawdown": f"{average_drawdown:.2f}%",
    "Max Drawdown Duration": f"{max_drawdown_duration_days:.2f} days",
    "Average Drawdown Duration": f"{average_drawdown_duration_days:.2f} days",
}

for key, value in results.items():
    print(f"{key:25}: {value:>15}")

# Disconnect from TWS
ib.disconnect()

'''
bollinger_period = 15
bollinger_stddev = 2
stop_loss_points = 3
take_profit_points = 23
commission_per_side = 0.47
total_commission = commission_per_side * 2
initial_cash = 5000

Performance Summary:
Start Date               :      2023-12-18
End Date                 :      2024-12-08
Exposure Time            :          31.41%
Final Account Balance    :       $9,669.40
Equity Peak              :       $9,823.48
Total Return             :          93.39%
Annualized Return        :          59.50%
Benchmark Return         :          23.89%
Volatility (Annual)      :          10.77%
Total Trades             :             490
Winning Trades           :              96
Losing Trades            :             394
Win Rate                 :          19.59%
Profit Factor            :            1.74
Sharpe Ratio             :            3.20
Sortino Ratio            :           59.35
Calmar Ratio             : 18.073190509060993
Max Drawdown             :          -5.17%
Average Drawdown         :          -1.46%
Max Drawdown Duration    :       2.75 days
Average Drawdown Duration:       0.15 days
'''