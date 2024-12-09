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
bollinger_period = 20
bollinger_stddev = 2
stop_loss_points = 10
take_profit_points = 20
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
df['date'] = pd.to_datetime(df['date'])  # Convert to datetime if not already
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

# Initialize variables
exposure_bars = 0  # Count active bars

# Backtesting loop
for i in range(bollinger_period, len(df)):
    current_price = df['close'].iloc[i]
    high_price = df['high'].iloc[i]
    low_price = df['low'].iloc[i]

    # Count exposure when position is active
    if position_size != 0:
        exposure_bars += 1

    if position_size == 0:
        if current_price < df['lower_band'].iloc[i]:
            position_size = 1
            entry_price = current_price
            position_type = 'long'

        elif current_price > df['upper_band'].iloc[i]:
            position_size = 1
            entry_price = current_price
            position_type = 'short'

    elif position_type == 'long':
        price_change_profit = high_price - entry_price
        price_change_loss = entry_price - low_price

        if price_change_profit >= take_profit_points:
            pnl = (take_profit_points * position_size * 5) - total_commission
            cash += pnl
            trade_results.append(pnl)
            balance_series.append(cash)
            position_size = 0

        elif price_change_loss >= stop_loss_points:
            pnl = (-stop_loss_points * position_size * 5) - total_commission
            cash += pnl
            trade_results.append(pnl)
            balance_series.append(cash)
            position_size = 0

    elif position_type == 'short':
        price_change_profit = entry_price - low_price
        price_change_loss = high_price - entry_price

        if price_change_profit >= take_profit_points:
            pnl = (take_profit_points * position_size * 5) - total_commission
            cash += pnl
            trade_results.append(pnl)
            balance_series.append(cash)
            position_size = 0

        elif price_change_loss >= stop_loss_points:
            pnl = (-stop_loss_points * position_size * 5) - total_commission
            cash += pnl
            trade_results.append(pnl)
            balance_series.append(cash)
            position_size = 0

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
    profit_factor = float('inf') if winning_trades else 0  # Avoid 'inf' when no trades


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
    "Win Rate": f"{len(winning_trades) / len(trade_results) * 100:.2f}%" if trade_results else "0.00%",
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