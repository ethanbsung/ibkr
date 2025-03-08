import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from ib_insync import IB, Future, util

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# -------------------------------
# Parameters & User Settings
# -------------------------------

# IBKR connection parameters (adjust host/port/clientId as needed)
IB_HOST = '127.0.0.1'
IB_PORT = 4002           # use appropriate port (e.g., 7497 for TWS paper trading)
CLIENT_ID = 1

# Define continuous contract for ES futures.
# (Adjust these details if your continuous symbol is different.)
contract = Future(symbol='ES', lastTradeDateOrContractMonth='202503', exchange='CME', currency='USD')

# Backtest parameters
initial_capital = 10000.0           # starting account balance in dollars
lookback_breakout = 10              # number of bars for breakout calculation
volume_lookback = 10                # number of bars for volume average
volume_multiplier = 1.5             # condition: current volume > 1.5 * average volume
multiplier = 5                      # $5 per point
commission_per_order = 1.24         # commission per order (applied at entry and exit)
atr_period = 14                     # ATR period (in bars)
atr_multiplier = 2                  # trailing stop = ATR multiplier * ATR
# For 1-hour bars, a 200-day MA is computed using approximately 1200 bars.
MASLength = 1200                   

# Backtest date range (format: 'YYYY-MM-DD')
start_date = '2024-09-01'
end_date   = '2025-02-23'

# -------------------------------
# Connect to IBKR and Request Data
# -------------------------------
ib = IB()
ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
logger.info("Connected to IBKR.")

# Request historical 1-hour bars for the backtest.
endDateTime = ''  # empty string = current time
bars_1h = ib.reqHistoricalData(
    contract,
    endDateTime=endDateTime,
    durationStr='1 M',              # Request one month of data (adjust if needed)
    barSizeSetting='1 hour',
    whatToShow='TRADES',
    useRTH=True,
    formatDate=1
)
data_1h = util.df(bars_1h)
# Rename columns (IBKR returns lowercase names)
data_1h.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Last',
    'volume': 'Volume'
}, inplace=True)
logger.info(f"Retrieved {len(data_1h)} hourly bars.")

# Request historical daily bars for the 200-day MA.
bars_daily = ib.reqHistoricalData(
    contract,
    endDateTime=endDateTime,
    durationStr='2 Y',              # Request 2 years of daily data
    barSizeSetting='1 day',
    whatToShow='TRADES',
    useRTH=True,
    formatDate=1
)
data_daily = util.df(bars_daily)
data_daily.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Last',
    'volume': 'Volume'
}, inplace=True)
logger.info(f"Retrieved {len(data_daily)} daily bars.")

ib.disconnect()
logger.info("Disconnected from IBKR.")

# -------------------------------
# Data Preparation
# -------------------------------

# Use the hourly data for the backtest.
data = data_1h.copy()

# Convert the IB date field to datetime and sort.
data['Time'] = pd.to_datetime(data['date'])
data.sort_values('Time', inplace=True)
data.reset_index(drop=True, inplace=True)

# Filter data based on the custom date range.
data = data[(data['Time'] >= start_date) & (data['Time'] <= end_date)].reset_index(drop=True)

# For daily MA, add a Date column.
data_daily['Date'] = pd.to_datetime(data_daily['date']).dt.date

# Merge daily MA onto hourly data by date.
data['Date'] = data['Time'].dt.date
data = pd.merge(data, data_daily[['Date', 'Last']], on='Date', how='left', suffixes=('', '_daily'))
# Compute 200-day MA on the daily "Last" price.
data_daily['MA200'] = data_daily['Last'].rolling(window=200, min_periods=1).mean()
# Merge the most recent daily MA into the hourly data.
data = pd.merge(data, data_daily[['Date', 'MA200']], on='Date', how='left')
data.drop(columns=['Date', 'date'], inplace=True, errors='ignore')

# Calculate breakout levels and average volume on the hourly data.
data['prev_10_high'] = data['High'].shift(1).rolling(window=lookback_breakout, min_periods=lookback_breakout).max()
data['prev_10_low'] = data['Low'].shift(1).rolling(window=lookback_breakout, min_periods=lookback_breakout).min()
data['avg_volume_10'] = data['Volume'].shift(1).rolling(window=volume_lookback, min_periods=volume_lookback).mean()

# Calculate the 200-day MA on the hourly chart (using 1200 bars).
data['MA_Hourly'] = data['Last'].rolling(window=MASLength, min_periods=MASLength).mean()

# -------------------------------
# ATR Calculation (for trailing stop)
# -------------------------------
data['prev_close'] = data['Last'].shift(1)
data['tr1'] = data['High'] - data['Low']
data['tr2'] = abs(data['High'] - data['prev_close'])
data['tr3'] = abs(data['Low'] - data['prev_close'])
data['TR'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
data['ATR'] = data['TR'].rolling(window=atr_period, min_periods=atr_period).mean()

# -------------------------------
# Backtest Simulation
# -------------------------------
# Here we simulate entries at the candle close and exits are determined at the close
# if the candle's low (for longs) or high (for shorts) indicates the stop was hit.
capital = initial_capital
in_position = False
position = None  # will hold: { 'direction', 'entry_price', 'entry_time', 'trailing_stop' }
trade_results = []
exposure_bars = 0
equity_curve = []

for i, row in data.iterrows():
    current_time = row['Time']
    current_close = row['Last']
    
    # Skip bars until ATR is available.
    if np.isnan(row['ATR']):
        equity_curve.append((current_time, capital))
        continue

    # --- Exit Logic: Update trailing stop and check if stop was hit during the bar ---
    if in_position:
        exposure_bars += 1
        if position['direction'] == 'long':
            # Update trailing stop at end of bar: use this bar's High.
            new_stop = row['High'] - atr_multiplier * row['ATR']
            position['trailing_stop'] = max(position['trailing_stop'], new_stop)
            # Check if the bar's Low is below the trailing stop.
            if row['Low'] <= position['trailing_stop']:
                # Exit at the close of this bar.
                exit_price = current_close
                trade_profit = (exit_price - position['entry_price']) * multiplier
                trade_profit -= commission_per_order
                trade_results.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'direction': 'long',
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': trade_profit
                })
                logger.info(f"Exiting LONG trade at {current_time} | Exit Price: {exit_price:.2f} | Profit: {trade_profit:.2f}")
                capital += trade_profit
                in_position = False
                position = None

        elif position['direction'] == 'short':
            new_stop = row['Low'] + atr_multiplier * row['ATR']
            position['trailing_stop'] = min(position['trailing_stop'], new_stop)
            if row['High'] >= position['trailing_stop']:
                exit_price = current_close
                trade_profit = (position['entry_price'] - exit_price) * multiplier
                trade_profit -= commission_per_order
                trade_results.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'direction': 'short',
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': trade_profit
                })
                logger.info(f"Exiting SHORT trade at {current_time} | Exit Price: {exit_price:.2f} | Profit: {trade_profit:.2f}")
                capital += trade_profit
                in_position = False
                position = None

    # --- Entry Logic: Evaluate at the candle close ---
    if not in_position and i >= lookback_breakout:
        # Check volume condition at close.
        if row['Volume'] > volume_multiplier * row['avg_volume_10']:
            # For long entries: if close > previous 10-bar high and close > both the daily MA and the hourly 200-day MA.
            if (current_close > row['prev_10_high']) and (current_close > row['MA200']) and (current_close > row['MA_Hourly']):
                entry_price = current_close
                in_position = True
                capital -= commission_per_order
                position = {
                    'direction': 'long',
                    'entry_price': entry_price,
                    'entry_time': current_time,
                    'trailing_stop': entry_price - atr_multiplier * row['ATR']
                }
                logger.info(f"Entering LONG trade at {current_time} | Entry Price: {entry_price:.2f} | Initial Stop: {position['trailing_stop']:.2f}")
            # For short entries: if close < previous 10-bar low and close < both daily MA and hourly MA.
            elif (current_close < row['prev_10_low']) and (current_close < row['MA200']) and (current_close < row['MA_Hourly']):
                entry_price = current_close
                in_position = True
                capital -= commission_per_order
                position = {
                    'direction': 'short',
                    'entry_price': entry_price,
                    'entry_time': current_time,
                    'trailing_stop': entry_price + atr_multiplier * row['ATR']
                }
                logger.info(f"Entering SHORT trade at {current_time} | Entry Price: {entry_price:.2f} | Initial Stop: {position['trailing_stop']:.2f}")
    
    # --- Mark-to-Market Equity Calculation ---
    if in_position:
        if position['direction'] == 'long':
            unrealized = (current_close - position['entry_price']) * multiplier
        else:
            unrealized = (position['entry_price'] - current_close) * multiplier
        equity = capital + unrealized
    else:
        equity = capital
    equity_curve.append((current_time, equity))

# --- Close any open position at the end ---
if in_position:
    row = data.iloc[-1]
    current_time = row['Time']
    current_close = row['Last']
    if position['direction'] == 'long':
        exit_price = current_close
        trade_profit = (exit_price - position['entry_price']) * multiplier
    else:
        exit_price = current_close
        trade_profit = (position['entry_price'] - exit_price) * multiplier
    trade_profit -= commission_per_order
    trade_results.append({
        'entry_time': position['entry_time'],
        'exit_time': current_time,
        'direction': position['direction'],
        'entry_price': position['entry_price'],
        'exit_price': exit_price,
        'profit': trade_profit
    })
    logger.info(f"Closing open {position['direction'].upper()} trade at end {current_time} | Exit Price: {exit_price:.2f} | Profit: {trade_profit:.2f}")
    capital += trade_profit
    equity = capital
    in_position = False
    position = None
    equity_curve[-1] = (current_time, equity)

# -------------------------------
# Convert Equity Curve to DataFrame
# -------------------------------
equity_df = pd.DataFrame(equity_curve, columns=['Time', 'Equity'])
equity_df.set_index('Time', inplace=True)

# -------------------------------
# Performance Metrics Calculation
# -------------------------------
total_bars = len(data)
exposure_time_percentage = (exposure_bars / total_bars) * 100
final_account_balance = capital
total_return_percentage = ((final_account_balance / initial_capital) - 1) * 100

start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)
years = (end_dt - start_dt).days / 365.25
annualized_return_percentage = ((final_account_balance / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else np.nan

initial_close = data['Last'].iloc[0]
benchmark_equity = (data.set_index('Time')['Last'] / initial_close) * initial_capital
benchmark_equity = benchmark_equity.reindex(equity_df.index, method='ffill').fillna(method='ffill')

equity_df['returns'] = equity_df['Equity'].pct_change()
volatility_annual = equity_df['returns'].std() * np.sqrt(1512) * 100

total_trades = len(trade_results)
winning_trades = [t for t in trade_results if t['profit'] > 0]
losing_trades = [t for t in trade_results if t['profit'] <= 0]
win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
profit_factor = (sum(t['profit'] for t in winning_trades) / abs(sum(t['profit'] for t in losing_trades))
                 if losing_trades and sum(t['profit'] for t in losing_trades) != 0 else np.nan)

avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else np.nan
avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else np.nan

sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(1512)
                if equity_df['returns'].std() != 0 else np.nan)
downside_std = equity_df[equity_df['returns'] < 0]['returns'].std()
sortino_ratio = (equity_df['returns'].mean() / downside_std * np.sqrt(1512)
                 if downside_std != 0 else np.nan)

equity_df['EquityPeak'] = equity_df['Equity'].cummax()
equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['EquityPeak']) / equity_df['EquityPeak']
max_drawdown_percentage = equity_df['Drawdown'].min() * 100
equity_df['DrawdownDollar'] = equity_df['EquityPeak'] - equity_df['Equity']
max_drawdown_dollar = equity_df['DrawdownDollar'].max()
average_drawdown_dollar = equity_df.loc[equity_df['DrawdownDollar'] > 0, 'DrawdownDollar'].mean()
average_drawdown_percentage = equity_df['Drawdown'].mean() * 100
calmar_ratio = (annualized_return_percentage / abs(max_drawdown_percentage)
                if max_drawdown_percentage != 0 else np.nan)

drawdown_durations = []
current_duration = 0
for eq, peak in zip(equity_df['Equity'], equity_df['EquityPeak']):
    if eq < peak:
        current_duration += 1
    else:
        if current_duration > 0:
            drawdown_durations.append(current_duration)
        current_duration = 0
if current_duration > 0:
    drawdown_durations.append(current_duration)
if drawdown_durations:
    max_drawdown_duration_days = (max(drawdown_durations) * 4) / 24
    average_drawdown_duration_days = (np.mean(drawdown_durations) * 4) / 24
else:
    max_drawdown_duration_days = 0
    average_drawdown_duration_days = 0

results = {
    "Start Date": start_date,
    "End Date": end_date,
    "Exposure Time": f"{exposure_time_percentage:.2f}%",
    "Final Account Balance": f"${final_account_balance:,.2f}",
    "Equity Peak": f"${equity_df['Equity'].cummax().iloc[-1]:,.2f}",
    "Total Return": f"{total_return_percentage:.2f}%",
    "Annualized Return": f"{annualized_return_percentage:.2f}%",
    "Benchmark Return": f"{((data['Last'].iloc[-1]/initial_close)-1)*100:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Total Trades": total_trades,
    "Winning Trades": len(winning_trades),
    "Losing Trades": len(losing_trades),
    "Win Rate": f"{win_rate:.2f}%",
    "Profit Factor": f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "NaN",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "NaN",
    "Sortino Ratio": f"{sortino_ratio:.2f}" if not np.isnan(sortino_ratio) else "NaN",
    "Calmar Ratio": f"{calmar_ratio:.2f}" if not np.isnan(calmar_ratio) else "NaN",
    "Max Drawdown (%)": f"{max_drawdown_percentage:.2f}%",
    "Average Drawdown (%)": f"{average_drawdown_percentage:.2f}%",
    "Max Drawdown ($)": f"${max_drawdown_dollar:,.2f}",
    "Average Drawdown ($)": f"${average_drawdown_dollar:,.2f}",
    "Max Drawdown Duration": f"{max_drawdown_duration_days:.2f} days",
    "Average Drawdown Duration": f"{average_drawdown_duration_days:.2f} days",
    "Average Win ($)": f"${avg_win:,.2f}",
    "Average Loss ($)": f"${avg_loss:,.2f}",
}

print("\nPerformance Summary:")
for key, value in results.items():
    print(f"{key:30}: {value:>15}")

plt.figure(figsize=(14, 7))
plt.plot(equity_df.index, equity_df['Equity'], label='Strategy Equity')
plt.plot(benchmark_equity.index, benchmark_equity, label='Benchmark Equity (Buy & Hold)', alpha=0.7)
plt.title('Equity Curve: Strategy vs Benchmark')
plt.xlabel('Time')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()