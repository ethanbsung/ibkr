import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# -------------------------------
# Parameters & User Settings
# -------------------------------
data_file = "Data/mes_daily_data.csv"  # Path to your CSV file (format: Symbol,Time,Open,High,Low,Last,...)
initial_capital = 10000.0
multiplier = 5
commission_per_order = 1.24
contracts = 1

# Bollinger Band Settings
bb_period = 20         # e.g., 20-period Bollinger band
bb_stddev = 2.0        # e.g., 2 standard deviations

# ATR Settings
atr_period = 14        # ATR lookback period
atr_stop_multiplier = 5.0  # Stop loss will be set at 1 ATR away (change as needed)

start_date = '2000-01-01'
end_date   = '2020-01-01'

# -------------------------------
# Data Preparation
# -------------------------------
df = pd.read_csv(data_file, parse_dates=['Time'])
logger.info(f"Loaded {len(df)} rows from {data_file}")
logger.info(f"Date range in data: {df['Time'].min()} to {df['Time'].max()}")

# Rename "Last" to "Close" if needed
if 'Last' in df.columns and 'Close' not in df.columns:
    df.rename(columns={'Last': 'Close'}, inplace=True)

# Verify required columns exist
required_columns = ['Time', 'Close', 'High', 'Low']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Convert key columns to numeric (in case they were read as strings)
for col in ['Close', 'High', 'Low']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.sort_values('Time', inplace=True)

# Filter by date range
logger.info(f"Filtering data between {start_date} and {end_date}")
df = df[(df['Time'] >= start_date) & (df['Time'] <= end_date)].copy()
logger.info(f"After filtering: {len(df)} rows")

if len(df) < bb_period:
    raise ValueError(
        f"Insufficient data ({len(df)} rows) for analysis. Need at least {bb_period} rows. "
        "Please adjust your 'start_date'/'end_date' or check the CSV."
    )

df.reset_index(drop=True, inplace=True)

# --- Compute Bollinger Bands ---
df['BB_Middle'] = df['Close'].rolling(bb_period).mean()
df['BB_Std']    = df['Close'].rolling(bb_period).std()
df['BB_Upper']  = df['BB_Middle'] + bb_stddev * df['BB_Std']
df['BB_Lower']  = df['BB_Middle'] - bb_stddev * df['BB_Std']

# --- Compute ATR (14-period) ---
df['PrevClose'] = df['Close'].shift(1)
df['TR_1'] = df['High'] - df['Low']
df['TR_2'] = (df['High'] - df['PrevClose']).abs()
df['TR_3'] = (df['Low'] - df['PrevClose']).abs()
df['TR']   = df[['TR_1', 'TR_2', 'TR_3']].max(axis=1)
df['ATR']  = df['TR'].rolling(atr_period).mean()

# Log the row count before dropping NaN values
logger.info(f"Before dropping NaN values: {len(df)} rows")

# Store initial close value before dropping NaN values
initial_close = df['Close'].iloc[0]

# Drop NaNs only from the columns we use for calculations
df.dropna(subset=['Close', 'High', 'Low', 'BB_Middle', 'BB_Std', 'BB_Upper', 'BB_Lower', 'ATR'], inplace=True)
logger.info(f"After dropping NaN values: {len(df)} rows")
df.reset_index(drop=True, inplace=True)

# -------------------------------
# Backtest Simulation
# -------------------------------
capital = initial_capital
in_position = False
position = None
trade_results = []
equity_curve = []

for i in range(len(df)):
    row = df.loc[i]
    current_time  = row['Time']
    current_price = row['Close']

    # Calculate mark-to-market equity
    if in_position:
        if position['direction'] == 'long':
            unrealized = (current_price - position['entry_price']) * multiplier * position['contracts']
        else:  # short
            unrealized = (position['entry_price'] - current_price) * multiplier * position['contracts']
        equity = capital + unrealized
    else:
        equity = capital
    equity_curve.append((current_time, equity))

    # Check exit conditions if in a position
    if in_position:
        if position['direction'] == 'long':
            # Exit if High touches or exceeds take_profit (price reverts to mean)
            if row['High'] >= position['take_profit']:
                exit_price = position['take_profit']
                trade_profit = ((exit_price - position['entry_price']) * multiplier * position['contracts']) - commission_per_order
                trade_results.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'direction': 'long',
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': trade_profit
                })
                logger.info(f"Long TP hit at {current_time} | Exit: {exit_price:.2f} | Entry: {position['entry_price']:.2f} | Profit: {trade_profit:.2f}")
                capital += trade_profit
                in_position = False
                position = None

            # Exit if Low touches or falls below stop_loss
            elif row['Low'] <= position['stop_loss']:
                exit_price = position['stop_loss']
                trade_profit = ((exit_price - position['entry_price']) * multiplier * position['contracts']) - commission_per_order
                trade_results.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'direction': 'long',
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': trade_profit
                })
                logger.info(f"Long SL hit at {current_time} | Exit: {exit_price:.2f} | Entry: {position['entry_price']:.2f} | Profit: {trade_profit:.2f}")
                capital += trade_profit
                in_position = False
                position = None

        else:  # Short position
            if row['Low'] <= position['take_profit']:
                exit_price = position['take_profit']
                trade_profit = ((position['entry_price'] - exit_price) * multiplier * position['contracts']) - commission_per_order
                trade_results.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'direction': 'short',
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': trade_profit
                })
                logger.info(f"Short TP hit at {current_time} | Exit: {exit_price:.2f} | Entry: {position['entry_price']:.2f} | Profit: {trade_profit:.2f}")
                capital += trade_profit
                in_position = False
                position = None

            elif row['High'] >= position['stop_loss']:
                exit_price = position['stop_loss']
                trade_profit = ((position['entry_price'] - exit_price) * multiplier * position['contracts']) - commission_per_order
                trade_results.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'direction': 'short',
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': trade_profit
                })
                logger.info(f"Short SL hit at {current_time} | Exit: {exit_price:.2f} | Entry: {position['entry_price']:.2f} | Profit: {trade_profit:.2f}")
                capital += trade_profit
                in_position = False
                position = None

    # Check entry signals if not in a position
    if not in_position:
        if row['Close'] < row['BB_Lower']:
            # Enter long when price is below lower band
            entry_price = row['Close']
            take_profit = row['BB_Middle']  # take profit when price reverts to the mean (middle band)
            stop_loss   = entry_price - atr_stop_multiplier * row['ATR']  # adjustable ATR stop loss
            capital -= commission_per_order
            in_position = True
            position = {
                'direction': 'long',
                'entry_time': current_time,
                'entry_price': entry_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'contracts': contracts
            }
            logger.info(f"Entering LONG at {current_time} | Price: {entry_price:.2f} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}")
        elif row['Close'] > row['BB_Upper']:
            # Enter short when price is above upper band
            entry_price = row['Close']
            take_profit = row['BB_Middle']  # take profit when price reverts to the mean (middle band)
            stop_loss   = entry_price + atr_stop_multiplier * row['ATR']  # adjustable ATR stop loss
            capital -= commission_per_order
            in_position = True
            position = {
                'direction': 'short',
                'entry_time': current_time,
                'entry_price': entry_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'contracts': contracts
            }
            logger.info(f"Entering SHORT at {current_time} | Price: {entry_price:.2f} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}")

# Close any open position at the end of the data series
if in_position:
    last_idx = df.index[-1]
    row = df.loc[last_idx]
    current_time  = row['Time']
    current_price = row['Close']
    if position['direction'] == 'long':
        exit_price = current_price
        trade_profit = ((exit_price - position['entry_price']) * multiplier * position['contracts']) - commission_per_order
    else:
        exit_price = current_price
        trade_profit = ((position['entry_price'] - exit_price) * multiplier * position['contracts']) - commission_per_order
    trade_results.append({
        'entry_time': position['entry_time'],
        'exit_time': current_time,
        'direction': position['direction'],
        'entry_price': position['entry_price'],
        'exit_price': exit_price,
        'profit': trade_profit
    })
    logger.info(f"Closing open {position['direction'].upper()} at end {current_time} | Exit: {exit_price:.2f} | Entry: {position['entry_price']:.2f} | Profit: {trade_profit:.2f}")
    capital += trade_profit
    equity_curve[-1] = (current_time, capital)
    in_position = False
    position = None

# Convert equity curve to DataFrame
equity_df = pd.DataFrame(equity_curve, columns=['Time', 'Equity'])
equity_df.set_index('Time', inplace=True)

# Benchmark calculation (move after equity curve creation)
df.set_index('Time', inplace=True)
benchmark_equity = (df['Close'] / initial_close) * initial_capital
benchmark_equity = benchmark_equity.reindex(equity_df.index).ffill()

# -------------------------------
# Performance Metrics Calculation
# -------------------------------
final_account_balance = capital
total_return_percentage = ((final_account_balance / initial_capital) - 1) * 100

start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)
years = (end_dt - start_dt).days / 365.25
if years <= 0:
    annualized_return_percentage = np.nan
else:
    annualized_return_percentage = ((final_account_balance / initial_capital) ** (1 / years) - 1) * 100

# Calculate returns and volatility
equity_df['returns'] = equity_df['Equity'].pct_change()
volatility_annual = equity_df['returns'].std() * np.sqrt(252) * 100  # approximate annualized volatility

total_trades = len(trade_results)
winning_trades = [t for t in trade_results if t['profit'] > 0]
losing_trades  = [t for t in trade_results if t['profit'] <= 0]
win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

if losing_trades and sum(t['profit'] for t in losing_trades) != 0:
    profit_factor = sum(t['profit'] for t in winning_trades) / abs(sum(t['profit'] for t in losing_trades))
else:
    profit_factor = np.nan

avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else np.nan
avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else np.nan

if equity_df['returns'].std() != 0:
    sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std()) * np.sqrt(252)
else:
    sharpe_ratio = np.nan

downside_std = equity_df[equity_df['returns'] < 0]['returns'].std()
if downside_std != 0:
    sortino_ratio = (equity_df['returns'].mean() / downside_std) * np.sqrt(252)
else:
    sortino_ratio = np.nan

equity_df['EquityPeak'] = equity_df['Equity'].cummax()
equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['EquityPeak']) / equity_df['EquityPeak']
max_drawdown_percentage = equity_df['Drawdown'].min() * 100
equity_df['DrawdownDollar'] = equity_df['EquityPeak'] - equity_df['Equity']
max_drawdown_dollar = equity_df['DrawdownDollar'].max()

results = {
    "Start Date": start_date,
    "End Date": end_date,
    "Final Account Balance": f"${final_account_balance:,.2f}",
    "Total Return": f"{total_return_percentage:.2f}%",
    "Annualized Return": f"{annualized_return_percentage:.2f}%",
    "Benchmark Return": f"{((df['Close'].iloc[-1]/initial_close)-1)*100:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Total Trades": total_trades,
    "Winning Trades": len(winning_trades),
    "Losing Trades": len(losing_trades),
    "Win Rate": f"{win_rate:.2f}%",
    "Profit Factor": f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "NaN",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "NaN",
    "Sortino Ratio": f"{sortino_ratio:.2f}" if not np.isnan(sortino_ratio) else "NaN",
    "Max Drawdown (%)": f"{max_drawdown_percentage:.2f}%",
    "Max Drawdown ($)": f"${max_drawdown_dollar:,.2f}",
    "Average Win ($)": f"${avg_win:,.2f}" if not np.isnan(avg_win) else "NaN",
    "Average Loss ($)": f"${avg_loss:,.2f}" if not np.isnan(avg_loss) else "NaN",
}

print("\nPerformance Summary:")
for key, value in results.items():
    print(f"{key:30}: {value:>15}")

# -------------------------------
# Plotting the Equity Curve vs. Benchmark
# -------------------------------
plt.figure(figsize=(14, 7))
plt.plot(equity_df.index, equity_df['Equity'], label='Strategy Equity')
#plt.plot(benchmark_equity.index, benchmark_equity, label='Benchmark Equity (Buy & Hold)', alpha=0.7)
plt.title('Equity Curve: Bollinger Band Mean Reversion Strategy vs Benchmark')
plt.xlabel('Time')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()