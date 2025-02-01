import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import time, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Parameters
# ----------------------------
ATR_PERIOD = 14                # ATR lookback period
THRESHOLD_MULTIPLIER = 1.0       # Multiplier for ATR to determine entry threshold
INITIAL_CASH = 5000            # Starting cash (equity)
RF_RATE = 0.0                  # Risk-free rate for Sharpe ratio (annualized)
MULTIPLIER = 5                 # $5 per point for MES futures
COMMISSION = 1.24              # Commission per trade leg ($1.24 per entry or exit)
CONTRACTS = 1                  # Number of contracts per trade

# ----------------------------
# 1. Read and preprocess data
# ----------------------------
df = pd.read_csv("Data/es_1m_data.csv")

# Ensure the timestamp column is in datetime format and set as index.
# (Assuming your CSV has a column named "timestamp". Adjust if necessary.)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Filter to regular trading hours (assumed 09:30 to 16:00 local time)
df = df.between_time('09:30', '16:00')
logger.info(f"Data filtered to regular trading hours: {df.index.min()} to {df.index.max()}")

# Drop any rows with missing values (if any)
df = df.dropna()

# ----------------------------
# 2. Compute Intraday VWAP
# ----------------------------
# For each day, compute cumulative (price*volume) and cumulative volume, then VWAP.
df['pv'] = df['close'] * df['volume']
# Group by the date part of the index.
df['cum_pv'] = df.groupby(df.index.date)['pv'].cumsum()
df['cum_volume'] = df.groupby(df.index.date)['volume'].cumsum()
df['vwap'] = df['cum_pv'] / df['cum_volume']

# ----------------------------
# 3. Compute ATR
# ----------------------------
df['prev_close'] = df['close'].shift(1)
df['tr1'] = df['high'] - df['low']
df['tr2'] = (df['high'] - df['prev_close']).abs()
df['tr3'] = (df['low'] - df['prev_close']).abs()
df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
df['ATR'] = df['true_range'].rolling(window=ATR_PERIOD, min_periods=1).mean()

# Set threshold for entry signals using ATR
df['threshold'] = THRESHOLD_MULTIPLIER * df['ATR']

# ----------------------------
# 4. Generate Trading Signals
# ----------------------------
# Signal definitions:
#   +1 : Long (price is below VWAP by at least threshold)
#   -1 : Short (price is above VWAP by at least threshold)
#    0 : No signal
df['signal'] = 0
df.loc[df['close'] < (df['vwap'] - df['threshold']), 'signal'] = 1   # long signal
df.loc[df['close'] > (df['vwap'] + df['threshold']), 'signal'] = -1  # short signal

# ----------------------------
# 5. Backtest the Strategy
# ----------------------------
# We assume trading one MES futures contract per trade.
trades = []           # List to hold trade details
equity_timeline = []  # List of tuples (timestamp, equity_value)
position = 0          # 0 means no position, 1 means long, -1 means short
entry_price = 0.0
entry_time = None

cash = INITIAL_CASH

# For exposure time, record all periods in which we are in a trade.
in_trade_flags = []

# Iterate over each minute bar
for timestamp, row in df.iterrows():
    price = row['close']
    sig = row['signal']
    current_vwap = row['vwap']
    
    # Record if we are in a trade this minute
    in_trade_flags.append(1 if position != 0 else 0)
    
    # If not in a position, look to enter a trade
    if position == 0:
        if sig == 1:
            # Enter long position
            entry_price = price
            entry_time = timestamp
            position = 1
            logger.info(f"Enter LONG at {timestamp} price: {price:.2f}")
        elif sig == -1:
            # Enter short position
            entry_price = price
            entry_time = timestamp
            position = -1
            logger.info(f"Enter SHORT at {timestamp} price: {price:.2f}")
    else:
        # Already in a position: look to exit when price reverts to VWAP
        if position == 1 and price >= current_vwap:
            exit_price = price
            exit_time = timestamp
            # For MES futures, PnL = (exit_price - entry_price)*MULTIPLIER*CONTRACTS,
            # subtract commission on entry and exit.
            pnl = (exit_price - entry_price) * MULTIPLIER * CONTRACTS - (COMMISSION * 2)
            cash += pnl
            trades.append({
                'Entry Time': entry_time,
                'Exit Time': exit_time,
                'Position': 'Long',
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'PnL': pnl
            })
            logger.info(f"Exit LONG at {timestamp} price: {price:.2f} | PnL: {pnl:.2f}")
            position = 0
        elif position == -1 and price <= current_vwap:
            exit_price = price
            exit_time = timestamp
            pnl = (entry_price - exit_price) * MULTIPLIER * CONTRACTS - (COMMISSION * 2)
            cash += pnl
            trades.append({
                'Entry Time': entry_time,
                'Exit Time': exit_time,
                'Position': 'Short',
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'PnL': pnl
            })
            logger.info(f"Exit SHORT at {timestamp} price: {price:.2f} | PnL: {pnl:.2f}")
            position = 0

    # Mark-to-market equity calculation:
    # If in a trade, update unrealized PnL; otherwise, equity is just cash.
    if position == 1:
        mtm = (price - entry_price) * MULTIPLIER * CONTRACTS
        equity_value = cash + mtm
    elif position == -1:
        mtm = (entry_price - price) * MULTIPLIER * CONTRACTS
        equity_value = cash + mtm
    else:
        equity_value = cash
    equity_timeline.append((timestamp, equity_value))

# If still in a position at the end of data, force an exit.
if position != 0:
    final_price = df.iloc[-1]['close']
    exit_time = df.index[-1]
    if position == 1:
        pnl = (final_price - entry_price) * MULTIPLIER * CONTRACTS - (COMMISSION * 2)
        trade_side = 'Long'
    else:
        pnl = (entry_price - final_price) * MULTIPLIER * CONTRACTS - (COMMISSION * 2)
        trade_side = 'Short'
    cash += pnl
    trades.append({
        'Entry Time': entry_time,
        'Exit Time': exit_time,
        'Position': trade_side,
        'Entry Price': entry_price,
        'Exit Price': final_price,
        'PnL': pnl
    })
    logger.info(f"Force exit {trade_side} at {exit_time} price: {final_price:.2f} | PnL: {pnl:.2f}")
    equity_timeline.append((exit_time, cash))
    in_trade_flags.append(0)
    position = 0

# Convert equity timeline into a pandas Series
balance_series = pd.Series(
    data=[equity for ts, equity in equity_timeline],
    index=[ts for ts, equity in equity_timeline]
)

# ----------------------------
# 6. Compute Performance Metrics
# ----------------------------
# Resample the original data to 30-minute bars for the summary (if needed)
df_30m_full = df.resample('30T').last()

# Exposure Time: percentage of time in a trade (using our in_trade_flags list)
total_minutes = len(df)
exposure_minutes = sum(in_trade_flags)
exposure_time_percentage = (exposure_minutes / total_minutes) * 100

# Final Account Balance
final_balance = cash

# Equity Peak (max of balance_series)
equity_peak = balance_series.cummax().max()

# Total Return Percentage
total_return_percentage = ((final_balance - INITIAL_CASH) / INITIAL_CASH) * 100

# Annualized Return (approximation)
days = (balance_series.index[-1] - balance_series.index[0]).total_seconds() / (3600*24)
annualized_return_percentage = ((final_balance / INITIAL_CASH) ** (365/days) - 1) * 100 if days > 0 else 0

# Benchmark: Buy and hold from first close to last close
initial_close = df_30m_full['close'].iloc[0]
final_close = df_30m_full['close'].iloc[-1]
benchmark_return = ((final_close / initial_close) - 1) * 100

# Calculate returns from balance_series for volatility and risk metrics.
balance_returns = balance_series.pct_change().dropna()
volatility_annual = balance_returns.std() * np.sqrt(252 * 6.5 * 60) * 100  # Annualizing using minutes in trading day

# Trade Stats
total_trades = len(trades)
winning_trades = [t for t in trades if t['PnL'] > 0]
losing_trades = [t for t in trades if t['PnL'] < 0]
win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
profit_factor = (sum(t['PnL'] for t in winning_trades) / abs(sum(t['PnL'] for t in losing_trades))
                 if sum(t['PnL'] for t in losing_trades) != 0 else np.nan)

# Sharpe Ratio (assume risk-free rate = 0)
sharpe_ratio = (balance_returns.mean() / balance_returns.std()) * np.sqrt(252 * 6.5 * 60) if balance_returns.std() != 0 else np.nan

# For Sortino Ratio, compute downside deviation
downside_returns = balance_returns[balance_returns < 0]
downside_std = downside_returns.std() if not downside_returns.empty else np.nan
sortino_ratio = (balance_returns.mean() / downside_std) * np.sqrt(252 * 6.5 * 60) if downside_std and downside_std != 0 else np.nan

# Calculate drawdowns from the equity curve
equity_df = pd.DataFrame({'Equity': balance_series})
equity_df['Peak'] = equity_df['Equity'].cummax()
equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak'] * 100
max_drawdown = equity_df['Drawdown'].min()
average_drawdown = equity_df['Drawdown'][equity_df['Drawdown'] < 0].mean() if (equity_df['Drawdown'] < 0).any() else 0

# Drawdown Duration: count consecutive minutes in drawdown, then convert to days.
drawdown_durations = []
current_duration = 0
for dd in equity_df['Drawdown']:
    if dd < 0:
        current_duration += 1
    else:
        if current_duration > 0:
            drawdown_durations.append(current_duration)
        current_duration = 0
if current_duration > 0:
    drawdown_durations.append(current_duration)

if drawdown_durations:
    max_drawdown_duration_minutes = max(drawdown_durations)
    average_drawdown_duration_minutes = np.mean(drawdown_durations)
else:
    max_drawdown_duration_minutes = average_drawdown_duration_minutes = 0

# Convert drawdown duration from minutes to days (assuming 6.5 hours = 390 minutes per trading day)
max_drawdown_duration_days = max_drawdown_duration_minutes / 390
average_drawdown_duration_days = average_drawdown_duration_minutes / 390

# Calmar Ratio: Annualized return divided by absolute max drawdown percentage.
calmar_ratio = (annualized_return_percentage / abs(max_drawdown)) if max_drawdown != 0 else np.nan

# ----------------------------
# 7. Print Performance Summary
# ----------------------------
print("\nPerformance Summary:")
results = {
    "Start Date": df_30m_full.index.min().strftime("%Y-%m-%d"),
    "End Date": df_30m_full.index.max().strftime("%Y-%m-%d"),
    "Exposure Time": f"{exposure_time_percentage:.2f}%",
    "Final Account Balance": f"${final_balance:,.2f}",
    "Equity Peak": f"${equity_peak:,.2f}",
    "Total Return": f"{total_return_percentage:.2f}%",
    "Annualized Return": f"{annualized_return_percentage:.2f}%",
    "Benchmark Return": f"{benchmark_return:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Total Trades": total_trades,
    "Winning Trades": len(winning_trades),
    "Losing Trades": len(losing_trades),
    "Win Rate": f"{win_rate:.2f}%",
    "Profit Factor": f"{profit_factor:.2f}",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}",
    "Sortino Ratio": f"{sortino_ratio:.2f}",
    "Calmar Ratio": f"{calmar_ratio:.2f}",
    "Max Drawdown": f"{max_drawdown:.2f}%",
    "Average Drawdown": f"{average_drawdown:.2f}%",
    "Max Drawdown Duration": f"{max_drawdown_duration_days:.2f} days",
    "Average Drawdown Duration": f"{average_drawdown_duration_days:.2f} days",
}

for key, value in results.items():
    print(f"{key:25}: {value:>15}")

# ----------------------------
# 8. Debug: Print Trade Details
# ----------------------------
print("\nTrade Details:")
for t in trades:
    print(f"{t['Position']:6} | Entry: {t['Entry Time']} @ {t['Entry Price']:.2f}  -->  Exit: {t['Exit Time']} @ {t['Exit Price']:.2f} | PnL: {t['PnL']:.2f}")

# ----------------------------
# 9. Plot Equity Curves
# ----------------------------
if len(balance_series) < 2:
    logger.warning("Not enough data points to plot equity curves.")
else:
    # Create a benchmark equity curve (buy-and-hold) based on the 30-minute close prices.
    initial_close = df_30m_full['close'].iloc[0]
    benchmark_equity = (df_30m_full['close'] / initial_close) * INITIAL_CASH
    benchmark_equity = benchmark_equity.reindex(balance_series.index, method='ffill')
    logger.info(f"Benchmark Equity Range: {benchmark_equity.index.min()} to {benchmark_equity.index.max()}")
    num_benchmark_nans = benchmark_equity.isna().sum()
    if num_benchmark_nans > 0:
        logger.warning(f"Benchmark equity has {num_benchmark_nans} NaN values. Filling with forward fill.")
        benchmark_equity = benchmark_equity.fillna(method='ffill')

    equity_df_plot = pd.DataFrame({
        'Strategy': balance_series,
        'Benchmark': benchmark_equity
    })

    plt.figure(figsize=(14, 7))
    plt.plot(equity_df_plot['Strategy'], label='Strategy Equity')
    plt.plot(equity_df_plot['Benchmark'], label='Benchmark Equity')
    plt.title('Equity Curve: Strategy vs Benchmark')
    plt.xlabel('Time')
    plt.ylabel('Account Balance ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()