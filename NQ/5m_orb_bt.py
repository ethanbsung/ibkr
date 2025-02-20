import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
import logging
import sys
import os

# -----------------------------
# Configuration Parameters
# -----------------------------
DATA_FILE = 'Data/nq_5m_data.csv'      # Path to your 5-minute data CSV
INITIAL_CAPITAL = 10000         # Starting capital in USD
COMMISSION_PER_SIDE = 0.85      # Commission per contract per leg (round-trip = $1.70)
SLIPPAGE = 0.25                  # 1 point slippage per leg
# Multiplier for MNQ futures contracts: $2 per point
POINT_VALUE = 2.0
# Custom Backtest Dates (inclusive)
START_DATE = '2016-01-01'       # Format: 'YYYY-MM-DD'
END_DATE   = '2024-12-31'       # Format: 'YYYY-MM-DD'

# -----------------------------
# Setup Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to reduce verbosity if desired
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# -----------------------------
# Helper Functions
# -----------------------------
def calculate_trade_outcome(trade_data, direction, entry_price, stop_loss, profit_target):
    """
    Given intraday candles (DataFrame) after entry, determine whether stop loss or profit target was hit.
    If neither is hit, exit at EOD close.
    """
    exit_price = None
    exit_time = None
    trade_exit = None  # 'target', 'stop', or 'eod'
    
    for idx, row in trade_data.iterrows():
        if direction == 'long':
            if row['Low'] <= stop_loss:
                exit_price = stop_loss  # assume fill at stop loss
                exit_time = row['Time']
                trade_exit = 'stop'
                break
            if row['High'] >= profit_target:
                exit_price = profit_target
                exit_time = row['Time']
                trade_exit = 'target'
                break
        elif direction == 'short':
            if row['High'] >= stop_loss:
                exit_price = stop_loss
                exit_time = row['Time']
                trade_exit = 'stop'
                break
            if row['Low'] <= profit_target:
                exit_price = profit_target
                exit_time = row['Time']
                trade_exit = 'target'
                break

    if exit_price is None:
        exit_price = trade_data.iloc[-1]['Close']
        exit_time = trade_data.iloc[-1]['Time']
        trade_exit = 'eod'
    
    return exit_price, exit_time, trade_exit

def compute_statistics(equity_curve, trade_results, initial_capital, trading_days):
    final_account_balance = equity_curve['Equity'].iloc[-1]
    total_return_percentage = ((final_account_balance / initial_capital) - 1) * 100

    years = trading_days / 252.0  # assuming 252 trading days per year
    annualized_return_percentage = ((final_account_balance / initial_capital) ** (1/years) - 1) * 100

    equity_curve['Daily Return'] = equity_curve['Equity'].pct_change().fillna(0)
    volatility_annual = equity_curve['Daily Return'].std() * np.sqrt(252) * 100

    equity_curve['Equity Peak'] = equity_curve['Equity'].cummax()
    equity_curve['Drawdown ($)'] = equity_curve['Equity Peak'] - equity_curve['Equity']
    equity_curve['Drawdown (%)'] = equity_curve['Drawdown ($)'] / equity_curve['Equity Peak'] * 100
    max_drawdown_dollar = equity_curve['Drawdown ($)'].max()
    max_drawdown_percentage = equity_curve['Drawdown (%)'].max()
    
    # Drawdown durations (in days)
    drawdown_periods = []
    duration_periods = []
    in_drawdown = False
    start_dd = None
    
    for idx, row in equity_curve.iterrows():
        if row['Equity'] < row['Equity Peak']:
            if not in_drawdown:
                in_drawdown = True
                start_dd = idx
        else:
            if in_drawdown:
                in_drawdown = False
                end_dd = idx
                dd_duration = (end_dd - start_dd).days
                duration_periods.append(dd_duration)
                drawdown_periods.append(row['Equity Peak'] - row['Equity'])
    average_drawdown_dollar = np.mean(equity_curve['Drawdown ($)'])
    average_drawdown_percentage = np.mean(equity_curve['Drawdown (%)'])
    max_drawdown_duration_days = max(duration_periods) if duration_periods else 0
    average_drawdown_duration_days = np.mean(duration_periods) if duration_periods else 0

    total_trades = len(trade_results)
    winning_trades = [t for t in trade_results if t['PnL'] > 0]
    losing_trades = [t for t in trade_results if t['PnL'] <= 0]
    win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

    sum_gains = sum([t['PnL'] for t in winning_trades])
    sum_losses = abs(sum([t['PnL'] for t in losing_trades]))
    profit_factor = (sum_gains / sum_losses) if sum_losses > 0 else np.nan

    sharpe_ratio = (equity_curve['Daily Return'].mean() / equity_curve['Daily Return'].std() * np.sqrt(252)
                    if equity_curve['Daily Return'].std() != 0 else np.nan)
    downside_std = equity_curve['Daily Return'][equity_curve['Daily Return'] < 0].std()
    sortino_ratio = (equity_curve['Daily Return'].mean() / downside_std * np.sqrt(252)
                     if downside_std != 0 else np.nan)
    calmar_ratio = (annualized_return_percentage / max_drawdown_percentage
                    if max_drawdown_percentage != 0 else np.nan)

    stats = {
        "Final Account Balance": final_account_balance,
        "Total Return (%)": total_return_percentage,
        "Annualized Return (%)": annualized_return_percentage,
        "Volatility (Annual %)": volatility_annual,
        "Total Trades": total_trades,
        "Winning Trades": len(winning_trades),
        "Losing Trades": len(losing_trades),
        "Win Rate (%)": win_rate,
        "Profit Factor": profit_factor,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
        "Max Drawdown (%)": max_drawdown_percentage,
        "Average Drawdown (%)": average_drawdown_percentage,
        "Max Drawdown ($)": max_drawdown_dollar,
        "Average Drawdown ($)": average_drawdown_dollar,
        "Max Drawdown Duration (days)": max_drawdown_duration_days,
        "Average Drawdown Duration (days)": average_drawdown_duration_days,
    }
    
    return stats

# -----------------------------
# Load and Prepare Data
# -----------------------------
logger.info("Loading data...")
data = pd.read_csv(DATA_FILE)
# Rename columns as needed; assume CSV columns: Symbol,Time,Open,High,Low,Last,Change,%Chg,Volume,Open Int
data.rename(columns={'Last': 'Close'}, inplace=True)
logger.info("Data loaded successfully.")

# Convert 'Time' to datetime and sort
data['Time'] = pd.to_datetime(data['Time'])
data.sort_values('Time', inplace=True)

# Create a 'Date' column (date only)
data['Date'] = data['Time'].dt.date

# Filter data by custom start and end dates
start_date_dt = datetime.strptime(START_DATE, "%Y-%m-%d").date()
end_date_dt = datetime.strptime(END_DATE, "%Y-%m-%d").date()
data = data[(data['Date'] >= start_date_dt) & (data['Date'] <= end_date_dt)]
logger.info(f"Data filtered from {START_DATE} to {END_DATE}. Total bars: {len(data)}")

# -----------------------------
# Compute Daily ATR (14-day)
# -----------------------------
logger.info("Aggregating intraday data to daily bars and computing 14-day ATR...")
daily_data = data.groupby('Date').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last'
}).reset_index()

daily_data['Prev_Close'] = daily_data['Close'].shift(1)

def calc_tr(row):
    if pd.isna(row['Prev_Close']):
        return row['High'] - row['Low']
    return max(row['High'] - row['Low'],
               abs(row['High'] - row['Prev_Close']),
               abs(row['Low'] - row['Prev_Close']))
    
daily_data['TR'] = daily_data.apply(calc_tr, axis=1)
daily_data['ATR'] = daily_data['TR'].rolling(window=14).mean()
atr_dict = pd.Series(daily_data.ATR.values, index=daily_data.Date).to_dict()
logger.info("Daily ATR computed.")

# -----------------------------
# Backtesting Loop
# -----------------------------
logger.info("Starting backtest loop...")
account_balance = INITIAL_CAPITAL
equity_records = []   # To record daily equity
trade_results = []    # List to store trade details

dates = data['Date'].unique()
for current_date in dates:
    # Skip days without ATR value (fewer than 14 days)
    if current_date not in atr_dict or pd.isna(atr_dict[current_date]):
        day_data = data[data['Date'] == current_date]
        equity_records.append({'Time': datetime.combine(current_date, datetime.max.time()), 
                               'Equity': account_balance})
        continue

    day_data = data[data['Date'] == current_date].reset_index(drop=True)
    
    # Ensure we use the bar at 8:30 am as the opening candle.
    candle_830 = day_data[day_data['Time'].dt.time == time(8, 30)]
    if candle_830.empty:
        logger.debug(f"{current_date}: No 8:30 bar found. Skipping trade for this day.")
        equity_records.append({'Time': datetime.combine(current_date, datetime.max.time()), 
                               'Equity': account_balance})
        continue
    first_candle = candle_830.iloc[0]
    
    # Get the next bar after 8:30 for the entry price.
    subsequent = day_data[day_data['Time'] > first_candle['Time']]
    if subsequent.empty:
        logger.debug(f"{current_date}: No bar after 8:30 found. Skipping trade for this day.")
        equity_records.append({'Time': datetime.combine(current_date, datetime.max.time()), 
                               'Equity': account_balance})
        continue
    second_candle = subsequent.iloc[0]

    # Determine trade direction based on the 8:30 candle
    if first_candle['Close'] > first_candle['Open']:
        direction = 'long'
    elif first_candle['Close'] < first_candle['Open']:
        direction = 'short'
    else:
        logger.debug(f"{current_date}: 8:30 candle is a doji. Skipping trade.")
        equity_records.append({'Time': datetime.combine(current_date, datetime.max.time()), 
                               'Equity': account_balance})
        continue

    # Entry is at the open of the next candle (after 8:30)
    entry_price = second_candle['Open']

    # Get the ATR for the current day (computed from the previous 14 days)
    atr_value = atr_dict[current_date]
    atr_component = 0.075 * atr_value

    # Log ATR details for verification
    logger.debug(f"{current_date}: ATR = {atr_value:.4f}, 7.5% of ATR = {atr_component:.4f}")

    # Calculate stop loss as 7.5% of ATR from the entry price
    if direction == 'long':
        stop_loss = entry_price - atr_component
        R = entry_price - stop_loss
        profit_target = entry_price + 10 * R
    else:
        stop_loss = entry_price + atr_component
        R = stop_loss - entry_price
        profit_target = entry_price - 10 * R

    if R <= 0:
        logger.debug(f"{current_date}: Computed risk R <= 0. Skipping trade.")
        equity_records.append({'Time': datetime.combine(current_date, datetime.max.time()), 
                               'Equity': account_balance})
        continue

    # Trade exactly 1 contract
    contracts = 1

    logger.debug(f"{current_date}: {direction.capitalize()} trade. Entry: {entry_price}, Stop: {stop_loss}, Target: {profit_target}, Contracts: {contracts}")

    # Use intraday data starting from the second candle (entry bar)
    intraday_data = day_data[day_data['Time'] >= second_candle['Time']].copy()
    exit_price, exit_time, exit_type = calculate_trade_outcome(intraday_data, direction, entry_price, stop_loss, profit_target)
    
    # Apply slippage: for long, add slippage on entry and subtract on exit; for short, vice versa.
    if direction == 'long':
        effective_entry_price = entry_price + SLIPPAGE
        effective_exit_price = exit_price - SLIPPAGE
        pnl_per_contract = effective_exit_price - effective_entry_price
    else:
        effective_entry_price = entry_price - SLIPPAGE
        effective_exit_price = exit_price + SLIPPAGE
        pnl_per_contract = effective_entry_price - effective_exit_price

    # Multiply the price difference by the point value for MNQ ($2 per point)
    trade_pnl = pnl_per_contract * POINT_VALUE * contracts
    total_commission = 1.70 * contracts  # round-trip commission per contract
    net_trade_pnl = trade_pnl - total_commission

    account_balance += net_trade_pnl

    trade_results.append({
        'Date': current_date,
        'Direction': direction,
        'Entry Price': entry_price,
        'Effective Entry': effective_entry_price,
        'Exit Price': exit_price,
        'Effective Exit': effective_exit_price,
        'Contracts': contracts,
        'PnL': net_trade_pnl,
        'Exit Type': exit_type
    })
    logger.debug(f"{current_date}: Trade completed with {exit_type}. PnL: {net_trade_pnl:.2f}")

    # Record end-of-day equity using last candle's close
    equity_records.append({'Time': datetime.combine(current_date, datetime.max.time()), 
                           'Equity': account_balance})

logger.info("Backtest loop completed.")

# Build equity curve DataFrame
equity_df = pd.DataFrame(equity_records)
equity_df.sort_values('Time', inplace=True)
equity_df.set_index('Time', inplace=True)

# -----------------------------
# Benchmark Calculation
# -----------------------------
logger.info("Calculating benchmark (Buy & Hold) equity curve...")
daily_close = data.groupby('Date').last()['Close']
initial_close = daily_close.iloc[0]
benchmark_equity = (daily_close / initial_close) * INITIAL_CAPITAL
benchmark_equity = benchmark_equity.reindex(equity_df.index, method='ffill').fillna(method='ffill')

# -----------------------------
# Compute Statistics
# -----------------------------
trading_days = len(equity_df)
stats = compute_statistics(equity_df.copy(), trade_results, INITIAL_CAPITAL, trading_days)

# -----------------------------
# Display Results
# -----------------------------
logger.info("Backtest Results:")
results = {
    "Start Date": equity_df.index[0].strftime("%Y-%m-%d"),
    "End Date": equity_df.index[-1].strftime("%Y-%m-%d"),
    "Final Account Balance": f"${stats['Final Account Balance']:,.2f}",
    "Total Return": f"{stats['Total Return (%)']:.2f}%",
    "Annualized Return": f"{stats['Annualized Return (%)']:.2f}%",
    "Benchmark Return": f"{(benchmark_equity.iloc[-1]/INITIAL_CAPITAL - 1)*100:.2f}%",
    "Volatility (Annual)": f"{stats['Volatility (Annual %)']:.2f}%",
    "Total Trades": stats['Total Trades'],
    "Winning Trades": stats['Winning Trades'],
    "Losing Trades": stats['Losing Trades'],
    "Win Rate": f"{stats['Win Rate (%)']:.2f}%",
    "Profit Factor": f"{stats['Profit Factor']:.2f}",
    "Sharpe Ratio": f"{stats['Sharpe Ratio']:.2f}",
    "Sortino Ratio": f"{stats['Sortino Ratio']:.2f}",
    "Calmar Ratio": f"{stats['Calmar Ratio']:.2f}",
    "Max Drawdown (%)": f"{stats['Max Drawdown (%)']:.2f}%",
    "Average Drawdown (%)": f"{stats['Average Drawdown (%)']:.2f}%",
    "Max Drawdown ($)": f"${stats['Max Drawdown ($)']:,.2f}",
    "Average Drawdown ($)": f"${stats['Average Drawdown ($)']:,.2f}",
    "Max Drawdown Duration": f"{stats['Max Drawdown Duration (days)']:.2f} days",
    "Average Drawdown Duration": f"{stats['Average Drawdown Duration (days)']:.2f} days",
}

for key, value in results.items():
    print(f"{key:30}: {value:>15}")

# -----------------------------
# Plot Equity Curve vs Benchmark
# -----------------------------
plt.figure(figsize=(14, 7))
plt.plot(equity_df.index, equity_df['Equity'], label='Strategy Equity')
#plt.plot(benchmark_equity.index, benchmark_equity, label='Benchmark Equity (Buy & Hold)', alpha=0.7)
plt.title('Equity Curve: Strategy vs Benchmark')
plt.xlabel('Time')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()