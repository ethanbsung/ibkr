import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import re
from datetime import datetime, time

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# -------------------------------
# Parameters & User Settings
# -------------------------------

# Input file paths
data_file_5m = "Data/es_5m_data.csv"     # 5-minute data for intraday signals
data_file_daily = "Data/es_daily_data.csv"  # daily data for ATR

# Backtest parameters
initial_capital = 10000.0          # starting account balance in dollars
commission_per_order = 1.24        # commission per order (per contract)
num_contracts = 1                  # number of contracts to trade
multiplier = 5                   # multiplier for contract value (adjust if needed)

# ATR & Strategy parameters
ATRLength = 14
ATRStopMultiplier = 0.075  # 7.5% of ATR
market_open_time = time(8, 30)   # 9:30 AM
market_close_time = time(15, 0)  # 4:00 PM

# Custom start and end date (format: 'YYYY-MM-DD')
start_date = '2024-01-01'
end_date   = '2024-12-31'

# -------------------------------
# Helper Functions
# -------------------------------

def conv(x):
    """
    Converter function to extract numeric value from a string.
    It uses a regular expression to find the first occurrence of a numeric pattern.
    """
    # Convert x to string in case it's not
    x_str = str(x)
    match = re.search(r"[-+]?\d*\.?\d+", x_str)
    if match:
        return float(match.group(0))
    else:
        return np.nan

def compute_atr(df, period=14):
    """
    Compute ATR (Average True Range) for a daily DataFrame.
    Assumes columns: 'High', 'Low', 'Last'.
    """
    df = df.copy()
    df['prev_close'] = df['Last'].shift(1)
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = (df['High'] - df['prev_close']).abs()
    df['tr3'] = (df['Low'] - df['prev_close']).abs()
    df['TrueRange'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR'] = df['TrueRange'].rolling(period).mean()
    return df

def get_date(dt):
    """Return just the date (YYYY-MM-DD) from a pd.Timestamp or datetime."""
    return dt.date()

# -------------------------------
# Load Daily Data & Compute ATR
# -------------------------------
daily_data = pd.read_csv(data_file_daily, parse_dates=['Time'])
daily_data.sort_values('Time', inplace=True)

# Filter daily data by date range
daily_data = daily_data[(daily_data['Time'] >= start_date) & (daily_data['Time'] <= end_date)].reset_index(drop=True)

# Compute the 14-day ATR
daily_data = compute_atr(daily_data, ATRLength)

# Create a dictionary { date: ATR_value } for easy lookup
atr_dict = {}
for i, row in daily_data.iterrows():
    atr_date = get_date(row['Time'])
    atr_value = row['ATR']
    if not np.isnan(atr_value):
        atr_dict[atr_date] = atr_value

# -------------------------------
# Load 5-Minute Data
# -------------------------------
# Use converters to clean numeric fields that may contain non-numeric characters.
converters = {
    'Open': conv,
    'High': conv,
    'Low': conv,
    'Last': conv
}

data_5m = pd.read_csv(
    data_file_5m, 
    parse_dates=['Time'], 
    converters=converters,
    low_memory=False
)
data_5m.sort_values('Time', inplace=True)

# Optionally drop rows with missing values in the key columns
cols_to_check = ['Open', 'High', 'Low', 'Last']
data_5m.dropna(subset=cols_to_check, inplace=True)

# Filter 5-minute data by date range
data_5m = data_5m[(data_5m['Time'] >= start_date) & (data_5m['Time'] <= end_date)].reset_index(drop=True)

# -------------------------------
# Backtest Initialization
# -------------------------------
capital = initial_capital  # realized account equity
in_position = False        # flag if a trade is active
position = None            # dictionary to hold trade details
trade_results = []         # list to record completed trades
equity_curve = []          # list of (Time, mark-to-market Equity)

# We'll track daily first-bar detection
current_day = None
first_bar_open = None
first_bar_close = None
entered_for_the_day = False

# -------------------------------
# Main Backtest Loop
# -------------------------------
for i, row in data_5m.iterrows():
    current_time = row['Time']
    bar_date = get_date(current_time)
    bar_open = row['Open']
    bar_close = row['Last']
    bar_high = row['High']
    bar_low = row['Low']

    # If we're on a new day, reset daily logic
    if bar_date != current_day:
        current_day = bar_date
        first_bar_open = None
        first_bar_close = None
        entered_for_the_day = False

        # Force an exit of any open position from the previous day
        if in_position:
            exit_price = position['last_bar_price']
            profit = (exit_price - position['entry_price']) * multiplier * position['direction'] * num_contracts
            profit -= commission_per_order * num_contracts
            capital += profit
            trade_results.append({
                'entry_time': position['entry_time'],
                'exit_time': current_time,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'profit': profit,
                'contracts': num_contracts
            })
            logger.info(f"Forced EOD exit on new day {current_day}. Profit: {profit:.2f}")
            in_position = False
            position = None

    # Update the last bar price if in position
    if in_position:
        position['last_bar_price'] = bar_close

    # Process only Regular Trading Hours (RTH): 9:30 to 16:00
    bar_time = current_time.time()
    if bar_time < market_open_time or bar_time > market_close_time:
        if in_position:
            unrealized = (bar_close - position['entry_price']) * multiplier * position['direction'] * num_contracts
            equity = capital + unrealized
        else:
            equity = capital
        equity_curve.append((current_time, equity))
        continue

    # 1) Identify the FIRST 5-minute bar of the day (09:30 bar)
    if bar_time == time(9, 30):
        first_bar_open = bar_open

    # At the 09:35 bar, decide entry direction if not already entered
    if bar_time == time(9, 35) and first_bar_open is not None and (not entered_for_the_day):
        first_bar_close = bar_close

        # 2) Determine direction based on the first bar's move
        if first_bar_close > first_bar_open:
            entry_price = bar_close
            capital -= commission_per_order * num_contracts
            in_position = True
            entered_for_the_day = True
            position = {
                'entry_price': entry_price,
                'entry_time': current_time,
                'direction': 1,  # long
                'contracts': num_contracts,
                'last_bar_price': entry_price
            }
            logger.info(f"LONG entry on {current_time} at {entry_price:.2f}")
        elif first_bar_close < first_bar_open:
            entry_price = bar_close
            capital -= commission_per_order * num_contracts
            in_position = True
            entered_for_the_day = True
            position = {
                'entry_price': entry_price,
                'entry_time': current_time,
                'direction': -1,  # short
                'contracts': num_contracts,
                'last_bar_price': entry_price
            }
            logger.info(f"SHORT entry on {current_time} at {entry_price:.2f}")
        else:
            # No directional move; do nothing
            pass

    # 3) If in a position, check for STOP LOSS or EOD exit
    if in_position:
        today_atr = atr_dict.get(bar_date, 0.0)
        stop_loss_amt = today_atr * ATRStopMultiplier

        if position['direction'] == 1:
            stop_price = position['entry_price'] - stop_loss_amt
            if bar_low <= stop_price:
                exit_price = stop_price
                profit = (exit_price - position['entry_price']) * multiplier * num_contracts
                profit -= commission_per_order * num_contracts
                capital += profit
                trade_results.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': profit,
                    'contracts': num_contracts
                })
                logger.info(f"STOP LOSS hit (LONG) on {current_time} at {exit_price:.2f}, Profit: {profit:.2f}")
                in_position = False
                position = None
            elif bar_time == market_close_time:
                exit_price = bar_close
                profit = (exit_price - position['entry_price']) * multiplier * num_contracts
                profit -= commission_per_order * num_contracts
                capital += profit
                trade_results.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': profit,
                    'contracts': num_contracts
                })
                logger.info(f"EOD exit (LONG) on {current_time} at {exit_price:.2f}, Profit: {profit:.2f}")
                in_position = False
                position = None

        elif position['direction'] == -1:
            stop_price = position['entry_price'] + stop_loss_amt
            if bar_high >= stop_price:
                exit_price = stop_price
                profit = (exit_price - position['entry_price']) * multiplier * position['direction'] * num_contracts
                profit -= commission_per_order * num_contracts
                capital += profit
                trade_results.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': profit,
                    'contracts': num_contracts
                })
                logger.info(f"STOP LOSS hit (SHORT) on {current_time} at {exit_price:.2f}, Profit: {profit:.2f}")
                in_position = False
                position = None
            elif bar_time == market_close_time:
                exit_price = bar_close
                profit = (exit_price - position['entry_price']) * multiplier * position['direction'] * num_contracts
                profit -= commission_per_order * num_contracts
                capital += profit
                trade_results.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': profit,
                    'contracts': num_contracts
                })
                logger.info(f"EOD exit (SHORT) on {current_time} at {exit_price:.2f}, Profit: {profit:.2f}")
                in_position = False
                position = None

    # Mark-to-market equity calculation
    if in_position:
        unrealized = (bar_close - position['entry_price']) * multiplier * position['direction'] * num_contracts
        equity = capital + unrealized
    else:
        equity = capital
    equity_curve.append((current_time, equity))

# -------------------------------
# Close any open position at the end
# -------------------------------
if in_position and position is not None:
    row = data_5m.iloc[-1]
    current_time = row['Time']
    bar_close = row['Last']
    exit_price = bar_close
    profit = (exit_price - position['entry_price']) * multiplier * position['direction'] * num_contracts
    profit -= commission_per_order * num_contracts
    capital += profit
    trade_results.append({
        'entry_time': position['entry_time'],
        'exit_time': current_time,
        'entry_price': position['entry_price'],
        'exit_price': exit_price,
        'profit': profit,
        'contracts': num_contracts
    })
    logger.info(f"Final forced exit on {current_time} at {exit_price:.2f}, Profit: {profit:.2f}")
    equity = capital
    equity_curve.append((current_time, equity))
    in_position = False
    position = None

# -------------------------------
# Convert equity curve to DataFrame
# -------------------------------
equity_df = pd.DataFrame(equity_curve, columns=['Time', 'Equity'])
equity_df.set_index('Time', inplace=True)

# -------------------------------
# Performance Metrics
# -------------------------------
final_account_balance = capital
total_return_percentage = ((final_account_balance / initial_capital) - 1) * 100

start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)
years = (end_dt - start_dt).days / 365.25
annualized_return_percentage = ((final_account_balance / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else np.nan

equity_df['returns'] = equity_df['Equity'].pct_change()
volatility_annual = equity_df['returns'].std() * np.sqrt(252 * 78) * 100  # ~78 bars per day

trade_count = len(trade_results)
winning_trades = [t for t in trade_results if t['profit'] > 0]
losing_trades = [t for t in trade_results if t['profit'] <= 0]
win_rate = (len(winning_trades) / trade_count * 100) if trade_count > 0 else 0
profit_factor = (sum(t['profit'] for t in winning_trades) / abs(sum(t['profit'] for t in losing_trades))
                 if losing_trades and sum(t['profit'] for t in losing_trades) != 0 else np.nan)

avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else np.nan
avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else np.nan

if equity_df['returns'].std() != 0:
    sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std()) * np.sqrt(252 * 78)
else:
    sharpe_ratio = np.nan

downside_std = equity_df[equity_df['returns'] < 0]['returns'].std()
if downside_std != 0 and not np.isnan(downside_std):
    sortino_ratio = (equity_df['returns'].mean() / downside_std) * np.sqrt(252 * 78)
else:
    sortino_ratio = np.nan

equity_df['EquityPeak'] = equity_df['Equity'].cummax()
equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['EquityPeak']) / equity_df['EquityPeak']
max_drawdown_percentage = equity_df['Drawdown'].min() * 100
equity_df['DrawdownDollar'] = equity_df['EquityPeak'] - equity_df['Equity']
max_drawdown_dollar = equity_df['DrawdownDollar'].max()
average_drawdown_dollar = equity_df.loc[equity_df['DrawdownDollar'] > 0, 'DrawdownDollar'].mean()
average_drawdown_percentage = equity_df['Drawdown'].mean() * 100
calmar_ratio = annualized_return_percentage / abs(max_drawdown_percentage) if max_drawdown_percentage != 0 else np.nan

results = {
    "Start Date": start_date,
    "End Date": end_date,
    "Final Account Balance": f"${final_account_balance:,.2f}",
    "Total Return": f"{total_return_percentage:.2f}%",
    "Annualized Return": f"{annualized_return_percentage:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Total Trades": trade_count,
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
    "Average Drawdown ($)": f"${average_drawdown_dollar:,.2f}" if not np.isnan(average_drawdown_dollar) else "NaN",
    "Average Win ($)": f"${avg_win:,.2f}" if not np.isnan(avg_win) else "NaN",
    "Average Loss ($)": f"${avg_loss:,.2f}" if not np.isnan(avg_loss) else "NaN",
}

print("\nPerformance Summary:")
for key, value in results.items():
    print(f"{key:30}: {value:>15}")

# -------------------------------
# Plot Equity Curve
# -------------------------------
plt.figure(figsize=(14, 7))
plt.plot(equity_df.index, equity_df['Equity'], label='Strategy Equity')
plt.title('Equity Curve: 5-Minute Opening Range Breakout Strategy')
plt.xlabel('Time')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()