import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import math
import pytz

# ----------------------------- #
#          Parameters           #
# ----------------------------- #

# Backtest Period
BACKTEST_START = '2022-01-01'  # Start date in 'YYYY-MM-DD' format
BACKTEST_END = '2024-12-23'    # End date in 'YYYY-MM-DD' format

# SMA Parameters
SMA_TIMEFRAME = '1h'  # 1-hour timeframe for SMA ('h' stands for hours)
SMA_PERIOD = 25        # Number of SMA periods (each period is 1 hour)

# ATR Parameters
ATR_TIMEFRAME = '1h'    # ATR calculated on 1-hour timeframe
ATR_PERIOD = 14            # Standard ATR period
STOP_LOSS_ATR_MULTIPLE = 0.5  # Stop loss multiple of ATR
TAKE_PROFIT_ATR_MULTIPLE = 1  # Take profit multiple of ATR

# Trading Strategy Parameters
THRESHOLD = 50          # Points deviation to trigger trades
INITIAL_BALANCE = 5000  # Initial account balance in $
CONTRACT_SIZE = 1       # Number of MES contracts
MES_TICK_VALUE = 5      # $ per tick

# Slippage Parameters
SLIPPAGE_POINTS = 0     # Slippage in points (1 tick for ES futures)

# Commission Parameters
COMMISSION_PER_TRADE = 1.24  # Commission per round-trip trade in $

# File Path
DATA_FILE_PATH = 'Data/es_5m_data.csv'  # Replace with your actual file path

# ----------------------------- #
#       Load and Prepare Data    #
# ----------------------------- #

# Load ES Futures 5-Minute Data
# The CSV file should have columns: Symbol, Time, Open, High, Low, Last, Change, %Chg, Volume, Open Int
try:
    es_data = pd.read_csv(DATA_FILE_PATH, parse_dates=['Time'])
except FileNotFoundError:
    raise FileNotFoundError(f"The data file at '{DATA_FILE_PATH}' was not found.")
except Exception as e:
    raise Exception(f"An error occurred while reading the data file: {e}")

# Ensure data is sorted by Time
es_data.sort_values('Time', inplace=True)

# Rename columns for consistency
es_data.rename(columns={'Time': 'date', 'Last': 'close'}, inplace=True)

# Optional: Filter for a specific symbol if multiple symbols are present
# Uncomment and modify the line below if needed
# es_data = es_data[es_data['Symbol'] == 'ESU08']

# Filter data based on the backtest period
es_data = es_data[(es_data['date'] >= pd.to_datetime(BACKTEST_START)) & 
                (es_data['date'] <= pd.to_datetime(BACKTEST_END))].reset_index(drop=True)

# Check if data is sufficient
if es_data.empty:
    raise ValueError("No data available for the specified backtest period. Please check the dates and data file.")

# ----------------------------- #
#    Filter Regular Trading Hours (RTH)  #
# ----------------------------- #

# Define US/Eastern timezone
eastern = pytz.timezone('US/Eastern')

# Ensure 'date' is timezone-aware. Assuming the data is in UTC.
# If data is not in UTC, adjust the localization accordingly.
es_data['date'] = es_data['date'].dt.tz_localize('UTC').dt.tz_convert(eastern)

# Filter for weekdays (Monday=0 to Friday=4)
es_data = es_data[es_data['date'].dt.weekday < 5]

# Set 'date' as index to use between_time
es_data.set_index('date', inplace=True)

# Filter for RTH hours: 09:30 to 16:00 ET
es_data = es_data.between_time('09:30', '16:00')

# Convert back to UTC for backtesting
es_data = es_data.tz_convert('UTC')

# Reset index
es_data = es_data.reset_index()

# ----------------------------- #
#      Calculate 1-Hour SMA and ATR    #
# ----------------------------- #

# Resample to 1-hour bars
resampled_1h = es_data.set_index('date').resample(SMA_TIMEFRAME).agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'close': 'last',
    'Volume': 'sum'
}).dropna()

# Calculate 1-Hour SMA
resampled_1h['SMA_1H'] = resampled_1h['close'].rolling(window=SMA_PERIOD).mean()

# Calculate 1-Hour ATR
# ATR = max(high - low, abs(high - previous close), abs(low - previous close))
resampled_1h['prev_close'] = resampled_1h['close'].shift(1)
resampled_1h['TR'] = resampled_1h.apply(
    lambda row: max(row['High'] - row['Low'], 
                   abs(row['High'] - row['prev_close']), 
                   abs(row['Low'] - row['prev_close'])), axis=1)
resampled_1h['ATR_1H'] = resampled_1h['TR'].rolling(window=ATR_PERIOD).mean()

# Drop rows with NaN ATR
resampled_1h.dropna(subset=['ATR_1H'], inplace=True)

# Merge SMA and ATR back to 5-minute data
es_data = es_data.set_index('date').join(resampled_1h[['SMA_1H', 'ATR_1H']], how='left').reset_index()

# Forward-fill the SMA and ATR values to align with 5-minute data
es_data['SMA_1H'] = es_data['SMA_1H'].ffill()
es_data['ATR_1H'] = es_data['ATR_1H'].ffill()

# Calculate Deviation from 1-Hour SMA
es_data['Deviation'] = es_data['close'] - es_data['SMA_1H']

# ----------------------------- #
#       Initialize Variables     #
# ----------------------------- #

balance = INITIAL_BALANCE
position = 0  # 1 for long, -1 for short, 0 for no position
entry_price = 0
signal_price = 0
balance_history = []
positions_history = []
equity_history = []
trade_results = []
winning_trades = []
losing_trades = []
exposure_time = 0  # Number of periods in position
total_periods = len(es_data)
current_drawdown = 0
max_drawdown = 0
equity_peak = INITIAL_BALANCE
drawdowns = []
drawdown_durations = []
drawdown_start = None

# Initialize Benchmark Variables
benchmark_balance = INITIAL_BALANCE
benchmark_balance_history = []

# ----------------------------- #
#        Backtest Logic          #
# ----------------------------- #

# Iterate over the es_data to generate signals and execute trades
for idx, row in es_data.iterrows():
    # Update Benchmark Equity (Buy & Hold)
    if idx == 0:
        benchmark_balance_history.append(benchmark_balance)
    else:
        # Calculate benchmark return based on close price change
        prev_close = es_data.at[idx - 1, 'close']
        if prev_close != 0:
            benchmark_return = (row['close'] / prev_close) - 1
        else:
            benchmark_return = 0
        benchmark_balance *= (1 + benchmark_return)
        benchmark_balance_history.append(benchmark_balance)
    
    if np.isnan(row['SMA_1H']) or np.isnan(row['ATR_1H']):
        # Not enough data to compute SMA or ATR
        balance_history.append(balance)
        equity_history.append(balance)
        positions_history.append(position)
        benchmark_balance_history[-1] = benchmark_balance  # Keep benchmark_balance
        continue

    if position == 0:
        # Check for entry signals based on deviation crossing the threshold
        if row['Deviation'] <= -THRESHOLD:
            # Long Entry: Price is below SMA by THRESHOLD
            position = 1
            signal_price = row['SMA_1H'] - THRESHOLD  # SMA - THRESHOLD
            entry_price = signal_price + SLIPPAGE_POINTS  # Adjust for slippage (buy higher)
            
            # Calculate Stop Loss and Take Profit based on ATR
            stop_loss_price = entry_price - (row['ATR_1H'] * STOP_LOSS_ATR_MULTIPLE)
            take_profit_price = entry_price + (row['ATR_1H'] * TAKE_PROFIT_ATR_MULTIPLE)
            
            trade = {
                'Entry_Time': row['date'],
                'Signal_Price': signal_price,
                'Entry_Price': entry_price,
                'Position': 'Long',
                'Stop_Loss': stop_loss_price,
                'Take_Profit': take_profit_price,
                'Exit_Time': None,
                'Exit_Price': None,
                'Profit': None,
                'Reason': None
            }
            trade_results.append(trade)
            exposure_time += 1
            print(f"Long Entry Signal at {trade['Entry_Time']} | Signal Price: {signal_price:.2f} | Actual Entry Price: {entry_price:.2f}")
        elif row['Deviation'] >= THRESHOLD:
            # Short Entry: Price is above SMA by THRESHOLD
            position = -1
            signal_price = row['SMA_1H'] + THRESHOLD  # SMA + THRESHOLD
            entry_price = signal_price - SLIPPAGE_POINTS  # Adjust for slippage (sell lower)
            
            # Calculate Stop Loss and Take Profit based on ATR
            stop_loss_price = entry_price + (row['ATR_1H'] * STOP_LOSS_ATR_MULTIPLE)
            take_profit_price = entry_price - (row['ATR_1H'] * TAKE_PROFIT_ATR_MULTIPLE)
            
            trade = {
                'Entry_Time': row['date'],
                'Signal_Price': signal_price,
                'Entry_Price': entry_price,
                'Position': 'Short',
                'Stop_Loss': stop_loss_price,
                'Take_Profit': take_profit_price,
                'Exit_Time': None,
                'Exit_Price': None,
                'Profit': None,
                'Reason': None
            }
            trade_results.append(trade)
            exposure_time += 1
            print(f"Short Entry Signal at {trade['Entry_Time']} | Signal Price: {signal_price:.2f} | Actual Entry Price: {entry_price:.2f}")
    else:
        # Update exposure time
        exposure_time += 1

        # Check for exit signals based on take profit or stop loss
        exit_reason = None
        exit_price = None
        profit = 0

        if position == 1:
            # Long Position
            # Take Profit: Reached take profit price
            if row['close'] >= take_profit_price:
                exit_price = take_profit_price
                exit_reason = 'Take Profit'
            # Stop Loss: Reached stop loss price
            elif row['close'] <= stop_loss_price:
                exit_price = stop_loss_price
                exit_reason = 'Stop Loss'
        elif position == -1:
            # Short Position
            # Take Profit: Reached take profit price
            if row['close'] <= take_profit_price:
                exit_price = take_profit_price
                exit_reason = 'Take Profit'
            # Stop Loss: Reached stop loss price
            elif row['close'] >= stop_loss_price:
                exit_price = stop_loss_price
                exit_reason = 'Stop Loss'

        if exit_reason:
            # Calculate Profit
            if position == 1:
                profit = (exit_price - entry_price) * CONTRACT_SIZE * MES_TICK_VALUE
            elif position == -1:
                profit = (entry_price - exit_price) * CONTRACT_SIZE * MES_TICK_VALUE
            balance += profit
            # Deduct commission
            balance -= COMMISSION_PER_TRADE
            # Update trade details with net profit after commission
            trade_results[-1].update({
                'Exit_Time': row['date'],
                'Exit_Price': exit_price,
                'Profit': profit - COMMISSION_PER_TRADE,  # Net profit after commission
                'Reason': exit_reason
            })
            # Categorize trade based on net profit
            if (profit - COMMISSION_PER_TRADE) > 0:
                winning_trades.append(trade_results[-1])
            else:
                losing_trades.append(trade_results[-1])
            print(f"{trade_results[-1]['Position']} Exit at {trade_results[-1]['Exit_Time']} | "
                  f"Exit Price: {exit_price:.2f} | Profit: {trade_results[-1]['Profit']:.2f} | Reason: {exit_reason}")
            # Reset position
            position = 0
            entry_price = 0
            signal_price = 0

    # Update equity
    equity = balance
    if position != 0:
        # Mark to market
        if position == 1:
            unrealized_profit = (row['close'] - entry_price) * CONTRACT_SIZE * MES_TICK_VALUE
        elif position == -1:
            unrealized_profit = (entry_price - row['close']) * CONTRACT_SIZE * MES_TICK_VALUE
        equity = balance + unrealized_profit
    equity_history.append(equity)
    balance_history.append(balance if position == 0 else balance + unrealized_profit)
    positions_history.append(position)

    # Update Equity Peak and Drawdown
    if equity > equity_peak:
        equity_peak = equity
        if current_drawdown != 0:
            drawdowns.append(current_drawdown)
            if drawdown_start:
                drawdown_duration = (row['date'] - drawdown_start).days
                drawdown_durations.append(drawdown_duration)
                drawdown_start = None
            current_drawdown = 0
    else:
        drawdown = equity_peak - equity
        if drawdown > current_drawdown:
            current_drawdown = drawdown
            if drawdown_start is None:
                drawdown_start = row['date']
        if drawdown > max_drawdown:
            max_drawdown = drawdown

# Final Equity Peak Check
if current_drawdown != 0:
    drawdowns.append(current_drawdown)
    if drawdown_start:
        drawdown_duration = (es_data['date'].iloc[-1] - drawdown_start).days
        drawdown_durations.append(drawdown_duration)

# ----------------------------- #
#      Add Backtest Results      #
# ----------------------------- #

# Add balance, equity, and position to es_data
es_data['Balance'] = balance_history
es_data['Equity'] = equity_history
es_data['Position'] = positions_history

# Add Benchmark Equity to es_data
es_data['Benchmark_Equity'] = benchmark_balance_history

# Calculate Returns
es_data['Returns'] = es_data['Equity'].pct_change().fillna(0)
es_data['Benchmark_Return'] = es_data['close'].pct_change().fillna(0)
es_data['Benchmark_Equity'] = INITIAL_BALANCE * (1 + es_data['Benchmark_Return']).cumprod()

# ----------------------------- #
#     Performance Metrics        #
# ----------------------------- #

# Define start and end dates
start_date = es_data['date'].min().strftime("%Y-%m-%d %H:%M")
end_date = es_data['date'].max().strftime("%Y-%m-%d %H:%M")

# Exposure Time Percentage
exposure_time_percentage = (exposure_time / total_periods) * 100

# Final Balance and Equity Peak
final_balance = balance
equity_peak_final = equity_peak

# Total Return
total_return = (final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100

# Annualized Return
total_days = (es_data['date'].iloc[-1] - es_data['date'].iloc[0]).days
annualized_return = ((final_balance / INITIAL_BALANCE) ** (365 / total_days) - 1) * 100 if total_days > 0 else 0

# Benchmark Return
benchmark_return = (es_data['Benchmark_Equity'].iloc[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100

# Volatility (Annual)
# Adjusted for 5-minute data: 252 trading days * 6.5 trading hours * 12 5-min periods per hour = 252*6.5*12=19656 periods per year
trading_periods_per_year = 252 * 6.5 * 12  # 5-minute data
volatility_annual = es_data['Returns'].std() * math.sqrt(trading_periods_per_year)

# Win Rate and Profit Factor
total_trades = len(trade_results)
winning_trades_count = len(winning_trades)
losing_trades_count = len(losing_trades)
win_rate = (winning_trades_count / total_trades * 100) if total_trades > 0 else 0
sum_profits = sum([trade['Profit'] for trade in winning_trades])
sum_losses = abs(sum([trade['Profit'] for trade in losing_trades]))
profit_factor = (sum_profits / sum_losses) if sum_losses != 0 else math.inf

# Sharpe Ratio
risk_free_rate = 0.0  # Assuming risk-free rate is 0
sharpe_ratio = (es_data['Returns'].mean() - risk_free_rate) / es_data['Returns'].std() * math.sqrt(trading_periods_per_year) if es_data['Returns'].std() != 0 else 0

# Sortino Ratio
downside_returns = es_data['Returns'][es_data['Returns'] < 0]
sortino_ratio = (es_data['Returns'].mean() - risk_free_rate) / downside_returns.std() * math.sqrt(trading_periods_per_year) if downside_returns.std() != 0 else 0

# Calmar Ratio
calmar_ratio = (total_return / (max_drawdown / MES_TICK_VALUE)) if max_drawdown != 0 else math.inf

# Max Drawdown and Average Drawdown
max_drawdown_percentage = (max_drawdown / equity_peak_final) * 100 if equity_peak_final != 0 else 0
average_drawdown = (np.mean(drawdowns) / equity_peak_final) * 100 if drawdowns else 0

# Drawdown Durations
max_drawdown_duration_days = max(drawdown_durations) if drawdown_durations else 0
average_drawdown_duration_days = (np.mean(drawdown_durations) if drawdown_durations else 0)

# ----------------------------- #
#        Results Summary         #
# ----------------------------- #

# Prepare Results
results = {
    "Start Date": start_date,
    "End Date": end_date,
    "Exposure Time": f"{exposure_time_percentage:.2f}%",
    "Final Account Balance": f"${final_balance:,.2f}",
    "Equity Peak": f"${equity_peak_final:,.2f}",
    "Total Return": f"{total_return:.2f}%",
    "Annualized Return": f"{annualized_return:.2f}%",
    "Benchmark Return": f"{benchmark_return:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Total Trades": total_trades,
    "Winning Trades": winning_trades_count,
    "Losing Trades": losing_trades_count,
    "Win Rate": f"{win_rate:.2f}%",
    "Profit Factor": f"{profit_factor:.2f}",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}",
    "Sortino Ratio": f"{sortino_ratio:.2f}",
    "Calmar Ratio": f"{calmar_ratio:.2f}",
    "Max Drawdown": f"{max_drawdown_percentage:.2f}%",
    "Average Drawdown": f"{average_drawdown:.2f}%",
    "Max Drawdown Duration": f"{max_drawdown_duration_days:.2f} days",
    "Average Drawdown Duration": f"{average_drawdown_duration_days:.2f} days",
}

# Print Performance Summary
print("\nPerformance Summary:")
for key, value in results.items():
    print(f"{key:25}: {value:>15}")

# ----------------------------- #
#           Visualization        #
# ----------------------------- #

# Plot Equity Curve with Benchmark
plt.figure(figsize=(14, 7))
plt.plot(es_data['date'], equity_history, label='Strategy Equity', color='blue')
plt.plot(es_data['date'], benchmark_balance_history, label='Benchmark Equity (Buy & Hold)', color='orange', linestyle='--')
plt.title('Equity Curve Comparison')
plt.xlabel('Date')
plt.ylabel('Equity ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Drawdowns
plt.figure(figsize=(14, 4))
plt.plot(es_data['date'], equity_history, label='Equity', color='blue')
plt.fill_between(es_data['date'], equity_history, equity_peak_final - max_drawdown, 
                 where=np.array(equity_history) < (equity_peak_final - max_drawdown), 
                 color='red', alpha=0.3, label='Drawdown')
plt.title('Drawdown Visualization')
plt.xlabel('Date')
plt.ylabel('Equity ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------- #
#        Trade Log (Optional)    #
# ----------------------------- #

# Optional: Save trade results to a CSV file
# trades_df = pd.DataFrame(trade_results)
# trades_df.to_csv('trade_results_atr_adjusted.csv', index=False)
# print("Trade log saved to 'trade_results_atr_adjusted.csv'.")

# ----------------------------- #
#             End                #
# ----------------------------- #