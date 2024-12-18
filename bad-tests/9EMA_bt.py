import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

# ------------------------
# Backtest Date Parameters
# ------------------------
START_DATE = '2022-12-18'  # Format: 'YYYY-MM-DD'
END_DATE = '2023-01-04'    # Format: 'YYYY-MM-DD'

# Convert start and end dates to datetime and localize to UTC
START_DATE = pd.to_datetime(START_DATE).tz_localize('UTC')
END_DATE = pd.to_datetime(END_DATE).tz_localize('UTC')

# ------------------------
# Parameters
# ------------------------
EMA_PERIOD = 9
UPTREND_LOOKBACK = 3  # Number of candles to check EMA slope
STOP_LOSS_POINTS = 5.00   # Stop-loss in points
TAKE_PROFIT_POINTS = 10.00 # Take-profit in points

INITIAL_CAPITAL = 5000.00  # Starting with $5,000
POSITION_SIZE = 1            # Number of contracts per trade
CONTRACT_MULTIPLIER = 5      # MES Futures multiplier

RISK_FREE_RATE = 0.0        # Assume 0% for Sharpe and Sortino calculations

# Transaction costs (optional)
COMMISSION_PER_TRADE = 1.24  # $1 per trade
SLIPPAGE = 0.25              # 0.05 points slippage

# ------------------------
# Data Load
# ------------------------
data = pd.read_csv("es_5m_data.csv")

# Ensure 'date' is converted to a proper datetime format if not already
if not pd.api.types.is_datetime64_any_dtype(data['date']):
    data['date'] = pd.to_datetime(data['date'])

data.set_index('date', inplace=True)
data = data.sort_index()  # Ensure chronological order

# ------------------------
# Verify Data Timezone
# ------------------------
print(f"Data Index Timezone: {data.index.tz}")  # For debugging purposes

# ------------------------
# Filter Data Based on Start and End Dates
# ------------------------
filtered_data = data.loc[(data.index >= START_DATE) & (data.index <= END_DATE)]

# Check if filtered_data is empty
if filtered_data.empty:
    raise ValueError("No data available in the specified date range. Please check the START_DATE and END_DATE.")

# Reset index for positional access
filtered_data = filtered_data.reset_index()

# ------------------------
# Indicator Calculation
# ------------------------
filtered_data['EMA9'] = filtered_data['close'].ewm(span=EMA_PERIOD, adjust=False).mean()

# ------------------------
# Backtest Logic and Performance Tracking
# ------------------------
position = None
entry_price = None
entry_time = None
trades = []
holding_periods = 0  # Tracks holding duration

# Performance tracking variables
cash = INITIAL_CAPITAL
equity = INITIAL_CAPITAL
equity_curve = []  # To track equity over time
drawdowns = []
max_equity = INITIAL_CAPITAL
total_exposure_periods = 0  # Total number of periods in a trade

for i in range(EMA_PERIOD, len(filtered_data)):
    current_time = filtered_data.loc[i, 'date']
    current_row = filtered_data.loc[i]
    
    # Record equity before any trade action
    equity_curve.append({'date': current_time, 'equity': equity})
    
    if position is None:
        # Look for buy signal
        # ------------------------ Buy Condition ------------------------
        def in_uptrend(idx):
            if idx < UPTREND_LOOKBACK:
                return False
            recent_emas = filtered_data['EMA9'].iloc[idx - UPTREND_LOOKBACK:idx+1]
            slope_positive = recent_emas.iloc[-1] > recent_emas.iloc[0]
            current_close = filtered_data.loc[idx, 'close']
            current_ema = filtered_data.loc[idx, 'EMA9']
            return (current_close > current_ema) and slope_positive
        
        def buy_condition(idx):
            if idx <= 0:
                return False
            if not in_uptrend(idx - 1):
                return False
            current = filtered_data.loc[idx]
            previous = filtered_data.loc[idx - 1]
            ema = current['EMA9']
            touched = (current['low'] <= ema) and (current['high'] >= ema)
            return previous['close'] > previous['EMA9'] and touched
        
        if buy_condition(i):
            # Enter long position
            position = 'long'
            entry_price = current_row['close'] + SLIPPAGE  # Adjust for slippage
            entry_time = current_time
            cash -= COMMISSION_PER_TRADE  # Subtract commission
            total_exposure_periods += 1  # Start counting exposure time
            holding_periods = 0  # Reset holding periods
            print(f"Entered long at {entry_price:.2f} on {entry_time}")
    else:
        # Manage open position
        if position == 'long':
            current_price = current_row['close']
            high = current_row['high']
            low = current_row['low']
            ema = current_row['EMA9']
            
            # Calculate potential exit prices
            stop_loss_price = entry_price - STOP_LOSS_POINTS
            take_profit_price = entry_price + TAKE_PROFIT_POINTS
            
            
            # Flags for exit conditions
            hit_stop_loss = low <= stop_loss_price
            hit_take_profit = high >= take_profit_price
            
            
            
            # Initialize exit variables
            exit_triggered = False
            exit_reason = ""
            exit_price = current_price  # Default exit price
            
            # Check Stop Loss
            if hit_stop_loss:
                exit_price = stop_loss_price - SLIPPAGE  # Adjust for slippage
                exit_reason = "Stop Loss hit"
                exit_triggered = True
            # Check Take Profit
            elif hit_take_profit:
                exit_price = take_profit_price + SLIPPAGE  # Adjust for slippage
                exit_reason = "Take Profit hit"
                exit_triggered = True
            
            
            
            if exit_triggered:
                if exit_reason in ["Exit Buffer hit", "Price below EMA", "Max Holding Period"]:
                    exit_price = current_price - SLIPPAGE  # Exit at current close price minus slippage
                # Calculate P&L with multiplier
                profit = (exit_price - entry_price) * POSITION_SIZE * CONTRACT_MULTIPLIER
                cash += profit
                cash -= COMMISSION_PER_TRADE  # Subtract commission
                equity = cash
                trades.append({
                    'EntryTime': entry_time,
                    'EntryPrice': entry_price,
                    'ExitTime': current_time,
                    'ExitPrice': exit_price,
                    'P&L': profit
                })
                print(f"{exit_reason}: Exited at {exit_price:.2f} on {current_time}, P&L: {profit:.2f}")
                position = None
                entry_price = None
                holding_periods = 0  # Reset holding periods
                continue  # Move to next candle
            
            # If none of the exit conditions are met, continue holding
            total_exposure_periods += 1  # Increment exposure time
            holding_periods += 1  # Increment holding periods
            # Update equity assuming mark-to-market
            equity = cash + (current_price - entry_price) * POSITION_SIZE * CONTRACT_MULTIPLIER
            # Optionally, record returns per period
            # returns.append((current_price - entry_price) / entry_price)

# After loop ends, close any open positions at the last close price
if position == 'long':
    exit_price = filtered_data.loc[len(filtered_data)-1, 'close'] - SLIPPAGE  # Adjust for slippage
    profit = (exit_price - entry_price) * POSITION_SIZE * CONTRACT_MULTIPLIER
    cash += profit
    cash -= COMMISSION_PER_TRADE  # Subtract commission
    equity = cash
    trades.append({
        'EntryTime': entry_time,
        'EntryPrice': entry_price,
        'ExitTime': filtered_data.loc[len(filtered_data)-1, 'date'],
        'ExitPrice': exit_price,
        'P&L': profit
    })
    print(f"Final Exit: Exited at {exit_price:.2f} on {filtered_data.loc[len(filtered_data)-1, 'date']}, P&L: {profit:.2f}")

# ------------------------
# Convert trades to a DataFrame for analysis
# ------------------------
trades_df = pd.DataFrame(trades)

# ------------------------
# Calculate Equity Curve
# ------------------------
equity_df = pd.DataFrame(equity_curve)
equity_df.set_index('date', inplace=True)
equity_df['equity'] = equity_df['equity'].ffill()

# ------------------------
# Calculate Drawdowns
# ------------------------
equity_df['cum_max'] = equity_df['equity'].cummax()
equity_df['drawdown'] = (equity_df['equity'] - equity_df['cum_max']) / equity_df['cum_max']
max_drawdown = equity_df['drawdown'].min() * 100  # in percentage
average_drawdown = equity_df['drawdown'].mean() * 100  # in percentage

# Calculate drawdown durations
equity_df['in_drawdown'] = equity_df['drawdown'] < 0
equity_df['drawdown_change'] = equity_df['in_drawdown'] != equity_df['in_drawdown'].shift(1)
drawdown_starts = equity_df[equity_df['drawdown_change'] & equity_df['in_drawdown']].index
drawdown_ends = equity_df[equity_df['drawdown_change'] & (~equity_df['in_drawdown'])].index

# Pair starts and ends
drawdown_durations = []
for start in drawdown_starts:
    end = drawdown_ends[drawdown_ends > start]
    end = end[0] if not end.empty else equity_df.index[-1]
    duration = (end - start).total_seconds() / (60 * 60 * 24)  # Convert to days
    drawdown_durations.append(duration)

max_drawdown_duration_days = max(drawdown_durations) if drawdown_durations else 0
average_drawdown_duration_days = np.mean(drawdown_durations) if drawdown_durations else 0

# ------------------------
# Calculate Returns
# ------------------------
equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)
total_return = (equity_df['equity'].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
total_return_percentage = total_return

# Calculate the number of years in the backtest
num_days = (equity_df.index[-1] - equity_df.index[0]).days + 1  # +1 to include the last day
num_years = num_days / 365.25

# Annualized Return
if num_years > 0:
    annualized_return = (1 + total_return / 100) ** (1 / num_years) - 1
    annualized_return_percentage = annualized_return * 100
else:
    annualized_return_percentage = 0.0

# Volatility (Annualized)
volatility_daily = equity_df['returns'].std()
volatility_annual = volatility_daily * np.sqrt(252 * 78) * 100  # 252 trading days, 78 5-min bars per day

# Sharpe Ratio
if volatility_daily != 0:
    sharpe_ratio = (equity_df['returns'].mean() - RISK_FREE_RATE / (252 * 78)) / volatility_daily * np.sqrt(252 * 78)
else:
    sharpe_ratio = 0.0

# Sortino Ratio
downside_returns = equity_df['returns'][equity_df['returns'] < 0]
if downside_returns.std() != 0:
    sortino_ratio = (equity_df['returns'].mean() - RISK_FREE_RATE / (252 * 78)) / downside_returns.std() * np.sqrt(252 * 78)
else:
    sortino_ratio = 0.0

# Calmar Ratio
if max_drawdown != 0:
    calmar_ratio = annualized_return / abs(max_drawdown / 100)
else:
    calmar_ratio = 0.0

# Profit Factor
total_wins = trades_df[trades_df['P&L'] > 0]['P&L'].sum()
total_losses = abs(trades_df[trades_df['P&L'] < 0]['P&L'].sum())
profit_factor = total_wins / total_losses if total_losses != 0 else np.inf

# Exposure Time Percentage
total_periods = len(filtered_data) - EMA_PERIOD
exposure_time_percentage = (total_exposure_periods / total_periods) * 100 if total_periods > 0 else 0

# Equity Peak
equity_peak = equity_df['equity'].max()

# Benchmark Return (Assume buy at first close and sell at last close)
benchmark_return = (filtered_data['close'].iloc[-1] - filtered_data['close'].iloc[EMA_PERIOD]) / filtered_data['close'].iloc[EMA_PERIOD] * 100

# ------------------------
# Identify Winning and Losing Trades
# ------------------------
winning_trades = trades_df[trades_df['P&L'] > 0]
losing_trades = trades_df[trades_df['P&L'] < 0]

# ------------------------
# Print Performance Summary
# ------------------------
print("\n--- Results Summary ---")

print("\nPerformance Summary:")
results = {
    "Start Date": equity_df.index.min().strftime("%Y-%m-%d"),
    "End Date": equity_df.index.max().strftime("%Y-%m-%d"),
    "Exposure Time": f"{exposure_time_percentage:.2f}%",
    "Final Account Balance": f"${cash:,.2f}",
    "Equity Peak": f"${equity_peak:,.2f}",
    "Total Return": f"{total_return_percentage:.2f}%",
    "Annualized Return": f"{annualized_return_percentage:.2f}%",
    "Benchmark Return": f"{benchmark_return:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Total Trades": len(trades_df),
    "Winning Trades": len(winning_trades),
    "Losing Trades": len(losing_trades),
    "Win Rate": f"{(len(winning_trades)/len(trades_df)*100) if len(trades_df) > 0 else 0:.2f}%",
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

# ------------------------
# Optional: Save trade details to a CSV file
# ------------------------
# trades_df.to_csv("backtest_results.csv", index=False)

# ------------------------
# Optional: Plot Equity Curve
# ------------------------
equity_df['equity'].plot(title='Equity Curve')
plt.xlabel('Date')
plt.ylabel('Equity ($)')
plt.show()