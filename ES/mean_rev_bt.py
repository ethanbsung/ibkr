import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# ------------- PARAMETERS -------------
initial_capital = 5000
# timeframe to test on (can be changed to '15T', '60T', etc.)
timeframe = '30min'  
bollinger_window = 20       # periods for Bollinger Bands
bollinger_std = 2           # standard deviation multiplier
stop_loss_points = 7        # fixed stop loss in points
take_profit_points = 15     # fixed take profit in points
contract_multiplier = 5     # assume 1 point = $1 (adjust if needed)
data_file = "ib_es_1m_data.csv"
# --------------------------------------

# --- 1. Load the 1-minute data ---
df = pd.read_csv(data_file, parse_dates=['date'])
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

# --- 2. Compute daily cumulative VWAP on the 1-minute data ---
# Typical price = (high + low + close) / 3
df['typical'] = (df['high'] + df['low'] + df['close']) / 3
df['tpv'] = df['typical'] * df['volume']
# Group by day (using index.date) and compute cumulative sums:
df['cum_tpv'] = df.groupby(df.index.date)['tpv'].cumsum()
df['cum_volume'] = df.groupby(df.index.date)['volume'].cumsum()
df['vwap'] = df['cum_tpv'] / df['cum_volume']

# --- 3. Resample data to the desired timeframe ---
# We aggregate open, high, low, close, volume and take the last VWAP value in the period.
agg_dict = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'vwap': 'last'
}
df_resampled = df.resample(timeframe).agg(agg_dict).dropna()

# --- 4. Calculate Bollinger Bands on the resampled data (using close price) ---
df_resampled['ma'] = df_resampled['close'].rolling(window=bollinger_window).mean()
df_resampled['std'] = df_resampled['close'].rolling(window=bollinger_window).std()
df_resampled['upper_band'] = df_resampled['ma'] + bollinger_std * df_resampled['std']
df_resampled['lower_band'] = df_resampled['ma'] - bollinger_std * df_resampled['std']
df_resampled.dropna(inplace=True)

# --- 5. Backtest Simulation ---
account_balance = initial_capital
equity_list = []     # to record account equity at each bar
equity_index = []    # corresponding timestamps
trade_results = []   # list to hold trade details
wins = []            # list of positive trade profits
losses = []          # list of negative trade profits

# state variables
in_trade = False
trade_entry_price = None
trade_entry_time = None
trade_direction = None  # either 'long' or 'short'
bars_in_trade = 0       # count bars with an open position

# We will use a “next‐bar entry” approach:
# On each bar (starting at index 1), check the *previous* bar for an entry signal.
# If a signal exists and no trade is active, then we “enter” at the current bar's open.
# Then, while in a trade, on each new bar we check if the bar’s high/low trigger our stop loss or take profit.

resampled_index = df_resampled.index
i = 1
while i < len(df_resampled):
    current_bar = df_resampled.iloc[i]
    current_time = resampled_index[i]
    
    # Mark-to-market equity: if in a trade, compute unrealized PnL using the current close.
    if in_trade:
        if trade_direction == 'long':
            unrealized = (current_bar['close'] - trade_entry_price) * contract_multiplier
        else:  # short
            unrealized = (trade_entry_price - current_bar['close']) * contract_multiplier
        current_equity = account_balance + unrealized
    else:
        current_equity = account_balance
        
    equity_list.append(current_equity)
    equity_index.append(current_time)
    
    # If not in a trade, check if the previous bar generated an entry signal.
    if not in_trade:
        prev_bar = df_resampled.iloc[i-1]
        # Long signal: previous bar’s close above its VWAP AND its low dipped below its lower Bollinger band.
        long_signal = (prev_bar['close'] > prev_bar['vwap']) and (prev_bar['low'] < prev_bar['lower_band'])
        # Short signal: previous bar’s close below its VWAP AND its high exceeded its upper Bollinger band.
        short_signal = (prev_bar['close'] < prev_bar['vwap']) and (prev_bar['high'] > prev_bar['upper_band'])
        
        if long_signal or short_signal:
            # Enter trade at the current bar's open.
            trade_entry_price = current_bar['open']
            trade_entry_time = current_time
            trade_direction = 'long' if long_signal else 'short'
            in_trade = True
            entry_bar = i  # record the bar index at which entry occurs

            # Immediately check the entry bar for an exit (if price gaps through the stop or TP)
            if trade_direction == 'long':
                sl_price = trade_entry_price - stop_loss_points
                tp_price = trade_entry_price + take_profit_points
                if current_bar['low'] <= sl_price:
                    # Stop loss triggered in the entry bar.
                    exit_price = sl_price
                    profit = (exit_price - trade_entry_price) * contract_multiplier
                    trade_exit_time = current_time
                    account_balance += profit
                    trade_results.append({
                        'entry_time': trade_entry_time,
                        'exit_time': trade_exit_time,
                        'direction': trade_direction,
                        'entry_price': trade_entry_price,
                        'exit_price': exit_price,
                        'profit': profit
                    })
                    (wins if profit > 0 else losses).append(profit)
                    in_trade = False
                    i += 1
                    continue  # move to the next bar
                elif current_bar['high'] >= tp_price:
                    # Take profit triggered.
                    exit_price = tp_price
                    profit = (exit_price - trade_entry_price) * contract_multiplier
                    trade_exit_time = current_time
                    account_balance += profit
                    trade_results.append({
                        'entry_time': trade_entry_time,
                        'exit_time': trade_exit_time,
                        'direction': trade_direction,
                        'entry_price': trade_entry_price,
                        'exit_price': exit_price,
                        'profit': profit
                    })
                    (wins if profit > 0 else losses).append(profit)
                    in_trade = False
                    i += 1
                    continue
            else:  # short trade entry
                sl_price = trade_entry_price + stop_loss_points
                tp_price = trade_entry_price - take_profit_points
                if current_bar['high'] >= sl_price:
                    exit_price = sl_price
                    profit = (trade_entry_price - exit_price) * contract_multiplier
                    trade_exit_time = current_time
                    account_balance += profit
                    trade_results.append({
                        'entry_time': trade_entry_time,
                        'exit_time': trade_exit_time,
                        'direction': trade_direction,
                        'entry_price': trade_entry_price,
                        'exit_price': exit_price,
                        'profit': profit
                    })
                    (wins if profit > 0 else losses).append(profit)
                    in_trade = False
                    i += 1
                    continue
                elif current_bar['low'] <= tp_price:
                    exit_price = tp_price
                    profit = (trade_entry_price - exit_price) * contract_multiplier
                    trade_exit_time = current_time
                    account_balance += profit
                    trade_results.append({
                        'entry_time': trade_entry_time,
                        'exit_time': trade_exit_time,
                        'direction': trade_direction,
                        'entry_price': trade_entry_price,
                        'exit_price': exit_price,
                        'profit': profit
                    })
                    (wins if profit > 0 else losses).append(profit)
                    in_trade = False
                    i += 1
                    continue
    else:
        # If already in a trade, check for exit conditions.
        if trade_direction == 'long':
            sl_price = trade_entry_price - stop_loss_points
            tp_price = trade_entry_price + take_profit_points
            if current_bar['low'] <= sl_price:
                exit_price = sl_price
                profit = (exit_price - trade_entry_price) * contract_multiplier
                trade_exit_time = current_time
                account_balance += profit
                trade_results.append({
                    'entry_time': trade_entry_time,
                    'exit_time': trade_exit_time,
                    'direction': trade_direction,
                    'entry_price': trade_entry_price,
                    'exit_price': exit_price,
                    'profit': profit
                })
                (wins if profit > 0 else losses).append(profit)
                in_trade = False
            elif current_bar['high'] >= tp_price:
                exit_price = tp_price
                profit = (exit_price - trade_entry_price) * contract_multiplier
                trade_exit_time = current_time
                account_balance += profit
                trade_results.append({
                    'entry_time': trade_entry_time,
                    'exit_time': trade_exit_time,
                    'direction': trade_direction,
                    'entry_price': trade_entry_price,
                    'exit_price': exit_price,
                    'profit': profit
                })
                (wins if profit > 0 else losses).append(profit)
                in_trade = False
        else:  # short trade
            sl_price = trade_entry_price + stop_loss_points
            tp_price = trade_entry_price - take_profit_points
            if current_bar['high'] >= sl_price:
                exit_price = sl_price
                profit = (trade_entry_price - exit_price) * contract_multiplier
                trade_exit_time = current_time
                account_balance += profit
                trade_results.append({
                    'entry_time': trade_entry_time,
                    'exit_time': trade_exit_time,
                    'direction': trade_direction,
                    'entry_price': trade_entry_price,
                    'exit_price': exit_price,
                    'profit': profit
                })
                (wins if profit > 0 else losses).append(profit)
                in_trade = False
            elif current_bar['low'] <= tp_price:
                exit_price = tp_price
                profit = (trade_entry_price - exit_price) * contract_multiplier
                trade_exit_time = current_time
                account_balance += profit
                trade_results.append({
                    'entry_time': trade_entry_time,
                    'exit_time': trade_exit_time,
                    'direction': trade_direction,
                    'entry_price': trade_entry_price,
                    'exit_price': exit_price,
                    'profit': profit
                })
                (wins if profit > 0 else losses).append(profit)
                in_trade = False

    if in_trade:
        bars_in_trade += 1  # count this bar as time in trade
        
    i += 1

# If a trade remains open at the end, close it at the final bar's close.
if in_trade:
    last_bar = df_resampled.iloc[-1]
    if trade_direction == 'long':
        profit = (last_bar['close'] - trade_entry_price) * contract_multiplier
    else:
        profit = (trade_entry_price - last_bar['close']) * contract_multiplier
    account_balance += profit
    trade_results.append({
        'entry_time': trade_entry_time,
        'exit_time': df_resampled.index[-1],
        'direction': trade_direction,
        'entry_price': trade_entry_price,
        'exit_price': last_bar['close'],
        'profit': profit
    })
    (wins if profit > 0 else losses).append(profit)
    in_trade = False
    # update last equity value:
    if equity_list:
        equity_list[-1] = account_balance

# Build an equity curve DataFrame
equity_df = pd.DataFrame({'Equity': equity_list}, index=equity_index)

# --- 6. Compute Performance Metrics ---
start_date = df_resampled.index[0]
end_date = df_resampled.index[-1]
total_bars = len(df_resampled)
exposure_percentage = (bars_in_trade / total_bars) * 100
final_balance = account_balance
total_return_pct = ((final_balance / initial_capital) - 1) * 100

# Annualized return calculation (using total days in the test period)
total_days = (end_date - start_date).days + 1
annualized_return = ((final_balance / initial_capital) ** (365 / total_days) - 1) * 100

# --- Benchmark: Buy & Hold on the resampled close ---
initial_close = df_resampled['close'].iloc[0]
benchmark_equity = (df_resampled['close'] / initial_close) * initial_capital
benchmark_return = ((benchmark_equity.iloc[-1] / initial_capital) - 1) * 100

# --- Risk Metrics ---
# Resample the equity curve to daily values.
daily_equity = equity_df['Equity'].resample('D').last()
daily_returns = daily_equity.pct_change().dropna()
volatility_annual = daily_returns.std() * np.sqrt(252)
sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else np.nan

# Sortino Ratio (using only negative daily returns)
downside_returns = daily_returns[daily_returns < 0]
downside_std = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else np.nan
sortino_ratio = (daily_returns.mean() * 252) / downside_std if downside_std not in [0, np.nan] else np.nan

# Drawdowns
running_max = equity_df['Equity'].cummax()
drawdown = (equity_df['Equity'] - running_max) / running_max
max_drawdown_pct = drawdown.min() * 100  # negative value
max_drawdown_dollar = (equity_df['Equity'] - running_max).min()

# Average drawdown (only when equity is below its running max)
if not drawdown[drawdown < 0].empty:
    avg_drawdown_pct = drawdown[drawdown < 0].mean() * 100
else:
    avg_drawdown_pct = 0
avg_drawdown_dollar = np.nan  # (calculation can be refined if desired)

# Calmar Ratio: annualized return divided by the absolute max drawdown (in %)
calmar_ratio = (annualized_return / abs(max_drawdown_pct)) if max_drawdown_pct != 0 else np.nan

# Drawdown durations (in days)
drawdown_durations = []
drawdown_start = None
peak_value = equity_df['Equity'].iloc[0]

for time, eq in equity_df['Equity'].items():
    if eq >= peak_value:
        # if a drawdown was active, record its duration
        if drawdown_start is not None:
            duration = (time - drawdown_start).days
            drawdown_durations.append(duration)
            drawdown_start = None
        peak_value = eq
    else:
        if drawdown_start is None:
            drawdown_start = time

max_drawdown_duration = max(drawdown_durations) if drawdown_durations else 0
avg_drawdown_duration = np.mean(drawdown_durations) if drawdown_durations else 0

win_rate = (len(wins) / len(trade_results)) * 100 if trade_results else 0
profit_factor = (sum(wins) / abs(sum(losses))) if sum(losses) != 0 else np.nan

results = {
    "Start Date": start_date,
    "End Date": end_date,
    "Exposure Time": f"{exposure_percentage:.2f}%",
    "Final Account Balance": f"${final_balance:,.2f}",
    "Total Return": f"{total_return_pct:.2f}%",
    "Annualized Return": f"{annualized_return:.2f}%",
    "Benchmark Return": f"{benchmark_return:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Total Trades": len(trade_results),
    "Winning Trades": len(wins),
    "Losing Trades": len(losses),
    "Win Rate": f"{win_rate:.2f}%",
    "Profit Factor": f"{profit_factor:.2f}",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}",
    "Sortino Ratio": f"{sortino_ratio:.2f}",
    "Calmar Ratio": f"{calmar_ratio:.2f}",
    "Max Drawdown (%)": f"{max_drawdown_pct:.2f}%",
    "Average Drawdown (%)": f"{avg_drawdown_pct:.2f}%",
    "Max Drawdown ($)": f"${max_drawdown_dollar:,.2f}",
    "Average Drawdown ($)": f"${avg_drawdown_dollar if not np.isnan(avg_drawdown_dollar) else 0:,.2f}",
    "Max Drawdown Duration": f"{max_drawdown_duration:.2f} days",
    "Average Drawdown Duration": f"{avg_drawdown_duration:.2f} days",
}

print("\nPerformance Summary:")
for key, value in results.items():
    print(f"{key:30}: {value:>15}")

# --- 7. Plot the Equity Curve vs. the Benchmark ---
plt.figure(figsize=(14, 7))
plt.plot(equity_df.index, equity_df['Equity'], label='Strategy Equity')
plt.plot(df_resampled.index, benchmark_equity, label='Benchmark Equity (Buy & Hold)', alpha=0.7)
plt.title('Equity Curve: Strategy vs Benchmark')
plt.xlabel('Time')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()