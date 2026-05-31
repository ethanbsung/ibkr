import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import timedelta, time

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)  # Set to DEBUG for detailed logs
logger = logging.getLogger(__name__)

# --- Function to Load and Prepare Data ---
def load_data(csv_file):
    try:
        df = pd.read_csv(
            csv_file,
            parse_dates=['Time'],
            dayfirst=False,
            na_values=['', 'NA', 'NaN']
        ).set_index('Time').sort_index()
        
        df.rename(columns={'Last': 'Close'}, inplace=True)
        df.drop(columns=[col for col in ['Symbol', 'Change', '%Chg', 'Open Int'] if col in df.columns], inplace=True)
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {csv_file}")
        exit(1)
    except pd.errors.EmptyDataError:
        logger.error("The CSV file is empty.")
        exit(1)
    except pd.errors.ParserError as e:
        logger.error(f"Parser error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)

# --- Configuration Parameters ---
INTRADAY_DATA_FILE = 'es_1m_data.csv'  # Path to 1-minute CSV

# Backtesting Parameters
INITIAL_CASH = 5000
ES_MULTIPLIER = 5  # 1 ES point = $5 P/L per contract
STOP_LOSS_POINTS = 3
TAKE_PROFIT_POINTS = 17
POSITION_SIZE = 1  # Number of contracts
COMMISSION = 1.24  # Per trade

# --- Load Intraday Dataset ---
df_intraday = load_data(INTRADAY_DATA_FILE)

# --- Define Backtest Period ---
start_date, end_date = "2016-01-01", "2024-12-23"
df = df_intraday.loc[start_date:end_date].copy()
print("\nFiltered Data Range:", df.index.min(), "to", df.index.max())
print("Number of Rows After Filtering:", len(df))

# --- Resample to 30-Minute Bars ---
df_30m = df.resample('30min').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()
print("\nResampled 30-Minute Data Range:", df_30m.index.min(), "to", df_30m.index.max())
print("Number of Rows After Resampling:", len(df_30m))

# --- Calculate Rolling High ---
rolling_window = 15
df_30m['Rolling_High'] = df_30m['High'].shift(1).rolling(window=rolling_window).max()
df_30m.dropna(subset=['Rolling_High'], inplace=True)
print("\nData Range After Rolling Calculations:", df_30m.index.min(), "to", df_30m.index.max())
print("Number of Rows After Rolling Calculations:", len(df_30m))

# --- Check for Missing Data ---
expected_freq = '30min'
full_index = pd.date_range(start=df_30m.index.min(), end=df_30m.index.max(), freq=expected_freq)
missing_dates = full_index.difference(df_30m.index)
print(f"\nMissing 30-Minute Bars: {len(missing_dates)}")
if not missing_dates.empty:
    print("Sample Missing Dates:", missing_dates[:5])

# --- Initialize Backtest Variables ---
cash = INITIAL_CASH
trade_results = []
balance_series = [INITIAL_CASH]
balance_dates = [df_30m.index.min()]
position = None
active_bars = 0
total_bars = len(df_30m)

# --- Backtesting Loop ---
for current_time, row in df_30m.iterrows():
    if position is None:
        # Entry Conditions
        if time(9, 30) <= current_time.time() < time(16, 0):
            if row['High'] > row['Rolling_High']:
                entry_price = row['Rolling_High']
                position = {
                    'entry_time': current_time,
                    'entry_price': entry_price,
                    'stop_loss': entry_price - STOP_LOSS_POINTS,
                    'take_profit': entry_price + TAKE_PROFIT_POINTS
                }
                active_bars += 1
                logger.info(f"[ENTRY] Long entered at {entry_price} on {current_time}")
    else:
        # Exit Conditions
        exit_triggered = False
        if row['Low'] <= position['stop_loss']:
            exit_price = position['stop_loss']
            exit_reason = "STOP LOSS"
            exit_triggered = True
        elif row['High'] >= position['take_profit']:
            exit_price = position['take_profit']
            exit_reason = "TAKE PROFIT"
            exit_triggered = True
        
        if exit_triggered:
            pnl = (exit_price - position['entry_price']) * POSITION_SIZE * ES_MULTIPLIER - COMMISSION
            cash += pnl
            trade_results.append(pnl)
            balance_series.append(cash)
            balance_dates.append(current_time)
            logger.info(f"[{exit_reason}] Exit at {exit_price} on {current_time}, PnL: ${pnl:.2f}")
            position = None
        else:
            # Update balance without changes
            balance_series.append(cash)
            balance_dates.append(current_time)

# --- Calculate Exposure Time ---
exposure_time_percentage = (active_bars / total_bars) * 100

# --- Create Balance DataFrame ---
balance_df = pd.DataFrame({'Equity': balance_series}, index=balance_dates)
print("\nBalance DataFrame Range:", balance_df.index.min(), "to", balance_df.index.max())

# --- Calculate Performance Metrics ---
rolling_max = balance_df['Equity'].cummax()
drawdown = (balance_df['Equity'] - rolling_max) / rolling_max
max_drawdown = drawdown.min() * 100

# Drawdown Duration
drawdown_periods = drawdown[drawdown < 0]
if not drawdown_periods.empty:
    drawdown_diff = drawdown_periods.index.to_series().diff() != timedelta(minutes=30)
    drawdown_group = drawdown_diff.cumsum()
    drawdown_durations = drawdown_periods.groupby(drawdown_group).size()
    max_drawdown_duration_days = drawdown_durations.max() * 0.0208333  # 30 minutes = 0.0208333 days
    average_drawdown_duration_days = drawdown_durations.mean() * 0.0208333
else:
    max_drawdown_duration_days = average_drawdown_duration_days = 0

# Profit Factor
gross_profit = sum(p for p in trade_results if p > 0)
gross_loss = abs(sum(p for p in trade_results if p < 0))
profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan

# Winning and Losing Trades
winning_trades = [p for p in trade_results if p > 0]
losing_trades = [p for p in trade_results if p < 0]

# Sortino Ratio
mar = 0
strategy_returns = np.array(trade_results) / INITIAL_CASH
downside_returns = strategy_returns[strategy_returns < mar]
sortino_ratio = (strategy_returns.mean() - mar) / downside_returns.std() * np.sqrt(252) if downside_returns.size > 0 else np.nan

# Calmar Ratio
days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
annualized_return = ((cash / INITIAL_CASH) ** (365.0 / days) - 1) * 100 if days > 0 else 0.0
calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

# Benchmark Return (Buy and Hold)
initial_close = df_30m.iloc[0]['Close']
final_close = df_30m.iloc[-1]['Close']
benchmark_return = ((final_close - initial_close) / initial_close) * 100

# --- Compile Results ---
results = {
    "Start Date": df_30m.index.min().strftime("%Y-%m-%d"),
    "End Date": df_30m.index.max().strftime("%Y-%m-%d"),
    "Exposure Time": f"{exposure_time_percentage:.2f}%",
    "Final Account Balance": f"${cash:,.2f}",
    "Equity Peak": f"${balance_df['Equity'].max():,.2f}",
    "Total Return": f"{((cash - INITIAL_CASH) / INITIAL_CASH) * 100:.2f}%",
    "Annualized Return": f"{annualized_return:.2f}%",
    "Benchmark Return": f"{benchmark_return:.2f}%",
    "Volatility (Annual)": f"{balance_df['Equity'].pct_change().std() * np.sqrt(252) * 100:.2f}%",
    "Total Trades": len(trade_results),
    "Winning Trades": len(winning_trades),
    "Losing Trades": len(losing_trades),
    "Win Rate": f"{(len(winning_trades)/len(trade_results)*100) if trade_results else 0:.2f}%",
    "Profit Factor": f"{profit_factor:.2f}",
    "Sortino Ratio": f"{sortino_ratio:.2f}",
    "Calmar Ratio": f"{calmar_ratio:.2f}",
    "Max Drawdown": f"{max_drawdown:.2f}%",
    "Max Drawdown Duration": f"{max_drawdown_duration_days:.2f} days",
    "Average Drawdown Duration": f"{average_drawdown_duration_days:.2f} days",
}

# --- Display Results ---
print("\n=== Performance Summary ===")
for key, value in results.items():
    print(f"{key:25}: {value}")

# --- Plot Equity Curves ---
benchmark_equity = (df_30m['Close'] / initial_close) * INITIAL_CASH
benchmark_equity = benchmark_equity.reindex(balance_df.index, method='ffill').fillna(method='ffill')

plt.figure(figsize=(14, 7))
plt.plot(balance_df.index, balance_df['Equity'], label='Strategy Equity')
plt.plot(balance_df.index, benchmark_equity, label='Benchmark Equity')
plt.title('Equity Curve: Strategy vs Benchmark')
plt.xlabel('Time')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()