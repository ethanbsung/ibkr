import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# --- Function to Load Data ---
def load_data(csv_file):
    try:
        df = pd.read_csv(
            csv_file,
            dtype={
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': float
            },
            parse_dates=['date']
        )
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
        df.sort_values('date', inplace=True)
        df.set_index('date', inplace=True)
        return df
    except FileNotFoundError:
        print(f"File not found: {csv_file}")
        exit(1)
    except pd.errors.EmptyDataError:
        print("The CSV file is empty.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        exit(1)

# --- Configuration Parameters ---
DAILY_DATA_FILE = 'es_daily_data.csv'    # Path to daily CSV
INTRADAY_DATA_FILE = 'es_5m_data.csv'    # Path to 5-min CSV

# Backtesting Parameters
INITIAL_CASH = 5000
ES_MULTIPLIER = 5  # 1 ES point = $5 profit/loss per contract
STOP_LOSS_POINTS = 10
TAKE_PROFIT_POINTS = 30
POSITION_SIZE = 1

# --- Load Datasets ---
df_daily = load_data(DAILY_DATA_FILE)
df_intraday = load_data(INTRADAY_DATA_FILE)

# --- Define Backtest Period ---
custom_start_date = "2022-09-25"
custom_end_date = "2024-12-11"
start_time = pd.to_datetime(custom_start_date).tz_localize(None)
end_time = pd.to_datetime(custom_end_date).tz_localize(None)

df_daily = df_daily.loc[start_time:end_time].copy()
df_intraday = df_intraday.loc[start_time:end_time].copy()

# --- Identify NR7 Days ---
df_daily['Range'] = df_daily['high'] - df_daily['low']
df_daily['NR7'] = df_daily['Range'] == df_daily['Range'].rolling(window=7, min_periods=7).min()
nr7_dates = df_daily[df_daily['NR7']].index

# --- Initialize Backtest Variables ---
cash = INITIAL_CASH
trade_results = []
long_pnl = []
balance_series = [INITIAL_CASH]
balance_dates = [df_intraday.index.min()]

# --- Backtesting Loop ---
for nr7_date in nr7_dates:
    nr7_high = df_daily.loc[nr7_date, 'high']
    nr7_volume = df_daily.loc[nr7_date, 'volume']

    next_trading_day = nr7_date + pd.Timedelta(days=1)
    intraday_start = next_trading_day.replace(hour=0, minute=0, second=0, microsecond=0)
    intraday_end = intraday_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df_next_day = df_intraday.loc[intraday_start:intraday_end].copy()

    if df_next_day.empty:
        print(f"No intraday data for {next_trading_day.strftime('%Y-%m-%d')}")
        continue

    # Trading Logic
    for current_time, row in df_next_day.iterrows():
        current_price = row['close']
        current_volume = row['volume']

        # Entry Condition: Raw Volume Check
        if current_price > nr7_high and current_volume > nr7_volume:
            entry_price = current_price
            stop_loss_price = entry_price - STOP_LOSS_POINTS
            take_profit_price = entry_price + TAKE_PROFIT_POINTS

            cash -= entry_price * POSITION_SIZE * ES_MULTIPLIER
            print(f"[ENTRY] Long entered at {entry_price} on {current_time}")

            # Exit Logic
            for exit_time, exit_row in df_next_day.loc[current_time:].iterrows():
                current_exit_price = exit_row['close']

                if current_exit_price <= stop_loss_price:
                    pnl = (current_exit_price - entry_price) * POSITION_SIZE * ES_MULTIPLIER
                    cash += pnl
                    trade_results.append(pnl)
                    long_pnl.append(pnl)
                    balance_series.append(cash)
                    balance_dates.append(exit_time)
                    print(f"[STOP LOSS] Exit at {current_exit_price} on {exit_time}, PnL: ${pnl:,.2f}")
                    break

                elif current_exit_price >= take_profit_price:
                    pnl = (current_exit_price - entry_price) * POSITION_SIZE * ES_MULTIPLIER
                    cash += pnl
                    trade_results.append(pnl)
                    long_pnl.append(pnl)
                    balance_series.append(cash)
                    balance_dates.append(exit_time)
                    print(f"[TAKE PROFIT] Exit at {current_exit_price} on {exit_time}, PnL: ${pnl:,.2f}")
                    break

                if exit_time == df_next_day.index[-1]:
                    pnl = (current_exit_price - entry_price) * POSITION_SIZE * ES_MULTIPLIER
                    cash += pnl
                    trade_results.append(pnl)
                    long_pnl.append(pnl)
                    balance_series.append(cash)
                    balance_dates.append(exit_time)
                    print(f"[END OF DAY] Exit at {current_exit_price} on {exit_time}, PnL: ${pnl:,.2f}")
                    break
            break  # Process only one trade per NR7 day

# --- Create Balance DataFrame ---
balance_df = pd.DataFrame({
    'Datetime': balance_dates,
    'Equity': balance_series
}).set_index('Datetime')

# --- Performance Metrics ---
total_return_percentage = ((cash - INITIAL_CASH) / INITIAL_CASH) * 100
num_days = (df_daily.index.max() - df_daily.index.min()).days
annualized_return_percentage = ((cash / INITIAL_CASH) ** (365.0 / num_days) - 1) * 100 if num_days > 0 else 0.0
balance_df['Returns'] = balance_df['Equity'].pct_change().fillna(0)
volatility_annual = balance_df['Returns'].std() * np.sqrt(252) * 100
sharpe_ratio = (balance_df['Returns'].mean() / balance_df['Returns'].std()) * np.sqrt(252) if balance_df['Returns'].std() != 0 else 0

# Print Performance Summary
results = {
    "Total Return": f"{total_return_percentage:.2f}%",
    "Annualized Return": f"{annualized_return_percentage:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}",
    "Final Account Balance": f"${cash:,.2f}",
    "Total Trades": len(trade_results),
    "Winning Trades": len([pnl for pnl in trade_results if pnl > 0]),
    "Losing Trades": len([pnl for pnl in trade_results if pnl < 0]),
    "Long PnL": f"${sum(long_pnl):,.2f}",
}

# Print Results
print("\nPerformance Summary:")
for key, value in results.items():
    print(f"{key:25}: {value:>15}")

