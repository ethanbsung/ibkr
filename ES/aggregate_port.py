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
initial_capital = 30000.0         # total capital ($30,000)
commission_per_order = 1.24       # commission per order (per contract)

# Date range for all strategies
start_date = '2020-01-01'
end_date   = '2025-03-12'

# -------------------------------
# Instrument & Strategy Settings
# -------------------------------
# IBS Strategy: half of total capital is allocated to IBS strategies,
# which now include 5 instruments. Each instrument gets an equal share.
capital_IBS_total = initial_capital / 2
ibs_instruments = 5
capital_IBS_each = capital_IBS_total / ibs_instruments

# Williams Strategy gets the other half of capital.
capital_Williams = initial_capital / 2

# IBS settings:
# ES: 1 contract, multiplier 5, file:
ibs_es_file = "Data/es_daily_data.csv"
ibs_es_contracts = 1
multiplier_es = 5

# YM: 2 contracts, multiplier 0.50, file:
ibs_ym_file = "Data/ym_daily_data.csv"
ibs_ym_contracts = 2
multiplier_ym = 0.50

# GC: 1 contract, multiplier 10, file:
ibs_gc_file = "Data/gc_daily_data.csv"
ibs_gc_contracts = 1
multiplier_gc = 10

# NQ: 1 contract, multiplier 2, file:
ibs_nq_file = "Data/nq_daily_data.csv"
ibs_nq_contracts = 1
multiplier_nq = 2

# ZQ: 2 contracts, multiplier 4167, file:
ibs_zq_file = "Data/zq_daily_data.csv"
ibs_zq_contracts = 0
multiplier_zq = 4167

# IBS entry/exit thresholds (common for all IBS instruments)
ibs_entry_threshold = 0.1       # Enter when IBS < 0.1
ibs_exit_threshold  = 0.9       # Exit when IBS > 0.9

# Williams %R strategy parameters (applied to ES only)
williams_period = 2             # 2-day lookback
wr_buy_threshold  = -90
wr_sell_threshold = -30
williams_contracts = 1          # Williams trades ES with 1 contract (multiplier_es)

# -------------------------------
# Data Preparation & Benchmark (ES data for benchmark and Williams)
# -------------------------------
# Load ES data (used for benchmark, IBS ES, and Williams strategies)
data_es = pd.read_csv(ibs_es_file, parse_dates=['Time'])
data_es.sort_values('Time', inplace=True)
data_es = data_es[(data_es['Time'] >= start_date) & (data_es['Time'] <= end_date)].reset_index(drop=True)

benchmark_initial_close = data_es['Last'].iloc[0]
benchmark_final_close   = data_es['Last'].iloc[-1]
benchmark_return = ((benchmark_final_close / benchmark_initial_close) - 1) * 100

benchmark_years = (data_es['Time'].iloc[-1] - data_es['Time'].iloc[0]).days / 365.25
benchmark_annualized_return = ((benchmark_final_close / benchmark_initial_close) ** (1 / benchmark_years) - 1) * 100

# -------------------------------
# Prepare IBS Data for Each Instrument
# -------------------------------

# IBS for ES
data_ibs_es = data_es.copy()
data_ibs_es['IBS'] = (data_ibs_es['Last'] - data_ibs_es['Low']) / (data_ibs_es['High'] - data_ibs_es['Low'])

# IBS for YM
data_ibs_ym = pd.read_csv(ibs_ym_file, parse_dates=['Time'])
data_ibs_ym.sort_values('Time', inplace=True)
data_ibs_ym = data_ibs_ym[(data_ibs_ym['Time'] >= start_date) & (data_ibs_ym['Time'] <= end_date)].reset_index(drop=True)
data_ibs_ym['IBS'] = (data_ibs_ym['Last'] - data_ibs_ym['Low']) / (data_ibs_ym['High'] - data_ibs_ym['Low'])

# IBS for GC
data_ibs_gc = pd.read_csv(ibs_gc_file, parse_dates=['Time'])
data_ibs_gc.sort_values('Time', inplace=True)
data_ibs_gc = data_ibs_gc[(data_ibs_gc['Time'] >= start_date) & (data_ibs_gc['Time'] <= end_date)].reset_index(drop=True)
data_ibs_gc['IBS'] = (data_ibs_gc['Last'] - data_ibs_gc['Low']) / (data_ibs_gc['High'] - data_ibs_gc['Low'])

# IBS for NQ
data_ibs_nq = pd.read_csv(ibs_nq_file, parse_dates=['Time'])
data_ibs_nq.sort_values('Time', inplace=True)
data_ibs_nq = data_ibs_nq[(data_ibs_nq['Time'] >= start_date) & (data_ibs_nq['Time'] <= end_date)].reset_index(drop=True)
data_ibs_nq['IBS'] = (data_ibs_nq['Last'] - data_ibs_nq['Low']) / (data_ibs_nq['High'] - data_ibs_nq['Low'])

# IBS for ZQ
data_ibs_zq = pd.read_csv(ibs_zq_file, parse_dates=['Time'])
data_ibs_zq.sort_values('Time', inplace=True)
data_ibs_zq = data_ibs_zq[(data_ibs_zq['Time'] >= start_date) & (data_ibs_zq['Time'] <= end_date)].reset_index(drop=True)
data_ibs_zq['IBS'] = (data_ibs_zq['Last'] - data_ibs_zq['Low']) / (data_ibs_zq['High'] - data_ibs_zq['Low'])

# -------------------------------
# Prepare Williams Data (ES only)
# -------------------------------
data_williams = data_es.copy()
data_williams['HighestHigh'] = data_williams['High'].rolling(window=williams_period).max()
data_williams['LowestLow'] = data_williams['Low'].rolling(window=williams_period).min()
data_williams.dropna(subset=['HighestHigh', 'LowestLow'], inplace=True)
data_williams.reset_index(drop=True, inplace=True)
data_williams['WilliamsR'] = -100 * (data_williams['HighestHigh'] - data_williams['Last']) / (data_williams['HighestHigh'] - data_williams['LowestLow'])

# -------------------------------
# Backtest Simulation for IBS (ES)
# -------------------------------
capital_es = capital_IBS_each
in_position_es = False
position_es = None
equity_curve_es = []

for i, row in data_ibs_es.iterrows():
    current_time = row['Time']
    current_price = row['Last']
    
    if in_position_es:
        if row['IBS'] > ibs_exit_threshold:
            exit_price = current_price
            profit = (exit_price - position_es['entry_price']) * multiplier_es * ibs_es_contracts - commission_per_order * ibs_es_contracts
            capital_es += profit
            in_position_es = False
            position_es = None
    else:
        if row['IBS'] < ibs_entry_threshold:
            entry_price = current_price
            in_position_es = True
            capital_es -= commission_per_order * ibs_es_contracts
            position_es = {'entry_price': entry_price, 'entry_time': current_time}
    
    if in_position_es:
        unrealized = (current_price - position_es['entry_price']) * multiplier_es * ibs_es_contracts
        equity = capital_es + unrealized
    else:
        equity = capital_es
    equity_curve_es.append((current_time, equity))

if in_position_es:
    row = data_ibs_es.iloc[-1]
    current_time = row['Time']
    current_price = row['Last']
    exit_price = current_price
    profit = (exit_price - position_es['entry_price']) * multiplier_es * ibs_es_contracts - commission_per_order * ibs_es_contracts
    capital_es += profit
    equity_curve_es[-1] = (current_time, capital_es)
    in_position_es = False
    position_es = None

equity_df_es = pd.DataFrame(equity_curve_es, columns=['Time', 'Equity'])
equity_df_es.set_index('Time', inplace=True)

# -------------------------------
# Backtest Simulation for IBS (YM)
# -------------------------------
capital_ym = capital_IBS_each
in_position_ym = False
position_ym = None
equity_curve_ym = []

for i, row in data_ibs_ym.iterrows():
    current_time = row['Time']
    current_price = row['Last']
    
    if in_position_ym:
        if row['IBS'] > ibs_exit_threshold:
            exit_price = current_price
            profit = (exit_price - position_ym['entry_price']) * multiplier_ym * ibs_ym_contracts - commission_per_order * ibs_ym_contracts
            capital_ym += profit
            in_position_ym = False
            position_ym = None
    else:
        if row['IBS'] < ibs_entry_threshold:
            entry_price = current_price
            in_position_ym = True
            capital_ym -= commission_per_order * ibs_ym_contracts
            position_ym = {'entry_price': entry_price, 'entry_time': current_time}
    
    if in_position_ym:
        unrealized = (current_price - position_ym['entry_price']) * multiplier_ym * ibs_ym_contracts
        equity = capital_ym + unrealized
    else:
        equity = capital_ym
    equity_curve_ym.append((current_time, equity))

if in_position_ym:
    row = data_ibs_ym.iloc[-1]
    current_time = row['Time']
    current_price = row['Last']
    exit_price = current_price
    profit = (exit_price - position_ym['entry_price']) * multiplier_ym * ibs_ym_contracts - commission_per_order * ibs_ym_contracts
    capital_ym += profit
    equity_curve_ym[-1] = (current_time, capital_ym)
    in_position_ym = False
    position_ym = None

equity_df_ym = pd.DataFrame(equity_curve_ym, columns=['Time', 'Equity'])
equity_df_ym.set_index('Time', inplace=True)

# -------------------------------
# Backtest Simulation for IBS (GC)
# -------------------------------
capital_gc = capital_IBS_each
in_position_gc = False
position_gc = None
equity_curve_gc = []

for i, row in data_ibs_gc.iterrows():
    current_time = row['Time']
    current_price = row['Last']
    
    if in_position_gc:
        if row['IBS'] > ibs_exit_threshold:
            exit_price = current_price
            profit = (exit_price - position_gc['entry_price']) * multiplier_gc * ibs_gc_contracts - commission_per_order * ibs_gc_contracts
            capital_gc += profit
            in_position_gc = False
            position_gc = None
    else:
        if row['IBS'] < ibs_entry_threshold:
            entry_price = current_price
            in_position_gc = True
            capital_gc -= commission_per_order * ibs_gc_contracts
            position_gc = {'entry_price': entry_price, 'entry_time': current_time}
    
    if in_position_gc:
        unrealized = (current_price - position_gc['entry_price']) * multiplier_gc * ibs_gc_contracts
        equity = capital_gc + unrealized
    else:
        equity = capital_gc
    equity_curve_gc.append((current_time, equity))

if in_position_gc:
    row = data_ibs_gc.iloc[-1]
    current_time = row['Time']
    current_price = row['Last']
    exit_price = current_price
    profit = (exit_price - position_gc['entry_price']) * multiplier_gc * ibs_gc_contracts - commission_per_order * ibs_gc_contracts
    capital_gc += profit
    equity_curve_gc[-1] = (current_time, capital_gc)
    in_position_gc = False
    position_gc = None

equity_df_gc = pd.DataFrame(equity_curve_gc, columns=['Time', 'Equity'])
equity_df_gc.set_index('Time', inplace=True)

# -------------------------------
# Backtest Simulation for IBS (NQ)
# -------------------------------
capital_nq = capital_IBS_each
in_position_nq = False
position_nq = None
equity_curve_nq = []

for i, row in data_ibs_nq.iterrows():
    current_time = row['Time']
    current_price = row['Last']
    
    if in_position_nq:
        if row['IBS'] > ibs_exit_threshold:
            exit_price = current_price
            profit = (exit_price - position_nq['entry_price']) * multiplier_nq * ibs_nq_contracts - commission_per_order * ibs_nq_contracts
            capital_nq += profit
            in_position_nq = False
            position_nq = None
    else:
        if row['IBS'] < ibs_entry_threshold:
            entry_price = current_price
            in_position_nq = True
            capital_nq -= commission_per_order * ibs_nq_contracts
            position_nq = {'entry_price': entry_price, 'entry_time': current_time}
    
    if in_position_nq:
        unrealized = (current_price - position_nq['entry_price']) * multiplier_nq * ibs_nq_contracts
        equity = capital_nq + unrealized
    else:
        equity = capital_nq
    equity_curve_nq.append((current_time, equity))

if in_position_nq:
    row = data_ibs_nq.iloc[-1]
    current_time = row['Time']
    current_price = row['Last']
    exit_price = current_price
    profit = (exit_price - position_nq['entry_price']) * multiplier_nq * ibs_nq_contracts - commission_per_order * ibs_nq_contracts
    capital_nq += profit
    equity_curve_nq[-1] = (current_time, capital_nq)
    in_position_nq = False
    position_nq = None

equity_df_nq = pd.DataFrame(equity_curve_nq, columns=['Time', 'Equity'])
equity_df_nq.set_index('Time', inplace=True)

# -------------------------------
# Backtest Simulation for IBS (ZQ)
# -------------------------------
capital_zq = capital_IBS_each
in_position_zq = False
position_zq = None
equity_curve_zq = []

for i, row in data_ibs_zq.iterrows():
    current_time = row['Time']
    current_price = row['Last']
    
    if in_position_zq:
        if row['IBS'] > ibs_exit_threshold:
            exit_price = current_price
            profit = (exit_price - position_zq['entry_price']) * multiplier_zq * ibs_zq_contracts - commission_per_order * ibs_zq_contracts
            capital_zq += profit
            in_position_zq = False
            position_zq = None
    else:
        if row['IBS'] < ibs_entry_threshold:
            entry_price = current_price
            in_position_zq = True
            capital_zq -= commission_per_order * ibs_zq_contracts
            position_zq = {'entry_price': entry_price, 'entry_time': current_time}
    
    if in_position_zq:
        unrealized = (current_price - position_zq['entry_price']) * multiplier_zq * ibs_zq_contracts
        equity = capital_zq + unrealized
    else:
        equity = capital_zq
    equity_curve_zq.append((current_time, equity))

if in_position_zq:
    row = data_ibs_zq.iloc[-1]
    current_time = row['Time']
    current_price = row['Last']
    exit_price = current_price
    profit = (exit_price - position_zq['entry_price']) * multiplier_zq * ibs_zq_contracts - commission_per_order * ibs_zq_contracts
    capital_zq += profit
    equity_curve_zq[-1] = (current_time, capital_zq)
    in_position_zq = False
    position_zq = None

equity_df_zq = pd.DataFrame(equity_curve_zq, columns=['Time', 'Equity'])
equity_df_zq.set_index('Time', inplace=True)

# -------------------------------
# Backtest Simulation for Williams Strategy (ES only)
# -------------------------------
capital_williams = capital_Williams
in_position_w = False
position_w = None
equity_curve_w = []

for i in range(len(data_williams)):
    row = data_williams.iloc[i]
    current_time = row['Time']
    current_price = row['Last']
    current_wr = row['WilliamsR']
    
    if in_position_w:
        if i > 0:
            yesterdays_high = data_williams['High'].iloc[i-1]
            if (current_price > yesterdays_high) or (current_wr > wr_sell_threshold):
                exit_price = current_price
                profit = (exit_price - position_w['entry_price']) * multiplier_es * williams_contracts - commission_per_order * williams_contracts
                capital_williams += profit
                in_position_w = False
                position_w = None
    else:
        if current_wr < wr_buy_threshold:
            entry_price = current_price
            in_position_w = True
            capital_williams -= commission_per_order * williams_contracts
            position_w = {'entry_price': entry_price, 'entry_time': current_time}
    
    if in_position_w:
        unrealized = (current_price - position_w['entry_price']) * multiplier_es * williams_contracts
        equity = capital_williams + unrealized
    else:
        equity = capital_williams
    equity_curve_w.append((current_time, equity))

if in_position_w:
    row = data_williams.iloc[-1]
    current_time = row['Time']
    current_price = row['Last']
    exit_price = current_price
    profit = (exit_price - position_w['entry_price']) * multiplier_es * williams_contracts - commission_per_order * williams_contracts
    capital_williams += profit
    equity_curve_w[-1] = (current_time, capital_williams)
    in_position_w = False
    position_w = None

equity_df_w = pd.DataFrame(equity_curve_w, columns=['Time', 'Equity'])
equity_df_w.set_index('Time', inplace=True)

# -------------------------------
# Aggregate Performance: Combine Equity Curves
# -------------------------------
# Reindex all equity DataFrames to a common daily date range.
common_dates = pd.date_range(start=start_date, end=end_date, freq='D')
equity_df_es = equity_df_es.reindex(common_dates, method='ffill')
equity_df_ym = equity_df_ym.reindex(common_dates, method='ffill')
equity_df_gc = equity_df_gc.reindex(common_dates, method='ffill')
equity_df_nq = equity_df_nq.reindex(common_dates, method='ffill')
equity_df_zq = equity_df_zq.reindex(common_dates, method='ffill')
equity_df_w = equity_df_w.reindex(common_dates, method='ffill')

# Combined IBS equity is the sum of ES, YM, GC, NQ, and ZQ IBS strategies.
combined_IBS = (equity_df_es['Equity'] + equity_df_ym['Equity'] +
                equity_df_gc['Equity'] + equity_df_nq['Equity'] +
                equity_df_zq['Equity'])
# Overall combined equity is IBS + Williams
combined_equity = combined_IBS + equity_df_w['Equity']
combined_equity_df = pd.DataFrame({'Equity': combined_equity}, index=common_dates)

# -------------------------------
# Calculate Aggregate Performance Metrics
# -------------------------------
initial_capital_combined = initial_capital
final_account_balance = combined_equity_df['Equity'].iloc[-1]
total_return_percentage = ((final_account_balance / initial_capital_combined) - 1) * 100

years = (combined_equity_df.index[-1] - combined_equity_df.index[0]).days / 365.25
annualized_return_percentage = ((final_account_balance / initial_capital_combined) ** (1 / years) - 1) * 100 if years > 0 else np.nan

combined_equity_df['returns'] = combined_equity_df['Equity'].pct_change()
volatility_annual = combined_equity_df['returns'].std() * np.sqrt(252) * 100

combined_equity_df['EquityPeak'] = combined_equity_df['Equity'].cummax()
combined_equity_df['Drawdown'] = (combined_equity_df['Equity'] - combined_equity_df['EquityPeak']) / combined_equity_df['EquityPeak']
max_drawdown_percentage = combined_equity_df['Drawdown'].min() * 100

combined_equity_df['DrawdownAmount'] = combined_equity_df['EquityPeak'] - combined_equity_df['Equity']
max_drawdown_dollar = combined_equity_df['DrawdownAmount'].max()

sharpe_ratio = (combined_equity_df['returns'].mean() / combined_equity_df['returns'].std() * np.sqrt(252)
                if combined_equity_df['returns'].std() != 0 else np.nan)
downside_std = combined_equity_df[combined_equity_df['returns'] < 0]['returns'].std()
sortino_ratio = (combined_equity_df['returns'].mean() / downside_std * np.sqrt(252)
                 if downside_std != 0 else np.nan)
calmar_ratio = (annualized_return_percentage / abs(max_drawdown_percentage)
                if max_drawdown_percentage != 0 else np.nan)

results = {
    "Start Date": start_date,
    "End Date": end_date,
    "Final Account Balance": f"${final_account_balance:,.2f}",
    "Total Return": f"{total_return_percentage:.2f}%",
    "Benchmark Total Return": f"{benchmark_return:.2f}%",
    "Benchmark Annualized Return": f"{benchmark_annualized_return:.2f}%",
    "Annualized Return": f"{annualized_return_percentage:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "NaN",
    "Sortino Ratio": f"{sortino_ratio:.2f}" if not np.isnan(sortino_ratio) else "NaN",
    "Calmar Ratio": f"{calmar_ratio:.2f}" if not np.isnan(calmar_ratio) else "NaN",
    "Max Drawdown (%)": f"{max_drawdown_percentage:.2f}%",
    "Max Drawdown ($)": f"${max_drawdown_dollar:.2f}"
}

print(f"Max Drawdown ($): ${max_drawdown_dollar:,.2f}\n")
print("Aggregate Performance Summary:")
for key, value in results.items():
    print(f"{key:30}: {value:>15}")

# -------------------------------
# Plot the Combined Equity Curve
# -------------------------------
plt.figure(figsize=(14, 7))
plt.plot(combined_equity_df.index, combined_equity_df['Equity'], label='Combined Strategy Equity')
plt.title('Aggregate Equity Curve of Combined Strategies (IBS: ES, YM, GC, NQ, ZQ + Williams ES)')
plt.xlabel('Time')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()