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

# File that contains symbols and multipliers
symbols_file = "Data/symbols.csv"

# Backtest parameters (applied to each symbol)
initial_capital = 10000.0         # starting account balance in dollars
commission_per_order = 1.24       # commission per order (per contract)
num_contracts = 1                 # number of contracts to trade

# Custom start and end date (format: 'YYYY-MM-DD')
start_date = '2000-01-01'
end_date   = '2022-03-12'

# IBS Strategy parameters
ibs_entry_threshold = 0.1  # enter when IBS < 0.1
ibs_exit_threshold  = 0.9  # exit when IBS > 0.9

# -------------------------------
# Helper Function: Run Backtest for a Symbol
# -------------------------------
def run_backtest(data_file, multiplier):
    """
    Runs the IBS strategy backtest on the provided data file using the given multiplier.
    Returns a dictionary with performance metrics including the Sharpe Ratio.
    """
    try:
        data = pd.read_csv(data_file, parse_dates=['Time'])
    except Exception as e:
        logger.error(f"Error reading {data_file}: {e}")
        return None

    data.sort_values('Time', inplace=True)
    # Filter data based on the custom date range
    data = data[(data['Time'] >= start_date) & (data['Time'] <= end_date)].reset_index(drop=True)
    if data.empty:
        logger.warning(f"No data for {data_file} in the given date range.")
        return None

    # Calculate Internal Bar Strength (IBS)
    # IBS = (Last - Low) / (High - Low)
    data['IBS'] = (data['Last'] - data['Low']) / (data['High'] - data['Low'])

    # -------------------------------
    # Backtest Simulation
    # -------------------------------
    capital = initial_capital  # realized account equity
    in_position = False        # flag if a trade is active
    position = None            # dictionary to hold trade details
    trade_results = []         # list to record completed trades
    equity_curve = []          # list of (Time, mark-to-market Equity)

    # For benchmark purposes we note the first close (not used in filtering)
    initial_close = data['Last'].iloc[0]

    for i, row in data.iterrows():
        current_time = row['Time']
        current_price = row['Last']

        if in_position:
            # Exit condition: IBS above exit threshold
            if row['IBS'] > ibs_exit_threshold:
                exit_price = current_price  # exit at the close price
                profit = (exit_price - position['entry_price']) * multiplier * num_contracts - commission_per_order * num_contracts
                trade_results.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': profit,
                    'contracts': num_contracts
                })
                logger.info(f"SELL signal for {data_file} at {current_time} | Exit Price: {exit_price:.2f} | Profit: {profit:.2f}")
                capital += profit
                in_position = False
                position = None
        else:
            # Entry condition: IBS below entry threshold
            if row['IBS'] < ibs_entry_threshold:
                entry_price = current_price
                in_position = True
                capital -= commission_per_order * num_contracts  # deduct entry commission
                position = {
                    'entry_price': entry_price,
                    'entry_time': current_time,
                    'contracts': num_contracts
                }
                logger.info(f"BUY signal for {data_file} at {current_time} | Entry Price: {entry_price:.2f}")

        # Mark-to-market equity calculation
        if in_position:
            unrealized = (current_price - position['entry_price']) * multiplier * num_contracts
            equity = capital + unrealized
        else:
            equity = capital
        equity_curve.append((current_time, equity))

    # Close any open position at the end
    if in_position:
        row = data.iloc[-1]
        current_time = row['Time']
        current_price = row['Last']
        exit_price = current_price
        profit = (exit_price - position['entry_price']) * multiplier * num_contracts - commission_per_order * num_contracts
        trade_results.append({
            'entry_time': position['entry_time'],
            'exit_time': current_time,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'profit': profit,
            'contracts': num_contracts
        })
        logger.info(f"Closing open position for {data_file} at end {current_time} | Exit Price: {exit_price:.2f} | Profit: {profit:.2f}")
        capital += profit
        equity = capital
        in_position = False
        position = None
        equity_curve[-1] = (current_time, equity)

    # Convert equity curve to DataFrame
    equity_df = pd.DataFrame(equity_curve, columns=['Time', 'Equity'])
    equity_df.set_index('Time', inplace=True)

    # -------------------------------
    # Performance Metrics Calculation
    # -------------------------------
    final_account_balance = capital
    total_return_percentage = ((final_account_balance / initial_capital) - 1) * 100

    # Annualized return calculation
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    years = (end_dt - start_dt).days / 365.25
    annualized_return_percentage = ((final_account_balance / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else np.nan

    equity_df['returns'] = equity_df['Equity'].pct_change()
    # Assume ~252 trading days per year
    volatility_annual = equity_df['returns'].std() * np.sqrt(252) * 100

    # Calculate Sharpe Ratio
    sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252)
                    if equity_df['returns'].std() != 0 else np.nan)

    performance = {
        'Data File': data_file,
        'Final Account Balance': final_account_balance,
        'Total Return (%)': total_return_percentage,
        'Annualized Return (%)': annualized_return_percentage,
        'Volatility (Annual %)': volatility_annual,
        'Sharpe Ratio': sharpe_ratio,
        'Total Trades': len(trade_results)
    }
    return performance

# -------------------------------
# Read symbols from the symbols file
# -------------------------------
symbols_list = []
try:
    with open(symbols_file, 'r') as f:
        lines = f.read().splitlines()
    # Skip header
    for line in lines[1:]:
        if line.strip():
            parts = line.split(',')
            symbol_path = parts[0].strip()
            # Extract multiplier (ignore inline comments)
            multiplier_str = parts[1].split()[0].strip()
            try:
                symbol_multiplier = float(multiplier_str)
            except Exception as e:
                logger.error(f"Error parsing multiplier for {symbol_path}: {e}")
                continue
            symbols_list.append((symbol_path, symbol_multiplier))
except Exception as e:
    logger.error(f"Error reading symbols file: {e}")

# -------------------------------
# Run backtest for each symbol and filter by Sharpe Ratio > 0.5
# -------------------------------
results_list = []
for symbol_path, sym_multiplier in symbols_list:
    logger.info(f"Running backtest for {symbol_path} with multiplier {sym_multiplier}")
    perf = run_backtest(symbol_path, sym_multiplier)
    if perf is not None and not np.isnan(perf['Sharpe Ratio']):
        if perf['Sharpe Ratio'] > 0.5:
            results_list.append(perf)

# -------------------------------
# Display the symbols with Sharpe Ratio > 0.5
# -------------------------------
if results_list:
    print("Symbols with Sharpe Ratio > 0.5:")
    df_results = pd.DataFrame(results_list)
    print(df_results[['Data File', 'Final Account Balance', 'Total Return (%)',
                      'Annualized Return (%)', 'Volatility (Annual %)', 'Sharpe Ratio', 'Total Trades']])
else:
    print("No symbols met the Sharpe Ratio > 0.5 criterion.")

# -------------------------------
# (Optional) Plot Equity Curve for one symbol if desired
# -------------------------------
# Uncomment the block below to plot the equity curve for the first symbol that meets the criterion
#
# if results_list:
#     sample_file = results_list[0]['Data File']
#     sample_multiplier = [m for s, m in symbols_list if s == sample_file][0]
#     performance = run_backtest(sample_file, sample_multiplier)
#     # For a full equity curve plot, you could modify run_backtest to return the equity_df as well.