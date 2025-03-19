# es, gc, ym, sq
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
end_date   = '2022-01-01'

# Williams %R Strategy parameters
williams_period = 2   # 2-day lookback period
buy_threshold = -90   # Buy when Williams %R is below -90
sell_threshold = -30  # Sell when Williams %R is above -30

# -------------------------------
# Helper Function: Run Williams %R Backtest
# -------------------------------
def run_williams_backtest(data_file, multiplier):
    """
    Runs the Williams %R strategy backtest on the provided data file using the given multiplier.
    Returns a dictionary with performance metrics including the Sharpe Ratio.
    
    The CSV file must contain at least the columns: Time, High, Low, Last.
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
        logger.warning(f"No data in {data_file} for the given date range.")
        return None

    # -------------------------------
    # Williams %R Calculation
    # -------------------------------
    # Calculate rolling highest high and lowest low
    data['HighestHigh'] = data['High'].rolling(window=williams_period).max()
    data['LowestLow'] = data['Low'].rolling(window=williams_period).min()
    data.dropna(subset=['HighestHigh', 'LowestLow'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Williams %R formula: -100 * (HighestHigh - Last) / (HighestHigh - LowestLow)
    data['WilliamsR'] = -100 * (data['HighestHigh'] - data['Last']) / (data['HighestHigh'] - data['LowestLow'])

    # -------------------------------
    # Backtest Simulation
    # -------------------------------
    capital = initial_capital  # realized account equity
    in_position = False        # flag if a trade is active
    position = None            # dictionary to hold trade details
    trade_results = []         # list to record completed trades
    equity_curve = []          # list of tuples (Time, mark-to-market Equity)

    # For benchmark: Buy and Hold (enter at first available close)
    initial_close = data['Last'].iloc[0]

    for i, row in data.iterrows():
        current_time = row['Time']
        current_price = row['Last']
        current_wr = row['WilliamsR']

        if in_position:
            # Exit condition: today's price > yesterday's high OR Williams %R > sell_threshold
            if i > 0:
                yesterdays_high = data['High'].iloc[i-1]
                if (current_price > yesterdays_high) or (current_wr > sell_threshold):
                    exit_price = current_price
                    profit = (exit_price - position['entry_price']) * multiplier * num_contracts
                    # Subtract exit commission
                    profit -= commission_per_order * num_contracts
                    trade_results.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'profit': profit,
                        'contracts': num_contracts
                    })
                    logger.info(f"SELL signal in {data_file} at {current_time} | Exit Price: {exit_price:.2f} | Profit: {profit:.2f}")
                    capital += profit
                    in_position = False
                    position = None
        else:
            # Buy condition: Williams %R below buy_threshold
            if current_wr < buy_threshold:
                entry_price = current_price
                in_position = True
                # Deduct commission on entry
                capital -= commission_per_order * num_contracts
                position = {
                    'entry_price': entry_price,
                    'entry_time': current_time,
                    'contracts': num_contracts
                }
                logger.info(f"BUY signal in {data_file} at {current_time} | Entry Price: {entry_price:.2f}")

        # Mark-to-market equity calculation
        if in_position:
            unrealized = (current_price - position['entry_price']) * multiplier * num_contracts
            equity = capital + unrealized
        else:
            equity = capital
        equity_curve.append((current_time, equity))

    # Close any open position at the end of the period
    if in_position:
        row = data.iloc[-1]
        current_time = row['Time']
        current_price = row['Last']
        exit_price = current_price
        profit = (exit_price - position['entry_price']) * multiplier * num_contracts
        profit -= commission_per_order * num_contracts
        trade_results.append({
            'entry_time': position['entry_time'],
            'exit_time': current_time,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'profit': profit,
            'contracts': num_contracts
        })
        logger.info(f"Closing open position in {data_file} at end {current_time} | Exit Price: {exit_price:.2f} | Profit: {profit:.2f}")
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

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    years = (end_dt - start_dt).days / 365.25
    annualized_return_percentage = ((final_account_balance / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else np.nan

    equity_df['returns'] = equity_df['Equity'].pct_change()
    # Using ~252 trading days per year
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
            # Extract multiplier (ignoring any inline comments)
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
    logger.info(f"Running Williams %R backtest for {symbol_path} with multiplier {sym_multiplier}")
    perf = run_williams_backtest(symbol_path, sym_multiplier)
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
#     # Optionally modify run_williams_backtest to also return equity_df for plotting
#     perf = run_williams_backtest(sample_file, sample_multiplier)
#     # For demonstration purposes, you would need to integrate the equity_df return in run_williams_backtest
#     # Then, plot the equity curve as shown below:
#     # plt.figure(figsize=(14, 7))
#     # plt.plot(equity_df.index, equity_df['Equity'], label=f'{sample_file} Equity Curve')
#     # plt.title(f'Equity Curve for {sample_file}')
#     # plt.xlabel('Time')
#     # plt.ylabel('Account Balance ($)')
#     # plt.legend()
#     # plt.grid(True)
#     # plt.tight_layout()
#     # plt.show()