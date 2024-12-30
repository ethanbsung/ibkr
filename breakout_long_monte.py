import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import timedelta, time
import sys

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)  # Change to DEBUG for more detailed logs
logger = logging.getLogger(__name__)

# =============================================================================
#                       FUNCTION: LOAD THE DATA
# =============================================================================
def load_data(csv_file):
    """
    Loads intraday data from CSV, parses the Time column as datetime, 
    sorts by time, and does some basic cleanup.
    """
    try:
        # Read the CSV
        df = pd.read_csv(
            csv_file,
            parse_dates=['Time'],
            dayfirst=False,  # Adjust if your data uses day-first format
            na_values=['', 'NA', 'NaN']  # Handle missing values
        )
        
        # Check if 'Time' column exists
        if 'Time' not in df.columns:
            logger.error("The CSV file does not contain a 'Time' column.")
            sys.exit(1)
        
        # Verify 'Time' column data type
        if not np.issubdtype(df['Time'].dtype, np.datetime64):
            logger.error("The 'Time' column was not parsed as datetime. Please check the date format.")
            sys.exit(1)
        
        # Handle timezone information if present
        if df['Time'].dt.tz is not None:
            df['Time'] = df['Time'].dt.tz_convert(None)
        
        # Sort by 'Time' and set it as the index
        df.sort_values('Time', inplace=True)
        df.set_index('Time', inplace=True)
        
        # Drop the 'Symbol' column as it's typically redundant for backtesting
        if 'Symbol' in df.columns:
            df.drop(columns=['Symbol'], inplace=True)
        
        # Rename 'Last' to 'Close' if needed
        if 'Last' in df.columns:
            df.rename(columns={'Last': 'Close'}, inplace=True)
        
        # Optional: Drop unnecessary columns
        unnecessary_cols = ['Change', '%Chg', 'Open Int']
        df.drop(columns=[col for col in unnecessary_cols if col in df.columns], inplace=True, errors='ignore')
        
        return df
    
    except FileNotFoundError:
        logger.error(f"File not found: {csv_file}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logger.error("The CSV file is empty.")
        sys.exit(1)
    except pd.errors.ParserError as e:
        logger.error(f"Parser error while reading the CSV: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading the CSV: {e}")
        sys.exit(1)

# =============================================================================
#                       BACKTEST CONFIGURATION
# =============================================================================
INTRADAY_DATA_FILE = 'es_1m_data.csv'  # <-- Update path to your 1-minute CSV file

# Backtesting Parameters
INITIAL_CASH = 5000
ES_MULTIPLIER = 5    # 1 ES point = $5 profit/loss per contract
STOP_LOSS_POINTS = 3
TAKE_PROFIT_POINTS = 17
POSITION_SIZE = 1    # Can be fractional if desired
COMMISSION = 1.24    # Commission per trade

# =============================================================================
#                       LOAD AND PREPARE DATA
# =============================================================================
df_intraday = load_data(INTRADAY_DATA_FILE)

# --- Verify Full Data Range ---
print("\nFull Data Range:")
print(df_intraday.index.min(), "to", df_intraday.index.max())

# --- Define Backtest Period ---
custom_start_date = "2016-01-01"
custom_end_date   = "2024-12-23"
start_time = pd.to_datetime(custom_start_date).tz_localize(None)
end_time   = pd.to_datetime(custom_end_date).tz_localize(None)

# --- Filter Data by Backtest Period ---
df_intraday_filtered = df_intraday.loc[start_time:end_time].copy()
print("\nFiltered Data Range:")
print(df_intraday_filtered.index.min(), "to", df_intraday_filtered.index.max())
print("Number of Rows After Filtering:", len(df_intraday_filtered))

# --- Resample to 30-minute bars ---
df_30m = df_intraday_filtered.resample('30min').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

print("\nResampled 30-Minute Data Range:")
print(df_30m.index.min(), "to", df_30m.index.max())
print("Number of Rows After Resampling:", len(df_30m))

# --- Compute Rolling High of Last 15 (30-min) bars (excluding current bar) ---
rolling_window = 15
df_30m['Rolling_High'] = (
    df_30m['High'].shift(1)
                  .rolling(window=rolling_window, min_periods=rolling_window)
                  .max()
)

# Remove rows where rolling high is NaN
df_30m.dropna(subset=['Rolling_High'], inplace=True)

print("\nData Range After Rolling Calculations:")
print(df_30m.index.min(), "to", df_30m.index.max())
print("Number of Rows After Rolling Calculations:", len(df_30m))

# --- Verify presence of data ---
if df_30m.empty:
    logger.error("No data available after applying rolling calculations. Check your data and rolling window.")
    sys.exit(1)

# --- Check for Missing Data ---
expected_freq = '30min'
full_index = pd.date_range(start=df_30m.index.min(), end=df_30m.index.max(), freq=expected_freq)
missing_dates = full_index.difference(df_30m.index)

print(f"\nNumber of Missing 30-Minute Bars: {len(missing_dates)}")
if not missing_dates.empty:
    print("Sample Missing Dates:")
    print(missing_dates[:5])

# --- Check Timezone Information ---
print("\nTimezone Information:")
print("Filtered Data Timezone:", df_intraday_filtered.index.tz)
print("Resampled Data Timezone:", df_30m.index.tz)

# =============================================================================
#                       BACKTEST INITIALIZATION
# =============================================================================
cash = INITIAL_CASH
trade_results = []          # List of PnLs for each trade
balance_series = [INITIAL_CASH]
balance_dates = [df_30m.index.min()]
position = None             # Track open position (if any)

# Variables for Exposure Time
total_bars = len(df_30m)
active_bars = 0

# =============================================================================
#                        BACKTEST LOOP
# =============================================================================
for idx, (current_time, row) in enumerate(df_30m.iterrows()):
    
    # If no open position, check for entry
    if position is None:
        # Strategy: Enter long if price breaks the 15-bar rolling high
        # Only check this during a specific session time (e.g. 09:30 to 16:00)
        if time(9, 30) <= current_time.time() < time(16, 0):
            breakout_level = row['Rolling_High']
            if row['High'] > breakout_level:
                entry_price = breakout_level
                stop_loss_price = entry_price - STOP_LOSS_POINTS
                take_profit_price = entry_price + TAKE_PROFIT_POINTS

                # Enter the trade
                position = {
                    'entry_time': current_time,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price
                }
                active_bars += 1
                logger.info(f"[ENTRY] Long entered at {entry_price} on {current_time}")
    
    # If a position is open, manage it
    else:
        current_high = row['High']
        current_low  = row['Low']
        exit_time    = current_time
        exit_price   = row['Close']

        # Check Stop Loss
        if current_low <= position['stop_loss']:
            exit_price = position['stop_loss']
            pnl = ((exit_price - position['entry_price']) 
                   * POSITION_SIZE * ES_MULTIPLIER) - COMMISSION
            cash += pnl
            trade_results.append(pnl)
            balance_series.append(cash)
            balance_dates.append(exit_time)
            logger.info(f"[STOP LOSS] Exit at {exit_price} on {exit_time}, PnL: ${pnl:,.2f}")
            position = None
        
        # Check Take Profit
        elif current_high >= position['take_profit']:
            exit_price = position['take_profit']
            pnl = ((exit_price - position['entry_price']) 
                   * POSITION_SIZE * ES_MULTIPLIER) - COMMISSION
            cash += pnl
            trade_results.append(pnl)
            balance_series.append(cash)
            balance_dates.append(exit_time)
            logger.info(f"[TAKE PROFIT] Exit at {exit_price} on {exit_time}, PnL: ${pnl:,.2f}")
            position = None
        
        # If neither SL nor TP is hit, position remains open (no EOD exit in this example)

    # If no trade action occurred, optionally record the same balance
    if position is None:
        # Ensure we don't duplicate the last index, which would cause plotting misalignment
        if len(balance_series) == len(balance_dates):
            balance_series.append(cash)
            balance_dates.append(current_time)

# =============================================================================
#                        POST-BACKTEST CALCULATIONS
# =============================================================================
exposure_time_percentage = (active_bars / total_bars) * 100

# --- Create Balance DataFrame ---
balance_df = pd.DataFrame({
    'Datetime': balance_dates,
    'Equity': balance_series
}).set_index('Datetime').sort_index()

# --- Calculate Equity Peak ---
equity_peak = balance_df['Equity'].max()

# --- Calculate Maximum Drawdown ---
rolling_max = balance_df['Equity'].cummax()
drawdown = (balance_df['Equity'] - rolling_max) / rolling_max
max_drawdown = drawdown.min() * 100  # as percentage

# --- Calculate Drawdown Duration (in days) ---
drawdown_periods = drawdown[drawdown < 0]
if not drawdown_periods.empty:
    end_dates = drawdown_periods.index.to_series().diff().ne(timedelta(minutes=30)).cumsum()
    drawdown_groups = drawdown_periods.groupby(end_dates)
    drawdown_durations = drawdown_groups.size()
    # 30 min = 0.5 hr, 1 day trading has 8–6.5 hours, 
    # but we'll just do approximate conversion for demonstration:
    max_drawdown_duration_days = drawdown_durations.max() * (30.0 / (60*24))
    average_drawdown_duration_days = drawdown_durations.mean() * (30.0 / (60*24))
else:
    max_drawdown_duration_days = 0
    average_drawdown_duration_days = 0

average_drawdown = drawdown.min() * 100  # simplified approach

# --- Calculate Profit Factor ---
gross_profit = sum(p for p in trade_results if p > 0)
gross_loss   = abs(sum(p for p in trade_results if p < 0))
profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan

# --- Winning / Losing Trades ---
winning_trades = [pnl for pnl in trade_results if pnl > 0]
losing_trades  = [pnl for pnl in trade_results if pnl < 0]
win_rate = (len(winning_trades) / len(trade_results) * 100) if trade_results else 0

# --- Sharpe Ratio (simple approximation) ---
# We'll treat each 30-min bar's equity change as a 'return'.
returns = balance_df['Equity'].pct_change().dropna()
if returns.std() != 0:
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252*13)  # 13 approx 30-min bars in a day
else:
    sharpe_ratio = 0

# --- Sortino Ratio ---
mar = 0
strategy_returns = np.array(trade_results) / INITIAL_CASH
downside_returns = np.where(strategy_returns < mar, strategy_returns - mar, 0)
expected_return  = np.mean(strategy_returns) if len(strategy_returns) > 0 else 0
downside_deviation = np.std(downside_returns)
sortino_ratio = (
    ((expected_return - mar) / downside_deviation) * np.sqrt(252) 
    if downside_deviation != 0 else np.nan
)

# --- Calmar Ratio ---
days_in_period = (end_time - start_time).days
final_return_pct = ((cash - INITIAL_CASH) / INITIAL_CASH) * 100
if days_in_period > 0:
    annualized_return_percentage = ((cash / INITIAL_CASH)**(365.0 / days_in_period) - 1) * 100
else:
    annualized_return_percentage = 0.0
calmar_ratio = (
    annualized_return_percentage / abs(max_drawdown)
    if max_drawdown != 0 else np.nan
)

# --- Benchmark Return (Buy & Hold) ---
initial_close = df_30m.iloc[0]['Close']
final_close   = df_30m.iloc[-1]['Close']
benchmark_return = ((final_close - initial_close) / initial_close) * 100

# =============================================================================
#                       BACKTEST PERFORMANCE SUMMARY
# =============================================================================
results = {
    "Start Date"            : df_30m.index.min().strftime("%Y-%m-%d"),
    "End Date"              : df_30m.index.max().strftime("%Y-%m-%d"),
    "Exposure Time"         : f"{exposure_time_percentage:.2f}%",
    "Final Account Balance" : f"${cash:,.2f}",
    "Equity Peak"           : f"${equity_peak:,.2f}",
    "Total Return"          : f"{final_return_pct:.2f}%",
    "Annualized Return"     : f"{annualized_return_percentage:.2f}%",
    "Benchmark Return"      : f"{benchmark_return:.2f}%",
    "Volatility (Annual)"   : f"{returns.std() * np.sqrt(252)*100:.2f}%",
    "Total Trades"          : len(trade_results),
    "Winning Trades"        : len(winning_trades),
    "Losing Trades"         : len(losing_trades),
    "Win Rate"              : f"{win_rate:.2f}%",
    "Profit Factor"         : f"{profit_factor:.2f}",
    "Sharpe Ratio"          : f"{sharpe_ratio:.2f}",
    "Sortino Ratio"         : f"{sortino_ratio:.2f}",
    "Calmar Ratio"          : f"{calmar_ratio:.2f}",
    "Max Drawdown"          : f"{max_drawdown:.2f}%",
    "Average Drawdown"      : f"{average_drawdown:.2f}%",
    "Max DD Duration"       : f"{max_drawdown_duration_days:.2f} days",
    "Avg DD Duration"       : f"{average_drawdown_duration_days:.2f} days",
}

print("\nPerformance Summary:")
for key, value in results.items():
    print(f"{key:25}: {value:>15}")

# =============================================================================
#                       PLOT EQUITY CURVES (STRATEGY VS BENCHMARK)
# =============================================================================
if len(balance_series) < 2:
    logger.warning("Not enough data points to plot equity curves.")
else:
    # Create benchmark equity curve
    initial_close = df_30m.iloc[0]['Close']
    benchmark_equity = (df_30m['Close'] / initial_close) * INITIAL_CASH

    # Align the benchmark to the strategy's balance_df index
    benchmark_equity = benchmark_equity.reindex(balance_df.index, method='ffill')
    benchmark_equity.fillna(method='ffill', inplace=True)

    # Combine into a single DataFrame for plotting
    equity_plot_df = pd.DataFrame({
        'Strategy': balance_df['Equity'],
        'Benchmark': benchmark_equity
    })

    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(equity_plot_df.index, equity_plot_df['Strategy'], label='Strategy Equity')
    plt.plot(equity_plot_df.index, equity_plot_df['Benchmark'], label='Benchmark Equity')
    plt.title('Equity Curve: Strategy vs Benchmark')
    plt.xlabel('Time')
    plt.ylabel('Account Balance ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# =============================================================================
#                    MONTE CARLO SIMULATION WITH EQUITY CURVES
# =============================================================================
def monte_carlo_simulation_with_equity_curves(trade_results, initial_cash, num_simulations=1000, random_seed=42):
    """
    Perform a Monte Carlo (bootstrap) simulation on the distribution of
    trade_results. Returns all equity curves (one per simulation) for analysis.

    Parameters
    ----------
    trade_results : list of float
        The list of per-trade PnLs from the backtest.
    initial_cash : float
        The initial cash used in the backtest.
    num_simulations : int
        Number of Monte Carlo runs.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    all_equity_curves : np.array
        Shape = (num_simulations, n_trades + 1). Each row is one sim's equity path.
    """
    if not trade_results:
        logging.warning("No trades were made. Monte Carlo simulation is not applicable.")
        return np.array([])

    np.random.seed(random_seed)
    
    n_trades = len(trade_results)
    all_equity_curves = []

    for _ in range(num_simulations):
        # Bootstrap sample of trade_results (with replacement)
        sample_trades = np.random.choice(trade_results, size=n_trades, replace=True)
        
        # Construct an equity curve by cumulatively adding each trade
        equity_curve = np.zeros(n_trades + 1)
        equity_curve[0] = initial_cash
        for i, trade_pnl in enumerate(sample_trades, start=1):
            equity_curve[i] = equity_curve[i-1] + trade_pnl
        
        all_equity_curves.append(equity_curve)

    return np.array(all_equity_curves)

# --- Run Monte Carlo ---
num_simulations = 5000
mc_equity_curves = monte_carlo_simulation_with_equity_curves(
    trade_results, INITIAL_CASH, num_simulations=num_simulations, random_seed=42
)

if mc_equity_curves.size > 0:
    final_balances = mc_equity_curves[:, -1]  # Final equity of each simulation

    # Basic distribution stats on final balances
    mc_mean   = np.mean(final_balances)
    mc_median = np.median(final_balances)
    mc_std    = np.std(final_balances)
    mc_min    = np.min(final_balances)
    mc_max    = np.max(final_balances)
    percentile_5  = np.percentile(final_balances, 5)
    percentile_95 = np.percentile(final_balances, 95)

    # Monte Carlo Results
    print("\nMonte Carlo Simulation Results:")
    print(f"Number of Simulations            : {num_simulations}")
    print(f"Mean Final Balance               : ${mc_mean:,.2f}")
    print(f"Median Final Balance             : ${mc_median:,.2f}")
    print(f"Std Dev of Final Balance         : ${mc_std:,.2f}")
    print(f"Min Final Balance (worst case)   : ${mc_min:,.2f}")
    print(f"Max Final Balance (best case)    : ${mc_max:,.2f}")
    print(f"5th Percentile                   : ${percentile_5:,.2f}")
    print(f"95th Percentile                  : ${percentile_95:,.2f}")

    # ----- Optional: Compute drawdowns, Sharpe, win rate, profit factor for each simulation -----
    # For demonstration, we'll compute them in a naive way, treating each trade as if it were
    # one discrete 'period'. You can adapt to your time-based approach as needed.
    all_drawdowns      = []
    all_total_returns  = []
    all_sharpes        = []
    all_win_rates      = []
    all_profit_factors = []

    for sim_idx in range(num_simulations):
        equity_curve = mc_equity_curves[sim_idx]
        trades_this_sim = np.diff(equity_curve)  # trade by trade PnLs
        # Drawdown
        rolling_max_sim = np.maximum.accumulate(equity_curve)
        dd_sim = (equity_curve - rolling_max_sim) / rolling_max_sim
        max_dd_sim = dd_sim.min()

        # Total Return
        total_return_sim = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

        # Sharpe Ratio
        # We'll treat each trade as a 'period', ignoring commissions/time for simplicity
        trade_returns = trades_this_sim / equity_curve[:-1]
        if trade_returns.std() != 0:
            sharpe_sim = (trade_returns.mean() / trade_returns.std()) * np.sqrt(len(trades_this_sim))
        else:
            sharpe_sim = 0
        
        # Win Rate
        wins_sim = [x for x in trades_this_sim if x > 0]
        if len(trades_this_sim) > 0:
            win_rate_sim = (len(wins_sim) / len(trades_this_sim)) * 100
        else:
            win_rate_sim = 0
        
        # Profit Factor
        gross_profit_sim = sum(x for x in trades_this_sim if x > 0)
        gross_loss_sim   = abs(sum(x for x in trades_this_sim if x < 0))
        if gross_loss_sim != 0:
            pf_sim = gross_profit_sim / gross_loss_sim
        else:
            pf_sim = np.nan

        all_drawdowns.append(max_dd_sim)
        all_total_returns.append(total_return_sim)
        all_sharpes.append(sharpe_sim)
        all_win_rates.append(win_rate_sim)
        all_profit_factors.append(pf_sim)

    print("\n-- Monte Carlo Distribution of Key Metrics --")
    print(f"Avg. Max Drawdown       : {np.mean(all_drawdowns)*100:.2f}%")
    print(f"Avg. Total Return       : {np.mean(all_total_returns)*100:.2f}%")
    print(f"Avg. Sharpe Ratio       : {np.mean(all_sharpes):.2f}")
    print(f"Avg. Win Rate           : {np.mean(all_win_rates):.2f}%")
    print(f"Avg. Profit Factor      : {np.mean(all_profit_factors):.2f}")

    # ----- Plot histogram of final balances -----
    plt.figure(figsize=(10, 6))
    plt.hist(final_balances, bins=50, color='skyblue', edgecolor='black')
    plt.title('Monte Carlo Distribution of Final Balances')
    plt.xlabel('Final Account Balance ($)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ----- Plot all equity curves to visualize variance -----
    plt.figure(figsize=(10, 6))
    for i in range(num_simulations):
        plt.plot(mc_equity_curves[i], color='blue', alpha=0.02)  # very light lines
    plt.title('Monte Carlo Equity Curves')
    plt.xlabel('Trade Count')
    plt.ylabel('Account Balance ($)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()