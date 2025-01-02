import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from datetime import timedelta, time

# =============== LOGGING SETUP ===============
logging.basicConfig(level=logging.INFO)  # Change to DEBUG for more detailed logs
logger = logging.getLogger(__name__)

# =============== MONTE CARLO IMPORTS ===============
# (None special needed beyond standard libraries above)

# =============================================================================
#                           FUNCTION: LOAD DATA
# =============================================================================
def load_data(csv_file):
    """
    Loads CSV data for CL futures, parses 'Time' as datetime, handles missing values, 
    and sets 'Time' as the index.
    Adjust column renames/cleanup to match your CSV format.
    """
    try:
        df = pd.read_csv(
            csv_file,
            parse_dates=['Time'],
            dayfirst=False,  # Adjust if your data uses day-first format (e.g. dayfirst=True)
            na_values=['', 'NA', 'NaN']  # Handle missing values
        )
        
        if 'Time' not in df.columns:
            logger.error("The CSV file does not contain a 'Time' column.")
            sys.exit(1)
        
        # Ensure 'Time' is actually datetime
        if not np.issubdtype(df['Time'].dtype, np.datetime64):
            logger.error("The 'Time' column was not parsed as datetime. Please check the date format.")
            sys.exit(1)
        
        # Convert any timezone to naive
        if df['Time'].dt.tz is not None:
            df['Time'] = df['Time'].dt.tz_convert(None)
        
        # Sort by time and set as index
        df.sort_values('Time', inplace=True)
        df.set_index('Time', inplace=True)
        
        # Example column cleanup
        # If your CSV uses 'Last' for close prices, rename to 'close'
        # Adjust as needed
        if 'Last' in df.columns:
            df.rename(columns={'Last': 'close'}, inplace=True)
        if 'Open' in df.columns:
            df.rename(columns={'Open': 'open'}, inplace=True)
        if 'High' in df.columns:
            df.rename(columns={'High': 'high'}, inplace=True)
        if 'Low' in df.columns:
            df.rename(columns={'Low': 'low'}, inplace=True)
        if 'Volume' in df.columns:
            df.rename(columns={'Volume': 'volume'}, inplace=True)
        
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
# Update to your actual file paths for CL data.
CSV_5M_FILE   = 'Data/cl_5m_data.csv'   # 5-minute CSV for stop-loss/take-profit checks
CSV_30M_FILE  = 'Data/cl_30m_data.csv'  # 30-minute CSV for Bollinger calculations

INITIAL_CASH        = 5000
POSITION_SIZE       = 3
CONTRACT_MULTIPLIER = 100  # CL: $1 per 1 cent move or $100 per $1 move
STOP_LOSS_DISTANCE  = 0.05
TAKE_PROFIT_DISTANCE= 0.15
COMMISSION          = 0.77

BOLLINGER_PERIOD = 15
BOLLINGER_STDDEV = 2

# =============================================================================
#                         LOAD AND PREPARE DATA
# =============================================================================
logger.info("Loading 5-minute and 30-minute data for CL...")
df_5m        = load_data(CSV_5M_FILE)
df_30m_full  = load_data(CSV_30M_FILE)

# --- Verify the full data range ---
print("\n5-Minute Data Range:")
print(df_5m.index.min(), "to", df_5m.index.max())
print("Rows:", len(df_5m))

print("\n30-Minute Data Range:")
print(df_30m_full.index.min(), "to", df_30m_full.index.max())
print("Rows:", len(df_30m_full))

# --- Define Backtest Period ---
custom_start_date = "2019-01-01"
custom_end_date   = "2024-12-24"

start_time = pd.to_datetime(custom_start_date)
end_time   = pd.to_datetime(custom_end_date)

# --- Filter data by the backtest period ---
df_5m       = df_5m.loc[start_time:end_time].copy()
df_30m_full = df_30m_full.loc[start_time:end_time].copy()

print("\nFiltered 5-Min Data Range:")
print(df_5m.index.min(), "to", df_5m.index.max(), "| Rows:", len(df_5m))

print("\nFiltered 30-Min Data Range:")
print(df_30m_full.index.min(), "to", df_30m_full.index.max(), "| Rows:", len(df_30m_full))

if df_30m_full.empty:
    logger.error("No 30-min data available after filtering. Exiting.")
    sys.exit(1)

# --- Calculate Bollinger Bands on 30-Min Full (including extended hours) ---
df_30m_full['ma']         = df_30m_full['close'].rolling(window=BOLLINGER_PERIOD).mean()
df_30m_full['std']        = df_30m_full['close'].rolling(window=BOLLINGER_PERIOD).std()
df_30m_full['upper_band'] = df_30m_full['ma'] + (BOLLINGER_STDDEV * df_30m_full['std'])
df_30m_full['lower_band'] = df_30m_full['ma'] - (BOLLINGER_STDDEV * df_30m_full['std'])

df_30m_full.dropna(subset=['ma', 'std', 'upper_band', 'lower_band'], inplace=True)
logger.info(f"30-Min data points after Bollinger band dropna: {len(df_30m_full)}")

# --- Filter for Regular Trading Hours (09:30 - 16:00 ET) on 30-min data for trade signals ---
def filter_rth(df):
    """
    Filters the DataFrame to RTH (09:30-16:00) Monday-Friday.
    Assumes your index is in naive or some uniform timezone (adjust if needed).
    """
    # Here we do a simple approach: keep Monday(0) to Friday(4) & between 09:30-16:00 local.
    valid_times = []
    for ts in df.index:
        if ts.weekday() < 5:  # Monday=0, ... Friday=4
            # Check time of day
            if time(9,30) <= ts.time() < time(16,0):
                valid_times.append(True)
            else:
                valid_times.append(False)
        else:
            valid_times.append(False)
    return df[valid_times]

df_30m_rth = filter_rth(df_30m_full)
if df_30m_rth.empty:
    logger.error("No 30-min RTH data points. Exiting.")
    sys.exit(1)

logger.info(f"30-Min RTH data points: {len(df_30m_rth)}")

# =============================================================================
#                         BACKTEST LOGIC
# =============================================================================
cash = INITIAL_CASH
position_size = 0
entry_price   = None
position_type = None
trade_results = []
balance_series = []
balance_dates  = []

# For tracking how many bars we're exposed in a position
exposure_bars = 0
total_bars    = len(df_30m_rth)  # We'll define exposure % as fraction of these RTH bars

def evaluate_exit_anytime(position_type, entry_price, stop_loss, take_profit, df_high_freq, entry_time):
    """
    Checks all subsequent 5m bars from entry_time forward to see if SL or TP was hit intrabar.
    Returns (exit_price, exit_time, hit_take_profit) or (None, None, None) if not hit.
    """
    df_period = df_high_freq.loc[entry_time:]
    for ts, row_5m in df_period.iterrows():
        high = row_5m['high']
        low  = row_5m['low']
        
        if position_type == 'long':
            # If both TP & SL can be hit in the same bar, check order:
            if high >= take_profit and low <= stop_loss:
                # Decide which was hit first by comparing the 'open' to SL or TP
                if row_5m['open'] <= stop_loss:
                    return (stop_loss, ts, False)
                else:
                    return (take_profit, ts, True)
            elif high >= take_profit:
                return (take_profit, ts, True)
            elif low <= stop_loss:
                return (stop_loss, ts, False)
        else:  # 'short'
            if low <= take_profit and high >= stop_loss:
                if row_5m['open'] >= stop_loss:
                    return (stop_loss, ts, False)
                else:
                    return (take_profit, ts, True)
            elif low <= take_profit:
                return (take_profit, ts, True)
            elif high >= stop_loss:
                return (stop_loss, ts, False)

    # If never triggered
    return None, None, None

# Which high-frequency data to use for checking intrabar hits
df_high_freq = df_5m

# Sort to ensure chronological order
df_high_freq.sort_index(inplace=True)

logger.info("Starting backtest loop over 5-min data...")

for tstamp, bar_5m in df_high_freq.iterrows():
    current_time  = tstamp
    current_price = bar_5m['close']

    # If we have an open position, increment exposure
    if position_size != 0:
        exposure_bars += 1

    # If no position open, check for new signals on the 30-min RTH index
    if position_size == 0:
        if current_time in df_30m_rth.index:
            # We retrieve the Bollinger band info from df_30m_full at the same timestamp
            row_30 = df_30m_full.loc[current_time]
            upper_band = row_30['upper_band']
            lower_band = row_30['lower_band']

            # Simple Mean Reversion Logic:
            # if current_price < lower_band => go long
            # if current_price > upper_band => go short
            if current_price < lower_band:
                position_size = POSITION_SIZE
                entry_price   = current_price
                position_type = 'long'
                stop_loss_price   = entry_price - STOP_LOSS_DISTANCE
                take_profit_price = entry_price + TAKE_PROFIT_DISTANCE
                entry_time        = current_time

            elif current_price > upper_band:
                position_size = POSITION_SIZE
                entry_price   = current_price
                position_type = 'short'
                stop_loss_price   = entry_price + STOP_LOSS_DISTANCE
                take_profit_price = entry_price - TAKE_PROFIT_DISTANCE
                entry_time        = current_time

        # Record the current balance with no changes
        balance_series.append(cash)
        balance_dates.append(current_time)

    else:
        # We do have a position, so check exit logic
        exit_price, exit_time, hit_tp = evaluate_exit_anytime(
            position_type,
            entry_price,
            stop_loss_price,
            take_profit_price,
            df_high_freq,
            entry_time
        )

        if exit_price is not None:
            # Calculate PnL
            if position_type == 'long':
                trade_pnl = (exit_price - entry_price) * CONTRACT_MULTIPLIER * position_size
            else:  # short
                trade_pnl = (entry_price - exit_price) * CONTRACT_MULTIPLIER * position_size
            
            # Subtract commissions (round-turn)
            trade_pnl -= (COMMISSION * position_size * 2)
            
            cash += trade_pnl
            trade_results.append(trade_pnl)

            position_size = 0
            position_type = None
            entry_price   = None
            exit_price    = None
            entry_time    = None
            stop_loss_price   = None
            take_profit_price = None

            balance_series.append(cash)
            balance_dates.append(exit_time if exit_time else current_time)
        else:
            # Still in position, no exit
            balance_series.append(cash)
            balance_dates.append(current_time)

logger.info("Backtest loop completed.")

# =============================================================================
#                      POST-BACKTEST METRICS & SUMMARY
# =============================================================================
balance_df = pd.DataFrame({
    'Datetime': balance_dates,
    'Equity': balance_series
}).drop_duplicates(subset=['Datetime']).set_index('Datetime').sort_index()

# If still in a position at the end, we do not forcibly exit. 
# (Alternatively, you could do a forced exit at last bar.)

final_balance = balance_df['Equity'].iloc[-1]
final_return_pct = ((final_balance - INITIAL_CASH) / INITIAL_CASH) * 100

equity_peak = balance_df['Equity'].max()

# Drawdown calculations
rolling_max = balance_df['Equity'].cummax()
drawdown = (balance_df['Equity'] - rolling_max) / rolling_max
max_drawdown = drawdown.min() * 100

# Drawdown durations
drawdown_periods = drawdown[drawdown < 0]
if not drawdown_periods.empty:
    # Identify contiguous segments of drawdown
    end_dates = drawdown_periods.index.to_series().diff().ne(pd.Timedelta('5min')).cumsum()
    drawdown_groups = drawdown_periods.groupby(end_dates)
    drawdown_durations_bars = drawdown_groups.size()
    # Convert bars to days (this is approximate if you want exact time)
    # We know each bar is 5min, so 1 day = 6.5 (RTH hours?) * 12 bars = 78 bars
    # This is approximate. Adjust if your data includes 24h or partial sessions
    bars_per_day = 78
    max_drawdown_duration_days = drawdown_durations_bars.max() / bars_per_day
    average_drawdown_duration_days = drawdown_durations_bars.mean() / bars_per_day
else:
    max_drawdown_duration_days    = 0
    average_drawdown_duration_days= 0

average_drawdown = drawdown[drawdown < 0].mean() * 100 if not drawdown_periods.empty else 0

# Exposure
exposure_time_percentage = (exposure_bars / total_bars) * 100 if total_bars else 0

# Profit factor
wins = [p for p in trade_results if p > 0]
losses = [p for p in trade_results if p < 0]
gross_profit = sum(wins)
gross_loss   = abs(sum(losses))
profit_factor= (gross_profit / gross_loss) if gross_loss != 0 else np.nan

win_rate = (len(wins) / len(trade_results) * 100) if trade_results else 0

# Sharpe ratio: We'll treat each 5-min equity step as a 'period' for a rough estimate
returns = balance_df['Equity'].pct_change().dropna()
if returns.std() != 0:
    # 252 trading days, ~78 bars (5-min) per day of RTH => ~ 252*78 periods = 19656
    # This is an approximation. Adjust as you like.
    annual_factor = np.sqrt(252 * 78)
    sharpe_ratio = (returns.mean() / returns.std()) * annual_factor
else:
    sharpe_ratio = 0

# Sortino ratio
def calculate_sortino_ratio(daily_returns, mar=0):
    """
    Sortino ratio. daily_returns can be any frequency, but we must scale correctly.
    Here we do a rough approach with the same annual_factor above.
    """
    if daily_returns.empty:
        return np.nan
    excess = daily_returns - mar
    downside = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return np.inf
    # Use same annual scaling factor
    ann_mean_excess = daily_returns.mean() * annual_factor
    ann_downside_std= downside.std() * annual_factor
    return ann_mean_excess / ann_downside_std

sortino_ratio = calculate_sortino_ratio(returns)

# Calmar ratio
days_in_period = (balance_df.index[-1] - balance_df.index[0]).days
if days_in_period > 0:
    annualized_return_pct = ((final_balance / INITIAL_CASH)**(365.0 / days_in_period) - 1)*100
else:
    annualized_return_pct = 0.0
calmar_ratio = (annualized_return_pct / abs(max_drawdown)) if max_drawdown != 0 else np.nan

# Benchmark Return (Assume a naive buy & hold on the 30-min close)
if not df_30m_full.empty:
    initial_bench = df_30m_full['close'].iloc[0]
    final_bench   = df_30m_full['close'].iloc[-1]
    benchmark_return = ((final_bench - initial_bench) / initial_bench) * 100
else:
    benchmark_return = 0

# =============================================================================
#                      PRINT PERFORMANCE SUMMARY
# =============================================================================
results = {
    "Start Date"             : balance_df.index.min().strftime("%Y-%m-%d %H:%M:%S"),
    "End Date"               : balance_df.index.max().strftime("%Y-%m-%d %H:%M:%S"),
    "Exposure Time"          : f"{exposure_time_percentage:.2f}%",
    "Final Account Balance"  : f"${final_balance:,.2f}",
    "Equity Peak"            : f"${equity_peak:,.2f}",
    "Total Return"           : f"{final_return_pct:.2f}%",
    "Annualized Return"      : f"{annualized_return_pct:.2f}%",
    "Benchmark Return"       : f"{benchmark_return:.2f}%",
    "Volatility (Annual)"    : f"{(returns.std()*annual_factor*100):.2f}%",
    "Total Trades"           : len(trade_results),
    "Winning Trades"         : len(wins),
    "Losing Trades"          : len(losses),
    "Win Rate"               : f"{win_rate:.2f}%",
    "Profit Factor"          : f"{profit_factor:.2f}",
    "Sharpe Ratio"           : f"{sharpe_ratio:.2f}",
    "Sortino Ratio"          : f"{sortino_ratio:.2f}",
    "Calmar Ratio"           : f"{calmar_ratio:.2f}",
    "Max Drawdown"           : f"{max_drawdown:.2f}%",
    "Average Drawdown"       : f"{average_drawdown:.2f}%",
    "Max DD Duration"        : f"{max_drawdown_duration_days:.2f} days",
    "Avg DD Duration"        : f"{average_drawdown_duration_days:.2f} days",
}

print("\nPerformance Summary:")
for k, v in results.items():
    print(f"{k:25}: {v:>15}")

# =============================================================================
#                    PLOT EQUITY CURVE VS. BENCHMARK
# =============================================================================
if len(balance_df) > 1:
    # Create a benchmark equity curve
    if not df_30m_full.empty:
        initial_close = df_30m_full['close'].iloc[0]
        benchmark_equity = (df_30m_full['close'] / initial_close) * INITIAL_CASH
        # Align the benchmark to the strategy's timeline
        benchmark_equity = benchmark_equity.reindex(balance_df.index, method='ffill').fillna(method='ffill')
    else:
        # If no benchmark data
        benchmark_equity = pd.Series(index=balance_df.index, data=INITIAL_CASH)
    
    equity_plot_df = pd.DataFrame({
        'Strategy': balance_df['Equity'],
        'Benchmark': benchmark_equity
    })
    
    plt.figure(figsize=(14, 7))
    plt.plot(equity_plot_df.index, equity_plot_df['Strategy'], label='Strategy Equity')
    plt.plot(equity_plot_df.index, equity_plot_df['Benchmark'], label='Benchmark Equity')
    plt.title('Equity Curve: Strategy vs. Benchmark')
    plt.xlabel('Time')
    plt.ylabel('Account Balance ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    logger.warning("Not enough data points to plot equity curve.")

# =============================================================================
#               MONTE CARLO SIMULATION WITH EQUITY CURVES
# =============================================================================
def monte_carlo_simulation_with_metrics(trade_results, initial_cash, num_simulations=1000, random_seed=42):
    """
    Perform a bootstrap Monte Carlo simulation on the distribution of trade_results.
    Each simulation re-samples the per-trade PnLs with replacement, building an
    equity curve and collecting various performance metrics.
    
    Returns:
        final_balances (np.array): Final account balances from each simulation.
        metrics_dict (dict): Dictionary containing arrays of various metrics from each simulation.
    """
    if not trade_results:
        logger.warning("No trades were made. Monte Carlo simulation is not applicable.")
        return np.array([]), {}
    
    np.random.seed(random_seed)
    
    n_trades = len(trade_results)
    all_equity_curves = []
    
    # Initialize lists to collect metrics
    all_final_balances = []
    all_drawdowns = []
    all_total_returns = []
    all_sharpes = []
    all_sortinos = []
    all_calmar_ratios = []
    all_win_rates = []
    all_profit_factors = []
    
    for _ in range(num_simulations):
        # Bootstrap sample of trade_results (with replacement)
        sample_trades = np.random.choice(trade_results, size=n_trades, replace=True)
        
        # Construct equity curve
        equity_curve = np.zeros(n_trades + 1)
        equity_curve[0] = initial_cash
        for i, trade_pnl in enumerate(sample_trades, start=1):
            equity_curve[i] = equity_curve[i-1] + trade_pnl
        
        all_equity_curves.append(equity_curve)
        
        # Calculate Metrics
        final_balance = equity_curve[-1]
        all_final_balances.append(final_balance)
        
        # Max Drawdown
        roll_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - roll_max) / roll_max
        max_dd = drawdown.min()
        all_drawdowns.append(max_dd)
        
        # Total Return
        tot_ret = (final_balance - initial_cash) / initial_cash
        all_total_returns.append(tot_ret)
        
        # Sharpe Ratio
        trade_returns = sample_trades / equity_curve[:-1]
        if trade_returns.std() != 0:
            sharpe = (trade_returns.mean() / trade_returns.std()) * np.sqrt(252 * 78)  # Adjust annualization factor as needed
        else:
            sharpe = 0
        all_sharpes.append(sharpe)
        
        # Sortino Ratio
        downside = trade_returns[trade_returns < 0]
        if downside.size > 0:
            sortino = (trade_returns.mean() / downside.std()) * np.sqrt(252 * 78)
        else:
            sortino = np.inf
        all_sortinos.append(sortino)
        
        # Calmar Ratio
        days_in_period = (end_time - start_time).days
        if days_in_period > 0 and abs(max_dd) > 0:
            annualized_return = ((final_balance / initial_cash) ** (365.0 / days_in_period) - 1)
            calmar = annualized_return / abs(max_dd)
        else:
            calmar = np.nan
        all_calmar_ratios.append(calmar)
        
        # Win Rate
        wins = sample_trades[sample_trades > 0]
        losses = sample_trades[sample_trades < 0]
        win_rate = (len(wins) / len(sample_trades) * 100) if len(sample_trades) > 0 else 0
        all_win_rates.append(win_rate)
        
        # Profit Factor
        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss != 0 else np.nan
        all_profit_factors.append(profit_factor)
    
    # Organize metrics into a dictionary
    metrics_dict = {
        "Final Balance": np.array(all_final_balances),
        "Max Drawdown": np.array(all_drawdowns),
        "Total Return": np.array(all_total_returns),
        "Sharpe Ratio": np.array(all_sharpes),
        "Sortino Ratio": np.array(all_sortinos),
        "Calmar Ratio": np.array(all_calmar_ratios),
        "Win Rate": np.array(all_win_rates),
        "Profit Factor": np.array(all_profit_factors),
    }
    
    return np.array(all_final_balances), metrics_dict

# --- Run Monte Carlo ---
num_simulations = 2000
mc_final_balances, mc_metrics = monte_carlo_simulation_with_metrics(
    trade_results, INITIAL_CASH, num_simulations=num_simulations, random_seed=42
)

if mc_final_balances.size > 0:
    # Define a helper function to calculate statistics
    def calculate_stats(data):
        return {
            "Mean": np.mean(data),
            "Median": np.median(data),
            "Min": np.min(data),
            "Max": np.max(data),
            "Std Dev": np.std(data),
            "5th Percentile": np.percentile(data, 5),
            "95th Percentile": np.percentile(data, 95),
        }
    
    # Calculate stats for each metric
    stats_summary = {metric: calculate_stats(values) for metric, values in mc_metrics.items()}
    
    # Convert to a DataFrame for better readability
    stats_df = pd.DataFrame(stats_summary).T
    stats_df = stats_df[['Mean', 'Median', 'Min', 'Max', 'Std Dev', '5th Percentile', '95th Percentile']]
    
    print("\nMonte Carlo Simulation Detailed Statistics:")
    print(stats_df.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x))
    
    # =============================================================================
    #                      PLOT MONTE CARLO FINAL Balances
    # =============================================================================
    # -- Plot histogram of final balances --
    plt.figure(figsize=(10, 6))
    plt.hist(mc_final_balances, bins=50, color='skyblue', edgecolor='black')
    plt.title('Monte Carlo Distribution of Final Balances')
    plt.xlabel('Final Account Balance ($)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # -- Plot all equity curves (light lines) --
    # Optional: Plot a subset to avoid performance issues
    plt.figure(figsize=(10, 6))
    subset = np.random.choice(mc_final_balances.size, size=min(num_simulations, 500), replace=False)
    for i in subset:
        plt.plot(mc_metrics["Final Balance"][i], color='blue', alpha=0.02)
    plt.title('Monte Carlo Equity Curves (Subset)')
    plt.xlabel('Trade Count')
    plt.ylabel('Account Balance ($)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # =============================================================================
    #               ADDITIONAL MONTE CARLO METRICS
    # =============================================================================
    # Example: Display percentiles for each metric
    for metric, stats in stats_summary.items():
        print(f"\n{metric} Statistics:")
        for stat_name, value in stats.items():
            if "Percentile" in stat_name:
                print(f"  {stat_name}: ${value:,.2f}" if "Balance" in metric else f"  {stat_name}: {value:.2f}%")
            else:
                print(f"  {stat_name}: ${value:,.2f}" if "Balance" in metric else f"  {stat_name}: {value:.2f}")