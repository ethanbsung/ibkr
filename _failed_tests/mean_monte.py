import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from sklearn.utils import resample
import multiprocessing
import warnings
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================
# 1. Helper Functions
# ==========================

def load_data(csv_file):
    """
    Loads CSV data into a Pandas DataFrame with appropriate data types and datetime parsing.
    """
    try:
        df = pd.read_csv(
            csv_file,
            dtype={
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': float,
                'average': float,
                'barCount': int,
                'contract': str
            },
            parse_dates=['date'],
            date_format="%Y-%m-%d %H:%M:%S%z"
        )
        df.sort_values('date', inplace=True)
        df.set_index('date', inplace=True)
        return df
    except FileNotFoundError:
        print(f"File not found: {csv_file}")
        exit(1)
    except pd.errors.EmptyDataError:
        print("No data: The CSV file is empty.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        exit(1)

def calculate_sortino_ratio(daily_returns, target_return=0):
    """
    Calculate the annualized Sortino Ratio.
    """
    if daily_returns.empty:
        return np.nan
    
    # Calculate excess returns
    excess_returns = daily_returns - target_return
    
    # Calculate downside returns (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]

    # Handle cases where there are no downside returns
    if downside_returns.empty or downside_returns.std() == 0:
        return np.inf  # No downside risk means infinite Sortino Ratio

    # Annualize downside standard deviation
    downside_std = downside_returns.std() * np.sqrt(252)

    # Annualize mean excess return
    annualized_mean_excess_return = daily_returns.mean() * 252

    # Return Sortino Ratio
    return annualized_mean_excess_return / downside_std

def evaluate_exit(position_type, entry_price, stop_loss, take_profit, df_high_freq, entry_time):
    """
    Determines whether the stop-loss or take-profit is hit using higher-frequency data.
    Returns exit_price, exit_time, and hit_take_profit flag.
    """
    df_period = df_high_freq.loc[entry_time:]

    # Iterate through each higher-frequency bar after entry_time
    for timestamp, row in df_period.iterrows():
        high = row['high']
        low = row['low']

        if position_type == 'long':
            if high >= take_profit and low <= stop_loss:
                # Determine which was hit first
                # Assuming the open of the bar is the first price, check sequence
                if row['open'] <= stop_loss:
                    return stop_loss, timestamp, False
                else:
                    return take_profit, timestamp, True
            elif high >= take_profit:
                return take_profit, timestamp, True
            elif low <= stop_loss:
                return stop_loss, timestamp, False

        elif position_type == 'short':
            if low <= take_profit and high >= stop_loss:
                if row['open'] >= stop_loss:
                    return stop_loss, timestamp, False
                else:
                    return take_profit, timestamp, True
            elif low <= take_profit:
                return take_profit, timestamp, True
            elif high >= stop_loss:
                return stop_loss, timestamp, False

    # If neither condition is met, wait for the next bar
    return None, None, None

# ==========================
# 2. Backtest Function
# ==========================

def run_backtest(
    df_1m,
    df_30m,
    start_time,
    end_time,
    initial_cash=5000,
    position_multiplier=5,
    spread_cost=0.47 * 2,
    stop_loss_offset=5,
    take_profit_offset=10,
    bollinger_period=15,
    bollinger_stddev=2,
    df_high_freq_choice='1m'
):
    """
    Runs the backtest on provided data and returns performance metrics.
    """
    # Make copies to avoid modifying original data
    df_1m = df_1m.copy()
    df_30m = df_30m.copy()

    # Slice data
    df_1m = df_1m.loc[start_time:end_time]
    df_30m = df_30m.loc[start_time:end_time]

    # Calculate Bollinger Bands on 30m data
    df_30m['ma'] = df_30m['close'].rolling(window=bollinger_period).mean()
    df_30m['std'] = df_30m['close'].rolling(window=bollinger_period).std()
    df_30m['upper_band'] = df_30m['ma'] + (bollinger_stddev * df_30m['std'])
    df_30m['lower_band'] = df_30m['ma'] - (bollinger_stddev * df_30m['std'])

    df_30m.dropna(inplace=True)

    # Initialize backtest variables
    position_size = 0
    entry_price = None
    position_type = None  
    cash = initial_cash
    trade_results = []
    balance_series = [initial_cash]  # Keep as a list
    exposure_bars = 0

    # For Drawdown Duration Calculations
    in_drawdown = False
    drawdown_start = None
    drawdown_durations = []

    # Define which high-frequency data to use
    if df_high_freq_choice == '1m':
        df_high_freq = df_1m
    elif df_high_freq_choice == '5m':
        df_high_freq = df_1m.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'average': 'mean',
            'barCount': 'sum',
            'contract': 'first'
        })
    else:
        df_high_freq = df_1m  # Default to 1m

    # Backtesting loop
    for i in range(len(df_30m)):
        current_bar = df_30m.iloc[i]
        current_time = df_30m.index[i]
        current_price = current_bar['close']

        # Count exposure when position is active
        if position_size != 0:
            exposure_bars += 1

        if position_size == 0:
            # No open position, check for entry signals based on 30m bar
            if current_price < current_bar['lower_band']:
                # Enter Long
                position_size = 1
                entry_price = current_price
                position_type = 'long'
                stop_loss_price = entry_price - stop_loss_offset
                take_profit_price = entry_price + take_profit_offset
                entry_time = current_time
                #print(f"Entered LONG at {entry_price} on {entry_time} UTC")

            elif current_price > current_bar['upper_band']:
                # Enter Short
                position_size = 1
                entry_price = current_price
                position_type = 'short'
                stop_loss_price = entry_price + stop_loss_offset
                take_profit_price = entry_price - take_profit_offset
                entry_time = current_time
                #print(f"Entered SHORT at {entry_price} on {entry_time} UTC")

        else:
            # Position is open, check high-frequency data until exit
            exit_price, exit_time, hit_take_profit = evaluate_exit(
                position_type,
                entry_price,
                stop_loss_price,
                take_profit_price,
                df_high_freq,
                entry_time
            )

            if exit_price is not None and exit_time is not None:
                # Calculate P&L based on the exit condition
                if position_type == 'long':
                    pnl = ((exit_price - entry_price) * position_multiplier) - spread_cost
                elif position_type == 'short':
                    pnl = ((entry_price - exit_price) * position_multiplier) - spread_cost
                
                trade_results.append(pnl)
                cash += pnl
                balance_series.append(cash)  # Append to list

                # Reset position variables
                position_size = 0
                position_type = None
                entry_price = None
                stop_loss_price = None
                take_profit_price = None
                entry_time = None

    # After the Backtesting Loop

    # Convert balance_series to a Pandas Series
    balance_series = pd.Series(balance_series, index=df_30m.index[:len(balance_series)])

    # Drawdown Duration Tracking
    for i in range(len(balance_series)):
        current_balance = balance_series.iloc[i]
        running_max = balance_series.iloc[:i+1].max()

        if current_balance < running_max:
            if not in_drawdown:
                in_drawdown = True
                drawdown_start = balance_series.index[i]
        else:
            if in_drawdown:
                in_drawdown = False
                drawdown_end = balance_series.index[i]
                duration = (drawdown_end - drawdown_start).total_seconds() / 86400  # Duration in days
                drawdown_durations.append(duration)

    # Handle if still in drawdown at the end of the data
    if in_drawdown:
        drawdown_end = balance_series.index[-1]
        duration = (drawdown_end - drawdown_start).total_seconds() / 86400
        drawdown_durations.append(duration)

    # Fix the FutureWarning by specifying fill_method=None
    daily_returns = balance_series.resample('D').last().pct_change(fill_method=None).dropna()

    # Performance Metrics
    total_return_percentage = ((cash - initial_cash) / initial_cash) * 100
    trading_days = max((df_30m.index.max() - df_30m.index.min()).days, 1)
    annualized_return_percentage = ((cash / initial_cash) ** (252 / trading_days)) - 1
    benchmark_return = ((df_30m['close'].iloc[-1] - df_30m['close'].iloc[0]) / df_30m['close'].iloc[0]) * 100
    equity_peak = balance_series.max()

    volatility_annual = daily_returns.std() * np.sqrt(252) * 100
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    sortino_ratio = calculate_sortino_ratio(daily_returns)

    # Drawdown Calculations
    running_max_series = balance_series.cummax()
    drawdowns = (balance_series - running_max_series) / running_max_series
    max_drawdown = drawdowns.min() * 100
    average_drawdown = drawdowns[drawdowns < 0].mean() * 100

    # Exposure Time
    exposure_time_percentage = (exposure_bars / len(df_30m)) * 100

    # Profit Factor
    winning_trades = [pnl for pnl in trade_results if pnl > 0]
    losing_trades = [pnl for pnl in trade_results if pnl <= 0]
    profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf')

    # Calmar Ratio Calculation
    calmar_ratio = (total_return_percentage / abs(max_drawdown)) if max_drawdown != 0 else float('inf')

    # Drawdown Duration Calculations
    if drawdown_durations:
        max_drawdown_duration_days = max(drawdown_durations)
        average_drawdown_duration_days = np.mean(drawdown_durations)
    else:
        max_drawdown_duration_days = 0
        average_drawdown_duration_days = 0

    # Collect all metrics in a dictionary
    results = {
        "Final Account Balance": cash,
        "Total Return (%)": total_return_percentage,
        "Annualized Return (%)": annualized_return_percentage * 100,
        "Benchmark Return (%)": benchmark_return,
        "Equity Peak": equity_peak,
        "Volatility (Annual %)": volatility_annual,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
        "Max Drawdown (%)": max_drawdown,
        "Average Drawdown (%)": average_drawdown,
        "Max Drawdown Duration (days)": max_drawdown_duration_days,
        "Average Drawdown Duration (days)": average_drawdown_duration_days,
        "Exposure Time (%)": exposure_time_percentage,
        "Total Trades": len(trade_results),
        "Winning Trades": len(winning_trades),
        "Losing Trades": len(losing_trades),
        "Win Rate (%)": (len(winning_trades)/len(trade_results)*100) if trade_results else 0,
        "Profit Factor": profit_factor,
        "Trade PnL": trade_results,
        "Balance Series": balance_series
    }

    return results

# ==========================
# 3. Monte Carlo Simulation
# ==========================

def block_bootstrap(df, block_size):
    """
    Perform block bootstrapping on a DataFrame.
    Divides the DataFrame into blocks of size 'block_size' and resamples these blocks with replacement.
    """
    blocks = [df.iloc[i:i + block_size] for i in range(0, len(df), block_size)]
    resampled_blocks = resample(blocks, replace=True, n_samples=int(np.ceil(len(blocks))))
    df_resampled = pd.concat(resampled_blocks).iloc[:len(df)]  # Ensure the same length
    df_resampled = df_resampled.sort_index()
    return df_resampled

def run_single_simulation(sim_number, df_1m, df_30m, initial_cash, position_multiplier, spread_cost, stop_loss_offset, take_profit_offset, bollinger_period, bollinger_stddev, df_high_freq_choice, block_size):
    """
    Runs a single Monte Carlo simulation by resampling the 30m data with block bootstrapping.
    """
    # Define the number of blocks based on block_size
    n_blocks = int(np.ceil(len(df_30m) / block_size))

    # Perform block bootstrapping
    df_30m_resampled = block_bootstrap(df_30m, block_size)

    # Align the 1m data to the resampled 30m data's date range
    start_time = df_30m_resampled.index.min()
    end_time = df_30m_resampled.index.max()
    df_1m_resampled = df_1m.loc[start_time:end_time]

    # Handle cases where the resampled 1m data might be empty
    if df_1m_resampled.empty:
        # Return a balance_series filled with initial_cash aligned with df_30m_resampled
        balance_series = pd.Series([initial_cash] * len(df_30m_resampled), index=df_30m_resampled.index)
        return {
            "Final Account Balance": initial_cash,
            "Total Return (%)": 0.0,
            "Annualized Return (%)": 0.0,
            "Benchmark Return (%)": 0.0,
            "Equity Peak": initial_cash,
            "Volatility (Annual %)": 0.0,
            "Sharpe Ratio": np.nan,
            "Sortino Ratio": np.inf,
            "Calmar Ratio": np.inf,
            "Max Drawdown (%)": 0.0,
            "Average Drawdown (%)": 0.0,
            "Max Drawdown Duration (days)": 0.0,
            "Average Drawdown Duration (days)": 0.0,
            "Exposure Time (%)": 0.0,
            "Total Trades": 0,
            "Winning Trades": 0,
            "Losing Trades": 0,
            "Win Rate (%)": 0.0,
            "Profit Factor": np.nan,
            "Trade PnL": [],
            "Balance Series": balance_series  # Properly aligned balance_series
        }

    # Run backtest on resampled data
    backtest_result = run_backtest(
        df_1m_resampled,
        df_30m_resampled,
        start_time,
        end_time,
        initial_cash=initial_cash,
        position_multiplier=position_multiplier,
        spread_cost=spread_cost,
        stop_loss_offset=stop_loss_offset,
        take_profit_offset=take_profit_offset,
        bollinger_period=bollinger_period,
        bollinger_stddev=bollinger_stddev,
        df_high_freq_choice=df_high_freq_choice
    )

    return backtest_result


def monte_carlo_simulation(
    df_1m,
    df_30m,
    num_simulations=200,
    initial_cash=5000,
    position_multiplier=5,
    spread_cost=0.47 * 2,
    stop_loss_offset=5,
    take_profit_offset=10,
    bollinger_period=15,
    bollinger_stddev=2,
    df_high_freq_choice='1m',
    block_size=10  # Size of blocks for block bootstrapping
):
    """
    Runs multiple backtests with randomized scenarios and collects performance metrics.
    Utilizes multiprocessing for faster execution with a progress bar.
    """
    print(f"Starting Monte Carlo simulation with {num_simulations} simulations...")
    simulation_results = []

    # Prepare arguments for parallel processing
    args = [
        (
            sim,
            df_1m,
            df_30m,
            initial_cash,
            position_multiplier,
            spread_cost,
            stop_loss_offset,
            take_profit_offset,
            bollinger_period,
            bollinger_stddev,
            df_high_freq_choice,
            block_size
        )
        for sim in range(num_simulations)
    ]

    # Use multiprocessing Pool with imap_unordered and tqdm for progress
    with multiprocessing.Pool() as pool:
        for result in tqdm(pool.imap_unordered(run_single_simulation, args), total=num_simulations, desc="Simulations", unit="sim"):
            simulation_results.append(result)

    print("All Monte Carlo simulations completed.")
    return simulation_results

# ==========================
# 4. Plotting Functions
# ==========================


def plot_equity_curves(equity_df):
    """
    Plots equity curves for Monte Carlo simulations.

    Parameters:
    equity_df (DataFrame): DataFrame where rows represent time and columns are individual simulation equity curves.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(equity_df, alpha=0.3, color='blue')  # Plot all simulations
    plt.title('Monte Carlo Simulation: Equity Curves')
    plt.xlabel('Time (Periods)')
    plt.ylabel('Account Balance')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Mean Equity Curve
    mean_equity = equity_df.mean(axis=1)
    plt.figure(figsize=(12, 8))
    plt.plot(mean_equity, color='red', label='Mean Equity Curve')
    plt.fill_between(
        equity_df.index, 
        equity_df.min(axis=1), 
        equity_df.max(axis=1), 
        color='blue', alpha=0.2, label='Range (Min/Max)'
    )
    plt.title('Mean Equity Curve with Min/Max Range')
    plt.xlabel('Time (Periods)')
    plt.ylabel('Account Balance')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# ==========================
# 5. Main Function
# ==========================

def main():
    # Load datasets
    csv_file_1m = 'es_1m_data.csv'
    csv_file_5m = 'es_5m_data.csv'
    csv_file_30m = 'es_30m_data.csv'

    df_1m = load_data(csv_file_1m)
    df_5m = load_data(csv_file_5m)
    df_30m = load_data(csv_file_30m)

    # Option 1: Custom Backtest Period (Replace These Dates)
    custom_start_date = "2022-10-01"
    custom_end_date = "2024-12-11"

    # Option 2: Use Full Available Data (if custom dates are not set)
    if custom_start_date and custom_end_date:
        start_time = pd.to_datetime(custom_start_date, utc=True)
        end_time = pd.to_datetime(custom_end_date, utc=True)
    else:
        start_time = pd.to_datetime(df_30m.index.min(), utc=True)
        end_time = pd.to_datetime(df_30m.index.max(), utc=True)

    # Ensure the 1-minute DataFrame index is in UTC
    df_1m.index = pd.to_datetime(df_1m.index, utc=True)

    # Slice the 1-minute and 30-minute DataFrames using the chosen backtest period
    df_1m = df_1m.loc[start_time:end_time]
    df_30m = df_30m.loc[start_time:end_time]

    print(f"Backtesting from {start_time} to {end_time}")

    # Run a single backtest to ensure everything is working
    single_backtest = run_backtest(
        df_1m,
        df_30m,
        start_time,
        end_time,
        initial_cash=5000,
        position_multiplier=5,
        spread_cost=0.47 * 2,
        stop_loss_offset=5,
        take_profit_offset=10,
        bollinger_period=15,
        bollinger_stddev=2,
        df_high_freq_choice='1m'
    )

    # Print single backtest results
    print("\nSingle Backtest Performance:")
    for key, value in single_backtest.items():
        if key != "Trade PnL":
            print(f"{key:30}: {value:>15}")

    # Define Monte Carlo simulation parameters
    num_simulations = 200  # Number of Monte Carlo runs

    # Run Monte Carlo Simulations
    print(f"\nRunning {num_simulations} Monte Carlo simulations...")
    simulation_results = monte_carlo_simulation(
        df_1m=df_1m,
        df_30m=df_30m,
        num_simulations=num_simulations,
        initial_cash=5000,
        position_multiplier=5,
        spread_cost=0.47 * 2,
        stop_loss_offset=5,
        take_profit_offset=10,
        bollinger_period=15,
        bollinger_stddev=2,
        df_high_freq_choice='1m',
        block_size=10  # Adjust block size as needed
    )
    print("Monte Carlo simulations completed.")

    # Convert simulation results to DataFrame for analysis
    simulation_df = pd.DataFrame(simulation_results)

    equity_curves = pd.DataFrame(
        [result["Balance Series"] for result in simulation_results if "Balance Series" in result and isinstance(result["Balance Series"], pd.Series)]
    ).T

    # Drop any columns with all NaNs
    equity_curves.dropna(axis=1, how='all', inplace=True)

    # Convert all data to numeric, coercing errors
    equity_curves = equity_curves.apply(pd.to_numeric, errors='coerce')


    # Display summary statistics
    print("\nMonte Carlo Simulation Summary:")
    summary_metrics = [
        "Final Account Balance",
        "Total Return (%)",
        "Annualized Return (%)",
        "Benchmark Return (%)",
        "Equity Peak",
        "Volatility (Annual %)",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Calmar Ratio",
        "Max Drawdown (%)",
        "Average Drawdown (%)",
        "Max Drawdown Duration (days)",
        "Average Drawdown Duration (days)",
        "Exposure Time (%)",
        "Total Trades",
        "Winning Trades",
        "Losing Trades",
        "Win Rate (%)",
        "Profit Factor"
    ]

    for metric in summary_metrics:
        if metric in simulation_df.columns:
            print(f"\n{metric}:")
            print(simulation_df[metric].describe())

    # Plot Monte Carlo Simulation Results
    plot_equity_curves(simulation_df)

if __name__ == "__main__":
    main()