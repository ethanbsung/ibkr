import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pytz
import logging
import sys
import gc

# --- Configuration Parameters ---
DATA_PATH = 'Data/es_1m_data.csv'  # Path to your 1-minute data CSV
INITIAL_CAPITAL = 5000             # Starting cash in USD
POSITION_SIZE = 1                   # Number of contracts per trade
CONTRACT_MULTIPLIER = 5             # Contract multiplier for MES

TIMEFRAME = '15T'                    # 15-minute timeframe ('15T' for 15 minutes)
RISK_REWARD_RATIO = 2               # Risk-Reward Ratio for take profit

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,  # Set to INFO or DEBUG for more verbosity
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# --- Helper Function to Filter Regular Trading Hours (RTH) ---
def filter_rth(df):
    """
    Filters the DataFrame to include only Regular Trading Hours (09:30 - 16:00 ET) on weekdays.

    Parameters:
        df (pd.DataFrame): The input DataFrame with a timezone-aware datetime index.

    Returns:
        pd.DataFrame: The filtered DataFrame containing only RTH data.
    """
    # Define US/Eastern timezone
    eastern = pytz.timezone('US/Eastern')

    # Ensure the index is timezone-aware
    if df.index.tz is None:
        # Assume the data is in US/Eastern if no timezone is set
        df = df.tz_localize(eastern)
        logger.debug("Localized naive datetime index to US/Eastern.")
    else:
        # Convert to US/Eastern to standardize
        df = df.tz_convert(eastern)
        logger.debug("Converted timezone-aware datetime index to US/Eastern.")

    # Filter for weekdays (Monday=0 to Friday=4)
    df_eastern = df[df.index.weekday < 5]

    # Filter for RTH hours: 09:30 to 16:00
    df_rth = df_eastern.between_time('09:30', '16:00')

    # Convert back to UTC for consistency in further processing
    df_rth = df_rth.tz_convert('UTC')

    return df_rth

# --- Function to Load Data ---
def load_data(csv_file):
    """
    Loads CSV data into a pandas DataFrame with appropriate data types and datetime parsing.

    Parameters:
        csv_file (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded and indexed DataFrame.
    """
    try:
        # Define converters for specific columns
        converters = {
            '%Chg': lambda x: float(x.strip('%')) if isinstance(x, str) else np.nan
        }

        df = pd.read_csv(
            csv_file,
            dtype={
                'Symbol': str,
                'Open': float,
                'High': float,
                'Low': float,
                'Last': float,
                'Change': float,
                'Volume': float,
                'Open Int': float
            },
            parse_dates=['Time'],
            # Removed 'infer_datetime_format' to eliminate FutureWarning
            # Let pandas infer the datetime format
            converters=converters
        )

        # Strip column names to remove leading/trailing spaces
        df.columns = df.columns.str.strip()

        # Log the columns present in the CSV
        logger.info(f"Loaded '{csv_file}' with columns: {df.columns.tolist()}")

        # Check if 'Time' column exists
        if 'Time' not in df.columns:
            logger.error(f"The 'Time' column is missing in the file: {csv_file}")
            sys.exit(1)

        df.sort_values('Time', inplace=True)
        df.set_index('Time', inplace=True)

        # Rename columns to match expected names
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Last': 'close',
            'Volume': 'volume',
            'Symbol': 'contract',
            '%Chg': 'pct_chg'  # Rename to avoid issues with '%' in column name
        }, inplace=True)

        # Ensure 'close' is numeric and handle any non-convertible values
        df['close'] = pd.to_numeric(df['close'], errors='coerce')

        # Check for NaNs in 'close' and log if any
        num_close_nans = df['close'].isna().sum()
        if num_close_nans > 0:
            logger.warning(f"'close' column has {num_close_nans} NaN values in file: {csv_file}")
            # Optionally, drop rows with NaN 'close' or handle them as needed
            df = df.dropna(subset=['close'])
            logger.info(f"Dropped rows with NaN 'close'. Remaining data points: {len(df)}")

        # Add missing columns with default values
        df['average'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['barCount'] = 1  # Assuming each row is a single bar

        # Ensure 'contract' is a string
        df['contract'] = df['contract'].astype(str)

        # Validate that all required columns are present
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'contract', 'pct_chg', 'average', 'barCount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns {missing_columns} in file: {csv_file}")
            sys.exit(1)

        # Final check for 'close' column
        if df['close'].isna().any():
            logger.error(f"After processing, 'close' column still contains NaNs in file: {csv_file}")
            sys.exit(1)

        # Additional Data Validation
        logger.debug(f"'close' column statistics:\n{df['close'].describe()}")
        logger.debug(f"Unique 'close' values: {df['close'].nunique()}")

        return df
    except FileNotFoundError:
        logger.error(f"File not found: {csv_file}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logger.error("No data: The CSV file is empty.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred while reading the CSV '{csv_file}': {e}")
        sys.exit(1)

# --- Backtest Class ---
class RangeBreakoutBacktest:
    def __init__(self, data_path, timeframe='15T', initial_capital=5000, risk_reward=2):
        """
        Initializes the backtest.

        Parameters:
        - data_path (str): Path to the CSV data file.
        - timeframe (str): Resampling timeframe for range breakout (e.g., '15T' for 15 minutes).
        - initial_capital (float): Starting capital for the backtest.
        - risk_reward (float): Risk-reward ratio for take profit.
        """
        self.data_path = data_path
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.risk_reward = risk_reward
        self.load_data()
        self.prepare_data()
        self.results = []
        self.equity_curve = []
        self.cash = initial_capital
        self.position_size = POSITION_SIZE  # Number of contracts per trade

    def load_data(self):
        """Loads CSV data into a pandas DataFrame with appropriate data types and datetime parsing."""
        self.data = load_data(self.data_path)
        logger.info("Data loaded successfully.")

    def prepare_data(self):
        """Prepares data by resampling to the desired timeframe and calculating range boundaries."""
        logger.info("Preparing data for backtest...")

        # Filter Regular Trading Hours
        self.data_rth = filter_rth(self.data)
        logger.info(f"Filtered data to Regular Trading Hours. Total data points: {len(self.data_rth)}")

        # Resample to 15-minute bars
        self.data_15m = self.data_rth.resample(self.timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'average': 'mean',
            'barCount': 'sum'
        }).dropna()
        logger.info(f"Resampled data to {self.timeframe}. Total bars: {len(self.data_15m)}")

        # Define the first 15-minute period of each day as the range period
        self.data_15m['date'] = self.data_15m.index.date
        self.data_15m['time'] = self.data_15m.index.time

        # Extract the first 15-minute bar's high and low for each day
        first_15m = self.data_15m.groupby('date').first()
        self.range_df = first_15m[['high', 'low']].rename(columns={'high': 'OR_high', 'low': 'OR_low'})
        logger.info("Range boundaries (OR_high and OR_low) calculated based on the first 15-minute period each day.")

    def run_backtest(self):
        """Executes the backtest by iterating over each 15-minute bar."""
        logger.info("Starting backtest execution...")
        position = None  # Can be 'long' or 'short'
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        trade_log = []
        equity = self.initial_capital
        equity_peak = self.initial_capital
        drawdowns = []
        drawdown = 0
        in_drawdown = False
        drawdown_start = None
        drawdown_durations = []
        total_bars = len(self.data_15m)
        exposed_bars = 0

        # For calculating returns over time
        equity_time_series = []

        for idx, row in self.data_15m.iterrows():
            current_time = idx
            price = row['close']

            # Get the date for the current bar
            current_date = current_time.date()

            # Get OR_high and OR_low for the current day
            OR_high = self.range_df.loc[current_date, 'OR_high']
            OR_low = self.range_df.loc[current_date, 'OR_low']

            # Check for entry signal
            if position is None:
                if price > OR_high:
                    # Enter Long
                    position = 'long'
                    entry_price = price
                    stop_loss = OR_low
                    take_profit = entry_price + (self.risk_reward * (entry_price - stop_loss))
                    trade_log.append({
                        'type': 'Long',
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'exit_time': None,
                        'exit_price': None,
                        'result': None,
                        'profit': 0
                    })
                    exposed_bars += 1
                    logger.debug(f"Long entered at {entry_price} on {current_time}")
                elif price < OR_low:
                    # Enter Short
                    position = 'short'
                    entry_price = price
                    stop_loss = OR_high
                    take_profit = entry_price - (self.risk_reward * (stop_loss - entry_price))
                    trade_log.append({
                        'type': 'Short',
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'exit_time': None,
                        'exit_price': None,
                        'result': None,
                        'profit': 0
                    })
                    exposed_bars += 1
                    logger.debug(f"Short entered at {entry_price} on {current_time}")
            else:
                # Position is open, check for exit conditions
                exit_triggered = False
                exit_type = None
                exit_price = None

                if position == 'long':
                    if price <= stop_loss:
                        # Stop loss hit
                        exit_triggered = True
                        exit_type = 'Stop Loss'
                        exit_price = stop_loss
                    elif price >= take_profit:
                        # Take profit hit
                        exit_triggered = True
                        exit_type = 'Take Profit'
                        exit_price = take_profit
                elif position == 'short':
                    if price >= stop_loss:
                        # Stop loss hit
                        exit_triggered = True
                        exit_type = 'Stop Loss'
                        exit_price = stop_loss
                    elif price <= take_profit:
                        # Take profit hit
                        exit_triggered = True
                        exit_type = 'Take Profit'
                        exit_price = take_profit

                if exit_triggered:
                    # Calculate P&L
                    if position == 'long':
                        pnl = (exit_price - entry_price) * self.position_size * CONTRACT_MULTIPLIER
                    elif position == 'short':
                        pnl = (entry_price - exit_price) * self.position_size * CONTRACT_MULTIPLIER

                    # Update equity
                    equity += pnl

                    # Update trade log
                    trade_log[-1].update({
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'result': exit_type,
                        'profit': pnl
                    })

                    logger.debug(f"{position.capitalize()} exited at {exit_price} on {current_time} via {exit_type} for P&L: ${pnl:.2f}")

                    # Reset position
                    position = None
                    entry_price = 0
                    stop_loss = 0
                    take_profit = 0
                else:
                    # Position is still open
                    exposed_bars += 1

            # Update equity peak
            if equity > equity_peak:
                equity_peak = equity
                if in_drawdown:
                    drawdown_durations.append((current_time - drawdown_start).total_seconds() / 86400)  # in days
                    in_drawdown = False
                    drawdown = 0

            # Calculate drawdown
            current_drawdown = (equity_peak - equity) / equity_peak * 100
            if current_drawdown > drawdown:
                drawdown = current_drawdown
                if not in_drawdown:
                    in_drawdown = True
                    drawdown_start = current_time
            if in_drawdown and current_drawdown < drawdown:
                # Continuing drawdown
                pass
            elif in_drawdown and current_drawdown >= drawdown:
                # Recovery from drawdown
                drawdowns.append(drawdown)
                drawdown = 0
                in_drawdown = False
                drawdown_durations.append((current_time - drawdown_start).total_seconds() / 86400)  # in days
                drawdown_start = None

            # Append current equity to time series
            equity_time_series.append({'timestamp': current_time, 'equity': equity})

        # Close any open positions at the end of the backtest
        if position is not None:
            last_price = self.data_15m['close'].iloc[-1]
            current_time = self.data_15m.index[-1]
            if position == 'long':
                pnl = (last_price - entry_price) * self.position_size * CONTRACT_MULTIPLIER
            elif position == 'short':
                pnl = (entry_price - last_price) * self.position_size * CONTRACT_MULTIPLIER
            equity += pnl
            trade_log[-1].update({
                'exit_time': current_time,
                'exit_price': last_price,
                'result': 'End of Data',
                'profit': pnl
            })
            logger.debug(f"{position.capitalize()} exited at {last_price} on {current_time} via End of Data for P&L: ${pnl:.2f}")

        # Final calculations
        self.results = pd.DataFrame(trade_log)
        self.final_capital = equity
        self.equity_curve = pd.DataFrame(equity_time_series).set_index('timestamp')
        self.equity_peak = equity_peak
        self.drawdowns = drawdowns
        self.max_drawdown = min(drawdowns) if drawdowns else 0
        self.average_drawdown = np.mean(drawdowns) if drawdowns else 0
        self.drawdown_durations = drawdown_durations
        self.max_drawdown_duration = max(drawdown_durations) if drawdown_durations else 0
        self.average_drawdown_duration = np.mean(drawdown_durations) if drawdown_durations else 0
        self.exposure_time_percentage = (exposed_bars / total_bars) * 100 if total_bars > 0 else 0

        logger.info("Backtest execution completed.")

    def analyze_results(self):
        """Analyzes and prints the backtest results."""
        if self.results.empty:
            print("No trades were executed.")
            return

        # Calculate Start and End Dates
        start_date = self.data_15m.index.min().strftime("%Y-%m-%d")
        end_date = self.data_15m.index.max().strftime("%Y-%m-%d")

        # Total Return
        total_return = ((self.final_capital - self.initial_capital) / self.initial_capital) * 100

        # Time delta in years for annualized return
        delta = self.data_15m.index.max() - self.data_15m.index.min()
        years = delta.days / 365.25 if delta.days > 0 else 0

        # Annualized Return
        if years > 0:
            annualized_return = ((self.final_capital / self.initial_capital) ** (1 / years) - 1) * 100
        else:
            annualized_return = 0.0

        # Benchmark Return (Buy & Hold)
        benchmark_start_price = self.data_15m['open'].iloc[0]
        benchmark_end_price = self.data_15m['close'].iloc[-1]
        benchmark_return = ((benchmark_end_price - benchmark_start_price) / benchmark_start_price) * 100

        # Daily Returns for volatility and Sharpe Ratio
        equity_daily = self.equity_curve['equity'].resample('D').last().dropna()
        daily_returns = equity_daily.pct_change().dropna()

        # Volatility (Annual)
        volatility_annual = daily_returns.std() * np.sqrt(252) * 100  # Assuming 252 trading days

        # Sharpe Ratio
        risk_free_rate = 0  # Example: 0 for simplicity or use the current rate
        sharpe_ratio = ((daily_returns.mean() - risk_free_rate) / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0

        # Sortino Ratio
        def calculate_sortino_ratio(daily_returns, target_return=0):
            """
            Calculate the annualized Sortino Ratio.

            Parameters:
                daily_returns (pd.Series): Series of daily returns.
                target_return (float): Minimum acceptable return.

            Returns:
                float: Annualized Sortino Ratio.
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

        sortino_ratio = calculate_sortino_ratio(daily_returns)

        # Drawdown Calculations
        running_max_series = self.equity_curve['equity'].cummax()
        drawdowns = (self.equity_curve['equity'] - running_max_series) / running_max_series
        max_drawdown = drawdowns.min() * 100
        average_drawdown = drawdowns[drawdowns < 0].mean() * 100 if not drawdowns[drawdowns < 0].empty else 0

        # Exposure Time
        exposure_time_percentage = self.exposure_time_percentage

        # Profit Factor
        winning_trades = self.results[self.results['result'] == 'Take Profit']['profit']
        losing_trades = self.results[self.results['result'] == 'Stop Loss']['profit']
        profit_factor = winning_trades.sum() / abs(losing_trades.sum()) if not losing_trades.empty else np.inf

        # Calmar Ratio
        calmar_ratio = (total_return / abs(max_drawdown)) if abs(max_drawdown) != 0 else np.inf

        # Drawdown Duration Calculations
        drawdown_periods = self.drawdown_durations
        if drawdown_periods:
            max_drawdown_duration_days = max(drawdown_periods)
            average_drawdown_duration_days = np.mean(drawdown_periods)
        else:
            max_drawdown_duration_days = 0
            average_drawdown_duration_days = 0

        # Win Rate
        total_trades = len(self.results)
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0.0

        # Prepare Results Dictionary
        results_summary = {
            "Start Date": start_date,
            "End Date": end_date,
            "Exposure Time": f"{exposure_time_percentage:.2f}%",
            "Final Account Balance": f"${self.final_capital:,.2f}",
            "Equity Peak": f"${self.equity_peak:,.2f}",
            "Total Return": f"{total_return:.2f}%",
            "Annualized Return": f"{annualized_return:.2f}%",
            "Benchmark Return": f"{benchmark_return:.2f}%",
            "Volatility (Annual)": f"{volatility_annual:.2f}%",
            "Total Trades": total_trades,
            "Winning Trades": len(winning_trades),
            "Losing Trades": len(losing_trades),
            "Win Rate": f"{win_rate:.2f}%",
            "Profit Factor": f"{profit_factor:.2f}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Sortino Ratio": f"{sortino_ratio:.2f}",
            "Calmar Ratio": f"{calmar_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2f}%",
            "Average Drawdown": f"{average_drawdown:.2f}%",
            "Max Drawdown Duration": f"{max_drawdown_duration_days:.2f} days",
            "Average Drawdown Duration": f"{average_drawdown_duration_days:.2f} days",
        }

        # Print Results Summary
        print("\n--- Performance Summary ---")
        for key, value in results_summary.items():
            print(f"{key:25}: {value:>15}")

        # --- Plot Equity Curves ---
        if len(self.equity_curve) < 2:
            logger.warning("Not enough data points to plot equity curves.")
        else:
            # Create benchmark equity curve (Buy & Hold)
            initial_close = self.data_15m['close'].iloc[0]
            benchmark_equity = (self.data_15m['close'] / initial_close) * self.initial_capital

            # Align the benchmark to the strategy's equity_curve
            benchmark_equity = benchmark_equity.reindex(self.equity_curve.index, method='ffill')

            # Ensure no NaNs in benchmark_equity
            benchmark_equity = benchmark_equity.fillna(method='ffill')

            # Create a DataFrame for plotting
            equity_df = pd.DataFrame({
                'Strategy': self.equity_curve['equity'],
                'Benchmark': benchmark_equity
            })

            # Plotting
            plt.figure(figsize=(14, 7))
            plt.plot(equity_df['Strategy'], label='Strategy Equity')
            plt.plot(equity_df['Benchmark'], label='Benchmark Equity (Buy & Hold)')
            plt.title('Equity Curve: Strategy vs Benchmark')
            plt.xlabel('Time')
            plt.ylabel('Account Balance ($)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the data file exists
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file {DATA_PATH} does not exist. Please check the path and filename.")
        sys.exit(1)
    else:
        try:
            backtester = RangeBreakoutBacktest(
                data_path=DATA_PATH,
                timeframe=TIMEFRAME,
                initial_capital=INITIAL_CAPITAL,
                risk_reward=RISK_REWARD_RATIO
            )
            backtester.run_backtest()
            backtester.analyze_results()

            # Optional: Save Trade Results to CSV
            # Uncomment the following lines if you wish to save trade results for further analysis
            # backtester.results.to_csv('trade_results.csv', index=False)
            # logger.info("Trade results saved to 'trade_results.csv'.")

            # Optional: Save Equity Curve to CSV
            # Uncomment the following lines if you wish to save the equity curve
            # backtester.equity_curve.to_csv('equity_curve.csv')
            # logger.info("Equity curve saved to 'equity_curve.csv'.")

        except Exception as e:
            logger.error(f"An error occurred during backtesting: {e}")
            sys.exit(1)