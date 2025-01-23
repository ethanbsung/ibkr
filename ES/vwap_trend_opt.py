import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import pytz
import logging
import sys
from itertools import product

# --- Configuration Parameters ---
DATA_PATH = 'Data/es_1m_data.csv'  # Path to your 1-minute data CSV
INITIAL_CAPITAL = 5000             # Starting cash in USD
POSITION_SIZE = 1                  # Number of contracts per trade
CONTRACT_MULTIPLIER = 5            # Contract multiplier for MES

TIMEFRAME = '15min'                   # 15-minute timeframe

STOP_LOSS_POINTS = 10               # Stop loss in points
TAKE_PROFIT_POINTS = 20             # Take profit in points

# Custom Backtest Dates (inclusive)
START_DATE = '2021-01-01'           # Format: 'YYYY-MM-DD'
END_DATE = '2024-12-31'             # Format: 'YYYY-MM-DD'

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,  # Set to INFO or DEBUG for more verbosity
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# --- Helper Functions ---

def filter_rth(df):
    """
    Filters the DataFrame to include only Regular Trading Hours (09:30 - 16:00 ET) on weekdays.

    Parameters:
        df (pd.DataFrame): The input DataFrame with a timezone-aware datetime index.

    Returns:
        pd.DataFrame: The filtered DataFrame containing only RTH data.
    """
    eastern = pytz.timezone('US/Eastern')

    # Localize to US/Eastern if naive, else convert to US/Eastern
    if df.index.tz is None:
        df = df.tz_localize(eastern)
        logger.debug("Localized naive datetime index to US/Eastern.")
    else:
        df = df.tz_convert(eastern)
        logger.debug("Converted timezone-aware datetime index to US/Eastern.")

    # Filter for weekdays (Monday=0 to Friday=4)
    df = df[df.index.weekday < 5]

    # Filter for RTH hours: 09:30 to 16:00
    df = df.between_time('09:30', '16:00')

    # Convert back to UTC for consistency
    df = df.tz_convert('UTC')

    return df

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

        # Sort and set 'Time' as index
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
            # Drop rows with NaN 'close'
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

def calculate_rsi(series, period=14):
    """
    Calculates the Relative Strength Index (RSI) for a given pandas Series.

    Parameters:
        series (pd.Series): Series of prices.
        period (int): Period for RSI calculation.

    Returns:
        pd.Series: RSI values.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Calculate exponential moving averages
    avg_gain = gain.ewm(com=(period - 1), min_periods=period).mean()
    avg_loss = loss.ewm(com=(period - 1), min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# --- Backtest Class ---
class MESFuturesBacktest:
    def __init__(self, data_path, start_date, end_date, timeframe, initial_capital,
                 position_size, contract_multiplier, stop_loss_points, take_profit_points,
                 rsi_upper, rsi_lower):
        """
        Initializes the backtest.

        Parameters:
            data_path (str): Path to the CSV data file.
            start_date (str): Start date for the backtest in 'YYYY-MM-DD' format.
            end_date (str): End date for the backtest in 'YYYY-MM-DD' format.
            timeframe (str): Resampling timeframe for the strategy (e.g., '15T' for 15 minutes).
            initial_capital (float): Starting capital for the backtest.
            risk_reward (float): Risk-reward ratio for take profit.
            position_size (int): Number of contracts per trade.
            contract_multiplier (int): Contract multiplier for MES.
            stop_loss_points (int): Stop loss in points.
            take_profit_points (int): Take profit in points.
            rsi_upper (int): Upper RSI threshold for long entry.
            rsi_lower (int): Lower RSI threshold for short entry.
        """
        self.data_path = data_path
        self.start_date = pd.to_datetime(start_date).tz_localize('UTC')
        self.end_date = pd.to_datetime(end_date).tz_localize('UTC')
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.contract_multiplier = contract_multiplier
        self.stop_loss_points = stop_loss_points
        self.take_profit_points = take_profit_points
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower

        self.load_data()
        self.prepare_data()

        # Initialize backtest variables
        self.cash = initial_capital
        self.equity = initial_capital
        self.equity_curve = []
        self.trade_log = []
        self.position = None  # None, 'long', 'short'
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.exposed_bars = 0
        self.total_bars = len(self.data_15m)
        self.equity_peak = initial_capital
        self.drawdowns = []
        self.current_drawdown = 0
        self.in_drawdown = False
        self.drawdown_start = None
        self.drawdown_durations = []

    def load_data(self):
        """Loads and preprocesses the data."""
        logger.info("Loading data...")
        self.data = load_data(self.data_path)
        logger.info("Data loaded successfully.")

    def prepare_data(self):
        """Prepares data by filtering RTH, resampling, calculating indicators, and applying date filters."""
        logger.info("Preparing data for backtest...")

        # Filter Regular Trading Hours
        self.data_rth = filter_rth(self.data)
        logger.info(f"Filtered data to Regular Trading Hours. Total data points: {len(self.data_rth)}")

        # Filter by Start and End Dates
        self.data_rth = self.data_rth[(self.data_rth.index >= self.start_date) & (self.data_rth.index <= self.end_date)]
        logger.info(f"Filtered data by date from {self.start_date.date()} to {self.end_date.date()}. Total data points: {len(self.data_rth)}")

        # Resample to specified timeframe
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

        # Calculate VWAP
        typical_price = (self.data_15m['high'] + self.data_15m['low'] + self.data_15m['close']) / 3
        self.data_15m['vwap'] = (typical_price * self.data_15m['volume']).cumsum() / self.data_15m['volume'].cumsum()

        # Calculate RSI
        self.data_15m['rsi'] = calculate_rsi(self.data_15m['close'])

        # Drop initial rows with NaN RSI
        self.data_15m.dropna(inplace=True)
        logger.info("Calculated VWAP and RSI.")

    def run_backtest(self):
        """Executes the backtest by iterating over each 15-minute bar."""
        logger.info("Starting backtest execution...")
        for idx, row in self.data_15m.iterrows():
            price = row['close']
            vwap = row['vwap']
            rsi = row['rsi']

            # Check if we are in a position
            if self.position is None:
                # Check for Long Entry
                if price > vwap and rsi > self.rsi_upper:
                    self.position = 'long'
                    self.entry_price = price
                    self.stop_loss = price - self.stop_loss_points
                    self.take_profit = price + self.take_profit_points
                    self.trade_log.append({
                        'Type': 'Long',
                        'Entry Time': idx,
                        'Entry Price': price,
                        'Exit Time': None,
                        'Exit Price': None,
                        'Result': None,
                        'Profit': 0
                    })
                    logger.debug(f"Entered LONG at {price} on {idx}")
                    self.exposed_bars += 1
                # Check for Short Entry
                elif price < vwap and rsi < self.rsi_lower:
                    self.position = 'short'
                    self.entry_price = price
                    self.stop_loss = price + self.stop_loss_points
                    self.take_profit = price - self.take_profit_points
                    self.trade_log.append({
                        'Type': 'Short',
                        'Entry Time': idx,
                        'Entry Price': price,
                        'Exit Time': None,
                        'Exit Price': None,
                        'Result': None,
                        'Profit': 0
                    })
                    logger.debug(f"Entered SHORT at {price} on {idx}")
                    self.exposed_bars += 1
            else:
                # Check for Exit Conditions
                exit_signal = False
                result = None
                exit_price = None

                if self.position == 'long':
                    if price <= self.stop_loss:
                        exit_signal = True
                        result = 'Stop Loss'
                        exit_price = self.stop_loss
                    elif price >= self.take_profit:
                        exit_signal = True
                        result = 'Take Profit'
                        exit_price = self.take_profit
                elif self.position == 'short':
                    if price >= self.stop_loss:
                        exit_signal = True
                        result = 'Stop Loss'
                        exit_price = self.stop_loss
                    elif price <= self.take_profit:
                        exit_signal = True
                        result = 'Take Profit'
                        exit_price = self.take_profit

                if exit_signal:
                    # Calculate Profit & Loss
                    if self.position == 'long':
                        pnl = (exit_price - self.entry_price) * self.position_size * self.contract_multiplier
                    elif self.position == 'short':
                        pnl = (self.entry_price - exit_price) * self.position_size * self.contract_multiplier

                    self.cash += pnl
                    self.equity += pnl

                    # Update Trade Log
                    self.trade_log[-1].update({
                        'Exit Time': idx,
                        'Exit Price': exit_price,
                        'Result': result,
                        'Profit': pnl
                    })
                    logger.debug(f"Exited {self.position.upper()} at {exit_price} on {idx} via {result} for P&L: ${pnl:.2f}")

                    # Reset Position
                    self.position = None
                    self.entry_price = 0
                    self.stop_loss = 0
                    self.take_profit = 0
                else:
                    self.exposed_bars += 1

            # Update Equity Curve
            self.equity_curve.append({'Time': idx, 'Equity': self.equity})

            # Update Equity Peak
            if self.equity > self.equity_peak:
                self.equity_peak = self.equity
                if self.in_drawdown:
                    drawdown_duration = (idx - self.drawdown_start).total_seconds() / 86400  # in days
                    self.drawdown_durations.append(drawdown_duration)
                    self.in_drawdown = False
                    self.current_drawdown = 0

            # Calculate Drawdown
            current_drawdown = (self.equity_peak - self.equity) / self.equity_peak * 100
            if current_drawdown > self.current_drawdown:
                self.current_drawdown = current_drawdown
                if not self.in_drawdown:
                    self.in_drawdown = True
                    self.drawdown_start = idx

            if self.in_drawdown and current_drawdown < self.current_drawdown:
                # Continuing drawdown
                pass
            elif self.in_drawdown and current_drawdown >= self.current_drawdown:
                # Recovery from drawdown
                self.drawdowns.append(self.current_drawdown)
                self.current_drawdown = 0
                self.in_drawdown = False
                drawdown_duration = (idx - self.drawdown_start).total_seconds() / 86400  # in days
                self.drawdown_durations.append(drawdown_duration)
                self.drawdown_start = None

    def close_open_position(self, backtester):
        """Closes any open position at the end of the backtest."""
        if backtester.position is not None:
            exit_price = backtester.data_15m['close'].iloc[-1]
            idx = backtester.data_15m.index[-1]
            if backtester.position == 'long':
                pnl = (exit_price - backtester.entry_price) * backtester.position_size * backtester.contract_multiplier
            elif backtester.position == 'short':
                pnl = (backtester.entry_price - exit_price) * backtester.position_size * backtester.contract_multiplier
            backtester.cash += pnl
            backtester.equity += pnl
            backtester.trade_log[-1].update({
                'Exit Time': idx,
                'Exit Price': exit_price,
                'Result': 'End of Data',
                'Profit': pnl
            })
            logger.debug(f"Exited {backtester.position.upper()} at {exit_price} on {idx} via End of Data for P&L: ${pnl:.2f}")
            backtester.equity_curve.append({'Time': idx, 'Equity': backtester.equity})
            backtester.position = None

    def analyze_results(self):
        """Analyzes and returns the backtest results without plotting."""
        # Close any open position
        self.close_open_position(self)

        if not self.trade_log:
            logger.warning("No trades were executed.")
            return None

        # Convert trade log to DataFrame
        trade_results = pd.DataFrame(self.trade_log)

        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('Time', inplace=True)

        # Calculate performance metrics
        start_date = equity_df.index.min().strftime("%Y-%m-%d")
        end_date = equity_df.index.max().strftime("%Y-%m-%d")
        exposure_time_percentage = (self.exposed_bars / self.total_bars) * 100
        final_account_balance = self.equity
        equity_peak = self.equity_peak
        total_return_percentage = ((final_account_balance - self.initial_capital) / self.initial_capital) * 100

        # Calculate time delta in years
        delta = equity_df.index.max() - equity_df.index.min()
        years = delta.days / 365.25 if delta.days > 0 else 0
        annualized_return_percentage = ((final_account_balance / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Benchmark Return (Buy and Hold)
        benchmark_start_price = self.data_15m['close'].iloc[0]
        benchmark_end_price = self.data_15m['close'].iloc[-1]
        benchmark_return = ((benchmark_end_price - benchmark_start_price) / benchmark_start_price) * 100

        # Daily Returns for volatility and Sharpe Ratio
        equity_daily = equity_df['Equity'].resample('D').last().dropna()
        daily_returns = equity_daily.pct_change().dropna()

        # Volatility (Annual)
        volatility_annual = daily_returns.std() * np.sqrt(252) * 100  # Assuming 252 trading days

        # Sharpe Ratio
        risk_free_rate = 0  # Assuming risk-free rate is 0
        sharpe_ratio = ((daily_returns.mean() - risk_free_rate) / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0

        # Sortino Ratio
        downside_returns = daily_returns[daily_returns < 0]
        sortino_ratio = (daily_returns.mean() * 252) / (downside_returns.std() * np.sqrt(252)) if not downside_returns.empty else np.inf

        # Drawdown Calculations
        running_max = equity_df['Equity'].cummax()
        drawdowns = (equity_df['Equity'] - running_max) / running_max * 100
        max_drawdown = drawdowns.min()
        average_drawdown = drawdowns[drawdowns < 0].mean() if not drawdowns[drawdowns < 0].empty else 0

        # Profit Factor
        winning_trades = trade_results[trade_results['Result'] == 'Take Profit']['Profit']
        losing_trades = trade_results[trade_results['Result'] == 'Stop Loss']['Profit']
        profit_factor = winning_trades.sum() / abs(losing_trades.sum()) if not losing_trades.empty else np.inf

        # Calmar Ratio
        calmar_ratio = (total_return_percentage / abs(max_drawdown)) if abs(max_drawdown) != 0 else np.inf

        # Drawdown Durations
        max_drawdown_duration_days = max(self.drawdown_durations) if self.drawdown_durations else 0
        average_drawdown_duration_days = np.mean(self.drawdown_durations) if self.drawdown_durations else 0

        # Win Rate
        total_trades = len(trade_results)
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0.0

        # Prepare Results Dictionary
        results = {
            "Start Date": start_date,
            "End Date": end_date,
            "Exposure Time (%)": f"{exposure_time_percentage:.2f}%",
            "Final Account Balance": final_account_balance,
            "Equity Peak": equity_peak,
            "Total Return (%)": total_return_percentage,
            "Annualized Return (%)": annualized_return_percentage,
            "Benchmark Return (%)": benchmark_return,
            "Volatility (Annual %)": volatility_annual,
            "Total Trades": total_trades,
            "Winning Trades": len(winning_trades),
            "Losing Trades": len(losing_trades),
            "Win Rate (%)": win_rate,
            "Profit Factor": profit_factor,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Calmar Ratio": calmar_ratio,
            "Max Drawdown (%)": max_drawdown,
            "Average Drawdown (%)": average_drawdown,
            "Max Drawdown Duration (days)": max_drawdown_duration_days,
            "Average Drawdown Duration (days)": average_drawdown_duration_days,
        }

        return results

# --- Optimization Function ---
def optimize_strategy(param_grid, data_path, start_date, end_date):
    """
    Optimizes the MES Futures strategy by grid searching over parameter combinations.

    Parameters:
        param_grid (dict): Dictionary where keys are parameter names and values are lists of parameter settings to try.
        data_path (str): Path to the CSV data file.
        start_date (str): Start date for the backtest.
        end_date (str): End date for the backtest.

    Returns:
        pd.DataFrame: DataFrame containing performance metrics for each parameter combination.
    """
    logger.info("Starting optimization process...")
    # Generate all combinations of parameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    logger.info(f"Total parameter combinations to evaluate: {len(param_combinations)}")

    results_list = []

    for idx, params in enumerate(param_combinations, start=1):
        logger.info(f"Evaluating combination {idx}/{len(param_combinations)}: {params}")
        try:
            backtester = MESFuturesBacktest(
                data_path=data_path,
                start_date=start_date,
                end_date=end_date,
                timeframe=TIMEFRAME,
                initial_capital=INITIAL_CAPITAL,
                position_size=POSITION_SIZE,
                contract_multiplier=CONTRACT_MULTIPLIER,
                stop_loss_points=params['stop_loss_points'],
                take_profit_points=params['take_profit_points'],
                rsi_upper=params['rsi_upper'],
                rsi_lower=params['rsi_lower']
            )
            backtester.run_backtest()
            metrics = backtester.analyze_results()

            if metrics is not None:
                # Store the relevant metrics for optimization
                result = {
                    'rsi_upper': params['rsi_upper'],
                    'rsi_lower': params['rsi_lower'],
                    'stop_loss_points': params['stop_loss_points'],
                    'take_profit_points': params['take_profit_points'],
                    'Total Return (%)': metrics['Total Return (%)'],
                    'Sharpe Ratio': metrics['Sharpe Ratio']
                }
                results_list.append(result)
            else:
                # No trades executed; append NaNs or appropriate values
                result = {
                    'rsi_upper': params['rsi_upper'],
                    'rsi_lower': params['rsi_lower'],
                    'stop_loss_points': params['stop_loss_points'],
                    'take_profit_points': params['take_profit_points'],
                    'Total Return (%)': np.nan,
                    'Sharpe Ratio': np.nan
                }
                results_list.append(result)

        except Exception as e:
            logger.error(f"Error evaluating combination {params}: {e}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)
    logger.info("Optimization process completed.")
    return results_df

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the data file exists
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file {DATA_PATH} does not exist. Please check the path and filename.")
        sys.exit(1)
    else:
        try:
            # Define parameter grid for optimization
            param_grid = {
                'rsi_upper': [70, 80],        # Upper RSI threshold for long entry
                'rsi_lower': [20, 30],            # Lower RSI threshold for short entry
                'stop_loss_points': range(4, 10, 1),      # Stop loss in points
                'take_profit_points': range(15, 30, 1)    # Take profit in points
            }

            # Run optimization
            optimization_results = optimize_strategy(
                param_grid=param_grid,
                data_path=DATA_PATH,
                start_date=START_DATE,
                end_date=END_DATE
            )

            # Drop rows with NaN Sharpe Ratios
            optimization_results = optimization_results.dropna(subset=['Sharpe Ratio'])

            if not optimization_results.empty:
                # Identify the best parameter set based on Sharpe Ratio
                best_params = optimization_results.loc[optimization_results['Sharpe Ratio'].idxmax()]
                print("\n--- Optimal Parameters Based on Sharpe Ratio ---")
                print(best_params)
            else:
                logger.warning("No valid optimization results to display.")

            # Save the optimization results to CSV
            optimization_results.to_csv('optimization_results.csv', index=False)
            logger.info("Optimization results saved to 'optimization_results.csv'.")

        except Exception as e:
            logger.error(f"An error occurred during backtesting: {e}")
            sys.exit(1)