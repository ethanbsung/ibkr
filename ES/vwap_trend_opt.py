import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Added for enhanced visualization
import os
from datetime import datetime, time, timedelta
import pytz
import logging
import sys
from itertools import product
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# --- Configuration Parameters ---
DATA_PATH = 'Data/es_1m_data.csv'  # Path to your 1-minute data CSV
INITIAL_CAPITAL = 5000             # Starting cash in USD
POSITION_SIZE = 1                  # Number of contracts per trade
CONTRACT_MULTIPLIER = 5            # Contract multiplier for MES

TIMEFRAME = '1min'                 # Keep original 1-minute timeframe for the main data

# Optimization Parameters (Define ranges here)
STOP_LOSS_POINTS_RANGE = range(3, 10, 1)      # 3 to 9 inclusive
TAKE_PROFIT_POINTS_RANGE = range(10, 30, 1)  # 10 to 29 inclusive

COMMISSION = 0.62                  # Commission per trade (entry or exit)
SLIPPAGE = 1                       # Slippage in points on entry

# Custom Backtest Dates (inclusive)
START_DATE = '2015-01-01'          # Format: 'YYYY-MM-DD'
END_DATE = '2024-12-25'            # Format: 'YYYY-MM-DD'

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to reduce verbosity
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# --- Helper Functions ---

def mark_rth(df):
    """
    Adds a column to the DataFrame indicating whether each bar is within Regular Trading Hours (RTH).
    RTH: 09:30 - 16:00 ET on weekdays.
    """
    eastern = pytz.timezone('US/Eastern')

    # Localize to US/Eastern if naive, else convert to US/Eastern
    if df.index.tz is None:
        df = df.tz_localize(eastern)
        logger.debug("Localized naive datetime index to US/Eastern.")
    else:
        df = df.tz_convert(eastern)
        logger.debug("Converted timezone-aware datetime index to US/Eastern.")

    # Mark RTH: weekdays and between 09:30-16:00
    df['is_rth'] = df.index.weekday < 5  # Monday=0, Sunday=6
    df['time'] = df.index.time
    df.loc[(df['time'] < time(9, 30)) | (df['time'] > time(16, 0)), 'is_rth'] = False

    # Remove the temporary 'time' column
    df.drop(['time'], axis=1, inplace=True)

    # Convert back to UTC for consistency
    df = df.tz_convert('UTC')
    return df

def load_data(csv_file):
    """
    Loads CSV data into a pandas DataFrame with appropriate data types and datetime parsing.
    """
    try:
        # Define converters for specific columns
        converters = {
            '%Chg': lambda x: float(x.strip('%')) if isinstance(x, str) else np.nan
        }

        df = pd.read_csv(
            csv_file,
            dtype={
                'Symbol': 'category',
                'Open': 'float32',
                'High': 'float32',
                'Low': 'float32',
                'Last': 'float32',
                'Change': 'float32',
                'Volume': 'float32',
                'Open Int': 'float32'
            },
            parse_dates=['Time'],
            converters=converters
        )

        # Strip column names to remove leading/trailing spaces
        df.columns = df.columns.str.strip()

        #logger.info(f"Loaded '{csv_file}' with columns: {df.columns.tolist()}")

        if 'Time' not in df.columns:
            logger.error(f"The 'Time' column is missing in the file: {csv_file}")
            sys.exit(1)

        # Sort and set 'Time' as index
        df.sort_values('Time', inplace=True)
        df.set_index('Time', inplace=True)

        # Rename columns
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Last': 'close',
            'Volume': 'volume',
            'Symbol': 'contract',
            '%Chg': 'pct_chg'
        }, inplace=True)

        # Ensure 'close' is numeric
        df['close'] = pd.to_numeric(df['close'], errors='coerce')

        # Drop rows with NaN in 'close'
        num_close_nans = df['close'].isna().sum()
        if num_close_nans > 0:
            logger.warning(f"'close' column has {num_close_nans} NaN values. Dropping those rows.")
            df = df.dropna(subset=['close'])
            #logger.info(f"Remaining data points: {len(df)}")

        # Add missing columns
        df['average'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['barCount'] = 1  # Assume each row is a single bar

        # Validate required columns
        required_columns = ['open', 'high', 'close', 'low', 'volume', 'contract', 'pct_chg', 'average', 'barCount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns {missing_columns} in file: {csv_file}")
            sys.exit(1)

        # Final check
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

def rsi(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate the RSI on the given DataFrame (expects 'close')."""
    delta = ohlc['close'].diff()

    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    _gain = up.ewm(com=(period - 1), min_periods=period).mean()
    _loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()

    RS = _gain / _loss
    rsi_series = 100 - (100 / (1 + RS))
    rsi_series.name = "RSI"
    return rsi_series

def calculate_vwap(ohlc: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP), resetting each day,
    on the provided time-indexed DataFrame.
    """
    ohlc = ohlc.copy()
    
    # Convert to US/Eastern
    ohlc_eastern = ohlc.tz_convert('US/Eastern')
    
    # Extract date for grouping
    ohlc_eastern['date'] = ohlc_eastern.index.date
    
    # Typical Price
    typical_price = (ohlc_eastern['high'] + ohlc_eastern['low'] + ohlc_eastern['close']) / 3
    
    # TPV and daily cumsum
    ohlc_eastern['tpv'] = typical_price * ohlc_eastern['volume']
    ohlc_eastern['cum_tpv'] = ohlc_eastern.groupby('date')['tpv'].cumsum()
    ohlc_eastern['cum_vol'] = ohlc_eastern.groupby('date')['volume'].cumsum()
    
    # VWAP
    ohlc_eastern['vwap'] = ohlc_eastern['cum_tpv'] / ohlc_eastern['cum_vol']
    
    # Assign back
    ohlc['vwap'] = ohlc_eastern['vwap']
    return ohlc['vwap']

# --- Backtest Class ---
class MESFuturesBacktest:
    def __init__(self, data_path, start_date, end_date, timeframe, initial_capital,
                 position_size, contract_multiplier, stop_loss_points, take_profit_points,
                 commission, slippage, rsi_period=14):
        """
        Initializes the backtest.
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
        self.commission = commission
        self.slippage = slippage
        self.rsi_period = rsi_period  # RSI period on the 15-minute bars

        self.load_data()
        self.prepare_data()

        # Initialize backtest state
        self.cash = initial_capital
        self.equity = initial_capital
        self.equity_curve = []
        self.trade_log = []
        self.position = None  # 'long' or 'short'
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.exposed_bars = 0
        self.total_bars = len(self.data_prepared)
        self.equity_peak = initial_capital
        self.drawdowns = []
        self.current_drawdown = 0
        self.in_drawdown = False
        self.drawdown_start = None
        self.drawdown_durations = []

    def load_data(self):
        """Loads and preprocesses the CSV data."""
        #logger.info("Loading data...")
        self.data = load_data(self.data_path)
        #logger.info("Data loaded successfully.")

    def prepare_data(self):
        """
        Prepares 1-minute data: mark RTH, filter by date, resample to 1-min,
        calculate VWAP, and precompute 15-minute RSI.
        """
        #logger.info("Preparing data for backtest...")

        # Mark RTH
        self.data = mark_rth(self.data)
        #logger.info("Marked Regular Trading Hours in the data.")

        # Filter date range
        self.data = self.data[(self.data.index >= self.start_date) & (self.data.index <= self.end_date)]
        #logger.info(f"Filtered data from {self.start_date.date()} to {self.end_date.date()}. Points: {len(self.data)}")

        # Sort
        self.data.sort_index(inplace=True)

        # Resample to the chosen timeframe (likely '1min', so this is effectively a no-op if data is 1min)
        #logger.info(f"Resampling data to {self.timeframe} bars if needed...")
        self.data_prepared = self.data.resample(self.timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'contract': 'first',
            'pct_chg': 'last',
            'average': 'mean',
            'barCount': 'sum',
            'is_rth': 'last'
        }).dropna()
        #logger.info(f"Resampled data: {len(self.data_prepared)} bars.")

        # Compute VWAP on the 1-min data (we keep VWAP updated every minute)
        #logger.info("Calculating 1-minute VWAP (reset daily)...")
        self.data_prepared['vwap'] = calculate_vwap(self.data_prepared)
        #logger.info("Finished calculating VWAP.")

        # Precompute 15-minute RSI
        #logger.info("Calculating 15-minute RSI...")
        bars_15m = self.data_prepared.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        bars_15m['RSI'] = rsi(bars_15m, period=self.rsi_period)
        #logger.info("Finished calculating 15-minute RSI.")

        # Forward-fill RSI to 1-minute bars
        #logger.info("Aligning 15-minute RSI with 1-minute data...")
        self.data_prepared['RSI'] = bars_15m['RSI'].reindex(self.data_prepared.index, method='ffill')
        #logger.info("Finished aligning RSI.")

        # Optional: Fill initial NaNs if any
        self.data_prepared['RSI'].fillna(method='bfill', inplace=True)

    def get_unrealized_pnl(self, current_price):
        """Calculates unrealized P&L based on the current price and open position."""
        if self.position == 'long':
            return (current_price - self.entry_price) * self.position_size * self.contract_multiplier
        elif self.position == 'short':
            return (self.entry_price - current_price) * self.position_size * self.contract_multiplier
        else:
            return 0

    def run_backtest(self):
        """
        Main loop: Iterate over each 1-minute bar using precomputed RSI and perform trading logic.
        """
        #logger.info("Starting backtest execution...")

        # Extract necessary columns as NumPy arrays for faster access
        times = self.data_prepared.index.to_numpy()
        closes = self.data_prepared['close'].to_numpy()
        highs = self.data_prepared['high'].to_numpy()
        lows = self.data_prepared['low'].to_numpy()
        vwaps = self.data_prepared['vwap'].to_numpy()
        is_rths = self.data_prepared['is_rth'].astype(bool).to_numpy()
        rsis = self.data_prepared['RSI'].to_numpy()

        for i in range(len(self.data_prepared)):
            current_idx = times[i]
            price = closes[i]
            high = highs[i]
            low = lows[i]
            vwap = vwaps[i]
            is_rth = is_rths[i]
            rsi_value = rsis[i]

            # Trading logic
            if not np.isnan(rsi_value):
                if self.position is None:
                    if is_rth:
                        if price > vwap and rsi_value > 70:
                            # Enter Long
                            self.enter_position('long', price, current_idx)
                        elif price < vwap and rsi_value < 30:
                            # Enter Short
                            self.enter_position('short', price, current_idx)
                else:
                    # Check for exit signals
                    exit_info = self.check_exit_conditions(price, high, low, current_idx)
                    if exit_info:
                        self.exit_position(exit_info, current_idx)

            # Update equity curve
            unrealized_pnl = self.get_unrealized_pnl(price)
            total_equity = self.equity + unrealized_pnl
            self.equity_curve.append({
                'Time': current_idx,
                'Equity': total_equity
            })

            # Update drawdown
            self.update_drawdown(total_equity, current_idx)

        #logger.info("Backtest execution completed.")

    def enter_position(self, position_type, price, current_idx):
        """Handles entering a position."""
        self.position = position_type
        if position_type == 'long':
            self.entry_price = price + self.slippage
            self.stop_loss = self.entry_price - self.stop_loss_points
            self.take_profit = self.entry_price + self.take_profit_points
        else:  # 'short'
            self.entry_price = price - self.slippage
            self.stop_loss = self.entry_price + self.stop_loss_points
            self.take_profit = self.entry_price - self.take_profit_points

        self.cash -= self.commission
        self.equity -= self.commission  # Commission impacts equity immediately
        self.trade_log.append({
            'Type': position_type.capitalize(),
            'Entry Time': current_idx,
            'Entry Price': self.entry_price,
            'Exit Time': None,
            'Exit Price': None,
            'Result': None,
            'Profit': 0
        })
        self.exposed_bars += 1
        logger.debug(f"{position_type.capitalize()} entered at {self.entry_price} on {current_idx}")

    def check_exit_conditions(self, price, high, low, current_idx):
        """Checks if exit conditions are met for the current position."""
        if self.position == 'long':
            # Check if both stop loss and take profit are hit within the bar
            if low <= self.stop_loss and high >= self.take_profit:
                # Determine which was hit first
                if (self.take_profit - self.entry_price) < (self.entry_price - self.stop_loss):
                    return {'result': 'Take Profit', 'price': self.take_profit}
                else:
                    return {'result': 'Stop Loss', 'price': self.stop_loss}
            elif high >= self.take_profit:
                return {'result': 'Take Profit', 'price': self.take_profit}
            elif low <= self.stop_loss:
                return {'result': 'Stop Loss', 'price': self.stop_loss}
        elif self.position == 'short':
            if high >= self.stop_loss and low <= self.take_profit:
                if (self.entry_price - self.take_profit) < (self.stop_loss - self.entry_price):
                    return {'result': 'Take Profit', 'price': self.take_profit}
                else:
                    return {'result': 'Stop Loss', 'price': self.stop_loss}
            elif low <= self.take_profit:
                return {'result': 'Take Profit', 'price': self.take_profit}
            elif high >= self.stop_loss:
                return {'result': 'Stop Loss', 'price': self.stop_loss}
        return None

    def exit_position(self, exit_info, current_idx):
        """Handles exiting a position."""
        result = exit_info['result']
        exit_price = exit_info['price']

        # Calculate PnL
        if self.position == 'long':
            pnl = (exit_price - self.entry_price) * self.position_size * self.contract_multiplier
        else:  # 'short'
            pnl = (self.entry_price - exit_price) * self.position_size * self.contract_multiplier

        pnl -= self.commission  # Exit commission
        self.cash += pnl
        self.equity += pnl

        # Update trade log
        self.trade_log[-1].update({
            'Exit Time': current_idx,
            'Exit Price': exit_price,
            'Result': result,
            'Profit': pnl
        })

        logger.debug(f"{self.position.capitalize()} exited with {result} at {exit_price} on {current_idx} PROFIT: {pnl}")

        # Reset position
        self.position = None
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0

    def update_drawdown(self, current_total_equity, current_idx):
        """Updates drawdown metrics."""
        if current_total_equity > self.equity_peak:
            self.equity_peak = current_total_equity
            if self.in_drawdown:
                drawdown_duration = (current_idx - self.drawdown_start).total_seconds() / 86400
                self.drawdown_durations.append(drawdown_duration)
                self.in_drawdown = False
                self.current_drawdown = 0
        else:
            dd = (self.equity_peak - current_total_equity) / self.equity_peak * 100
            if dd > self.current_drawdown:
                self.current_drawdown = dd
                if not self.in_drawdown:
                    self.in_drawdown = True
                    self.drawdown_start = current_idx
            # Optionally track further drawdown details here

    def get_trade_results(self):
        """Returns trade results as a DataFrame."""
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log)

    def analyze_results(self):
        """Analyzes and prints the backtest results."""
        if not self.trade_log:
            print("No trades were executed.")
            return

        trade_results = self.get_trade_results()
        equity_df = pd.DataFrame(self.equity_curve).set_index('Time')

        start_date = equity_df.index.min().strftime("%Y-%m-%d")
        end_date = equity_df.index.max().strftime("%Y-%m-%d")
        exposure_time_percentage = (self.exposed_bars / self.total_bars) * 100
        final_account_balance = self.equity
        equity_peak = self.equity_peak
        total_return_percentage = ((final_account_balance - self.initial_capital) / self.initial_capital) * 100

        # Annualized return
        delta = equity_df.index.max() - equity_df.index.min()
        years = delta.days / 365.25 if delta.days > 0 else 0
        if years > 0:
            annualized_return_percentage = ((final_account_balance / self.initial_capital) ** (1 / years) - 1) * 100
        else:
            annualized_return_percentage = 0

        # Benchmark (Buy & Hold) on the 1-min data
        benchmark_start_price = self.data_prepared['close'].iloc[0]
        benchmark_end_price = self.data_prepared['close'].iloc[-1]
        benchmark_return = ((benchmark_end_price - benchmark_start_price) / benchmark_start_price) * 100

        # Daily returns for Sharpe/Sortino
        equity_daily = equity_df['Equity'].resample('D').last().dropna()
        daily_returns = equity_daily.pct_change().dropna()

        volatility_annual = daily_returns.std() * np.sqrt(252) * 100
        risk_free_rate = 0
        if daily_returns.std() != 0:
            sharpe_ratio = ((daily_returns.mean() - risk_free_rate) / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        downside_returns = daily_returns[daily_returns < 0]
        if not downside_returns.empty:
            sortino_ratio = (daily_returns.mean() * np.sqrt(252)) / (downside_returns.std() * np.sqrt(252))
        else:
            sortino_ratio = np.inf

        # Drawdowns
        running_max = equity_df['Equity'].cummax()
        drawdowns_percentage = (equity_df['Equity'] - running_max) / running_max * 100
        max_drawdown_percentage = drawdowns_percentage.min()
        if not drawdowns_percentage[drawdowns_percentage < 0].empty:
            average_drawdown_percentage = drawdowns_percentage[drawdowns_percentage < 0].mean()
        else:
            average_drawdown_percentage = 0

        # Dollar drawdowns
        drawdowns_dollar = running_max - equity_df['Equity']
        max_drawdown_dollar = drawdowns_dollar.max()
        if (drawdowns_dollar > 0).any():
            average_drawdown_dollar = drawdowns_dollar[drawdowns_dollar > 0].mean()
        else:
            average_drawdown_dollar = 0

        # Profit factor
        winning_trades = trade_results[trade_results['Result'] == 'Take Profit']['Profit']
        losing_trades = trade_results[trade_results['Result'] == 'Stop Loss']['Profit']
        if not losing_trades.empty and losing_trades.sum() != 0:
            profit_factor = winning_trades.sum() / abs(losing_trades.sum())
        else:
            profit_factor = np.inf

        # Calmar ratio
        if abs(max_drawdown_percentage) != 0:
            calmar_ratio = (total_return_percentage / abs(max_drawdown_percentage))
        else:
            calmar_ratio = np.inf

        # Drawdown durations
        max_drawdown_duration_days = max(self.drawdown_durations) if self.drawdown_durations else 0
        average_drawdown_duration_days = np.mean(self.drawdown_durations) if self.drawdown_durations else 0

        # Win rate
        total_trades = len(trade_results)
        if total_trades > 0:
            win_rate = (len(winning_trades) / total_trades) * 100
        else:
            win_rate = 0.0

        # Gather results
        results = {
            "Start Date": start_date,
            "End Date": end_date,
            "Exposure Time (%)": exposure_time_percentage,
            "Final Account Balance": final_account_balance,
            "Equity Peak": equity_peak,
            "Total Return (%)": total_return_percentage,
            "Annualized Return (%)": annualized_return_percentage,
            "Benchmark Return (%)": benchmark_return,
            "Volatility (Annual %)": volatility_annual,
            "Total Trades": len(trade_results),
            "Winning Trades": len(winning_trades),
            "Losing Trades": len(losing_trades),
            "Win Rate (%)": win_rate,
            "Profit Factor": profit_factor,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Calmar Ratio": calmar_ratio,
            "Max Drawdown (%)": max_drawdown_percentage,
            "Average Drawdown (%)": average_drawdown_percentage,
            "Max Drawdown ($)": max_drawdown_dollar,
            "Average Drawdown ($)": average_drawdown_dollar,
            "Max Drawdown Duration (days)": max_drawdown_duration_days,
            "Average Drawdown Duration (days)": average_drawdown_duration_days,
        }

        # Print performance summary
        performance_summary = "\nPerformance Summary:\n"
        for key, value in results.items():
            if isinstance(value, float):
                performance_summary += f"{key:35}: {value:>15.2f}\n"
            elif isinstance(value, int):
                performance_summary += f"{key:35}: {value:>15}\n"
            else:
                performance_summary += f"{key:35}: {value:>15}\n"
        #logger.info(performance_summary)

        # Return results for optimization
        return results

    def analyze_results_plot(self):
        """Analyzes and plots the backtest results."""
        if not self.trade_log:
            print("No trades were executed.")
            return

        trade_results = self.get_trade_results()
        equity_df = pd.DataFrame(self.equity_curve).set_index('Time')

        # Plot Equity Curve vs Benchmark
        initial_close = self.data_prepared['close'].iloc[0]
        benchmark_equity = (self.data_prepared['close'] / initial_close) * self.initial_capital
        benchmark_equity = benchmark_equity.reindex(equity_df.index, method='ffill').fillna(method='ffill')

        plt.figure(figsize=(14, 7))
        plt.plot(equity_df['Equity'], label='Strategy Equity')
        plt.plot(benchmark_equity, label='Benchmark Equity (Buy & Hold)', alpha=0.7)
        plt.title('Equity Curve: Strategy vs Benchmark')
        plt.xlabel('Time')
        plt.ylabel('Account Balance ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot Price and VWAP (from 1-min data)
        plt.figure(figsize=(14, 7))
        plt.plot(self.data_prepared.index, self.data_prepared['close'], label='Close Price')
        plt.plot(self.data_prepared.index, self.data_prepared['vwap'], label='VWAP', linestyle='--')
        plt.title('Price vs 1-min VWAP')
        plt.xlabel('Time')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot RSI (15-min) Over Full Date Range
        bars_15m_final = self.data_prepared.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        bars_15m_final['RSI'] = rsi(bars_15m_final, period=self.rsi_period)

        plt.figure(figsize=(14, 4))
        plt.plot(bars_15m_final.index, bars_15m_final['RSI'], label='15-min RSI', color='orange')
        plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        plt.title('RSI (15-min) Over Full Date Range')
        plt.xlabel('Time')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Optionally save trade results or equity curve:
        # trade_results.to_csv('trade_results.csv', index=False)
        # equity_df.to_csv('equity_curve.csv')

# --- Optimization Function ---
def run_backtest(params):
    """
    Function to run backtest with a given set of parameters.
    This is designed to be used with multiprocessing.
    """
    stop_loss, take_profit = params
    backtester = MESFuturesBacktest(
        data_path=DATA_PATH,
        start_date=START_DATE,
        end_date=END_DATE,
        timeframe=TIMEFRAME,  # 1-min bars for main loop
        initial_capital=INITIAL_CAPITAL,
        position_size=POSITION_SIZE,
        contract_multiplier=CONTRACT_MULTIPLIER,
        stop_loss_points=stop_loss,
        take_profit_points=take_profit,
        commission=COMMISSION,
        slippage=SLIPPAGE,
        rsi_period=14  # Fixed RSI period
    )
    backtester.run_backtest()
    results = backtester.analyze_results()
    # Include parameters in the results
    results['Stop Loss Points'] = stop_loss
    results['Take Profit Points'] = take_profit
    # No RSI Period since it's fixed
    return results

def optimize_parameters():
    """
    Optimizes the trading strategy by testing different combinations of parameters.
    """
    logger.info("Starting optimization process...")

    # Create parameter grid (only stop_loss and take_profit)
    parameter_grid = list(product(STOP_LOSS_POINTS_RANGE, TAKE_PROFIT_POINTS_RANGE))
    total_combinations = len(parameter_grid)
    logger.info(f"Total parameter combinations to test: {total_combinations}")

    # To store optimization results
    optimization_results = []

    # Utilize multiprocessing for faster optimization
    pool = Pool(processes=cpu_count())

    try:
        for i, result in enumerate(pool.imap(run_backtest, parameter_grid), 1):
            optimization_results.append(result)
            logger.info(f"Completed {i}/{total_combinations}: Stop Loss={result['Stop Loss Points']}, "
                        f"Take Profit={result['Take Profit Points']}, "
                        f"Sharpe Ratio={result['Sharpe Ratio']:.2f}")
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user.")
        pool.terminate()
        pool.join()
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during optimization: {e}")
        pool.terminate()
        pool.join()
        sys.exit(1)
    finally:
        pool.close()
        pool.join()

    # Convert results to DataFrame
    results_df = pd.DataFrame(optimization_results)

    # Save optimization results to CSV
    results_df.to_csv('optimization_results.csv', index=False)
    logger.info("Optimization results saved to 'optimization_results.csv'.")

    # Analyze optimization results
    # Find the parameter set with the highest Sharpe Ratio
    if results_df['Sharpe Ratio'].isnull().all():
        logger.error("All backtests resulted in NaN Sharpe Ratios. Check your strategy and data.")
        sys.exit(1)
    best_result = results_df.loc[results_df['Sharpe Ratio'].idxmax()]
    logger.info("\nBest Parameter Set Based on Sharpe Ratio:")
    logger.info(f"Stop Loss Points: {best_result['Stop Loss Points']}")
    logger.info(f"Take Profit Points: {best_result['Take Profit Points']}")
    logger.info(f"Sharpe Ratio: {best_result['Sharpe Ratio']:.2f}")
    logger.info(f"Total Return (%): {best_result['Total Return (%)']:.2f}%")
    logger.info(f"Profit Factor: {best_result['Profit Factor']:.2f}")

    # Optionally, plot the top N results based on Sharpe Ratio
    top_n = 5
    top_results = results_df.sort_values(by='Sharpe Ratio', ascending=False).head(top_n)

    # Plotting can be added here if desired

    return results_df, best_result

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file {DATA_PATH} does not exist.")
        sys.exit(1)
    else:
        try:
            # Run optimization
            optimization_df, best_params = optimize_parameters()

            # Optionally, rerun backtest with best parameters for detailed analysis
            logger.info("Running backtest with the best parameter set for detailed analysis...")
            best_backtester = MESFuturesBacktest(
                data_path=DATA_PATH,
                start_date=START_DATE,
                end_date=END_DATE,
                timeframe=TIMEFRAME,  # 1-min bars for main loop
                initial_capital=INITIAL_CAPITAL,
                position_size=POSITION_SIZE,
                contract_multiplier=CONTRACT_MULTIPLIER,
                stop_loss_points=best_params['Stop Loss Points'],
                take_profit_points=best_params['Take Profit Points'],
                commission=COMMISSION,
                slippage=SLIPPAGE,
                rsi_period=14  # Fixed RSI period
            )
            best_backtester.run_backtest()
            best_backtester.analyze_results()
            # best_backtester.analyze_results_plot()  # Uncomment to plot

        except Exception as e:
            logger.error(f"An error occurred during optimization: {e}")
            sys.exit(1)