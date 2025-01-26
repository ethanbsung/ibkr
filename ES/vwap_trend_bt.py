import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import pytz
import logging
import sys

# --- Configuration Parameters ---
DATA_PATH = 'Data/es_1m_data.csv'  # Path to your 1-minute data CSV
INITIAL_CAPITAL = 5000             # Starting cash in USD
POSITION_SIZE = 1                  # Number of contracts per trade
CONTRACT_MULTIPLIER = 5            # Contract multiplier for MES

TIMEFRAME = '1T'                    # 1-minute timeframe

STOP_LOSS_POINTS = 4                # Stop loss in points
TAKE_PROFIT_POINTS = 18             # Take profit in points

COMMISSION = 0.62                   # Commission per trade (entry or exit)
SLIPPAGE = 0.5                      # Slippage in points on entry

# Custom Backtest Dates (inclusive)
START_DATE = '2021-01-15'           # Format: 'YYYY-MM-DD'
END_DATE = '2024-12-30'             # Format: 'YYYY-MM-DD'

# --- Setup Logging ---
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more verbosity
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
    df = df.between_time('09:30', '16:00')  # Changed to include 16:00

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

def rsi(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI)."""
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

# --- Backtest Class ---
class MESFuturesBacktest:
    def __init__(self, data_path, start_date, end_date, timeframe, initial_capital,
                 position_size, contract_multiplier, stop_loss_points, take_profit_points,
                 commission, slippage, rsi_period=210):
        """
        Initializes the backtest.

        Parameters:
            data_path (str): Path to the CSV data file.
            start_date (str): Start date for the backtest in 'YYYY-MM-DD' format.
            end_date (str): End date for the backtest in 'YYYY-MM-DD' format.
            timeframe (str): Resampling timeframe for the strategy (e.g., '15T' for 15 minutes).
            initial_capital (float): Starting capital for the backtest.
            position_size (int): Number of contracts per trade.
            contract_multiplier (int): Contract multiplier for MES.
            stop_loss_points (int): Stop loss in points.
            take_profit_points (int): Take profit in points.
            commission (float): Commission per trade.
            slippage (float): Slippage in points on entry.
            rsi_period (int): Period for RSI calculation based on 1-minute bars.
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
        self.rsi_period = rsi_period  # Set RSI period to 210

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
        self.total_bars = len(self.data_1m)
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
        """Prepares data by filtering RTH, calculating indicators, and applying date filters."""
        logger.info("Preparing data for backtest...")

        # Filter Regular Trading Hours
        self.data_rth = filter_rth(self.data)
        logger.info(f"Filtered data to Regular Trading Hours. Total data points: {len(self.data_rth)}")

        # Filter by Start and End Dates
        self.data_rth = self.data_rth[(self.data_rth.index >= self.start_date) & (self.data_rth.index <= self.end_date)]
        logger.info(f"Filtered data by date from {self.start_date.date()} to {self.end_date.date()}. Total data points: {len(self.data_rth)}")

        # Ensure data is sorted
        self.data_rth.sort_index(inplace=True)

        # Calculate VWAP per day using Typical Price
        self.data_rth['date'] = self.data_rth.index.date
        self.data_rth['typical_price'] = (self.data_rth['high'] + self.data_rth['low'] + self.data_rth['close']) / 3
        self.data_rth['cum_typical_price_volume'] = self.data_rth['typical_price'] * self.data_rth['volume']
        self.data_rth['cum_typical_price_volume'] = self.data_rth.groupby('date')['cum_typical_price_volume'].cumsum()
        self.data_rth['cum_volume'] = self.data_rth.groupby('date')['volume'].cumsum()
        self.data_rth['vwap'] = self.data_rth['cum_typical_price_volume'] / self.data_rth['cum_volume']

        # Calculate RSI using the last 210 1-minute bars
        self.data_rth['rsi'] = rsi(self.data_rth, period=self.rsi_period)
        # Drop initial rows with NaN RSI
        initial_rsi_nans = self.data_rth['rsi'].isna().sum()
        if initial_rsi_nans > 0:
            logger.info(f"Dropping first {initial_rsi_nans} bars due to NaN RSI.")
            self.data_rth = self.data_rth.dropna(subset=['rsi'])
            logger.info(f"Remaining data points after dropping NaN RSI: {len(self.data_rth)}")

        # Assign to data_1m for clarity
        self.data_1m = self.data_rth.copy()
        logger.info("Calculated VWAP and RSI.")

    def run_backtest(self):
        """Executes the backtest by iterating over each 1-minute bar."""
        logger.info("Starting backtest execution...")
        for idx, row in self.data_1m.iterrows():
            price = row['close']
            vwap = row['vwap']
            rsi_value = row['rsi']

            # Check if we are in a position
            if self.position is None:
                # Check for Long Entry
                if price > vwap and rsi_value > 70:
                    self.position = 'long'
                    # Apply slippage: buy at price + slippage
                    self.entry_price = price + self.slippage
                    self.stop_loss = self.entry_price - self.stop_loss_points
                    self.take_profit = self.entry_price + self.take_profit_points
                    self.cash -= self.commission  # Deduct commission for entry
                    self.trade_log.append({
                        'Type': 'Long',
                        'Entry Time': idx,
                        'Entry Price': self.entry_price,
                        'Exit Time': None,
                        'Exit Price': None,
                        'Result': None,
                        'Profit': 0
                    })
                    logger.debug(f"Entered LONG at {self.entry_price} on {idx}")
                    self.exposed_bars += 1
                # Check for Short Entry
                elif price < vwap and rsi_value < 30:
                    self.position = 'short'
                    # Apply slippage: sell at price - slippage
                    self.entry_price = price - self.slippage
                    self.stop_loss = self.entry_price + self.stop_loss_points
                    self.take_profit = self.entry_price - self.take_profit_points
                    self.cash -= self.commission  # Deduct commission for entry
                    self.trade_log.append({
                        'Type': 'Short',
                        'Entry Time': idx,
                        'Entry Price': self.entry_price,
                        'Exit Time': None,
                        'Exit Price': None,
                        'Result': None,
                        'Profit': 0
                    })
                    logger.debug(f"Entered SHORT at {self.entry_price} on {idx}")
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

                    pnl -= self.commission  # Deduct commission for exit

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

            # Update Equity Curve with Unrealized P&L
            self.equity_curve.append({'Time': idx, 'Equity': self.equity + self.get_unrealized_pnl(price)})

            # Update Equity Peak
            current_total_equity = self.equity + self.get_unrealized_pnl(price)
            if current_total_equity > self.equity_peak:
                self.equity_peak = current_total_equity
                if self.in_drawdown:
                    drawdown_duration = (idx - self.drawdown_start).total_seconds() / 86400  # in days
                    self.drawdown_durations.append(drawdown_duration)
                    self.in_drawdown = False
                    self.current_drawdown = 0

            # Calculate Drawdown
            current_drawdown = (self.equity_peak - current_total_equity) / self.equity_peak * 100
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
                drawdown_duration = (idx - self.drawdown_start).total_seconds() / 86400  # in days
                self.drawdown_durations.append(drawdown_duration)

        # Close any open position at the end of the backtest
        if self.position is not None:
            exit_price = self.data_1m['close'].iloc[-1]
            idx = self.data_1m.index[-1]
            if self.position == 'long':
                pnl = (exit_price - self.entry_price) * self.position_size * self.contract_multiplier
            elif self.position == 'short':
                pnl = (self.entry_price - exit_price) * self.position_size * self.contract_multiplier
            pnl -= self.commission  # Deduct commission for exit
            self.cash += pnl
            self.equity += pnl
            self.trade_log[-1].update({
                'Exit Time': idx,
                'Exit Price': exit_price,
                'Result': 'End of Data',
                'Profit': pnl
            })
            logger.debug(f"Exited {self.position.upper()} at {exit_price} on {idx} via End of Data for P&L: ${pnl:.2f}")
            self.equity_curve.append({'Time': idx, 'Equity': self.equity})

    def get_unrealized_pnl(self, current_price):
        """Calculates unrealized P&L based on current price and open position."""
        if self.position == 'long':
            return (current_price - self.entry_price) * self.position_size * self.contract_multiplier
        elif self.position == 'short':
            return (self.entry_price - current_price) * self.position_size * self.contract_multiplier
        else:
            return 0

    def analyze_results(self):
        """Analyzes and prints the backtest results."""
        if not self.trade_log:
            print("No trades were executed.")
            return

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
        benchmark_start_price = self.data_1m['close'].iloc[0]
        benchmark_end_price = self.data_1m['close'].iloc[-1]
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
        sortino_ratio = (daily_returns.mean() * np.sqrt(252)) / (downside_returns.std() * np.sqrt(252)) if not downside_returns.empty else np.inf

        # Drawdown Calculations
        running_max = equity_df['Equity'].cummax()
        drawdowns_percentage = (equity_df['Equity'] - running_max) / running_max * 100
        max_drawdown_percentage = drawdowns_percentage.min()
        average_drawdown_percentage = drawdowns_percentage[drawdowns_percentage < 0].mean() if not drawdowns_percentage[drawdowns_percentage < 0].empty else 0

        # Dollar Drawdown Calculations
        drawdowns_dollar = running_max - equity_df['Equity']
        max_drawdown_dollar = drawdowns_dollar.max()
        average_drawdown_dollar = drawdowns_dollar[drawdowns_dollar > 0].mean() if not drawdowns_dollar[drawdowns_dollar > 0].empty else 0

        # Profit Factor
        winning_trades = trade_results[trade_results['Result'] == 'Take Profit']['Profit']
        losing_trades = trade_results[trade_results['Result'] == 'Stop Loss']['Profit']
        profit_factor = winning_trades.sum() / abs(losing_trades.sum()) if not losing_trades.empty else np.inf

        # Calmar Ratio
        calmar_ratio = (total_return_percentage / abs(max_drawdown_percentage)) if abs(max_drawdown_percentage) != 0 else np.inf

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
            "Exposure Time": f"{exposure_time_percentage:.2f}%",
            "Final Account Balance": f"${final_account_balance:,.2f}",
            "Equity Peak": f"${equity_peak:,.2f}",
            "Total Return": f"{total_return_percentage:.2f}%",
            "Annualized Return": f"{annualized_return_percentage:.2f}%",
            "Benchmark Return": f"{benchmark_return:.2f}%",
            "Volatility (Annual)": f"{volatility_annual:.2f}%",
            "Total Trades": len(trade_results),
            "Winning Trades": len(winning_trades),
            "Losing Trades": len(losing_trades),
            "Win Rate": f"{win_rate:.2f}%",
            "Profit Factor": f"{profit_factor:.2f}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Sortino Ratio": f"{sortino_ratio:.2f}",
            "Calmar Ratio": f"{calmar_ratio:.2f}",
            "Max Drawdown (%)": f"{max_drawdown_percentage:.2f}%",
            "Average Drawdown (%)": f"{average_drawdown_percentage:.2f}%",
            "Max Drawdown ($)": f"${max_drawdown_dollar:,.2f}",
            "Average Drawdown ($)": f"${average_drawdown_dollar:,.2f}",
            "Max Drawdown Duration": f"{max_drawdown_duration_days:.2f} days",
            "Average Drawdown Duration": f"{average_drawdown_duration_days:.2f} days",
        }

        # Print Results Summary
        print("\nPerformance Summary:")
        for key, value in results.items():
            print(f"{key:30}: {value:>15}")

        # --- Plot Equity Curve vs Benchmark ---
        # Create benchmark equity curve (Buy & Hold)
        initial_close = self.data_1m['close'].iloc[0]
        benchmark_equity = (self.data_1m['close'] / initial_close) * self.initial_capital

        # Align the benchmark to the strategy's equity_curve
        benchmark_equity = benchmark_equity.reindex(equity_df.index, method='ffill')

        # Ensure no NaNs in benchmark_equity
        benchmark_equity = benchmark_equity.fillna(method='ffill')

        # Plotting Equity Curve and Benchmark
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

        # --- Plot Price and VWAP ---
        plt.figure(figsize=(14, 7))
        plt.plot(self.data_1m.index, self.data_1m['close'], label='Close Price')
        plt.plot(self.data_1m.index, self.data_1m['vwap'], label='VWAP', linestyle='--')
        plt.title('Price vs VWAP')
        plt.xlabel('Time')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Plot RSI ---
        plt.figure(figsize=(14, 4))
        plt.plot(self.data_1m.index, self.data_1m['rsi'], label='RSI', color='orange')
        plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        plt.title('Relative Strength Index (RSI)')
        plt.xlabel('Time')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Optional: Save Trade Results and Equity Curve
        # Uncomment the following lines if you wish to save trade results and equity curve
        # trade_results.to_csv('trade_results.csv', index=False)
        # logger.info("Trade results saved to 'trade_results.csv'.")

        # equity_df.to_csv('equity_curve.csv')
        # logger.info("Equity curve saved to 'equity_curve.csv'.")

    # --- Optional: Additional Methods for Advanced Metrics ---
    # You can add more methods here if needed for further analysis

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the data file exists
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file {DATA_PATH} does not exist. Please check the path and filename.")
        sys.exit(1)
    else:
        try:
            # Initialize Backtest
            backtester = MESFuturesBacktest(
                data_path=DATA_PATH,
                start_date=START_DATE,
                end_date=END_DATE,
                timeframe=TIMEFRAME,
                initial_capital=INITIAL_CAPITAL,
                position_size=POSITION_SIZE,
                contract_multiplier=CONTRACT_MULTIPLIER,
                stop_loss_points=STOP_LOSS_POINTS,
                take_profit_points=TAKE_PROFIT_POINTS,
                commission=COMMISSION,
                slippage=SLIPPAGE,
                rsi_period=210  # Set RSI period to 210 to match 14 periods of 15-minute bars
            )
            # Run Backtest
            backtester.run_backtest()
            # Analyze Results
            backtester.analyze_results()

            # Optional: Save Trade Results to CSV
            # Uncomment the following lines if you wish to save trade results for further analysis
            # trade_results = pd.DataFrame(backtester.trade_log)
            # trade_results.to_csv('trade_results.csv', index=False)
            # logger.info("Trade results saved to 'trade_results.csv'.")

            # Optional: Save Equity Curve to CSV
            # Uncomment the following lines if you wish to save the equity curve
            # equity_df = pd.DataFrame(backtester.equity_curve)
            # equity_df.to_csv('equity_curve.csv', index=False)
            # logger.info("Equity curve saved to 'equity_curve.csv'.")

        except Exception as e:
            logger.error(f"An error occurred during backtesting: {e}")
            sys.exit(1)