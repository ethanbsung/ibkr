import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, time, timedelta
import pytz
import logging
import sys

# --- Configuration Parameters ---
DATA_PATH = 'Data/es_1m_data.csv'  # Path to your 1-minute data CSV
INITIAL_CAPITAL = 5000             # Starting cash in USD
POSITION_SIZE = 1                  # Number of contracts per trade
CONTRACT_MULTIPLIER = 5            # Contract multiplier for MES

TIMEFRAME = '1min'                 # Keep original 1-minute timeframe for the main data
STOP_LOSS_POINTS = 4               # Stop loss in points
TAKE_PROFIT_POINTS = 18            # Take profit in points

COMMISSION = 0.62                  # Commission per trade (entry or exit)
SLIPPAGE = 1                       # Slippage in points on entry

# Custom Backtest Dates (inclusive)
START_DATE = '2024-12-01'          # Format: 'YYYY-MM-DD'
END_DATE = '2024-12-25'            # Format: 'YYYY-MM-DD'

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbosity
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

        logger.info(f"Loaded '{csv_file}' with columns: {df.columns.tolist()}")

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
            logger.info(f"Remaining data points: {len(df)}")

        # Add missing columns
        df['average'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['barCount'] = 1  # Assume each row is a single bar
        df['contract'] = df['contract'].astype(str)

        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'contract', 'pct_chg', 'average', 'barCount']
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
        
        We keep the 1-minute data in self.data_prepared for the main loop,
        but compute a 15-minute RSI on the fly inside run_backtest.
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
        self.rsi_period = rsi_period  # We'll compute RSI(14) on 15-min bars.

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
        logger.info("Loading data...")
        self.data = load_data(self.data_path)
        logger.info("Data loaded successfully.")

    def prepare_data(self):
        """
        Prepares 1-minute data: mark RTH, filter by date, resample to 1-min (no-op if already 1-min),
        and calculate VWAP so that we can read it in the loop.
        """
        logger.info("Preparing data for backtest...")

        # Mark RTH
        self.data = mark_rth(self.data)
        logger.info("Marked Regular Trading Hours in the data.")

        # Filter date range
        self.data = self.data[(self.data.index >= self.start_date) & (self.data.index <= self.end_date)]
        logger.info(f"Filtered data from {self.start_date.date()} to {self.end_date.date()}. Points: {len(self.data)}")

        # Sort
        self.data.sort_index(inplace=True)

        # Resample to the chosen timeframe (likely '1min', so this is effectively a no-op if data is 1min)
        logger.info(f"Resampling data to {self.timeframe} bars if needed...")
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
        logger.info(f"Resampled data: {len(self.data_prepared)} bars.")

        # Compute VWAP on the 1-min data (we keep VWAP updated every minute)
        logger.info("Calculating 1-minute VWAP (reset daily)...")
        self.data_prepared['vwap'] = calculate_vwap(self.data_prepared)
        logger.info("Finished calculating VWAP.")

        # We do NOT compute RSI here, because we want a 15-minute RSI computed on-the-fly.
        # The user wants an RSI that is effectively on 15-minute bars, updated each 1-minute step.

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
        Main loop: 
         - For each 1-minute bar, we resample all data up to that bar to 15 minutes,
           compute RSI(14) on those 15-min bars, and get the latest RSI value.
         - Then we trade based on that RSI and our usual logic.
        """
        logger.info("Starting backtest execution...")

        # We'll iterate over each 1-minute bar in chronological order
        for current_idx, row in self.data_prepared.iterrows():
            price = row['close']
            vwap = row['vwap']
            is_rth = row['is_rth']

            # 1) Resample up to the current index in 15-minute bars
            #    This effectively simulates "live" building of 15-min bars.
            up_to_now = self.data_prepared.loc[:current_idx]
            bars_15m = up_to_now.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            # 2) Compute RSI(14) on those 15-min bars
            bars_15m['RSI'] = rsi(bars_15m, period=self.rsi_period)

            # 3) Extract the latest 15-minute RSI value
            if len(bars_15m) < 1 or pd.isna(bars_15m['RSI'].iloc[-1]):
                # Not enough data to compute RSI, skip entry logic
                rsi_value = None
            else:
                rsi_value = bars_15m['RSI'].iloc[-1]

            # Proceed with trading logic if we have an RSI reading
            if rsi_value is not None:
                # Check if we are in a position
                if self.position is None:
                    # Only consider entering during RTH
                    if is_rth:
                        # Check for a Long Entry
                        if price > vwap and rsi_value > 70:
                            self.position = 'long'
                            self.entry_price = price + self.slippage  # add slippage
                            self.stop_loss = self.entry_price - self.stop_loss_points
                            self.take_profit = self.entry_price + self.take_profit_points
                            self.cash -= self.commission  # entry commission
                            self.trade_log.append({
                                'Type': 'Long',
                                'Entry Time': current_idx,
                                'Entry Price': self.entry_price,
                                'Exit Time': None,
                                'Exit Price': None,
                                'Result': None,
                                'Profit': 0
                            })
                            self.exposed_bars += 1
                        # Check for a Short Entry
                        elif price < vwap and rsi_value < 30:
                            self.position = 'short'
                            self.entry_price = price - self.slippage  # slippage
                            self.stop_loss = self.entry_price + self.stop_loss_points
                            self.take_profit = self.entry_price - self.take_profit_points
                            self.cash -= self.commission
                            self.trade_log.append({
                                'Type': 'Short',
                                'Entry Time': current_idx,
                                'Entry Price': self.entry_price,
                                'Exit Time': None,
                                'Exit Price': None,
                                'Result': None,
                                'Profit': 0
                            })
                            self.exposed_bars += 1
                else:
                    # Already in a position, check for exits
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
                    else:  # self.position == 'short'
                        if price >= self.stop_loss:
                            exit_signal = True
                            result = 'Stop Loss'
                            exit_price = self.stop_loss
                        elif price <= self.take_profit:
                            exit_signal = True
                            result = 'Take Profit'
                            exit_price = self.take_profit

                    if exit_signal:
                        # Calculate realized PnL
                        if self.position == 'long':
                            pnl = (exit_price - self.entry_price) * self.position_size * self.contract_multiplier
                        else:  # short
                            pnl = (self.entry_price - exit_price) * self.position_size * self.contract_multiplier

                        pnl -= self.commission  # exit commission
                        self.cash += pnl
                        self.equity += pnl

                        # Update trade log
                        self.trade_log[-1].update({
                            'Exit Time': current_idx,
                            'Exit Price': exit_price,
                            'Result': result,
                            'Profit': pnl
                        })

                        # Reset position
                        self.position = None
                        self.entry_price = 0
                        self.stop_loss = 0
                        self.take_profit = 0
                    else:
                        self.exposed_bars += 1
            else:
                # No RSI -> skip entry logic
                if self.position is not None:
                    # If we are in a position, we do not exit (no RSI signal).
                    # You could optionally do time-based exit here, if desired.
                    self.exposed_bars += 1

            # Update the equity curve with the unrealized PnL
            self.equity_curve.append({
                'Time': current_idx,
                'Equity': self.equity + self.get_unrealized_pnl(price)
            })

            # Track new peak or update drawdown
            current_total_equity = self.equity + self.get_unrealized_pnl(price)
            if current_total_equity > self.equity_peak:
                self.equity_peak = current_total_equity
                if self.in_drawdown:
                    drawdown_duration = (current_idx - self.drawdown_start).total_seconds() / 86400
                    self.drawdown_durations.append(drawdown_duration)
                    self.in_drawdown = False
                    self.current_drawdown = 0

            # Current drawdown from peak
            dd = (self.equity_peak - current_total_equity) / self.equity_peak * 100
            if dd > self.current_drawdown:
                self.current_drawdown = dd
                if not self.in_drawdown:
                    self.in_drawdown = True
                    self.drawdown_start = current_idx
            if self.in_drawdown and dd < self.current_drawdown:
                # continuing drawdown, do nothing
                pass
            elif self.in_drawdown and dd >= self.current_drawdown:
                # recovering from drawdown
                self.drawdowns.append(self.current_drawdown)
                self.current_drawdown = 0
                drawdown_duration = (current_idx - self.drawdown_start).total_seconds() / 86400
                self.drawdown_durations.append(drawdown_duration)

    def analyze_results(self):
        """Analyzes and prints the backtest results."""
        if not self.trade_log:
            print("No trades were executed.")
            return

        trade_results = pd.DataFrame(self.trade_log)
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

        # Print performance summary
        print("\nPerformance Summary:")
        for key, value in results.items():
            print(f"{key:30}: {value:>15}")

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

        # Because RSI is computed on 15-min bars (on the fly), we don't have an RSI column in data_prepared.
        # If you'd like to store those RSI values for plotting, you'd need to do so in the loop.
        # Or you can re-run a 15-min aggregator for the entire date range *after* backtest and plot RSI.
        # For demonstration, let's just show how you'd plot it if you re-aggregate afterwards:

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

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file {DATA_PATH} does not exist.")
        sys.exit(1)
    else:
        try:
            backtester = MESFuturesBacktest(
                data_path=DATA_PATH,
                start_date=START_DATE,
                end_date=END_DATE,
                timeframe=TIMEFRAME,  # 1-min bars for main loop
                initial_capital=INITIAL_CAPITAL,
                position_size=POSITION_SIZE,
                contract_multiplier=CONTRACT_MULTIPLIER,
                stop_loss_points=STOP_LOSS_POINTS,
                take_profit_points=TAKE_PROFIT_POINTS,
                commission=COMMISSION,
                slippage=SLIPPAGE,
                rsi_period=14  # RSI period on the 15-minute bars
            )
            backtester.run_backtest()
            backtester.analyze_results()
        except Exception as e:
            logger.error(f"An error occurred during backtesting: {e}")
            sys.exit(1)