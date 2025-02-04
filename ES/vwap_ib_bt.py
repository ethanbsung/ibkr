import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, time, timedelta
import pytz
import logging
import sys
import time as tm  # Import the time module as 'tm' to avoid conflict with datetime.time

# Import IBKR API via ib_insync
from ib_insync import *

# --- Configuration Parameters ---

# Backtest settings
INITIAL_CAPITAL = 5000             # Starting cash in USD
POSITION_SIZE = 1                  # Number of contracts per trade
CONTRACT_MULTIPLIER = 5            # Contract multiplier for MES

TIMEFRAME = '5 secs'               # Use 5‑second bars for IBKR data request
RESAMPLE_FREQ = '5s'               # Pandas-compatible frequency for resampling

STOP_LOSS_POINTS = 9               # Stop loss in points
TAKE_PROFIT_POINTS = 10            # Take profit in points

COMMISSION = 0.62                  # Commission per trade (entry or exit)
SLIPPAGE = 0.25                    # Slippage in points on entry

# Custom Backtest Dates (inclusive)
START_DATE = '2024-10-01'          # Format: 'YYYY-MM-DD'
END_DATE = '2025-02-04'            # Format: 'YYYY-MM-DD'

# IBKR Connection Parameters
IB_HOST = '127.0.0.1'
IB_PORT = 4002
IB_CLIENT_ID = 1

# Define the MES futures contract.
# (Here we use the local symbol 'ESH5'; adjust parameters if needed.)
ES_CONTRACT = Future(localSymbol='ESH5', exchange='CME', currency='USD')

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more details
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# --- Historical Data Fetch Function ---
def fetch_ibkr_data(ib, contract, bar_size, start_time, end_time, useRTH=False):
    """
    Fetch historical data from IBKR using ib_insync in chunks.
    This function works by iterating backward in time from end_time to start_time,
    requesting data in chunks (whose duration depends on bar_size).
    
    Parameters:
      - ib: an active IB instance.
      - contract: an IBKR contract (e.g. Future with localSymbol 'ESH5')
      - bar_size: string for the bar size (e.g. '5 secs')
      - start_time: pd.Timestamp (tz-aware) for the start of data.
      - end_time: pd.Timestamp (tz-aware) for the end of data.
      - useRTH: Boolean; if True then only Regular Trading Hours are returned.
    
    Returns:
      A DataFrame containing the historical bars.
    """
    # Set maximum chunk duration based on bar size.
    if bar_size == '1 min':
        max_chunk = pd.Timedelta(days=7)  # For 1‑min bars, IBKR may allow about 7 days.
    elif bar_size == '30 mins':
        max_chunk = pd.Timedelta(days=365)
    elif bar_size == '5 secs':
        max_chunk = pd.Timedelta(days=1)  # IBKR typically limits sub‑minute bars to a shorter duration (here 1 day)
    else:
        max_chunk = pd.Timedelta(days=30)
    
    current_end = end_time
    all_bars = []
    while current_end > start_time:
        current_start = max(start_time, current_end - max_chunk)
        delta = current_end - current_start
        # For durations less than one day, request "1 D"
        if delta < pd.Timedelta(days=1):
            duration_str = "1 D"
        else:
            duration_days = delta.days
            duration_str = f"{duration_days} D"
        
        # Format endDateTime as required by IBKR (YYYYMMDD HH:MM:SS)
        end_dt_str = current_end.strftime("%Y%m%d %H:%M:%S")
        logger.info(f"Requesting {bar_size} data from {current_start} to {current_end} "
                    f"(Duration: {duration_str}) for contract {contract.localSymbol}")
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_dt_str,
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=useRTH,
                formatDate=1
            )
        except Exception as e:
            logger.error(f"Error requesting historical data: {e}")
            break

        if not bars:
            logger.warning("No bars returned for this chunk.")
            break

        # Convert bars to a DataFrame using ib_insync's util.df helper.
        df_chunk = util.df(bars)
        all_bars.append(df_chunk)
        # Update current_end to just before the earliest bar in this chunk
        earliest_bar_time = pd.to_datetime(df_chunk['date'].min())
        current_end = earliest_bar_time - pd.Timedelta(seconds=1)
        tm.sleep(1)  # Pause to respect IBKR pacing limits

    if all_bars:
        df_all = pd.concat(all_bars, ignore_index=True)
        df_all.sort_values('date', inplace=True)
        return df_all
    else:
        logger.error("No historical data fetched.")
        return pd.DataFrame()

# --- Helper Functions ---

def mark_rth(df):
    """
    Mark each bar as inside Regular Trading Hours (RTH: 09:30-16:00 ET on weekdays).
    """
    eastern = pytz.timezone('US/Eastern')
    if df.index.tz is None:
        df = df.tz_localize(eastern)
        logger.debug("Localized naive datetime index to US/Eastern.")
    else:
        df = df.tz_convert(eastern)
        logger.debug("Converted timezone-aware datetime index to US/Eastern.")

    df['is_rth'] = df.index.weekday < 5  # Monday=0, Sunday=6
    df['time'] = df.index.time
    df.loc[(df['time'] < time(9, 30)) | (df['time'] > time(16, 0)), 'is_rth'] = False
    df.drop(['time'], axis=1, inplace=True)
    df = df.tz_convert('UTC')  # Convert back to UTC
    return df

def rsi(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate the RSI on the given DataFrame (expects a 'close' column)."""
    delta = ohlc['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    _gain = up.ewm(com=(period - 1), min_periods=period).mean()
    _loss = down.ewm(com=(period - 1), min_periods=period).mean()
    RS = _gain / _loss
    rsi_series = 100 - (100 / (1 + RS))
    rsi_series.name = "RSI"
    return rsi_series

def calculate_vwap(ohlc: pd.DataFrame) -> pd.Series:
    """
    Calculate the Volume Weighted Average Price (VWAP) on the provided DataFrame.
    VWAP resets at the start of each trading day.
    """
    ohlc = ohlc.copy()
    ohlc_eastern = ohlc.tz_convert('US/Eastern')
    ohlc_eastern['date'] = ohlc_eastern.index.date
    typical_price = (ohlc_eastern['high'] + ohlc_eastern['low'] + ohlc_eastern['close']) / 3
    ohlc_eastern['tpv'] = typical_price * ohlc_eastern['volume']
    ohlc_eastern['cum_tpv'] = ohlc_eastern.groupby('date')['tpv'].cumsum()
    ohlc_eastern['cum_vol'] = ohlc_eastern.groupby('date')['volume'].cumsum()
    ohlc_eastern['vwap'] = ohlc_eastern['cum_tpv'] / ohlc_eastern['cum_vol']
    ohlc['vwap'] = ohlc_eastern['vwap']
    return ohlc['vwap']

# --- Backtest Class Using Dynamic RSI Update (Like the Live Code) ---

class MESFuturesBacktest:
    def __init__(self, contract, start_date, end_date, timeframe, initial_capital,
                 position_size, contract_multiplier, stop_loss_points, take_profit_points,
                 commission, slippage, rsi_period=14):
        """
        Initializes the backtest.
        """
        # Convert dates to timezone-aware UTC timestamps.
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
        self.rsi_period = rsi_period

        # Fetch data from IBKR using the new method.
        self.load_data(contract)
        self.prepare_data()

        # Initialize backtest state.
        self.cash = initial_capital
        self.equity = initial_capital
        self.equity_curve = []
        self.trade_log = []
        self.position = None  # "long" or "short"
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

    def load_data(self, contract):
        """Fetch historical data from IBKR using the new fetch_ibkr_data function."""
        logger.info("Fetching historical data from IBKR using new fetch method...")
        ib = IB()
        try:
            ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
        except Exception as e:
            logger.error(f"Could not connect to IBKR: {e}")
            sys.exit(1)

        # Use the new function to fetch 5‑second data.
        df_bars = fetch_ibkr_data(
            ib,
            contract,
            bar_size=self.timeframe,  # now "5 secs"
            start_time=self.start_date,
            end_time=self.end_date,
            useRTH=False
        )
        ib.disconnect()

        if df_bars.empty:
            logger.error("No historical data fetched from IBKR.")
            sys.exit(1)

        # Rename and set the index appropriately.
        df_bars.rename(columns={'date': 'Time'}, inplace=True)
        df_bars.set_index('Time', inplace=True)
        df_bars.index = pd.to_datetime(df_bars.index)

        # Save the raw data and add extra columns required for backtesting.
        self.data = df_bars.copy()
        self.data['contract'] = contract.localSymbol
        self.data['pct_chg'] = self.data['close'].pct_change() * 100
        self.data['average'] = (self.data['open'] + self.data['high'] + self.data['low'] + self.data['close']) / 4
        self.data['barCount'] = 1
        self.data['pct_chg'] = self.data['pct_chg'].fillna(0)
        logger.info(f"Fetched {len(self.data)} bars of data from IBKR.")

    def prepare_data(self):
        """
        Prepare 5‑second data by marking RTH and resampling if needed.
        Also calculates the 5‑second VWAP.
        Note: RSI is now calculated dynamically (using a 15‑minute resample) as in the live implementation.
        """
        logger.info("Preparing data for backtest...")
        # Mark Regular Trading Hours.
        self.data = mark_rth(self.data)
        logger.info("Marked Regular Trading Hours in the data.")

        # Filter the data by the backtest date range.
        self.data = self.data[(self.data.index >= self.start_date) & (self.data.index <= self.end_date)]
        logger.info(f"Filtered data from {self.start_date.date()} to {self.end_date.date()}. Points: {len(self.data)}")
        self.data.sort_index(inplace=True)

        # Resample to the chosen timeframe (for 5‑second bars this is a no‑op if already in that frequency).
        logger.info(f"Resampling data to {RESAMPLE_FREQ} bars if needed...")
        self.data_prepared = self.data.resample(RESAMPLE_FREQ).agg({
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

        # Compute VWAP on the 5‑second data.
        logger.info("Calculating 5‑second VWAP (reset daily)...")
        self.data_prepared['vwap'] = calculate_vwap(self.data_prepared)
        logger.info("Finished calculating VWAP.")

    def get_unrealized_pnl(self, current_price):
        """Calculate the unrealized profit and loss based on the current price and open position."""
        if self.position == 'long':
            return (current_price - self.entry_price) * self.position_size * self.contract_multiplier
        elif self.position == 'short':
            return (self.entry_price - current_price) * self.position_size * self.contract_multiplier
        else:
            return 0

    def run_backtest(self):
        """
        Iterate over each 5‑second bar and apply the trading logic.
        For each bar, update the RSI as follows:
          1. Take all 5‑second bars up to the current time.
          2. Resample that subset to 15‑minute bars (dropping incomplete bars).
          3. Compute the RSI on the 15‑minute OHLC data.
          4. Use the latest RSI value for trade decisions.
        """
        logger.info("Starting backtest execution...")
        times = self.data_prepared.index
        closes = self.data_prepared['close']
        highs = self.data_prepared['high']
        lows = self.data_prepared['low']
        vwaps = self.data_prepared['vwap']
        is_rths = self.data_prepared['is_rth'].astype(bool)

        for current_idx in times:
            # Get current values from the 5‑second data.
            price = closes.loc[current_idx]
            high = highs.loc[current_idx]
            low = lows.loc[current_idx]
            vwap = vwaps.loc[current_idx]
            is_rth = is_rths.loc[current_idx]

            # Dynamically calculate the 15‑minute RSI exactly like in the live code.
            # 1. Select all bars up to the current index.
            subset = self.data_prepared.loc[:current_idx]
            # 2. Resample these bars to 15‑minute OHLC data.
            bars_15m = subset.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            if len(bars_15m) < 1:
                rsi_value = np.nan
            else:
                # 3. Compute the RSI on the 15‑minute data.
                bars_15m['RSI'] = rsi(bars_15m, period=self.rsi_period)
                # 4. Use the latest complete bar's RSI.
                rsi_value = bars_15m['RSI'].iloc[-1]

            # Trading logic.
            if not np.isnan(rsi_value):
                if self.position is None:
                    if is_rth:
                        if price > vwap and rsi_value > 70:
                            self.enter_position('long', price, current_idx, vwap, rsi_value)
                        elif price < vwap and rsi_value < 30:
                            self.enter_position('short', price, current_idx, vwap, rsi_value)
                else:
                    exit_info = self.check_exit_conditions(price, high, low, current_idx)
                    if exit_info:
                        self.exit_position(exit_info, current_idx, vwap, rsi_value)

            # Update equity curve.
            unrealized_pnl = self.get_unrealized_pnl(price)
            total_equity = self.equity + unrealized_pnl
            self.equity_curve.append({
                'Time': current_idx,
                'Equity': total_equity
            })

            # Update drawdown metrics.
            self.update_drawdown(total_equity, current_idx)

        logger.info("Backtest execution completed.")

    def enter_position(self, position_type, price, current_idx, vwap, rsi_value):
        """Enter a long or short position."""
        self.position = position_type
        if position_type == 'long':
            self.entry_price = price + self.slippage
            self.stop_loss = self.entry_price - self.stop_loss_points
            self.take_profit = self.entry_price + self.take_profit_points
        else:  # short
            self.entry_price = price - self.slippage
            self.stop_loss = self.entry_price + self.stop_loss_points
            self.take_profit = self.entry_price - self.take_profit_points

        self.cash -= self.commission
        self.equity -= self.commission  # Deduct commission immediately.
        # Log additional entry details: VWAP and RSI.
        self.trade_log.append({
            'Type': position_type.capitalize(),
            'Entry Time': current_idx,
            'Entry Price': self.entry_price,
            'Entry VWAP': vwap,
            'Entry RSI': rsi_value,
            'Exit Time': None,
            'Exit Price': None,
            'Exit VWAP': None,
            'Exit RSI': None,
            'Result': None,
            'Profit': 0
        })
        self.exposed_bars += 1
        msg = (f"Trade Entry: {position_type.capitalize()} at {self.entry_price:.2f} on {current_idx}. "
               f"Price: {price:.2f}, VWAP: {vwap:.2f}, RSI: {rsi_value:.2f}")
        logger.info(msg)

    def check_exit_conditions(self, price, high, low, current_idx):
        """Check if exit conditions are met for the current position."""
        if self.position == 'long':
            if low <= self.stop_loss and high >= self.take_profit:
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

    def exit_position(self, exit_info, current_idx, vwap, rsi_value):
        """Exit the open position."""
        result = exit_info['result']
        exit_price = exit_info['price']
        if self.position == 'long':
            pnl = (exit_price - self.entry_price) * self.position_size * self.contract_multiplier
        else:  # short
            pnl = (self.entry_price - exit_price) * self.position_size * self.contract_multiplier

        pnl -= self.commission  # Deduct exit commission.
        self.cash += pnl
        self.equity += pnl

        # Update the last trade log with exit details including VWAP and RSI at exit.
        self.trade_log[-1].update({
            'Exit Time': current_idx,
            'Exit Price': exit_price,
            'Exit VWAP': vwap,
            'Exit RSI': rsi_value,
            'Result': result,
            'Profit': pnl
        })

        msg = (f"Trade Exit: {self.trade_log[-1]['Type']} exit at {exit_price:.2f} on {current_idx}. "
               f"VWAP: {vwap:.2f}, RSI: {rsi_value:.2f}. Result: {result}. Profit: {pnl:.2f}")
        logger.info(msg)

        # Reset position.
        self.position = None
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0

    def update_drawdown(self, current_total_equity, current_idx):
        """Update drawdown metrics."""
        if current_total_equity > self.equity_peak:
            self.equity_peak = current_total_equity
            if self.in_drawdown:
                dd_duration = (current_idx - self.drawdown_start).total_seconds() / 86400
                self.drawdown_durations.append(dd_duration)
                self.in_drawdown = False
                self.current_drawdown = 0
        else:
            dd = (self.equity_peak - current_total_equity) / self.equity_peak * 100
            if dd > self.current_drawdown:
                self.current_drawdown = dd
                if not self.in_drawdown:
                    self.in_drawdown = True
                    self.drawdown_start = current_idx

    def get_trade_results(self):
        """Return trade log as a DataFrame."""
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log)

    def analyze_results(self):
        """Analyze the backtest performance and plot results."""
        trade_results = self.get_trade_results()
        if trade_results.empty:
            print("No trades were executed.")
            return

        equity_df = pd.DataFrame(self.equity_curve).set_index('Time')
        start_date = equity_df.index.min().strftime("%Y-%m-%d")
        end_date = equity_df.index.max().strftime("%Y-%m-%d")
        exposure_percentage = (self.exposed_bars / self.total_bars) * 100
        final_balance = self.equity
        total_return_pct = ((final_balance - self.initial_capital) / self.initial_capital) * 100

        # Annualized return calculation.
        delta = equity_df.index.max() - equity_df.index.min()
        years = delta.days / 365.25 if delta.days > 0 else 0
        annualized_return = ((final_balance / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Benchmark (Buy & Hold) calculation.
        benchmark_start = self.data_prepared['close'].iloc[0]
        benchmark_end = self.data_prepared['close'].iloc[-1]
        benchmark_return = ((benchmark_end - benchmark_start) / benchmark_start) * 100

        # Daily returns for Sharpe/Sortino.
        equity_daily = equity_df['Equity'].resample('D').last().dropna()
        daily_returns = equity_daily.pct_change().dropna()
        volatility_annual = daily_returns.std() * np.sqrt(252) * 100
        risk_free_rate = 0
        sharpe_ratio = ((daily_returns.mean() - risk_free_rate) / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
        downside = daily_returns[daily_returns < 0]
        sortino_ratio = (daily_returns.mean() * np.sqrt(252)) / (downside.std() * np.sqrt(252)) if not downside.empty else np.inf

        # Drawdowns.
        running_max = equity_df['Equity'].cummax()
        drawdowns_pct = (equity_df['Equity'] - running_max) / running_max * 100
        max_drawdown_pct = drawdowns_pct.min()
        avg_drawdown_pct = drawdowns_pct[drawdowns_pct < 0].mean() if not drawdowns_pct[drawdowns_pct < 0].empty else 0
        drawdowns_dollar = running_max - equity_df['Equity']
        max_drawdown_dollar = drawdowns_dollar.max()
        avg_drawdown_dollar = drawdowns_dollar[drawdowns_dollar > 0].mean() if (drawdowns_dollar > 0).any() else 0

        # Profit factor.
        wins = trade_results[trade_results['Result'] == 'Take Profit']['Profit']
        losses = trade_results[trade_results['Result'] == 'Stop Loss']['Profit']
        profit_factor = wins.sum() / abs(losses.sum()) if (not losses.empty and losses.sum() != 0) else np.inf

        # Calmar ratio.
        calmar_ratio = total_return_pct / abs(max_drawdown_pct) if abs(max_drawdown_pct) != 0 else np.inf

        # Win rate.
        win_rate = (len(wins) / len(trade_results)) * 100 if len(trade_results) > 0 else 0

        results = {
            "Start Date": start_date,
            "End Date": end_date,
            "Exposure Time": f"{exposure_percentage:.2f}%",
            "Final Account Balance": f"${final_balance:,.2f}",
            "Total Return": f"{total_return_pct:.2f}%",
            "Annualized Return": f"{annualized_return:.2f}%",
            "Benchmark Return": f"{benchmark_return:.2f}%",
            "Volatility (Annual)": f"{volatility_annual:.2f}%",
            "Total Trades": len(trade_results),
            "Winning Trades": len(wins),
            "Losing Trades": len(losses),
            "Win Rate": f"{win_rate:.2f}%",
            "Profit Factor": f"{profit_factor:.2f}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Sortino Ratio": f"{sortino_ratio:.2f}",
            "Calmar Ratio": f"{calmar_ratio:.2f}",
            "Max Drawdown (%)": f"{max_drawdown_pct:.2f}%",
            "Average Drawdown (%)": f"{avg_drawdown_pct:.2f}%",
            "Max Drawdown ($)": f"${max_drawdown_dollar:,.2f}",
            "Average Drawdown ($)": f"${avg_drawdown_dollar:,.2f}",
            "Max Drawdown Duration": f"{max(self.drawdown_durations) if self.drawdown_durations else 0:.2f} days",
            "Average Drawdown Duration": f"{np.mean(self.drawdown_durations) if self.drawdown_durations else 0:.2f} days",
        }

        print("\nPerformance Summary:")
        for key, value in results.items():
            print(f"{key:30}: {value:>15}")

        # Plot the equity curve vs. a buy & hold benchmark.
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

        # Plot Price and VWAP from the 5‑second data.
        plt.figure(figsize=(14, 7))
        plt.plot(self.data_prepared.index, self.data_prepared['close'], label='Close Price')
        plt.plot(self.data_prepared.index, self.data_prepared['vwap'], label='VWAP', linestyle='--')
        plt.title('Price vs 5‑second VWAP')
        plt.xlabel('Time')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def get_trade_results(self):
        """Return the trade log as a DataFrame."""
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log)

# --- Main Execution ---

if __name__ == "__main__":
    try:
        backtester = MESFuturesBacktest(
            contract=ES_CONTRACT,
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
            rsi_period=14  # RSI period for the 15‑minute bars
        )
        backtester.run_backtest()
        backtester.analyze_results()
    except Exception as e:
        logger.error(f"An error occurred during backtesting: {e}")
        sys.exit(1)