import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pytz
import logging
import sys
import gc
import time
import calendar

# --- IBKR API Imports ---
from ib_insync import *

# --- Configuration Parameters ---
IB_HOST = '127.0.0.1'        # IBKR Gateway/TWS host
IB_PORT = 7497               # IBKR Gateway/TWS paper trading port
CLIENT_ID = 2                # Unique client ID (ensure it's different from other scripts)

# Instrument to backtest (for example, ES for the E-mini S&P 500)
EXEC_SYMBOL = 'ES'
EXEC_EXCHANGE = 'CME'        # Exchange for the futures
CURRENCY = 'USD'

INITIAL_CASH = 5000          # Starting cash
POSITION_SIZE = 1            # Number of contracts per trade
CONTRACT_MULTIPLIER = 50     # For ES the multiplier is 50

BOLLINGER_PERIOD = 15
BOLLINGER_STDDEV = 2
STOP_LOSS_DISTANCE = 5       # Points away from entry
TAKE_PROFIT_DISTANCE = 10    # Points away from entry

ROLL_DAYS = 3  # Number of days before expiry to roll the contract

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,  # or DEBUG as needed
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# --- Helper Function to Filter RTH ---
def filter_rth(df):
    """
    Filters the DataFrame to include only Regular Trading Hours (09:30 - 16:00 ET) on weekdays.
    Assumes the DataFrame index is a timezone-aware datetime.
    """
    eastern = pytz.timezone('US/Eastern')
    if df.index.tz is None:
        df = df.tz_localize(eastern)
        logger.debug("Localized naive datetime index to US/Eastern.")
    else:
        df = df.tz_convert(eastern)
        logger.debug("Converted timezone-aware datetime index to US/Eastern.")
    df_eastern = df[df.index.weekday < 5]
    df_rth = df_eastern.between_time('09:30', '16:00')
    df_rth = df_rth.tz_convert('UTC')
    return df_rth

def fetch_ibkr_data(ib, contract, bar_size, start_time, end_time, useRTH=False):
    """
    Fetch historical data from IBKR using ib_insync over the specified time range.
    Data is fetched in chunks to satisfy IBKR’s limits.
    """
    # Set maximum chunk duration based on bar size.
    if bar_size == '1 min':
        max_chunk = pd.Timedelta(days=7)  # IBKR limit for 1-min bars is about 7 days.
    elif bar_size == '30 mins':
        max_chunk = pd.Timedelta(days=365)  # 30-min bars allow for a longer duration.
    else:
        max_chunk = pd.Timedelta(days=30)
    
    current_end = end_time
    all_bars = []
    while current_end > start_time:
        current_start = max(start_time, current_end - max_chunk)
        delta = current_end - current_start
        # Instead of using hours ("H"), use days ("D").
        if delta < pd.Timedelta(days=1):
            # For durations less than one day, request 1 day.
            duration_str = "1 D"
        else:
            duration_days = delta.days
            duration_str = f"{duration_days} D"
        
        # Format endDateTime as required by IBKR (YYYYMMDD HH:MM:SS)
        end_dt_str = current_end.strftime("%Y%m%d %H:%M:%S")
        logger.info(f"Requesting {bar_size} bars from {current_start} to {current_end} (Duration: {duration_str}) for contract expiry {contract.lastTradeDateOrContractMonth}")
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_dt_str,
            durationStr=duration_str,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=useRTH,
            formatDate=1
        )
        if not bars:
            logger.warning("No bars returned for this chunk.")
            break
        df_chunk = util.df(bars)
        all_bars.append(df_chunk)
        # Update current_end to just before the earliest bar in this chunk
        earliest_bar_time = pd.to_datetime(df_chunk['date'].min())
        current_end = earliest_bar_time - pd.Timedelta(seconds=1)
        time.sleep(1)  # Pause to respect IBKR pacing limits
    if all_bars:
        df_all = pd.concat(all_bars, ignore_index=True)
        df_all.sort_values('date', inplace=True)
        df_all.rename(columns={'date': 'Time'}, inplace=True)
        df_all.set_index('Time', inplace=True)
        return df_all
    else:
        logger.error("No historical data fetched from IBKR.")
        return pd.DataFrame()

# --- Vectorized Exit Evaluation Function ---
def evaluate_exit_vectorized(position_type, entry_price, stop_loss, take_profit, df_high_freq, entry_time):
    """
    Determines the exit price and time using vectorized operations.
    """
    df_period = df_high_freq.loc[entry_time:]
    if df_period.empty:
        return None, None, None
    max_valid_timestamp = pd.Timestamp('2262-04-11 23:47:16.854775807', tz='UTC')
    if position_type == 'long':
        hit_sl = df_period[df_period['low'] <= stop_loss]
        hit_tp = df_period[df_period['high'] >= take_profit]
        first_sl = hit_sl.index.min() if not hit_sl.empty else max_valid_timestamp
        first_tp = hit_tp.index.min() if not hit_tp.empty else max_valid_timestamp
        if first_sl < first_tp:
            return stop_loss, first_sl, False
        elif first_tp < first_sl:
            return take_profit, first_tp, True
        elif first_sl == first_tp and first_sl != max_valid_timestamp:
            row = df_period.loc[first_sl]
            if row['low'] <= stop_loss:
                return stop_loss, first_sl, False
            else:
                return take_profit, first_sl, True
    elif position_type == 'short':
        hit_sl = df_period[df_period['high'] >= stop_loss]
        hit_tp = df_period[df_period['low'] <= take_profit]
        first_sl = hit_sl.index.min() if not hit_sl.empty else max_valid_timestamp
        first_tp = hit_tp.index.min() if not hit_tp.empty else max_valid_timestamp
        if first_sl < first_tp:
            return stop_loss, first_sl, False
        elif first_tp < first_sl:
            return take_profit, first_tp, True
        elif first_sl == first_tp and first_sl != max_valid_timestamp:
            row = df_period.loc[first_sl]
            if row['open'] >= stop_loss:
                return stop_loss, first_sl, False
            else:
                return take_profit, first_sl, True
    return None, None, None

# --- Manual Contract Selection Functions ---
def get_third_friday(year, month):
    """
    Returns a timezone-aware Timestamp for the third Friday of the given year and month (US/Eastern).
    """
    cal = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in cal if week[calendar.FRIDAY] != 0]
    third_friday = fridays[2]  # third Friday (0-indexed)
    return pd.Timestamp(year=year, month=month, day=third_friday, tz='US/Eastern')

def generate_sorted_contracts(symbol, exchange, currency, S, E, roll_days):
    """
    Manually generate a list of candidate futures contracts (tuples of (expiry, roll_date, contract))
    for quarterly expiries (assumed in March, June, September, December) over a period covering S to E.
    We use the full expiry date (YYYYMMDD) and set includeExpired=True.
    Note: Removed the condition on roll_date < E so that candidates (like March 2025) are included.
    """
    valid_months = [3, 6, 9, 12]
    contracts = []
    # Generate a range wide enough to cover S to E (from one year before S to one year after E)
    for year in range(S.year - 1, E.year + 2):
        for month in valid_months:
            expiry = get_third_friday(year, month)
            roll_date = expiry - pd.Timedelta(days=roll_days)
            if expiry > S:  # Only check that the expiry is after the start date.
                expiry_str = expiry.strftime("%Y%m%d")
                contract = Future(symbol=symbol,
                                  lastTradeDateOrContractMonth=expiry_str,
                                  exchange=exchange,
                                  currency=currency)
                contract.includeExpired = True
                contract.multiplier = str(CONTRACT_MULTIPLIER)
                contracts.append((expiry, roll_date, contract))
    contracts.sort(key=lambda x: x[1])  # sort by roll_date
    return contracts

def generate_contract_segments_manual(symbol, exchange, currency, S, E, roll_days):
    """
    From the manually generated sorted contracts, create segments for the backtest.
    Each segment is a tuple (segment_start, segment_end, contract) indicating the time
    period during which that contract was assumed to be most liquid.
    """
    sorted_contracts = generate_sorted_contracts(symbol, exchange, currency, S, E, roll_days)
    segments = []
    segment_start = S
    for expiry, roll_date, contract in sorted_contracts:
        if roll_date > segment_start:
            segment_end = min(E, roll_date)
            segments.append((segment_start, segment_end, contract))
            segment_start = roll_date  # new segment starts at roll_date
        if segment_start >= E:
            break
    if segment_start < E and sorted_contracts:
        last_contract = sorted_contracts[-1][2]
        segments.append((segment_start, E, last_contract))
    return segments

# --- Connect to IBKR ---
logger.info("Connecting to IBKR...")
ib = IB()
try:
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
except Exception as e:
    logger.error(f"Could not connect to IBKR: {e}")
    sys.exit(1)

# --- Define Backtest Period: Last 2 Years ---
end_time = pd.Timestamp.now(tz=pytz.UTC)
start_time = end_time - pd.Timedelta(days=730)
logger.info(f"Backtest Period: {start_time} to {end_time}")

# --- Get Active Contract Segments (Manual Roll) ---
segments = generate_contract_segments_manual(EXEC_SYMBOL, EXEC_EXCHANGE, CURRENCY, start_time, end_time, roll_days=ROLL_DAYS)
if not segments:
    logger.error("No contract segments available. Exiting.")
    sys.exit(1)

logger.info("Contract segments determined:")
for seg_start, seg_end, contract in segments:
    logger.info(f"Segment: {seg_start} to {seg_end} using contract expiry {contract.lastTradeDateOrContractMonth}")

# --- Fetch Historical Data for Each Segment and Combine ---
df_1m_list = []
df_30m_list = []

for seg_start, seg_end, contract in segments:
    logger.info(f"Fetching 1-Minute data for contract expiry {contract.lastTradeDateOrContractMonth} from {seg_start} to {seg_end}...")
    df1 = fetch_ibkr_data(ib, contract, '1 min', seg_start, seg_end, useRTH=False)
    if not df1.empty:
        df_1m_list.append(df1)
    else:
        logger.warning(f"No 1-Minute data for segment {seg_start} to {seg_end} using contract expiry {contract.lastTradeDateOrContractMonth}.")
    
    logger.info(f"Fetching 30-Minute data for contract expiry {contract.lastTradeDateOrContractMonth} from {seg_start} to {seg_end}...")
    df30 = fetch_ibkr_data(ib, contract, '30 mins', seg_start, seg_end, useRTH=False)
    if not df30.empty:
        df_30m_list.append(df30)
    else:
        logger.warning(f"No 30-Minute data for segment {seg_start} to {seg_end} using contract expiry {contract.lastTradeDateOrContractMonth}.")

if df_1m_list:
    df_1m = pd.concat(df_1m_list).drop_duplicates().sort_index()
else:
    logger.error("No 1-minute data collected from any segment.")
    sys.exit(1)

if df_30m_list:
    df_30m_full = pd.concat(df_30m_list).drop_duplicates().sort_index()
else:
    logger.error("No 30-minute data collected from any segment.")
    sys.exit(1)

# --- Localize and Convert Timezones ---
eastern = pytz.timezone('US/Eastern')
if df_1m.index.tz is None:
    df_1m = df_1m.tz_localize(eastern).tz_convert('UTC')
else:
    df_1m = df_1m.tz_convert('UTC')
if df_30m_full.index.tz is None:
    df_30m_full = df_30m_full.tz_localize(eastern).tz_convert('UTC')
else:
    df_30m_full = df_30m_full.tz_convert('UTC')

logger.info(f"1-Minute Data Range: {df_1m.index.min()} to {df_1m.index.max()}")
logger.info(f"30-Minute Data Range: {df_30m_full.index.min()} to {df_30m_full.index.max()}")

# --- Adjust 1-Minute Data Timestamp Alignment ---
df_1m.index = df_1m.index - pd.Timedelta(minutes=30)

# --- Calculate Bollinger Bands on Full 30-Minute Data (Including Extended Hours) ---
logger.info("Calculating Bollinger Bands on full 30-minute data (including extended hours)...")
df_30m_full['ma'] = df_30m_full['close'].rolling(window=BOLLINGER_PERIOD, min_periods=BOLLINGER_PERIOD).mean()
df_30m_full['std'] = df_30m_full['close'].rolling(window=BOLLINGER_PERIOD, min_periods=BOLLINGER_PERIOD).std()
df_30m_full['upper_band'] = df_30m_full['ma'] + (BOLLINGER_STDDEV * df_30m_full['std'])
df_30m_full['lower_band'] = df_30m_full['ma'] - (BOLLINGER_STDDEV * df_30m_full['std'])
logger.info(f"'ma' column has {df_30m_full['ma'].isna().sum()} NaN values after rolling calculation.")
logger.info(f"'std' column has {df_30m_full['std'].isna().sum()} NaN values after rolling calculation.")
df_30m_full.dropna(subset=['ma', 'std', 'upper_band', 'lower_band'], inplace=True)
logger.info(f"After Bollinger Bands calculation, 30-Minute Full Data Points: {len(df_30m_full)}")

# --- Filter RTH Data for Trade Execution ---
logger.info("Applying RTH filter to 30-minute data for trade execution...")
df_30m_rth = filter_rth(df_30m_full)
logger.info(f"30-Minute RTH Data Points after Filtering: {len(df_30m_rth)}")
if df_30m_rth.empty:
    logger.warning("No 30-minute RTH data points after filtering. Exiting backtest.")
    sys.exit(1)

print(f"\nBacktesting from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
print(f"1-Minute Data Points: {len(df_1m)}")
print(f"30-Minute Full Data Points: {len(df_30m_full)}")
print(f"30-Minute RTH Data Points: {len(df_30m_rth)}")

# --- Initialize Backtest Variables ---
position_size = 0
entry_price = None
position_type = None  
cash = INITIAL_CASH
trade_results = []
balance_series = []  # record account balance at each 30-min bar
exposure_bars = 0

# --- Prepare High-Frequency Data ---
df_high_freq = df_1m.sort_index()
df_high_freq = df_high_freq.astype({
    'open': 'float32',
    'high': 'float32',
    'low': 'float32',
    'close': 'float32',
    'volume': 'float32',
    'average': 'float32',
    'barCount': 'int32'
})

# --- Backtesting Loop ---
logger.info("Starting backtesting loop...")
for i, current_time in enumerate(df_30m_rth.index):
    current_bar = df_30m_rth.loc[current_time]
    current_price = current_bar['close']
    if position_size != 0:
        exposure_bars += 1
    if position_size == 0:
        upper_band = df_30m_full.loc[current_time, 'upper_band']
        lower_band = df_30m_full.loc[current_time, 'lower_band']
        if current_price < lower_band:
            position_size = POSITION_SIZE
            entry_price = current_price
            position_type = 'long'
            stop_loss_price = entry_price - STOP_LOSS_DISTANCE
            take_profit_price = entry_price + TAKE_PROFIT_DISTANCE
            entry_time = current_time
            logger.info(f"Entering LONG at {entry_price} on {entry_time} UTC; Bollinger Bands -> Lower: {lower_band}, Upper: {upper_band}")
        elif current_price > upper_band:
            position_size = POSITION_SIZE
            entry_price = current_price
            position_type = 'short'
            stop_loss_price = entry_price + STOP_LOSS_DISTANCE
            take_profit_price = entry_price - TAKE_PROFIT_DISTANCE
            entry_time = current_time
            logger.info(f"Entering SHORT at {entry_price} on {entry_time} UTC; Bollinger Bands -> Lower: {lower_band}, Upper: {upper_band}")
    else:
        exit_price, exit_time, hit_tp = evaluate_exit_vectorized(
            position_type,
            entry_price,
            stop_loss_price,
            take_profit_price,
            df_high_freq,
            entry_time
        )
        if exit_price is not None and exit_time is not None:
            try:
                exit_candle = df_high_freq.loc[exit_time]
                logger.info(f"Exit Candle Timestamp: {exit_time} | Open: {exit_candle['open']}, High: {exit_candle['high']}, Low: {exit_candle['low']}, Close: {exit_candle['close']}")
            except KeyError:
                logger.warning(f"Exit time {exit_time} not found in 1-min data.")
            if position_type == 'long':
                pnl = ((exit_price - entry_price) * CONTRACT_MULTIPLIER * position_size) - (0.62 * 2)
            elif position_type == 'short':
                pnl = ((entry_price - exit_price) * CONTRACT_MULTIPLIER * position_size) - (0.62 * 2)
            trade_results.append(pnl)
            cash += pnl
            exit_type = "TAKE PROFIT" if hit_tp else "STOP LOSS"
            logger.info(f"Exiting {position_type.upper()} at {exit_price} on {exit_time} UTC via {exit_type} for P&L: ${pnl:.2f}")
            position_size = 0
            position_type = None
            entry_price = None
            stop_loss_price = None
            take_profit_price = None
            entry_time = None
    balance_series.append(cash)
    if (i + 1) % 100000 == 0:
        logger.info(f"Processed {i + 1} out of {len(df_30m_rth)} 30-minute bars.")

del df_high_freq
gc.collect()
logger.info("Backtesting loop completed.")

# --- Post-Backtest Calculations ---
balance_series = pd.Series(balance_series, index=df_30m_rth.index)
total_return_percentage = ((cash - INITIAL_CASH) / INITIAL_CASH) * 100
trading_days = max((df_30m_full.index.max() - df_30m_full.index.min()).days, 1)
annualized_return_percentage = ((cash / INITIAL_CASH) ** (252 / trading_days) - 1) * 100
benchmark_return = ((df_30m_full['close'].iloc[-1] - df_30m_full['close'].iloc[0]) / df_30m_full['close'].iloc[0]) * 100
equity_peak = balance_series.max()
daily_equity = balance_series.resample('D').ffill()
daily_returns = daily_equity.pct_change().dropna()
volatility_annual = daily_returns.std() * np.sqrt(252) * 100
risk_free_rate = 0
sharpe_ratio = ((daily_returns.mean() - risk_free_rate) / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0

def calculate_sortino_ratio(daily_returns, target_return=0):
    if daily_returns.empty:
        return np.nan
    excess_returns = daily_returns - target_return
    downside_returns = excess_returns[excess_returns < 0]
    if downside_returns.empty or downside_returns.std() == 0:
        return np.inf
    downside_std = downside_returns.std() * np.sqrt(252)
    annualized_mean_excess_return = daily_returns.mean() * 252
    return annualized_mean_excess_return / downside_std

sortino_ratio = calculate_sortino_ratio(daily_returns)
running_max_series = balance_series.cummax()
drawdowns = (balance_series - running_max_series) / running_max_series
max_drawdown = drawdowns.min() * 100
average_drawdown = drawdowns[drawdowns < 0].mean() * 100 if not drawdowns[drawdowns < 0].empty else 0
exposure_time_percentage = (exposure_bars / len(df_30m_rth)) * 100 if len(df_30m_rth) > 0 else 0
winning_trades = [pnl for pnl in trade_results if pnl > 0]
losing_trades = [pnl for pnl in trade_results if pnl <= 0]
profit_factor = (sum(winning_trades) / abs(sum(losing_trades))) if losing_trades else float('inf')
calmar_ratio = (total_return_percentage / abs(max_drawdown)) if max_drawdown != 0 else float('inf')

drawdown_periods = drawdowns[drawdowns < 0]
if not drawdown_periods.empty:
    is_drawdown = drawdowns < 0
    drawdown_changes = is_drawdown.ne(is_drawdown.shift())
    drawdown_groups = drawdown_changes.cumsum()
    drawdown_groups = is_drawdown.groupby(drawdown_groups)
    drawdown_durations = []
    for name, group in drawdown_groups:
        if group.iloc[0]:
            duration = (group.index[-1] - group.index[0]).total_seconds() / 86400
            drawdown_durations.append(duration)
    if drawdown_durations:
        max_drawdown_duration_days = max(drawdown_durations)
        average_drawdown_duration_days = np.mean(drawdown_durations)
    else:
        max_drawdown_duration_days = 0
        average_drawdown_duration_days = 0
else:
    max_drawdown_duration_days = 0
    average_drawdown_duration_days = 0

print("\nPerformance Summary:")
results = {
    "Start Date": df_30m_full.index.min().strftime("%Y-%m-%d"),
    "End Date": df_30m_full.index.max().strftime("%Y-%m-%d"),
    "Exposure Time": f"{exposure_time_percentage:.2f}%",
    "Final Account Balance": f"${cash:,.2f}",
    "Equity Peak": f"${equity_peak:,.2f}",
    "Total Return": f"{total_return_percentage:.2f}%",
    "Annualized Return": f"{annualized_return_percentage:.2f}%",
    "Benchmark Return": f"{benchmark_return:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Total Trades": len(trade_results),
    "Winning Trades": len(winning_trades),
    "Losing Trades": len(losing_trades),
    "Win Rate": f"{(len(winning_trades)/len(trade_results)*100) if trade_results else 0:.2f}%",
    "Profit Factor": f"{profit_factor:.2f}",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}",
    "Sortino Ratio": f"{sortino_ratio:.2f}",
    "Calmar Ratio": f"{calmar_ratio:.2f}",
    "Max Drawdown": f"{max_drawdown:.2f}%",
    "Average Drawdown": f"{average_drawdown:.2f}%",
    "Max Drawdown Duration": f"{max_drawdown_duration_days:.2f} days",
    "Average Drawdown Duration": f"{average_drawdown_duration_days:.2f} days",
}
for key, value in results.items():
    print(f"{key:25}: {value:>15}")

# --- Print Raw Bar Data at a Specific Timestamp ---
target_timestamp = pd.Timestamp("2025-01-31 19:30:00", tz="UTC")
if target_timestamp in df_1m.index:
    raw_bar_1m = df_1m.loc[target_timestamp]
    logger.info(f"1-Minute Bar at {target_timestamp} - Open: {raw_bar_1m['open']}, High: {raw_bar_1m['high']}, Low: {raw_bar_1m['low']}, Close: {raw_bar_1m['close']}, Volume: {raw_bar_1m['volume']}, Bar Count: {raw_bar_1m['barCount']}")
else:
    logger.warning(f"No 1-minute data found for timestamp {target_timestamp}.")
if target_timestamp in df_30m_full.index:
    raw_bar_30m = df_30m_full.loc[target_timestamp]
    logger.info(f"30-Minute Bar at {target_timestamp} - Open: {raw_bar_30m['open']}, High: {raw_bar_30m['high']}, Low: {raw_bar_30m['low']}, Close: {raw_bar_30m['close']}, Volume: {raw_bar_30m['volume']}, Bar Count: {raw_bar_30m['barCount']}")
else:
    logger.warning(f"No 30-minute data found for timestamp {target_timestamp}.")

# --- Plot Equity Curves ---
if len(balance_series) < 2:
    logger.warning("Not enough data points to plot equity curves.")
else:
    initial_close = df_30m_full['close'].iloc[0]
    benchmark_equity = (df_30m_full['close'] / initial_close) * INITIAL_CASH
    benchmark_equity = benchmark_equity.reindex(balance_series.index, method='ffill')
    if benchmark_equity.isna().sum() > 0:
        benchmark_equity = benchmark_equity.fillna(method='ffill')
    equity_df = pd.DataFrame({
        'Strategy': balance_series,
        'Benchmark': benchmark_equity
    })
    plt.figure(figsize=(14, 7))
    plt.plot(equity_df['Strategy'], label='Strategy Equity')
    plt.plot(equity_df['Benchmark'], label='Benchmark Equity')
    plt.title('Equity Curve: Strategy vs Benchmark')
    plt.xlabel('Time')
    plt.ylabel('Account Balance ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Disconnect from IBKR ---
ib.disconnect()
logger.info("Disconnected from IBKR.")