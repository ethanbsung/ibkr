import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time as dt_time, timedelta
import pytz
import yfinance as yf
import logging
import sys
import time

# --- Configuration Parameters ---
CSV_FILE_PATH = 'es_15m_data.csv'      # Replace with your actual CSV file path
TIMEFRAME = '15min'                    # Aggregation timeframe
DONCHIAN_PERIOD = 20                   # Number of periods for Donchian Channels
ATR_PERIOD = 14                        # Number of periods for ATR
VOLUME_MULTIPLIER = 1.5                # Current volume must be 1.5 times the average volume
STOP_LOSS_ATR_MULTIPLIER = 0.5         # Stop loss multiplier
TAKE_PROFIT_ATR_MULTIPLIER = 1         # Take profit multiplier
SESSION_START = dt_time(9, 30)         # RTH start
SESSION_END = dt_time(16, 0)           # RTH end
EASTERN = pytz.timezone('US/Eastern')
INITIAL_CAPITAL = 5000
COMMISSION_PER_TRADE = 1.24
SLIPPAGE = 0.0
MULTIPLIER = 5                      # MES contract multiplier
BACKTEST_START_DATE = '2022-12-18'  # Start date of backtest
BACKTEST_END_DATE = '2023-12-17'    # End date of backtest
BENCHMARK_TICKER = '^GSPC'

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# --- Helper Functions ---

def load_data(file_path):
    """
    Load historical data from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        # Convert 'date' column to DateTime with timezone
        df['DateTime'] = pd.to_datetime(df['date'])
        # Set DateTime as the index
        df.set_index('DateTime', inplace=True)
        # Sort the index
        df.sort_index(inplace=True)
        # Drop original 'date' column
        df.drop(['date'], axis=1, inplace=True)
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logger.error("No data: The CSV file is empty.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred while reading the CSV: {e}")
        sys.exit(1)

def calculate_donchian_channels(df, period):
    """
    Calculate Donchian Channels.
    """
    df['Donchian_High'] = df['high'].rolling(window=period).max()
    df['Donchian_Low'] = df['low'].rolling(window=period).min()
    return df

def calculate_atr(df, period):
    """
    Calculate Average True Range (ATR).
    """
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    df.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, inplace=True)
    return df

def calculate_volume_average(df, period=20):
    """
    Calculate the moving average of volume.
    """
    df['Volume_Avg'] = df['volume'].rolling(window=period).mean()
    return df

def filter_sessions(df, start_date, end_date):
    """
    Filter data to include only RTH and within the specified date range.
    """
    if df.index.tz is None:
        df = df.tz_localize('UTC')
    else:
        df = df.tz_convert('UTC')
    
    df_eastern = df.copy()
    df_eastern.index = df_eastern.index.tz_convert(EASTERN)
    mask = (df_eastern.index.date >= pd.to_datetime(start_date).date()) & \
           (df_eastern.index.date <= pd.to_datetime(end_date).date())
    df_eastern = df_eastern.loc[mask]
    df_eastern = df_eastern[df_eastern.index.weekday < 5]  # Weekdays only
    df_rth = df_eastern.between_time(SESSION_START, SESSION_END)
    df_rth.index = df_rth.index.tz_convert('UTC')
    return df_rth

def generate_signals(df):
    """
    Generate buy and sell signals based on the strategy.
    """
    df['Position'] = 0  # 1 for Long, -1 for Short, 0 for Flat
    df['Session'] = df.index.time
    df['Date'] = df.index.date
    sessions = df.groupby('Date')
    
    previous_session = None
    for current_date, group in sessions:
        if previous_session is not None:
            prev_high = previous_session['high'].max()
            prev_low = previous_session['low'].min()
            df.loc[group.index, 'Prev_Session_High'] = prev_high
            df.loc[group.index, 'Prev_Session_Low'] = prev_low
        previous_session = group
    
    # Drop rows where previous session data is not available
    df.dropna(subset=['Prev_Session_High', 'Prev_Session_Low'], inplace=True)
    
    # Buy Signal
    df['Buy_Signal'] = ((df['close'] > df['Prev_Session_High']) & 
                        (df['volume'] > df['Volume_Avg'] * VOLUME_MULTIPLIER))
    # Sell Signal
    df['Sell_Signal'] = df['close'] < df['Prev_Session_Low']
    
    return df

def calculate_benchmark_return(start_date, end_date, ticker=BENCHMARK_TICKER):
    """
    Calculate the benchmark return for the given ticker and date range.
    """
    benchmark_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if benchmark_data.empty:
        logger.warning("Benchmark data is empty. Benchmark return set to 0.0")
        return 0.0
    benchmark_start_price = benchmark_data['Close'].iloc[0]
    benchmark_end_price = benchmark_data['Close'].iloc[-1]
    benchmark_return = ((benchmark_end_price - benchmark_start_price) / benchmark_start_price) * 100
    return float(benchmark_return)

# --- Backtest Execution ---
def backtest(df):
    capital = INITIAL_CAPITAL
    equity_peak = INITIAL_CAPITAL
    position = 0  # 1 for Long, -1 for Short, 0 for Flat
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    trades = []
    equity_curve = []
    exposure_time = 0
    total_time = len(df)
    
    equity_over_time = []
    drawdowns = []
    peak = INITIAL_CAPITAL
    max_drawdown = 0
    
    for idx, row in df.iterrows():
        # Entry
        if position == 0:
            if row['Buy_Signal']:
                position = 1
                entry_price = row['close']
                stop_loss = entry_price - row['ATR'] * STOP_LOSS_ATR_MULTIPLIER
                take_profit = entry_price + row['ATR'] * TAKE_PROFIT_ATR_MULTIPLIER
                capital -= (entry_price * MULTIPLIER) + COMMISSION_PER_TRADE
                trades.append({'Entry_Date': idx, 'Type': 'Long', 'Entry_Price': entry_price, 
                               'Exit_Date': None, 'Exit_Price': None, 'Profit': None})
            elif row['Sell_Signal']:
                position = -1
                entry_price = row['close']
                stop_loss = entry_price + row['ATR'] * STOP_LOSS_ATR_MULTIPLIER
                take_profit = entry_price - row['ATR'] * TAKE_PROFIT_ATR_MULTIPLIER
                capital += (entry_price * MULTIPLIER) - COMMISSION_PER_TRADE
                trades.append({'Entry_Date': idx, 'Type': 'Short', 'Entry_Price': entry_price, 
                               'Exit_Date': None, 'Exit_Price': None, 'Profit': None})
        
        # Exit
        elif position == 1:
            # Long position exit conditions
            if row['low'] <= stop_loss:
                exit_price = stop_loss
                profit = ((exit_price - entry_price) * MULTIPLIER) - (COMMISSION_PER_TRADE * 2)
                capital += (exit_price * MULTIPLIER) - COMMISSION_PER_TRADE
                trades[-1].update({'Exit_Date': idx, 'Exit_Price': exit_price, 'Profit': profit})
                position = 0
            elif row['high'] >= take_profit:
                exit_price = take_profit
                profit = ((exit_price - entry_price) * MULTIPLIER) - (COMMISSION_PER_TRADE * 2)
                capital += (exit_price * MULTIPLIER) - COMMISSION_PER_TRADE
                trades[-1].update({'Exit_Date': idx, 'Exit_Price': exit_price, 'Profit': profit})
                position = 0
                
        elif position == -1:
            # Short position exit conditions
            if row['high'] >= stop_loss:
                exit_price = stop_loss
                profit = ((entry_price - exit_price) * MULTIPLIER) - (COMMISSION_PER_TRADE * 2)
                capital -= (exit_price * MULTIPLIER) + COMMISSION_PER_TRADE
                trades[-1].update({'Exit_Date': idx, 'Exit_Price': exit_price, 'Profit': profit})
                position = 0
            elif row['low'] <= take_profit:
                exit_price = take_profit
                profit = ((entry_price - exit_price) * MULTIPLIER) - (COMMISSION_PER_TRADE * 2)
                capital -= (exit_price * MULTIPLIER) + COMMISSION_PER_TRADE
                trades[-1].update({'Exit_Date': idx, 'Exit_Price': exit_price, 'Profit': profit})
                position = 0
        
        # Track equity and drawdowns
        if capital > equity_peak:
            equity_peak = capital
        if capital < peak:
            current_drawdown = (peak - capital) / peak * 100
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
            drawdowns.append(current_drawdown)
        else:
            peak = capital
        
        equity_over_time.append(capital)
        if position != 0:
            exposure_time += 1
        equity_curve.append({'DateTime': idx, 'Capital': capital})
    
    equity_df = pd.DataFrame(equity_curve)
    equity_df.set_index('DateTime', inplace=True)
    
    # Drawdown duration calculations
    drawdown_durations = []
    in_drawdown = False
    peak_cap = INITIAL_CAPITAL
    duration = 0
    for eq_val in equity_over_time:
        if eq_val < peak_cap:
            if not in_drawdown:
                in_drawdown = True
                duration = 1
            else:
                duration += 1
        else:
            if in_drawdown:
                drawdown_durations.append(duration)
                in_drawdown = False
            peak_cap = eq_val
    if in_drawdown:
        drawdown_durations.append(duration)
    
    if drawdown_durations:
        # Convert number of bars to days
        # TIMEFRAME is '15min', 15 minutes = 0.0104167 days
        timeframe_minutes = 15
        max_drawdown_duration_days = max(drawdown_durations) * (timeframe_minutes / (60*24))
        average_drawdown_duration_days = np.mean(drawdown_durations) * (timeframe_minutes / (60*24))
    else:
        max_drawdown_duration_days = 0
        average_drawdown_duration_days = 0
    
    exposure_time_percentage = (exposure_time / total_time) * 100
    total_return = capital - INITIAL_CAPITAL
    total_return_percentage = (total_return / INITIAL_CAPITAL) * 100
    duration_days = (df.index.max().date() - df.index.min().date()).days
    if duration_days > 0:
        annualized_return_percentage = ((capital / INITIAL_CAPITAL) ** (365.25 / duration_days) - 1) * 100
    else:
        annualized_return_percentage = 0.0
    
    # Benchmark
    benchmark_return = calculate_benchmark_return(BACKTEST_START_DATE, BACKTEST_END_DATE)
    benchmark_return_percentage = float(benchmark_return)
    
    returns = equity_df['Capital'].pct_change().dropna()
    volatility = returns.std()
    # Approximation: 252 trading days * (6.5 hours per day * 4 bars per hour for 15-min bars)
    periods_per_year = 252 * (6.5 * 4)
    volatility_annual = volatility * np.sqrt(periods_per_year) * 100
    
    winning_trades = [t for t in trades if t['Profit'] is not None and t['Profit'] > 0]
    losing_trades = [t for t in trades if t['Profit'] is not None and t['Profit'] <= 0]
    
    gross_profit = sum(t['Profit'] for t in winning_trades)
    gross_loss = -sum(t['Profit'] for t in losing_trades)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
    
    risk_free_rate = 0.0
    if returns.std() != 0:
        sharpe_ratio = ((returns.mean() - risk_free_rate) / returns.std()) * np.sqrt(periods_per_year)
    else:
        sharpe_ratio = np.nan
    
    downside_returns = returns[returns < 0]
    if not downside_returns.empty and downside_returns.std() != 0:
        sortino_ratio = ((returns.mean() - risk_free_rate) / downside_returns.std()) * np.sqrt(periods_per_year)
    else:
        sortino_ratio = np.nan
    
    if max_drawdown != 0:
        calmar_ratio = total_return_percentage / max_drawdown
    else:
        calmar_ratio = np.nan
    
    max_drawdown_percentage = max_drawdown
    average_drawdown_percentage = np.mean(drawdowns) if drawdowns else 0.0
    
    # --- Results Summary ---
    logger.info("Backtest Completed.\n")
    
    print("Performance Summary:")
    results = {
        "Start Date": df.index.min().strftime("%Y-%m-%d"),
        "End Date": df.index.max().strftime("%Y-%m-%d"),
        "Exposure Time": f"{exposure_time_percentage:.2f}%",
        "Final Account Balance": f"${capital:,.2f}",
        "Equity Peak": f"${equity_peak:,.2f}",
        "Total Return": f"{total_return_percentage:.2f}%",
        "Annualized Return": f"{annualized_return_percentage:.2f}%",
        "Benchmark Return": f"{benchmark_return_percentage:.2f}%",
        "Volatility (Annual)": f"{volatility_annual:.2f}%",
        "Total Trades": len(trades),
        "Winning Trades": len(winning_trades),
        "Losing Trades": len(losing_trades),
        "Win Rate": f"{(len(winning_trades)/len(trades)*100) if trades else 0:.2f}%",
        "Profit Factor": f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "NaN",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "NaN",
        "Sortino Ratio": f"{sortino_ratio:.2f}" if not np.isnan(sortino_ratio) else "NaN",
        "Calmar Ratio": f"{calmar_ratio:.2f}" if not np.isnan(calmar_ratio) else "NaN",
        "Max Drawdown": f"{max_drawdown_percentage:.2f}%",
        "Average Drawdown": f"{average_drawdown_percentage:.2f}%",
        "Max Drawdown Duration": f"{max_drawdown_duration_days:.2f} days",
        "Average Drawdown Duration": f"{average_drawdown_duration_days:.2f} days",
    }
    
    for key, value in results.items():
        print(f"{key:25}: {value:>15}")

    # Optional: Plot equity curve
    # equity_df['Capital'].plot(title='Equity Curve')
    # plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    try:
        logger.info("Loading data...")
        df = load_data(CSV_FILE_PATH)
        
        logger.info("Filtering sessions...")
        df = filter_sessions(df, BACKTEST_START_DATE, BACKTEST_END_DATE)
        
        # If resampling is needed (the CSV is already in 15m, so not necessary here)
        # If your CSV is not in 15min, uncomment and adjust:
        """
        df = df.resample(TIMEFRAME).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'average': 'mean',
            'barCount': 'sum',
            'contract': 'first'
        }).dropna()
        """

        logger.info("Calculating Indicators...")
        df = calculate_donchian_channels(df, DONCHIAN_PERIOD)
        df = calculate_atr(df, ATR_PERIOD)
        df = calculate_volume_average(df, period=20)
        
        logger.info("Generating signals...")
        df = generate_signals(df)
        
        logger.info("Starting backtest...")
        backtest(df)
        
    except Exception as e:
        logger.error(f"An error occurred during backtesting: {e}")