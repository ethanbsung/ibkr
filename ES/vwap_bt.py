import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import logging
import sys
import gc

# --- Configuration Parameters ---
DATA_FILE = "Data/es_5m_data.csv"  # Path to your data file
TIMEFRAME = '5T'  # 5-minute timeframe
INITIAL_BALANCE = 5000  # Starting with $5,000
POSITION_SIZE = 1  # Number of contracts per trade
TICK_SIZE = 0.25  # ES tick size (0.25 for ES futures)
TICK_VALUE = 1.25  # ES tick value per tick (adjusted for mini ES)
COMMISSION_PER_TRADE = 0.62  # Commission per contract per side

# Backtest Period (Set Start and End Dates Here)
START_DATE = "2021-05-04"
END_DATE = "2024-12-10"

# Regular Trading Hours (Eastern Time)
TRADING_START_TIME = "09:30"
TRADING_END_TIME = "16:00"

# Timezone Information
TRADING_TIMEZONE = 'US/Eastern'  # Adjust if necessary

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# --- Helper Functions ---

def calculate_vwap(df):
    """
    Calculate the Volume-Weighted Average Price (VWAP) resetting daily.
    
    Parameters:
        df (pd.DataFrame): DataFrame with price and volume data indexed by datetime.
    
    Returns:
        pd.DataFrame: DataFrame with an added 'VWAP' column.
    """
    # Ensure the index is in UTC and localized
    if df.index.tz is None:
        df = df.tz_localize('UTC')
    
    # Reset VWAP daily
    df = df.copy()
    df['Date'] = df.index.date  # Extract date for grouping
    
    # Calculate Typical Price
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    # Calculate Cumulative TP_Vol and Cumulative Volume per day
    df['Cumulative_TP_Vol'] = df.groupby('Date')['Typical_Price'].transform(lambda x: (x * df.loc[x.index, 'Volume']).cumsum())
    df['Cumulative_Volume'] = df.groupby('Date')['Volume'].transform('cumsum')
    
    # Calculate VWAP
    df['VWAP'] = df['Cumulative_TP_Vol'] / df['Cumulative_Volume']
    
    # Drop helper columns
    df.drop(columns=['Date', 'Typical_Price', 'Cumulative_TP_Vol', 'Cumulative_Volume'], inplace=True)
    
    return df

def filter_trading_hours(df):
    """
    Filters the DataFrame to include only Regular Trading Hours (09:30 AM to 04:00 PM ET) on weekdays.

    Parameters:
        df (pd.DataFrame): The input DataFrame with a datetime index.

    Returns:
        pd.DataFrame: The filtered DataFrame containing only RTH data.
    """
    # Define US/Eastern timezone
    eastern = pytz.timezone(TRADING_TIMEZONE)

    # Ensure the index is timezone-aware
    if df.index.tzinfo is None or df.index.tz is None:
        # Localize to Eastern Time if naive
        df = df.tz_localize(eastern)
        logger.debug("Localized naive datetime index to US/Eastern.")
    else:
        # Convert to Eastern Time if already timezone-aware
        df = df.tz_convert(eastern)
        logger.debug("Converted timezone-aware datetime index to US/Eastern.")

    # Filter for weekdays (Monday=0 to Friday=4)
    df_eastern = df[df.index.weekday < 5]

    # Filter for RTH hours: 09:30 to 16:00
    df_rth = df_eastern.between_time(TRADING_START_TIME, TRADING_END_TIME)

    # Convert back to UTC for consistency in further processing
    df_rth = df_rth.tz_convert('UTC')

    return df_rth

def load_data(file_path, start_date, end_date):
    """Load and preprocess the data."""
    try:
        logger.info(f"Attempting to load data from {file_path}...")
        df = pd.read_csv(file_path, parse_dates=['Time'])

        # Rename columns for consistency
        df = df.rename(columns={
            'Time': 'Datetime',
            'Last': 'Close'  # Ensure 'Close' is correctly named
        })

        # Ensure required columns are present
        required_columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Error: Missing required columns: {required_columns}")
            return pd.DataFrame()

        # Select and clean relevant columns
        df = df[required_columns]
        df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
        df['High'] = pd.to_numeric(df['High'], errors='coerce')
        df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

        # Handle missing or invalid values
        initial_len = len(df)
        df.dropna(inplace=True)  # Drop rows with NaN values
        logger.info(f"Dropped {initial_len - len(df)} rows with invalid data.")

        # Set index and filter by date range
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.set_index('Datetime')
        df = df.loc[start_date:end_date]

        # Filter to trading hours
        df = filter_trading_hours(df)

        # Calculate VWAP
        df = calculate_vwap(df)

        if df.empty:
            logger.warning("No data available after preprocessing and filtering.")
        return df

    except FileNotFoundError:
        logger.error(f"Error: The file {file_path} was not found.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        logger.error(f"Error: The file {file_path} is empty.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading the CSV '{file_path}': {e}")
        return pd.DataFrame()

def calculate_metrics(trades, equity_curve, benchmark_curve):
    """
    Calculate performance metrics.

    Parameters:
        trades (list of dict): List containing trade details.
        equity_curve (pd.Series): Series representing account balance over time.
        benchmark_curve (pd.Series): Series representing benchmark performance.

    Returns:
        dict: Dictionary containing performance metrics.
    """
    results = {}

    # Check if trades list is empty
    if not trades:
        logger.warning("No trades were executed during the backtest period.")
        results = {
            "Start Date": equity_curve.index.min().strftime('%Y-%m-%d') if not equity_curve.empty else "N/A",
            "End Date": equity_curve.index.max().strftime('%Y-%m-%d') if not equity_curve.empty else "N/A",
            "Exposure Time": "0.00%",
            "Final Account Balance": f"${INITIAL_BALANCE:,.2f}",
            "Equity Peak": f"${INITIAL_BALANCE:,.2f}",
            "Total Return": "0.00%",
            "Annualized Return": "0.00%",
            "Benchmark Return": f"{((benchmark_curve.iloc[-1] - benchmark_curve.iloc[0]) / benchmark_curve.iloc[0] * 100):.2f}%" if not benchmark_curve.empty else "N/A",
            "Volatility (Annual)": "0.00%",
            "Total Trades": 0,
            "Winning Trades": 0,
            "Losing Trades": 0,
            "Win Rate": "0.00%",
            "Profit Factor": "N/A",
            "Sharpe Ratio": "N/A",
            "Sortino Ratio": "N/A",
            "Calmar Ratio": "N/A",
            "Max Drawdown": "0.00%",
            "Average Drawdown": "0.00%",
            "Max Drawdown Duration": "0.00 days",
            "Average Drawdown Duration": "0.00 days",
        }
        return results

    # Convert trades list to DataFrame
    trades_df = pd.DataFrame(trades)

    # Remove trades without exit (if any)
    trades_df = trades_df.dropna(subset=['Exit Time'])

    # Start and End Dates
    results["Start Date"] = equity_curve.index.min().strftime('%Y-%m-%d')
    results["End Date"] = equity_curve.index.max().strftime('%Y-%m-%d')

    # Exposure Time
    total_time = (equity_curve.index.max() - equity_curve.index.min()).total_seconds()
    exposure_time = len(trades_df) * (5 * 60)  # 5 minutes per trade
    exposure_time_percentage = (exposure_time / total_time) * 100 if total_time > 0 else 0
    results["Exposure Time"] = f"{exposure_time_percentage:.2f}%"

    # Final Account Balance
    final_balance = equity_curve.iloc[-1]
    results["Final Account Balance"] = f"${final_balance:,.2f}"

    # Equity Peak
    equity_peak_final = equity_curve.max()
    results["Equity Peak"] = f"${equity_peak_final:,.2f}"

    # Total Return
    total_return = ((final_balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    results["Total Return"] = f"{total_return:.2f}%"

    # Annualized Return
    days = (equity_curve.index.max() - equity_curve.index.min()).days
    if days > 0:
        annualized_return = (1 + total_return / 100) ** (365.0 / days) - 1
        annualized_return *= 100
    else:
        annualized_return = 0
    results["Annualized Return"] = f"{annualized_return:.2f}%"

    # Benchmark Return (Buy & Hold)
    if not benchmark_curve.empty:
        benchmark_return = ((benchmark_curve.iloc[-1] - benchmark_curve.iloc[0]) / benchmark_curve.iloc[0]) * 100
        results["Benchmark Return"] = f"{benchmark_return:.2f}%"
    else:
        results["Benchmark Return"] = "N/A"

    # Volatility (Annual)
    returns = equity_curve.pct_change().dropna()
    if not returns.empty:
        volatility_daily = returns.std()
        # Assuming 252 trading days per year and 78 5-minute bars per day
        volatility_annual = volatility_daily * np.sqrt(252 * 78)
        results["Volatility (Annual)"] = f"{volatility_annual:.2f}%"
    else:
        results["Volatility (Annual)"] = "0.00%"

    # Total Trades
    total_trades = len(trades_df)
    results["Total Trades"] = total_trades

    # Winning and Losing Trades
    winning_trades = trades_df[trades_df['Profit'] > 0]
    losing_trades = trades_df[trades_df['Profit'] <= 0]
    winning_trades_count = len(winning_trades)
    losing_trades_count = len(losing_trades)
    results["Winning Trades"] = winning_trades_count
    results["Losing Trades"] = losing_trades_count

    # Win Rate
    win_rate = (winning_trades_count / total_trades) * 100 if total_trades > 0 else 0
    results["Win Rate"] = f"{win_rate:.2f}%"

    # Profit Factor
    gross_profit = winning_trades['Profit'].sum()
    gross_loss = abs(losing_trades['Profit'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss != 0 else np.inf
    results["Profit Factor"] = f"{profit_factor:.2f}" if gross_loss != 0 else "Infinity"

    # Sharpe Ratio
    if not returns.empty and returns.std() != 0:
        risk_free_rate = 0.0  # Assuming risk-free rate is 0
        sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252 * 78)
        results["Sharpe Ratio"] = f"{sharpe_ratio:.2f}"
    else:
        results["Sharpe Ratio"] = "N/A"

    # Sortino Ratio
    if not returns.empty:
        downside_returns = returns[returns < 0]
        if not downside_returns.empty and downside_returns.std() != 0:
            sortino_ratio = (returns.mean() - 0.0) / downside_returns.std() * np.sqrt(252 * 78)
            results["Sortino Ratio"] = f"{sortino_ratio:.2f}"
        else:
            results["Sortino Ratio"] = "Infinity"
    else:
        results["Sortino Ratio"] = "N/A"

    # Calmar Ratio
    min_equity = equity_curve.min()
    if min_equity < INITIAL_BALANCE and (initial_drawdown := ((min_equity - INITIAL_BALANCE) / INITIAL_BALANCE * 100)) != 0:
        calmar_ratio = total_return / abs(initial_drawdown)
    else:
        calmar_ratio = np.inf
    results["Calmar Ratio"] = f"{calmar_ratio:.2f}" if min_equity < INITIAL_BALANCE else "Infinity"

    # Max Drawdown
    equity_peak = equity_curve.cummax()
    drawdown = (equity_curve - equity_peak) / equity_peak
    max_drawdown = drawdown.min() * 100
    results["Max Drawdown"] = f"{max_drawdown:.2f}%" if not drawdown.empty else "0.00%"

    # Average Drawdown
    average_drawdown = drawdown.mean() * 100 if not drawdown.empty else 0
    results["Average Drawdown"] = f"{average_drawdown:.2f}%" if not drawdown.empty else "0.00%"

    # Max Drawdown Duration
    drawdown_duration = 0
    max_drawdown_duration = 0
    temp_duration = 0
    for dd in drawdown:
        if dd < 0:
            temp_duration += 1
            if temp_duration > max_drawdown_duration:
                max_drawdown_duration = temp_duration
        else:
            temp_duration = 0

    max_drawdown_duration_minutes = max_drawdown_duration * 5  # 5 minutes per bar
    max_drawdown_duration_days = max_drawdown_duration_minutes / (60 * 24)
    results["Max Drawdown Duration"] = f"{max_drawdown_duration_days:.2f} days"

    # Average Drawdown Duration
    drawdown_durations = []
    temp_duration = 0
    for dd in drawdown:
        if dd < 0:
            temp_duration += 1
        else:
            if temp_duration > 0:
                drawdown_durations.append(temp_duration)
                temp_duration = 0
    if temp_duration > 0:
        drawdown_durations.append(temp_duration)
    if drawdown_durations:
        average_drawdown_duration = (np.mean(drawdown_durations) * (5 / 60 / 24))
    else:
        average_drawdown_duration = 0
    results["Average Drawdown Duration"] = f"{average_drawdown_duration:.2f} days"

    return results

# --- Define Backtest Function ---
def backtest_vwap_bounce(df):
    """
    Backtest the VWAP Bounce Strategy based on sequential bar closes and a 30-minute window.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing price and VWAP data.
    
    Returns:
        tuple: (trades, equity_curve, benchmark_curve)
    """
    balance = INITIAL_BALANCE
    equity_curve = []
    position = None
    trades = []
    benchmark = df['Close']

    logger.info("Starting backtest loop...")
    df = df.copy()
    df['Prev_Close'] = df['Close'].shift(1)

    # Calculate Rolling Min and Max over the last 30 minutes (6 bars of 5 minutes each)
    df['Rolling_Min30'] = df['Low'].rolling(window=6, min_periods=1).min()
    df['Rolling_Max30'] = df['High'].rolling(window=6, min_periods=1).max()

    for idx, row in df.iterrows():
        # Update Equity Curve
        equity_curve.append(balance)
        current_price = row['Close']
        prev_close = row['Prev_Close']
        current_vwap = row['VWAP']
        rolling_min30 = row['Rolling_Min30']
        rolling_max30 = row['Rolling_Max30']

        if position:
            # Check for Stop Loss or Take Profit
            if position['type'] == 'long':
                if current_price <= position['stop_loss']:
                    # Exit at Stop Loss
                    exit_price = position['stop_loss']  # Set exit price to stop loss
                    profit_ticks = (exit_price - position['entry_price']) / TICK_SIZE
                    profit = profit_ticks * TICK_VALUE * POSITION_SIZE
                    balance += profit
                    balance -= COMMISSION_PER_TRADE  # Exit commission
                    # Update the last trade with exit details
                    trades[-1].update({
                        'Exit Time': idx,
                        'Exit Price': exit_price,
                        'Profit': profit - COMMISSION_PER_TRADE
                    })
                    logger.info(f"Long Exit at Stop Loss: {idx} | Entry: {position['entry_price']} | Exit: {exit_price} | Profit: {profit - COMMISSION_PER_TRADE:.2f}")
                    position = None
                elif current_price >= position['take_profit']:
                    # Exit at Take Profit
                    exit_price = position['take_profit']  # Set exit price to take profit
                    profit_ticks = (exit_price - position['entry_price']) / TICK_SIZE
                    profit = profit_ticks * TICK_VALUE * POSITION_SIZE
                    balance += profit
                    balance -= COMMISSION_PER_TRADE  # Exit commission
                    # Update the last trade with exit details
                    trades[-1].update({
                        'Exit Time': idx,
                        'Exit Price': exit_price,
                        'Profit': profit - COMMISSION_PER_TRADE
                    })
                    logger.info(f"Long Exit at Take Profit: {idx} | Entry: {position['entry_price']} | Exit: {exit_price} | Profit: {profit - COMMISSION_PER_TRADE:.2f}")
                    position = None

            elif position['type'] == 'short':
                if current_price >= position['stop_loss']:
                    # Exit at Stop Loss
                    exit_price = position['stop_loss']  # Set exit price to stop loss
                    profit_ticks = (position['entry_price'] - exit_price) / TICK_SIZE
                    profit = profit_ticks * TICK_VALUE * POSITION_SIZE
                    balance += profit
                    balance -= COMMISSION_PER_TRADE  # Exit commission
                    # Update the last trade with exit details
                    trades[-1].update({
                        'Exit Time': idx,
                        'Exit Price': exit_price,
                        'Profit': profit - COMMISSION_PER_TRADE
                    })
                    logger.info(f"Short Exit at Stop Loss: {idx} | Entry: {position['entry_price']} | Exit: {exit_price} | Profit: {profit - COMMISSION_PER_TRADE:.2f}")
                    position = None
                elif current_price <= position['take_profit']:
                    # Exit at Take Profit
                    exit_price = position['take_profit']  # Set exit price to take profit
                    profit_ticks = (position['entry_price'] - exit_price) / TICK_SIZE
                    profit = profit_ticks * TICK_VALUE * POSITION_SIZE
                    balance += profit
                    balance -= COMMISSION_PER_TRADE  # Exit commission
                    # Update the last trade with exit details
                    trades[-1].update({
                        'Exit Time': idx,
                        'Exit Price': exit_price,
                        'Profit': profit - COMMISSION_PER_TRADE
                    })
                    logger.info(f"Short Exit at Take Profit: {idx} | Entry: {position['entry_price']} | Exit: {exit_price} | Profit: {profit - COMMISSION_PER_TRADE:.2f}")
                    position = None

        # Strategy Logic: Entry Conditions
        if not position and not pd.isna(prev_close):
            # Define Conditions for Long Entry
            cond1_long = (prev_close <= current_vwap) and (current_price > current_vwap)
            cond2_long = (rolling_min30 <= current_vwap) and (current_price > current_vwap)

            # Define Conditions for Short Entry
            cond1_short = (prev_close >= current_vwap) and (current_price < current_vwap)
            cond2_short = (rolling_max30 >= current_vwap) and (current_price < current_vwap)

            # Enter Long Position if either condition is met
            if cond1_long or cond2_long:
                entry_price = current_price
                entry_time = idx
                stop_loss = rolling_min30 - TICK_SIZE  # 1-tick below the low of the last 30 minutes
                distance = entry_price - stop_loss
                take_profit = entry_price + (2 * distance)  # 2x risk
                balance -= COMMISSION_PER_TRADE  # Entry commission
                position = {
                    'type': 'long',
                    'entry_price': entry_price,
                    'entry_time': entry_time,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
                trades.append({
                    'Entry Time': entry_time,
                    'Exit Time': None,
                    'Position': 'long',
                    'Entry Price': entry_price,
                    'Exit Price': None,
                    'Profit': 0  # Placeholder, updated upon exit
                })
                logger.info(f"Long Entry: {entry_time} | Entry Price: {entry_price} | Stop Loss: {stop_loss} | Take Profit: {take_profit}")

            # Enter Short Position if either condition is met
            elif cond1_short or cond2_short:
                entry_price = current_price
                entry_time = idx
                stop_loss = rolling_max30 + TICK_SIZE  # 1-tick above the high of the last 30 minutes
                distance = stop_loss - entry_price
                take_profit = entry_price - (2 * distance)  # 2x risk
                balance -= COMMISSION_PER_TRADE  # Entry commission
                position = {
                    'type': 'short',
                    'entry_price': entry_price,
                    'entry_time': entry_time,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
                trades.append({
                    'Entry Time': entry_time,
                    'Exit Time': None,
                    'Position': 'short',
                    'Entry Price': entry_price,
                    'Exit Price': None,
                    'Profit': 0  # Placeholder, updated upon exit
                })
                logger.info(f"Short Entry: {entry_time} | Entry Price: {entry_price} | Stop Loss: {stop_loss} | Take Profit: {take_profit}")

    logger.info("Backtest loop completed.")
    return trades, pd.Series(equity_curve, index=df.index), benchmark

# --- Main Function ---
def main():
    logger.info("Loading data...")
    df = load_data(DATA_FILE, START_DATE, END_DATE)
    if df.empty:
        logger.error("No data available for the specified date range after filtering. Exiting backtest.")
        sys.exit(1)
    else:
        logger.info(f"Data Loaded: {df.shape[0]} bars within trading hours.")

    # Visual Inspection (Optional)
    plt.figure(figsize=(14, 7))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['VWAP'], label='VWAP', color='orange')
    plt.title('Price vs VWAP')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    logger.info("Starting backtest...")
    trades, equity_curve, benchmark_curve = backtest_vwap_bounce(df)

    logger.info("Calculating performance metrics...")
    metrics = calculate_metrics(trades, equity_curve, benchmark_curve)

    print("\nPerformance Summary:")
    for key, value in metrics.items():
        print(f"{key:25}: {value:>15}")

    # Optional: Plot Equity Curves
    if not equity_curve.empty and not benchmark_curve.empty:
        plt.figure(figsize=(14, 7))
        plt.plot(equity_curve, label='Strategy Equity', color='blue')
        plt.plot(
            (benchmark_curve / benchmark_curve.iloc[0]) * INITIAL_BALANCE,
            label='Benchmark (Buy & Hold)',
            color='orange'
        )
        plt.title('Equity Curve: VWAP Bounce Strategy vs Benchmark')
        plt.xlabel('Date')
        plt.ylabel('Account Balance ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Optional: Print Trades
    if trades:
        trades_df = pd.DataFrame(trades)
        # Drop trades that haven't exited yet
        trades_df = trades_df.dropna(subset=['Exit Time'])
        print("\nTrades:")
        print(trades_df)
    else:
        print("\nNo trades were executed during the backtest period.")

    logger.info("Backtest completed.")

# --- Execute ---
if __name__ == "__main__":
    main()