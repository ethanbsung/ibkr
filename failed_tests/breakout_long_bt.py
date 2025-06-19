import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import time, timedelta

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO,  # Set to DEBUG for detailed logs
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler()  # Logs will be printed to console
                    ])
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
#              CONFIGURATION & USER PARAMETERS
# -------------------------------------------------------------

# IMPORTANT FIXES APPLIED TO PREVENT LOOKAHEAD BIAS:
# 1. Rolling High Timing: 30m rolling high values are now only available 
#    to 1m bars AFTER the 30m period completes (timestamp shifted by +30min)
# 2. Entry Price Modeling: Added slippage to entry price for more realistic fills
# 3. Continuous Equity Tracking: Added unrealized P&L tracking for better drawdown analysis
# 4. PnL Calculation: Fixed to properly account for slippage without double-counting
# 
# REMAINING LIMITATIONS:
# - Assumes perfect execution at breakout level when 1m high touches trigger
# - May still be optimistic due to intrabar execution assumptions

INTRADAY_1M_DATA_FILE = 'Data/es_1m_data.csv'    # Path to 1-minute CSV data
INTRADAY_30M_DATA_FILE = 'Data/es_30m_data.csv'  # Path to 30-minute CSV data

# General Backtesting Parameters
INITIAL_CASH        = 10000
ES_MULTIPLIER       = 5       # 1 ES point = $5 per contract
STOP_LOSS_POINTS    = 8       # Stop loss in points
TAKE_PROFIT_POINTS  = 23      # Take profit in points
POSITION_SIZE       = 1       # Number of contracts per trade
COMMISSION          = 1.24    # Commission per trade
ONE_TICK            = 0.25    # Tick size for ES
SLIPPAGE            = 0.25    # Slippage in points

# Rolling window for the 30-minute bars
ROLLING_WINDOW = 30  # Number of 30-minute bars in the rolling window

# Backtest date range
BACKTEST_START = "2000-01-01"
BACKTEST_END   = "2020-01-01"

# -------------------------------------------------------------
#              STEP 1: LOAD DATA
# -------------------------------------------------------------
def load_data(csv_file, data_type='1m'):
    """
    Loads intraday data from CSV, parses the Time column as datetime,
    sorts by time, sets index, and performs basic cleanup.

    Parameters:
    - csv_file: Path to the CSV file.
    - data_type: '1m' for 1-minute data, '30m' for 30-minute data.

    Returns:
    - df: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(
            csv_file,
            parse_dates=['Time'],
            dayfirst=False,  # Change to True if your CSV uses day-first format
            na_values=['', 'NA', 'NaN']
        )
        
        if 'Time' not in df.columns:
            logger.error(f"CSV {csv_file} does not contain a 'Time' column.")
            raise ValueError("Missing 'Time' column.")
        
        if not np.issubdtype(df['Time'].dtype, np.datetime64):
            logger.error(f"'Time' column in {csv_file} not parsed as datetime. Check the date format.")
            raise TypeError("'Time' column not datetime.")
        
        # If there's a timezone, remove it
        if df['Time'].dt.tz is not None:
            df['Time'] = df['Time'].dt.tz_convert(None)
            logger.debug("Removed timezone from 'Time' column.")
        
        # Sort by 'Time' and set as index
        df.sort_values('Time', inplace=True)
        df.set_index('Time', inplace=True)
        logger.info(f"Loaded data from {csv_file} with {len(df)} rows.")
        
        # Drop unnecessary columns
        columns_to_drop = ['Symbol', 'Change', '%Chg', 'Open Int']
        for col in columns_to_drop:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
                logger.debug(f"Dropped column: {col}")
        
        # Rename 'Last' to 'Close' if present
        if 'Last' in df.columns:
            df.rename(columns={'Last': 'Close'}, inplace=True)
            logger.debug("Renamed 'Last' column to 'Close'.")
        
        return df
    except Exception as e:
        logger.error(f"Error loading CSV {csv_file}: {e}")
        raise

# -------------------------------------------------------------
#   STEP 2: PREPARE DATA USING 30-MIN DATA FOR ROLLING HIGH
# -------------------------------------------------------------
def prepare_data(df_1m, df_30m, rolling_window=ROLLING_WINDOW):
    """
    1) Computes Rolling_High over the previous 'rolling_window' 30-minute bars, excluding the current bar.
    2) Computes Prev_30m_High as the High of the previous 30-minute bar.
    3) Merges Rolling_High and Prev_30m_High into the 1-minute DataFrame properly without lookahead bias.
    4) Adds a '30m_bar' column to track the corresponding 30-minute bar for each 1-minute bar.

    Parameters:
    - df_1m: 1-minute DataFrame.
    - df_30m: 30-minute DataFrame.
    - rolling_window: Number of 30-minute bars in the rolling window.

    Returns:
    - df_1m: Updated 1-minute DataFrame with Rolling_High, Prev_30m_High, and 30m_bar.
    """
    # Compute Rolling_High (Exclude current 30-minute bar)
    df_30m['Rolling_High'] = (
        df_30m['High']
        .shift(1)  # Exclude current bar's high
        .rolling(window=rolling_window, min_periods=rolling_window)
        .max()
    )
    
    # Compute Prev_30m_High as the High of the previous 30-minute bar
    df_30m['Prev_30m_High'] = df_30m['High'].shift(1)
    
    # Drop rows where Rolling_High or Prev_30m_High is NaN
    df_30m.dropna(subset=['Rolling_High', 'Prev_30m_High'], inplace=True)
    
    logger.info(f"Computed Rolling_High and Prev_30m_High with {len(df_30m)} valid 30-minute bars.")
    
    # FIXED: Proper merge without lookahead bias
    # Create a copy of 30m data with timestamps shifted to the END of each 30m bar
    # This ensures rolling high is only available AFTER the 30m bar completes
    df_30m_shifted = df_30m[['Rolling_High', 'Prev_30m_High']].copy()
    
    # Shift timestamps to the end of each 30-minute period (next bar start)
    # This ensures rolling high calculated at 9:30 is only available at 10:00
    df_30m_shifted.index = df_30m_shifted.index + pd.Timedelta('30min')
    
    # Use reindex with forward fill, but now without lookahead bias
    df_1m['Rolling_High'] = df_30m_shifted['Rolling_High'].reindex(df_1m.index, method='pad')
    df_1m['Prev_30m_High'] = df_30m_shifted['Prev_30m_High'].reindex(df_1m.index, method='pad')
    
    logger.debug("Forward-filled Rolling_High and Prev_30m_High into 1-minute data without lookahead bias.")
    
    # Add a '30m_bar' column indicating the 30-minute bar each 1-minute bar belongs to
    df_1m['30m_bar'] = df_1m.index.floor('30min')
    
    logger.debug("Added '30m_bar' column to 1-minute data.")
    
    return df_1m

# -------------------------------------------------------------
#             STEP 3: BACKTEST ON 1-MIN DATA
# -------------------------------------------------------------
def backtest_1m(df_1m, 
                initial_cash=INITIAL_CASH,
                es_multiplier=ES_MULTIPLIER,
                stop_loss_points=STOP_LOSS_POINTS,
                take_profit_points=TAKE_PROFIT_POINTS,
                position_size=POSITION_SIZE,
                commission=COMMISSION,
                one_tick=ONE_TICK,
                start_date=BACKTEST_START,
                end_date=BACKTEST_END):
    """
    Run a backtest on 1-minute data where:
    - Enter a long position 1 tick above the 30-minute Rolling High when broken.
    - Manage stops and take profits on a 1-minute basis.
    - Limit to one trade per 30-minute bar.

    Parameters:
    - df_1m: Prepared 1-minute DataFrame with Rolling_High, Prev_30m_High, and 30m_bar.
    - Other parameters as defined.

    Returns:
    - result_dict: Dictionary containing backtest results and data.
    """
    # Filter date range AFTER preparing data
    start_time = pd.to_datetime(start_date)
    end_time   = pd.to_datetime(end_date)
    df_filtered = df_1m.loc[start_time:end_time].copy()
    
    if df_filtered.empty:
        logger.error("No data after filtering by date range.")
        return None
    
    # Verify that Rolling_High and Prev_30m_High were computed
    if 'Rolling_High' not in df_filtered.columns or 'Prev_30m_High' not in df_filtered.columns:
        logger.error("Rolling_High or Prev_30m_High column not found. Did you run prepare_data()?")
        return None
    
    logger.info(f"Backtesting from {start_time} to {end_time} with {len(df_filtered)} 1-minute bars.")
    
    # Initialize backtest variables
    cash = initial_cash
    position = None
    trade_results = []
    balance_series = [cash]
    balance_dates  = [df_filtered.index[0]]
    
    total_bars = len(df_filtered)
    active_trades = 0  # For measuring "exposure"
    
    # Initialize last_trade_30m_bar to None
    last_trade_30m_bar = None
    
    # Initialize previous_rolling_high to track new Rolling Highs
    previous_rolling_high = -np.inf
    
    # For plotting/debugging purposes, store points where breakout should occur
    breakout_points = []
    
    for idx, (current_time, row) in enumerate(df_filtered.iterrows()):
        rolling_high_value = row['Rolling_High']
        prev_30m_high = row['Prev_30m_High']
        current_30m_bar = row['30m_bar']
        
        # Skip if Rolling High or Prev_30m_High is NaN (shouldn't happen if we've forward-filled + dropped NaN)
        if pd.isna(rolling_high_value) or pd.isna(prev_30m_high):
            logger.debug(f"Skipped Time: {current_time} due to NaN in Rolling_High or Prev_30m_High.")
            continue
        
        # Determine current breakout price
        breakout_price = rolling_high_value + one_tick
        
        # Check eligibility:
        # 1. Breakout Price > Prev_30m_High
        # 2. No trade has been taken in the current 30-minute bar
        # 3. Rolling_High has increased since the last trade (OPTIONAL)
        # To increase trade frequency, consider removing this condition
        # eligible_for_entry = (breakout_price > prev_30m_high + one_tick) and (current_30m_bar != last_trade_30m_bar) and (rolling_high_value > previous_rolling_high)
        eligible_for_entry = (breakout_price > prev_30m_high + one_tick) and (current_30m_bar != last_trade_30m_bar)
        # Optionally, you can keep the rolling_high_value > previous_rolling_high condition
        # but it's causing fewer trades. You may want to experiment with it.
        
        # Debug logs for current bar
        logger.debug(f"Time: {current_time}")
        logger.debug(f"30m_bar: {current_30m_bar}")
        logger.debug(f"Rolling_High: {rolling_high_value}, Prev_30m_High: {prev_30m_high}")
        logger.debug(f"Breakout_Price: {breakout_price}")
        logger.debug(f"Eligible for Entry: {eligible_for_entry}")
        
        position_closed = False  # Flag to track if a position was closed in this iteration
        
        if position is not None:
            # Manage open position
            current_high = row['High']
            current_low  = row['Low']
            exit_time = current_time
            
            # Check Stop Loss
            if current_low <= position['stop_loss']:
                exit_price = position['stop_loss']
                # Entry price already includes slippage, so use it directly
                pnl = ((exit_price - position['entry_price']) 
                       * position_size * es_multiplier) - commission
                cash += pnl
                trade_results.append(pnl)
                balance_series.append(cash)
                balance_dates.append(exit_time)
                logger.info(f"[STOP LOSS] Exit at {exit_price} on {exit_time}, PnL: ${pnl:,.2f}")
                position = None
                position_closed = True  # Set flag
                # Optionally reset previous_rolling_high if needed
            
            # If still open, check Take Profit
            elif current_high >= position['take_profit']:
                exit_price = position['take_profit']
                # Entry price already includes slippage, so use it directly
                pnl = ((exit_price - position['entry_price']) 
                       * position_size * es_multiplier) - commission
                cash += pnl
                trade_results.append(pnl)
                balance_series.append(cash)
                balance_dates.append(exit_time)
                logger.info(f"[TAKE PROFIT] Exit at {exit_price} on {exit_time}, PnL: ${pnl:,.2f}")
                position = None
                position_closed = True  # Set flag
                # Optionally reset previous_rolling_high if needed
        
        if not position_closed and position is None and eligible_for_entry:
            # Only trade during Regular Trading Hours (09:30 - 16:00)
            if time(9, 30) <= current_time.time() < time(16, 0):
                # Entry Condition: High price >= breakout_price
                if row['High'] >= breakout_price:
                    # NOTE: This assumes we can get filled when high touches breakout level
                    # In reality, execution might be worse due to momentum and market impact
                    logger.debug(f"Attempting to enter trade at {current_time} with breakout_price {breakout_price}")
                    
                    # More realistic entry: assume we get filled with some slippage above breakout
                    entry_price = breakout_price + SLIPPAGE
                    stop_price  = entry_price - stop_loss_points
                    target_price= entry_price + take_profit_points
                    
                    position = {
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_price,
                        'take_profit': target_price
                    }
                    active_trades += 1
                    last_trade_30m_bar = current_30m_bar  # Update last trade 30m bar
                    # Optionally, track previous_rolling_high
                    # previous_rolling_high = rolling_high_value
                    logger.info(f"[ENTRY] Long entered at {entry_price} on {current_time}")
                    
                    # For debugging: mark this breakout
                    breakout_points.append(current_time)
                else:
                    logger.debug(f"No entry: High {row['High']} < Breakout Price {breakout_price}")
            else:
                logger.debug(f"No entry: Outside Regular Trading Hours at {current_time.time()}")
        
        # Record equity for every bar (including unrealized P&L when in position)
        if position is not None:
            # Calculate unrealized P&L using current close price
            current_close = row['Close']
            unrealized_pnl = ((current_close - position['entry_price']) 
                             * position_size * es_multiplier)
            current_equity = cash + unrealized_pnl
        else:
            current_equity = cash
        
        # Always update equity curve
        balance_series.append(current_equity)
        balance_dates.append(current_time)
    
    exposure_time_percentage = (active_trades / total_bars) * 100
    logger.info(f"Total Bars: {total_bars}, Active Trades (Trades Entered): {active_trades}")
    logger.info(f"Exposure Time Percentage: {exposure_time_percentage:.2f}%")
    
    balance_df = pd.DataFrame({
        'Datetime': balance_dates,
        'Equity': balance_series
    }).set_index('Datetime').sort_index()
    
    # For debugging: Save breakout points to a CSV
    if breakout_points:
        breakout_df = pd.DataFrame({'Breakout_Time': breakout_points})
        breakout_df.to_csv('breakout_points.csv', index=False)
        logger.info(f"Saved {len(breakout_points)} breakout points to 'breakout_points.csv'.")
    
    return {
        'cash': cash,
        'trade_results': trade_results,
        'balance_df': balance_df,
        'exposure_time_pct': exposure_time_percentage,
        'df_filtered': df_filtered,  # We'll use this for benchmark calculations
        'breakout_points': breakout_points  # For further analysis if needed
    }

# -------------------------------------------------------------
#             STEP 4: COMPUTE METRICS & PLOT
# -------------------------------------------------------------
def compute_and_plot_metrics(result_dict):
    """
    Takes the dictionary from backtest_1m() and computes:
      - Full suite of performance metrics
      - Benchmark equity curve
      - Plot of Strategy vs Benchmark
      - Plot of Rolling_High vs Close to visualize breakouts

    Parameters:
    - result_dict: Dictionary containing backtest results and data.
    """
    if not result_dict:
        logger.error("Result dictionary is empty. Cannot compute metrics.")
        return

    cash         = result_dict['cash']
    trade_results= result_dict['trade_results']
    balance_df   = result_dict['balance_df']
    exposure_pct = result_dict['exposure_time_pct']
    df_filtered  = result_dict['df_filtered']
    breakout_points = result_dict.get('breakout_points', [])

    if len(balance_df) < 2:
        logger.warning("Not enough points in balance_df to compute metrics or plot.")
        return

    # Basic Metrics
    initial_cash = INITIAL_CASH
    final_cash   = cash
    total_return_pct = ((final_cash - initial_cash) / initial_cash) * 100

    # Compute Rolling Max for Drawdown
    rolling_max = balance_df['Equity'].cummax()
    drawdown = (balance_df['Equity'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100  # percentage

    # Calculate drawdown durations using a robust method
    def calculate_drawdown_durations(drawdown_series):
        """
        Calculates maximum and average drawdown durations in days.

        Parameters:
        - drawdown_series: Pandas Series of drawdowns.

        Returns:
        - max_duration_days: Maximum drawdown duration in days.
        - avg_duration_days: Average drawdown duration in days.
        """
        # Identify the start and end of each drawdown period
        drawdown = drawdown_series
        drawdown_shift = drawdown.shift(1).fillna(0)

        # Start of drawdown: drawdown < 0 and previous >= 0
        start_drawdown = (drawdown < 0) & (drawdown_shift >= 0)

        # End of drawdown: drawdown >= 0 and previous < 0
        end_drawdown = (drawdown >= 0) & (drawdown_shift < 0)

        drawdown_starts = drawdown[start_drawdown].index
        drawdown_ends = drawdown[end_drawdown].index

        # Ensure that every start has an end
        if drawdown_starts.empty:
            return 0, 0

        if drawdown_ends.empty or (drawdown_starts[-1] > drawdown_ends[-1]):
            # If the last drawdown didn't end, assume it ends at the last timestamp
            drawdown_ends = drawdown_ends.append(pd.Index([drawdown.index[-1]]))

        # Calculate durations in days
        durations = (drawdown_ends - drawdown_starts) / pd.Timedelta('1D')  # Converts to float days

        # Convert durations to a numpy array for mean calculation
        durations = durations.to_numpy()

        max_duration_days = durations.max()
        avg_duration_days = durations.mean() if len(durations) > 0 else 0

        return max_duration_days, avg_duration_days

    max_drawdown_duration_days, average_drawdown_duration_days = calculate_drawdown_durations(drawdown)

    # We'll define 'average_drawdown' as the same as min drawdown for simplicity
    average_drawdown = drawdown.min() * 100

    # Profit Factor
    gross_profit = sum(p for p in trade_results if p > 0)
    gross_loss   = abs(sum(p for p in trade_results if p < 0))
    profit_factor= gross_profit / gross_loss if gross_loss != 0 else np.nan

    # Winning & Losing Trades
    winning_trades = [p for p in trade_results if p > 0]
    losing_trades  = [p for p in trade_results if p < 0]
    total_trades   = len(trade_results)
    win_rate = (len(winning_trades) / total_trades * 100) if total_trades else 0

    # Strategy returns for ratio calculations
    returns = balance_df['Equity'].pct_change().dropna()

    # Sharpe Ratio (annualized)
    if returns.std() != 0:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 6.5 * 60)  # Adjusted for intraday
    else:
        sharpe_ratio = 0

    # Sortino Ratio
    mar = 0  # Minimum Acceptable Return
    strategy_returns = np.array(trade_results) / initial_cash
    downside_returns = np.where(strategy_returns < mar, strategy_returns - mar, 0)
    expected_return  = np.mean(strategy_returns) if len(strategy_returns) > 0 else 0
    downside_deviation = np.std(downside_returns)
    if downside_deviation != 0:
        sortino_ratio = (expected_return - mar) / downside_deviation * np.sqrt(252 * 6.5 * 60)
    else:
        sortino_ratio = np.nan

    # Calmar Ratio (annualized return / abs(max_drawdown))
    # Approximate the number of days in the dataset
    days_in_period = (df_filtered.index[-1] - df_filtered.index[0]).days
    if days_in_period > 0:
        annualized_return_percentage = ((final_cash / initial_cash)**(365.0 / days_in_period) - 1) * 100
    else:
        annualized_return_percentage = 0.0

    if max_drawdown != 0:
        calmar_ratio = annualized_return_percentage / abs(max_drawdown)
    else:
        calmar_ratio = np.nan

    # Benchmark: Simple buy & hold on the same 1-min close, from start to end of df_filtered
    initial_close = df_filtered['Close'].iloc[0]
    final_close   = df_filtered['Close'].iloc[-1]
    benchmark_return = ((final_close - initial_close) / initial_close) * 100

    # Create a 1-min benchmark equity curve: (price / initial_price) * initial_cash
    benchmark_equity = (df_filtered['Close'] / initial_close) * initial_cash
    # Align it with balance_df (the strategy equity)
    benchmark_equity = benchmark_equity.reindex(balance_df.index, method='pad')
    # Fill any remaining NaNs
    benchmark_equity.fillna(method='ffill', inplace=True)

    # Volatility (Annual)
    vol_annual = returns.std() * np.sqrt(252 * 6.5 * 60) * 100  # Adjusted for intraday

    # Create results dictionary
    results = {
        "Start Date": df_filtered.index.min().strftime("%Y-%m-%d"),
        "End Date": df_filtered.index.max().strftime("%Y-%m-%d"),
        "Exposure Time": f"{exposure_pct:.2f}%",
        "Final Account Balance": f"${final_cash:,.2f}",
        "Equity Peak": f"${balance_df['Equity'].max():,.2f}",
        "Total Return": f"{total_return_pct:.2f}%",
        "Annualized Return": f"{annualized_return_percentage:.2f}%",
        "Benchmark Return": f"{benchmark_return:.2f}%",
        "Volatility (Annual)": f"{vol_annual:.2f}%",
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
        "Max Drawdown Duration": f"{max_drawdown_duration_days:.4f} days",
        "Average Drawdown Duration": f"{average_drawdown_duration_days:.4f} days",
    }

    # Print the Performance Summary
    print("\nPerformance Summary:")
    for key, value in results.items():
        print(f"{key:25}: {value:>15}")

    # Plot Strategy vs Benchmark
    if len(balance_df) < 2:
        logger.warning("Not enough data points to plot equity curves.")
        return

    equity_plot_df = pd.DataFrame({
        'Strategy': balance_df['Equity'],
        'Benchmark': benchmark_equity
    })

    plt.figure(figsize=(14, 7))
    plt.plot(equity_plot_df.index, equity_plot_df['Strategy'], label='Strategy Equity')
    plt.plot(equity_plot_df.index, equity_plot_df['Benchmark'], label='Benchmark Equity')
    plt.title('Equity Curve: Strategy vs Benchmark')
    plt.xlabel('Time')
    plt.ylabel('Account Balance ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Rolling_High vs Close to visualize breakouts
    plt.figure(figsize=(14, 7))
    plt.plot(df_filtered.index, df_filtered['Close'], label='Close Price', alpha=0.5)
    plt.plot(df_filtered.index, df_filtered['Rolling_High'], label='Rolling High', alpha=0.7)
    if breakout_points:
        # Ensure breakout_points are within df_filtered index
        valid_breakouts = [pt for pt in breakout_points if pt in df_filtered.index]
        if valid_breakouts:
            plt.scatter(valid_breakouts, df_filtered.loc[valid_breakouts, 'Close'], 
                        color='red', marker='^', label='Breakouts')
    plt.title('Close Price and Rolling High with Breakouts')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------
#               STEP 5: MAIN SCRIPT
# -------------------------------------------------------------
def main():
    # 1) Load the 1-minute data
    try:
        df_intraday_1m = load_data(INTRADAY_1M_DATA_FILE, data_type='1m')
        print("1-Minute Data Range:", df_intraday_1m.index.min(), "to", df_intraday_1m.index.max())
    except Exception as e:
        logger.error("Failed to load 1-minute data. Exiting.")
        return
    
    # 2) Load the 30-minute data
    try:
        df_intraday_30m = load_data(INTRADAY_30M_DATA_FILE, data_type='30m')
        print("30-Minute Data Range:", df_intraday_30m.index.min(), "to", df_intraday_30m.index.max())
    except Exception as e:
        logger.error("Failed to load 30-minute data. Exiting.")
        return
    
    # 3) Prepare data (merge Rolling_High and Prev_30m_High into 1-minute data)
    df_prepared = prepare_data(df_intraday_1m, df_intraday_30m, rolling_window=ROLLING_WINDOW)
    
    # 4) Run the backtest on 1-minute data
    backtest_result = backtest_1m(
        df_1m=df_prepared,
        initial_cash=INITIAL_CASH,
        es_multiplier=ES_MULTIPLIER,
        stop_loss_points=STOP_LOSS_POINTS,
        take_profit_points=TAKE_PROFIT_POINTS,
        position_size=POSITION_SIZE,
        commission=COMMISSION,
        one_tick=ONE_TICK,
        start_date=BACKTEST_START,
        end_date=BACKTEST_END
    )
    
    if not backtest_result:
        logger.error("Backtest returned None. Please check your data and parameters.")
        return
    
    # 5) Compute extended metrics and plot
    compute_and_plot_metrics(backtest_result)

if __name__ == '__main__':
    main()