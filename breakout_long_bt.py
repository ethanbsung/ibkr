import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import time, timedelta

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)  # Change to DEBUG for more detailed logs
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
#              CONFIGURATION & USER PARAMETERS
# -------------------------------------------------------------
INTRADAY_DATA_FILE = 'es_1m_data.csv'  # 1-minute CSV path

# General Backtesting Parameters
INITIAL_CASH       = 5000
ES_MULTIPLIER      = 5      # 1 ES point = $5 per contract for ES
STOP_LOSS_POINTS   = 1
TAKE_PROFIT_POINTS = 12
POSITION_SIZE      = 1      # Can be fractional if desired
COMMISSION         = 1.24   # Commission per trade
ONE_TICK           = 0.25   # For ES, 1 tick = 0.25

# Rolling window for the 30-minute bars
ROLLING_WINDOW = 7

# Backtest date range
BACKTEST_START = "2012-01-01"
BACKTEST_END   = "2019-12-23"

# -------------------------------------------------------------
#              STEP 1: LOAD 1-MIN DATA
# -------------------------------------------------------------
def load_data(csv_file):
    """
    Loads 1-minute intraday data from CSV, parses the Time column as datetime,
    sorts by time, sets index, and does basic cleanup.
    """
    try:
        df = pd.read_csv(
            csv_file,
            parse_dates=['Time'],
            dayfirst=False,  # Change to True if your CSV uses day-first format
            na_values=['', 'NA', 'NaN']
        )
        
        if 'Time' not in df.columns:
            logger.error("CSV does not contain a 'Time' column.")
            raise ValueError("Missing 'Time' column.")
        
        if not np.issubdtype(df['Time'].dtype, np.datetime64):
            logger.error("'Time' column not parsed as datetime. Check the date format.")
            raise TypeError("'Time' column not datetime.")
        
        # If there's a timezone, remove it
        if df['Time'].dt.tz is not None:
            df['Time'] = df['Time'].dt.tz_convert(None)
        
        # Sort by 'Time' and set as index
        df.sort_values('Time', inplace=True)
        df.set_index('Time', inplace=True)
        
        # Drop columns if not needed
        if 'Symbol' in df.columns:
            df.drop(columns=['Symbol'], inplace=True)
        if 'Last' in df.columns:
            df.rename(columns={'Last': 'Close'}, inplace=True)
        
        # Optional: drop some columns
        for col in ['Change', '%Chg', 'Open Int']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        
        return df
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise

# -------------------------------------------------------------
#   STEP 2: PREPARE DATA (30-MIN ROLLING HIGH + MERGE TO 1-MIN)
# -------------------------------------------------------------
def prepare_data(df_1m, rolling_window=ROLLING_WINDOW):
    """
    1) Resample df_1m to 30-min bars
    2) Compute the rolling high over 'rolling_window' previous 30-min bars
    3) Forward-fill that rolling high back onto the 1-min DataFrame
    """
    # Resample to 30-minute bars
    df_30m = df_1m.resample('30min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    # Rolling high of the past 'rolling_window' 30-min bars (excluding current bar => shift(1))
    df_30m['Rolling_High'] = (
        df_30m['High'].shift(1)
                      .rolling(window=rolling_window, min_periods=rolling_window)
                      .max()
    )
    # Drop rows where Rolling_High is NaN
    df_30m.dropna(subset=['Rolling_High'], inplace=True)
    
    # We'll merge the 30-min Rolling_High into 1-min data via forward-fill
    df_1m['Rolling_High'] = df_30m['Rolling_High'].reindex(df_1m.index, method='ffill')
    
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
    - We buy 1 tick above the 30-min Rolling High when broken.
    - We manage stops and targets on a 1-minute basis.
    """
    # Filter date range
    start_time = pd.to_datetime(start_date)
    end_time   = pd.to_datetime(end_date)
    df_filtered = df_1m.loc[start_time:end_time].copy()
    
    if df_filtered.empty:
        logger.error("No data after filtering by date range.")
        return None
    
    # Verify that Rolling_High was computed
    if 'Rolling_High' not in df_filtered.columns:
        logger.error("Rolling_High column not found. Did you run prepare_data()?")
        return None
    
    # Initialize backtest variables
    cash = initial_cash
    position = None
    trade_results = []
    balance_series = [cash]
    balance_dates  = [df_filtered.index[0]]
    
    total_bars = len(df_filtered)
    active_bars = 0  # For measuring "exposure"

    for idx, (current_time, row) in enumerate(df_filtered.iterrows()):
        rolling_high_value = row['Rolling_High']
        
        # Skip if Rolling High is NaN (shouldn't happen if we've forward-filled + dropped NaN)
        if pd.isna(rolling_high_value):
            continue
        
        if position is None:
            # Only trade between 09:30 and 16:00
            if time(9, 30) <= current_time.time() < time(16, 0):
                breakout_price = rolling_high_value + one_tick
                # If the 1-min high >= breakout_price => fill
                if row['High'] >= breakout_price:
                    entry_price = breakout_price
                    stop_price  = entry_price - stop_loss_points
                    target_price= entry_price + take_profit_points
                    
                    position = {
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_price,
                        'take_profit': target_price
                    }
                    active_bars += 1
                    logger.info(f"[ENTRY] Long entered at {entry_price} on {current_time}")
        else:
            # Manage open position
            current_high = row['High']
            current_low  = row['Low']
            exit_time    = current_time
            
            # Check Stop Loss
            if current_low <= position['stop_loss']:
                exit_price = position['stop_loss']
                pnl = ((exit_price - position['entry_price']) 
                       * position_size * es_multiplier) - commission
                cash += pnl
                trade_results.append(pnl)
                balance_series.append(cash)
                balance_dates.append(exit_time)
                logger.info(f"[STOP LOSS] Exit at {exit_price} on {exit_time}, PnL: ${pnl:,.2f}")
                position = None
            
            # If still open, check Take Profit
            elif current_high >= position['take_profit']:
                exit_price = position['take_profit']
                pnl = ((exit_price - position['entry_price']) 
                       * position_size * es_multiplier) - commission
                cash += pnl
                trade_results.append(pnl)
                balance_series.append(cash)
                balance_dates.append(exit_time)
                logger.info(f"[TAKE PROFIT] Exit at {exit_price} on {exit_time}, PnL: ${pnl:,.2f}")
                position = None
        
        # Record equity if no position or if we just closed
        if position is None:
            if len(balance_series) == len(balance_dates):
                balance_series.append(cash)
                balance_dates.append(current_time)
    
    exposure_time_percentage = (active_bars / total_bars) * 100
    
    balance_df = pd.DataFrame({
        'Datetime': balance_dates,
        'Equity': balance_series
    }).set_index('Datetime').sort_index()
    
    return {
        'cash': cash,
        'trade_results': trade_results,
        'balance_df': balance_df,
        'exposure_time_pct': exposure_time_percentage,
        'df_filtered': df_filtered  # We'll use this for benchmark calculations
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
    """
    if not result_dict:
        logger.error("Result dictionary is empty. Cannot compute metrics.")
        return
    
    cash         = result_dict['cash']
    trade_results= result_dict['trade_results']
    balance_df   = result_dict['balance_df']
    exposure_pct = result_dict['exposure_time_pct']
    df_filtered  = result_dict['df_filtered']
    
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
    
    # Calculate drawdown durations
    drawdown_periods = drawdown[drawdown < 0]
    if not drawdown_periods.empty:
        # Group consecutive negative drawdown periods
        # We'll treat each 1-min step as 1 'period'. For 30-min, we used 0.0208333 days,
        # but here we can do 1 minute = 1/1440 days as a rough approach if we want days.
        end_dates = drawdown_periods.index.to_series().diff().ne(pd.Timedelta('1min')).cumsum()
        drawdown_groups = drawdown_periods.groupby(end_dates)
        drawdown_durations = drawdown_groups.size()
        
        # 1 minute = 1/1440 days
        max_drawdown_duration_days    = drawdown_durations.max() * (1.0 / 1440.0)
        average_drawdown_duration_days= drawdown_durations.mean() * (1.0 / 1440.0)
    else:
        max_drawdown_duration_days    = 0
        average_drawdown_duration_days= 0
    
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
    
    # Sharpe Ratio (approx, using ~ 252 trading days * ~6.5 hrs * 60 min = ~ 98k minutes
    # This is not an exact formula, but a ballpark. Adjust to your preference.
    if returns.std() != 0:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 6.5 * 60)
    else:
        sharpe_ratio = 0
    
    # Sortino Ratio
    mar = 0
    strategy_returns = np.array(trade_results) / initial_cash
    downside_returns = np.where(strategy_returns < mar, strategy_returns - mar, 0)
    expected_return  = np.mean(strategy_returns) if len(strategy_returns) > 0 else 0
    downside_deviation = np.std(downside_returns)
    if downside_deviation != 0:
        sortino_ratio = (expected_return - mar) / downside_deviation * np.sqrt(252 * 6.5 * 60)
    else:
        sortino_ratio = np.nan
    
    # Calmar Ratio (annualized return / abs(max_drawdown))
    # We'll approximate the days in the dataset:
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
    benchmark_equity = benchmark_equity.reindex(balance_df.index, method='ffill')
    # Fill any remaining NaNs
    benchmark_equity.fillna(method='ffill', inplace=True)
    
    # Volatility (Annual)
    # We'll again approximate by scaling the std dev of daily returns.
    vol_annual = returns.std() * np.sqrt(252 * 6.5 * 60) * 100
    
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
        "Max Drawdown Duration": f"{max_drawdown_duration_days:.2f} days",
        "Average Drawdown Duration": f"{average_drawdown_duration_days:.2f} days",
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

# -------------------------------------------------------------
#               STEP 5: MAIN SCRIPT
# -------------------------------------------------------------
def main():
    # 1) Load the 1-minute data
    df_intraday = load_data(INTRADAY_DATA_FILE)
    print("Full Data Range:", df_intraday.index.min(), "to", df_intraday.index.max())
    
    # 2) Prepare data (30-min Rolling High -> forward-fill -> 1-min)
    df_prepared = prepare_data(df_intraday, rolling_window=ROLLING_WINDOW)
    df_prepared.dropna(subset=['Rolling_High'], inplace=True)
    
    # 3) Run the backtest on 1-minute data
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
    
    # 4) Compute extended metrics and plot
    compute_and_plot_metrics(backtest_result)


if __name__ == '__main__':
    main()