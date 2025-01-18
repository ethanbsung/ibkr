import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz

# ============================
# Configuration Parameters
# ============================

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

# ============================
# Helper Functions
# ============================

def calculate_vwap(df):
    """Calculate the Volume-Weighted Average Price (VWAP)."""
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Cumulative_TP_Vol'] = (df['Typical_Price'] * df['Volume']).cumsum()
    df['Cumulative_Volume'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cumulative_TP_Vol'] / df['Cumulative_Volume']
    return df

def filter_trading_hours(df):
    """Filter DataFrame to include only regular trading hours (09:30 AM to 04:00 PM Eastern Time)."""
    # Check if the index is timezone-aware
    if df.index.tzinfo is None or df.index.tz is None:
        # Localize to Eastern Time if naive
        df = df.tz_localize(TRADING_TIMEZONE)
    else:
        # Convert to Eastern Time if already timezone-aware
        df = df.tz_convert(TRADING_TIMEZONE)
    
    # Filter based on time
    df = df.between_time(TRADING_START_TIME, TRADING_END_TIME)
    return df

def load_data(file_path, start_date, end_date):
    """Load and preprocess the data."""
    try:
        df = pd.read_csv(file_path, parse_dates=['Time'])
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()
    
    df = df.rename(columns={
        'Time': 'Datetime',
        'Last': 'Close'  # Ensure 'Close' is correctly named
    })
    
    # Ensure all required columns are present
    required_columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Data file must contain the following columns: {required_columns}")
        return pd.DataFrame()
    
    df = df[required_columns]  # Select relevant columns
    df = df.set_index('Datetime')
    
    # Localize or convert timezone to Eastern Time
    df = filter_trading_hours(df)
    
    # Filter data based on date range
    df = df.loc[start_date:end_date]
    
    if df.empty:
        print("Warning: No data available for the specified date range after filtering.")
    
    # Calculate VWAP
    df = calculate_vwap(df)
    
    return df

def calculate_metrics(trades, equity_curve, benchmark_curve):
    """Calculate performance metrics."""
    results = {}

    # Check if trades DataFrame is empty
    if trades.empty:
        print("No trades were executed during the backtest period.")
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

    # Start and End Dates
    results["Start Date"] = equity_curve.index.min().strftime('%Y-%m-%d')
    results["End Date"] = equity_curve.index.max().strftime('%Y-%m-%d')

    # Exposure Time
    total_time = (equity_curve.index.max() - equity_curve.index.min()).total_seconds()
    exposure_time = trades['Entry Time'].count() * (15 * 60)  # 15 minutes per trade
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

    # Benchmark Return (Assuming Buy and Hold)
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
    total_trades = trades.shape[0]
    results["Total Trades"] = total_trades

    # Winning and Losing Trades
    winning_trades = trades[trades['Profit'] > 0]
    losing_trades = trades[trades['Profit'] <= 0]
    winning_trades_count = winning_trades.shape[0]
    losing_trades_count = losing_trades.shape[0]
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
    max_drawdown_duration_days = max_drawdown_duration * (15 / 60 / 24)  # 15 minutes per trade
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
        average_drawdown_duration = (np.mean(drawdown_durations) * (15 / 60 / 24))
    else:
        average_drawdown_duration = 0
    results["Average Drawdown Duration"] = f"{average_drawdown_duration:.2f} days"

    return results

# ============================
# Backtest Strategy
# ============================

def backtest_vwap_pullback(df):
    """Backtest the VWAP Pullback Strategy with dynamic stop loss and take profit."""
    balance = INITIAL_BALANCE
    equity_curve = []
    position = None
    trades = []
    benchmark = df['Close']

    # Iterate over each bar
    for idx, row in df.iterrows():
        # Update Equity Curve
        equity_curve.append(balance)
        current_price = row['Close']

        if position:
            # Check for Stop Loss or Take Profit
            if position['type'] == 'long':
                if current_price <= position['stop_loss'] or current_price >= position['take_profit']:
                    # Exit Long Position
                    exit_price = current_price
                    profit_ticks = (exit_price - position['entry_price']) / TICK_SIZE
                    profit = profit_ticks * TICK_VALUE * POSITION_SIZE
                    balance += profit - (COMMISSION_PER_TRADE * 2)
                    trades.append({
                        'Entry Time': position['entry_time'],
                        'Exit Time': idx,
                        'Position': 'long',
                        'Entry Price': position['entry_price'],
                        'Exit Price': exit_price,
                        'Profit': profit
                    })
                    print(f"Long Exit: {idx} | Entry: {position['entry_price']} | Exit: {exit_price} | Profit: {profit:.2f}")
                    position = None
            elif position['type'] == 'short':
                if current_price >= position['stop_loss'] or current_price <= position['take_profit']:
                    # Exit Short Position
                    exit_price = current_price
                    profit_ticks = (position['entry_price'] - exit_price) / TICK_SIZE
                    profit = profit_ticks * TICK_VALUE * POSITION_SIZE
                    balance += profit - (COMMISSION_PER_TRADE * 2)
                    trades.append({
                        'Entry Time': position['entry_time'],
                        'Exit Time': idx,
                        'Position': 'short',
                        'Entry Price': position['entry_price'],
                        'Exit Price': exit_price,
                        'Profit': profit
                    })
                    print(f"Short Exit: {idx} | Entry: {position['entry_price']} | Exit: {exit_price} | Profit: {profit:.2f}")
                    position = None

        # Strategy Logic: Entry Conditions
        if not position:
            # Determine Trend
            if current_price > row['VWAP']:
                trend = 'bullish'
            elif current_price < row['VWAP']:
                trend = 'bearish'
            else:
                trend = 'neutral'

            # Check for Pullback within the last 15 minutes (3 bars)
            if trend == 'bullish':
                # Check if any of the last 3 bars have Low <= VWAP
                window = df.loc[:idx].tail(3)
                if (window['Low'] <= window['VWAP']).any():
                    # Check if current bar closes above VWAP
                    if current_price > row['VWAP']:
                        # Enter Long Position
                        entry_price = current_price
                        entry_time = idx
                        # Identify the bar where Low <= VWAP occurred
                        pullback_bar = window[window['Low'] <= window['VWAP']].iloc[-1]
                        # Set Stop Loss below the Low of the pullback bar with a 1-tick buffer
                        stop_loss = pullback_bar['Low'] - (TICK_SIZE * 1)
                        # Set Take Profit at 2x the distance from entry to stop loss
                        distance = entry_price - stop_loss
                        take_profit = entry_price + (2 * distance)
                        # Update balance for commissions
                        balance -= (COMMISSION_PER_TRADE * 2)  # Entry and planned exit commissions
                        # Record Position
                        position = {
                            'type': 'long',
                            'entry_price': entry_price,
                            'entry_time': entry_time,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        }
                        print(f"Long Entry: {entry_time} | Entry Price: {entry_price} | Stop Loss: {stop_loss} | Take Profit: {take_profit}")

            elif trend == 'bearish':
                # Check if any of the last 3 bars have High >= VWAP
                window = df.loc[:idx].tail(3)
                if (window['High'] >= window['VWAP']).any():
                    # Check if current bar closes below VWAP
                    if current_price < row['VWAP']:
                        # Enter Short Position
                        entry_price = current_price
                        entry_time = idx
                        # Identify the bar where High >= VWAP occurred
                        pullback_bar = window[window['High'] >= window['VWAP']].iloc[-1]
                        # Set Stop Loss above the High of the pullback bar with a 1-tick buffer
                        stop_loss = pullback_bar['High'] + (TICK_SIZE * 1)
                        # Set Take Profit at 2x the distance from entry to stop loss
                        distance = stop_loss - entry_price
                        take_profit = entry_price - (2 * distance)
                        # Update balance for commissions
                        balance -= (COMMISSION_PER_TRADE * 2)  # Entry and planned exit commissions
                        # Record Position
                        position = {
                            'type': 'short',
                            'entry_price': entry_price,
                            'entry_time': entry_time,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        }
                        print(f"Short Entry: {entry_time} | Entry Price: {entry_price} | Stop Loss: {stop_loss} | Take Profit: {take_profit}")

    # ============================
    # Main Function
    # ============================

def main():
    # Set Start and End Dates
    start_date = pd.to_datetime(START_DATE)
    end_date = pd.to_datetime(END_DATE)

    # Load Data
    print("Loading data...")
    df = load_data(DATA_FILE, start_date, end_date)
    if df.empty:
        print("No data available for the specified date range after filtering. Exiting backtest.")
        return
    else:
        print(f"Data Loaded: {df.shape[0]} bars within trading hours.")

    # Handle potential missing Volume data
    if df['Volume'].isnull().any():
        df['Volume'] = df['Volume'].fillna(0)
        print("Warning: Missing Volume data found and filled with 0.")

    # Backtest Strategy
    print("Starting backtest...")
    trades, equity_curve, benchmark_curve = backtest_vwap_pullback(df)

    # Calculate Metrics
    print("Calculating performance metrics...")
    metrics = calculate_metrics(trades, equity_curve, benchmark_curve)

    # Print Performance Summary
    print("\nPerformance Summary:")
    for key, value in metrics.items():
        print(f"{key:25}: {value:>15}")

    # Optional: Plot Equity Curve
    if not equity_curve.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve, label='Equity Curve')
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Balance ($)')
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print("Equity curve is empty. No plot to display.")

    # Optional: Plot Benchmark Curve
    if not benchmark_curve.empty and not equity_curve.empty:
        plt.figure(figsize=(12, 6))
        benchmark_scaled = benchmark_curve / benchmark_curve.iloc[0] * INITIAL_BALANCE
        plt.plot(benchmark_scaled, label='Benchmark (Buy & Hold)', alpha=0.7)
        plt.plot(equity_curve, label='Equity Curve', alpha=0.7)
        plt.title('Equity Curve vs Benchmark')
        plt.xlabel('Date')
        plt.ylabel('Balance ($)')
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print("Benchmark or Equity curve is empty. No comparison plot to display.")

    # Optional: Print Trades
    if not trades.empty:
        print("\nTrades:")
        print(trades)
    else:
        print("\nNo trades were executed during the backtest period.")

# ============================
# Execute Main Function
# ============================

if __name__ == "__main__":
    main()