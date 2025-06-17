import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, time

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# -------------------------------
# Parameters & User Settings
# -------------------------------

# Input file path for ES futures 1-minute data
data_file = "Data/es_5m_data.csv"  # File should include: Symbol, Time, Open, High, Low, Last, Volume

# Backtest parameters
initial_capital = 100000.0         # starting account balance in dollars
commission_per_trade = 1.24       # commission per round-trip trade (total)
commission_per_order = commission_per_trade / 2  # commission per order (entry or exit)
risk_target = 0.20                # 20% annual risk target (similar to rob_port chapters)
multiplier = 5                    # each point move is worth $5 per contract for full ES
tick_size = 0.25                  # ES minimum tick size

# Custom start and end date (format: 'YYYY-MM-DD')
start_date = '2008-08-01'
end_date   = '2025-01-01'

# Strategy parameters
opening_range_minutes = 1         # 1-minute opening range period
market_open_time = time(8, 30)    # 8:30 AM CT - equity market open
market_close_time = time(15, 0)   # 3:00 PM CT - close all positions

# Volatility calculation parameters (from rob_port)
rolling_vol_window = 256          # 256-day rolling volatility window
business_days_per_year = 256      # Business days per year for annualization
min_vol_floor = 0.05              # Minimum volatility floor (5% annually)

# -------------------------------
# Dynamic Position Sizing Functions (from rob_port)
# -------------------------------

def calculate_rolling_volatility(returns, window=256, min_vol_floor=0.05):
    """
    Calculate rolling annualized volatility with minimum floor.
    Similar to rob_port chapter 2-3 implementation.
    """
    # Calculate rolling standard deviation
    rolling_std = returns.rolling(window=window, min_periods=1).std()
    
    # Annualize the volatility
    annualized_vol = rolling_std * np.sqrt(business_days_per_year)
    
    # Apply minimum volatility floor
    annualized_vol = annualized_vol.clip(lower=min_vol_floor)
    
    return annualized_vol

def calculate_position_size(capital, multiplier, price, annualized_volatility, risk_target=0.2):
    """
    Calculate dynamic position size based on capital and volatility.
    
    Formula from rob_port: N = (Capital × τ) ÷ (Multiplier × Price × σ%)
    
    Parameters:
        capital (float): Current trading capital
        multiplier (float): Contract multiplier  
        price (float): Current price
        annualized_volatility (float): Annualized volatility
        risk_target (float): Target risk fraction
    
    Returns:
        float: Number of contracts (can be fractional)
    """
    if np.isnan(annualized_volatility) or annualized_volatility <= 0:
        return 0.0
    
    position_size = (capital * risk_target) / (multiplier * price * annualized_volatility)
    
    # Protect against extremely large positions
    if np.isinf(position_size) or position_size > 1000:
        return 0.0
    
    return position_size

# -------------------------------
# Optimized Data Preparation
# -------------------------------

print("Loading and preparing data...")
# Read 1-minute ES futures data
data = pd.read_csv(data_file)
data['Time'] = pd.to_datetime(data['Time'])
data.sort_values('Time', inplace=True)

# Filter data based on custom date range
data = data[(data['Time'] >= start_date) & (data['Time'] <= end_date)].copy()

# Add date and time columns for easier filtering
data['Date'] = data['Time'].dt.date
data['TimeOnly'] = data['Time'].dt.time

# Filter for relevant trading hours (8:30 AM to 3:00 PM CT)
data = data[
    (data['TimeOnly'] >= market_open_time) & 
    (data['TimeOnly'] <= market_close_time)
].reset_index(drop=True)

logger.info(f"Loaded {len(data)} 1-minute bars from {data['Time'].min()} to {data['Time'].max()}")

# Create minute number within each day for faster processing
data['MinuteOfDay'] = data.groupby('Date').cumcount() + 1

# -------------------------------
# Daily Data Preparation for Volatility Calculation
# -------------------------------

# Create daily closing prices for volatility calculation
daily_data = data.groupby('Date').agg({
    'Last': 'last',  # Use last price of each day
    'Time': 'last'   # Keep the timestamp
}).copy()

# Calculate daily returns
daily_data['daily_returns'] = daily_data['Last'].pct_change()

# Calculate rolling volatility forecast (similar to rob_port)
# The volatility forecast for day T uses data up to day T-1
daily_data['volatility_forecast'] = calculate_rolling_volatility(
    daily_data['daily_returns'].dropna(), 
    window=rolling_vol_window, 
    min_vol_floor=min_vol_floor
)

# Shift volatility forecast to align properly (forecast for day T made using data up to T-1)
daily_data['volatility_forecast'] = daily_data['volatility_forecast'].shift(1).fillna(min_vol_floor)

logger.info(f"Calculated volatility forecasts for {len(daily_data)} trading days")
logger.info(f"Average volatility forecast: {daily_data['volatility_forecast'].mean():.2%}")

# Merge volatility forecasts back to minute data
data = data.merge(daily_data[['volatility_forecast']], left_on='Date', right_index=True, how='left')

# -------------------------------
# Vectorized Opening Range Calculation
# -------------------------------

print("Calculating opening ranges...")
# Get the first 5 minutes of each day
opening_data = data[data['MinuteOfDay'] <= opening_range_minutes].copy()

# Calculate opening ranges using groupby (much faster)
opening_ranges = opening_data.groupby('Date').agg({
    'High': 'max',
    'Low': 'min'
}).rename(columns={'High': 'range_high', 'Low': 'range_low'})

# Only keep dates with complete opening ranges (5 minutes)
complete_days = opening_data.groupby('Date').size()
complete_days = complete_days[complete_days >= opening_range_minutes].index
opening_ranges = opening_ranges.loc[complete_days]

logger.info(f"Calculated opening ranges for {len(opening_ranges)} trading days")

# Merge opening ranges back to main data for faster lookup
data = data.merge(opening_ranges, left_on='Date', right_index=True, how='left')

# -------------------------------
# Dynamic Position Sizing Backtest Simulation  
# -------------------------------

print("Running dynamic position sizing backtest simulation...")

# Initialize results storage
trade_results = []
equity_values = []
equity_times = []

# Track state
current_equity = initial_capital
daily_equity_tracking = {}  # Track equity by date for position sizing

# Group data by date for efficient processing
grouped_data = data.groupby('Date')

print(f"Initial Capital: ${current_equity:,.2f}")
print(f"Risk Target: {risk_target:.1%}")
print(f"Rolling Volatility Window: {rolling_vol_window} days")

for date_idx, (date, day_data) in enumerate(grouped_data):
    # Get volatility forecast for position sizing (made using data up to previous day)
    volatility_forecast = day_data['volatility_forecast'].iloc[0]
    
    if pd.isna(volatility_forecast):
        volatility_forecast = min_vol_floor
    
    # Get capital for position sizing (current equity at start of day)
    capital_for_sizing = current_equity
    
    # Only look at bars after the opening range (minute 2 and onwards for 1-minute range)
    post_opening_data = day_data[day_data['MinuteOfDay'] > opening_range_minutes].copy()
    
    if len(post_opening_data) == 0:
        # Still need to add equity points for the day
        for idx, row in day_data.iterrows():
            equity_values.append(current_equity)
            equity_times.append(row['Time'])
        continue
    
    # Calculate opening range from first minute(s)
    opening_data = day_data[day_data['MinuteOfDay'] <= opening_range_minutes]
    if len(opening_data) == 0:
        continue
        
    range_high = opening_data['High'].max()
    range_low = opening_data['Low'].min()
    
    # Find the first breakout bar
    breakout_bar = post_opening_data[
        (post_opening_data['High'] > range_high) |
        (post_opening_data['Low'] < range_low)
    ].head(1)
    
    # If no breakout bar found, no trade that day
    if breakout_bar.empty:
        # Still need to add equity points for the day
        for idx, row in post_opening_data.iterrows():
            equity_values.append(current_equity)
            equity_times.append(row['Time'])
        continue
    
    # === DYNAMIC POSITION SIZING ===
    # Use breakout bar's last price for position sizing
    entry_price = breakout_bar['Last'].iat[0]
    entry_time = breakout_bar['Time'].iat[0]
    
    # Calculate dynamic position size based on current equity
    num_contracts = calculate_position_size(
        capital=capital_for_sizing,
        multiplier=multiplier, 
        price=entry_price,
        annualized_volatility=volatility_forecast,
        risk_target=risk_target
    )
    
    # Determine direction: 1 for long, -1 for short
    direction = 1 if breakout_bar['High'].iat[0] > range_high else -1
    
    # Calculate exit price (15:00 CT bar - last bar of the day)
    exit_price = day_data['Last'].iloc[-1]
    exit_time = day_data['Time'].iloc[-1]
    
    # Calculate daily return as per Tsai et al. (2019)
    daily_ret = direction * (exit_price - entry_price) / entry_price
    
    # Convert to dollar P&L (accounting for position direction and dynamic sizing)
    if direction == 1:  # Long
        profit = (exit_price - entry_price) * multiplier * num_contracts - commission_per_trade * num_contracts
        position_direction = 'long'
    else:  # Short  
        profit = (entry_price - exit_price) * multiplier * num_contracts - commission_per_trade * num_contracts
        position_direction = 'short'
    
    # Record the trade
    trade_results.append({
        'entry_time': entry_time,
        'exit_time': exit_time,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'direction': position_direction,
        'profit': profit,
        'daily_return': daily_ret,
        'contracts': num_contracts,
        'volatility_forecast': volatility_forecast,
        'capital_used': capital_for_sizing,
        'exit_reason': 'end_of_day'
    })
    
    # Update equity (this becomes capital for next day's position sizing)
    current_equity += profit
    daily_equity_tracking[date] = current_equity
    
    # Add equity points for the day (simplified since we only have one trade per day)
    # Add equity at entry and throughout the day
    for idx, row in post_opening_data.iterrows():
        if row['Time'] <= entry_time:
            equity_values.append(capital_for_sizing)  # Pre-trade capital
            equity_times.append(row['Time'])
        else:
            # After entry, mark-to-market based on current price
            current_price = row['Last']
            if direction == 1:  # Long
                unrealized = (current_price - entry_price) * multiplier * num_contracts
            else:  # Short
                unrealized = (entry_price - current_price) * multiplier * num_contracts
            equity = capital_for_sizing + unrealized  # Unrealized P&L
            equity_values.append(equity)
            equity_times.append(row['Time'])
    
    # Debug output for first few days
    if date_idx < 5:
        logger.debug(f"Day {date_idx+1} ({date}): {position_direction.upper()} {num_contracts:.2f} contracts")
        logger.debug(f"  Capital: ${capital_for_sizing:,.2f}, Vol Forecast: {volatility_forecast:.2%}")
        logger.debug(f"  Entry: {entry_price:.2f} @ {entry_time}, Exit: {exit_price:.2f} @ {exit_time}")
        logger.debug(f"  Profit: ${profit:.2f}, New Equity: ${current_equity:,.2f}")

print("Backtest simulation completed.")

# -------------------------------
# Performance Metrics Calculation
# -------------------------------

final_account_balance = current_equity
total_return_percentage = ((final_account_balance / initial_capital) - 1) * 100

start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)
years = (end_dt - start_dt).days / 365.25
annualized_return_percentage = ((final_account_balance / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else np.nan

equity_df = pd.DataFrame({'Time': equity_times, 'Equity': equity_values})
equity_df.set_index('Time', inplace=True)

equity_df['returns'] = equity_df['Equity'].pct_change()
# Using the number of trading days per year for futures (assume ~252)
volatility_annual = equity_df['returns'].std() * np.sqrt(252) * 100

total_trades = len(trade_results)
winning_trades = [t for t in trade_results if t['profit'] > 0]
losing_trades = [t for t in trade_results if t['profit'] <= 0]
win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

long_trades = [t for t in trade_results if t['direction'] == 'long']
short_trades = [t for t in trade_results if t['direction'] == 'short']

profit_factor = (sum(t['profit'] for t in winning_trades) / abs(sum(t['profit'] for t in losing_trades))
                 if losing_trades and sum(t['profit'] for t in losing_trades) != 0 else np.nan)

avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else np.nan
avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else np.nan

# Calculate average daily return from trades
avg_daily_return = np.mean([t['daily_return'] for t in trade_results]) if trade_results else np.nan
std_daily_return = np.std([t['daily_return'] for t in trade_results]) if trade_results else np.nan

sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252)
                if equity_df['returns'].std() != 0 else np.nan)

downside_std = equity_df[equity_df['returns'] < 0]['returns'].std()
sortino_ratio = (equity_df['returns'].mean() / downside_std * np.sqrt(252)
                 if downside_std != 0 else np.nan)

equity_df['EquityPeak'] = equity_df['Equity'].cummax()
equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['EquityPeak']) / equity_df['EquityPeak']
max_drawdown_percentage = equity_df['Drawdown'].min() * 100
equity_df['DrawdownDollar'] = equity_df['EquityPeak'] - equity_df['Equity']
max_drawdown_dollar = equity_df['DrawdownDollar'].max()
average_drawdown_dollar = equity_df.loc[equity_df['DrawdownDollar'] > 0, 'DrawdownDollar'].mean()
average_drawdown_percentage = equity_df['Drawdown'].mean() * 100
calmar_ratio = (annualized_return_percentage / abs(max_drawdown_percentage)
                if max_drawdown_percentage != 0 else np.nan)

# Calculate benchmark (buy and hold)
initial_close = data['Last'].iloc[0]
final_close = data['Last'].iloc[-1]
benchmark_return = ((final_close / initial_close) - 1) * 100

results = {
    "Strategy": "Dynamic TORB (Timely Open Range Breakout) - Rob Carver Style",
    "Position Sizing": "Dynamic based on current equity and volatility forecast",
    "Opening Range Period": f"{opening_range_minutes} minute(s)",
    "Timezone": "CT (Central Time)",
    "Start Date": start_date,
    "End Date": end_date,
    "Initial Capital": f"${initial_capital:,.2f}",
    "Final Account Balance": f"${final_account_balance:,.2f}",
    "Total Return": f"{total_return_percentage:.2f}%",
    "Annualized Return": f"{annualized_return_percentage:.2f}%",
    "Benchmark Return (B&H)": f"{benchmark_return:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Risk Target": f"{risk_target:.1%}",
    "Volatility Window": f"{rolling_vol_window} days",
    "Average Volatility Forecast": f"{daily_data['volatility_forecast'].mean():.2%}",
    "Min Volatility Forecast": f"{daily_data['volatility_forecast'].min():.2%}",
    "Max Volatility Forecast": f"{daily_data['volatility_forecast'].max():.2%}",
    "Total Trades": total_trades,
    "Long Trades": len(long_trades),
    "Short Trades": len(short_trades),
    "Winning Trades": len(winning_trades),
    "Losing Trades": len(losing_trades),
    "Win Rate": f"{win_rate:.2f}%",
    "Profit Factor": f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "NaN",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "NaN",
    "Sortino Ratio": f"{sortino_ratio:.2f}" if not np.isnan(sortino_ratio) else "NaN",
    "Calmar Ratio": f"{calmar_ratio:.2f}" if not np.isnan(calmar_ratio) else "NaN",
    "Max Drawdown (%)": f"{max_drawdown_percentage:.2f}%",
    "Average Drawdown (%)": f"{average_drawdown_percentage:.2f}%",
    "Max Drawdown ($)": f"${max_drawdown_dollar:,.2f}",
    "Average Drawdown ($)": f"${average_drawdown_dollar:,.2f}",
    "Average Win ($)": f"${avg_win:,.2f}" if not np.isnan(avg_win) else "NaN",
    "Average Loss ($)": f"${avg_loss:,.2f}" if not np.isnan(avg_loss) else "NaN",
    "Average Daily Return": f"{avg_daily_return:.6f}" if not np.isnan(avg_daily_return) else "NaN",
    "Std Daily Return": f"{std_daily_return:.6f}" if not np.isnan(std_daily_return) else "NaN",
}

# Add position sizing statistics if trades exist
if trade_results:
    trades_df = pd.DataFrame(trade_results)
    avg_contracts = trades_df['contracts'].mean()
    min_contracts = trades_df['contracts'].min()
    max_contracts = trades_df['contracts'].max()
    avg_capital_used = trades_df['capital_used'].mean()
    
    results.update({
        "Average Contracts per Trade": f"{avg_contracts:.2f}",
        "Min Contracts per Trade": f"{min_contracts:.2f}",
        "Max Contracts per Trade": f"{max_contracts:.2f}",
        "Average Capital Used": f"${avg_capital_used:,.2f}",
    })

print("\nDynamic TORB Strategy Performance Summary:")
print("=" * 70)
for key, value in results.items():
    print(f"{key:35}: {value:>25}")

# Trade details summary with position sizing info
if trade_results:
    trades_df = pd.DataFrame(trade_results)
    print(f"\nFirst 10 trades with position sizing:")
    display_cols = ['entry_time', 'direction', 'contracts', 'entry_price', 'exit_price', 
                   'volatility_forecast', 'capital_used', 'daily_return', 'profit']
    print(trades_df.head(10)[display_cols].to_string(index=False, float_format='%.4f'))
    
    print(f"\nPosition Sizing Statistics:")
    print(f"Average Position Size: {trades_df['contracts'].mean():.2f} contracts")
    print(f"Position Size Range: {trades_df['contracts'].min():.2f} to {trades_df['contracts'].max():.2f}")
    print(f"Average Volatility Forecast: {trades_df['volatility_forecast'].mean():.2%}")
    print(f"Capital Growth: ${initial_capital:,.2f} → ${final_account_balance:,.2f} ({total_return_percentage:.1f}%)")

# Visualization
plt.figure(figsize=(15, 12))

# Equity curve
plt.subplot(3, 1, 1)
plt.plot(equity_df.index, equity_df['Equity'], label='Dynamic TORB Strategy Equity', linewidth=1.5, color='blue')
plt.title('Dynamic TORB Strategy Performance (Rob Carver Style Position Sizing)', fontsize=14, fontweight='bold')
plt.xlabel('Time')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Position sizing evolution
plt.subplot(3, 1, 2)
if trade_results:
    trades_df = pd.DataFrame(trade_results)
    plt.plot(trades_df['entry_time'], trades_df['contracts'], 'o-', linewidth=1, markersize=3, 
             color='green', label='Contracts per Trade')
    plt.title('Position Size Evolution (Contracts per Trade)')
    plt.xlabel('Time')
    plt.ylabel('Number of Contracts')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Drawdown chart
plt.subplot(3, 1, 3)
plt.fill_between(equity_df.index, equity_df['Drawdown'] * 100, 0, 
                 color='red', alpha=0.3, label='Drawdown')
plt.title('Drawdown Chart')
plt.xlabel('Time')
plt.ylabel('Drawdown (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

logger.info(f"Dynamic TORB backtest completed. Total trades: {total_trades}, Final capital: ${final_account_balance:,.2f}")

# Additional volatility forecast chart if we have trade data
if trade_results:
    plt.figure(figsize=(12, 8))
    
    # Volatility forecast evolution
    plt.subplot(2, 1, 1)
    trades_df = pd.DataFrame(trade_results)
    plt.plot(trades_df['entry_time'], trades_df['volatility_forecast'] * 100, 'r-', linewidth=1, 
             label='Volatility Forecast (%)')
    plt.axhline(y=min_vol_floor * 100, color='gray', linestyle='--', alpha=0.7, 
                label=f'Min Vol Floor ({min_vol_floor:.1%})')
    plt.title('Volatility Forecast Evolution')
    plt.xlabel('Time')
    plt.ylabel('Annualized Volatility (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Capital utilization
    plt.subplot(2, 1, 2)
    plt.plot(trades_df['entry_time'], trades_df['capital_used'] / 1000, 'purple', linewidth=1, 
             label='Capital Used (Thousands)')
    plt.title('Capital Evolution (Used for Position Sizing)')
    plt.xlabel('Time')
    plt.ylabel('Capital ($K)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
