import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# -------------------------------
# Parameters & User Settings
# -------------------------------

# Input file path for ES futures 30-minute data
data_file = "Data/es_30m_data.csv"  # 30-minute ES data with exact 3PM and 8:30AM entries

# Backtest parameters
initial_capital = 50000.0         # starting account balance in dollars
commission_per_order = 0.62       # commission per order (per contract)
num_contracts = 2                # number of contracts to trade
multiplier = 5                   # each point move is worth $5 per contract (full MES)

start_date = '2008-01-01'
end_date   = '2025-01-01'

# -------------------------------
# Data Preparation
# -------------------------------

# Read 30-minute ES futures data
logger.info("Loading ES 30-minute data...")
data = pd.read_csv(data_file, parse_dates=['Time'])
data.sort_values('Time', inplace=True)

# Filter data based on custom date range
data = data[(data['Time'] >= start_date) & (data['Time'] <= end_date)].reset_index(drop=True)

# Extract time component for filtering
data['hour'] = data['Time'].dt.hour
data['minute'] = data['Time'].dt.minute
data['date'] = data['Time'].dt.date

# Filter for only 3 PM (15:00) and 8:30 AM (08:30) entries
entry_data = data[(data['hour'] == 15) & (data['minute'] == 0)].copy()  # 3 PM CT entries
exit_data = data[(data['hour'] == 8) & (data['minute'] == 30)].copy()   # 8:30 AM CT exits

logger.info(f"Found {len(entry_data)} potential entry points at 3 PM CT")
logger.info(f"Found {len(exit_data)} potential exit points at 8:30 AM CT")

# Create entry-exit pairs for overnight trades
trades = []
entry_data = entry_data.reset_index(drop=True)
exit_data = exit_data.reset_index(drop=True)

# Set exit_data index on date for efficient lookup
exit_data.set_index('date', inplace=True)

for i, entry_row in entry_data.iterrows():
    entry_date = entry_row['date']
    entry_time = entry_row['Time']
    entry_price = entry_row['Last']
    
    # Find next trading day's 8:30 AM exit
    # Look for exit on the next trading day (could be 1-3 days later due to weekends)
    exit_found = False
    for days_ahead in range(1, 4):  # Check up to 3 days ahead for next trading day
        target_exit_date = entry_date + timedelta(days=days_ahead)
        
        if target_exit_date in exit_data.index:
            exit_row = exit_data.loc[target_exit_date]
            if isinstance(exit_row, pd.DataFrame):  # Multiple entries for same date
                exit_row = exit_row.iloc[0]  # Take first one
                
            exit_time = exit_row['Time']
            exit_price = exit_row['Last']
            
            # Calculate overnight return
            price_change = exit_price - entry_price
            gross_profit = price_change * multiplier * num_contracts
            net_profit = gross_profit - (2 * commission_per_order * num_contracts)  # Entry + Exit commission
            
            trades.append({
                'entry_date': entry_date,
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_date': target_exit_date,
                'exit_time': exit_time,
                'exit_price': exit_price,
                'price_change': price_change,
                'gross_profit': gross_profit,
                'net_profit': net_profit,
                'contracts': num_contracts
            })
            
            exit_found = True
            break
    
    if not exit_found:
        logger.warning(f"No exit found for entry on {entry_date}")

logger.info(f"Created {len(trades)} overnight trade pairs")

# Convert trades to DataFrame for easier manipulation
trades_df = pd.DataFrame(trades)

if len(trades_df) == 0:
    logger.error("No valid trades found. Please check date range and data availability.")
    exit()

# -------------------------------
# Backtest Simulation
# -------------------------------

capital = initial_capital
equity_curve = []
cumulative_trades = []

# Create daily equity curve
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
daily_returns = pd.Series(0.0, index=all_dates)

# Calculate cumulative performance
running_capital = initial_capital

for i, trade in trades_df.iterrows():
    # Update capital with trade result
    running_capital += trade['net_profit']
    
    # Record equity on exit date
    exit_date = pd.to_datetime(trade['exit_date'])
    daily_returns.loc[exit_date] = trade['net_profit'] / running_capital * 100  # Daily return percentage
    
    equity_curve.append({
        'date': exit_date,
        'trade_number': i + 1,
        'capital': running_capital,
        'trade_profit': trade['net_profit'],
        'cumulative_return': ((running_capital / initial_capital) - 1) * 100
    })
    
    cumulative_trades.append(trade)
    
    logger.info(f"Trade {i+1}: {trade['entry_date']} -> {trade['exit_date']} | "
               f"Entry: ${trade['entry_price']:.2f} | Exit: ${trade['exit_price']:.2f} | "
               f"P&L: ${trade['net_profit']:.2f} | Capital: ${running_capital:,.2f}")

# Convert equity curve to DataFrame
equity_df = pd.DataFrame(equity_curve)
if len(equity_df) > 0:
    equity_df.set_index('date', inplace=True)

# -------------------------------
# Benchmark Calculation (Buy & Hold)
# -------------------------------

# Use first and last available prices for buy & hold benchmark
first_price = data['Last'].iloc[0]
last_price = data['Last'].iloc[-1]
benchmark_return = ((last_price / first_price) - 1) * 100

# -------------------------------
# Performance Metrics Calculation
# -------------------------------

final_account_balance = running_capital
total_return_percentage = ((final_account_balance / initial_capital) - 1) * 100

start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)
years = (end_dt - start_dt).days / 365.25
annualized_return_percentage = ((final_account_balance / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else np.nan

# Calculate returns from equity curve
if len(equity_df) > 1:
    equity_df['returns'] = equity_df['capital'].pct_change()
    returns_series = equity_df['returns'].dropna()
    
    # Annualized volatility (assuming trades happen roughly daily)
    volatility_annual = returns_series.std() * np.sqrt(252) * 100
    
    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = (returns_series.mean() / returns_series.std() * np.sqrt(252)
                    if returns_series.std() != 0 else np.nan)
    
    # Sortino ratio
    downside_returns = returns_series[returns_series < 0]
    downside_std = downside_returns.std()
    sortino_ratio = (returns_series.mean() / downside_std * np.sqrt(252)
                     if downside_std != 0 and not np.isnan(downside_std) else np.nan)
    
    # Drawdown analysis
    equity_df['equity_peak'] = equity_df['capital'].cummax()
    equity_df['drawdown'] = (equity_df['capital'] - equity_df['equity_peak']) / equity_df['equity_peak']
    max_drawdown_percentage = equity_df['drawdown'].min() * 100
    
    # Calmar ratio
    calmar_ratio = (annualized_return_percentage / abs(max_drawdown_percentage)
                    if max_drawdown_percentage != 0 else np.nan)
else:
    volatility_annual = np.nan
    sharpe_ratio = np.nan
    sortino_ratio = np.nan
    max_drawdown_percentage = 0
    calmar_ratio = np.nan

# Trade statistics
total_trades = len(trades_df)
winning_trades = trades_df[trades_df['net_profit'] > 0]
losing_trades = trades_df[trades_df['net_profit'] <= 0]

win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
avg_win = winning_trades['net_profit'].mean() if len(winning_trades) > 0 else 0
avg_loss = losing_trades['net_profit'].mean() if len(losing_trades) > 0 else 0
profit_factor = (winning_trades['net_profit'].sum() / abs(losing_trades['net_profit'].sum())
                 if len(losing_trades) > 0 and losing_trades['net_profit'].sum() != 0 else np.nan)

largest_win = trades_df['net_profit'].max() if total_trades > 0 else 0
largest_loss = trades_df['net_profit'].min() if total_trades > 0 else 0

# -------------------------------
# Results Summary
# -------------------------------

results = {
    "Strategy": "Overnight ES (3PM CT Entry -> 8:30AM CT Exit)",
    "Data File": data_file,
    "Start Date": start_date,
    "End Date": end_date,
    "Initial Capital": f"${initial_capital:,.2f}",
    "Final Account Balance": f"${final_account_balance:,.2f}",
    "Total Return": f"{total_return_percentage:.2f}%",
    "Annualized Return": f"{annualized_return_percentage:.2f}%",
    "Benchmark Return (Buy & Hold)": f"{benchmark_return:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}",
    "Sortino Ratio": f"{sortino_ratio:.2f}",
    "Max Drawdown": f"{max_drawdown_percentage:.2f}%",
    "Calmar Ratio": f"{calmar_ratio:.2f}",
    "Total Trades": total_trades,
    "Win Rate": f"{win_rate:.1f}%",
    "Profit Factor": f"{profit_factor:.2f}",
    "Average Win": f"${avg_win:.2f}",
    "Average Loss": f"${avg_loss:.2f}",
    "Largest Win": f"${largest_win:.2f}",
    "Largest Loss": f"${largest_loss:.2f}",
    "Commission per Trade": f"${commission_per_order * 2 * num_contracts:.2f}",
    "Total Commission Paid": f"${total_trades * commission_per_order * 2 * num_contracts:.2f}"
}

# -------------------------------
# Display Results
# -------------------------------

print("\n" + "="*60)
print("         OVERNIGHT ES FUTURES BACKTEST RESULTS")
print("="*60)

for key, value in results.items():
    print(f"{key:<25}: {value}")

print("="*60)

# -------------------------------
# Visualization
# -------------------------------

if len(equity_df) > 1:
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Equity Curve
    ax1.plot(equity_df.index, equity_df['capital'], 'b-', linewidth=2, label='Overnight Strategy')
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
    ax1.set_title('Overnight ES Strategy: Equity Curve (3PM Entry -> 8:30AM Exit)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Account Balance ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Drawdown
    ax2.fill_between(equity_df.index, equity_df['drawdown'] * 100, 0, 
                     color='red', alpha=0.3, label='Drawdown')
    ax2.plot(equity_df.index, equity_df['drawdown'] * 100, 'r-', linewidth=1)
    ax2.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Trade P&L Distribution
    ax3.hist(trades_df['net_profit'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax3.axvline(x=trades_df['net_profit'].mean(), color='green', linestyle='-', linewidth=2, 
                label=f'Average: ${trades_df["net_profit"].mean():.2f}')
    ax3.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Trade Profit/Loss ($)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# -------------------------------
# Save Results
# -------------------------------

print(f"\nBacktest completed successfully!")
print(f"Processed {total_trades} overnight trades from {start_date} to {end_date}")
