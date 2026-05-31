import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# -------------------------------
# Parameters & User Settings
# -------------------------------

# Input file path for ES 30-minute data
data_file = "Data/es_30m_data.csv"

# Backtest parameters
initial_capital = 30000.0         # starting account balance in dollars
commission_per_order = 0.62       # commission per order (per contract)
num_contracts = 5                # number of contracts to trade
multiplier = 5                   # each point move is worth $50 per contract for ES

# Custom start and end date (format: 'YYYY-MM-DD')
start_date = '2008-01-01'
end_date   = '2020-01-01'

# -------------------------------
# Data Preparation
# -------------------------------

# Read 30-minute ES futures data
data = pd.read_csv(data_file, parse_dates=['Time'])
data.sort_values('Time', inplace=True)

# Filter data based on custom date range
data = data[(data['Time'] >= start_date) & (data['Time'] <= end_date)].reset_index(drop=True)

# Filter to regular trading hours only (8:30-15:00 CT)
data = data.set_index('Time')                       # make index a DatetimeIndex
data = data.between_time("08:30", "15:00")          # keep only 08:30-15:00 CT
data.reset_index(inplace=True)

# Add date column for grouping by trading day
data['Date'] = data['Time'].dt.date

# Add time component for intraday filtering
data['TimeOnly'] = data['Time'].dt.time

# -------------------------------
# Strategy Implementation
# -------------------------------

def get_first_30min_return(daily_data):
    """
    Calculate return for the 8:30 bar (covers 8:30-9:00 AM)
    Uses open and close of the single 8:30 bar
    """
    bar = daily_data[daily_data['TimeOnly'] == pd.to_datetime("08:30:00").time()]
    if bar.empty: 
        return np.nan
    return (bar['Last'].iloc[0] - bar['Open'].iloc[0]) / bar['Open'].iloc[0]

def get_trade_bar(daily_data):
    """
    Get the 14:30 bar (covers 14:30-15:00, the last RTH bar)
    Entry at open, exit at close of same bar
    """
    bar = daily_data[daily_data['TimeOnly'] == pd.to_datetime("14:30:00").time()]
    return bar.iloc[0] if not bar.empty else None

# -------------------------------
# Backtest Simulation
# -------------------------------

capital = initial_capital
trade_results = []
equity_curve = []

# For benchmark: Buy and Hold ES (enter at first available close)
initial_close = data['Last'].iloc[0]

# Group data by trading day
daily_groups = data.groupby('Date')

logger.info(f"Starting backtest with {len(daily_groups)} trading days")

for date, daily_data in daily_groups:
    # Calculate first 30-minute return
    first_30min_return = get_first_30min_return(daily_data)
    
    # Skip if we can't calculate the return
    if pd.isna(first_30min_return):
        # Record equity for the day (no trading)
        last_price = daily_data.iloc[-1]['Last']
        last_time = daily_data.iloc[-1]['Time']
        equity_curve.append((last_time, capital))
        continue
    
    # Get entry and exit times
    entry_bar = get_trade_bar(daily_data)
    
    if entry_bar is None:
        # Record equity for the day (no trading)
        last_price = daily_data.iloc[-1]['Last']
        last_time = daily_data.iloc[-1]['Time']
        equity_curve.append((last_time, capital))
        continue
    
    # Determine trade direction based on first 30-min return
    if first_30min_return > 0:
        # Go long at 14:30 bar open, exit at 14:30 bar close
        direction = 'long'
        entry_price = entry_bar['Open']   # open of 14:30 bar
        exit_price = entry_bar['Last']    # close of 14:30 bar
        
        # Calculate profit for long position
        price_change = exit_price - entry_price
        profit = price_change * multiplier * num_contracts - (2 * commission_per_order * num_contracts)
        
        logger.info(f"{date} | LONG | 30min return: {first_30min_return:.4f} | Entry: {entry_price:.2f} | Exit: {exit_price:.2f} | P&L: {profit:.2f}")
        
    elif first_30min_return < 0:
        # Go short at 14:30 bar open, exit at 14:30 bar close
        direction = 'short'
        entry_price = entry_bar['Open']   # open of 14:30 bar
        exit_price = entry_bar['Last']    # close of 14:30 bar
        
        # Calculate profit for short position
        price_change = entry_price - exit_price  # Reversed for short
        profit = price_change * multiplier * num_contracts - (2 * commission_per_order * num_contracts)
        
        logger.info(f"{date} | SHORT | 30min return: {first_30min_return:.4f} | Entry: {entry_price:.2f} | Exit: {exit_price:.2f} | P&L: {profit:.2f}")
        
    else:
        # No trade if return is exactly 0
        last_price = daily_data.iloc[-1]['Last']
        last_time = daily_data.iloc[-1]['Time']
        equity_curve.append((last_time, capital))
        continue
    
    # Record the trade
    trade_results.append({
        'date': date,
        'direction': direction,
        'first_30min_return': first_30min_return,
        'entry_time': entry_bar['Time'],
        'exit_time': entry_bar['Time'],
        'entry_price': entry_price,
        'exit_price': exit_price,
        'profit': profit,
        'contracts': num_contracts
    })
    
    # Update capital
    capital += profit
    
    # Record equity at end of day
    equity_curve.append((entry_bar['Time'], capital))

# Convert equity curve to DataFrame
equity_df = pd.DataFrame(equity_curve, columns=['Time', 'Equity'])
equity_df.set_index('Time', inplace=True)

# Create benchmark equity curve
benchmark_data = data.groupby('Date').last().reset_index()
benchmark_equity = (benchmark_data['Last'] / initial_close) * initial_capital
benchmark_df = pd.DataFrame({
    'Time': benchmark_data['Time'],
    'Equity': benchmark_equity.values
})
benchmark_df.set_index('Time', inplace=True)

# -------------------------------
# Performance Metrics Calculation
# -------------------------------

final_account_balance = capital
total_return_percentage = ((final_account_balance / initial_capital) - 1) * 100

start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)
years = (end_dt - start_dt).days / 365.25
annualized_return_percentage = ((final_account_balance / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else np.nan

# Calculate returns
equity_df['returns'] = equity_df['Equity'].pct_change()
volatility_annual = equity_df['returns'].std() * np.sqrt(252) * 100

# Trade statistics
total_trades = len(trade_results)
if total_trades > 0:
    winning_trades = [t for t in trade_results if t['profit'] > 0]
    losing_trades = [t for t in trade_results if t['profit'] <= 0]
    win_rate = (len(winning_trades) / total_trades * 100)
    
    total_profit = sum(t['profit'] for t in winning_trades)
    total_loss = abs(sum(t['profit'] for t in losing_trades))
    profit_factor = total_profit / total_loss if total_loss > 0 else np.nan
    
    avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else np.nan
    avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else np.nan
else:
    win_rate = 0
    profit_factor = np.nan
    avg_win = np.nan
    avg_loss = np.nan

# Risk metrics
sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252)
                if equity_df['returns'].std() != 0 else np.nan)

downside_returns = equity_df[equity_df['returns'] < 0]['returns']
downside_std = downside_returns.std()
sortino_ratio = (equity_df['returns'].mean() / downside_std * np.sqrt(252)
                 if downside_std != 0 else np.nan)

# Drawdown analysis
equity_df['EquityPeak'] = equity_df['Equity'].cummax()
equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['EquityPeak']) / equity_df['EquityPeak']
max_drawdown_percentage = equity_df['Drawdown'].min() * 100

# Calculate Calmar ratio
calmar_ratio = (annualized_return_percentage / abs(max_drawdown_percentage)
                if max_drawdown_percentage != 0 else np.nan)

# Benchmark performance
benchmark_return = ((benchmark_df['Equity'].iloc[-1] / benchmark_df['Equity'].iloc[0]) - 1) * 100

# -------------------------------
# Results Summary
# -------------------------------

results = {
    "Strategy": "Intraday 30-Min Momentum",
    "Start Date": start_date,
    "End Date": end_date,
    "Final Account Balance": f"${final_account_balance:,.2f}",
    "Total Return": f"{total_return_percentage:.2f}%",
    "Annualized Return": f"{annualized_return_percentage:.2f}%",
    "Benchmark Return": f"{benchmark_return:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}",
    "Sortino Ratio": f"{sortino_ratio:.2f}",
    "Max Drawdown (%)": f"{max_drawdown_percentage:.2f}%",
    "Calmar Ratio": f"{calmar_ratio:.2f}",
    "Total Trades": total_trades,
    "Win Rate": f"{win_rate:.2f}%",
    "Profit Factor": f"{profit_factor:.2f}",
    "Average Win": f"${avg_win:.2f}" if not pd.isna(avg_win) else "N/A",
    "Average Loss": f"${avg_loss:.2f}" if not pd.isna(avg_loss) else "N/A",
}

# Print results
print("\n" + "="*60)
print("INTRADAY 30-MINUTE MOMENTUM STRATEGY BACKTEST RESULTS")
print("="*60)
for key, value in results.items():
    print(f"{key:.<30} {value}")
print("="*60)

# -------------------------------
# Visualization
# -------------------------------

plt.figure(figsize=(14, 10))

# Plot 1: Equity Curves
plt.subplot(2, 2, 1)
plt.plot(equity_df.index, equity_df['Equity'], label='Strategy', linewidth=2)
plt.plot(benchmark_df.index, benchmark_df['Equity'], label='Buy & Hold', linewidth=2, alpha=0.7)
plt.title('Equity Curve: 30-Min Momentum vs Buy & Hold')
plt.xlabel('Time')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Drawdown
plt.subplot(2, 2, 2)
plt.fill_between(equity_df.index, equity_df['Drawdown'] * 100, 0, alpha=0.3, color='red')
plt.plot(equity_df.index, equity_df['Drawdown'] * 100, color='red', linewidth=1)
plt.title('Drawdown (%)')
plt.xlabel('Time')
plt.ylabel('Drawdown (%)')
plt.grid(True, alpha=0.3)

# Plot 3: Daily Returns Distribution
plt.subplot(2, 2, 3)
returns_clean = equity_df['returns'].dropna()
plt.hist(returns_clean, bins=50, alpha=0.7, edgecolor='black')
plt.title('Daily Returns Distribution')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Plot 4: Monthly Returns Heatmap (if sufficient data)
plt.subplot(2, 2, 4)
monthly_returns = equity_df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
years_available = monthly_returns.index.year.unique()

if len(years_available) > 1:
    monthly_pivot = monthly_returns.reset_index()
    monthly_pivot['Year'] = monthly_pivot['Time'].dt.year
    monthly_pivot['Month'] = monthly_pivot['Time'].dt.month
    pivot_table = monthly_pivot.pivot(index='Year', columns='Month', values='returns')
    
    import seaborn as sns
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0, cbar_kws={'label': 'Monthly Return (%)'})
    plt.title('Monthly Returns Heatmap (%)')
else:
    plt.text(0.5, 0.5, 'Insufficient data\nfor monthly heatmap', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Monthly Returns Heatmap')

plt.tight_layout()
plt.show()

# Print sample trades
print(f"\nFirst 10 Trades:")
print("-" * 80)
for i, trade in enumerate(trade_results[:10]):
    print(f"{trade['date']} | {trade['direction'].upper():5} | "
          f"30min ret: {trade['first_30min_return']:6.3f} | "
          f"Entry: {trade['entry_price']:7.2f} | "
          f"Exit: {trade['exit_price']:7.2f} | "
          f"P&L: ${trade['profit']:7.2f}")

print(f"\nTotal trading days: {len(daily_groups)}")
print(f"Days with trades: {len(trade_results)}")
print(f"Trade frequency: {len(trade_results)/len(daily_groups)*100:.1f}%")
