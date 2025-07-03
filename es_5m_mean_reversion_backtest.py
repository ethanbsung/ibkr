import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Configuration Parameters
# ----------------------------
DATA_FILE = 'Data/es_5m_data.csv'
INITIAL_CAPITAL = 10000
COMMISSION_PER_RT = 1.24  # Commission per round trip
MULTIPLIER = 5  # ES futures multiplier
CONTRACT_SIZE = 1  # Base position size

# Backtest time period (set to None to use all available data)
START_DATE = '2019-01-01'  # Format: 'YYYY-MM-DD' or None for earliest date
END_DATE = '2019-12-31'    # Format: 'YYYY-MM-DD' or None for latest date

# Trading hours (CT timezone - data is already in CT)
SESSION_START = time(8, 30)  # VWAP calculation starts
TRADING_START = time(8, 45)  # First trades allowed
TRADING_END = time(14, 45)   # Last entries allowed
FORCE_CLOSE = time(14, 55)   # Force close all positions

# Strategy parameters
STD_MULTIPLIER_1 = 2.0  # First entry threshold
STD_MULTIPLIER_2 = 3.5  # Additional contract threshold

print("Starting ES 5-Minute Mean Reversion Strategy Backtest")
print("=" * 60)
print(f"Backtest Period: {START_DATE or 'Earliest'} to {END_DATE or 'Latest'}")
print("=" * 60)

# ----------------------------
# Load and Prepare Data
# ----------------------------
print("Loading data...")
df = pd.read_csv(DATA_FILE)

# Convert Time column to datetime
df['Time'] = pd.to_datetime(df['Time'])
df = df.sort_values('Time').reset_index(drop=True)

print(f"Full dataset: {len(df)} bars from {df['Time'].min()} to {df['Time'].max()}")

# Filter data by backtest period
if START_DATE is not None:
    start_datetime = pd.to_datetime(START_DATE)
    df = df[df['Time'] >= start_datetime]
    print(f"Filtered from start date: {START_DATE}")

if END_DATE is not None:
    end_datetime = pd.to_datetime(END_DATE) + pd.Timedelta(days=1)  # Include full end date
    df = df[df['Time'] < end_datetime]
    print(f"Filtered to end date: {END_DATE}")

# Reset index after filtering
df = df.reset_index(drop=True)

# Extract time of day for session logic
df['time_of_day'] = df['Time'].dt.time
df['date'] = df['Time'].dt.date

# Calculate typical price for VWAP calculation
df['typical_price'] = (df['High'] + df['Low'] + df['Last']) / 3

print(f"Backtest period: {len(df)} bars from {df['Time'].min()} to {df['Time'].max()}")

# Validate we have sufficient data
if len(df) == 0:
    raise ValueError("No data available for the specified backtest period!")
elif len(df) < 100:
    print(f"WARNING: Limited data available ({len(df)} bars). Results may not be reliable.")

# ----------------------------
# VWAP and Standard Deviation Calculation
# ----------------------------
print("Calculating VWAP and standard deviation bands...")

def calculate_session_indicators(group):
    """Calculate VWAP and cumulative std for each session"""
    # Filter to session hours (08:30 onwards)
    session_data = group[group['time_of_day'] >= SESSION_START].copy()
    
    if len(session_data) == 0:
        return group
    
    # Calculate cumulative volume and price*volume
    session_data['cum_volume'] = session_data['Volume'].cumsum()
    session_data['cum_pv'] = (session_data['typical_price'] * session_data['Volume']).cumsum()
    
    # Calculate VWAP
    session_data['VWAP'] = session_data['cum_pv'] / session_data['cum_volume']
    
    # ------------------------------------------------
    # Calculate cumulative, volume‑weighted std. dev.
    # σ_t = sqrt( E_w[P²]_t − VWAP_t² )
    # where E_w[P²]_t = Σ(P²·V) / ΣV
    # ------------------------------------------------
    session_data['cum_pv2'] = (session_data['typical_price']**2 * session_data['Volume']).cumsum()
    session_data['std_dev'] = np.sqrt(
        (session_data['cum_pv2'] / session_data['cum_volume']) - session_data['VWAP']**2
    )
    
    # Calculate bands
    session_data['upper_band_1'] = session_data['VWAP'] + (STD_MULTIPLIER_1 * session_data['std_dev'])
    session_data['lower_band_1'] = session_data['VWAP'] - (STD_MULTIPLIER_1 * session_data['std_dev'])
    session_data['upper_band_2'] = session_data['VWAP'] + (STD_MULTIPLIER_2 * session_data['std_dev'])
    session_data['lower_band_2'] = session_data['VWAP'] - (STD_MULTIPLIER_2 * session_data['std_dev'])
    
    # Merge back to original group
    result = group.copy()
    result.loc[session_data.index, ['VWAP', 'std_dev', 'upper_band_1', 'lower_band_1', 
                                   'upper_band_2', 'lower_band_2']] = session_data[['VWAP', 'std_dev', 'upper_band_1', 'lower_band_1', 
                                                                                    'upper_band_2', 'lower_band_2']]
    
    return result

# Apply calculations by date
df = df.groupby('date').apply(calculate_session_indicators).reset_index(drop=True)

# Forward fill VWAP and bands for consistency
df[['VWAP', 'std_dev', 'upper_band_1', 'lower_band_1', 'upper_band_2', 'lower_band_2']] = \
    df[['VWAP', 'std_dev', 'upper_band_1', 'lower_band_1', 'upper_band_2', 'lower_band_2']].fillna(method='ffill')

# ----------------------------
# Initialize Backtest Variables
# ----------------------------
print("Initializing backtest...")

# Portfolio tracking
capital = INITIAL_CAPITAL
position = 0  # Current net position (positive = long, negative = short)
entry_prices = []  # Track entry prices for each contract
trades = []  # Record all trades
equity_curve = []
daily_pnl = []

# Position management tracking
MAX_POSITION = 2  # Maximum contracts per side
long_1std_triggered = False  # Track if long 1 std entry already used
long_2std_triggered = False  # Track if long 2 std entry already used
short_1std_triggered = False  # Track if short 1 std entry already used
short_2std_triggered = False  # Track if short 2 std entry already used

# Performance tracking
total_trades = 0
winning_trades = 0
total_pnl_points = 0

# ----------------------------
# Backtest Main Loop
# ----------------------------
print("Running backtest...")

for i in range(1, len(df)):  # Start from 1 to allow for lookback
    current_bar = df.iloc[i]
    prev_bar = df.iloc[i-1]
    
    current_time = current_bar['Time']
    current_price = current_bar['Last']
    current_close = current_bar['Last']
    prev_close = prev_bar['Last']
    
    # Skip if VWAP not available
    if pd.isna(current_bar['VWAP']):
        equity_curve.append(capital)
        continue
    
    vwap = current_bar['VWAP']
    upper_1 = current_bar['upper_band_1']
    lower_1 = current_bar['lower_band_1'] 
    upper_2 = current_bar['upper_band_2']
    lower_2 = current_bar['lower_band_2']
    
    time_of_day = current_bar['time_of_day']
    
    # Track unrealized P&L
    unrealized_pnl = 0
    if position != 0 and len(entry_prices) > 0:
        for entry_price in entry_prices:
            if position > 0:  # Long position
                unrealized_pnl += (current_price - entry_price) * MULTIPLIER
            else:  # Short position  
                unrealized_pnl += (entry_price - current_price) * MULTIPLIER
    
    # Force close positions at 14:55 CT
    if time_of_day >= FORCE_CLOSE and position != 0:
        # Close all positions
        if position > 0:
            pnl_points = sum([(current_price - entry_price) for entry_price in entry_prices])
        else:
            pnl_points = sum([(entry_price - current_price) for entry_price in entry_prices])
        
        total_pnl = (pnl_points * MULTIPLIER) - (COMMISSION_PER_RT * abs(position))
        capital += total_pnl
        total_pnl_points += pnl_points
        
        trades.append({
            'time': current_time,
            'action': 'FORCE_CLOSE',
            'price': current_price,
            'contracts': abs(position),
            'pnl_points': pnl_points,
            'pnl_dollars': total_pnl
        })
        
        if pnl_points > 0:
            winning_trades += 1
        total_trades += 1
        
        position = 0
        entry_prices = []
        
        # Reset position triggers for next session
        long_1std_triggered = False
        long_2std_triggered = False
        short_1std_triggered = False
        short_2std_triggered = False
        
        print(f"FORCE CLOSE at {current_time}: {abs(position)} contracts at {current_price:.2f}, PnL: {pnl_points:.2f} points")
    
    # Check for VWAP exit (only during trading hours)
    elif position != 0 and TRADING_START <= time_of_day <= TRADING_END:
        exit_triggered = False
        
        # Exit long positions when price crosses back above VWAP
        if position > 0 and prev_close <= vwap and current_close > vwap:
            exit_triggered = True
        
        # Exit short positions when price crosses back below VWAP
        elif position < 0 and prev_close >= vwap and current_close < vwap:
            exit_triggered = True
        
        if exit_triggered:
            # Calculate P&L
            if position > 0:
                pnl_points = sum([(vwap - entry_price) for entry_price in entry_prices])
            else:
                pnl_points = sum([(entry_price - vwap) for entry_price in entry_prices])
            
            total_pnl = (pnl_points * MULTIPLIER) - (COMMISSION_PER_RT * abs(position))
            capital += total_pnl
            total_pnl_points += pnl_points
            
            trades.append({
                'time': current_time,
                'action': 'EXIT_VWAP',
                'price': vwap,
                'contracts': abs(position),
                'pnl_points': pnl_points,
                'pnl_dollars': total_pnl
            })
            
            if pnl_points > 0:
                winning_trades += 1
            total_trades += 1
            
            print(f"EXIT at VWAP {current_time}: {abs(position)} contracts at {vwap:.2f}, PnL: {pnl_points:.2f} points")
            
            position = 0
            entry_prices = []
            
            # Reset position triggers after VWAP exit
            long_1std_triggered = False
            long_2std_triggered = False
            short_1std_triggered = False
            short_2std_triggered = False
    
    # Entry logic (only during trading hours)
    if TRADING_START <= time_of_day <= TRADING_END:
        
        # Long entries
        if current_close < lower_1 and prev_close >= lower_1 and not long_1std_triggered and position == 0:
            # First long entry at 1 std (only if no position and not already triggered)
            position += CONTRACT_SIZE
            entry_prices.append(current_close)
            long_1std_triggered = True
            
            trades.append({
                'time': current_time,
                'action': 'LONG_ENTRY_1STD',
                'price': current_close,
                'contracts': CONTRACT_SIZE,
                'pnl_points': 0,
                'pnl_dollars': 0
            })
            
            print(f"LONG ENTRY (1 std) at {current_time}: 1 contract at {current_close:.2f}")
        
        elif current_close < lower_2 and prev_close >= lower_2 and not long_2std_triggered and position > 0 and position < MAX_POSITION:
            # Additional long entry at 2 std (only if already long, under max position, and not already triggered)
            position += CONTRACT_SIZE
            entry_prices.append(current_close)
            long_2std_triggered = True
            
            trades.append({
                'time': current_time,
                'action': 'LONG_ADD_2STD',
                'price': current_close,
                'contracts': CONTRACT_SIZE,
                'pnl_points': 0,
                'pnl_dollars': 0
            })
            
            print(f"LONG ADD (2 std) at {current_time}: 1 contract at {current_close:.2f}")
        
        # Short entries
        elif current_close > upper_1 and prev_close <= upper_1 and not short_1std_triggered and position == 0:
            # First short entry at 1 std (only if no position and not already triggered)
            position -= CONTRACT_SIZE
            entry_prices.append(current_close)
            short_1std_triggered = True
            
            trades.append({
                'time': current_time,
                'action': 'SHORT_ENTRY_1STD',
                'price': current_close,
                'contracts': CONTRACT_SIZE,
                'pnl_points': 0,
                'pnl_dollars': 0
            })
            
            print(f"SHORT ENTRY (1 std) at {current_time}: 1 contract at {current_close:.2f}")
        
        elif current_close > upper_2 and prev_close <= upper_2 and not short_2std_triggered and position < 0 and abs(position) < MAX_POSITION:
            # Additional short entry at 2 std (only if already short, under max position, and not already triggered)
            position -= CONTRACT_SIZE
            entry_prices.append(current_close)
            short_2std_triggered = True
            
            trades.append({
                'time': current_time,
                'action': 'SHORT_ADD_2STD',
                'price': current_close,
                'contracts': CONTRACT_SIZE,
                'pnl_points': 0,
                'pnl_dollars': 0
            })
            
            print(f"SHORT ADD (2 std) at {current_time}: 1 contract at {current_close:.2f}")
    
    # Update equity curve
    current_equity = capital + unrealized_pnl
    equity_curve.append(current_equity)

# ----------------------------
# Calculate Performance Metrics
# ----------------------------
print("\nCalculating performance metrics...")

final_capital = equity_curve[-1]
total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

# Create equity series for analysis (align with data starting from index 1)
equity_series = pd.Series(equity_curve, index=df['Time'][1:len(equity_curve)+1])
returns = equity_series.pct_change().dropna()

# Calculate key metrics
win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
max_drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max() * 100
sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 78) if returns.std() > 0 else 0  # 78 bars per day

# ----------------------------
# Performance Summary
# ----------------------------
print("\n" + "=" * 60)
print("PERFORMANCE SUMMARY")
print("=" * 60)
print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"Final Capital: ${final_capital:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Total P&L (Points): {total_pnl_points:.2f}")
print(f"Total P&L (Dollars): ${final_capital - INITIAL_CAPITAL:,.2f}")
print(f"Total Trades: {total_trades}")
print(f"Winning Trades: {winning_trades}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Max Drawdown: {max_drawdown:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# ----------------------------
# Plot Equity Curve
# ----------------------------
plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
plt.plot(df['Time'][1:len(equity_curve)+1], equity_curve, 'b-', linewidth=2)
plt.title('ES 5-Minute Mean Reversion Strategy - Equity Curve', fontsize=14)
plt.ylabel('Equity ($)')
plt.grid(True, alpha=0.3)
plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', alpha=0.7, label='Initial Capital')
plt.legend()

# ----------------------------
# Example Day Visualization
# ----------------------------
# Find a day with good trading activity
trade_dates = [trade['time'].date() for trade in trades if 'ENTRY' in trade['action']]
if trade_dates:
    example_date = max(set(trade_dates), key=trade_dates.count)  # Most active day
    
    # Filter data for example day
    day_data = df[df['date'] == example_date].copy()
    day_trades = [t for t in trades if t['time'].date() == example_date]
    
    if len(day_data) > 0:
        plt.subplot(2, 1, 2)
        
        # Plot price and bands
        plt.plot(day_data['Time'], day_data['Last'], 'k-', linewidth=1, label='Price')
        plt.plot(day_data['Time'], day_data['VWAP'], 'b-', linewidth=2, label='VWAP')
        plt.plot(day_data['Time'], day_data['upper_band_1'], 'r--', alpha=0.7, label='±1 Std')
        plt.plot(day_data['Time'], day_data['lower_band_1'], 'r--', alpha=0.7)
        plt.plot(day_data['Time'], day_data['upper_band_2'], 'orange', linestyle=':', alpha=0.7, label='±2 Std')
        plt.plot(day_data['Time'], day_data['lower_band_2'], 'orange', linestyle=':', alpha=0.7)
        
        # Plot trades
        for trade in day_trades:
            if 'LONG' in trade['action']:
                plt.scatter(trade['time'], trade['price'], color='green', s=100, marker='^', 
                           label='Long Entry' if 'Long Entry' not in plt.gca().get_legend_handles_labels()[1] else "")
            elif 'SHORT' in trade['action']:
                plt.scatter(trade['time'], trade['price'], color='red', s=100, marker='v',
                           label='Short Entry' if 'Short Entry' not in plt.gca().get_legend_handles_labels()[1] else "")
            elif 'EXIT' in trade['action']:
                plt.scatter(trade['time'], trade['price'], color='blue', s=100, marker='x',
                           label='Exit' if 'Exit' not in plt.gca().get_legend_handles_labels()[1] else "")
        
        plt.title(f'Example Trading Day: {example_date}', fontsize=12)
        plt.ylabel('Price')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print(f"\nBacktest completed successfully!")
print(f"Example day shown: {example_date if 'example_date' in locals() else 'No trades found'}") 