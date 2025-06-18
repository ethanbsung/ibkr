import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Strategy Parameters - CORRECTED VERSION WITH STOP LOSS
initial_capital = 30000
commission_per_order = 0.62  # Per contract per side  
multiplier = 5  # MES multiplier
breakout_offset = 0.5  # Points above/below range

# Stop Loss Parameters
use_stop_loss = True  # Enable/disable stop loss
stop_loss_points = 2.0  # Fixed point stop loss (alternative to OR-based)
stop_loss_type = 'OR_BASED'  # 'OR_BASED' or 'FIXED_POINTS'
# OR_BASED: Uses opposite breakout level as stop
# FIXED_POINTS: Uses fixed point distance from entry

# Data file path
data_file = '../Data/es_1m_data.csv'

# Date range for backtest
start_date = '2008-01-01'
end_date = '2020-01-01'

def load_and_prepare_data(file_path):
    """Load 1-minute ES data and prepare for analysis"""
    print("Loading 1-minute ES data...")
    
    # Load data
    data = pd.read_csv(file_path)
    
    # Convert Time column to datetime
    data['Time'] = pd.to_datetime(data['Time'])
    
    # Sort by time
    data.sort_values('Time', inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # Filter by date range if specified
    if start_date and end_date:
        data = data[(data['Time'] >= start_date) & (data['Time'] <= end_date)]
        data.reset_index(drop=True, inplace=True)
    
    print(f"Loaded {len(data)} 1-minute bars from {data['Time'].min()} to {data['Time'].max()}")
    return data

def create_daily_bars(minute_data):
    """Create daily OHLC bars from 1-minute data for NR4 identification"""
    daily_data = []
    
    # Group by date (using CT timezone convention: 5PM-4PM CT daily session)
    minute_data['Date'] = minute_data['Time'].dt.date
    
    for date, group in minute_data.groupby('Date'):
        if len(group) == 0:
            continue
            
        daily_bar = {
            'Date': date,
            'Open': group.iloc[0]['Open'],
            'High': group['High'].max(),
            'Low': group['Low'].min(),
            'Close': group.iloc[-1]['Last'],
            'Volume': group['Volume'].sum()
        }
        daily_data.append(daily_bar)
    
    daily_df = pd.DataFrame(daily_data)
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    daily_df['Range'] = daily_df['High'] - daily_df['Low']
    
    return daily_df

def identify_nr4_days(daily_df):
    """Identify NR4 days (day has smallest range of last 4 days)"""
    daily_df['NR4'] = False
    
    for i in range(3, len(daily_df)):  # Need at least 4 days
        current_range = daily_df.iloc[i]['Range']
        previous_3_ranges = daily_df.iloc[i-3:i]['Range'].values
        
        # NR4: Current day has smallest range of the last 4 days
        if current_range < min(previous_3_ranges):
            daily_df.iloc[i, daily_df.columns.get_loc('NR4')] = True
    
    return daily_df

def calculate_opening_range(minute_data, date):
    """Calculate 5-minute opening range for a specific date - OPTIMIZED"""
    # Pre-filter data for the specific date (more efficient than filtering inside loop)
    date_mask = minute_data['Time'].dt.date == date
    date_data = minute_data[date_mask].copy()
    
    if len(date_data) == 0:
        return None
    
    # Find opening range (8:30-8:35 AM CT) using vectorized operations
    time_only = date_data['Time'].dt.time
    or_mask = (time_only >= time(8, 30)) & (time_only < time(8, 35))
    opening_range_data = date_data[or_mask]
    
    if len(opening_range_data) == 0:
        return None
    
    or_high = opening_range_data['High'].max()
    or_low = opening_range_data['Low'].min()
    or_range = or_high - or_low
    
    # Skip days with very small opening ranges (less than 1 point)
    if or_range < 1.0:
        return None
    
    return {
        'or_high': or_high,
        'or_low': or_low,
        'or_range': or_range,
        'buy_stop': or_high + breakout_offset,
        'sell_stop': or_low - breakout_offset,
        'date_data': date_data  # Cache filtered data for reuse
    }

def calculate_stop_loss(entry_price, entry_type, or_data):
    """Calculate stop loss based on configured method"""
    if not use_stop_loss:
        return None
    
    if stop_loss_type == 'OR_BASED':
        # Use opposite breakout level as stop loss
        if entry_type == 'LONG':
            return or_data['sell_stop']
        else:  # SHORT
            return or_data['buy_stop']
    
    elif stop_loss_type == 'FIXED_POINTS':
        # Use fixed point distance from entry
        if entry_type == 'LONG':
            return entry_price - stop_loss_points
        else:  # SHORT
            return entry_price + stop_loss_points
    
    return None

def run_backtest(minute_data):
    """Run the ORB backtest - OPTIMIZED VERSION WITH NR4 FILTER"""
    
    # Create daily bars for NR4 identification
    print("Creating daily bars for NR4 analysis...")
    daily_df = create_daily_bars(minute_data)
    daily_df = identify_nr4_days(daily_df)
    
    nr4_days = daily_df['NR4'].sum()
    print(f"Found {nr4_days} NR4 days out of {len(daily_df)} total days")
    
    # Get NR4 dates for filtering
    nr4_dates = set(daily_df[daily_df['NR4']]['Date'].dt.date.tolist())
    
    # Initialize tracking variables
    capital = initial_capital
    trade_results = []
    equity_curve = []
    
    # Pre-compute all dates to avoid repeated operations
    all_dates = sorted(minute_data['Time'].dt.date.unique())
    
    # Filter to only NR4 days + next day (since we trade the day AFTER NR4)
    nr4_next_day_dates = []
    daily_dates = sorted(daily_df['Date'].dt.date.tolist())
    
    for i, date in enumerate(daily_dates[:-1]):  # Exclude last day since it has no "next day"
        if date in nr4_dates:
            next_day = daily_dates[i + 1]
            nr4_next_day_dates.append(next_day)
    
    qualifying_dates = [d for d in all_dates if d in nr4_next_day_dates]
    
    print(f"Found {len(qualifying_dates)} trading days (day after NR4)")
    print(f"Stop Loss Enabled: {use_stop_loss}")
    if use_stop_loss:
        print(f"Stop Loss Type: {stop_loss_type}")
        if stop_loss_type == 'FIXED_POINTS':
            print(f"Stop Loss Distance: {stop_loss_points} points")
    
    total_trades = 0
    successful_setups = 0
    
    # Process each qualifying date
    for i, trade_date in enumerate(qualifying_dates):
        # Progress indicator for long backtests
        if i % 100 == 0 and i > 0:
            print(f"Processed {i}/{len(qualifying_dates)} NR4 days ({i/len(qualifying_dates)*100:.1f}%)")
        
        # Calculate opening range (includes cached day data)
        or_data = calculate_opening_range(minute_data, trade_date)
        if or_data is None:
            continue
        
        successful_setups += 1
        
        # Use cached day data from opening range calculation
        day_data = or_data['date_data'].copy()
        day_data['Time_Only'] = day_data['Time'].dt.time
        
        # Pre-filter to relevant trading hours only (8:35 AM - 3:00 PM CT) - CHANGED TO 3PM
        relevant_mask = (day_data['Time_Only'] >= time(8, 35)) & (day_data['Time_Only'] <= time(15, 0))
        day_data = day_data[relevant_mask].reset_index(drop=True)
        
        if len(day_data) == 0:
            continue
        
        # Track daily state
        daily_position = None
        daily_equity = capital
        trade_taken = False
        
        # OPTIMIZED: Process only trading window (8:35-9:30) for entries
        entry_window_mask = day_data['Time_Only'] <= time(9, 30)
        entry_window_data = day_data[entry_window_mask]
        
        # Look for breakout in entry window
        if len(entry_window_data) > 0:
            # Vectorized breakout detection
            long_breakout_mask = entry_window_data['High'] >= or_data['buy_stop']
            short_breakout_mask = entry_window_data['Low'] <= or_data['sell_stop']
            
            long_breakout_idx = None
            short_breakout_idx = None
            
            if long_breakout_mask.any():
                long_breakout_idx = entry_window_data[long_breakout_mask].index[0]
            if short_breakout_mask.any():
                short_breakout_idx = entry_window_data[short_breakout_mask].index[0]
            
            # Determine which breakout occurred first
            entry_idx = None
            entry_type = None
            entry_price = None
            
            if long_breakout_idx is not None and short_breakout_idx is not None:
                # Both breakouts occurred - take the first one
                if long_breakout_idx <= short_breakout_idx:
                    entry_idx = long_breakout_idx
                    entry_type = 'LONG'
                    entry_price = or_data['buy_stop']
                else:
                    entry_idx = short_breakout_idx
                    entry_type = 'SHORT'
                    entry_price = or_data['sell_stop']
            elif long_breakout_idx is not None:
                entry_idx = long_breakout_idx
                entry_type = 'LONG'
                entry_price = or_data['buy_stop']
            elif short_breakout_idx is not None:
                entry_idx = short_breakout_idx
                entry_type = 'SHORT'
                entry_price = or_data['sell_stop']
            
            # If we have an entry, process the trade
            if entry_type:
                entry_row = day_data.loc[entry_idx]
                current_time = entry_row['Time']
                
                # Calculate stop loss
                stop_loss = calculate_stop_loss(entry_price, entry_type, or_data)
                
                # Enter position
                num_contracts = 1
                capital -= commission_per_order * num_contracts
                
                daily_position = {
                    'type': entry_type,
                    'entry_price': entry_price,
                    'entry_time': current_time,
                    'stop_loss': stop_loss,
                    'contracts': num_contracts
                }
                trade_taken = True
                total_trades += 1
                
                # Reduced logging for performance
                if total_trades <= 10 or total_trades % 50 == 0:
                    stop_info = f" | Stop: {stop_loss:.2f}" if stop_loss else " | No Stop"
                    logger.info(f"ENTRY {total_trades}: {entry_type} at {current_time} | Price: {entry_price:.2f}{stop_info}")
                
                # OPTIMIZED EXIT PROCESSING: Look for exit after entry
                exit_data = day_data[day_data.index > entry_idx].copy()
                
                exit_found = False
                exit_price = None
                exit_time = None
                exit_reason = None
                
                if use_stop_loss and stop_loss:
                    # Vectorized stop loss detection
                    if entry_type == 'LONG':
                        stop_hit_mask = exit_data['Low'] <= stop_loss
                    else:  # SHORT
                        stop_hit_mask = exit_data['High'] >= stop_loss
                    
                    if stop_hit_mask.any():
                        exit_idx = exit_data[stop_hit_mask].index[0]
                        exit_row = day_data.loc[exit_idx]
                        exit_price = stop_loss
                        exit_time = exit_row['Time']
                        exit_reason = 'STOP_LOSS'
                        exit_found = True
                
                # If no stop loss hit, exit at 3:00 PM CT (CHANGED FROM 4PM)
                if not exit_found:
                    exit_row = day_data.iloc[-1]
                    exit_price = exit_row['Last']
                    exit_time = exit_row['Time']
                    exit_reason = 'END_OF_DAY_3PM'
                
                # Calculate profit
                if entry_type == 'LONG':
                    price_diff = exit_price - entry_price
                else:  # SHORT
                    price_diff = entry_price - exit_price
                
                profit = price_diff * multiplier * num_contracts - commission_per_order * num_contracts
                
                # Record trade
                trade_results.append({
                    'entry_time': current_time,
                    'exit_time': exit_time,
                    'type': entry_type,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit': profit,
                    'contracts': num_contracts,
                    'exit_reason': exit_reason,
                    'or_range': or_data['or_range'],
                    'stop_loss': stop_loss
                })
                
                capital += profit
                daily_equity = capital
                
                # Reduced logging for performance
                if total_trades <= 10 or total_trades % 50 == 0:
                    logger.info(f"EXIT {total_trades}: {exit_reason} | Price: {exit_price:.2f} | P&L: ${profit:.2f}")
        
        # Record daily equity (use final capital for the day)
        if len(day_data) > 0:
            equity_curve.append((day_data.iloc[-1]['Time'], capital))
    
    print(f"\nSetup Success Rate: {successful_setups}/{len(qualifying_dates)} = {(successful_setups/len(qualifying_dates)*100):.1f}%")
    print(f"Trade Execution Rate: {total_trades}/{successful_setups} = {(total_trades/successful_setups*100):.1f}%" if successful_setups > 0 else "No valid setups")
    
    return trade_results, equity_curve, capital

def calculate_performance_metrics(trade_results, equity_curve, initial_capital, final_capital):
    """Calculate comprehensive performance metrics"""
    
    if len(trade_results) == 0:
        print("No trades executed - cannot calculate performance metrics")
        return {}
    
    # Convert equity curve to DataFrame
    equity_df = pd.DataFrame(equity_curve, columns=['Time', 'Equity'])
    equity_df.set_index('Time', inplace=True)
    
    # Calculate returns
    equity_df['returns'] = equity_df['Equity'].pct_change()
    returns = equity_df['returns'].dropna()
    
    # Time period
    start_date = equity_df.index.min()
    end_date = equity_df.index.max()
    days = (end_date - start_date).days
    years = days / 365.25
    
    # Basic metrics
    total_return = ((final_capital / initial_capital) - 1) * 100
    annualized_return = ((final_capital / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # Risk metrics
    if len(returns) > 1:
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else 0
        sortino_ratio = (returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    else:
        volatility = 0
        sharpe_ratio = 0
        sortino_ratio = 0
    
    # Drawdown analysis
    equity_df['EquityPeak'] = equity_df['Equity'].cummax()
    equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['EquityPeak']) / equity_df['EquityPeak']
    max_drawdown = equity_df['Drawdown'].min() * 100
    
    # Trade analysis
    profits = [trade['profit'] for trade in trade_results]
    winning_trades = [p for p in profits if p > 0]
    losing_trades = [p for p in profits if p < 0]
    
    win_rate = (len(winning_trades) / len(profits)) * 100 if len(profits) > 0 else 0
    avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
    avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0
    profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if len(losing_trades) > 0 and sum(losing_trades) < 0 else float('inf')
    
    # Stop loss analysis (if enabled)
    stop_loss_exits = [t for t in trade_results if t['exit_reason'] == 'STOP_LOSS']
    stop_loss_rate = (len(stop_loss_exits) / len(trade_results)) * 100 if len(trade_results) > 0 else 0
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    metrics = {
        'Start Date': start_date.strftime('%Y-%m-%d'),
        'End Date': end_date.strftime('%Y-%m-%d'),
        'Total Trades': len(trade_results),
        'Final Account Balance': f"${final_capital:,.2f}",
        'Total Return': f"{total_return:.2f}%",
        'Annualized Return': f"{annualized_return:.2f}%",
        'Volatility (Annual)': f"{volatility:.2f}%",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Sortino Ratio': f"{sortino_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown:.2f}%",
        'Calmar Ratio': f"{calmar_ratio:.2f}",
        'Win Rate': f"{win_rate:.2f}%",
        'Average Win': f"${avg_win:.2f}",
        'Average Loss': f"${avg_loss:.2f}",
        'Profit Factor': f"{profit_factor:.2f}",
        'Winning Trades': len(winning_trades),
        'Losing Trades': len(losing_trades)
    }
    
    # Add stop loss metrics if enabled
    if use_stop_loss:
        metrics['Stop Loss Rate'] = f"{stop_loss_rate:.2f}%"
        metrics['Stop Loss Exits'] = len(stop_loss_exits)
    
    return metrics

def create_equity_curve_plot(equity_curve):
    """Create and display equity curve plot"""
    equity_df = pd.DataFrame(equity_curve, columns=['Time', 'Equity'])
    equity_df.set_index('Time', inplace=True)
    
    plt.figure(figsize=(14, 8))
    plt.plot(equity_df.index, equity_df['Equity'], label='ORB Strategy', linewidth=2)
    plt.axhline(y=initial_capital, color='red', linestyle='--', alpha=0.7, label='Starting Capital')
    
    stop_label = f" (Stop Loss: {stop_loss_type})" if use_stop_loss else " (No Stop Loss)"
    plt.title(f'Opening Range Breakout Strategy - Equity Curve{stop_label}', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Account Balance ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function"""
    print("=== Opening Range Breakout (ORB) Strategy Backtest ===")
    print(f"Initial Capital: ${initial_capital:,}")
    print(f"Commission per order: ${commission_per_order}")
    print(f"Contract Multiplier: {multiplier}")
    print(f"Breakout Offset: {breakout_offset} points")
    print(f"Filter: NR4 (trade day after narrow range 4)")
    print(f"Exit Time: 3:00 PM CT (instead of 4:00 PM)")
    print(f"Stop Loss Enabled: {use_stop_loss}")
    if use_stop_loss:
        print(f"Stop Loss Type: {stop_loss_type}")
        if stop_loss_type == 'FIXED_POINTS':
            print(f"Stop Loss Distance: {stop_loss_points} points")
    print("=" * 60)
    
    # Load and prepare data
    minute_data = load_and_prepare_data(data_file)
    
    # Run backtest
    trade_results, equity_curve, final_capital = run_backtest(minute_data)
    
    # Calculate and display performance metrics
    if len(trade_results) > 0:
        performance = calculate_performance_metrics(trade_results, equity_curve, initial_capital, final_capital)
        
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        for key, value in performance.items():
            print(f"{key}: {value}")
        
        # Create equity curve plot
        if len(equity_curve) > 0:
            create_equity_curve_plot(equity_curve)
        
        # Display sample trades
        print("\n" + "=" * 60)
        print("SAMPLE TRADES (First 10)")
        print("=" * 60)
        for i, trade in enumerate(trade_results[:10]):
            stop_info = f" | Stop: {trade['stop_loss']:.2f}" if trade['stop_loss'] else " | No Stop"
            print(f"Trade {i+1}: {trade['type']} | Entry: {trade['entry_price']:.2f} | Exit: {trade['exit_price']:.2f} | P&L: ${trade['profit']:.2f} | Reason: {trade['exit_reason']}{stop_info} | OR: {trade['or_range']:.2f}")
        
        # Show exit reason breakdown
        print("\n" + "=" * 60)
        print("EXIT REASON BREAKDOWN")
        print("=" * 60)
        exit_reasons = {}
        for trade in trade_results:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        for reason, count in exit_reasons.items():
            percentage = (count / len(trade_results)) * 100
            print(f"{reason}: {count} trades ({percentage:.1f}%)")
    
    else:
        print("No trades were executed during the backtest period.")

if __name__ == "__main__":
    main()


