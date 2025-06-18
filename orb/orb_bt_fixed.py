import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Strategy Parameters - FIXED VERSION
initial_capital = 30000
commission_per_order = 1.24  # Per contract per side  
multiplier = 5  # MES multiplier
breakout_offset = 0.5  # Reduced from 1.0 - more sensitive to breakouts

# Data file path
data_file = '../Data/es_1m_data.csv'

# Date range for backtest
start_date = '2000-01-01'
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
    """Create daily OHLC bars from 1-minute data for signal identification"""
    daily_data = []
    
    # Group by date
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

def identify_inside_days(daily_df):
    """Identify inside days (day's range is inside previous day's range)"""
    daily_df['Inside_Day'] = False
    
    for i in range(1, len(daily_df)):
        current = daily_df.iloc[i]
        previous = daily_df.iloc[i-1]
        
        # Inside day: High < Previous High AND Low > Previous Low
        if current['High'] < previous['High'] and current['Low'] > previous['Low']:
            daily_df.iloc[i, daily_df.columns.get_loc('Inside_Day')] = True
    
    return daily_df

def calculate_opening_range(minute_data, date):
    """Calculate 5-minute opening range for a specific date"""
    # Filter data for the specific date
    date_data = minute_data[minute_data['Time'].dt.date == date].copy()
    
    if len(date_data) == 0:
        return None
    
    # Find opening range (8:30-8:35 AM CT)
    date_data['Time_Only'] = date_data['Time'].dt.time
    opening_range_data = date_data[
        (date_data['Time_Only'] >= time(8, 30)) & 
        (date_data['Time_Only'] < time(8, 35))
    ]
    
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
        'sell_stop': or_low - breakout_offset
    }

def run_backtest(minute_data, daily_df):
    """Run the ORB backtest - FIXED VERSION"""
    
    # Initialize tracking variables
    capital = initial_capital
    in_position = False
    position = None
    trade_results = []
    equity_curve = []
    
    # FIXED: Use only Inside Days filter (not the restrictive combo)
    # Based on diagnosis: Inside days alone perform better than the combo
    qualifying_dates = daily_df[daily_df['Inside_Day']]['Date'].dt.date.tolist()
    
    # Alternative: Use no filter at all for maximum trades
    # qualifying_dates = daily_df['Date'].dt.date.tolist()
    
    print(f"Found {len(qualifying_dates)} qualifying days (Inside Days only)")
    
    if len(qualifying_dates) == 0:
        print("No qualifying trading days found!")
        return trade_results, equity_curve, capital
    
    # Process each qualifying date
    for trade_date in qualifying_dates:
        # Calculate opening range
        or_data = calculate_opening_range(minute_data, trade_date)
        if or_data is None:
            continue
        
        # Get minute data for this trading day
        day_data = minute_data[minute_data['Time'].dt.date == trade_date].copy()
        day_data['Time_Only'] = day_data['Time'].dt.time
        
        # Track daily state
        daily_position = None
        daily_equity = capital
        
        # Process each minute bar for this day
        for idx, row in day_data.iterrows():
            current_time = row['Time']
            current_time_only = row['Time_Only']
            current_price = row['Last']
            high = row['High']
            low = row['Low']
            
            # FIXED: Extended trading window - trades until 11:00 AM instead of 9:30 AM
            in_trading_window = (current_time_only >= time(8, 35)) and (current_time_only <= time(11, 0))
            
            # Entry logic - only during trading window and not in position
            if not in_position and in_trading_window and daily_position is None:
                entry_type = None
                entry_price = None
                
                # Check for breakout
                if high >= or_data['buy_stop']:
                    # Long breakout
                    entry_type = 'LONG'
                    entry_price = or_data['buy_stop']
                    stop_loss = or_data['sell_stop']
                elif low <= or_data['sell_stop']:
                    # Short breakout
                    entry_type = 'SHORT'
                    entry_price = or_data['sell_stop']
                    stop_loss = or_data['buy_stop']
                
                if entry_type:
                    # Enter position
                    num_contracts = 1  # Single contract per trade
                    capital -= commission_per_order * num_contracts  # Entry commission
                    
                    daily_position = {
                        'type': entry_type,
                        'entry_price': entry_price,
                        'entry_time': current_time,
                        'stop_loss': stop_loss,
                        'contracts': num_contracts
                    }
                    in_position = True
                    
                    logger.info(f"{entry_type} entry at {current_time} | Price: {entry_price:.2f} | Stop: {stop_loss:.2f} | OR Range: {or_data['or_range']:.2f}")
            
            # Exit logic - FIXED: Only exit at end of day (let winners run!)
            if in_position and daily_position:
                exit_triggered = False
                exit_price = None
                exit_reason = None
                
                # REMOVED: Tight stop loss exits - let trades run to EOD
                # This was causing premature exits and poor performance
                
                # Check end of day (4:00 PM CT)
                if current_time_only >= time(16, 0):
                    exit_triggered = True
                    exit_price = current_price
                    exit_reason = 'END_OF_DAY'
                
                if exit_triggered:
                    # Calculate profit
                    if daily_position['type'] == 'LONG':
                        price_diff = exit_price - daily_position['entry_price']
                    else:  # SHORT
                        price_diff = daily_position['entry_price'] - exit_price
                    
                    profit = price_diff * multiplier * daily_position['contracts'] - commission_per_order * daily_position['contracts']
                    
                    # Record trade
                    trade_results.append({
                        'entry_time': daily_position['entry_time'],
                        'exit_time': current_time,
                        'type': daily_position['type'],
                        'entry_price': daily_position['entry_price'],
                        'exit_price': exit_price,
                        'profit': profit,
                        'contracts': daily_position['contracts'],
                        'exit_reason': exit_reason,
                        'or_range': or_data['or_range']
                    })
                    
                    capital += profit
                    in_position = False
                    daily_position = None
                    
                    logger.info(f"{exit_reason} exit at {current_time} | Price: {exit_price:.2f} | Profit: {profit:.2f}")
            
            # Calculate mark-to-market equity
            if in_position and daily_position:
                if daily_position['type'] == 'LONG':
                    unrealized = (current_price - daily_position['entry_price']) * multiplier * daily_position['contracts']
                else:  # SHORT
                    unrealized = (daily_position['entry_price'] - current_price) * multiplier * daily_position['contracts']
                daily_equity = capital + unrealized
            else:
                daily_equity = capital
        
        # Record daily equity
        if len(day_data) > 0:
            equity_curve.append((day_data.iloc[-1]['Time'], daily_equity))
        
        # Force close any remaining position at end of day
        if in_position and daily_position:
            final_price = day_data.iloc[-1]['Last']
            
            if daily_position['type'] == 'LONG':
                price_diff = final_price - daily_position['entry_price']
            else:  # SHORT
                price_diff = daily_position['entry_price'] - final_price
            
            profit = price_diff * multiplier * daily_position['contracts'] - commission_per_order * daily_position['contracts']
            
            trade_results.append({
                'entry_time': daily_position['entry_time'],
                'exit_time': day_data.iloc[-1]['Time'],
                'type': daily_position['type'],
                'entry_price': daily_position['entry_price'],
                'exit_price': final_price,
                'profit': profit,
                'contracts': daily_position['contracts'],
                'exit_reason': 'FORCE_CLOSE',
                'or_range': or_data['or_range']
            })
            
            capital += profit
            in_position = False
            daily_position = None
            
            # Update final equity
            if len(equity_curve) > 0:
                equity_curve[-1] = (equity_curve[-1][0], capital)
    
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
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
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

def create_equity_curve_plot(equity_curve):
    """Create and display equity curve plot"""
    equity_df = pd.DataFrame(equity_curve, columns=['Time', 'Equity'])
    equity_df.set_index('Time', inplace=True)
    
    plt.figure(figsize=(14, 8))
    plt.plot(equity_df.index, equity_df['Equity'], label='ORB Strategy (Fixed)', linewidth=2, color='green')
    plt.axhline(y=initial_capital, color='red', linestyle='--', alpha=0.7, label='Starting Capital')
    
    plt.title('Opening Range Breakout (ORB) Strategy - FIXED VERSION', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Account Balance ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function"""
    print("=== Opening Range Breakout (ORB) Strategy Backtest - FIXED VERSION ===")
    print(f"Initial Capital: ${initial_capital:,}")
    print(f"Commission per order: ${commission_per_order}")
    print(f"Contract Multiplier: {multiplier}")
    print(f"Breakout Offset: {breakout_offset} points (REDUCED)")
    print("FIXES APPLIED:")
    print("- Removed overly restrictive Inside+NR4 filter")
    print("- Reduced breakout offset from 1.0 to 0.5 points")
    print("- Extended trading window to 11:00 AM")
    print("- Removed tight stop losses (let winners run to EOD)")
    print("- Added minimum opening range filter (>1 point)")
    print("=" * 80)
    
    # Load and prepare data
    minute_data = load_and_prepare_data(data_file)
    
    # Create daily bars for signal identification
    daily_df = create_daily_bars(minute_data)
    print(f"Created {len(daily_df)} daily bars")
    
    # Identify inside days
    daily_df = identify_inside_days(daily_df)
    
    inside_days = daily_df['Inside_Day'].sum()
    print(f"Inside Days: {inside_days}")
    
    # Run backtest
    trade_results, equity_curve, final_capital = run_backtest(minute_data, daily_df)
    
    # Calculate and display performance metrics
    if len(trade_results) > 0:
        performance = calculate_performance_metrics(trade_results, equity_curve, initial_capital, final_capital)
        
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY - FIXED VERSION")
        print("=" * 80)
        for key, value in performance.items():
            print(f"{key}: {value}")
        
        # Create equity curve plot
        if len(equity_curve) > 0:
            create_equity_curve_plot(equity_curve)
        
        # Display sample trades
        print("\n" + "=" * 80)
        print("SAMPLE TRADES (First 10)")
        print("=" * 80)
        for i, trade in enumerate(trade_results[:10]):
            print(f"Trade {i+1}: {trade['type']} | Entry: {trade['entry_price']:.2f} | Exit: {trade['exit_price']:.2f} | P&L: ${trade['profit']:.2f} | OR Range: {trade['or_range']:.2f}")
    
    else:
        print("No trades were executed during the backtest period.")

if __name__ == "__main__":
    main() 