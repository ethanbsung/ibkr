import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time

def load_and_analyze_data():
    """Load data and analyze opening range characteristics"""
    print("=== ORB Strategy Diagnosis ===\n")
    
    # Load data
    data = pd.read_csv('../Data/es_1m_data.csv')
    data['Time'] = pd.to_datetime(data['Time'])
    data.sort_values('Time', inplace=True)
    
    # Filter to test period
    start_date = '2000-01-01'
    end_date = '2020-01-01'
    data = data[(data['Time'] >= start_date) & (data['Time'] <= end_date)]
    print(f"Data period: {data['Time'].min()} to {data['Time'].max()}")
    print(f"Total bars: {len(data)}")
    
    return data

def analyze_opening_ranges(data):
    """Analyze opening range characteristics"""
    print("\n=== Opening Range Analysis ===")
    
    data['Date'] = data['Time'].dt.date
    data['Time_Only'] = data['Time'].dt.time
    
    opening_ranges = []
    
    for date in data['Date'].unique()[:100]:  # Analyze first 100 days
        day_data = data[data['Date'] == date]
        
        # Get opening range (8:30-8:35 AM)
        or_data = day_data[
            (day_data['Time_Only'] >= time(8, 30)) & 
            (day_data['Time_Only'] < time(8, 35))
        ]
        
        if len(or_data) > 0:
            or_high = or_data['High'].max()
            or_low = or_data['Low'].min()
            or_range = or_high - or_low
            
            # Get full day data
            full_day = day_data[
                (day_data['Time_Only'] >= time(8, 30)) & 
                (day_data['Time_Only'] <= time(16, 0))
            ]
            
            if len(full_day) > 0:
                day_high = full_day['High'].max()
                day_low = full_day['Low'].min()
                
                # Calculate breakout statistics
                broke_above = day_high > (or_high + 1.0)
                broke_below = day_low < (or_low - 1.0)
                
                opening_ranges.append({
                    'date': date,
                    'or_range': or_range,
                    'or_high': or_high,
                    'or_low': or_low,
                    'day_high': day_high,
                    'day_low': day_low,
                    'broke_above': broke_above,
                    'broke_below': broke_below,
                    'range_vs_breakout': or_range / 1.0 if or_range > 0 else 0
                })
    
    or_df = pd.DataFrame(opening_ranges)
    
    print(f"Average opening range: {or_df['or_range'].mean():.2f} points")
    print(f"Median opening range: {or_df['or_range'].median():.2f} points")
    print(f"Breakout above rate: {or_df['broke_above'].mean()*100:.1f}%")
    print(f"Breakout below rate: {or_df['broke_below'].mean()*100:.1f}%")
    print(f"Either breakout rate: {(or_df['broke_above'] | or_df['broke_below']).mean()*100:.1f}%")
    print(f"Range/Breakout ratio: {or_df['range_vs_breakout'].mean():.2f}")
    
    return or_df

def test_different_configurations(data):
    """Test different ORB configurations"""
    print("\n=== Testing Different Configurations ===")
    
    configs = [
        {'filter': 'none', 'offset': 0.5, 'name': 'No Filter, 0.5pt offset'},
        {'filter': 'none', 'offset': 1.0, 'name': 'No Filter, 1.0pt offset'},
        {'filter': 'none', 'offset': 2.0, 'name': 'No Filter, 2.0pt offset'},
        {'filter': 'inside', 'offset': 1.0, 'name': 'Inside Days Only, 1.0pt'},
        {'filter': 'nr4', 'offset': 1.0, 'name': 'NR4 Days Only, 1.0pt'},
        {'filter': 'both', 'offset': 1.0, 'name': 'Inside + NR4, 1.0pt (Original)'},
    ]
    
    # Create daily bars for signal analysis
    daily_data = create_daily_bars(data)
    daily_data = identify_signals(daily_data)
    
    results = []
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        
        # Filter qualifying days
        if config['filter'] == 'none':
            qualifying_dates = daily_data['Date'].dt.date.tolist()
        elif config['filter'] == 'inside':
            qualifying_dates = daily_data[daily_data['Inside_Day']]['Date'].dt.date.tolist()
        elif config['filter'] == 'nr4':
            qualifying_dates = daily_data[daily_data['NR4']]['Date'].dt.date.tolist()
        else:  # both
            qualifying_dates = daily_data[daily_data['Inside_Day'] & daily_data['NR4']]['Date'].dt.date.tolist()
        
        print(f"Qualifying days: {len(qualifying_dates)}")
        
        if len(qualifying_dates) > 0:
            # Run quick simulation
            trades = simulate_trades(data, qualifying_dates[:50], config['offset'])  # Test first 50 days
            
            if len(trades) > 0:
                profits = [t['profit'] for t in trades]
                win_rate = len([p for p in profits if p > 0]) / len(profits) * 100
                avg_profit = np.mean(profits)
                total_profit = sum(profits)
                
                results.append({
                    'config': config['name'],
                    'trades': len(trades),
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'total_profit': total_profit
                })
                
                print(f"Trades: {len(trades)}, Win Rate: {win_rate:.1f}%, Avg P&L: ${avg_profit:.2f}")
    
    return results

def create_daily_bars(minute_data):
    """Create daily bars from minute data"""
    daily_data = []
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

def identify_signals(daily_df):
    """Identify inside days and NR4 days"""
    daily_df['Inside_Day'] = False
    daily_df['NR4'] = False
    
    # Inside days
    for i in range(1, len(daily_df)):
        current = daily_df.iloc[i]
        previous = daily_df.iloc[i-1]
        
        if current['High'] < previous['High'] and current['Low'] > previous['Low']:
            daily_df.iloc[i, daily_df.columns.get_loc('Inside_Day')] = True
    
    # NR4 days
    for i in range(3, len(daily_df)):
        current_range = daily_df.iloc[i]['Range']
        previous_3_ranges = daily_df.iloc[i-3:i]['Range'].values
        
        if current_range < min(previous_3_ranges):
            daily_df.iloc[i, daily_df.columns.get_loc('NR4')] = True
    
    return daily_df

def simulate_trades(minute_data, qualifying_dates, breakout_offset):
    """Quick trade simulation"""
    trades = []
    
    for trade_date in qualifying_dates[:20]:  # Limit for speed
        # Calculate opening range
        date_data = minute_data[minute_data['Time'].dt.date == trade_date].copy()
        if len(date_data) == 0:
            continue
            
        date_data['Time_Only'] = date_data['Time'].dt.time
        or_data = date_data[
            (date_data['Time_Only'] >= time(8, 30)) & 
            (date_data['Time_Only'] < time(8, 35))
        ]
        
        if len(or_data) == 0:
            continue
        
        or_high = or_data['High'].max()
        or_low = or_data['Low'].min()
        buy_stop = or_high + breakout_offset
        sell_stop = or_low - breakout_offset
        
        # Check for breakouts during trading window
        trading_data = date_data[
            (date_data['Time_Only'] >= time(8, 35)) & 
            (date_data['Time_Only'] <= time(9, 30))
        ]
        
        entry_found = False
        for _, row in trading_data.iterrows():
            if not entry_found:
                if row['High'] >= buy_stop:
                    # Long entry
                    entry_price = buy_stop
                    stop_loss = sell_stop
                    
                    # Find exit (EOD or stop)
                    exit_price = find_exit(date_data, row['Time'], stop_loss, 'LONG')
                    profit = (exit_price - entry_price) * 5 - 2.48  # MES multiplier and commission
                    
                    trades.append({
                        'date': trade_date,
                        'type': 'LONG',
                        'entry': entry_price,
                        'exit': exit_price,
                        'profit': profit
                    })
                    entry_found = True
                    
                elif row['Low'] <= sell_stop:
                    # Short entry
                    entry_price = sell_stop
                    stop_loss = buy_stop
                    
                    # Find exit
                    exit_price = find_exit(date_data, row['Time'], stop_loss, 'SHORT')
                    profit = (entry_price - exit_price) * 5 - 2.48
                    
                    trades.append({
                        'date': trade_date,
                        'type': 'SHORT',
                        'entry': entry_price,
                        'exit': exit_price,
                        'profit': profit
                    })
                    entry_found = True
    
    return trades

def find_exit(day_data, entry_time, stop_loss, position_type):
    """Find exit price (stop loss or EOD)"""
    remaining_data = day_data[day_data['Time'] > entry_time]
    
    for _, row in remaining_data.iterrows():
        # Check stop loss
        if position_type == 'LONG' and row['Low'] <= stop_loss:
            return stop_loss
        elif position_type == 'SHORT' and row['High'] >= stop_loss:
            return stop_loss
    
    # Exit at end of day
    if len(remaining_data) > 0:
        return remaining_data.iloc[-1]['Last']
    else:
        return stop_loss  # Fallback

def main():
    # Load and analyze data
    data = load_and_analyze_data()
    
    # Analyze opening ranges
    or_analysis = analyze_opening_ranges(data)
    
    # Test different configurations
    test_results = test_different_configurations(data)
    
    print("\n=== CONFIGURATION COMPARISON ===")
    for result in test_results:
        print(f"{result['config']}: {result['trades']} trades, {result['win_rate']:.1f}% win rate, ${result['total_profit']:.2f} total P&L")

if __name__ == "__main__":
    main() 