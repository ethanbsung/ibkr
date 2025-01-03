import pandas as pd
import datetime

def generate_pinescript_code(entries_df, exits_df):
    script = """
//@version=6
indicator("Backtest Trade Visualization", overlay=true)

// === Trade Entries ===
var entry_timestamps = array.new<int>()
var entry_prices = array.new<float>()
var entry_types = array.new<string>()

if barstate.isfirst
"""
    # Generate entries
    for index, row in entries_df.iterrows():
        # Convert timestamp string to datetime and then to Unix timestamp (milliseconds)
        trade_time = pd.to_datetime(row['timestamp'])
        unix_timestamp = int(trade_time.timestamp() * 1000)
        trade_price = row['price']
        trade_type = row['type']
        script += f'    array.push(entry_timestamps, {unix_timestamp})\n'
        script += f'    array.push(entry_prices, {trade_price})\n'
        script += f'    array.push(entry_types, "{trade_type}")\n'

    script += """
    
// === Trade Exits ===
var exit_timestamps = array.new<int>()
var exit_prices = array.new<float>()
var exit_types = array.new<string>()

if barstate.isfirst
"""
    # Generate exits
    for index, row in exits_df.iterrows():
        # Convert timestamp string to datetime and then to Unix timestamp (milliseconds)
        trade_time = pd.to_datetime(row['timestamp'])
        unix_timestamp = int(trade_time.timestamp() * 1000)
        trade_price = row['price']
        trade_type = row['type']
        script += f'    array.push(exit_timestamps, {unix_timestamp})\n'
        script += f'    array.push(exit_prices, {trade_price})\n'
        script += f'    array.push(exit_types, "{trade_type}")\n'

    script += """
    
// === Plot Trade Entries ===
for i = 0 to array.size(entry_timestamps) - 1
    entry_time = array.get(entry_timestamps, i)
    entry_price = array.get(entry_prices, i)
    entry_type = array.get(entry_types, i)
    
    label.new(x=entry_time, y=entry_price, text=entry_type, style=label.style_label_up, color=(entry_type == "BUY" ? color.green : color.red), textcolor=color.white, size=size.small, tooltip="Trade Entry")

// === Plot Trade Exits and Draw Lines ===
for i = 0 to array.size(exit_timestamps) - 1
    exit_time = array.get(exit_timestamps, i)
    exit_price = array.get(exit_prices, i)
    exit_type = array.get(exit_types, i)
    
    label.new(x=exit_time, y=exit_price, text=exit_type, style=label.style_label_down, color=(exit_type == "TAKE PROFIT" ? color.blue : color.orange), textcolor=color.white, size=size.small, tooltip="Trade Exit")
    
    if i < array.size(entry_timestamps)
        entry_time = array.get(entry_timestamps, i)
        entry_price = array.get(entry_prices, i)
        line.new(x1=entry_time, y1=entry_price, x2=exit_time, y2=exit_price, color=(exit_type == "TAKE PROFIT" ? color.blue : color.orange), width=1, style=line.style_dashed)
"""

    return script

# Example usage
entries_df = pd.read_csv('trade_entries.csv')
exits_df = pd.read_csv('trade_exits.csv')

# Generate the complete Pine Script
complete_pinescript = generate_pinescript_code(entries_df, exits_df)

# Save to a .pine file
with open('complete_pinescript_trade_visualization.pine', 'w') as file:
    file.write(complete_pinescript)

print("Complete Pine Script generated and saved to 'complete_pinescript_trade_visualization.pine'.")