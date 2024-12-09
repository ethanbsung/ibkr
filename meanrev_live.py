import pandas as pd
import numpy as np
import datetime
import time
from ib_insync import *

# --- Configuration Parameters ---
IB_HOST = '127.0.0.1'        # IB Gateway/TWS host
IB_PORT = 7497               # IB Gateway/TWS port (7497 for TWS Paper)
CLIENT_ID = 1                # Unique client ID
DATA_SYMBOL = 'ES'            # E-mini S&P 500 for data
DATA_EXPIRY = '202412'        # December 2024
DATA_EXCHANGE = 'GLOBEX'      # Exchange for ES

EXEC_SYMBOL = 'MES'           # Micro E-mini S&P 500 for execution
EXEC_EXPIRY = '202412'        # December 2024
EXEC_EXCHANGE = 'GLOBEX'      # Exchange for MES

CURRENCY = 'USD'

INITIAL_CASH = 5000           # Starting cash
POSITION_SIZE = 1             # Number of MES contracts per trade

BOLLINGER_PERIOD = 15
BOLLINGER_STDDEV = 2
STOP_LOSS_DISTANCE = 5        # Points away from entry
TAKE_PROFIT_DISTANCE = 10     # Points away from entry

# --- Connect to IBKR ---
ib = IB()
print("Connecting to IBKR...")
ib.connect(host=IB_HOST, port=IB_PORT, clientId=CLIENT_ID)
print("Connected.")

# --- Define Contracts ---
# ES Contract for Data
es_contract = Future(symbol=DATA_SYMBOL, 
                     lastTradeDateOrContractMonth=DATA_EXPIRY,
                     exchange=DATA_EXCHANGE,
                     currency=CURRENCY)

# MES Contract for Execution
mes_contract = Future(symbol=EXEC_SYMBOL, 
                      lastTradeDateOrContractMonth=EXEC_EXPIRY,
                      exchange=EXEC_EXCHANGE,
                      currency=CURRENCY)

# --- Request Historical Data for ES (Data Contract) ---
print("Requesting historical ES data...")
bars_30m = ib.reqHistoricalData(
    es_contract, 
    endDateTime='', 
    durationStr='60 D',            # Last 60 days
    barSizeSetting='30 mins', 
    whatToShow='TRADES', 
    useRTH=False, 
    formatDate=1,
    keepUpToDate=False
)

bars_1m = ib.reqHistoricalData(
    es_contract, 
    endDateTime='', 
    durationStr='2 D',
    barSizeSetting='1 min', 
    whatToShow='TRADES', 
    useRTH=False, 
    formatDate=1,
    keepUpToDate=False
)

# Convert to DataFrames
df_30m = util.df(bars_30m)
df_30m.set_index('date', inplace=True)
df_30m.sort_index(inplace=True)

df_1m = util.df(bars_1m)
df_1m.set_index('date', inplace=True)
df_1m.sort_index(inplace=True)

# --- Calculate Bollinger Bands for ES ---
def calculate_bollinger_bands(df, period=15, stddev=2):
    df['ma'] = df['close'].rolling(window=period).mean()
    df['std'] = df['close'].rolling(window=period).std()
    df['upper_band'] = df['ma'] + (stddev * df['std'])
    df['lower_band'] = df['ma'] - (stddev * df['std'])
    return df

df_30m = calculate_bollinger_bands(df_30m, BOLLINGER_PERIOD, BOLLINGER_STDDEV)
df_30m.dropna(inplace=True)

# --- Initialize Variables ---
cash = INITIAL_CASH
balance_series = [INITIAL_CASH]
position = None  # No open position initially

# --- Define Bracket Order Function ---
def create_bracket_order(action, quantity, limit_price, stop_loss_price, take_profit_price):
    """
    Creates a bracket order with a parent order, a take profit limit order, and a stop loss order.
    """
    # Main order: Market order to enter the position
    parent_order = MarketOrder(action, quantity)
    
    # Determine actions for take profit and stop loss based on the main action
    if action == 'BUY':
        take_profit_action = 'SELL'
        stop_loss_action = 'SELL'
    else:
        take_profit_action = 'BUY'
        stop_loss_action = 'BUY'
    
    # Take Profit Order: Limit Order
    take_profit_order = LimitOrder(
        take_profit_action, 
        quantity, 
        take_profit_price
    )
    
    # Stop Loss Order: Stop Order
    stop_loss_order = StopOrder(
        stop_loss_action, 
        quantity, 
        stop_loss_price
    )
    
    # Attach parentId to child orders
    take_profit_order.parentId = parent_order.orderId
    stop_loss_order.parentId = parent_order.orderId
    
    return [parent_order, take_profit_order, stop_loss_order]

# --- Order Filled Callback ---
def on_order_filled(trade, fill, position_type, take_profit_order, stop_loss_order):
    """
    Handles the event when the parent order is filled and places the take profit and stop loss orders.
    """
    global cash, position
    if fill:
        print(f"Parent order filled: {trade.order.action} {trade.order.totalQuantity} @ {fill.price}")
        # Update cash based on position type
        if position_type == 'long':
            entry_price = fill.price
            print(f"Entered LONG at {entry_price}")
        elif position_type == 'short':
            entry_price = fill.price
            print(f"Entered SHORT at {entry_price}")
        
        # Place take profit and stop loss orders
        ib.placeOrder(mes_contract, take_profit_order)
        ib.placeOrder(mes_contract, stop_loss_order)
        print(f"Placed TAKE PROFIT and STOP LOSS orders for {position_type.upper()} position.")

# --- Subscribe to Real-Time ES Bars ---
print("Subscribing to real-time ES bars...")
ib.reqRealTimeBars(es_contract, whatToShow='TRADES', useRTH=False, realTimeBarsOptions=[])

# --- Initialize Real-Time Bar Aggregation ---
current_minute = None
current_min_bars = []
thirty_min_bars = df_30m.copy()

# --- Define Real-Time Bar Handler ---
def on_realtime_bar(tick):
    global current_minute, current_min_bars, thirty_min_bars, position, cash
    
    # Convert timestamp to UTC
    bar_time = datetime.datetime.fromtimestamp(tick.time, datetime.timezone.utc)
    bar_minute = bar_time.replace(second=0, microsecond=0)
    
    # Check if a new minute has started
    if current_minute and bar_minute != current_minute:
        if current_min_bars:
            # Finalize the 1-min bar
            df_new = pd.DataFrame(current_min_bars)
            open_ = df_new.iloc[0]['open']
            high_ = df_new['high'].max()
            low_ = df_new['low'].min()
            close_ = df_new.iloc[-1]['close']
            volume_ = df_new['volume'].sum()
            
            new_bar = pd.DataFrame([{
                'open': open_,
                'high': high_,
                'low': low_,
                'close': close_,
                'volume': volume_
            }], index=[current_minute])
            
            # Append to 1-min DataFrame
            global df_1m
            df_1m = pd.concat([df_1m, new_bar])
            df_1m = df_1m[~df_1m.index.duplicated(keep='last')]
            
            # Aggregate into 30-min bars
            if current_minute.minute % 30 == 0:
                start_30min = current_minute - pd.Timedelta(minutes=29)
                last_30min = df_1m.loc[start_30min:current_minute]
                
                if len(last_30min) == 30:
                    open_30 = last_30min.iloc[0]['open']
                    high_30 = last_30min['high'].max()
                    low_30 = last_30min['low'].min()
                    close_30 = last_30min.iloc[-1]['close']
                    volume_30 = last_30min['volume'].sum()
                    
                    new_30_bar = pd.DataFrame([{
                        'open': open_30,
                        'high': high_30,
                        'low': low_30,
                        'close': close_30,
                        'volume': volume_30
                    }], index=[current_minute])
                    
                    thirty_min_bars = pd.concat([thirty_min_bars, new_30_bar])
                    thirty_min_bars.sort_index(inplace=True)
                    
                    # Recalculate Bollinger Bands
                    thirty_min_bars = calculate_bollinger_bands(thirty_min_bars, BOLLINGER_PERIOD, BOLLINGER_STDDEV)
                    thirty_min_bars.dropna(inplace=True)
                    
                    # Get the latest 30-min bar
                    current_30_bar = thirty_min_bars.iloc[-1]
                    current_price = current_30_bar['close']
                    current_time = thirty_min_bars.index[-1]
                    
                    print(f"\nNew 30-min bar closed at {current_time} UTC with close price: {current_price}")
                    print(f"Bollinger Bands - Upper: {current_30_bar['upper_band']}, Lower: {current_30_bar['lower_band']}")
                    
                    # --- Trading Logic ---
                    if position is None:
                        # No open position, check for entry signals
                        if current_price < current_30_bar['lower_band']:
                            # Enter Long
                            print("Entry Signal: LONG")
                            bracket = create_bracket_order(
                                action='BUY',
                                quantity=POSITION_SIZE,
                                limit_price=current_price,
                                stop_loss_price=current_price - STOP_LOSS_DISTANCE,
                                take_profit_price=current_price + TAKE_PROFIT_DISTANCE
                            )
                            # Place Bracket Order
                            parent_order = bracket[0]
                            take_profit_order = bracket[1]
                            stop_loss_order = bracket[2]
                            
                            trade = ib.placeOrder(mes_contract, parent_order)
                            # Attach callback for when the parent order is filled
                            trade.filledEvent += lambda fill: on_order_filled(trade, fill, 'long', take_profit_order, stop_loss_order)
                            
                            position = {
                                'type': 'long',
                                'entry_price': current_price,
                                'entry_time': current_time
                            }
                            
                        elif current_price > current_30_bar['upper_band']:
                            # Enter Short
                            print("Entry Signal: SHORT")
                            bracket = create_bracket_order(
                                action='SELL',
                                quantity=POSITION_SIZE,
                                limit_price=current_price,
                                stop_loss_price=current_price + STOP_LOSS_DISTANCE,
                                take_profit_price=current_price - TAKE_PROFIT_DISTANCE
                            )
                            # Place Bracket Order
                            parent_order = bracket[0]
                            take_profit_order = bracket[1]
                            stop_loss_order = bracket[2]
                            
                            trade = ib.placeOrder(mes_contract, parent_order)
                            # Attach callback for when the parent order is filled
                            trade.filledEvent += lambda fill: on_order_filled(trade, fill, 'short', take_profit_order, stop_loss_order)
                            
                            position = {
                                'type': 'short',
                                'entry_price': current_price,
                                'entry_time': current_time
                            }
                    
                    # Reset current_min_bars for the new minute
        # Reset current_min_bars for the new minute
        current_min_bars = []
    
    # Append current 5-second bar
    current_minute = bar_minute
    current_min_bars.append({
        'open': tick.open,
        'high': tick.high,
        'low': tick.low,
        'close': tick.close,
        'volume': tick.volume
    })

# --- Assign the Real-Time Bar Handler ---
# Real-time bars are received as RealTimeBar objects
def handle_real_time_bars(ticker, tickType, value, attrib):
    """
    Handles incoming real-time bars for ES.
    """
    if tickType == 'RT_BID_SIZE':
        return  # Ignore non-price ticks
    if hasattr(ticker, 'rtBar'):
        on_realtime_bar(ticker.rtBar)

ib.pendingTickersEvent += lambda tickers: [handle_real_time_bars(ticker, None, None, None) for ticker in tickers]

# --- Start the Event Loop ---
print("Starting event loop...")
try:
    ib.run()
except KeyboardInterrupt:
    print("Interrupt received, shutting down...")
finally:
    ib.disconnect()
    print("Disconnected from IBKR.")