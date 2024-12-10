import time
import pandas as pd
import numpy as np
from ib_insync import *

# ----------------------------
# User Configurations
# ----------------------------
HOST = '127.0.0.1'
PORT = 7497            # Paper TWS
CLIENT_ID = 1         

# Strategy Parameters
bollinger_period = 15
bollinger_stddev = 2
stop_loss_points = 3
take_profit_points = 23
commission_per_side = 0.47
total_commission = commission_per_side * 2
initial_cash = 5000

# Contracts
es_contract = Future(
    symbol='ES',
    exchange='CME',
    currency='USD',
    lastTradeDateOrContractMonth='202412'  # Nearest Contract
)

mes_contract = Future(
    symbol='MES',
    exchange='CME',
    currency='USD',
    lastTradeDateOrContractMonth='202412'  # Nearest Contract
)

# ----------------------------
# Connect to IBKR
# ----------------------------
ib = IB()
ib.connect(HOST, PORT, CLIENT_ID)

# Qualify Contracts
es_contract = ib.qualifyContracts(es_contract)[0]
mes_contract = ib.qualifyContracts(mes_contract)[0]

# ----------------------------
# Initialize Historical Data
# ----------------------------
bars = ib.reqHistoricalData(
    es_contract,
    endDateTime='',
    durationStr='3 D',
    barSizeSetting='30 mins',
    whatToShow='TRADES',
    useRTH=False,
    formatDate=1
)

# Initialize DataFrame
df = util.df(bars)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# ----------------------------
# Initialize Strategy State Variables
# ----------------------------
position_size = 0
entry_price = None
position_type = None
cash = initial_cash
trade_results = []
balance_series = [initial_cash]
stop_loss_price = None
take_profit_price = None
last_signal_bar_time = None

# ----------------------------
# Helper Functions
# ----------------------------
def update_bollinger_bands(dataframe):
    dataframe['ma'] = dataframe['close'].rolling(window=bollinger_period).mean()
    dataframe['std'] = dataframe['close'].rolling(window=bollinger_period).std()
    dataframe['upper_band'] = dataframe['ma'] + (bollinger_stddev * dataframe['std'])
    dataframe['lower_band'] = dataframe['ma'] - (bollinger_stddev * dataframe['std'])

def place_bracket_order(action, quantity, entry, take_profit_offset, stop_loss_offset):
    """
    Place a bracket order on MES (execution contract)
    """
    if action == 'BUY':
        take_profit = entry + take_profit_offset
        stop_loss = entry - stop_loss_offset
    else:
        take_profit = entry - take_profit_offset
        stop_loss = entry + stop_loss_offset

    parent = MarketOrder(action, quantity)
    parent.orderId = ib.client.getReqId()

    tp = LimitOrder('SELL' if action == 'BUY' else 'BUY', quantity, take_profit)
    tp.orderId = ib.client.getReqId()
    tp.parentId = parent.orderId

    sl = StopOrder('SELL' if action == 'BUY' else 'BUY', quantity, stop_loss)
    sl.orderId = ib.client.getReqId()
    sl.parentId = parent.orderId

    # OCA Group ensures only one of TP/SL fills
    oca_group = f"OCA_{parent.orderId}"
    tp.ocaGroup = oca_group
    tp.ocaType = 1
    sl.ocaGroup = oca_group
    sl.ocaType = 1

    # Place orders
    trade = ib.placeOrder(mes_contract, parent)
    time.sleep(1)  # Ensure parent registers
    ib.placeOrder(mes_contract, tp)
    ib.placeOrder(mes_contract, sl)

    return parent.orderId, tp.orderId, sl.orderId

def current_position():
    """
    Check if we have an open MES position.
    """
    for p in ib.positions():
        if p.contract.conId == mes_contract.conId:
            return p
    return None

# ----------------------------
# Real-Time Data Subscription
# ----------------------------
realtime_bars = ib.reqRealTimeBars(es_contract, whatToShow='TRADES', useRTH=False, barSize=5)
partial_bars = []

def onBarUpdate(bars, hasNewBar):
    if not hasNewBar:
        return

    latest_bar = bars[-1]
    bar_dt = pd.to_datetime(latest_bar.time, unit='s', tz='UTC')

    # Add partial bars for aggregation
    partial_bars.append(latest_bar)

    # Check if we have a full 30-minute bar
    if bar_dt.minute % 30 == 0 and bar_dt.second == 0:
        finalize_30min_bar()

def finalize_30min_bar():
    if not partial_bars:
        return

    # Create a 30-minute bar from partial bars
    highs = [b.high for b in partial_bars]
    lows = [b.low for b in partial_bars]
    opens = partial_bars[0].open
    closes = partial_bars[-1].close
    volumes = sum(b.volume for b in partial_bars)
    bar_dt = pd.to_datetime(partial_bars[-1].time, unit='s', tz='UTC')

    # Update DataFrame
    global df, position_size, position_type
    global entry_price, stop_loss_price, take_profit_price
    global trade_results, balance_series, cash

    new_row = {
        'open': opens,
        'high': max(highs),
        'low': min(lows),
        'close': closes,
        'volume': volumes
    }
    df.loc[bar_dt] = new_row

    # Keep a limited DataFrame size
    if len(df) > bollinger_period * 10:
        df = df.iloc[-(bollinger_period * 10):]

    # Recalculate Bollinger Bands
    update_bollinger_bands(df)

    # Check if we can trade
    if len(df) < bollinger_period:
        partial_bars.clear()
        return

    current_price = df['close'].iloc[-1]
    upper_band = df['upper_band'].iloc[-1]
    lower_band = df['lower_band'].iloc[-1]

    # Check current MES position
    pos = current_position()
    if pos:
        pass  # Position already open; rely on IBKR orders
    else:
        # Enter long if price < lower band
        if current_price < lower_band:
            print(f"Signal: Enter LONG on MES at {current_price:.2f}")
            place_bracket_order('BUY', 1, current_price, take_profit_points, stop_loss_points)
            position_size = 1
            position_type = 'long'
            entry_price = current_price

        # Enter short if price > upper band
        elif current_price > upper_band:
            print(f"Signal: Enter SHORT on MES at {current_price:.2f}")
            place_bracket_order('SELL', 1, current_price, take_profit_points, stop_loss_points)
            position_size = 1
            position_type = 'short'
            entry_price = current_price

    # Reset partial bars
    partial_bars.clear()

# Assign callback
realtime_bars.updateEvent += onBarUpdate

# ----------------------------
# Main Trading Loop
# ----------------------------
try:
    while True:
        ib.sleep(1)
        # Check position status
        if position_size != 0 and current_position() is None:
            exit_price = df['close'].iloc[-1]
            pnl = ((exit_price - entry_price) * 5 if position_type == 'long' else 
                   (entry_price - exit_price) * 5) - total_commission
            cash += pnl
            balance_series.append(cash)
            print(f"Position closed | PnL: {pnl:.2f}")

            position_size = 0
            entry_price = None
            stop_loss_price = None
            take_profit_price = None

except KeyboardInterrupt:
    print("Algorithm stopped.")
finally:
    ib.disconnect()