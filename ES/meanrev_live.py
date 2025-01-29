import pandas as pd
import numpy as np
import datetime
from ib_insync import *
import logging
import sys
import pytz
import json
from datetime import timedelta
import os

# --- Configuration Parameters ---
IB_HOST = '127.0.0.1'        # IBKR Gateway/TWS host
IB_PORT = 7497               # IBKR Gateway/TWS paper trading port
CLIENT_ID = 1                # Unique client ID

DATA_SYMBOL = 'ES'           # E-mini S&P 500 for data
DATA_EXPIRY = '202503'       # March 2025
DATA_EXCHANGE = 'CME'        # Exchange for ES

EXEC_SYMBOL = 'MES'          # Micro E-mini S&P 500 for execution
EXEC_EXPIRY = '202503'       # March 2025
EXEC_EXCHANGE = 'CME'        # Exchange for MES
CURRENCY = 'USD'

INITIAL_CASH = 5000          # Starting cash
POSITION_SIZE = 1            # Number of MES contracts per trade

BOLLINGER_PERIOD = 15
BOLLINGER_STDDEV = 2
STOP_LOSS_DISTANCE = 5        # Points away from entry
TAKE_PROFIT_DISTANCE = 10     # Points away from entry

# RTH: 09:30 - 16:00 ET, Monday to Friday
RTH_START = datetime.time(9, 29)
RTH_END = datetime.time(15, 59)
EASTERN = pytz.timezone('US/Eastern')

# --- Setup Logging ---
logging.basicConfig(
    level=logging.WARNING,  # Set to WARNING to reduce verbosity; change to DEBUG for more details
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# --- Performance Tracking ---
PERFORMANCE_FILE = 'ES/mr_aggregate_performance.json'
EQUITY_CURVE_FILE = 'ES/mr_aggregate_equity_curve.csv'

# Initialize or load aggregate performance
if os.path.exists(PERFORMANCE_FILE):
    try:
        with open(PERFORMANCE_FILE, 'r') as f:
            aggregate_performance = json.load(f)
        # Convert equity_curve timestamps from strings to datetime objects
        if os.path.exists(EQUITY_CURVE_FILE):
            try:
                aggregate_equity_curve = pd.read_csv(
                    EQUITY_CURVE_FILE,
                    parse_dates=['Timestamp'],
                    index_col='Timestamp'
                )
                # Verify that the necessary columns exist
                if not {'Equity'}.issubset(aggregate_equity_curve.columns):
                    logger.warning(f"Equity curve file {EQUITY_CURVE_FILE} is missing required columns. Reinitializing.")
                    aggregate_equity_curve = pd.DataFrame(columns=['Equity'])
            except pd.errors.EmptyDataError:
                logger.warning(f"Equity curve file {EQUITY_CURVE_FILE} is empty. Initializing new equity curve.")
                aggregate_equity_curve = pd.DataFrame(columns=['Equity'])
            except Exception as e:
                logger.error(f"Error loading equity curve file: {e}. Initializing new equity curve.")
                aggregate_equity_curve = pd.DataFrame(columns=['Equity'])
        else:
            aggregate_equity_curve = pd.DataFrame(columns=['Equity'])
        print("Loaded existing aggregate performance data.")
    except json.JSONDecodeError:
        logger.warning(f"Performance file {PERFORMANCE_FILE} is empty or invalid. Initializing new performance data.")
        aggregate_performance = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "equity_curve": []  # List of dicts with 'Timestamp' and 'Equity'
        }
        aggregate_equity_curve = pd.DataFrame(columns=['Equity'])
    except Exception as e:
        logger.error(f"Error loading performance file: {e}")
        sys.exit(1)
else:
    aggregate_performance = {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "total_pnl": 0.0,
        "equity_curve": []  # List of dicts with 'Timestamp' and 'Equity'
    }
    aggregate_equity_curve = pd.DataFrame(columns=['Equity'])
    print("Initialized new aggregate performance data.")

current_entry_price = None
current_position_size = 0

# --- Connect to IBKR ---
ib = IB()
try:
    ib.connect(host=IB_HOST, port=IB_PORT, clientId=CLIENT_ID)
    print("Connected to IBKR.")
except Exception as e:
    logger.error(f"Failed to connect to IBKR: {e}")
    sys.exit(1)

# --- Define Contracts ---
es_contract = Future(symbol=DATA_SYMBOL, lastTradeDateOrContractMonth=DATA_EXPIRY, exchange=DATA_EXCHANGE, currency=CURRENCY)
mes_contract = Future(symbol=EXEC_SYMBOL, lastTradeDateOrContractMonth=EXEC_EXPIRY, exchange=EXEC_EXCHANGE, currency=CURRENCY)

# Qualify Contracts
try:
    qualified_contracts = ib.qualifyContracts(es_contract, mes_contract)
    es_contract = qualified_contracts[0]
    mes_contract = qualified_contracts[1]
    print(f"Qualified ES Contract: {es_contract}")
    print(f"Qualified MES Contract: {mes_contract}")
except Exception as e:
    logger.error(f"Error qualifying contracts: {e}")
    ib.disconnect()
    sys.exit(1)

# --- Request Historical Data for ES (Full Data) ---
try:
    logger.info("Requesting historical ES data (including ETH)...")
    bars_30m_full = ib.reqHistoricalData(
        contract=es_contract,
        endDateTime='',
        durationStr='90 D',
        barSizeSetting='30 mins',
        whatToShow='TRADES',
        useRTH=False,                     
        formatDate=1,
        keepUpToDate=False
    )

    if bars_30m_full:
        df_30m_full = util.df(bars_30m_full)
        df_30m_full.set_index('date', inplace=True)
        df_30m_full.sort_index(inplace=True)
        # Ensure the index is timezone-aware (UTC)
        df_30m_full.index = pd.to_datetime(df_30m_full.index, utc=True)
        print("Successfully retrieved full 30m historical data (including ETH).")
    else:
        logger.warning("No 30m historical data received.")
        df_30m_full = pd.DataFrame()

except Exception as e:
    logger.error(f"Error requesting historical data: {e}")
    ib.disconnect()
    sys.exit(1)

def calculate_bollinger_bands(df, period=15, stddev=2):
    if len(df) < period:
        return df
    df['ma'] = df['close'].rolling(window=period).mean()
    df['std'] = df['close'].rolling(window=period).std()
    df['upper_band'] = df['ma'] + (stddev * df['std'])
    df['lower_band'] = df['ma'] - (stddev * df['std'])
    return df

def filter_rth(df):
    if df.index.tz is None:
        df = df.tz_localize('UTC')
    else:
        df = df.tz_convert('UTC')

    df_eastern = df.copy()
    df_eastern.index = df_eastern.index.tz_convert(EASTERN)

    df_eastern = df_eastern[df_eastern.index.weekday < 5]
    df_rth = df_eastern.between_time(RTH_START, RTH_END)
    df_rth.index = df_rth.index.tz_convert('UTC')
    return df_rth

# Calculate Bollinger Bands on Full Data
df_30m_full = calculate_bollinger_bands(df_30m_full, BOLLINGER_PERIOD, BOLLINGER_STDDEV)

# Filter RTH Data Separately for Trade Execution
df_30m_rth = filter_rth(df_30m_full)

# --- Initialize Backtest Variables ---
cash = INITIAL_CASH + aggregate_performance.get("total_pnl", 0.0)  # Start with initial cash plus cumulative P&L
balance_series = [cash]
position = None
pending_order = False
current_30min_start = None
current_30min_bars = []

def save_performance():
    """Save aggregate performance to a JSON file and equity curve to a CSV."""
    try:
        # Update equity_curve in aggregate_performance
        aggregate_performance['equity_curve'] = aggregate_performance.get('equity_curve', [])
        if not aggregate_equity_curve.empty:
            latest_equity = aggregate_equity_curve['Equity'].iloc[-1]
            timestamp = aggregate_equity_curve.index[-1].isoformat()
            aggregate_performance['equity_curve'].append({"Timestamp": timestamp, "Equity": latest_equity})
        
        # Save to JSON
        with open(PERFORMANCE_FILE, 'w') as f:
            json.dump(aggregate_performance, f, indent=4)
        
        # Save equity curve to CSV
        if not aggregate_equity_curve.empty:
            aggregate_equity_curve.to_csv(EQUITY_CURVE_FILE)
        
        print("Aggregate performance data saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save performance data: {e}")

def on_trade_filled(trade):
    global position, pending_order, aggregate_performance, cash, current_entry_price, current_position_size, aggregate_equity_curve
    try:
        if not trade.fills:
            logger.warning("Trade filled event received but no fills are present.")
            return

        fill = trade.fills[-1]  # Get the latest fill
        print(f"Trade Filled - Order ID {trade.order.orderId}: {trade.order.action} {fill.execution.shares} @ {fill.execution.price}")

        # Debugging: Check the type of trade.filled
        print(f"trade.filled: {trade.filled} (type: {type(trade.filled)})")

        # Correctly assign filled_quantity based on whether 'filled' is a method or a property
        if callable(trade.filled):
            filled_quantity = trade.filled()  # If 'filled' is a method, call it
        else:
            filled_quantity = trade.filled  # If 'filled' is a property, access it directly

        print(f"Filled Quantity: {filled_quantity} (type: {type(filled_quantity)})")

        if filled_quantity > 0:
            action = trade.order.action.upper()
            if trade.order.parentId == 0:
                # This is an entry order
                position_type = 'LONG' if action == 'BUY' else 'SHORT'
                entry_price = fill.execution.price
                print(f"Entered {position_type} position at {entry_price}")
                position = position_type
                current_entry_price = entry_price
                current_position_size = filled_quantity
                pending_order = False
            else:
                # This is an exit order
                exit_price = fill.execution.price
                # Determine exit type based on order type and action
                if trade.order.orderType == 'LIMIT' and action == 'SELL':
                    exit_type = "TAKE PROFIT"
                elif trade.order.orderType == 'STOP':
                    exit_type = "STOP LOSS"
                else:
                    exit_type = "EXIT"
                print(f"Exited position via {exit_type} at {exit_price}")

                # Calculate P&L
                if position == 'LONG':
                    pnl = (exit_price - current_entry_price) * current_position_size
                elif position == 'SHORT':
                    pnl = (current_entry_price - exit_price) * current_position_size
                else:
                    pnl = 0.0  # Should not happen

                print(f"Trade P&L: {pnl}")

                # Update aggregate performance metrics
                aggregate_performance["total_trades"] += 1
                aggregate_performance["total_pnl"] += pnl
                cash += pnl
                balance_series.append(cash)

                # Update equity curve
                timestamp = datetime.datetime.utcnow().isoformat()
                aggregate_performance["equity_curve"].append({"Timestamp": timestamp, "Equity": cash})
                new_entry = pd.DataFrame({'Equity': [cash]}, index=[pd.to_datetime(timestamp)])
                aggregate_equity_curve = pd.concat([aggregate_equity_curve, new_entry])

                if pnl > 0:
                    aggregate_performance["winning_trades"] += 1
                else:
                    aggregate_performance["losing_trades"] += 1

                # Save performance after each trade
                save_performance()

                # Reset position
                position = None
                current_entry_price = None
                current_position_size = 0
                pending_order = False
    except Exception as e:
        logger.error(f"Error in on_trade_filled handler: {e}")

def on_order_status(trade):
    global position, pending_order
    try:
        print(f"Trade Status Update - Order ID {trade.order.orderId}: {trade.orderStatus.status}")
        if trade.orderStatus.status in ('Cancelled', 'Inactive', 'Filled'):
            print(f"Order ID {trade.order.orderId} has been {trade.orderStatus.status.lower()}.")
            pending_order = False
    except Exception as e:
        logger.error(f"Error in on_order_status handler: {e}")

def place_bracket_order(action, current_price):
    global pending_order
    print(f"Placing bracket order: Action={action}, Current Price={current_price}")
    if action.upper() not in ['BUY', 'SELL']:
        logger.error(f"Invalid action: {action}. Must be 'BUY' or 'SELL'.")
        return

    if action.upper() == 'BUY':
        take_profit_price = current_price + TAKE_PROFIT_DISTANCE
        stop_loss_price = current_price - STOP_LOSS_DISTANCE
    else:
        take_profit_price = current_price - TAKE_PROFIT_DISTANCE
        stop_loss_price = current_price + STOP_LOSS_DISTANCE

    try:
        # Create a standard bracket order with the parent as a limit order
        bracket = ib.bracketOrder(
            action=action.upper(),
            quantity=POSITION_SIZE,
            limitPrice=current_price,      # Parent is a limit order at current_price
            takeProfitPrice=take_profit_price,
            stopLossPrice=stop_loss_price
        )

        # Place the parent order and get the Trade object
        parent_trade = ib.placeOrder(mes_contract, bracket[0])
        print(f"Placed Parent {bracket[0].orderType} Order ID {bracket[0].orderId} for {bracket[0].action} at {bracket[0].lmtPrice}")

        # Attach event handlers to the parent Trade object
        parent_trade.filledEvent += on_trade_filled
        parent_trade.statusEvent += on_order_status

        # Place Take-Profit and Stop-Loss Orders and attach event handlers
        take_profit_trade = ib.placeOrder(mes_contract, bracket[1])
        print(f"Placed Take-Profit {bracket[1].orderType} Order ID {bracket[1].orderId} for {bracket[1].action} at {bracket[1].lmtPrice}")

        take_profit_trade.filledEvent += on_trade_filled
        take_profit_trade.statusEvent += on_order_status

        stop_loss_trade = ib.placeOrder(mes_contract, bracket[2])
        print(f"Placed Stop-Loss {bracket[2].orderType} Order ID {bracket[2].orderId} for {bracket[2].action} at {bracket[2].auxPrice}")

        stop_loss_trade.filledEvent += on_trade_filled
        stop_loss_trade.statusEvent += on_order_status

        pending_order = True
        print("Bracket order placed successfully and event handlers attached.")

    except Exception as e:
        logger.error(f"Failed to place bracket order: {e}")
        pending_order = False

def is_rth(timestamp):
    if timestamp is None:
        return False
    ts_eastern = timestamp.astimezone(EASTERN)
    return ts_eastern.weekday() < 5 and RTH_START <= ts_eastern.time() < RTH_END

def execute_trade(action, current_price, current_time):
    global pending_order  # Declare global to modify the variable

    if pending_order:
        print("There is already a pending order. Skipping new trade execution.")
        return

    if not is_rth(current_time):
        print("Current time is not in RTH. Skipping trade entry.")
        return

    print(f"Entry Signal: {action}")
    print(f"Current Price: {current_price}")

    if current_time not in df_30m_full.index:
        logger.warning(f"No data row for {current_time} in df_30m_full. Skipping trade.")
        return

    row = df_30m_full.loc[current_time]
    upper_band = row.get('upper_band', np.nan)
    lower_band = row.get('lower_band', np.nan)

    if pd.isna(upper_band) or pd.isna(lower_band):
        logger.warning(f"No valid Bollinger Bands data (NaN) available for {current_time}. Skipping trade.")
        return

    print(f"Lower Threshold (Bollinger Band): {lower_band}")
    print(f"Upper Threshold (Bollinger Band): {upper_band}")

    place_bracket_order(action, current_price)

def on_realtime_bar(ticker, hasNewBar):
    global current_30min_start, current_30min_bars, df_30m_full, df_30m_rth, position, cash, pending_order, aggregate_equity_curve

    try:
        if hasNewBar:
            if len(ticker) == 0:
                logger.warning("No bars received in RealTimeBarList.")
                return

            bar = ticker[-1]
            bar_time = pd.Timestamp(bar.time).tz_localize('UTC')

            candle_start_time = bar_time.floor('30min')

            # If we detect a new 30-minute candle
            if current_30min_start != candle_start_time:
                # Finalize the previous candle when a new one starts
                if current_30min_start is not None and current_30min_bars:
                    # Aggregate the 30-minute candle data
                    open_30 = current_30min_bars[0]['open']
                    high_30 = max(b['high'] for b in current_30min_bars)
                    low_30 = min(b['low'] for b in current_30min_bars)
                    close_30 = current_30min_bars[-1]['close']
                    volume_30 = sum(b['volume'] for b in current_30min_bars)

                    # Update final values for the just-closed candle
                    df_30m_full.loc[current_30min_start, ['open', 'high', 'low', 'close', 'volume']] = [open_30, high_30, low_30, close_30, volume_30]

                    # Recalculate Bollinger Bands only if we have enough data
                    if len(df_30m_full) >= BOLLINGER_PERIOD:
                        df_30m_full = calculate_bollinger_bands(df_30m_full, BOLLINGER_PERIOD, BOLLINGER_STDDEV)

                    # Re-filter RTH data
                    df_30m_rth = filter_rth(df_30m_full)

                    if is_rth(current_30min_start):
                        current_price = close_30
                        current_time = current_30min_start

                        if current_time in df_30m_full.index:
                            row = df_30m_full.loc[current_time]
                            upper_band = row.get('upper_band', np.nan)
                            lower_band = row.get('lower_band', np.nan)
                        else:
                            upper_band = np.nan
                            lower_band = np.nan

                        if not pd.isna(upper_band) and not pd.isna(lower_band):
                            print(f"\nBollinger Bands - Upper: {upper_band}, Lower: {lower_band}")
                            print(f"Current Price: {current_price}")

                            print(f"Evaluating trade signals: Position={position}, Pending Order={pending_order}")

                            if position is None and not pending_order:
                                if current_price < lower_band:
                                    # Enter Long during RTH
                                    execute_trade('BUY', current_price, current_time)
                                elif current_price > upper_band:
                                    # Enter Short during RTH
                                    execute_trade('SELL', current_price, current_time)
                                else:
                                    print("No trading signal detected.")
                        else:
                            logger.warning(f"No Bollinger Bands data available for {current_time}. Skipping trade.")
                    else:
                        print(f"New 30-min bar at {current_30min_start} UTC is outside RTH. No trade executed.")

                # Start a new 30-minute candle
                current_30min_start = candle_start_time
                current_30min_bars = []

            # Append the current 5-second bar data to the ongoing 30-min candle
            current_30min_bars.append({
                'open': bar.open_,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            })

            # Continuously update the current candle row in df_30m_full
            # This ensures that if we lose connection or something, we always have the latest partial data.
            if current_30min_start is not None:
                open_30 = current_30min_bars[0]['open']
                high_30 = max(b['high'] for b in current_30min_bars)
                low_30 = min(b['low'] for b in current_30min_bars)
                close_30 = current_30min_bars[-1]['close']
                volume_30 = sum(b['volume'] for b in current_30min_bars)

                # Update or create the row for the current candle in progress
                df_30m_full.loc[current_30min_start, ['open', 'high', 'low', 'close', 'volume']] = [open_30, high_30, low_30, close_30, volume_30]

                logger.debug(f"Updated current candle: {current_30min_start} - O:{open_30}, H:{high_30}, L:{low_30}, C:{close_30}, V:{volume_30}")

    except Exception as e:
        logger.error(f"Error in on_realtime_bar handler: {e}")

try:
    print("Requesting real-time 5-second bars (including ETH)...")
    ticker = ib.reqRealTimeBars(
        contract=es_contract,
        barSize=5,
        whatToShow='TRADES',
        useRTH=False,
        realTimeBarsOptions=[]
    )
    ticker.updateEvent += on_realtime_bar
    print("Real-time bar handler assigned.")
except Exception as e:
    logger.error(f"Failed to subscribe to real-time bars: {e}")
    ib.disconnect()
    sys.exit(1)

print("Starting mean reversion bot...")
try:
    ib.run()
except KeyboardInterrupt:
    print("Interrupt received, shutting down...")
finally:
    # Print Aggregate Performance Summary
    print("\n--- Aggregate Performance Summary ---")
    total_trades = aggregate_performance.get("total_trades", 0)
    winning_trades = aggregate_performance.get("winning_trades", 0)
    losing_trades = aggregate_performance.get("losing_trades", 0)
    total_pnl = aggregate_performance.get("total_pnl", 0.0)
    equity_curve_data = aggregate_performance.get("equity_curve", [])

    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Final Cash Balance: ${cash:.2f}")
    print("----------------------------\n")
    
    # Optionally, create/update the aggregate equity curve DataFrame and save it
    if not aggregate_equity_curve.empty:
        aggregate_equity_curve.to_csv(EQUITY_CURVE_FILE)
        print("Aggregate equity curve saved to 'aggregate_equity_curve.csv'.")
    else:
        print("No equity data to save.")
    
    # Save the aggregate performance data one last time
    save_performance()

    ib.disconnect()
    print("Disconnected from IBKR.")