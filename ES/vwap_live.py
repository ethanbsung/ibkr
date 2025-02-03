import pandas as pd
import numpy as np
from ib_insync import *
import logging
import sys
import pytz
from datetime import datetime, time, timedelta, timezone
from threading import Lock
import json
import os

# --- Configuration Parameters ---
IB_HOST = '127.0.0.1'        # IBKR Gateway/TWS host
IB_PORT = 7497               # IBKR Gateway/TWS paper trading port
CLIENT_ID = 2                # Unique client ID

DATA_SYMBOL = 'ES'           # E-mini S&P 500 for data
DATA_EXPIRY = '202503'       # March 2025 (example)
DATA_EXCHANGE = 'CME'        # Exchange for ES
CURRENCY = 'USD'

EXEC_SYMBOL = 'MES'          # Micro E-mini S&P 500 for execution
EXEC_EXPIRY = '202503'       # March 2025 (example)
EXEC_EXCHANGE = 'CME'        # Exchange for MES

INITIAL_CASH = 5000          # Starting cash in USD
POSITION_SIZE = 1            # Number of MES contracts per trade
CONTRACT_MULTIPLIER = 5      # Contract multiplier for MES

# RSI thresholds
RSI_PERIOD_15M = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Risk Management
STOP_LOSS_POINTS = 9
TAKE_PROFIT_POINTS = 10

# Regular Trading Hours (RTH) for logic gating
RTH_START = datetime.strptime("09:30", "%H:%M").time()
RTH_END   = datetime.strptime("15:59", "%H:%M").time()
EASTERN   = pytz.timezone('US/Eastern')

# --- Performance Tracking Files ---
PERFORMANCE_FILE = 'ES/vwap_performance_data.json'      # JSON file to store aggregate performance
EQUITY_CURVE_FILE = 'ES/vwap_equity_curve.csv'           # CSV file to store equity curve

# --- Setup Logging ---
logging.basicConfig(
    level=logging.WARNING,  # Set to INFO or DEBUG for more verbosity
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# -----------------------------------------------------------------------------
# Helper functions for RSI & VWAP
# -----------------------------------------------------------------------------

def rsi(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) on the given DataFrame (expects 'close').
    """
    delta = ohlc['close'].diff()

    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    _gain = up.ewm(com=(period - 1), min_periods=period).mean()
    _loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()

    RS = _gain / _loss
    return pd.Series(100 - (100 / (1 + RS)), name="RSI")

def vwap(ohlc: pd.DataFrame) -> pd.Series:
    """
    Calculate the Volume Weighted Average Price (VWAP) since the *latest* market open
    (defaulted to 6:00 PM Eastern) within the same session.
    """
    df = ohlc.copy()

    # Convert index to US/Eastern
    df['date_eastern'] = df.index.tz_convert('US/Eastern')

    # For each row, define the "session open" as 18:00 ET of the same day if the bar time is >= 18:00,
    # otherwise use the previous day's 18:00.
    def session_open(dt_eastern):
        if dt_eastern.time() >= time(18, 0):
            return dt_eastern.replace(hour=18, minute=0, second=0, microsecond=0)
        else:
            return (dt_eastern - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)

    df['session_open'] = df['date_eastern'].apply(session_open)
    df = df[df['date_eastern'] >= df['session_open']]

    typical_price = (df['high'] + df['low'] + df['close']) / 3
    csum_tpvol = (typical_price * df['volume']).cumsum()
    csum_vol   = df['volume'].cumsum()

    return pd.Series(csum_tpvol / csum_vol, name="VWAP")


# -----------------------------------------------------------------------------
# Live Trading Strategy Class with Performance Tracking
# -----------------------------------------------------------------------------
class MESFuturesLiveStrategy:
    def __init__(
        self, 
        ib,
        es_contract,
        mes_contract,
        initial_cash=INITIAL_CASH,
        position_size=POSITION_SIZE,
        contract_multiplier=CONTRACT_MULTIPLIER,
        stop_loss_points=STOP_LOSS_POINTS,
        take_profit_points=TAKE_PROFIT_POINTS,
        rsi_period_15m=RSI_PERIOD_15M,
        rsi_overbought=RSI_OVERBOUGHT,
        rsi_oversold=RSI_OVERSOLD,
    ):
        """
        Initializes the live trading strategy with performance tracking.
        """
        self.ib = ib
        self.ib.autoReconnect = False  # We'll handle reconnection manually

        self.es_contract = es_contract
        self.mes_contract = mes_contract
        self.initial_cash = initial_cash
        self.position_size = position_size
        self.contract_multiplier = contract_multiplier
        self.stop_loss_points = stop_loss_points
        self.take_profit_points = take_profit_points
        self.rsi_period_15m = rsi_period_15m
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

        # Strategy state
        self.cash = initial_cash
        self.equity = initial_cash
        self.position = None  # None, 'LONG', or 'SHORT'
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.pending_order = False

        # For tracking equity curve and trade log
        self.equity_curve = []
        self.trade_log = []

        # Lock for thread safety if desired
        self.lock = Lock()

        # We will store all 5-second bars in this list (and convert to DF on each update)
        self.realtime_5s_data = []

        # Initialize last_log_time to current UTC time (timezone-aware)
        self.last_log_time = datetime.now(timezone.utc)

        self.latest_5s_close = None

        # Performance Tracking Initialization
        self.performance_file = PERFORMANCE_FILE
        self.equity_curve_file = EQUITY_CURVE_FILE
        self.load_performance()

        # Bind the events to handlers (note the on_disconnect now accepts extra args)
        self.ib.disconnectedEvent += self.on_disconnect
        self.ib.connectedEvent += self.on_reconnected

    def load_performance(self):
        """Load existing performance data from files or initialize new."""
        if os.path.exists(self.performance_file):
            try:
                with open(self.performance_file, 'r') as f:
                    self.aggregate_performance = json.load(f)
                if 'equity_curve' not in self.aggregate_performance:
                    self.aggregate_performance['equity_curve'] = []
                logger.warning("Loaded existing aggregate performance data.")
            except json.JSONDecodeError:
                logger.warning(f"Performance file {self.performance_file} is empty or invalid. Initializing new performance data.")
                self.initialize_performance()
            except Exception as e:
                logger.error(f"Error loading performance file: {e}. Initializing new performance data.")
                self.initialize_performance()
        else:
            self.initialize_performance()

        if os.path.exists(self.equity_curve_file):
            try:
                self.aggregate_equity_curve = pd.read_csv(
                    self.equity_curve_file,
                    parse_dates=['Timestamp'],
                    index_col='Timestamp'
                )
                if not {'Equity'}.issubset(self.aggregate_equity_curve.columns):
                    logger.warning(f"Equity curve file {self.equity_curve_file} is missing required columns. Reinitializing.")
                    self.aggregate_equity_curve = pd.DataFrame(columns=['Equity'])
            except pd.errors.EmptyDataError:
                logger.warning(f"Equity curve file {self.equity_curve_file} is empty. Initializing new equity curve.")
                self.aggregate_equity_curve = pd.DataFrame(columns=['Equity'])
            except Exception as e:
                logger.error(f"Error loading equity curve file: {e}. Initializing new equity curve.")
                self.aggregate_equity_curve = pd.DataFrame(columns=['Equity'])
        else:
            self.aggregate_equity_curve = pd.DataFrame(columns=['Equity'])

    def initialize_performance(self):
        """Initialize performance metrics."""
        self.aggregate_performance = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "equity_curve": []
        }
        logger.warning("Initialized new aggregate performance data.")
        self.aggregate_equity_curve = pd.DataFrame(columns=['Equity'])

    def save_performance(self):
        """Save aggregate performance to a JSON file and equity curve to a CSV."""
        try:
            if not self.aggregate_equity_curve.empty:
                latest_equity = self.aggregate_equity_curve['Equity'].iloc[-1]
                timestamp = self.aggregate_equity_curve.index[-1].isoformat()
                self.aggregate_performance['equity_curve'].append({"Timestamp": timestamp, "Equity": latest_equity})

            with open(self.performance_file, 'w') as f:
                json.dump(self.aggregate_performance, f, indent=4)

            if not self.aggregate_equity_curve.empty:
                self.aggregate_equity_curve.to_csv(self.equity_curve_file)

            logger.warning("Aggregate performance data saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")

    def fetch_historical_data(self, duration='3 D', bar_size='15 mins'):
        """
        (Optional) Fetch historical data to initialize indicators if desired.
        """
        logger.warning("Fetching historical ES data for indicator initialization...")
        try:
            bars_15m = self.ib.reqHistoricalData(
                contract=self.es_contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1,
                keepUpToDate=False
            )
            if not bars_15m:
                logger.warning("No 15-minute historical data fetched.")
                return

            df = util.df(bars_15m)
            df.set_index('date', inplace=True)
            if df.index.tz is None:
                df.index = pd.to_datetime(df.index).tz_localize('UTC')
            else:
                df.index = pd.to_datetime(df.index).tz_convert('UTC')

            logger.warning(f"Fetched {len(df)} historical 15-minute ES bars for warm-up.")
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")

    def subscribe_to_realtime_bars(self):
        """
        Subscribes to real-time 5-second bars for ES.
        """
        logger.warning("Requesting real-time 5-second bars for ES...")
        try:
            # Unsubscribe previous subscription if exists
            if hasattr(self, 'ticker_5s') and self.ticker_5s is not None:
                self.ticker_5s.updateEvent -= self.on_bar_update
                self.ib.cancelHistoricalData(self.ticker_5s)

            self.ticker_5s = self.ib.reqHistoricalData(
                contract=self.es_contract,
                endDateTime='',
                durationStr='1 D',
                barSizeSetting='5 secs',
                whatToShow='TRADES',
                useRTH=False,
                keepUpToDate=True,
                formatDate=1
            )
            self.ticker_5s.updateEvent += self.on_bar_update
            logger.warning("Real-time bar subscription set up.")
        except Exception as e:
            logger.error(f"Failed to subscribe to real-time bars: {e}")

    # -------------------------------------------------------------------------
    # Callback to handle disconnections. Accept extra args to avoid errors.
    def on_disconnect(self, *args):
        logger.warning("Disconnected from IBKR. Will try to reconnect in 5 seconds...")
        self.ib.schedule(5, self.try_reconnect)

    def try_reconnect(self):
        """
        Attempts to reconnect to IBKR, re-qualify contracts, and re-subscribe to real-time bars.
        """
        if not self.ib.isConnected():
            logger.warning("Attempting to reconnect to IBKR...")
            try:
                self.ib.connect(host=IB_HOST, port=IB_PORT, clientId=CLIENT_ID)
                logger.warning("Reconnected to IBKR.")

                # Re-qualify contracts
                qualified_contracts = self.ib.qualifyContracts(self.es_contract, self.mes_contract)
                if not qualified_contracts:
                    raise ValueError("Failed to qualify contracts after reconnection.")
                self.es_contract, self.mes_contract = qualified_contracts
                logger.warning(f"Re-qualified ES Contract: {self.es_contract}")
                logger.warning(f"Re-qualified MES Contract: {self.mes_contract}")

                # Optionally clear the old data buffer if needed
                self.realtime_5s_data = []

                # Resubscribe to real-time bars
                self.subscribe_to_realtime_bars()

                logger.warning("Reconnection and resubscription successful.")
            except Exception as e:
                logger.error(f"Reconnection attempt failed: {e}. Scheduling another attempt in 5 seconds.")
                self.ib.schedule(5, self.try_reconnect)

    def on_reconnected(self):
        logger.warning("Successfully reconnected to IBKR.")

    def on_bar_update(self, bars: RealTimeBarList, hasNewBar: bool):
        with self.lock:
            if not hasNewBar or len(bars) == 0:
                return

            for bar in bars:
                # Ensure bar.date is timezone-aware (UTC)
                bar_time = bar.date
                if bar_time.tzinfo is None:
                    bar_time = bar_time.replace(tzinfo=timezone.utc)
                else:
                    bar_time = bar_time.astimezone(timezone.utc)

                self.latest_5s_close = bar.close

                self.realtime_5s_data.append({
                    'date': bar_time,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })

            # Convert to DataFrame and process for 15-minute bars
            df_5s = pd.DataFrame(self.realtime_5s_data)
            df_5s.set_index('date', inplace=True)
            df_5s.sort_index(inplace=True)

            ohlc_15m = df_5s.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            if len(ohlc_15m) < 1:
                return

            ohlc_15m['RSI'] = rsi(ohlc_15m, period=self.rsi_period_15m)
            ohlc_15m['VWAP'] = vwap(ohlc_15m)

            latest_bar_time = ohlc_15m.index[-1]
            latest_rsi      = ohlc_15m['RSI'].iloc[-1]
            latest_vwap     = ohlc_15m['VWAP'].iloc[-1]

            if self.latest_5s_close is not None:
                current_price = self.latest_5s_close
                self.apply_trading_logic(current_price, latest_vwap, latest_rsi, latest_bar_time)
            else:
                logger.debug("Latest 5-second close price not available yet.")

            self.equity_curve.append({'Time': latest_bar_time.isoformat(), 'Equity': self.equity})

            current_time = datetime.now(timezone.utc)
            if current_time >= self.last_log_time + timedelta(minutes=5):
                logger.warning(f"Periodic Update: Price: {current_price}, RSI={latest_rsi:.2f}, VWAP={latest_vwap:.2f}")
                self.last_log_time = current_time

    def apply_trading_logic(self, current_price, current_vwap, current_rsi, current_time):
        """
        Logs the current price, RSI, and VWAP values.
        Only places orders if the current time (in Eastern Time) is within RTH.
        Also, no new order will be placed if there is an active position or any open MES orders.
        """
        # Check if there are any open orders for MES
        if self.ib.openOrders(self.mes_contract):
            logger.warning("Existing open orders detected for MES. Not placing a new trade.")
            return

        # Check if a trade is already open
        if self.position is not None:
            logger.warning("A trade is already open. Not entering a new trade.")
            return

        # Check if there is a pending order already
        if self.pending_order:
            logger.debug("An order is already pending. Skipping new entry.")
            return

        # Convert current_time to Eastern Time for display and checking
        current_time_et = current_time.astimezone(EASTERN).time()

        # Only log the indicator values if needed
        logger.debug(
            f"Current Time (ET): {current_time_et}, Price: {current_price:.2f}, "
            f"RSI: {current_rsi:.2f}, VWAP: {current_vwap:.2f}"
        )

        # Check if we are in RTH before attempting to enter a trade
        if not (RTH_START <= current_time_et <= RTH_END):
            logger.warning(f"Time {current_time_et} is outside RTH. No trading action will be taken.")
            return

        # Proceed with trading logic only if within RTH
        logger.debug(f"Checking signals: Price={current_price:.2f}, VWAP={current_vwap:.2f}, RSI={current_rsi:.2f}")

        # Long Entry Condition
        if (current_price > current_vwap) and (current_rsi > self.rsi_overbought):
            logger.warning(f"Signal: Enter LONG at price {current_price:.2f}, RSI: {current_rsi:.2f}, VWAP: {current_vwap:.2f}")
            self.place_bracket_order('BUY', current_price, current_time)
        # Short Entry Condition
        elif (current_price < current_vwap) and (current_rsi < self.rsi_oversold):
            logger.warning(f"Signal: Enter SHORT at price {current_price:.2f}, RSI: {current_rsi:.2f}, VWAP: {current_vwap:.2f}")
            self.place_bracket_order('SELL', current_price, current_time)
        else:
            logger.debug("No entry signal detected.")

    def place_bracket_order(self, action, current_price, current_time):
        """
        Places a bracket (parent + take-profit + stop-loss) order using ib_insync's bracketOrder method.
        """
        try:
            action = action.upper()
            if action == 'BUY':
                take_profit_price = current_price + self.take_profit_points
                stop_loss_price   = current_price - self.stop_loss_points
                order_action      = 'BUY'
            elif action == 'SELL':
                take_profit_price = current_price - self.take_profit_points
                stop_loss_price   = current_price + self.stop_loss_points
                order_action      = 'SELL'
            else:
                logger.error(f"Invalid action: {action}. Must be 'BUY' or 'SELL'.")
                return

            logger.debug(
                f"Placing bracket: Action {order_action} x{self.position_size}, "
                f"TP={take_profit_price:.2f}, SL={stop_loss_price:.2f}"
            )

            trades = self.ib.bracketOrder(
                action=order_action,
                quantity=self.position_size,
                limitPrice=current_price,
                takeProfitPrice=take_profit_price,
                stopLossPrice=stop_loss_price
            )

            parent_trade = self.ib.placeOrder(self.mes_contract, trades[0])
            logger.warning(f"Placed Parent {trades[0].orderType} Order ID {trades[0].orderId} for {trades[0].action} at {trades[0].lmtPrice}")
            parent_trade.filledEvent += self.on_trade_filled
            parent_trade.statusEvent += self.on_order_status

            take_profit_trade = self.ib.placeOrder(self.mes_contract, trades[1])
            logger.warning(f"Placed Take-Profit {trades[1].orderType} Order ID {trades[1].orderId} for {trades[1].action} at {trades[1].lmtPrice}")
            take_profit_trade.filledEvent += self.on_trade_filled
            take_profit_trade.statusEvent += self.on_order_status

            stop_loss_trade = self.ib.placeOrder(self.mes_contract, trades[2])
            logger.warning(f"Placed Stop-Loss {trades[2].orderType} Order ID {trades[2].orderId} for {trades[2].action} at {trades[2].auxPrice}")
            stop_loss_trade.filledEvent += self.on_trade_filled
            stop_loss_trade.statusEvent += self.on_order_status

            self.pending_order = True
            logger.warning("Bracket order placed successfully and event handlers attached.")

        except Exception as e:
            logger.error(f"Failed to place bracket order: {e}")
            self.pending_order = False

    def on_trade_filled(self, trade):
        """
        Callback for trade fill events.
        """
        try:
            fill = trade.fills[-1]
            fill_price = fill.execution.price
            fill_qty   = fill.execution.shares
            fill_time  = fill.execution.time
            order_action = trade.order.action.upper()

            logger.debug(f"Trade Filled: {order_action} {fill_qty} @ {fill_price} on {fill_time}")

            # Parent order fill -> position entry
            if trade.order.orderType == 'LMT' and not trade.order.parentId:
                if order_action == 'BUY' and self.position is None:
                    self.position = 'LONG'
                    self.entry_price = fill_price
                    self.stop_loss   = self.entry_price - self.stop_loss_points
                    self.take_profit = self.entry_price + self.take_profit_points
                    self.trade_log.append({
                        'Type': 'LONG',
                        'Entry Time': fill_time.isoformat(),
                        'Entry Price': self.entry_price,
                        'Exit Time': None,
                        'Exit Price': None,
                        'Result': None,
                        'Profit': 0
                    })
                    logger.warning(f"Entered LONG @ {self.entry_price}")
                elif order_action == 'SELL' and self.position is None:
                    self.position = 'SHORT'
                    self.entry_price = fill_price
                    self.stop_loss   = self.entry_price + self.stop_loss_points
                    self.take_profit = self.entry_price - self.take_profit_points
                    self.trade_log.append({
                        'Type': 'SHORT',
                        'Entry Time': fill_time.isoformat(),
                        'Entry Price': self.entry_price,
                        'Exit Time': None,
                        'Exit Price': None,
                        'Result': None,
                        'Profit': 0
                    })
                    logger.warning(f"Entered SHORT @ {self.entry_price}")

            # Child order fill -> position exit
            elif trade.order.orderType in ['TAKE_PROFIT_LIMIT', 'STOP']:
                if self.position == 'LONG' and order_action == 'SELL':
                    pnl = (fill_price - self.entry_price) * self.position_size * self.contract_multiplier - 1.24
                    self.cash += pnl
                    self.equity += pnl
                    result = 'Take Profit' if fill_price >= self.take_profit else 'Stop Loss'
                    self.trade_log[-1].update({
                        'Exit Time': fill_time.isoformat(),
                        'Exit Price': fill_price,
                        'Result': result,
                        'Profit': pnl
                    })
                    logger.warning(f"Exited LONG @ {fill_price} PnL=${pnl:.2f} ({result})")
                    self.position = None

                elif self.position == 'SHORT' and order_action == 'BUY':
                    pnl = (self.entry_price - fill_price) * self.position_size * self.contract_multiplier - 1.24
                    self.cash += pnl
                    self.equity += pnl
                    result = 'Take Profit' if fill_price <= self.take_profit else 'Stop Loss'
                    self.trade_log[-1].update({
                        'Exit Time': fill_time.isoformat(),
                        'Exit Price': fill_price,
                        'Result': result,
                        'Profit': pnl
                    })
                    logger.warning(f"Exited SHORT @ {fill_price} PnL=${pnl:.2f} ({result})")
                    self.position = None

            # If position is closed, clear the pending order flag
            if self.position is None:
                self.pending_order = False

            self.equity_curve.append({'Time': fill_time.isoformat(), 'Equity': self.equity})

            # Update aggregate performance if a child order was filled
            if trade.order.orderType in ['TAKE_PROFIT_LIMIT', 'STOP']:
                self.aggregate_performance["total_trades"] += 1
                self.aggregate_performance["total_pnl"] += pnl
                if pnl > 0:
                    self.aggregate_performance["winning_trades"] += 1
                else:
                    self.aggregate_performance["losing_trades"] += 1

                new_entry = pd.DataFrame({'Equity': [self.equity]}, index=[pd.to_datetime(fill_time.isoformat())])
                if not new_entry.empty:
                    self.aggregate_equity_curve = pd.concat([self.aggregate_equity_curve, new_entry])

                self.save_performance()

        except Exception as e:
            logger.error(f"Error in on_trade_filled handler: {e}")

    def on_order_status(self, trade):
        """
        Callback when an order status changes.
        """
        try:
            logger.debug(f"Order Status: ID={trade.order.orderId}, Status={trade.orderStatus.status}")
            if trade.orderStatus.status in ['Cancelled', 'Inactive', 'Filled']:
                if trade.order.orderType == 'LMT' and not trade.order.parentId and self.position is None:
                    logger.warning(f"Parent order {trade.order.orderId} cancelled or inactive, no position entered.")
                if self.position is None:
                    self.pending_order = False
        except Exception as e:
            logger.error(f"Error in on_order_status handler: {e}")

    def run(self):
        """
        Runs the strategy:
          1) Optionally fetch historical data for warmup.
          2) Subscribe to 5-second real-time bars.
          3) Start the IB event loop in a resilient loop that attempts reconnection if needed.
        """
        self.fetch_historical_data(duration='3 D', bar_size='15 mins')
        self.subscribe_to_realtime_bars()

        # Log initial RSI and VWAP after waiting briefly for data
        import time as time_module
        time_module.sleep(5)

        if self.realtime_5s_data:
            df_initial = pd.DataFrame(self.realtime_5s_data)
            df_initial.set_index('date', inplace=True)
            df_initial.sort_index(inplace=True)

            ohlc_initial = df_initial.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            if not ohlc_initial.empty:
                ohlc_initial['RSI'] = rsi(ohlc_initial, period=self.rsi_period_15m)
                ohlc_initial['VWAP'] = vwap(ohlc_initial)

                latest_rsi_initial = ohlc_initial['RSI'].iloc[-1]
                latest_vwap_initial = ohlc_initial['VWAP'].iloc[-1]

                logger.warning(f"Initial Indicators: RSI={latest_rsi_initial:.2f}, VWAP={latest_vwap_initial:.2f}")

        logger.warning("Starting IB event loop. Press Ctrl+C to exit.")
        try:
            # Keep the event loop running and attempt reconnection if disconnected
            while True:
                self.ib.run()
                if not self.ib.isConnected():
                    logger.warning("IB is disconnected. Attempting to reconnect...")
                    self.try_reconnect()
                    time_module.sleep(5)
                else:
                    break
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received, shutting down...")
        except Exception as e:
            logger.error(f"Exception in IB event loop: {e}")
        finally:
            logger.debug(f"Final Equity: ${self.equity:.2f}")
            if self.trade_log:
                trade_df = pd.DataFrame(self.trade_log)
                logger.debug(f"Trade Log:\n{trade_df}")
            else:
                logger.debug("No trades were executed.")
            try:
                self.ib.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
            logger.warning("Disconnected from IBKR.")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ib = IB()
    try:
        ib.connect(host=IB_HOST, port=IB_PORT, clientId=CLIENT_ID)
        logger.warning("Connected to IBKR.")
    except Exception as e:
        logger.error(f"Failed to connect to IBKR: {e}")
        sys.exit(1)

    # Define Contracts
    es_contract = Future(
        symbol=DATA_SYMBOL, 
        lastTradeDateOrContractMonth=DATA_EXPIRY,
        exchange=DATA_EXCHANGE, 
        currency=CURRENCY
    )
    mes_contract = Future(
        symbol=EXEC_SYMBOL, 
        lastTradeDateOrContractMonth=EXEC_EXPIRY,
        exchange=EXEC_EXCHANGE, 
        currency=CURRENCY
    )

    # Qualify the contracts
    try:
        qualified_contracts = ib.qualifyContracts(es_contract, mes_contract)
        if not qualified_contracts or len(qualified_contracts) < 2:
            raise ValueError("Failed to qualify contracts.")
        es_contract, mes_contract = qualified_contracts
        logger.warning(f"Qualified ES Contract: {es_contract}")
        logger.warning(f"Qualified MES Contract: {mes_contract}")
    except Exception as e:
        logger.error(f"Error qualifying contracts: {e}")
        ib.disconnect()
        sys.exit(1)

    # Initialize and run the strategy
    strategy = MESFuturesLiveStrategy(
        ib=ib,
        es_contract=es_contract,
        mes_contract=mes_contract,
        initial_cash=INITIAL_CASH,
        position_size=POSITION_SIZE,
        contract_multiplier=CONTRACT_MULTIPLIER,
        stop_loss_points=STOP_LOSS_POINTS,
        take_profit_points=TAKE_PROFIT_POINTS,
        rsi_period_15m=RSI_PERIOD_15M,
        rsi_overbought=RSI_OVERBOUGHT,
        rsi_oversold=RSI_OVERSOLD
    )

    strategy.run()