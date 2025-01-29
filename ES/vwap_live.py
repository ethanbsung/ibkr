import pandas as pd
import numpy as np
from ib_insync import *
import logging
import sys
import pytz
from datetime import datetime, time, timedelta, timezone
from threading import Lock

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

# --- Setup Logging ---
logging.basicConfig(
    level=logging.WARNING,  # Set to INFO or DEBUG for more verbosity
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# -----------------------------------------------------------------------------
# Helper functions for RSI & VWAP (from your snippet)
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
    # Copy so as not to mutate original
    df = ohlc.copy()

    # Convert index to US/Eastern
    df['date_eastern'] = df.index.tz_convert('US/Eastern')

    # Define "market open" as 18:00 (6 PM ET) on the same or previous day
    # For each row, find the most recent 18:00:00
    def session_open(dt_eastern):
        # If the bar time is >= 18:00 that day, session open is that day 18:00
        # else session open is the *previous* day at 18:00
        if dt_eastern.time() >= time(18,0):
            return dt_eastern.replace(hour=18, minute=0, second=0, microsecond=0)
        else:
            # Subtract one day, set to 18:00
            return (dt_eastern - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)

    df['session_open'] = df['date_eastern'].apply(session_open)

    # Filter rows only from the current session's open
    df = df[df['date_eastern'] >= df['session_open']]

    # Typical Price
    typical_price = (df['high'] + df['low'] + df['close']) / 3

    # Cumulative TP * Volume / Cumulative Volume
    csum_tpvol = (typical_price * df['volume']).cumsum()
    csum_vol   = df['volume'].cumsum()

    return pd.Series(csum_tpvol / csum_vol, name="VWAP")


# -----------------------------------------------------------------------------
# Live Trading Strategy Class
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
        Initializes the live trading strategy.
        """
        self.ib = ib
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

    def fetch_historical_data(self, duration='3 D', bar_size='15 mins'):
        """
        (Optional) Fetch some historical data to initialize indicators if desired.
        Adjust the logic if you want a warm-up for RSI, etc.
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
            # Optionally do a warm-up RSI / VWAP if needed
            # ...
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")

    def on_bar_update(self, bars: RealTimeBarList, hasNewBar: bool):
        """
        Callback that receives new 5-second bars. 
        We accumulate them, then resample to 15-min bars, recalc RSI/VWAP, and apply logic.
        """
        with self.lock:
            if not hasNewBar or len(bars) == 0:
                return

            # Accumulate the new bars in an in-memory list
            for bar in bars:
                # Convert bar.date to UTC (ib_insync BarData has 'date' attribute with timezone)
                bar_time = bar.date  # Changed from bar.time to bar.date
                if bar_time.tzinfo is None:
                    bar_time = bar_time.replace(tzinfo=timezone.utc)
                else:
                    bar_time = bar_time.astimezone(timezone.utc)
                # Store fields
                self.realtime_5s_data.append({
                    'date': bar_time,
                    'open': bar.open,       # Fixed attribute access
                    'high': bar.high,       # Fixed attribute access
                    'low': bar.low,         # Fixed attribute access
                    'close': bar.close,     # Fixed attribute access
                    'volume': bar.volume
                })

            # Convert to DataFrame
            df_5s = pd.DataFrame(self.realtime_5s_data)
            df_5s.set_index('date', inplace=True)
            df_5s.sort_index(inplace=True)

            # Resample to 15-minute bars
            ohlc_15m = df_5s.resample('15T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            if len(ohlc_15m) < 1:
                return

            # Calculate RSI on 15-minute data
            ohlc_15m['RSI'] = rsi(ohlc_15m, period=self.rsi_period_15m)

            # Calculate VWAP since the "most recent" session open (6 PM ET)
            ohlc_15m['VWAP'] = vwap(ohlc_15m)

            # Get the latest row for signals
            latest_bar_time = ohlc_15m.index[-1]
            latest_close    = ohlc_15m['close'].iloc[-1]
            latest_rsi      = ohlc_15m['RSI'].iloc[-1]
            latest_vwap     = ohlc_15m['VWAP'].iloc[-1]


            # Apply trading logic (on each new 15-min bar — but triggered by 5s data arrival)
            self.apply_trading_logic(latest_close, latest_vwap, latest_rsi, latest_bar_time)

            # Optionally, you can store the equity curve with the *latest close*
            # (includes unrealized PnL, if you want to track that).
            self.equity_curve.append({'Time': latest_bar_time, 'Equity': self.equity})

            # --- New Code to Log RSI and VWAP Every 5 Minutes ---
            current_time = datetime.now(timezone.utc)
            if current_time >= self.last_log_time + timedelta(minutes=5):
                logger.warning(f"Periodic Update: RSI={latest_rsi:.2f}, VWAP={latest_vwap:.2f}")
                self.last_log_time = current_time
            # --- End of New Code ---

    def apply_trading_logic(self, current_price, current_vwap, current_rsi, current_time):
        """
        Decide whether to place a new trade or not, based on RSI & VWAP.
        """
        # Convert current_time to ET to check if within RTH
        current_time_et = current_time.astimezone(EASTERN).time()
        if not (RTH_START <= current_time_et <= RTH_END):
            logger.warning(f"Time {current_time_et} outside RTH. No trading action.")
            return

        logger.warning(f"Checking signals @ {current_time}: Price={current_price:.2f}, VWAP={current_vwap:.2f}, RSI={current_rsi:.2f}")

        # If no position and no pending order, consider new entries
        if self.position is None and not self.pending_order:
            # Long Entry Condition
            if (current_price > current_vwap) and (current_rsi > self.rsi_overbought):
                logger.warning("Signal: Enter LONG")
                self.place_bracket_order('BUY', current_price, current_time)
            # Short Entry Condition
            elif (current_price < current_vwap) and (current_rsi < self.rsi_oversold):
                logger.warning("Signal: Enter SHORT")
                self.place_bracket_order('SELL', current_price, current_time)
            else:
                logger.debug("No entry signal detected.")
        else:
            logger.warning(f"Position={self.position}, PendingOrder={self.pending_order}. No new entry.")

    def place_bracket_order(self, action, current_price, current_time):
        """
        Places a bracket (parent + take-profit + stop-loss) order using ib_insync's bracketOrder method correctly.
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

            logger.warning(
                f"Placing bracket: Action {order_action} x{self.position_size}, "
                f"TP={take_profit_price:.2f}, SL={stop_loss_price:.2f}"
            )

            # Correct usage of bracketOrder
            trades = self.ib.bracketOrder(
                action=order_action,
                quantity=self.position_size,
                takeProfitPrice=take_profit_price,
                stopLossPrice=stop_loss_price
            )

            # Attach event handlers to all orders in the bracket
            for trade in trades:
                trade.filledEvent += self.on_trade_filled
                trade.statusEvent += self.on_order_status

            self.pending_order = True

        except Exception as e:
            logger.error(f"Failed to place bracket order: {e}")
            self.pending_order = False

    def on_trade_filled(self, trade):
        """
        Callback for trade fill event.
        """
        try:
            fill = trade.fills[-1]
            fill_price = fill.execution.price
            fill_qty   = fill.execution.shares
            fill_time  = fill.execution.time
            order_action = trade.order.action.upper()

            logger.warning(f"Trade Filled: {order_action} {fill_qty} @ {fill_price} on {fill_time}")

            # Parent order fill => position entry
            if trade.order.orderType == 'MARKET':
                if order_action == 'BUY' and self.position is None:
                    self.position = 'LONG'
                    self.entry_price = fill_price
                    self.stop_loss   = self.entry_price - self.stop_loss_points
                    self.take_profit = self.entry_price + self.take_profit_points
                    self.trade_log.append({
                        'Type': 'LONG',
                        'Entry Time': fill_time,
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
                        'Entry Time': fill_time,
                        'Entry Price': self.entry_price,
                        'Exit Time': None,
                        'Exit Price': None,
                        'Result': None,
                        'Profit': 0
                    })
                    logger.warning(f"Entered SHORT @ {self.entry_price}")

            # Child order fill => position exit
            elif trade.order.orderType in ['LIMIT', 'STOP']:
                if self.position == 'LONG' and order_action == 'SELL':
                    # Exiting a long
                    pnl = (fill_price - self.entry_price) * self.position_size * self.contract_multiplier
                    self.cash += pnl
                    self.equity += pnl
                    result = 'Take Profit' if fill_price >= self.take_profit else 'Stop Loss'
                    self.trade_log[-1].update({
                        'Exit Time': fill_time,
                        'Exit Price': fill_price,
                        'Result': result,
                        'Profit': pnl
                    })
                    logger.warning(f"Exited LONG @ {fill_price} PnL=${pnl:.2f} ({result})")
                    self.position = None

                elif self.position == 'SHORT' and order_action == 'BUY':
                    # Exiting a short
                    pnl = (self.entry_price - fill_price) * self.position_size * self.contract_multiplier
                    self.cash += pnl
                    self.equity += pnl
                    result = 'Take Profit' if fill_price <= self.take_profit else 'Stop Loss'
                    self.trade_log[-1].update({
                        'Exit Time': fill_time,
                        'Exit Price': fill_price,
                        'Result': result,
                        'Profit': pnl
                    })
                    logger.warning(f"Exited SHORT @ {fill_price} PnL=${pnl:.2f} ({result})")
                    self.position = None

            # If position is flattened, we can reset pending_order
            if self.position is None:
                self.pending_order = False

            # Track equity curve
            self.equity_curve.append({'Time': fill_time, 'Equity': self.equity})

        except Exception as e:
            logger.error(f"Error in on_trade_filled: {e}")

    def on_order_status(self, trade):
        """
        Callback when an order status changes.
        """
        try:
            logger.warning(f"Order Status: ID={trade.order.orderId}, Status={trade.orderStatus.status}")
            if trade.orderStatus.status in ['Cancelled', 'Inactive', 'Filled']:
                # If parent order is cancelled, reset
                if trade.order.orderType == 'MARKET' and self.position is None:
                    logger.warning(f"Parent order {trade.order.orderId} cancelled or inactive, no position entered.")
                # We can reset pending_order if there's no position
                if self.position is None:
                    self.pending_order = False
        except Exception as e:
            logger.error(f"Error in on_order_status: {e}")

    def run(self):
        """
        Runs the strategy:
          1) Optionally fetch some historical data for warmup
          2) Subscribe to 5-second real-time bars
          3) Start the IB event loop
        """
        # Optional: warm up indicators
        self.fetch_historical_data(duration='3 D', bar_size='15 mins')

        # Request live 5-second bars
        logger.warning("Requesting real-time 5-second bars for ES...")
        ticker_5s = self.ib.reqHistoricalData(
            contract=self.es_contract,
            endDateTime='',
            durationStr='1 D',
            barSizeSetting='5 secs',
            whatToShow='TRADES',
            useRTH=False,
            keepUpToDate=True
        )
        ticker_5s.updateEvent += self.on_bar_update
        logger.warning("Real-time bar subscription set up.")

        # --- New Code to Log RSI and VWAP at Startup ---
        # Wait briefly to ensure some data is received
        import time as time_module
        time_module.sleep(5)  # Sleep for 5 seconds

        # If there are already 15-minute bars, calculate and log initial RSI and VWAP
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
        # --- End of New Code ---

        logger.warning("Starting IB event loop. Press Ctrl+C to exit.")
        try:
            self.ib.run()
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received, shutting down...")
        finally:
            if self.equity_curve:
                self.plot_equity_curve()
            logger.warning(f"Final Equity: ${self.equity:.2f}")
            if self.trade_log:
                trade_df = pd.DataFrame(self.trade_log)
                logger.warning(f"Trade Log:\n{trade_df}")
            else:
                logger.warning("No trades were executed.")
            self.ib.disconnect()


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
        es_contract  = qualified_contracts[0]
        mes_contract = qualified_contracts[1]
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