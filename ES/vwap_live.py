import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ib_insync import *
import logging
import sys
import pytz
from datetime import datetime, timedelta
from threading import Lock

# --- Configuration Parameters ---
IB_HOST = '127.0.0.1'        # IBKR Gateway/TWS host
IB_PORT = 7497               # IBKR Gateway/TWS paper trading port
CLIENT_ID = 2                # Unique client ID

DATA_SYMBOL = 'ES'            # E-mini S&P 500 for data
DATA_EXPIRY = '202503'        # March 2025 (example)
DATA_EXCHANGE = 'CME'         # Exchange for ES
CURRENCY = 'USD'

EXEC_SYMBOL = 'MES'            # Micro E-mini S&P 500 for execution
EXEC_EXPIRY = '202503'        # March 2025 (example)
EXEC_EXCHANGE = 'CME'         # Exchange for MES

INITIAL_CASH = 5000          # Starting cash in USD
POSITION_SIZE = 1            # Number of MES contracts per trade
CONTRACT_MULTIPLIER = 5      # Contract multiplier for MES

VWAP_PERIOD = 15             # Number of 15-minute bars to calculate VWAP
RSI_PERIOD = 14              # RSI period
RSI_OVERBOUGHT = 70          # RSI threshold for overbought
RSI_OVERSOLD = 30            # RSI threshold for oversold

STOP_LOSS_POINTS = 4         # Stop loss in points
TAKE_PROFIT_POINTS = 18      # Take profit in points

# RTH: 09:30 - 16:00 ET, Monday to Friday
RTH_START = datetime.strptime("09:30", "%H:%M").time()
RTH_END = datetime.strptime("16:00", "%H:%M").time()
EASTERN = pytz.timezone('US/Eastern')

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,  # Set to INFO or DEBUG for more verbosity
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# --- Helper Functions ---

def filter_rth(df):
    """
    Filters the DataFrame to include only Regular Trading Hours (09:30 - 16:00 ET) on weekdays.

    Parameters:
        df (pd.DataFrame): The input DataFrame with a timezone-aware datetime index.

    Returns:
        pd.DataFrame: The filtered DataFrame containing only RTH data.
    """
    eastern = pytz.timezone('US/Eastern')

    # Localize to US/Eastern if naive, else convert to US/Eastern
    if df.index.tz is None:
        df = df.tz_localize(eastern)
        logger.debug("Localized naive datetime index to US/Eastern.")
    else:
        df = df.tz_convert(eastern)
        logger.debug("Converted timezone-aware datetime index to US/Eastern.")

    # Filter for weekdays (Monday=0 to Friday=4)
    df = df[df.index.weekday < 5]

    # Filter for RTH hours: 09:30 to 16:00
    df = df.between_time('09:30', '16:00')  # Changed to include 16:00

    # Convert back to UTC for consistency
    df = df.tz_convert('UTC')

    return df

def calculate_rsi(series, period=14):
    """
    Calculates the Relative Strength Index (RSI) for a given pandas Series.

    Parameters:
        series (pd.Series): Series of prices.
        period (int): Period for RSI calculation.

    Returns:
        pd.Series: RSI values.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Calculate exponential moving averages
    avg_gain = gain.ewm(com=(period - 1), min_periods=period).mean()
    avg_loss = loss.ewm(com=(period - 1), min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# --- Live Trading Strategy Class ---
class MESFuturesLiveStrategy:
    def __init__(self, ib, es_contract, mes_contract, initial_cash=5000, position_size=1,
                 contract_multiplier=5, stop_loss_points=4, take_profit_points=18,
                 vwap_period=15, rsi_period=14,
                 rsi_overbought=70, rsi_oversold=30):
        """
        Initializes the live trading strategy.

        Parameters:
            ib (IB): ib_insync IB instance.
            es_contract (Future): E-mini S&P 500 contract for data.
            mes_contract (Future): Micro E-mini S&P 500 contract for execution.
            initial_cash (float): Starting cash in USD.
            position_size (int): Number of contracts per trade.
            contract_multiplier (int): Contract multiplier for MES.
            stop_loss_points (int): Stop loss in points.
            take_profit_points (int): Take profit in points.
            vwap_period (int): Number of 15-minute bars to calculate VWAP.
            rsi_period (int): RSI period.
            rsi_overbought (int): RSI threshold for overbought.
            rsi_oversold (int): RSI threshold for oversold.
        """
        self.ib = ib
        self.es_contract = es_contract
        self.mes_contract = mes_contract
        self.initial_cash = initial_cash
        self.position_size = position_size
        self.contract_multiplier = contract_multiplier
        self.stop_loss_points = stop_loss_points
        self.take_profit_points = take_profit_points
        self.vwap_period = vwap_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

        # Initialize cumulative VWAP and volume
        self.cumulative_vwap = 0
        self.cumulative_volume = 0

        # Strategy state
        self.cash = initial_cash
        self.equity = initial_cash
        self.position = None  # None, 'LONG', 'SHORT'
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.pending_order = False

        # Equity Curve
        self.equity_curve = []
        self.trade_log = []

        # Data Buffer for 15-minute bars
        self.lock = Lock()
        self.current_bar_start = None
        self.current_bar = {
            'open': None,
            'high': -np.inf,
            'low': np.inf,
            'close': None,
            'volume': 0
        }

        # Historical Data for Indicators
        self.historical_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'rsi'])

    def fetch_historical_data(self, duration='2 D', bar_size='15 mins'):
        """
        Fetches historical ES data to initialize indicators.

        Parameters:
            duration (str): Duration string (e.g., '2 D' for 2 days).
            bar_size (str): Bar size (e.g., '15 mins').
        """
        logger.info("Fetching historical ES data for indicator initialization...")
        try:
            bars = self.ib.reqHistoricalData(
                contract=self.es_contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=False,  # Include all trading hours
                formatDate=1,
                keepUpToDate=False
            )

            if not bars:
                logger.error("No historical data fetched. Exiting.")
                sys.exit(1)

            df = util.df(bars)
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            # Ensure the index is timezone-aware (UTC)
            if df.index.tz is None:
                df.index = pd.to_datetime(df.index).tz_localize('UTC')
            else:
                df.index = pd.to_datetime(df.index).tz_convert('UTC')

            logger.info(f"Fetched {len(df)} historical ES bars.")

            # Resample to 15-minute bars
            df_15m = df.resample('15T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            # Initialize historical data
            self.historical_data = df_15m.copy()

            # Calculate initial RSI
            self.historical_data['rsi'] = calculate_rsi(self.historical_data['close'], self.rsi_period)
            initial_rsi = self.historical_data['rsi'].iloc[-1]

            # Calculate initial VWAP using Typical Price
            recent_bars = self.historical_data.tail(self.vwap_period)
            typical_price = (recent_bars['high'] + recent_bars['low'] + recent_bars['close']) / 3
            initial_vwap = (typical_price * recent_bars['volume']).sum() / recent_bars['volume'].sum()

            logger.info(f"Initialized indicators with VWAP: {initial_vwap:.2f}, RSI: {initial_rsi:.2f}")

            # Print Initial VWAP and RSI
            print(f"Initial VWAP: {initial_vwap:.2f}, Initial RSI: {initial_rsi:.2f}")

            # Initialize cumulative VWAP and volume
            self.cumulative_vwap = (typical_price * recent_bars['volume']).sum()
            self.cumulative_volume = recent_bars['volume'].sum()

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            sys.exit(1)

    def on_realtime_bar(self, ticker, hasNewBar):
        """
        Callback function to handle real-time 5-second ES bars.

        Parameters:
            ticker (RealTimeBarList): The real-time bar list.
            hasNewBar (bool): Indicates if a new bar has been received.
        """
        with self.lock:
            if not hasNewBar:
                return

            if len(ticker) == 0:
                logger.warning("No bars received in RealTimeBarList.")
                return

            bar = ticker[-1]
            # Ensure bar time is timezone-aware (UTC)
            if bar.time.tzinfo is None:
                bar_time = pytz.UTC.localize(bar.time.replace(second=0, microsecond=0))
            else:
                bar_time = bar.time.astimezone(pytz.UTC).replace(second=0, microsecond=0)

            # Determine the start time of the current 15-minute bar
            minute = bar_time.minute
            candle_start_minute = (minute // 15) * 15
            candle_start_time = bar_time.replace(minute=candle_start_minute, second=0, microsecond=0)

            # Convert candle_start_time to ET to determine if a new trading day has started
            candle_start_time_et = candle_start_time.astimezone(EASTERN)
            candle_date_et = candle_start_time_et.date()

            if self.current_bar_start is not None:
                previous_time_et = self.current_bar_start.astimezone(EASTERN)
                previous_date_et = previous_time_et.date()

                if candle_date_et != previous_date_et:
                    # New trading day detected, reset cumulative VWAP and volume
                    self.cumulative_vwap = 0
                    self.cumulative_volume = 0
                    logger.info(f"New trading day started on {candle_date_et}. Resetting VWAP.")

            if self.current_bar_start != candle_start_time:
                # Finalize the previous 15-minute bar
                if self.current_bar_start is not None and self.current_bar['open'] is not None:
                    finalized_bar = self.current_bar.copy()
                    # Calculate Typical Price
                    typical_price = (finalized_bar['high'] + finalized_bar['low'] + finalized_bar['close']) / 3
                    # Update cumulative VWAP and volume
                    self.cumulative_vwap += (typical_price * finalized_bar['volume'])
                    self.cumulative_volume += finalized_bar['volume']
                    # Calculate VWAP
                    vwap = self.cumulative_vwap / self.cumulative_volume if self.cumulative_volume != 0 else 0

                    # Append to historical_data
                    new_row = pd.Series({
                        'open': finalized_bar['open'],
                        'high': finalized_bar['high'],
                        'low': finalized_bar['low'],
                        'close': finalized_bar['close'],
                        'volume': finalized_bar['volume']
                    }, name=self.current_bar_start)

                    # Append the new row
                    self.historical_data = pd.concat([self.historical_data, new_row.to_frame().T])

                    # Recalculate RSI on the entire 'close' series
                    self.historical_data['rsi'] = calculate_rsi(self.historical_data['close'], self.rsi_period)

                    # Get the latest RSI value
                    latest_rsi = self.historical_data['rsi'].iloc[-1]

                    logger.debug(f"Finalized 15-minute bar starting at {self.current_bar_start}: {finalized_bar}")
                    logger.debug(f"Updated cumulative VWAP: {self.cumulative_vwap:.2f}, Volume: {self.cumulative_volume}")

                    # Calculate and apply indicators
                    if self.cumulative_volume != 0:
                        vwap = self.cumulative_vwap / self.cumulative_volume
                        logger.info(f"Calculated VWAP: {vwap:.2f}, RSI: {latest_rsi:.2f}")
                        # Print the indicators and trade conditions
                        self.print_status(vwap, latest_rsi)
                        self.apply_trading_logic(vwap, latest_rsi)
                    else:
                        logger.warning("Cumulative volume is zero. Cannot calculate VWAP.")

                # Start a new 15-minute bar
                self.current_bar_start = candle_start_time
                self.current_bar = {
                    'open': bar.open_,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                }
                logger.debug(f"Started new 15-minute bar at {self.current_bar_start}")
            else:
                # Update the current 15-minute bar
                self.current_bar['high'] = max(self.current_bar['high'], bar.high)
                self.current_bar['low'] = min(self.current_bar['low'], bar.low)
                self.current_bar['close'] = bar.close
                self.current_bar['volume'] += bar.volume
                logger.debug(f"Updated current 15-minute bar at {self.current_bar_start}: {self.current_bar}")

    def print_status(self, vwap, rsi):
        """
        Prints the current status of the strategy at each 15-minute interval.

        Parameters:
            vwap (float): The calculated VWAP.
            rsi (float): The calculated RSI.
        """
        position_status = self.position if self.position else "No Position"
        logger.info(f"--- 15-Minute Interval ---")
        logger.info(f"Position Status: {position_status}")
        logger.info(f"Current Price: {self.historical_data['close'].iloc[-1]:.2f}")
        logger.info(f"VWAP: {vwap:.2f}, RSI: {rsi:.2f}")
        #logger.info(f"Trade Conditions:")
        #logger.info(f"  Long Entry: Price > VWAP and RSI > {self.rsi_overbought}")
        #logger.info(f"  Short Entry: Price < VWAP and RSI < {self.rsi_oversold}")
        logger.info(f"--------------------------")

    def apply_trading_logic(self, vwap, rsi):
        """
        Applies trading logic based on VWAP and RSI.

        Parameters:
            vwap (float): The calculated VWAP.
            rsi (float): The calculated RSI.
        """
        current_price = self.historical_data['close'].iloc[-1]
        current_time = self.historical_data.index[-1]

        # Convert current_time to ET to check if within RTH
        current_time_et = current_time.astimezone(EASTERN).time()

        # Check if current time is within RTH
        if not (RTH_START <= current_time_et <= RTH_END):
            logger.debug(f"Current time {current_time_et} is outside RTH. No trading action taken.")
            return

        logger.info(f"Evaluating Trading Signals at {current_time}: Price={current_price}, VWAP={vwap}, RSI={rsi}")

        if self.position is None and not self.pending_order:
            # Long Entry Condition
            if current_price > vwap and rsi > self.rsi_overbought:
                logger.info("Long Entry Signal Detected.")
                self.place_bracket_order('BUY', current_price, current_time)
            # Short Entry Condition
            elif current_price < vwap and rsi < self.rsi_oversold:
                logger.info("Short Entry Signal Detected.")
                self.place_bracket_order('SELL', current_price, current_time)
            else:
                logger.debug("No Entry Signal Detected.")
        else:
            logger.debug(f"Already in position: {self.position} or Pending Order: {self.pending_order}")

    def place_bracket_order(self, action, current_price, current_time):
        """
        Places a bracket order based on the action.

        Parameters:
            action (str): 'BUY' or 'SELL'.
            current_price (float): The current price for placing the parent order.
            current_time (Timestamp): The timestamp of the current bar.
        """
        try:
            # Define take-profit and stop-loss prices
            if action.upper() == 'BUY':
                take_profit_price = current_price + self.take_profit_points
                stop_loss_price = current_price - self.stop_loss_points
                order_action = 'BUY'
            elif action.upper() == 'SELL':
                take_profit_price = current_price - self.take_profit_points
                stop_loss_price = current_price + self.stop_loss_points
                order_action = 'SELL'
            else:
                logger.error(f"Invalid action: {action}. Must be 'BUY' or 'SELL'.")
                return

            # Create a standard bracket order with the parent as a limit order
            parent_order = LimitOrder(action=order_action, totalQuantity=self.position_size, lmtPrice=current_price)
            take_profit_order = LimitOrder('SELL' if action.upper() == 'BUY' else 'BUY',
                                          totalQuantity=self.position_size,
                                          lmtPrice=take_profit_price)
            stop_loss_order = StopOrder('SELL' if action.upper() == 'BUY' else 'BUY',
                                        totalQuantity=self.position_size,
                                        stopPrice=stop_loss_price)

            # Place the parent order and get the Trade object
            parent_trade = self.ib.placeOrder(self.mes_contract, parent_order)
            logger.info(f"Placed Parent {parent_order.orderType} Order ID {parent_order.orderId} for {parent_order.action} at {parent_order.lmtPrice}")

            # Attach event handlers to the parent Trade object
            parent_trade.filledEvent += self.on_trade_filled
            parent_trade.statusEvent += self.on_order_status

            # Place Take-Profit and Stop-Loss Orders and attach event handlers
            take_profit_trade = self.ib.placeOrder(self.mes_contract, take_profit_order)
            logger.info(f"Placed Take-Profit {take_profit_order.orderType} Order ID {take_profit_order.orderId} for {take_profit_order.action} at {take_profit_order.lmtPrice}")

            take_profit_trade.filledEvent += self.on_trade_filled
            take_profit_trade.statusEvent += self.on_order_status

            stop_loss_trade = self.ib.placeOrder(self.mes_contract, stop_loss_order)
            logger.info(f"Placed Stop-Loss {stop_loss_order.orderType} Order ID {stop_loss_order.orderId} for {stop_loss_order.action} at {stop_loss_order.stopPrice}")

            stop_loss_trade.filledEvent += self.on_trade_filled
            stop_loss_trade.statusEvent += self.on_order_status

            self.pending_order = True
            logger.info("Bracket order placed successfully and event handlers attached.")

        except Exception as e:
            logger.error(f"Failed to place bracket order: {e}")
            self.pending_order = False

    def on_trade_filled(self, trade):
        """
        Callback function when a trade is filled.

        Parameters:
            trade (Trade): The trade that was filled.
        """
        try:
            fill = trade.fills[-1]
            fill_price = fill.execution.price
            fill_qty = fill.execution.shares
            fill_time = fill.execution.time

            logger.info(f"Trade Filled: {trade.order.action} {fill_qty} @ {fill_price} on {fill_time}")

            if trade.order.action.upper() == 'BUY':
                if self.position is None:
                    # Entering Long Position
                    self.position = 'LONG'
                    self.entry_price = fill_price
                    self.stop_loss = self.entry_price - self.stop_loss_points
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
                    logger.debug(f"Entered LONG Position at {self.entry_price}")
                else:
                    # Exiting Long Position via Take Profit or Stop Loss
                    if self.position == 'LONG':
                        pnl = (fill_price - self.entry_price) * self.position_size * self.contract_multiplier
                        self.cash += pnl
                        self.equity += pnl
                        self.trade_log[-1].update({
                            'Exit Time': fill_time,
                            'Exit Price': fill_price,
                            'Result': 'Take Profit' if fill_price >= self.take_profit else 'Stop Loss',
                            'Profit': pnl
                        })
                        logger.info(f"Exited LONG Position at {fill_price} for P&L: ${pnl:.2f}")
                        self.position = None
            elif trade.order.action.upper() == 'SELL':
                if self.position is None:
                    # Entering Short Position
                    self.position = 'SHORT'
                    self.entry_price = fill_price
                    self.stop_loss = self.entry_price + self.stop_loss_points
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
                    logger.debug(f"Entered SHORT Position at {self.entry_price}")
                else:
                    # Exiting Short Position via Take Profit or Stop Loss
                    if self.position == 'SHORT':
                        pnl = (self.entry_price - fill_price) * self.position_size * self.contract_multiplier
                        self.cash += pnl
                        self.equity += pnl
                        self.trade_log[-1].update({
                            'Exit Time': fill_time,
                            'Exit Price': fill_price,
                            'Result': 'Take Profit' if fill_price <= self.take_profit else 'Stop Loss',
                            'Profit': pnl
                        })
                        logger.info(f"Exited SHORT Position at {fill_price} for P&L: ${pnl:.2f}")
                        self.position = None

            # Update Equity Curve
            self.equity_curve.append({'Time': fill_time, 'Equity': self.equity})

            self.pending_order = False

        except Exception as e:
            logger.error(f"Error in on_trade_filled handler: {e}")

    def on_order_status(self, trade):
        """
        Callback function when an order status changes.

        Parameters:
            trade (Trade): The trade with updated status.
        """
        try:
            logger.info(f"Order Status Update: {trade.order.orderId} - {trade.orderStatus.status}")
            if trade.order.orderId and trade.orderStatus.status in ['Cancelled', 'Inactive']:
                logger.warning(f"Order {trade.order.orderId} has been {trade.orderStatus.status.lower()}.")
                self.pending_order = False
        except Exception as e:
            logger.error(f"Error in on_order_status handler: {e}")

    def run(self):
        """
        Runs the live trading strategy by fetching historical data and subscribing to real-time ES bars.
        """
        try:
            # Fetch historical data to initialize indicators
            self.fetch_historical_data(duration='3 D', bar_size='15 mins')  # Adjust duration as needed

            # Subscribe to real-time ES bars
            logger.info("Requesting real-time 5-second bars for ES...")
            ticker = self.ib.reqRealTimeBars(
                contract=self.es_contract,
                barSize=5,
                whatToShow='TRADES',
                useRTH=False,  # Include all trading hours
                realTimeBarsOptions=[]
            )
            ticker.updateEvent += self.on_realtime_bar
            logger.info("Real-time bar handler assigned.")

            # Start the event loop
            logger.info("Starting live trading strategy...")
            self.ib.run()
        except Exception as e:
            logger.error(f"Failed to start live trading strategy: {e}")
            self.ib.disconnect()

    def plot_equity_curve(self):
        """
        Plots the equity curve.
        """
        if not self.equity_curve:
            logger.warning("No equity data to plot.")
            return

        df_equity = pd.DataFrame(self.equity_curve)
        df_equity.set_index('Time', inplace=True)

        plt.figure(figsize=(14, 7))
        plt.plot(df_equity['Equity'], label='Equity Curve')
        plt.title('Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Account Balance ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # --- Connect to IBKR ---
    ib = IB()
    try:
        ib.connect(host=IB_HOST, port=IB_PORT, clientId=CLIENT_ID)
        logger.info("Connected to IBKR.")
    except Exception as e:
        logger.error(f"Failed to connect to IBKR: {e}")
        sys.exit(1)

    # --- Define Contracts ---
    es_contract = Future(symbol=DATA_SYMBOL, lastTradeDateOrContractMonth=DATA_EXPIRY,
                        exchange=DATA_EXCHANGE, currency=CURRENCY)
    mes_contract = Future(symbol=EXEC_SYMBOL, lastTradeDateOrContractMonth=EXEC_EXPIRY,
                         exchange=EXEC_EXCHANGE, currency=CURRENCY)

    # Qualify Contracts
    try:
        qualified_contracts = ib.qualifyContracts(es_contract, mes_contract)
        es_contract = qualified_contracts[0]
        mes_contract = qualified_contracts[1]
        logger.info(f"Qualified ES Contract: {es_contract}")
        logger.info(f"Qualified MES Contract: {mes_contract}")
    except Exception as e:
        logger.error(f"Error qualifying contracts: {e}")
        ib.disconnect()
        sys.exit(1)

    # --- Initialize Strategy ---
    strategy = MESFuturesLiveStrategy(
        ib=ib,
        es_contract=es_contract,
        mes_contract=mes_contract,
        initial_cash=INITIAL_CASH,
        position_size=POSITION_SIZE,
        contract_multiplier=CONTRACT_MULTIPLIER,
        stop_loss_points=STOP_LOSS_POINTS,
        take_profit_points=TAKE_PROFIT_POINTS,
        vwap_period=VWAP_PERIOD,
        rsi_period=RSI_PERIOD,
        rsi_overbought=RSI_OVERBOUGHT,
        rsi_oversold=RSI_OVERSOLD
    )

    # --- Run Strategy ---
    try:
        strategy.run()
    except KeyboardInterrupt:
        logger.info("Interrupt received, shutting down...")
    finally:
        # Optional: Plot Equity Curve
        if strategy.equity_curve:
            strategy.plot_equity_curve()

        # Log final equity
        logger.info(f"Final Equity: ${strategy.equity:.2f}")
        # Optionally, print trade log
        if strategy.trade_log:
            trade_df = pd.DataFrame(strategy.trade_log)
            logger.info(f"Trade Log:\n{trade_df}")
        else:
            logger.info("No trades were executed.")

        ib.disconnect()
        logger.info("Disconnected from IBKR.")