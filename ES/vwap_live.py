import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ib_insync import *
import logging
import sys
import pytz
from datetime import datetime, time, timedelta
from threading import Lock

# --- Configuration Parameters ---
IB_HOST = '127.0.0.1'        # IBKR Gateway/TWS host
IB_PORT = 7497               # IBKR Gateway/TWS paper trading port
CLIENT_ID = 1                # Unique client ID

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

VWAP_PERIOD_15M = 15         # Number of 15-minute bars to calculate VWAP for trading logic
RSI_PERIOD_15M = 14          # RSI period for 15-minute bars used in trading logic

VWAP_PERIOD_1M = 1           # Number of 1-minute bars to calculate VWAP for monitoring
RSI_PERIOD_1M = 14           # RSI period for 1-minute bars used in monitoring
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

def rsi(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI)."""
    delta = ohlc['close'].diff()

    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    _gain = up.ewm(com=(period - 1), min_periods=period).mean()
    _loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()

    RS = _gain / _loss
    rsi = pd.Series(100 - (100 / (1 + RS)), name="RSI")

    return rsi

def vwap(ohlc: pd.DataFrame) -> pd.Series:
    """Calculate the Volume Weighted Average Price (VWAP) since market open."""
    # Convert index to US/Eastern time
    ohlc = ohlc.copy()
    ohlc['date_et'] = ohlc.index.tz_convert('US/Eastern')

    # Determine the start of the trading day (09:30 ET)
    ohlc['date_only_et'] = ohlc['date_et'].dt.date
    ohlc['time_et'] = ohlc['date_et'].dt.time

    # Define market open for each day
    ohlc['market_open_et'] = ohlc['date_et'].apply(
        lambda x: x.replace(hour=9, minute=30, second=0, microsecond=0)
    )

    # Calculate if each row is after market open
    ohlc['is_after_open'] = ohlc['date_et'] >= ohlc['market_open_et']

    # Filter data since market open
    ohlc = ohlc[ohlc['is_after_open']]

    # Calculate Typical Price
    typical_price = (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3

    # Calculate TPV and cumulative sums since market open
    ohlc['tpv'] = typical_price * ohlc['volume']
    ohlc['cum_tpv'] = ohlc.groupby('date_only_et')['tpv'].cumsum()
    ohlc['cum_vol'] = ohlc.groupby('date_only_et')['volume'].cumsum()

    # Calculate VWAP
    ohlc['vwap'] = ohlc['cum_tpv'] / ohlc['cum_vol']

    # Assign VWAP back to original DataFrame
    ohlc['vwap_original'] = ohlc['vwap']
    ohlc = ohlc.tz_convert('UTC')  # Convert back to UTC

    return ohlc['vwap_original']

# --- Live Trading Strategy Class ---
class MESFuturesLiveStrategy:
    def __init__(self, ib, es_contract, mes_contract, initial_cash=5000, position_size=1,
                 contract_multiplier=5, stop_loss_points=4, take_profit_points=18,
                 vwap_period_15m=15, rsi_period_15m=14,
                 vwap_period_1m=1, rsi_period_1m=14,
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
            vwap_period_15m (int): Number of 15-minute bars to calculate VWAP for trading logic.
            rsi_period_15m (int): RSI period for 15-minute bars used in trading logic.
            vwap_period_1m (int): Number of 1-minute bars to calculate VWAP for monitoring.
            rsi_period_1m (int): RSI period for 1-minute bars used in monitoring.
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
        self.vwap_period_15m = vwap_period_15m
        self.rsi_period_15m = rsi_period_15m
        self.vwap_period_1m = vwap_period_1m
        self.rsi_period_1m = rsi_period_1m
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

        # Initialize cumulative VWAP and volume for 15-minute bars
        self.cumulative_vwap_15m = 0
        self.cumulative_volume_15m = 0

        # Initialize cumulative VWAP and volume for 1-minute bars
        self.cumulative_vwap_1m = 0
        self.cumulative_volume_1m = 0

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
        self.current_bar_start_15m = None
        self.current_bar_15m = {
            'open': None,
            'high': -np.inf,
            'low': np.inf,
            'close': None,
            'volume': 0
        }

        # Data Buffer for 1-minute bars
        self.current_bar_start_1m = None
        self.current_bar_1m = {
            'open': None,
            'high': -np.inf,
            'low': np.inf,
            'close': None,
            'volume': 0
        }

        # Historical Data for 15-minute Indicators
        self.historical_data_15m = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'rsi'])

        # Historical Data for 1-minute Indicators
        self.historical_data_1m = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'rsi'])

    def fetch_historical_data(self, duration='3 D', bar_size='15 mins'):
        """
        Fetches historical ES data to initialize indicators.

        Parameters:
            duration (str): Duration string (e.g., '3 D' for 3 days).
            bar_size (str): Bar size (e.g., '15 mins').
        """
        logger.info("Fetching historical ES data for indicator initialization...")
        try:
            # Fetch 15-minute historical data
            bars_15m = self.ib.reqHistoricalData(
                contract=self.es_contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=False,  # Include all trading hours
                formatDate=1,
                keepUpToDate=False
            )

            if not bars_15m:
                logger.error("No historical 15-minute data fetched. Exiting.")
                sys.exit(1)

            df_15m = util.df(bars_15m)
            df_15m.set_index('date', inplace=True)
            df_15m.sort_index(inplace=True)
            # Ensure the index is timezone-aware (UTC)
            if df_15m.index.tz is None:
                df_15m.index = pd.to_datetime(df_15m.index).tz_localize('UTC')
            else:
                df_15m.index = pd.to_datetime(df_15m.index).tz_convert('UTC')

            logger.info(f"Fetched {len(df_15m)} historical 15-minute ES bars.")

            # Resample to 15-minute bars (if not already)
            ohlc_15m = df_15m.resample('15T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            # Initialize historical data for 15-minute bars
            self.historical_data_15m = ohlc_15m.copy()

            # Calculate initial RSI for 15-minute bars
            self.historical_data_15m['rsi'] = rsi(self.historical_data_15m, self.rsi_period_15m)
            initial_rsi_15m = self.historical_data_15m['rsi'].iloc[-1]

            # Calculate initial VWAP for 15-minute bars
            self.historical_data_15m['vwap'] = vwap(self.historical_data_15m)
            initial_vwap_15m = self.historical_data_15m['vwap'].iloc[-1]

            logger.info(f"Initialized 15-minute indicators with VWAP: {initial_vwap_15m:.2f}, RSI: {initial_rsi_15m:.2f}")

            # Print Initial 15-minute VWAP and RSI
            print(f"Initial 15-minute VWAP: {initial_vwap_15m:.2f}, Initial 15-minute RSI: {initial_rsi_15m:.2f}")

            # Initialize cumulative VWAP and volume for 15-minute bars
            recent_bars_15m = self.historical_data_15m.tail(self.vwap_period_15m)
            typical_price_15m = (recent_bars_15m['high'] + recent_bars_15m['low'] + recent_bars_15m['close']) / 3
            self.cumulative_vwap_15m = (typical_price_15m * recent_bars_15m['volume']).sum()
            self.cumulative_volume_15m = recent_bars_15m['volume'].sum()

            # Fetch 1-minute historical data for monitoring
            bars_1m = self.ib.reqHistoricalData(
                contract=self.es_contract,
                endDateTime='',
                durationStr='1 D',
                barSizeSetting='1 min',
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1,
                keepUpToDate=False
            )

            if not bars_1m:
                logger.error("No historical 1-minute data fetched. Exiting.")
                sys.exit(1)

            df_1m = util.df(bars_1m)
            df_1m.set_index('date', inplace=True)
            df_1m.sort_index(inplace=True)
            # Ensure the index is timezone-aware (UTC)
            if df_1m.index.tz is None:
                df_1m.index = pd.to_datetime(df_1m.index).tz_localize('UTC')
            else:
                df_1m.index = pd.to_datetime(df_1m.index).tz_convert('UTC')

            logger.info(f"Fetched {len(df_1m)} historical 1-minute ES bars.")

            # Initialize historical data for 1-minute bars
            self.historical_data_1m = df_1m.copy()

            # Calculate initial RSI for 1-minute bars
            self.historical_data_1m['rsi'] = rsi(self.historical_data_1m, self.rsi_period_1m)
            initial_rsi_1m = self.historical_data_1m['rsi'].iloc[-1]

            # Calculate initial VWAP for 1-minute bars
            self.historical_data_1m['vwap'] = vwap(self.historical_data_1m)
            initial_vwap_1m = self.historical_data_1m['vwap'].iloc[-1]

            logger.info(f"Initialized 1-minute indicators with VWAP: {initial_vwap_1m:.2f}, RSI: {initial_rsi_1m:.2f}")

            # Print Initial 1-minute VWAP and RSI
            print(f"Initial 1-minute VWAP: {initial_vwap_1m:.2f}, Initial 1-minute RSI: {initial_rsi_1m:.2f}")

            # Initialize cumulative VWAP and volume for 1-minute bars
            recent_bars_1m = self.historical_data_1m.tail(self.vwap_period_1m)
            typical_price_1m = (recent_bars_1m['high'] + recent_bars_1m['low'] + recent_bars_1m['close']) / 3
            self.cumulative_vwap_1m = (typical_price_1m * recent_bars_1m['volume']).sum()
            self.cumulative_volume_1m = recent_bars_1m['volume'].sum()
        except:
            print("Failure fetching historical data...")
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

                # ----------------------------
                # Process 1-minute bars for monitoring
                # ----------------------------
                minute = bar_time.minute
                candle_start_minute = minute  # 1-minute bars
                candle_start_time_1m = bar_time.replace(second=0, microsecond=0)

                if self.current_bar_start_1m != candle_start_time_1m:
                    # Finalize the previous 1-minute bar
                    if self.current_bar_start_1m is not None and self.current_bar_1m['open'] is not None:
                        finalized_bar_1m = self.current_bar_1m.copy()
                        # Calculate Typical Price
                        typical_price_1m = (finalized_bar_1m['high'] + finalized_bar_1m['low'] + finalized_bar_1m['close']) / 3
                        # Update cumulative VWAP and volume for 1-minute VWAP
                        self.cumulative_vwap_1m += (typical_price_1m * finalized_bar_1m['volume'])
                        self.cumulative_volume_1m += finalized_bar_1m['volume']
                        # Calculate VWAP
                        vwap_1m = self.cumulative_vwap_1m / self.cumulative_volume_1m if self.cumulative_volume_1m != 0 else 0

                        # Append to historical_data_1m
                        new_row_1m = pd.Series({
                            'open': finalized_bar_1m['open'],
                            'high': finalized_bar_1m['high'],
                            'low': finalized_bar_1m['low'],
                            'close': finalized_bar_1m['close'],
                            'volume': finalized_bar_1m['volume']
                        }, name=self.current_bar_start_1m)

                        # Append the new row
                        self.historical_data_1m = pd.concat([self.historical_data_1m, new_row_1m.to_frame().T])

                        # Recalculate RSI for 1-minute bars
                        self.historical_data_1m['rsi'] = rsi(self.historical_data_1m, self.rsi_period_1m)

                        # Recalculate VWAP using user's logic
                        vwap_series = vwap(self.historical_data_1m)
                        latest_vwap_1m = vwap_series.iloc[-1]
                        self.historical_data_1m['vwap'] = vwap_series

                        # Get the latest RSI value
                        latest_rsi_1m = self.historical_data_1m['rsi'].iloc[-1]

                        logger.debug(f"Finalized 1-minute bar starting at {self.current_bar_start_1m}: {finalized_bar_1m}")
                        logger.debug(f"Updated cumulative VWAP (1m): {self.cumulative_vwap_1m:.2f}, Volume: {self.cumulative_volume_1m}")

                        # Calculate and print indicators
                        if self.cumulative_volume_1m != 0:
                            vwap_1m = self.cumulative_vwap_1m / self.cumulative_volume_1m
                            logger.info(f"1-Minute VWAP: {vwap_1m:.2f}, RSI: {latest_rsi_1m:.2f}")
                        else:
                            logger.warning("Cumulative volume for 1-minute bars is zero. Cannot calculate VWAP.")

                    # Start a new 1-minute bar
                    self.current_bar_start_1m = candle_start_time_1m
                    self.current_bar_1m = {
                        'open': bar.open_,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume
                    }
                    logger.debug(f"Started new 1-minute bar at {self.current_bar_start_1m}")
                else:
                    # Update the current 1-minute bar
                    self.current_bar_1m['high'] = max(self.current_bar_1m['high'], bar.high)
                    self.current_bar_1m['low'] = min(self.current_bar_1m['low'], bar.low)
                    self.current_bar_1m['close'] = bar.close
                    self.current_bar_1m['volume'] += bar.volume
                    logger.debug(f"Updated current 1-minute bar at {self.current_bar_start_1m}: {self.current_bar_1m}")

                # ----------------------------
                # Process 15-minute bars for trading logic
                # ----------------------------
                minute_15m = bar_time.minute
                candle_start_minute_15m = (minute_15m // 15) * 15
                candle_start_time_15m = bar_time.replace(minute=candle_start_minute_15m, second=0, microsecond=0)

                if self.current_bar_start_15m != candle_start_time_15m:
                    # Finalize the previous 15-minute bar
                    if self.current_bar_start_15m is not None and self.current_bar_15m['open'] is not None:
                        finalized_bar_15m = self.current_bar_15m.copy()
                        # Calculate Typical Price
                        typical_price_15m = (finalized_bar_15m['high'] + finalized_bar_15m['low'] + finalized_bar_15m['close']) / 3
                        # Update cumulative VWAP and volume for 15-minute VWAP
                        self.cumulative_vwap_15m += (typical_price_15m * finalized_bar_15m['volume'])
                        self.cumulative_volume_15m += finalized_bar_15m['volume']
                        # Calculate VWAP
                        vwap_15m = self.cumulative_vwap_15m / self.cumulative_volume_15m if self.cumulative_volume_15m != 0 else 0

                        # Append to historical_data_15m
                        new_row_15m = pd.Series({
                            'open': finalized_bar_15m['open'],
                            'high': finalized_bar_15m['high'],
                            'low': finalized_bar_15m['low'],
                            'close': finalized_bar_15m['close'],
                            'volume': finalized_bar_15m['volume']
                        }, name=self.current_bar_start_15m)

                        # Append the new row
                        self.historical_data_15m = pd.concat([self.historical_data_15m, new_row_15m.to_frame().T])

                        # Recalculate RSI for 15-minute bars
                        self.historical_data_15m['rsi'] = rsi(self.historical_data_15m, self.rsi_period_15m)

                        # Recalculate VWAP using user's logic
                        vwap_series_15m = vwap(self.historical_data_15m)
                        latest_vwap_15m = vwap_series_15m.iloc[-1]
                        self.historical_data_15m['vwap'] = vwap_series_15m

                        # Get the latest RSI value
                        latest_rsi_15m = self.historical_data_15m['rsi'].iloc[-1]

                        logger.debug(f"Finalized 15-minute bar starting at {self.current_bar_start_15m}: {finalized_bar_15m}")
                        logger.debug(f"Updated cumulative VWAP (15m): {self.cumulative_vwap_15m:.2f}, Volume: {self.cumulative_volume_15m}")

                        # Calculate and apply indicators for trading logic
                        if self.cumulative_volume_15m != 0:
                            vwap_15m = self.cumulative_vwap_15m / self.cumulative_volume_15m
                            logger.info(f"15-Minute VWAP: {vwap_15m:.2f}, RSI: {latest_rsi_15m:.2f}")
                            # Print the indicators and trade conditions
                            self.print_status(vwap_15m, latest_rsi_15m)
                            self.apply_trading_logic(vwap_15m, latest_rsi_15m)
                        else:
                            logger.warning("Cumulative volume for 15-minute bars is zero. Cannot calculate VWAP.")

                    # Start a new 15-minute bar
                    self.current_bar_start_15m = candle_start_time_15m
                    self.current_bar_15m = {
                        'open': bar.open_,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume
                    }
                    logger.debug(f"Started new 15-minute bar at {self.current_bar_start_15m}")
                else:
                    # Update the current 15-minute bar
                    self.current_bar_15m['high'] = max(self.current_bar_15m['high'], bar.high)
                    self.current_bar_15m['low'] = min(self.current_bar_15m['low'], bar.low)
                    self.current_bar_15m['close'] = bar.close
                    self.current_bar_15m['volume'] += bar.volume
                    logger.debug(f"Updated current 15-minute bar at {self.current_bar_start_15m}: {self.current_bar_15m}")

    def print_status(self, vwap, rsi):
        """
        Prints the current status of the strategy at each 15-minute interval.

        Parameters:
            vwap (float): The calculated VWAP.
            rsi (float): The calculated RSI.
        """
        position_status = self.position if self.position else "No Position"
        logger.info(f"--- 15-Minute Interval ---")
        logger.info(f"Position Status : {position_status}")
        logger.info(f"Current Price   : {self.historical_data_15m['close'].iloc[-1]:.2f}")
        logger.info(f"VWAP            : {vwap:.2f}, RSI: {rsi:.2f}")
        logger.info(f"--------------------------")

    def apply_trading_logic(self, vwap, rsi):
        """
        Applies trading logic based on VWAP and RSI.

        Parameters:
            vwap (float): The calculated VWAP.
            rsi (float): The calculated RSI.
        """
        current_price = self.historical_data_15m['close'].iloc[-1]
        current_time = self.historical_data_15m.index[-1]

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
                take_profit_order_action = 'SELL'
                stop_loss_order_action = 'SELL'
            elif action.upper() == 'SELL':
                take_profit_price = current_price - self.take_profit_points
                stop_loss_price = current_price + self.stop_loss_points
                order_action = 'SELL'
                take_profit_order_action = 'BUY'
                stop_loss_order_action = 'BUY'
            else:
                logger.error(f"Invalid action: {action}. Must be 'BUY' or 'SELL'.")
                return

            # Create a market parent order
            parent_order = MarketOrder(action=order_action, totalQuantity=self.position_size)
            # Create take-profit and stop-loss orders
            take_profit_order = LimitOrder(action=take_profit_order_action,
                                           totalQuantity=self.position_size,
                                           lmtPrice=take_profit_price)
            stop_loss_order = StopOrder(action=stop_loss_order_action,
                                        totalQuantity=self.position_size,
                                        stopPrice=stop_loss_price)

            # Place the bracket order
            logger.info(f"Placing bracket order: Parent ({order_action} {self.position_size} @ Market), "
                        f"Take-Profit ({take_profit_order_action} {self.position_size} @ {take_profit_price}), "
                        f"Stop-Loss ({stop_loss_order_action} {self.position_size} @ {stop_loss_price})")
            trades = self.ib.bracketOrder(parent_order, take_profit_order, stop_loss_order)

            # Attach event handlers to the parent Trade object
            parent_trade = trades[0]
            parent_trade.filledEvent += self.on_trade_filled
            parent_trade.statusEvent += self.on_order_status

            # Attach event handlers to child trades
            tp_trade = trades[1]
            tp_trade.filledEvent += self.on_trade_filled
            tp_trade.statusEvent += self.on_order_status

            sl_trade = trades[2]
            sl_trade.filledEvent += self.on_trade_filled
            sl_trade.statusEvent += self.on_order_status

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

            if trade.order.action.upper() in ['BUY', 'SELL']:
                if trade.order.orderType == 'MARKET':
                    # Parent order filled, entering a position
                    if trade.order.action.upper() == 'BUY' and self.position is None:
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
                    elif trade.order.action.upper() == 'SELL' and self.position is None:
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
                    # Child order filled, exiting a position
                    if trade.order.orderType in ['LIMIT', 'STOP']:
                        if self.position == 'LONG' and trade.order.action.upper() == 'SELL':
                            # Exiting Long Position via Take-Profit or Stop-Loss
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
                            logger.info(f"Exited LONG Position at {fill_price} for P&L: ${pnl:.2f}")
                            self.position = None
                        elif self.position == 'SHORT' and trade.order.action.upper() == 'BUY':
                            # Exiting Short Position via Take-Profit or Stop-Loss
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
                            logger.info(f"Exited SHORT Position at {fill_price} for P&L: ${pnl:.2f}")
                            self.position = None

            # Update Equity Curve
            self.equity_curve.append({'Time': fill_time, 'Equity': self.equity})

            # Reset pending order flag if all orders are filled
            if self.position is None:
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
            logger.info(f"Order Status Update: Order ID {trade.order.orderId} - {trade.orderStatus.status}")
            if trade.order.orderId and trade.orderStatus.status in ['Cancelled', 'Inactive', 'Filled']:
                # If order is cancelled or inactive, reset pending_order flag
                if trade.order.orderType == 'MARKET':
                    # If parent order is cancelled, reset position
                    if self.position is not None:
                        logger.warning(f"Parent order {trade.order.orderId} cancelled. Resetting position.")
                        self.position = None
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
        vwap_period_15m=VWAP_PERIOD_15M,
        rsi_period_15m=RSI_PERIOD_15M,
        vwap_period_1m=VWAP_PERIOD_1M,
        rsi_period_1m=RSI_PERIOD_1M,
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