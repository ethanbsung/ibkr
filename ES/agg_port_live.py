#!/usr/bin/env python
import pandas as pd
import numpy as np
from ib_insync import *
import logging
import sys
from datetime import datetime, timedelta, time
import pytz

# -------------------------------
# Configuration Parameters
# -------------------------------
IB_HOST = '127.0.0.1'
IB_PORT = 4002
CLIENT_ID = 1

# Instruments & Contract Settings for daily signals:
# For IBS instruments, we now trade the micro contracts (prefixed with 'M')
DATA_SYMBOL = 'MES'          # Micro E-mini S&P 500 for data
DATA_EXPIRY = '202506'
DATA_EXCHANGE = 'CME'
CURRENCY = 'USD'

# For IBS instruments we trade: MES, MYM, MGC, MNQ.
# For Williams %R strategy, we use MES.
# (Adjust expiries and exchanges as needed.)
# -------------------------------
# Strategy Parameters
# -------------------------------
# IBS thresholds
IBS_ENTRY_THRESHOLD = 0.1    # Enter when IBS < 0.1
IBS_EXIT_THRESHOLD  = 0.9    # Exit when IBS > 0.9

# Williams %R thresholds (for MES)
WILLR_PERIOD = 2            # Use yesterday's bar and today's current data
WR_BUY_THRESHOLD  = -90     # Buy if Williams %R < -90
WR_SELL_THRESHOLD = -30     # Sell if Williams %R > -30

# -------------------------------
# Position Tracking
# -------------------------------
# For each instrument we track the current position.
positions = {
    'MES_IBS': None,
    'MYM_IBS': None,
    'MGC_IBS': None,
    'MNQ_IBS': None,
    'MES_W': None,  # Williams for MES
}

# -------------------------------
# Helper Functions
# -------------------------------
def format_end_datetime(dt, tz):
    """
    Format the end datetime in UTC using the format yyyymmdd-HH:MM:SS.
    dt: datetime object in tz timezone.
    tz: pytz timezone of dt.
    """
    dt = dt.replace(hour=23, minute=59, second=59, microsecond=0)
    dt_utc = dt.astimezone(pytz.UTC)
    return dt_utc.strftime("%Y%m%d-%H:%M:%S")

def getDailyBar(ib, contract, endDateTime):
    """
    Request the most recent completed daily bar for the contract.
    endDateTime: string in 'yyyymmdd-HH:MM:SS' format in UTC.
    """
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=endDateTime,
        durationStr='1 D',
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    if bars:
        return bars[-1]
    return None

def get_current_day_data(ib, contract):
    """
    Request current market data for the day using the ticker.
    Returns a simple object with attributes: open, high, low, close, date.
    """
    ticker = ib.reqMktData(contract, '', False, False)
    # Allow time for market data to update
    IB.sleep(1)
    
    # Create a simple bar-like object.
    class Bar:
        pass
    bar = Bar()
    # Use ticker.last as the close price.
    bar.close = ticker.last if ticker.last is not None else 0
    # Use ticker.high and ticker.low; if not available, fallback to close.
    bar.high = ticker.high if ticker.high is not None else bar.close
    bar.low  = ticker.low  if ticker.low is not None else bar.close
    # If available, use ticker.open; otherwise fallback.
    bar.open = ticker.open if hasattr(ticker, 'open') and ticker.open is not None else bar.close
    bar.date = datetime.now(pytz.timezone('US/Eastern')).date()
    ib.cancelMktData(ticker)
    return bar

def compute_IBS(bar):
    """Calculate IBS = (close - low) / (high - low)."""
    if bar.high == bar.low:
        return 0
    return (bar.close - bar.low) / (bar.high - bar.low)

def compute_Williams(twoBars):
    """
    Compute Williams %R using two bars.
    Formula: -100 * (highestHigh - current_close) / (highestHigh - lowestLow)
    """
    highs = [bar.high for bar in twoBars]
    lows = [bar.low for bar in twoBars]
    highestHigh = max(highs)
    lowestLow = min(lows)
    if highestHigh == lowestLow:
        return 0
    return -100 * (highestHigh - twoBars[-1].close) / (highestHigh - lowestLow)

def wait_until_close(target_hour=17, target_minute=0, timezone='US/Eastern', lead_seconds=5):
    """
    Wait until the specified time minus lead_seconds.
    For example, for a target of 17:00 and lead_seconds=5,
    this function will block until 16:59:55 US/Eastern.
    """
    tz = pytz.timezone(timezone)
    while True:
        now = datetime.now(tz)
        close_today = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        target_time = close_today - timedelta(seconds=lead_seconds)
        
        if now >= target_time:
            return
        else:
            sleep_time = (target_time - now).total_seconds()
            if sleep_time < 0:
                sleep_time = 30  # fallback
            IB.sleep(sleep_time)

def print_recent_ohlc(ib):
    """
    Print the daily OHLC data for MES for the past 2 days.
    """
    logger = logging.getLogger()
    tz = pytz.timezone('US/Eastern')
    today = datetime.now(tz)
    endDateTimeStr = format_end_datetime(today, tz)
    
    # Define and qualify the MES contract
    mes_contract  = Future(symbol=DATA_SYMBOL, lastTradeDateOrContractMonth=DATA_EXPIRY, exchange=DATA_EXCHANGE, currency=CURRENCY)
    qualified = ib.qualifyContracts(mes_contract)
    if not qualified:
        logger.error("Could not qualify MES contract for OHLC printing.")
        return
    mes_contract = qualified[0]
    
    # Get the past 2 daily bars
    bars = ib.reqHistoricalData(
        mes_contract,
        endDateTime=endDateTimeStr,
        durationStr='2 D',
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    if bars:
        logger.info("Recent 2-day OHLC data for MES:")
        for bar in bars:
            logger.info(f"Date: {bar.date}, Open: {bar.open}, High: {bar.high}, Low: {bar.low}, Close: {bar.close}")
    else:
        logger.warning("Could not retrieve recent OHLC data for MES.")

# -------------------------------
# Main Trading Logic (run daily)
# -------------------------------
def run_daily_signals(ib):
    logger = logging.getLogger()

    # Define micro Futures contracts for IBS instruments.
    mes_contract  = Future(symbol='MES', lastTradeDateOrContractMonth=DATA_EXPIRY, exchange=DATA_EXCHANGE, currency=CURRENCY)
    mym_contract  = Future(symbol='MYM', lastTradeDateOrContractMonth='202506', exchange='CBOT', currency=CURRENCY)
    mgc_contract  = Future(symbol='MGC', lastTradeDateOrContractMonth='202504', exchange='COMEX', currency=CURRENCY)
    mnq_contract  = Future(symbol='MNQ', lastTradeDateOrContractMonth='202506', exchange='CME', currency=CURRENCY)

    # Qualify all contracts.
    try:
        qualified = ib.qualifyContracts(mes_contract, mym_contract, mgc_contract, mnq_contract)
        if len(qualified) < 4:
            raise ValueError("Not all contracts qualified.")
        mes_contract, mym_contract, mgc_contract, mnq_contract = qualified
        logger.info(f"Qualified MES: {mes_contract}")
        logger.info(f"Qualified MYM: {mym_contract}")
        logger.info(f"Qualified MGC: {mgc_contract}")
        logger.info(f"Qualified MNQ: {mnq_contract}")
    except Exception as e:
        logger.error(f"Contract qualification error: {e}")
        return  # Don't exit; just skip this iteration

    tz = pytz.timezone('US/Eastern')
    current_dt = datetime.now(tz)

    # Mapping for IBS Strategy instruments (excluding ZQ)
    ibs_contracts = {
        'MES': mes_contract,
        'MYM': mym_contract,
        'MGC': mgc_contract,
        'MNQ': mnq_contract
    }

    # Process IBS Strategy for each instrument using current market data at 4:59:55 PM.
    for sym, contract in ibs_contracts.items():
        current_bar = get_current_day_data(ib, contract)
        if current_bar is None:
            logger.warning(f"No current market data for {sym}. Skipping.")
            continue

        ibs_value = compute_IBS(current_bar)
        logger.info(f"{sym} current data - Price: {current_bar.close}, High: {current_bar.high}, Low: {current_bar.low}, IBS: {ibs_value:.2f}")
        pos_key = f"{sym}_IBS"
        # If no position and IBS < entry threshold, place a MOC BUY order.
        if positions[pos_key] is None and ibs_value < IBS_ENTRY_THRESHOLD:
            order = Order(action='BUY', totalQuantity=1, orderType='MOC', tif='DAY')
            trade = ib.placeOrder(contract, order)
            positions[pos_key] = trade
            logger.info(f"Placed IBS MOC BUY order for {sym}")
        # If a position exists and IBS > exit threshold, place a MOC SELL order.
        elif positions[pos_key] is not None and ibs_value > IBS_EXIT_THRESHOLD:
            order = Order(action='SELL', totalQuantity=1, orderType='MOC', tif='DAY')
            trade = ib.placeOrder(contract, order)
            positions[pos_key] = None
            logger.info(f"Placed IBS MOC SELL order for {sym}")

    # Process Williams %R Strategy for MES using current market data.
    # For Williams %R we need yesterday's completed bar and today's current data.
    current_bar_mes = get_current_day_data(ib, mes_contract)
    yesterday_dt = current_dt - timedelta(days=1)
    endDateTimeStr_yesterday = format_end_datetime(yesterday_dt, tz)
    yesterday_bar = getDailyBar(ib, mes_contract, endDateTimeStr_yesterday)
    
    if yesterday_bar is None:
        logger.warning("Could not retrieve yesterday's bar for MES for Williams %R calculation.")
    else:
        twoBars = [yesterday_bar, current_bar_mes]
        wr_value = compute_Williams(twoBars)
        logger.info(f"MES Williams %R computed as {wr_value:.2f}")
        pos_key = 'MES_W'
        # If no position and WR < buy threshold, place a MOC BUY order.
        if positions[pos_key] is None and wr_value < WR_BUY_THRESHOLD:
            order = Order(action='BUY', totalQuantity=1, orderType='MOC', tif='DAY')
            trade = ib.placeOrder(mes_contract, order)
            positions[pos_key] = trade
            logger.info("Placed Williams MOC BUY order for MES")
        # If in position and (current price > yesterday's high or WR > sell threshold), place a MOC SELL order.
        elif positions[pos_key] is not None:
            if current_bar_mes.close > yesterday_bar.high or wr_value > WR_SELL_THRESHOLD:
                order = Order(action='SELL', totalQuantity=1, orderType='MOC', tif='DAY')
                trade = ib.placeOrder(mes_contract, order)
                positions[pos_key] = None
                logger.info("Placed Williams MOC SELL order for MES")

def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    
    ib = IB()
    # Attempt to connect once at startup:
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
        logger.info("Connected to IBKR.")
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return

    # On startup, print the daily OHLC data for the past 2 days for MES.
    print_recent_ohlc(ib)

    # Main loop: run forever, evaluating signals once each day just before the futures market close
    while True:
        # 1) Wait until 5 seconds before the futures market close time (5 PM Eastern)
        wait_until_close(target_hour=17, target_minute=0, timezone='US/Eastern', lead_seconds=5)

        # 2) Once we're at the target time, run the daily signal logic using the current market data
        run_daily_signals(ib)

        # 3) Sleep a bit to avoid hammering right around the close time repeatedly
        logger.info("Finished running daily signals. Waiting for next trading day...")
        IB.sleep(60 * 5)  # Sleep 5 minutes or adjust as needed

if __name__ == '__main__':
    main()