import pandas as pd
import numpy as np
import talib  # pip install ta-lib
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------------
# 1. DATA LOADING
# --------------------------------------------------------------------------------
def load_data(csv_file):
    print("Loading data from file:", csv_file)
    df = pd.read_csv(csv_file)
    print("Original columns:", df.columns.tolist())
    
    # Convert all column names to lowercase
    df.columns = df.columns.str.lower()
    print("Converted columns to lowercase:", df.columns.tolist())
    
    # Convert 'date' column to datetime and set as index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        print("Converted 'date' column to datetime and set as index.")
    
    df.sort_index(inplace=True)
    print("Data sorted by date.")
    
    # Ensure numeric columns for open, high, low, close, and volume
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"Converted column '{col}' to numeric.")
    
    initial_rows = len(df)
    df.dropna(inplace=True)
    final_rows = len(df)
    print(f"Dropped rows with missing data. Rows before: {initial_rows}, after: {final_rows}")
    return df

# --------------------------------------------------------------------------------
# 2. SIMPLE STRATEGY DEFINITIONS
# --------------------------------------------------------------------------------
# Instead of generating many random strategies, we define a small set of simple,
# commonly used strategies with fixed (and generic) parameters.
candidate_strategies = [
    # Moving Average Crossover: Buy when short MA is above long MA.
    {"type": "ma_crossover", "short_window": 20, "long_window": 50},
    
    # RSI Threshold: Buy when RSI is below 30, sell when above 70.
    {"type": "rsi_threshold", "rsi_period": 14, "overbought": 70, "oversold": 30},
    
    # Bollinger Band Reversion: Buy when price is below lower band, sell above upper band.
    {"type": "bollinger_reversion", "window": 20, "nbdev": 2},
    
    # Donchian Channel Breakout: Buy when price exceeds the highest high over the last 20 bars.
    {"type": "donchian_breakout", "channel_period": 20, "hold_period": 10},
    
    # Price Breakout: Buy when price exceeds the recent high (last 10 bars).
    {"type": "price_breakout", "n_days": 10},
    
    # VWAP Reversion: Buy when price is below the rolling VWAP over a lookback period.
    {"type": "vwap_reversion", "lookback_window": 60}
]

# --------------------------------------------------------------------------------
# 3. BACKTESTING LOGIC
# --------------------------------------------------------------------------------
def backtest_strategy(df, strategy):
    """
    Applies the given simple strategy to the DataFrame and computes strategy returns.
    This simplified backtester:
      - Generates a 'signal' column (1 for long, -1 for short, 0 for flat).
      - Computes per-bar returns from the percent change in 'close'.
      - Computes strategy returns = signal * returns.
    
    Note: This is a very basic backtest and is not a full simulation of live trading.
    """
    data = df.copy()
    data['signal'] = 0
    
    if 'close' not in data.columns:
        raise ValueError("DataFrame must contain 'close' prices.")
    
    close = data['close']
    stype = strategy['type']
    
    # --- 1. Moving Average Crossover ---
    if stype == 'ma_crossover':
        short_w = strategy['short_window']
        long_w = strategy['long_window']
        data['ma_short'] = close.rolling(window=short_w).mean()
        data['ma_long'] = close.rolling(window=long_w).mean()
        data['signal'] = np.where(data['ma_short'] > data['ma_long'], 1, -1)
    
    # --- 2. RSI Threshold ---
    elif stype == 'rsi_threshold':
        period = strategy['rsi_period']
        overbought = strategy['overbought']
        oversold = strategy['oversold']
        data['rsi'] = talib.RSI(close, timeperiod=period)
        data['signal'] = np.where(data['rsi'] < oversold, 1,
                           np.where(data['rsi'] > overbought, -1, 0))
    
    # --- 3. Bollinger Band Reversion ---
    elif stype == 'bollinger_reversion':
        w = strategy['window']
        nbdev = strategy['nbdev']
        data['middle'] = close.rolling(w).mean()
        data['std'] = close.rolling(w).std()
        data['upper'] = data['middle'] + nbdev * data['std']
        data['lower'] = data['middle'] - nbdev * data['std']
        data['signal'] = np.where(close < data['lower'], 1,
                           np.where(close > data['upper'], -1, 0))
    
    # --- 4. Donchian Channel Breakout ---
    elif stype == 'donchian_breakout':
        channel_period = strategy['channel_period']
        data['donchian_high'] = data['high'].rolling(channel_period).max()
        data['donchian_low'] = data['low'].rolling(channel_period).min()
        data['signal'] = np.where(close > data['donchian_high'], 1,
                           np.where(close < data['donchian_low'], -1, 0))
    
    # --- 5. Price Breakout (above recent high) ---
    elif stype == 'price_breakout':
        n_days = strategy['n_days']
        data['recent_high'] = data['high'].rolling(n_days).max()
        data['signal'] = np.where(close > data['recent_high'].shift(1), 1, 0)
    
    # --- 6. VWAP Reversion ---
    elif stype == 'vwap_reversion':
        lw = strategy['lookback_window']
        if 'volume' not in data.columns:
            data['volume'] = 1  # default if volume is missing
        data['cum_pv'] = (close * data['volume']).rolling(lw).sum()
        data['cum_vol'] = data['volume'].rolling(lw).sum()
        data['rolling_vwap'] = data['cum_pv'] / data['cum_vol']
        data['signal'] = np.where(close < data['rolling_vwap'], 1, -1)
    
    # Shift the signal to simulate entering on the next bar.
    data['signal'] = data['signal'].shift(1).fillna(0)
    
    # Compute per-bar returns
    data['returns'] = close.pct_change().fillna(0)
    data['strategy_returns'] = data['signal'] * data['returns']
    
    return data['strategy_returns']

# --------------------------------------------------------------------------------
# 4. SHARPE RATIO CALCULATION
# --------------------------------------------------------------------------------
def sharpe_ratio(returns, freq_per_year=None):
    """
    Calculate the annualized Sharpe ratio.
    For 1-minute data (typical US equity hours ~390 minutes/day, ~252 days/year):
      freq_per_year ~ 390 * 252 = 98,280.
    This calculation is simplified.
    """
    if freq_per_year is None:
        freq_per_year = 98280  # for 1-minute bars
    mean_ret = returns.mean()
    std_ret = returns.std()
    if std_ret == 0:
        return 0
    return (mean_ret / std_ret) * np.sqrt(freq_per_year)

# --------------------------------------------------------------------------------
# 5. MAIN: APPLY SIMPLE STRATEGIES & REPORT RESULTS
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting backtesting process for simple candidate strategies...")
    df = load_data("ib_es_1m_data.csv")
    print("Data loaded. Total rows after cleaning:", len(df))
    
    print("\nTesting candidate strategies:")
    for strat in candidate_strategies:
        print("\n-----------------------------------")
        print("Testing strategy:", strat)
        returns = backtest_strategy(df, strat)
        sr = sharpe_ratio(returns)
        total_return = (1 + returns).prod() - 1
        print(f"Sharpe Ratio: {sr:.2f}")
        print(f"Total Return: {total_return*100:.2f}%")
    
    print("\nNote: These strategies are very simple ideas and have not been optimized.")
    print("They are intended as starting points for generating trading ideas rather than final systems.")