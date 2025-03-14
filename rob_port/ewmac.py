import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# --------------- SETTINGS AND ASSUMPTIONS --------------------
# Backtest account parameters
initial_capital = 250000.0
risk_target = 0.20  # 20% risk target
IDM = risk_target   # Using risk target as a risk adjustment factor
tau = 1.0           # Time factor for position sizing

# Define your EWMAC filters
filters = [
    {'name': 'EWMAC(4,16)',   'fast': 4,   'slow': 16,   'weight': 0.20, 'FDM': 1.20},
    {'name': 'EWMAC(8,32)',   'fast': 8,   'slow': 32,   'weight': 0.20, 'FDM': 1.15},
    {'name': 'EWMAC(16,64)',  'fast': 16,  'slow': 64,   'weight': 0.20, 'FDM': 1.10},
    {'name': 'EWMAC(32,128)', 'fast': 32,  'slow': 128,  'weight': 0.20, 'FDM': 1.05},
    {'name': 'EWMAC(64,256)', 'fast': 64,  'slow': 256,  'weight': 0.20, 'FDM': 1.00},
]

forecast_cap = 20       # Cap for forecast
vol_window = 20         # Rolling window for daily volatility
instrument_weight = 0.25
FX = 1.0

# Contract multipliers for each instrument
contract_multipliers = {
    'es': 5,
    'nq': 2,
    'cl': 100,
    'ng': 1000
}

# CSV file paths
data_files = {
    'es': 'Data/es_daily_data.csv',
    'nq': 'Data/nq_daily_data.csv',
    'cl': 'Data/cl_daily_data.csv',
    'ng': 'Data/ng_daily_data.csv'
}

# --------------- HELPER FUNCTIONS --------------------
def compute_EWMA(series, span):
    """Compute EWMA using pandas ewm with given span."""
    return series.ewm(span=span, adjust=False).mean()

def compute_volatility(returns, window):
    """Compute rolling std dev of returns."""
    return returns.rolling(window=window).std()

def capped_forecast(raw_forecast, cap):
    """Cap forecast between -cap and +cap."""
    return raw_forecast.clip(lower=-cap, upper=cap)

def load_and_prepare_data(file_path):
    """
    Load CSV with columns: Symbol,Time,Open,High,Low,Last,Change,%Chg,Volume,"Open Int"
    Parse 'Time' as date, rename 'Last' -> 'Close'.
    """
    df = pd.read_csv(file_path, parse_dates=['Time'])
    df.sort_values('Time', inplace=True)
    df.set_index('Time', inplace=True)
    
    df.rename(columns={'Last': 'Close'}, inplace=True)
    df['Close'] = df['Close'].astype(float)
    return df

# --------------- BACKTEST FUNCTIONS --------------------
def single_filter_backtest(df, fast, slow, multiplier, capital, show_name="Filter", vol_window=20):
    """
    Backtest a single EWMAC(fast, slow) on a single instrument DataFrame 'df'.
    Returns a DataFrame with columns: ['Position','Trade','PnL','Equity'].
    Mark-to-market approach.
    """
    # Compute daily returns
    df['Return'] = df['Close'].pct_change()
    # Rolling daily std dev
    df['Vol_daily'] = compute_volatility(df['Return'], vol_window)
    # Annualize (assuming 256 trading days)
    df['Vol_annual'] = df['Vol_daily'] * np.sqrt(256)
    
    # Compute EWMAs
    df['EWMA_fast'] = compute_EWMA(df['Close'], fast)
    df['EWMA_slow'] = compute_EWMA(df['Close'], slow)
    
    # Raw forecast
    df['RawForecast'] = (df['EWMA_fast'] - df['EWMA_slow']) / df['Vol_daily'].replace(0, np.nan)
    
    # For single filter, treat weight & FDM as 1.0
    df['CappedForecast'] = capped_forecast(df['RawForecast'], forecast_cap)
    
    # Position sizing
    # N_opt = (CappedForecast * capital * IDM * instrument_weight * tau) / (multiplier * price * FX * Vol_annual)
    df['N_opt'] = (
        df['CappedForecast'] 
        * capital 
        * IDM 
        * instrument_weight 
        * tau
        / (multiplier * df['Close'] * FX * df['Vol_annual'].replace(0, np.nan))
    )
    
    # Buffer width
    df['BufferWidth'] = (
        0.1 * capital * IDM * instrument_weight * tau
        / (multiplier * df['Close'] * FX * df['Vol_annual'].replace(0, np.nan))
    )
    
    df['B_L'] = (df['N_opt'] - df['BufferWidth']).round()
    df['B_U'] = (df['N_opt'] + df['BufferWidth']).round()
    
    # Prepare results
    out = pd.DataFrame(index=df.index, columns=['Position','Trade','PnL','Equity'], dtype=float)
    out['Position'] = 0.0
    out['Trade'] = 0.0
    out['PnL'] = 0.0
    current_position = 0.0
    
    # Mark-to-market daily
    for t in range(len(df.index)):
        date = df.index[t]
        
        if t == 0:
            out.loc[date, 'Position'] = 0.0
            out.loc[date, 'Trade'] = 0.0
            out.loc[date, 'PnL'] = 0.0
            continue
        
        prev_date = df.index[t-1]
        price_today = df.loc[date, 'Close']
        price_prev = df.loc[prev_date, 'Close']
        
        # Realize PnL from holding current_position overnight
        daily_pnl = (price_today - price_prev) * current_position * multiplier
        out.loc[date, 'PnL'] = daily_pnl
        
        # Compute new position
        lower_buffer = df.loc[date, 'B_L']
        upper_buffer = df.loc[date, 'B_U']
        
        if current_position < lower_buffer:
            trade = lower_buffer - current_position
        elif current_position > upper_buffer:
            trade = upper_buffer - current_position
        else:
            trade = 0.0
        
        new_position = current_position + trade
        out.loc[date, 'Trade'] = trade
        out.loc[date, 'Position'] = new_position
        
        current_position = new_position
    
    # Compute running equity for this filter
    out['Equity'] = initial_capital + out['PnL'].cumsum()
    return out

def combined_filter_backtest(df, filter_defs, multiplier, capital, show_name="Combined", vol_window=20):
    """
    Combine multiple filters' raw forecasts with their weights & FDM, then do
    the same mark-to-market approach as above.
    """
    # Daily returns
    df['Return'] = df['Close'].pct_change()
    # Rolling daily std dev
    df['Vol_daily'] = compute_volatility(df['Return'], vol_window)
    # Annualize
    df['Vol_annual'] = df['Vol_daily'] * np.sqrt(256)
    
    # Compute each filter's raw forecast
    df['RawCombinedForecast'] = 0.0
    for f in filter_defs:
        fast = f['fast']
        slow = f['slow']
        w = f['weight']
        fdm = f['FDM']
        
        col_fast = f'EWMA_fast_{fast}'
        col_slow = f'EWMA_slow_{slow}'
        df[col_fast] = compute_EWMA(df['Close'], fast)
        df[col_slow] = compute_EWMA(df['Close'], slow)
        
        col_rf = f'RawForecast_{fast}_{slow}'
        df[col_rf] = (df[col_fast] - df[col_slow]) / df['Vol_daily'].replace(0, np.nan)
        
        # Weighted sum
        df['RawCombinedForecast'] += w * fdm * df[col_rf]
    
    # Cap
    df['CappedCombinedForecast'] = capped_forecast(df['RawCombinedForecast'], forecast_cap)
    
    # Position sizing (no "/10", use annual vol)
    df['N_opt'] = (
        df['CappedCombinedForecast'] 
        * capital 
        * IDM 
        * instrument_weight 
        * tau
        / (multiplier * df['Close'] * FX * df['Vol_annual'].replace(0, np.nan))
    )
    
    # Buffer
    df['BufferWidth'] = (
        0.1 * capital * IDM * instrument_weight * tau
        / (multiplier * df['Close'] * FX * df['Vol_annual'].replace(0, np.nan))
    )
    df['B_L'] = (df['N_opt'] - df['BufferWidth']).round()
    df['B_U'] = (df['N_opt'] + df['BufferWidth']).round()
    
    # Simulate
    out = pd.DataFrame(index=df.index, columns=['Position','Trade','PnL','Equity'], dtype=float)
    out['Position'] = 0.0
    out['Trade'] = 0.0
    out['PnL'] = 0.0
    current_position = 0.0
    
    for t in range(len(df.index)):
        date = df.index[t]
        
        if t == 0:
            out.loc[date, 'Position'] = 0.0
            out.loc[date, 'Trade'] = 0.0
            out.loc[date, 'PnL'] = 0.0
            continue
        
        prev_date = df.index[t-1]
        price_today = df.loc[date, 'Close']
        price_prev = df.loc[prev_date, 'Close']
        
        # Realize PnL from holding current_position overnight
        daily_pnl = (price_today - price_prev) * current_position * multiplier
        out.loc[date, 'PnL'] = daily_pnl
        
        # Determine new position
        lower_buffer = df.loc[date, 'B_L']
        upper_buffer = df.loc[date, 'B_U']
        
        if current_position < lower_buffer:
            trade = lower_buffer - current_position
        elif current_position > upper_buffer:
            trade = upper_buffer - current_position
        else:
            trade = 0.0
        
        new_position = current_position + trade
        out.loc[date, 'Trade'] = trade
        out.loc[date, 'Position'] = new_position
        
        current_position = new_position
    
    # Equity
    out['Equity'] = initial_capital + out['PnL'].cumsum()
    return out

# --------------- LOAD DATA --------------------
instruments = {}
for inst, file_path in data_files.items():
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    instruments[inst] = load_and_prepare_data(file_path)

# Common dates
common_dates = set.intersection(*(set(df.index) for df in instruments.values()))
common_dates = sorted(list(common_dates))
if len(common_dates) == 0:
    raise ValueError("No overlapping dates found among the CSV files. Check your data.")

# Trim
for inst in instruments:
    instruments[inst] = instruments[inst].loc[common_dates].copy()

# --------------- RUN BACKTESTS --------------------
results = {}
portfolio_df = pd.DataFrame(index=common_dates, columns=['PnL','Equity'], dtype=float)
portfolio_df[['PnL','Equity']] = 0.0

for inst, df in instruments.items():
    multiplier = contract_multipliers[inst]
    
    # 1) Backtest each filter individually
    for f in filters:
        f_name = f['name']
        fast = f['fast']
        slow = f['slow']
        
        single_res = single_filter_backtest(df.copy(), fast, slow, multiplier,
                                            initial_capital,
                                            show_name=f_name,
                                            vol_window=vol_window)
        results[(inst, f_name)] = single_res
    
    # 2) Backtest combined
    combined_res = combined_filter_backtest(df.copy(), filters, multiplier,
                                            initial_capital,
                                            show_name="Combined",
                                            vol_window=vol_window)
    results[(inst, 'Combined')] = combined_res
    
    # 3) Add the "Combined" strategy's PnL to the portfolio-level PnL
    portfolio_df['PnL'] += combined_res['PnL']

# 4) Compute final portfolio equity
portfolio_df['PnL'] = portfolio_df['PnL'].fillna(0)
portfolio_df['Equity'] = initial_capital + portfolio_df['PnL'].cumsum()

# --------------- PLOTTING --------------------
# Plot each instrument individually:
for inst in instruments.keys():
    plt.figure(figsize=(10,6))
    plt.title(f"Equity Curves for {inst.upper()}")
    
    # Plot each filter's equity
    for f in filters:
        f_name = f['name']
        single_res = results[(inst, f_name)]
        plt.plot(single_res.index, single_res['Equity'], label=f_name, alpha=0.7)
    
    # Plot combined
    comb_res = results[(inst, 'Combined')]
    plt.plot(comb_res.index, comb_res['Equity'], label='Combined', color='black', linewidth=2)
    
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    
    # Turn off scientific notation on the y-axis:
    plt.ticklabel_format(style='plain', axis='y')
    
    plt.legend()
    plt.show()

# Plot portfolio-level equity
plt.figure(figsize=(10,6))
plt.title("Portfolio Equity (All Instruments Combined)")
plt.plot(portfolio_df.index, portfolio_df['Equity'], label='Portfolio', color='red')
plt.xlabel('Date')
plt.ylabel('Equity ($)')

# Ensure we see the actual scale (no scientific notation):
plt.ticklabel_format(style='plain', axis='y')

plt.legend()
plt.show()

final_equity = portfolio_df['Equity'].iloc[-1]
print(f"\nFinal Portfolio Equity (Combined): ${final_equity:,.2f}")