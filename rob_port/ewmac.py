import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# --------------- SETTINGS AND ASSUMPTIONS --------------------
# Backtest account parameters
initial_capital = 1_000_000.0
IDM = 1.56        # Using IDM = 1.56 (from your strategy explanation)
tau = 1.0         # Time factor for position sizing

# Define your EWMAC filters
filters = [
    {'name': 'EWMAC(4,16)',   'fast': 4,   'slow': 16,   'weight': 0.20, 'FDM': 1.20},
    {'name': 'EWMAC(8,32)',   'fast': 8,   'slow': 32,   'weight': 0.20, 'FDM': 1.15},
    {'name': 'EWMAC(16,64)',  'fast': 16,  'slow': 64,   'weight': 0.20, 'FDM': 1.10},
    {'name': 'EWMAC(32,128)', 'fast': 32,  'slow': 128,  'weight': 0.20, 'FDM': 1.05},
    {'name': 'EWMAC(64,256)', 'fast': 64,  'slow': 256,  'weight': 0.20, 'FDM': 1.00},
]

forecast_cap = 20       # Cap for forecast values (±20)
vol_window = 20         # Rolling window for daily volatility
instrument_weight = 0.25
FX = 1.0

# Contract multipliers for each instrument
contract_multipliers = {
    'es': 5,
    'nq': 2,
    'cl': 100,
    'ng': 1000,
}

# CSV file paths
data_files = {
    'es': 'Data/mes_daily_data.csv',
    'nq': 'Data/mnq_daily_data.csv',
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

def compute_performance_metrics(equity_series, days_per_year=252):
    """
    Compute basic performance metrics given a daily equity series:
      - Total Return
      - CAGR
      - Annualized Volatility
      - Sharpe Ratio (0% risk-free)
      - Max Drawdown
    """
    returns = equity_series.pct_change().dropna()
    if len(equity_series) < 2:
        return {
            'Total Return': np.nan,
            'CAGR': np.nan,
            'Annual Volatility': np.nan,
            'Sharpe Ratio': np.nan,
            'Max Drawdown': np.nan
        }
    
    total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1
    num_days = (equity_series.index[-1] - equity_series.index[0]).days
    if num_days <= 0:
        return {
            'Total Return': total_return,
            'CAGR': np.nan,
            'Annual Volatility': np.nan,
            'Sharpe Ratio': np.nan,
            'Max Drawdown': np.nan
        }
    
    years = num_days / 365.0
    cagr = (1 + total_return)**(1 / years) - 1 if years > 0 else np.nan
    
    ann_vol = returns.std() * np.sqrt(days_per_year)
    sharpe = cagr / ann_vol if ann_vol != 0 else np.nan
    
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Annual Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd
    }

# --------------- BACKTEST FUNCTIONS --------------------
def single_filter_backtest(df, fast, slow, multiplier, capital,
                           show_name="Filter", vol_window=20):
    """
    Backtest a single EWMAC(fast, slow) on a single instrument DataFrame 'df'.
    Returns a DataFrame with columns: ['Position','Trade','PnL','Equity'].
    Mark-to-market approach. Incorporates:
      - Capped forecast
      - IDM-based position sizing
      - Buffer approach for trade execution
    """
    # Compute daily returns
    df['Return'] = df['Close'].pct_change()
    # Rolling daily std dev
    df['Vol_daily'] = compute_volatility(df['Return'], vol_window)
    # Annualize (assuming ~256 trading days)
    df['Vol_annual'] = df['Vol_daily'] * np.sqrt(256)
    
    # Compute EWMAs
    df['EWMA_fast'] = compute_EWMA(df['Close'], fast)
    df['EWMA_slow'] = compute_EWMA(df['Close'], slow)
    
    # Raw forecast
    df['RawForecast'] = (df['EWMA_fast'] - df['EWMA_slow']) / df['Vol_daily'].replace(0, np.nan)
    
    # Cap the forecast at ±20
    df['CappedForecast'] = capped_forecast(df['RawForecast'], forecast_cap)
    
    # Position sizing formula (from the strategy references):
    #   N_opt = (capital * IDM * instrument_weight * tau * Forecast)
    #           / (multiplier * price * FX * vol_annual)
    df['N_opt'] = (
        capital * IDM * instrument_weight * tau
        * df['CappedForecast']
        / (multiplier * df['Close'] * FX * df['Vol_annual'].replace(0, np.nan))
    )
    
    # Buffer width:
    #   BufferWidth = 0.1 * capital * IDM * instrument_weight * tau
    #                 / (multiplier * price * FX * vol_annual)
    df['BufferWidth'] = (
        0.1 * capital * IDM * instrument_weight * tau
        / (multiplier * df['Close'] * FX * df['Vol_annual'].replace(0, np.nan))
    )
    
    # Lower and upper buffer bounds
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
        
        # Determine if we cross the buffer and need to trade
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

def combined_filter_backtest(df, filter_defs, multiplier, capital,
                             show_name="Combined", vol_window=20):
    """
    Combine multiple EWMAC filters' raw forecasts with their weights & FDM, 
    then do a mark-to-market approach. Incorporates:
      - Weighted & scaled forecasts
      - Capped combined forecast
      - IDM-based position sizing
      - Buffer approach for trade execution
    """
    # Daily returns
    df['Return'] = df['Close'].pct_change()
    # Rolling daily std dev
    df['Vol_daily'] = compute_volatility(df['Return'], vol_window)
    # Annualize
    df['Vol_annual'] = df['Vol_daily'] * np.sqrt(256)
    
    # Combine each filter's raw forecast
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
        
        # Weighted sum (plus FDM scaling)
        df['RawCombinedForecast'] += w * fdm * df[col_rf]
    
    # Cap the combined forecast at ±20
    df['CappedCombinedForecast'] = capped_forecast(df['RawCombinedForecast'], forecast_cap)
    
    # Position sizing with IDM factor:
    #   N_opt = (capital * IDM * instrument_weight * tau * CappedCombinedForecast)
    #           / (multiplier * price * FX * vol_annual)
    df['N_opt'] = (
        capital * IDM * instrument_weight * tau
        * df['CappedCombinedForecast']
        / (multiplier * df['Close'] * FX * df['Vol_annual'].replace(0, np.nan))
    )
    
    # Buffer width
    df['BufferWidth'] = (
        0.1 * capital * IDM * instrument_weight * tau
        / (multiplier * df['Close'] * FX * df['Vol_annual'].replace(0, np.nan))
    )
    df['B_L'] = (df['N_opt'] - df['BufferWidth']).round()
    df['B_U'] = (df['N_opt'] + df['BufferWidth']).round()
    
    # Simulate mark-to-market
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
        
        # Determine new position (buffer logic)
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
    
    # Compute running equity
    out['Equity'] = initial_capital + out['PnL'].cumsum()
    return out

# --------------- LOAD DATA --------------------
instruments = {}
for inst, file_path in data_files.items():
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    instruments[inst] = load_and_prepare_data(file_path)

# Common dates across all instruments
common_dates = set.intersection(*(set(df.index) for df in instruments.values()))
common_dates = sorted(list(common_dates))
if len(common_dates) == 0:
    raise ValueError("No overlapping dates found among the CSV files. Check your data.")

# Trim dataframes to common dates
for inst in instruments:
    instruments[inst] = instruments[inst].loc[common_dates].copy()

# --------------- RUN BACKTESTS --------------------
results = {}
portfolio_df = pd.DataFrame(index=common_dates, columns=['PnL','Equity'], dtype=float)
portfolio_df[['PnL','Equity']] = 0.0

for inst, df in instruments.items():
    multiplier = contract_multipliers[inst]
    
    # 1) Backtest each filter individually (optional to see each filter’s curve)
    for f in filters:
        f_name = f['name']
        fast = f['fast']
        slow = f['slow']
        
        single_res = single_filter_backtest(df.copy(), fast, slow, multiplier,
                                            initial_capital,
                                            show_name=f_name,
                                            vol_window=vol_window)
        results[(inst, f_name)] = single_res
    
    # 2) Backtest combined strategy
    combined_res = combined_filter_backtest(df.copy(), filters, multiplier,
                                            initial_capital,
                                            show_name="Combined",
                                            vol_window=vol_window)
    results[(inst, 'Combined')] = combined_res
    
    # 3) Aggregate the "Combined" strategy's PnL into the portfolio-level PnL
    portfolio_df['PnL'] += combined_res['PnL']

# 4) Compute final portfolio equity
portfolio_df['PnL'] = portfolio_df['PnL'].fillna(0)
portfolio_df['Equity'] = initial_capital + portfolio_df['PnL'].cumsum()

# --------------- PLOTTING --------------------
# Plot each instrument individually (filters + combined)
for inst in instruments.keys():
    plt.figure(figsize=(10,6))
    plt.title(f"Equity Curves for {inst.upper()}")
    
    # Plot each filter's equity curve
    for f in filters:
        f_name = f['name']
        single_res = results[(inst, f_name)]
        plt.plot(single_res.index, single_res['Equity'], label=f_name, alpha=0.7)
    
    # Plot combined strategy equity
    comb_res = results[(inst, 'Combined')]
    plt.plot(comb_res.index, comb_res['Equity'], label='Combined', color='black', linewidth=2)
    
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.ticklabel_format(style='plain', axis='y')
    plt.legend()
    plt.show()

# Plot portfolio-level equity
plt.figure(figsize=(10,6))
plt.title("Portfolio Equity (All Instruments Combined)")
plt.plot(portfolio_df.index, portfolio_df['Equity'], label='Portfolio', color='red')
plt.xlabel('Date')
plt.ylabel('Equity ($)')
plt.ticklabel_format(style='plain', axis='y')
plt.legend()
plt.show()

final_equity = portfolio_df['Equity'].iloc[-1]
print(f"\nFinal Portfolio Equity (Combined): ${final_equity:,.2f}")

# --------------- PERFORMANCE METRICS --------------------
metrics = compute_performance_metrics(portfolio_df['Equity'], days_per_year=252)
print("\n--- Portfolio Performance Metrics ---")
print(f"Total Return:      {metrics['Total Return']:.2%}")
print(f"CAGR:              {metrics['CAGR']:.2%}")
print(f"Annual Volatility: {metrics['Annual Volatility']:.2%}")
print(f"Sharpe Ratio:      {metrics['Sharpe Ratio']:.2f}")
print(f"Max Drawdown:      {metrics['Max Drawdown']:.2%}")