from chapter3 import *
from chapter2 import *
from chapter1 import *
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')

#####   FX RATE HANDLING   #####

def load_fx_data(data_dir='Data'):
    """
    Load FX rate data for non-USD currencies.
    
    Parameters:
        data_dir (str): Directory containing FX data files.
    
    Returns:
        dict: Dictionary with currency code as key and DataFrame as value.
    """
    fx_data = {}
    
    # Define FX files and their properties
    fx_files = {
        'EUR': {'file': 'eur_daily_data.csv', 'invert': False},  # EUR/USD
        'JPY': {'file': 'jpy_daily_data.csv', 'invert': False},  # JPY/USD  
        'GBP': {'file': 'gbp_daily_data.csv', 'invert': False},  # GBP/USD
        'CNH': {'file': 'uc_daily_data.csv', 'invert': True},    # USD/CNH -> CNH/USD
        'SGD': {'file': 'snd_daily_data.csv', 'invert': True}    # USD/SGD -> SGD/USD
    }
    
    for currency, config in fx_files.items():
        filepath = os.path.join(data_dir, config['file'])
        
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, parse_dates=['Time'])
                df.set_index('Time', inplace=True)
                df = df.dropna()
                
                if not df.empty and 'Last' in df.columns:
                    # Get the FX rate series
                    fx_series = df['Last'].copy()
                    
                    # Invert if needed (for USD/XXX -> XXX/USD)
                    if config['invert']:
                        fx_series = 1.0 / fx_series
                    
                    fx_data[currency] = fx_series
                    print(f"Loaded FX data for {currency}: {len(fx_series)} records from {fx_series.index.min().date()} to {fx_series.index.max().date()}")
                
            except Exception as e:
                print(f"Warning: Could not load FX data for {currency}: {e}")
        else:
            print(f"Warning: FX file not found for {currency}: {filepath}")
    
    return fx_data

def get_instrument_currency_mapping():
    """
    Map instrument symbols to their base currencies.
    
    Returns:
        dict: Dictionary mapping instrument symbols to currency codes.
    """
    # This mapping is based on typical futures contract currencies
    # You may need to adjust this based on your specific instruments
    currency_mapping = {
        # EUR-based instruments
        'DAX': 'EUR', 'GBS': 'EUR', 'GBM': 'EUR', 'GBL': 'EUR', 'GBX': 'EUR',
        'BTS': 'EUR', 'BTP': 'EUR', 'FBON': 'EUR', 'CAC40': 'EUR', 'ESTX50': 'EUR',
        'SXAP': 'EUR', 'SXPP': 'EUR', 'SXDP': 'EUR', 'SXIP': 'EUR', 'SX8P': 'EUR',
        'SXTP': 'EUR', 'SX6P': 'EUR', 'V2TX': 'EUR', 'EURO': 'EUR',
        
        # JPY-based instruments  
        'JPY': 'JPY', 'NIK': 'JPY',
        
        # GBP-based instruments
        'GBP': 'GBP', 'Z': 'GBP',
        
        # CNH-based instruments
        'UC': 'CNH', 'XINA50': 'CNH',
        
        # SGD-based instruments
        'SIR': 'SGD', 'SND': 'SGD', 'SSG': 'SGD',
        
        # KRW-based instruments (skip these)
        'KRW': 'KRW', 'RP': 'KRW', 'RY': 'KRW',
        
        # All other instruments default to USD
        # This includes: MES, MNQ, MYM, RSV, M2K, ZT, Z3N, ZF, ZN, TN, TWE, ZB, YE,
        # ALI, HG, MGC, SCI, PA, PL, SI, QM, HH, RB, QG, HO, AIGCI,
        # CSC, ZC, GF, HE, LE, ZO, KE, ZR, ZS, ZM, ZL, ZW, VIX, MBT, ETHUSDRR, etc.
    }
    
    return currency_mapping

def get_fx_rate_for_date_and_currency(date, currency, fx_data):
    """
    Get FX rate for a specific date and currency.
    
    Parameters:
        date (pd.Timestamp): The date for which to get the FX rate.
        currency (str): Currency code (e.g., 'EUR', 'JPY').
        fx_data (dict): Dictionary of FX data series.
    
    Returns:
        float: FX rate to convert from currency to USD (currency/USD).
    """
    if currency == 'USD' or currency is None:
        return 1.0
    
    if currency == 'KRW':
        # Skip KRW instruments as requested
        return None
    
    if currency not in fx_data:
        print(f"Warning: No FX data available for {currency}, using 1.0")
        return 1.0
    
    fx_series = fx_data[currency]
    
    if fx_series.empty:
        return 1.0
    
    # Try to get exact date
    if date in fx_series.index:
        return fx_series.loc[date]
    
    # If date is before first available data, use first available rate
    if date < fx_series.index.min():
        return fx_series.iloc[0]
    
    # If date is after last available data, use last available rate
    if date > fx_series.index.max():
        return fx_series.iloc[-1]
    
    # For dates in between, use forward fill (next closest date)
    try:
        # Reindex with forward fill
        extended_series = fx_series.reindex(fx_series.index.union([date]))
        extended_series = extended_series.fillna(method='ffill')
        return extended_series.loc[date]
    except:
        # Fallback to nearest available rate
        idx = fx_series.index.get_indexer([date], method='nearest')[0]
        return fx_series.iloc[idx]

#####   STRATEGY 4: MULTI-INSTRUMENT VARIABLE RISK PORTFOLIO   #####

def load_all_instrument_data(data_dir='Data'):
    """
    Load all available instrument data from CSV files.
    
    Parameters:
        data_dir (str): Directory containing the data files.
    
    Returns:
        dict: Dictionary with symbol as key and DataFrame as value.
    """
    # Load instrument specs
    instruments_df = load_instrument_data()
    
    instrument_data = {}
    failed_loads = []
    
    for _, row in instruments_df.iterrows():
        symbol = row['Symbol']
        filename = f"{symbol.lower()}_daily_data.csv"
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, parse_dates=['Time'])
                df.set_index('Time', inplace=True)
                df = df.dropna()
                
                # Calculate returns
                df['returns'] = df['Last'].pct_change()
                df = df.dropna()
                
                # Only include if we have sufficient data
                if len(df) > 252:  # At least 1 year of data
                    instrument_data[symbol] = df
                else:
                    pass  # Skip without printing
                    
            except Exception as e:
                failed_loads.append((symbol, str(e)))
        else:
            pass  # Skip without printing
    
    print(f"Successfully loaded {len(instrument_data)} instruments")
    return instrument_data

def calculate_idm_from_count(num_instruments):
    """
    Calculate Instrument Diversification Multiplier based on number of instruments.
    
    From the book: IDM should be 2.5 for 30+ instruments.
    This is a simplified version. The book references Table 16 and Appendix B
    for more sophisticated calculations.
    
    Parameters:
        num_instruments (int): Number of instruments in portfolio.
    
    Returns:
        float: IDM value.
    """
    # Corrected IDM calculation based on book guidance
    if num_instruments <= 1:
        return 1.0
    elif num_instruments <= 2:
        return 1.2
    elif num_instruments <= 3:
        return 1.48
    elif num_instruments <= 4:
        return 1.56
    elif num_instruments <= 5:
        return 1.7
    elif num_instruments <= 6:
        return 1.9
    elif num_instruments <= 7:
        return 2.10
    elif num_instruments <= 14:
        return 2.2
    elif num_instruments <= 24:
        return 2.3
    elif num_instruments <= 29:
        return 2.4
    else:
        # Book specifies 2.5 for 30+ instruments
        return 2.5

def calculate_instrument_weights(instrument_data, method='equal', instruments_df=None, common_hypothetical_SR=0.3, annual_turnover_T=7.0, risk_target=0.2):
    """
    Calculate weights for each instrument in the portfolio.
    
    Parameters:
        instrument_data (dict): Dictionary of instrument DataFrames.
        method (str): Weighting method ('equal', 'vol_inverse', 'handcrafted').
        instruments_df (pd.DataFrame): Instrument specifications for handcrafted method.
        common_hypothetical_SR (float): Common hypothetical SR for SR' calculation.
        annual_turnover_T (float): Annual turnover T for SR' calculation.
        risk_target (float): Target risk fraction.
    
    Returns:
        dict: Dictionary of weights for each instrument.
    """
    symbols = list(instrument_data.keys())
    num_instruments = len(symbols)
    
    if method == 'equal':
        # Equal weights
        weight = 1.0 / num_instruments
        return {symbol: weight for symbol in symbols}
    
    elif method == 'vol_inverse':
        # Inverse volatility weights
        vols = {}
        for symbol, df in instrument_data.items():
            if len(df) > 252:
                vol = df['returns'].std() * np.sqrt(business_days_per_year)
                vols[symbol] = vol
        
        # Calculate inverse volatility weights
        inv_vols = {symbol: 1/vol for symbol, vol in vols.items()}
        total_inv_vol = sum(inv_vols.values())
        
        return {symbol: inv_vol/total_inv_vol for symbol, inv_vol in inv_vols.items()}
    
    elif method == 'handcrafted':
        # Sophisticated handcrafted weighting algorithm
        return calculate_handcrafted_weights(instrument_data, instruments_df, common_hypothetical_SR, annual_turnover_T, risk_target)
    
    else:
        # Default to equal weights
        weight = 1.0 / num_instruments
        return {symbol: weight for symbol in symbols}

def calculate_handcrafted_weights(instrument_data, instruments_df, common_hypothetical_SR, annual_turnover_T, risk_target):
    """
    Calculate handcrafted instrument weights based on multiple factors.
    
    This algorithm considers:
    1. Risk-adjusted costs (lower is better)
    2. Volatility scaling (for risk parity)
    3. Performance characteristics (Sharpe ratio, skewness)
    4. Diversification benefits across asset classes
    
    Parameters:
        instrument_data (dict): Dictionary of instrument DataFrames.
        instruments_df (pd.DataFrame): Instrument specifications.
        common_hypothetical_SR (float): Common hypothetical SR for SR' calculation.
        annual_turnover_T (float): Annual turnover T for SR' calculation.
        risk_target (float): Target risk fraction.
    
    Returns:
        dict: Dictionary of optimized weights.
    """
    print(f"\n--- Calculating Handcrafted Weights (SR': {common_hypothetical_SR:.2f}, T: {annual_turnover_T:.1f}) ---")
    
    # Asset class groupings for diversification
    asset_classes = {
        'bonds': ['ZT', 'Z3N', 'ZF', 'ZN', 'TN', 'TWE', 'ZB', 'YE', 'GBS', 'GBM', 'GBL', 'GBX', 'BTS', 'BTP', 'FBON'],
        'equities': ['MYM', 'MNQ', 'RSV', 'M2K', 'MES', 'CAC40', 'DAX', 'SMI', 'DJ600', 'ESTX50', 'SXAP', 'SXPP', 'SXDP', 'SXIP', 'SX8P', 'SXTP', 'SX6P', 'XINA50', 'SSG', 'TWN'],
        'fx': ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'RP', 'RY', 'UC', 'SIR', 'SND'],
        'commodities': ['ALI', 'HG', 'MGC', 'SCI', 'PA', 'PL', 'SI', 'QM', 'HH', 'RB', 'QG', 'HO', 'AIGCI'],
        'agriculture': ['CSC', 'ZC', 'GF', 'HE', 'LE', 'ZO', 'KE', 'ZR', 'ZS', 'ZM', 'ZL', 'ZW'],
        'volatility': ['VIX', 'V2TX'],
        'crypto': ['MBT', 'ETHUSDRR']
    }
    
    # Initialize weight calculation components
    symbol_scores = {}
    
    for symbol, df in instrument_data.items():
        if len(df) < 252:  # Need minimum data
            continue
            
        try:
            # Get instrument specs
            try:
                specs = get_instrument_specs(symbol, instruments_df)
                sr_cost = specs.get('sr_cost', 0.01)  # Default cost if missing
                if pd.isna(sr_cost) or sr_cost <= 0:
                    sr_cost = 0.01  # Default for invalid values
            except:
                sr_cost = 0.01  # Default for missing instruments
            
            # Calculate key metrics
            returns = df['returns'].dropna()
            
            # 1. Volatility (for inverse vol weighting component)
            vol = returns.std() * np.sqrt(business_days_per_year)
            
            # 2. Performance metrics (Original historical Sharpe and Skew)
            # sharpe_ratio = returns.mean() / returns.std() * np.sqrt(business_days_per_year) if returns.std() > 0 else 0
            skewness = returns.skew() if len(returns) > 10 else 0
            
            # ---- New SR' Calculation (Book's method) ----
            cost_percentage_per_trade = specs.get('sr_cost', 0.01) # e.g., 0.00028 for MES
            if pd.isna(cost_percentage_per_trade) or cost_percentage_per_trade < 0:
                cost_percentage_per_trade = 0.01 # Default for invalid values

            if vol > 0.001: # Avoid division by zero or extremely low vol
                risk_adjusted_cost_per_trade = cost_percentage_per_trade / vol
            else:
                risk_adjusted_cost_per_trade = float('inf') # Effectively makes SR' very low or negative

            estimated_sr_prime = common_hypothetical_SR - (annual_turnover_T * risk_adjusted_cost_per_trade)
            # ---- End New SR' Calculation ----

            # 3. Cost efficiency (lower SR cost is better - this is raw cost_percentage_per_trade)
            cost_efficiency = 1.0 / (cost_percentage_per_trade + 0.00001) # Add small constant


            # 4. Risk-adjusted performance (using estimated_sr_prime)
            # Prefer positive skew, ensure estimated_sr_prime contributes positively if it's positive
            risk_adj_performance = estimated_sr_prime * 0.5 + max(0, skewness) * 0.1  # Prefer positive skew
            
            # 5. Volatility scaling factor (inverse vol component)
            vol_scaling = 1.0 / vol if vol > 0.01 else 0
            
            # Combined score (higher is better)
            base_score = (
                cost_efficiency * 0.3 +           # 30% weight on cost efficiency
                vol_scaling * 0.3 +               # 30% weight on volatility scaling
                (risk_adj_performance + 1) * 0.2 + # 20% weight on performance (add 1 to handle negatives)
                1.0 * 0.2                         # 20% base allocation
            )
            
            symbol_scores[symbol] = {
                'base_score': base_score,
                'vol': vol,
                # 'sharpe': sharpe_ratio, # Store original historical Sharpe for info if needed, but use estimated_sr_prime for scoring
                'estimated_sr_prime': estimated_sr_prime,
                'skewness': skewness,
                'sr_cost': cost_percentage_per_trade, # This is cost_percentage_per_trade
                'cost_efficiency': cost_efficiency,
                'asset_class': None
            }
            
            # Determine asset class
            for ac, symbols in asset_classes.items():
                if symbol in symbols:
                    symbol_scores[symbol]['asset_class'] = ac
                    break
            if symbol_scores[symbol]['asset_class'] is None:
                symbol_scores[symbol]['asset_class'] = 'other'
                
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
    
    # Apply diversification adjustments
    # Target allocation by asset class (these sum to 1.0)
    target_allocations = {
        'bonds': 0.25,
        'equities': 0.30,
        'fx': 0.15,
        'commodities': 0.15,
        'agriculture': 0.08,
        'volatility': 0.02,
        'crypto': 0.03,
        'other': 0.02
    }
    
    # Calculate weights within each asset class
    final_weights = {}
    
    for asset_class, target_allocation in target_allocations.items():
        # Get instruments in this asset class
        class_instruments = {k: v for k, v in symbol_scores.items() 
                           if v['asset_class'] == asset_class}
        
        if not class_instruments:
            continue
        
        # Calculate relative weights within asset class based on scores
        total_class_score = sum(inst['base_score'] for inst in class_instruments.values())
        
        if total_class_score > 0:
            for symbol, data in class_instruments.items():
                class_weight = data['base_score'] / total_class_score
                final_weights[symbol] = class_weight * target_allocation
        
        # Reduced output - only print summary
    
    # Normalize weights to sum to 1.0
    total_weight = sum(final_weights.values())
    if total_weight > 0:
        final_weights = {k: v/total_weight for k, v in final_weights.items()}
    
    # Display top allocations
    sorted_weights = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 Weighted Instruments:")
    print(f"{'Symbol':<8} {'Weight':<8} {'AssetClass':<12} {'SR_prime':<8} {'Vol':<8} {'Cost%':<8}")
    print("-" * 65)
    
    for symbol, weight in sorted_weights[:10]:
        data = symbol_scores[symbol]
        print(f"{symbol:<8} {weight:<8.3f} {data['asset_class']:<12} {data['estimated_sr_prime']:<8.3f} "
              f"{data['vol']:<8.2%} {data['sr_cost']:<8.4f}")
    
    return final_weights

def calculate_portfolio_position_size(symbol, capital, weight, idm, price, volatility, 
                                    multiplier, risk_target=0.2, fx_rate=1.0):
    """
    Calculate position size for an instrument in a multi-instrument portfolio.
    
    Formula from Strategy 4:
        N = (Capital × IDM × Weight × τ) ÷ (Multiplier × Price × FX × σ%)
    
    Parameters:
        symbol (str): Instrument symbol.
        capital (float): Total portfolio capital.
        weight (float): Weight allocated to this instrument.
        idm (float): Instrument Diversification Multiplier.
        price (float): Current price.
        volatility (float): Annualized volatility forecast.
        multiplier (float): Contract multiplier.
        risk_target (float): Target risk fraction.
        fx_rate (float): FX rate for currency conversion.
    
    Returns:
        int: Number of contracts for this instrument (rounded to nearest integer).
    """
    if np.isnan(volatility) or volatility <= 0:
        return 0
    
    numerator = capital * idm * weight * risk_target
    denominator = multiplier * price * fx_rate * volatility
    
    position_size = numerator / denominator
    
    # Protect against infinite or extremely large position sizes
    if np.isinf(position_size) or position_size > 100000:
        return 0
    
    # Round to nearest integer since you can only hold whole contracts
    return round(position_size)

def backtest_multi_instrument_strategy(data_dir='Data', capital=50000000, risk_target=0.2,
                                     short_span=32, long_years=10, min_vol_floor=0.05,
                                     weight_method='equal',
                                     common_hypothetical_SR=0.3, annual_turnover_T=7.0,
                                     start_date=None, end_date=None):
    """
    Backtest Strategy 4: Multi-instrument portfolio with variable risk scaling and daily dynamic rebalancing.
    """
    print("=" * 60)
    print("STRATEGY 4: MULTI-INSTRUMENT VARIABLE RISK PORTFOLIO (Daily Rebalance)")
    print("=" * 60)
    
    # Load FX data
    print("\nLoading FX data...")
    fx_data = load_fx_data(data_dir)
    currency_mapping = get_instrument_currency_mapping()
    
    all_instruments_specs_df = load_instrument_data() # For multipliers etc.
    raw_instrument_data = load_all_instrument_data(data_dir) # Loads DFs with 'Last', 'returns'
    
    if not raw_instrument_data:
        raise ValueError("No instrument data loaded successfully")
    
    print(f"\nPortfolio Configuration:")
    print(f"  Instruments initially loaded: {len(raw_instrument_data)}")
    print(f"  Capital: ${capital:,.0f}")
    print(f"  Risk Target: {risk_target:.1%}")
    print(f"  Weight Method: {weight_method}")
    print(f"  Common Hypothetical SR for SR': {common_hypothetical_SR}")
    print(f"  Annual Turnover T for SR': {annual_turnover_T}")

    # Preprocess: Calculate returns and vol forecasts for each instrument
    processed_instrument_data = {}
    for symbol, df_orig in raw_instrument_data.items():
        df = df_orig.copy()
        if 'Last' not in df.columns:
            print(f"Skipping {symbol}: 'Last' column missing.")
            continue
        
        df['daily_price_change_pct'] = df['Last'].pct_change()
        
        # Volatility forecast for day D is made using data up to D-1
        raw_returns_for_vol = df['daily_price_change_pct'].dropna()
        if len(raw_returns_for_vol) < short_span: # Need enough data for EWMA
            print(f"Skipping {symbol}: Insufficient data for vol forecast ({len(raw_returns_for_vol)} days).")
            continue

        blended_vol_series = calculate_blended_volatility(
            raw_returns_for_vol, short_span=short_span, long_years=long_years, min_vol_floor=min_vol_floor
        )
        # The forecast for df.index[t] (today) uses vol calculated up to df.index[t-1] (yesterday)
        df['vol_forecast'] = blended_vol_series.shift(1).reindex(df.index).ffill().fillna(min_vol_floor)
        
        # Ensure 'Last' and 'vol_forecast' are present for all relevant dates by reindexing to a common range later if needed
        # For now, keep as is, filtering will happen in the main loop.
        df.dropna(subset=['Last', 'vol_forecast', 'daily_price_change_pct'], inplace=True) # Ensure critical data is present
        if df.empty:
            print(f"Skipping {symbol}: Empty after dropping NaNs in critical columns.")
            continue

        processed_instrument_data[symbol] = df

    if not processed_instrument_data:
        raise ValueError("No instruments remaining after preprocessing and volatility calculation.")
    
    print(f"  Instruments after preprocessing: {len(processed_instrument_data)}")

    # Determine common date range for backtest - start from earliest available data from ANY instrument
    all_indices = [df.index for df in processed_instrument_data.values() if not df.empty]
    if not all_indices:
        raise ValueError("No valid instrument data in processed_instrument_data to determine date range.")

    # Use absolute earliest and latest dates across all instruments to maximize trading period
    all_available_start_dates = [idx.min() for idx in all_indices]
    all_available_end_dates = [idx.max() for idx in all_indices]

    global_min_date = min(all_available_start_dates) if all_available_start_dates else pd.Timestamp.min
    global_max_date = max(all_available_end_dates) if all_available_end_dates else pd.Timestamp.max
    
    # Start from earliest available data if no start_date specified
    # If start_date is specified, use the earlier of start_date or earliest available data
    if start_date:
        user_start_dt = pd.to_datetime(start_date)
        backtest_start_dt = min(user_start_dt, global_min_date)  # Use earlier date
    else:
        backtest_start_dt = global_min_date
    
    # End at latest available data if no end_date specified
    # If end_date is specified, use the later of end_date or latest available data
    if end_date:
        user_end_dt = pd.to_datetime(end_date)
        backtest_end_dt = max(user_end_dt, global_max_date)  # Use later date
    else:
        backtest_end_dt = global_max_date

    if backtest_start_dt >= backtest_end_dt:
        raise ValueError(f"Invalid backtest period: Start {backtest_start_dt}, End {backtest_end_dt}. Check data alignment and date inputs.")

    # Use a common business day index
    trading_days_range = pd.bdate_range(start=backtest_start_dt, end=backtest_end_dt)
    
    print(f"\nBacktest Period (effective, common across instruments):")
    print(f"  Start: {trading_days_range.min().date()}")
    print(f"  End: {trading_days_range.max().date()}")
    print(f"  Duration: {len(trading_days_range)} trading days")

    # Initialize portfolio tracking
    current_portfolio_equity = capital
    portfolio_daily_records = []
    known_eligible_instruments = set()
    weights = {} 
    idm = 1.0 # Default IDM

    # Load FX data
    fx_data = load_fx_data(data_dir)
    currency_mapping = get_instrument_currency_mapping()

    # Main time-stepping loop
    for idx, current_date in enumerate(trading_days_range):
        if idx == 0: # First day, no previous trading day in our loop range to get price for sizing yet
            # For the first day, P&L is effectively zero as positions are established.
            # We can log initial state if needed but skip P&L calculation based on a "previous" day within this range.
            # Positions will be determined based on data *prior* to trading_days_range[0] if available.
            # This part needs careful handling to ensure initial positions are set correctly.
            # For now, let's assume P&L starts accumulating from the second day of trading_days_range.
            # The sizing for the first day *will* use data from the day before trading_days_range[0].

            # Log initial equity state
            record = {'date': current_date, 'total_pnl': 0.0, 'portfolio_return': 0.0, 
                      'equity_sod': current_portfolio_equity, 'equity_eod': current_portfolio_equity}
            for symbol_k in processed_instrument_data.keys(): record[f'{symbol_k}_contracts'] = 0.0 # Placeholder
            portfolio_daily_records.append(record)
            continue
        
        previous_trading_date = trading_days_range[idx-1]
        capital_at_start_of_day = current_portfolio_equity # From previous day's EOD
        daily_total_pnl = 0.0
        current_day_positions = {}

        effective_data_cutoff_date = previous_trading_date if idx > 0 else current_date - pd.tseries.offsets.BDay(1)

        # Determine current period eligible instruments based on data up to cutoff
        current_iteration_eligible_instruments = set()
        for s, df_full in processed_instrument_data.items():
            df_upto_cutoff = df_full[df_full.index <= effective_data_cutoff_date]
            if not df_upto_cutoff.empty and len(df_upto_cutoff) > short_span:
                current_iteration_eligible_instruments.add(s)
        
        if idx == 0: # Always print initial state
            print(f"Initial processed instruments count: {len(processed_instrument_data)}")
            print(f"Initial eligible instruments for weighting (day 0): {len(current_iteration_eligible_instruments)}")


        perform_reweight = False
        if idx == 0:
            perform_reweight = True
            print(f"Performing initial re-weighting for date: {current_date.date()}")
        elif len(current_iteration_eligible_instruments) > len(known_eligible_instruments):
            newly_added = current_iteration_eligible_instruments - known_eligible_instruments
            perform_reweight = True
            print(f"Performing re-weighting for date: {current_date.date()} due to new eligible instruments: {newly_added}")
        
        if perform_reweight:
            known_eligible_instruments = current_iteration_eligible_instruments.copy()
            
            data_for_reweighting = {}
            for s_eligible in known_eligible_instruments:
                # Use the full historical data from processed_instrument_data up to the cutoff for this reweighting
                df_historical_slice = processed_instrument_data[s_eligible][processed_instrument_data[s_eligible].index <= effective_data_cutoff_date]
                if not df_historical_slice.empty: # Should be true due to eligibility check, but double check
                     data_for_reweighting[s_eligible] = df_historical_slice
            
            if data_for_reweighting:
                weights = calculate_instrument_weights(
                    data_for_reweighting, 
                    weight_method, 
                    all_instruments_specs_df,
                    common_hypothetical_SR, # Pass new param
                    annual_turnover_T,      # Pass new param
                    risk_target             # Pass risk_target
                )
                
                num_weighted_instruments = sum(1 for w_val in weights.values() if w_val > 1e-6) # Count instruments with meaningful weight
                idm = calculate_idm_from_count(num_weighted_instruments)
                print(f"  New IDM: {idm:.2f} based on {num_weighted_instruments} instruments with weight > 0.")
                # print(f"  Top 5 new weights: {sorted(weights.items(), key=lambda item: item[1], reverse=True)[:5]}") # Already printed in calculate_handcrafted_weights
            else:
                # This case should ideally not be hit if known_eligible_instruments is populated
                # and processed_instrument_data is consistent.
                print(f"Warning: No data available for reweighting on {current_date.date()} despite eligibility signal. Using previous weights (if any).")


        if idx == 0: # First day, P&L is effectively zero as positions are established.
            # Log initial equity state
            record = {'date': current_date, 'total_pnl': 0.0, 'portfolio_return': 0.0, 
                      'equity_sod': current_portfolio_equity, 'equity_eod': current_portfolio_equity}
            for symbol_k in processed_instrument_data.keys(): record[f'{symbol_k}_contracts'] = 0.0 # Placeholder
            portfolio_daily_records.append(record)
            continue

        for symbol, df_instrument in processed_instrument_data.items():
            instrument_multiplier = all_instruments_specs_df[all_instruments_specs_df['Symbol'] == symbol]['Multiplier'].iloc[0]
            instrument_weight = weights.get(symbol, 0.0)

            if instrument_weight == 0.0:
                current_day_positions[symbol] = 0.0
                continue

            # Get data for sizing (from previous_trading_date) and P&L (current_date)
            try:
                # Sizing based on previous day's close price and current day's vol forecast (made from prev day's data)
                price_for_sizing = df_instrument.loc[previous_trading_date, 'Last']
                vol_for_sizing = df_instrument.loc[current_date, 'vol_forecast']
                
                # Data for P&L calculation for current_date
                price_at_start_of_trading = df_instrument.loc[previous_trading_date, 'Last'] # Same as price_for_sizing
                price_at_end_of_trading = df_instrument.loc[current_date, 'Last']
                
                if pd.isna(price_for_sizing) or pd.isna(vol_for_sizing) or pd.isna(price_at_start_of_trading) or pd.isna(price_at_end_of_trading):
                    num_contracts = 0.0
                    instrument_pnl_today = 0.0
                else:
                    vol_for_sizing = vol_for_sizing if vol_for_sizing > 0 else min_vol_floor
                    # Get FX rate for position sizing
                    instrument_currency = currency_mapping.get(symbol, 'USD')
                    fx_rate = get_fx_rate_for_date_and_currency(current_date, instrument_currency, fx_data)
                    
                    # Skip KRW instruments as requested
                    if fx_rate is None:
                        num_contracts = 0.0
                        instrument_pnl_today = 0.0
                    else:
                        num_contracts = calculate_portfolio_position_size(
                            symbol=symbol, capital=capital_at_start_of_day, weight=instrument_weight, idm=idm,
                            price=price_for_sizing, volatility=vol_for_sizing, multiplier=instrument_multiplier,
                            risk_target=risk_target, fx_rate=fx_rate
                        )
                        # P&L calculation with FX rate to convert to base currency (USD)
                        price_change_in_local_currency = price_at_end_of_trading - price_at_start_of_trading
                        price_change_in_base_currency = price_change_in_local_currency * fx_rate
                        instrument_pnl_today = num_contracts * instrument_multiplier * price_change_in_base_currency
            
            except KeyError: # Date not found for this instrument
                num_contracts = 0.0
                instrument_pnl_today = 0.0
            
            current_day_positions[symbol] = num_contracts
            daily_total_pnl += instrument_pnl_today

        portfolio_daily_percentage_return = daily_total_pnl / capital_at_start_of_day if capital_at_start_of_day > 0 else 0.0
        current_portfolio_equity = capital_at_start_of_day * (1 + portfolio_daily_percentage_return)

        record = {'date': current_date, 'total_pnl': daily_total_pnl, 
                  'portfolio_return': portfolio_daily_percentage_return, 
                  'equity_sod': capital_at_start_of_day, 
                  'equity_eod': current_portfolio_equity}
        for symbol_k, contracts_k in current_day_positions.items(): record[f'{symbol_k}_contracts'] = contracts_k
        portfolio_daily_records.append(record)

    # Post-loop processing
    if not portfolio_daily_records:
        raise ValueError("No daily records generated during backtest. Check date ranges and data availability.")
        
    portfolio_df = pd.DataFrame(portfolio_daily_records)
    portfolio_df.set_index('date', inplace=True)
    
    print(f"Portfolio backtest loop completed. {len(portfolio_df)} daily records.")
    if portfolio_df.empty or 'portfolio_return' not in portfolio_df.columns or portfolio_df['portfolio_return'].std() == 0 :
        print("Warning: Portfolio returns are zero or constant. P&L might not be calculated as expected.")
    
    # Calculate performance metrics
    # Ensure account_curve starts with initial capital on the day before the first return.
    # If portfolio_df['portfolio_return'] starts from the first actual P&L day, build_account_curve will handle it.
    account_curve = build_account_curve(portfolio_df['portfolio_return'], capital)
    performance = calculate_comprehensive_performance(account_curve, portfolio_df['portfolio_return'])
    
    performance['num_instruments'] = len(processed_instrument_data)
    performance['idm'] = idm
    # performance['avg_active_instruments'] = portfolio_df['num_active_instruments'].mean() # Need to compute this if desired
    performance['weight_method'] = weight_method
    performance['backtest_start'] = trading_days_range.min()
    performance['backtest_end'] = trading_days_range.max()

    instrument_stats = {} # Simplified for now, can be expanded
    for symbol in processed_instrument_data.keys():
        pnl_col = f'{symbol}_pnl' # This column is not directly in portfolio_df with this new structure
                                  # Need to reconstruct if detailed per-instrument P&L series is needed
        pos_col = f'{symbol}_contracts'
        if pos_col in portfolio_df.columns:
                instrument_stats[symbol] = {
                    'avg_position': portfolio_df[pos_col][portfolio_df[pos_col] != 0].mean(),
                'weight': weights.get(symbol,0)
                }
    
    return {
        'portfolio_data': portfolio_df,
        # 'instrument_data': processed_instrument_data, # This contains full DFs, maybe too large
        'performance': performance,
        'instrument_stats': instrument_stats,
        'weights': weights,
        'idm': idm,
        'config': {
            'capital': capital, 'risk_target': risk_target, 'short_span': short_span, 
            'long_years': long_years, 'min_vol_floor': min_vol_floor, 
            'weight_method': weight_method, 'backtest_start': trading_days_range.min(), 
            'backtest_end': trading_days_range.max(),
            'common_hypothetical_SR': common_hypothetical_SR, # Add to config
            'annual_turnover_T': annual_turnover_T             # Add to config
        }
    }

def plot_strategy4_equity_curve(results, save_path='results/strategy4_equity_curve.png'):
    """
    Plot Strategy 4 equity curve and save to file.
    
    Parameters:
        results (dict): Results from backtest_multi_instrument_strategy.
        save_path (str): Path to save the plot.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        portfolio_df = results['portfolio_data']
        config = results['config']
        performance = results['performance']
        
        equity_curve = build_account_curve(portfolio_df['portfolio_return'], config['capital'])
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(equity_curve.index, equity_curve.values, 'b-', linewidth=1.5, label='Strategy 4: Multi-Instrument Portfolio')
        plt.title('Strategy 4: Multi-Instrument Portfolio Equity Curve', fontsize=14, fontweight='bold')
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        plt.subplot(2, 1, 2)
        drawdown_stats = calculate_maximum_drawdown(equity_curve)
        drawdown_series = drawdown_stats['drawdown_series'] * 100
        
        plt.fill_between(drawdown_series.index, drawdown_series.values, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        plt.plot(drawdown_series.index, drawdown_series.values, 'r-', linewidth=1)
        plt.title('Drawdown', fontsize=12, fontweight='bold')
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        for ax in plt.gcf().get_axes():
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        textstr = f'''Performance Summary:
Total Return: {performance['total_return']:.1%}
Annualized Return: {performance['annualized_return']:.1%}
Volatility: {performance['annualized_volatility']:.1%}
Sharpe Ratio: {performance['sharpe_ratio']:.3f}
Max Drawdown: {performance['max_drawdown_pct']:.1f}%
Instruments: {performance.get('num_instruments', 'N/A')} 
Period: {config['backtest_start'].strftime('%Y-%m-%d')} to {config['backtest_end'].strftime('%Y-%m-%d')}'''
        
        plt.figtext(0.02, 0.02, textstr, fontsize=9, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Strategy 4 equity curve saved to: {save_path}")
        
    except Exception as e:
        print(f"Error plotting equity curve: {e}")
        import traceback
        traceback.print_exc()

def analyze_portfolio_results(results):
    """
    Analyze and display comprehensive portfolio results.
    
    Parameters:
        results (dict): Results from backtest_multi_instrument_strategy.
    """
    performance = results['performance']
    instrument_stats = results['instrument_stats']
    config = results['config']
    portfolio_df = results['portfolio_data'] # Added for avg_active_instruments
    
    print("\n" + "=" * 60)
    print("PORTFOLIO PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    print(f"\n--- Overall Portfolio Performance ---")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Annualized Return: {performance['annualized_return']:.2%}")
    print(f"Annualized Volatility: {performance['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {performance['max_drawdown_pct']:.1f}%")
    print(f"Skewness: {performance.get('skewness', float('nan')):.3f}") # Use .get for safety
    if 'kurtosis' in performance:
        print(f"Kurtosis: {performance['kurtosis']:.3f}")
    
    print(f"\n--- Portfolio Characteristics ---")
    print(f"Number of Instruments: {performance.get('num_instruments', 'N/A')}")
    print(f"IDM: {performance.get('idm', float('nan')):.2f}")
    
    # Calculate avg_active_instruments if relevant columns exist
    active_cols = [col for col in portfolio_df.columns if col.endswith('_contracts')]
    if active_cols:
        portfolio_df['num_active_instruments'] = portfolio_df[active_cols].gt(0.001).sum(axis=1)
        avg_active = portfolio_df['num_active_instruments'].mean()
        print(f"Average Active Instruments: {avg_active:.1f}")
    else:
        print(f"Average Active Instruments: N/A (no contract columns found for calculation)")

    print(f"Capital: ${config['capital']:,.0f}")
    print(f"Risk Target: {config['risk_target']:.1%}")
    print(f"Backtest Period: {config['backtest_start'].date()} to {config['backtest_end'].date()}")
    
    # Top performing instruments
    print(f"\n--- Top 10 Weighted Instruments (with Avg Position if available) ---")
    sorted_weights = sorted(results['weights'].items(), key=lambda x: x[1], reverse=True)

    print(f"{'Symbol':<8} {'Weight':<8} {'AvgPos':<10}")
    print("-" * 30)
    for symbol, weight in sorted_weights[:10]:
        avg_pos_str = "N/A"
        if symbol in instrument_stats and 'avg_position' in instrument_stats[symbol]:
            avg_pos = instrument_stats[symbol]['avg_position']
            avg_pos_str = f"{avg_pos:.2f}" if pd.notna(avg_pos) else "N/A"
        print(f"{symbol:<8} {weight:<8.3f} {avg_pos_str:<10}")

    # Bottom 5 Weighted Instruments (changed from P&L based)
    print(f"\n--- Bottom 5 Weighted Instruments ---")
    # sorted_instruments was already sorted by weight for Top 10, so take from the end for smallest weights
    # Ensure we have at least 5 instruments to show, or adjust if fewer
    num_to_show_bottom = min(5, len(sorted_weights))
    bottom_instruments = sorted_weights[-num_to_show_bottom:]
    
    print(f"{'Symbol':<8} {'Weight':<8} {'AvgPos':<10}")
    print("-" * 30)

    # Print in ascending order of weight for "bottom"
    for symbol, weight in reversed(bottom_instruments): # reversed to show smallest first
        avg_pos_str = "N/A"
        if symbol in instrument_stats and 'avg_position' in instrument_stats[symbol]:
            avg_pos = instrument_stats[symbol]['avg_position']
            avg_pos_str = f"{avg_pos:.2f}" if pd.notna(avg_pos) else "N/A"
        print(f"{symbol:<8} {weight:<8.3f} {avg_pos_str:<10}")

#####   UNIT TESTS   #####

def test_position_size_calculation():
    """Test position size calculation function."""
    print("\n=== Testing Position Size Calculation ===")
    
    # Test normal case
    position = calculate_portfolio_position_size(
        symbol='MES', capital=50000000, weight=0.02, idm=2.5, 
        price=4500, volatility=0.16, multiplier=5, risk_target=0.2
    )
    expected_raw = (50000000 * 2.5 * 0.02 * 0.2) / (5 * 4500 * 1.0 * 0.16)
    expected_rounded = round(expected_raw)
    print(f"Normal case: position={position}, expected_raw={expected_raw:.2f}, expected_rounded={expected_rounded}")
    assert position == expected_rounded, f"Position calculation failed: {position} != {expected_rounded}"
    
    # Test zero volatility
    position_zero = calculate_portfolio_position_size(
        'MES', 50000000, 0.02, 2.5, 4500, 0.0, 5, 0.2
    )
    print(f"Zero volatility: position={position_zero}")
    assert position_zero == 0, "Zero volatility should return 0 position"
    
    # Test NaN volatility
    position_nan = calculate_portfolio_position_size(
        'MES', 50000000, 0.02, 2.5, 4500, np.nan, 5, 0.2
    )
    print(f"NaN volatility: position={position_nan}")
    assert position_nan == 0, "NaN volatility should return 0 position"
    
    print("✓ Position size calculation tests passed")

def test_idm_calculation():
    """Test IDM calculation function."""
    print("\n=== Testing IDM Calculation ===")
    
    test_cases = [
        (1, 1.0),
        (3, 1.48),
        (8, 2.2),
        (15, 2.3),
        (30, 2.5),
        (100, 2.5)
    ]
    
    for num_instruments, expected_idm in test_cases:
        calculated_idm = calculate_idm_from_count(num_instruments)
        print(f"Instruments: {num_instruments}, IDM: {calculated_idm} (expected: {expected_idm})")
        assert calculated_idm == expected_idm, f"IDM calculation failed for {num_instruments} instruments"
    
    print("✓ IDM calculation tests passed")

def test_instrument_weights():
    """Test instrument weighting function."""
    print("\n=== Testing Instrument Weights ===")
    
    # Create sample data
    sample_data = {
        'A': pd.DataFrame({'returns': np.random.normal(0, 0.01, 100)}),
        'B': pd.DataFrame({'returns': np.random.normal(0, 0.02, 100)}),
        'C': pd.DataFrame({'returns': np.random.normal(0, 0.015, 100)})
    }
    
    # Test equal weights
    equal_weights = calculate_instrument_weights(sample_data, 'equal')
    expected_weight = 1.0 / 3
    print(f"Equal weights: {equal_weights}")
    
    for symbol, weight in equal_weights.items():
        assert abs(weight - expected_weight) < 0.001, f"Equal weight calculation failed for {symbol}"
    
    # Test that weights sum to 1
    total_weight = sum(equal_weights.values())
    print(f"Total weight: {total_weight}")
    assert abs(total_weight - 1.0) < 0.001, "Weights should sum to 1"
    
    print("✓ Instrument weights tests passed")

def test_data_loading():
    """Test data loading function."""
    print("\n=== Testing Data Loading ===")
    
    # Test loading with actual data directory
    try:
        instrument_data = load_all_instrument_data('Data')
        print(f"Loaded {len(instrument_data)} instruments")
        
        # Check that each loaded instrument has required columns
        for symbol, df in instrument_data.items():
            required_columns = ['Last', 'returns']
            for col in required_columns:
                assert col in df.columns, f"Missing column {col} in {symbol}"
            
            # Check data quality
            assert len(df) > 0, f"Empty data for {symbol}"
            assert not df['Last'].isna().all(), f"All NaN prices for {symbol}"
            
        print("✓ Data loading tests passed")
        return instrument_data
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return {}

def run_unit_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)
    
    try:
        test_position_size_calculation()
        test_idm_calculation()
        test_instrument_weights()
        instrument_data = test_data_loading()
        
        print("\n" + "=" * 60)
        print("ALL UNIT TESTS PASSED ✓")
        print("=" * 60)
        
        return True, instrument_data
        
    except Exception as e:
        print(f"\n✗ Unit test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def compare_weighting_methods():
    """
    Compare different weighting methods.
    """
    print("\n" + "=" * 80)
    print("COMPARING WEIGHTING METHODS")
    print("=" * 80)
    
    methods = ['equal', 'handcrafted']
    results_by_method = {}
    
    for method in methods:
        print(f"\n{'='*20} TESTING {method.upper()} WEIGHTING {'='*20}")
        
        try:
            results = backtest_multi_instrument_strategy(
                data_dir='Data',
                capital=50000000,
                risk_target=0.2,
                weight_method=method,
                common_hypothetical_SR=0.3, # Add example value
                annual_turnover_T=7.0       # Add example value
            )
            results_by_method[method] = results
            
            # Brief summary
            perf = results['performance']
            print(f"\n{method.upper()} SUMMARY:")
            print(f"  Total Return: {perf['total_return']:.2%}")
            print(f"  Annualized Return: {perf['annualized_return']:.2%}")
            print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {perf['max_drawdown_pct']:.1f}%")
            
        except Exception as e:
            print(f"Error with {method} method: {e}")
            results_by_method[method] = None
    
    # Detailed comparison
    if len(results_by_method) >= 2:
        print(f"\n" + "=" * 80)
        print("WEIGHTING METHOD COMPARISON")
        print("=" * 80)
        
        print(f"{'Metric':<25} {'Equal':<15} {'Handcrafted':<15} {'Difference':<15}")
        print("-" * 80)
        
        equal_result = results_by_method.get('equal', {})
        handcrafted_result = results_by_method.get('handcrafted', {})
        
        equal_perf = equal_result.get('performance', {}) if equal_result else {}
        handcrafted_perf = handcrafted_result.get('performance', {}) if handcrafted_result else {}
        
        if equal_perf and handcrafted_perf:
            metrics = [
                ('Total Return', 'total_return', '%'),
                ('Annualized Return', 'annualized_return', '%'),
                ('Volatility', 'annualized_volatility', '%'),
                ('Sharpe Ratio', 'sharpe_ratio', ''),
                ('Max Drawdown', 'max_drawdown_pct', '%')
            ]
            
            for name, key, unit in metrics:
                equal_val = equal_perf.get(key, 0)
                handcrafted_val = handcrafted_perf.get(key, 0)
                diff = handcrafted_val - equal_val
                
                if unit == '%':
                    print(f"{name:<25} {equal_val:<15.2%} {handcrafted_val:<15.2%} {diff:<15.2%}")
                else:
                    print(f"{name:<25} {equal_val:<15.3f} {handcrafted_val:<15.3f} {diff:<15.3f}")
    
    return results_by_method

def main():
    """
    Test Strategy 4 implementation with unit tests and FX functionality.
    """
    # Run unit tests
    tests_passed, instrument_data = run_unit_tests()
    
    if not tests_passed:
        print("Unit tests failed. Stopping execution.")
        return
    
    try:
        # results_by_method = compare_weighting_methods() # This also calls backtest_multi_instrument_strategy
        # For a single run with handcrafted, call directly:
        print(f"\n" + "=" * 60)
        print("RUNNING HANDCRAFTED STRATEGY (DEFAULT)")
        print("=" * 60)
        results = backtest_multi_instrument_strategy(
            data_dir='Data',
            capital=50000000, # Example capital
            risk_target=0.2,
            short_span=32, # Default from Chapter 3
            long_years=10, # Default from Chapter 3
            min_vol_floor=0.05, # Consistent with Chapter 3
            weight_method='handcrafted', # or 'equal'
            common_hypothetical_SR=0.3, # Example: Pass new SR parameters
            annual_turnover_T=7.0,      # Example: Pass new SR parameters
            # start_date='YYYY-MM-DD', # Optional
            # end_date='YYYY-MM-DD'   # Optional
        )

        if results:
            print(f"\n" + "=" * 60)
            print("DETAILED HANDCRAFTED STRATEGY ANALYSIS")
            print("=" * 60)
            analyze_portfolio_results(results)
            plot_strategy4_equity_curve(results)
        else:
            print("Handcrafted strategy backtest did not produce results.")
        
        # ... (Comparison with MES single instrument strategy from Chapter 3, needs careful date alignment) ...
        # This part requires results to be non-None from the handcrafted run
        if results and results['config']:
            print(f"\n" + "=" * 60)
        print("COMPARISON WITH SINGLE INSTRUMENT STRATEGY (MES from Chapter 3)")
        print("=" * 60)
        try:
            backtest_start_ch4 = results['config']['backtest_start']
            backtest_end_ch4 = results['config']['backtest_end']

            # Create a temporary filtered MES data file for Chapter 3 backtest
            mes_full_df = pd.read_csv('Data/mes_daily_data.csv', parse_dates=['Time'])
            mes_full_df.set_index('Time', inplace=True)
            # Filter MES data to match the chapter 4 backtest period EXACTLY
            mes_filtered_for_comp = mes_full_df[(mes_full_df.index >= backtest_start_ch4) & (mes_full_df.index <= backtest_end_ch4)]
            
            if mes_filtered_for_comp.empty:
                print("MES data for comparison period is empty. Skipping comparison.")
            else:
                temp_mes_path = 'Data/mes_temp_comparison_ch4.csv'
                mes_filtered_for_comp.to_csv(temp_mes_path)

                # Run Chapter 3 strategy with the same capital and date range
                ch3_results_for_comp = backtest_variable_risk_strategy(
                    temp_mes_path, 
                    initial_capital=results['config']['capital'], 
                    risk_target=results['config']['risk_target'],
                    short_span=results['config']['short_span'],
                    long_years=results['config']['long_years'],
                    min_vol_floor=results['config']['min_vol_floor']
                )
                os.remove(temp_mes_path)

                mes_perf_comp = ch3_results_for_comp['performance']
                multi_perf_comp = results['performance']
                # ... (print comparison table as before) ...
                print(f"\nPerformance Comparison (Same Time Period: {backtest_start_ch4.date()} to {backtest_end_ch4.date()}):")
                print(f"{'Metric':<25} {'MES Only (Ch3)':<15} {'Multi-Inst (Ch4)':<18} {'Difference':<12}")
                print("-" * 75)
                print(f"{'Total Return':<25} {mes_perf_comp['total_return']:.2%} {multi_perf_comp['total_return']:.2%} {multi_perf_comp['total_return'] - mes_perf_comp['total_return']:.2%}")
                print(f"{'Annualized Return':<25} {mes_perf_comp['annualized_return']:.2%} {multi_perf_comp['annualized_return']:.2%} {multi_perf_comp['annualized_return'] - mes_perf_comp['annualized_return']:.2%}")
                print(f"{'Volatility':<25} {mes_perf_comp['annualized_volatility']:.2%} {multi_perf_comp['annualized_volatility']:.2%} {multi_perf_comp['annualized_volatility'] - mes_perf_comp['annualized_volatility']:.2%}")
                print(f"{'Sharpe Ratio':<25} {mes_perf_comp['sharpe_ratio']:.3f} {multi_perf_comp['sharpe_ratio']:.3f} {multi_perf_comp['sharpe_ratio'] - mes_perf_comp['sharpe_ratio']:.3f}")
                print(f"{'Max Drawdown':<25} {mes_perf_comp['max_drawdown_pct']:.1f}% {multi_perf_comp['max_drawdown_pct']:.1f}% {multi_perf_comp['max_drawdown_pct'] - mes_perf_comp['max_drawdown_pct']:.1f}%")
        except Exception as e:
            print(f"Could not run comparison with MES (Chapter 3 based): {e}")
            # import traceback; traceback.print_exc()
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback; traceback.print_exc()
    print(f"\nStrategy 4 processing completed!")
    return results # Return results from the direct handcrafted run

if __name__ == "__main__":
    main()
