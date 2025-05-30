from chapter3 import *
from dynamic_optimization import *
import numpy as np
import pandas as pd
from itertools import combinations
import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

#####   DATA LOADING AND PREPARATION   #####

def validate_returns_data(returns_series, symbol):
    """
    Validate returns data for reasonable values.
    
    Parameters:
        returns_series (pd.Series): Daily returns series.
        symbol (str): Instrument symbol for reporting.
    
    Returns:
        pd.Series: Cleaned returns series.
    """
    # Remove extreme outliers (returns > 50% or < -50% daily)
    valid_mask = (returns_series.abs() <= 0.5)
    
    if valid_mask.sum() < len(returns_series) * 0.95:  # More than 5% outliers
        print(f"Warning: {symbol} has {(~valid_mask).sum()} extreme outliers ({(~valid_mask).mean():.1%})")
    
    return returns_series[valid_mask]

def get_available_instruments(instruments_df, data_directory='Data'):
    """
    Get list of instruments that have corresponding data files.
    
    Parameters:
        instruments_df (pd.DataFrame): Instruments data.
        data_directory (str): Directory containing data files.
    
    Returns:
        list: List of symbols with available data files.
    """
    available_instruments = []
    
    for _, instrument in instruments_df.iterrows():
        symbol = instrument['Symbol']
        filename = f"{symbol.lower()}_daily_data.csv"
        filepath = os.path.join(data_directory, filename)
        
        if os.path.exists(filepath):
            available_instruments.append(symbol)
    
    return available_instruments

def load_instrument_data_files(symbols, data_directory='Data', start_date='2000-01-01', end_date='2025-01-01'):
    """
    Load price data for multiple instruments.
    
    Parameters:
        symbols (list): List of instrument symbols.
        data_directory (str): Directory containing data files.
        start_date (str): Start date for data.
        end_date (str): End date for data.
    
    Returns:
        dict: Dictionary with symbol -> DataFrame mapping.
    """
    data = {}
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    for symbol in symbols:
        try:
            filename = f"{symbol.lower()}_daily_data.csv"
            filepath = os.path.join(data_directory, filename)
            
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, parse_dates=['Time'])
                df.set_index('Time', inplace=True)
                
                # Sort the index to make it monotonic
                df = df.sort_index()
                
                # Remove duplicates if any
                df = df[~df.index.duplicated(keep='first')]
                
                # Filter by date range using boolean indexing
                mask = (df.index >= start_dt) & (df.index <= end_dt)
                df = df[mask].dropna()
                
                if len(df) > 0:
                    df['returns'] = df['Last'].pct_change()
                    df = df.dropna()
                    
                    # Validate returns data
                    df['returns'] = validate_returns_data(df['returns'], symbol)
                    df = df.dropna()
                    
                    # Final check for sufficient data
                    if len(df) > 100:  # Need at least 100 days of data
                        data[symbol] = df
                    
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            continue
    
    return data

def create_returns_matrix(data):
    """
    Create aligned returns matrix from individual instrument data.
    
    Parameters:
        data (dict): Dictionary of instrument DataFrames.
    
    Returns:
        pd.DataFrame: Aligned returns matrix with instruments as columns.
    """
    returns_list = []
    
    for symbol, df in data.items():
        returns_series = df['returns'].copy()
        returns_series.name = symbol
        returns_list.append(returns_series)
    
    if returns_list:
        returns_matrix = pd.concat(returns_list, axis=1, join='inner')
        return returns_matrix.dropna()
    else:
        return pd.DataFrame()

#####   CORRELATION AND DIVERSIFICATION   #####

def calculate_correlation_matrix(returns_matrix):
    """
    Calculate correlation matrix from returns.
    
    Parameters:
        returns_matrix (pd.DataFrame): Returns matrix with instruments as columns.
    
    Returns:
        pd.DataFrame: Correlation matrix.
    """
    return returns_matrix.corr()

def calculate_idm_from_correlations(weights, correlation_matrix):
    """
    Calculate exact IDM from correlation matrix and weights.
    
    Formula: IDM = 1 / sqrt(w^T * Σ * w) where Σ is correlation matrix
    
    Parameters:
        weights (pd.Series or np.array): Portfolio weights.
        correlation_matrix (pd.DataFrame): Correlation matrix.
    
    Returns:
        float: Instrument Diversification Multiplier.
    """
    if isinstance(weights, dict):
        weights = pd.Series(weights)
    
    # Align weights with correlation matrix
    aligned_weights = weights.reindex(correlation_matrix.index).fillna(0)
    w = aligned_weights.values
    
    # Calculate portfolio variance: w^T * Σ * w
    portfolio_variance = np.dot(w, np.dot(correlation_matrix.values, w))
    
    # IDM = 1 / sqrt(portfolio_variance)
    idm = 1.0 / np.sqrt(portfolio_variance) if portfolio_variance > 0 else 1.0
    
    return idm

def estimate_portfolio_volatility(weights, volatilities, correlation_matrix):
    """
    Estimate portfolio volatility using weights, individual volatilities, and correlations.
    
    Parameters:
        weights (dict): Portfolio weights.
        volatilities (dict): Individual instrument volatilities.
        correlation_matrix (pd.DataFrame): Correlation matrix.
    
    Returns:
        float: Estimated portfolio volatility.
    """
    if isinstance(weights, dict):
        weights = pd.Series(weights)
    if isinstance(volatilities, dict):
        volatilities = pd.Series(volatilities)
    
    # Align all data
    instruments = correlation_matrix.index
    w = weights.reindex(instruments).fillna(0)
    vol = volatilities.reindex(instruments).fillna(0)
    
    # Create covariance matrix: Σ_cov = D * Σ_corr * D where D is diagonal of volatilities
    vol_matrix = np.outer(vol.values, vol.values)
    covariance_matrix = vol_matrix * correlation_matrix.values
    
    # Portfolio variance: w^T * Σ_cov * w
    portfolio_variance = np.dot(w.values, np.dot(covariance_matrix, w.values))
    
    return np.sqrt(portfolio_variance)

#####   PORTFOLIO CONSTRUCTION FUNCTIONS   #####

def calculate_position_size_with_idm(capital, weight, idm, multiplier, price, fx_rate, sigma_pct, risk_target=0.2, max_leverage=3.0, min_vol_floor=0.05):
    """
    Calculate position size for an instrument considering IDM.

    Parameters:
        capital (float): Total capital.
        weight (float): Target weight for the instrument.
        idm (float): Instrument diversification multiplier.
        multiplier (float): Contract multiplier.
        price (float): Current price of the instrument.
        fx_rate (float): FX rate (relevant for non-USD instruments, 1.0 for USD).
        sigma_pct (float): Volatility (as a percentage, e.g., 0.2 for 20%).
        risk_target (float): Overall portfolio risk target (e.g., 0.2 for 20% annual vol).
        max_leverage (float): Maximum leverage cap (3.0 = 3x gross notional).
        min_vol_floor (float): Minimum volatility floor (0.05 = 5% annually).

    Returns:
        float: Number of contracts (can be fractional).
    """
    # Apply volatility floor to prevent near-zero denominators
    sigma_pct = max(sigma_pct, min_vol_floor)
    
    # Ensure sigma_pct and price are valid positive numbers to avoid division by zero or nonsensical results
    if sigma_pct <= 0 or pd.isna(sigma_pct) or price <= 0 or pd.isna(price) or multiplier <= 0 or pd.isna(multiplier):
        # print(f"DEBUG calc_pos_size_idm: Invalid input for position calc. Price: {price}, Sigma: {sigma_pct}, Mult: {multiplier}. Returning 0 contracts.")
        return 0
    
    # Ensure other critical inputs are sensible
    if capital <= 0 or idm <= 0 or weight < 0 or risk_target <= 0 or fx_rate <= 0:
        # print(f"DEBUG calc_pos_size_idm: Invalid capital/idm/weight/risk_target/fx_rate. Returning 0 contracts.")
        return 0

    numerator = capital * idm * weight * risk_target
    denominator = multiplier * price * fx_rate * sigma_pct
    
    if denominator == 0: # Should be caught by above checks, but as a safeguard
        # print(f"DEBUG calc_pos_size_idm: Denominator is zero. Price: {price}, Sigma: {sigma_pct}, Mult: {multiplier}. Returning 0 contracts.")
        return 0
    
    position = numerator / denominator
    
    return position

def create_jumbo_portfolio(instruments_df, data, min_instruments=50, max_instruments=100):
    """
    Create a jumbo portfolio with many instruments as described in the book.
    
    Parameters:
        instruments_df (pd.DataFrame): Instruments data.
        data (dict): Individual instrument data.
        min_instruments (int): Minimum number of instruments.
        max_instruments (int): Maximum number of instruments.
    
    Returns:
        dict: Jumbo portfolio weights.
    """
    # Get all instruments with data
    available_instruments = list(data.keys())
    
    # Filter by cost efficiency (SR cost <= 0.01)
    cost_efficient = []
    for symbol in available_instruments:
        try:
            specs = get_instrument_specs(symbol, instruments_df)
            if specs['sr_cost'] <= 0.01:
                cost_efficient.append(symbol)
        except:
            continue
    
    # Select instruments up to max_instruments limit
    selected_instruments = cost_efficient[:max_instruments]
    
    if len(selected_instruments) < min_instruments:
        print(f"Warning: Only {len(selected_instruments)} instruments available (minimum {min_instruments} requested)")
    
    # Equal weights for all selected instruments
    if selected_instruments:
        weight = 1.0 / len(selected_instruments)
        return {symbol: weight for symbol in selected_instruments}
    else:
        return {}

def backtest_portfolio_with_individual_data(portfolio_weights, data, instruments_df, capital=50000000, 
                                           risk_target=0.2, start_date='2000-01-01', end_date='2025-01-01'):
    """
    Backtest portfolio strategy using individual instrument data with improved logic for date handling.
    
    Parameters:
        portfolio_weights (dict): Portfolio weights by instrument.
        data (dict): Individual instrument data.
        instruments_df (pd.DataFrame): Instruments specifications.
        capital (float): Initial capital.
        risk_target (float): Target risk fraction.
        start_date (str): Start date (used by data loading, backtest range derived from data).
        end_date (str): End date (used by data loading, backtest range derived from data).
    
    Returns:
        dict: Backtest results.
    """
    if not portfolio_weights:
        return {'error': 'No portfolio weights provided'}

    # 1. Filter portfolio_weights to those available in 'data' and normalize them
    active_portfolio_weights = {
        s: w for s, w in portfolio_weights.items() if s in data
    }
    if not active_portfolio_weights:
        return {'error': 'None of the instruments in portfolio_weights are available in loaded data'}
    
    total_active_weight = sum(active_portfolio_weights.values())
    if total_active_weight <= 0:
        return {'error': 'Total weight of active portfolio instruments is not positive.'}
    normalized_weights = {s: w / total_active_weight for s, w in active_portfolio_weights.items()}

    # 2. Determine correlation_matrix and IDM using common data (inner join)
    returns_for_idm_list = []
    for s_key in normalized_weights.keys():
        if 'returns' in data[s_key] and isinstance(data[s_key]['returns'], pd.Series):
            s_returns = data[s_key]['returns'].copy()
            s_returns.name = s_key
            returns_for_idm_list.append(s_returns)

    correlation_matrix = pd.DataFrame() # Default empty
    idm = 1.0  # Default IDM

    if not returns_for_idm_list:
        print("Warning: No returns series available for IDM calculation. Using IDM=1.0.")
    else:
        common_returns_matrix = pd.concat(returns_for_idm_list, axis=1, join='inner')
        common_returns_matrix.dropna(inplace=True) # Ensure no NaNs rows

        if common_returns_matrix.empty or common_returns_matrix.shape[0] < 2:
            print(f"Warning: Not enough common data (shape: {common_returns_matrix.shape}) to calculate IDM from correlations. Using IDM=1.0.")
            if list(normalized_weights.keys()): # Create dummy identity matrix if symbols exist
                 symbols_for_dummy_corr = list(normalized_weights.keys())
                 correlation_matrix = pd.DataFrame(np.eye(len(symbols_for_dummy_corr)), 
                                                  index=symbols_for_dummy_corr, columns=symbols_for_dummy_corr)
        elif common_returns_matrix.shape[1] == 1: # Single instrument
            idm = 1.0
            symbol = common_returns_matrix.columns[0]
            correlation_matrix = pd.DataFrame([[1.0]], index=[symbol], columns=[symbol])
            print(f"Note: Only one instrument ({symbol}) in common data for IDM. IDM set to 1.0.")
        else:
            try:
                # Calculate IDM using actual portfolio weights (not equal weights)
                correlation_matrix_calculated = calculate_correlation_matrix(common_returns_matrix)
                # Use actual portfolio weights for IDM calculation
                weights_for_idm_series = pd.Series(normalized_weights).reindex(correlation_matrix_calculated.index).fillna(0)
                # Re-normalize these weights if some instruments were dropped
                if weights_for_idm_series.sum() > 0:
                    weights_for_idm_series = weights_for_idm_series / weights_for_idm_series.sum()
                else:
                    # Fallback to equal weights if all weights are zero (shouldn't happen)
                    weights_for_idm_series = pd.Series({symbol: 1.0/len(normalized_weights) for symbol in normalized_weights.keys()})
                    weights_for_idm_series = weights_for_idm_series.reindex(correlation_matrix_calculated.index).fillna(0)
                    weights_for_idm_series = weights_for_idm_series / weights_for_idm_series.sum()
                
                idm_calculated = calculate_idm_from_correlations(weights_for_idm_series, correlation_matrix_calculated)
                idm = idm_calculated
                correlation_matrix = correlation_matrix_calculated
                print(f"Calculated IDM for no-trend strategy using portfolio weights: {idm:.2f}")
            except Exception as e_idm:
                print(f"Error calculating IDM from correlations: {e_idm}. Using IDM=1.0.")
                # correlation_matrix remains empty or becomes dummy if symbols_for_dummy_corr was set
                if list(normalized_weights.keys()) and correlation_matrix.empty:
                     symbols_for_dummy_corr = list(normalized_weights.keys())
                     correlation_matrix = pd.DataFrame(np.eye(len(symbols_for_dummy_corr)), 
                                                      index=symbols_for_dummy_corr, columns=symbols_for_dummy_corr)
    
    # 3. Determine `aligned_dates` for the main loop from an outer join of all active instruments
    all_individual_returns_for_loop = []
    for s_key in normalized_weights.keys(): # Use instruments confirmed to be in portfolio and data
        if 'returns' in data[s_key] and isinstance(data[s_key]['returns'], pd.Series):
            s_returns = data[s_key]['returns'].copy()
            s_returns.name = s_key
            all_individual_returns_for_loop.append(s_returns)

    if not all_individual_returns_for_loop:
        return {'error': 'No returns series available for outer join for backtest loop.'}
        
    outer_join_loop_matrix = pd.concat(all_individual_returns_for_loop, axis=1, join='outer')
    # Filter by overall start_date and end_date if specified, otherwise use full range from data
    # This ensures the loop doesn't go beyond the intended scope of the study period.
    # However, start_date/end_date in this function are currently for data loading guidance.
    # The effective range is determined by the data itself.
    aligned_dates = outer_join_loop_matrix.index.sort_values().drop_duplicates()
    
    # Filter aligned_dates by the function's start_date and end_date parameters
    # This is important if the user wants to constrain the backtest to a specific sub-period
    # of the loaded data.
    param_start_dt = pd.to_datetime(start_date)
    param_end_dt = pd.to_datetime(end_date)
    aligned_dates = aligned_dates[(aligned_dates >= param_start_dt) & (aligned_dates <= param_end_dt)]

    if aligned_dates.empty:
        return {'error': 'No dates available for backtest loop after filtering by start/end_date parameters.'}

    # Pre-calculate volatilities for each instrument over the full aligned_dates
    volatilities = {}
    vol_diagnostics = {}  # Track volatility statistics for debugging
    
    for symbol in normalized_weights.keys():
        symbol_data_df = data[symbol]
        symbol_returns = symbol_data_df['returns'] 
        # Reindex to ensure it covers all aligned_dates, ffill for missing vol at start
        blended_vol_series = calculate_blended_volatility(symbol_returns).reindex(aligned_dates, method='ffill')
        volatilities[symbol] = blended_vol_series
        
        # Track volatility statistics for debugging
        vol_stats = blended_vol_series.describe()
        vol_diagnostics[symbol] = {
            'min': vol_stats['min'],
            'max': vol_stats['max'],
            'mean': vol_stats['mean'],
            'count_below_5pct': (blended_vol_series < 0.05).sum(),
            'count_below_1pct': (blended_vol_series < 0.01).sum()
        }
    
    # Print volatility diagnostics
    print(f"\n=== VOLATILITY DIAGNOSTICS ===")
    for symbol, stats in vol_diagnostics.items():
        print(f"{symbol}: min={stats['min']:.6f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}, <5%={stats['count_below_5pct']}, <1%={stats['count_below_1pct']}")
    
    min_vol_across_all = min(stats['min'] for stats in vol_diagnostics.values())
    print(f"Minimum volatility across all instruments: {min_vol_across_all:.8e}")
    if min_vol_across_all < 0.01:
        print(f"WARNING: Extremely low volatilities detected! This will cause position size explosions.")
    
    # Initialize tracking arrays
    portfolio_returns = []
    positions_data = {symbol: [] for symbol in normalized_weights.keys()}
    position_diagnostics = {symbol: [] for symbol in normalized_weights.keys()}  # Track position sizes
    
    for i, date_val in enumerate(aligned_dates):
        if i == 0:
            for symbol in normalized_weights.keys():
                positions_data[symbol].append(0)
            portfolio_returns.append(0)
            continue
        
        prev_date = aligned_dates[i-1]
        current_date = date_val # Renamed from 'date' to avoid conflict with datetime module
        
        daily_portfolio_return = 0
        
        for symbol, weight in normalized_weights.items(): # Use normalized_weights
            try:
                specs = get_instrument_specs(symbol, instruments_df)
                multiplier = specs['multiplier']
                symbol_data_df = data[symbol] # Renamed from symbol_data to avoid conflict
                
                # Check if PREVIOUS date had price data for sizing, and CURRENT date has return data
                if prev_date in symbol_data_df.index and symbol_data_df.loc[prev_date, 'Last'] is not np.nan \
                   and current_date in symbol_data_df.index and symbol_data_df.loc[current_date, 'returns'] is not np.nan:
                    
                    prev_price = symbol_data_df.loc[prev_date, 'Last']
                    current_return = symbol_data_df.loc[current_date, 'returns']
                    
                    # Get pre-calculated blended volatility for prev_date
                    if prev_date in volatilities[symbol].index and not pd.isna(volatilities[symbol].loc[prev_date]):
                        blended_vol = volatilities[symbol].loc[prev_date]
                    else: # Fallback if somehow still NaN after ffill (e.g., very start of series)
                        # This fallback should be less needed with reindex and ffill
                        fallback_returns = symbol_data_df['returns'].loc[:prev_date]
                        if len(fallback_returns) >= 22: # check for sufficient data for rolling
                           blended_vol = fallback_returns.rolling(22).std().iloc[-1] * np.sqrt(business_days_per_year)
                        elif len(fallback_returns) > 0: # use std of what's available if less than 22
                           blended_vol = fallback_returns.std() * np.sqrt(business_days_per_year)
                        else: # no returns to calculate vol
                           blended_vol = np.nan # or some default high vol to prevent large positions
                           
                    position = 0
                    if not pd.isna(blended_vol) and blended_vol > 0 and not pd.isna(prev_price) and prev_price > 0:
                        position = calculate_position_size_with_idm(
                            capital, weight, idm, multiplier, prev_price, 1.0, blended_vol, risk_target
                        )
                    
                    positions_data[symbol].append(position)
                    position_diagnostics[symbol].append(position)  # Track for diagnostics
                    
                    if not pd.isna(current_return) and position != 0:
                        specs = get_instrument_specs(symbol, instruments_df)
                        multiplier = specs['multiplier']
                        
                        # Calculate P&L: position * multiplier * percentage_return * previous_price
                        if prev_date in symbol_data_df.index:
                            prev_price = symbol_data_df.loc[prev_date, 'Last']
                            if not pd.isna(prev_price):
                                # current_return is already a percentage, so multiply by notional to get P&L
                                notional_exposure = position * multiplier * prev_price
                                instrument_pnl = notional_exposure * current_return
                                instrument_return_contrib = instrument_pnl / capital
                                
                                # Check for overflow/extreme values before adding to portfolio return
                                if np.isinf(instrument_return_contrib) or np.isnan(instrument_return_contrib):
                                    print(f"OVERFLOW DETECTED on {current_date}: {symbol} pos={position} mult={multiplier} price={prev_price:.2f} return={current_return:.6f}")
                                    print(f"  notional={notional_exposure:.2e} pnl={instrument_pnl:.2e} contrib={instrument_return_contrib}")
                                    raise ValueError(f"Overflow detected in P&L calculation for {symbol} on {current_date}")
                                
                                # Debug output for extreme values (>50% daily return is suspicious)
                                if abs(instrument_return_contrib) > 0.50:
                                    print(f"EXTREME RETURN {current_date}: {symbol} pos={position} mult={multiplier} price={prev_price:.2f} return={current_return:.6f}")
                                    print(f"  notional={notional_exposure:.2e} pnl={instrument_pnl:.2e} contrib={instrument_return_contrib:.6f}")
                                
                                # Debug output for large positions (previous threshold was 10%)
                                if abs(instrument_return_contrib) > 0.1:  # Return > 10% in a day
                                    print(f"DEBUG PnL: {symbol} pos={position} mult={multiplier} price={prev_price:.2f} return={current_return:.4f} pnl={instrument_pnl:.2f} contrib={instrument_return_contrib:.4f}")
                                
                                daily_portfolio_return += instrument_return_contrib
                else:
                    positions_data[symbol].append(0) # No position or no data to calc P&L
                    
            except Exception as e:
                # print(f"Error processing {symbol} on {current_date}: {e}") # Optional: for debugging
                positions_data[symbol].append(0)
        
        portfolio_returns.append(daily_portfolio_return)
    
    # Create results dataframe
    results_df = pd.DataFrame(index=aligned_dates) # Use the potentially longer aligned_dates
    # Trim initial zero P&L day for correct performance calc if it's the only entry
    if len(portfolio_returns) == len(aligned_dates) and len(aligned_dates) > 1:
        results_df['portfolio_returns'] = portfolio_returns
        for symbol, pos_list in positions_data.items():
             if len(pos_list) == len(aligned_dates):
                results_df[f'position_{symbol}'] = pos_list
    elif len(aligned_dates) == 1 and len(portfolio_returns) ==1 : # Single date case
        results_df['portfolio_returns'] = portfolio_returns
        for symbol, pos_list in positions_data.items():
             if len(pos_list) == len(aligned_dates):
                results_df[f'position_{symbol}'] = pos_list
    else: # Mismatch, indicates an issue
        return {'error': f'Length mismatch between aligned_dates ({len(aligned_dates)}) and results.'}


    # Drop initial rows if they are all zeros until first valid P&L or position
    # Find the first row where portfolio_returns is not zero or any position is not zero
    first_activity_index = results_df[(results_df['portfolio_returns'] != 0) | \
                                      results_df.filter(like='position_').ne(0).any(axis=1)].index.min()
    if pd.notna(first_activity_index):
        results_df = results_df.loc[first_activity_index:]
    else: # All zero returns and positions
        results_df = results_df.iloc[1:] # Keep at least one row if all are zero, but drop first for perf calc

    if results_df.empty: # If all data was before start_date or after end_date or only one row of zeros
         return {'error': 'No valid backtest data after date filtering and initial zero removal'}
    
    # Print position diagnostics summary
    print(f"\n=== POSITION SIZE DIAGNOSTICS ===")
    for symbol in normalized_weights.keys():
        if position_diagnostics[symbol]:
            pos_series = pd.Series(position_diagnostics[symbol])
            pos_stats = pos_series.describe()
            max_notional = abs(pos_stats['max']) * get_instrument_specs(symbol, instruments_df)['multiplier'] * data[symbol]['Last'].iloc[-1] if len(data[symbol]) > 0 else 0
            print(f"{symbol}: min={pos_stats['min']:.2f}, max={pos_stats['max']:.2f}, mean={pos_stats['mean']:.2f}, max_notional=${max_notional:.0f}")
            
            # Check for extreme positions
            if abs(pos_stats['max']) > 1000:
                print(f"  WARNING: Extremely large position detected for {symbol}!")
    
    # Calculate performance metrics
    equity_curve_series = build_account_curve(results_df['portfolio_returns'], capital)
    
    performance = calculate_comprehensive_performance(
        equity_curve_series,
        results_df['portfolio_returns']
    )
    
    performance['num_instruments'] = len(normalized_weights)
    performance['idm'] = idm
    # Storing the full correlation matrix might be too large for many instruments.
    # Storing a summary or path to it might be better if needed. For now, as is.
    performance['correlation_matrix'] = correlation_matrix 
    performance['available_weights'] = normalized_weights # Store the actual weights used
    
    return {
        'data': results_df,
        'performance': performance,
        'portfolio_weights': normalized_weights,
        'correlation_matrix': correlation_matrix,
        'idm': idm
    }

#####   DYNAMIC OPTIMIZATION BACKTESTING   #####

def backtest_portfolio_with_dynamic_optimization(portfolio_weights, data, instruments_df, capital=50000000, 
                                                risk_target=0.2, start_date='2000-01-01', end_date='2025-01-01',
                                                cost_multiplier=50, use_buffering=True, buffer_fraction=0.05,
                                                rebalance_frequency='daily'):
    """
    Backtest portfolio strategy using dynamic optimization for position sizing.
    
    This implements Chapter 25's dynamic optimization strategy which optimizes positions
    daily using the greedy algorithm to minimize tracking error while accounting for costs.
    
    Parameters:
        portfolio_weights (dict): Portfolio weights by instrument.
        data (dict): Individual instrument data.
        instruments_df (pd.DataFrame): Instruments specifications.
        capital (float): Initial capital.
        risk_target (float): Target risk fraction.
        start_date (str): Start date for backtest.
        end_date (str): End date for backtest.
        cost_multiplier (float): Cost penalty multiplier for optimization.
        use_buffering (bool): Whether to use buffering to reduce turnover.
        buffer_fraction (float): Buffer fraction for tracking error.
        rebalance_frequency (str): How often to reoptimize ('daily', 'weekly', 'monthly').
    
    Returns:
        dict: Backtest results with dynamic optimization metrics.
    """
    if not portfolio_weights:
        return {'error': 'No portfolio weights provided'}

    # 1. Filter and normalize portfolio weights
    active_portfolio_weights = {
        s: w for s, w in portfolio_weights.items() if s in data
    }
    if not active_portfolio_weights:
        return {'error': 'None of the instruments in portfolio_weights are available in loaded data'}
    
    total_active_weight = sum(active_portfolio_weights.values())
    if total_active_weight <= 0:
        return {'error': 'Total weight of active portfolio instruments is not positive.'}
    normalized_weights = {s: w / total_active_weight for s, w in active_portfolio_weights.items()}

    # 2. Create returns matrix for covariance calculation
    returns_for_optimization = []
    for s_key in normalized_weights.keys():
        if 'returns' in data[s_key] and isinstance(data[s_key]['returns'], pd.Series):
            s_returns = data[s_key]['returns'].copy()
            s_returns.name = s_key
            returns_for_optimization.append(s_returns)

    if not returns_for_optimization:
        return {'error': 'No returns series available for dynamic optimization.'}
        
    # Use outer join for maximum date coverage
    returns_matrix = pd.concat(returns_for_optimization, axis=1, join='outer')
    
    # Filter by date range
    param_start_dt = pd.to_datetime(start_date)
    param_end_dt = pd.to_datetime(end_date)
    returns_matrix = returns_matrix[(returns_matrix.index >= param_start_dt) & 
                                   (returns_matrix.index <= param_end_dt)]
    
    if returns_matrix.empty:
        return {'error': 'No data available for optimization after date filtering.'}

    # 3. Determine rebalancing dates
    if rebalance_frequency == 'daily':
        rebalance_dates = returns_matrix.index
    elif rebalance_frequency == 'weekly':
        # Use business days that are closest to weekly intervals
        weekly_markers = returns_matrix.resample('W').last().index
        rebalance_dates = []
        for week_end in weekly_markers:
            # Find the actual business day closest to this week end
            available_dates = returns_matrix.index
            closest_date = available_dates[available_dates <= week_end]
            if len(closest_date) > 0:
                rebalance_dates.append(closest_date[-1])
        rebalance_dates = pd.Index(rebalance_dates).drop_duplicates()
    elif rebalance_frequency == 'monthly':
        # Use business days that are closest to monthly intervals  
        monthly_markers = returns_matrix.resample('M').last().index
        rebalance_dates = []
        for month_end in monthly_markers:
            # Find the actual business day closest to this month end
            available_dates = returns_matrix.index
            closest_date = available_dates[available_dates <= month_end]
            if len(closest_date) > 0:
                rebalance_dates.append(closest_date[-1])
        rebalance_dates = pd.Index(rebalance_dates).drop_duplicates()
    else:
        rebalance_dates = returns_matrix.index  # Default to daily

    # 4. Pre-calculate volatilities for all dates
    volatilities = {}
    for symbol in normalized_weights.keys():
        symbol_data_df = data[symbol]
        symbol_returns = symbol_data_df['returns'] 
        blended_vol_series = calculate_blended_volatility(symbol_returns).reindex(returns_matrix.index, method='ffill')
        volatilities[symbol] = blended_vol_series
    
    # 5. Initialize tracking variables
    portfolio_returns = []
    positions_data = {symbol: [] for symbol in normalized_weights.keys()}
    position_diagnostics = {symbol: [] for symbol in normalized_weights.keys()}  # Track position sizes for diagnostics
    current_positions = {symbol: 0 for symbol in normalized_weights.keys()}
    optimization_metrics = []
    
    # Track dynamic optimization specific metrics
    tracking_errors = []
    adjustment_factors = []
    turnover_data = []
    
    for i, date_val in enumerate(returns_matrix.index):
        if i == 0:
            # Initialize first day
            for symbol in normalized_weights.keys():
                positions_data[symbol].append(0)
            portfolio_returns.append(0)
            tracking_errors.append(0)
            adjustment_factors.append(1.0)
            turnover_data.append(0)
            continue
        
        prev_date = returns_matrix.index[i-1]
        current_date = date_val
        
        # Check if we should reoptimize positions
        if current_date in rebalance_dates:
            # Prepare instrument data for optimization
            instruments_data = {}
            for symbol in normalized_weights.keys():
                if prev_date in data[symbol].index and current_date in data[symbol].index:
                    try:
                        specs = get_instrument_specs(symbol, instruments_df)
                        prev_price = data[symbol].loc[prev_date, 'Last']
                        
                        # Get volatility forecast
                        if prev_date in volatilities[symbol].index:
                            vol_forecast = volatilities[symbol].loc[prev_date]
                        else:
                            vol_forecast = 0.16  # fallback
                        
                        if not pd.isna(prev_price) and not pd.isna(vol_forecast) and vol_forecast > 0:
                            instruments_data[symbol] = {
                                'price': prev_price,
                                'volatility': vol_forecast,
                                'specs': specs
                            }
                    except Exception as e:
                        continue
            
            # Run dynamic optimization if we have sufficient data
            if len(instruments_data) >= 2:  # Need at least 2 instruments for correlation
                try:
                    # Get recent returns for covariance calculation (last 252 days)
                    recent_returns = returns_matrix.loc[:prev_date].tail(252)
                    if len(recent_returns) < 50:  # Need minimum data
                        recent_returns = returns_matrix.loc[:prev_date]
                    
                    # Filter to available instruments
                    available_instruments = list(instruments_data.keys())
                    recent_returns_filtered = recent_returns[available_instruments].dropna()
                    
                    if len(recent_returns_filtered) > 20:
                        optimization_result = calculate_dynamic_portfolio_positions(
                            instruments_data, capital, current_positions, 
                            normalized_weights, recent_returns_filtered,
                            risk_target, cost_multiplier, use_buffering, buffer_fraction
                        )
                        
                        # Update positions from optimization
                        new_positions = optimization_result['positions']
                        
                        # Calculate turnover
                        daily_turnover = sum(abs(new_positions.get(s, 0) - current_positions.get(s, 0)) 
                                           for s in normalized_weights.keys())
                        turnover_data.append(daily_turnover)
                        
                        # Update current positions
                        for symbol in normalized_weights.keys():
                            current_positions[symbol] = new_positions.get(symbol, 0)
                        
                        # Store optimization metrics
                        tracking_errors.append(optimization_result.get('tracking_error', 0))
                        adjustment_factors.append(optimization_result.get('adjustment_factor', 1.0))
                        
                    else:
                        # Fallback to simple position sizing
                        for symbol in normalized_weights.keys():
                            if symbol in instruments_data:
                                data_item = instruments_data[symbol]
                                position = calculate_position_size_with_idm(
                                    capital, normalized_weights[symbol], 1.0, 
                                    data_item['specs']['multiplier'], 
                                    data_item['price'], 1.0, data_item['volatility'], risk_target
                                )
                                current_positions[symbol] = round(position)
                        
                        tracking_errors.append(0)
                        adjustment_factors.append(1.0)
                        turnover_data.append(0)
                        
                except Exception as e:
                    # Fallback on optimization failure
                    tracking_errors.append(0)
                    adjustment_factors.append(1.0)
                    turnover_data.append(0)
            else:
                # Not enough instruments for correlation-based optimization
                tracking_errors.append(0)
                adjustment_factors.append(1.0)
                turnover_data.append(0)
        else:
            # No rebalancing, keep same metrics
            tracking_errors.append(tracking_errors[-1] if tracking_errors else 0)
            adjustment_factors.append(adjustment_factors[-1] if adjustment_factors else 1.0)
            turnover_data.append(0)
        
        # Calculate daily P&L with current positions (MOVED OUTSIDE REBALANCING CHECK)
        daily_portfolio_return = 0
        
        for symbol in normalized_weights.keys():
            try:
                symbol_data_df = data[symbol]
                
                # Check if we have return data for current date
                if current_date in symbol_data_df.index:
                    current_return = symbol_data_df.loc[current_date, 'returns']
                    position = current_positions.get(symbol, 0)
                    
                    if not pd.isna(current_return) and position != 0:
                        specs = get_instrument_specs(symbol, instruments_df)
                        multiplier = specs['multiplier']
                        
                        # Calculate P&L: position * multiplier * percentage_return * previous_price
                        if prev_date in symbol_data_df.index:
                            prev_price = symbol_data_df.loc[prev_date, 'Last']
                            if not pd.isna(prev_price):
                                # current_return is already a percentage, so multiply by notional to get P&L
                                notional_exposure = position * multiplier * prev_price
                                instrument_pnl = notional_exposure * current_return
                                instrument_return_contrib = instrument_pnl / capital
                                
                                # Check for overflow/extreme values before adding to portfolio return
                                if np.isinf(instrument_return_contrib) or np.isnan(instrument_return_contrib):
                                    print(f"DYNAMIC OVERFLOW DETECTED on {current_date}: {symbol} pos={position} mult={multiplier} price={prev_price:.2f} return={current_return:.6f}")
                                    print(f"  notional={notional_exposure:.2e} pnl={instrument_pnl:.2e} contrib={instrument_return_contrib}")
                                    raise ValueError(f"Dynamic optimization overflow detected in P&L calculation for {symbol} on {current_date}")
                                
                                # Debug output for extreme values (>50% daily return is suspicious)
                                if abs(instrument_return_contrib) > 0.50:
                                    print(f"DYNAMIC EXTREME RETURN {current_date}: {symbol} pos={position} mult={multiplier} price={prev_price:.2f} return={current_return:.6f}")
                                    print(f"  notional={notional_exposure:.2e} pnl={instrument_pnl:.2e} contrib={instrument_return_contrib:.6f}")
                                
                                # Debug output for problematic values
                                if abs(instrument_return_contrib) > 0.1:  # Return > 10% in a day
                                    print(f"DEBUG PnL: {symbol} pos={position} mult={multiplier} price={prev_price:.2f} return={current_return:.4f} pnl={instrument_pnl:.2f} contrib={instrument_return_contrib:.4f}")
                                
                                daily_portfolio_return += instrument_return_contrib
                
                # Store position data
                positions_data[symbol].append(current_positions.get(symbol, 0))
                position_diagnostics[symbol].append(current_positions.get(symbol, 0))  # Track for diagnostics
                    
            except Exception as e:
                positions_data[symbol].append(0)
                position_diagnostics[symbol].append(0)  # Track zeros for diagnostics too
        
        portfolio_returns.append(daily_portfolio_return)
    
    # Create results dataframe
    results_df = pd.DataFrame(index=returns_matrix.index)
    results_df['portfolio_returns'] = portfolio_returns
    
    for symbol, pos_list in positions_data.items():
        if len(pos_list) == len(results_df):
            results_df[f'position_{symbol}'] = pos_list
    
    # Add dynamic optimization specific columns
    results_df['tracking_error'] = tracking_errors
    results_df['adjustment_factor'] = adjustment_factors
    results_df['daily_turnover'] = turnover_data
    
    # Remove initial zero return day if needed
    if len(results_df) > 1 and results_df['portfolio_returns'].iloc[0] == 0:
        results_df = results_df.iloc[1:]
    
    if results_df.empty:
        return {'error': 'No valid backtest data after processing'}
    
    # Print position diagnostics summary
    print(f"\n=== POSITION SIZE DIAGNOSTICS ===")
    for symbol in normalized_weights.keys():
        if position_diagnostics[symbol]:
            pos_series = pd.Series(position_diagnostics[symbol])
            pos_stats = pos_series.describe()
            max_notional = abs(pos_stats['max']) * get_instrument_specs(symbol, instruments_df)['multiplier'] * data[symbol]['Last'].iloc[-1] if len(data[symbol]) > 0 else 0
            print(f"{symbol}: min={pos_stats['min']:.2f}, max={pos_stats['max']:.2f}, mean={pos_stats['mean']:.2f}, max_notional=${max_notional:.0f}")
            
            # Check for extreme positions
            if abs(pos_stats['max']) > 1000:
                print(f"  WARNING: Extremely large position detected for {symbol}!")
    
    # Calculate performance metrics
    equity_curve_series = build_account_curve(results_df['portfolio_returns'], capital)
    
    # Debug: Check for problematic values in dynamic optimization
    print(f"DEBUG: Portfolio returns stats - Min: {results_df['portfolio_returns'].min():.6f}, Max: {results_df['portfolio_returns'].max():.6f}")
    print(f"DEBUG: Portfolio returns NaN count: {results_df['portfolio_returns'].isna().sum()}")
    print(f"DEBUG: Portfolio returns inf count: {np.isinf(results_df['portfolio_returns']).sum()}")
    print(f"DEBUG: Final positions: {current_positions}")
    
    performance = calculate_comprehensive_performance(
        equity_curve_series,
        results_df['portfolio_returns']
    )
    
    # Add dynamic optimization specific metrics
    performance['num_instruments'] = len(normalized_weights)
    performance['avg_tracking_error'] = results_df['tracking_error'].mean()
    performance['avg_adjustment_factor'] = results_df['adjustment_factor'].mean()
    performance['total_turnover'] = results_df['daily_turnover'].sum()
    performance['avg_daily_turnover'] = results_df['daily_turnover'].mean()
    performance['annual_turnover'] = performance['avg_daily_turnover'] * business_days_per_year
    performance['rebalance_frequency'] = rebalance_frequency
    performance['cost_multiplier'] = cost_multiplier
    performance['use_buffering'] = use_buffering
    
    return {
        'data': results_df,
        'performance': performance,
        'portfolio_weights': normalized_weights,
        'optimization_type': 'dynamic',
        'final_positions': current_positions
    }

#####   JUMBO PORTFOLIO FUNCTIONS   #####

def test_jumbo_portfolio(instruments_df, data, capital=50000000):
    """
    Test the jumbo portfolio strategy as described in the book.
    
    Parameters:
        instruments_df (pd.DataFrame): Instruments data.
        data (dict): Individual instrument data.
        capital (float): Initial capital.
    
    Returns:
        dict: Jumbo portfolio backtest results.
    """
    # Create jumbo portfolio
    jumbo_weights = create_jumbo_portfolio(instruments_df, data)
    
    if not jumbo_weights:
        return None
    
    # Backtest the jumbo portfolio
    results = backtest_portfolio_with_individual_data(
        jumbo_weights, data, instruments_df, capital
    )
    
    if 'error' in results:
        return None
    
    return results

def run_portfolio_comparison(capital=50000000, risk_target=0.2, max_instruments=100):
    """
    Run jumbo portfolio strategy using real data.
    
    Parameters:
        capital (float): Initial capital.
        risk_target (float): Target risk fraction.
        max_instruments (int): Maximum instruments for jumbo portfolio.
    
    Returns:
        dict: Portfolio results.
    """
    # Load instruments data
    instruments_df = load_instrument_data()
    
    # Get available instruments with data files
    available_instruments = get_available_instruments(instruments_df)
    
    if len(available_instruments) == 0:
        return {}
    
    # Load data for available instruments
    data = load_instrument_data_files(available_instruments)
    
    if len(data) == 0:
        return {}
    
    # Create jumbo portfolio strategy
    jumbo_weights = create_jumbo_portfolio(instruments_df, data, max_instruments=max_instruments)
    
    strategies = {}
    if jumbo_weights:
        strategies['Jumbo Portfolio'] = jumbo_weights
    
    return strategies, data, instruments_df, {}

#####   UTILITY FUNCTIONS FOR FUTURE CHAPTERS   #####

def get_instrument_universe_stats(instruments_df, data):
    """
    Get comprehensive statistics about the instrument universe.
    
    Parameters:
        instruments_df (pd.DataFrame): Instruments data.
        data (dict): Individual instrument data.
    
    Returns:
        dict: Universe statistics.
    """
    stats = {
        'total_instruments_available': len(data),
        'asset_classes': {},
        'cost_distribution': {},
        'data_quality': {}
    }
    
    # Asset class distribution
    asset_classes = create_asset_class_groups(instruments_df, list(data.keys()))
    stats['asset_classes'] = {k: len(v) for k, v in asset_classes.items()}
    
    # Cost distribution
    costs = []
    for symbol in data.keys():
        try:
            specs = get_instrument_specs(symbol, instruments_df)
            costs.append(specs['sr_cost'])
        except:
            continue
    
    if costs:
        stats['cost_distribution'] = {
            'mean': np.mean(costs),
            'median': np.median(costs),
            'min': np.min(costs),
            'max': np.max(costs),
            'std': np.std(costs)
        }
    
    # Data quality metrics
    data_lengths = [len(df) for df in data.values()]
    stats['data_quality'] = {
        'avg_data_points': np.mean(data_lengths),
        'min_data_points': np.min(data_lengths),
        'max_data_points': np.max(data_lengths),
        'instruments_with_full_data': sum(1 for length in data_lengths if length > 6000)  # ~24 years
    }
    
    return stats

def calculate_portfolio_risk_decomposition(weights, correlation_matrix, volatilities):
    """
    Calculate risk decomposition for portfolio analysis.
    
    Parameters:
        weights (dict): Portfolio weights.
        correlation_matrix (pd.DataFrame): Correlation matrix.
        volatilities (dict): Individual volatilities.
    
    Returns:
        dict: Risk decomposition analysis.
    """
    if isinstance(weights, dict):
        weights = pd.Series(weights)
    if isinstance(volatilities, dict):
        volatilities = pd.Series(volatilities)
    
    # Align data
    instruments = correlation_matrix.index
    w = weights.reindex(instruments).fillna(0)
    vol = volatilities.reindex(instruments).fillna(0)
    
    # Calculate portfolio variance
    portfolio_vol = estimate_portfolio_volatility(weights, volatilities, correlation_matrix)
    
    # Calculate marginal contributions to risk
    covar_matrix = np.outer(vol.values, vol.values) * correlation_matrix.values
    marginal_contrib = np.dot(covar_matrix, w.values) / portfolio_vol if portfolio_vol > 0 else np.zeros_like(w.values)
    
    # Calculate component contributions to risk
    component_contrib = w.values * marginal_contrib / portfolio_vol if portfolio_vol > 0 else np.zeros_like(w.values)
    
    return {
        'portfolio_volatility': portfolio_vol,
        'marginal_contributions': pd.Series(marginal_contrib, index=instruments),
        'component_contributions': pd.Series(component_contrib, index=instruments),
        'risk_concentration': max(component_contrib) if len(component_contrib) > 0 else 0
    }

def create_custom_portfolio(instrument_weights, instruments_df, data):
    """
    Create a custom portfolio with specified weights - utility for future chapters.
    
    Parameters:
        instrument_weights (dict): Dictionary of symbol: weight pairs.
        instruments_df (pd.DataFrame): Instruments data.
        data (dict): Individual instrument data.
    
    Returns:
        dict: Portfolio analysis results.
    """
    # Validate that all instruments exist
    available_weights = {k: v for k, v in instrument_weights.items() if k in data}
    
    if not available_weights:
        return {'error': 'No valid instruments in portfolio'}
    
    # Normalize weights
    total_weight = sum(available_weights.values())
    if total_weight > 0:
        available_weights = {k: v/total_weight for k, v in available_weights.items()}
    
    # Calculate correlation matrix
    returns_matrix = create_returns_matrix({k: data[k] for k in available_weights.keys()})
    if returns_matrix.empty:
        return {'error': 'No returns data for portfolio'}
    
    correlation_matrix = calculate_correlation_matrix(returns_matrix)
    idm = calculate_idm_from_correlations(available_weights, correlation_matrix)
    
    # Calculate individual volatilities
    volatilities = {}
    for symbol in available_weights.keys():
        symbol_returns = data[symbol]['returns']
        volatilities[symbol] = symbol_returns.std() * np.sqrt(business_days_per_year)
    
    # Risk decomposition
    risk_decomp = calculate_portfolio_risk_decomposition(available_weights, correlation_matrix, volatilities)
    
    return {
        'weights': available_weights,
        'correlation_matrix': correlation_matrix,
        'idm': idm,
        'risk_decomposition': risk_decomp,
        'individual_volatilities': volatilities
    }

def calculate_strategy_turnover(positions_df, multipliers_dict):
    """
    Calculate portfolio turnover for transaction cost analysis.
    
    Parameters:
        positions_df (pd.DataFrame): DataFrame with position columns.
        multipliers_dict (dict): Dictionary of symbol: multiplier pairs.
    
    Returns:
        dict: Turnover statistics.
    """
    position_cols = [col for col in positions_df.columns if col.startswith('position_')]
    
    if len(position_cols) == 0:
        return {'daily_turnover': 0, 'annual_turnover': 0}
    
    # Calculate daily position changes
    daily_turnover = 0
    
    for col in position_cols:
        symbol = col.replace('position_', '')
        if symbol in multipliers_dict:
            position_changes = positions_df[col].diff().abs()
            daily_turnover += position_changes.mean()
    
    annual_turnover = daily_turnover * business_days_per_year
    
    return {
        'daily_turnover': daily_turnover,
        'annual_turnover': annual_turnover,
        'avg_daily_trades': len(position_cols) * daily_turnover
    }

def main():
    """
    Run jumbo portfolio strategy only.
    """
    print("=" * 70)
    print("CHAPTER 4: JUMBO PORTFOLIO STRATEGY")
    print("=" * 70)
    
    # Run portfolio comparison to get the setup
    try:
        strategies, data, instruments_df, _ = run_portfolio_comparison(capital=50000000)
        
        if not strategies:
            print("No strategies to test!")
            return
        
        # Only run Jumbo Portfolio strategy
        print(f"\n----- Testing Jumbo Portfolio Strategy -----")
        
        if 'Jumbo Portfolio' in strategies:
            strategy_name = 'Jumbo Portfolio'
            weights = strategies[strategy_name]
            
            print(f"\n--- {strategy_name} ---")
            print(f"Number of instruments being traded: {len(weights)}")
            print(f"Instruments: {list(weights.keys())}")
            
            # Backtest the strategy
            result = backtest_portfolio_with_individual_data(
                weights, data, instruments_df, capital=50000000
            )
            
            if 'error' in result:
                print(f"Error: {result['error']}")
                return
            
            perf = result['performance']
            
            print(f"\nPerformance:")
            print(f"  Annual Return: {perf['annualized_return']:.2%}")
            print(f"  Volatility: {perf['annualized_volatility']:.2%}")
            print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {perf['max_drawdown_pct']:.1f}%")
            print(f"  IDM: {result['idm']:.2f}")
        else:
            print("Jumbo Portfolio strategy not available!")
        
    except Exception as e:
        print(f"Error in portfolio analysis: {e}")

if __name__ == "__main__":
    main()

