from chapter3 import *
import numpy as np
import pandas as pd
from itertools import combinations
import os
import warnings
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

def calculate_position_size_with_idm(capital, weight, idm, multiplier, price, fx_rate, sigma_pct, risk_target=0.2):
    """
    Calculate position size for an instrument in a portfolio with IDM.
    
    Formula:
        N_i = (Capital × IDM × Weight_i × τ) ÷ (Multiplier_i × Price_i × FX_i × σ_i)
    
    Parameters:
        capital (float): Total capital.
        weight (float): Instrument weight in portfolio.
        idm (float): Instrument diversification multiplier.
        multiplier (float): Contract multiplier.
        price (float): Current price.
        fx_rate (float): FX rate.
        sigma_pct (float): Volatility forecast.
        risk_target (float): Target risk fraction.
    
    Returns:
        float: Position size in contracts.
    """
    if sigma_pct <= 0 or np.isnan(sigma_pct):
        return 0
    
    numerator = capital * idm * weight * risk_target
    denominator = multiplier * price * fx_rate * sigma_pct
    
    return numerator / denominator

def select_instruments_by_criteria(instruments_df, available_instruments, capital, 
                                 max_cost_sr=0.01, min_volume_usd=1000000):
    """
    Select suitable instruments based on cost, liquidity, and data availability.
    
    Parameters:
        instruments_df (pd.DataFrame): Instruments data.
        available_instruments (list): Instruments with available data.
        capital (float): Available capital.
        max_cost_sr (float): Maximum acceptable SR cost.
        min_volume_usd (float): Minimum daily volume requirement.
    
    Returns:
        list: List of suitable instrument symbols.
    """
    suitable_instruments = []
    
    for symbol in available_instruments:
        try:
            instrument = instruments_df[instruments_df['Symbol'] == symbol].iloc[0]
            sr_cost = instrument['SR_cost']
            
            # Skip instruments with missing SR cost
            if pd.isna(sr_cost):
                continue
                
            # Check cost criterion
            if sr_cost <= max_cost_sr:
                suitable_instruments.append(symbol)
                
        except Exception:
            continue
    
    return suitable_instruments

def create_asset_class_groups(instruments_df, suitable_instruments):
    """
    Create asset class groupings based on instrument names and characteristics.
    
    Parameters:
        instruments_df (pd.DataFrame): Instruments data.
        suitable_instruments (list): List of suitable instrument symbols.
    
    Returns:
        dict: Asset class groupings.
    """
    asset_classes = {
        'equity': [],
        'bonds': [],
        'commodities': [],
        'fx': [],
        'volatility': []
    }
    
    for symbol in suitable_instruments:
        try:
            instrument = instruments_df[instruments_df['Symbol'] == symbol].iloc[0]
            name = instrument['Name'].lower()
            
            # Classify instruments based on name patterns
            if any(term in name for term in ['treasury', 'bond', 'note', 'bund', 'btp', 'schatz', 'bobl']):
                asset_classes['bonds'].append(symbol)
            elif any(term in name for term in ['s&p', 'dow', 'nasdaq', 'russell', 'nikkei', 'dax', 'stoxx', 'kospi', 'aex', 'cac', 'smi']):
                asset_classes['equity'].append(symbol)
            elif any(term in name for term in ['usd', 'eur', 'gbp', 'jpy', 'aud', 'cad', 'chf', 'nok', 'nzd', 'sek']):
                asset_classes['fx'].append(symbol)
            elif 'vix' in name or 'volatility' in name or 'vstoxx' in name:
                asset_classes['volatility'].append(symbol)
            else:
                # Default to commodities for metals, energy, agriculture
                asset_classes['commodities'].append(symbol)
                
        except Exception:
            continue
    
    # Remove empty asset classes
    asset_classes = {k: v for k, v in asset_classes.items() if v}
    
    return asset_classes

def create_risk_parity_weights(asset_classes):
    """
    Create risk parity portfolio weights with equal risk allocation across asset classes.
    
    Parameters:
        asset_classes (dict): Asset class groupings.
    
    Returns:
        dict: Portfolio weights by instrument.
    """
    portfolio_weights = {}
    num_asset_classes = len(asset_classes)
    
    if num_asset_classes == 0:
        return portfolio_weights
    
    # Equal risk allocation across asset classes
    risk_per_asset_class = 1.0 / num_asset_classes
    
    for asset_class, instruments in asset_classes.items():
        num_instruments = len(instruments)
        if num_instruments > 0:
            weight_per_instrument = risk_per_asset_class / num_instruments
            for instrument in instruments:
                portfolio_weights[instrument] = weight_per_instrument
    
    return portfolio_weights

def create_all_weather_weights(asset_classes):
    """
    Create All Weather style portfolio based on book's methodology.
    
    Target allocations:
    - 25% Equities  
    - 25% Bonds
    - 25% Commodities
    - 25% Other (FX, Volatility)
    
    Parameters:
        asset_classes (dict): Asset class groupings.
    
    Returns:
        dict: Portfolio weights by instrument.
    """
    portfolio_weights = {}
    
    # Define target allocations per asset class
    target_allocations = {
        'equity': 0.25,
        'bonds': 0.25,
        'commodities': 0.25,
        'fx': 0.125,
        'volatility': 0.125
    }
    
    # Calculate total available allocation
    available_classes = set(asset_classes.keys())
    total_target = sum(target_allocations.get(ac, 0) for ac in available_classes)
    
    if total_target == 0:
        return portfolio_weights
    
    # Normalize and allocate
    for asset_class, instruments in asset_classes.items():
        if asset_class in target_allocations and len(instruments) > 0:
            allocation = target_allocations[asset_class] / total_target
            weight_per_instrument = allocation / len(instruments)
            
            for instrument in instruments:
                portfolio_weights[instrument] = weight_per_instrument
    
    return portfolio_weights

def optimize_instrument_selection(instruments_df, suitable_instruments, target_instruments=10):
    """
    Select optimal instruments based on cost efficiency and diversification.
    
    Parameters:
        instruments_df (pd.DataFrame): Instruments data.
        suitable_instruments (list): Pre-filtered suitable instruments.
        target_instruments (int): Target number of instruments.
    
    Returns:
        dict: Selected instruments with equal weights.
    """
    if len(suitable_instruments) == 0:
        return {}
    
    # Sort by SR cost (ascending)
    instrument_costs = []
    for symbol in suitable_instruments:
        try:
            instrument = instruments_df[instruments_df['Symbol'] == symbol].iloc[0]
            sr_cost = instrument['SR_cost']
            if not pd.isna(sr_cost):
                instrument_costs.append((symbol, sr_cost))
        except:
            continue
    
    # Sort by cost and select top instruments
    instrument_costs.sort(key=lambda x: x[1])
    selected_instruments = [symbol for symbol, _ in instrument_costs[:target_instruments]]
    
    # Equal weights
    if selected_instruments:
        weight = 1.0 / len(selected_instruments)
        return {symbol: weight for symbol in selected_instruments}
    else:
        return {}

#####   PORTFOLIO BACKTESTING   #####

def backtest_portfolio_with_individual_data(portfolio_weights, data, instruments_df, capital=100000, 
                                           risk_target=0.2, start_date='2000-01-01', end_date='2025-01-01'):
    """
    Backtest portfolio strategy using individual instrument data.
    
    Parameters:
        portfolio_weights (dict): Portfolio weights by instrument.
        data (dict): Individual instrument data.
        instruments_df (pd.DataFrame): Instruments specifications.
        capital (float): Initial capital.
        risk_target (float): Target risk fraction.
        start_date (str): Start date.
        end_date (str): End date.
    
    Returns:
        dict: Backtest results.
    """
    if not portfolio_weights:
        return {'error': 'No portfolio weights provided'}
    
    # Create aligned returns matrix
    returns_matrix = create_returns_matrix(data)
    
    if returns_matrix.empty:
        return {'error': 'No aligned returns data'}
    
    # Filter weights to only include instruments with data
    available_weights = {k: v for k, v in portfolio_weights.items() if k in returns_matrix.columns}
    
    if not available_weights:
        return {'error': 'No instruments with both weights and data'}
    
    # Normalize weights
    total_weight = sum(available_weights.values())
    if total_weight > 0:
        available_weights = {k: v/total_weight for k, v in available_weights.items()}
    
    # Calculate correlations and IDM
    correlation_matrix = calculate_correlation_matrix(returns_matrix[list(available_weights.keys())])
    idm = calculate_idm_from_correlations(available_weights, correlation_matrix)
    
    # Initialize tracking arrays
    portfolio_returns = []
    positions_data = {symbol: [] for symbol in available_weights.keys()}
    
    # Get first valid date range
    aligned_dates = returns_matrix.index
    
    for i, date in enumerate(aligned_dates):
        if i == 0:
            # No positions on first day
            for symbol in available_weights.keys():
                positions_data[symbol].append(0)
            portfolio_returns.append(0)
            continue
        
        # Get previous day data for position sizing
        prev_date = aligned_dates[i-1]
        current_date = date
        
        daily_portfolio_return = 0
        
        for symbol, weight in available_weights.items():
            try:
                # Get instrument specs
                specs = get_instrument_specs(symbol, instruments_df)
                multiplier = specs['multiplier']
                
                # Get price and volatility data
                symbol_data = data[symbol]
                
                if prev_date in symbol_data.index and current_date in symbol_data.index:
                    prev_price = symbol_data.loc[prev_date, 'Last']
                    current_return = symbol_data.loc[current_date, 'returns']
                    
                    # Calculate blended volatility for this instrument
                    symbol_returns = symbol_data['returns'].loc[:prev_date]
                    if len(symbol_returns) > 50:  # Need sufficient history
                        blended_vol = calculate_blended_volatility(symbol_returns).iloc[-1]
                    else:
                        # Use simple rolling volatility as fallback
                        blended_vol = symbol_returns.rolling(22).std().iloc[-1] * np.sqrt(business_days_per_year)
                    
                    # Calculate position size with IDM
                    if not pd.isna(blended_vol) and blended_vol > 0:
                        position = calculate_position_size_with_idm(
                            capital, weight, idm, multiplier, prev_price, 1.0, blended_vol, risk_target
                        )
                    else:
                        position = 0
                    
                    positions_data[symbol].append(position)
                    
                    # Calculate contribution to portfolio return
                    instrument_pnl = position * multiplier * current_return * prev_price
                    instrument_return = instrument_pnl / capital
                    daily_portfolio_return += instrument_return
                else:
                    positions_data[symbol].append(0)
                    
            except Exception as e:
                positions_data[symbol].append(0)
        
        portfolio_returns.append(daily_portfolio_return)
    
    # Create results dataframe
    results_df = pd.DataFrame(index=aligned_dates)
    results_df['portfolio_returns'] = portfolio_returns
    
    for symbol, positions in positions_data.items():
        results_df[f'position_{symbol}'] = positions
    
    results_df = results_df.dropna()
    
    if len(results_df) == 0:
        return {'error': 'No valid backtest data'}
    
    # Calculate performance metrics
    performance = calculate_comprehensive_performance(
        build_account_curve(results_df['portfolio_returns'], capital),
        results_df['portfolio_returns']
    )
    
    # Add portfolio-specific metrics
    performance['num_instruments'] = len(available_weights)
    performance['idm'] = idm
    performance['correlation_matrix'] = correlation_matrix
    performance['available_weights'] = available_weights
    
    return {
        'data': results_df,
        'performance': performance,
        'portfolio_weights': available_weights,
        'correlation_matrix': correlation_matrix,
        'idm': idm
    }

#####   MAIN COMPARISON FUNCTION   #####

def run_portfolio_comparison(capital=100000, risk_target=0.2, max_instruments=20):
    """
    Run comprehensive portfolio strategy comparison using real data.
    
    Parameters:
        capital (float): Initial capital.
        risk_target (float): Target risk fraction.
        max_instruments (int): Maximum instruments for optimized selection.
    
    Returns:
        dict: Comparison results.
    """
    print("Loading instruments and data...")
    
    # Load instruments data
    instruments_df = load_instrument_data()
    
    # Get available instruments with data files
    available_instruments = get_available_instruments(instruments_df)
    print(f"Found {len(available_instruments)} instruments with data files")
    
    # Select suitable instruments
    suitable_instruments = select_instruments_by_criteria(
        instruments_df, available_instruments, capital
    )
    print(f"Selected {len(suitable_instruments)} suitable instruments")
    
    if len(suitable_instruments) == 0:
        print("No suitable instruments found!")
        return {}
    
    # Load data for suitable instruments
    print("Loading individual instrument data...")
    data = load_instrument_data_files(suitable_instruments)
    print(f"Successfully loaded data for {len(data)} instruments")
    
    if len(data) == 0:
        print("No data loaded!")
        return {}
    
    # Create asset class groups
    asset_classes = create_asset_class_groups(instruments_df, list(data.keys()))
    
    print("\nAsset class distribution:")
    for asset_class, instruments in asset_classes.items():
        print(f"  {asset_class}: {len(instruments)} instruments")
    
    # Create portfolio strategies
    strategies = {}
    
    # 1. Risk Parity Portfolio
    if asset_classes:
        risk_parity_weights = create_risk_parity_weights(asset_classes)
        if risk_parity_weights:
            strategies['Risk Parity'] = risk_parity_weights
    
    # 2. All Weather Portfolio
    if asset_classes:
        all_weather_weights = create_all_weather_weights(asset_classes)
        if all_weather_weights:
            strategies['All Weather'] = all_weather_weights
    
    # 3. Optimized Selection
    optimized_weights = optimize_instrument_selection(
        instruments_df, list(data.keys()), target_instruments=min(max_instruments, len(data))
    )
    if optimized_weights:
        strategies['Optimized Selection'] = optimized_weights
    
    # 4. Single Best Instrument (lowest cost)
    if len(data) > 0:
        best_instrument = min(data.keys(), key=lambda x: get_instrument_specs(x, instruments_df)['sr_cost'])
        strategies['Single Best'] = {best_instrument: 1.0}
    
    return strategies, data, instruments_df, asset_classes

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
    Run comprehensive Chapter 4 portfolio analysis.
    """
    print("=" * 70)
    print("CHAPTER 4: PORTFOLIO STRATEGIES WITH REAL DATA")
    print("=" * 70)
    
    # Run portfolio comparison
    try:
        strategies, data, instruments_df, asset_classes = run_portfolio_comparison()
        
        if not strategies:
            print("No strategies to test!")
            return
        
        # Print universe statistics
        universe_stats = get_instrument_universe_stats(instruments_df, data)
        print(f"\n----- Instrument Universe Statistics -----")
        print(f"Total instruments with data: {universe_stats['total_instruments_available']}")
        print(f"Asset class distribution:")
        for asset_class, count in universe_stats['asset_classes'].items():
            print(f"  {asset_class}: {count} instruments")
        
        if 'cost_distribution' in universe_stats and universe_stats['cost_distribution']:
            cost_stats = universe_stats['cost_distribution']
            print(f"\nCost distribution (SR units):")
            print(f"  Mean: {cost_stats['mean']:.6f}")
            print(f"  Median: {cost_stats['median']:.6f}")
            print(f"  Range: {cost_stats['min']:.6f} - {cost_stats['max']:.6f}")
        
        print(f"\n----- Testing {len(strategies)} Portfolio Strategies -----")
        
        results = {}
        
        for strategy_name, weights in strategies.items():
            print(f"\n--- {strategy_name} ---")
            print(f"Instruments: {len(weights)} - {list(weights.keys())[:10]}{'...' if len(weights) > 10 else ''}")
            
            # Backtest the strategy
            result = backtest_portfolio_with_individual_data(
                weights, data, instruments_df, capital=100000
            )
            
            if 'error' in result:
                print(f"Error: {result['error']}")
                continue
            
            results[strategy_name] = result
            perf = result['performance']
            
            print(f"Performance:")
            print(f"  Annual Return: {perf['annualized_return']:.2%}")
            print(f"  Volatility: {perf['annualized_volatility']:.2%}")
            print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {perf['max_drawdown_pct']:.1f}%")
            print(f"  IDM: {result['idm']:.2f}")
        
        # Summary comparison
        if results:
            print("\n" + "="*80)
            print("PORTFOLIO PERFORMANCE SUMMARY")
            print("="*80)
            
            print(f"\n{'Strategy':<20} {'Return':<10} {'Vol':<8} {'Sharpe':<8} {'MaxDD':<8} {'Instruments':<12} {'IDM':<6}")
            print("-" * 80)
            
            for strategy_name, result in results.items():
                perf = result['performance']
                print(f"{strategy_name:<20} {perf['annualized_return']:<10.1%} "
                      f"{perf['annualized_volatility']:<8.1%} {perf['sharpe_ratio']:<8.3f} "
                      f"{perf['max_drawdown_pct']:<8.1f}% {perf['num_instruments']:<12} "
                      f"{result['idm']:<6.2f}")
        
        # Correlation analysis
        if 'Risk Parity' in results and len(results['Risk Parity']['correlation_matrix']) > 1:
            corr_matrix = results['Risk Parity']['correlation_matrix']
            print(f"\n----- Correlation Analysis -----")
            print(f"Portfolio instruments: {len(corr_matrix)}")
            correlations = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            print(f"Average correlation: {correlations.mean():.3f}")
            print(f"Min correlation: {correlations.min():.3f}")
            print(f"Max correlation: {correlations.max():.3f}")
            print(f"Correlation std dev: {correlations.std():.3f}")
        
        # Key insights
        print(f"\n----- Key Insights -----")
        print("1. Diversification Analysis:")
        if results:
            best_sharpe = max(results.values(), key=lambda x: x['performance']['sharpe_ratio'])
            best_strategy = [k for k, v in results.items() if v == best_sharpe][0]
            print(f"   - Best Sharpe ratio: {best_strategy} ({best_sharpe['performance']['sharpe_ratio']:.3f})")
            print(f"   - IDM range: {min(r['idm'] for r in results.values()):.2f} - {max(r['idm'] for r in results.values()):.2f}")
        
        print("2. Portfolio Construction:")
        print("   - Risk parity: Equal risk allocation across asset classes")
        print("   - All Weather: Balanced allocation across economic environments")
        print("   - Optimized: Focus on cost-efficient instruments")
        
        print("3. Implementation Notes:")
        print("   - All calculations use actual individual instrument data")
        print("   - IDM calculated from real correlation matrices")
        print("   - Position sizing includes variable volatility forecasting")
        print("   - Functions are reusable for future strategy development")
        
    except Exception as e:
        print(f"Error in portfolio analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

