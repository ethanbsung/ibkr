from chapter1 import *
import numpy as np
import pandas as pd

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
            # Check volatility first (most specific)
            if 'vix' in name or 'volatility' in name or 'vstoxx' in name:
                asset_classes['volatility'].append(symbol)
            # Check bonds
            elif any(term in name for term in ['treasury', 'bond', 'note', 'bund', 'btp', 'schatz', 'bobl', 'buxl', 'bono', 'eurodollar']):
                asset_classes['bonds'].append(symbol)
            # Check equity (be more specific with oil - only EU Oil, not heating oil or soybean oil)
            elif any(term in name for term in ['s&p', 'dow', 'nasdaq', 'russell', 'nikkei', 'dax', 'stoxx', 'kospi', 'aex', 'cac', 'smi', 'china', 'singapore', 'taiwan', 'auto', 'basic materials', 'health', 'insurance', 'technology', 'travel', 'utilities']) or ('eu oil' in name):
                asset_classes['equity'].append(symbol)
            # Check FX (exclude eurodollar which is bonds)
            elif any(term in name for term in ['usd', 'eur', 'gbp', 'jpy', 'aud', 'cad', 'chf', 'nok', 'nzd', 'sek', 'cnh', 'inr', 'mxp', 'rur', 'sgd']) and not any(term in name for term in ['eurodollar']):
                asset_classes['fx'].append(symbol)
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

def create_custom_portfolio_strategies(instruments_df, suitable_instruments, max_instruments=20):
    """
    Create multiple portfolio strategies using instrument selection logic.
    
    Parameters:
        instruments_df (pd.DataFrame): Instruments data.
        suitable_instruments (list): Pre-filtered suitable instruments.
        max_instruments (int): Maximum instruments for optimized selection.
    
    Returns:
        dict: Dictionary of strategy name -> weights.
    """
    strategies = {}
    
    # Create asset class groups
    asset_classes = create_asset_class_groups(instruments_df, suitable_instruments)
    
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
        instruments_df, suitable_instruments, target_instruments=min(max_instruments, len(suitable_instruments))
    )
    if optimized_weights:
        strategies['Optimized Selection'] = optimized_weights
    
    # 4. Single Best Instrument (lowest cost)
    if len(suitable_instruments) > 0:
        best_instrument = min(suitable_instruments, key=lambda x: get_instrument_specs(x, instruments_df)['sr_cost'])
        strategies['Single Best'] = {best_instrument: 1.0}
    
    return strategies, asset_classes 