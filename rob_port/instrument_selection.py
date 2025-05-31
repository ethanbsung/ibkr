from chapter1 import *
import numpy as np
import pandas as pd

def select_instruments_by_criteria(instruments_df, available_instruments, capital, 
                                 max_cost_sr=0.01, min_volume_usd=1500000):
    """
    Select suitable instruments based on criteria from Chapter 4 of the book:
    - Risk adjusted cost below 0.01 SR units
    - Average daily volume of at least 100 contracts, and annualized standard deviation 
      in dollar terms of greater than $1.5 million
    - Minimum position of at least four contracts given available capital
    
    Parameters:
        instruments_df (pd.DataFrame): Instruments data.
        available_instruments (list): Instruments with available data.
        capital (float): Available capital.
        max_cost_sr (float): Maximum acceptable SR cost (0.01 from book).
        min_volume_usd (float): Minimum daily volume requirement in USD risk ($1.5M from book).
    
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
                
            # Check cost criterion (0.01 SR units from book)
            if sr_cost <= max_cost_sr:
                suitable_instruments.append(symbol)
                
        except Exception:
            continue
    
    return suitable_instruments

def create_asset_class_groups(instruments_df, suitable_instruments):
    """
    Create asset class groupings based on instrument names and characteristics.
    Enhanced classification based on the book's methodology.
    
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
        'volatility': [],
        'energy': [],
        'metals': [],
        'agriculture': []
    }
    
    for symbol in suitable_instruments:
        try:
            instrument = instruments_df[instruments_df['Symbol'] == symbol].iloc[0]
            name = instrument['Name'].lower()
            symbol_lower = symbol.lower()
            
            # Volatility (highest priority as it's most specific)
            if any(term in name for term in ['vix', 'volatility', 'vstoxx']) or 'vix' in symbol_lower:
                asset_classes['volatility'].append(symbol)
            
            # Bonds and interest rates
            elif any(term in name for term in ['treasury', 'bond', 'note', 'bund', 'btp', 'schatz', 'bobl', 
                                             'buxl', 'bono', 'eurodollar', 'gilt', 'government']):
                asset_classes['bonds'].append(symbol)
            elif any(term in symbol_lower for term in ['zn', 'zb', 'zf', 'zt', 'zc']):  # Common bond symbols
                asset_classes['bonds'].append(symbol)
                
            # Energy
            elif any(term in name for term in ['crude', 'oil', 'gas', 'heating', 'gasoline', 'natural gas', 'rbob']) and 'soybean' not in name:
                asset_classes['energy'].append(symbol)
            elif any(term in symbol_lower for term in ['cl', 'ng', 'rb', 'ho']):  # Common energy symbols
                asset_classes['energy'].append(symbol)
                
            # Metals  
            elif any(term in name for term in ['gold', 'silver', 'copper', 'platinum', 'palladium']):
                asset_classes['metals'].append(symbol)
            elif any(term in symbol_lower for term in ['gc', 'si', 'hg', 'pl', 'pa']):  # Common metal symbols
                asset_classes['metals'].append(symbol)
                
            # Agriculture
            elif any(term in name for term in ['corn', 'wheat', 'soybean', 'sugar', 'cotton', 'coffee', 'cocoa', 
                                             'cattle', 'hogs', 'feeder', 'lean', 'live', 'orange', 'milk']):
                asset_classes['agriculture'].append(symbol)
            elif any(term in symbol_lower for term in ['zc', 'zw', 'zs', 'sb', 'ct', 'kc', 'cc', 'lc', 'lh', 'fc']):
                asset_classes['agriculture'].append(symbol)
                
            # Equity indices (be more specific)
            elif any(term in name for term in ['s&p', 'dow', 'nasdaq', 'russell', 'nikkei', 'dax', 'stoxx', 'kospi', 
                                             'aex', 'cac', 'smi', 'china', 'singapore', 'taiwan', 'ftse', 'euro stoxx']):
                asset_classes['equity'].append(symbol)
            elif any(term in symbol_lower for term in ['es', 'ym', 'nq', 'rty', 'mes', 'mym', 'mnq', 'nk']):
                asset_classes['equity'].append(symbol)
                
            # FX (exclude eurodollar which is bonds)
            elif any(term in name for term in ['usd', 'eur', 'gbp', 'jpy', 'aud', 'cad', 'chf', 'nok', 'nzd', 
                                             'sek', 'cnh', 'inr', 'mxp', 'rur', 'sgd', 'currency']) and not any(term in name for term in ['eurodollar']):
                asset_classes['fx'].append(symbol)
            elif any(term in symbol_lower for term in ['6e', '6b', '6j', '6a', '6c', '6s', '6n']):
                asset_classes['fx'].append(symbol)
                
            else:
                # Default to commodities for anything else
                asset_classes['commodities'].append(symbol)
                
        except Exception:
            continue
    
    # Merge smaller asset classes into commodities if they have few instruments
    if len(asset_classes['energy']) < 2:
        asset_classes['commodities'].extend(asset_classes['energy'])
        asset_classes['energy'] = []
        
    if len(asset_classes['metals']) < 2:
        asset_classes['commodities'].extend(asset_classes['metals'])
        asset_classes['metals'] = []
        
    if len(asset_classes['agriculture']) < 2:
        asset_classes['commodities'].extend(asset_classes['agriculture'])
        asset_classes['agriculture'] = []
    
    # Remove empty asset classes
    asset_classes = {k: v for k, v in asset_classes.items() if v}
    
    return asset_classes

def create_risk_parity_weights(asset_classes):
    """
    Create risk parity portfolio weights with equal risk allocation across asset classes.
    Follows the book's methodology for Strategy 4.
    
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
    
    Target allocations from the book:
    - Equities: Variable based on available classes
    - Bonds: Higher allocation for stability
    - Commodities: Diversification across sub-classes
    - Other: FX, Volatility
    
    Parameters:
        asset_classes (dict): Asset class groupings.
    
    Returns:
        dict: Portfolio weights by instrument.
    """
    portfolio_weights = {}
    
    # Define target allocations per asset class (book's approach)
    base_allocations = {
        'equity': 0.20,
        'bonds': 0.30,
        'commodities': 0.25,
        'energy': 0.08,
        'metals': 0.08,
        'agriculture': 0.09,
        'fx': 0.10,
        'volatility': 0.10
    }
    
    # Get available asset classes
    available_classes = set(asset_classes.keys())
    
    # Calculate total target allocation for available classes
    total_target = sum(base_allocations.get(ac, 0) for ac in available_classes)
    
    if total_target == 0:
        return portfolio_weights
    
    # Normalize and allocate
    for asset_class, instruments in asset_classes.items():
        if asset_class in base_allocations and len(instruments) > 0:
            allocation = base_allocations[asset_class] / total_target
            weight_per_instrument = allocation / len(instruments)
            
            for instrument in instruments:
                portfolio_weights[instrument] = weight_per_instrument
    
    return portfolio_weights

def optimize_instrument_selection(instruments_df, suitable_instruments, target_instruments=10):
    """
    Select optimal instruments based on cost efficiency and diversification.
    Enhanced to follow book's methodology.
    
    Parameters:
        instruments_df (pd.DataFrame): Instruments data.
        suitable_instruments (list): Pre-filtered suitable instruments.
        target_instruments (int): Target number of instruments.
    
    Returns:
        dict: Selected instruments with equal weights.
    """
    if len(suitable_instruments) == 0:
        return {}
    
    # Create asset class groups first
    asset_classes = create_asset_class_groups(instruments_df, suitable_instruments)
    
    # Select best instruments from each asset class
    selected_instruments = []
    
    for asset_class, instruments in asset_classes.items():
        if not instruments:
            continue
            
        # Sort by SR cost within asset class
        instrument_costs = []
        for symbol in instruments:
            try:
                instrument = instruments_df[instruments_df['Symbol'] == symbol].iloc[0]
                sr_cost = instrument['SR_cost']
                if not pd.isna(sr_cost):
                    instrument_costs.append((symbol, sr_cost))
            except:
                continue
        
        # Take best instruments from this asset class
        instrument_costs.sort(key=lambda x: x[1])
        max_from_class = min(3, len(instrument_costs))  # Max 3 per asset class initially
        selected_instruments.extend([symbol for symbol, _ in instrument_costs[:max_from_class]])
    
    # If we have too few instruments, add more from the remaining pool
    remaining = [sym for sym in suitable_instruments if sym not in selected_instruments]
    while len(selected_instruments) < target_instruments and remaining:
        # Add next best by cost
        remaining_costs = []
        for symbol in remaining:
            try:
                instrument = instruments_df[instruments_df['Symbol'] == symbol].iloc[0]
                sr_cost = instrument['SR_cost']
                if not pd.isna(sr_cost):
                    remaining_costs.append((symbol, sr_cost))
            except:
                continue
        
        if not remaining_costs:
            break
            
        remaining_costs.sort(key=lambda x: x[1])
        next_symbol = remaining_costs[0][0]
        selected_instruments.append(next_symbol)
        remaining.remove(next_symbol)
    
    # Limit to target number
    selected_instruments = selected_instruments[:target_instruments]
    
    # Equal weights
    if selected_instruments:
        weight = 1.0 / len(selected_instruments)
        return {symbol: weight for symbol in selected_instruments}
    else:
        return {}

def calculate_portfolio_minimum_capital(instruments_df, portfolio_weights, instrument_prices, 
                                      instrument_volatilities, risk_target=0.2, min_contracts=4):
    """
    Calculate total minimum capital for a portfolio using Strategy 4 methodology.
    
    Parameters:
        instruments_df (pd.DataFrame): Instruments data.
        portfolio_weights (dict): Portfolio weights.
        instrument_prices (dict): Current prices.
        instrument_volatilities (dict): Volatilities.
        risk_target (float): Target risk fraction.
        min_contracts (int): Minimum contracts per instrument.
    
    Returns:
        dict: Minimum capital analysis.
    """
    from chapter4 import get_idm_for_instruments, calculate_strategy4_min_capital
    
    num_instruments = len(portfolio_weights)
    idm = get_idm_for_instruments(num_instruments)
    
    individual_min_capitals = {}
    total_min_capital = 0
    
    for symbol, weight in portfolio_weights.items():
        if symbol in instrument_prices and symbol in instrument_volatilities:
            min_capital = calculate_strategy4_min_capital(
                symbol, instrument_prices[symbol], instrument_volatilities[symbol],
                weight, idm, instruments_df, risk_target, min_contracts
            )
            individual_min_capitals[symbol] = min_capital
            total_min_capital += min_capital
    
    return {
        'individual_min_capitals': individual_min_capitals,
        'total_min_capital': total_min_capital,
        'idm': idm,
        'num_instruments': num_instruments
    }

def create_custom_portfolio_strategies(instruments_df, suitable_instruments, max_instruments=20):
    """
    Create multiple portfolio strategies using instrument selection logic.
    Enhanced for Chapter 4 methodologies.
    
    Parameters:
        instruments_df (pd.DataFrame): Instruments data.
        suitable_instruments (list): Pre-filtered suitable instruments.
        max_instruments (int): Maximum instruments for optimized selection.
    
    Returns:
        tuple: (strategies dict, asset_classes dict)
    """
    strategies = {}
    
    # Create asset class groups
    asset_classes = create_asset_class_groups(instruments_df, suitable_instruments)
    
    # 1. Risk Parity Portfolio (Strategy 4 approach)
    if asset_classes:
        risk_parity_weights = create_risk_parity_weights(asset_classes)
        if risk_parity_weights:
            strategies['Strategy 4 Risk Parity'] = risk_parity_weights
    
    # 2. All Weather Portfolio
    if asset_classes:
        all_weather_weights = create_all_weather_weights(asset_classes)
        if all_weather_weights:
            strategies['All Weather'] = all_weather_weights
    
    # 3. Optimized Selection (using book's methodology)
    optimized_weights = optimize_instrument_selection(
        instruments_df, suitable_instruments, target_instruments=min(max_instruments, len(suitable_instruments))
    )
    if optimized_weights:
        strategies['Optimized Selection'] = optimized_weights
    
    # 4. Single Best Instrument (lowest cost)
    if len(suitable_instruments) > 0:
        best_instrument = min(suitable_instruments, key=lambda x: get_instrument_specs(x, instruments_df)['sr_cost'])
        strategies['Single Best'] = {best_instrument: 1.0}
    
    # 5. Equal Weight Portfolio
    if len(suitable_instruments) > 0:
        equal_weight = 1.0 / min(max_instruments, len(suitable_instruments))
        equal_weight_instruments = suitable_instruments[:min(max_instruments, len(suitable_instruments))]
        strategies['Equal Weight'] = {symbol: equal_weight for symbol in equal_weight_instruments}
    
    return strategies, asset_classes

def display_asset_class_analysis(asset_classes, instruments_df):
    """
    Display detailed asset class analysis as shown in the book.
    
    Parameters:
        asset_classes (dict): Asset class groupings.
        instruments_df (pd.DataFrame): Instruments data.
    """
    print("\n----- Asset Class Analysis -----")
    
    total_instruments = sum(len(instruments) for instruments in asset_classes.values())
    
    for asset_class, instruments in asset_classes.items():
        allocation = len(instruments) / total_instruments if total_instruments > 0 else 0
        print(f"\n{asset_class.upper()} ({allocation:.1%}):")
        
        for symbol in instruments:
            try:
                specs = get_instrument_specs(symbol, instruments_df)
                print(f"  {symbol:>6}: {specs['name'][:40]:40} | SR Cost: {specs['sr_cost']:.6f}")
            except:
                print(f"  {symbol:>6}: Error getting specs")

def main():
    """
    Test the enhanced instrument selection functionality.
    """
    print("ENHANCED INSTRUMENT SELECTION FOR STRATEGY 4")
    print("=" * 60)
    
    # Load instruments data
    instruments_df = load_instrument_data()
    
    # Test selection
    available_instruments = instruments_df['Symbol'].tolist()
    capital = 50000000
    
    # Apply selection criteria
    suitable_instruments = select_instruments_by_criteria(
        instruments_df, available_instruments, capital, max_cost_sr=0.01
    )
    
    print(f"Total available instruments: {len(available_instruments)}")
    print(f"Suitable instruments: {len(suitable_instruments)}")
    
    # Create asset class analysis
    asset_classes = create_asset_class_groups(instruments_df, suitable_instruments)
    display_asset_class_analysis(asset_classes, instruments_df)
    
    # Create strategies
    strategies, _ = create_custom_portfolio_strategies(instruments_df, suitable_instruments, 15)
    
    print(f"\n----- Strategy Comparison -----")
    for strategy_name, weights in strategies.items():
        print(f"\n{strategy_name}: {len(weights)} instruments")
        for symbol, weight in sorted(weights.items())[:5]:  # Show top 5
            print(f"  {symbol}: {weight:.2%}")

if __name__ == "__main__":
    main() 