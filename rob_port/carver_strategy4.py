#!/usr/bin/env python3
"""
Strategy 4 implementation based on Robert Carver's actual code methodology
from "Advanced Futures Trading Strategies"

This follows his static_instrument_selection.py and handcrafting.py approach.
"""

from chapter3 import *
from chapter2 import *
from chapter1 import *
from instrument_selection import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from scipy.cluster import hierarchy as sch

#####   ROBERT CARVER'S CLASSES   #####

class CorrelationEstimate:
    """Correlation matrix handler following Carver's approach"""
    
    def __init__(self, values: pd.DataFrame):
        columns = values.columns
        values = values.values
        self._values = values
        self._columns = list(columns)

    def __repr__(self):
        return str(self.as_pd())

    def __len__(self):
        return len(self.columns)

    def as_pd(self) -> pd.DataFrame:
        return pd.DataFrame(self._values, index=self._columns, columns=self._columns)

    @property
    def values(self) -> np.array:
        return self._values

    @property
    def columns(self) -> list:
        return self._columns

    @property
    def size(self) -> int:
        return len(self.columns)

    def subset(self, subset_of_asset_names: list):
        as_pd = self.as_pd()
        subset_pd = as_pd.loc[subset_of_asset_names, subset_of_asset_names]
        return CorrelationEstimate(subset_pd)

class PortfolioWeights(dict):
    """Portfolio weights handler following Carver's approach"""
    
    @property
    def weights(self):
        return list(self.values())

    @property
    def assets(self):
        return list(self.keys())

    def multiply_by_float(self, multiplier: float):
        list_of_assets = self.assets
        list_of_weights = [self[asset] for asset in list_of_assets]
        list_of_weights_multiplied = [weight * multiplier for weight in list_of_weights]
        return PortfolioWeights.from_weights_and_keys(
            list_of_weights=list_of_weights_multiplied,
            list_of_keys=list_of_assets
        )

    @classmethod
    def from_list_of_subportfolios(cls, list_of_portfolio_weights):
        list_of_unique_asset_names = list(set(
            [asset for subportfolio in list_of_portfolio_weights 
             for asset in subportfolio.assets]
        ))

        portfolio_weights = cls.allzeros(list_of_unique_asset_names)

        for subportfolio_weights in list_of_portfolio_weights:
            for asset_name in subportfolio_weights.assets:
                portfolio_weights[asset_name] = (
                    portfolio_weights[asset_name] + subportfolio_weights[asset_name]
                )

        return portfolio_weights

    @classmethod
    def allzeros(cls, list_of_keys: list):
        return cls.from_weights_and_keys(
            list_of_weights=[0.0] * len(list_of_keys), 
            list_of_keys=list_of_keys
        )

    @classmethod
    def from_weights_and_keys(cls, list_of_weights: list, list_of_keys: list):
        assert len(list_of_keys) == len(list_of_weights)
        return cls([(key, weight) for key, weight in zip(list_of_keys, list_of_weights)])

def one_over_n_weights_given_asset_names(list_of_asset_names: list) -> PortfolioWeights:
    weight = 1.0 / len(list_of_asset_names)
    return PortfolioWeights([(asset_name, weight) for asset_name in list_of_asset_names])

def cluster_correlation_matrix(corr_matrix: CorrelationEstimate, cluster_size: int = 2):
    """Cluster correlation matrix using hierarchical clustering"""
    try:
        corr_as_np = corr_matrix.values
        d = sch.distance.pdist(corr_as_np)
        L = sch.linkage(d, method="complete")
        
        N = len(corr_as_np)
        cutoff = L[N - cluster_size][2] - 0.000001
        
        ind = sch.fcluster(L, cutoff, "distance")
        ind = list(ind)
        
        # Convert cluster indices to asset names
        all_clusters = list(set(ind))
        asset_names = corr_matrix.columns
        list_of_asset_clusters = [
            [asset for asset, cluster in zip(asset_names, ind) if cluster == cluster_id]
            for cluster_id in all_clusters
        ]
        
        return list_of_asset_clusters
    except:
        # Fallback to arbitrary split
        count_assets = len(corr_matrix.columns)
        asset_names = corr_matrix.columns
        clusters = [(x % cluster_size) + 1 for x in range(count_assets)]
        
        all_clusters = list(set(clusters))
        list_of_asset_clusters = [
            [asset for asset, cluster in zip(asset_names, clusters) if cluster == cluster_id]
            for cluster_id in all_clusters
        ]
        
        return list_of_asset_clusters

class HandcraftPortfolio:
    """Handcraft portfolio weights using correlation clustering"""
    
    def __init__(self, correlation: CorrelationEstimate):
        self._correlation = correlation

    @property
    def correlation(self) -> CorrelationEstimate:
        return self._correlation

    @property
    def size(self) -> int:
        return len(self.correlation)

    @property
    def asset_names(self) -> list:
        return list(self.correlation.columns)

    def weights(self) -> PortfolioWeights:
        if self.size <= 2:
            return self.risk_weights_this_portfolio()
        else:
            return self.aggregated_risk_weights()

    def risk_weights_this_portfolio(self) -> PortfolioWeights:
        return one_over_n_weights_given_asset_names(self.asset_names)

    def aggregated_risk_weights(self):
        clusters_as_names = cluster_correlation_matrix(self.correlation)
        sub_portfolios = [self.subset(cluster) for cluster in clusters_as_names]
        
        # Equal allocation to each cluster
        asset_count = len(sub_portfolios)
        weights_for_each_subportfolio = [1.0/asset_count] * asset_count
        
        risk_weights_by_portfolio = [sub_portfolio.weights() for sub_portfolio in sub_portfolios]
        
        multiplied_risk_weights_by_portfolio = [
            sub_portfolio_weights.multiply_by_float(weight_for_subportfolio) 
            for weight_for_subportfolio, sub_portfolio_weights in 
            zip(weights_for_each_subportfolio, risk_weights_by_portfolio)
        ]
        
        return PortfolioWeights.from_list_of_subportfolios(multiplied_risk_weights_by_portfolio)

    def subset(self, subset_of_asset_names: list):
        return HandcraftPortfolio(self.correlation.subset(subset_of_asset_names))

#####   CARVER'S CALCULATIONS   #####

def calculate_idm_from_correlation(portfolio_weights: PortfolioWeights, 
                                 correlation_matrix: CorrelationEstimate) -> float:
    """Calculate IDM using actual correlation matrix (Carver's method)"""
    if len(portfolio_weights.assets) == 1:
        return 1.0

    aligned_correlation_matrix = correlation_matrix.subset(portfolio_weights.assets)
    weights_np = np.array(portfolio_weights.weights)
    corr_np = aligned_correlation_matrix.values
    
    variance = weights_np.dot(corr_np).dot(weights_np)
    risk = variance ** 0.5
    
    return 1.0 / risk

def minimum_capital_for_sub_strategy(fx: float, idm: float, weight: float, 
                                   instrument_risk_ann_perc: float, price: float, 
                                   multiplier: float, risk_target: float, 
                                   min_contracts: int = 4) -> float:
    """Calculate minimum capital using Carver's formula"""
    numerator = min_contracts * multiplier * price * fx * instrument_risk_ann_perc
    denominator = idm * weight * risk_target
    
    if denominator == 0:
        return float('inf')
    
    return numerator / denominator

def risk_adjusted_cost_for_instrument(symbol: str, instruments_df: pd.DataFrame, 
                                    position_turnover: float, rolls_per_year: float = 4) -> float:
    """Calculate risk-adjusted cost for instrument"""
    specs = get_instrument_specs(symbol, instruments_df)
    SR_cost_per_trade = specs['sr_cost']
    return SR_cost_per_trade * (rolls_per_year + position_turnover)

def get_instrument_characteristics(symbol: str):
    """Get realistic price and volatility estimates for instruments"""
    if 'VIX' in symbol:
        return 20, 0.80
    elif 'BTC' in symbol or 'MBT' in symbol:
        return 50000, 0.60
    elif any(term in symbol for term in ['ZN', 'ZB', 'ZF', 'ZT', '3KTB', 'TN', 'GBL', 'BTP', 'OAT']):
        return 110, 0.06
    elif any(curr in symbol for curr in ['EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD', 'NOK', 'SEK']):
        return 1.1, 0.10
    elif any(metal in symbol for metal in ['GC', 'MGC', 'SI', 'HG', 'PL', 'PA']):
        if 'GC' in symbol or 'MGC' in symbol:
            return 2000, 0.18
        elif 'SI' in symbol:
            return 25, 0.25
        else:
            return 4, 0.22
    elif any(energy in symbol for energy in ['CL', 'QM', 'RB', 'HO', 'NG', 'BZ']):
        return 70, 0.30
    elif any(ag in symbol for ag in ['ZC', 'ZS', 'ZW', 'ZM', 'ZL', 'ZO', 'ZR']):
        return 500, 0.25
    else:
        return 4000, 0.18  # Equity indices

def minimum_capital_okay_for_instrument(symbol: str, instruments_df: pd.DataFrame, 
                                      idm: float, weight: float, risk_target: float, 
                                      capital: float) -> bool:
    """Check if minimum capital requirements are met"""
    try:
        specs = get_instrument_specs(symbol, instruments_df)
        price, volatility = get_instrument_characteristics(symbol)
        
        minimum_capital = minimum_capital_for_sub_strategy(
            fx=1.0,
            idm=idm,
            weight=weight,
            instrument_risk_ann_perc=volatility,
            price=price,
            multiplier=specs['multiplier'],
            risk_target=risk_target
        )
        
        return minimum_capital <= capital
    except:
        return False

def calculate_portfolio_weights(selected_instruments: list, 
                               correlation_matrix: CorrelationEstimate) -> PortfolioWeights:
    """Calculate portfolio weights using handcrafting method"""
    if len(selected_instruments) == 1:
        return PortfolioWeights.from_weights_and_keys([1.0], selected_instruments)

    subset_matrix = correlation_matrix.subset(selected_instruments)
    handcraft_portfolio = HandcraftPortfolio(subset_matrix)
    return handcraft_portfolio.weights()

def calculate_expected_mean_for_portfolio(portfolio_weights: PortfolioWeights,
                                        pre_cost_SR: float,
                                        instruments_df: pd.DataFrame,
                                        position_turnover: float) -> float:
    """Calculate expected mean return for portfolio"""
    instrument_means = []
    for instrument_code in portfolio_weights.assets:
        weight = portfolio_weights[instrument_code]
        costs_SR_units = risk_adjusted_cost_for_instrument(
            instrument_code, instruments_df, position_turnover
        )
        SR_for_instrument = pre_cost_SR - costs_SR_units
        instrument_means.append(weight * SR_for_instrument)
    
    return sum(instrument_means)

def calculate_expected_std_for_portfolio(portfolio_weights: PortfolioWeights,
                                       correlation_matrix: CorrelationEstimate) -> float:
    """Calculate expected standard deviation for portfolio"""
    subset_aligned_correlation = correlation_matrix.subset(portfolio_weights.assets)
    weights_np = np.array(portfolio_weights.weights)
    sigma = subset_aligned_correlation.values
    
    variance = weights_np.dot(sigma).dot(weights_np.transpose())
    return variance ** 0.5

def calculate_SR_for_selected_instruments(selected_instruments: list,
                                        pre_cost_SR: float,
                                        instruments_df: pd.DataFrame,
                                        position_turnover: float,
                                        correlation_matrix: CorrelationEstimate,
                                        capital: float,
                                        risk_target: float) -> float:
    """Calculate Sharpe ratio for selected instruments"""
    
    # Calculate portfolio weights
    portfolio_weights = calculate_portfolio_weights(selected_instruments, correlation_matrix)
    
    # Check minimum capital requirements
    idm = calculate_idm_from_correlation(portfolio_weights, correlation_matrix)
    
    for instrument_code in portfolio_weights.assets:
        weight = portfolio_weights[instrument_code]
        okay_for_instrument = minimum_capital_okay_for_instrument(
            instrument_code, instruments_df, idm, weight, risk_target, capital
        )
        if not okay_for_instrument:
            return -999999999999  # Large negative number
    
    # Calculate expected mean and std
    expected_mean = calculate_expected_mean_for_portfolio(
        portfolio_weights, pre_cost_SR, instruments_df, position_turnover
    )
    expected_std = calculate_expected_std_for_portfolio(
        portfolio_weights, correlation_matrix
    )
    
    if expected_std == 0:
        return -999999999999
    
    return expected_mean / expected_std

#####   CORRELATION MATRIX FROM DATA   #####

def create_correlation_matrix_from_data(selected_instruments: list):
    """Create correlation matrix from actual return data"""
    
    # Load actual return data
    returns_data = {}
    
    for symbol in selected_instruments:
        file_path = f"Data/{symbol.lower()}_daily_data.csv"
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, parse_dates=['Time'])
                df.set_index('Time', inplace=True)
                df['returns'] = df['Last'].pct_change()
                df = df.dropna()
                
                if len(df) > 252:  # At least 1 year of data
                    returns_data[symbol] = df['returns']
            except:
                continue
    
    if len(returns_data) < 2:
        # Fallback to synthetic correlation matrix
        size = len(selected_instruments)
        corr_matrix = np.eye(size) * 0.8 + np.ones((size, size)) * 0.2
        corr_df = pd.DataFrame(corr_matrix, index=selected_instruments, columns=selected_instruments)
        return CorrelationEstimate(corr_df)
    
    # Create DataFrame of returns with common dates
    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.dropna()
    
    if len(returns_df) < 100:
        # Not enough overlapping data, use synthetic
        size = len(selected_instruments)
        corr_matrix = np.eye(size) * 0.8 + np.ones((size, size)) * 0.2
        corr_df = pd.DataFrame(corr_matrix, index=selected_instruments, columns=selected_instruments)
        return CorrelationEstimate(corr_df)
    
    # Calculate correlation matrix
    correlation_df = returns_df.corr()
    
    # Ensure all selected instruments are included
    for symbol in selected_instruments:
        if symbol not in correlation_df.columns:
            # Add synthetic correlations for missing instruments
            correlation_df.loc[symbol, :] = 0.3
            correlation_df.loc[:, symbol] = 0.3
            correlation_df.loc[symbol, symbol] = 1.0
    
    # Subset to selected instruments
    correlation_df = correlation_df.loc[selected_instruments, selected_instruments]
    
    return CorrelationEstimate(correlation_df)

#####   INSTRUMENT SELECTION   #####

def select_first_static_instrument(instruments_df: pd.DataFrame,
                                 suitable_instruments: list,
                                 approx_number_of_instruments: int,
                                 approx_IDM: float,
                                 capital: float,
                                 risk_target: float,
                                 position_turnover: float):
    """Select first instrument following Carver's method"""
    
    approx_initial_weight = 1.0 / approx_number_of_instruments
    
    # Filter instruments that meet minimum capital requirements
    instruments_okay_for_minimum_capital = []
    for symbol in suitable_instruments:
        okay = minimum_capital_okay_for_instrument(
            symbol, instruments_df, approx_IDM, approx_initial_weight, risk_target, capital
        )
        if okay:
            instruments_okay_for_minimum_capital.append(symbol)
    
    if not instruments_okay_for_minimum_capital:
        return None
    
    # Find cheapest instrument
    best_symbol = None
    best_cost = float('inf')
    
    for symbol in instruments_okay_for_minimum_capital:
        cost = risk_adjusted_cost_for_instrument(symbol, instruments_df, position_turnover)
        if cost < best_cost:
            best_cost = cost
            best_symbol = symbol
    
    return best_symbol

def choose_next_instrument(selected_instruments: list,
                         suitable_instruments: list,
                         pre_cost_SR: float,
                         capital: float,
                         risk_target: float,
                         instruments_df: pd.DataFrame,
                         position_turnover: float,
                         correlation_matrix: CorrelationEstimate) -> str:
    """Choose next best instrument to add"""
    
    remaining_instruments = [inst for inst in suitable_instruments 
                           if inst not in selected_instruments]
    
    if not remaining_instruments:
        return None
    
    best_instrument = None
    best_SR = -float('inf')
    
    for instrument_code in remaining_instruments:
        trial_instruments = selected_instruments + [instrument_code]
        
        # Update correlation matrix for trial
        trial_correlation_matrix = create_correlation_matrix_from_data(trial_instruments)
        
        SR = calculate_SR_for_selected_instruments(
            trial_instruments,
            pre_cost_SR=pre_cost_SR,
            instruments_df=instruments_df,
            position_turnover=position_turnover,
            correlation_matrix=trial_correlation_matrix,
            capital=capital,
            risk_target=risk_target
        )
        
        if SR > best_SR:
            best_SR = SR
            best_instrument = instrument_code
    
    return best_instrument

#####   CARVER'S STRATEGY 4 IMPLEMENTATION   #####

def implement_carver_strategy4(instruments_df: pd.DataFrame, 
                             capital: float, 
                             risk_target: float = 0.2,
                             pre_cost_SR: float = 0.4,
                             position_turnover: float = 5,
                             approx_number_of_instruments: int = 5,
                             approx_IDM: float = 2.5):
    """Implement Strategy 4 using Robert Carver's exact methodology"""
    
    print("=" * 80)
    print("CARVER'S STRATEGY 4: STATIC INSTRUMENT SELECTION")
    print("=" * 80)
    
    print(f"Capital: ${capital:,.0f}")
    print(f"Risk target: {risk_target:.1%}")
    print(f"Pre-cost SR: {pre_cost_SR}")
    print(f"Position turnover: {position_turnover}")
    
    # Step 1: Get suitable instruments (pre-filtered for cost and liquidity)
    print("\n----- Step 1: Filtering Suitable Instruments -----")
    available_instruments = instruments_df['Symbol'].tolist()
    
    suitable_instruments = select_instruments_by_criteria(
        instruments_df, available_instruments, capital, max_cost_sr=0.01
    )
    
    print(f"Suitable instruments: {len(suitable_instruments)}")
    
    if len(suitable_instruments) < 2:
        print("Not enough suitable instruments!")
        return {}
    
    # Step 2: Select first instrument
    print("\n----- Step 2: Selecting First Instrument -----")
    
    first_instrument = select_first_static_instrument(
        instruments_df=instruments_df,
        suitable_instruments=suitable_instruments,
        approx_number_of_instruments=approx_number_of_instruments,
        approx_IDM=approx_IDM,
        capital=capital,
        risk_target=risk_target,
        position_turnover=position_turnover
    )
    
    if not first_instrument:
        print("No instruments meet minimum capital requirements!")
        return {}
    
    selected_instruments = [first_instrument]
    
    # Create initial correlation matrix
    correlation_matrix = create_correlation_matrix_from_data(selected_instruments)
    
    current_SR = calculate_SR_for_selected_instruments(
        selected_instruments,
        pre_cost_SR=pre_cost_SR,
        instruments_df=instruments_df,
        position_turnover=position_turnover,
        correlation_matrix=correlation_matrix,
        capital=capital,
        risk_target=risk_target
    )
    
    max_SR_achieved = current_SR
    print(f"Starting with {first_instrument}, SR: {current_SR:.4f}")
    
    # Step 3: Iteratively add instruments (Carver's stopping condition)
    print("\n----- Step 3: Iterative Instrument Selection -----")
    
    iteration = 0
    max_iterations = 20  # Safety limit
    
    while current_SR > (max_SR_achieved * 0.9) and iteration < max_iterations:
        iteration += 1
        
        next_instrument = choose_next_instrument(
            selected_instruments=selected_instruments,
            suitable_instruments=suitable_instruments,
            pre_cost_SR=pre_cost_SR,
            capital=capital,
            risk_target=risk_target,
            instruments_df=instruments_df,
            position_turnover=position_turnover,
            correlation_matrix=correlation_matrix
        )
        
        if not next_instrument:
            print("No more instruments to add")
            break
        
        selected_instruments.append(next_instrument)
        
        # Update correlation matrix
        correlation_matrix = create_correlation_matrix_from_data(selected_instruments)
        
        current_SR = calculate_SR_for_selected_instruments(
            selected_instruments,
            pre_cost_SR=pre_cost_SR,
            instruments_df=instruments_df,
            position_turnover=position_turnover,
            correlation_matrix=correlation_matrix,
            capital=capital,
            risk_target=risk_target
        )
        
        if current_SR > max_SR_achieved:
            max_SR_achieved = current_SR
        
        print(f"Added {next_instrument}: {len(selected_instruments)} instruments, SR: {current_SR:.4f}")
    
    # Step 4: Calculate final portfolio weights and positions
    print(f"\n----- Step 4: Final Portfolio -----")
    
    final_correlation_matrix = create_correlation_matrix_from_data(selected_instruments)
    portfolio_weights = calculate_portfolio_weights(selected_instruments, final_correlation_matrix)
    idm = calculate_idm_from_correlation(portfolio_weights, final_correlation_matrix)
    
    print(f"Final instruments: {len(selected_instruments)}")
    print(f"Final SR: {current_SR:.4f}")
    print(f"IDM: {idm:.2f}")
    
    # Calculate position sizes
    position_sizes = {}
    min_capitals = {}
    total_notional = 0
    
    for symbol in selected_instruments:
        weight = portfolio_weights[symbol]
        price, vol = get_instrument_characteristics(symbol)
        
        try:
            specs = get_instrument_specs(symbol, instruments_df)
            
            # Calculate position size
            position_size = (capital * idm * weight * risk_target) / (specs['multiplier'] * price * vol)
            
            # Calculate minimum capital
            min_capital = minimum_capital_for_sub_strategy(
                fx=1.0, idm=idm, weight=weight, 
                instrument_risk_ann_perc=vol, price=price,
                multiplier=specs['multiplier'], risk_target=risk_target
            )
            
            # Calculate notional exposure
            notional = position_size * specs['multiplier'] * price
            total_notional += notional
            
            position_sizes[symbol] = position_size
            min_capitals[symbol] = min_capital
            
            print(f"  {symbol}: Weight {weight:.3f}, Position {position_size:.2f}, Min Cap ${min_capital:,.0f}")
            
        except Exception as e:
            print(f"  Warning: Error calculating {symbol}: {e}")
            position_sizes[symbol] = 0
            min_capitals[symbol] = 0
    
    # Summary
    print(f"\n----- Portfolio Summary -----")
    total_min_capital = sum(min_capitals.values())
    leverage = total_notional / capital if capital > 0 else 0
    
    print(f"Total minimum capital: ${total_min_capital:,.0f}")
    print(f"Total notional exposure: ${total_notional:,.0f}")
    print(f"Leverage ratio: {leverage:.2f}x")
    
    return {
        'selected_instruments': selected_instruments,
        'portfolio_weights': dict(portfolio_weights),
        'position_sizes': position_sizes,
        'min_capitals': min_capitals,
        'idm': idm,
        'final_SR': current_SR,
        'total_min_capital': total_min_capital,
        'total_notional': total_notional,
        'leverage': leverage,
        'correlation_matrix': final_correlation_matrix
    }

def main():
    """Test Robert Carver's Strategy 4 implementation"""
    print("ROBERT CARVER'S STRATEGY 4 IMPLEMENTATION")
    print("=" * 80)
    
    # Load instruments data
    instruments_df = load_instrument_data()
    
    # Test parameters from Carver's code
    capital_levels = [1000000, 5000000, 25000000, 50000000]
    
    for capital in capital_levels:
        print(f"\n{'='*80}")
        print(f"TESTING WITH ${capital:,.0f} CAPITAL")
        print(f"{'='*80}")
        
        # Implement Carver's Strategy 4
        strategy_results = implement_carver_strategy4(
            instruments_df=instruments_df,
            capital=capital,
            risk_target=0.2,
            pre_cost_SR=0.4,
            position_turnover=5,
            approx_number_of_instruments=5,
            approx_IDM=2.5
        )
        
        if strategy_results:
            print(f"Selected {len(strategy_results['selected_instruments'])} instruments")
            print(f"Portfolio weights: {strategy_results['portfolio_weights']}")

if __name__ == "__main__":
    main() 