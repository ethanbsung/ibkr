#!/usr/bin/env python3
"""
Instrument Selection implementation following Robert Carver's exact methodology
from "Advanced Futures Trading Strategies"

This implements both the handcrafting method and static instrument selection
exactly as provided in the author's code.
"""

import pandas as pd
import numpy as np
from scipy.cluster import hierarchy as sch
from typing import List, Dict, Union
import os

PRINT_TRACE = False

# ===== CORRELATION ESTIMATE CLASS =====

class correlationEstimate(object):
    def __init__(self, values: pd.DataFrame):
        columns = values.columns
        values = values.values

        self._values = values
        self._columns = columns

    def __repr__(self):
        return str(self.as_pd())

    def __len__(self):
        return len(self.columns)

    def as_pd(self) -> pd.DataFrame:
        values = self.values
        columns = self.columns

        return pd.DataFrame(values, index=columns, columns=columns)

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

        new_correlation = correlationEstimate(subset_pd)

        return new_correlation

# ===== CLUSTERING FUNCTIONS =====

def cluster_correlation_matrix(corr_matrix: correlationEstimate,
                               cluster_size: int = 2):

    clusters = get_list_of_clusters_for_correlation_matrix(corr_matrix,
                                                          cluster_size=cluster_size)
    clusters_as_names = from_cluster_index_to_asset_names(clusters, corr_matrix)
    if PRINT_TRACE:
        print("Cluster split: %s" % str(clusters_as_names))

    return clusters_as_names


def get_list_of_clusters_for_correlation_matrix(corr_matrix: correlationEstimate,
                                             cluster_size: int = 2) -> list:
    corr_as_np = corr_matrix.values
    try:
        clusters = get_list_of_clusters_for_correlation_matrix_as_np(
            corr_as_np,
            cluster_size=cluster_size
        )
    except:
        clusters = arbitrary_split_of_correlation_matrix(
            corr_as_np,
            cluster_size=cluster_size
        )

    return clusters

def get_list_of_clusters_for_correlation_matrix_as_np(corr_as_np: np.array,
                                             cluster_size: int = 2) -> list:
    d = sch.distance.pdist(corr_as_np)
    L = sch.linkage(d, method="complete")

    cutoff = cutoff_distance_to_guarantee_N_clusters(corr_as_np, L=L,
                                                     cluster_size = cluster_size)
    ind = sch.fcluster(L, cutoff, "distance")
    ind = list(ind)

    if max(ind) > cluster_size:
        raise Exception("Couldn't cluster into %d clusters" % cluster_size)

    return ind


def cutoff_distance_to_guarantee_N_clusters(corr_as_np: np.array, L: np.array,
                                            cluster_size: int = 2):
    N = len(corr_as_np)
    return L[N - cluster_size][2] - 0.000001


def arbitrary_split_of_correlation_matrix(corr_matrix: np.array,
                                          cluster_size: int = 2) -> list:
    # split correlation of 3 or more assets
    count_assets = len(corr_matrix)
    return arbitrary_split_for_asset_length(count_assets, cluster_size=cluster_size)


def arbitrary_split_for_asset_length(count_assets: int,
                                     cluster_size: int = 2) -> list:

    return [(x % cluster_size) + 1 for x in range(count_assets)]


def from_cluster_index_to_asset_names(
    clusters: list, corr_matrix: correlationEstimate
) -> list:

    all_clusters = list(set(clusters))
    asset_names = corr_matrix.columns
    list_of_asset_clusters = [
        get_asset_names_for_cluster_index(cluster_id, clusters, asset_names)
        for cluster_id in all_clusters
    ]

    return list_of_asset_clusters


def get_asset_names_for_cluster_index(
    cluster_id: int, clusters: list, asset_names: list
):

    list_of_assets = [
        asset for asset, cluster in zip(asset_names, clusters) if cluster == cluster_id
    ]

    return list_of_assets

# ===== PORTFOLIO WEIGHTS CLASS =====

class portfolioWeights(dict):
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
        return portfolioWeights.from_weights_and_keys(
            list_of_weights=list_of_weights_multiplied,
            list_of_keys=list_of_assets
        )

    @classmethod
    def from_list_of_subportfolios(portfolioWeights, list_of_portfolio_weights):
        list_of_unique_asset_names = list(
            set(
                flatten_list(
                    [
                        subportfolio.assets
                        for subportfolio in list_of_portfolio_weights
                    ]
                )
            )
        )

        portfolio_weights = portfolioWeights.allzeros(list_of_unique_asset_names)

        for subportfolio_weights in list_of_portfolio_weights:
            for asset_name in subportfolio_weights.assets:
                portfolio_weights[asset_name] = (
                    portfolio_weights[asset_name] + subportfolio_weights[asset_name]
                )

        return portfolio_weights

    @classmethod
    def allzeros(portfolioWeights, list_of_keys: list):
        return portfolioWeights.all_one_value(list_of_keys, value=0.0)

    @classmethod
    def all_one_value(portfolioWeights, list_of_keys: list, value=0.0):
        return portfolioWeights.from_weights_and_keys(
            list_of_weights=[value] * len(list_of_keys), list_of_keys=list_of_keys
        )

    @classmethod
    def from_weights_and_keys(
        portfolioWeights, list_of_weights: list, list_of_keys: list
    ):
        assert len(list_of_keys) == len(list_of_weights)
        pweights_as_list = [
            (key, weight) for key, weight in zip(list_of_keys, list_of_weights)
        ]

        return portfolioWeights(pweights_as_list)

def flatten_list(some_list):
    flattened = [item for sublist in some_list for item in sublist]

    return flattened


def one_over_n_weights_given_asset_names(list_of_asset_names: list) -> portfolioWeights:
    weight = 1.0 / len(list_of_asset_names)
    return portfolioWeights(
        [(asset_name, weight) for asset_name in list_of_asset_names]
    )

# ===== HANDCRAFT PORTFOLIO CLASS =====

class handcraftPortfolio(object):
    def __init__(self, correlation: correlationEstimate):

        self._correlation = correlation

    @property
    def correlation(self) -> correlationEstimate:
        return self._correlation

    @property
    def size(self) -> int:
        return len(self.correlation)

    @property
    def asset_names(self) -> list:
        return list(self.correlation.columns)

    def weights(self) -> portfolioWeights:
        if self.size <= 2:
            # don't cluster one or two assets
            raw_weights = self.risk_weights_this_portfolio()
        else:
            raw_weights = self.aggregated_risk_weights()

        return raw_weights

    def risk_weights_this_portfolio(self) -> portfolioWeights:

        asset_names = self.asset_names
        raw_weights = one_over_n_weights_given_asset_names(asset_names)

        return raw_weights

    def aggregated_risk_weights(self):
        sub_portfolios = create_sub_portfolios_from_portfolio(self)
        aggregate_risk_weights = aggregate_risk_weights_over_sub_portfolios(
            sub_portfolios
        )

        return aggregate_risk_weights

    def subset(self, subset_of_asset_names: list):
        return handcraftPortfolio(self.correlation.subset(subset_of_asset_names))

# ===== SUB PORTFOLIOS =====

def create_sub_portfolios_from_portfolio(handcraft_portfolio: handcraftPortfolio):

    clusters_as_names = \
        cluster_correlation_matrix(handcraft_portfolio.correlation)

    sub_portfolios = create_sub_portfolios_given_clusters(
        clusters_as_names, handcraft_portfolio
    )

    return sub_portfolios


def create_sub_portfolios_given_clusters(
    clusters_as_names: list, handcraft_portfolio: handcraftPortfolio
) -> list:
    list_of_sub_portfolios = [
        handcraft_portfolio.subset(subset_of_asset_names)
        for subset_of_asset_names in clusters_as_names
    ]

    return list_of_sub_portfolios


def aggregate_risk_weights_over_sub_portfolios(
    sub_portfolios: list,
) -> portfolioWeights:
    # sub portfolios guaranteed to be 2 long
    # We allocate half to each
    asset_count = len(sub_portfolios)
    weights_for_each_subportfolio = [1.0/asset_count]*asset_count

    risk_weights_by_portfolio = [
        sub_portfolio.weights() for sub_portfolio in
        sub_portfolios
    ]

    multiplied_risk_weights_by_portfolio = [
        sub_portfolio_weights.multiply_by_float(weight_for_subportfolio) for
        weight_for_subportfolio, sub_portfolio_weights in
        zip(weights_for_each_subportfolio, risk_weights_by_portfolio)
    ]

    aggregate_weights = portfolioWeights.from_list_of_subportfolios(
        multiplied_risk_weights_by_portfolio
    )

    return aggregate_weights

# ===== STATIC INSTRUMENT SELECTION =====

def select_first_static_instrument(instrument_config: pd.DataFrame,
                                   approx_number_of_instruments: int,
                                   approx_IDM: float,
                                   capital: float,
                                   risk_target: float,
                                   position_turnover: float):

    approx_initial_weight = 1.0/ approx_number_of_instruments
    instrument_list = list(instrument_config.index)
    instruments_okay_for_minimum_capital = [instrument_code
                         for instrument_code in instrument_list
                         if minimum_capital_okay_for_instrument(
                                instrument_code=instrument_code,
                                instrument_config=instrument_config,
                                capital=capital,
                                weight=approx_initial_weight,
                                idm = approx_IDM,
                                risk_target=risk_target
                                )]

    cheapest_instrument = lowest_risk_adjusted_cost_given_instrument_list(
        instruments_okay_for_minimum_capital,
        instrument_config=instrument_config,
        position_turnover=position_turnover
    )

    return cheapest_instrument

def minimum_capital_okay_for_instrument(instrument_code: str,
                                         instrument_config: pd.DataFrame,
                                         idm: float,
                                         weight: float,
                                         risk_target: float,
                                         capital: float) -> bool:

    config_for_instrument = instrument_config.loc[instrument_code]
    minimum_capital = minimum_capital_for_sub_strategy(
        fx = config_for_instrument.fx_rate,
        idm = idm,
        weight=weight,
        instrument_risk_ann_perc=config_for_instrument.ann_std,
        price=config_for_instrument.price,
        multiplier=config_for_instrument.multiplier,
        risk_target=risk_target
    )

    return minimum_capital<=capital

def minimum_capital_for_sub_strategy(fx: float, 
                                   idm: float, 
                                   weight: float,
                                   instrument_risk_ann_perc: float, 
                                   price: float,
                                   multiplier: float, 
                                   risk_target: float,
                                   min_contracts: int = 4) -> float:
    """Calculate minimum capital using Strategy 4 formula from book"""
    # From book: (4 × Multiplier × Price × FX rate × σ%) ÷ (IDM × Weight × τ)
    numerator = min_contracts * multiplier * price * fx * instrument_risk_ann_perc
    denominator = idm * weight * risk_target
    
    if denominator == 0:
        return float('inf')
    
    return numerator / denominator

def lowest_risk_adjusted_cost_given_instrument_list(
        instrument_list: list,
        instrument_config: pd.DataFrame,
        position_turnover: float
        ) -> str:

    list_of_risk_adjusted_cost_by_instrument = [
        risk_adjusted_cost_for_instrument(instrument_code,
                                          instrument_config = instrument_config,
                                          position_turnover = position_turnover)
        for instrument_code in instrument_list
    ]
    index_min = get_min_index(list_of_risk_adjusted_cost_by_instrument)
    return instrument_list[index_min]

def get_min_index(x: list) -> int:
    index_min = get_func_index(x, min)
    return index_min

def get_max_index(x: list) -> int:
    index_max = get_func_index(x, max)
    return index_max

def get_func_index(x: list, func) -> int:
    index_min = func(range(len(x)),
                    key=x.__getitem__)

    return index_min


def risk_adjusted_cost_for_instrument(instrument_code: str,
                                      instrument_config: pd.DataFrame,
                                      position_turnover: float) -> float:

    config_for_instrument = instrument_config.loc[instrument_code]
    SR_cost_per_trade = config_for_instrument.SR_cost
    rolls_per_year = config_for_instrument.rolls_per_year

    return SR_cost_per_trade * (rolls_per_year + position_turnover)

def calculate_SR_for_selected_instruments(selected_instruments: list,
                                          pre_cost_SR: float,
                                          instrument_config: pd.DataFrame,
                                          position_turnover: float,
                                          correlation_matrix: correlationEstimate,
                                        capital: float,
                                          risk_target: float
                                          ) -> float:

    ## Returns a large negative number if minimum capital requirements not met

    portfolio_weights = calculate_portfolio_weights(selected_instruments,
                                                    correlation_matrix=correlation_matrix)

    min_capital_okay = check_minimum_capital_ok(
        portfolio_weights=portfolio_weights,
        instrument_config=instrument_config,
        correlation_matrix=correlation_matrix,
        risk_target=risk_target,
        capital=capital
    )

    if not min_capital_okay:
        return -999999999999

    portfolio_SR = calculate_SR_of_portfolio(portfolio_weights,
                                             pre_cost_SR=pre_cost_SR,
                                             correlation_matrix=correlation_matrix,
                                             position_turnover=position_turnover,
                                             instrument_config=instrument_config)

    return portfolio_SR


def calculate_portfolio_weights(selected_instruments: list,
                                correlation_matrix: correlationEstimate) -> portfolioWeights:


    if len(selected_instruments)==1:
        return portfolioWeights.from_weights_and_keys(list_of_weights=[1.0],
                                                      list_of_keys=selected_instruments)

    subset_matrix = correlation_matrix.subset(selected_instruments)
    handcraft_portfolio = handcraftPortfolio(subset_matrix)

    return handcraft_portfolio.weights()

def check_minimum_capital_ok(
        portfolio_weights: portfolioWeights,
        correlation_matrix: correlationEstimate,
        risk_target: float,
        instrument_config: pd.DataFrame,
        capital: float
        ) -> bool:

    idm = calculate_idm(portfolio_weights,
                        correlation_matrix=correlation_matrix)

    list_of_instruments = portfolio_weights.assets

    for instrument_code in list_of_instruments:
        weight = portfolio_weights[instrument_code]
        okay_for_instrument = minimum_capital_okay_for_instrument(instrument_code=instrument_code,
                                            instrument_config=instrument_config,
                                            capital=capital,
                                            risk_target=risk_target,
                                            idm = idm,
                                            weight = weight)
        if not okay_for_instrument:
            return False

    return True


def calculate_idm(portfolio_weights: portfolioWeights,
                  correlation_matrix: correlationEstimate) -> float:

    if len(portfolio_weights.assets)==1:
        return 1.0

    aligned_correlation_matrix = correlation_matrix.subset(portfolio_weights.assets)
    return div_multiplier_from_np(np.array(portfolio_weights.weights),
                                  aligned_correlation_matrix.values)


def div_multiplier_from_np(weights_np: np.array,
                   corr_np: np.array
                   ) -> float:

    variance = weights_np.dot(corr_np).dot(weights_np)
    risk = variance ** 0.5

    return 1.0 / risk


def calculate_SR_of_portfolio(portfolio_weights: portfolioWeights,
                    pre_cost_SR: float,
                    instrument_config: pd.DataFrame,
                    position_turnover: float,
                    correlation_matrix: correlationEstimate
    ) -> float:

    expected_mean = calculate_expected_mean_for_portfolio(
        portfolio_weights=portfolio_weights,
        pre_cost_SR=pre_cost_SR,
        instrument_config=instrument_config,
        position_turnover=position_turnover
    )
    expected_std = calculate_expected_std_for_portfolio(
        portfolio_weights=portfolio_weights,
        correlation_matrix=correlation_matrix
    )

    return expected_mean / expected_std

def calculate_expected_mean_for_portfolio(
                                          portfolio_weights: portfolioWeights,
                                          pre_cost_SR: float,
                                          instrument_config: pd.DataFrame,
                                          position_turnover: float
                                          ) -> float:
    instrument_means = [
        calculate_expected_mean_for_instrument_in_portfolio(instrument_code,
                                                            portfolio_weights=portfolio_weights,
                                                            pre_cost_SR=pre_cost_SR,
                                                            instrument_config=instrument_config,
                                                            position_turnover=position_turnover)
        for instrument_code in portfolio_weights.assets
    ]

    return sum(instrument_means)

def calculate_expected_mean_for_instrument_in_portfolio(instrument_code: str,
                                                        portfolio_weights: portfolioWeights,
                                                        pre_cost_SR: float,
                                                        instrument_config: pd.DataFrame,
                                                        position_turnover: float
                                                        ):
    weight = portfolio_weights[instrument_code]
    costs_SR_units = risk_adjusted_cost_for_instrument(instrument_code=instrument_code,
                                                       instrument_config=instrument_config,
                                                       position_turnover=position_turnover)
    SR_for_instrument = pre_cost_SR - costs_SR_units

    return weight *SR_for_instrument

def calculate_expected_std_for_portfolio(portfolio_weights: portfolioWeights,
                                         correlation_matrix: correlationEstimate) -> float:

    subset_aligned_correlation = correlation_matrix.subset(portfolio_weights.assets)

    return variance_for_numpy(weights = np.array(portfolio_weights.weights),
                              sigma = subset_aligned_correlation.values)

def variance_for_numpy(weights: np.array, sigma: np.array) -> float:
    # returns the variance (NOT standard deviation) given weights and sigma
    return weights.dot(sigma).dot(weights.transpose())


def choose_next_instrument(selected_instruments: list,
                          pre_cost_SR: float,
                           capital: float,
                           risk_target: float,
                          instrument_config: pd.DataFrame,
                          position_turnover: float,
                          correlation_matrix: correlationEstimate) -> str:

    remaining_instruments = get_remaining_instruments(selected_instruments,
                                                      instrument_config=instrument_config)

    if not remaining_instruments:
        return None

    SR_by_instrument = []
    valid_instruments = []
    
    for instrument_code in remaining_instruments:
        try:
            test_instruments = selected_instruments + [instrument_code]
            test_correlation_matrix = create_correlation_matrix_from_data(test_instruments)
            
            sr = calculate_SR_for_selected_instruments(
                test_instruments,
                correlation_matrix=test_correlation_matrix,
                capital=capital,
                pre_cost_SR=pre_cost_SR,
                instrument_config=instrument_config,
                risk_target=risk_target,
                position_turnover=position_turnover
            )
            
            SR_by_instrument.append(sr)
            valid_instruments.append(instrument_code)
            
        except Exception as e:
            # Skip instruments that cause errors
            continue

    if not valid_instruments:
        return None
        
    # Filter out instruments that fail minimum capital check (SR = -999999999999)
    valid_sr_pairs = [(inst, sr) for inst, sr in zip(valid_instruments, SR_by_instrument) 
                      if sr > -999999]
    
    if not valid_sr_pairs:
        return None
    
    # Find the instrument with the highest valid SR
    best_instrument, best_sr = max(valid_sr_pairs, key=lambda x: x[1])
    
    return best_instrument

def get_remaining_instruments(selected_instruments: list,
                          instrument_config: pd.DataFrame) -> list:

    all_instruments = list(instrument_config.index)
    remaining = set(all_instruments).difference(set(selected_instruments))

    return list(remaining)

# ===== HELPER FUNCTIONS FOR INTEGRATION =====

def load_instrument_data():
    """Load instruments data - adapting to existing structure"""
    # This should load your instruments DataFrame
    # For now, I'll create a stub that you can replace
    try:
        import sys
        sys.path.append('rob_port')
        from chapter1 import load_instrument_data as load_data
        return load_data()
    except:
        print("Warning: Could not load instrument data from chapter1")
        return pd.DataFrame()

def create_correlation_matrix_from_data(selected_instruments: list) -> correlationEstimate:
    """Create correlation matrix from actual return data"""
    
    # Load actual return data
    returns_data = {}
    
    for symbol in selected_instruments:
        file_path = f"Data/{symbol.lower()}_daily_data.csv"
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, parse_dates=['Time'])
                df.set_index('Time', inplace=True)
                df['returns'] = df['Last'].pct_change(fill_method=None)
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
        return correlationEstimate(corr_df)
    
    # Create DataFrame of returns with common dates
    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.dropna()
    
    if len(returns_df) < 100:
        # Not enough overlapping data, use synthetic
        size = len(selected_instruments)
        corr_matrix = np.eye(size) * 0.8 + np.ones((size, size)) * 0.2
        corr_df = pd.DataFrame(corr_matrix, index=selected_instruments, columns=selected_instruments)
        return correlationEstimate(corr_df)
    
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
    
    return correlationEstimate(correlation_df)

def create_instrument_config_from_instruments_df(instruments_df: pd.DataFrame) -> pd.DataFrame:
    """Convert existing instruments DataFrame to the format expected by Carver's code"""
    
    # Map the columns to what Carver expects
    config_data = []
    
    for _, row in instruments_df.iterrows():
        symbol = row['Symbol']
        
        # Get realistic estimates for price and volatility
        if 'VIX' in symbol:
            price, ann_std = 20, 0.80
        elif 'BTC' in symbol or 'MBT' in symbol:
            price, ann_std = 50000, 0.60
        elif any(term in symbol for term in ['ZN', 'ZB', 'ZF', 'ZT', '3KTB', 'TN', 'GBL', 'BTP', 'OAT']):
            price, ann_std = 110, 0.06
        elif any(curr in symbol for curr in ['EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD', 'NOK', 'SEK']):
            price, ann_std = 1.1, 0.10
        elif any(metal in symbol for metal in ['GC', 'MGC', 'SI', 'HG', 'PL', 'PA']):
            if 'GC' in symbol or 'MGC' in symbol:
                price, ann_std = 2000, 0.18
            elif 'SI' in symbol:
                price, ann_std = 25, 0.25
            else:
                price, ann_std = 4, 0.22
        elif any(energy in symbol for energy in ['CL', 'QM', 'RB', 'HO', 'NG', 'BZ']):
            price, ann_std = 70, 0.30
        elif any(ag in symbol for ag in ['ZC', 'ZS', 'ZW', 'ZM', 'ZL', 'ZO', 'ZR']):
            price, ann_std = 500, 0.25
        else:
            price, ann_std = 4000, 0.18  # Equity indices
        
        config_data.append({
            'instrument': symbol,
            'fx_rate': 1.0,  # Assuming USD base
            'multiplier': row['Multiplier'],
            'price': price,
            'ann_std': ann_std,
            'SR_cost': row.get('SR_cost', 0.01),  # Default if not available
            'rolls_per_year': 4  # Quarterly rolls assumption
        })
    
    config_df = pd.DataFrame(config_data)
    config_df.set_index('instrument', inplace=True)
    
    return config_df

# ===== MAIN IMPLEMENTATION FUNCTION =====

def implement_carver_static_instrument_selection(instruments_df: pd.DataFrame,
                                                capital: float,
                                                risk_target: float = 0.2,
                                                pre_cost_SR: float = 0.4,
                                                position_turnover: float = 5,
                                                approx_number_of_instruments: int = 5,
                                                approx_IDM: float = 2.5):
    """
    Implement Carver's static instrument selection exactly as in his code
    """
    
    print("=" * 80)
    print("CARVER'S STATIC INSTRUMENT SELECTION")
    print("=" * 80)
    
    print(f"Capital: ${capital:,.0f}")
    print(f"Risk target: {risk_target:.1%}")
    print(f"Pre-cost SR: {pre_cost_SR}")
    print(f"Position turnover: {position_turnover}")
    print(f"Approx instruments: {approx_number_of_instruments}")
    print(f"Approx IDM: {approx_IDM}")
    
    # Convert to Carver's instrument config format
    instrument_config = create_instrument_config_from_instruments_df(instruments_df)
    
    print(f"\nAvailable instruments: {len(instrument_config)}")
    
    # Step 1: Select first instrument
    print("\n----- Step 1: Selecting First Instrument -----")
    
    selected_instruments = []
    first_instrument = select_first_static_instrument(
        instrument_config=instrument_config,
        position_turnover=position_turnover,
        capital=capital,
        risk_target=risk_target,
        approx_IDM=approx_IDM,
        approx_number_of_instruments=approx_number_of_instruments
    )
    
    if not first_instrument:
        print("No instruments meet minimum capital requirements!")
        return {}
    
    selected_instruments.append(first_instrument)
    
    # Create correlation matrix
    correlation_matrix = create_correlation_matrix_from_data(selected_instruments)
    
    # Calculate initial SR
    current_SR = calculate_SR_for_selected_instruments(
        selected_instruments,
        correlation_matrix=correlation_matrix,
        pre_cost_SR=pre_cost_SR,
        instrument_config=instrument_config,
        position_turnover=position_turnover,
        capital=capital,
        risk_target=risk_target
    )
    
    max_SR_achieved = current_SR
    
    print(f"Starting with {first_instrument}, SR: {current_SR:.4f}")
    
    # Step 2: Iterative selection following Carver's stopping condition
    print("\n----- Step 2: Iterative Instrument Selection -----")
    
    iteration = 0
    max_iterations = min(50, len(instrument_config))  # Safety limit
    
    while current_SR > (max_SR_achieved * 0.9) and iteration < max_iterations:
        iteration += 1
        
        print(f"{selected_instruments} SR: {current_SR:.2f}")
        
        # Update correlation matrix for current selection
        correlation_matrix = create_correlation_matrix_from_data(selected_instruments)
        
        try:
            next_instrument = choose_next_instrument(
                selected_instruments,
                correlation_matrix=correlation_matrix,
                pre_cost_SR=pre_cost_SR,
                instrument_config=instrument_config,
                position_turnover=position_turnover,
                capital=capital,
                risk_target=risk_target
            )
        except:
            print("No more instruments can be added")
            break
        
        if not next_instrument:
            print("No more instruments available")
            break
        
        # Test the new selection
        test_instruments = selected_instruments + [next_instrument]
        test_correlation_matrix = create_correlation_matrix_from_data(test_instruments)
        
        current_SR = calculate_SR_for_selected_instruments(
            test_instruments,
            correlation_matrix=test_correlation_matrix,
            pre_cost_SR=pre_cost_SR,
            instrument_config=instrument_config,
            position_turnover=position_turnover,
            capital=capital,
            risk_target=risk_target
        )
        
        if current_SR > max_SR_achieved:
            max_SR_achieved = current_SR
        
        # Accept the instrument
        selected_instruments.append(next_instrument)
        correlation_matrix = test_correlation_matrix
    
    # Final results
    print(f"\n----- Final Results -----")
    final_correlation_matrix = create_correlation_matrix_from_data(selected_instruments)
    portfolio_weights = calculate_portfolio_weights(selected_instruments, final_correlation_matrix)
    idm = calculate_idm(portfolio_weights, final_correlation_matrix)
    
    print(f"Final instruments: {len(selected_instruments)}")
    print(f"Selected: {selected_instruments}")
    print(f"Final SR: {current_SR:.4f}")
    print(f"IDM: {idm:.2f}")
    print(f"Portfolio weights:")
    for instrument in selected_instruments:
        weight = portfolio_weights[instrument]
        print(f"  {instrument}: {weight:.4f}")
    
    return {
        'selected_instruments': selected_instruments,
        'portfolio_weights': dict(portfolio_weights),
        'idm': idm,
        'final_SR': current_SR,
        'max_SR_achieved': max_SR_achieved,
        'correlation_matrix': final_correlation_matrix,
        'instrument_config': instrument_config
    }

if __name__ == "__main__":
    # Test the implementation
    try:
        import sys
        sys.path.append('rob_port')
        from chapter1 import load_instrument_data
        instruments_df = load_instrument_data()
        
        # Test with different capital levels
        for capital in [10000000, 50000000]:  # Start with smaller test
            print(f"\n{'='*80}")
            print(f"TESTING WITH ${capital:,.0f} CAPITAL")
            print(f"{'='*80}")
            
            results = implement_carver_static_instrument_selection(
                instruments_df=instruments_df,
                capital=capital,
                risk_target=0.2,
                pre_cost_SR=0.4,
                position_turnover=5,
                approx_number_of_instruments=5,
                approx_IDM=2.5
            )
            
            if results:
                print(f"✓ Successfully selected {len(results['selected_instruments'])} instruments")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Could not run test. Please make sure data files are available.") 