from chapter3 import *
import pandas as pd
import numpy as np
from scipy.cluster import hierarchy as sch
import os
import matplotlib.pyplot as plt

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
    return [item for sublist in some_list for item in sublist]

def one_over_n_weights_given_asset_names(list_of_asset_names: list) -> portfolioWeights:
    weight = 1.0 / len(list_of_asset_names)
    return portfolioWeights(
        [(asset_name, weight) for asset_name in list_of_asset_names]
    )

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

def cluster_correlation_matrix(corr_matrix: correlationEstimate,
                             cluster_size: int = 2):
    clusters = get_list_of_clusters_for_correlation_matrix(corr_matrix,
                                                          cluster_size=cluster_size)
    clusters_as_names = from_cluster_index_to_asset_names(clusters, corr_matrix)
    return clusters_as_names

def get_list_of_clusters_for_correlation_matrix(corr_matrix: np.array,
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
                                                    cluster_size=cluster_size)
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

def create_sub_portfolios_from_portfolio(handcraft_portfolio: handcraftPortfolio):
    clusters_as_names = cluster_correlation_matrix(handcraft_portfolio.correlation)
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

def calculate_portfolio_weights(selected_instruments: list,
                              correlation_matrix: correlationEstimate) -> portfolioWeights:
    if len(selected_instruments)==1:
        return portfolioWeights.from_weights_and_keys(list_of_weights=[1.0],
                                                    list_of_keys=selected_instruments)

    subset_matrix = correlation_matrix.subset(selected_instruments)
    handcraft_portfolio = handcraftPortfolio(subset_matrix)
    return handcraft_portfolio.weights()

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
    # Set costs to zero temporarily
    costs_SR_units = 0.0
    SR_for_instrument = pre_cost_SR - costs_SR_units
    return weight * SR_for_instrument

def calculate_expected_std_for_portfolio(portfolio_weights: portfolioWeights,
                                       correlation_matrix: correlationEstimate) -> float:
    subset_aligned_correlation = correlation_matrix.subset(portfolio_weights.assets)
    return variance_for_numpy(weights = np.array(portfolio_weights.weights),
                            sigma = subset_aligned_correlation.values)

def variance_for_numpy(weights: np.array, sigma: np.array) -> float:
    return weights.dot(sigma).dot(weights.transpose())

def risk_adjusted_cost_for_instrument(instrument_code: str,
                                    instrument_config: pd.DataFrame,
                                    position_turnover: float) -> float:
    """
    Calculate the Sharpe ratio cost for an instrument based on commission and spread.
    Temporarily set to zero to exclude costs from calculations.
    """
    return 0.0

def get_remaining_instruments(selected_instruments: list,
                            instrument_config: pd.DataFrame) -> list:
    all_instruments = list(instrument_config.index)
    remaining = set(all_instruments).difference(set(selected_instruments))
    return list(remaining)

def choose_next_instrument(selected_instruments: list,
                         pre_cost_SR: float,
                         capital: float,
                         risk_target: float,
                         instrument_config: pd.DataFrame,
                         position_turnover: float,
                         correlation_matrix: correlationEstimate) -> str:
    remaining_instruments = get_remaining_instruments(selected_instruments,
                                                    instrument_config=instrument_config)

    SR_by_instrument = [
        calculate_SR_for_selected_instruments(selected_instruments+[instrument_code],
                                            correlation_matrix=correlation_matrix,
                                            capital=capital,
                                            pre_cost_SR=pre_cost_SR,
                                            instrument_config=instrument_config,
                                            risk_target=risk_target,
                                            position_turnover=position_turnover)
        for instrument_code in remaining_instruments
    ]

    index_of_max_SR = get_max_index(SR_by_instrument)
    return remaining_instruments[index_of_max_SR]

def get_max_index(x: list) -> int:
    return get_func_index(x, max)

def get_min_index(x: list) -> int:
    return get_func_index(x, min)

def get_func_index(x: list, func) -> int:
    return func(range(len(x)), key=x.__getitem__)

def calculate_SR_for_selected_instruments(selected_instruments: list,
                                        pre_cost_SR: float,
                                        instrument_config: pd.DataFrame,
                                        position_turnover: float,
                                        correlation_matrix: correlationEstimate,
                                        capital: float,
                                        risk_target: float
                                        ) -> float:
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
                                                                idm=idm,
                                                                weight=weight)
        if not okay_for_instrument:
            return False

    return True

def minimum_capital_okay_for_instrument(instrument_code: str,
                                      instrument_config: pd.DataFrame,
                                      idm: float,
                                      weight: float,
                                      risk_target: float,
                                      capital: float) -> bool:
    config_for_instrument = instrument_config.loc[instrument_code]
    
    # Get actual price and calculate volatility
    price = config_for_instrument['price']
    
    # Read the full data file to get price series
    try:
        df = pd.read_csv(f'Data/{instrument_code}_daily_data.csv', parse_dates=['Time'])
        df.set_index('Time', inplace=True)
        
        # Calculate variable standard deviation using the function from chapter3
        sigma_pct = calculate_variable_standard_deviation_for_risk_targeting(
            adjusted_price=df['Last'],
            current_price=df['Last'],
            use_perc_returns=True,
            annualise_stdev=True
        ).iloc[-1]  # Get the most recent volatility estimate
        
        # Get FX rate based on currency
        currency = config_for_instrument['Currency']
        fx_rate = get_fx_rate(currency)
        
        minimum_capital = calculate_minimum_capital(
            multiplier=config_for_instrument['Multiplier'],
            price=price,
            fx_rate=fx_rate,
            sigma_pct=sigma_pct,
            idm=idm,
            weight=weight,
            risk_target=risk_target
        )
        
        if minimum_capital > capital:
            print(f"\nInstrument {instrument_code} rejected due to capital requirements:")
            print(f"Required capital: ${minimum_capital:,.2f}")
            print(f"Available capital: ${capital:,.2f}")
            print(f"Multiplier: {config_for_instrument['Multiplier']}")
            print(f"Price: ${price:,.2f}")
            print(f"Volatility: {sigma_pct:.2%}")
            print(f"FX Rate: {fx_rate:.4f}")
            print(f"IDM: {idm:.2f}")
            print(f"Weight: {weight:.2%}")
            print(f"Risk Target: {risk_target:.2%}")
        
        return minimum_capital <= capital
    except FileNotFoundError:
        print(f"No data found for {instrument_code}")
        return False

def get_fx_rate(currency: str) -> float:
    """
    Get the current FX rate for a given currency.
    For USD-denominated instruments, return 1.0.
    For other currencies, you'll need to implement actual FX rate retrieval.
    
    Parameters:
    - currency (str): The currency code (e.g., 'USD', 'EUR', 'JPY')
    
    Returns:
    - float: The FX rate relative to USD
    """
    if currency == 'USD':
        return 1.0
    elif currency == 'EUR':
        return 1.08  # Example rate, replace with actual data
    elif currency == 'JPY':
        return 0.0067  # Example rate, replace with actual data
    elif currency == 'GBP':
        return 1.26  # Example rate, replace with actual data
    elif currency == 'CNH':
        return 0.14  # Example rate, replace with actual data
    elif currency == 'KRW':
        return 0.00075  # Example rate, replace with actual data
    elif currency == 'SGD':
        return 0.74  # Example rate, replace with actual data
    else:
        print(f"Warning: No FX rate found for {currency}, using 1.0")
        return 1.0

def read_instrument_data(symbol):
    """
    Read daily data for a given instrument symbol.
    
    Parameters:
    - symbol (str): The instrument symbol
    
    Returns:
    - tuple: (pd.Series, float) - (returns series, last price)
    """
    try:
        df = pd.read_csv(f'Data/{symbol}_daily_data.csv', parse_dates=['Time'])
        df.set_index('Time', inplace=True)
        df['returns'] = df['Last'].pct_change(fill_method=None)
        return df['returns'], df['Last'].iloc[-1]  # Return both returns and last price
    except FileNotFoundError:
        print(f"No data found for {symbol}")
        return None, None

def create_correlation_matrix(instruments_df):
    """
    Create correlation matrix from instrument returns.
    
    Parameters:
    - instruments_df (pd.DataFrame): DataFrame with returns for all instruments
    
    Returns:
    - pd.DataFrame: Correlation matrix
    """
    return instruments_df.corr()

def calculate_minimum_capital(multiplier, price, fx_rate, sigma_pct, idm, weight, risk_target):
    """
    Calculate minimum capital required for an instrument.
    
    Parameters:
    - multiplier (float): Contract multiplier
    - price (float): Current price
    - fx_rate (float): FX conversion rate
    - sigma_pct (float): Annualized standard deviation
    - idm (float): Instrument Diversification Multiplier
    - weight (float): Portfolio weight
    - risk_target (float): Risk target
    
    Returns:
    - float: Minimum capital required
    """
    numerator = 4 * multiplier * price * fx_rate * sigma_pct
    denominator = idm * weight * risk_target
    return numerator / denominator

def calculate_risk_adjusted_cost(commission, spread_points, multiplier, price, annualized_std_dev, rolls_per_year, turnover):
    """
    Calculate risk-adjusted cost for an instrument.
    
    Parameters:
    - commission (float): Commission per trade
    - spread_points (float): Bid-ask spread in points
    - multiplier (float): Contract multiplier
    - price (float): Current price
    - annualized_std_dev (float): Annualized standard deviation
    - rolls_per_year (int): Number of rolls per year
    - turnover (int): Number of trades per year
    
    Returns:
    - float: Risk-adjusted cost in Sharpe ratio units
    """
    spread_cost = multiplier * (spread_points / 2)
    total_cost = spread_cost + commission
    cost_percent = total_cost / (price * multiplier)
    risk_adjusted_cost = cost_percent / annualized_std_dev
    
    return risk_adjusted_cost * (rolls_per_year + turnover)

def calculate_SR_cost_for_instrument(instrument_code: str,
                                   instrument_config: pd.DataFrame,
                                   position_turnover: float) -> float:
    """
    Calculate the Sharpe ratio cost for an instrument based on commission and spread.
    Temporarily set to zero to exclude costs from calculations.
    """
    return 0.0

def select_instruments(capital, risk_target, approx_number_of_instruments, approx_idm, position_turnover, pre_cost_sr):
    """
    Select instruments using the handcrafting algorithm.
    """
    print(f"\nStarting instrument selection with:")
    print(f"Capital: ${capital:,.2f}")
    print(f"Risk Target: {risk_target:.2%}")
    print(f"Pre-cost SR: {pre_cost_sr:.2f}")
    print(f"Position Turnover: {position_turnover}")
    
    # Read instruments configuration
    instruments_df = pd.read_csv('Data/instruments.csv')
    instruments_df.set_index('Symbol', inplace=True)
    
    # Create returns DataFrame and price dictionary for all instruments
    returns_dict = {}
    price_dict = {}
    for symbol in instruments_df.index:
        returns, price = read_instrument_data(symbol)
        if returns is not None:
            returns_dict[symbol] = returns
            price_dict[symbol] = price
    
    # Filter instruments_df to only include instruments with data
    instruments_with_data = list(returns_dict.keys())
    instruments_df = instruments_df.loc[instruments_with_data]
    
    print(f"\nFound {len(instruments_with_data)} instruments with data")
    
    # Add price and rolls_per_year columns
    instruments_df['price'] = instruments_df.index.map(price_dict)
    instruments_df['rolls_per_year'] = 4
    
    # Add SR_cost column to instruments_df
    instruments_df['SR_cost'] = instruments_df.index.map(
        lambda x: calculate_SR_cost_for_instrument(x, instruments_df, position_turnover)
    )
    
    # Create returns DataFrame
    returns_df = pd.DataFrame(returns_dict)
    returns_df.dropna(inplace=True)
    
    # Calculate correlation matrix
    corr_matrix = correlationEstimate(returns_df.corr())
    
    # Select first instrument based on minimum capital and risk-adjusted cost
    selected_instruments = []
    approx_initial_weight = 1.0 / approx_number_of_instruments
    
    print("\nSelecting first instrument...")
    # Select first instrument
    first_instrument = choose_next_instrument(
        selected_instruments=[],
        pre_cost_SR=pre_cost_sr,
        capital=capital,
        risk_target=risk_target,
        instrument_config=instruments_df,
        position_turnover=position_turnover,
        correlation_matrix=corr_matrix
    )
    selected_instruments.append(first_instrument)
    
    # Calculate initial SR
    current_SR = calculate_SR_for_selected_instruments(
        selected_instruments,
        correlation_matrix=corr_matrix,
        pre_cost_SR=pre_cost_sr,
        instrument_config=instruments_df,
        position_turnover=position_turnover,
        capital=capital,
        risk_target=risk_target
    )
    
    max_SR_achieved = current_SR
    print(f"\nFirst instrument selected: {first_instrument}")
    print(f"Initial Sharpe Ratio: {current_SR:.2f}")
    
    # Continue selecting instruments until no more improvements can be made
    iteration = 1
    improvement_found = True
    
    while improvement_found:
        print(f"\nIteration {iteration}:")
        print(f"Current portfolio: {selected_instruments}")
        print(f"Current SR: {current_SR:.2f}")
        print(f"Max SR achieved: {max_SR_achieved:.2f}")
        print(f"Threshold (90% of max): {max_SR_achieved * 0.9:.2f}")
        
        # Get all remaining instruments
        remaining_instruments = get_remaining_instruments(selected_instruments, instruments_df)
        print(f"Remaining instruments to try: {len(remaining_instruments)}")
        
        # Try each remaining instrument
        best_next_instrument = None
        best_next_SR = current_SR
        
        for next_instrument in remaining_instruments:
            print(f"\nTrying to add: {next_instrument}")
            
            # Calculate SR with the new instrument
            test_instruments = selected_instruments + [next_instrument]
            test_SR = calculate_SR_for_selected_instruments(
                test_instruments,
                correlation_matrix=corr_matrix,
                pre_cost_SR=pre_cost_sr,
                instrument_config=instruments_df,
                position_turnover=position_turnover,
                capital=capital,
                risk_target=risk_target
            )
            
            print(f"SR with new instrument: {test_SR:.2f}")
            
            if test_SR > best_next_SR:
                best_next_SR = test_SR
                best_next_instrument = next_instrument
                print(f"New best SR found: {best_next_SR:.2f}")
        
        # If we found an improvement, add the best instrument
        if best_next_instrument is not None and best_next_SR > current_SR:
            selected_instruments.append(best_next_instrument)
            current_SR = best_next_SR
            if current_SR > max_SR_achieved:
                max_SR_achieved = current_SR
                print(f"New max SR achieved: {max_SR_achieved:.2f}")
        else:
            improvement_found = False
            print("\nNo more improvements found with remaining instruments")
        
        iteration += 1
    
    print(f"\nSelection process stopped after {len(selected_instruments)} instruments")
    print(f"Final portfolio: {selected_instruments}")
    print(f"Final Sharpe Ratio: {current_SR:.2f}")
    print(f"Maximum Sharpe Ratio achieved: {max_SR_achieved:.2f}")
    
    return selected_instruments

def get_asset_class(instrument_code: str, instrument_config: pd.DataFrame) -> str:
    """
    Categorize instruments into asset classes based on the book's tables.
    
    Asset Classes:
    1. Bonds - Government bonds and interest rate futures
    2. Equities - Stock indices (US, European, Asian)
    3. Volatility - VIX and VSTOXX
    4. FX - Major and cross-rate currency futures
    5. Metals - Including crypto
    6. Energies - Oil, gas, etc.
    7. Agricultural - Grains, meats, etc.
    """
    # Bonds and Interest Rates (Tables 172 & 173)
    if instrument_code in ['ZT', 'Z3N', 'ZF', 'ZN', 'TN', 'ZB', 'UB', 'LIW', 'N1U', 'GE',  # US
                          'OAT', 'GBS', 'GBM', 'GBL', 'GBX', 'BTS', 'BTP', 'JGB',           # European & Japanese
                          '3KTB', 'FLKTB', 'FBON']:                                          # Korean & Spanish
        return 'Bonds'
    
    # US Equity Indices (Table 174)
    elif instrument_code in ['MYM', 'MNQ', 'RSV', 'M2K', 'EMD', 'MES']:
        return 'US_Equity'
    
    # European Equity Indices and Sectors (Tables 175 & 176)
    elif instrument_code in ['EOE', 'CAC40', 'DAX', 'SMI', 'DJ200S', 'DJSD', 'DJ600', 'ESTX50',
                           'SXAP', 'SXPP', 'SXDP', 'SXIP', 'SXEP', 'SX8P', 'SXTP', 'SX6P']:
        return 'EU_Equity'
    
    # Asian Equity Indices (Table 177)
    elif instrument_code in ['M1MS', 'XINA50', 'XINO1', 'NIFTY', 'N225M', 'JPNK400',
                           'TSEMOTHR', 'MNTPX', 'KOSDQ150', 'K200', 'SSG', 'TWN']:
        return 'Asia_Equity'
    
    # Volatility (Table 178)
    elif instrument_code in ['VIX', 'V2TX']:
        return 'Volatility'
    
    # FX - Major (Table 179)
    elif instrument_code in ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NOK', 'NZD', 'SEK']:
        return 'Major_FX'
    
    # FX - Cross and EM (Table 180)
    elif instrument_code in ['RP', 'RY', 'BRE', 'UC', 'SIR', 'MXP', 'RUR', 'SND']:
        return 'EM_FX'
    
    # Metals and Crypto (Table 181)
    elif instrument_code in ['ALI', 'HG', 'MGC', 'SCI', 'PA', 'PL', 'SI', 'MBT', 'ETHUSDRR']:
        return 'Metals'
    
    # Energy (Table 182)
    elif instrument_code in ['BZ', 'QM', 'HH', 'RB', 'QG', 'HO']:
        return 'Energy'
    
    # Agricultural (Table 183)
    elif instrument_code in ['AIGCI', 'CSC', 'ZC', 'GF', 'HE', 'LE', 'ZO', 'KE', 'ZR',
                           'ZS', 'ZM', 'ZL', 'ZW']:
        return 'Agricultural'
    
    else:
        print(f"Warning: Unknown asset class for instrument {instrument_code}")
        return 'Other'

def get_idm_for_instrument_count(count: int) -> float:
    """
    Returns the IDM value based on number of instruments as per the book's table 16.
    """
    if count == 1:
        return 1.00
    elif count == 2:
        return 1.20
    elif count == 3:
        return 1.48
    elif count == 4:
        return 1.56
    elif count == 5:
        return 1.70
    elif count == 6:
        return 1.90
    elif count == 7:
        return 2.10
    elif count <= 14:
        return 2.20
    elif count <= 24:
        return 2.30
    elif count <= 29:
        return 2.40
    else:
        return 2.50

def handcraft_portfolio_weights(selected_instruments: list, instrument_config: pd.DataFrame) -> dict:
    """
    Implement the book's handcrafting method for portfolio weights:
    1. Group instruments by major asset class
    2. Give each major asset class equal weight
    3. Within each asset class, subdivide by region/type if applicable
    4. Within each subgroup, divide equally among instruments
    """
    # Group instruments by asset class
    instruments_by_class = {}
    for inst in selected_instruments:
        asset_class = get_asset_class(inst, instrument_config)
        if asset_class not in instruments_by_class:
            instruments_by_class[asset_class] = []
        instruments_by_class[asset_class].append(inst)
    
    # Remove 'Other' class if it exists and is empty
    if 'Other' in instruments_by_class and not instruments_by_class['Other']:
        del instruments_by_class['Other']
    
    # Calculate weights
    weights = {}
    n_classes = len(instruments_by_class)
    class_weight = 1.0 / n_classes
    
    # Special handling for equities - combine US, EU, and Asia
    equity_classes = ['US_Equity', 'EU_Equity', 'Asia_Equity']
    total_equity_instruments = sum(len(instruments_by_class.get(ec, [])) for ec in equity_classes)
    
    print("\nPortfolio Weight Allocation:")
    print(f"Number of asset classes: {n_classes}")
    
    for asset_class, instruments in instruments_by_class.items():
        if asset_class in equity_classes:
            # All equity instruments share the same class weight
            instrument_weight = class_weight / total_equity_instruments if total_equity_instruments > 0 else 0
        else:
            # Normal weight calculation for non-equity classes
            instrument_weight = class_weight / len(instruments)
        
        print(f"\n{asset_class} ({len(instruments)} instruments, {class_weight:.1%} class weight):")
        for inst in instruments:
            weights[inst] = instrument_weight
            print(f"  {inst}: {weights[inst]:.2%}")
    
    return weights

def calculate_portfolio_sr(selected_instruments: list,
                         weights: dict,
                         correlation_matrix: correlationEstimate,
                         pre_cost_sr: float,
                         position_turnover: float = 5.1) -> float:
    """
    Calculate the expected Sharpe ratio for a portfolio using the book's methodology.
    """
    if not selected_instruments:
        return 0.0
    
    # Get the subset correlation matrix for selected instruments
    subset_correlation = correlation_matrix.subset(selected_instruments)
    
    # Calculate portfolio variance using correlation matrix and weights
    weight_array = np.array([weights[inst] for inst in selected_instruments])
    portfolio_variance = weight_array.dot(subset_correlation.values).dot(weight_array)
    portfolio_vol = np.sqrt(portfolio_variance)
    
    # Calculate expected return (pre-cost SR adjusted for costs)
    costs = 0.0  # Temporarily set to 0 as per current implementation
    portfolio_sr = pre_cost_sr - costs
    
    return portfolio_sr / portfolio_vol

def select_instruments_v2(capital: float,
                        risk_target: float,
                        instrument_config: pd.DataFrame,
                        correlation_matrix: correlationEstimate,
                        pre_cost_sr: float = 0.3,
                        position_turnover: float = 5.1) -> list:
    """
    Implement the book's instrument selection algorithm
    """
    print("\nStarting Portfolio Construction:")
    print(f"Capital: ${capital:,.0f}")
    print(f"Risk Target: {risk_target:.1%}")
    print(f"Pre-cost SR: {pre_cost_sr:.2f}")
    
    selected_instruments = []
    max_sr = 0.0
    current_sr = 0.0
    
    # Get all available instruments
    all_instruments = list(instrument_config.index)
    
    iteration = 0
    while True:
        iteration += 1
        best_new_sr = current_sr
        best_instrument = None
        
        # Try each remaining instrument
        remaining = set(all_instruments) - set(selected_instruments)
        for instrument in remaining:
            test_portfolio = selected_instruments + [instrument]
            test_weights = handcraft_portfolio_weights(test_portfolio, instrument_config)
            
            # Calculate SR for test portfolio
            test_sr = calculate_portfolio_sr(
                test_portfolio,
                test_weights,
                correlation_matrix,
                pre_cost_sr,
                position_turnover
            )
            
            if test_sr > best_new_sr:
                best_new_sr = test_sr
                best_instrument = instrument
        
        # If we found an improvement
        if best_instrument and best_new_sr > current_sr:
            selected_instruments.append(best_instrument)
            current_sr = best_new_sr
            if current_sr > max_sr:
                max_sr = current_sr
            print(f"\nAdded {best_instrument} (SR: {current_sr:.3f})")
        else:
            print("\nNo further improvements found")
            break
        
        # Check if SR has declined by more than 10% from maximum
        if current_sr < 0.9 * max_sr:
            print(f"\nStopping: SR ({current_sr:.3f}) has declined more than 10% from maximum ({max_sr:.3f})")
            break
    
    print("\nFinal Portfolio Summary:")
    print(f"Number of instruments: {len(selected_instruments)}")
    print(f"Final SR: {current_sr:.3f}")
    print(f"Maximum SR achieved: {max_sr:.3f}")
    print("\nSelected Instruments by Asset Class:")
    
    # Group and display instruments by asset class
    instruments_by_class = {}
    for inst in selected_instruments:
        asset_class = get_asset_class(inst, instrument_config)
        if asset_class not in instruments_by_class:
            instruments_by_class[asset_class] = []
        instruments_by_class[asset_class].append(inst)
    
    for asset_class, instruments in instruments_by_class.items():
        print(f"{asset_class}: {', '.join(instruments)}")
    
    return selected_instruments

def load_all_instrument_data(instrument_config: pd.DataFrame) -> tuple:
    """
    Load price and return data for all instruments in the config.
    
    Returns:
    - tuple: (price_data: pd.DataFrame, returns_data: pd.DataFrame, instruments_with_data: list)
    """
    print("\nLoading instrument data:")
    price_data = {}
    returns_data = {}
    instruments_with_data = []
    
    for symbol in instrument_config.index:
        try:
            df = pd.read_csv(f'Data/{symbol}_daily_data.csv', parse_dates=['Time'])
            df.set_index('Time', inplace=True)
            
            # Store price and calculate returns
            price_data[symbol] = df['Last']
            returns_data[symbol] = df['Last'].pct_change()
            
            instruments_with_data.append(symbol)
            print(f"Loaded data for {symbol}: {len(df)} rows")
            
        except FileNotFoundError:
            print(f"No data found for {symbol}")
            continue
        except Exception as e:
            print(f"Error loading {symbol}: {str(e)}")
            continue
    
    # Convert to DataFrames
    price_df = pd.DataFrame(price_data)
    returns_df = pd.DataFrame(returns_data)
    
    print(f"\nSuccessfully loaded data for {len(instruments_with_data)} instruments")
    print(f"Date range: {price_df.index[0]} to {price_df.index[-1]}")
    
    return price_df, returns_df, instruments_with_data

def calculate_position_sizes(capital: float,
                           selected_instruments: list,
                           weights: dict,
                           idm: float,
                           risk_target: float,
                           price_data: pd.DataFrame,
                           instrument_config: pd.DataFrame,
                           volatility_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate position sizes using the book's formula:
    Ni = Capital × IDM × Weighti × τ ÷ (Multiplieri × Pricei,t × FX ratei,t × σ_%i,t)
    
    Returns DataFrame with positions for each instrument over time.
    """
    print("\nCalculating position sizes:")
    positions = {}
    
    for instrument in selected_instruments:
        print(f"\nProcessing {instrument}:")
        
        # Get instrument config
        config = instrument_config.loc[instrument]
        multiplier = config['Multiplier']
        currency = config['Currency']
        
        # Get current FX rate (simplified - should be time series)
        fx_rate = get_fx_rate(currency)
        
        # Get weight
        weight = weights[instrument]
        
        # Calculate positions over time
        price_series = price_data[instrument]
        vol_series = volatility_data[instrument]
        
        positions[instrument] = (capital * idm * weight * risk_target / 
                               (multiplier * price_series * fx_rate * vol_series))
        
        # Round to nearest whole number
        positions[instrument] = positions[instrument].round()
        
        print(f"Average position size: {positions[instrument].mean():.1f}")
        print(f"Max position size: {positions[instrument].max():.0f}")
        print(f"Min position size: {positions[instrument].min():.0f}")
    
    positions_df = pd.DataFrame(positions)
    return positions_df

def calculate_portfolio_returns(positions: pd.DataFrame,
                              returns: pd.DataFrame,
                              price_data: pd.DataFrame,
                              instrument_config: pd.DataFrame,
                              capital: float) -> pd.DataFrame:
    """
    Calculate portfolio returns based on positions and instrument returns.
    """
    print("\nBacktest Results:")
    
    # Calculate returns for each instrument
    instrument_returns = {}
    for instrument in positions.columns:
        config = instrument_config.loc[instrument]
        multiplier = config['Multiplier']
        currency = config['Currency']
        fx_rate = get_fx_rate(currency)
        
        price_series = price_data[instrument]
        pos_series = positions[instrument].shift(1)
        returns_series = returns[instrument]
        
        pnl_series = pos_series * multiplier * price_series.shift(1) * fx_rate * returns_series
        instrument_returns[instrument] = pnl_series / capital
    
    portfolio_returns = pd.DataFrame(instrument_returns)
    portfolio_returns['total'] = portfolio_returns.sum(axis=1)
    
    # Calculate key statistics
    ann_factor = np.sqrt(256)  # Annualization factor
    mean_return = portfolio_returns['total'].mean() * 256
    vol = portfolio_returns['total'].std() * ann_factor
    sharpe = mean_return / vol if vol != 0 else 0
    
    # Calculate drawdown
    cum_returns = (1 + portfolio_returns['total']).cumprod()
    drawdown = 1 - cum_returns / cum_returns.cummax()
    max_drawdown = drawdown.max()
    
    print(f"Annualized Return: {mean_return:.1%}")
    print(f"Annualized Volatility: {vol:.1%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.1%}")
    
    return portfolio_returns

def run_strategy(capital: float = 50_000_000,
                risk_target: float = 0.20,
                pre_cost_sr: float = 0.3,
                position_turnover: float = 5.1):
    """
    Run the complete strategy as described in the book.
    """
    print(f"\nInitializing strategy with:")
    print(f"Capital: ${capital:,.0f}")
    print(f"Risk target: {risk_target:.1%}")
    print(f"Pre-cost SR: {pre_cost_sr:.2f}")
    print(f"Position turnover: {position_turnover:.1f}")
    
    # Load instrument configuration
    instrument_config = pd.read_csv('Data/instruments.csv')
    instrument_config.set_index('Symbol', inplace=True)
    
    # Load all instrument data
    price_data, returns_data, available_instruments = load_all_instrument_data(instrument_config)
    
    # Filter instrument_config to only include instruments with data
    instrument_config = instrument_config.loc[available_instruments]
    
    # Calculate correlation matrix
    correlation_matrix = correlationEstimate(returns_data.corr())
    
    # Select instruments using the book's methodology
    selected_instruments = select_instruments_v2(
        capital=capital,
        risk_target=risk_target,
        instrument_config=instrument_config,
        correlation_matrix=correlation_matrix,
        pre_cost_sr=pre_cost_sr,
        position_turnover=position_turnover
    )
    
    # Calculate portfolio weights using handcrafting method
    weights = handcraft_portfolio_weights(selected_instruments, instrument_config)
    
    # Get IDM based on number of instruments
    idm = get_idm_for_instrument_count(len(selected_instruments))
    print(f"\nUsing IDM: {idm:.2f} for {len(selected_instruments)} instruments")
    
    # Calculate volatility for each instrument
    volatility_data = pd.DataFrame()
    for inst in selected_instruments:
        volatility_data[inst] = calculate_variable_standard_deviation_for_risk_targeting(
            adjusted_price=price_data[inst],
            current_price=price_data[inst]
        )
    
    # Calculate position sizes
    positions = calculate_position_sizes(
        capital=capital,
        selected_instruments=selected_instruments,
        weights=weights,
        idm=idm,
        risk_target=risk_target,
        price_data=price_data,
        instrument_config=instrument_config,
        volatility_data=volatility_data
    )
    
    # Calculate portfolio returns
    portfolio_returns = calculate_portfolio_returns(
        positions=positions,
        returns=returns_data,
        price_data=price_data,
        instrument_config=instrument_config,
        capital=capital
    )
    
    # Calculate and plot equity curve
    equity_curve = (1 + portfolio_returns['total']).cumprod()
    
    print("\nStrategy Performance:")
    print(f"Final equity: {equity_curve.iloc[-1]:.2f}")
    print(f"Max drawdown: {(1 - equity_curve / equity_curve.cummax()).max()*100:.1f}%")
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    equity_curve.plot(title='Strategy Equity Curve')
    plt.grid(True)
    plt.show()
    
    return positions, portfolio_returns, equity_curve

if __name__ == "__main__":
    # Run the strategy with the book's parameters
    positions, returns, equity = run_strategy(
        capital=50_000_000,  # $50M as in the book's example
        risk_target=0.20,    # 20% annual risk target
        pre_cost_sr=0.3,     # Pre-cost Sharpe ratio assumption
        position_turnover=5.1 # From the book's tables
    )

