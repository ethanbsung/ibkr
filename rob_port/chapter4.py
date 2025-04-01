from chapter3 import *
import pandas as pd
import numpy as np
from scipy.cluster import hierarchy as sch
import os

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
    costs_SR_units = risk_adjusted_cost_for_instrument(instrument_code=instrument_code,
                                                      instrument_config=instrument_config,
                                                      position_turnover=position_turnover)
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
    
    Parameters:
    - instrument_code (str): The instrument symbol
    - instrument_config (pd.DataFrame): DataFrame with instrument configurations
    - position_turnover (float): Position turnover per year
    
    Returns:
    - float: Sharpe ratio cost per trade
    """
    # Default values for commission and spread
    commission = 1.24  # Default commission per trade
    spread_points = 0.625  # Default spread in points
    
    # Get instrument multiplier and current price
    multiplier = instrument_config.loc[instrument_code, 'Multiplier']
    price = 1000  # Default price - you should replace this with actual price data
    
    # Calculate risk-adjusted cost
    spread_cost = multiplier * (spread_points / 2)
    total_cost = spread_cost + commission
    cost_percent = total_cost / (price * multiplier)
    
    # Get rolls_per_year from config or use default
    rolls_per_year = instrument_config.loc[instrument_code, 'rolls_per_year']
    
    # Annualize the cost
    annual_cost = cost_percent * (rolls_per_year + position_turnover)
    
    return annual_cost

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
    minimum_capital = calculate_minimum_capital(
        multiplier=config_for_instrument['Multiplier'],
        price=1000,  # You'll need to get actual prices
        fx_rate=1.0,  # You'll need to get actual FX rates
        sigma_pct=0.16,  # You'll need to calculate actual volatility
        idm=idm,
        weight=weight,
        risk_target=risk_target
    )
    return minimum_capital <= capital

def read_instrument_data(symbol):
    """
    Read daily data for a given instrument symbol.
    
    Parameters:
    - symbol (str): The instrument symbol
    
    Returns:
    - pd.DataFrame: DataFrame with daily returns
    """
    try:
        df = pd.read_csv(f'Data/{symbol}_daily_data.csv', parse_dates=['Time'])
        df.set_index('Time', inplace=True)
        df['returns'] = df['Last'].pct_change(fill_method=None)
        return df['returns']
    except FileNotFoundError:
        print(f"No data found for {symbol}")
        return None

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
    
    Parameters:
    - instrument_code (str): The instrument symbol
    - instrument_config (pd.DataFrame): DataFrame with instrument configurations
    - position_turnover (float): Position turnover per year
    
    Returns:
    - float: Sharpe ratio cost per trade
    """
    # Default values for commission and spread
    commission = 1.24  # Default commission per trade
    spread_points = 0.625  # Default spread in points
    
    # Get instrument multiplier and current price
    multiplier = instrument_config.loc[instrument_code, 'Multiplier']
    price = 1000  # Default price - you should replace this with actual price data
    
    # Calculate risk-adjusted cost
    spread_cost = multiplier * (spread_points / 2)
    total_cost = spread_cost + commission
    cost_percent = total_cost / (price * multiplier)
    
    # Annualize the cost
    annual_cost = cost_percent * (position_turnover + 4)  # 4 rolls per year
    
    return annual_cost

def select_instruments(capital, risk_target, approx_number_of_instruments, approx_idm, position_turnover, pre_cost_sr):
    """
    Select instruments using the handcrafting algorithm.
    
    Parameters:
    - capital (float): Total capital
    - risk_target (float): Risk target
    - approx_number_of_instruments (int): Approximate number of instruments to select
    - approx_idm (float): Approximate IDM
    - position_turnover (int): Position turnover per year
    - pre_cost_sr (float): Pre-cost Sharpe ratio
    
    Returns:
    - list: Selected instruments
    """
    # Read instruments configuration
    instruments_df = pd.read_csv('Data/instruments.csv')
    instruments_df.set_index('Symbol', inplace=True)
    
    # Create returns DataFrame for all instruments
    returns_dict = {}
    for symbol in instruments_df.index:
        returns = read_instrument_data(symbol)
        if returns is not None:
            returns_dict[symbol] = returns
    
    # Filter instruments_df to only include instruments with data
    instruments_with_data = list(returns_dict.keys())
    instruments_df = instruments_df.loc[instruments_with_data]
    
    # Add rolls_per_year column with default value of 4
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
    
    # Continue selecting instruments until SR starts declining
    while current_SR > (max_SR_achieved * 0.9):
        print(f"{selected_instruments} SR: {current_SR:.2f}")
        next_instrument = choose_next_instrument(
            selected_instruments,
            correlation_matrix=corr_matrix,
            pre_cost_SR=pre_cost_sr,
            instrument_config=instruments_df,
            position_turnover=position_turnover,
            capital=capital,
            risk_target=risk_target
        )
        selected_instruments.append(next_instrument)
        
        current_SR = calculate_SR_for_selected_instruments(
            selected_instruments,
            correlation_matrix=corr_matrix,
            pre_cost_SR=pre_cost_sr,
            instrument_config=instruments_df,
            position_turnover=position_turnover,
            capital=capital,
            risk_target=risk_target
        )
        
        if current_SR > max_SR_achieved:
            max_SR_achieved = current_SR
    
    return selected_instruments

if __name__ == "__main__":
    # Example usage
    capital = 1000000
    risk_target = 0.2
    approx_number_of_instruments = 5
    approx_idm = 2.5
    position_turnover = 5
    pre_cost_sr = 0.4
    
    selected = select_instruments(
        capital=capital,
        risk_target=risk_target,
        approx_number_of_instruments=approx_number_of_instruments,
        approx_idm=approx_idm,
        position_turnover=position_turnover,
        pre_cost_sr=pre_cost_sr
    )
    
    print("Selected instruments:", selected)

