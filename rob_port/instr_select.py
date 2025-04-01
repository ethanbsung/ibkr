import pandas as pd
import numpy as np
from scipy.cluster import hierarchy as sch
import os
import matplotlib.pyplot as plt

PRINT_TRACE = False

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

def cluster_correlation_matrix(corr_matrix: correlationEstimate,
                             cluster_size: int = 2):
    clusters = get_list_of_clusters_for_correlation_matrix(corr_matrix,
                                                          cluster_size=cluster_size)
    clusters_as_names = from_cluster_index_to_asset_names(clusters, corr_matrix)
    if PRINT_TRACE:
        print("Cluster split: %s" % str(clusters_as_names))
    return clusters_as_names

def get_list_of_clusters_for_correlation_matrix(corr_matrix, cluster_size: int = 2) -> list:
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

def load_instrument_returns(instrument_config: pd.DataFrame) -> pd.DataFrame:
    """Load returns data for all instruments from their daily data files."""
    returns_dict = {}
    
    for symbol in instrument_config.index:
        try:
            df = pd.read_csv(f'Data/{symbol}_daily_data.csv', parse_dates=['Time'])
            df.set_index('Time', inplace=True)
            df['returns'] = df['Last'].pct_change(fill_method=None)
            df.dropna(inplace=True)
            returns_dict[symbol] = df['returns']
            print(f"✓ Loaded {symbol}")
        except FileNotFoundError:
            print(f"✗ No data found for {symbol}")
            continue
        except Exception as e:
            print(f"✗ Error loading {symbol}: {str(e)}")
            continue
    
    if not returns_dict:
        raise ValueError("No data available for any instruments")
    
    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna(how='all')
    return returns_df

def get_asset_class(instrument_code: str, instrument_config: pd.DataFrame) -> str:
    """Determine the asset class for an instrument based on its code."""
    if instrument_code in ['ZT', 'Z3N', 'ZF', 'ZN', 'TN', 'TWE', 'ZB', 'LIW', 'N1U', 'YE',
                          'OAT', 'GBS', 'GBM', 'GBL', 'GBX', 'BTS', 'BTP', '3KTB', 'FLKTB', 'FBON']:
        return 'Bonds'
    elif instrument_code in ['MYM', 'MNQ', 'RSV', 'M2K', 'EMD', 'MES']:
        return 'US_Equity'
    elif instrument_code in ['EOE', 'CAC40', 'DAX', 'SMI', 'DJ200S', 'DJSD', 'DJ600', 'ESTX50',
                           'SXAP', 'SXPP', 'SXDP', 'SXIP', 'SXEP', 'SX8P', 'SXTP', 'SX6P']:
        return 'EU_Equity'
    elif instrument_code in ['M1MS', 'XINA50', 'XINO1', 'NIFTY', 'N225M', 'JPNK400',
                           'TSEMOTHR', 'MNTPX', 'KOSDQ150', 'K200', 'SSG', 'TWN']:
        return 'Asia_Equity'
    elif instrument_code in ['VIX', 'V2TX']:
        return 'Volatility'
    elif instrument_code in ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NOK', 'NZD', 'SEK']:
        return 'Major_FX'
    elif instrument_code in ['RP', 'RY', 'BRE', 'UC', 'SIR', 'MXP', 'RUR', 'SND']:
        return 'EM_FX'
    elif instrument_code in ['ALI', 'HG', 'MGC', 'SCI', 'PA', 'PL', 'SI', 'MBT', 'ETHUSDRR']:
        return 'Metals'
    elif instrument_code in ['BZ', 'QM', 'HH', 'RB', 'QG', 'HO']:
        return 'Energy'
    elif instrument_code in ['AIGCI', 'CSC', 'ZC', 'GF', 'HE', 'LE', 'ZO', 'KE', 'ZR',
                           'ZS', 'ZM', 'ZL', 'ZW']:
        return 'Agricultural'
    else:
        print(f"Warning: Unknown asset class for instrument {instrument_code}")
        return 'Other'

def analyze_portfolio_weights(weights: portfolioWeights, instrument_config: pd.DataFrame):
    """Analyze and print portfolio weights by asset class."""
    asset_classes = {}
    for instrument in weights.assets:
        asset_class = get_asset_class(instrument, instrument_config)
        if asset_class not in asset_classes:
            asset_classes[asset_class] = []
        asset_classes[asset_class].append(instrument)
    
    print("\nPortfolio Weight Allocation:")
    print(f"Number of asset classes: {len(asset_classes)}")
    
    for asset_class, instruments in asset_classes.items():
        class_weight = sum(weights[inst] for inst in instruments)
        print(f"\n{asset_class} ({len(instruments)} instruments, {class_weight*100:.1f}% class weight):")
        for instrument in instruments:
            print(f"  {instrument}: {weights[instrument]*100:.2f}%")

def main():
    print("\nLoading instrument configuration...")
    instrument_config = pd.read_csv('Data/instruments.csv')
    instrument_config.set_index('Symbol', inplace=True)
    
    print("\nLoading instrument returns data...")
    returns_df = load_instrument_returns(instrument_config)
    
    print("\nCalculating correlation matrix...")
    corr_matrix = correlationEstimate(returns_df.corr())
    
    print("\nCreating handcraft portfolio...")
    handcraft_portfolio = handcraftPortfolio(corr_matrix)
    
    print("\nCalculating portfolio weights...")
    PRINT_TRACE = True
    weights = handcraft_portfolio.weights()
    
    print("\nSelected Instruments Summary:")
    print(f"Total number of instruments selected: {len(weights.assets)}")
    print("\nSelected instruments:")
    for instrument in sorted(weights.assets):
        print(f"  {instrument}: {weights[instrument]*100:.2f}%")
    
    print("\nAnalyzing portfolio weights by asset class...")
    analyze_portfolio_weights(weights, instrument_config)
    
    return weights

if __name__ == "__main__":
    weights = main()
