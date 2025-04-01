import matplotlib
matplotlib.use("TkAgg")

import pandas as pd
import numpy as np
from enum import Enum
from scipy.stats import norm
from copy import copy

from chapter1 import (
    calculate_standard_deviation,
    annualized_standard_deviation,
    calculate_fat_tails,
    business_days_per_year
)

# Constants
DEFAULT_DATE_FORMAT = "%Y-%m-%d"
BUSINESS_DAYS_IN_YEAR = 256
WEEKS_PER_YEAR = 52.25
MONTHS_PER_YEAR = 12
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60

# Frequency Enum
Frequency = Enum(
    "Frequency",
    "Natural Year Month Week BDay",
)

NATURAL = Frequency.Natural
YEAR = Frequency.Year
MONTH = Frequency.Month
WEEK = Frequency.Week

PERIODS_PER_YEAR = {
    MONTH: MONTHS_PER_YEAR,
    WEEK: WEEKS_PER_YEAR,
    YEAR: 1
}

def periods_per_year(at_frequency: Frequency):
    if at_frequency == NATURAL:
        return BUSINESS_DAYS_IN_YEAR
    else:
        return PERIODS_PER_YEAR[at_frequency]

def years_in_data(some_data: pd.Series) -> float:
    datediff = some_data.index[-1] - some_data.index[0]
    seconds_in_data = datediff.total_seconds()
    return seconds_in_data / SECONDS_PER_YEAR

def sum_at_frequency(perc_return: pd.Series,
                     at_frequency: Frequency = NATURAL) -> pd.Series:
    if at_frequency == NATURAL:
        return perc_return

    at_frequency_str_dict = {
        YEAR: "Y",
        WEEK: "7D",
        MONTH: "1M"
    }
    at_frequency_str = at_frequency_str_dict[at_frequency]

    perc_return_at_freq = perc_return.resample(at_frequency_str).sum()
    return perc_return_at_freq

def ann_mean_given_frequency(perc_return_at_freq: pd.Series,
                             at_frequency: Frequency) -> float:
    mean_at_frequency = perc_return_at_freq.mean()
    periods_per_year_for_frequency = periods_per_year(at_frequency)
    annualised_mean = mean_at_frequency * periods_per_year_for_frequency
    return annualised_mean

def ann_std_given_frequency(perc_return_at_freq: pd.Series,
                             at_frequency: Frequency) -> float:
    std_at_frequency = perc_return_at_freq.std()
    periods_per_year_for_frequency = periods_per_year(at_frequency)
    annualised_std = std_at_frequency * (periods_per_year_for_frequency**.5)
    return annualised_std

def calculate_drawdown(perc_return):
    cum_perc_return = perc_return.cumsum()
    max_cum_perc_return = cum_perc_return.rolling(len(perc_return)+1,
                                                  min_periods=1).max()
    return max_cum_perc_return - cum_perc_return

QUANT_PERCENTILE_EXTREME = 0.01
QUANT_PERCENTILE_STD = 0.3
NORMAL_DISTR_RATIO = norm.ppf(QUANT_PERCENTILE_EXTREME) / norm.ppf(QUANT_PERCENTILE_STD)

def calculate_quant_ratio_lower(x):
    x_dm = demeaned_remove_zeros(x)
    raw_ratio = x_dm.quantile(QUANT_PERCENTILE_EXTREME) / x_dm.quantile(
        QUANT_PERCENTILE_STD
    )
    return raw_ratio / NORMAL_DISTR_RATIO

def calculate_quant_ratio_upper(x):
    x_dm = demeaned_remove_zeros(x)
    raw_ratio = x_dm.quantile(1 - QUANT_PERCENTILE_EXTREME) / x_dm.quantile(
        1 - QUANT_PERCENTILE_STD
    )
    return raw_ratio / NORMAL_DISTR_RATIO

def demeaned_remove_zeros(x):
    x[x == 0] = np.nan
    return x - x.mean()

def calculate_stats(perc_return: pd.Series,
                at_frequency: Frequency = NATURAL) -> dict:
    perc_return_at_freq = sum_at_frequency(perc_return, at_frequency=at_frequency)

    ann_mean = ann_mean_given_frequency(perc_return_at_freq, at_frequency=at_frequency)
    ann_std = ann_std_given_frequency(perc_return_at_freq, at_frequency=at_frequency)
    sharpe_ratio = ann_mean / ann_std

    skew_at_freq = perc_return_at_freq.skew()
    drawdowns = calculate_drawdown(perc_return_at_freq)
    avg_drawdown = drawdowns.mean()
    max_drawdown = drawdowns.max()
    quant_ratio_lower = calculate_quant_ratio_upper(perc_return_at_freq)
    quant_ratio_upper = calculate_quant_ratio_upper(perc_return_at_freq)

    return dict(
        ann_mean = ann_mean,
        ann_std = ann_std,
        sharpe_ratio = sharpe_ratio,
        skew = skew_at_freq,
        avg_drawdown = avg_drawdown,
        max_drawdown = max_drawdown,
        quant_ratio_lower = quant_ratio_lower,
        quant_ratio_upper = quant_ratio_upper
    )

def calculate_perc_returns(position_contracts_held: pd.Series,
                            adjusted_price: pd.Series,
                           fx_series: pd.Series,
                           multiplier: float,
                           capital_required: pd.Series,
                           ) -> pd.Series:
    return_price_points = (adjusted_price - adjusted_price.shift(1))*position_contracts_held.shift(1)
    return_instrument_currency = return_price_points * multiplier
    fx_series_aligned = fx_series.reindex(return_instrument_currency.index, method="ffill")
    return_base_currency = return_instrument_currency * fx_series_aligned
    perc_return = return_base_currency / capital_required
    return perc_return

def calculate_variable_standard_deviation_for_risk_targeting(
    adjusted_price: pd.Series,
    current_price: pd.Series,
    use_perc_returns: bool = True,
    annualise_stdev: bool = True,
) -> pd.Series:
    if use_perc_returns:
        daily_returns = calculate_percentage_returns(
            adjusted_price=adjusted_price, current_price=current_price
        )
    else:
        daily_returns = calculate_daily_returns(adjusted_price=adjusted_price)

    daily_exp_std_dev = daily_returns.ewm(span=32).std()

    if annualise_stdev:
        annualisation_factor = BUSINESS_DAYS_IN_YEAR ** 0.5
    else:
        annualisation_factor = 1

    annualised_std_dev = daily_exp_std_dev * annualisation_factor

    ten_year_vol = annualised_std_dev.rolling(
        BUSINESS_DAYS_IN_YEAR * 10, min_periods=1
    ).mean()
    weighted_vol = 0.3 * ten_year_vol + 0.7 * annualised_std_dev

    return weighted_vol

def calculate_percentage_returns(
    adjusted_price: pd.Series, current_price: pd.Series
) -> pd.Series:
    daily_price_changes = calculate_daily_returns(adjusted_price)
    percentage_changes = daily_price_changes / current_price.shift(1)
    return percentage_changes

def calculate_daily_returns(adjusted_price: pd.Series) -> pd.Series:
    return adjusted_price.diff()

class standardDeviation(pd.Series):
    def __init__(
        self,
        adjusted_price: pd.Series,
        current_price: pd.Series,
        use_perc_returns: bool = True,
        annualise_stdev: bool = True,
    ):
        stdev = calculate_variable_standard_deviation_for_risk_targeting(
            adjusted_price=adjusted_price,
            current_price=current_price,
            annualise_stdev=annualise_stdev,
            use_perc_returns=use_perc_returns,
        )
        super().__init__(stdev)

        self._use_perc_returns = use_perc_returns
        self._annualised = annualise_stdev
        self._current_price = current_price

    def daily_risk_price_terms(self):
        stdev = copy(self)
        if self.annualised:
            stdev = stdev / (BUSINESS_DAYS_IN_YEAR ** 0.5)

        if self.use_perc_returns:
            stdev = stdev * self.current_price

        return stdev

    def annual_risk_price_terms(self):
        stdev = copy(self)
        if not self.annualised:
            stdev = stdev * (BUSINESS_DAYS_IN_YEAR ** 0.5)

        if self.use_perc_returns:
            stdev = stdev * self.current_price

        return stdev

    @property
    def annualised(self) -> bool:
        return self._annualised

    @property
    def use_perc_returns(self) -> bool:
        return self._use_perc_returns

    @property
    def current_price(self) -> pd.Series:
        return self._current_price

def calculate_position_series_given_variable_risk(
    capital: float,
    risk_target_tau: float,
    fx: pd.Series,
    multiplier: float,
    instrument_risk: standardDeviation,
) -> pd.Series:
    daily_risk_price_terms = instrument_risk.daily_risk_price_terms()
    return (
        capital
        * risk_target_tau
        / (multiplier * fx * daily_risk_price_terms * (BUSINESS_DAYS_IN_YEAR ** 0.5))
    )

def calculate_turnover(position, average_position):
    daily_trades = position.diff()
    as_proportion_of_average = daily_trades.abs() / average_position.shift(1)
    average_daily = as_proportion_of_average.mean()
    annualised_turnover = average_daily * BUSINESS_DAYS_IN_YEAR
    return annualised_turnover

def get_data_dict():
    """Load data for all instruments from their daily data files."""
    all_data = {}
    adjusted_prices = {}
    current_prices = {}
    
    # Load instrument configuration
    instrument_config = pd.read_csv('Data/instruments.csv')
    instrument_config.set_index('Symbol', inplace=True)
    
    for symbol in instrument_config.index:
        try:
            df = pd.read_csv(f'Data/{symbol}_daily_data.csv', parse_dates=['Time'])
            df.set_index('Time', inplace=True)
            all_data[symbol] = df
            adjusted_prices[symbol] = df['Last']  # Using Last price as adjusted price
            current_prices[symbol] = df['Last']   # Using Last price as current price
            print(f"✓ Loaded {symbol}")
        except FileNotFoundError:
            print(f"✗ No data found for {symbol}")
            continue
        except Exception as e:
            print(f"✗ Error loading {symbol}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("No data available for any instruments")
    
    return adjusted_prices, current_prices

def create_fx_series_given_adjusted_prices_dict(adjusted_prices_dict: dict) -> dict:
    """Create FX rate series for each instrument."""
    fx_series_dict = {}
    
    for instrument_code, adjusted_prices in adjusted_prices_dict.items():
        try:
            # Determine currency from instrument code
            if instrument_code in ['EOE', 'CAC40', 'DAX', 'SMI', 'DJ200S', 'DJSD', 'DJ600', 
                                 'ESTX50', 'SXAP', 'SXPP', 'SXDP', 'SXIP', 'SXEP', 'SX8P', 
                                 'SXTP', 'SX6P', 'OAT', 'GBS', 'GBM', 'GBL', 'GBX', 'BTS', 
                                 'BTP', 'FBON', 'V2TX']:
                # EUR instruments
                fx_df = pd.read_csv('Data/eur_daily_data.csv', parse_dates=['Time'])
                fx_df.set_index('Time', inplace=True)
                fx_series = fx_df['Last']
            elif instrument_code in ['N225M', 'JPNK400', 'TSEMOTHR', 'MNTPX', 'JPY']:
                # JPY instruments
                fx_df = pd.read_csv('Data/jpy_daily_data.csv', parse_dates=['Time'])
                fx_df.set_index('Time', inplace=True)
                fx_series = fx_df['Last']
            else:
                # USD instruments
                fx_series = pd.Series(1, index=adjusted_prices.index)
            
            # Align FX series with price series
            fx_series_aligned = fx_series.reindex(adjusted_prices.index).ffill()
            fx_series_dict[instrument_code] = fx_series_aligned
            
        except Exception as e:
            print(f"✗ Error creating FX series for {instrument_code}: {str(e)}")
            fx_series_dict[instrument_code] = pd.Series(1, index=adjusted_prices.index)
    
    return fx_series_dict

def calculate_variable_standard_deviation_for_risk_targeting_from_dict(
    adjusted_prices: dict,
    current_prices: dict,
    use_perc_returns: bool = True,
    annualise_stdev: bool = True,
) -> dict:
    std_dev_dict = {}
    
    for instrument_code in adjusted_prices.keys():
        std_dev_dict[instrument_code] = standardDeviation(
            adjusted_price=adjusted_prices[instrument_code],
            current_price=current_prices[instrument_code],
            use_perc_returns=use_perc_returns,
            annualise_stdev=annualise_stdev,
        )
    
    return std_dev_dict

def calculate_position_series_given_variable_risk_for_dict(
    capital: float,
    risk_target_tau: float,
    idm: float,
    weights: dict,
    fx_series_dict: dict,
    multipliers: dict,
    std_dev_dict: dict,
) -> dict:
    position_series_dict = {}
    
    for instrument_code in std_dev_dict.keys():
        position_series_dict[instrument_code] = calculate_position_series_given_variable_risk(
            capital=capital * idm * weights[instrument_code],
            risk_target_tau=risk_target_tau,
            multiplier=multipliers[instrument_code],
            fx=fx_series_dict[instrument_code],
            instrument_risk=std_dev_dict[instrument_code],
        )
    
    return position_series_dict

def calculate_perc_returns_for_dict(
    position_contracts_dict: dict,
    adjusted_prices: dict,
    multipliers: dict,
    fx_series: dict,
    capital: float,
) -> dict:
    perc_returns_dict = {}
    
    for instrument_code in position_contracts_dict.keys():
        perc_returns_dict[instrument_code] = calculate_perc_returns(
            position_contracts_held=position_contracts_dict[instrument_code],
            adjusted_price=adjusted_prices[instrument_code],
            multiplier=multipliers[instrument_code],
            fx_series=fx_series[instrument_code],
            capital_required=capital,
        )
    
    return perc_returns_dict

def aggregate_returns(perc_returns_dict: dict) -> pd.Series:
    both_returns = pd.concat(perc_returns_dict, axis=1)
    both_returns = both_returns.dropna(how="all")
    agg = both_returns.sum(axis=1)
    return agg

def minimum_capital_for_sub_strategy(
    multiplier: float,
    price: float,
    fx: float,
    instrument_risk_ann_perc: float,
    risk_target: float,
    idm: float,
    weight: float,
    contracts: int = 4,
):
    return (
        contracts
        * multiplier
        * price
        * fx
        * instrument_risk_ann_perc
        / (risk_target * idm * weight)
    )

if __name__ == "__main__":
    # Load instrument configuration
    instrument_config = pd.read_csv('Data/instruments.csv')
    instrument_config.set_index('Symbol', inplace=True)
    
    # Get data for all instruments
    adjusted_prices, current_prices = get_data_dict()
    
    # Create multipliers dictionary from instrument config
    multipliers = instrument_config['Multiplier'].to_dict()
    
    # Set risk parameters
    risk_target_tau = 0.2
    capital = 1000000
    idm = 1.5
    
    # Create equal weights for all instruments
    instrument_weights = {symbol: 1.0/len(adjusted_prices) for symbol in adjusted_prices.keys()}
    
    # Create FX series for all instruments
    fx_series_dict = create_fx_series_given_adjusted_prices_dict(adjusted_prices)
    
    # Calculate standard deviations
    std_dev_dict = calculate_variable_standard_deviation_for_risk_targeting_from_dict(
        adjusted_prices=adjusted_prices,
        current_prices=current_prices,
        annualise_stdev=True,
        use_perc_returns=True,
    )
    
    # Calculate position sizes
    position_contracts_dict = calculate_position_series_given_variable_risk_for_dict(
        capital=capital,
        risk_target_tau=risk_target_tau,
        idm=idm,
        weights=instrument_weights,
        std_dev_dict=std_dev_dict,
        fx_series_dict=fx_series_dict,
        multipliers=multipliers,
    )
    
    # Calculate returns
    perc_return_dict = calculate_perc_returns_for_dict(
        position_contracts_dict=position_contracts_dict,
        fx_series=fx_series_dict,
        multipliers=multipliers,
        capital=capital,
        adjusted_prices=adjusted_prices,
    )
    
    # Print statistics for each instrument
    for instrument_code, returns in perc_return_dict.items():
        print(f"\nStatistics for {instrument_code}:")
        print(calculate_stats(returns))
    
    # Calculate and print aggregate returns
    perc_return_agg = aggregate_returns(perc_return_dict)
    print("\nAggregate Portfolio Statistics:")
    print(calculate_stats(perc_return_agg))
    
    # Calculate minimum capital for each instrument
    print("\nMinimum Capital Requirements:")
    for instrument_code in position_contracts_dict.keys():
        min_cap = minimum_capital_for_sub_strategy(
            multiplier=multipliers[instrument_code],
            risk_target=risk_target_tau,
            fx=fx_series_dict[instrument_code][-1],
            idm=idm,
            weight=instrument_weights[instrument_code],
            instrument_risk_ann_perc=std_dev_dict[instrument_code][-1],
            price=current_prices[instrument_code][-1],
        )
        print(f"{instrument_code}: {min_cap:,.2f}")