#!/usr/bin/env python
"""
Strategy 9 – Complete Code File

This file includes all helper functions from your chapters (1, 3, 4, 5, 7, 8)
integrated into one file. Data is loaded from your symbols CSV file.

Your Data/symbols.csv file should have the format:

Symbol,Multiplier
Data/audmicro_daily_data.csv,10000
Data/ba_daily_data.csv,0.1  # Micro Bitcoin Futures
...
"""

import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
from scipy.stats import norm, linregress
from enum import Enum
from copy import copy
from typing import Tuple
import os

# =============================================================================
# Chapter 1: Basic Returns, Statistics, and Frequency Aggregation
# =============================================================================

DEFAULT_DATE_FORMAT = "%Y-%m-%d"
BUSINESS_DAYS_IN_YEAR = 256  # Trading days per year

def pd_readcsv(filename: str, date_format=DEFAULT_DATE_FORMAT, date_index_name: str = "Time") -> pd.DataFrame:
    """
    Reads a CSV file and sets its index using the given date column.
    Expects a column named "Time" (or change date_index_name if needed).
    """
    df = pd.read_csv(filename)
    if date_index_name not in df.columns:
        raise ValueError(f"Expected date column '{date_index_name}' not found in {filename}.")
    df.index = pd.to_datetime(df[date_index_name], format=date_format).values
    del df[date_index_name]
    df.index.name = None
    return df

def calculate_perc_returns(position_contracts_held: pd.Series,
                           adjusted_price: pd.Series,
                           fx_series: pd.Series,
                           multiplier: float,
                           capital_required: float) -> pd.Series:
    return_price_points = (adjusted_price - adjusted_price.shift(1)) * position_contracts_held.shift(1)
    return_instrument_currency = return_price_points * multiplier
    fx_series_aligned = fx_series.reindex(return_instrument_currency.index, method="ffill")
    return_base_currency = return_instrument_currency * fx_series_aligned
    perc_return = return_base_currency / capital_required
    return perc_return

# Frequency definitions for resampling
Frequency = Enum("Frequency", "Natural Year Month Week BDay")
NATURAL = Frequency.Natural
YEAR = Frequency.Year
MONTH = Frequency.Month
WEEK = Frequency.Week

def sum_at_frequency(perc_return: pd.Series, at_frequency: Frequency = NATURAL) -> pd.Series:
    if at_frequency == NATURAL:
        return perc_return
    freq_map = {YEAR: "Y", WEEK: "7D", MONTH: "1M"}
    return perc_return.resample(freq_map[at_frequency]).sum()

def ann_mean_given_frequency(perc_return_at_freq: pd.Series, at_frequency: Frequency) -> float:
    periods = BUSINESS_DAYS_IN_YEAR if at_frequency == NATURAL else {YEAR: 1, WEEK: 52.25, MONTH: 12}[at_frequency]
    return perc_return_at_freq.mean() * periods

def ann_std_given_frequency(perc_return_at_freq: pd.Series, at_frequency: Frequency) -> float:
    periods = BUSINESS_DAYS_IN_YEAR if at_frequency == NATURAL else {YEAR: 1, WEEK: 52.25, MONTH: 12}[at_frequency]
    return perc_return_at_freq.std() * (periods ** 0.5)

def calculate_drawdown(perc_return: pd.Series) -> pd.Series:
    cum = perc_return.cumsum()
    running_max = cum.rolling(len(cum) + 1, min_periods=1).max()
    return running_max - cum

QUANT_PERCENTILE_EXTREME = 0.01
QUANT_PERCENTILE_STD = 0.3
NORMAL_DISTR_RATIO = norm.ppf(QUANT_PERCENTILE_EXTREME) / norm.ppf(QUANT_PERCENTILE_STD)

def demeaned_remove_zeros(x: pd.Series) -> pd.Series:
    x = x.copy()
    x[x == 0] = np.nan
    return x - x.mean()

def calculate_quant_ratio_lower(x: pd.Series) -> float:
    x_dm = demeaned_remove_zeros(x)
    raw_ratio = x_dm.quantile(QUANT_PERCENTILE_EXTREME) / x_dm.quantile(QUANT_PERCENTILE_STD)
    return raw_ratio / NORMAL_DISTR_RATIO

def calculate_quant_ratio_upper(x: pd.Series) -> float:
    x_dm = demeaned_remove_zeros(x)
    raw_ratio = x_dm.quantile(1 - QUANT_PERCENTILE_EXTREME) / x_dm.quantile(1 - QUANT_PERCENTILE_STD)
    return raw_ratio / NORMAL_DISTR_RATIO

def calculate_stats(perc_return: pd.Series, at_frequency: Frequency = NATURAL) -> dict:
    pr_freq = sum_at_frequency(perc_return, at_frequency)
    ann_mean = ann_mean_given_frequency(pr_freq, at_frequency)
    ann_std = ann_std_given_frequency(pr_freq, at_frequency)
    sharpe_ratio = ann_mean / ann_std if ann_std != 0 else np.nan
    skew = pr_freq.skew()
    drawdowns = calculate_drawdown(pr_freq)
    avg_drawdown = drawdowns.mean()
    max_drawdown = drawdowns.max()
    quant_ratio_lower = calculate_quant_ratio_lower(pr_freq)
    quant_ratio_upper = calculate_quant_ratio_upper(pr_freq)
    return {
        "ann_mean": ann_mean,
        "ann_std": ann_std,
        "sharpe_ratio": sharpe_ratio,
        "skew": skew,
        "avg_drawdown": avg_drawdown,
        "max_drawdown": max_drawdown,
        "quant_ratio_lower": quant_ratio_lower,
        "quant_ratio_upper": quant_ratio_upper,
    }

# =============================================================================
# Chapter 3 & 4: Data Loading, Volatility, and Position Sizing
# =============================================================================

def load_data_from_symbols(symbols_csv: str) -> Tuple[dict, dict, dict]:
    """
    Reads your Data/symbols.csv file and loads data for each instrument.
    Returns three dictionaries:
      - adjusted_prices: keys are instrument IDs (derived from the file basename),
        values are the price series.
      - current_prices: same as adjusted_prices.
      - multipliers: keys as instrument IDs, values as float multipliers.
    The symbols CSV is expected to have columns "Symbol" and "Multiplier".
    """
    df = pd.read_csv(symbols_csv, comment='#')
    # Clean the multiplier column (take first token)
    df["Multiplier"] = df["Multiplier"].apply(lambda x: float(str(x).split()[0]))
    adjusted_prices = {}
    current_prices = {}
    multipliers = {}
    for _, row in df.iterrows():
        file_path = row["Symbol"]
        if not os.path.exists(file_path):
            print(f"File not found: {file_path} – skipping.")
            continue
        try:
            data = pd_readcsv(file_path)  # uses "Time" column for dates
            data = data.dropna()
            # Use 'adjusted' and 'underlying' if present, else use "Last"
            if "adjusted" in data.columns and "underlying" in data.columns:
                price_adj = data["adjusted"]
                price_curr = data["underlying"]
            elif "Last" in data.columns:
                price_adj = data["Last"]
                price_curr = data["Last"]
            else:
                raise ValueError(f"Expected price columns not found in {file_path}")
            # Derive an instrument ID from the file name.
            # E.g., "Data/es_daily_data.csv" becomes "es"
            base = os.path.basename(file_path)
            inst_id = base.split("_")[0].lower()
            adjusted_prices[inst_id] = price_adj
            current_prices[inst_id] = price_curr
            multipliers[inst_id] = row["Multiplier"]
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return adjusted_prices, current_prices, multipliers

def create_fx_series_given_adjusted_prices_dict(adjusted_prices_dict: dict) -> dict:
    fx_series = {}
    for inst, series in adjusted_prices_dict.items():
        fx_series[inst] = pd.Series(1, index=series.index)
    return fx_series

def calculate_variable_standard_deviation_for_risk_targeting(adjusted_price: pd.Series,
                                                             current_price: pd.Series,
                                                             use_perc_returns: bool = True,
                                                             annualise_stdev: bool = True) -> pd.Series:
    if use_perc_returns:
        daily_returns = adjusted_price.diff() / current_price.shift(1)
    else:
        daily_returns = adjusted_price.diff()
    daily_exp_std = daily_returns.ewm(span=32).std()
    factor = BUSINESS_DAYS_IN_YEAR ** 0.5 if annualise_stdev else 1
    annualised_std = daily_exp_std * factor
    ten_year_vol = annualised_std.rolling(BUSINESS_DAYS_IN_YEAR * 10, min_periods=1).mean()
    weighted_vol = 0.3 * ten_year_vol + 0.7 * annualised_std
    return weighted_vol

class standardDeviation(pd.Series):
    def __init__(self,
                 adjusted_price: pd.Series,
                 current_price: pd.Series,
                 use_perc_returns: bool = True,
                 annualise_stdev: bool = True):
        stdev_series = calculate_variable_standard_deviation_for_risk_targeting(
            adjusted_price, current_price, use_perc_returns, annualise_stdev
        )
        super().__init__(stdev_series)
        self._use_perc_returns = use_perc_returns
        self._annualised = annualise_stdev
        self._current_price = current_price

    @property
    def annualised(self) -> bool:
        return self._annualised

    @property
    def use_perc_returns(self) -> bool:
        return self._use_perc_returns

    @property
    def current_price(self) -> pd.Series:
        return self._current_price

    def daily_risk_price_terms(self) -> pd.Series:
        stdev = self.copy()
        if self.annualised:
            stdev = stdev / (BUSINESS_DAYS_IN_YEAR ** 0.5)
        if self.use_perc_returns:
            stdev = stdev * self.current_price
        return stdev

def calculate_variable_standard_deviation_for_risk_targeting_from_dict(adjusted_prices: dict,
                                                                       current_prices: dict,
                                                                       use_perc_returns: bool = True,
                                                                       annualise_stdev: bool = True) -> dict:
    std_dev_dict = {}
    for inst in adjusted_prices.keys():
        std_dev_dict[inst] = standardDeviation(
            adjusted_price=adjusted_prices[inst],
            current_price=current_prices[inst],
            use_perc_returns=use_perc_returns,
            annualise_stdev=annualise_stdev
        )
    return std_dev_dict

def calculate_position_series_given_variable_risk(capital: float,
                                                  risk_target_tau: float,
                                                  fx: pd.Series,
                                                  multiplier: float,
                                                  instrument_risk: standardDeviation) -> pd.Series:
    daily_risk = instrument_risk.daily_risk_price_terms()
    return capital * risk_target_tau / (multiplier * fx * daily_risk * (BUSINESS_DAYS_IN_YEAR ** 0.5))

def calculate_position_series_given_variable_risk_for_dict(capital: float,
                                                           risk_target_tau: float,
                                                           idm: float,
                                                           weights: dict,
                                                           fx_series_dict: dict,
                                                           multipliers: dict,
                                                           std_dev_dict: dict) -> dict:
    pos_dict = {}
    for inst in std_dev_dict.keys():
        effective_capital = capital * idm * weights.get(inst, 1)
        pos_dict[inst] = calculate_position_series_given_variable_risk(
            capital=effective_capital,
            risk_target_tau=risk_target_tau,
            fx=fx_series_dict[inst],
            multiplier=multipliers[inst],
            instrument_risk=std_dev_dict[inst]
        )
    return pos_dict

def perc_returns_to_df(perc_returns_dict: dict) -> pd.DataFrame:
    df = pd.concat(perc_returns_dict, axis=1)
    return df.dropna(how="all")

def aggregate_returns(perc_returns_dict: dict) -> pd.Series:
    return perc_returns_to_df(perc_returns_dict).sum(axis=1)

# =============================================================================
# Chapter 5: Cost-Adjusted Percentage Returns
# =============================================================================

def calculate_deflated_costs(stddev_series: standardDeviation, cost_per_contract: float) -> pd.Series:
    stdev_daily = stddev_series.daily_risk_price_terms()
    final_stdev = stdev_daily.iloc[-1]
    cost_deflator = stdev_daily / final_stdev
    historic_cost_per_contract = cost_per_contract * cost_deflator
    return historic_cost_per_contract

def calculate_costs_deflated_for_vol(stddev_series: standardDeviation,
                                     cost_per_contract: float,
                                     position_contracts_held: pd.Series) -> pd.Series:
    rounded_positions = position_contracts_held.round()
    position_change = rounded_positions - rounded_positions.shift(1)
    abs_trades = position_change.abs()
    historic_cost = calculate_deflated_costs(stddev_series, cost_per_contract)
    historic_cost_aligned = historic_cost.reindex(abs_trades.index, method="ffill")
    return abs_trades * historic_cost_aligned

def calculate_perc_returns_with_costs(position_contracts_held: pd.Series,
                                      adjusted_price: pd.Series,
                                      fx_series: pd.Series,
                                      stdev_series: standardDeviation,
                                      multiplier: float,
                                      capital_required: float,
                                      cost_per_contract: float) -> pd.Series:
    precost_return = (adjusted_price - adjusted_price.shift(1)) * position_contracts_held.shift(1)
    precost_return_currency = precost_return * multiplier
    historic_costs = calculate_costs_deflated_for_vol(stddev_series, cost_per_contract, position_contracts_held)
    historic_costs_aligned = historic_costs.reindex(precost_return_currency.index, method="ffill")
    net_return = precost_return_currency - historic_costs_aligned
    fx_aligned = fx_series.reindex(net_return.index, method="ffill")
    return_base = net_return * fx_aligned
    return return_base / capital_required

def calculate_perc_returns_for_dict_with_costs(position_contracts_dict: dict,
                                               adjusted_prices: dict,
                                               multipliers: dict,
                                               fx_series: dict,
                                               capital: float,
                                               cost_per_contract_dict: dict,
                                               std_dev_dict: dict) -> dict:
    pr_dict = {}
    for inst in position_contracts_dict.keys():
        pr_dict[inst] = calculate_perc_returns_with_costs(
            position_contracts_held=position_contracts_dict[inst],
            adjusted_price=adjusted_prices[inst],
            fx_series=fx_series[inst],
            stdev_series=std_dev_dict[inst],
            multiplier=multipliers[inst],
            capital_required=capital,
            cost_per_contract=cost_per_contract_dict[inst]
        )
    return pr_dict

# =============================================================================
# Chapter 7 & 8: EWMA Trend Forecasting and Buffering
# =============================================================================

def ewmac(adjusted_price: pd.Series, fast_span=16, slow_span=64) -> pd.Series:
    slow_ewma = adjusted_price.ewm(span=slow_span, min_periods=2).mean()
    fast_ewma = adjusted_price.ewm(span=fast_span, min_periods=2).mean()
    return fast_ewma - slow_ewma

def calculate_risk_adjusted_forecast_for_ewmac(adjusted_price: pd.Series,
                                               stdev_ann_perc: standardDeviation,
                                               fast_span: int = 64) -> pd.Series:
    ewmac_vals = ewmac(adjusted_price, fast_span=fast_span, slow_span=fast_span * 4)
    daily_vol = stdev_ann_perc.daily_risk_price_terms()
    return ewmac_vals / daily_vol

def calculate_scaled_forecast_for_ewmac(adjusted_price: pd.Series,
                                        stdev_ann_perc: standardDeviation,
                                        fast_span: int = 64) -> pd.Series:
    scalar_dict = {64: 1.91, 32: 2.79, 16: 4.1, 8: 5.95, 4: 8.53, 2: 12.1}
    risk_adjusted = calculate_risk_adjusted_forecast_for_ewmac(adjusted_price, stdev_ann_perc, fast_span)
    forecast_scalar = scalar_dict.get(fast_span, 1.0)
    return risk_adjusted * forecast_scalar

def calculate_forecast_for_ewmac(adjusted_price: pd.Series,
                                 stdev_ann_perc: standardDeviation,
                                 fast_span: int = 64) -> pd.Series:
    scaled = calculate_scaled_forecast_for_ewmac(adjusted_price, stdev_ann_perc, fast_span)
    return scaled.clip(-20, 20)

def calculate_combined_ewmac_forecast(adjusted_price: pd.Series,
                                      stdev_ann_perc: standardDeviation,
                                      fast_spans: list) -> pd.Series:
    forecasts = [calculate_forecast_for_ewmac(adjusted_price, stdev_ann_perc, fs) for fs in fast_spans]
    df_forecasts = pd.concat(forecasts, axis=1)
    average_forecast = df_forecasts.mean(axis=1)
    rule_count = len(fast_spans)
    FDM_DICT = {1: 1.0, 2: 1.03, 3: 1.08, 4: 1.13, 5: 1.19, 6: 1.26}
    fdm = FDM_DICT.get(rule_count, 1.0)
    scaled = average_forecast * fdm
    return scaled.clip(-20, 20)

def calculate_position_with_multiple_trend_forecast_applied(adjusted_price: pd.Series,
                                                            average_position: pd.Series,
                                                            stdev_ann_perc: standardDeviation,
                                                            fast_spans: list) -> pd.Series:
    forecast = calculate_combined_ewmac_forecast(adjusted_price, stdev_ann_perc, fast_spans)
    return forecast * average_position / 10

def calculate_position_dict_with_multiple_trend_forecast_applied(adjusted_prices_dict: dict,
                                                                 average_position_contracts_dict: dict,
                                                                 std_dev_dict: dict,
                                                                 fast_spans: list) -> dict:
    pos = {}
    for inst in adjusted_prices_dict.keys():
        pos[inst] = calculate_position_with_multiple_trend_forecast_applied(
            adjusted_price=adjusted_prices_dict[inst],
            average_position=average_position_contracts_dict[inst],
            stdev_ann_perc=std_dev_dict[inst],
            fast_spans=fast_spans
        )
    return pos

def apply_buffer_single_period(last_position: float, top_pos: float, bot_pos: float) -> float:
    if last_position > top_pos:
        return top_pos
    elif last_position < bot_pos:
        return bot_pos
    else:
        return last_position

def apply_buffer(optimal_position: pd.Series, upper_buffer: pd.Series, lower_buffer: pd.Series) -> pd.Series:
    upper_buffer = upper_buffer.ffill().round()
    lower_buffer = lower_buffer.ffill().round()
    use_optimal = optimal_position.ffill()
    current_pos = use_optimal.iloc[0]
    if pd.isna(current_pos):
        current_pos = 0.0
    buffered = [current_pos]
    for i in range(1, len(optimal_position.index)):
        current_pos = apply_buffer_single_period(current_pos, upper_buffer.iloc[i], lower_buffer.iloc[i])
        buffered.append(current_pos)
    return pd.Series(buffered, index=optimal_position.index)

def apply_buffering_to_positions(position_contracts: pd.Series,
                                 average_position_contracts: pd.Series,
                                 buffer_size: float = 0.10) -> pd.Series:
    buffer = average_position_contracts.abs() * buffer_size
    upper_buffer = position_contracts + buffer
    lower_buffer = position_contracts - buffer
    return apply_buffer(position_contracts, upper_buffer, lower_buffer)

def apply_buffering_to_position_dict(position_contracts_dict: dict,
                                     average_position_contracts_dict: dict) -> dict:
    buffered = {}
    for inst, pos in position_contracts_dict.items():
        buffered[inst] = apply_buffering_to_positions(pos, average_position_contracts_dict[inst])
    return buffered

# =============================================================================
# Strategy 9 Main Execution
# =============================================================================

if __name__ == "__main__":
    # Step 1: Load instrument data from the symbols file.
    symbols_file = "Data/symbols.csv"  # Path to your symbols file
    adjusted_prices_dict, current_prices_dict, file_multipliers = load_data_from_symbols(symbols_file)
    
    # For demonstration, you can print the loaded instruments:
    print("Loaded instruments:", list(adjusted_prices_dict.keys()))
    
    # Step 2: Use the multipliers from the symbols file.
    multipliers = file_multipliers  # keys are instrument IDs derived from filenames
    
    # Step 3: Create FX series (assume all USD).
    fx_series_dict = create_fx_series_given_adjusted_prices_dict(adjusted_prices_dict)
    
    # Step 4: Define other strategy parameters.
    capital = 1000000
    risk_target_tau = 0.2
    idm = 1.5
    # Define instrument weights (if not provided, use equal weights)
    instrument_weights = {inst: 1/len(adjusted_prices_dict) for inst in adjusted_prices_dict.keys()}
    # Define cost per contract (you can adjust these; here we set default values)
    cost_per_contract_dict = {inst: 1.0 for inst in adjusted_prices_dict.keys()}
    
    # Step 5: Calculate instrument risk (volatility).
    std_dev_dict = calculate_variable_standard_deviation_for_risk_targeting_from_dict(
        adjusted_prices=adjusted_prices_dict,
        current_prices=current_prices_dict,
        use_perc_returns=True,
        annualise_stdev=True,
    )
    
    # Step 6: Determine average risk-based position sizing.
    average_position_contracts_dict = calculate_position_series_given_variable_risk_for_dict(
        capital=capital,
        risk_target_tau=risk_target_tau,
        idm=idm,
        weights=instrument_weights,
        fx_series_dict=fx_series_dict,
        multipliers=multipliers,
        std_dev_dict=std_dev_dict,
    )
    
    # Step 7: Apply multiple trend forecast (Strategy 9).
    fast_spans = [16, 32, 64]  # Adjustable list of fast spans
    position_contracts_dict = calculate_position_dict_with_multiple_trend_forecast_applied(
        adjusted_prices_dict=adjusted_prices_dict,
        average_position_contracts_dict=average_position_contracts_dict,
        std_dev_dict=std_dev_dict,
        fast_spans=fast_spans,
    )
    
    # Step 8: Buffer positions to reduce turnover.
    buffered_position_dict = apply_buffering_to_position_dict(
        position_contracts_dict=position_contracts_dict,
        average_position_contracts_dict=average_position_contracts_dict,
    )
    
    # Step 9: Compute cost-adjusted percentage returns.
    perc_return_dict = calculate_perc_returns_for_dict_with_costs(
        position_contracts_dict=buffered_position_dict,
        adjusted_prices=adjusted_prices_dict,
        multipliers=multipliers,
        fx_series=fx_series_dict,
        capital=capital,
        cost_per_contract_dict=cost_per_contract_dict,
        std_dev_dict=std_dev_dict,
    )
    
    # Step 10: Output performance statistics.
    for inst, stats in calculate_stats(perc_return_dict[list(perc_return_dict.keys())[0]]).items():
        print(f"Stats for {list(perc_return_dict.keys())[0]}: {stats}")
    
    perc_return_agg = aggregate_returns(perc_return_dict)
    print("Aggregated portfolio stats:", calculate_stats(perc_return_agg))
    
    # Optional: Linear regression (beta estimation)
    time_numeric = pd.Series(perc_return_agg.index.astype(np.int64))
    reg_results = linregress(time_numeric, perc_return_agg.values)
    print("Beta slope: {:.4f}".format(reg_results.slope))
    daily_alpha = reg_results.intercept
    annual_alpha = 100 * daily_alpha * BUSINESS_DAYS_IN_YEAR
    print("Annual alpha: {:.2f}%".format(annual_alpha))