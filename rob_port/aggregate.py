#!/usr/bin/env python
"""
Strategy 9 Backtest – Multiple EWMA Trend Filters
Refactored to match the earlier logic and produce a single final
portfolio curve plus an ES multi-strategy plot.
"""

import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
from scipy.stats import norm, linregress
import os
import matplotlib.pyplot as plt

# =============================================================================
# Chapter 1: Basic Returns, Statistics, and Frequency Aggregation
# =============================================================================

DEFAULT_DATE_FORMAT = "%Y-%m-%d"
BUSINESS_DAYS_IN_YEAR = 256  # Trading days per year

def pd_readcsv(filename: str, date_format=DEFAULT_DATE_FORMAT, date_index_name: str = "Time") -> pd.DataFrame:
    df = pd.read_csv(filename)
    if date_index_name not in df.columns:
        raise ValueError(f"Expected date column '{date_index_name}' not found in {filename}.")
    df.index = pd.to_datetime(df[date_index_name], format=date_format).values
    del df[date_index_name]
    df.index.name = None
    return df

def sum_at_frequency(perc_return: pd.Series, freq: str = "NATURAL") -> pd.Series:
    """
    freq='NATURAL' means no resampling (use daily returns as-is).
    Otherwise use 'Y','M','W' for year/month/week.
    """
    if freq.upper() == "NATURAL":
        return perc_return
    elif freq.upper() == "Y":
        return perc_return.resample("Y").sum()
    elif freq.upper() == "M":
        return perc_return.resample("M").sum()
    elif freq.upper() == "W":
        return perc_return.resample("W").sum()
    else:
        return perc_return

def ann_mean_std(perc_return: pd.Series, freq: str = "NATURAL") -> tuple[float, float]:
    """
    Compute annualized mean & std from daily or resampled returns.
    """
    ret = sum_at_frequency(perc_return, freq)
    if freq.upper() == "NATURAL":
        periods_per_year = BUSINESS_DAYS_IN_YEAR
    elif freq.upper() == "Y":
        periods_per_year = 1
    elif freq.upper() == "M":
        periods_per_year = 12
    elif freq.upper() == "W":
        periods_per_year = 52
    else:
        periods_per_year = BUSINESS_DAYS_IN_YEAR
    mu = ret.mean() * periods_per_year
    sigma = ret.std() * np.sqrt(periods_per_year)
    return mu, sigma

def calculate_drawdown(perc_return: pd.Series) -> pd.Series:
    cum = perc_return.cumsum()
    running_max = cum.cummax()
    return running_max - cum

QUANT_PERCENTILE_EXTREME = 0.01
QUANT_PERCENTILE_STD = 0.30
NORMAL_DISTR_RATIO = norm.ppf(QUANT_PERCENTILE_EXTREME) / norm.ppf(QUANT_PERCENTILE_STD)

def demeaned_remove_zeros(x: pd.Series) -> pd.Series:
    x = x.copy()
    x[x == 0] = np.nan
    return x - x.mean()

def calculate_quant_ratio_lower(x: pd.Series) -> float:
    x_dm = demeaned_remove_zeros(x)
    raw = x_dm.quantile(QUANT_PERCENTILE_EXTREME) / x_dm.quantile(QUANT_PERCENTILE_STD)
    return raw / NORMAL_DISTR_RATIO

def calculate_quant_ratio_upper(x: pd.Series) -> float:
    x_dm = demeaned_remove_zeros(x)
    raw = x_dm.quantile(1 - QUANT_PERCENTILE_EXTREME) / x_dm.quantile(1 - QUANT_PERCENTILE_STD)
    return raw / NORMAL_DISTR_RATIO

def calculate_stats(perc_return: pd.Series, freq: str = "NATURAL") -> dict:
    ann_mu, ann_sigma = ann_mean_std(perc_return, freq)
    sharpe = ann_mu / ann_sigma if ann_sigma != 0 else np.nan
    skew = sum_at_frequency(perc_return, freq).skew()
    dd = calculate_drawdown(perc_return)
    avg_dd = dd.mean()
    max_dd = dd.max()
    qr_lower = calculate_quant_ratio_lower(perc_return)
    qr_upper = calculate_quant_ratio_upper(perc_return)
    return {
        "ann_mean": ann_mu,
        "ann_std": ann_sigma,
        "sharpe_ratio": sharpe,
        "skew": skew,
        "avg_drawdown": avg_dd,
        "max_drawdown": max_dd,
        "quant_ratio_lower": qr_lower,
        "quant_ratio_upper": qr_upper
    }

# =============================================================================
# Chapter 3 & 4: Data Loading, Volatility, and Position Sizing
# =============================================================================

def load_data_from_symbols(symbols_csv: str) -> tuple[dict, dict, dict]:
    """
    Reads your symbols CSV file and loads data for each instrument.
    Returns dictionaries:
      - adjusted_prices: keys are instrument IDs (e.g., 'es' from "Data/es_daily_data.csv"),
      - current_prices: same as adjusted_prices,
      - multipliers: from the CSV.
    """
    df = pd.read_csv(symbols_csv, comment='#')
    df["Multiplier"] = df["Multiplier"].apply(lambda x: float(str(x).split()[0]))
    adjusted_prices = {}
    current_prices = {}
    multipliers = {}
    for _, row in df.iterrows():
        file_path = row["Symbol"]
        if not os.path.exists(file_path):
            print(f"File not found: {file_path} – skipping.")
            continue
        data = pd_readcsv(file_path)
        data = data.dropna()
        if "adjusted" in data.columns and "underlying" in data.columns:
            price_adj = data["adjusted"]
            price_curr = data["underlying"]
        elif "Last" in data.columns:
            price_adj = data["Last"]
            price_curr = data["Last"]
        else:
            raise ValueError(f"Expected price columns not found in {file_path}")
        base = os.path.basename(file_path)
        inst_id = base.split("_")[0].lower()
        adjusted_prices[inst_id] = price_adj
        current_prices[inst_id] = price_curr
        multipliers[inst_id] = row["Multiplier"]
    return adjusted_prices, current_prices, multipliers

def create_fx_series_given_adjusted_prices_dict(adjusted_prices_dict: dict) -> dict:
    fx_series = {}
    for inst, series in adjusted_prices_dict.items():
        fx_series[inst] = pd.Series(1.0, index=series.index)
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
    factor = (BUSINESS_DAYS_IN_YEAR ** 0.5) if annualise_stdev else 1.0
    annualised_std = daily_exp_std * factor
    ten_year_vol = annualised_std.rolling(BUSINESS_DAYS_IN_YEAR*10, min_periods=1).mean()
    weighted_vol = 0.3 * ten_year_vol + 0.7 * annualised_std
    return weighted_vol

class standardDeviation(pd.Series):
    def __init__(self,
                 adjusted_price: pd.Series,
                 current_price: pd.Series,
                 use_perc_returns: bool = True,
                 annualise_stdev: bool = True):
        stddev_series = calculate_variable_standard_deviation_for_risk_targeting(
            adjusted_price, current_price, use_perc_returns, annualise_stdev
        )
        super().__init__(stddev_series)
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
        stddev = self.copy()
        if self.annualised:
            stddev /= (BUSINESS_DAYS_IN_YEAR ** 0.5)
        if self.use_perc_returns:
            stddev = stddev * self.current_price
        return stddev

def calculate_variable_standard_deviation_for_risk_targeting_from_dict(adjusted_prices: dict,
                                                                       current_prices: dict,
                                                                       use_perc_returns: bool = True,
                                                                       annualise_stdev: bool = True) -> dict:
    std_dev_dict = {}
    for inst in adjusted_prices.keys():
        std_dev_dict[inst] = standardDeviation(
            adjusted_prices[inst],
            current_prices[inst],
            use_perc_returns,
            annualise_stdev
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
        effective_capital = capital * idm * weights.get(inst, 1.0)
        pos_dict[inst] = calculate_position_series_given_variable_risk(
            capital=effective_capital,
            risk_target_tau=risk_target_tau,
            fx=fx_series_dict[inst],
            multiplier=multipliers[inst],
            instrument_risk=std_dev_dict[inst]
        )
    return pos_dict

# =============================================================================
# Chapter 5: Cost-Adjusted Percentage Returns
# =============================================================================

def calculate_deflated_costs(stddev_series: standardDeviation, cost_per_contract: float) -> pd.Series:
    stdev_daily = stddev_series.daily_risk_price_terms()
    final_stdev = stdev_daily.iloc[-1]
    cost_deflator = stdev_daily / final_stdev
    return cost_per_contract * cost_deflator

def calculate_costs_deflated_for_vol(stddev_series: standardDeviation,
                                     cost_per_contract: float,
                                     position_contracts_held: pd.Series) -> pd.Series:
    rounded_positions = position_contracts_held.round()
    position_change = rounded_positions.diff()
    abs_trades = position_change.abs()
    historic_cost = calculate_deflated_costs(stddev_series, cost_per_contract)
    historic_cost_aligned = historic_cost.reindex(abs_trades.index, method="ffill")
    return abs_trades * historic_cost_aligned

def calculate_perc_returns_with_costs(position_contracts_held: pd.Series,
                                      adjusted_price: pd.Series,
                                      fx_series: pd.Series,
                                      stddev_series: standardDeviation,
                                      multiplier: float,
                                      capital_required: float,
                                      cost_per_contract: float) -> pd.Series:
    pl_points = (adjusted_price - adjusted_price.shift(1)) * position_contracts_held.shift(1)
    pl_currency = pl_points * multiplier
    costs = calculate_costs_deflated_for_vol(stddev_series, cost_per_contract, position_contracts_held)
    costs_aligned = costs.reindex(pl_currency.index, method="ffill")
    net_pl = pl_currency - costs_aligned
    fx_aligned = fx_series.reindex(net_pl.index, method="ffill")
    net_pl_in_base = net_pl * fx_aligned
    return net_pl_in_base / capital_required

def calculate_perc_returns_for_dict_with_costs(position_contracts_dict: dict,
                                               adjusted_prices: dict,
                                               multipliers: dict,
                                               fx_series_dict: dict,
                                               capital: float,
                                               cost_per_contract_dict: dict,
                                               std_dev_dict: dict) -> dict:
    pr_dict = {}
    for inst in position_contracts_dict.keys():
        pr_dict[inst] = calculate_perc_returns_with_costs(
            position_contracts_held=position_contracts_dict[inst],
            adjusted_price=adjusted_prices[inst],
            fx_series=fx_series_dict[inst],
            stddev_series=std_dev_dict[inst],
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
                                               stddev_ann_perc: standardDeviation,
                                               fast_span: int = 64) -> pd.Series:
    ewmac_vals = ewmac(adjusted_price, fast_span=fast_span, slow_span=fast_span*4)
    daily_vol = stddev_ann_perc.daily_risk_price_terms()
    return ewmac_vals / daily_vol

def calculate_scaled_forecast_for_ewmac(adjusted_price: pd.Series,
                                        stddev_ann_perc: standardDeviation,
                                        fast_span: int = 64) -> pd.Series:
    scalar_dict = {2: 12.1, 4: 8.53, 8: 5.95, 16: 4.10, 32: 2.79, 64: 1.91}
    raw_forecast = calculate_risk_adjusted_forecast_for_ewmac(adjusted_price, stddev_ann_perc, fast_span)
    forecast_scalar = scalar_dict.get(fast_span, 1.0)
    scaled = raw_forecast * forecast_scalar
    return scaled

def calculate_forecast_for_ewmac(adjusted_price: pd.Series,
                                 stddev_ann_perc: standardDeviation,
                                 fast_span: int = 64) -> pd.Series:
    scaled = calculate_scaled_forecast_for_ewmac(adjusted_price, stddev_ann_perc, fast_span)
    capped = scaled.clip(-20, 20)
    return capped

def calculate_combined_ewmac_forecast(adjusted_price: pd.Series,
                                      stddev_ann_perc: standardDeviation,
                                      fast_spans: list) -> pd.Series:
    forecasts = []
    for fs in fast_spans:
        f = calculate_forecast_for_ewmac(adjusted_price, stddev_ann_perc, fs)
        forecasts.append(f)
    df_f = pd.concat(forecasts, axis=1)
    avg_forecast = df_f.mean(axis=1)
    rule_count = len(fast_spans)
    FDM_DICT = {1: 1.0, 2: 1.03, 3: 1.08, 4: 1.13, 5: 1.19, 6: 1.26}
    fdm = FDM_DICT.get(rule_count, 1.0)
    scaled = avg_forecast * fdm
    return scaled.clip(-20, 20)

def calculate_position_with_multiple_trend_forecast_applied(adjusted_price: pd.Series,
                                                            average_position: pd.Series,
                                                            stddev_ann_perc: standardDeviation,
                                                            fast_spans: list) -> pd.Series:
    combined_forecast = calculate_combined_ewmac_forecast(adjusted_price, stddev_ann_perc, fast_spans)
    return combined_forecast * average_position / 10.0

def calculate_position_dict_with_multiple_trend_forecast_applied(adjusted_prices_dict: dict,
                                                                 average_position_contracts_dict: dict,
                                                                 std_dev_dict: dict,
                                                                 fast_spans: list) -> dict:
    pos_dict = {}
    for inst in adjusted_prices_dict.keys():
        pos_dict[inst] = calculate_position_with_multiple_trend_forecast_applied(
            adjusted_price=adjusted_prices_dict[inst],
            average_position=average_position_contracts_dict[inst],
            stddev_ann_perc=std_dev_dict[inst],
            fast_spans=fast_spans
        )
    return pos_dict

def apply_buffer_single_period(last_position: float, top_pos: float, bot_pos: float) -> float:
    if last_position > top_pos:
        return top_pos
    elif last_position < bot_pos:
        return bot_pos
    else:
        return last_position

def apply_buffer(optimal_position: pd.Series, top_buffer: pd.Series, bot_buffer: pd.Series) -> pd.Series:
    top_buffer = top_buffer.ffill().round()
    bot_buffer = bot_buffer.ffill().round()
    pos_optimal = optimal_position.ffill()
    current = pos_optimal.iloc[0] if not pd.isna(pos_optimal.iloc[0]) else 0.0
    out = [current]
    for i in range(1, len(pos_optimal)):
        current = apply_buffer_single_period(current, top_buffer.iloc[i], bot_buffer.iloc[i])
        out.append(current)
    return pd.Series(out, index=pos_optimal.index)

def apply_buffering_to_positions(position_contracts: pd.Series,
                                 average_position_contracts: pd.Series,
                                 buffer_size: float = 0.10) -> pd.Series:
    buf = average_position_contracts.abs() * buffer_size
    top = position_contracts + buf
    bot = position_contracts - buf
    return apply_buffer(position_contracts, top, bot)

def apply_buffering_to_position_dict(position_contracts_dict: dict,
                                     average_position_contracts_dict: dict,
                                     buffer_size: float = 0.10) -> dict:
    buffered = {}
    for inst, pos in position_contracts_dict.items():
        buffered[inst] = apply_buffering_to_positions(pos, average_position_contracts_dict[inst], buffer_size)
    return buffered

# =============================================================================
# MAIN: Strategy 9 Backtest
# =============================================================================

if __name__ == "__main__":
    # Step 1: Load symbols from CSV.
    symbols_file = "Data/symbols.csv"  # Update path if needed.
    adjusted_prices_dict, current_prices_dict, file_multipliers = load_data_from_symbols(symbols_file)
    instruments = list(adjusted_prices_dict.keys())
    print("Loaded instruments:", instruments)

    # Step 2: Use multipliers from the CSV.
    multipliers = file_multipliers

    # Step 3: Create FX series (assume USD 1:1).
    fx_series_dict = create_fx_series_given_adjusted_prices_dict(adjusted_prices_dict)

    # Step 4: Define strategy parameters.
    capital = 1_000_000.0
    risk_target_tau = 0.20
    idm = 1.5
    instrument_weights = {inst: 1.0/len(instruments) for inst in instruments}

    # Set cost per contract (adjust as needed; e.g., sp500=0.875, us10=5)
    cost_per_contract_dict = {inst: 1.0 for inst in instruments}

    # Step 5: Calculate instrument risk.
    std_dev_dict = calculate_variable_standard_deviation_for_risk_targeting_from_dict(
        adjusted_prices=adjusted_prices_dict,
        current_prices=current_prices_dict,
        use_perc_returns=True,
        annualise_stdev=True
    )

    # Step 6: Compute average risk-based position sizing.
    average_position_contracts_dict = calculate_position_series_given_variable_risk_for_dict(
        capital=capital,
        risk_target_tau=risk_target_tau,
        idm=idm,
        weights=instrument_weights,
        fx_series_dict=fx_series_dict,
        multipliers=multipliers,
        std_dev_dict=std_dev_dict,
    )

    # Step 7: Apply multiple EWMAC trend filters.
    fast_spans = [4, 8, 16, 32, 64]
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
        buffer_size=0.10
    )

    # Step 9: Compute cost-adjusted percentage returns.
    perc_return_dict = calculate_perc_returns_for_dict_with_costs(
        position_contracts_dict=buffered_position_dict,
        adjusted_prices=adjusted_prices_dict,
        multipliers=multipliers,
        fx_series_dict=fx_series_dict,
        capital=capital,
        cost_per_contract_dict=cost_per_contract_dict,
        std_dev_dict=std_dev_dict,
    )

    # Step 10: Plot aggregated portfolio equity curve.
    combined_df = pd.concat(perc_return_dict, axis=1)
    portfolio_returns = combined_df.sum(axis=1).dropna()
    portfolio_stats = calculate_stats(portfolio_returns)
    print("Portfolio stats:", portfolio_stats)

    portfolio_equity = capital + (capital * portfolio_returns.cumsum())

    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_equity, label="Aggregated Portfolio Equity")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.title("Strategy 9: Aggregated Portfolio Equity Curve")
    plt.legend()
    plt.ticklabel_format(style='plain', axis='y')
    plt.show()

    # Step 11: Plot ES multi-EWMAC equity curves.
    if "es" in instruments:
        es_adj = adjusted_prices_dict["es"]
        es_std = std_dev_dict["es"]
        all_equities = {}
        for fs in fast_spans:
            single_forecast = calculate_forecast_for_ewmac(es_adj, es_std, fs)
            single_pos = (single_forecast * average_position_contracts_dict["es"] / 10.0)
            single_pos_buffered = apply_buffering_to_positions(single_pos, average_position_contracts_dict["es"], 0.10)
            single_ret = calculate_perc_returns_with_costs(
                position_contracts_held=single_pos_buffered,
                adjusted_price=es_adj,
                fx_series=fx_series_dict["es"],
                stddev_series=es_std,
                multiplier=multipliers["es"],
                capital_required=capital,
                cost_per_contract=cost_per_contract_dict["es"]
            )
            single_equity = capital + (capital * single_ret.cumsum())
            all_equities[f"EWMAC({fs},{fs*4})"] = single_equity

        plt.figure(figsize=(10, 6))
        for label, eq in all_equities.items():
            plt.plot(eq, label=label)
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.title("ES - EWMAC Single-Speed Equity Curves")
        plt.legend()
        plt.ticklabel_format(style='plain', axis='y')
        plt.show()
    else:
        print("No ES instrument found to plot multi-EWMAC speeds.")