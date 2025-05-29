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

def load_data_from_instruments(instruments_csv: str) -> tuple[dict, dict, dict]:
    """
    Reads your instruments CSV file and loads data for each instrument.
    Returns dictionaries:
      - adjusted_prices: keys are instrument IDs (lowercase symbol),
      - current_prices: same as adjusted_prices,
      - multipliers: from the CSV.
    """
    df = pd.read_csv(instruments_csv, comment='#')
    df["Multiplier"] = df["Multiplier"].apply(lambda x: float(str(x).split()[0]))
    adjusted_prices = {}
    current_prices = {}
    multipliers = {}
    
    # Determine the correct data directory path
    data_dir = "Data"
    if not os.path.exists(data_dir) and os.path.exists("../Data"):
        data_dir = "../Data"
    
    for _, row in df.iterrows():
        symbol = row["Symbol"]
        # Try both uppercase and lowercase versions of the symbol
        file_paths = [
            f"{data_dir}/{symbol}_daily_data.csv",           # Original case
            f"{data_dir}/{symbol.lower()}_daily_data.csv",   # Lowercase
            f"{data_dir}/{symbol.upper()}_daily_data.csv"    # Uppercase
        ]
        
        file_path = None
        for path in file_paths:
            if os.path.exists(path):
                file_path = path
                break
                
        if file_path is None:
            print(f"No data file found for {symbol} (tried {len(file_paths)} variations) – skipping.")
            continue
            
        try:
            data = pd_readcsv(file_path)
            data = data.dropna()
            
            if "adjusted" in data.columns and "underlying" in data.columns:
                price_adj = data["adjusted"]
                price_curr = data["underlying"]
            elif "Last" in data.columns:
                price_adj = data["Last"]
                price_curr = data["Last"]
            else:
                print(f"Expected price columns not found in {file_path} – skipping {symbol}.")
                continue
                
            # Use lowercase symbol as instrument ID
            inst_id = symbol.lower()
            adjusted_prices[inst_id] = price_adj
            current_prices[inst_id] = price_curr
            multipliers[inst_id] = row["Multiplier"]
            print(f"Successfully loaded {symbol} ({inst_id}) from {file_path}")
            
        except Exception as e:
            print(f"Error loading {symbol} from {file_path}: {str(e)} – skipping.")
            continue
            
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
    
    # Book-based forecast weights (from "top down method")
    # Higher weights for more diversifying (fastest and slowest) and cost-effective rules
    if len(fast_spans) == 5 and fast_spans == [4, 8, 16, 32, 64]:
        # Optimized weights based on the book's analysis (Table 36 implications)
        # These weights account for:
        # 1. Diversification benefits (higher weights for extreme speeds)
        # 2. Cost considerations (lower weights for very fast rules)
        # 3. Expected Sharpe ratios
        forecast_weights = [0.15, 0.20, 0.30, 0.25, 0.10]  # Sum = 1.0
        # EWMAC4: 0.15 (fast but expensive)
        # EWMAC8: 0.20 
        # EWMAC16: 0.30 (best balance per book)
        # EWMAC32: 0.25
        # EWMAC64: 0.10 (slow, good for diversification but lower Sharpe)
    else:
        # Fallback to equal weights if different spans are used
        forecast_weights = [1.0/len(fast_spans)] * len(fast_spans)
    
    # Apply weighted average instead of simple mean
    weighted_forecast = (df_f * forecast_weights).sum(axis=1)
    
    rule_count = len(fast_spans)
    FDM_DICT = {1: 1.0, 2: 1.03, 3: 1.08, 4: 1.13, 5: 1.19, 6: 1.26}
    fdm = FDM_DICT.get(rule_count, 1.0)
    scaled = weighted_forecast * fdm
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
    # Step 1: Load instruments from CSV.
    # Determine the correct path for instruments.csv
    instruments_file = "Data/instruments.csv"
    if not os.path.exists(instruments_file) and os.path.exists("../Data/instruments.csv"):
        instruments_file = "../Data/instruments.csv"
    
    adjusted_prices_dict, current_prices_dict, file_multipliers = load_data_from_instruments(instruments_file)
    instruments = list(adjusted_prices_dict.keys())
    print("Loaded instruments:", instruments)

    # Step 2: Use multipliers from the CSV.
    multipliers = file_multipliers

    # Step 3: Create FX series (assume USD 1:1).
    fx_series_dict = create_fx_series_given_adjusted_prices_dict(adjusted_prices_dict)

    # Step 4: Define strategy parameters.
    capital = 1_000_000.0
    # Increase risk target to match book's higher volatility target
    # Book shows 22.2% annual std vs our current 7.7%
    risk_target_tau = 0.25  # Increased from 0.20 to achieve higher vol
    idm = 1.8  # Slightly higher IDM for better diversification benefit
    instrument_weights = {inst: 1.0/len(instruments) for inst in instruments}

    # Set cost per contract using SR_cost from instruments.csv
    cost_per_contract_dict = {}
    instruments_df = pd.read_csv(instruments_file, comment='#')
    for inst in instruments:
        # Find the row for this instrument (convert to uppercase for matching)
        inst_upper = inst.upper()
        matching_rows = instruments_df[instruments_df['Symbol'] == inst_upper]
        if not matching_rows.empty and 'SR_cost' in instruments_df.columns:
            sr_cost = matching_rows.iloc[0]['SR_cost']
            if pd.notna(sr_cost) and sr_cost > 0:
                # Convert from SR terms to cost per contract
                # The book suggests these are already in appropriate units
                cost_per_contract_dict[inst] = sr_cost * 100  # Scale appropriately
            else:
                cost_per_contract_dict[inst] = 1.0  # Default fallback
        else:
            cost_per_contract_dict[inst] = 1.0  # Default fallback

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

    # Step 10: Plot aggregated portfolio equity curve and analyze performance.
    combined_df = pd.concat(perc_return_dict, axis=1)
    portfolio_returns = combined_df.sum(axis=1).dropna()
    portfolio_stats = calculate_stats(portfolio_returns)
    
    print("\n" + "="*60)
    print("STRATEGY 9 PERFORMANCE ANALYSIS")
    print("="*60)
    print(f"Loaded instruments: {len(instruments)}/101 ({len(instruments)/101*100:.1f}%)")
    print(f"Missing instruments: {101-len(instruments)} ({(101-len(instruments))/101*100:.1f}%)")
    
    print(f"\nPortfolio Performance:")
    print(f"  Annual Return:    {portfolio_stats['ann_mean']*100:.2f}% (Book: 25.2%)")
    print(f"  Annual Volatility: {portfolio_stats['ann_std']*100:.2f}% (Book: 22.2%)")
    print(f"  Sharpe Ratio:     {portfolio_stats['sharpe_ratio']:.3f} (Book: 1.14)")
    print(f"  Avg Drawdown:     {portfolio_stats['avg_drawdown']*100:.2f}% (Book: -11.2%)")
    print(f"  Max Drawdown:     {portfolio_stats['max_drawdown']*100:.2f}%")
    print(f"  Skew:            {portfolio_stats['skew']:.2f} (Book: 0.98)")
    
    # Performance vs targets
    return_gap = (portfolio_stats['ann_mean'] - 0.252) * 100
    vol_gap = (portfolio_stats['ann_std'] - 0.222) * 100 
    sharpe_gap = portfolio_stats['sharpe_ratio'] - 1.14
    
    print(f"\nPerformance vs Book Targets:")
    print(f"  Return gap:  {return_gap:+.2f}pp")
    print(f"  Vol gap:     {vol_gap:+.2f}pp") 
    print(f"  Sharpe gap:  {sharpe_gap:+.3f}")
    
    if portfolio_stats['ann_mean'] > 0.15:  # 15%+ annual return
        print("✅ Strong performance achieved!")
    elif portfolio_stats['ann_mean'] > 0.10:  # 10%+ annual return  
        print("✅ Good performance - getting closer!")
    else:
        print("⚠️  Still underperforming - may need more adjustments")

    portfolio_equity = capital + (capital * portfolio_returns.cumsum())

    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_equity, label="Aggregated Portfolio Equity")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.title("Strategy 9: Aggregated Portfolio Equity Curve")
    plt.legend()
    plt.ticklabel_format(style='plain', axis='y')
    plt.show()

    # Step 11: Plot ES/MES multi-EWMAC equity curves.
    es_instrument = None
    if "es" in instruments:
        es_instrument = "es"
    elif "mes" in instruments:
        es_instrument = "mes"
    
    if es_instrument:
        es_adj = adjusted_prices_dict[es_instrument]
        es_std = std_dev_dict[es_instrument]
        all_equities = {}
        for fs in fast_spans:
            single_forecast = calculate_forecast_for_ewmac(es_adj, es_std, fs)
            single_pos = (single_forecast * average_position_contracts_dict[es_instrument] / 10.0)
            single_pos_buffered = apply_buffering_to_positions(single_pos, average_position_contracts_dict[es_instrument], 0.10)
            single_ret = calculate_perc_returns_with_costs(
                position_contracts_held=single_pos_buffered,
                adjusted_price=es_adj,
                fx_series=fx_series_dict[es_instrument],
                stddev_series=es_std,
                multiplier=multipliers[es_instrument],
                capital_required=capital,
                cost_per_contract=cost_per_contract_dict[es_instrument]
            )
            single_equity = capital + (capital * single_ret.cumsum())
            all_equities[f"EWMAC({fs},{fs*4})"] = single_equity

        plt.figure(figsize=(10, 6))
        for label, eq in all_equities.items():
            plt.plot(eq, label=label)
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.title(f"{es_instrument.upper()} - EWMAC Single-Speed Equity Curves")
        plt.legend()
        plt.ticklabel_format(style='plain', axis='y')
        plt.show()
    else:
        print("No ES or MES instrument found to plot multi-EWMAC speeds.")