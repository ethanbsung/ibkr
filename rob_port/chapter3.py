from chapter2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chapter1 import calculate_standard_deviation, annualized_standard_deviation, calculate_fat_tails

# Use the business days constant (256)
business_days_in_year = 256

def calculate_variable_standard_deviation_for_risk_targeting(
    adjusted_price: pd.Series,
    current_price: pd.Series,
    use_perc_returns: bool = True,
    annualise_stdev: bool = True,
) -> pd.Series:
    """
    Calculate the variable (EWMA-based) standard deviation for risk targeting,
    using the author's method:
    
      1. Compute daily percentage returns if use_perc_returns is True.
      2. Compute the EWMA standard deviation over a 32-day span.
      3. Annualise the EWMA standard deviation by multiplying by √(business_days_in_year).
      4. Compute a 10-year rolling average of the annualised EWMA volatility.
      5. Blend the long-run (10-year) vol and the short-run (annualised EWMA) vol.
      
    Returns a Series of weighted (blended) annual volatility estimates.
    """
    if use_perc_returns:
        # Daily percentage returns: (Price_t - Price_{t-1}) / Price_{t-1}
        daily_returns = adjusted_price.diff() / current_price.shift(1)
    else:
        daily_returns = adjusted_price.diff()
    
    # Compute the EWMA standard deviation over a 32-day span using pandas' built-in ewm
    daily_exp_std_dev = daily_returns.ewm(span=32).std()
    
    if annualise_stdev:
        annualisation_factor = business_days_in_year ** 0.5
    else:
        annualisation_factor = 1
        
    # Annualise the short-run EWMA volatility
    annualised_std_dev = daily_exp_std_dev * annualisation_factor
    
    # Compute the 10-year average volatility using a rolling window
    ten_year_vol = annualised_std_dev.rolling(business_days_in_year * 10, min_periods=1).mean()
    
    # Blend: 30% weight to long-run vol and 70% to short-run annualised EWMA vol
    weighted_vol = 0.3 * ten_year_vol + 0.7 * annualised_std_dev
    
    return weighted_vol

def calculate_position_size_with_variable_risk(capital, multiplier, price, fx_rate, risk_target, sigma_pct):
    """
    Calculate position size using the formula with variable risk.

    N = (Capital × τ) ÷ (Multiplier × Price × FX × σ_%)

    Parameters:
    - capital (float): Account capital.
    - multiplier (float): Futures contract multiplier.
    - price (float): Instrument price.
    - fx_rate (float or pd.Series): FX conversion rate.
    - risk_target (float): Target risk as a fraction (e.g. 0.2 for 20%).
    - sigma_pct (float): Forecasted annual volatility (as a decimal, e.g. 0.25 for 25%).

    Returns:
    - float: Number of contracts (rounded).
    """
    denominator = multiplier * price * fx_rate * sigma_pct
    return round((capital * risk_target) / denominator)

def calculate_annual_risk_adjusted_cost(price, multiplier, commission, spread_points, annualized_std_dev,
                                        rolls_per_year, turnover):
    """
    Calculate the annual risk-adjusted cost.

    Parameters:
    - price (float): Current price of the instrument.
    - multiplier (float): Futures contract multiplier.
    - commission (float): Commission per contract in currency.
    - spread_points (float): Bid-ask spread in price points.
    - annualized_std_dev (float): Annualized standard deviation as a decimal (e.g., 0.16 for 16%).
    - rolls_per_year (int): Number of times the position is rolled annually (default is 4).
    - turnover (int): Number of other trades made per year (default is 6).

    Returns:
    - float: Annual risk-adjusted cost in Sharpe Ratio units.
    """
    spread_cost_currency = multiplier * (spread_points / 2)
    total_cost_per_trade_currency = spread_cost_currency + commission
    total_cost_per_trade_percent = total_cost_per_trade_currency / (price * multiplier)
    risk_adjusted_cost_per_trade = total_cost_per_trade_percent / annualized_std_dev

    holding_cost = risk_adjusted_cost_per_trade * rolls_per_year * 2
    transaction_cost = risk_adjusted_cost_per_trade * turnover
    annual_risk_adjusted_cost = holding_cost + transaction_cost

    return annual_risk_adjusted_cost

def calculate_position_series_with_variable_risk(csv_path, capital, multiplier, fx_rate=1.0, risk_target=0.2, lambda_=0.060061):
    """
    Calculate and visualize position sizes using the EWMA-based variable risk forecast.
    
    This version uses the author's method to compute variable volatility:
      - It computes daily percentage returns,
      - Uses a 32-day EWMA to get short-run volatility,
      - Annualises it,
      - Then computes a 10-year rolling average,
      - And blends them (30% long-run, 70% short-run) to obtain the final annual volatility estimate.
    
    Position size is computed as:
      N = (Capital × τ) ÷ (Multiplier × Price × FX × σ_annual)
    where Price is the previous day's close.
    
    It then computes strategy returns (using the notional exposure) and prints performance metrics,
    and plots both the position size series and the equity curve.
    
    Returns the position on the second last trading day.
    """
    df = pd.read_csv(csv_path, parse_dates=['Time'])
    df.set_index('Time', inplace=True)
    df = df.iloc[:-1]  # Drop last row (incomplete)
    
    # Compute daily returns from 'Last' price.
    df['returns'] = df['Last'].pct_change()
    df.dropna(inplace=True)
    
    # Compute the variable (EWMA-based) annual volatility series using the author's method.
    # Here we use the 'Last' column as both adjusted and current price.
    weighted_vol_series = calculate_variable_standard_deviation_for_risk_targeting(
        adjusted_price=df['Last'],
        current_price=df['Last'],
        use_perc_returns=True,
        annualise_stdev=True,
    )
    
    # For each day (starting after sufficient data), compute position size using previous day's price and volatility.
    position_sizes = []
    # For day i, use i-1 values:
    for i in range(len(df)):
        if i < 1:
            position_sizes.append(0)
            continue
        price_prev = df['Last'].iloc[i-1]
        vol_prev = weighted_vol_series.iloc[i-1]
        # If volatility is NaN, set position to 0 to avoid error.
        if np.isnan(vol_prev):
            pos = 0
        else:
            pos = calculate_position_size_with_variable_risk(
                capital=capital,
                multiplier=multiplier,
                price=price_prev,
                fx_rate=fx_rate,
                risk_target=risk_target,
                sigma_pct=vol_prev
            )
        position_sizes.append(pos)
    
    df['position'] = position_sizes
    df['position'].plot(title="Position Size Over Time", figsize=(12, 5))
    plt.xlabel("Date")
    plt.ylabel("Contracts")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Compute daily strategy PnL and returns.
    # Use previous day's position and previous day's price to compute notional change.
    df['position_shifted'] = df['position'].shift(1)
    df['strategy_pnl'] = df['position_shifted'] * multiplier * df['returns'] * df['Last'].shift(1)
    # Strategy return as a percentage of capital.
    df['strategy_returns'] = df['strategy_pnl'] / capital
    df.dropna(subset=['strategy_returns'], inplace=True)
    
    # Plot equity curve.
    cum_returns = (1 + df['strategy_returns']).cumprod()
    cum_returns.plot(title="Equity Curve", figsize=(12, 5))
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Compute performance metrics.
    daily_returns_array = df['strategy_returns'].values
    daily_mean_return = np.mean(daily_returns_array)
    daily_std_dev = calculate_standard_deviation(daily_returns_array)
    ann_mean_return = daily_mean_return * business_days_in_year
    ann_std_dev = annualized_standard_deviation(daily_std_dev, business_days_in_year)
    sharpe_ratio = ann_mean_return / ann_std_dev if ann_std_dev != 0 else 0.0
    
    # Average drawdown: compute running cumulative returns, then drawdowns.
    cum_returns_series = (1 + df['strategy_returns']).cumprod()
    running_max = cum_returns_series.cummax()
    drawdown = (cum_returns_series - running_max) / running_max
    avg_drawdown = drawdown.mean()
    
    # Skew and tail statistics.
    skewness = pd.Series(daily_returns_array).skew()
    fat_tail_stats = calculate_fat_tails(df[['Last']].copy())
    
    print("\n----- Strategy Metrics -----")
    print("Mean Annual Return:", round(ann_mean_return, 6))
    print("Annualized Std Dev:", round(ann_std_dev, 6))
    print("Sharpe Ratio:", round(sharpe_ratio, 6))
    print("Skew:", round(skewness, 6))
    print("Average Drawdown:", round(avg_drawdown, 6))
    print("\n----- Fat Tail Statistics -----")
    for k, v in fat_tail_stats.items():
        print(f"  {k}: {v}")
    
    print("\nPosition on second last day:", df['position'].iloc[-2])
    return df['position'].iloc[-2]

def min_liquidity_met(avg_vol, annualized_std_percentage, price, multiplier, fx=1):
    return (fx * avg_vol * annualized_std_percentage * price * multiplier) > 1250000

# Example usage with S&P 500 micro futures values.
if __name__ == "__main__":
    price = 4500
    multiplier = 5
    commission = 0.62
    spread_points = 0.25
    annualized_std_dev_val = 0.16

    spread_cost_currency = multiplier * (spread_points / 2)
    total_cost_per_trade_currency = spread_cost_currency + commission

    print("Spread Cost (Currency):", round(spread_cost_currency, 4))
    print("Total Cost per Trade (Currency):", round(total_cost_per_trade_currency, 4))

    # Restore the risk-adjusted cost function call if needed.
    result = calculate_annual_risk_adjusted_cost(
        price=price,
        multiplier=multiplier,
        commission=0.62,
        spread_points=0.25,
        annualized_std_dev=0.16,
        rolls_per_year=4,
        turnover=6
    )
    print("Annual Risk Adjusted Cost (S&P 500 Micro):", round(result, 6))  # Should be ~0.0034

    print("\nCalculating position series with variable risk...")
    calculate_position_series_with_variable_risk(
        csv_path="Data/mes_daily_data.csv",
        capital=100000,
        multiplier=multiplier,
        fx_rate=1.0,
        risk_target=0.2,
        lambda_=0.060061
    )