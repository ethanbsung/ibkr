from chapter2 import *
import numpy as np

def calculate_ewma_std(returns, lambda_=0.06):
    """
    Calculate the exponentially weighted moving standard deviation.

    Parameters:
    - returns (list or np.array): A list or numpy array of returns.
    - lambda_ (float): The decay factor λ used for EWMA (default is 0.06).

    Returns:
    - float: The exponentially weighted standard deviation.
    """
    returns = np.array(returns)
    r_mean = 0
    weighted_sq_diffs = 0
    weight = 1
    total_weight = 0

    for r in reversed(returns):
        r_mean = lambda_ * r + (1 - lambda_) * r_mean
        deviation = r - r_mean
        weighted_sq_diffs += weight * deviation ** 2
        total_weight += weight
        weight *= (1 - lambda_)

    ewma_variance = weighted_sq_diffs / total_weight
    return np.sqrt(ewma_variance)

def calculate_blended_volatility(returns, long_run_std, lambda_=0.060061):
    """
    Calculate a blended estimate of volatility using both long-run and short-run (EWMA) estimates.

    Parameters:
    - returns (list or np.array): A list or numpy array of returns for short-run EWMA volatility.
    - long_run_std (float): The long-run standard deviation (e.g., 10-year average of volatility).
    - lambda_ (float): The decay factor λ for EWMA (default is 0.060061).

    Returns:
    - float: The blended volatility estimate.
    """
    short_run_std = calculate_ewma_std(returns, lambda_)
    blended_std = 0.3 * long_run_std + 0.7 * short_run_std
    return blended_std

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

# Example usage with S&P 500 micro futures values
if __name__ == "__main__":
    price = 4500
    multiplier = 5
    commission = 0.25
    spread_points = 0.25
    annualized_std_dev = 0.16

    spread_cost_currency = multiplier * (spread_points / 2)
    total_cost_per_trade_currency = spread_cost_currency + commission

    print("Spread Cost (Currency):", round(spread_cost_currency, 4))
    print("Total Cost per Trade (Currency):", round(total_cost_per_trade_currency, 4))

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
