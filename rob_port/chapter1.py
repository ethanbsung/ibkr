import pandas as pd
import numpy as np

# Variables
multiplier = 5
price = 0  # Placeholder for price
notional_exposure_per_contract = multiplier * price
commission = 1.24
slippage = 0.625

# We assume 256 business days in a year for your calculations
business_days_per_year = 256

#####   STANDARD DEVIATION  #####
def calculate_standard_deviation(returns):
    """
    Calculates the standard deviation of a series of returns.

    Parameters:
    returns (list or np.array): A list or numpy array of returns.

    Returns:
    float: The standard deviation of returns.
    """
    T = len(returns)  # Number of periods
    r_mean = np.sum(returns) / T  # Mean return

    variance = np.sum([(1 / T) * (r - r_mean) ** 2 for r in returns])  # Variance formula
    std_dev = np.sqrt(variance)  # Standard deviation

    return std_dev

def annualized_standard_deviation(daily_std_dev, business_days_per_year=256):
    """
    Converts daily standard deviation to an annualized standard deviation.

    Parameters:
    daily_std_dev (float): Daily standard deviation of returns.
    business_days_per_year (int): Number of trading days in a year (default 256).

    Returns:
    float: Annualized standard deviation.
    """
    return daily_std_dev * np.sqrt(business_days_per_year)

#####   FAT TAILS   #####
def calculate_fat_tails(df):
    """
    Calculate fat tail statistics for given OHLCV data.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'Last' prices indexed by date.
    
    Returns:
        dict: Fat tail statistics including percentiles, ratios, and relative ratios.
    """
    # Compute daily returns
    df['returns'] = df['Last'].pct_change().dropna()
    
    # Demean the return series
    df['demeaned_returns'] = df['returns'] - df['returns'].mean()
    
    # Calculate percentiles
    percentiles = {
        '1st': np.percentile(df['demeaned_returns'].dropna(), 1),
        '30th': np.percentile(df['demeaned_returns'].dropna(), 30),
        '70th': np.percentile(df['demeaned_returns'].dropna(), 70),
        '99th': np.percentile(df['demeaned_returns'].dropna(), 99),
    }

    # Calculate percentile ratios
    lower_percentile_ratio = percentiles['1st'] / percentiles['30th']
    upper_percentile_ratio = percentiles['99th'] / percentiles['70th']

    # Gaussian normal benchmark ratio
    gaussian_ratio = 4.43

    # Compute relative fat tail ratios
    relative_lower_fat_tail_ratio = lower_percentile_ratio / gaussian_ratio
    relative_upper_fat_tail_ratio = upper_percentile_ratio / gaussian_ratio

    return {
        'percentiles': percentiles,
        'lower_percentile_ratio': lower_percentile_ratio,
        'upper_percentile_ratio': upper_percentile_ratio,
        'relative_lower_fat_tail_ratio': relative_lower_fat_tail_ratio,
        'relative_upper_fat_tail_ratio': relative_upper_fat_tail_ratio
    }

def main():
    """
    Main function that reads the CSV, calculates daily/annualized standard deviation,
    Sharpe ratio, and fat-tail metrics, then prints the results.
    """

    # Read the dataset 'Data/es_daily_data.csv'
    df = pd.read_csv('Data/mes_daily_data.csv', parse_dates=['Time'])
    df.set_index('Time', inplace=True)

    # Calculate daily returns for standard deviation
    df['daily_returns'] = df['Last'].pct_change()

    # Drop the first NaN row after pct_change
    df.dropna(subset=['daily_returns'], inplace=True)

    # Convert daily_returns to a numpy array
    daily_returns_array = df['daily_returns'].values

    # Calculate daily mean return
    daily_mean_return = df['daily_returns'].mean()

    # Calculate daily standard deviation
    daily_std_dev = calculate_standard_deviation(daily_returns_array)

    # Annualize the mean return and standard deviation
    annual_mean_return = daily_mean_return * business_days_per_year
    annual_std_dev = annualized_standard_deviation(daily_std_dev, business_days_per_year)

    # Calculate the annualized Sharpe ratio
    # If you have a risk-free rate, you can subtract it from annual_mean_return here
    sharpe_ratio = annual_mean_return / annual_std_dev if annual_std_dev != 0 else 0.0

    # Calculate fat-tail metrics
    fat_tail_stats = calculate_fat_tails(df.copy())

    # Print results
    print("----- Standard Deviation -----")
    print("Daily Standard Deviation:", daily_std_dev)
    print("Annualized Standard Deviation:", annual_std_dev)

    print("\n----- Sharpe Ratio -----")
    print("Daily Mean Return:", daily_mean_return)
    print("Annual Mean Return:", annual_mean_return)
    print("Annualized Sharpe Ratio:", sharpe_ratio)

    print("\n----- Fat Tail Statistics -----")
    for k, v in fat_tail_stats.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()