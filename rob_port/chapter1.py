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

#####   INSTRUMENT DATA LOADING   #####
def load_instrument_data(instruments_file='Data/instruments.csv'):
    """
    Load instrument data including multipliers and SR costs.
    
    Parameters:
        instruments_file (str): Path to the instruments CSV file.
    
    Returns:
        pd.DataFrame: DataFrame with instrument specifications.
    """
    return pd.read_csv(instruments_file)

def get_instrument_specs(symbol, instruments_df=None):
    """
    Get specifications for a specific instrument.
    
    Parameters:
        symbol (str): Instrument symbol to look up.
        instruments_df (pd.DataFrame): Instruments data. If None, loads from file.
    
    Returns:
        dict: Dictionary containing instrument specifications.
    """
    if instruments_df is None:
        instruments_df = load_instrument_data()
    
    instrument = instruments_df[instruments_df['Symbol'] == symbol]
    
    if instrument.empty:
        raise ValueError(f"Instrument {symbol} not found in instruments data")
    
    return {
        'symbol': symbol,
        'multiplier': instrument['Multiplier'].iloc[0],
        'name': instrument['Name'].iloc[0],
        'currency': instrument['Currency'].iloc[0],
        'sr_cost': instrument['SR_cost'].iloc[0]
    }

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

#####   MAXIMUM DRAWDOWN   #####
def calculate_maximum_drawdown(equity_curve):
    """
    Calculate maximum drawdown from an equity curve.
    
    Parameters:
        equity_curve (pd.Series): Series of equity values over time.
    
    Returns:
        dict: Dictionary containing max drawdown percentage, duration, and other stats.
    """
    # Calculate running maximum (peak)
    rolling_max = equity_curve.cummax()
    
    # Calculate drawdown at each point
    drawdown = (equity_curve - rolling_max) / rolling_max
    
    # Maximum drawdown percentage
    max_drawdown_pct = drawdown.min() * 100
    
    # Calculate drawdown durations
    in_drawdown = drawdown < 0
    drawdown_periods = []
    start_idx = None
    
    for idx, is_dd in in_drawdown.items():
        if is_dd and start_idx is None:
            start_idx = idx
        elif not is_dd and start_idx is not None:
            duration = (idx - start_idx).total_seconds() / 86400  # days
            drawdown_periods.append(duration)
            start_idx = None
    
    # Handle case where drawdown continues to end
    if start_idx is not None:
        duration = (equity_curve.index[-1] - start_idx).total_seconds() / 86400
        drawdown_periods.append(duration)
    
    max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
    avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
    avg_drawdown_pct = drawdown[drawdown < 0].mean() * 100 if (drawdown < 0).any() else 0
    
    return {
        'max_drawdown_pct': max_drawdown_pct,
        'avg_drawdown_pct': avg_drawdown_pct,
        'max_drawdown_duration_days': max_drawdown_duration,
        'avg_drawdown_duration_days': avg_drawdown_duration,
        'drawdown_series': drawdown
    }

#####   SKEW CALCULATIONS   #####
def calculate_skew(returns):
    """
    Calculate skewness of returns distribution.
    
    Parameters:
        returns (pd.Series or np.array): Series of returns.
    
    Returns:
        float: Skewness value.
    """
    if isinstance(returns, pd.Series):
        return returns.skew()
    else:
        # Manual calculation for numpy arrays
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        n = len(returns)
        
        if std_return == 0:
            return 0
        
        skew = (n / ((n-1) * (n-2))) * np.sum(((returns - mean_return) / std_return) ** 3)
        return skew

#####   CURRENCY CONVERSIONS   #####
def calculate_notional_exposure(multiplier, price, fx_rate=1.0):
    """
    Calculate notional exposure in base currency.
    
    Parameters:
        multiplier (float): Futures contract multiplier.
        price (float): Current price of the instrument.
        fx_rate (float): FX rate to convert to base currency (default 1.0).
    
    Returns:
        float: Notional exposure in base currency.
    """
    return multiplier * price * fx_rate

def convert_to_base_currency(instrument_return, multiplier, fx_rate):
    """
    Convert instrument return to base currency return.
    
    Parameters:
        instrument_return (float): Return in instrument currency.
        multiplier (float): Futures contract multiplier.
        fx_rate (float): FX rate to base currency.
    
    Returns:
        float: Return in base currency.
    """
    return instrument_return * multiplier * fx_rate

#####   ACCOUNT CURVE AND PERCENTAGE RETURNS   #####
def calculate_percentage_returns_from_capital(price_returns, capital_required):
    """
    Convert price point returns to percentage returns based on capital.
    
    Parameters:
        price_returns (pd.Series): Returns in price points.
        capital_required (float or pd.Series): Capital required for position.
    
    Returns:
        pd.Series: Percentage returns based on capital.
    """
    if isinstance(capital_required, (int, float)):
        return price_returns / capital_required
    else:
        return price_returns / capital_required

def build_account_curve(returns, initial_capital=100):
    """
    Build cumulative account curve from returns.
    
    Parameters:
        returns (pd.Series): Series of percentage returns.
        initial_capital (float): Starting capital value (default 100).
    
    Returns:
        pd.Series: Cumulative account curve.
    """
    return initial_capital * (1 + returns).cumprod()

#####   TRADING COSTS WITH SR_COST   #####
def get_sr_cost_for_instrument(symbol, instruments_df=None):
    """
    Get the SR cost for a specific instrument from the instruments data.
    
    Parameters:
        symbol (str): Instrument symbol.
        instruments_df (pd.DataFrame): Instruments dataframe. If None, loads from file.
    
    Returns:
        float: SR cost for the instrument.
    """
    if instruments_df is None:
        instruments_df = load_instrument_data()
    
    instrument_specs = get_instrument_specs(symbol, instruments_df)
    return instrument_specs['sr_cost']

def calculate_portfolio_sr_cost(instruments_weights, instruments_df=None):
    """
    Calculate weighted SR cost for a portfolio of instruments.
    
    Parameters:
        instruments_weights (dict): Dictionary with symbol: weight pairs.
        instruments_df (pd.DataFrame): Instruments dataframe. If None, loads from file.
    
    Returns:
        float: Weighted average SR cost for the portfolio.
    """
    if instruments_df is None:
        instruments_df = load_instrument_data()
    
    total_sr_cost = 0
    total_weight = 0
    
    for symbol, weight in instruments_weights.items():
        sr_cost = get_sr_cost_for_instrument(symbol, instruments_df)
        total_sr_cost += sr_cost * weight
        total_weight += weight
    
    return total_sr_cost / total_weight if total_weight != 0 else 0

#####   TICK VALUE CALCULATIONS   #####
def calculate_tick_value(multiplier, tick_size):
    """
    Calculate the monetary value of one tick.
    
    Parameters:
        multiplier (float): Contract multiplier.
        tick_size (float): Minimum price fluctuation.
    
    Returns:
        float: Value of one tick in currency.
    """
    return multiplier * tick_size

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

#####   COMPREHENSIVE PERFORMANCE METRICS   #####
def calculate_comprehensive_performance(equity_curve, returns, risk_free_rate=0.0):
    """
    Calculate comprehensive performance metrics for a trading strategy.
    
    Parameters:
        equity_curve (pd.Series): Equity curve over time.
        returns (pd.Series): Series of returns.
        risk_free_rate (float): Risk-free rate for Sharpe calculation.
    
    Returns:
        dict: Comprehensive performance metrics.
    """
    # Basic return metrics
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    
    # Time period
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = days / 365.25
    
    # Annualized return
    if years > 0:
        annualized_return = ((1 + total_return) ** (1/years)) - 1
    else:
        annualized_return = np.nan
    
    # Volatility metrics
    daily_vol = returns.std()
    annualized_vol = daily_vol * np.sqrt(business_days_per_year)
    
    # Risk-adjusted metrics
    excess_returns = returns - (risk_free_rate / business_days_per_year)
    sharpe_ratio = excess_returns.mean() / daily_vol * np.sqrt(business_days_per_year) if daily_vol != 0 else 0
    
    # Downside risk metrics
    downside_returns = returns[returns < 0]
    sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(business_days_per_year) if len(downside_returns) > 0 and downside_returns.std() != 0 else np.inf
    
    # Drawdown metrics
    drawdown_stats = calculate_maximum_drawdown(equity_curve)
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(drawdown_stats['max_drawdown_pct'] / 100) if drawdown_stats['max_drawdown_pct'] != 0 else np.inf
    
    # Skewness
    skewness = calculate_skew(returns)
    
    # Fat tails (if we have price data)
    # Note: This would need actual price data, not just returns
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_vol,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'skewness': skewness,
        'max_drawdown_pct': drawdown_stats['max_drawdown_pct'],
        'avg_drawdown_pct': drawdown_stats['avg_drawdown_pct'],
        'max_drawdown_duration_days': drawdown_stats['max_drawdown_duration_days'],
        'avg_drawdown_duration_days': drawdown_stats['avg_drawdown_duration_days'],
        'days_traded': days,
        'years_traded': years
    }

def main():
    """
    Main function that reads the CSV, calculates daily/annualized standard deviation,
    Sharpe ratio, and fat-tail metrics, then prints the results.
    """

    # Load instruments data
    instruments_df = load_instrument_data()
    print("Available instruments:")
    print(instruments_df[['Symbol', 'Name', 'Currency', 'SR_cost']].head(10))
    
    # Read the dataset 'Data/mes_daily_data.csv'
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

    # Create a simple equity curve for demonstration
    initial_capital = 10000
    equity_curve = build_account_curve(df['daily_returns'], initial_capital)

    # Calculate comprehensive performance metrics
    comprehensive_stats = calculate_comprehensive_performance(equity_curve, df['daily_returns'])

    # Get MES instrument specifications and SR cost
    try:
        mes_specs = get_instrument_specs('MES', instruments_df)
        mes_sr_cost = mes_specs['sr_cost']
        mes_multiplier = mes_specs['multiplier']
        
        print(f"\n----- MES Instrument Specifications -----")
        print(f"Name: {mes_specs['name']}")
        print(f"Multiplier: {mes_multiplier}")
        print(f"Currency: {mes_specs['currency']}")
        print(f"SR Cost: {mes_sr_cost}")
        
        # Calculate tick value for MES
        tick_value = calculate_tick_value(mes_multiplier, 0.25)  # MES tick size is 0.25
        
        # Example notional exposure calculation
        current_price = df['Last'].iloc[-1] if len(df) > 0 else 4500  # fallback price
        notional_exposure = calculate_notional_exposure(mes_multiplier, current_price)
        
    except ValueError as e:
        print(f"Error getting MES specifications: {e}")
        mes_sr_cost = 0.00028  # fallback value from the CSV
        tick_value = 1.25  # 5 * 0.25
        notional_exposure = 5 * 4500  # fallback calculation

    # Print results
    print("\n----- Basic Calculations -----")
    print("Daily Standard Deviation:", daily_std_dev)
    print("Annualized Standard Deviation:", annual_std_dev)
    print("Daily Mean Return:", daily_mean_return)
    print("Annual Mean Return:", annual_mean_return)
    print("Annualized Sharpe Ratio:", sharpe_ratio)
    
    print("\n----- Trading Costs -----")
    print("Tick Value:", tick_value)
    print("SR Cost (from instruments file):", mes_sr_cost)
    print("Notional Exposure:", notional_exposure)

    print("\n----- Comprehensive Performance -----")
    for key, value in comprehensive_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print("\n----- Fat Tail Statistics -----")
    for k, v in fat_tail_stats.items():
        print(f"  {k}: {v}")
    
    # Example: Portfolio SR cost calculation
    example_portfolio = {'MES': 0.5, 'MYM': 0.3, 'MNQ': 0.2}
    try:
        portfolio_sr_cost = calculate_portfolio_sr_cost(example_portfolio, instruments_df)
        print(f"\n----- Example Portfolio SR Cost -----")
        print(f"Portfolio weights: {example_portfolio}")
        print(f"Weighted SR Cost: {portfolio_sr_cost:.6f}")
    except Exception as e:
        print(f"Error calculating portfolio SR cost: {e}")

if __name__ == "__main__":
    main()