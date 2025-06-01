from chapter2 import *
from chapter1 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

#####   VARIABLE VOLATILITY FORECASTING   #####

def calculate_ewma_volatility(returns, span=32, annualize=True):
    """
    Calculate Exponentially Weighted Moving Average volatility.
    
    Formula from book:
        σ_ewma(λ) = sqrt(λ(r_t - r̄)² + λ(1-λ)(r_{t-1} - r̄)² + λ(1-λ)²(r_{t-2} - r̄)² + ...)
    
    Parameters:
        returns (pd.Series): Daily returns series.
        span (int): EWMA span in days (default 32 as per book).
        annualize (bool): Whether to annualize the volatility.
    
    Returns:
        pd.Series: EWMA volatility series.
    """
    # Calculate lambda from span: λ = 2 / (span + 1)
    lambda_param = 2 / (span + 1)
    
    # Calculate EWMA volatility using pandas ewm
    ewma_vol = returns.ewm(span=span, adjust=False).std()
    
    if annualize:
        ewma_vol = ewma_vol * np.sqrt(business_days_per_year)
    
    return ewma_vol

def calculate_blended_volatility(returns, short_span=32, long_years=10, short_weight=0.7, long_weight=0.3, min_vol_floor=0.05):
    """
    Calculate blended volatility forecast using short-run EWMA and long-run average.
    
    From book:
        σ_blend = 0.3 × (Ten year average of σ_t) + 0.7 × σ_t
    
    Parameters:
        returns (pd.Series): Daily returns series.
        short_span (int): EWMA span for short-run volatility.
        long_years (int): Years for long-run rolling average.
        short_weight (float): Weight for short-run volatility (0.7 in book).
        long_weight (float): Weight for long-run volatility (0.3 in book).
        min_vol_floor (float): Minimum volatility floor (0.05 = 5% annually).
    
    Returns:
        pd.Series: Blended volatility forecast.
    """
    # Calculate short-run EWMA volatility
    short_vol = calculate_ewma_volatility(returns, span=short_span, annualize=True)
    
    # Calculate long-run rolling average volatility
    long_window = long_years * business_days_per_year
    long_vol = short_vol.rolling(window=long_window, min_periods=1).mean()
    
    # Blend the two estimates
    blended_vol = long_weight * long_vol + short_weight * short_vol
    
    # Apply minimum volatility floor to prevent position size explosions
    blended_vol = blended_vol.clip(lower=min_vol_floor)
    
    return blended_vol

def calculate_variable_position_size(capital, multiplier, price, blended_volatility, risk_target=0.2, fx_rate=1.0):
    """
    Calculate position size using variable volatility forecast.
    
    Formula:
        N = (Capital × τ) ÷ (Multiplier × Price × FX × σ_blended)
    
    Parameters:
        capital (float): Trading capital.
        multiplier (float): Contract multiplier.
        price (float): Current price.
        blended_volatility (float): Blended volatility forecast.
        risk_target (float): Target risk fraction.
        fx_rate (float): FX rate for currency conversion.
    
    Returns:
        float: Number of contracts (can be fractional).
    """
    if np.isnan(blended_volatility) or blended_volatility == 0:
        return 0
    
    denominator = multiplier * price * fx_rate * blended_volatility
    position_size = (capital * risk_target) / denominator
    
    return position_size

#####   RISK-ADJUSTED TRADING COSTS   #####

def calculate_spread_cost(spread_points, multiplier):
    """
    Calculate spread cost per trade.
    
    Formula:
        Spread cost = (Bid-Offer spread ÷ 2) × Multiplier
    
    Parameters:
        spread_points (float): Bid-ask spread in price points.
        multiplier (float): Contract multiplier.
    
    Returns:
        float: Spread cost in currency.
    """
    return (spread_points / 2) * multiplier

def calculate_total_cost_per_trade(spread_points, multiplier, commission):
    """
    Calculate total cost per trade including spread and commission.
    
    Parameters:
        spread_points (float): Bid-ask spread in price points.
        multiplier (float): Contract multiplier.
        commission (float): Commission per contract.
    
    Returns:
        float: Total cost per trade in currency.
    """
    spread_cost = calculate_spread_cost(spread_points, multiplier)
    return spread_cost + commission

def calculate_cost_as_percentage(total_cost_currency, price, multiplier):
    """
    Calculate cost as percentage of notional exposure.
    
    Formula:
        Cost % = Total cost ÷ (Price × Multiplier)
    
    Parameters:
        total_cost_currency (float): Total cost in currency.
        price (float): Instrument price.
        multiplier (float): Contract multiplier.
    
    Returns:
        float: Cost as percentage (decimal).
    """
    notional_exposure = price * multiplier
    return total_cost_currency / notional_exposure

def calculate_risk_adjusted_cost_per_trade(total_cost_percentage, annualized_volatility):
    """
    Calculate risk-adjusted cost per trade in Sharpe ratio units.
    
    Formula:
        Risk adjusted cost = Cost % ÷ σ_annual
    
    Parameters:
        total_cost_percentage (float): Total cost as percentage.
        annualized_volatility (float): Annualized volatility.
    
    Returns:
        float: Risk-adjusted cost in SR units.
    """
    if annualized_volatility == 0:
        return float('inf')
    return total_cost_percentage / annualized_volatility

def calculate_annual_risk_adjusted_cost(price, multiplier, commission, spread_points, 
                                      annualized_std_dev, rolls_per_year=4, turnover=6):
    """
    Calculate total annual risk-adjusted trading costs.
    
    Parameters:
        price (float): Current price.
        multiplier (float): Contract multiplier.
        commission (float): Commission per trade.
        spread_points (float): Bid-ask spread in points.
        annualized_std_dev (float): Annualized standard deviation.
        rolls_per_year (int): Contract rolls per year.
        turnover (int): Other trades per year.
    
    Returns:
        float: Annual risk-adjusted cost in SR units.
    """
    # Calculate cost per trade
    total_cost_currency = calculate_total_cost_per_trade(spread_points, multiplier, commission)
    cost_percentage = calculate_cost_as_percentage(total_cost_currency, price, multiplier)
    risk_adjusted_cost_per_trade = calculate_risk_adjusted_cost_per_trade(cost_percentage, annualized_std_dev)
    
    # Calculate holding costs (rolling costs - 2 trades per roll)
    holding_cost = risk_adjusted_cost_per_trade * rolls_per_year * 2
    
    # Calculate transaction costs (other trading)
    transaction_cost = risk_adjusted_cost_per_trade * turnover
    
    return holding_cost + transaction_cost

#####   INSTRUMENT SELECTION CRITERIA   #####

def calculate_minimum_capital_for_instrument(symbol, instruments_df, price, volatility, risk_target=0.2, min_contracts=4):
    """
    Calculate minimum capital required to trade an instrument.
    
    Parameters:
        symbol (str): Instrument symbol.
        instruments_df (pd.DataFrame): Instruments data.
        price (float): Current price.
        volatility (float): Annualized volatility.
        risk_target (float): Target risk fraction.
        min_contracts (int): Minimum contracts to trade.
    
    Returns:
        float: Minimum capital required.
    """
    specs = get_instrument_specs(symbol, instruments_df)
    multiplier = specs['multiplier']
    
    # Calculate minimum capital for minimum contracts
    min_capital = calculate_min_capital_n_contracts(
        min_contracts, multiplier, price, volatility, risk_target
    )
    
    return min_capital

def calculate_liquidity_threshold(price, multiplier, volatility, fx_rate=1.0, min_daily_volume_usd=1250000):
    """
    Calculate minimum daily volume required for liquidity.
    
    Formula from book:
        Average daily volume in USD risk = FX rate × Average daily volume × σ_% × Price × Multiplier
    
    Parameters:
        price (float): Instrument price.
        multiplier (float): Contract multiplier.
        volatility (float): Annualized volatility.
        fx_rate (float): FX rate to USD.
        min_daily_volume_usd (float): Minimum daily volume in USD risk.
    
    Returns:
        float: Required minimum daily volume in contracts.
    """
    risk_per_contract = price * multiplier * volatility
    volume_usd_risk_per_contract = fx_rate * risk_per_contract
    
    return min_daily_volume_usd / volume_usd_risk_per_contract

def evaluate_instrument_suitability(symbol, instruments_df, price, volatility, 
                                  capital, risk_target=0.2, max_cost_sr=0.10):
    """
    Evaluate if an instrument is suitable for trading based on multiple criteria.
    
    Parameters:
        symbol (str): Instrument symbol.
        instruments_df (pd.DataFrame): Instruments data.
        price (float): Current price.
        volatility (float): Annualized volatility.
        capital (float): Available capital.
        risk_target (float): Target risk fraction.
        max_cost_sr (float): Maximum acceptable cost in SR units.
    
    Returns:
        dict: Evaluation results.
    """
    try:
        specs = get_instrument_specs(symbol, instruments_df)
        multiplier = specs['multiplier']
        sr_cost = specs['sr_cost']
        
        # Check minimum capital requirement
        min_capital = calculate_minimum_capital_for_instrument(symbol, instruments_df, price, volatility, risk_target)
        capital_ok = capital >= min_capital
        
        # Check cost efficiency
        cost_ok = sr_cost <= max_cost_sr
        
        # Calculate theoretical position size
        position_size = calculate_variable_position_size(
            capital, multiplier, price, volatility, risk_target
        )
        
        # Calculate liquidity requirement
        min_volume_contracts = calculate_liquidity_threshold(price, multiplier, volatility)
        
        return {
            'symbol': symbol,
            'name': specs['name'],
            'multiplier': multiplier,
            'sr_cost': sr_cost,
            'min_capital': min_capital,
            'capital_ok': capital_ok,
            'cost_ok': cost_ok,
            'position_size': position_size,
            'min_volume_contracts': min_volume_contracts,
            'suitable': capital_ok and cost_ok
        }
        
    except Exception as e:
        return {
            'symbol': symbol,
            'error': str(e),
            'suitable': False
        }

#####   STRATEGY IMPLEMENTATION   #####

def backtest_variable_risk_strategy(csv_path, capital, risk_target=0.2, 
                                   short_span=32, long_years=10):
    """
    Backtest buy-and-hold strategy with variable risk scaling.
    
    Parameters:
        csv_path (str): Path to price data CSV.
        capital (float): Initial capital.
        risk_target (float): Target risk fraction.
        short_span (int): EWMA span for short-run volatility.
        long_years (int): Years for long-run volatility average.
    
    Returns:
        dict: Backtest results and performance metrics.
    """
    # Load data
    df = pd.read_csv(csv_path, parse_dates=['Time'])
    df.set_index('Time', inplace=True)
    df = df.dropna()
    
    # Calculate returns
    df['returns'] = df['Last'].pct_change()
    df = df.dropna()
    
    # Calculate blended volatility forecast
    df['blended_vol'] = calculate_blended_volatility(
        df['returns'], short_span=short_span, long_years=long_years
    )
    
    # Get instrument specs (assuming MES)
    instruments_df = load_instrument_data()
    mes_specs = get_instrument_specs('MES', instruments_df)
    multiplier = mes_specs['multiplier']
    
    # Calculate position sizes (using previous day's price and volatility)
    positions = []
    for i in range(len(df)):
        if i == 0:
            positions.append(0)  # No position on first day
        else:
            prev_price = df['Last'].iloc[i-1]
            prev_vol = df['blended_vol'].iloc[i-1]
            
            if np.isnan(prev_vol) or prev_vol <= 0:
                position = 0
            else:
                position = calculate_variable_position_size(
                    capital, multiplier, prev_price, prev_vol, risk_target
                )
            positions.append(position)
    
    df['position'] = positions
    df['position_discrete'] = df['position'].round()
    
    # Calculate strategy returns
    df['position_lag'] = df['position'].shift(1)
    df['notional_exposure'] = df['position_lag'] * multiplier * df['Last'].shift(1)
    df['strategy_pnl'] = df['position_lag'] * multiplier * df['returns'] * df['Last'].shift(1)
    df['strategy_returns'] = df['strategy_pnl'] / capital
    
    # Remove NaN values
    df = df.dropna()
    
    # Calculate performance metrics
    performance = calculate_comprehensive_performance(
        build_account_curve(df['strategy_returns'], capital),
        df['strategy_returns']
    )
    
    # Add strategy-specific metrics
    performance['avg_position'] = df['position'].mean()
    performance['max_position'] = df['position'].max()
    performance['min_position'] = df['position'].min()
    performance['avg_volatility'] = df['blended_vol'].mean()
    performance['vol_min'] = df['blended_vol'].min()
    performance['vol_max'] = df['blended_vol'].max()
    
    return {
        'data': df,
        'performance': performance,
        'final_position': df['position'].iloc[-1]
    }

def plot_chapter3_variable_risk_results(results, save_path='results/chapter3_variable_risk.png'):
    """
    Plot Chapter 3 variable risk strategy results including volatility and position size evolution.
    
    Parameters:
        results (dict): Results from backtest_variable_risk_strategy.
        save_path (str): Path to save the plot.
    """
    try:
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        df = results['data']
        performance = results['performance']
        
        # Build equity curve
        equity_curve = build_account_curve(df['strategy_returns'], 50000000)
        
        # Create the plot
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Equity curve
        plt.subplot(4, 1, 1)
        plt.plot(equity_curve.index, equity_curve.values/1e6, 'b-', linewidth=1.5, label='Variable Risk Strategy')
        plt.title('Chapter 3: Variable Risk Scaling Strategy', fontsize=14, fontweight='bold')
        plt.ylabel('Portfolio Value ($M)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Volatility evolution
        plt.subplot(4, 1, 2)
        plt.plot(df.index, df['blended_vol'] * 100, 'r-', linewidth=1, label='Blended Volatility Forecast')
        plt.ylabel('Volatility (%)', fontsize=12)
        plt.title('Volatility Forecast Evolution', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 3: Position size evolution
        plt.subplot(4, 1, 3)
        plt.plot(df.index, df['position'], 'g-', linewidth=1, label='Position Size (Contracts)')
        plt.plot(df.index, df['position_discrete'], 'g--', linewidth=1, alpha=0.7, label='Discrete Position')
        plt.ylabel('Contracts', fontsize=12)
        plt.title('Position Size Evolution', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 4: Drawdown
        plt.subplot(4, 1, 4)
        drawdown_stats = calculate_maximum_drawdown(equity_curve)
        drawdown_series = drawdown_stats['drawdown_series'] * 100
        
        plt.fill_between(drawdown_series.index, drawdown_series.values, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        plt.plot(drawdown_series.index, drawdown_series.values, 'r-', linewidth=1)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.title('Drawdown', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis dates for all subplots
        for ax in plt.gcf().get_axes():
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add performance metrics as text
        start_date = df.index[0].strftime('%Y-%m-%d')
        end_date = df.index[-1].strftime('%Y-%m-%d')
        
        textstr = f'''Performance Summary:
Total Return: {performance['total_return']:.1%}
Annualized Return: {performance['annualized_return']:.1%}
Volatility: {performance['annualized_volatility']:.1%}
Sharpe Ratio: {performance['sharpe_ratio']:.3f}
Max Drawdown: {performance['max_drawdown_pct']:.1f}%
Avg Position: {performance['avg_position']:.1f} contracts
Avg Volatility: {performance['avg_volatility']:.1%}
Period: {start_date} to {end_date}'''
        
        plt.figtext(0.02, 0.02, textstr, fontsize=9, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)  # Make room for performance text
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Chapter 3 variable risk results saved to: {save_path}")
        
    except Exception as e:
        print(f"Error plotting Chapter 3 results: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Test Chapter 3 variable risk scaling implementation.
    """
    print("=" * 60)
    print("CHAPTER 3: BUY AND HOLD WITH VARIABLE RISK SCALING")
    print("=" * 60)
    
    # Load instruments data
    instruments_df = load_instrument_data()
    
    # Test risk-adjusted cost calculations
    print("\n----- Risk-Adjusted Cost Analysis -----")
    
    # MES example from book
    price = 4500
    multiplier = 5
    commission = 0.62
    spread_points = 0.25
    volatility = 0.16
    
    total_cost = calculate_total_cost_per_trade(spread_points, multiplier, commission)
    cost_pct = calculate_cost_as_percentage(total_cost, price, multiplier)
    risk_adj_cost = calculate_risk_adjusted_cost_per_trade(cost_pct, volatility)
    annual_cost = calculate_annual_risk_adjusted_cost(
        price, multiplier, commission, spread_points, volatility, 4, 6
    )
    
    print(f"MES Risk-Adjusted Cost Analysis:")
    print(f"  Total Cost per Trade: ${total_cost:.4f}")
    print(f"  Cost as %: {cost_pct:.6f}")
    print(f"  Risk-Adjusted Cost per Trade: {risk_adj_cost:.6f} SR units")
    print(f"  Annual Risk-Adjusted Cost: {annual_cost:.6f} SR units")
    
    # Compare with book's value (should be ~0.0034)
    expected_book_value = 0.0034
    print(f"  Book Expected Value: {expected_book_value:.6f}")
    print(f"  Difference: {abs(annual_cost - expected_book_value):.6f}")
    
    # Test instrument evaluation
    print("\n----- Instrument Suitability Analysis -----")
    
    capital = 50000000
    test_instruments = ['MES', 'MYM', 'MNQ', 'ZN', 'VIX']
    
    for symbol in test_instruments:
        # Use different price/vol estimates for different instruments
        if symbol == 'VIX':
            test_price, test_vol = 20, 0.80  # High volatility
        elif symbol == 'ZN':
            test_price, test_vol = 110, 0.08  # Low volatility bond
        else:
            test_price, test_vol = 4000, 0.20  # Equity index
        
        evaluation = evaluate_instrument_suitability(
            symbol, instruments_df, test_price, test_vol, capital
        )
        
        if 'error' not in evaluation:
            print(f"\n{symbol} ({evaluation['name'][:40]}):")
            print(f"  Multiplier: {evaluation['multiplier']}")
            print(f"  SR Cost: {evaluation['sr_cost']:.6f}")
            print(f"  Min Capital: ${evaluation['min_capital']:,.0f}")
            print(f"  Position Size: {evaluation['position_size']:.2f}")
            print(f"  Capital OK: {evaluation['capital_ok']}")
            print(f"  Cost OK: {evaluation['cost_ok']}")
            print(f"  Suitable: {evaluation['suitable']}")
        else:
            print(f"\n{symbol}: Error - {evaluation['error']}")
    
    # Test variable risk strategy
    print("\n----- Variable Risk Strategy Backtest -----")
    
    try:
        results = backtest_variable_risk_strategy(
            'Data/mes_daily_data.csv', 
            capital=50000000, 
            risk_target=0.2
        )
        
        perf = results['performance']
        
        print(f"Performance Summary:")
        print(f"  Total Return: {perf['total_return']:.2%}")
        print(f"  Annualized Return: {perf['annualized_return']:.2%}")
        print(f"  Volatility: {perf['annualized_volatility']:.2%}")
        print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {perf['max_drawdown_pct']:.1f}%")
        print(f"  Skewness: {perf['skewness']:.3f}")
        
        print(f"\nPosition Statistics:")
        print(f"  Average Position: {perf['avg_position']:.2f} contracts")
        print(f"  Max Position: {perf['max_position']:.2f} contracts")
        print(f"  Min Position: {perf['min_position']:.2f} contracts")
        
        print(f"\nVolatility Statistics:")
        print(f"  Average Volatility: {perf['avg_volatility']:.2%}")
        print(f"  Min Volatility: {perf['vol_min']:.2%}")
        print(f"  Max Volatility: {perf['vol_max']:.2%}")
        
        # Compare with fixed risk strategy from Chapter 2
        print(f"\n----- Comparison with Fixed Risk (Chapter 2) -----")
        
        # Calculate fixed risk strategy performance for comparison
        fixed_position = calculate_variable_position_size(50000000, 5, 4500, 0.16, 0.2)
        print(f"Fixed Risk Position: {fixed_position:.2f} contracts")
        print(f"Variable Risk Final Position: {results['final_position']:.2f} contracts")
        
        # Plot the results
        plot_chapter3_variable_risk_results(results)
        
    except Exception as e:
        print(f"Error in backtest: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 