from chapter4 import *
from chapter3 import *
from chapter2 import *
from chapter1 import *
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

#####   STRATEGY 5: SLOW TREND FOLLOWING, LONG ONLY   #####

def calculate_ewma_trend(prices: pd.Series, fast_span: int = 64, slow_span: int = 256) -> pd.Series:
    """
    Calculate EWMA trend filter (EWMAC) using fast and slow exponentially weighted moving averages.
    
    From book:
        EWMA(N = 64, λ = 0.031) for fast trend
        EWMA(N = 256, λ = 0.0078) for slow trend
        EWMAC(64,256) = EWMA(64) - EWMA(256)
        Go long if EWMAC > 0, else remain flat
    
    Parameters:
        prices (pd.Series): Price series.
        fast_span (int): Fast EWMA span (default 64).
        slow_span (int): Slow EWMA span (default 256).
    
    Returns:
        pd.Series: EWMAC trend signal (positive = uptrend, negative/zero = downtrend).
    """
    # Calculate EWMA with specified spans
    fast_ewma = prices.ewm(span=fast_span, adjust=False).mean()
    slow_ewma = prices.ewm(span=slow_span, adjust=False).mean()
    
    # EWMAC = fast - slow
    ewmac = fast_ewma - slow_ewma
    
    return ewmac

def calculate_trend_signal(prices: pd.Series, fast_span: int = 64, slow_span: int = 256) -> pd.Series:
    """
    Calculate binary trend signal from EWMAC.
    
    Parameters:
        prices (pd.Series): Price series.
        fast_span (int): Fast EWMA span.
        slow_span (int): Slow EWMA span.
    
    Returns:
        pd.Series: Binary signal (1 = long, 0 = flat).
    """
    ewmac = calculate_ewma_trend(prices, fast_span, slow_span)
    
    # Go long if EWMAC > 0, else flat
    trend_signal = (ewmac > 0).astype(int)
    
    return trend_signal

def calculate_strategy5_position_size(symbol, capital, weight, idm, price, volatility, 
                                    multiplier, trend_signal, risk_target=0.2, fx_rate=1.0):
    """
    Calculate position size for Strategy 5 with trend filter.
    
    From book: "Every day we'd calculate our optimal position using the standard strategy 
    four position sizing equations, accounting for current levels of volatility, price 
    and exchange rates. However, if the trend was showing a downward movement, then 
    we'd set our optimal position to zero."
    
    Formula:
        N = (Capital × IDM × Weight × τ × TrendSignal) ÷ (Multiplier × Price × FX × σ%)
    
    Parameters:
        symbol (str): Instrument symbol.
        capital (float): Total portfolio capital.
        weight (float): Weight allocated to this instrument.
        idm (float): Instrument Diversification Multiplier.
        price (float): Current price.
        volatility (float): Annualized volatility forecast.
        multiplier (float): Contract multiplier.
        trend_signal (float): Trend signal (1 = long, 0 = flat).
        risk_target (float): Target risk fraction.
        fx_rate (float): FX rate for currency conversion.
    
    Returns:
        float: Number of contracts for this instrument.
    """
    if np.isnan(volatility) or volatility <= 0 or np.isnan(trend_signal):
        return 0
    
    # Calculate base position size (same as Strategy 4)
    base_position = calculate_portfolio_position_size(
        symbol, capital, weight, idm, price, volatility, 
        multiplier, risk_target, fx_rate
    )
    
    # Apply trend filter: multiply by 1 (long) or 0 (flat)
    position_with_trend = base_position * trend_signal
    
    return position_with_trend

def backtest_trend_following_strategy(data_dir='Data', capital=50000000, risk_target=0.2,
                                    short_span=32, long_years=10, 
                                    trend_fast_span=64, trend_slow_span=256,
                                    weight_method='handcrafted',
                                    start_date=None, end_date=None):
    """
    Backtest Strategy 5: Trend following multi-instrument portfolio.
    
    Implementation follows book exactly: "Buy and hold a portfolio of one or more 
    instruments when they have been in a long uptrend, each with positions scaled 
    for a variable risk estimate."
    
    Uses dynamic position sizing as stated in book: "positions are continuously 
    managed after opening to ensure their risk is correct."
    
    Parameters:
        data_dir (str): Directory containing price data files.
        capital (float): Initial capital.
        risk_target (float): Target risk fraction.
        short_span (int): EWMA span for short-run volatility.
        long_years (int): Years for long-run volatility average.
        trend_fast_span (int): Fast EWMA span for trend filter.
        trend_slow_span (int): Slow EWMA span for trend filter.
        weight_method (str): Method for calculating instrument weights.
        start_date (str): Start date for backtest (YYYY-MM-DD).
        end_date (str): End date for backtest (YYYY-MM-DD).
    
    Returns:
        dict: Comprehensive backtest results.
    """
    print("=" * 60)
    print("STRATEGY 5: SLOW TREND FOLLOWING, LONG ONLY")
    print("=" * 60)
    
    # Load all instrument data
    instrument_data = load_all_instrument_data(data_dir)
    
    if len(instrument_data) == 0:
        raise ValueError("No instrument data loaded successfully")
    
    # Load instrument specifications
    instruments_df = load_instrument_data()
    
    print(f"\nPortfolio Configuration:")
    print(f"  Instruments: {len(instrument_data)}")
    print(f"  Capital: ${capital:,.0f}")
    print(f"  Risk Target: {risk_target:.1%}")
    print(f"  Weight Method: {weight_method}")
    print(f"  Trend Filter: EWMA({trend_fast_span},{trend_slow_span})")
    
    # Calculate IDM
    idm = calculate_idm_from_count(len(instrument_data))
    print(f"  IDM: {idm:.2f}")
    
    # Calculate instrument weights
    weights = calculate_instrument_weights(instrument_data, weight_method, instruments_df)
    
    # Determine the full date range for backtest
    all_start_dates = [df.index.min() for df in instrument_data.values()]
    all_end_dates = [df.index.max() for df in instrument_data.values()]
    
    # Use the earliest available data to latest available data
    backtest_start = start_date if start_date else min(all_start_dates)
    backtest_end = end_date if end_date else max(all_end_dates)
    
    if isinstance(backtest_start, str):
        backtest_start = pd.to_datetime(backtest_start)
    if isinstance(backtest_end, str):
        backtest_end = pd.to_datetime(backtest_end)
    
    print(f"\nBacktest Period:")
    print(f"  Start: {backtest_start.date()}")
    print(f"  End: {backtest_end.date()}")
    print(f"  Duration: {(backtest_end - backtest_start).days} days")
    
    # Create full date range for backtest
    full_date_range = pd.date_range(backtest_start, backtest_end, freq='D')
    full_date_range = full_date_range[full_date_range.weekday < 5]  # Business days only
    
    # Process each instrument and calculate volatility forecasts + trend signals
    processed_data = {}
    
    for symbol, df in instrument_data.items():
        # Get instrument specs
        try:
            specs = get_instrument_specs(symbol, instruments_df)
            multiplier = specs['multiplier']
        except:
            continue
        
        # Filter data to backtest period
        df_filtered = df[(df.index >= backtest_start) & (df.index <= backtest_end)].copy()
        
        if len(df_filtered) < 300:  # Need more data for trend filter (256 + buffer)
            continue
        
        # Calculate blended volatility forecast (same as Strategy 4)
        df_filtered['blended_vol'] = calculate_blended_volatility(
            df_filtered['returns'], short_span=short_span, long_years=long_years
        )
        
        # Calculate trend signal using EWMAC
        df_filtered['trend_signal'] = calculate_trend_signal(
            df_filtered['Last'], trend_fast_span, trend_slow_span
        )
        
        # Calculate position sizes with trend filter
        positions = []
        for i in range(len(df_filtered)):
            if i == 0:
                positions.append(0)  # No position on first day
            else:
                prev_price = df_filtered['Last'].iloc[i-1]
                prev_vol = df_filtered['blended_vol'].iloc[i-1]
                prev_trend = df_filtered['trend_signal'].iloc[i-1]
                
                if (np.isnan(prev_vol) or prev_vol <= 0 or 
                    np.isnan(prev_trend) or prev_trend == 0):
                    position = 0
                else:
                    position = calculate_strategy5_position_size(
                        symbol, capital, weights[symbol], idm, 
                        prev_price, prev_vol, multiplier, prev_trend, risk_target
                    )
                positions.append(position)
        
        df_filtered['position'] = positions
        df_filtered['position_lag'] = df_filtered['position'].shift(1)
        df_filtered['multiplier'] = multiplier
        df_filtered['weight'] = weights[symbol]
        
        # Calculate P&L for this instrument
        df_filtered['instrument_pnl'] = (
            df_filtered['position_lag'] * 
            multiplier * 
            df_filtered['returns'] * 
            df_filtered['Last'].shift(1)
        )
        
        processed_data[symbol] = df_filtered
    
    print(f"\nCombining portfolio...")
    print(f"Successfully processed {len(processed_data)} instruments")
    
    # Create portfolio DataFrame with full date range
    portfolio_df = pd.DataFrame(index=full_date_range)
    portfolio_df['total_pnl'] = 0.0
    portfolio_df['num_active_instruments'] = 0
    portfolio_df['num_long_signals'] = 0
    
    # Aggregate P&L across all instruments for each day
    for symbol, df in processed_data.items():
        # Initialize columns if they don't exist
        portfolio_df[f'{symbol}_position'] = 0.0
        portfolio_df[f'{symbol}_pnl'] = 0.0
        portfolio_df[f'{symbol}_trend'] = 0.0
        
        # Add P&L only for dates where we have actual data
        actual_dates = df.index.intersection(full_date_range)
        
        for date in actual_dates:
            if date in df.index and not pd.isna(df.loc[date, 'instrument_pnl']):
                portfolio_df.loc[date, 'total_pnl'] += df.loc[date, 'instrument_pnl']
                portfolio_df.loc[date, f'{symbol}_pnl'] = df.loc[date, 'instrument_pnl']
                portfolio_df.loc[date, f'{symbol}_position'] = df.loc[date, 'position_lag']
                portfolio_df.loc[date, f'{symbol}_trend'] = df.loc[date, 'trend_signal']
                
                if abs(df.loc[date, 'position_lag']) > 0.01:
                    portfolio_df.loc[date, 'num_active_instruments'] += 1
                
                if df.loc[date, 'trend_signal'] > 0.5:
                    portfolio_df.loc[date, 'num_long_signals'] += 1
    
    # Calculate portfolio returns
    portfolio_df['strategy_returns'] = portfolio_df['total_pnl'] / capital
    
    # Remove rows with no activity (weekends, holidays)
    portfolio_df = portfolio_df[portfolio_df.index.weekday < 5]  # Business days only
    portfolio_df = portfolio_df.dropna(subset=['strategy_returns'])
    
    print(f"Final portfolio data: {len(portfolio_df)} observations")
    print(f"Average active instruments: {portfolio_df['num_active_instruments'].mean():.1f}")
    print(f"Average instruments with long signals: {portfolio_df['num_long_signals'].mean():.1f}")
    
    # Calculate performance metrics
    account_curve = build_account_curve(portfolio_df['strategy_returns'], capital)
    performance = calculate_comprehensive_performance(account_curve, portfolio_df['strategy_returns'])
    
    # Add strategy-specific metrics
    performance['num_instruments'] = len(processed_data)
    performance['idm'] = idm
    performance['avg_active_instruments'] = portfolio_df['num_active_instruments'].mean()
    performance['avg_long_signals'] = portfolio_df['num_long_signals'].mean()
    performance['weight_method'] = weight_method
    performance['backtest_start'] = backtest_start
    performance['backtest_end'] = backtest_end
    performance['trend_fast_span'] = trend_fast_span
    performance['trend_slow_span'] = trend_slow_span
    
    # Calculate per-instrument statistics
    instrument_stats = {}
    for symbol in processed_data.keys():
        pnl_col = f'{symbol}_pnl'
        pos_col = f'{symbol}_position'
        trend_col = f'{symbol}_trend'
        
        if pnl_col in portfolio_df.columns:
            # Get only non-zero P&L periods for this instrument
            inst_pnl = portfolio_df[pnl_col][portfolio_df[pnl_col] != 0]
            inst_trend = portfolio_df[trend_col][portfolio_df[pnl_col] != 0]
            
            if len(inst_pnl) > 10:  # Need minimum observations
                inst_returns = inst_pnl / capital
                inst_performance = calculate_comprehensive_performance(
                    build_account_curve(inst_returns, capital), inst_returns
                )
                
                instrument_stats[symbol] = {
                    'total_return': inst_performance['total_return'],
                    'sharpe_ratio': inst_performance['sharpe_ratio'],
                    'volatility': inst_performance['annualized_volatility'],
                    'max_drawdown': inst_performance['max_drawdown_pct'],
                    'avg_position': portfolio_df[pos_col][portfolio_df[pos_col] != 0].mean(),
                    'weight': weights[symbol],
                    'active_days': len(inst_pnl),
                    'total_pnl': inst_pnl.sum(),
                    'avg_trend_signal': inst_trend.mean(),
                    'percent_time_long': (inst_trend > 0.5).mean()
                }
    
    return {
        'portfolio_data': portfolio_df,
        'instrument_data': processed_data,
        'performance': performance,
        'instrument_stats': instrument_stats,
        'weights': weights,
        'idm': idm,
        'config': {
            'capital': capital,
            'risk_target': risk_target,
            'short_span': short_span,
            'long_years': long_years,
            'trend_fast_span': trend_fast_span,
            'trend_slow_span': trend_slow_span,
            'weight_method': weight_method,
            'backtest_start': backtest_start,
            'backtest_end': backtest_end
        }
    }

def analyze_trend_following_results(results):
    """
    Analyze and display comprehensive trend following results.
    
    Parameters:
        results (dict): Results from backtest_trend_following_strategy.
    """
    performance = results['performance']
    instrument_stats = results['instrument_stats']
    config = results['config']
    
    print("\n" + "=" * 60)
    print("TREND FOLLOWING PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Overall performance
    print(f"\n--- Overall Portfolio Performance ---")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Annualized Return: {performance['annualized_return']:.2%}")
    print(f"Annualized Volatility: {performance['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {performance['max_drawdown_pct']:.1f}%")
    print(f"Skewness: {performance['skewness']:.3f}")
    
    # Trend following characteristics
    print(f"\n--- Trend Following Characteristics ---")
    print(f"Average Active Instruments: {performance['avg_active_instruments']:.1f}")
    print(f"Average Long Signals: {performance['avg_long_signals']:.1f}")
    print(f"Percent Time in Market: {(performance['avg_long_signals'] / performance['num_instruments']):.1%}")
    print(f"Trend Filter: EWMA({config['trend_fast_span']},{config['trend_slow_span']})")
    
    # Portfolio characteristics
    print(f"\n--- Portfolio Characteristics ---")
    print(f"Number of Instruments: {performance['num_instruments']}")
    print(f"IDM: {performance['idm']:.2f}")
    print(f"Capital: ${config['capital']:,.0f}")
    print(f"Risk Target: {config['risk_target']:.1%}")
    print(f"Backtest Period: {config['backtest_start'].date()} to {config['backtest_end'].date()}")
    
    # Top performing instruments
    print(f"\n--- Top 10 Performing Instruments (by Total P&L) ---")
    sorted_instruments = sorted(
        instrument_stats.items(), 
        key=lambda x: x[1]['total_pnl'], 
        reverse=True
    )
    
    print(f"{'Symbol':<8} {'Weight':<8} {'Return':<10} {'Sharpe':<8} {'%Long':<8} {'TotalPnL':<12} {'Days':<6}")
    print("-" * 75)
    
    for symbol, stats in sorted_instruments[:10]:
        print(f"{symbol:<8} {stats['weight']:<8.3f} {stats['total_return']:<10.2%} "
              f"{stats['sharpe_ratio']:<8.3f} {stats['percent_time_long']:<8.1%} "
              f"${stats['total_pnl']:<11,.0f} {stats['active_days']:<6}")

def compare_strategies():
    """
    Compare Strategy 4 (no trend filter) vs Strategy 5 (with trend filter).
    """
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON: STRATEGY 4 vs STRATEGY 5")
    print("=" * 80)
    
    try:
        # Strategy 4 (no trend filter)
        print("Running Strategy 4 (no trend filter)...")
        strategy4_results = backtest_multi_instrument_strategy(
            data_dir='Data',
            capital=50000000,
            risk_target=0.2,
            weight_method='handcrafted'
        )
        
        # Strategy 5 (with trend filter)
        print("Running Strategy 5 (with trend filter)...")
        strategy5_results = backtest_trend_following_strategy(
            data_dir='Data',
            capital=50000000,
            risk_target=0.2,
            weight_method='handcrafted'
        )
        
        if strategy4_results and strategy5_results:
            s4_perf = strategy4_results['performance']
            s5_perf = strategy5_results['performance']
            
            print(f"\n{'Strategy':<15} {'Ann. Return':<12} {'Volatility':<12} {'Sharpe':<8} {'Max DD':<8} {'Time in Market':<15}")
            print("-" * 85)
            print(f"{'Strategy 4':<15} {s4_perf['annualized_return']:<12.2%} "
                  f"{s4_perf['annualized_volatility']:<12.2%} "
                  f"{s4_perf['sharpe_ratio']:<8.3f} "
                  f"{s4_perf['max_drawdown_pct']:<8.1f}% "
                  f"{'100.0%':<15}")
            
            time_in_market = (s5_perf['avg_long_signals'] / s5_perf['num_instruments']) * 100
            print(f"{'Strategy 5':<15} {s5_perf['annualized_return']:<12.2%} "
                  f"{s5_perf['annualized_volatility']:<12.2%} "
                  f"{s5_perf['sharpe_ratio']:<8.3f} "
                  f"{s5_perf['max_drawdown_pct']:<8.1f}% "
                  f"{time_in_market:<15.1f}%")
            
            # Calculate differences
            return_diff = s5_perf['annualized_return'] - s4_perf['annualized_return']
            vol_diff = s5_perf['annualized_volatility'] - s4_perf['annualized_volatility']
            sharpe_diff = s5_perf['sharpe_ratio'] - s4_perf['sharpe_ratio']
            dd_diff = s5_perf['max_drawdown_pct'] - s4_perf['max_drawdown_pct']
            
            print(f"{'Difference':<15} {return_diff:<+12.2%} "
                  f"{vol_diff:<+12.2%} "
                  f"{sharpe_diff:<+8.3f} "
                  f"{dd_diff:<+8.1f}% "
                  f"{time_in_market - 100:<+15.1f}%")
            
            print(f"\n--- Strategy 5 Benefits ---")
            if vol_diff < 0:
                print(f"✓ Lower volatility by {abs(vol_diff):.2%}")
            if dd_diff < 0:
                print(f"✓ Better max drawdown by {abs(dd_diff):.1f}%")
            if sharpe_diff > 0:
                print(f"✓ Better Sharpe ratio by {sharpe_diff:.3f}")
            if return_diff > 0:
                print(f"✓ Higher returns by {return_diff:.2%}")
            
            print(f"✓ Reduces market exposure to {time_in_market:.1f}% of time")
            
            return {
                'strategy4': strategy4_results,
                'strategy5': strategy5_results
            }
        
    except Exception as e:
        print(f"Error in strategy comparison: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Test Strategy 5 implementation.
    """
    print("=" * 60)
    print("TESTING STRATEGY 5: TREND FOLLOWING")
    print("=" * 60)
    
    try:
        # Run Strategy 5 backtest
        results = backtest_trend_following_strategy(
            data_dir='Data',
            capital=50000000,
            risk_target=0.2,
            weight_method='handcrafted'
        )
        
        # Analyze results
        analyze_trend_following_results(results)
        
        # Compare strategies
        comparison = compare_strategies()
        
        print(f"\nStrategy 5 backtest completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error in Strategy 5 backtest: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
