from chapter6 import *
from chapter5 import *
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

#####   STRATEGY 7: SLOW TREND FOLLOWING WITH TREND STRENGTH   #####

def calculate_raw_forecast(prices: pd.Series, fast_span: int = 64, slow_span: int = 256) -> pd.Series:
    """
    Calculate raw forecast for EWMAC trend following.
    
    From book:
        Raw forecast = (Fast EWMA - Slow EWMA) ÷ σp
        where σp = Price × σ% ÷ 16 (daily price volatility)
    
    Parameters:
        prices (pd.Series): Price series.
        fast_span (int): Fast EWMA span (default 64).
        slow_span (int): Slow EWMA span (default 256).
    
    Returns:
        pd.Series: Raw forecast values.
    """
    # Calculate EWMA crossover (same as before)
    ewmac = calculate_ewma_trend(prices, fast_span, slow_span)
    
    # Calculate daily price volatility: σp = Price × σ% ÷ 16
    returns = prices.pct_change().dropna()
    
    # Calculate annualized volatility and convert to daily price volatility
    rolling_vol = returns.ewm(span=32).std() * np.sqrt(business_days_per_year)
    daily_price_vol = prices * rolling_vol / 16
    
    # Reindex to match EWMAC
    daily_price_vol = daily_price_vol.reindex(ewmac.index, method='ffill')
    
    # Calculate raw forecast: EWMAC ÷ σp
    raw_forecast = ewmac / daily_price_vol
    
    # Handle division by zero or very small volatility
    raw_forecast = raw_forecast.replace([np.inf, -np.inf], 0)
    raw_forecast = raw_forecast.fillna(0)
    
    return raw_forecast

def calculate_scaled_forecast(raw_forecast: pd.Series, forecast_scalar: float = 1.9) -> pd.Series:
    """
    Calculate scaled forecast from raw forecast.
    
    From book:
        Scaled forecast = Raw forecast × Forecast scalar
        where Forecast scalar = 1.9 for EWMAC(64,256)
    
    Parameters:
        raw_forecast (pd.Series): Raw forecast values.
        forecast_scalar (float): Forecast scalar (default 1.9).
    
    Returns:
        pd.Series: Scaled forecast values.
    """
    return raw_forecast * forecast_scalar

def calculate_capped_forecast(scaled_forecast: pd.Series, cap: float = 20.0) -> pd.Series:
    """
    Calculate capped forecast to limit extreme positions.
    
    From book:
        Capped forecast = Max(Min(Scaled forecast, +20), -20)
    
    Parameters:
        scaled_forecast (pd.Series): Scaled forecast values.
        cap (float): Maximum absolute forecast value (default 20.0).
    
    Returns:
        pd.Series: Capped forecast values.
    """
    return np.clip(scaled_forecast, -cap, cap)

def calculate_forecast_for_instrument(prices: pd.Series, fast_span: int = 64, slow_span: int = 256,
                                    forecast_scalar: float = 1.9, cap: float = 20.0) -> pd.Series:
    """
    Calculate complete forecast pipeline for an instrument.
    
    Parameters:
        prices (pd.Series): Price series.
        fast_span (int): Fast EWMA span.
        slow_span (int): Slow EWMA span.
        forecast_scalar (float): Forecast scalar.
        cap (float): Maximum absolute forecast value.
    
    Returns:
        pd.Series: Capped forecast values.
    """
    raw_forecast = calculate_raw_forecast(prices, fast_span, slow_span)
    scaled_forecast = calculate_scaled_forecast(raw_forecast, forecast_scalar)
    capped_forecast = calculate_capped_forecast(scaled_forecast, cap)
    
    return capped_forecast

def calculate_strategy7_position_size(symbol, capital, weight, idm, price, volatility, 
                                    multiplier, forecast, risk_target=0.2, fx_rate=1.0):
    """
    Calculate position size for Strategy 7 with forecast scaling.
    
    From book:
        N = Capped forecast × Capital × IDM × Weight × τ ÷ (10 × Multiplier × Price × FX × σ%)
    
    The key difference is the division by 10 and multiplication by forecast instead of ±1.
    
    Parameters:
        symbol (str): Instrument symbol.
        capital (float): Total portfolio capital.
        weight (float): Weight allocated to this instrument.
        idm (float): Instrument Diversification Multiplier.
        price (float): Current price.
        volatility (float): Annualized volatility forecast.
        multiplier (float): Contract multiplier.
        forecast (float): Capped forecast value.
        risk_target (float): Target risk fraction.
        fx_rate (float): FX rate for currency conversion.
    
    Returns:
        float: Number of contracts for this instrument.
    """
    if np.isnan(volatility) or volatility <= 0 or np.isnan(forecast):
        return 0
    
    # Calculate position size with forecast scaling
    numerator = forecast * capital * idm * weight * risk_target
    denominator = 10 * multiplier * price * fx_rate * volatility
    
    position_size = numerator / denominator
    
    # Protect against infinite or extremely large position sizes
    if np.isinf(position_size) or abs(position_size) > 100000:
        return 0
    
    return position_size

def backtest_forecast_trend_strategy(data_dir='Data', capital=50000000, risk_target=0.2,
                                   short_span=32, long_years=10, 
                                   trend_fast_span=64, trend_slow_span=256,
                                   forecast_scalar=1.9, forecast_cap=20.0,
                                   weight_method='handcrafted',
                                   start_date=None, end_date=None):
    """
    Backtest Strategy 7: Forecast-based trend following multi-instrument portfolio.
    
    Implementation follows book exactly: "Trade a portfolio of one or more instruments, 
    each with positions scaled for a variable risk estimate. Hold a long position when 
    they are in an uptrend, and hold a short position in a downtrend. Scale the size 
    of the position according to the strength of the trend."
    
    Parameters:
        data_dir (str): Directory containing price data files.
        capital (float): Initial capital.
        risk_target (float): Target risk fraction.
        short_span (int): EWMA span for short-run volatility.
        long_years (int): Years for long-run volatility average.
        trend_fast_span (int): Fast EWMA span for trend filter.
        trend_slow_span (int): Slow EWMA span for trend filter.
        forecast_scalar (float): Forecast scaling factor (default 1.9).
        forecast_cap (float): Maximum absolute forecast value (default 20.0).
        weight_method (str): Method for calculating instrument weights.
        start_date (str): Start date for backtest (YYYY-MM-DD).
        end_date (str): End date for backtest (YYYY-MM-DD).
    
    Returns:
        dict: Comprehensive backtest results.
    """
    print("=" * 60)
    print("STRATEGY 7: SLOW TREND FOLLOWING WITH TREND STRENGTH")
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
    print(f"  Trend Filter: EWMAC({trend_fast_span},{trend_slow_span}) with Forecasts")
    print(f"  Forecast Scalar: {forecast_scalar}")
    print(f"  Forecast Cap: ±{forecast_cap}")
    
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
    
    # Process each instrument and calculate volatility forecasts + trend forecasts
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
        
        # Calculate blended volatility forecast (same as previous strategies)
        df_filtered['blended_vol'] = calculate_blended_volatility(
            df_filtered['returns'], short_span=short_span, long_years=long_years
        )
        
        # Calculate forecast using trend strength
        df_filtered['forecast'] = calculate_forecast_for_instrument(
            df_filtered['Last'], trend_fast_span, trend_slow_span, 
            forecast_scalar, forecast_cap
        )
        
        # Calculate position sizes with forecast scaling
        positions = []
        for i in range(len(df_filtered)):
            if i == 0:
                positions.append(0)  # No position on first day
            else:
                prev_price = df_filtered['Last'].iloc[i-1]
                prev_vol = df_filtered['blended_vol'].iloc[i-1]
                prev_forecast = df_filtered['forecast'].iloc[i-1]
                
                if (np.isnan(prev_vol) or prev_vol <= 0 or np.isnan(prev_forecast)):
                    position = 0
                else:
                    position = calculate_strategy7_position_size(
                        symbol, capital, weights[symbol], idm, 
                        prev_price, prev_vol, multiplier, prev_forecast, risk_target
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
    portfolio_df['avg_forecast'] = 0.0
    portfolio_df['avg_abs_forecast'] = 0.0
    
    # Aggregate P&L across all instruments for each day
    for symbol, df in processed_data.items():
        # Initialize columns if they don't exist
        portfolio_df[f'{symbol}_position'] = 0.0
        portfolio_df[f'{symbol}_pnl'] = 0.0
        portfolio_df[f'{symbol}_forecast'] = 0.0
        
        # Add P&L only for dates where we have actual data
        actual_dates = df.index.intersection(full_date_range)
        
        for date in actual_dates:
            if date in df.index and not pd.isna(df.loc[date, 'instrument_pnl']):
                portfolio_df.loc[date, 'total_pnl'] += df.loc[date, 'instrument_pnl']
                portfolio_df.loc[date, f'{symbol}_pnl'] = df.loc[date, 'instrument_pnl']
                portfolio_df.loc[date, f'{symbol}_position'] = df.loc[date, 'position_lag']
                portfolio_df.loc[date, f'{symbol}_forecast'] = df.loc[date, 'forecast']
                
                if abs(df.loc[date, 'position_lag']) > 0.01:
                    portfolio_df.loc[date, 'num_active_instruments'] += 1
    
    # Calculate average forecast metrics
    forecast_cols = [col for col in portfolio_df.columns if col.endswith('_forecast')]
    if forecast_cols:
        portfolio_df['avg_forecast'] = portfolio_df[forecast_cols].mean(axis=1)
        portfolio_df['avg_abs_forecast'] = portfolio_df[forecast_cols].abs().mean(axis=1)
    
    # Calculate portfolio returns
    portfolio_df['strategy_returns'] = portfolio_df['total_pnl'] / capital
    
    # Remove rows with no activity (weekends, holidays)
    portfolio_df = portfolio_df[portfolio_df.index.weekday < 5]  # Business days only
    portfolio_df = portfolio_df.dropna(subset=['strategy_returns'])
    
    print(f"Final portfolio data: {len(portfolio_df)} observations")
    print(f"Average active instruments: {portfolio_df['num_active_instruments'].mean():.1f}")
    print(f"Average forecast: {portfolio_df['avg_forecast'].mean():.2f}")
    print(f"Average absolute forecast: {portfolio_df['avg_abs_forecast'].mean():.2f}")
    
    # Calculate performance metrics
    account_curve = build_account_curve(portfolio_df['strategy_returns'], capital)
    performance = calculate_comprehensive_performance(account_curve, portfolio_df['strategy_returns'])
    
    # Add strategy-specific metrics
    performance['num_instruments'] = len(processed_data)
    performance['idm'] = idm
    performance['avg_active_instruments'] = portfolio_df['num_active_instruments'].mean()
    performance['avg_forecast'] = portfolio_df['avg_forecast'].mean()
    performance['avg_abs_forecast'] = portfolio_df['avg_abs_forecast'].mean()
    performance['weight_method'] = weight_method
    performance['backtest_start'] = backtest_start
    performance['backtest_end'] = backtest_end
    performance['trend_fast_span'] = trend_fast_span
    performance['trend_slow_span'] = trend_slow_span
    performance['forecast_scalar'] = forecast_scalar
    performance['forecast_cap'] = forecast_cap
    
    # Calculate per-instrument statistics
    instrument_stats = {}
    for symbol in processed_data.keys():
        pnl_col = f'{symbol}_pnl'
        pos_col = f'{symbol}_position'
        forecast_col = f'{symbol}_forecast'
        
        if pnl_col in portfolio_df.columns:
            # Get only non-zero P&L periods for this instrument
            inst_pnl = portfolio_df[pnl_col][portfolio_df[pnl_col] != 0]
            inst_forecast = portfolio_df[forecast_col][portfolio_df[pnl_col] != 0]
            
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
                    'avg_forecast': inst_forecast.mean(),
                    'avg_abs_forecast': inst_forecast.abs().mean(),
                    'max_forecast': inst_forecast.max(),
                    'min_forecast': inst_forecast.min()
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
            'forecast_scalar': forecast_scalar,
            'forecast_cap': forecast_cap,
            'weight_method': weight_method,
            'backtest_start': backtest_start,
            'backtest_end': backtest_end
        }
    }

def analyze_forecast_results(results):
    """
    Analyze and display comprehensive forecast trend following results.
    
    Parameters:
        results (dict): Results from backtest_forecast_trend_strategy.
    """
    performance = results['performance']
    instrument_stats = results['instrument_stats']
    config = results['config']
    
    print("\n" + "=" * 60)
    print("FORECAST TREND FOLLOWING PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Overall performance
    print(f"\n--- Overall Portfolio Performance ---")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Annualized Return: {performance['annualized_return']:.2%}")
    print(f"Annualized Volatility: {performance['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {performance['max_drawdown_pct']:.1f}%")
    print(f"Skewness: {performance['skewness']:.3f}")
    
    # Forecast characteristics
    print(f"\n--- Forecast Characteristics ---")
    print(f"Average Active Instruments: {performance['avg_active_instruments']:.1f}")
    print(f"Average Forecast: {performance['avg_forecast']:.2f}")
    print(f"Average Absolute Forecast: {performance['avg_abs_forecast']:.2f}")
    print(f"Forecast Scalar: {config['forecast_scalar']}")
    print(f"Forecast Cap: ±{config['forecast_cap']}")
    print(f"Trend Filter: EWMAC({config['trend_fast_span']},{config['trend_slow_span']}) with Forecasts")
    
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
    
    print(f"{'Symbol':<8} {'Weight':<8} {'Return':<10} {'Sharpe':<8} {'AvgFcst':<8} {'MaxFcst':<8} {'TotalPnL':<12} {'Days':<6}")
    print("-" * 95)
    
    for symbol, stats in sorted_instruments[:10]:
        print(f"{symbol:<8} {stats['weight']:<8.3f} {stats['total_return']:<10.2%} "
              f"{stats['sharpe_ratio']:<8.3f} {stats['avg_forecast']:<8.2f} "
              f"{stats['max_forecast']:<8.2f} ${stats['total_pnl']:<11,.0f} {stats['active_days']:<6}")

def compare_all_forecast_strategies():
    """
    Compare Strategy 4 (no trend) vs Strategy 5 (long only) vs Strategy 6 (long/short) vs Strategy 7 (forecasts).
    """
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON: STRATEGY 4 vs 5 vs 6 vs 7")
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
        
        # Strategy 5 (with trend filter, long only)
        print("Running Strategy 5 (trend filter, long only)...")
        strategy5_results = backtest_trend_following_strategy(
            data_dir='Data',
            capital=50000000,
            risk_target=0.2,
            weight_method='handcrafted'
        )
        
        # Strategy 6 (with trend filter, long/short)
        print("Running Strategy 6 (trend filter, long/short)...")
        strategy6_results = backtest_long_short_trend_strategy(
            data_dir='Data',
            capital=50000000,
            risk_target=0.2,
            weight_method='handcrafted'
        )
        
        # Strategy 7 (with forecasts)
        print("Running Strategy 7 (trend filter with forecasts)...")
        strategy7_results = backtest_forecast_trend_strategy(
            data_dir='Data',
            capital=50000000,
            risk_target=0.2,
            weight_method='handcrafted'
        )
        
        if strategy4_results and strategy5_results and strategy6_results and strategy7_results:
            s4_perf = strategy4_results['performance']
            s5_perf = strategy5_results['performance']
            s6_perf = strategy6_results['performance']
            s7_perf = strategy7_results['performance']
            
            print(f"\n{'Strategy':<15} {'Ann. Return':<12} {'Volatility':<12} {'Sharpe':<8} {'Max DD':<8} {'Special':<15}")
            print("-" * 85)
            
            print(f"{'Strategy 4':<15} {s4_perf['annualized_return']:<12.2%} "
                  f"{s4_perf['annualized_volatility']:<12.2%} "
                  f"{s4_perf['sharpe_ratio']:<8.3f} "
                  f"{s4_perf['max_drawdown_pct']:<8.1f}% "
                  f"{'Always Long':<15}")
            
            print(f"{'Strategy 5':<15} {s5_perf['annualized_return']:<12.2%} "
                  f"{s5_perf['annualized_volatility']:<12.2%} "
                  f"{s5_perf['sharpe_ratio']:<8.3f} "
                  f"{s5_perf['max_drawdown_pct']:<8.1f}% "
                  f"{'Long/Flat':<15}")
            
            print(f"{'Strategy 6':<15} {s6_perf['annualized_return']:<12.2%} "
                  f"{s6_perf['annualized_volatility']:<12.2%} "
                  f"{s6_perf['sharpe_ratio']:<8.3f} "
                  f"{s6_perf['max_drawdown_pct']:<8.1f}% "
                  f"{'Long/Short':<15}")
            
            print(f"{'Strategy 7':<15} {s7_perf['annualized_return']:<12.2%} "
                  f"{s7_perf['annualized_volatility']:<12.2%} "
                  f"{s7_perf['sharpe_ratio']:<8.3f} "
                  f"{s7_perf['max_drawdown_pct']:<8.1f}% "
                  f"{'Forecasts':<15}")
            
            print(f"\n--- Strategy 7 vs Strategy 6 Analysis ---")
            return_diff = s7_perf['annualized_return'] - s6_perf['annualized_return']
            vol_diff = s7_perf['annualized_volatility'] - s6_perf['annualized_volatility']
            sharpe_diff = s7_perf['sharpe_ratio'] - s6_perf['sharpe_ratio']
            dd_diff = s7_perf['max_drawdown_pct'] - s6_perf['max_drawdown_pct']
            
            print(f"Return Difference: {return_diff:+.2%}")
            print(f"Volatility Difference: {vol_diff:+.2%}")
            print(f"Sharpe Difference: {sharpe_diff:+.3f}")
            print(f"Max Drawdown Difference: {dd_diff:+.1f}%")
            
            if 'avg_forecast' in s7_perf:
                print(f"\nStrategy 7 Forecast Characteristics:")
                print(f"  Average Forecast: {s7_perf['avg_forecast']:.2f}")
                print(f"  Average Absolute Forecast: {s7_perf['avg_abs_forecast']:.2f}")
            
            return {
                'strategy4': strategy4_results,
                'strategy5': strategy5_results,
                'strategy6': strategy6_results,
                'strategy7': strategy7_results
            }
        
    except Exception as e:
        print(f"Error in strategy comparison: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Test Strategy 7 implementation.
    """
    print("=" * 60)
    print("TESTING STRATEGY 7: FORECAST TREND FOLLOWING")
    print("=" * 60)
    
    try:
        # Run Strategy 7 backtest
        results = backtest_forecast_trend_strategy(
            data_dir='Data',
            capital=50000000,
            risk_target=0.2,
            weight_method='handcrafted'
        )
        
        # Analyze results
        analyze_forecast_results(results)
        
        # Compare all strategies
        comparison = compare_all_forecast_strategies()
        
        print(f"\nStrategy 7 backtest completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error in Strategy 7 backtest: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
