from chapter6 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#####   FORECAST CALCULATION FUNCTIONS   #####

def calculate_price_volatility(price, percentage_volatility):
    """
    Calculate price-based volatility for forecast calculation.
    
    From Chapter 7:
        σp = Price × σ% ÷ 16
    
    Parameters:
        price (float or pd.Series): Current price(s).
        percentage_volatility (float or pd.Series): Volatility as percentage (e.g., 0.16 for 16%).
    
    Returns:
        float or pd.Series: Price-based volatility.
    """
    return price * percentage_volatility / 16

def calculate_raw_forecast(fast_ma, slow_ma, price, percentage_volatility):
    """
    Calculate raw forecast using EWMAC methodology from Chapter 7.
    
    Formula:
        Raw forecast = (Fast EWMA - Slow EWMA) ÷ σp
        where σp = Price × σ% ÷ 16
    
    Parameters:
        fast_ma (pd.Series): Fast moving average series.
        slow_ma (pd.Series): Slow moving average series.
        price (pd.Series): Price series.
        percentage_volatility (pd.Series): Volatility as percentage.
    
    Returns:
        pd.Series: Raw forecast series.
    """
    ewmac_signal = fast_ma - slow_ma
    price_vol = calculate_price_volatility(price, percentage_volatility)
    
    # Avoid division by zero
    price_vol = price_vol.replace(0, np.nan)
    raw_forecast = ewmac_signal / price_vol
    
    # Fill NaN values with 0 (neutral forecast)
    raw_forecast.fillna(0, inplace=True)
    
    return raw_forecast

def scale_forecast(raw_forecast, scale_factor=1.9):
    """
    Scale raw forecast by a constant factor.
    
    From Chapter 7:
        Scaled forecast = Raw forecast × 1.9
    
    Parameters:
        raw_forecast (pd.Series): Raw forecast series.
        scale_factor (float): Scaling factor (1.9 for EWMAC(64,256) in book).
    
    Returns:
        pd.Series: Scaled forecast series.
    """
    return raw_forecast * scale_factor

def cap_forecast(scaled_forecast, max_forecast=20, min_forecast=-20):
    """
    Cap forecast values to prevent extreme positions.
    
    From Chapter 7:
        Capped forecast = Max(Min(Scaled forecast, +20), -20)
    
    Parameters:
        scaled_forecast (pd.Series): Scaled forecast series.
        max_forecast (float): Maximum forecast value (default 20).
        min_forecast (float): Minimum forecast value (default -20).
    
    Returns:
        pd.Series: Capped forecast series.
    """
    return scaled_forecast.clip(lower=min_forecast, upper=max_forecast)

def calculate_forecast_signals(prices, fast_window=64, slow_window=256, use_ewma=True, 
                             scale_factor=1.9, max_forecast=20, min_forecast=-20):
    """
    Calculate complete forecast signals for trend following with strength.
    
    This implements the full Chapter 7 forecast methodology:
    1. Calculate EWMAC signal (Fast MA - Slow MA)
    2. Calculate raw forecast = EWMAC ÷ σp
    3. Scale forecast by factor (1.9)
    4. Cap forecast between -20 and +20
    
    Parameters:
        prices (pd.Series): Price series.
        fast_window (int): Fast moving average window.
        slow_window (int): Slow moving average window.
        use_ewma (bool): Use EWMA instead of SMA.
        scale_factor (float): Forecast scaling factor.
        max_forecast (float): Maximum forecast value.
        min_forecast (float): Minimum forecast value.
    
    Returns:
        dict: Dictionary containing all forecast components.
    """
    # Calculate moving averages
    if use_ewma:
        fast_ma = calculate_ewma_trend(prices, fast_window)
        slow_ma = calculate_ewma_trend(prices, slow_window)
    else:
        fast_ma = calculate_simple_moving_average(prices, fast_window)
        slow_ma = calculate_simple_moving_average(prices, slow_window)
    
    # Calculate returns and volatility
    returns = prices.pct_change()
    percentage_volatility = calculate_blended_volatility(returns)
    
    # Calculate raw forecast
    raw_forecast = calculate_raw_forecast(fast_ma, slow_ma, prices, percentage_volatility)
    
    # Scale and cap forecast
    scaled_forecast = scale_forecast(raw_forecast, scale_factor)
    capped_forecast = cap_forecast(scaled_forecast, max_forecast, min_forecast)
    
    # Position direction (for analysis)
    position_direction = pd.Series(index=capped_forecast.index)
    position_direction[capped_forecast > 0] = 1.0   # Long
    position_direction[capped_forecast < 0] = -1.0  # Short
    position_direction[capped_forecast == 0] = 0.0  # Flat (rare)
    position_direction.fillna(0.0, inplace=True)
    
    return {
        'fast_ma': fast_ma,
        'slow_ma': slow_ma,
        'percentage_volatility': percentage_volatility,
        'raw_forecast': raw_forecast,
        'scaled_forecast': scaled_forecast,
        'capped_forecast': capped_forecast,
        'position_direction': position_direction,
        'forecast_strength': capped_forecast.abs(),  # Absolute forecast for analysis
        'uptrend': capped_forecast > 0,
        'downtrend': capped_forecast < 0
    }

#####   FORECAST-BASED POSITION SIZING   #####

def calculate_forecast_position_size(capital, weight, idm, multiplier, price, fx_rate, 
                                   volatility_pct, capped_forecast, risk_target=0.2):
    """
    Calculate position size using forecast scaling methodology from Chapter 7.
    
    Formula:
        N = Capped_forecast × Capital × IDM × Weight × τ ÷ (10 × Multiplier × Price × FX × σ%)
    
    Parameters:
        capital (float): Total capital.
        weight (float): Target weight for instrument.
        idm (float): Instrument diversification multiplier.
        multiplier (float): Contract multiplier.
        price (float): Current price.
        fx_rate (float): FX rate.
        volatility_pct (float): Volatility as percentage.
        capped_forecast (float): Capped forecast value (-20 to +20).
        risk_target (float): Risk target.
    
    Returns:
        float: Number of contracts (signed based on forecast).
    """
    if (np.isnan(capped_forecast) or capped_forecast == 0 or 
        volatility_pct <= 0 or price <= 0 or multiplier <= 0):
        return 0
    
    # Chapter 7 position sizing formula
    numerator = capped_forecast * capital * idm * weight * risk_target
    denominator = 10 * multiplier * price * fx_rate * volatility_pct
    
    position = numerator / denominator
    
    # Apply reasonable position limits
    max_reasonable_position = 1000
    if abs(position) > max_reasonable_position:
        position = np.sign(position) * max_reasonable_position
    
    return position

def apply_forecast_scaling_to_weights(portfolio_weights, forecast_signals_dict):
    """
    Apply forecast scaling to determine position magnitudes and directions.
    
    Parameters:
        portfolio_weights (dict): Original portfolio weights by instrument.
        forecast_signals_dict (dict): Forecast signals for each instrument.
    
    Returns:
        dict: Dictionary of {date: {symbol: forecast_value}} where forecast_value is -20 to +20.
    """
    daily_forecasts = {}
    
    # Get all dates from forecast signals
    all_dates = set()
    for signals in forecast_signals_dict.values():
        if 'capped_forecast' in signals:
            all_dates.update(signals['capped_forecast'].index)
    
    all_dates = sorted(list(all_dates))
    
    # For each date, get forecast value for each instrument
    for date in all_dates:
        date_forecasts = {}
        for symbol in portfolio_weights.keys():
            if symbol in forecast_signals_dict:
                signals = forecast_signals_dict[symbol]
                if date in signals['capped_forecast'].index:
                    forecast_value = signals['capped_forecast'].loc[date]
                    date_forecasts[symbol] = forecast_value if not pd.isna(forecast_value) else 0.0
                else:
                    # Try to forward fill from previous date
                    prev_forecast = signals['capped_forecast'].asof(date)
                    date_forecasts[symbol] = prev_forecast if not pd.isna(prev_forecast) else 0.0
            else:
                # No forecast signals for this instrument
                date_forecasts[symbol] = 0.0
        
        daily_forecasts[date] = date_forecasts
    
    return daily_forecasts

#####   FORECAST-BASED PORTFOLIO BACKTESTING   #####

def backtest_forecast_trend_following_portfolio(portfolio_weights, data, instruments_df, capital=1000000, 
                                              risk_target=0.2, start_date='2000-01-01', end_date='2025-01-01',
                                              fast_window=64, slow_window=256, use_ewma=True,
                                              scale_factor=1.9, max_forecast=20, min_forecast=-20):
    """
    Backtest trend following strategy with forecast scaling (Chapter 7).
    
    This implements Strategy 7: positions scaled according to forecast strength,
    where forecasts range from -20 to +20 based on trend strength.
    
    Parameters:
        portfolio_weights (dict): Portfolio weights by instrument.
        data (dict): Individual instrument data.
        instruments_df (pd.DataFrame): Instruments specifications.
        capital (float): Initial capital.
        risk_target (float): Target risk fraction.
        start_date (str): Start date for backtest.
        end_date (str): End date for backtest.
        fast_window (int): Fast moving average window.
        slow_window (int): Slow moving average window.
        use_ewma (bool): Use EWMA instead of SMA.
        scale_factor (float): Forecast scaling factor.
        max_forecast (float): Maximum forecast value.
        min_forecast (float): Minimum forecast value.
    
    Returns:
        dict: Backtest results including forecast-specific metrics.
    """
    if not portfolio_weights:
        return {'error': 'No portfolio weights provided'}

    # Filter and normalize portfolio weights
    active_portfolio_weights = {
        s: w for s, w in portfolio_weights.items() if s in data
    }
    if not active_portfolio_weights:
        return {'error': 'No instruments available in data'}
    
    total_active_weight = sum(active_portfolio_weights.values())
    normalized_weights = {s: w / total_active_weight for s, w in active_portfolio_weights.items()}

    # Calculate forecast signals for each instrument
    print("Calculating forecast signals for each instrument...")
    forecast_signals_dict = {}
    
    for symbol in normalized_weights.keys():
        try:
            symbol_data = data[symbol]
            prices = symbol_data['Last']
            
            forecast_signals = calculate_forecast_signals(
                prices, fast_window, slow_window, use_ewma, 
                scale_factor, max_forecast, min_forecast
            )
            forecast_signals_dict[symbol] = forecast_signals
            
        except Exception as e:
            print(f"Error calculating forecast signals for {symbol}: {e}")
            continue

    if not forecast_signals_dict:
        return {'error': 'No forecast signals calculated'}

    # Calculate correlation matrix and IDM
    returns_for_idm_list = []
    for s_key in normalized_weights.keys():
        if 'returns' in data[s_key]:
            s_returns = data[s_key]['returns'].copy()
            s_returns.name = s_key
            returns_for_idm_list.append(s_returns)

    correlation_matrix = pd.DataFrame()
    base_idm = 1.0

    if returns_for_idm_list and len(returns_for_idm_list) > 1:
        try:
            common_returns_matrix = pd.concat(returns_for_idm_list, axis=1, join='inner').dropna()
            
            if not common_returns_matrix.empty and common_returns_matrix.shape[1] > 1:
                correlation_matrix = calculate_correlation_matrix(common_returns_matrix)
                equal_weights_series = pd.Series({symbol: 1.0/len(normalized_weights) for symbol in normalized_weights.keys()})
                equal_weights_series = equal_weights_series.reindex(correlation_matrix.index).fillna(0)
                equal_weights_series = equal_weights_series / equal_weights_series.sum()
                base_idm = calculate_idm_from_correlations(equal_weights_series, correlation_matrix)
                print(f"Calculated IDM for forecast strategy: {base_idm:.2f}")
        except Exception as e:
            print(f"Error calculating IDM: {e}")

    # Apply forecast scaling to get daily forecast values
    print("Applying forecast scaling to determine position sizes...")
    daily_forecasts = apply_forecast_scaling_to_weights(normalized_weights, forecast_signals_dict)
    
    # Determine backtest date range
    returns_for_backtest = []
    for s_key in normalized_weights.keys():
        if 'returns' in data[s_key]:
            s_returns = data[s_key]['returns'].copy()
            s_returns.name = s_key
            returns_for_backtest.append(s_returns)

    if not returns_for_backtest:
        return {'error': 'No returns data for backtest'}
        
    returns_matrix = pd.concat(returns_for_backtest, axis=1, join='outer')
    param_start_dt = pd.to_datetime(start_date)
    param_end_dt = pd.to_datetime(end_date)
    returns_matrix = returns_matrix[(returns_matrix.index >= param_start_dt) & 
                                   (returns_matrix.index <= param_end_dt)]
    
    if returns_matrix.empty:
        return {'error': 'No data in specified date range'}

    # Pre-calculate volatilities
    volatilities = {}
    for symbol in normalized_weights.keys():
        symbol_returns = data[symbol]['returns']
        blended_vol = calculate_blended_volatility(symbol_returns)
        volatilities[symbol] = blended_vol.reindex(returns_matrix.index, method='ffill')

    # Initialize tracking variables
    portfolio_returns = []
    positions_data = {symbol: [] for symbol in normalized_weights.keys()}
    forecast_data = {symbol: [] for symbol in normalized_weights.keys()}
    long_exposure_data = []
    short_exposure_data = []
    net_exposure_data = []
    gross_exposure_data = []
    idm_data = []

    # Main backtesting loop
    for i, date_val in enumerate(returns_matrix.index):
        if i == 0:
            # Initialize first day
            for symbol in normalized_weights.keys():
                positions_data[symbol].append(0)
                forecast_data[symbol].append(0)
            portfolio_returns.append(0)
            long_exposure_data.append(0)
            short_exposure_data.append(0)
            net_exposure_data.append(0)
            gross_exposure_data.append(0)
            idm_data.append(base_idm)
            continue

        prev_date = returns_matrix.index[i-1]
        current_date = date_val

        # Get forecast values for previous date
        if prev_date in daily_forecasts:
            current_forecasts = daily_forecasts[prev_date]
        else:
            available_dates_forecasts = [d for d in daily_forecasts.keys() if d <= prev_date]
            if available_dates_forecasts:
                closest_date_forecasts = max(available_dates_forecasts)
                current_forecasts = daily_forecasts[closest_date_forecasts]
            else:
                current_forecasts = {symbol: 0.0 for symbol in normalized_weights.keys()}

        # Calculate exposure metrics
        long_exposure = 0
        short_exposure = 0
        net_exposure = 0
        gross_exposure = 0

        for symbol, base_weight in normalized_weights.items():
            forecast_value = current_forecasts.get(symbol, 0.0)
            forecast_data[symbol].append(forecast_value)
            
            # Calculate effective exposure based on forecast
            # Forecast ranges from -20 to +20, so normalize to get exposure
            effective_exposure = abs(forecast_value) / 20.0 * base_weight  # Scale by forecast strength
            gross_exposure += effective_exposure
            
            if forecast_value > 0:
                long_exposure += effective_exposure
                net_exposure += effective_exposure
            elif forecast_value < 0:
                short_exposure += effective_exposure
                net_exposure -= effective_exposure

        long_exposure_data.append(long_exposure)
        short_exposure_data.append(short_exposure)
        net_exposure_data.append(net_exposure)
        gross_exposure_data.append(gross_exposure)
        idm_data.append(base_idm)

        # Calculate positions and P&L
        daily_portfolio_return = 0

        for symbol, base_weight in normalized_weights.items():
            try:
                specs = get_instrument_specs(symbol, instruments_df)
                multiplier = specs['multiplier']
                symbol_data_df = data[symbol]

                forecast_value = current_forecasts.get(symbol, 0.0)

                # Calculate forecast-based position
                if (prev_date in symbol_data_df.index and current_date in symbol_data_df.index and
                    not pd.isna(symbol_data_df.loc[prev_date, 'Last']) and 
                    not pd.isna(symbol_data_df.loc[current_date, 'returns'])):

                    prev_price = symbol_data_df.loc[prev_date, 'Last']
                    current_return = symbol_data_df.loc[current_date, 'returns']

                    # Get volatility
                    if prev_date in volatilities[symbol].index:
                        blended_vol = volatilities[symbol].loc[prev_date]
                    else:
                        blended_vol = 0.16

                    position = 0
                    if blended_vol > 0 and prev_price > 0 and forecast_value != 0:
                        position = calculate_forecast_position_size(
                            capital, base_weight, base_idm, multiplier, prev_price, 1.0, 
                            blended_vol, forecast_value, risk_target
                        )

                    positions_data[symbol].append(position)

                    # Calculate P&L
                    if position != 0 and not pd.isna(current_return):
                        notional_exposure = position * multiplier * prev_price
                        instrument_pnl = notional_exposure * current_return
                        instrument_return_contrib = instrument_pnl / capital

                        if np.isinf(instrument_return_contrib) or np.isnan(instrument_return_contrib):
                            print(f"FORECAST OVERFLOW {current_date}: {symbol} pos={position:.2f}")
                            raise ValueError(f"Overflow for {symbol}")

                        daily_portfolio_return += instrument_return_contrib
                else:
                    positions_data[symbol].append(0)

            except Exception as e:
                positions_data[symbol].append(0)

        portfolio_returns.append(daily_portfolio_return)

    # Create results dataframe
    results_df = pd.DataFrame(index=returns_matrix.index)
    results_df['portfolio_returns'] = portfolio_returns
    results_df['long_exposure'] = long_exposure_data
    results_df['short_exposure'] = short_exposure_data
    results_df['net_exposure'] = net_exposure_data
    results_df['gross_exposure'] = gross_exposure_data
    results_df['idm'] = idm_data

    for symbol, pos_list in positions_data.items():
        if len(pos_list) == len(results_df):
            results_df[f'position_{symbol}'] = pos_list

    for symbol, forecast_list in forecast_data.items():
        if len(forecast_list) == len(results_df):
            results_df[f'forecast_{symbol}'] = forecast_list

    # Remove initial zero return day
    if len(results_df) > 1 and results_df['portfolio_returns'].iloc[0] == 0:
        results_df = results_df.iloc[1:]

    if results_df.empty:
        return {'error': 'No valid backtest data'}

    # Calculate performance metrics
    equity_curve_series = build_account_curve(results_df['portfolio_returns'], capital)
    performance = calculate_comprehensive_performance(equity_curve_series, results_df['portfolio_returns'])

    # Add forecast-specific metrics
    performance['num_instruments'] = len(normalized_weights)
    performance['avg_gross_exposure'] = results_df['gross_exposure'].mean()
    performance['max_gross_exposure'] = results_df['gross_exposure'].max()
    performance['avg_net_exposure'] = results_df['net_exposure'].mean()
    performance['avg_long_exposure'] = results_df['long_exposure'].mean()
    performance['avg_short_exposure'] = results_df['short_exposure'].mean()
    performance['avg_idm'] = results_df['idm'].mean()
    performance['fast_window'] = fast_window
    performance['slow_window'] = slow_window
    performance['scale_factor'] = scale_factor
    performance['max_forecast'] = max_forecast

    # Calculate average forecast statistics
    forecast_cols = [col for col in results_df.columns if col.startswith('forecast_')]
    if forecast_cols:
        all_forecasts = results_df[forecast_cols].values.flatten()
        all_forecasts = all_forecasts[~np.isnan(all_forecasts)]
        performance['avg_forecast_magnitude'] = np.mean(np.abs(all_forecasts))
        performance['max_forecast_used'] = np.max(np.abs(all_forecasts))
        performance['percent_time_long'] = np.mean(all_forecasts > 0)
        performance['percent_time_short'] = np.mean(all_forecasts < 0)
        performance['percent_time_flat'] = np.mean(all_forecasts == 0)

    return {
        'data': results_df,
        'performance': performance,
        'portfolio_weights': normalized_weights,
        'forecast_signals': forecast_signals_dict,
        'correlation_matrix': correlation_matrix,
        'idm': base_idm
    }

#####   STRATEGY COMPARISON FUNCTIONS   #####

def compare_four_strategies(portfolio_weights, data, instruments_df, capital=1000000, 
                          risk_target=0.2, fast_window=64, slow_window=256):
    """
    Compare four strategies: No trend, Long-only trend, Long/short trend, Forecast trend.
    
    This implements the comparison shown in Chapter 7 between:
    - Strategy 4: No trend filter (buy and hold)
    - Strategy 5: Long-only trend filter  
    - Strategy 6: Long/short trend filter
    - Strategy 7: Forecast-based trend filter
    
    Parameters:
        portfolio_weights (dict): Portfolio weights.
        data (dict): Individual instrument data.
        instruments_df (pd.DataFrame): Instruments specifications.
        capital (float): Initial capital.
        risk_target (float): Target risk fraction.
        fast_window (int): Fast MA window.
        slow_window (int): Slow MA window.
    
    Returns:
        dict: Comparison results for all four strategies.
    """
    print("=" * 80)
    print("FOUR STRATEGY COMPARISON: No Trend vs Long-Only vs Long/Short vs Forecast")
    print("=" * 80)
    
    comparison_results = {}

    # Strategy 4: No trend filter (buy and hold)
    print("\nRunning Strategy 4: Buy and Hold (No Trend Filter)...")
    scaled_risk_target = risk_target * (20.0 / 14.9)  # Scale to achieve ~20% volatility
    no_trend_result = backtest_portfolio_with_individual_data(
        portfolio_weights, data, instruments_df, capital, scaled_risk_target
    )
    comparison_results['Strategy 4 (No Trend)'] = no_trend_result

    # Strategy 5: Long-only trend filter
    print("\nRunning Strategy 5: Long-Only Trend Following...")
    long_only_result = backtest_trend_following_portfolio(
        portfolio_weights, data, instruments_df, capital, risk_target, 
        fast_window=fast_window, slow_window=slow_window
    )
    comparison_results['Strategy 5 (Long-Only)'] = long_only_result

    # Strategy 6: Long/short trend filter
    print("\nRunning Strategy 6: Long/Short Trend Following...")
    long_short_result = backtest_long_short_trend_following_portfolio(
        portfolio_weights, data, instruments_df, capital, risk_target,
        fast_window=fast_window, slow_window=slow_window
    )
    comparison_results['Strategy 6 (Long/Short)'] = long_short_result

    # Strategy 7: Forecast-based trend filter
    print("\nRunning Strategy 7: Forecast-Based Trend Following...")
    forecast_result = backtest_forecast_trend_following_portfolio(
        portfolio_weights, data, instruments_df, capital, risk_target,
        fast_window=fast_window, slow_window=slow_window
    )
    comparison_results['Strategy 7 (Forecast)'] = forecast_result

    # Display comparison results
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON RESULTS")
    print(f"{'='*80}")
    
    # Summary table
    print(f"\n{'Strategy':<25} {'Return':<10} {'Vol':<8} {'Sharpe':<8} {'MaxDD':<8} {'Exposure':<10}")
    print("-" * 80)
    
    for strategy_name, result in comparison_results.items():
        if 'error' not in result:
            perf = result['performance']
            exposure = perf.get('avg_gross_exposure', perf.get('avg_trend_exposure', 1.0))
            
            print(f"{strategy_name:<25} {perf['annualized_return']:<10.1%} "
                  f"{perf['annualized_volatility']:<8.1%} {perf['sharpe_ratio']:<8.3f} "
                  f"{perf['max_drawdown_pct']:<8.1f}% {exposure:<10.1%}")
        else:
            print(f"{strategy_name:<25} ERROR: {result['error']}")

    # Detailed metrics comparison
    print(f"\n{'-'*50}")
    print("DETAILED METRICS COMPARISON")
    print(f"{'-'*50}")
    
    for strategy_name, result in comparison_results.items():
        if 'error' not in result:
            perf = result['performance']
            print(f"\n{strategy_name}:")
            print(f"  Annual Return: {perf['annualized_return']:.2%}")
            print(f"  Volatility: {perf['annualized_volatility']:.2%}")
            print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {perf['max_drawdown_pct']:.1f}%")
            print(f"  IDM: {result.get('idm', 'N/A')}")
            
            # Strategy-specific metrics
            if 'avg_forecast_magnitude' in perf:
                print(f"  Avg Forecast Magnitude: {perf['avg_forecast_magnitude']:.1f}")
                print(f"  Time Long: {perf.get('percent_time_long', 0):.1%}")
                print(f"  Time Short: {perf.get('percent_time_short', 0):.1%}")
            elif 'avg_long_exposure' in perf:
                print(f"  Long Exposure: {perf['avg_long_exposure']:.1%}")
                print(f"  Short Exposure: {perf['avg_short_exposure']:.1%}")
                print(f"  Net Exposure: {perf['avg_net_exposure']:.1%}")
            elif 'avg_trend_exposure' in perf:
                print(f"  Trend Exposure: {perf['avg_trend_exposure']:.1%}")

    return comparison_results

def test_forecast_trend_following_strategy(capital=1000000, risk_target=0.2, 
                                         max_instruments=25, fast_window=64, slow_window=256):
    """
    Test forecast-based trend following strategy on optimized portfolio.
    """
    print("=" * 80)
    print("CHAPTER 7: FORECAST-BASED TREND FOLLOWING STRATEGY TEST")
    print("=" * 80)
    
    # Get optimized portfolio setup
    try:
        strategies, data, instruments_df, asset_classes = run_portfolio_comparison(capital, risk_target, max_instruments)
        
        if not strategies or 'Optimized Selection' not in strategies:
            print("Error: Could not create optimized portfolio")
            return
        
        optimized_weights = strategies['Optimized Selection']
        
        print(f"Testing on optimized portfolio:")
        print(f"  Instruments: {len(optimized_weights)}")
        print(f"  Symbols: {list(optimized_weights.keys())}")
        print(f"  Capital: ${capital:,.0f}")
        print(f"  Risk Target: {risk_target:.1%}")
        
        # Run four-strategy comparison
        comparison_results = compare_four_strategies(
            optimized_weights, data, instruments_df, capital, risk_target, fast_window, slow_window
        )
        
        return comparison_results
        
    except Exception as e:
        print(f"Error in forecast trend following test: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function to test Chapter 7 forecast-based trend following.
    """
    print("=" * 70)
    print("CHAPTER 7: SLOW TREND FOLLOWING WITH TREND STRENGTH (FORECASTS)")
    print("=" * 70)
    
    # Test forecast trend following strategy
    test_forecast_trend_following_strategy()

if __name__ == "__main__":
    main()
