from chapter5 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#####   LONG/SHORT TREND FOLLOWING LOGIC   #####

def calculate_long_short_trend_signals(prices, fast_window=64, slow_window=256, use_ewma=True):
    """
    Calculate long/short trend signals using EWMAC methodology from the book.
    
    From Chapter 6:
        EWMAC(64,256) = EWMA(N=64) - EWMA(N=256)
        Go long if: EWMAC(64,256) >= 0 (Book implies hold/go long if zero)
        Go short if: EWMAC(64,256) < 0
        
    Key Change: Never go flat - always take either long or short position.
                 Signal must be robustly +1 or -1.
    
    Parameters:
        prices (pd.Series): Price series (back-adjusted futures prices recommended).
        fast_window (int): Fast moving average window (default 64).
        slow_window (int): Slow moving average window (default 256).
        use_ewma (bool): Use EWMA instead of SMA (recommended in book).
    
    Returns:
        dict: Dictionary containing trend signals and robust position multipliers (+1 or -1).
    """
    if use_ewma:
        fast_ma = calculate_ewma_trend(prices, fast_window)
        slow_ma = calculate_ewma_trend(prices, slow_window)
    else:
        fast_ma = calculate_simple_moving_average(prices, fast_window)
        slow_ma = calculate_simple_moving_average(prices, slow_window)
    
    # EWMAC signal
    ewmac_signal = fast_ma - slow_ma
    
    # Position multiplier: +1 for long, -1 for short. Robustly handle all cases.
    # Default to long (1.0) if ewmac_signal is NaN (e.g., at the start of the series before MAs are defined)
    position_multiplier = pd.Series(np.ones(len(ewmac_signal)), index=ewmac_signal.index)
    position_multiplier[ewmac_signal < 0] = -1.0  # Short if EWMAC is negative
    # ewmac_signal >= 0 remains 1.0 (long)
    
    # Ensure no NaNs remain from MA calculations, fill forward then backward, then default to 1.0
    if position_multiplier.isnull().any():
        position_multiplier = position_multiplier.ffill().bfill().fillna(1.0)

    # Trend strength for analysis
    # Avoid division by zero if slow_ma is zero or NaN
    trend_strength = ewmac_signal.copy()
    valid_slow_ma = slow_ma.replace(0, np.nan).ffill().bfill()
    trend_strength = ewmac_signal / valid_slow_ma
    trend_strength.fillna(0, inplace=True) # Fill any remaining NaNs with 0 strength
    
    return {
        'fast_ma': fast_ma,
        'slow_ma': slow_ma,
        'ewmac_signal': ewmac_signal,
        'position_multiplier': position_multiplier, # Should always be +1 or -1
        'trend_strength': trend_strength,
        'uptrend': position_multiplier > 0, # Consistent with multiplier
        'downtrend': position_multiplier < 0 # Consistent with multiplier
    }

def apply_long_short_trend_filter_to_weights(portfolio_weights, trend_signals_dict):
    """
    Determine the target position sign (+1 for long, -1 for short) for each instrument on each date.
    The actual weighting (magnitude) is handled by the main backtest loop using normalized_weights.
    
    Parameters:
        portfolio_weights (dict): Original portfolio weights by instrument (used to get the list of symbols).
        trend_signals_dict (dict): Long/short trend signals for each instrument 
                                 (must contain 'position_multiplier' which is always +1 or -1).
    
    Returns:
        dict: Dictionary of {date: {symbol: sign}} where sign is +1 or -1.
    """
    daily_target_signs = {}
    
    # Get all dates from trend signals
    all_dates = set()
    for signals in trend_signals_dict.values():
        if 'position_multiplier' in signals:
            all_dates.update(signals['position_multiplier'].index)
    
    all_dates = sorted(list(all_dates))
    
    # For each date, determine target sign for each instrument
    for date in all_dates:
        date_signs = {}
        for symbol in portfolio_weights.keys(): # Iterate over all instruments in the target portfolio
            if symbol in trend_signals_dict:
                signals = trend_signals_dict[symbol]
                if date in signals['position_multiplier'].index:
                    # position_multiplier is guaranteed to be +1 or -1 by calculate_long_short_trend_signals
                    target_sign = signals['position_multiplier'].loc[date]
                    date_signs[symbol] = target_sign
                else:
                    # If no signal for this specific date (should be rare with robust signal calc),
                    # try to ffill from previous day, else default to long (1.0)
                    # This instrument might be new or data is patchy.
                    prev_signal = signals['position_multiplier'].asof(date)
                    date_signs[symbol] = prev_signal if not pd.isna(prev_signal) else 1.0
            else:
                # If instrument has no trend signals at all (e.g. very new, insufficient data for MA),
                # default to long. This ensures it's included if in portfolio_weights.
                date_signs[symbol] = 1.0
        
        daily_target_signs[date] = date_signs
    
    return daily_target_signs

#####   LONG/SHORT PORTFOLIO BACKTESTING   #####

def backtest_long_short_trend_following_portfolio(portfolio_weights, data, instruments_df, capital=1000000, 
                                                 risk_target=0.2, start_date='2000-01-01', end_date='2025-01-01',
                                                 fast_window=64, slow_window=256, use_ewma=True):
    """
    Backtest long/short trend-following portfolio strategy.
    
    This implements Chapter 6's long/short trend following approach:
    1. Calculate trend signals for each instrument using EWMAC
    2. Go long when EWMAC > 0, short when EWMAC < 0
    3. Maintain full portfolio exposure with appropriate signs
    
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
    
    Returns:
        dict: Backtest results with long/short trend following metrics.
    """
    if not portfolio_weights:
        return {'error': 'No portfolio weights provided'}

    # Filter portfolio_weights to those available in data
    active_portfolio_weights = {
        s: w for s, w in portfolio_weights.items() if s in data
    }
    if not active_portfolio_weights:
        return {'error': 'None of the instruments in portfolio_weights are available in loaded data'}
    
    total_active_weight = sum(active_portfolio_weights.values())
    if total_active_weight <= 0:
        return {'error': 'Total weight of active portfolio instruments is not positive.'}
    normalized_weights = {s: w / total_active_weight for s, w in active_portfolio_weights.items()}

    # Calculate long/short trend signals for each instrument
    print(f"Calculating long/short trend signals for {len(normalized_weights)} instruments...")
    trend_signals_dict = {}
    
    for symbol in normalized_weights.keys():
        try:
            symbol_data = data[symbol]
            if 'Last' in symbol_data.columns and len(symbol_data) > slow_window:
                # Use back-adjusted prices for trend calculation (book recommendation)
                prices = symbol_data['Last'].copy()
                
                # Calculate long/short trend signals
                trend_signals = calculate_long_short_trend_signals(
                    prices, fast_window, slow_window, use_ewma
                )
                trend_signals_dict[symbol] = trend_signals
            else:
                print(f"Warning: Insufficient data for trend calculation in {symbol}")
        except Exception as e:
            print(f"Error calculating trend signals for {symbol}: {e}")
            continue
    
    if not trend_signals_dict:
        return {'error': 'No trend signals could be calculated'}

    # Calculate correlation matrix for IDM calculation (same as Chapter 5)
    returns_for_idm_list = []
    for s_key in normalized_weights.keys():
        if 'returns' in data[s_key] and isinstance(data[s_key]['returns'], pd.Series):
            s_returns = data[s_key]['returns'].copy()
            s_returns.name = s_key
            returns_for_idm_list.append(s_returns)

    correlation_matrix = pd.DataFrame()
    base_idm = 1.0  # Default IDM

    if not returns_for_idm_list:
        print("Warning: No returns series available for IDM calculation. Using IDM=1.0.")
    else:
        common_returns_matrix = pd.concat(returns_for_idm_list, axis=1, join='inner')
        common_returns_matrix.dropna(inplace=True)

        if common_returns_matrix.empty or common_returns_matrix.shape[0] < 2:
            print(f"Warning: Not enough common data for IDM calculation. Using IDM=1.0.")
            if list(normalized_weights.keys()):
                symbols_for_dummy_corr = list(normalized_weights.keys())
                correlation_matrix = pd.DataFrame(np.eye(len(symbols_for_dummy_corr)), 
                                                index=symbols_for_dummy_corr, columns=symbols_for_dummy_corr)
        elif common_returns_matrix.shape[1] == 1:
            base_idm = 1.0
            symbol = common_returns_matrix.columns[0]
            correlation_matrix = pd.DataFrame([[1.0]], index=[symbol], columns=[symbol])
            print(f"Note: Only one instrument ({symbol}) in common data for IDM. IDM set to 1.0.")
        else:
            try:
                correlation_matrix_calculated = calculate_correlation_matrix(common_returns_matrix)
                # Calculate base IDM using equal weights for all available instruments
                equal_weights_series = pd.Series({symbol: 1.0/len(normalized_weights) for symbol in normalized_weights.keys()})
                equal_weights_series = equal_weights_series.reindex(correlation_matrix_calculated.index).fillna(0)
                equal_weights_series = equal_weights_series / equal_weights_series.sum()
                base_idm_calculated = calculate_idm_from_correlations(equal_weights_series, correlation_matrix_calculated)
                base_idm = base_idm_calculated
                correlation_matrix = correlation_matrix_calculated
                print(f"Calculated base IDM for long/short strategy: {base_idm:.2f}")
            except Exception as e_idm:
                print(f"Error calculating IDM from correlations: {e_idm}. Using IDM=1.0.")
                if list(normalized_weights.keys()) and correlation_matrix.empty:
                    symbols_for_dummy_corr = list(normalized_weights.keys())
                    correlation_matrix = pd.DataFrame(np.eye(len(symbols_for_dummy_corr)), 
                                                    index=symbols_for_dummy_corr, columns=symbols_for_dummy_corr)

    # Apply long/short trend filters to get dynamic target signs (+1 or -1)
    print("Applying long/short trend filters to determine target signs...")
    daily_target_signs = apply_long_short_trend_filter_to_weights(normalized_weights, trend_signals_dict)
    
    # Determine backtest date range from available data
    returns_for_backtest = []
    for s_key in normalized_weights.keys():
        if 'returns' in data[s_key] and isinstance(data[s_key]['returns'], pd.Series):
            s_returns = data[s_key]['returns'].copy()
            s_returns.name = s_key
            returns_for_backtest.append(s_returns)

    if not returns_for_backtest:
        return {'error': 'No returns series available for backtest.'}
        
    # Use outer join for maximum date coverage
    returns_matrix = pd.concat(returns_for_backtest, axis=1, join='outer')
    
    # Filter by date range
    param_start_dt = pd.to_datetime(start_date)
    param_end_dt = pd.to_datetime(end_date)
    returns_matrix = returns_matrix[(returns_matrix.index >= param_start_dt) & 
                                   (returns_matrix.index <= param_end_dt)]
    
    if returns_matrix.empty:
        return {'error': 'No data available after date filtering.'}

    # Pre-calculate volatilities for position sizing
    volatilities = {}
    for symbol in normalized_weights.keys():
        symbol_data_df = data[symbol]
        symbol_returns = symbol_data_df['returns'] 
        blended_vol_series = calculate_blended_volatility(symbol_returns).reindex(returns_matrix.index, method='ffill')
        volatilities[symbol] = blended_vol_series
    
    # Initialize tracking variables
    portfolio_returns = []
    positions_data = {symbol: [] for symbol in normalized_weights.keys()}
    long_exposure_data = []   # Track percentage of portfolio long
    short_exposure_data = []  # Track percentage of portfolio short
    net_exposure_data = []    # Track net exposure (long - short)
    position_multipliers_data = {symbol: [] for symbol in normalized_weights.keys()}
    idm_data = []  # Track IDM values over time
    
    for i, date_val in enumerate(returns_matrix.index):
        if i == 0:
            # Initialize first day
            for symbol in normalized_weights.keys():
                positions_data[symbol].append(0)
                position_multipliers_data[symbol].append(0)
            portfolio_returns.append(0)
            long_exposure_data.append(0)
            short_exposure_data.append(0)
            net_exposure_data.append(0)
            idm_data.append(base_idm)
            continue
        
        prev_date = returns_matrix.index[i-1]
        current_date = date_val
        
        # Get target signs for previous date (for position sizing)
        if prev_date in daily_target_signs:
            current_target_signs = daily_target_signs[prev_date]
        else:
            # Find closest available date for signs
            available_dates_signs = [d for d in daily_target_signs.keys() if d <= prev_date]
            if available_dates_signs:
                closest_date_signs = max(available_dates_signs)
                current_target_signs = daily_target_signs[closest_date_signs]
            else:
                # Default to long for all if no signs found (should be rare)
                current_target_signs = {symbol: 1.0 for symbol in normalized_weights.keys()}
        
        # Use the pre-calculated base_idm for the long/short strategy
        # This IDM should reflect the diversification of a portfolio that *can* go long/short.
        # The book suggests a single, higher IDM for Strategy 6 (e.g., 2.89 for jumbo).
        # For now, we use the 'base_idm' calculated earlier which uses equal weights on the full correlation matrix.
        # A more advanced IDM would consider the *signed* correlation matrix as attempted before for 'current_idm'.
        # However, for simplicity and to align with a single IDM per strategy, we'll use base_idm.
        current_idm_to_use = base_idm 
        idm_data.append(current_idm_to_use)

        # Calculate exposure metrics based on target signs and normalized weights
        long_exposure = 0
        short_exposure = 0
        net_exposure = 0
        gross_exposure = 0
        
        for symbol, base_weight in normalized_weights.items():
            target_sign = current_target_signs.get(symbol, 1.0) # Default to long if symbol somehow missing
            gross_exposure += base_weight # Each instrument contributes its full base weight to gross exposure
            if target_sign > 0:
                long_exposure += base_weight
                net_exposure += base_weight
            elif target_sign < 0:
                short_exposure += base_weight
                net_exposure -= base_weight
        
        long_exposure_data.append(long_exposure)
        short_exposure_data.append(short_exposure) 
        net_exposure_data.append(net_exposure)
        # Gross exposure should ideally be 1.0 if all instruments are always active
        # print(f"DEBUG {prev_date}: Gross Exp: {gross_exposure}, Long: {long_exposure}, Short: {short_exposure}, Net: {net_exposure}")

        # Calculate positions and P&L
        daily_portfolio_return = 0
        
        for symbol, base_weight in normalized_weights.items(): # Iterate through ALL instruments in portfolio
            try:
                specs = get_instrument_specs(symbol, instruments_df)
                multiplier = specs['multiplier']
                symbol_data_df = data[symbol]
                
                # Get target sign for this instrument
                target_sign = current_target_signs.get(symbol, 1.0) # Default to long
                position_multipliers_data[symbol].append(target_sign)
                
                # Calculate position: use base_weight and apply target_sign
                if (prev_date in symbol_data_df.index and symbol_data_df.loc[prev_date, 'Last'] is not np.nan and 
                    current_date in symbol_data_df.index and symbol_data_df.loc[current_date, 'returns'] is not np.nan):
                    # target_sign is always +1 or -1, so we always attempt to trade if data exists
                    
                    prev_price = symbol_data_df.loc[prev_date, 'Last']
                    current_return = symbol_data_df.loc[current_date, 'returns']
                    
                    if prev_date in volatilities[symbol].index and not pd.isna(volatilities[symbol].loc[prev_date]):
                        blended_vol = volatilities[symbol].loc[prev_date]
                    else:
                        blended_vol = 0.16  # fallback
                    
                    position = 0
                    if blended_vol > 0 and not pd.isna(prev_price) and prev_price > 0 and base_weight > 0:
                        # Position size based on full base_weight
                        abs_position = calculate_position_size_with_idm(
                            capital, base_weight, current_idm_to_use, multiplier, prev_price, 1.0, blended_vol, risk_target
                        )
                        # Apply trend direction sign
                        position = abs_position * target_sign
                    
                    positions_data[symbol].append(position)
                    
                    if not pd.isna(current_return) and position != 0:
                        notional_exposure = position * multiplier * prev_price
                        instrument_pnl = notional_exposure * current_return
                        instrument_return_contrib = instrument_pnl / capital
                        
                        if np.isinf(instrument_return_contrib) or np.isnan(instrument_return_contrib):
                            print(f"LS OVERFLOW {current_date}: {symbol} pos={position:.2f} ret={current_return:.4f} pnl={instrument_pnl:.2f}")
                            raise ValueError(f"Overflow for {symbol}")
                        
                        daily_portfolio_return += instrument_return_contrib
                else:
                    positions_data[symbol].append(0) # No trade if no price/return data
                    
            except Exception as e:
                # print(f"Error processing {symbol} on {current_date} in L/S: {e}")
                positions_data[symbol].append(0)
                if symbol not in position_multipliers_data or len(position_multipliers_data[symbol]) < i:
                     position_multipliers_data[symbol].append(0) # ensure list length matches
        
        portfolio_returns.append(daily_portfolio_return)
    
    # Create results dataframe
    results_df = pd.DataFrame(index=returns_matrix.index)
    results_df['portfolio_returns'] = portfolio_returns
    results_df['long_exposure'] = long_exposure_data
    results_df['short_exposure'] = short_exposure_data
    results_df['net_exposure'] = net_exposure_data
    results_df['gross_exposure'] = np.array(long_exposure_data) + np.array(short_exposure_data)
    results_df['idm'] = idm_data
    
    for symbol, pos_list in positions_data.items():
        if len(pos_list) == len(results_df):
            results_df[f'position_{symbol}'] = pos_list
            
    for symbol, mult_list in position_multipliers_data.items():
        if len(mult_list) == len(results_df):
            results_df[f'pos_mult_{symbol}'] = mult_list
    
    # Remove initial zero return day if needed
    if len(results_df) > 1 and results_df['portfolio_returns'].iloc[0] == 0:
        results_df = results_df.iloc[1:]
    
    if results_df.empty:
        return {'error': 'No valid backtest data after processing'}
    
    # Calculate performance metrics
    equity_curve_series = build_account_curve(results_df['portfolio_returns'], capital)
    
    performance = calculate_comprehensive_performance(
        equity_curve_series,
        results_df['portfolio_returns']
    )
    
    # Add long/short specific metrics
    performance['num_instruments'] = len(normalized_weights)
    performance['avg_long_exposure'] = results_df['long_exposure'].mean()
    performance['avg_short_exposure'] = results_df['short_exposure'].mean()
    performance['avg_net_exposure'] = results_df['net_exposure'].mean()
    performance['avg_gross_exposure'] = results_df['gross_exposure'].mean()
    performance['avg_idm'] = results_df['idm'].mean()
    performance['max_idm'] = results_df['idm'].max()
    performance['min_idm'] = results_df['idm'].min()
    performance['fast_window'] = fast_window
    performance['slow_window'] = slow_window
    performance['use_ewma'] = use_ewma
    
    # Calculate position statistics
    position_stats = {}
    for symbol in normalized_weights.keys():
        if f'pos_mult_{symbol}' in results_df.columns:
            multipliers = results_df[f'pos_mult_{symbol}']
            position_stats[symbol] = {
                'long_pct': (multipliers == 1.0).mean(),
                'short_pct': (multipliers == -1.0).mean(),
                'flat_pct': (multipliers == 0.0).mean(),
                'signal_changes': multipliers.diff().abs().sum() / 2  # Divide by 2 since changes are ±2
            }
    performance['position_statistics'] = position_stats
    
    return {
        'data': results_df,
        'performance': performance,
        'portfolio_weights': normalized_weights,
        'trend_signals': trend_signals_dict,
        'strategy_type': 'long_short_trend_following'
    }

#####   THREE-WAY STRATEGY COMPARISON   #####

def compare_three_strategies(portfolio_weights, data, instruments_df, capital=1000000, 
                           risk_target=0.2, fast_window=64, slow_window=256):
    """
    Compare all three strategies: No trend (Chapter 4), Long-only trend (Chapter 5), 
    and Long/short trend (Chapter 6).
    
    Parameters:
        portfolio_weights (dict): Portfolio weights.
        data (dict): Individual instrument data.
        instruments_df (pd.DataFrame): Instruments specifications.
        capital (float): Initial capital.
        risk_target (float): Target risk fraction.
        fast_window (int): Fast MA window for trend filter.
        slow_window (int): Slow MA window for trend filter.
    
    Returns:
        dict: Comprehensive comparison results.
    """
    print("=" * 90)
    print("THREE-STRATEGY COMPARISON: NO TREND vs LONG-ONLY TREND vs LONG/SHORT TREND")
    print("=" * 90)
    
    # Strategy 4: No trend filter (buy and hold)
    print("\nRunning Strategy 4: Buy and Hold (No Trend Filter)...")
    # Scale up risk target for no-trend strategy to achieve ~20% volatility
    scaled_risk_target = risk_target * (20.0 / 14.9)  # Scale from observed 14.9% to target 20%
    print(f"Using scaled risk target of {scaled_risk_target:.1%} for no-trend strategy to achieve ~20% volatility")
    
    no_trend_result = backtest_portfolio_with_individual_data(
        portfolio_weights, data, instruments_df, capital, scaled_risk_target
    )
    
    # Strategy 5: Long-only trend following
    print("\nRunning Strategy 5: Long-Only Trend Following...")
    long_only_result = backtest_trend_following_portfolio(
        portfolio_weights, data, instruments_df, capital, risk_target,
        fast_window=fast_window, slow_window=slow_window
    )
    
    # Strategy 6: Long/short trend following
    print("\nRunning Strategy 6: Long/Short Trend Following...")
    long_short_result = backtest_long_short_trend_following_portfolio(
        portfolio_weights, data, instruments_df, capital, risk_target,
        fast_window=fast_window, slow_window=slow_window
    )
    
    if 'error' in no_trend_result or 'error' in long_only_result or 'error' in long_short_result:
        return {
            'no_trend_error': no_trend_result.get('error'),
            'long_only_error': long_only_result.get('error'),
            'long_short_error': long_short_result.get('error')
        }
    
    # Performance comparison
    no_trend_perf = no_trend_result['performance']
    long_only_perf = long_only_result['performance']
    long_short_perf = long_short_result['performance']
    
    print(f"\n{'='*90}")
    print("COMPREHENSIVE PERFORMANCE COMPARISON")
    print(f"{'='*90}")
    
    print(f"\n{'Metric':<25} {'No Trend':<15} {'Long-Only':<15} {'Long/Short':<15}")
    print("-" * 80)
    
    # Key performance metrics
    metrics = [
        ('Annual Return', 'annualized_return', '%'),
        ('Volatility', 'annualized_volatility', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', ''),
        ('Max Drawdown', 'max_drawdown_pct', '%'),
        ('Calmar Ratio', 'calmar_ratio', ''),
        ('Skewness', 'skewness', '')
    ]
    
    for metric_name, metric_key, suffix in metrics:
        no_trend_val = no_trend_perf[metric_key]
        long_only_val = long_only_perf[metric_key]
        long_short_val = long_short_perf[metric_key]
        
        if suffix == '%':
            no_trend_str = f"{no_trend_val:.1%}"
            long_only_str = f"{long_only_val:.1%}"
            long_short_str = f"{long_short_val:.1%}"
        elif metric_key == 'max_drawdown_pct':
            no_trend_str = f"{no_trend_val:.1f}%"
            long_only_str = f"{long_only_val:.1f}%"
            long_short_str = f"{long_short_val:.1f}%"
        else:
            no_trend_str = f"{no_trend_val:.3f}"
            long_only_str = f"{long_only_val:.3f}"
            long_short_str = f"{long_short_val:.3f}"
        
        print(f"{metric_name:<25} {no_trend_str:<15} {long_only_str:<15} {long_short_str:<15}")
    
    # Long/Short specific metrics
    print(f"\n{'='*60}")
    print("LONG/SHORT STRATEGY SPECIFIC METRICS")
    print(f"{'='*60}")
    print(f"Average Long Exposure: {long_short_perf.get('avg_long_exposure', 0):.1%}")
    print(f"Average Short Exposure: {long_short_perf.get('avg_short_exposure', 0):.1%}")
    print(f"Average Net Exposure: {long_short_perf.get('avg_net_exposure', 0):.1%}")
    print(f"Average Gross Exposure: {long_short_perf.get('avg_gross_exposure', 0):.1%}")
    print(f"Average IDM: {long_short_perf.get('avg_idm', 1.0):.2f}")
    
    # Position analysis for long/short strategy
    if 'position_statistics' in long_short_perf:
        print(f"\n{'='*60}")
        print("INDIVIDUAL INSTRUMENT LONG/SHORT STATISTICS")
        print(f"{'='*60}")
        print(f"{'Instrument':<10} {'Long %':<8} {'Short %':<8} {'Flat %':<8} {'Changes':<8}")
        print("-" * 50)
        
        position_stats = long_short_perf['position_statistics']
        for symbol, stats in position_stats.items():
            print(f"{symbol:<10} {stats['long_pct']:<8.1%} {stats['short_pct']:<8.1%} "
                  f"{stats['flat_pct']:<8.1%} {stats['signal_changes']:<8.0f}")
    
    return {
        'no_trend': no_trend_result,
        'long_only_trend': long_only_result,
        'long_short_trend': long_short_result,
        'performance_comparison': {
            'strategies': ['No Trend', 'Long-Only', 'Long/Short'],
            'returns': [no_trend_perf['annualized_return'], long_only_perf['annualized_return'], long_short_perf['annualized_return']],
            'sharpes': [no_trend_perf['sharpe_ratio'], long_only_perf['sharpe_ratio'], long_short_perf['sharpe_ratio']],
            'drawdowns': [no_trend_perf['max_drawdown_pct'], long_only_perf['max_drawdown_pct'], long_short_perf['max_drawdown_pct']],
            'volatilities': [no_trend_perf['annualized_volatility'], long_only_perf['annualized_volatility'], long_short_perf['annualized_volatility']]
        }
    }

def plot_three_strategy_comparison(comparison_results, capital=1000000):
    """
    Plot comprehensive comparison of all three strategies.
    
    Parameters:
        comparison_results (dict): Results from compare_three_strategies.
        capital (float): Initial capital for plotting.
    """
    if not all(key in comparison_results for key in ['no_trend', 'long_only_trend', 'long_short_trend']):
        print("Error: Missing strategy data for plotting")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Colors for strategies
    colors = ['blue', 'green', 'red']
    labels = ['No Trend (Strategy 4)', 'Long-Only Trend (Strategy 5)', 'Long/Short Trend (Strategy 6)']
    
    # Equity curves
    strategies = ['no_trend', 'long_only_trend', 'long_short_trend']
    for i, (strategy_key, label, color) in enumerate(zip(strategies, labels, colors)):
        equity_curve = build_account_curve(comparison_results[strategy_key]['data']['portfolio_returns'], capital)
        ax1.plot(equity_curve.index, equity_curve.values, label=label, linewidth=2, color=color)
    
    ax1.set_title('Equity Curve Comparison (All Strategies)')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Drawdown comparison
    for i, (strategy_key, label, color) in enumerate(zip(strategies, labels, colors)):
        equity_curve = build_account_curve(comparison_results[strategy_key]['data']['portfolio_returns'], capital)
        drawdown = calculate_maximum_drawdown(equity_curve)['drawdown_series'] * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, label=label, color=color)
    
    ax2.set_title('Drawdown Comparison')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Long/Short exposure over time (Strategy 6 only)
    long_short_data = comparison_results['long_short_trend']['data']
    ax3.plot(long_short_data.index, long_short_data['long_exposure'] * 100, 
             label='Long Exposure', linewidth=1, color='green')
    ax3.plot(long_short_data.index, long_short_data['short_exposure'] * 100, 
             label='Short Exposure', linewidth=1, color='red')
    ax3.plot(long_short_data.index, long_short_data['net_exposure'] * 100, 
             label='Net Exposure', linewidth=1, color='blue')
    ax3.set_title('Long/Short Strategy Exposure Over Time')
    ax3.set_ylabel('Exposure (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Performance comparison bar chart
    perf_comp = comparison_results['performance_comparison']
    x_pos = np.arange(len(perf_comp['strategies']))
    
    ax4.bar(x_pos - 0.25, [r * 100 for r in perf_comp['returns']], 0.25, 
            label='Annual Return (%)', alpha=0.8, color='lightblue')
    ax4.bar(x_pos, [s * 10 for s in perf_comp['sharpes']], 0.25, 
            label='Sharpe Ratio (×10)', alpha=0.8, color='lightgreen')
    ax4.bar(x_pos + 0.25, [abs(d) for d in perf_comp['drawdowns']], 0.25, 
            label='Max Drawdown (%)', alpha=0.8, color='lightcoral')
    
    ax4.set_title('Performance Metrics Comparison')
    ax4.set_ylabel('Value')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(perf_comp['strategies'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

#####   MAIN CHAPTER 6 IMPLEMENTATION   #####

def test_long_short_trend_following_strategy(capital=1000000, risk_target=0.2, 
                                           max_instruments=25, fast_window=64, slow_window=256):
    """
    Test long/short trend following strategy on optimized portfolio.
    
    This implements Chapter 6's approach of going both long and short
    based on trend signals.
    
    Parameters:
        capital (float): Initial capital.
        risk_target (float): Target risk fraction.
        max_instruments (int): Maximum instruments for portfolio.
        fast_window (int): Fast moving average window.
        slow_window (int): Slow moving average window.
    
    Returns:
        dict: Comprehensive test results comparing all three strategies.
    """
    print("=" * 80)
    print("CHAPTER 6: LONG/SHORT TREND FOLLOWING STRATEGY")
    print("=" * 80)
    
    # Load data and create portfolio using optimized selection (same as Chapter 5)
    print("\nLoading instruments and data...")
    instruments_df = load_instrument_data()
    available_instruments = get_available_instruments(instruments_df)
    print(f"Found {len(available_instruments)} instruments with data files")
    
    # Select suitable instruments
    suitable_instruments = select_instruments_by_criteria(
        instruments_df, available_instruments, capital, max_cost_sr=0.01
    )
    print(f"Selected {len(suitable_instruments)} suitable instruments")
    
    if len(suitable_instruments) == 0:
        print("No suitable instruments found!")
        return {}
    
    # Load data
    print("Loading individual instrument data...")
    data = load_instrument_data_files(suitable_instruments)
    print(f"Successfully loaded data for {len(data)} instruments")
    
    if len(data) == 0:
        print("No data loaded!")
        return {}
    
    # Create optimized selection portfolio
    optimized_weights = optimize_instrument_selection(
        instruments_df, list(data.keys()), target_instruments=min(max_instruments, len(data))
    )
    
    if not optimized_weights:
        print("Could not create optimized portfolio!")
        return {}
    
    print(f"\nOptimized Portfolio created with {len(optimized_weights)} instruments")
    print(f"Instruments: {list(optimized_weights.keys())}")
    
    # Compare all three strategies
    comparison_results = compare_three_strategies(
        optimized_weights, data, instruments_df, capital, risk_target, fast_window, slow_window
    )
    
    # Plot results
    if all(key in comparison_results for key in ['no_trend', 'long_only_trend', 'long_short_trend']):
        plot_three_strategy_comparison(comparison_results, capital)
        
        # Summary statistics
        print(f"\n{'='*90}")
        print("CHAPTER 6 SUMMARY INSIGHTS")
        print(f"{'='*90}")
        
        perf_comp = comparison_results['performance_comparison']
        
        print(f"\n🔹 **Strategy Performance Summary:**")
        for i, strategy in enumerate(perf_comp['strategies']):
            ret = perf_comp['returns'][i] * 100
            sharpe = perf_comp['sharpes'][i]
            dd = perf_comp['drawdowns'][i]
            vol = perf_comp['volatilities'][i] * 100
            print(f"   {strategy}: {ret:.1f}% return, {sharpe:.3f} Sharpe, {dd:.1f}% maxDD, {vol:.1f}% vol")
        
        # Key insights from the book
        print(f"\n🔹 **Key Insights from Chapter 6:**")
        print("   1. Long/short trend following enables profit in falling markets")
        print("   2. Provides better diversification than long-only strategies")
        print("   3. Higher turnover due to position reversals on trend changes")
        print("   4. Similar volatility to no-trend but different risk characteristics")
        print("   5. The 'magical diversification machine' - creates negative correlations")
        
        # Long/short specific insights
        long_short_perf = comparison_results['long_short_trend']['performance']
        print(f"\n🔹 **Long/Short Strategy Characteristics:**")
        print(f"   Average Long Exposure: {long_short_perf.get('avg_long_exposure', 0):.1%}")
        print(f"   Average Short Exposure: {long_short_perf.get('avg_short_exposure', 0):.1%}")
        print(f"   Average Net Exposure: {long_short_perf.get('avg_net_exposure', 0):.1%}")
        print(f"   Average Gross Exposure: {long_short_perf.get('avg_gross_exposure', 0):.1%}")
        
    return comparison_results

def main():
    """
    Main function to run Chapter 6 long/short trend following strategy.
    """
    print("=" * 70)
    print("CHAPTER 6: SLOW TREND FOLLOWING, LONG AND SHORT")
    print("=" * 70)
    
    # Test the long/short trend following strategy
    results = test_long_short_trend_following_strategy(
        capital=1000000,
        risk_target=0.2,
        max_instruments=25,  # Manageable size for testing
        fast_window=64,      # Book's recommended fast window
        slow_window=256      # Book's recommended slow window
    )
    
    if results:
        print("\nChapter 6 long/short trend following strategy implementation completed successfully!")
    else:
        print("\nError: Could not complete long/short trend following strategy test.")

if __name__ == "__main__":
    main()
