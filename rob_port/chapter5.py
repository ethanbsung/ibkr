from chapter4 import *
from instrument_selection import optimize_instrument_selection
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def calculate_simple_moving_average(prices, window):
    """
    Calculate Simple Moving Average for trend identification.
    
    Formula: Moving average (256) = sum(p_t-255 + ... + p_t) ÷ 256
    
    Parameters:
        prices (pd.Series): Price series.
        window (int): Moving average window (e.g., 64, 256).
    
    Returns:
        pd.Series: Simple moving average series.
    """
    return prices.rolling(window=window, min_periods=1).mean()

def calculate_ewma_trend(prices, span):
    """
    Calculate Exponentially Weighted Moving Average for trend following.
    
    Formula: EWMA(λ) = λp_t + λ(1-λ)p_t-1 + λ(1-λ)²p_t-2 + ...
    where λ = 2 / (span + 1)
    
    Parameters:
        prices (pd.Series): Price series.
        span (int): EWMA span (e.g., 64, 256).
    
    Returns:
        pd.Series: EWMA series.
    """
    return prices.ewm(span=span, adjust=False).mean()

def calculate_moving_average_crossover(prices, fast_window=64, slow_window=256, use_ewma=True):
    """
    Calculate moving average crossover signals for trend following.
    Go long if fast MA > slow MA, otherwise reduce/close positions.
    
    Parameters:
        prices (pd.Series): Price series (back-adjusted futures prices recommended).
        fast_window (int): Fast moving average window (default 64).
        slow_window (int): Slow moving average window (default 256).
        use_ewma (bool): Use EWMA instead of SMA (recommended).
    
    Returns:
        dict: Dictionary containing fast MA, slow MA, and trend signals.
    """
    if use_ewma:
        fast_ma = calculate_ewma_trend(prices, fast_window)
        slow_ma = calculate_ewma_trend(prices, slow_window)
    else:
        fast_ma = calculate_simple_moving_average(prices, fast_window)
        slow_ma = calculate_simple_moving_average(prices, slow_window)
    
    # Trend signal: 1 for uptrend (fast > slow), 0 for downtrend/flat
    trend_signal = (fast_ma > slow_ma).astype(int)
    trend_strength = (fast_ma / slow_ma - 1)  # Relative strength measure
    
    return {
        'fast_ma': fast_ma,
        'slow_ma': slow_ma,
        'trend_signal': trend_signal,
        'trend_strength': trend_strength,
        'uptrend': fast_ma > slow_ma
    }

def apply_trend_filter_to_weights(portfolio_weights, trend_signals_dict):
    """
    Apply trend filters to portfolio weights.
    
    Parameters:
        portfolio_weights (dict): Original portfolio weights by instrument.
        trend_signals_dict (dict): Trend signals for each instrument.
    
    Returns:
        dict: Trend-adjusted weights for each date.
    """
    trend_adjusted_weights = {}
    
    # Get all dates from trend signals
    all_dates = set()
    for signals in trend_signals_dict.values():
        if 'trend_signal' in signals:
            all_dates.update(signals['trend_signal'].index)
    
    all_dates = sorted(all_dates)
    
    # For each date, calculate trend-adjusted weights
    for date in all_dates:
        date_weights = {}
        total_trend_weight = 0
        
        # First pass: calculate trend-adjusted weights
        for symbol, base_weight in portfolio_weights.items():
            if symbol in trend_signals_dict:
                signals = trend_signals_dict[symbol]
                if date in signals['trend_signal'].index:
                    trend_mult = signals['trend_signal'].loc[date]
                    trend_weight = base_weight * trend_mult
                    date_weights[symbol] = trend_weight
                    total_trend_weight += trend_weight
                else:
                    date_weights[symbol] = 0
            else:
                date_weights[symbol] = 0
        
        # Renormalize weights to sum to 1 (for instruments in uptrends)
        if total_trend_weight > 0:
            date_weights = {k: v / total_trend_weight for k, v in date_weights.items()}
        
        trend_adjusted_weights[date] = date_weights
    
    return trend_adjusted_weights

def backtest_trend_following_portfolio(portfolio_weights, data, instruments_df, capital=50000000, 
                                     risk_target=0.2, start_date='2000-01-01', end_date='2025-01-01',
                                     fast_window=64, slow_window=256, use_ewma=True):
    """
    Backtest trend-following portfolio strategy.
    
    This implements Chapter 5's trend following approach:
    1. Calculate trend signals for each instrument
    2. Apply trend filters to position sizing
    3. Go long only when in uptrend, flat when in downtrend
    
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
        dict: Backtest results with trend following metrics.
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

    # Calculate trend signals for each instrument
    trend_signals_dict = {}
    
    for symbol in normalized_weights.keys():
        try:
            symbol_data = data[symbol]
            if 'Last' in symbol_data.columns and len(symbol_data) > slow_window:
                # Use back-adjusted prices for trend calculation
                prices = symbol_data['Last'].copy()
                
                # Calculate trend signals
                trend_signals = calculate_moving_average_crossover(
                    prices, fast_window, slow_window, use_ewma
                )
                trend_signals_dict[symbol] = trend_signals
        except Exception as e:
            continue
    
    if not trend_signals_dict:
        return {'error': 'No trend signals could be calculated'}

    # Apply trend filters to get dynamic weights
    trend_adjusted_weights = apply_trend_filter_to_weights(normalized_weights, trend_signals_dict)
    
    # Calculate correlation matrix for IDM calculation
    returns_for_idm_list = []
    for s_key in normalized_weights.keys():
        if 'returns' in data[s_key] and isinstance(data[s_key]['returns'], pd.Series):
            s_returns = data[s_key]['returns'].copy()
            s_returns.name = s_key
            returns_for_idm_list.append(s_returns)

    correlation_matrix = pd.DataFrame()
    base_idm = 1.0  # Default IDM

    if returns_for_idm_list:
        common_returns_matrix = pd.concat(returns_for_idm_list, axis=1, join='inner')
        common_returns_matrix.dropna(inplace=True)

        if not common_returns_matrix.empty and common_returns_matrix.shape[0] >= 2 and common_returns_matrix.shape[1] > 1:
            try:
                correlation_matrix_calculated = calculate_correlation_matrix(common_returns_matrix)
                # Calculate base IDM using equal weights for all available instruments
                equal_weights_series = pd.Series({symbol: 1.0/len(normalized_weights) for symbol in normalized_weights.keys()})
                equal_weights_series = equal_weights_series.reindex(correlation_matrix_calculated.index).fillna(0)
                equal_weights_series = equal_weights_series / equal_weights_series.sum()
                base_idm_calculated = calculate_idm_from_correlations(equal_weights_series, correlation_matrix_calculated)
                base_idm = base_idm_calculated
                correlation_matrix = correlation_matrix_calculated
            except:
                if list(normalized_weights.keys()):
                    symbols_for_dummy_corr = list(normalized_weights.keys())
                    correlation_matrix = pd.DataFrame(np.eye(len(symbols_for_dummy_corr)), 
                                                    index=symbols_for_dummy_corr, columns=symbols_for_dummy_corr)
    
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
    trend_exposure_data = []  # Track percentage of portfolio in trends
    trend_signals_data = {symbol: [] for symbol in normalized_weights.keys()}
    idm_data = []  # Track IDM values over time
    
    for i, date_val in enumerate(returns_matrix.index):
        if i == 0:
            # Initialize first day
            for symbol in normalized_weights.keys():
                positions_data[symbol].append(0)
                trend_signals_data[symbol].append(0)
            portfolio_returns.append(0)
            trend_exposure_data.append(0)
            idm_data.append(base_idm)
            continue
        
        prev_date = returns_matrix.index[i-1]
        current_date = date_val
        
        # Get trend-adjusted weights for previous date (for position sizing)
        if prev_date in trend_adjusted_weights:
            current_weights = trend_adjusted_weights[prev_date]
        else:
            # Find closest available date
            available_dates = [d for d in trend_adjusted_weights.keys() if d <= prev_date]
            if available_dates:
                closest_date = max(available_dates)
                current_weights = trend_adjusted_weights[closest_date]
            else:
                current_weights = {symbol: 0 for symbol in normalized_weights.keys()}
        
        # Calculate dynamic IDM for instruments currently in uptrend
        active_instruments = [symbol for symbol, weight in current_weights.items() if weight > 0]
        if len(active_instruments) > 1 and not correlation_matrix.empty:
            try:
                # Get weights for active instruments only
                active_weights = {symbol: current_weights[symbol] for symbol in active_instruments}
                active_weights_series = pd.Series(active_weights)
                
                # Calculate IDM for active instruments
                active_correlation_matrix = correlation_matrix.reindex(index=active_instruments, columns=active_instruments)
                if active_correlation_matrix.shape[0] > 1:
                    current_idm = calculate_idm_from_correlations(active_weights_series, active_correlation_matrix)
                else:
                    current_idm = 1.0
            except:
                current_idm = base_idm
        else:
            current_idm = 1.0  # Single instrument or no correlation data
        
        # Calculate total trend exposure
        total_trend_exposure = sum(current_weights.values())
        trend_exposure_data.append(total_trend_exposure)
        idm_data.append(current_idm)
        
        # Calculate positions and P&L
        daily_portfolio_return = 0
        
        for symbol in normalized_weights.keys():
            try:
                specs = get_instrument_specs(symbol, instruments_df)
                multiplier = specs['multiplier']
                symbol_data_df = data[symbol]
                
                # Get trend-adjusted weight
                trend_weight = current_weights.get(symbol, 0)
                
                # Store trend signal
                if symbol in trend_signals_dict and prev_date in trend_signals_dict[symbol]['trend_signal'].index:
                    trend_signal = trend_signals_dict[symbol]['trend_signal'].loc[prev_date]
                else:
                    trend_signal = 0
                trend_signals_data[symbol].append(trend_signal)
                
                # Calculate position based on trend-adjusted weight
                if (prev_date in symbol_data_df.index and symbol_data_df.loc[prev_date, 'Last'] is not np.nan and 
                    current_date in symbol_data_df.index and symbol_data_df.loc[current_date, 'returns'] is not np.nan and
                    trend_weight > 0):
                    
                    prev_price = symbol_data_df.loc[prev_date, 'Last']
                    current_return = symbol_data_df.loc[current_date, 'returns']
                    
                    # Get volatility for position sizing
                    if prev_date in volatilities[symbol].index and not pd.isna(volatilities[symbol].loc[prev_date]):
                        blended_vol = volatilities[symbol].loc[prev_date]
                    else:
                        blended_vol = 0.16  # fallback volatility
                    
                    position = 0
                    if blended_vol > 0 and not pd.isna(prev_price) and prev_price > 0:
                        # Calculate position using trend-adjusted weight and dynamic IDM
                        position = calculate_position_size_with_idm(
                            capital, trend_weight, current_idm, multiplier, prev_price, 1.0, blended_vol, risk_target
                        )
                    
                    positions_data[symbol].append(position)
                    
                    # Calculate P&L
                    if not pd.isna(current_return) and position != 0:
                        notional_exposure = position * multiplier * prev_price
                        instrument_pnl = notional_exposure * current_return
                        instrument_return_contrib = instrument_pnl / capital
                        
                        # Safeguard against extreme values
                        if np.isinf(instrument_return_contrib) or np.isnan(instrument_return_contrib):
                            raise ValueError(f"Overflow in trend following P&L calculation for {symbol}")
                        
                        daily_portfolio_return += instrument_return_contrib
                else:
                    positions_data[symbol].append(0)
                    
            except Exception as e:
                positions_data[symbol].append(0)
                trend_signals_data[symbol].append(0)
        
        portfolio_returns.append(daily_portfolio_return)
    
    # Create results dataframe
    results_df = pd.DataFrame(index=returns_matrix.index)
    results_df['portfolio_returns'] = portfolio_returns
    results_df['trend_exposure'] = trend_exposure_data
    results_df['idm'] = idm_data
    
    for symbol, pos_list in positions_data.items():
        if len(pos_list) == len(results_df):
            results_df[f'position_{symbol}'] = pos_list
            
    for symbol, signal_list in trend_signals_data.items():
        if len(signal_list) == len(results_df):
            results_df[f'trend_signal_{symbol}'] = signal_list
    
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
    
    # Add trend-following specific metrics
    performance['num_instruments'] = len(normalized_weights)
    performance['avg_trend_exposure'] = results_df['trend_exposure'].mean()
    performance['max_trend_exposure'] = results_df['trend_exposure'].max()
    performance['min_trend_exposure'] = results_df['trend_exposure'].min()
    performance['percent_time_in_trends'] = (results_df['trend_exposure'] > 0.1).mean()
    performance['avg_idm'] = results_df['idm'].mean()
    performance['max_idm'] = results_df['idm'].max()
    performance['min_idm'] = results_df['idm'].min()
    performance['fast_window'] = fast_window
    performance['slow_window'] = slow_window
    performance['use_ewma'] = use_ewma
    
    # Calculate trend signal statistics
    trend_stats = {}
    for symbol in normalized_weights.keys():
        if f'trend_signal_{symbol}' in results_df.columns:
            signals = results_df[f'trend_signal_{symbol}']
            trend_stats[symbol] = {
                'uptrend_pct': signals.mean(),
                'signal_changes': signals.diff().abs().sum()
            }
    performance['trend_statistics'] = trend_stats
    
    return {
        'data': results_df,
        'performance': performance,
        'portfolio_weights': normalized_weights,
        'trend_signals': trend_signals_dict,
        'strategy_type': 'trend_following'
    }

def test_jumbo_portfolio_with_trend_following(capital=50000000, risk_target=0.2, 
                                            max_instruments=50, fast_window=64, slow_window=256):
    """
    Test trend following strategy on jumbo portfolio using optimized selection.
    
    This implements the book's Chapter 5 approach of applying trend filters to a 
    diversified portfolio selection.
    
    Parameters:
        capital (float): Initial capital.
        risk_target (float): Target risk fraction.
        max_instruments (int): Maximum instruments for jumbo portfolio.
        fast_window (int): Fast moving average window.
        slow_window (int): Slow moving average window.
    
    Returns:
        dict: Backtest results for trend following strategy.
    """
    # Load data and create jumbo portfolio using optimized selection
    instruments_df = load_instrument_data()
    available_instruments = get_available_instruments(instruments_df)
    
    # Select suitable instruments
    suitable_instruments = select_instruments_by_criteria(
        instruments_df, available_instruments, capital, max_cost_sr=0.01
    )
    
    if len(suitable_instruments) == 0:
        return {'error': 'No suitable instruments found'}
    
    # Load data
    data = load_instrument_data_files(suitable_instruments)
    
    if len(data) == 0:
        return {'error': 'No data loaded'}
    
    # Create optimized selection portfolio (Chapter 4 method)
    optimized_weights = optimize_instrument_selection(
        instruments_df, list(data.keys()), target_instruments=min(max_instruments, len(data))
    )
    
    if not optimized_weights:
        return {'error': 'Could not create optimized portfolio'}
    
    # Run trend following backtest
    results = backtest_trend_following_portfolio(
        optimized_weights, data, instruments_df, capital, risk_target,
        fast_window=fast_window, slow_window=slow_window
    )
    
    return results

def main():
    """
    Main function to run Chapter 5 trend following strategy.
    """
    # Run the trend following strategy on jumbo portfolio
    results = test_jumbo_portfolio_with_trend_following(
        capital=50000000,
        risk_target=0.2,
        max_instruments=25,  # Manageable size for testing
        fast_window=64,      # Book's recommended fast window
        slow_window=256      # Book's recommended slow window
    )
    
    if 'error' in results:
        print(f"Error: {results['error']}")
    else:
        print("Chapter 5 trend following strategy completed successfully!")
        print(f"Portfolio includes {results['performance']['num_instruments']} instruments")
        print(f"Annual Return: {results['performance']['annualized_return']:.1%}")
        print(f"Volatility: {results['performance']['annualized_volatility']:.1%}")
        print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['performance']['max_drawdown_pct']:.1f}%")
        print(f"Average Trend Exposure: {results['performance']['avg_trend_exposure']:.1%}")

if __name__ == "__main__":
    main()
