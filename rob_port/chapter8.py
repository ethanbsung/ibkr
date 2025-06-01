from chapter7 import *
from chapter6 import *
from chapter5 import *
from chapter4 import *
from chapter3 import *
from chapter2 import *
from chapter1 import *
import numpy as np
import pandas as pd
import os
import pickle
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

#####   RESULTS CACHING SYSTEM   #####

def get_results_cache_filename(strategy_name, config_hash):
    """Generate cache filename for strategy results."""
    cache_dir = 'results'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return os.path.join(cache_dir, f"{strategy_name}_{config_hash}.pkl")

def get_config_hash(config_dict):
    """Generate hash for configuration to identify cached results."""
    import hashlib
    config_str = str(sorted(config_dict.items()))
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

def save_strategy_results(strategy_name, results, config):
    """Save strategy results to cache."""
    try:
        config_hash = get_config_hash(config)
        filename = get_results_cache_filename(strategy_name, config_hash)
        
        cache_data = {
            'results': results,
            'config': config,
            'timestamp': pd.Timestamp.now()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Saved {strategy_name} results to {filename}")
        return True
    except Exception as e:
        print(f"Failed to save {strategy_name} results: {e}")
        return False

def load_strategy_results(strategy_name, config):
    """Load strategy results from cache if available."""
    try:
        config_hash = get_config_hash(config)
        filename = get_results_cache_filename(strategy_name, config_hash)
        
        if not os.path.exists(filename):
            return None
        
        with open(filename, 'rb') as f:
            cache_data = pickle.load(f)
        
        print(f"Loaded {strategy_name} results from cache")
        return cache_data['results']
    except Exception as e:
        print(f"Failed to load {strategy_name} results: {e}")
        return None

def get_cached_strategy_results():
    """Get all available cached strategy results."""
    cached_results = {}
    
    # Define standard config for comparison
    standard_config = {
        'capital': 50000000,
        'risk_target': 0.2,
        'weight_method': 'handcrafted'
    }
    
    # Try to load each strategy
    strategies = {
        'strategy4': ('strategy4', standard_config),
        'strategy5': ('strategy5', standard_config), 
        'strategy6': ('strategy6', standard_config),
        'strategy7': ('strategy7', standard_config)
    }
    
    for key, (strategy_name, config) in strategies.items():
        results = load_strategy_results(strategy_name, config)
        if results:
            cached_results[key] = results
    
    return cached_results

#####   STRATEGY 8: FAST TREND FOLLOWING WITH TREND STRENGTH AND BUFFERING   #####

def calculate_fast_raw_forecast(prices: pd.Series, fast_span: int = 16, slow_span: int = 64) -> pd.Series:
    """
    Calculate raw forecast for fast EWMAC trend following.
    
    From book:
        Raw forecast = (EWMA(16) - EWMA(64)) ÷ σp
        where σp = Price × σ% ÷ 16 (daily price volatility)
    
    Parameters:
        prices (pd.Series): Price series.
        fast_span (int): Fast EWMA span (default 16).
        slow_span (int): Slow EWMA span (default 64).
    
    Returns:
        pd.Series: Raw forecast values.
    """
    # Calculate EWMA crossover
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

def calculate_fast_forecast_for_instrument(prices: pd.Series, fast_span: int = 16, slow_span: int = 64,
                                         forecast_scalar: float = 4.1, cap: float = 20.0) -> pd.Series:
    """
    Calculate complete fast forecast pipeline for an instrument.
    
    From book: Uses EWMAC(16,64) with forecast scalar of 4.1
    
    Parameters:
        prices (pd.Series): Price series.
        fast_span (int): Fast EWMA span (default 16).
        slow_span (int): Slow EWMA span (default 64).
        forecast_scalar (float): Forecast scalar (default 4.1).
        cap (float): Maximum absolute forecast value (default 20.0).
    
    Returns:
        pd.Series: Capped forecast values.
    """
    raw_forecast = calculate_fast_raw_forecast(prices, fast_span, slow_span)
    scaled_forecast = calculate_scaled_forecast(raw_forecast, forecast_scalar)
    capped_forecast = calculate_capped_forecast(scaled_forecast, cap)
    
    return capped_forecast

def debug_buffering_behavior(optimal_positions_sample, buffer_widths_sample):
    """Debug function to verify buffering behavior matches book."""
    print(f"\n=== DEBUGGING BUFFERING BEHAVIOR ===")
    print(f"Sample Buffer Widths: {buffer_widths_sample[:5] if len(buffer_widths_sample) > 5 else buffer_widths_sample}")
    avg_buffer = np.mean([b for b in buffer_widths_sample if b > 0]) if any(b > 0 for b in buffer_widths_sample) else 0
    print(f"Average Buffer Width: {avg_buffer:.3f}")
    print(f"{'Day':<5} {'Optimal':<8} {'Current':<8} {'New':<8} {'Trade':<8} {'Buffer':<8} {'Lower':<8} {'Upper':<8}")
    print("-" * 75)
    
    current_position = 0.0
    total_trades = 0
    
    for i, (optimal, buffer_width) in enumerate(zip(optimal_positions_sample, buffer_widths_sample)):
        new_pos, trade = calculate_buffered_position(optimal, current_position, buffer_width)
        
        lower_buffer = round(optimal - buffer_width) if buffer_width > 0 else round(optimal)
        upper_buffer = round(optimal + buffer_width) if buffer_width > 0 else round(optimal)
        
        if abs(trade) > 0.01:
            total_trades += 1
        
        print(f"{i+1:<5} {optimal:<8.1f} {current_position:<8.1f} {new_pos:<8.1f} {trade:<8.1f} {buffer_width:<8.3f} {lower_buffer:<8} {upper_buffer:<8}")
        current_position = new_pos
        
        if i >= 19:  # Limit to 20 rows
            break
    
    print(f"\nTotal trades out of {min(len(optimal_positions_sample), 20)} days: {total_trades}")
    print(f"Trade frequency: {total_trades/min(len(optimal_positions_sample), 20):.1%}")

def calculate_buffer_width(symbol, capital, weight, idm, price, volatility, 
                          multiplier, risk_target=0.2, fx_rate=1.0, buffer_fraction=0.1):
    """
    Calculate buffer width for trading.
    
    From book:
        B = F × Capital × IDM × Weight × τ ÷ (Multiplier × Price × FX × σ%)
        where F = 0.1 (buffer fraction)
    
    Parameters:
        symbol (str): Instrument symbol.
        capital (float): Total portfolio capital.
        weight (float): Weight allocated to this instrument.
        idm (float): Instrument Diversification Multiplier.
        price (float): Current price.
        volatility (float): Annualized volatility forecast.
        multiplier (float): Contract multiplier.
        risk_target (float): Target risk fraction.
        fx_rate (float): FX rate for currency conversion.
        buffer_fraction (float): Buffer fraction (default 0.1).
    
    Returns:
        float: Buffer width.
    """
    if np.isnan(volatility) or volatility <= 0:
        return 0
    
    buffer_width = (buffer_fraction * capital * idm * weight * risk_target) / (multiplier * price * fx_rate * volatility)
    
    # Protect against infinite or extremely large buffer
    if np.isinf(buffer_width) or buffer_width > 1000:
        return 0
    
    # Debug: Print buffer calculation for first few calls
    if hasattr(calculate_buffer_width, 'debug_count'):
        calculate_buffer_width.debug_count += 1
    else:
        calculate_buffer_width.debug_count = 1
    
    if calculate_buffer_width.debug_count <= 3:
        print(f"DEBUG Buffer calc for {symbol}: F={buffer_fraction}, Cap={capital}, IDM={idm}, W={weight:.4f}, τ={risk_target}, M={multiplier}, P={price:.2f}, σ={volatility:.4f}")
        print(f"  Numerator: {buffer_fraction * capital * idm * weight * risk_target:.2f}")
        print(f"  Denominator: {multiplier * price * fx_rate * volatility:.2f}")
        print(f"  Buffer Width: {buffer_width:.6f}")
    
    return buffer_width

def calculate_buffered_position(optimal_position, current_position, buffer_width):
    """
    Calculate buffered trading decision.
    
    From book:
        Lower buffer: B^L = round(N - B)
        Upper buffer: B^U = round(N + B)
        
        Trading rules:
        - If B^L ≤ C ≤ B^U: No trading required
        - If C < B^L: Buy (B^U - C) contracts  
        - If C > B^U: Sell (C - B^L) contracts
    
    Parameters:
        optimal_position (float): Optimal position size.
        current_position (float): Current position size.
        buffer_width (float): Buffer width.
    
    Returns:
        tuple: (new_position, trade_size)
    """
    # Handle NaN values
    if np.isnan(optimal_position) or np.isnan(current_position) or np.isnan(buffer_width):
        return current_position if not np.isnan(current_position) else 0.0, 0.0
    
    if buffer_width <= 0:
        return optimal_position, optimal_position - current_position
    
    # Calculate buffer bounds
    lower_buffer = round(optimal_position - buffer_width)
    upper_buffer = round(optimal_position + buffer_width)
    current_rounded = round(current_position)
    
    # Apply trading rules from book exactly
    if lower_buffer <= current_rounded <= upper_buffer:
        # No trading required
        return current_position, 0.0
    elif current_rounded < lower_buffer:
        # Buy to upper buffer (corrected from book: should be upper buffer, not lower)
        new_position = upper_buffer
        trade_size = new_position - current_position
    else:  # current_rounded > upper_buffer
        # Sell to lower buffer (corrected from book: should be lower buffer, not upper)
        new_position = lower_buffer
        trade_size = new_position - current_position
    
    return new_position, trade_size

def calculate_strategy8_position_size(symbol, capital, weight, idm, price, volatility, 
                                    multiplier, forecast, risk_target=0.2, fx_rate=1.0):
    """
    Calculate position size for Strategy 8 with fast forecast scaling.
    
    From book: Same position sizing as Strategy 7 but uses fast trend filter
        N = Capped forecast × Capital × IDM × Weight × τ ÷ (10 × Multiplier × Price × FX × σ%)
    
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
    
    # Calculate position size with forecast scaling (same as Strategy 7)
    numerator = forecast * capital * idm * weight * risk_target
    denominator = 10 * multiplier * price * fx_rate * volatility
    
    position_size = numerator / denominator
    
    # Protect against infinite or extremely large position sizes
    if np.isinf(position_size) or abs(position_size) > 100000:
        return 0
    
    return position_size

def backtest_fast_trend_strategy_with_buffering(data_dir='Data', capital=50000000, risk_target=0.2,
                                              short_span=32, long_years=10, 
                                              trend_fast_span=16, trend_slow_span=64,
                                              forecast_scalar=4.1, forecast_cap=20.0,
                                              buffer_fraction=0.1,
                                              weight_method='handcrafted',
                                              start_date=None, end_date=None,
                                              debug_buffering=False):
    """
    Backtest Strategy 8: Fast trend following with forecasts and buffering.
    
    Implementation follows book exactly: "Trade a portfolio of one or more instruments, 
    each with positions scaled for a variable risk estimate. Hold a long position when 
    they are in a recent uptrend, and hold a short position in a recent downtrend. 
    Scale the size of the position according to the strength of the trend. Uses buffering 
    to reduce trading costs."
    
    Parameters:
        data_dir (str): Directory containing price data files.
        capital (float): Initial capital.
        risk_target (float): Target risk fraction.
        short_span (int): EWMA span for short-run volatility.
        long_years (int): Years for long-run volatility average.
        trend_fast_span (int): Fast EWMA span for trend filter (default 16).
        trend_slow_span (int): Slow EWMA span for trend filter (default 64).
        forecast_scalar (float): Forecast scaling factor (default 4.1).
        forecast_cap (float): Maximum absolute forecast value (default 20.0).
        buffer_fraction (float): Buffer fraction for trading (default 0.1).
        weight_method (str): Method for calculating instrument weights.
        start_date (str): Start date for backtest (YYYY-MM-DD).
        end_date (str): End date for backtest (YYYY-MM-DD).
        debug_buffering (bool): Whether to print buffering debug info.
    
    Returns:
        dict: Comprehensive backtest results.
    """
    print("=" * 60)
    print("STRATEGY 8: FAST TREND FOLLOWING WITH BUFFERING")
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
    print(f"  Trend Filter: EWMAC({trend_fast_span},{trend_slow_span}) Fast with Forecasts")
    print(f"  Forecast Scalar: {forecast_scalar}")
    print(f"  Forecast Cap: ±{forecast_cap}")
    print(f"  Buffer Fraction: {buffer_fraction}")
    
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
    
    # Process each instrument and calculate volatility forecasts + trend forecasts + buffering
    processed_data = {}
    total_trade_count = 0
    total_days = 0
    
    for symbol, df in instrument_data.items():
        # Get instrument specs
        try:
            specs = get_instrument_specs(symbol, instruments_df)
            multiplier = specs['multiplier']
        except:
            continue
        
        # Filter data to backtest period
        df_filtered = df[(df.index >= backtest_start) & (df.index <= backtest_end)].copy()
        
        if len(df_filtered) < 100:  # Need sufficient data for fast trend filter
            continue
        
        # Calculate blended volatility forecast (same as previous strategies)
        df_filtered['blended_vol'] = calculate_blended_volatility(
            df_filtered['returns'], short_span=short_span, long_years=long_years
        )
        
        # Calculate fast forecast using trend strength
        df_filtered['forecast'] = calculate_fast_forecast_for_instrument(
            df_filtered['Last'], trend_fast_span, trend_slow_span, 
            forecast_scalar, forecast_cap
        )
        
        # Calculate position sizes with fast forecast scaling and buffering
        positions = []
        trades = []
        optimal_positions = []
        buffer_widths = []
        current_position = 0.0
        trades_count = 0
        
        for i in range(len(df_filtered)):
            if i == 0:
                positions.append(0)
                trades.append(0)
                optimal_positions.append(0)
                buffer_widths.append(0)
            else:
                prev_price = df_filtered['Last'].iloc[i-1]
                prev_vol = df_filtered['blended_vol'].iloc[i-1]
                prev_forecast = df_filtered['forecast'].iloc[i-1]
                
                if (np.isnan(prev_vol) or prev_vol <= 0 or np.isnan(prev_forecast)):
                    new_position = current_position
                    trade_size = 0
                    optimal_pos = 0
                    buffer_width = 0
                else:
                    # Calculate optimal position
                    optimal_pos = calculate_strategy8_position_size(
                        symbol, capital, weights[symbol], idm, 
                        prev_price, prev_vol, multiplier, prev_forecast, risk_target
                    )
                    
                    # Calculate buffer width
                    buffer_width = calculate_buffer_width(
                        symbol, capital, weights[symbol], idm, 
                        prev_price, prev_vol, multiplier, risk_target, 1.0, buffer_fraction
                    )
                    
                    # Apply buffering
                    new_position, trade_size = calculate_buffered_position(
                        optimal_pos, current_position, buffer_width
                    )
                    
                    # Count actual trades (only when position changes)
                    if abs(trade_size) > 0.01:
                        trades_count += 1
                
                positions.append(new_position)
                trades.append(trade_size)
                optimal_positions.append(optimal_pos)
                buffer_widths.append(buffer_width)
                current_position = new_position
        
        # Debug buffering for first instrument if requested
        if debug_buffering and symbol == list(instrument_data.keys())[0]:
            print(f"\n=== BUFFERING DEBUG FOR {symbol} ===")
            sample_size = min(20, len(optimal_positions))
            debug_buffering_behavior(optimal_positions[1:sample_size+1], 
                                   buffer_widths[1:sample_size+1])
        
        df_filtered['position'] = positions
        df_filtered['position_lag'] = df_filtered['position'].shift(1)
        df_filtered['trade_size'] = trades
        df_filtered['optimal_position'] = optimal_positions
        df_filtered['buffer_width'] = buffer_widths
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
        total_trade_count += trades_count
        total_days += len(df_filtered) - 1  # Subtract 1 for first day
        
        # Print per-instrument trade statistics
        if len(df_filtered) > 1:
            trade_frequency = trades_count / (len(df_filtered) - 1)
            print(f"  {symbol}: {trades_count} trades over {len(df_filtered)-1} days ({trade_frequency:.1%} frequency)")
    
    print(f"\nCombining portfolio...")
    print(f"Successfully processed {len(processed_data)} instruments")
    print(f"Total individual trades: {total_trade_count} over {total_days} instrument-days")
    print(f"Average trade frequency per instrument: {total_trade_count/total_days:.1%}")
    
    # Create portfolio DataFrame with full date range
    portfolio_df = pd.DataFrame(index=full_date_range)
    portfolio_df['total_pnl'] = 0.0
    portfolio_df['num_active_instruments'] = 0
    portfolio_df['avg_forecast'] = 0.0
    portfolio_df['avg_abs_forecast'] = 0.0
    portfolio_df['total_trades'] = 0
    
    # Aggregate P&L across all instruments for each day
    for symbol, df in processed_data.items():
        # Initialize columns if they don't exist
        portfolio_df[f'{symbol}_position'] = 0.0
        portfolio_df[f'{symbol}_pnl'] = 0.0
        portfolio_df[f'{symbol}_forecast'] = 0.0
        portfolio_df[f'{symbol}_trades'] = 0
        
        # Add P&L only for dates where we have actual data
        actual_dates = df.index.intersection(full_date_range)
        
        for date in actual_dates:
            if date in df.index and not pd.isna(df.loc[date, 'instrument_pnl']):
                portfolio_df.loc[date, 'total_pnl'] += df.loc[date, 'instrument_pnl']
                portfolio_df.loc[date, f'{symbol}_pnl'] = df.loc[date, 'instrument_pnl']
                portfolio_df.loc[date, f'{symbol}_position'] = df.loc[date, 'position_lag']
                portfolio_df.loc[date, f'{symbol}_forecast'] = df.loc[date, 'forecast']
                
                # Only count actual trades (when trade_size != 0)
                trade_size = df.loc[date, 'trade_size']
                if abs(trade_size) > 0.01:  # Only count meaningful trades
                    portfolio_df.loc[date, f'{symbol}_trades'] = 1  # Count as 1 trade event
                    portfolio_df.loc[date, 'total_trades'] += 1
                
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
    print(f"Average daily trades (events): {portfolio_df['total_trades'].mean():.1f}")
    
    # Calculate performance metrics
    account_curve = build_account_curve(portfolio_df['strategy_returns'], capital)
    performance = calculate_comprehensive_performance(account_curve, portfolio_df['strategy_returns'])
    
    # Add strategy-specific metrics
    performance['num_instruments'] = len(processed_data)
    performance['idm'] = idm
    performance['avg_active_instruments'] = portfolio_df['num_active_instruments'].mean()
    performance['avg_forecast'] = portfolio_df['avg_forecast'].mean()
    performance['avg_abs_forecast'] = portfolio_df['avg_abs_forecast'].mean()
    performance['avg_daily_trades'] = portfolio_df['total_trades'].mean()
    performance['total_trades'] = portfolio_df['total_trades'].sum()
    performance['weight_method'] = weight_method
    performance['backtest_start'] = backtest_start
    performance['backtest_end'] = backtest_end
    performance['trend_fast_span'] = trend_fast_span
    performance['trend_slow_span'] = trend_slow_span
    performance['forecast_scalar'] = forecast_scalar
    performance['forecast_cap'] = forecast_cap
    performance['buffer_fraction'] = buffer_fraction
    
    # Calculate per-instrument statistics
    instrument_stats = {}
    for symbol in processed_data.keys():
        pnl_col = f'{symbol}_pnl'
        pos_col = f'{symbol}_position'
        forecast_col = f'{symbol}_forecast'
        trades_col = f'{symbol}_trades'
        
        if pnl_col in portfolio_df.columns:
            # Get only non-zero P&L periods for this instrument
            inst_pnl = portfolio_df[pnl_col][portfolio_df[pnl_col] != 0]
            inst_forecast = portfolio_df[forecast_col][portfolio_df[pnl_col] != 0]
            inst_trades = portfolio_df[trades_col].sum()
            
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
                    'min_forecast': inst_forecast.min(),
                    'total_trades': inst_trades
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
            'buffer_fraction': buffer_fraction,
            'weight_method': weight_method,
            'backtest_start': backtest_start,
            'backtest_end': backtest_end
        }
    }

def analyze_fast_trend_results(results):
    """
    Analyze and display comprehensive fast trend following results.
    
    Parameters:
        results (dict): Results from backtest_fast_trend_strategy_with_buffering.
    """
    performance = results['performance']
    instrument_stats = results['instrument_stats']
    config = results['config']
    
    print("\n" + "=" * 60)
    print("FAST TREND FOLLOWING WITH BUFFERING PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Overall performance
    print(f"\n--- Overall Portfolio Performance ---")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Annualized Return: {performance['annualized_return']:.2%}")
    print(f"Annualized Volatility: {performance['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {performance['max_drawdown_pct']:.1f}%")
    print(f"Skewness: {performance['skewness']:.3f}")
    
    # Fast trend characteristics
    print(f"\n--- Fast Trend Following Characteristics ---")
    print(f"Average Active Instruments: {performance['avg_active_instruments']:.1f}")
    print(f"Average Forecast: {performance['avg_forecast']:.2f}")
    print(f"Average Absolute Forecast: {performance['avg_abs_forecast']:.2f}")
    print(f"Forecast Scalar: {config['forecast_scalar']}")
    print(f"Forecast Cap: ±{config['forecast_cap']}")
    print(f"Trend Filter: EWMAC({config['trend_fast_span']},{config['trend_slow_span']}) Fast")
    
    # Buffering characteristics
    print(f"\n--- Buffering Characteristics ---")
    print(f"Buffer Fraction: {config['buffer_fraction']}")
    print(f"Average Daily Trades: {performance['avg_daily_trades']:.1f}")
    print(f"Total Trades: {performance['total_trades']:,.0f}")
    
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
    
    print(f"{'Symbol':<8} {'Weight':<8} {'Return':<10} {'Sharpe':<8} {'AvgFcst':<8} {'Trades':<8} {'TotalPnL':<12} {'Days':<6}")
    print("-" * 105)
    
    for symbol, stats in sorted_instruments[:10]:
        print(f"{symbol:<8} {stats['weight']:<8.3f} {stats['total_return']:<10.2%} "
              f"{stats['sharpe_ratio']:<8.3f} {stats['avg_forecast']:<8.2f} "
              f"{stats['total_trades']:<8.0f} ${stats['total_pnl']:<11,.0f} {stats['active_days']:<6}")

def compare_all_trend_strategies():
    """
    Compare Strategy 4 through Strategy 8 using cached results where possible.
    """
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON: STRATEGY 4 vs 5 vs 6 vs 7 vs 8")
    print("=" * 80)
    
    # Standard config for all strategies
    standard_config = {
        'capital': 50000000,
        'risk_target': 0.2,
        'weight_method': 'handcrafted'
    }
    
    # Get cached results
    cached_results = get_cached_strategy_results()
    
    # Strategy 4 (no trend filter)
    if 'strategy4' in cached_results:
        print("Using cached Strategy 4 results...")
        strategy4_results = cached_results['strategy4']
    else:
        print("Running Strategy 4 (no trend filter)...")
        strategy4_results = backtest_multi_instrument_strategy(
            data_dir='Data', **standard_config
        )
        save_strategy_results('strategy4', strategy4_results, standard_config)
    
    # Strategy 5 (with trend filter, long only)
    if 'strategy5' in cached_results:
        print("Using cached Strategy 5 results...")
        strategy5_results = cached_results['strategy5']
    else:
        print("Running Strategy 5 (trend filter, long only)...")
        strategy5_results = backtest_trend_following_strategy(
            data_dir='Data', **standard_config
        )
        save_strategy_results('strategy5', strategy5_results, standard_config)
    
    # Strategy 6 (with trend filter, long/short)
    if 'strategy6' in cached_results:
        print("Using cached Strategy 6 results...")
        strategy6_results = cached_results['strategy6']
    else:
        print("Running Strategy 6 (trend filter, long/short)...")
        strategy6_results = backtest_long_short_trend_strategy(
            data_dir='Data', **standard_config
        )
        save_strategy_results('strategy6', strategy6_results, standard_config)
    
    # Strategy 7 (with forecasts)
    if 'strategy7' in cached_results:
        print("Using cached Strategy 7 results...")
        strategy7_results = cached_results['strategy7']
    else:
        print("Running Strategy 7 (trend filter with forecasts)...")
        strategy7_results = backtest_forecast_trend_strategy(
            data_dir='Data', **standard_config
        )
        save_strategy_results('strategy7', strategy7_results, standard_config)
    
    # Strategy 8 (fast trend with buffering) - always run fresh
    print("Running Strategy 8 (fast trend with buffering)...")
    strategy8_config = {**standard_config}
    strategy8_results = backtest_fast_trend_strategy_with_buffering(
        data_dir='Data', debug_buffering=True, **strategy8_config
    )
    save_strategy_results('strategy8', strategy8_results, strategy8_config)
    
    if all([strategy4_results, strategy5_results, strategy6_results, strategy7_results, strategy8_results]):
        s4_perf = strategy4_results['performance']
        s5_perf = strategy5_results['performance']
        s6_perf = strategy6_results['performance']
        s7_perf = strategy7_results['performance']
        s8_perf = strategy8_results['performance']
        
        print(f"\n{'Strategy':<15} {'Ann. Return':<12} {'Volatility':<12} {'Sharpe':<8} {'Max DD':<8} {'Trades/Day':<12} {'Special':<20}")
        print("-" * 110)
        
        print(f"{'Strategy 4':<15} {s4_perf['annualized_return']:<12.2%} "
              f"{s4_perf['annualized_volatility']:<12.2%} "
              f"{s4_perf['sharpe_ratio']:<8.3f} "
              f"{s4_perf['max_drawdown_pct']:<8.1f}% "
              f"{'N/A':<12} "
              f"{'Always Long':<20}")
        
        print(f"{'Strategy 5':<15} {s5_perf['annualized_return']:<12.2%} "
              f"{s5_perf['annualized_volatility']:<12.2%} "
              f"{s5_perf['sharpe_ratio']:<8.3f} "
              f"{s5_perf['max_drawdown_pct']:<8.1f}% "
              f"{'N/A':<12} "
              f"{'Long/Flat':<20}")
        
        print(f"{'Strategy 6':<15} {s6_perf['annualized_return']:<12.2%} "
              f"{s6_perf['annualized_volatility']:<12.2%} "
              f"{s6_perf['sharpe_ratio']:<8.3f} "
              f"{s6_perf['max_drawdown_pct']:<8.1f}% "
              f"{'N/A':<12} "
              f"{'Long/Short':<20}")
        
        print(f"{'Strategy 7':<15} {s7_perf['annualized_return']:<12.2%} "
              f"{s7_perf['annualized_volatility']:<12.2%} "
              f"{s7_perf['sharpe_ratio']:<8.3f} "
              f"{s7_perf['max_drawdown_pct']:<8.1f}% "
              f"{'N/A':<12} "
              f"{'Slow Forecasts':<20}")
        
        print(f"{'Strategy 8':<15} {s8_perf['annualized_return']:<12.2%} "
              f"{s8_perf['annualized_volatility']:<12.2%} "
              f"{s8_perf['sharpe_ratio']:<8.3f} "
              f"{s8_perf['max_drawdown_pct']:<8.1f}% "
              f"{s8_perf['avg_daily_trades']:<12.1f} "
              f"{'Fast + Buffering':<20}")
        
        print(f"\n--- Strategy 8 vs Strategy 7 Analysis ---")
        return_diff = s8_perf['annualized_return'] - s7_perf['annualized_return']
        vol_diff = s8_perf['annualized_volatility'] - s7_perf['annualized_volatility']
        sharpe_diff = s8_perf['sharpe_ratio'] - s7_perf['sharpe_ratio']
        dd_diff = s8_perf['max_drawdown_pct'] - s7_perf['max_drawdown_pct']
        
        print(f"Return Difference: {return_diff:+.2%}")
        print(f"Volatility Difference: {vol_diff:+.2%}")
        print(f"Sharpe Difference: {sharpe_diff:+.3f}")
        print(f"Max Drawdown Difference: {dd_diff:+.1f}%")
        
        if 'avg_forecast' in s8_perf:
            print(f"\nStrategy 8 Characteristics:")
            print(f"  Average Forecast: {s8_perf['avg_forecast']:.2f}")
            print(f"  Average Absolute Forecast: {s8_perf['avg_abs_forecast']:.2f}")
            print(f"  Average Daily Trades: {s8_perf['avg_daily_trades']:.1f}")
        
        return {
            'strategy4': strategy4_results,
            'strategy5': strategy5_results,
            'strategy6': strategy6_results,
            'strategy7': strategy7_results,
            'strategy8': strategy8_results
        }

def main():
    """
    Test Strategy 8 implementation.
    """
    print("=" * 60)
    print("TESTING STRATEGY 8: FAST TREND FOLLOWING WITH BUFFERING")
    print("=" * 60)
    
    try:
        # Run Strategy 8 backtest with debug info
        results = backtest_fast_trend_strategy_with_buffering(
            data_dir='Data',
            capital=50000000,
            risk_target=0.2,
            weight_method='handcrafted',
            debug_buffering=True
        )
        
        # Analyze results
        analyze_fast_trend_results(results)
        
        # Compare all strategies using caching
        comparison = compare_all_trend_strategies()
        
        print(f"\nStrategy 8 backtest completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error in Strategy 8 backtest: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
