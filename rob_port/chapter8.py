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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')

#####   TRADING COST CALCULATIONS   #####

def calculate_trading_cost_from_sr(symbol, trade_size, price, volatility, multiplier, 
                                  sr_cost, capital, fx_rate=1.0):
    """
    Calculate trading cost in currency units from SR_cost.
    
    From the book: SR_cost represents the cost as a fraction of Sharpe ratio.
    To convert to currency units:
        Cost = |trade_size| × SR_cost × volatility × price × multiplier × fx_rate
    
    Parameters:
        symbol (str): Instrument symbol.
        trade_size (float): Absolute trade size in contracts.
        price (float): Current price.
        volatility (float): Annualized volatility forecast.
        multiplier (float): Contract multiplier.
        sr_cost (float): SR cost from instruments.csv.
        capital (float): Portfolio capital.
        fx_rate (float): FX rate for currency conversion.
    
    Returns:
        float: Trading cost in base currency (USD).
    """
    if trade_size == 0 or sr_cost == 0:
        return 0.0
    
    # Convert SR cost to currency cost
    # This is an approximation based on the book's cost methodology
    notional_per_contract = price * multiplier * fx_rate
    cost_per_contract = sr_cost * volatility * notional_per_contract
    total_cost = abs(trade_size) * cost_per_contract
    
    return total_cost

def calculate_position_change(previous_position, current_position):
    """
    Calculate position change and trade size.
    
    Parameters:
        previous_position (float): Previous position size.
        current_position (float): Current position size.
    
    Returns:
        float: Absolute trade size (always positive).
    """
    if pd.isna(previous_position) or pd.isna(current_position):
        return 0.0
    
    return abs(current_position - previous_position)

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

def calculate_fast_raw_forecast(prices: pd.Series, fast_span: int = 16, slow_span: int = 64,
                               short_span: int = 32, long_years: int = 10, min_vol_floor: float = 0.05) -> pd.Series:
    """
    Calculate raw forecast for fast EWMAC trend following.
    
    From book:
        Raw forecast = (EWMA(16) - EWMA(64)) ÷ σp
        where σp = Price × σ% ÷ 16 (daily price volatility)
    
    Parameters:
        prices (pd.Series): Price series.
        fast_span (int): Fast EWMA span (default 16).
        slow_span (int): Slow EWMA span (default 64).
        short_span (int): EWMA span for short-run volatility.
        long_years (int): Years for long-run volatility average.
        min_vol_floor (float): Minimum volatility floor.
    
    Returns:
        pd.Series: Raw forecast values.
    """
    # Calculate EWMA crossover
    ewmac = calculate_ewma_trend(prices, fast_span, slow_span)
    
    # Calculate blended volatility using the same method as other strategies
    returns = prices.pct_change().dropna()
    blended_vol = calculate_blended_volatility(returns, short_span, long_years, min_vol_floor)
    
    # Convert to daily price volatility: σp = Price × σ% ÷ 16
    daily_price_vol = prices * blended_vol / 16
    
    # Reindex to match EWMAC
    daily_price_vol = daily_price_vol.reindex(ewmac.index, method='ffill')
    
    # Calculate raw forecast: EWMAC ÷ σp
    raw_forecast = ewmac / daily_price_vol
    
    # Handle division by zero or very small volatility
    raw_forecast = raw_forecast.replace([np.inf, -np.inf], 0)
    raw_forecast = raw_forecast.fillna(0)
    
    return raw_forecast

def calculate_fast_forecast_for_instrument(prices: pd.Series, fast_span: int = 16, slow_span: int = 64,
                                         forecast_scalar: float = 4.1, cap: float = 20.0,
                                         short_span: int = 32, long_years: int = 10, min_vol_floor: float = 0.05) -> pd.Series:
    """
    Calculate complete fast forecast pipeline for an instrument.
    
    From book: Uses EWMAC(16,64) with forecast scalar of 4.1
    
    Parameters:
        prices (pd.Series): Price series.
        fast_span (int): Fast EWMA span (default 16).
        slow_span (int): Slow EWMA span (default 64).
        forecast_scalar (float): Forecast scalar (default 4.1).
        cap (float): Maximum absolute forecast value (default 20.0).
        short_span (int): EWMA span for short-run volatility.
        long_years (int): Years for long-run volatility average.
        min_vol_floor (float): Minimum volatility floor.
    
    Returns:
        pd.Series: Capped forecast values.
    """
    raw_forecast = calculate_fast_raw_forecast(prices, fast_span, slow_span, short_span, long_years, min_vol_floor)
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
                                              short_span=32, long_years=10, min_vol_floor=0.05,
                                              trend_fast_span=16, trend_slow_span=64,
                                              forecast_scalar=4.1, forecast_cap=20.0,
                                              buffer_fraction=0.1,
                                              weight_method='handcrafted',
                                              common_hypothetical_SR=0.3, annual_turnover_T=7.0,
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
        min_vol_floor (float): Minimum volatility floor.
        trend_fast_span (int): Fast EWMA span for trend filter (default 16).
        trend_slow_span (int): Slow EWMA span for trend filter (default 64).
        forecast_scalar (float): Forecast scaling factor (default 4.1).
        forecast_cap (float): Maximum absolute forecast value (default 20.0).
        buffer_fraction (float): Buffer fraction for trading (default 0.1).
        weight_method (str): Method for calculating instrument weights.
        common_hypothetical_SR (float): Common hypothetical SR.
        annual_turnover_T (float): Annual turnover T.
        start_date (str): Start date for backtest (YYYY-MM-DD).
        end_date (str): End date for backtest (YYYY-MM-DD).
        debug_buffering (bool): Whether to print buffering debug info.
    
    Returns:
        dict: Comprehensive backtest results.
    """
    print("=" * 60)
    print("STRATEGY 8: FAST TREND FOLLOWING WITH BUFFERING")
    print("=" * 60)
    
    # Load all instrument data using the same function as chapter 4-7
    all_instruments_specs_df = load_instrument_data()
    raw_instrument_data = load_all_instrument_data(data_dir)
    
    if not raw_instrument_data:
        raise ValueError("No instrument data loaded successfully")
    
    print(f"\nPortfolio Configuration:")
    print(f"  Instruments initially loaded: {len(raw_instrument_data)}")
    print(f"  Capital: ${capital:,.0f}")
    print(f"  Risk Target: {risk_target:.1%}")
    print(f"  Weight Method: {weight_method}")
    print(f"  Trend Filter: EWMAC({trend_fast_span},{trend_slow_span}) Fast with Forecasts")
    print(f"  Forecast Scalar: {forecast_scalar}")
    print(f"  Forecast Cap: ±{forecast_cap}")
    print(f"  Buffer Fraction: {buffer_fraction}")
    print(f"  Common Hypothetical SR for SR': {common_hypothetical_SR}")
    print(f"  Annual Turnover T for SR': {annual_turnover_T}")

    # Preprocess: Calculate returns, vol forecasts, and trend forecasts for each instrument
    processed_instrument_data = {}
    for symbol, df_orig in raw_instrument_data.items():
        df = df_orig.copy()
        if 'Last' not in df.columns:
            print(f"Skipping {symbol}: 'Last' column missing.")
            continue
        
        df['daily_price_change_pct'] = df['Last'].pct_change()
        
        # Volatility forecast for day D is made using data up to D-1 (no lookahead bias)
        raw_returns_for_vol = df['daily_price_change_pct'].dropna()
        if len(raw_returns_for_vol) < max(short_span, trend_slow_span):
            print(f"Skipping {symbol}: Insufficient data for vol forecast and trend ({len(raw_returns_for_vol)} days).")
            continue

        # Calculate blended volatility (same as Strategy 4-7)
        blended_vol_series = calculate_blended_volatility(
            raw_returns_for_vol, short_span=short_span, long_years=long_years, min_vol_floor=min_vol_floor
        )
        # Shift to prevent lookahead bias - forecast for day T uses data up to T-1
        df['vol_forecast'] = blended_vol_series.shift(1).reindex(df.index).ffill().fillna(min_vol_floor)
        
        # Calculate fast forecast using trend strength (no lookahead bias)
        forecast_series = calculate_fast_forecast_for_instrument(
            df['Last'], trend_fast_span, trend_slow_span, forecast_scalar, forecast_cap, short_span, long_years, min_vol_floor
        )
        # Shift to prevent lookahead bias - forecast for day T uses data up to T-1
        df['forecast'] = forecast_series.shift(1).reindex(df.index).fillna(0)
        
        # Ensure critical data is present
        df.dropna(subset=['Last', 'vol_forecast', 'daily_price_change_pct'], inplace=True)
        if df.empty:
            print(f"Skipping {symbol}: Empty after dropping NaNs in critical columns.")
            continue

        processed_instrument_data[symbol] = df

    if not processed_instrument_data:
        raise ValueError("No instruments remaining after preprocessing and volatility calculation.")
    
    print(f"  Instruments after preprocessing: {len(processed_instrument_data)}")

    # Determine common date range for backtest (same logic as chapter 4-7)
    all_indices = [df.index for df in processed_instrument_data.values() if not df.empty]
    if not all_indices:
        raise ValueError("No valid instrument data in processed_instrument_data to determine date range.")

    all_available_start_dates = [idx.min() for idx in all_indices]
    all_available_end_dates = [idx.max() for idx in all_indices]

    global_min_date = min(all_available_start_dates) if all_available_start_dates else pd.Timestamp.min
    global_max_date = max(all_available_end_dates) if all_available_end_dates else pd.Timestamp.max
    
    backtest_start_dt = pd.to_datetime(start_date) if start_date else global_min_date
    backtest_end_dt = pd.to_datetime(end_date) if end_date else global_max_date
    
    # Clamp user-defined dates to the absolute earliest/latest possible dates from data
    backtest_start_dt = max(backtest_start_dt, global_min_date)
    backtest_end_dt = min(backtest_end_dt, global_max_date)

    if backtest_start_dt >= backtest_end_dt:
        raise ValueError(f"Invalid backtest period: Start {backtest_start_dt}, End {backtest_end_dt}")

    # Use a common business day index
    trading_days_range = pd.bdate_range(start=backtest_start_dt, end=backtest_end_dt)
    
    print(f"\nBacktest Period (effective, common across instruments):")
    print(f"  Start: {trading_days_range.min().date()}")
    print(f"  End: {trading_days_range.max().date()}")
    print(f"  Duration: {len(trading_days_range)} trading days")

    # Initialize portfolio tracking (same structure as chapter 4-7)
    current_portfolio_equity = capital
    portfolio_daily_records = []
    known_eligible_instruments = set()
    weights = {} 
    idm = 1.0
    
    # Initialize buffering state - track current positions for each instrument
    current_positions = {}

    # Main time-stepping loop with daily position updates and buffering
    for idx, current_date in enumerate(trading_days_range):
        if idx == 0:
            # First day setup
            record = {'date': current_date, 'total_pnl': 0.0, 'portfolio_return': 0.0, 
                      'equity_sod': current_portfolio_equity, 'equity_eod': current_portfolio_equity,
                      'num_active_instruments': 0, 'avg_forecast': 0.0, 'avg_abs_forecast': 0.0, 'total_trades': 0}
            for symbol_k in processed_instrument_data.keys(): 
                record[f'{symbol_k}_contracts'] = 0.0
                record[f'{symbol_k}_forecast'] = 0.0
                record[f'{symbol_k}_trades'] = 0
                current_positions[symbol_k] = 0.0  # Initialize buffering state
            portfolio_daily_records.append(record)
            continue
        
        previous_trading_date = trading_days_range[idx-1]
        capital_at_start_of_day = current_portfolio_equity
        daily_total_pnl = 0.0
        current_day_positions_and_forecasts = {}
        num_active_instruments = 0
        daily_forecasts = []
        daily_trades = 0

        effective_data_cutoff_date = previous_trading_date

        # Determine current period eligible instruments based on data up to cutoff
        current_iteration_eligible_instruments = set()
        for s, df_full in processed_instrument_data.items():
            df_upto_cutoff = df_full[df_full.index <= effective_data_cutoff_date]
            if not df_upto_cutoff.empty and len(df_upto_cutoff) > max(short_span, trend_slow_span):
                current_iteration_eligible_instruments.add(s)
        
        # Check if reweighting is needed (same logic as chapter 4-7)
        perform_reweight = False
        if idx == 1:  # First actual trading day
            perform_reweight = True
            print(f"Performing initial re-weighting for date: {current_date.date()}")
        elif len(current_iteration_eligible_instruments) > len(known_eligible_instruments):
            newly_added = current_iteration_eligible_instruments - known_eligible_instruments
            perform_reweight = True
            print(f"Performing re-weighting for date: {current_date.date()} due to new eligible instruments: {newly_added}")
        
        if perform_reweight:
            known_eligible_instruments = current_iteration_eligible_instruments.copy()
            
            data_for_reweighting = {}
            for s_eligible in known_eligible_instruments:
                df_historical_slice = processed_instrument_data[s_eligible][processed_instrument_data[s_eligible].index <= effective_data_cutoff_date]
                if not df_historical_slice.empty:
                     data_for_reweighting[s_eligible] = df_historical_slice
            
            if data_for_reweighting:
                weights = calculate_instrument_weights(
                    data_for_reweighting, 
                    weight_method, 
                    all_instruments_specs_df,
                    common_hypothetical_SR,
                    annual_turnover_T,
                    risk_target
                )
                
                num_weighted_instruments = sum(1 for w_val in weights.values() if w_val > 1e-6)
                idm = calculate_idm_from_count(num_weighted_instruments)
                print(f"  New IDM: {idm:.2f} based on {num_weighted_instruments} instruments with weight > 0.")
            else:
                print(f"Warning: No data available for reweighting on {current_date.date()} despite eligibility signal.")

        # Calculate positions and P&L for each instrument with buffering
        for symbol, df_instrument in processed_instrument_data.items():
            try:
                specs = get_instrument_specs(symbol, all_instruments_specs_df)
                instrument_multiplier = specs['multiplier']
            except:
                continue
                
            instrument_weight = weights.get(symbol, 0.0)
            num_contracts = current_positions.get(symbol, 0.0)  # Start with current position
            instrument_pnl_today = 0.0
            actual_forecast_used = 0.0
            trade_size = 0.0

            if instrument_weight > 1e-6:
                try:
                    # Sizing based on previous day's close price and current day's forecasts
                    price_for_sizing = df_instrument.loc[previous_trading_date, 'Last']
                    vol_for_sizing = df_instrument.loc[current_date, 'vol_forecast'] / np.sqrt(business_days_per_year)
                    forecast_for_sizing = df_instrument.loc[current_date, 'forecast']
                    actual_forecast_used = forecast_for_sizing
                    
                    # Data for P&L calculation for current_date
                    price_at_start_of_trading = df_instrument.loc[previous_trading_date, 'Last']
                    price_at_end_of_trading = df_instrument.loc[current_date, 'Last']
                    
                    if (pd.isna(price_for_sizing) or pd.isna(vol_for_sizing) or 
                        pd.isna(price_at_start_of_trading) or pd.isna(price_at_end_of_trading) or
                        pd.isna(forecast_for_sizing)):
                        num_contracts = current_positions.get(symbol, 0.0)
                        instrument_pnl_today = 0.0
                        trade_size = 0.0
                    else:
                        vol_for_sizing = max(vol_for_sizing, min_vol_floor)
                        
                        # Calculate optimal position size with fast forecast scaling
                        optimal_position = calculate_strategy8_position_size(
                            symbol=symbol, capital=capital_at_start_of_day, weight=instrument_weight, 
                            idm=idm, price=price_for_sizing, volatility=vol_for_sizing, 
                            multiplier=instrument_multiplier, forecast=forecast_for_sizing, 
                            risk_target=risk_target
                        )
                        
                        # Calculate buffer width
                        buffer_width = calculate_buffer_width(
                            symbol, capital_at_start_of_day, instrument_weight, idm, 
                            price_for_sizing, vol_for_sizing, instrument_multiplier, 
                            risk_target, 1.0, buffer_fraction
                        )
                        
                        # Apply buffering to get actual position
                        current_pos = current_positions.get(symbol, 0.0)
                        num_contracts, trade_size = calculate_buffered_position(
                            optimal_position, current_pos, buffer_width
                        )
                        
                        # Calculate P&L based on the position we held during the day (BEFORE any trades)
                        # This is the position we entered the day with
                        gross_pnl = current_pos * instrument_multiplier * (price_at_end_of_trading - price_at_start_of_trading)
                        
                        # Calculate trading costs (only apply if there are actual trades)
                        trading_cost = 0.0
                        if abs(trade_size) > 0.01:  # Only apply costs when there are significant trades
                            sr_cost = specs.get('sr_cost', 0.0)
                            if not pd.isna(sr_cost) and sr_cost > 0:
                                trading_cost = calculate_trading_cost_from_sr(
                                    symbol, abs(trade_size), price_at_start_of_trading, vol_for_sizing * np.sqrt(business_days_per_year),
                                    instrument_multiplier, sr_cost, capital_at_start_of_day, 1.0
                                )
                        
                        # Net P&L after costs
                        instrument_pnl_today = gross_pnl - trading_cost
                        
                        # Update current position for next iteration (AFTER P&L calculation)
                        current_positions[symbol] = num_contracts
                        
                        # Count active instruments and collect forecasts
                        if abs(num_contracts) > 0.01:
                            num_active_instruments += 1
                        if not pd.isna(forecast_for_sizing):
                            daily_forecasts.append(forecast_for_sizing)
                        if abs(trade_size) > 0.01:
                            daily_trades += 1
                
                except KeyError:  # Date not found for this instrument
                    num_contracts = current_positions.get(symbol, 0.0)
                    instrument_pnl_today = 0.0
                    actual_forecast_used = 0.0
                    trade_size = 0.0
            
            current_day_positions_and_forecasts[symbol] = {
                'contracts': num_contracts, 
                'forecast': actual_forecast_used,
                'trades': 1 if abs(trade_size) > 0.01 else 0
            }
            daily_total_pnl += instrument_pnl_today

        # Calculate daily forecast metrics
        avg_forecast = np.mean(daily_forecasts) if daily_forecasts else 0.0
        avg_abs_forecast = np.mean([abs(f) for f in daily_forecasts]) if daily_forecasts else 0.0

        # Update portfolio equity (same as chapter 4-7)
        portfolio_daily_percentage_return = daily_total_pnl / capital_at_start_of_day if capital_at_start_of_day > 0 else 0.0
        current_portfolio_equity = capital_at_start_of_day * (1 + portfolio_daily_percentage_return)

        # Record daily results
        record = {'date': current_date, 'total_pnl': daily_total_pnl, 
                  'portfolio_return': portfolio_daily_percentage_return, 
                  'equity_sod': capital_at_start_of_day, 
                  'equity_eod': current_portfolio_equity,
                  'num_active_instruments': num_active_instruments,
                  'avg_forecast': avg_forecast,
                  'avg_abs_forecast': avg_abs_forecast,
                  'total_trades': daily_trades}
        
        for symbol_k, data_k in current_day_positions_and_forecasts.items(): 
            record[f'{symbol_k}_contracts'] = data_k['contracts']
            record[f'{symbol_k}_forecast'] = data_k['forecast']
            record[f'{symbol_k}_trades'] = data_k['trades']
        
        # Ensure all processed instruments have entries in the record
        for s_proc in processed_instrument_data.keys():
            if f'{s_proc}_contracts' not in record:
                record[f'{s_proc}_contracts'] = current_positions.get(s_proc, 0.0)
            if f'{s_proc}_forecast' not in record:
                forecast_val_fill = 0.0
                if current_date in processed_instrument_data[s_proc].index:
                    sig = processed_instrument_data[s_proc].loc[current_date, 'forecast']
                    if pd.notna(sig):
                        forecast_val_fill = sig
                record[f'{s_proc}_forecast'] = forecast_val_fill
            if f'{s_proc}_trades' not in record:
                record[f'{s_proc}_trades'] = 0
                
        portfolio_daily_records.append(record)

    # Post-loop processing (same as chapter 4-7)
    if not portfolio_daily_records:
        raise ValueError("No daily records generated during backtest.")
        
    portfolio_df = pd.DataFrame(portfolio_daily_records)
    portfolio_df.set_index('date', inplace=True)
    
    print(f"Portfolio backtest loop completed. {len(portfolio_df)} daily records.")
    if portfolio_df.empty or 'portfolio_return' not in portfolio_df.columns or portfolio_df['portfolio_return'].std() == 0:
        print("Warning: Portfolio returns are zero or constant. P&L might not be calculated as expected.")
    
    # Calculate performance metrics
    account_curve = build_account_curve(portfolio_df['portfolio_return'], capital)
    performance = calculate_comprehensive_performance(account_curve, portfolio_df['portfolio_return'])
    
    # Add strategy-specific metrics
    performance['num_instruments'] = len(processed_instrument_data)
    performance['idm'] = idm
    performance['avg_active_instruments'] = portfolio_df['num_active_instruments'].mean()
    performance['avg_forecast'] = portfolio_df['avg_forecast'].mean()
    performance['avg_abs_forecast'] = portfolio_df['avg_abs_forecast'].mean()
    performance['avg_daily_trades'] = portfolio_df['total_trades'].mean()
    performance['total_trades'] = portfolio_df['total_trades'].sum()
    performance['weight_method'] = weight_method
    performance['backtest_start'] = trading_days_range.min()
    performance['backtest_end'] = trading_days_range.max()
    performance['trend_fast_span'] = trend_fast_span
    performance['trend_slow_span'] = trend_slow_span
    performance['forecast_scalar'] = forecast_scalar
    performance['forecast_cap'] = forecast_cap
    performance['buffer_fraction'] = buffer_fraction

    # Calculate per-instrument statistics (simplified for now)
    instrument_stats = {}
    for symbol in processed_instrument_data.keys():
        pos_col = f'{symbol}_contracts'
        forecast_col = f'{symbol}_forecast'
        trades_col = f'{symbol}_trades'
        
        if pos_col in portfolio_df.columns:
            # Calculate basic statistics for instruments with positions
            inst_positions = portfolio_df[pos_col][portfolio_df[pos_col] != 0]
            inst_forecasts = portfolio_df[forecast_col][portfolio_df[pos_col] != 0]
            inst_trades = portfolio_df[trades_col].sum()
            
            if len(inst_positions) > 0:
                instrument_stats[symbol] = {
                    'avg_position': inst_positions.mean(),
                    'weight': weights.get(symbol, 0.0),
                    'active_days': len(inst_positions),
                    'avg_forecast': inst_forecasts.mean() if len(inst_forecasts) > 0 else 0.0,
                    'avg_abs_forecast': inst_forecasts.abs().mean() if len(inst_forecasts) > 0 else 0.0,
                    'max_forecast': inst_forecasts.max() if len(inst_forecasts) > 0 else 0.0,
                    'min_forecast': inst_forecasts.min() if len(inst_forecasts) > 0 else 0.0,
                    'total_trades': inst_trades
                }

    return {
        'portfolio_data': portfolio_df,
        'performance': performance,
        'instrument_stats': instrument_stats,
        'weights': weights,
        'idm': idm,
        'config': {
            'capital': capital,
            'risk_target': risk_target,
            'short_span': short_span,
            'long_years': long_years,
            'min_vol_floor': min_vol_floor,
            'trend_fast_span': trend_fast_span,
            'trend_slow_span': trend_slow_span,
            'forecast_scalar': forecast_scalar,
            'forecast_cap': forecast_cap,
            'buffer_fraction': buffer_fraction,
            'weight_method': weight_method,
            'common_hypothetical_SR': common_hypothetical_SR,
            'annual_turnover_T': annual_turnover_T,
            'backtest_start': trading_days_range.min(),
            'backtest_end': trading_days_range.max()
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
    
    # Fast trend characteristics with buffering
    print(f"\n--- Fast Trend + Buffering Characteristics ---")
    print(f"Average Active Instruments: {performance['avg_active_instruments']:.1f}")
    print(f"Average Forecast: {performance['avg_forecast']:.2f}")
    print(f"Average Absolute Forecast: {performance['avg_abs_forecast']:.2f}")
    print(f"Forecast Scalar: {config['forecast_scalar']}")
    print(f"Forecast Cap: ±{config['forecast_cap']}")
    print(f"Buffer Fraction: {config['buffer_fraction']:.1%}")
    print(f"Average Daily Trades: {performance['avg_daily_trades']:.1f}")
    print(f"Total Trades: {performance['total_trades']:,}")
    print(f"Trend Filter: EWMAC({config['trend_fast_span']},{config['trend_slow_span']}) Fast with Buffering")
    
    # Portfolio characteristics
    print(f"\n--- Portfolio Characteristics ---")
    print(f"Number of Instruments: {performance['num_instruments']}")
    print(f"IDM: {performance['idm']:.2f}")
    print(f"Capital: ${config['capital']:,.0f}")
    print(f"Risk Target: {config['risk_target']:.1%}")
    print(f"Backtest Period: {config['backtest_start'].date()} to {config['backtest_end'].date()}")
    
    # Top performing instruments (by weight since total_pnl is no longer calculated)
    print(f"\n--- Top 10 Performing Instruments (by Weight and Activity) ---")
    sorted_instruments = sorted(
        instrument_stats.items(), 
        key=lambda x: x[1]['weight'], 
        reverse=True
    )
    
    print(f"{'Symbol':<8} {'Weight':<8} {'Avg Pos':<10} {'AvgFcst':<8} {'MaxFcst':<8} {'Trades':<8} {'Days':<6}")
    print("-" * 75)
    
    for symbol, stats in sorted_instruments[:10]:
        print(f"{symbol:<8} {stats['weight']:<8.3f} {stats['avg_position']:<10.2f} "
              f"{stats['avg_forecast']:<8.2f} {stats['max_forecast']:<8.2f} {stats['total_trades']:<8} {stats['active_days']:<6}")
    
    # Show instruments with highest trade activity
    print(f"\n--- Top 10 Most Active Trading Instruments (by Total Trades) ---")
    sorted_by_trades = sorted(
        instrument_stats.items(), 
        key=lambda x: x[1]['total_trades'], 
        reverse=True
    )
    
    print(f"{'Symbol':<8} {'Trades':<8} {'Days':<6} {'AvgFcst':<8} {'AbsFcst':<8} {'Weight':<8} {'Avg Pos':<10}")
    print("-" * 75)
    
    for symbol, stats in sorted_by_trades[:10]:
        print(f"{symbol:<8} {stats['total_trades']:<8} {stats['active_days']:<6} {stats['avg_forecast']:<8.2f} "
              f"{stats['avg_abs_forecast']:<8.2f} {stats['weight']:<8.3f} {stats['avg_position']:<10.2f}")
    
    # Summary of buffering efficiency
    total_active_days = sum(stats['active_days'] for stats in instrument_stats.values())
    total_trades = sum(stats['total_trades'] for stats in instrument_stats.values())
    avg_forecast_all = sum(stats['avg_forecast'] for stats in instrument_stats.values()) / len(instrument_stats)
    avg_abs_forecast_all = sum(stats['avg_abs_forecast'] for stats in instrument_stats.values()) / len(instrument_stats)
    
    print(f"\n--- Buffering Summary ---")
    print(f"Total instrument-days with positions: {total_active_days:,}")
    print(f"Total trades across all instruments: {total_trades:,}")
    print(f"Trade frequency: {total_trades / total_active_days:.1%} (trades per instrument-day)")
    print(f"Average forecast across all instruments: {avg_forecast_all:.2f}")
    print(f"Average absolute forecast across all instruments: {avg_abs_forecast_all:.2f}")
    print(f"Instruments with any activity: {len(instrument_stats)}")

def plot_strategy8_equity_curve(results, save_path='results/strategy8_equity_curve.png'):
    """
    Plot Strategy 8 equity curve and save to file.
    
    Parameters:
        results (dict): Results from backtest_fast_trend_strategy_with_buffering.
        save_path (str): Path to save the plot.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        portfolio_df = results['portfolio_data']
        config = results['config']
        performance = results['performance']
        
        equity_curve = build_account_curve(portfolio_df['portfolio_return'], config['capital'])
        
        plt.figure(figsize=(14, 8))
        
        # Plot equity curve
        plt.plot(equity_curve.index, equity_curve.values/1e6, 'green', linewidth=2.5, 
                label=f'Strategy 8: Fast Trend Following with Buffering (SR: {performance["sharpe_ratio"]:.3f})')
        
        plt.title('Strategy 8: Fast Trend Following with Buffering Equity Curve', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Portfolio Value ($M)', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='upper left')
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Strategy 8 equity curve saved to: {save_path}")
        
    except Exception as e:
        print(f"Error plotting Strategy 8 equity curve: {e}")
        import traceback
        traceback.print_exc()

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
        'short_span': 32,
        'long_years': 10,
        'min_vol_floor': 0.05,
        'weight_method': 'handcrafted',
        'common_hypothetical_SR': 0.3,
        'annual_turnover_T': 7.0
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
    
    # Strategy 8 (fast trend with buffering)
    if 'strategy8' in cached_results:
        print("Using cached Strategy 8 results...")
        strategy8_results = cached_results['strategy8']
    else:
        print("Running Strategy 8 (fast trend with buffering)...")
        strategy8_results = backtest_fast_trend_strategy_with_buffering(
            data_dir='Data',
            capital=50000000,
            risk_target=0.2,
            short_span=32,
            long_years=10,
            min_vol_floor=0.05,
            trend_fast_span=16,
            trend_slow_span=64,
            forecast_scalar=4.1,
            forecast_cap=20.0,
            buffer_fraction=0.1,
            weight_method='handcrafted',
            common_hypothetical_SR=0.3,
            annual_turnover_T=7.0,
            debug_buffering=False
        )
        save_strategy_results('strategy8', strategy8_results, standard_config)
    
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
    # ===========================================
    # CONFIGURATION - MODIFY THESE AS NEEDED
    # ===========================================
    CAPITAL = 1000000               # Starting capital
    START_DATE = '2000-01-01'       # Backtest start date (YYYY-MM-DD) or None for earliest available
    END_DATE = '2020-01-01'         # Backtest end date (YYYY-MM-DD) or None for latest available
    RISK_TARGET = 0.2               # 20% annual risk target
    WEIGHT_METHOD = 'handcrafted'   # 'equal', 'vol_inverse', or 'handcrafted'
    TREND_FAST_SPAN = 16            # Fast EWMA span for trend filter
    TREND_SLOW_SPAN = 64            # Slow EWMA span for trend filter
    FORECAST_SCALAR = 4.1           # Forecast scaling factor
    FORECAST_CAP = 20.0             # Maximum absolute forecast value
    BUFFER_FRACTION = 0.1           # Buffer fraction for trading
    
    print("=" * 60)
    print("TESTING STRATEGY 8: FAST TREND FOLLOWING WITH BUFFERING")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Capital: ${CAPITAL:,}")
    print(f"  Date Range: {START_DATE or 'earliest'} to {END_DATE or 'latest'}")
    print(f"  Risk Target: {RISK_TARGET:.1%}")
    print(f"  Weight Method: {WEIGHT_METHOD}")
    print(f"  Trend Filter: EWMA({TREND_FAST_SPAN},{TREND_SLOW_SPAN}) Fast with Buffering")
    print(f"  Forecast Scalar: {FORECAST_SCALAR}")
    print(f"  Forecast Cap: ±{FORECAST_CAP}")
    print(f"  Buffer Fraction: {BUFFER_FRACTION:.1%}")
    print("=" * 60)
    
    try:
        # Run Strategy 8 backtest
        results = backtest_fast_trend_strategy_with_buffering(
            data_dir='Data',
            capital=CAPITAL,
            risk_target=RISK_TARGET,
            short_span=32,
            long_years=10,
            min_vol_floor=0.05,
            trend_fast_span=TREND_FAST_SPAN,
            trend_slow_span=TREND_SLOW_SPAN,
            forecast_scalar=FORECAST_SCALAR,
            forecast_cap=FORECAST_CAP,
            buffer_fraction=BUFFER_FRACTION,
            weight_method=WEIGHT_METHOD,
            common_hypothetical_SR=0.3,
            annual_turnover_T=7.0,
            start_date=START_DATE,
            end_date=END_DATE,
            debug_buffering=False
        )
        
        # Analyze results
        analyze_fast_trend_results(results)
        
        # Plot Strategy 8 equity curve
        plot_strategy8_equity_curve(results)
        
        # Compare all strategies using caching
        # comparison = compare_all_trend_strategies()
        
        print(f"\nStrategy 8 backtest completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error in Strategy 8 backtest: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
