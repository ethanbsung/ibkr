from chapter8 import *
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

#####   STRATEGY 9: MULTIPLE TREND FOLLOWING RULES   #####

def get_trend_filter_configs():
    """
    Get trend filter configurations from the book.
    
    From book: Uses multiple EWMAC variations with different speeds.
    Table 29 shows forecast scalars for different speeds.
    
    Returns:
        dict: Dictionary of trend filter configurations.
    """
    return {
        'EWMAC2': {'fast_span': 2, 'slow_span': 8, 'forecast_scalar': 12.1},
        'EWMAC4': {'fast_span': 4, 'slow_span': 16, 'forecast_scalar': 8.53},
        'EWMAC8': {'fast_span': 8, 'slow_span': 32, 'forecast_scalar': 5.95},
        'EWMAC16': {'fast_span': 16, 'slow_span': 64, 'forecast_scalar': 4.10},
        'EWMAC32': {'fast_span': 32, 'slow_span': 128, 'forecast_scalar': 2.79},
        'EWMAC64': {'fast_span': 64, 'slow_span': 256, 'forecast_scalar': 1.91}
    }

def get_forecast_weights_and_fdm():
    """
    Get forecast weights and FDM values from the book's Table 36.
    
    From book: Different combinations of trend filters with their weights and FDM.
    
    Returns:
        dict: Dictionary of forecast weight configurations.
    """
    return {
        'six_filters': {
            'filters': ['EWMAC2', 'EWMAC4', 'EWMAC8', 'EWMAC16', 'EWMAC32', 'EWMAC64'],
            'weights': [0.167, 0.167, 0.167, 0.167, 0.167, 0.167],
            'fdm': 1.26
        },
        'five_filters': {
            'filters': ['EWMAC4', 'EWMAC8', 'EWMAC16', 'EWMAC32', 'EWMAC64'],
            'weights': [0.2, 0.2, 0.2, 0.2, 0.2],
            'fdm': 1.19
        },
        'four_filters': {
            'filters': ['EWMAC8', 'EWMAC16', 'EWMAC32', 'EWMAC64'],
            'weights': [0.25, 0.25, 0.25, 0.25],
            'fdm': 1.13
        },
        'three_filters': {
            'filters': ['EWMAC16', 'EWMAC32', 'EWMAC64'],
            'weights': [0.333, 0.333, 0.333],
            'fdm': 1.08
        },
        'two_filters': {
            'filters': ['EWMAC32', 'EWMAC64'],
            'weights': [0.50, 0.50],
            'fdm': 1.03
        }
    }

def calculate_multiple_trend_forecasts(prices: pd.Series, filter_config: dict, 
                                     forecast_config: dict, cap: float = 20.0,
                                     short_span: int = 32, long_years: int = 10, min_vol_floor: float = 0.05) -> pd.Series:
    """
    Calculate combined forecast from multiple trend filters.
    
    From book:
        1. Calculate individual forecasts for each filter
        2. Take weighted average: Raw combined forecast = w1×f1 + w2×f2 + ...
        3. Apply FDM: Scaled combined forecast = Raw combined forecast × FDM
        4. Cap result: Capped combined forecast = Max(Min(Scaled, +20), -20)
    
    Parameters:
        prices (pd.Series): Price series.
        filter_config (dict): Trend filter configurations.
        forecast_config (dict): Forecast weights and FDM configuration.
        cap (float): Maximum absolute forecast value.
        short_span (int): EWMA span for short-run volatility.
        long_years (int): Years for long-run volatility average.
        min_vol_floor (float): Minimum volatility floor.
    
    Returns:
        pd.Series: Combined capped forecast.
    """
    individual_forecasts = {}
    
    # Calculate individual forecasts for each filter
    for filter_name in forecast_config['filters']:
        if filter_name in filter_config:
            config = filter_config[filter_name]
            
            # Calculate raw forecast using the standardized method
            raw_forecast = calculate_fast_raw_forecast(
                prices, 
                config['fast_span'], 
                config['slow_span'],
                short_span,
                long_years,
                min_vol_floor
            )
            
            # Scale and cap individual forecast
            scaled_forecast = raw_forecast * config['forecast_scalar']
            capped_forecast = np.clip(scaled_forecast, -cap, cap)
            
            individual_forecasts[filter_name] = capped_forecast
    
    # Combine forecasts using weights
    combined_forecast = pd.Series(0.0, index=prices.index)
    
    for i, filter_name in enumerate(forecast_config['filters']):
        if filter_name in individual_forecasts:
            weight = forecast_config['weights'][i]
            combined_forecast += weight * individual_forecasts[filter_name]
    
    # Apply forecast diversification multiplier (FDM)
    scaled_combined_forecast = combined_forecast * forecast_config['fdm']
    
    # Cap the final combined forecast
    capped_combined_forecast = np.clip(scaled_combined_forecast, -cap, cap)
    
    return capped_combined_forecast

def calculate_strategy9_position_size(symbol, capital, weight, idm, price, volatility, 
                                    multiplier, combined_forecast, risk_target=0.2, fx_rate=1.0):
    """
    Calculate position size for Strategy 9 with combined forecast scaling.
    
    From book: Same position sizing as Strategy 8 but uses combined forecast
        N = Combined forecast × Capital × IDM × Weight × τ ÷ (10 × Multiplier × Price × FX × σ%)
    
    Parameters:
        symbol (str): Instrument symbol.
        capital (float): Total portfolio capital.
        weight (float): Weight allocated to this instrument.
        idm (float): Instrument Diversification Multiplier.
        price (float): Current price.
        volatility (float): Annualized volatility forecast.
        multiplier (float): Contract multiplier.
        combined_forecast (float): Combined capped forecast value.
        risk_target (float): Target risk fraction.
        fx_rate (float): FX rate for currency conversion.
    
    Returns:
        float: Number of contracts for this instrument.
    """
    if np.isnan(volatility) or volatility <= 0 or np.isnan(combined_forecast):
        return 0
    
    # Calculate position size with combined forecast scaling
    numerator = combined_forecast * capital * idm * weight * risk_target
    denominator = 10 * multiplier * price * fx_rate * volatility
    
    position_size = numerator / denominator
    
    # Protect against infinite or extremely large position sizes
    if np.isinf(position_size) or abs(position_size) > 100000:
        return 0
    
    return position_size

def backtest_multiple_trend_strategy(data_dir='Data', capital=50000000, risk_target=0.2,
                                   short_span=32, long_years=10, min_vol_floor=0.05,
                                   forecast_combination='five_filters',
                                   buffer_fraction=0.1,
                                   weight_method='handcrafted',
                                   common_hypothetical_SR=0.3, annual_turnover_T=7.0,
                                   start_date=None, end_date=None,
                                   debug_forecasts=False):
    """
    Backtest Strategy 9: Multiple trend following rules with buffering.
    
    Implementation follows book exactly: "Trade a portfolio of one or more instruments, 
    each with positions scaled for a variable risk estimate. Calculate a number of 
    forecasts for different speeds of trend filter. Place a position based on the 
    combined forecast."
    
    Parameters:
        data_dir (str): Directory containing price data files.
        capital (float): Initial capital.
        risk_target (float): Target risk fraction.
        short_span (int): EWMA span for short-run volatility.
        long_years (int): Years for long-run volatility average.
        min_vol_floor (float): Minimum volatility floor.
        forecast_combination (str): Which forecast combination to use.
        buffer_fraction (float): Buffer fraction for trading.
        weight_method (str): Method for calculating instrument weights.
        common_hypothetical_SR (float): Common hypothetical Sharpe Ratio.
        annual_turnover_T (float): Annual turnover T.
        start_date (str): Start date for backtest (YYYY-MM-DD).
        end_date (str): End date for backtest (YYYY-MM-DD).
        debug_forecasts (bool): Whether to print forecast debug info.
    
    Returns:
        dict: Comprehensive backtest results.
    """
    print("=" * 60)
    print("STRATEGY 9: MULTIPLE TREND FOLLOWING RULES")
    print("=" * 60)
    
    # Load all instrument data using the same function as chapter 4-8
    all_instruments_specs_df = load_instrument_data()
    raw_instrument_data = load_all_instrument_data(data_dir)
    
    if not raw_instrument_data:
        raise ValueError("No instrument data loaded successfully")
    
    # Get trend filter and forecast configurations
    filter_config = get_trend_filter_configs()
    forecast_configs = get_forecast_weights_and_fdm()
    selected_config = forecast_configs[forecast_combination]
    
    print(f"\nPortfolio Configuration:")
    print(f"  Instruments initially loaded: {len(raw_instrument_data)}")
    print(f"  Capital: ${capital:,.0f}")
    print(f"  Risk Target: {risk_target:.1%}")
    print(f"  Weight Method: {weight_method}")
    print(f"  Forecast Combination: {forecast_combination}")
    print(f"  Trend Filters: {', '.join(selected_config['filters'])}")
    print(f"  Forecast Weights: {selected_config['weights']}")
    print(f"  FDM: {selected_config['fdm']}")
    print(f"  Buffer Fraction: {buffer_fraction}")
    print(f"  Common Hypothetical SR for SR': {common_hypothetical_SR}")
    print(f"  Annual Turnover T for SR': {annual_turnover_T}")

    # Preprocess: Calculate returns, vol forecasts, and combined trend forecasts for each instrument
    processed_instrument_data = {}
    for symbol, df_orig in raw_instrument_data.items():
        df = df_orig.copy()
        if 'Last' not in df.columns:
            print(f"Skipping {symbol}: 'Last' column missing.")
            continue
        
        df['daily_price_change_pct'] = df['Last'].pct_change()
        
        # Volatility forecast for day D is made using data up to D-1 (no lookahead bias)
        raw_returns_for_vol = df['daily_price_change_pct'].dropna()
        if len(raw_returns_for_vol) < 300:  # Need sufficient data for slowest trend filter (256 + buffer)
            print(f"Skipping {symbol}: Insufficient data for vol forecast and trend ({len(raw_returns_for_vol)} days).")
            continue

        # Calculate blended volatility (same as Strategy 4-8)
        blended_vol_series = calculate_blended_volatility(
            raw_returns_for_vol, short_span=short_span, long_years=long_years, min_vol_floor=min_vol_floor
        )
        # Shift to prevent lookahead bias - forecast for day T uses data up to T-1
        df['vol_forecast'] = blended_vol_series.shift(1).reindex(df.index).ffill().fillna(min_vol_floor)
        
        # Calculate combined forecast using multiple trend filters (no lookahead bias)
        combined_forecast_series = calculate_multiple_trend_forecasts(
            df['Last'], filter_config, selected_config, 20.0, short_span, long_years, min_vol_floor
        )
        # Shift to prevent lookahead bias - forecast for day T uses data up to T-1
        df['combined_forecast'] = combined_forecast_series.shift(1).reindex(df.index).fillna(0)
        
        # Debug first instrument forecasts if requested
        if debug_forecasts and symbol == list(raw_instrument_data.keys())[0]:
            print(f"\n=== FORECAST DEBUG FOR {symbol} ===")
            sample_forecasts = df['combined_forecast'].dropna()[:10]
            if len(sample_forecasts) > 0:
                print(f"Sample Combined Forecasts: {sample_forecasts.values}")
                print(f"Average Combined Forecast: {df['combined_forecast'].mean():.3f}")
                print(f"Average Absolute Forecast: {df['combined_forecast'].abs().mean():.3f}")
        
        # Ensure critical data is present
        df.dropna(subset=['Last', 'vol_forecast', 'daily_price_change_pct'], inplace=True)
        if df.empty:
            print(f"Skipping {symbol}: Empty after dropping NaNs in critical columns.")
            continue

        processed_instrument_data[symbol] = df

    if not processed_instrument_data:
        raise ValueError("No instruments remaining after preprocessing and volatility calculation.")
    
    print(f"  Instruments after preprocessing: {len(processed_instrument_data)}")

    # Determine common date range for backtest (same logic as chapter 4-8)
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

    # Initialize portfolio tracking (same structure as chapter 4-8)
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
            if not df_upto_cutoff.empty and len(df_upto_cutoff) > 300:  # Need sufficient data for slowest filter
                current_iteration_eligible_instruments.add(s)
        
        # Check if reweighting is needed (same logic as chapter 4-8)
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
                    forecast_for_sizing = df_instrument.loc[current_date, 'combined_forecast']
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
                        
                        # Calculate optimal position size with combined forecast scaling
                        optimal_position = calculate_strategy9_position_size(
                            symbol=symbol, capital=capital_at_start_of_day, weight=instrument_weight, 
                            idm=idm, price=price_for_sizing, volatility=vol_for_sizing, 
                            multiplier=instrument_multiplier, combined_forecast=forecast_for_sizing, 
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
                        instrument_pnl_today = current_pos * instrument_multiplier * (price_at_end_of_trading - price_at_start_of_trading)
                        
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

        # Update portfolio equity (same as chapter 4-8)
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
                    sig = processed_instrument_data[s_proc].loc[current_date, 'combined_forecast']
                    if pd.notna(sig):
                        forecast_val_fill = sig
                record[f'{s_proc}_forecast'] = forecast_val_fill
            if f'{s_proc}_trades' not in record:
                record[f'{s_proc}_trades'] = 0
                
        portfolio_daily_records.append(record)

    # Post-loop processing (same as chapter 4-8)
    if not portfolio_daily_records:
        raise ValueError("No daily records generated during backtest.")
        
    portfolio_df = pd.DataFrame(portfolio_daily_records)
    portfolio_df.set_index('date', inplace=True)
    
    print(f"Portfolio backtest loop completed. {len(portfolio_df)} daily records.")
    if portfolio_df.empty or 'portfolio_return' not in portfolio_df.columns or portfolio_df['portfolio_return'].std() == 0:
        print(f"Average active instruments: {portfolio_df['num_active_instruments'].mean():.1f}")
        print(f"Average combined forecast: {portfolio_df['avg_forecast'].mean():.2f}")
        print(f"Average absolute forecast: {portfolio_df['avg_abs_forecast'].mean():.2f}")
        print(f"Average daily trades (events): {portfolio_df['total_trades'].mean():.1f}")
    
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
    performance['forecast_combination'] = forecast_combination
    performance['selected_filters'] = selected_config['filters']
    performance['forecast_weights'] = selected_config['weights']
    performance['fdm'] = selected_config['fdm']
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
            'forecast_combination': forecast_combination,
            'selected_filters': selected_config['filters'],
            'forecast_weights': selected_config['weights'],
            'fdm': selected_config['fdm'],
            'buffer_fraction': buffer_fraction,
            'weight_method': weight_method,
            'common_hypothetical_SR': common_hypothetical_SR,
            'annual_turnover_T': annual_turnover_T,
            'backtest_start': trading_days_range.min(),
            'backtest_end': trading_days_range.max()
        }
    }

def plot_equity_curves(strategy_results_dict, save_path=None, figsize=(15, 10)):
    """
    Plot equity curves for multiple strategies comparison.
    
    Parameters:
        strategy_results_dict (dict): Dictionary of strategy results.
        save_path (str): Optional path to save the plot.
        figsize (tuple): Figure size.
    
    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Strategy Performance Comparison', fontsize=16, fontweight='bold')
    
    # Colors for different strategies
    colors = {
        'strategy4': '#1f77b4',  # Blue
        'strategy5': '#ff7f0e',  # Orange
        'strategy6': '#2ca02c',  # Green
        'strategy7': '#d62728',  # Red
        'strategy8': '#9467bd',  # Purple
        'strategy9': '#8c564b',  # Brown
    }
    
    strategy_names = {
        'strategy4': 'Strategy 4: Always Long',
        'strategy5': 'Strategy 5: Long/Flat',
        'strategy6': 'Strategy 6: Long/Short',
        'strategy7': 'Strategy 7: Slow Forecasts',
        'strategy8': 'Strategy 8: Fast + Buffering',
        'strategy9': 'Strategy 9: Multiple Trends'
    }
    
    # Plot 1: Equity Curves
    for strategy_name, results in strategy_results_dict.items():
        if 'portfolio_data' in results:
            portfolio_data = results['portfolio_data']
            equity_curve = build_account_curve(portfolio_data['portfolio_return'], 100)
            
            ax1.plot(equity_curve.index, equity_curve.values, 
                    label=strategy_names.get(strategy_name, strategy_name), 
                    color=colors.get(strategy_name, 'black'), linewidth=2)
    
    ax1.set_title('Cumulative Equity Curves', fontweight='bold')
    ax1.set_ylabel('Portfolio Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Plot 2: Annual Returns
    annual_returns = []
    strategy_labels = []
    for strategy_name, results in strategy_results_dict.items():
        if 'performance' in results:
            annual_returns.append(results['performance']['annualized_return'] * 100)
            strategy_labels.append(strategy_names.get(strategy_name, strategy_name).replace('Strategy ', 'S'))
    
    bars = ax2.bar(strategy_labels, annual_returns, color=[colors.get(f'strategy{i+4}', 'gray') for i in range(len(annual_returns))])
    ax2.set_title('Annualized Returns (%)', fontweight='bold')
    ax2.set_ylabel('Return (%)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, annual_returns):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Sharpe Ratios
    sharpe_ratios = []
    for strategy_name, results in strategy_results_dict.items():
        if 'performance' in results:
            sharpe_ratios.append(results['performance']['sharpe_ratio'])
    
    bars = ax3.bar(strategy_labels, sharpe_ratios, color=[colors.get(f'strategy{i+4}', 'gray') for i in range(len(sharpe_ratios))])
    ax3.set_title('Sharpe Ratios', fontweight='bold')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, sharpe_ratios):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Max Drawdowns
    max_drawdowns = []
    for strategy_name, results in strategy_results_dict.items():
        if 'performance' in results:
            max_drawdowns.append(abs(results['performance']['max_drawdown_pct']))
    
    bars = ax4.bar(strategy_labels, max_drawdowns, color=[colors.get(f'strategy{i+4}', 'gray') for i in range(len(max_drawdowns))])
    ax4.set_title('Maximum Drawdowns (%)', fontweight='bold')
    ax4.set_ylabel('Max Drawdown (%)')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, max_drawdowns):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Equity curves plot saved to: {save_path}")
    
    return fig

def analyze_multiple_trend_results(results):
    """
    Analyze and display comprehensive multiple trend following results.
    
    Parameters:
        results (dict): Results from backtest_multiple_trend_strategy.
    """
    performance = results['performance']
    instrument_stats = results['instrument_stats']
    config = results['config']
    
    print("\n" + "=" * 60)
    print("MULTIPLE TREND FOLLOWING PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Overall performance
    print(f"\n--- Overall Portfolio Performance ---")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Annualized Return: {performance['annualized_return']:.2%}")
    print(f"Annualized Volatility: {performance['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {performance['max_drawdown_pct']:.1f}%")
    print(f"Skewness: {performance['skewness']:.3f}")
    
    # Multiple trend characteristics
    print(f"\n--- Multiple Trend Characteristics ---")
    print(f"Average Active Instruments: {performance['avg_active_instruments']:.1f}")
    print(f"Average Combined Forecast: {performance['avg_forecast']:.2f}")
    print(f"Average Absolute Forecast: {performance['avg_abs_forecast']:.2f}")
    print(f"Total Trades: {performance['total_trades']:,}")
    print(f"Average Daily Trades: {performance['avg_daily_trades']:.1f}")
    print(f"Forecast Combination: {config['forecast_combination']}")
    print(f"Trend Filters: {', '.join(config['selected_filters'])}")
    print(f"Forecast Weights: {config['forecast_weights']}")
    print(f"FDM: {config['fdm']}")
    print(f"Buffer Fraction: {config['buffer_fraction']}")
    
    # Portfolio characteristics
    print(f"\n--- Portfolio Characteristics ---")
    print(f"Number of Instruments: {performance['num_instruments']}")
    print(f"IDM: {performance['idm']:.2f}")
    print(f"Capital: ${config['capital']:,.0f}")
    print(f"Risk Target: {config['risk_target']:.1%}")
    print(f"Backtest Period: {config['backtest_start'].date()} to {config['backtest_end'].date()}")
    
    # Top performing instruments
    print(f"\n--- Top 10 Performing Instruments (by Weight and Activity) ---")
    sorted_instruments = sorted(
        instrument_stats.items(), 
        key=lambda x: x[1]['weight'], 
        reverse=True
    )
    
    print(f"{'Symbol':<8} {'Weight':<8} {'Avg Pos':<10} {'AvgFcst':<8} {'Trades':<8} {'Days':<6}")
    print("-" * 70)
    
    for symbol, stats in sorted_instruments[:10]:
        print(f"{symbol:<8} {stats['weight']:<8.3f} {stats['avg_position']:<10.2f} "
              f"{stats['avg_forecast']:<8.2f} {stats['total_trades']:<8} {stats['active_days']:<6}")
    
    # Show instruments with highest forecast activity
    print(f"\n--- Top 10 Most Active Multiple Trend Instruments (by Days Active) ---")
    sorted_by_activity = sorted(
        instrument_stats.items(), 
        key=lambda x: x[1]['active_days'], 
        reverse=True
    )
    
    print(f"{'Symbol':<8} {'Days':<6} {'AvgFcst':<8} {'AbsFcst':<8} {'Trades':<8} {'Weight':<8} {'Avg Pos':<10}")
    print("-" * 80)
    
    for symbol, stats in sorted_by_activity[:10]:
        print(f"{symbol:<8} {stats['active_days']:<6} {stats['avg_forecast']:<8.2f} "
              f"{stats['avg_abs_forecast']:<8.2f} {stats['total_trades']:<8} {stats['weight']:<8.3f} {stats['avg_position']:<10.2f}")
    
    # Summary of multiple trend characteristics
    total_active_days = sum(stats['active_days'] for stats in instrument_stats.values())
    avg_forecast_all = sum(stats['avg_forecast'] for stats in instrument_stats.values()) / len(instrument_stats)
    avg_abs_forecast_all = sum(stats['avg_abs_forecast'] for stats in instrument_stats.values()) / len(instrument_stats)
    total_trades_all = sum(stats['total_trades'] for stats in instrument_stats.values())
    
    print(f"\n--- Multiple Trend Summary ---")
    print(f"Total instrument-days with positions: {total_active_days:,}")
    print(f"Average combined forecast across all instruments: {avg_forecast_all:.2f}")
    print(f"Average absolute forecast across all instruments: {avg_abs_forecast_all:.2f}")
    print(f"Total individual instrument trades: {total_trades_all:,}")
    print(f"Instruments with any activity: {len(instrument_stats)}")

def compare_all_strategies():
    """
    Compare all strategies (4 through 9) using cached results where possible.
    """
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON: STRATEGY 4 vs 5 vs 6 vs 7 vs 8 vs 9")
    print("=" * 80)
    
    # Standard config for all strategies
    standard_config = {
        'capital': 50000000,
        'risk_target': 0.2,
        'weight_method': 'handcrafted'
    }
    
    # Get cached results for strategies 4-8
    cached_results = get_cached_strategy_results()
    
    strategy_results = {}
    
    # Strategy 4 (no trend filter)
    if 'strategy4' in cached_results:
        print("Using cached Strategy 4 results...")
        strategy_results['strategy4'] = cached_results['strategy4']
    else:
        print("Running Strategy 4 (no trend filter)...")
        strategy4_results = backtest_multi_instrument_strategy(
            data_dir='Data', **standard_config
        )
        save_strategy_results('strategy4', strategy4_results, standard_config)
        strategy_results['strategy4'] = strategy4_results
    
    # Strategy 5 (with trend filter, long only)
    if 'strategy5' in cached_results:
        print("Using cached Strategy 5 results...")
        strategy_results['strategy5'] = cached_results['strategy5']
    else:
        print("Running Strategy 5 (trend filter, long only)...")
        strategy5_results = backtest_trend_following_strategy(
            data_dir='Data', **standard_config
        )
        save_strategy_results('strategy5', strategy5_results, standard_config)
        strategy_results['strategy5'] = strategy5_results
    
    # Strategy 6 (with trend filter, long/short)
    if 'strategy6' in cached_results:
        print("Using cached Strategy 6 results...")
        strategy_results['strategy6'] = cached_results['strategy6']
    else:
        print("Running Strategy 6 (trend filter, long/short)...")
        strategy6_results = backtest_long_short_trend_strategy(
            data_dir='Data', **standard_config
        )
        save_strategy_results('strategy6', strategy6_results, standard_config)
        strategy_results['strategy6'] = strategy6_results
    
    # Strategy 7 (with forecasts)
    if 'strategy7' in cached_results:
        print("Using cached Strategy 7 results...")
        strategy_results['strategy7'] = cached_results['strategy7']
    else:
        print("Running Strategy 7 (trend filter with forecasts)...")
        strategy7_results = backtest_forecast_trend_strategy(
            data_dir='Data', **standard_config
        )
        save_strategy_results('strategy7', strategy7_results, standard_config)
        strategy_results['strategy7'] = strategy7_results
    
    # Strategy 8 (fast trend with buffering)
    print("Running Strategy 8 (fast trend with buffering)...")
    strategy8_config = {**standard_config}
    strategy8_results = backtest_fast_trend_strategy_with_buffering(
        data_dir='Data', debug_buffering=False, **strategy8_config
    )
    save_strategy_results('strategy8', strategy8_results, strategy8_config)
    strategy_results['strategy8'] = strategy8_results
    
    # Strategy 9 (multiple trend with buffering) - always run fresh
    print("Running Strategy 9 (multiple trend with buffering)...")
    strategy9_config = {**standard_config, 'forecast_combination': 'five_filters'}
    strategy9_results = backtest_multiple_trend_strategy(
        data_dir='Data', debug_forecasts=False, **strategy9_config
    )
    save_strategy_results('strategy9', strategy9_results, strategy9_config)
    strategy_results['strategy9'] = strategy9_results
    
    # Performance comparison table
    if all(strategy_results.values()):
        performances = {name: results['performance'] for name, results in strategy_results.items()}
        
        print(f"\n{'Strategy':<15} {'Ann. Return':<12} {'Volatility':<12} {'Sharpe':<8} {'Max DD':<8} {'Trades/Day':<12} {'Special':<25}")
        print("-" * 115)
        
        strategy_descriptions = {
            'strategy4': 'Always Long',
            'strategy5': 'Long/Flat',
            'strategy6': 'Long/Short',
            'strategy7': 'Slow Forecasts',
            'strategy8': 'Fast + Buffering',
            'strategy9': 'Multiple Trends'
        }
        
        for strategy_name, perf in performances.items():
            trades_day = perf.get('avg_daily_trades', 'N/A')
            trades_str = f"{trades_day:.1f}" if trades_day != 'N/A' else 'N/A'
            
            print(f"{strategy_name.replace('strategy', 'Strategy '):<15} {perf['annualized_return']:<12.2%} "
                  f"{perf['annualized_volatility']:<12.2%} "
                  f"{perf['sharpe_ratio']:<8.3f} "
                  f"{perf['max_drawdown_pct']:<8.1f}% "
                  f"{trades_str:<12} "
                  f"{strategy_descriptions[strategy_name]:<25}")
        
        print(f"\n--- Strategy 9 vs Strategy 8 Analysis ---")
        s8_perf = performances['strategy8']
        s9_perf = performances['strategy9']
        
        return_diff = s9_perf['annualized_return'] - s8_perf['annualized_return']
        vol_diff = s9_perf['annualized_volatility'] - s8_perf['annualized_volatility']
        sharpe_diff = s9_perf['sharpe_ratio'] - s8_perf['sharpe_ratio']
        dd_diff = s9_perf['max_drawdown_pct'] - s8_perf['max_drawdown_pct']
        
        print(f"Return Difference: {return_diff:+.2%}")
        print(f"Volatility Difference: {vol_diff:+.2%}")
        print(f"Sharpe Difference: {sharpe_diff:+.3f}")
        print(f"Max Drawdown Difference: {dd_diff:+.1f}%")
        
        if 'avg_forecast' in s9_perf:
            print(f"\nStrategy 9 Characteristics:")
            print(f"  Average Combined Forecast: {s9_perf['avg_forecast']:.2f}")
            print(f"  Average Absolute Forecast: {s9_perf['avg_abs_forecast']:.2f}")
            print(f"  Average Daily Trades: {s9_perf['avg_daily_trades']:.1f}")
        
        # Create and display equity curves plot
        print(f"\n--- Generating Equity Curves Plot ---")
        try:
            fig = plot_equity_curves(strategy_results, save_path='results/equity_curves_comparison.png')
            plt.show()
        except Exception as e:
            print(f"Error creating plot: {e}")
        
        return strategy_results
    

def plot_strategy9_equity_curve(results, save_path='results/strategy9_equity_curve.png'):
    """
    Plot Strategy 9 equity curve and save to file.
    
    Parameters:
        results (dict): Results from backtest_multiple_trend_strategy.
        save_path (str): Path to save the plot.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        portfolio_df = results['portfolio_data']
        config = results['config']
        performance = results['performance']
        
        equity_curve = build_account_curve(portfolio_df['portfolio_return'], config['capital'])
        
        plt.figure(figsize=(15, 10))
        
        # Main equity curve
        plt.subplot(3, 1, 1)
        plt.plot(equity_curve.index, equity_curve.values/1e6, 'darkblue', linewidth=2, 
                label=f'Strategy 9: Multiple Trend Following (SR: {performance["sharpe_ratio"]:.3f})')
        plt.title('Strategy 9: Multiple Trend Following Equity Curve', fontsize=14, fontweight='bold')
        plt.ylabel('Portfolio Value ($M)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Drawdown
        plt.subplot(3, 1, 2)
        drawdown_stats = calculate_maximum_drawdown(equity_curve)
        drawdown_series = drawdown_stats['drawdown_series'] * 100
        
        plt.fill_between(drawdown_series.index, drawdown_series.values, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        plt.plot(drawdown_series.index, drawdown_series.values, 'r-', linewidth=1)
        plt.title('Drawdown', fontsize=12, fontweight='bold')
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Combined forecast and trading activity over time
        plt.subplot(3, 1, 3)
        plt.plot(portfolio_df.index, portfolio_df['avg_forecast'], 'green', linewidth=1, 
                label='Average Combined Forecast')
        plt.plot(portfolio_df.index, portfolio_df['avg_abs_forecast'], 'orange', linewidth=1, 
                label='Average Absolute Forecast')
        plt.plot(portfolio_df.index, portfolio_df['total_trades'], 'purple', linewidth=1, alpha=0.7,
                label='Daily Trades')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.title('Combined Forecast & Trading Activity Over Time', fontsize=12, fontweight='bold')
        plt.ylabel('Forecast Value / Trade Count', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis dates for all subplots
        for ax in plt.gcf().get_axes():
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Performance summary text
        textstr = f'''Strategy 9 Performance Summary:
Total Return: {performance['total_return']:.1%}
Annualized Return: {performance['annualized_return']:.1%}
Volatility: {performance['annualized_volatility']:.1%}
Sharpe Ratio: {performance['sharpe_ratio']:.3f}
Max Drawdown: {performance['max_drawdown_pct']:.1f}%
Instruments: {performance.get('num_instruments', 'N/A')}
Average Combined Forecast: {performance.get('avg_forecast', 0):.2f}
Total Trades: {performance.get('total_trades', 0):,}
Forecast Combination: {config.get('forecast_combination', 'N/A')}
Period: {config['backtest_start'].strftime('%Y-%m-%d')} to {config['backtest_end'].strftime('%Y-%m-%d')}'''
        
        plt.figtext(0.02, 0.02, textstr, fontsize=9, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.35)  # Make room for performance text
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Strategy 9 equity curve saved to: {save_path}")
        
    except Exception as e:
        print(f"Error plotting Strategy 9 equity curve: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Test Strategy 9 implementation and compare all strategies.
    """
    print("=" * 60)
    print("TESTING STRATEGY 9: MULTIPLE TREND FOLLOWING RULES")
    print("=" * 60)
    
    try:
        # Run Strategy 9 backtest
        results = backtest_multiple_trend_strategy(
            data_dir='Data',
            capital=1000000,
            risk_target=0.2,
            short_span=32,
            long_years=10,
            min_vol_floor=0.05,
            forecast_combination='five_filters',
            buffer_fraction=0.1,
            weight_method='handcrafted',
            common_hypothetical_SR=0.3,
            annual_turnover_T=7.0,
            debug_forecasts=False
        )
        
        # Analyze results
        analyze_multiple_trend_results(results)
        
        # Plot Strategy 9 equity curve
        plot_strategy9_equity_curve(results)
        
        print(f"\nStrategy 9 backtest completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error in Strategy 9 testing: {e}")
        import traceback
        traceback.print_exc()
        return None



if __name__ == "__main__":
    main() 