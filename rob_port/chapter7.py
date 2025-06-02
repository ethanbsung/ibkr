from chapter6 import *
from chapter5 import *
from chapter4 import *
from chapter3 import *
from chapter2 import *
from chapter1 import *
# Import FX functions from chapter 4
from chapter4 import load_fx_data, get_instrument_currency_mapping, get_fx_rate_for_date_and_currency
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')

#####   STRATEGY 7: SLOW TREND FOLLOWING WITH TREND STRENGTH   #####

def calculate_raw_forecast(prices: pd.Series, fast_span: int = 64, slow_span: int = 256,
                          short_span: int = 32, long_years: int = 10, min_vol_floor: float = 0.05) -> pd.Series:
    """
    Calculate raw forecast for EWMAC trend following.
    
    From book:
        Raw forecast = (Fast EWMA - Slow EWMA) ÷ σp
        where σp = Price × σ% ÷ 16 (daily price volatility)
    
    Parameters:
        prices (pd.Series): Price series.
        fast_span (int): Fast EWMA span (default 64).
        slow_span (int): Slow EWMA span (default 256).
        short_span (int): EWMA span for short-run volatility.
        long_years (int): Years for long-run volatility average.
        min_vol_floor (float): Minimum volatility floor.
    
    Returns:
        pd.Series: Raw forecast values.
    """
    # Calculate EWMA crossover (same as before)
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
                                    forecast_scalar: float = 1.9, cap: float = 20.0,
                                    short_span: int = 32, long_years: int = 10, min_vol_floor: float = 0.05) -> pd.Series:
    """
    Calculate complete forecast pipeline for an instrument.
    
    Parameters:
        prices (pd.Series): Price series.
        fast_span (int): Fast EWMA span.
        slow_span (int): Slow EWMA span.
        forecast_scalar (float): Forecast scalar.
        cap (float): Maximum absolute forecast value.
        short_span (int): EWMA span for short-run volatility.
        long_years (int): Years for long-run volatility average.
        min_vol_floor (float): Minimum volatility floor.
    
    Returns:
        pd.Series: Capped forecast values.
    """
    raw_forecast = calculate_raw_forecast(prices, fast_span, slow_span, short_span, long_years, min_vol_floor)
    scaled_forecast = calculate_scaled_forecast(raw_forecast, forecast_scalar)
    capped_forecast = calculate_capped_forecast(scaled_forecast, cap)
    
    return capped_forecast

# Note: calculate_strategy7_position_size function removed - now using new approach:
# 1. Calculate base position using calculate_portfolio_position_size  
# 2. Apply forecast as multiplier: forecast * base_position / 10

def get_forecast_scalar_for_ewmac(fast_span: int) -> float:
    """
    Get forecast scalar for EWMAC based on fast span.
    
    From author's code:
        scalar_dict = {64: 1.91, 32: 2.79, 16: 4.1, 8: 5.95, 4: 8.53, 2: 12.1}
    
    Parameters:
        fast_span (int): Fast EWMA span.
    
    Returns:
        float: Forecast scalar for this span.
    """
    scalar_dict = {64: 1.91, 32: 2.79, 16: 4.1, 8: 5.95, 4: 8.53, 2: 12.1}
    return scalar_dict.get(fast_span, 1.91)  # Default to 64-span scalar

def calculate_risk_adjusted_forecast_for_ewmac(prices: pd.Series, volatility_series: pd.Series, 
                                             fast_span: int = 64, slow_span: int = 256) -> pd.Series:
    """
    Calculate risk-adjusted forecast for EWMAC.
    
    From author's code:
        ewmac_values = ewmac(adjusted_price, fast_span=fast_span, slow_span=fast_span * 4)
        daily_price_vol = stdev_ann_perc.daily_risk_price_terms()
        risk_adjusted_ewmac = ewmac_values / daily_price_vol
    
    Parameters:
        prices (pd.Series): Price series.
        volatility_series (pd.Series): Daily price volatility series.
        fast_span (int): Fast EWMA span.
        slow_span (int): Slow EWMA span.
    
    Returns:
        pd.Series: Risk-adjusted forecast values.
    """
    # Calculate EWMAC 
    ewmac_values = calculate_ewma_trend(prices, fast_span, slow_span)
    
    # Convert percentage volatility to daily price volatility
    # daily_price_vol = price * vol_percentage / 16
    daily_price_vol = prices * volatility_series / 16
    
    # Reindex to match EWMAC
    daily_price_vol = daily_price_vol.reindex(ewmac_values.index, method='ffill')
    
    # Calculate risk-adjusted forecast
    risk_adjusted_forecast = ewmac_values / daily_price_vol
    
    # Handle division by zero or very small volatility
    risk_adjusted_forecast = risk_adjusted_forecast.replace([np.inf, -np.inf], 0)
    risk_adjusted_forecast = risk_adjusted_forecast.fillna(0)
    
    return risk_adjusted_forecast

def calculate_scaled_forecast_for_ewmac(prices: pd.Series, volatility_series: pd.Series,
                                      fast_span: int = 64, slow_span: int = 256) -> pd.Series:
    """
    Calculate scaled forecast for EWMAC.
    
    From author's code:
        risk_adjusted_ewmac = calculate_risk_adjusted_forecast_for_ewmac(...)
        forecast_scalar = scalar_dict[fast_span]
        scaled_ewmac = risk_adjusted_ewmac * forecast_scalar
    
    Parameters:
        prices (pd.Series): Price series.
        volatility_series (pd.Series): Daily volatility series.
        fast_span (int): Fast EWMA span.
        slow_span (int): Slow EWMA span.
    
    Returns:
        pd.Series: Scaled forecast values.
    """
    risk_adjusted_forecast = calculate_risk_adjusted_forecast_for_ewmac(prices, volatility_series, fast_span, slow_span)
    forecast_scalar = get_forecast_scalar_for_ewmac(fast_span)
    scaled_forecast = risk_adjusted_forecast * forecast_scalar
    
    return scaled_forecast

def calculate_forecast_for_ewmac(prices: pd.Series, volatility_series: pd.Series,
                               fast_span: int = 64, slow_span: int = 256, cap: float = 20.0) -> pd.Series:
    """
    Calculate complete forecast for EWMAC.
    
    From author's code:
        scaled_ewmac = calculate_scaled_forecast_for_ewmac(...)
        capped_ewmac = scaled_ewmac.clip(-20, 20)
    
    Parameters:
        prices (pd.Series): Price series.
        volatility_series (pd.Series): Daily volatility series.
        fast_span (int): Fast EWMA span.
        slow_span (int): Slow EWMA span.
        cap (float): Maximum absolute forecast value.
    
    Returns:
        pd.Series: Capped forecast values.
    """
    scaled_forecast = calculate_scaled_forecast_for_ewmac(prices, volatility_series, fast_span, slow_span)
    capped_forecast = calculate_capped_forecast(scaled_forecast, cap)
    
    return capped_forecast

def apply_forecast_to_position(base_position: pd.Series, prices: pd.Series, volatility_series: pd.Series,
                             fast_span: int = 64, slow_span: int = 256, cap: float = 20.0) -> pd.Series:
    """
    Apply trend forecast to base position using author's approach.
    
    From author's code:
        forecast = calculate_forecast_for_ewmac(...)
        return forecast * average_position / 10
    
    Parameters:
        base_position (pd.Series): Base position size without forecast.
        prices (pd.Series): Price series for forecast calculation.
        volatility_series (pd.Series): Daily volatility series.
        fast_span (int): Fast EWMA span.
        slow_span (int): Slow EWMA span.
        cap (float): Maximum absolute forecast value.
    
    Returns:
        pd.Series: Position with forecast applied.
    """
    forecast = calculate_forecast_for_ewmac(prices, volatility_series, fast_span, slow_span, cap)
    
    # Apply forecast to base position with division by 10 (as per author's code)
    position_with_forecast = forecast * base_position / 10
    
    return position_with_forecast

def backtest_forecast_trend_strategy(data_dir='Data', capital=50000000, risk_target=0.2,
                                   short_span=32, long_years=10, min_vol_floor=0.05,
                                   trend_fast_span=64, trend_slow_span=256,
                                   forecast_scalar=1.9, forecast_cap=20.0,
                                   weight_method='handcrafted',
                                   common_hypothetical_SR=0.3, annual_turnover_T=7.0,
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
        min_vol_floor (float): Minimum volatility floor for trend filter.
        trend_fast_span (int): Fast EWMA span for trend filter.
        trend_slow_span (int): Slow EWMA span for trend filter.
        forecast_scalar (float): Forecast scaling factor (default 1.9).
        forecast_cap (float): Maximum absolute forecast value (default 20.0).
        weight_method (str): Method for calculating instrument weights.
        common_hypothetical_SR (float): Common hypothetical Sharpe Ratio.
        annual_turnover_T (float): Annual turnover T.
        start_date (str): Start date for backtest (YYYY-MM-DD).
        end_date (str): End date for backtest (YYYY-MM-DD).
    
    Returns:
        dict: Comprehensive backtest results.
    """
    print("=" * 60)
    print("STRATEGY 7: SLOW TREND FOLLOWING WITH TREND STRENGTH")
    print("=" * 60)
    
    # Load FX data
    print("\nLoading FX data...")
    fx_data = load_fx_data(data_dir)
    currency_mapping = get_instrument_currency_mapping()
    
    # Load all instrument data using the same function as chapter 4-6
    all_instruments_specs_df = load_instrument_data()
    raw_instrument_data = load_all_instrument_data(data_dir)
    
    if not raw_instrument_data:
        raise ValueError("No instrument data loaded successfully")
    
    print(f"\nPortfolio Configuration:")
    print(f"  Instruments initially loaded: {len(raw_instrument_data)}")
    print(f"  Capital: ${capital:,.0f}")
    print(f"  Risk Target: {risk_target:.1%}")
    print(f"  Weight Method: {weight_method}")
    print(f"  Trend Filter: EWMAC({trend_fast_span},{trend_slow_span}) with Forecasts")
    print(f"  Forecast Scalar: {forecast_scalar}")
    print(f"  Forecast Cap: ±{forecast_cap}")
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

        # Calculate blended volatility (same as Strategy 4-6)
        blended_vol_series = calculate_blended_volatility(
            raw_returns_for_vol, short_span=short_span, long_years=long_years, min_vol_floor=min_vol_floor
        )
        # Shift to prevent lookahead bias - forecast for day T uses data up to T-1
        df['vol_forecast'] = blended_vol_series.shift(1).reindex(df.index).ffill().fillna(min_vol_floor)
        
        # No pre-calculation of forecasts - will calculate on-the-fly during backtest to prevent lookahead bias
        
        # Ensure critical data is present
        df.dropna(subset=['Last', 'vol_forecast', 'daily_price_change_pct'], inplace=True)
        if df.empty:
            print(f"Skipping {symbol}: Empty after dropping NaNs in critical columns.")
            continue

        processed_instrument_data[symbol] = df

    if not processed_instrument_data:
        raise ValueError("No instruments remaining after preprocessing and volatility calculation.")
    
    print(f"  Instruments after preprocessing: {len(processed_instrument_data)}")

    # Determine common date range for backtest (same logic as chapter 4-6)
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

    # Initialize portfolio tracking (same structure as chapter 4-6)
    current_portfolio_equity = capital
    portfolio_daily_records = []
    known_eligible_instruments = set()
    weights = {} 
    idm = 1.0

    # Main time-stepping loop with daily position updates
    for idx, current_date in enumerate(trading_days_range):
        if idx == 0:
            # First day setup
            record = {'date': current_date, 'total_pnl': 0.0, 'portfolio_return': 0.0, 
                      'equity_sod': current_portfolio_equity, 'equity_eod': current_portfolio_equity,
                      'num_active_instruments': 0, 'avg_forecast': 0.0, 'avg_abs_forecast': 0.0}
            for symbol_k in processed_instrument_data.keys(): 
                record[f'{symbol_k}_contracts'] = 0.0
                record[f'{symbol_k}_forecast'] = 0.0
            portfolio_daily_records.append(record)
            continue
        
        previous_trading_date = trading_days_range[idx-1]
        capital_at_start_of_day = current_portfolio_equity
        daily_total_pnl = 0.0
        current_day_positions_and_forecasts = {}
        num_active_instruments = 0
        daily_forecasts = []

        effective_data_cutoff_date = previous_trading_date

        # Determine current period eligible instruments based on data up to cutoff
        current_iteration_eligible_instruments = set()
        for s, df_full in processed_instrument_data.items():
            df_upto_cutoff = df_full[df_full.index <= effective_data_cutoff_date]
            if not df_upto_cutoff.empty and len(df_upto_cutoff) > max(short_span, trend_slow_span):
                current_iteration_eligible_instruments.add(s)
        
        # Check if reweighting is needed (same logic as chapter 4-6)
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

        # Calculate positions and P&L for each instrument
        for symbol, df_instrument in processed_instrument_data.items():
            try:
                specs = get_instrument_specs(symbol, all_instruments_specs_df)
                instrument_multiplier = specs['multiplier']
            except:
                continue
                
            instrument_weight = weights.get(symbol, 0.0)
            num_contracts = 0.0
            instrument_pnl_today = 0.0
            actual_forecast_used = 0.0

            if instrument_weight > 1e-6:
                try:
                    # Sizing based on previous day's close price and current day's vol forecasts
                    price_for_sizing = df_instrument.loc[previous_trading_date, 'Last']
                    vol_for_sizing = df_instrument.loc[current_date, 'vol_forecast']
                    
                    # Data for P&L calculation for current_date
                    price_at_start_of_trading = df_instrument.loc[previous_trading_date, 'Last']
                    price_at_end_of_trading = df_instrument.loc[current_date, 'Last']
                    
                    if (pd.isna(price_for_sizing) or pd.isna(vol_for_sizing) or 
                        pd.isna(price_at_start_of_trading) or pd.isna(price_at_end_of_trading)):
                        num_contracts = 0.0
                        instrument_pnl_today = 0.0
                        actual_forecast_used = 0.0
                    else:
                        vol_for_sizing = max(vol_for_sizing, min_vol_floor)
                        
                        # Get FX rate for position sizing
                        instrument_currency = currency_mapping.get(symbol, 'USD')
                        fx_rate = get_fx_rate_for_date_and_currency(current_date, instrument_currency, fx_data)
                        
                        # Skip KRW instruments as requested
                        if fx_rate is None:
                            num_contracts = 0.0
                            instrument_pnl_today = 0.0
                            actual_forecast_used = 0.0
                        else:
                            # Step 1: Calculate base position (without forecast) using Strategy 4 logic
                            base_position = calculate_portfolio_position_size(
                                symbol=symbol, capital=capital_at_start_of_day, weight=instrument_weight, 
                                idm=idm, price=price_for_sizing, volatility=vol_for_sizing, 
                                multiplier=instrument_multiplier, risk_target=risk_target, fx_rate=fx_rate
                            )
                            
                            # Step 2: Calculate forecast for current date using data up to previous date (no lookahead bias)
                            # Get price and volatility data up to previous trading date to avoid lookahead bias
                            price_data_for_forecast = df_instrument[df_instrument.index <= previous_trading_date]['Last']
                            vol_data_for_forecast = df_instrument[df_instrument.index <= previous_trading_date]['vol_forecast']
                            
                            if len(price_data_for_forecast) >= max(trend_fast_span, trend_slow_span) and len(vol_data_for_forecast) > 0:
                                # Calculate forecast using historical data only (as per author's approach)
                                # Use the lookup table approach from the author
                                forecast_scalar_from_table = get_forecast_scalar_for_ewmac(trend_fast_span)
                                forecast_values = calculate_forecast_for_ewmac(
                                    price_data_for_forecast, vol_data_for_forecast, trend_fast_span, trend_slow_span, forecast_cap
                                )
                                current_forecast = forecast_values.iloc[-1] if not forecast_values.empty else 0.0
                                actual_forecast_used = current_forecast
                                
                                # Apply forecast to base position with division by 10 (as per author's code)
                                num_contracts = current_forecast * base_position / 10
                            else:
                                # Insufficient data for forecast calculation
                                num_contracts = 0.0
                                actual_forecast_used = 0.0
                            
                            # P&L calculation with FX rate to convert to base currency (USD)
                            price_change_in_local_currency = price_at_end_of_trading - price_at_start_of_trading
                            price_change_in_base_currency = price_change_in_local_currency * fx_rate
                            instrument_pnl_today = num_contracts * instrument_multiplier * price_change_in_base_currency
                        
                        # Count active instruments and collect forecasts
                        if abs(num_contracts) > 0.01:
                            num_active_instruments += 1
                        if actual_forecast_used is not None and not pd.isna(actual_forecast_used):
                            daily_forecasts.append(actual_forecast_used)
                
                except KeyError:  # Date not found for this instrument
                    num_contracts = 0.0
                    instrument_pnl_today = 0.0
                    actual_forecast_used = 0.0
            
            current_day_positions_and_forecasts[symbol] = {
                'contracts': num_contracts, 
                'forecast': actual_forecast_used
            }
            daily_total_pnl += instrument_pnl_today

        # Calculate daily forecast metrics
        avg_forecast = np.mean(daily_forecasts) if daily_forecasts else 0.0
        avg_abs_forecast = np.mean([abs(f) for f in daily_forecasts]) if daily_forecasts else 0.0

        # Update portfolio equity (same as chapter 4-6)
        portfolio_daily_percentage_return = daily_total_pnl / capital_at_start_of_day if capital_at_start_of_day > 0 else 0.0
        current_portfolio_equity = capital_at_start_of_day * (1 + portfolio_daily_percentage_return)

        # Record daily results
        record = {'date': current_date, 'total_pnl': daily_total_pnl, 
                  'portfolio_return': portfolio_daily_percentage_return, 
                  'equity_sod': capital_at_start_of_day, 
                  'equity_eod': current_portfolio_equity,
                  'num_active_instruments': num_active_instruments,
                  'avg_forecast': avg_forecast,
                  'avg_abs_forecast': avg_abs_forecast}
        
        for symbol_k, data_k in current_day_positions_and_forecasts.items(): 
            record[f'{symbol_k}_contracts'] = data_k['contracts']
            record[f'{symbol_k}_forecast'] = data_k['forecast']
        
        # Ensure all processed instruments have entries in the record
        for s_proc in processed_instrument_data.keys():
            if f'{s_proc}_contracts' not in record:
                record[f'{s_proc}_contracts'] = 0.0
            if f'{s_proc}_forecast' not in record:
                # Calculate forecast for this instrument if not already recorded
                try:
                    df_proc = processed_instrument_data[s_proc]
                    price_data_for_forecast = df_proc[df_proc.index <= previous_trading_date]['Last']
                    vol_data_for_forecast = df_proc[df_proc.index <= previous_trading_date]['vol_forecast']
                    if len(price_data_for_forecast) >= max(trend_fast_span, trend_slow_span) and len(vol_data_for_forecast) > 0:
                        forecast_values = calculate_forecast_for_ewmac(
                            price_data_for_forecast, vol_data_for_forecast, trend_fast_span, trend_slow_span, forecast_cap
                        )
                        forecast_val_fill = forecast_values.iloc[-1] if not forecast_values.empty else 0.0
                    else:
                        forecast_val_fill = 0.0  # No forecast due to insufficient data
                except:
                    forecast_val_fill = 0.0  # Default neutral forecast
                record[f'{s_proc}_forecast'] = forecast_val_fill
                
        portfolio_daily_records.append(record)

    # Post-loop processing (same as chapter 4-6)
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
    performance['weight_method'] = weight_method
    performance['backtest_start'] = trading_days_range.min()
    performance['backtest_end'] = trading_days_range.max()
    performance['trend_fast_span'] = trend_fast_span
    performance['trend_slow_span'] = trend_slow_span
    performance['forecast_scalar'] = forecast_scalar
    performance['forecast_cap'] = forecast_cap

    # Calculate per-instrument statistics (simplified for now)
    instrument_stats = {}
    for symbol in processed_instrument_data.keys():
        pos_col = f'{symbol}_contracts'
        forecast_col = f'{symbol}_forecast'
        
        if pos_col in portfolio_df.columns:
            # Calculate basic statistics for instruments with positions
            inst_positions = portfolio_df[pos_col][portfolio_df[pos_col] != 0]
            inst_forecasts = portfolio_df[forecast_col][portfolio_df[pos_col] != 0]
            
            if len(inst_positions) > 0:
                instrument_stats[symbol] = {
                    'avg_position': inst_positions.mean(),
                    'weight': weights.get(symbol, 0.0),
                    'active_days': len(inst_positions),
                    'avg_forecast': inst_forecasts.mean() if len(inst_forecasts) > 0 else 0.0,
                    'avg_abs_forecast': inst_forecasts.abs().mean() if len(inst_forecasts) > 0 else 0.0,
                    'max_forecast': inst_forecasts.max() if len(inst_forecasts) > 0 else 0.0,
                    'min_forecast': inst_forecasts.min() if len(inst_forecasts) > 0 else 0.0
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
            'weight_method': weight_method,
            'common_hypothetical_SR': common_hypothetical_SR,
            'annual_turnover_T': annual_turnover_T,
            'backtest_start': trading_days_range.min(),
            'backtest_end': trading_days_range.max()
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
    print(f"\n--- Top 10 Performing Instruments (by Weight and Activity) ---")
    sorted_instruments = sorted(
        instrument_stats.items(), 
        key=lambda x: x[1]['weight'], 
        reverse=True
    )
    
    print(f"{'Symbol':<8} {'Weight':<8} {'Avg Pos':<10} {'AvgFcst':<8} {'MaxFcst':<8} {'Days':<6}")
    print("-" * 70)
    
    for symbol, stats in sorted_instruments[:10]:
        print(f"{symbol:<8} {stats['weight']:<8.3f} {stats['avg_position']:<10.2f} "
              f"{stats['avg_forecast']:<8.2f} {stats['max_forecast']:<8.2f} {stats['active_days']:<6}")
    
    # Show instruments with highest forecast activity
    print(f"\n--- Top 10 Most Active Forecast Instruments (by Days Active) ---")
    sorted_by_activity = sorted(
        instrument_stats.items(), 
        key=lambda x: x[1]['active_days'], 
        reverse=True
    )
    
    print(f"{'Symbol':<8} {'Days':<6} {'AvgFcst':<8} {'AbsFcst':<8} {'Weight':<8} {'Avg Pos':<10}")
    print("-" * 70)
    
    for symbol, stats in sorted_by_activity[:10]:
        print(f"{symbol:<8} {stats['active_days']:<6} {stats['avg_forecast']:<8.2f} "
              f"{stats['avg_abs_forecast']:<8.2f} {stats['weight']:<8.3f} {stats['avg_position']:<10.2f}")
    
    # Summary of forecast characteristics
    total_active_days = sum(stats['active_days'] for stats in instrument_stats.values())
    avg_forecast_all = sum(stats['avg_forecast'] for stats in instrument_stats.values()) / len(instrument_stats)
    avg_abs_forecast_all = sum(stats['avg_abs_forecast'] for stats in instrument_stats.values()) / len(instrument_stats)
    
    print(f"\n--- Forecast Summary ---")
    print(f"Total instrument-days with positions: {total_active_days:,}")
    print(f"Average forecast across all instruments: {avg_forecast_all:.2f}")
    print(f"Average absolute forecast across all instruments: {avg_abs_forecast_all:.2f}")
    print(f"Instruments with any activity: {len(instrument_stats)}")

def plot_strategy7_equity_curve(results, save_path='results/strategy7_equity_curve.png'):
    """
    Plot Strategy 7 equity curve and save to file.
    
    Parameters:
        results (dict): Results from backtest_forecast_trend_strategy.
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
        plt.plot(equity_curve.index, equity_curve.values/1e6, 'purple', linewidth=2, 
                label=f'Strategy 7: Forecast Trend Following (SR: {performance["sharpe_ratio"]:.3f})')
        plt.title('Strategy 7: Forecast-Based Trend Following Equity Curve', fontsize=14, fontweight='bold')
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
        
        # Forecast characteristics over time
        plt.subplot(3, 1, 3)
        plt.plot(portfolio_df.index, portfolio_df['avg_forecast'], 'green', linewidth=1, 
                label='Average Forecast')
        plt.plot(portfolio_df.index, portfolio_df['avg_abs_forecast'], 'orange', linewidth=1, 
                label='Average Absolute Forecast')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.axhline(y=10, color='blue', linestyle='--', alpha=0.5, label='Target Average (10)')
        
        plt.title('Forecast Characteristics Over Time', fontsize=12, fontweight='bold')
        plt.ylabel('Forecast Value', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis dates for all subplots
        for ax in plt.gcf().get_axes():
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Performance summary text
        textstr = f'''Strategy 7 Performance Summary:
Total Return: {performance['total_return']:.1%}
Annualized Return: {performance['annualized_return']:.1%}
Volatility: {performance['annualized_volatility']:.1%}
Sharpe Ratio: {performance['sharpe_ratio']:.3f}
Max Drawdown: {performance['max_drawdown_pct']:.1f}%
Instruments: {performance.get('num_instruments', 'N/A')}
Average Forecast: {performance.get('avg_forecast', 0):.2f}
Average Absolute Forecast: {performance.get('avg_abs_forecast', 0):.2f}
Period: {config['backtest_start'].strftime('%Y-%m-%d')} to {config['backtest_end'].strftime('%Y-%m-%d')}'''
        
        plt.figtext(0.02, 0.02, textstr, fontsize=9, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.35)  # Make room for performance text
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Strategy 7 equity curve saved to: {save_path}")
        
    except Exception as e:
        print(f"Error plotting Strategy 7 equity curve: {e}")
        import traceback
        traceback.print_exc()

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
            short_span=32,
            long_years=10,
            min_vol_floor=0.05,
            weight_method='handcrafted',
            common_hypothetical_SR=0.3,
            annual_turnover_T=7.0
        )
        
        # Strategy 5 (with trend filter, long only)
        print("Running Strategy 5 (trend filter, long only)...")
        strategy5_results = backtest_trend_following_strategy(
            data_dir='Data',
            capital=50000000,
            risk_target=0.2,
            short_span=32,
            long_years=10,
            min_vol_floor=0.05,
            trend_fast_span=64,
            trend_slow_span=256,
            weight_method='handcrafted',
            common_hypothetical_SR=0.3,
            annual_turnover_T=7.0
        )
        
        # Strategy 6 (with trend filter, long/short)
        print("Running Strategy 6 (trend filter, long/short)...")
        strategy6_results = backtest_long_short_trend_strategy(
            data_dir='Data',
            capital=50000000,
            risk_target=0.2,
            short_span=32,
            long_years=10,
            min_vol_floor=0.05,
            trend_fast_span=64,
            trend_slow_span=256,
            weight_method='handcrafted',
            common_hypothetical_SR=0.3,
            annual_turnover_T=7.0
        )
        
        # Strategy 7 (with forecasts)
        print("Running Strategy 7 (trend filter with forecasts)...")
        strategy7_results = backtest_forecast_trend_strategy(
            data_dir='Data',
            capital=50000000,
            risk_target=0.2,
            short_span=32,
            long_years=10,
            min_vol_floor=0.05,
            trend_fast_span=64,
            trend_slow_span=256,
            forecast_scalar=1.9,
            forecast_cap=20.0,
            weight_method='handcrafted',
            common_hypothetical_SR=0.3,
            annual_turnover_T=7.0
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
            short_span=32,
            long_years=10,
            min_vol_floor=0.05,
            trend_fast_span=64,
            trend_slow_span=256,
            forecast_scalar=1.9,
            forecast_cap=20.0,
            weight_method='handcrafted',
            common_hypothetical_SR=0.3,
            annual_turnover_T=7.0
        )
        
        # Analyze results
        analyze_forecast_results(results)
        
        # Plot Strategy 7 equity curve
        plot_strategy7_equity_curve(results)
        
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
