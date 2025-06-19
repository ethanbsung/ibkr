from chapter9 import *
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

#####   STRATEGY 10: BOLLINGER BANDS MEAN REVERSION   #####

def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands: SMA ± (std_dev * num_std).
    
    Parameters:
        prices (pd.Series): Price series.
        window (int): Rolling window for SMA and std calculation.
        num_std (float): Number of standard deviations for bands.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['sma', 'upper_band', 'lower_band', 'bb_width', 'bb_position'].
    """
    # Calculate Simple Moving Average
    sma = prices.rolling(window=window, min_periods=window).mean()
    
    # Calculate rolling standard deviation
    rolling_std = prices.rolling(window=window, min_periods=window).std()
    
    # Calculate Bollinger Bands
    upper_band = sma + (rolling_std * num_std)
    lower_band = sma - (rolling_std * num_std)
    
    # Calculate band width (normalized)
    bb_width = (upper_band - lower_band) / sma
    
    # Calculate position within bands (0 = lower band, 0.5 = middle, 1 = upper band)
    bb_position = (prices - lower_band) / (upper_band - lower_band)
    
    return pd.DataFrame({
        'sma': sma,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'bb_width': bb_width,
        'bb_position': bb_position
    }, index=prices.index)

def calculate_mean_reversion_forecast(prices: pd.Series, window: int = 20, num_std: float = 2.0,
                                    entry_threshold: float = 0.1, exit_threshold: float = 0.5,
                                    max_forecast: float = 20.0) -> pd.Series:
    """
    Calculate mean reversion forecast based on Bollinger Bands.
    
    Mean reversion logic:
    - Strong buy signal when price is below lower band (bb_position < entry_threshold)
    - Strong sell signal when price is above upper band (bb_position > 1 - entry_threshold)
    - Exit signals when price returns to middle (bb_position near exit_threshold)
    
    Parameters:
        prices (pd.Series): Price series.
        window (int): Bollinger Bands window.
        num_std (float): Number of standard deviations for bands.
        entry_threshold (float): Threshold for entry signals (0.1 = 10% into bands).
        exit_threshold (float): Threshold for exit signals (0.5 = middle of bands).
        max_forecast (float): Maximum absolute forecast value.
    
    Returns:
        pd.Series: Mean reversion forecast (-20 to +20).
    """
    bb_data = calculate_bollinger_bands(prices, window, num_std)
    bb_position = bb_data['bb_position']
    bb_width = bb_data['bb_width']
    
    # Initialize forecast series
    forecast = pd.Series(0.0, index=prices.index)
    
    # Mean reversion signals based on position within bands
    # Stronger signals when bands are wider (more volatile periods)
    volatility_multiplier = np.clip(bb_width / bb_width.rolling(window=window*2).median(), 0.5, 2.0)
    volatility_multiplier = volatility_multiplier.fillna(1.0)
    
    # Buy signals (positive forecast) when price is near lower band
    buy_signal_strength = np.where(
        bb_position < entry_threshold,
        (entry_threshold - bb_position) / entry_threshold,  # Stronger as we go below lower band
        0.0
    )
    
    # Sell signals (negative forecast) when price is near upper band  
    sell_signal_strength = np.where(
        bb_position > (1 - entry_threshold),
        (bb_position - (1 - entry_threshold)) / entry_threshold,  # Stronger as we go above upper band
        0.0
    )
    
    # Calculate raw forecast
    raw_forecast = (buy_signal_strength - sell_signal_strength) * volatility_multiplier
    
    # Scale to max forecast and apply to series
    forecast = raw_forecast * max_forecast
    
    # Smooth the forecast to reduce noise
    forecast = forecast.rolling(window=3, min_periods=1).mean()
    
    # Cap the forecast
    forecast = np.clip(forecast, -max_forecast, max_forecast)
    
    return forecast

def calculate_strategy10_position_size(symbol, capital, weight, idm, price, volatility, 
                                     multiplier, mean_reversion_forecast, risk_target=0.2, fx_rate=1.0):
    """
    Calculate position size for Strategy 10 with mean reversion forecast scaling.
    
    Formula: N = MR_forecast × Capital × IDM × Weight × τ ÷ (10 × Multiplier × Price × FX × σ%)
    
    Parameters:
        symbol (str): Instrument symbol.
        capital (float): Total portfolio capital.
        weight (float): Weight allocated to this instrument.
        idm (float): Instrument Diversification Multiplier.
        price (float): Current price.
        volatility (float): Annualized volatility forecast.
        multiplier (float): Contract multiplier.
        mean_reversion_forecast (float): Mean reversion forecast value (-20 to +20).
        risk_target (float): Target risk fraction.
        fx_rate (float): FX rate for currency conversion.
    
    Returns:
        float: Number of contracts for this instrument.
    """
    if np.isnan(volatility) or volatility <= 0 or np.isnan(mean_reversion_forecast):
        return 0
    
    # Calculate position size with mean reversion forecast scaling
    numerator = mean_reversion_forecast * capital * idm * weight * risk_target
    denominator = 10 * multiplier * price * fx_rate * volatility
    
    position_size = numerator / denominator
    
    # Protect against infinite or extremely large position sizes
    if np.isinf(position_size) or abs(position_size) > 100000:
        return 0
    
    return position_size

def backtest_bollinger_mean_reversion_strategy(data_dir='Data', capital=50000000, risk_target=0.2,
                                             short_span=32, long_years=10, min_vol_floor=0.05,
                                             bb_window=20, bb_std=2.0, entry_threshold=0.1, 
                                             exit_threshold=0.5, max_forecast=20.0,
                                             buffer_fraction=0.1,
                                             weight_method='handcrafted',
                                             common_hypothetical_SR=0.3, annual_turnover_T=7.0,
                                             start_date=None, end_date=None,
                                             debug_forecasts=False):
    """
    Backtest Strategy 10: Bollinger Bands mean reversion with buffering.
    
    Implementation: "Trade a portfolio of one or more instruments using mean reversion 
    signals from Bollinger Bands. Go long when price breaks below lower band, 
    go short when price breaks above upper band, with positions scaled by forecast strength."
    
    Parameters:
        data_dir (str): Directory containing price data files.
        capital (float): Initial capital.
        risk_target (float): Target risk fraction.
        short_span (int): EWMA span for short-run volatility.
        long_years (int): Years for long-run volatility average.
        min_vol_floor (float): Minimum volatility floor.
        bb_window (int): Bollinger Bands window (default 20).
        bb_std (float): Bollinger Bands standard deviations (default 2.0).
        entry_threshold (float): Entry threshold for mean reversion signals.
        exit_threshold (float): Exit threshold for mean reversion signals.
        max_forecast (float): Maximum absolute forecast value.
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
    print("STRATEGY 10: BOLLINGER BANDS MEAN REVERSION")
    print("=" * 60)
    
    # Load all instrument data using the same function as previous chapters
    all_instruments_specs_df = load_instrument_data()
    raw_instrument_data = load_all_instrument_data(data_dir)
    
    if not raw_instrument_data:
        raise ValueError("No instrument data loaded successfully")
    
    print(f"\nPortfolio Configuration:")
    print(f"  Instruments initially loaded: {len(raw_instrument_data)}")
    print(f"  Capital: ${capital:,.0f}")
    print(f"  Risk Target: {risk_target:.1%}")
    print(f"  Weight Method: {weight_method}")
    print(f"  Bollinger Bands: {bb_window}-period, {bb_std} std dev")
    print(f"  Entry Threshold: {entry_threshold}")
    print(f"  Exit Threshold: {exit_threshold}")
    print(f"  Max Forecast: ±{max_forecast}")
    print(f"  Buffer Fraction: {buffer_fraction}")
    print(f"  Common Hypothetical SR for SR': {common_hypothetical_SR}")
    print(f"  Annual Turnover T for SR': {annual_turnover_T}")

    # Preprocess: Calculate returns, vol forecasts, and mean reversion forecasts for each instrument
    processed_instrument_data = {}
    for symbol, df_orig in raw_instrument_data.items():
        df = df_orig.copy()
        if 'Last' not in df.columns:
            print(f"Skipping {symbol}: 'Last' column missing.")
            continue
        
        df['daily_price_change_pct'] = df['Last'].pct_change()
        
        # Volatility forecast for day D is made using data up to D-1 (no lookahead bias)
        raw_returns_for_vol = df['daily_price_change_pct'].dropna()
        if len(raw_returns_for_vol) < max(short_span, bb_window * 2):  # Need sufficient data
            print(f"Skipping {symbol}: Insufficient data for vol forecast and BB ({len(raw_returns_for_vol)} days).")
            continue

        # Calculate blended volatility (same as previous strategies)
        blended_vol_series = calculate_blended_volatility(
            raw_returns_for_vol, short_span=short_span, long_years=long_years, min_vol_floor=min_vol_floor
        )
        # Shift to prevent lookahead bias - forecast for day T uses data up to T-1
        df['vol_forecast'] = blended_vol_series.shift(1).reindex(df.index).ffill().fillna(min_vol_floor)
        
        # Calculate mean reversion forecast using Bollinger Bands (no lookahead bias)
        mr_forecast_series = calculate_mean_reversion_forecast(
            df['Last'], bb_window, bb_std, entry_threshold, exit_threshold, max_forecast
        )
        # Shift to prevent lookahead bias - forecast for day T uses data up to T-1
        df['mr_forecast'] = mr_forecast_series.shift(1).reindex(df.index).fillna(0)
        
        # Debug first instrument forecasts if requested
        if debug_forecasts and symbol == list(raw_instrument_data.keys())[0]:
            print(f"\n=== FORECAST DEBUG FOR {symbol} ===")
            sample_forecasts = df['mr_forecast'].dropna()[:10]
            if len(sample_forecasts) > 0:
                print(f"Sample MR Forecasts: {sample_forecasts.values}")
                print(f"Average MR Forecast: {df['mr_forecast'].mean():.3f}")
                print(f"Average Absolute Forecast: {df['mr_forecast'].abs().mean():.3f}")
        
        # Ensure critical data is present
        df.dropna(subset=['Last', 'vol_forecast', 'daily_price_change_pct'], inplace=True)
        if df.empty:
            print(f"Skipping {symbol}: Empty after dropping NaNs in critical columns.")
            continue

        processed_instrument_data[symbol] = df

    if not processed_instrument_data:
        raise ValueError("No instruments remaining after preprocessing and volatility calculation.")
    
    print(f"  Instruments after preprocessing: {len(processed_instrument_data)}")

    # Determine common date range for backtest (same logic as previous strategies)
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

    # Initialize portfolio tracking (same structure as previous strategies)
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
            if not df_upto_cutoff.empty and len(df_upto_cutoff) > max(short_span, bb_window * 2):
                current_iteration_eligible_instruments.add(s)
        
        # Check if reweighting is needed (same logic as previous strategies)
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
                    vol_for_sizing = df_instrument.loc[current_date, 'vol_forecast']
                    forecast_for_sizing = df_instrument.loc[current_date, 'mr_forecast']
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
                        
                        # Calculate optimal position size with mean reversion forecast scaling
                        optimal_position = calculate_strategy10_position_size(
                            symbol=symbol, capital=capital_at_start_of_day, weight=instrument_weight, 
                            idm=idm, price=price_for_sizing, volatility=vol_for_sizing, 
                            multiplier=instrument_multiplier, mean_reversion_forecast=forecast_for_sizing, 
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

        # Update portfolio equity (same as previous strategies)
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
                    sig = processed_instrument_data[s_proc].loc[current_date, 'mr_forecast']
                    if pd.notna(sig):
                        forecast_val_fill = sig
                record[f'{s_proc}_forecast'] = forecast_val_fill
            if f'{s_proc}_trades' not in record:
                record[f'{s_proc}_trades'] = 0
                
        portfolio_daily_records.append(record)

    # Post-loop processing (same as previous strategies)
    if not portfolio_daily_records:
        raise ValueError("No daily records generated during backtest.")
        
    portfolio_df = pd.DataFrame(portfolio_daily_records)
    portfolio_df.set_index('date', inplace=True)
    
    print(f"Portfolio backtest loop completed. {len(portfolio_df)} daily records.")
    if portfolio_df.empty or 'portfolio_return' not in portfolio_df.columns or portfolio_df['portfolio_return'].std() == 0:
        print(f"Average active instruments: {portfolio_df['num_active_instruments'].mean():.1f}")
        print(f"Average mean reversion forecast: {portfolio_df['avg_forecast'].mean():.2f}")
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
    performance['bb_window'] = bb_window
    performance['bb_std'] = bb_std
    performance['entry_threshold'] = entry_threshold
    performance['exit_threshold'] = exit_threshold
    performance['max_forecast'] = max_forecast
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
            'bb_window': bb_window,
            'bb_std': bb_std,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'max_forecast': max_forecast,
            'buffer_fraction': buffer_fraction,
            'weight_method': weight_method,
            'common_hypothetical_SR': common_hypothetical_SR,
            'annual_turnover_T': annual_turnover_T,
            'backtest_start': trading_days_range.min(),
            'backtest_end': trading_days_range.max()
        }
    }

def analyze_bollinger_mean_reversion_results(results):
    """
    Analyze and display comprehensive Bollinger Bands mean reversion results.
    
    Parameters:
        results (dict): Results from backtest_bollinger_mean_reversion_strategy.
    """
    performance = results['performance']
    instrument_stats = results['instrument_stats']
    config = results['config']
    
    print("\n" + "=" * 60)
    print("BOLLINGER BANDS MEAN REVERSION PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Overall performance
    print(f"\n--- Overall Portfolio Performance ---")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Annualized Return: {performance['annualized_return']:.2%}")
    print(f"Annualized Volatility: {performance['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {performance['max_drawdown_pct']:.1f}%")
    print(f"Skewness: {performance['skewness']:.3f}")
    
    # Mean reversion characteristics
    print(f"\n--- Mean Reversion Characteristics ---")
    print(f"Average Active Instruments: {performance['avg_active_instruments']:.1f}")
    print(f"Average MR Forecast: {performance['avg_forecast']:.2f}")
    print(f"Average Absolute Forecast: {performance['avg_abs_forecast']:.2f}")
    print(f"Total Trades: {performance['total_trades']:,}")
    print(f"Average Daily Trades: {performance['avg_daily_trades']:.1f}")
    print(f"Bollinger Bands: {config['bb_window']}-period, {config['bb_std']} std dev")
    print(f"Entry/Exit Thresholds: {config['entry_threshold']}/{config['exit_threshold']}")
    print(f"Max Forecast: ±{config['max_forecast']}")
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
    
    # Show instruments with highest mean reversion activity
    print(f"\n--- Top 10 Most Active Mean Reversion Instruments (by Days Active) ---")
    sorted_by_activity = sorted(
        instrument_stats.items(), 
        key=lambda x: x[1]['active_days'], 
        reverse=True
    )
    
    print(f"{'Symbol':<8} {'Days':<6} {'AvgFcst':<8} {'AbsFcst':<8} {'MinFcst':<8} {'MaxFcst':<8} {'Trades':<8} {'Weight':<8}")
    print("-" * 85)
    
    for symbol, stats in sorted_by_activity[:10]:
        print(f"{symbol:<8} {stats['active_days']:<6} {stats['avg_forecast']:<8.2f} "
              f"{stats['avg_abs_forecast']:<8.2f} {stats['min_forecast']:<8.2f} {stats['max_forecast']:<8.2f} "
              f"{stats['total_trades']:<8} {stats['weight']:<8.3f}")
    
    # Summary of mean reversion characteristics
    total_active_days = sum(stats['active_days'] for stats in instrument_stats.values())
    avg_forecast_all = sum(stats['avg_forecast'] for stats in instrument_stats.values()) / len(instrument_stats)
    avg_abs_forecast_all = sum(stats['avg_abs_forecast'] for stats in instrument_stats.values()) / len(instrument_stats)
    total_trades_all = sum(stats['total_trades'] for stats in instrument_stats.values())
    
    print(f"\n--- Mean Reversion Summary ---")
    print(f"Total instrument-days with positions: {total_active_days:,}")
    print(f"Average MR forecast across all instruments: {avg_forecast_all:.2f}")
    print(f"Average absolute forecast across all instruments: {avg_abs_forecast_all:.2f}")
    print(f"Total individual instrument trades: {total_trades_all:,}")
    print(f"Instruments with any activity: {len(instrument_stats)}")

def plot_strategy10_equity_curve(results, save_path='results/strategy10_equity_curve.png'):
    """
    Plot Strategy 10 equity curve and save to file.
    
    Parameters:
        results (dict): Results from backtest_bollinger_mean_reversion_strategy.
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
                label=f'Strategy 10: Bollinger Bands Mean Reversion (SR: {performance["sharpe_ratio"]:.3f})')
        plt.title('Strategy 10: Bollinger Bands Mean Reversion Equity Curve', fontsize=14, fontweight='bold')
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
        
        # Mean reversion forecast and trading activity over time
        plt.subplot(3, 1, 3)
        plt.plot(portfolio_df.index, portfolio_df['avg_forecast'], 'blue', linewidth=1, 
                label='Average MR Forecast')
        plt.plot(portfolio_df.index, portfolio_df['avg_abs_forecast'], 'orange', linewidth=1, 
                label='Average Absolute Forecast')
        plt.plot(portfolio_df.index, portfolio_df['total_trades'], 'green', linewidth=1, alpha=0.7,
                label='Daily Trades')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.title('Mean Reversion Forecast & Trading Activity Over Time', fontsize=12, fontweight='bold')
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
        textstr = f'''Strategy 10 Performance Summary:
Total Return: {performance['total_return']:.1%}
Annualized Return: {performance['annualized_return']:.1%}
Volatility: {performance['annualized_volatility']:.1%}
Sharpe Ratio: {performance['sharpe_ratio']:.3f}
Max Drawdown: {performance['max_drawdown_pct']:.1f}%
Instruments: {performance.get('num_instruments', 'N/A')}
Average MR Forecast: {performance.get('avg_forecast', 0):.2f}
Total Trades: {performance.get('total_trades', 0):,}
Bollinger Bands: {config.get('bb_window', 20)}-period, {config.get('bb_std', 2.0)} std
Entry/Exit: {config.get('entry_threshold', 0.1)}/{config.get('exit_threshold', 0.5)}
Period: {config['backtest_start'].strftime('%Y-%m-%d')} to {config['backtest_end'].strftime('%Y-%m-%d')}'''
        
        plt.figtext(0.02, 0.02, textstr, fontsize=9, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.35)  # Make room for performance text
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Strategy 10 equity curve saved to: {save_path}")
        
    except Exception as e:
        print(f"Error plotting Strategy 10 equity curve: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Test Strategy 10 implementation: Bollinger Bands mean reversion.
    """
    print("=" * 60)
    print("TESTING STRATEGY 10: BOLLINGER BANDS MEAN REVERSION")
    print("=" * 60)
    
    try:
        # Run Strategy 10 backtest
        results = backtest_bollinger_mean_reversion_strategy(
            data_dir='Data',
            capital=1000000,
            risk_target=0.2,
            short_span=32,
            long_years=10,
            min_vol_floor=0.05,
            bb_window=20,
            bb_std=2.0,
            entry_threshold=0.1,
            exit_threshold=0.5,
            max_forecast=20.0,
            buffer_fraction=0.1,
            weight_method='handcrafted',
            common_hypothetical_SR=0.3,
            annual_turnover_T=7.0,
            debug_forecasts=False
        )
        
        # Analyze results
        analyze_bollinger_mean_reversion_results(results)
        
        # Plot Strategy 10 equity curve
        plot_strategy10_equity_curve(results)
        
        print(f"\nStrategy 10 backtest completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error in Strategy 10 testing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 