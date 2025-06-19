from chapter4 import *
from chapter3 import *
from chapter2 import *
from chapter1 import *
import numpy as np
import pandas as pd
import os
from copy import copy
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
    # Calculate EWMA with specified spans and min_periods=2 (matching author's code)
    fast_ewma = prices.ewm(span=fast_span, min_periods=2, adjust=False).mean()
    slow_ewma = prices.ewm(span=slow_span, min_periods=2, adjust=False).mean()
    
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

def apply_trend_filter_to_position(base_position: pd.Series, prices: pd.Series, 
                                 fast_span: int = 64, slow_span: int = 256) -> pd.Series:
    """
    Apply trend filter to base position by zeroing out bearish positions.
    
    This matches the author's implementation approach:
    1. Calculate base position (without trend filter)
    2. Calculate EWMAC trend signal
    3. Zero out positions when EWMAC < 0 (bearish)
    
    Parameters:
        base_position (pd.Series): Base position size without trend filter.
        prices (pd.Series): Price series for trend calculation.
        fast_span (int): Fast EWMA span.
        slow_span (int): Slow EWMA span.
    
    Returns:
        pd.Series: Filtered position (zero when bearish, base position when bullish).
    """
    from copy import copy
    
    # Start with copy of base position
    filtered_position = copy(base_position)
    
    # Calculate EWMAC values
    ewmac_values = calculate_ewma_trend(prices, fast_span, slow_span)
    
    # Identify bearish periods (EWMAC < 0)
    bearish = ewmac_values < 0
    
    # Zero out positions during bearish periods
    filtered_position[bearish] = 0
    
    return filtered_position

# Note: calculate_strategy5_position_size function removed - now using new approach:
# 1. Calculate base position using calculate_portfolio_position_size
# 2. Apply trend filter by zeroing out bearish positions using apply_trend_filter_to_position

def backtest_trend_following_strategy(data_dir='Data', capital=1000000, risk_target=0.2,
                                    short_span=32, long_years=10, min_vol_floor=0.05,
                                    trend_fast_span=64, trend_slow_span=256,
                                    weight_method='handcrafted',
                                    common_hypothetical_SR=0.3, annual_turnover_T=7.0,
                                    start_date=None, end_date=None):
    """
    Backtest Strategy 5: Trend following multi-instrument portfolio with daily dynamic rebalancing.
    
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
        min_vol_floor (float): Minimum volatility floor.
        trend_fast_span (int): Fast EWMA span for trend filter.
        trend_slow_span (int): Slow EWMA span for trend filter.
        weight_method (str): Method for calculating instrument weights.
        common_hypothetical_SR (float): Common hypothetical SR for SR' calculation.
        annual_turnover_T (float): Annual turnover T for SR' calculation.
        start_date (str): Start date for backtest (YYYY-MM-DD).
        end_date (str): End date for backtest (YYYY-MM-DD).
    
    Returns:
        dict: Comprehensive backtest results.
    """
    print("=" * 60)
    print("STRATEGY 5: SLOW TREND FOLLOWING, LONG ONLY")
    print("=" * 60)
    
    # Load FX data
    print("\nLoading FX data...")
    fx_data = load_fx_data(data_dir)
    currency_mapping = get_instrument_currency_mapping()
    
    # Load all instrument data using the same function as chapter 4
    all_instruments_specs_df = load_instrument_data()
    raw_instrument_data = load_all_instrument_data(data_dir)
    
    if not raw_instrument_data:
        raise ValueError("No instrument data loaded successfully")
    
    print(f"\nPortfolio Configuration:")
    print(f"  Instruments initially loaded: {len(raw_instrument_data)}")
    print(f"  Capital: ${capital:,.0f}")
    print(f"  Risk Target: {risk_target:.1%}")
    print(f"  Weight Method: {weight_method}")
    print(f"  Trend Filter: EWMA({trend_fast_span},{trend_slow_span})")
    print(f"  Common Hypothetical SR for SR': {common_hypothetical_SR}")
    print(f"  Annual Turnover T for SR': {annual_turnover_T}")

    # Preprocess: Calculate returns, vol forecasts, and trend signals for each instrument
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

        # Calculate blended volatility (same as Strategy 4)
        blended_vol_series = calculate_blended_volatility(
            raw_returns_for_vol, short_span=short_span, long_years=long_years, min_vol_floor=min_vol_floor
        )
        # Shift to prevent lookahead bias - forecast for day T uses data up to T-1
        df['vol_forecast'] = blended_vol_series.shift(1).reindex(df.index).ffill().fillna(min_vol_floor)
        
        # Calculate trend signal using EWMAC (no lookahead bias)
        # No need to shift here since we'll apply the filter directly to positions later
        
        # Ensure critical data is present
        df.dropna(subset=['Last', 'vol_forecast', 'daily_price_change_pct'], inplace=True)
        if df.empty:
            print(f"Skipping {symbol}: Empty after dropping NaNs in critical columns.")
            continue

        processed_instrument_data[symbol] = df

    if not processed_instrument_data:
        raise ValueError("No instruments remaining after preprocessing and volatility calculation.")
    
    print(f"  Instruments after preprocessing: {len(processed_instrument_data)}")

    # Determine common date range for backtest (same logic as chapter 4)
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

    # Initialize portfolio tracking (same structure as chapter 4)
    current_portfolio_equity = capital
    portfolio_daily_records = []
    known_eligible_instruments = set()
    weights = {} 
    idm = 1.0
    
    # Initialize position tracking for cost calculation
    previous_positions = {}

    # Main time-stepping loop with daily position updates
    for idx, current_date in enumerate(trading_days_range):
        if idx == 0:
            # First day, no previous trading day in our loop range
            record = {'date': current_date, 'total_pnl': 0.0, 'portfolio_return': 0.0, 
                      'equity_sod': current_portfolio_equity, 'equity_eod': current_portfolio_equity,
                      'num_active_instruments': 0, 'num_long_signals': 0}
            for symbol_k in processed_instrument_data.keys(): 
                record[f'{symbol_k}_contracts'] = 0.0
                record[f'{symbol_k}_trend'] = 0.0
            portfolio_daily_records.append(record)
            continue
        
        previous_trading_date = trading_days_range[idx-1]
        capital_at_start_of_day = current_portfolio_equity
        daily_total_pnl = 0.0
        current_day_positions = {}
        num_active_instruments = 0
        num_long_signals = 0

        effective_data_cutoff_date = previous_trading_date if idx > 0 else current_date - pd.tseries.offsets.BDay(1)

        # Determine current period eligible instruments based on data up to cutoff
        current_iteration_eligible_instruments = set()
        for s, df_full in processed_instrument_data.items():
            df_upto_cutoff = df_full[df_full.index <= effective_data_cutoff_date]
            if not df_upto_cutoff.empty and len(df_upto_cutoff) > max(short_span, trend_slow_span):
                current_iteration_eligible_instruments.add(s)
        
        # Check if reweighting is needed (same logic as chapter 4)
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

            if instrument_weight == 0.0:
                current_day_positions[symbol] = 0.0
                continue

            # Get data for sizing (from previous_trading_date) and P&L (current_date)
            try:
                # Sizing based on previous day's close price and vol forecasts
                price_for_sizing = df_instrument.loc[previous_trading_date, 'Last']
                vol_for_sizing = df_instrument.loc[current_date, 'vol_forecast'] / np.sqrt(business_days_per_year)
                
                # Data for P&L calculation for current_date
                price_at_start_of_trading = df_instrument.loc[previous_trading_date, 'Last']
                price_at_end_of_trading = df_instrument.loc[current_date, 'Last']
                
                if (pd.isna(price_for_sizing) or pd.isna(vol_for_sizing) or 
                    pd.isna(price_at_start_of_trading) or pd.isna(price_at_end_of_trading)):
                    num_contracts = 0.0
                    instrument_pnl_today = 0.0
                    trend_signal_value = 0.0
                else:
                    vol_for_sizing = vol_for_sizing if vol_for_sizing > 0 else min_vol_floor
                    
                    # Get FX rate for position sizing
                    instrument_currency = currency_mapping.get(symbol, 'USD')
                    fx_rate = get_fx_rate_for_date_and_currency(current_date, instrument_currency, fx_data)
                    
                    # Skip KRW instruments as requested
                    if fx_rate is None:
                        num_contracts = 0.0
                        instrument_pnl_today = 0.0
                        trend_signal_value = 0.0
                    else:
                        # Step 1: Calculate base position (without trend filter) using Strategy 4 logic
                        base_position = calculate_portfolio_position_size(
                            symbol=symbol, capital=capital_at_start_of_day, weight=instrument_weight, 
                            idm=idm, price=price_for_sizing, volatility=vol_for_sizing, 
                            multiplier=instrument_multiplier, risk_target=risk_target, fx_rate=fx_rate
                        )
                        
                        # Step 2: Apply trend filter - calculate EWMAC for current date using data up to previous date
                        # Get price data up to previous trading date to avoid lookahead bias
                        price_data_for_trend = df_instrument[df_instrument.index <= previous_trading_date]['Last']
                        
                        if len(price_data_for_trend) >= max(trend_fast_span, trend_slow_span):
                            # Calculate EWMAC using historical data only
                            ewmac_values = calculate_ewma_trend(price_data_for_trend, trend_fast_span, trend_slow_span)
                            current_ewmac = ewmac_values.iloc[-1] if not ewmac_values.empty else 0.0
                            
                            # Apply trend filter: zero out position if bearish (EWMAC < 0)
                            if current_ewmac < 0:
                                num_contracts = 0.0  # Bearish - no position
                                trend_signal_value = 0.0
                            else:
                                num_contracts = base_position  # Bullish - full position
                                trend_signal_value = 1.0
                        else:
                            # Insufficient data for trend calculation
                            num_contracts = 0.0
                            trend_signal_value = 0.0
                        
                        # P&L calculation with FX rate to convert to base currency (USD)
                        price_change_in_local_currency = price_at_end_of_trading - price_at_start_of_trading
                        price_change_in_base_currency = price_change_in_local_currency * fx_rate
                        gross_pnl = num_contracts * instrument_multiplier * price_change_in_base_currency
                        
                        # Calculate trading costs
                        previous_position = previous_positions.get(symbol, 0.0)
                        trade_size = calculate_position_change(previous_position, num_contracts)
                        trading_cost = 0.0
                        
                        if trade_size > 0:  # Only apply costs when there are trades
                            sr_cost = specs.get('sr_cost', 0.0)
                            if not pd.isna(sr_cost) and sr_cost > 0:
                                trading_cost = calculate_trading_cost_from_sr(
                                    symbol, trade_size, price_at_start_of_trading, vol_for_sizing * np.sqrt(business_days_per_year),
                                    instrument_multiplier, sr_cost, capital_at_start_of_day, fx_rate
                                )
                        
                        # Net P&L after costs
                        instrument_pnl_today = gross_pnl - trading_cost
                        
                        # Count active instruments and long signals
                        if abs(num_contracts) > 0.01:
                            num_active_instruments += 1
                        if trend_signal_value > 0.5:
                            num_long_signals += 1
            
            except KeyError:  # Date not found for this instrument
                num_contracts = 0.0
                instrument_pnl_today = 0.0
                trend_signal_value = 0.0
            
            current_day_positions[symbol] = num_contracts
            daily_total_pnl += instrument_pnl_today
            
            # Update position tracking for next day's cost calculation
            previous_positions[symbol] = num_contracts

        # Update portfolio equity (same as chapter 4)
        portfolio_daily_percentage_return = daily_total_pnl / capital_at_start_of_day if capital_at_start_of_day > 0 else 0.0
        current_portfolio_equity = capital_at_start_of_day * (1 + portfolio_daily_percentage_return)

        # Record daily results
        record = {'date': current_date, 'total_pnl': daily_total_pnl, 
                  'portfolio_return': portfolio_daily_percentage_return, 
                  'equity_sod': capital_at_start_of_day, 
                  'equity_eod': current_portfolio_equity,
                  'num_active_instruments': num_active_instruments,
                  'num_long_signals': num_long_signals}
        
        for symbol_k, contracts_k in current_day_positions.items(): 
            record[f'{symbol_k}_contracts'] = contracts_k
        # Store trend signals calculated during position sizing
        trend_signals_today = {}
        for symbol_k, contracts_k in current_day_positions.items():
            # Get trend signal from the position calculation above
            if contracts_k > 0.01:
                trend_signals_today[f'{symbol_k}_trend'] = 1.0  # Was bullish
            else:
                # Need to check if zero due to bearish signal or other reasons
                try:
                    df_instrument = processed_instrument_data[symbol_k]
                    if current_date in df_instrument.index and previous_trading_date in df_instrument.index:
                        price_data_for_trend = df_instrument[df_instrument.index <= previous_trading_date]['Last']
                        if len(price_data_for_trend) >= max(trend_fast_span, trend_slow_span):
                            ewmac_values = calculate_ewma_trend(price_data_for_trend, trend_fast_span, trend_slow_span)
                            current_ewmac = ewmac_values.iloc[-1] if not ewmac_values.empty else 0.0
                            trend_signals_today[f'{symbol_k}_trend'] = 1.0 if current_ewmac >= 0 else 0.0
                        else:
                            trend_signals_today[f'{symbol_k}_trend'] = 0.0
                    else:
                        trend_signals_today[f'{symbol_k}_trend'] = 0.0
                except:
                    trend_signals_today[f'{symbol_k}_trend'] = 0.0

        for symbol in processed_instrument_data.keys():
            if symbol not in current_day_positions:
                record[f'{symbol}_contracts'] = 0.0
            # Use calculated trend signals
            record[f'{symbol}_trend'] = trend_signals_today.get(f'{symbol}_trend', 0.0)
                
        portfolio_daily_records.append(record)

    # Post-loop processing (same as chapter 4)
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
    performance['avg_long_signals'] = portfolio_df['num_long_signals'].mean()
    performance['weight_method'] = weight_method
    performance['backtest_start'] = trading_days_range.min()
    performance['backtest_end'] = trading_days_range.max()
    performance['trend_fast_span'] = trend_fast_span
    performance['trend_slow_span'] = trend_slow_span

    # Calculate per-instrument statistics (simplified for now)
    instrument_stats = {}
    for symbol in processed_instrument_data.keys():
        pos_col = f'{symbol}_contracts'
        trend_col = f'{symbol}_trend'
        
        if pos_col in portfolio_df.columns:
            # Calculate basic statistics for instruments with positions
            inst_positions = portfolio_df[pos_col][portfolio_df[pos_col] != 0]
            inst_trends = portfolio_df[trend_col][portfolio_df[pos_col] != 0]
            
            if len(inst_positions) > 0:
                instrument_stats[symbol] = {
                    'avg_position': inst_positions.mean(),
                    'weight': weights.get(symbol, 0.0),
                    'active_days': len(inst_positions),
                    'avg_trend_signal': inst_trends.mean() if len(inst_trends) > 0 else 0.0,
                    'percent_time_long': (inst_trends > 0.5).mean() if len(inst_trends) > 0 else 0.0
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
            'weight_method': weight_method,
            'common_hypothetical_SR': common_hypothetical_SR,
            'annual_turnover_T': annual_turnover_T,
            'backtest_start': trading_days_range.min(),
            'backtest_end': trading_days_range.max()
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
    
    # Top performing instruments (by weight since total_pnl is no longer calculated)
    print(f"\n--- Top 10 Instruments (by Weight and Activity) ---")
    sorted_instruments = sorted(
        instrument_stats.items(), 
        key=lambda x: x[1]['weight'], 
        reverse=True
    )
    
    print(f"{'Symbol':<8} {'Weight':<8} {'Avg Pos':<10} {'%Long':<8} {'Days':<6}")
    print("-" * 50)
    
    for symbol, stats in sorted_instruments[:10]:
        print(f"{symbol:<8} {stats['weight']:<8.3f} {stats['avg_position']:<10.2f} "
              f"{stats['percent_time_long']:<8.1%} {stats['active_days']:<6}")
    
    # Show instruments with highest trend following activity
    print(f"\n--- Top 10 Most Active Trend Followers (by Days Active) ---")
    sorted_by_activity = sorted(
        instrument_stats.items(), 
        key=lambda x: x[1]['active_days'], 
        reverse=True
    )
    
    print(f"{'Symbol':<8} {'Days':<6} {'%Long':<8} {'Weight':<8} {'Avg Pos':<10}")
    print("-" * 50)
    
    for symbol, stats in sorted_by_activity[:10]:
        print(f"{symbol:<8} {stats['active_days']:<6} {stats['percent_time_long']:<8.1%} "
              f"{stats['weight']:<8.3f} {stats['avg_position']:<10.2f}")
    
    # Summary of trend following efficiency
    total_long_days = sum(stats['active_days'] for stats in instrument_stats.values())
    avg_long_percentage = sum(stats['percent_time_long'] for stats in instrument_stats.values()) / len(instrument_stats)
    
    print(f"\n--- Trend Following Summary ---")
    print(f"Total instrument-days with positions: {total_long_days:,}")
    print(f"Average % time long across all instruments: {avg_long_percentage:.1%}")
    print(f"Instruments with >50% time long: {sum(1 for stats in instrument_stats.values() if stats['percent_time_long'] > 0.5)}")
    print(f"Instruments with any activity: {len(instrument_stats)}")

def plot_strategy_comparison(strategy4_results, strategy5_results, save_path='results/strategy_comparison.png'):
    """
    Plot Strategy 4 vs Strategy 5 equity curves for comparison.
    
    Parameters:
        strategy4_results (dict): Results from Strategy 4 backtest.
        strategy5_results (dict): Results from Strategy 5 backtest.
        save_path (str): Path to save the plot.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        s4_config = strategy4_results['config']
        s5_config = strategy5_results['config']
        s4_perf = strategy4_results['performance']
        s5_perf = strategy5_results['performance']
        
        # Build equity curves
        s4_equity = build_account_curve(strategy4_results['portfolio_data']['portfolio_return'], s4_config['capital'])
        s5_equity = build_account_curve(strategy5_results['portfolio_data']['portfolio_return'], s5_config['capital'])
        
        plt.figure(figsize=(15, 10))
        
        # Main equity curve comparison
        plt.subplot(3, 1, 1)
        plt.plot(s4_equity.index, s4_equity.values/1e6, 'b-', linewidth=2, 
                label=f'Strategy 4: Multi-Instrument (SR: {s4_perf["sharpe_ratio"]:.3f})')
        plt.plot(s5_equity.index, s5_equity.values/1e6, 'r-', linewidth=2, 
                label=f'Strategy 5: Trend Following (SR: {s5_perf["sharpe_ratio"]:.3f})')
        
        plt.title('Strategy Comparison: Multi-Instrument vs Trend Following', fontsize=14, fontweight='bold')
        plt.ylabel('Portfolio Value ($M)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Drawdown comparison
        plt.subplot(3, 1, 2)
        s4_drawdown = calculate_maximum_drawdown(s4_equity)['drawdown_series'] * 100
        s5_drawdown = calculate_maximum_drawdown(s5_equity)['drawdown_series'] * 100
        
        plt.fill_between(s4_drawdown.index, s4_drawdown.values, 0, 
                        color='blue', alpha=0.3, label='Strategy 4 Drawdown')
        plt.fill_between(s5_drawdown.index, s5_drawdown.values, 0, 
                        color='red', alpha=0.3, label='Strategy 5 Drawdown')
        plt.plot(s4_drawdown.index, s4_drawdown.values, 'b-', linewidth=1)
        plt.plot(s5_drawdown.index, s5_drawdown.values, 'r-', linewidth=1)
        
        plt.title('Drawdown Comparison', fontsize=12, fontweight='bold')
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Active instruments comparison (Strategy 5 specific)
        plt.subplot(3, 1, 3)
        s5_portfolio = strategy5_results['portfolio_data']
        if 'num_active_instruments' in s5_portfolio.columns:
            plt.plot(s5_portfolio.index, s5_portfolio['num_active_instruments'], 'g-', 
                    linewidth=1, label='Active Instruments')
        if 'num_long_signals' in s5_portfolio.columns:
            plt.plot(s5_portfolio.index, s5_portfolio['num_long_signals'], 'orange', 
                    linewidth=1, label='Long Signals')
        
        plt.title('Strategy 5: Market Exposure Over Time', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Instruments', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis dates for all subplots
        for ax in plt.gcf().get_axes():
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Performance comparison text
        time_in_market = (s5_perf['avg_long_signals'] / s5_perf['num_instruments']) * 100
        
        textstr = f'''Performance Comparison:
Strategy 4 (Multi-Instrument):
  Total Return: {s4_perf['total_return']:.1%}
  Ann. Return: {s4_perf['annualized_return']:.1%}
  Volatility: {s4_perf['annualized_volatility']:.1%}
  Sharpe: {s4_perf['sharpe_ratio']:.3f}
  Max DD: {s4_perf['max_drawdown_pct']:.1f}%
  
Strategy 5 (Trend Following):
  Total Return: {s5_perf['total_return']:.1%}
  Ann. Return: {s5_perf['annualized_return']:.1%}
  Volatility: {s5_perf['annualized_volatility']:.1%}
  Sharpe: {s5_perf['sharpe_ratio']:.3f}
  Max DD: {s5_perf['max_drawdown_pct']:.1f}%
  Time in Market: {time_in_market:.1f}%'''
        
        plt.figtext(0.02, 0.02, textstr, fontsize=9, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.35)  # Make room for performance text
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Strategy comparison chart saved to: {save_path}")
        
    except Exception as e:
        print(f"Error plotting strategy comparison: {e}")
        import traceback
        traceback.print_exc()

def plot_strategy5_equity_curve(results, save_path='results/strategy5.png'):
    """
    Plot Strategy 5 equity curve and save to file.
    
    Parameters:
        results (dict): Results from backtest_trend_following_strategy.
        save_path (str): Path to save the plot.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        portfolio_df = results['portfolio_data']
        config = results['config']
        performance = results['performance']
        
        equity_curve = build_account_curve(portfolio_df['portfolio_return'], config['capital'])
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(equity_curve.index, equity_curve.values, 'g-', linewidth=1.5, label='Strategy 5: Trend Following')
        plt.title('Strategy 5: Trend Following Portfolio Equity Curve', fontsize=14, fontweight='bold')
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        plt.subplot(2, 1, 2)
        drawdown_stats = calculate_maximum_drawdown(equity_curve)
        drawdown_series = drawdown_stats['drawdown_series'] * 100
        
        plt.fill_between(drawdown_series.index, drawdown_series.values, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        plt.plot(drawdown_series.index, drawdown_series.values, 'r-', linewidth=1)
        plt.title('Drawdown', fontsize=12, fontweight='bold')
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        for ax in plt.gcf().get_axes():
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        time_in_market = (performance['avg_long_signals'] / performance['num_instruments']) * 100
        
        textstr = f'''Performance Summary:
Total Return: {performance['total_return']:.1%}
Annualized Return: {performance['annualized_return']:.1%}
Volatility: {performance['annualized_volatility']:.1%}
Sharpe Ratio: {performance['sharpe_ratio']:.3f}
Max Drawdown: {performance['max_drawdown_pct']:.1f}%
Time in Market: {time_in_market:.1f}%
Instruments: {performance.get('num_instruments', 'N/A')} 
Period: {config['backtest_start'].strftime('%Y-%m-%d')} to {config['backtest_end'].strftime('%Y-%m-%d')}'''
        
        plt.figtext(0.02, 0.02, textstr, fontsize=9, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Strategy 5 equity curve saved to: {save_path}")
        
    except Exception as e:
        print(f"Error plotting equity curve: {e}")
        import traceback
        traceback.print_exc()

def plot_equity_curves_only(strategy4_results, strategy5_results, save_path='results/equity_curves_comparison.png'):
    """
    Plot only the equity curves for Strategy 4 vs Strategy 5 comparison.
    
    Parameters:
        strategy4_results (dict): Results from Strategy 4 backtest.
        strategy5_results (dict): Results from Strategy 5 backtest.
        save_path (str): Path to save the plot.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        s4_config = strategy4_results['config']
        s5_config = strategy5_results['config']
        s4_perf = strategy4_results['performance']
        s5_perf = strategy5_results['performance']
        
        # Build equity curves
        s4_equity = build_account_curve(strategy4_results['portfolio_data']['portfolio_return'], s4_config['capital'])
        s5_equity = build_account_curve(strategy5_results['portfolio_data']['portfolio_return'], s5_config['capital'])
        
        plt.figure(figsize=(14, 8))
        
        # Plot equity curves
        plt.plot(s4_equity.index, s4_equity.values/1e6, 'b-', linewidth=2.5, 
                label=f'Strategy 4: Multi-Instrument (SR: {s4_perf["sharpe_ratio"]:.3f})')
        plt.plot(s5_equity.index, s5_equity.values/1e6, 'r-', linewidth=2.5, 
                label=f'Strategy 5: Trend Following (SR: {s5_perf["sharpe_ratio"]:.3f})')
        
        plt.title('Equity Curve Comparison: Multi-Instrument vs Trend Following', 
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
        
        print(f"\n✅ Equity curves comparison saved to: {save_path}")
        
    except Exception as e:
        print(f"Error plotting equity curves: {e}")
        import traceback
        traceback.print_exc()

def compare_strategies(strategy5_results=None, capital=1000000, risk_target=0.2, 
                      start_date=None, end_date=None, weight_method='handcrafted'):
    """
    Compare Strategy 4 (no trend filter) vs Strategy 5 (with trend filter).
    
    Parameters:
        strategy5_results (dict): Pre-computed Strategy 5 results to avoid re-running.
        capital (float): Starting capital for Strategy 4 comparison.
        risk_target (float): Risk target for Strategy 4 comparison.
        start_date (str): Start date for Strategy 4 comparison.
        end_date (str): End date for Strategy 4 comparison.
        weight_method (str): Weight method for Strategy 4 comparison.
    """
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON: STRATEGY 4 vs STRATEGY 5")
    print("=" * 80)
    
    try:
        # Strategy 4 (no trend filter)
        print("Running Strategy 4 (no trend filter) for comparison...")
        strategy4_results = backtest_multi_instrument_strategy(
            data_dir='Data',
            capital=capital,
            risk_target=risk_target,
            short_span=32,
            long_years=10,
            min_vol_floor=0.05,
            weight_method=weight_method,
            common_hypothetical_SR=0.3,
            annual_turnover_T=7.0,
            start_date=start_date,
            end_date=end_date
        )
        
        # Use pre-computed Strategy 5 results if provided
        if strategy5_results is None:
            print("Running Strategy 5 (with trend filter) for comparison...")
            strategy5_results = backtest_trend_following_strategy(
                data_dir='Data',
                capital=capital,
                risk_target=risk_target,
                short_span=32,
                long_years=10,
                min_vol_floor=0.05,
                trend_fast_span=64,
                trend_slow_span=256,
                weight_method=weight_method,
                common_hypothetical_SR=0.3,
                annual_turnover_T=7.0,
                start_date=start_date,
                end_date=end_date
            )
        else:
            print("Using pre-computed Strategy 5 results...")
        
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
    # ===========================================
    # CONFIGURATION - MODIFY THESE AS NEEDED
    # ===========================================
    CAPITAL = 1000000               # Starting capital
    START_DATE = '2000-01-01'       # Backtest start date (YYYY-MM-DD) or None for earliest available
    END_DATE = '2025-12-31'         # Backtest end date (YYYY-MM-DD) or None for latest available
    RISK_TARGET = 0.2               # 20% annual risk target
    WEIGHT_METHOD = 'handcrafted'   # 'equal', 'vol_inverse', or 'handcrafted'
    TREND_FAST_SPAN = 64            # Fast EWMA span for trend filter
    TREND_SLOW_SPAN = 256           # Slow EWMA span for trend filter
    
    print("=" * 60)
    print("TESTING STRATEGY 5: TREND FOLLOWING")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Capital: ${CAPITAL:,}")
    print(f"  Date Range: {START_DATE or 'earliest'} to {END_DATE or 'latest'}")
    print(f"  Risk Target: {RISK_TARGET:.1%}")
    print(f"  Weight Method: {WEIGHT_METHOD}")
    print(f"  Trend Filter: EWMA({TREND_FAST_SPAN},{TREND_SLOW_SPAN})")
    print("=" * 60)
    
    try:
        # Run Strategy 5 backtest
        results = backtest_trend_following_strategy(
            data_dir='Data',
            capital=CAPITAL,
            risk_target=RISK_TARGET,
            short_span=32,
            long_years=10,
            min_vol_floor=0.05,
            trend_fast_span=TREND_FAST_SPAN,
            trend_slow_span=TREND_SLOW_SPAN,
            weight_method=WEIGHT_METHOD,
            common_hypothetical_SR=0.3,
            annual_turnover_T=7.0,
            start_date=START_DATE,
            end_date=END_DATE
        )
        
        # Analyze results
        analyze_trend_following_results(results)

        # Plot Strategy 5 equity curve
        plot_strategy5_equity_curve(results)

        '''
        # Compare strategies using pre-computed Strategy 5 results
        comparison = compare_strategies(
            strategy5_results=results,
            capital=CAPITAL,
            risk_target=RISK_TARGET,
            start_date=START_DATE,
            end_date=END_DATE,
            weight_method=WEIGHT_METHOD
        )
        
        
        
        # Plot strategy comparison
        if comparison and comparison['strategy4'] and comparison['strategy5']:
            plot_strategy_comparison(comparison['strategy4'], comparison['strategy5'])
        
        # Plot equity curves only
        plot_equity_curves_only(comparison['strategy4'], comparison['strategy5'])

        '''
        
        print(f"\nStrategy 5 backtest completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error in Strategy 5 backtest: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
