from chapter4 import *  # For multi-instrument infrastructure
from chapter3 import *  # For volatility forecasting
from chapter2 import *  # For position sizing
from chapter1 import *  # For basic calculations
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')

#####   RSI-BASED MEAN REVERSION SIGNAL GENERATION   #####

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Parameters:
        prices (pd.Series): Price series.
        window (int): RSI calculation window (default 14).
    
    Returns:
        pd.Series: RSI values (0-100).
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    
    # Avoid division by zero
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)  # Fill NaN with neutral RSI

def calculate_bollinger_bands_volatility_adjusted(prices: pd.Series, window: int = 20, 
                                                 vol_lookback: int = 252, num_std: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calculate volatility-adjusted Bollinger Bands using rolling volatility scaling.
    
    Unlike traditional Bollinger Bands that use a fixed standard deviation multiplier,
    this implementation adjusts the band width based on the current volatility regime.
    
    Parameters:
        prices (pd.Series): Price series.
        window (int): Moving average window for center line (default 20).
        vol_lookback (int): Lookback period for volatility calculation (default 252 = 1 year).
        num_std (float): Base number of standard deviations for bands (default 2.0).
    
    Returns:
        Dict[str, pd.Series]: Dictionary containing 'upper', 'lower', 'middle', 'width', 'vol_adj_factor'.
    """
    # Calculate rolling mean (middle band)
    middle_band = prices.rolling(window=window, min_periods=max(1, window//2)).mean()
    
    # Calculate rolling standard deviation for traditional bands
    rolling_std = prices.rolling(window=window, min_periods=max(1, window//2)).std()
    
    # Calculate volatility adjustment factor
    # Use longer-term volatility to adjust band width
    price_returns = prices.pct_change()
    # Ensure min_periods doesn't exceed window size
    min_periods_vol = min(30, max(10, vol_lookback // 4))
    long_term_vol = price_returns.rolling(window=vol_lookback, min_periods=min_periods_vol).std() * np.sqrt(252)
    short_term_vol = price_returns.rolling(window=window, min_periods=max(1, window//2)).std() * np.sqrt(252)
    
    # Volatility adjustment factor: expand bands in high vol periods, contract in low vol periods
    vol_adjustment = short_term_vol / long_term_vol
    vol_adjustment = vol_adjustment.fillna(1.0).clip(lower=0.5, upper=3.0)  # Reasonable bounds
    
    # Calculate volatility-adjusted bands
    adjusted_std = rolling_std * vol_adjustment
    upper_band = middle_band + (num_std * adjusted_std)
    lower_band = middle_band - (num_std * adjusted_std)
    
    # Band width as percentage of middle band
    band_width = (upper_band - lower_band) / middle_band
    
    return {
        'upper': upper_band,
        'lower': lower_band,
        'middle': middle_band,
        'width': band_width,
        'vol_adj_factor': vol_adjustment
    }

def calculate_rsi_position_signal(prices: pd.Series, rsi_window: int = 14, 
                                 oversold: float = 30, overbought: float = 70,
                                 signal_strength_method: str = 'linear') -> pd.Series:
    """
    Calculate mean reversion position signal from RSI.
    
    Signal interpretation:
    - Positive values (0 to +1): Mean reversion long signal when RSI is oversold
    - Negative values (-1 to 0): Mean reversion short signal when RSI is overbought
    - Zero: No signal when RSI is in neutral territory
    
    Parameters:
        prices (pd.Series): Price series.
        rsi_window (int): RSI calculation window (default 14).
        oversold (float): RSI oversold threshold (default 30).
        overbought (float): RSI overbought threshold (default 70).
        signal_strength_method (str): Method for calculating signal strength ('linear', 'sigmoid').
    
    Returns:
        pd.Series: Signal values from -1 to +1.
    """
    rsi = calculate_rsi(prices, window=rsi_window)
    
    if signal_strength_method == 'linear':
        # Linear signal based on RSI distance from thresholds
        signal = np.where(rsi < oversold, 
                         (oversold - rsi) / oversold,  # Stronger signal when more oversold
                         0)
        signal = np.where(rsi > overbought,
                         -(rsi - overbought) / (100 - overbought),  # Negative signal when overbought
                         signal)
        
    elif signal_strength_method == 'sigmoid':
        # Sigmoid-based signal for smoother transitions
        oversold_signal = 1 / (1 + np.exp((rsi - oversold + 10) / 5))  # Sigmoid around oversold
        overbought_signal = -1 / (1 + np.exp(-(rsi - overbought - 10) / 5))  # Sigmoid around overbought
        
        signal = np.where(rsi < oversold + 10, oversold_signal, 0)
        signal = np.where(rsi > overbought - 10, overbought_signal, signal)
        
    else:
        raise ValueError(f"Unknown signal_strength_method: {signal_strength_method}")
    
    # Clip final signal to [-1, +1] range
    signal = pd.Series(signal, index=prices.index).clip(lower=-1.0, upper=1.0)
    
    return signal

def calculate_bollinger_position_signal(prices: pd.Series, window: int = 20, 
                                      vol_lookback: int = 252, num_std: float = 2.0,
                                      signal_strength_method: str = 'linear') -> pd.Series:
    """
    DEPRECATED: Use calculate_rsi_position_signal instead.
    Calculate mean reversion position signal from volatility-adjusted Bollinger Bands.
    """
    bands = calculate_bollinger_bands_volatility_adjusted(prices, window, vol_lookback, num_std)
    
    upper_band = bands['upper']
    lower_band = bands['lower']
    middle_band = bands['middle']
    
    # Calculate position within bands (-1 = at lower band, 0 = at middle, +1 = at upper band)
    band_width = upper_band - lower_band
    band_position = (prices - middle_band) / (band_width / 2)
    band_position = band_position.clip(lower=-1.5, upper=1.5)  # Allow some overshoot
    
    if signal_strength_method == 'linear':
        # Linear mean reversion signal: opposite of band position
        signal = -band_position
        
    elif signal_strength_method == 'sigmoid':
        # Sigmoid-based signal for smoother transitions
        signal = -np.tanh(band_position * 2)  # More aggressive near bands
        
    else:
        raise ValueError(f"Unknown signal_strength_method: {signal_strength_method}")
    
    # Clip final signal to [-1, +1] range
    signal = signal.clip(lower=-1.0, upper=1.0)
    
    return signal

def calculate_rsi_long_only_signal(prices: pd.Series, rsi_window: int = 14, 
                                  oversold: float = 30, overbought: float = 70,
                                  signal_strength_method: str = 'linear') -> pd.Series:
    """
    Calculate long-only mean reversion signal from RSI.
    
    This version only generates buy signals when RSI is oversold,
    following a long-only mean reversion approach.
    
    Signal interpretation:
    - Positive values (0 to +1): Buy signal when RSI is oversold
    - Zero: No position when RSI is neutral or overbought
    
    Parameters:
        prices (pd.Series): Price series.
        rsi_window (int): RSI calculation window (default 14).
        oversold (float): RSI oversold threshold (default 30).
        overbought (float): RSI overbought threshold (default 70).
        signal_strength_method (str): Method for calculating signal strength.
    
    Returns:
        pd.Series: Signal values from 0 to +1.
    """
    # Get the full signal first
    full_signal = calculate_rsi_position_signal(prices, rsi_window, oversold, overbought, signal_strength_method)
    
    # Convert to long-only: keep only positive signals, set negative to zero
    long_only_signal = full_signal.clip(lower=0.0, upper=1.0)
    
    return long_only_signal

def calculate_bollinger_long_only_signal(prices: pd.Series, window: int = 20, 
                                       vol_lookback: int = 252, num_std: float = 2.0,
                                       signal_strength_method: str = 'linear') -> pd.Series:
    """
    DEPRECATED: Use calculate_rsi_long_only_signal instead.
    Calculate long-only mean reversion signal from Bollinger Bands.
    """
    # Get the full signal first
    full_signal = calculate_bollinger_position_signal(prices, window, vol_lookback, num_std, signal_strength_method)
    
    # Convert to long-only: keep only positive signals, set negative to zero
    long_only_signal = full_signal.clip(lower=0.0, upper=1.0)
    
    return long_only_signal

def apply_rsi_filter_to_position(base_position: pd.Series, prices: pd.Series,
                                rsi_window: int = 14, oversold: float = 30, overbought: float = 70,
                                signal_strength_method: str = 'linear', long_only: bool = True) -> pd.Series:
    """
    Apply RSI mean reversion filter to base position.
    
    Parameters:
        base_position (pd.Series): Base position size without signal filter.
        prices (pd.Series): Price series for signal calculation.
        rsi_window (int): RSI calculation window.
        oversold (float): RSI oversold threshold.
        overbought (float): RSI overbought threshold.
        signal_strength_method (str): Method for calculating signal strength.
        long_only (bool): Whether to use long-only signals.
    
    Returns:
        pd.Series: Signal-adjusted position (base_position * signal_strength).
    """
    if long_only:
        signal = calculate_rsi_long_only_signal(prices, rsi_window, oversold, overbought, signal_strength_method)
    else:
        signal = calculate_rsi_position_signal(prices, rsi_window, oversold, overbought, signal_strength_method)
    
    # Align signal with base_position index and handle NaN values
    signal_aligned = signal.reindex(base_position.index).fillna(0.0)
    
    # Apply signal to base position
    filtered_position = base_position * signal_aligned
    
    # For long_only, ensure no negative positions
    if long_only:
        filtered_position = filtered_position.clip(lower=0.0)
    
    return filtered_position

def apply_bollinger_filter_to_position(base_position: pd.Series, prices: pd.Series,
                                     window: int = 20, vol_lookback: int = 252, 
                                     num_std: float = 2.0, signal_strength_method: str = 'linear',
                                     long_only: bool = True) -> pd.Series:
    """
    DEPRECATED: Use apply_rsi_filter_to_position instead.
    Apply Bollinger Bands mean reversion filter to base position.
    """
    if long_only:
        signal = calculate_bollinger_long_only_signal(prices, window, vol_lookback, num_std, signal_strength_method)
    else:
        signal = calculate_bollinger_position_signal(prices, window, vol_lookback, num_std, signal_strength_method)
    
    # Align signal with base_position index and handle NaN values
    signal_aligned = signal.reindex(base_position.index).fillna(0.0)
    
    # Apply signal to base position
    filtered_position = base_position * signal_aligned
    
    # For long_only, ensure no negative positions
    if long_only:
        filtered_position = filtered_position.clip(lower=0.0)
    
    return filtered_position

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

#####   STRATEGY 6: RSI-BASED MEAN REVERSION   #####

def backtest_rsi_mean_reversion_strategy(data_dir='Data', capital=1000000, risk_target=0.2,
                                        short_span=32, long_years=10, min_vol_floor=0.05,
                                        rsi_window=14, oversold=30, overbought=70,
                                        signal_strength_method='linear', long_only=True,
                                        weight_method='handcrafted',
                                        common_hypothetical_SR=0.3, annual_turnover_T=7.0,
                                        start_date=None, end_date=None):
    """
    Backtest Strategy 6: RSI-based mean reversion portfolio with daily dynamic rebalancing.
    
    Implementation follows mean reversion principles: "Buy instruments when their RSI 
    indicates oversold conditions, each with positions scaled for a variable risk 
    estimate and volatility regime."
    
    Uses dynamic position sizing: "positions are continuously managed after opening 
    to ensure their risk is correct and signals remain valid."
    
    Parameters:
        data_dir (str): Directory containing price data files.
        capital (float): Initial capital.
        risk_target (float): Target risk fraction.
        short_span (int): EWMA span for short-run volatility.
        long_years (int): Years for long-run volatility average.
        min_vol_floor (float): Minimum volatility floor.
        rsi_window (int): RSI calculation window (default 14).
        oversold (float): RSI oversold threshold (default 30).
        overbought (float): RSI overbought threshold (default 70).
        signal_strength_method (str): Signal calculation method ('linear', 'sigmoid').
        long_only (bool): Whether to use long-only mean reversion signals.
        weight_method (str): Method for calculating instrument weights.
        common_hypothetical_SR (float): Common hypothetical SR for SR' calculation.
        annual_turnover_T (float): Annual turnover T for SR' calculation.
        start_date (str): Start date for backtest (YYYY-MM-DD).
        end_date (str): End date for backtest (YYYY-MM-DD).
    
    Returns:
        dict: Comprehensive backtest results.
    """
    print("=" * 60)
    print("STRATEGY 6: RSI-BASED MEAN REVERSION")
    print("=" * 60)
    
    # Load FX data
    print("\nLoading FX data...")
    fx_data = load_fx_data(data_dir)
    currency_mapping = get_instrument_currency_mapping()
    
    # Load all instrument data
    all_instruments_specs_df = load_instrument_data()
    raw_instrument_data = load_all_instrument_data(data_dir)
    
    if not raw_instrument_data:
        raise ValueError("No instrument data loaded successfully")
    
    print(f"\nPortfolio Configuration:")
    print(f"  Instruments initially loaded: {len(raw_instrument_data)}")
    print(f"  Capital: ${capital:,.0f}")
    print(f"  Risk Target: {risk_target:.1%}")
    print(f"  Weight Method: {weight_method}")
    print(f"  RSI Window: {rsi_window}")
    print(f"  RSI Oversold: {oversold}")
    print(f"  RSI Overbought: {overbought}")
    print(f"  Signal Method: {signal_strength_method}")
    print(f"  Long Only: {long_only}")

    # Preprocess instruments: calculate returns, volatility forecasts, and Bollinger signals
    processed_instrument_data = {}
    for symbol, df_orig in raw_instrument_data.items():
        df = df_orig.copy()
        if 'Last' not in df.columns:
            continue
        
        df['daily_price_change_pct'] = df['Last'].pct_change()
        
        # Calculate volatility forecast
        raw_returns_for_vol = df['daily_price_change_pct'].dropna()
        if len(raw_returns_for_vol) < max(short_span, rsi_window, 50):
            continue

        blended_vol_series = calculate_blended_volatility(
            raw_returns_for_vol, short_span=short_span, long_years=long_years, min_vol_floor=min_vol_floor
        )
        df['vol_forecast'] = blended_vol_series.shift(1).reindex(df.index).ffill().fillna(min_vol_floor)
        
        # Calculate RSI signals (no shift needed - signals use current and past prices)
        if long_only:
            df['rsi_signal'] = calculate_rsi_long_only_signal(
                df['Last'], rsi_window, oversold, overbought, signal_strength_method
            )
        else:
            df['rsi_signal'] = calculate_rsi_position_signal(
                df['Last'], rsi_window, oversold, overbought, signal_strength_method
            )
        
        # Store RSI for analysis
        df['rsi'] = calculate_rsi(df['Last'], window=rsi_window)
        
        # Ensure all critical data is present
        df.dropna(subset=['Last', 'vol_forecast', 'rsi_signal', 'daily_price_change_pct'], inplace=True)
        if df.empty:
            continue

        processed_instrument_data[symbol] = df

    if not processed_instrument_data:
        raise ValueError("No instruments remaining after preprocessing.")
    
    print(f"  Instruments after preprocessing: {len(processed_instrument_data)}")

    # Determine common date range for backtest
    all_indices = [df.index for df in processed_instrument_data.values() if not df.empty]
    if not all_indices:
        raise ValueError("No valid instrument data to determine date range.")

    global_min_date = min(idx.min() for idx in all_indices)
    global_max_date = max(idx.max() for idx in all_indices)
    
    if start_date:
        user_start_dt = pd.to_datetime(start_date)
        backtest_start_dt = max(user_start_dt, global_min_date)
    else:
        backtest_start_dt = global_min_date
    
    if end_date:
        user_end_dt = pd.to_datetime(end_date)
        backtest_end_dt = min(user_end_dt, global_max_date)
    else:
        backtest_end_dt = global_max_date

    if backtest_start_dt >= backtest_end_dt:
        raise ValueError(f"Invalid backtest period: Start {backtest_start_dt}, End {backtest_end_dt}")

    trading_days_range = pd.bdate_range(start=backtest_start_dt, end=backtest_end_dt)
    
    print(f"\nBacktest Period:")
    print(f"  Start: {trading_days_range.min().date()}")
    print(f"  End: {trading_days_range.max().date()}")
    print(f"  Duration: {len(trading_days_range)} trading days")

    # Initialize portfolio tracking
    current_portfolio_equity = capital
    portfolio_daily_records = []
    known_eligible_instruments = set()
    weights = {} 
    idm = 1.0
    previous_positions = {}

    # Main time-stepping loop
    for idx, current_date in enumerate(trading_days_range):
        if idx == 0:
            # Initialize on first day
            record = {'date': current_date, 'total_pnl': 0.0, 'total_costs': 0.0, 'portfolio_return': 0.0, 
                      'equity_sod': current_portfolio_equity, 'equity_eod': current_portfolio_equity}
            for symbol_k in processed_instrument_data.keys(): 
                record[f'{symbol_k}_contracts'] = 0.0
                record[f'{symbol_k}_signal'] = 0.0
            portfolio_daily_records.append(record)
            continue
        
        previous_date = trading_days_range[idx-1]
        capital_at_start = current_portfolio_equity
        daily_total_pnl = 0.0
        daily_total_costs = 0.0
        current_day_positions = {}

        effective_data_cutoff_date = previous_date

        # Determine eligible instruments
        current_iteration_eligible_instruments = set()
        for s, df_full in processed_instrument_data.items():
            df_upto_cutoff = df_full[df_full.index <= effective_data_cutoff_date]
            if not df_upto_cutoff.empty and len(df_upto_cutoff) > max(short_span, rsi_window):
                current_iteration_eligible_instruments.add(s)

        # Handle reweighting when new instruments become available
        perform_reweight = False
        if idx <= 5 or len(current_iteration_eligible_instruments) > len(known_eligible_instruments):
            if idx <= 5:
                print(f"Initial reweighting for date: {current_date.date()}")
            else:
                newly_added = current_iteration_eligible_instruments - known_eligible_instruments
                print(f"Reweighting for date: {current_date.date()} due to new instruments: {newly_added}")
            perform_reweight = True
        
        if perform_reweight:
            known_eligible_instruments = current_iteration_eligible_instruments.copy()
            
            data_for_reweighting = {}
            for s_eligible in known_eligible_instruments:
                df_historical_slice = processed_instrument_data[s_eligible][
                    processed_instrument_data[s_eligible].index <= effective_data_cutoff_date]
                if not df_historical_slice.empty:
                     data_for_reweighting[s_eligible] = df_historical_slice
            
            if data_for_reweighting:
                weights = calculate_instrument_weights(
                    data_for_reweighting, 
                    weight_method, 
                    all_instruments_specs_df,
                    common_hypothetical_SR,
                    annual_turnover_T,
                    risk_target,
                    capital=current_portfolio_equity,
                    fx_data=fx_data,
                    currency_mapping=currency_mapping,
                    filter_by_capital=True,
                    assumed_num_instruments=10
                )
                
                num_weighted_instruments = sum(1 for w_val in weights.values() if w_val > 1e-6)
                idm = calculate_idm_from_count(num_weighted_instruments)
                if idx <= 5:
                    print(f"  IDM: {idm:.2f} based on {num_weighted_instruments} instruments")

        # Process each instrument
        for symbol, df_instrument in processed_instrument_data.items():
            try:
                instrument_specs = get_instrument_specs(symbol, all_instruments_specs_df)
                instrument_multiplier = instrument_specs['multiplier']
                instrument_weight = weights.get(symbol, 0.0)
                sr_cost = instrument_specs.get('sr_cost', 0.01)

                if instrument_weight == 0.0:
                    current_day_positions[symbol] = 0.0
                    continue

                # Get data for sizing and P&L calculation
                price_for_sizing = df_instrument.loc[previous_date, 'Last']
                vol_for_sizing = df_instrument.loc[current_date, 'vol_forecast']
                rsi_signal = df_instrument.loc[current_date, 'rsi_signal']
                
                price_at_start = df_instrument.loc[previous_date, 'Last']
                price_at_end = df_instrument.loc[current_date, 'Last']
                
                if pd.isna(price_for_sizing) or pd.isna(vol_for_sizing) or pd.isna(rsi_signal):
                    num_contracts = 0.0
                    instrument_pnl_today = 0.0
                    trading_cost_today = 0.0
                else:
                    vol_for_sizing = max(vol_for_sizing, min_vol_floor)
                    
                    # Get FX rate
                    instrument_currency = currency_mapping.get(symbol, 'USD')
                    fx_rate = get_fx_rate_for_date_and_currency(current_date, instrument_currency, fx_data)
                    
                    if fx_rate is None:
                        num_contracts = 0.0
                        instrument_pnl_today = 0.0
                        trading_cost_today = 0.0
                    else:
                        # Calculate base position
                        base_position = calculate_portfolio_position_size(
                            symbol=symbol, capital=capital_at_start, weight=instrument_weight, idm=idm,
                            price=price_for_sizing, volatility=vol_for_sizing, multiplier=instrument_multiplier,
                            risk_target=risk_target, fx_rate=fx_rate
                        )
                        
                        # Apply RSI signal
                        num_contracts = base_position * rsi_signal
                        num_contracts = round(num_contracts)  # Round to whole contracts
                        
                        # Calculate P&L
                        price_change_in_local = price_at_end - price_at_start
                        price_change_in_base = price_change_in_local * fx_rate
                        instrument_pnl_today = num_contracts * instrument_multiplier * price_change_in_base
                        
                        # Calculate trading costs
                        prev_position = previous_positions.get(symbol, 0.0)
                        trade_size = calculate_position_change(prev_position, num_contracts)
                        trading_cost_today = calculate_trading_cost_from_sr(
                            symbol, trade_size, price_for_sizing, vol_for_sizing, 
                            instrument_multiplier, sr_cost, capital_at_start, fx_rate
                        )
            
            except KeyError:
                num_contracts = 0.0
                instrument_pnl_today = 0.0
                trading_cost_today = 0.0
                rsi_signal = 0.0
            
            current_day_positions[symbol] = num_contracts
            daily_total_pnl += instrument_pnl_today
            daily_total_costs += trading_cost_today

        # Update portfolio equity
        portfolio_return = (daily_total_pnl - daily_total_costs) / capital_at_start if capital_at_start > 0 else 0.0
        current_portfolio_equity = capital_at_start * (1 + portfolio_return)

        # Record daily results
        record = {'date': current_date, 'total_pnl': daily_total_pnl, 'total_costs': daily_total_costs,
                  'portfolio_return': portfolio_return, 'equity_sod': capital_at_start, 
                  'equity_eod': current_portfolio_equity}
        
        for symbol_k in processed_instrument_data.keys(): 
            record[f'{symbol_k}_contracts'] = current_day_positions.get(symbol_k, 0.0)
            try:
                record[f'{symbol_k}_signal'] = processed_instrument_data[symbol_k].loc[current_date, 'rsi_signal']
            except:
                record[f'{symbol_k}_signal'] = 0.0
        
        portfolio_daily_records.append(record)
        
        # Update previous positions for next iteration
        previous_positions = current_day_positions.copy()

    # Post-processing
    if not portfolio_daily_records:
        raise ValueError("No daily records generated during backtest.")
        
    portfolio_df = pd.DataFrame(portfolio_daily_records)
    portfolio_df.set_index('date', inplace=True)
    
    print(f"\nBacktest completed: {len(portfolio_df)} daily records generated.")
    
    # Calculate performance metrics
    account_curve = build_account_curve(portfolio_df['portfolio_return'], capital)
    performance = calculate_comprehensive_performance(account_curve, portfolio_df['portfolio_return'])
    
    # Add strategy-specific metrics
    performance['num_instruments'] = len(processed_instrument_data)
    performance['idm'] = idm
    performance['weight_method'] = weight_method
    performance['backtest_start'] = trading_days_range.min()
    performance['backtest_end'] = trading_days_range.max()
    performance['rsi_window'] = rsi_window
    performance['oversold'] = oversold
    performance['overbought'] = overbought
    performance['signal_strength_method'] = signal_strength_method
    performance['long_only'] = long_only
    performance['avg_daily_costs'] = portfolio_df['total_costs'].mean()
    performance['total_costs'] = portfolio_df['total_costs'].sum()

    # Calculate signal statistics
    signal_cols = [col for col in portfolio_df.columns if col.endswith('_signal')]
    if signal_cols:
        all_signals = portfolio_df[signal_cols].values.flatten()
        all_signals = all_signals[~pd.isna(all_signals)]
        performance['avg_signal_strength'] = np.mean(np.abs(all_signals))
        performance['signal_utilization'] = np.mean(all_signals != 0)

    return {
        'portfolio_data': portfolio_df,
        'processed_instruments': list(processed_instrument_data.keys()),
        'performance': performance,
        'weights': weights,
        'idm': idm,
        'config': {
            'capital': capital, 'risk_target': risk_target, 'short_span': short_span, 
            'long_years': long_years, 'min_vol_floor': min_vol_floor, 
            'rsi_window': rsi_window, 'oversold': oversold, 'overbought': overbought,
            'signal_strength_method': signal_strength_method, 'long_only': long_only,
            'weight_method': weight_method, 'backtest_start': trading_days_range.min(), 
            'backtest_end': trading_days_range.max(),
            'common_hypothetical_SR': common_hypothetical_SR,
            'annual_turnover_T': annual_turnover_T
        }
    } 

# Add backward compatibility function
def backtest_bollinger_mean_reversion_strategy(*args, **kwargs):
    """Backward compatibility wrapper - redirects to RSI strategy."""
    print("Note: Bollinger Bands strategy has been replaced with superior RSI strategy")
    return backtest_rsi_mean_reversion_strategy(*args, **kwargs)

#####   PERFORMANCE ANALYSIS AND PLOTTING FUNCTIONS   #####

def analyze_rsi_mean_reversion_results(results):
    """
    Analyze and display comprehensive RSI mean reversion strategy results.
    
    Parameters:
        results (dict): Results from backtest_bollinger_mean_reversion_strategy.
    """
    performance = results['performance']
    config = results['config']
    portfolio_df = results['portfolio_data']
    
    print("\n" + "=" * 60)
    print("STRATEGY 6: RSI MEAN REVERSION PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Overall Performance Metrics
    print(f"\n--- Overall Performance ---")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Annualized Return: {performance['annualized_return']:.2%}")
    print(f"Annualized Volatility: {performance['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {performance['max_drawdown_pct']:.1f}%")
    print(f"Skewness: {performance.get('skewness', 'N/A'):.3f}")
    
    # Strategy-Specific Metrics
    print(f"\n--- RSI Strategy Metrics ---")
    print(f"RSI Window: {performance.get('rsi_window', 'N/A')} days")
    print(f"RSI Oversold: {performance.get('oversold', 'N/A')}")
    print(f"RSI Overbought: {performance.get('overbought', 'N/A')}")
    print(f"Signal Method: {performance.get('signal_strength_method', 'N/A')}")
    print(f"Long Only: {performance.get('long_only', 'N/A')}")
    print(f"Average Signal Strength: {performance.get('avg_signal_strength', 'N/A'):.3f}")
    print(f"Signal Utilization: {performance.get('signal_utilization', 'N/A'):.1%}")
    
    # Portfolio Characteristics
    print(f"\n--- Portfolio Characteristics ---")
    print(f"Number of Instruments: {performance.get('num_instruments', 'N/A')}")
    print(f"IDM: {performance.get('idm', 'N/A'):.2f}")
    print(f"Weight Method: {performance.get('weight_method', 'N/A')}")
    print(f"Capital: ${config['capital']:,.0f}")
    print(f"Risk Target: {config['risk_target']:.1%}")
    
    # Cost Analysis
    print(f"\n--- Trading Cost Analysis ---")
    print(f"Total Trading Costs: ${performance.get('total_costs', 0):,.2f}")
    print(f"Average Daily Costs: ${performance.get('avg_daily_costs', 0):,.2f}")
    cost_as_pct_return = (performance.get('total_costs', 0) / (config['capital'] * performance['total_return'])) * 100 if performance['total_return'] != 0 else 0
    print(f"Costs as % of Total Return: {cost_as_pct_return:.2f}%")
    
    # Calculate position statistics
    position_cols = [col for col in portfolio_df.columns if col.endswith('_contracts')]
    if position_cols:
        portfolio_df['total_contracts'] = portfolio_df[position_cols].abs().sum(axis=1)
        portfolio_df['num_active_positions'] = portfolio_df[position_cols].ne(0).sum(axis=1)
        
        print(f"\n--- Position Statistics ---")
        print(f"Average Active Positions: {portfolio_df['num_active_positions'].mean():.1f}")
        print(f"Max Active Positions: {portfolio_df['num_active_positions'].max()}")
        print(f"Average Total Contracts: {portfolio_df['total_contracts'].mean():.1f}")
        print(f"Max Total Contracts: {portfolio_df['total_contracts'].max():.0f}")
    
    # Signal Analysis
    signal_cols = [col for col in portfolio_df.columns if col.endswith('_signal')]
    if signal_cols:
        print(f"\n--- Signal Analysis ---")
        print(f"Signal Columns Available: {len(signal_cols)}")
        
        # Calculate signal statistics per instrument
        for col in signal_cols[:5]:  # Show top 5 instruments
            signal_data = portfolio_df[col].dropna()
            if len(signal_data) > 0:
                symbol = col.replace('_signal', '')
                avg_signal = signal_data.mean()
                signal_utilization = (signal_data != 0).mean()
                print(f"  {symbol}: Avg Signal: {avg_signal:.3f}, Utilization: {signal_utilization:.1%}")
    
    print(f"\nBacktest Period: {config['backtest_start'].date()} to {config['backtest_end'].date()}")

def plot_rsi_mean_reversion_equity_curve(results, save_path='results/strategy6_rsi_mean_reversion.png'):
    """
    Plot RSI mean reversion strategy equity curve and key metrics.
    
    Parameters:
        results (dict): Results from backtest function.
        save_path (str): Path to save the plot.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        portfolio_df = results['portfolio_data']
        config = results['config']
        performance = results['performance']
        
        equity_curve = build_account_curve(portfolio_df['portfolio_return'], config['capital'])
        
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Equity curve
        plt.subplot(4, 1, 1)
        plt.plot(equity_curve.index, equity_curve.values/1e6, 'g-', linewidth=1.5, 
                label='RSI Mean Reversion Strategy')
        plt.title('Strategy 6: RSI-Based Mean Reversion', fontsize=14, fontweight='bold')
        plt.ylabel('Portfolio Value ($M)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Drawdown
        plt.subplot(4, 1, 2)
        drawdown_stats = calculate_maximum_drawdown(equity_curve)
        drawdown_series = drawdown_stats['drawdown_series'] * 100
        
        plt.fill_between(drawdown_series.index, drawdown_series.values, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        plt.plot(drawdown_series.index, drawdown_series.values, 'r-', linewidth=1)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.title('Portfolio Drawdown', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 3: Signal strength and position activity
        plt.subplot(4, 1, 3)
        
        # Calculate average signal strength per day
        signal_cols = [col for col in portfolio_df.columns if col.endswith('_signal')]
        if signal_cols:
            portfolio_df['avg_signal_strength'] = portfolio_df[signal_cols].abs().mean(axis=1)
            portfolio_df['signal_utilization'] = (portfolio_df[signal_cols] != 0).mean(axis=1)
            
            # Plot 30-day rolling average of signal metrics
            rolling_signal = portfolio_df['avg_signal_strength'].rolling(30).mean()
            rolling_utilization = portfolio_df['signal_utilization'].rolling(30).mean()
            
            plt.plot(rolling_signal.index, rolling_signal.values, 'b-', linewidth=1, 
                    label='Avg Signal Strength (30d MA)', alpha=0.7)
            plt.plot(rolling_utilization.index, rolling_utilization.values, 'orange', linewidth=1, 
                    label='Signal Utilization (30d MA)', alpha=0.7)
        
        plt.ylabel('Signal Metrics', fontsize=12)
        plt.title('Signal Strength and Utilization', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 4: Trading costs and active positions
        plt.subplot(4, 1, 4)
        
        # Plot cumulative costs
        cumulative_costs = portfolio_df['total_costs'].cumsum()
        plt.plot(cumulative_costs.index, cumulative_costs.values/1000, 'purple', linewidth=1, 
                label='Cumulative Costs ($K)')
        
        # Plot number of active positions (right y-axis)
        position_cols = [col for col in portfolio_df.columns if col.endswith('_contracts')]
        if position_cols:
            portfolio_df['num_active_positions'] = (portfolio_df[position_cols] != 0).sum(axis=1)
            
            ax2 = plt.gca().twinx()
            ax2.plot(portfolio_df.index, portfolio_df['num_active_positions'], 'green', 
                    linewidth=1, alpha=0.6, label='Active Positions')
            ax2.set_ylabel('Number of Active Positions', fontsize=12, color='green')
            ax2.tick_params(axis='y', labelcolor='green')
        
        plt.ylabel('Cumulative Costs ($K)', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.title('Trading Costs and Portfolio Activity', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        
        # Format x-axis dates for all subplots
        for ax in plt.gcf().get_axes()[:4]:  # Only main axes, not twin axes
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add performance metrics as text
        start_date = config['backtest_start'].strftime('%Y-%m-%d')
        end_date = config['backtest_end'].strftime('%Y-%m-%d')
        
        textstr = f'''Performance Summary:
Total Return: {performance['total_return']:.1%}
Annualized Return: {performance['annualized_return']:.1%}
Volatility: {performance['annualized_volatility']:.1%}
Sharpe Ratio: {performance['sharpe_ratio']:.3f}
Max Drawdown: {performance['max_drawdown_pct']:.1f}%
Avg Signal Strength: {performance.get('avg_signal_strength', 0):.3f}
Signal Utilization: {performance.get('signal_utilization', 0):.1%}
Total Costs: ${performance.get('total_costs', 0):,.0f}
Period: {start_date} to {end_date}'''
        
        plt.figtext(0.02, 0.02, textstr, fontsize=9, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ RSI mean reversion strategy results saved to: {save_path}")
        
    except Exception as e:
        print(f"Error plotting equity curve: {e}")
        import traceback
        traceback.print_exc()

def plot_bollinger_strategy_comparison(bollinger_results, baseline_results, 
                                     save_path='results/strategy6_vs_baseline_comparison.png'):
    """
    Compare Bollinger Bands strategy with baseline (typically Strategy 4).
    
    Parameters:
        bollinger_results (dict): Results from Bollinger Bands strategy.
        baseline_results (dict): Results from baseline strategy.
        save_path (str): Path to save comparison plot.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Build equity curves
        bb_equity = build_account_curve(bollinger_results['portfolio_data']['portfolio_return'], 
                                       bollinger_results['config']['capital'])
        baseline_equity = build_account_curve(baseline_results['portfolio_data']['portfolio_return'], 
                                            baseline_results['config']['capital'])
        
        # Align dates for comparison
        common_dates = bb_equity.index.intersection(baseline_equity.index)
        if len(common_dates) < 252:
            print("Warning: Limited overlapping data for comparison")
        
        bb_aligned = bb_equity.reindex(common_dates)
        baseline_aligned = baseline_equity.reindex(common_dates)
        
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Equity curves comparison
        plt.subplot(3, 1, 1)
        plt.plot(bb_aligned.index, bb_aligned.values/1e6, 'g-', linewidth=1.5, 
                label='Bollinger Bands Mean Reversion')
        plt.plot(baseline_aligned.index, baseline_aligned.values/1e6, 'b-', linewidth=1.5, 
                label='Baseline Strategy')
        plt.title('Strategy Comparison: Equity Curves', fontsize=14, fontweight='bold')
        plt.ylabel('Portfolio Value ($M)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Relative performance
        plt.subplot(3, 1, 2)
        relative_performance = (bb_aligned / baseline_aligned - 1) * 100
        plt.plot(relative_performance.index, relative_performance.values, 'purple', linewidth=1.5)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('Relative Performance: Bollinger Bands vs Baseline', fontsize=12, fontweight='bold')
        plt.ylabel('Outperformance (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Rolling Sharpe ratio comparison
        plt.subplot(3, 1, 3)
        window = 252  # 1-year rolling window
        
        bb_returns = bollinger_results['portfolio_data']['portfolio_return'].reindex(common_dates)
        baseline_returns = baseline_results['portfolio_data']['portfolio_return'].reindex(common_dates)
        
        bb_rolling_sharpe = bb_returns.rolling(window).mean() / bb_returns.rolling(window).std() * np.sqrt(252)
        baseline_rolling_sharpe = baseline_returns.rolling(window).mean() / baseline_returns.rolling(window).std() * np.sqrt(252)
        
        plt.plot(bb_rolling_sharpe.index, bb_rolling_sharpe.values, 'g-', linewidth=1.5, 
                label='Bollinger Bands', alpha=0.8)
        plt.plot(baseline_rolling_sharpe.index, baseline_rolling_sharpe.values, 'b-', linewidth=1.5, 
                label='Baseline', alpha=0.8)
        plt.title(f'Rolling Sharpe Ratio Comparison ({window}-day window)', fontsize=12, fontweight='bold')
        plt.ylabel('Sharpe Ratio', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis dates
        for ax in plt.gcf().get_axes():
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add comparison statistics
        bb_perf = bollinger_results['performance']
        baseline_perf = baseline_results['performance']
        
        comparison_text = f'''Performance Comparison Summary:
                    Bollinger Bands    Baseline      Difference
Total Return:       {bb_perf['total_return']:.1%}        {baseline_perf['total_return']:.1%}       {bb_perf['total_return'] - baseline_perf['total_return']:.1%}
Ann. Return:        {bb_perf['annualized_return']:.1%}        {baseline_perf['annualized_return']:.1%}       {bb_perf['annualized_return'] - baseline_perf['annualized_return']:.1%}
Volatility:         {bb_perf['annualized_volatility']:.1%}        {baseline_perf['annualized_volatility']:.1%}       {bb_perf['annualized_volatility'] - baseline_perf['annualized_volatility']:.1%}
Sharpe Ratio:       {bb_perf['sharpe_ratio']:.3f}         {baseline_perf['sharpe_ratio']:.3f}        {bb_perf['sharpe_ratio'] - baseline_perf['sharpe_ratio']:.3f}
Max Drawdown:       {bb_perf['max_drawdown_pct']:.1f}%         {baseline_perf['max_drawdown_pct']:.1f}%        {bb_perf['max_drawdown_pct'] - baseline_perf['max_drawdown_pct']:.1f}%'''
        
        plt.figtext(0.02, 0.02, comparison_text, fontsize=8, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Strategy comparison saved to: {save_path}")
        
    except Exception as e:
        print(f"Error plotting strategy comparison: {e}")
        import traceback
        traceback.print_exc()

def compare_with_baseline_strategy(bollinger_results=None, capital=1000000, risk_target=0.2, 
                                 start_date=None, end_date=None, weight_method='handcrafted'):
    """
    Compare Bollinger Bands strategy with baseline Strategy 4.
    
    Parameters:
        bollinger_results (dict): Results from Bollinger Bands strategy.
        capital (float): Capital for baseline comparison.
        risk_target (float): Risk target for baseline.
        start_date (str): Start date for baseline backtest.
        end_date (str): End date for baseline backtest.
        weight_method (str): Weight method for baseline.
    
    Returns:
        tuple: (bollinger_results, baseline_results)
    """
    print(f"\n" + "=" * 60)
    print("COMPARING BOLLINGER BANDS STRATEGY WITH BASELINE (STRATEGY 4)")
    print("=" * 60)
    
    # Run baseline strategy if not provided via bollinger_results
    if bollinger_results is None:
        print("Running Bollinger Bands strategy first...")
        bollinger_results = backtest_bollinger_mean_reversion_strategy(
            capital=capital, risk_target=risk_target, start_date=start_date, 
            end_date=end_date, weight_method=weight_method
        )
    
    # Run baseline strategy with same parameters and date range
    print("Running baseline Strategy 4 for comparison...")
    baseline_results = backtest_multi_instrument_strategy(
        capital=capital, risk_target=risk_target, 
        start_date=start_date or bollinger_results['config']['backtest_start'].strftime('%Y-%m-%d'),
        end_date=end_date or bollinger_results['config']['backtest_end'].strftime('%Y-%m-%d'),
        weight_method=weight_method
    )
    
    # Display comparison
    bb_perf = bollinger_results['performance']
    baseline_perf = baseline_results['performance']
    
    print(f"\nPerformance Comparison:")
    print(f"{'Metric':<25} {'Bollinger Bands':<18} {'Baseline':<15} {'Difference':<12}")
    print("-" * 75)
    print(f"{'Total Return':<25} {bb_perf['total_return']:<18.2%} {baseline_perf['total_return']:<15.2%} {bb_perf['total_return'] - baseline_perf['total_return']:<12.2%}")
    print(f"{'Annualized Return':<25} {bb_perf['annualized_return']:<18.2%} {baseline_perf['annualized_return']:<15.2%} {bb_perf['annualized_return'] - baseline_perf['annualized_return']:<12.2%}")
    print(f"{'Volatility':<25} {bb_perf['annualized_volatility']:<18.2%} {baseline_perf['annualized_volatility']:<15.2%} {bb_perf['annualized_volatility'] - baseline_perf['annualized_volatility']:<12.2%}")
    print(f"{'Sharpe Ratio':<25} {bb_perf['sharpe_ratio']:<18.3f} {baseline_perf['sharpe_ratio']:<15.3f} {bb_perf['sharpe_ratio'] - baseline_perf['sharpe_ratio']:<12.3f}")
    print(f"{'Max Drawdown':<25} {bb_perf['max_drawdown_pct']:<18.1f}% {baseline_perf['max_drawdown_pct']:<15.1f}% {bb_perf['max_drawdown_pct'] - baseline_perf['max_drawdown_pct']:<12.1f}%")
    
    # Generate comparison plots
    plot_bollinger_strategy_comparison(bollinger_results, baseline_results)
    
    return bollinger_results, baseline_results

#####   CONFIGURATION AND MAIN FUNCTION   #####

# ===========================================
# CONFIGURATION - MODIFY THESE AS NEEDED
# ===========================================
CAPITAL = 1000000               # Starting capital
START_DATE = '2018-01-01'       # Backtest start date or None
END_DATE = '2020-01-01'         # Backtest end date or None
RISK_TARGET = 0.2               # 20% annual risk target
WEIGHT_METHOD = 'handcrafted'   # 'equal', 'vol_inverse', 'handcrafted'

# RSI Strategy-specific parameters
RSI_WINDOW = 14                # RSI calculation window
OVERSOLD = 30                  # RSI oversold threshold
OVERBOUGHT = 70                # RSI overbought threshold
SIGNAL_STRENGTH_METHOD = 'linear'  # 'linear' or 'sigmoid'
LONG_ONLY = True               # Long-only mean reversion

def main():
    """
    Test RSI mean reversion strategy implementation with comprehensive analysis.
    """
    try:
        print(f"\n" + "=" * 60)
        print("RUNNING STRATEGY 6: RSI-BASED MEAN REVERSION")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Capital: ${CAPITAL:,}")
        print(f"  Date Range: {START_DATE or 'earliest'} to {END_DATE or 'latest'}")
        print(f"  Risk Target: {RISK_TARGET:.1%}")
        print(f"  Weight Method: {WEIGHT_METHOD}")
        print(f"  RSI Window: {RSI_WINDOW}")
        print(f"  RSI Oversold: {OVERSOLD}")
        print(f"  RSI Overbought: {OVERBOUGHT}")
        print(f"  Signal Method: {SIGNAL_STRENGTH_METHOD}")
        print(f"  Long Only: {LONG_ONLY}")
        print("=" * 60)
        
        # Run backtest
        results = backtest_rsi_mean_reversion_strategy(
            data_dir='Data',
            capital=CAPITAL,
            risk_target=RISK_TARGET,
            short_span=32,
            long_years=10,
            min_vol_floor=0.05,
            rsi_window=RSI_WINDOW,
            oversold=OVERSOLD,
            overbought=OVERBOUGHT,
            signal_strength_method=SIGNAL_STRENGTH_METHOD,
            long_only=LONG_ONLY,
            weight_method=WEIGHT_METHOD,
            common_hypothetical_SR=0.3,
            annual_turnover_T=7.0,
            start_date=START_DATE,
            end_date=END_DATE
        )
        
        # Analyze results
        if results:
            analyze_rsi_mean_reversion_results(results)
            plot_rsi_mean_reversion_equity_curve(results)
            
            print(f"\n✅ RSI mean reversion strategy analysis completed!")
        else:
            print("RSI strategy backtest did not produce results.")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    
    return results

if __name__ == "__main__":
    main() 