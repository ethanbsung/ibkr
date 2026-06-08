from rob_port.chapter4 import *  # For multi-instrument infrastructure
from rob_port.chapter3 import *  # For volatility forecasting
from rob_port.chapter2 import *  # For position sizing
from rob_port.chapter1 import *  # For basic calculations
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')

#####   STRATEGY: BREAKOUT CROSS-TIMEFRAME STRATEGY   #####

def calculate_breakout_signals_multi_period(prices: pd.Series, periods: List[int] = [10, 25, 50, 75, 100]) -> pd.DataFrame:
    """
    Calculate breakout signals for multiple periods.
    
    This strategy diversifies across timeframes by using different lookback periods for breakout detection.
    It goes long when price breaks above the rolling maximum of the specified periods.
    
    Parameters:
        prices (pd.Series): Price series.
        periods (List[int]): List of lookback periods for breakout detection.
    
    Returns:
        pd.DataFrame: DataFrame with breakout signals for each period.
    """
    signals_df = pd.DataFrame(index=prices.index)
    
    for period in periods:
        # Calculate rolling maximum over the period (excluding current day to avoid lookahead bias)
        rolling_high = prices.shift(1).rolling(window=period, min_periods=period).max()
        
        # Generate breakout signal: 1 if price breaks above rolling high, 0 otherwise
        breakout_signal = (prices > rolling_high).astype(int)
        
        # Store in DataFrame
        signals_df[f'breakout_{period}d'] = breakout_signal
    
    return signals_df

def calculate_combined_breakout_signal(signals_df: pd.DataFrame, combination_method: str = 'average') -> pd.Series:
    """
    Combine multiple breakout signals into a single signal.
    
    Parameters:
        signals_df (pd.DataFrame): DataFrame with individual breakout signals.
        combination_method (str): Method to combine signals ('average', 'max', 'weighted_average').
    
    Returns:
        pd.Series: Combined breakout signal (0 to 1).
    """
    if combination_method == 'average':
        # Simple average of all signals
        combined_signal = signals_df.mean(axis=1)
    
    elif combination_method == 'max':
        # Take maximum signal (most aggressive)
        combined_signal = signals_df.max(axis=1)
    
    elif combination_method == 'weighted_average':
        # Weight shorter periods more heavily (they react faster)
        periods = [int(col.split('_')[1].replace('d', '')) for col in signals_df.columns]
        # Inverse weighting: shorter periods get higher weights
        weights = [1.0 / period for period in periods]
        weights = np.array(weights) / sum(weights)  # Normalize to sum to 1
        
        combined_signal = (signals_df * weights).sum(axis=1)
    
    else:
        # Default to average
        combined_signal = signals_df.mean(axis=1)
    
    return combined_signal

def calculate_breakout_position_size(symbol: str, capital: float, weight: float, idm: float,
                                   price: float, volatility: float, multiplier: float,
                                   breakout_signal: float, risk_target: float = 0.2, fx_rate: float = 1.0) -> int:
    """
    Calculate position size for breakout strategy.
    
    Formula: Base position (from Strategy 4) × Breakout signal strength
    
    Parameters:
        symbol (str): Instrument symbol.
        capital (float): Current trading capital.
        weight (float): Instrument weight in portfolio.
        idm (float): Instrument Diversification Multiplier.
        price (float): Current price.
        volatility (float): Annualized volatility forecast.
        multiplier (float): Contract multiplier.
        breakout_signal (float): Breakout signal strength (0 to 1).
        risk_target (float): Target risk fraction.
        fx_rate (float): FX rate for currency conversion.
    
    Returns:
        int: Number of contracts (rounded to nearest integer).
    """
    if np.isnan(volatility) or volatility <= 0 or np.isnan(breakout_signal):
        return 0
    
    # Calculate base position using Strategy 4 methodology
    base_position = calculate_portfolio_position_size(
        symbol=symbol, capital=capital, weight=weight, idm=idm,
        price=price, volatility=volatility, multiplier=multiplier,
        risk_target=risk_target, fx_rate=fx_rate
    )
    
    # Apply breakout signal scaling
    scaled_position = base_position * breakout_signal
    
    # Round to nearest integer (can only hold whole contracts)
    return round(scaled_position)

def calculate_position_change(previous_position: float, new_position: float) -> float:
    """Calculate the change in position size for cost calculation."""
    return abs(new_position - previous_position)

def calculate_trading_cost_from_sr(symbol: str, trade_size: float, price: float, volatility: float,
                                  multiplier: float, sr_cost: float, capital: float, fx_rate: float = 1.0) -> float:
    """
    Calculate trading cost using SR_cost methodology.
    
    Parameters:
        symbol (str): Instrument symbol.
        trade_size (float): Number of contracts traded.
        price (float): Price at trade.
        volatility (float): Annualized volatility.
        multiplier (float): Contract multiplier.
        sr_cost (float): SR cost from instruments.csv.
        capital (float): Current capital.
        fx_rate (float): FX rate.
    
    Returns:
        float: Trading cost in base currency.
    """
    if trade_size == 0 or sr_cost == 0:
        return 0.0
    
    # Calculate notional exposure per contract
    notional_per_contract = price * multiplier * fx_rate
    
    # Calculate cost per contract using SR methodology
    cost_per_contract = sr_cost * volatility * notional_per_contract
    
    # Total cost for the trade
    total_cost = abs(trade_size) * cost_per_contract
    
    return total_cost

def backtest_breakout_cts_strategy(data_dir='Data', capital=1000000, risk_target=0.2,
                                 short_span=32, long_years=10, min_vol_floor=0.05,
                                 breakout_periods=[10, 25, 50, 75, 100],
                                 signal_combination_method='weighted_average',
                                 weight_method='handcrafted',
                                 common_hypothetical_SR=0.3, annual_turnover_T=7.0,
                                 start_date=None, end_date=None):
    """
    Backtest Breakout Cross-Timeframe Strategy (CTS): Multi-instrument portfolio with breakout signals 
    across multiple timeframes, providing diversification across both instruments and time horizons.
    
    This strategy:
    1. Calculates breakout signals for multiple periods (10, 25, 50, 75, 100 days)
    2. Combines signals to create a composite breakout strength indicator
    3. Goes long when prices break above rolling highs with position size scaled by signal strength
    4. Uses Strategy 4's risk parity framework for multi-instrument portfolio construction
    5. Applies proper volatility forecasting and FX handling
    
    Parameters:
        data_dir (str): Directory containing price data files.
        capital (float): Initial capital.
        risk_target (float): Target risk fraction.
        short_span (int): EWMA span for short-run volatility.
        long_years (int): Years for long-run volatility average.
        min_vol_floor (float): Minimum volatility floor.
        breakout_periods (List[int]): Periods for breakout signal calculation.
        signal_combination_method (str): Method to combine breakout signals.
        weight_method (str): Method for calculating instrument weights.
        common_hypothetical_SR (float): Common hypothetical SR for SR' calculation.
        annual_turnover_T (float): Annual turnover T for SR' calculation.
        start_date (str): Start date for backtest (YYYY-MM-DD).
        end_date (str): End date for backtest (YYYY-MM-DD).
    
    Returns:
        dict: Comprehensive backtest results.
    """
    print("=" * 60)
    print("BREAKOUT CROSS-TIMEFRAME STRATEGY (CTS)")
    print("=" * 60)
    
    # Load FX data for multi-currency support
    print("\nLoading FX data...")
    fx_data = load_fx_data(data_dir)
    currency_mapping = get_instrument_currency_mapping()
    
    # Load instrument specifications and price data
    all_instruments_specs_df = load_instrument_data()
    raw_instrument_data = load_all_instrument_data(data_dir)
    
    if not raw_instrument_data:
        raise ValueError("No instrument data loaded successfully")
    
    print(f"\nStrategy Configuration:")
    print(f"  Instruments initially loaded: {len(raw_instrument_data)}")
    print(f"  Capital: ${capital:,.0f}")
    print(f"  Risk Target: {risk_target:.1%}")
    print(f"  Breakout Periods: {breakout_periods}")
    print(f"  Signal Combination: {signal_combination_method}")
    print(f"  Weight Method: {weight_method}")
    print(f"  Common Hypothetical SR: {common_hypothetical_SR}")
    print(f"  Annual Turnover T: {annual_turnover_T}")

    # Preprocess instruments: Calculate returns, volatility forecasts, and breakout signals
    processed_instrument_data = {}
    max_period = max(breakout_periods)
    
    for symbol, df_orig in raw_instrument_data.items():
        df = df_orig.copy()
        if 'Last' not in df.columns:
            print(f"Skipping {symbol}: 'Last' column missing.")
            continue
        
        df['daily_price_change_pct'] = df['Last'].pct_change()
        
        # Calculate blended volatility forecast (no lookahead bias)
        raw_returns_for_vol = df['daily_price_change_pct'].dropna()
        if len(raw_returns_for_vol) < max(short_span, max_period):
            print(f"Skipping {symbol}: Insufficient data for vol forecast and breakout signals ({len(raw_returns_for_vol)} days).")
            continue

        blended_vol_series = calculate_blended_volatility(
            raw_returns_for_vol, short_span=short_span, long_years=long_years, min_vol_floor=min_vol_floor
        )
        # Shift to prevent lookahead bias
        df['vol_forecast'] = blended_vol_series.shift(1).reindex(df.index).ffill().fillna(min_vol_floor)
        
        # Calculate breakout signals for all periods
        breakout_signals_df = calculate_breakout_signals_multi_period(df['Last'], breakout_periods)
        
        # Combine breakout signals
        df['combined_breakout_signal'] = calculate_combined_breakout_signal(
            breakout_signals_df, signal_combination_method
        )
        
        # Ensure all critical data is present
        df.dropna(subset=['Last', 'vol_forecast', 'daily_price_change_pct', 'combined_breakout_signal'], inplace=True)
        if df.empty:
            print(f"Skipping {symbol}: Empty after dropping NaNs in critical columns.")
            continue

        processed_instrument_data[symbol] = df

    if not processed_instrument_data:
        raise ValueError("No instruments remaining after preprocessing.")
    
    print(f"  Instruments after preprocessing: {len(processed_instrument_data)}")

    # Determine backtest date range
    all_indices = [df.index for df in processed_instrument_data.values() if not df.empty]
    if not all_indices:
        raise ValueError("No valid instrument data to determine date range.")

    # Use user-specified dates or default to data availability
    if start_date:
        backtest_start_dt = pd.to_datetime(start_date)
    else:
        # Default to earliest available data across all instruments
        backtest_start_dt = min(idx.min() for idx in all_indices)
    
    if end_date:
        backtest_end_dt = pd.to_datetime(end_date)
    else:
        # Default to latest available data across all instruments  
        backtest_end_dt = max(idx.max() for idx in all_indices)

    # Validate date range
    if backtest_start_dt >= backtest_end_dt:
        raise ValueError(f"Invalid backtest period: Start {backtest_start_dt}, End {backtest_end_dt}")
    
    # Check if we have any data in the specified range
    data_available_in_range = False
    for df in processed_instrument_data.values():
        if not df.empty and df.index.min() <= backtest_end_dt and df.index.max() >= backtest_start_dt:
            data_available_in_range = True
            break
    
    if not data_available_in_range:
        raise ValueError(f"No instrument data available in the specified date range: {backtest_start_dt} to {backtest_end_dt}")

    # Create trading days range
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
    previous_positions = {}  # Track positions for cost calculation

    # Main time-stepping loop with daily position updates
    for idx, current_date in enumerate(trading_days_range):
        if idx == 0:
            # First day initialization
            record = {
                'date': current_date, 'total_pnl': 0.0, 'portfolio_return': 0.0,
                'equity_sod': current_portfolio_equity, 'equity_eod': current_portfolio_equity,
                'num_active_instruments': 0, 'avg_breakout_signal': 0.0, 'total_trading_costs': 0.0
            }
            for symbol_k in processed_instrument_data.keys():
                record[f'{symbol_k}_contracts'] = 0.0
                record[f'{symbol_k}_signal'] = 0.0
            portfolio_daily_records.append(record)
            continue
        
        previous_trading_date = trading_days_range[idx-1]
        capital_at_start_of_day = current_portfolio_equity
        daily_total_pnl = 0.0
        daily_total_costs = 0.0
        current_day_positions = {}
        num_active_instruments = 0
        daily_breakout_signals = []

        effective_data_cutoff_date = previous_trading_date

        # Determine current period eligible instruments
        current_iteration_eligible_instruments = set()
        for s, df_full in processed_instrument_data.items():
            df_upto_cutoff = df_full[df_full.index <= effective_data_cutoff_date]
            if not df_upto_cutoff.empty and len(df_upto_cutoff) > max(short_span, max_period):
                current_iteration_eligible_instruments.add(s)

        # Check if reweighting is needed
        perform_reweight = False
        if idx == 1:
            perform_reweight = True
            print(f"Performing initial reweighting for date: {current_date.date()}")
        elif len(current_iteration_eligible_instruments) > len(known_eligible_instruments):
            newly_added = current_iteration_eligible_instruments - known_eligible_instruments
            perform_reweight = True
            print(f"Reweighting due to new instruments: {newly_added}")

        if perform_reweight:
            known_eligible_instruments = current_iteration_eligible_instruments.copy()
            
            data_for_reweighting = {}
            for s_eligible in known_eligible_instruments:
                df_historical_slice = processed_instrument_data[s_eligible][
                    processed_instrument_data[s_eligible].index <= effective_data_cutoff_date
                ]
                if not df_historical_slice.empty:
                    data_for_reweighting[s_eligible] = df_historical_slice
            
            if data_for_reweighting:
                weights = calculate_instrument_weights(
                    data_for_reweighting, weight_method, all_instruments_specs_df,
                    common_hypothetical_SR, annual_turnover_T, risk_target,
                    capital=current_portfolio_equity, fx_data=fx_data,
                    currency_mapping=currency_mapping, filter_by_capital=True,
                    assumed_num_instruments=10
                )
                
                num_weighted_instruments = sum(1 for w_val in weights.values() if w_val > 1e-6)
                idm = calculate_idm_from_count(num_weighted_instruments)
                print(f"  New IDM: {idm:.2f} based on {num_weighted_instruments} instruments")

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

            # Get data for sizing and P&L calculation
            try:
                # CRITICAL FIX: Use previous day's signal for position sizing to prevent lookahead bias
                # Position sizing decision is made at start of day T based on information available up to day T-1
                price_for_sizing = df_instrument.loc[previous_trading_date, 'Last']
                vol_for_sizing = df_instrument.loc[current_date, 'vol_forecast'] / np.sqrt(business_days_per_year)
                
                # Use PREVIOUS day's breakout signal for current day's position sizing decision
                # This prevents lookahead bias since we can't know current day's breakout status before the day ends
                try:
                    breakout_signal = df_instrument.loc[previous_trading_date, 'combined_breakout_signal']
                except KeyError:
                    # If previous day's signal not available, use 0 (no position)
                    breakout_signal = 0.0
                
                # Data for P&L calculation (current day's actual price movement)
                price_at_start_of_trading = df_instrument.loc[previous_trading_date, 'Last']
                price_at_end_of_trading = df_instrument.loc[current_date, 'Last']
                
                if (pd.isna(price_for_sizing) or pd.isna(vol_for_sizing) or 
                    pd.isna(price_at_start_of_trading) or pd.isna(price_at_end_of_trading) or
                    pd.isna(breakout_signal)):
                    num_contracts = 0.0
                    instrument_pnl_today = 0.0
                    trading_cost = 0.0
                else:
                    vol_for_sizing = max(vol_for_sizing, min_vol_floor)
                    
                    # Get FX rate
                    instrument_currency = currency_mapping.get(symbol, 'USD')
                    fx_rate = get_fx_rate_for_date_and_currency(current_date, instrument_currency, fx_data)
                    
                    if fx_rate is None:  # Skip KRW instruments
                        num_contracts = 0.0
                        instrument_pnl_today = 0.0
                        trading_cost = 0.0
                    else:
                        # Calculate position size using breakout signal
                        num_contracts = calculate_breakout_position_size(
                            symbol=symbol, capital=capital_at_start_of_day, weight=instrument_weight,
                            idm=idm, price=price_for_sizing, volatility=vol_for_sizing,
                            multiplier=instrument_multiplier, breakout_signal=breakout_signal,
                            risk_target=risk_target, fx_rate=fx_rate
                        )
                        
                        # Calculate P&L
                        price_change_in_local_currency = price_at_end_of_trading - price_at_start_of_trading
                        price_change_in_base_currency = price_change_in_local_currency * fx_rate
                        gross_pnl = num_contracts * instrument_multiplier * price_change_in_base_currency
                        
                        # Calculate trading costs
                        previous_position = previous_positions.get(symbol, 0.0)
                        trade_size = calculate_position_change(previous_position, num_contracts)
                        trading_cost = 0.0
                        
                        if trade_size > 0:
                            sr_cost = specs.get('sr_cost', 0.0)
                            if not pd.isna(sr_cost) and sr_cost > 0:
                                trading_cost = calculate_trading_cost_from_sr(
                                    symbol, trade_size, price_at_start_of_trading,
                                    vol_for_sizing * np.sqrt(business_days_per_year),
                                    instrument_multiplier, sr_cost, capital_at_start_of_day, fx_rate
                                )
                        
                        # Net P&L after costs
                        instrument_pnl_today = gross_pnl
                        
                        # Track signals and active instruments
                        daily_breakout_signals.append(breakout_signal)
                        if abs(num_contracts) > 0.01:
                            num_active_instruments += 1
            
            except KeyError:  # Date not found
                num_contracts = 0.0
                instrument_pnl_today = 0.0
                trading_cost = 0.0
                breakout_signal = 0.0
            
            current_day_positions[symbol] = num_contracts
            daily_total_pnl += instrument_pnl_today
            daily_total_costs += trading_cost
            
            # Update position tracking
            previous_positions[symbol] = num_contracts

        # Update portfolio equity
        net_daily_pnl = daily_total_pnl - daily_total_costs
        portfolio_daily_percentage_return = net_daily_pnl / capital_at_start_of_day if capital_at_start_of_day > 0 else 0.0
        current_portfolio_equity = capital_at_start_of_day * (1 + portfolio_daily_percentage_return)

        # Calculate average breakout signal
        avg_breakout_signal = np.mean(daily_breakout_signals) if daily_breakout_signals else 0.0

        # Record daily results
        record = {
            'date': current_date, 'total_pnl': daily_total_pnl,
            'portfolio_return': portfolio_daily_percentage_return,
            'equity_sod': capital_at_start_of_day, 'equity_eod': current_portfolio_equity,
            'num_active_instruments': num_active_instruments,
            'avg_breakout_signal': avg_breakout_signal,
            'total_trading_costs': daily_total_costs
        }
        
        for symbol_k, contracts_k in current_day_positions.items():
            record[f'{symbol_k}_contracts'] = contracts_k
        
        # Add breakout signals to record (using previous day's signal that was used for position sizing)
        for symbol in processed_instrument_data.keys():
            if symbol not in current_day_positions:
                record[f'{symbol}_contracts'] = 0.0
            try:
                # Record the signal that was actually used for position sizing (previous day's signal)
                signal_value = processed_instrument_data[symbol].loc[previous_trading_date, 'combined_breakout_signal']
                record[f'{symbol}_signal'] = signal_value if not pd.isna(signal_value) else 0.0
            except:
                record[f'{symbol}_signal'] = 0.0
        
        portfolio_daily_records.append(record)

    # Post-processing and performance calculation
    if not portfolio_daily_records:
        raise ValueError("No daily records generated during backtest.")
        
    portfolio_df = pd.DataFrame(portfolio_daily_records)
    portfolio_df.set_index('date', inplace=True)
    
    print(f"Backtest completed: {len(portfolio_df)} daily records generated.")
    
    # Calculate performance metrics
    returns_series = portfolio_df['portfolio_return'].dropna()
    if returns_series.std() == 0:
        print("Warning: Portfolio returns have zero variance.")
    
    account_curve = build_account_curve(returns_series, capital)
    performance = calculate_comprehensive_performance(account_curve, returns_series)
    
    # Add strategy-specific performance metrics
    performance['num_instruments'] = len(processed_instrument_data)
    performance['idm'] = idm
    performance['avg_active_instruments'] = portfolio_df['num_active_instruments'].mean()
    performance['avg_breakout_signal'] = portfolio_df['avg_breakout_signal'].mean()
    performance['total_trading_costs'] = portfolio_df['total_trading_costs'].sum()
    performance['cost_as_pct_of_capital'] = performance['total_trading_costs'] / capital
    performance['weight_method'] = weight_method
    performance['signal_combination_method'] = signal_combination_method
    performance['breakout_periods'] = breakout_periods

    # Calculate instrument statistics
    instrument_stats = {}
    for symbol in processed_instrument_data.keys():
        pos_col = f'{symbol}_contracts'
        signal_col = f'{symbol}_signal'
        
        if pos_col in portfolio_df.columns:
            inst_positions = portfolio_df[pos_col][portfolio_df[pos_col] != 0]
            inst_signals = portfolio_df[signal_col]
            
            if len(inst_positions) > 0:
                instrument_stats[symbol] = {
                    'avg_position': inst_positions.mean(),
                    'weight': weights.get(symbol, 0.0),
                    'active_days': len(inst_positions),
                    'avg_signal': inst_signals.mean(),
                    'max_signal': inst_signals.max(),
                    'signal_above_50pct': (inst_signals > 0.5).mean()
                }

    return {
        'portfolio_data': portfolio_df,
        'performance': performance,
        'instrument_stats': instrument_stats,
        'weights': weights,
        'idm': idm,
        'config': {
            'capital': capital, 'risk_target': risk_target, 'short_span': short_span,
            'long_years': long_years, 'min_vol_floor': min_vol_floor,
            'breakout_periods': breakout_periods, 'signal_combination_method': signal_combination_method,
            'weight_method': weight_method, 'common_hypothetical_SR': common_hypothetical_SR,
            'annual_turnover_T': annual_turnover_T, 'backtest_start': trading_days_range.min(),
            'backtest_end': trading_days_range.max()
        }
    }

def analyze_breakout_cts_results(results):
    """
    Analyze and display comprehensive breakout CTS results.
    
    Parameters:
        results (dict): Results from backtest_breakout_cts_strategy.
    """
    performance = results['performance']
    instrument_stats = results['instrument_stats']
    config = results['config']
    
    print("\n" + "=" * 60)
    print("BREAKOUT CROSS-TIMEFRAME STRATEGY (CTS) PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Overall performance
    print(f"\n--- Overall Portfolio Performance ---")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Annualized Return: {performance['annualized_return']:.2%}")
    print(f"Annualized Volatility: {performance['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {performance['max_drawdown_pct']:.1f}%")
    print(f"Skewness: {performance['skewness']:.3f}")
    
    # Breakout strategy characteristics
    print(f"\n--- Breakout Strategy Characteristics ---")
    print(f"Breakout Periods: {config['breakout_periods']}")
    print(f"Signal Combination Method: {config['signal_combination_method']}")
    print(f"Average Active Instruments: {performance['avg_active_instruments']:.1f}")
    print(f"Average Breakout Signal Strength: {performance['avg_breakout_signal']:.3f}")
    print(f"Total Trading Costs: ${performance['total_trading_costs']:,.0f}")
    print(f"Trading Costs as % of Capital: {performance['cost_as_pct_of_capital']:.3%}")
    
    # Portfolio characteristics
    print(f"\n--- Portfolio Characteristics ---")
    print(f"Number of Instruments: {performance['num_instruments']}")
    print(f"IDM: {performance['idm']:.2f}")
    print(f"Capital: ${config['capital']:,.0f}")
    print(f"Risk Target: {config['risk_target']:.1%}")
    print(f"Weight Method: {config['weight_method']}")
    print(f"Backtest Period: {config['backtest_start'].date()} to {config['backtest_end'].date()}")
    
    # Top performing instruments by signal strength
    print(f"\n--- Top 10 Instruments by Signal Activity ---")
    if instrument_stats:
        sorted_instruments = sorted(instrument_stats.items(), 
                                  key=lambda x: x[1].get('avg_signal', 0), reverse=True)
        
        print(f"{'Symbol':<8} {'Weight':<8} {'AvgSignal':<10} {'MaxSignal':<10} {'ActiveDays':<10} {'Strong%':<8}")
        print("-" * 70)
        
        for symbol, stats in sorted_instruments[:10]:
            print(f"{symbol:<8} {stats['weight']:<8.3f} {stats['avg_signal']:<10.3f} "
                  f"{stats['max_signal']:<10.3f} {stats['active_days']:<10} "
                  f"{stats['signal_above_50pct']:<8.1%}")

def plot_breakout_cts_equity_curve(results, save_path='results/breakout_cts_equity_curve.png'):
    """
    Plot breakout CTS equity curve and strategy metrics.
    
    Parameters:
        results (dict): Results from backtest_breakout_cts_strategy.
        save_path (str): Path to save the plot.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        portfolio_df = results['portfolio_data']
        config = results['config']
        performance = results['performance']
        
        equity_curve = build_account_curve(portfolio_df['portfolio_return'], config['capital'])
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Plot 1: Equity curve
        axes[0].plot(equity_curve.index, equity_curve.values/1e6, 'b-', linewidth=1.5, 
                    label='Breakout CTS Strategy')
        axes[0].set_title('Breakout Cross-Timeframe Strategy Equity Curve', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Portfolio Value ($M)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Breakout signal strength
        axes[1].plot(portfolio_df.index, portfolio_df['avg_breakout_signal'], 'g-', linewidth=1, 
                    label='Average Breakout Signal')
        axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='50% Threshold')
        axes[1].set_ylabel('Signal Strength', fontsize=12)
        axes[1].set_title('Breakout Signal Evolution', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_ylim(0, 1)
        
        # Plot 3: Active instruments
        axes[2].plot(portfolio_df.index, portfolio_df['num_active_instruments'], 'orange', linewidth=1,
                    label='Active Instruments')
        axes[2].set_ylabel('Number of Instruments', fontsize=12)
        axes[2].set_title('Active Instruments Over Time', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # Plot 4: Drawdown
        drawdown_stats = calculate_maximum_drawdown(equity_curve)
        drawdown_series = drawdown_stats['drawdown_series'] * 100
        
        axes[3].fill_between(drawdown_series.index, drawdown_series.values, 0, 
                            color='red', alpha=0.3, label='Drawdown')
        axes[3].plot(drawdown_series.index, drawdown_series.values, 'r-', linewidth=1)
        axes[3].set_ylabel('Drawdown (%)', fontsize=12)
        axes[3].set_xlabel('Date', fontsize=12)
        axes[3].set_title('Drawdown', fontsize=12, fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()
        
        # Format x-axis dates for all subplots
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add performance metrics text
        start_date = config['backtest_start'].strftime('%Y-%m-%d')
        end_date = config['backtest_end'].strftime('%Y-%m-%d')
        
        textstr = f'''Breakout CTS Performance Summary:
Total Return: {performance['total_return']:.1%}
Annualized Return: {performance['annualized_return']:.1%}
Volatility: {performance['annualized_volatility']:.1%}
Sharpe Ratio: {performance['sharpe_ratio']:.3f}
Max Drawdown: {performance['max_drawdown_pct']:.1f}%
Periods: {config['breakout_periods']}
Avg Signal: {performance['avg_breakout_signal']:.3f}
Active Instruments: {performance['avg_active_instruments']:.1f}
Trading Costs: {performance['cost_as_pct_of_capital']:.2%} of capital
Period: {start_date} to {end_date}'''
        
        plt.figtext(0.02, 0.02, textstr, fontsize=9, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Breakout CTS equity curve saved to: {save_path}")
        
    except Exception as e:
        print(f"Error plotting equity curve: {e}")
        import traceback
        traceback.print_exc()

#####   COMPREHENSIVE UNIT TESTS AND VALIDATION   #####

def test_breakout_signal_calculation():
    """
    Test breakout signal calculation with known price patterns.
    """
    print("\n" + "=" * 60)
    print("UNIT TEST 1: BREAKOUT SIGNAL CALCULATION")
    print("=" * 60)
    
    # Create test price series with known breakout patterns
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Test 1: Clear breakout pattern
    print("\n--- Test 1: Clear Breakout Pattern ---")
    # Create a stable price series, then add a clear breakout
    stable_prices = pd.Series([100.0] * 100, index=dates[:100])
    breakout_prices = pd.Series([120.0] * 100, index=dates[100:])  # 20% jump
    uptrend_prices = pd.concat([stable_prices, breakout_prices])
    
    breakout_signals = calculate_breakout_signals_multi_period(uptrend_prices, [10, 25, 50])
    
    # Check signals at the exact moment of breakout
    # Need to account for rolling window requirements (at least 50 days)
    pre_breakout_signals = breakout_signals.iloc[80:100]  # Before breakout, after rolling windows established
    breakout_moment_signals = breakout_signals.iloc[100:102]  # The exact breakout moment
    
    pre_avg = pre_breakout_signals.mean().mean()
    breakout_avg = breakout_moment_signals.mean().mean()
    
    print(f"Pre-breakout avg signals (days 80-99): {pre_avg:.3f}")
    print(f"Breakout moment signals (days 100-101): {breakout_avg:.3f}")
    
    # For a stable then jumping pattern, we expect signals at the moment of breakout
    assert breakout_avg > pre_avg, "Expected higher signals at breakout moment"
    assert breakout_avg > 0.3, "Expected signals when price breaks above rolling high"
    print("✓ Clear breakout signals working correctly")
    
    # Test 2: Sideways market (should have few breakouts)
    print("\n--- Test 2: Sideways Market ---")
    sideways_prices = pd.Series(100 + 2 * np.sin(np.linspace(0, 10*np.pi, 200)) + np.random.normal(0, 0.3, 200), index=dates)
    
    sideways_signals = calculate_breakout_signals_multi_period(sideways_prices, [10, 25, 50])
    avg_sideways_signal = sideways_signals.mean().mean()
    
    print(f"Average signal in sideways market: {avg_sideways_signal:.3f}")
    assert avg_sideways_signal < 0.3, "Expected low signals in sideways market"
    print("✓ Sideways market signals working correctly")
    
    # Test 3: Signal combination methods
    print("\n--- Test 3: Signal Combination Methods ---")
    test_signals_df = pd.DataFrame({
        'breakout_10d': [1, 0, 1, 0, 1],
        'breakout_25d': [0, 1, 1, 0, 0],
        'breakout_50d': [0, 0, 1, 1, 0]
    })
    
    avg_combined = calculate_combined_breakout_signal(test_signals_df, 'average')
    max_combined = calculate_combined_breakout_signal(test_signals_df, 'max')
    weighted_combined = calculate_combined_breakout_signal(test_signals_df, 'weighted_average')
    
    print(f"Test signals DataFrame:\n{test_signals_df}")
    print(f"Average combination: {avg_combined.values}")
    print(f"Max combination: {max_combined.values}")
    print(f"Weighted average combination: {weighted_combined.values}")
    
    # Verify that weighted average gives more weight to shorter periods
    assert weighted_combined.iloc[0] > avg_combined.iloc[0], "Weighted average should favor short periods"
    print("✓ Signal combination methods working correctly")

def test_position_sizing_logic():
    """
    Test position sizing calculations and edge cases.
    """
    print("\n" + "=" * 60)
    print("UNIT TEST 2: POSITION SIZING LOGIC")
    print("=" * 60)
    
    # Test parameters
    symbol = 'TEST'
    capital = 100000
    weight = 0.1
    idm = 2.0
    price = 4500
    volatility = 0.20  # 20% annual volatility
    multiplier = 5
    risk_target = 0.2
    fx_rate = 1.0
    
    print(f"Test parameters:")
    print(f"  Capital: ${capital:,}")
    print(f"  Weight: {weight}")
    print(f"  IDM: {idm}")
    print(f"  Price: ${price}")
    print(f"  Volatility: {volatility:.1%}")
    print(f"  Multiplier: {multiplier}")
    
    # Test 1: Full signal (should get full position)
    print("\n--- Test 1: Full Breakout Signal ---")
    full_signal = 1.0
    full_position = calculate_breakout_position_size(
        symbol, capital, weight, idm, price, volatility, multiplier, full_signal, risk_target, fx_rate
    )
    
    # Calculate expected base position manually
    expected_base = calculate_portfolio_position_size(
        symbol, capital, weight, idm, price, volatility, multiplier, risk_target, fx_rate
    )
    
    print(f"Full signal position: {full_position}")
    print(f"Expected base position: {expected_base}")
    assert abs(full_position - expected_base) <= 1, "Full signal should equal base position"
    print("✓ Full signal position sizing correct")
    
    # Test 2: Half signal (should get half position)
    print("\n--- Test 2: Half Breakout Signal ---")
    half_signal = 0.5
    half_position = calculate_breakout_position_size(
        symbol, capital, weight, idm, price, volatility, multiplier, half_signal, risk_target, fx_rate
    )
    
    expected_half = expected_base * 0.5
    print(f"Half signal position: {half_position}")
    print(f"Expected half position: {expected_half:.1f}")
    assert abs(half_position - expected_half) <= 1, "Half signal should be half of base position"
    print("✓ Half signal position sizing correct")
    
    # Test 3: Zero signal (should get zero position)
    print("\n--- Test 3: Zero Breakout Signal ---")
    zero_signal = 0.0
    zero_position = calculate_breakout_position_size(
        symbol, capital, weight, idm, price, volatility, multiplier, zero_signal, risk_target, fx_rate
    )
    
    print(f"Zero signal position: {zero_position}")
    assert zero_position == 0, "Zero signal should result in zero position"
    print("✓ Zero signal position sizing correct")
    
    # Test 4: Edge cases
    print("\n--- Test 4: Edge Cases ---")
    
    # NaN volatility
    nan_vol_position = calculate_breakout_position_size(
        symbol, capital, weight, idm, price, np.nan, multiplier, full_signal, risk_target, fx_rate
    )
    assert nan_vol_position == 0, "NaN volatility should result in zero position"
    
    # NaN signal
    nan_signal_position = calculate_breakout_position_size(
        symbol, capital, weight, idm, price, volatility, multiplier, np.nan, risk_target, fx_rate
    )
    assert nan_signal_position == 0, "NaN signal should result in zero position"
    
    # Zero volatility
    zero_vol_position = calculate_breakout_position_size(
        symbol, capital, weight, idm, price, 0.0, multiplier, full_signal, risk_target, fx_rate
    )
    assert zero_vol_position == 0, "Zero volatility should result in zero position"
    
    print("✓ Edge cases handled correctly")

def test_trading_cost_calculation():
    """
    Test trading cost calculation using SR methodology.
    """
    print("\n" + "=" * 60)
    print("UNIT TEST 3: TRADING COST CALCULATION")
    print("=" * 60)
    
    # Test parameters
    symbol = 'MES'
    trade_size = 10.0  # 10 contracts
    price = 4500
    volatility = 0.20  # 20% annual
    multiplier = 5
    sr_cost = 0.00028  # From MES specification
    capital = 100000
    fx_rate = 1.0
    
    print(f"Test parameters:")
    print(f"  Trade size: {trade_size} contracts")
    print(f"  Price: ${price}")
    print(f"  Volatility: {volatility:.1%}")
    print(f"  SR cost: {sr_cost}")
    
    # Test 1: Normal trade cost calculation
    print("\n--- Test 1: Normal Trade Cost ---")
    trading_cost = calculate_trading_cost_from_sr(
        symbol, trade_size, price, volatility, multiplier, sr_cost, capital, fx_rate
    )
    
    # Manual calculation for verification
    notional_per_contract = price * multiplier * fx_rate  # 4500 * 5 = 22500
    cost_per_contract = sr_cost * volatility * notional_per_contract  # 0.00028 * 0.20 * 22500
    expected_cost = trade_size * cost_per_contract
    
    print(f"Calculated trading cost: ${trading_cost:.2f}")
    print(f"Expected cost: ${expected_cost:.2f}")
    print(f"Notional per contract: ${notional_per_contract:,.0f}")
    print(f"Cost per contract: ${cost_per_contract:.2f}")
    
    assert abs(trading_cost - expected_cost) < 0.01, "Trading cost calculation incorrect"
    print("✓ Trading cost calculation correct")
    
    # Test 2: Zero trade size (no cost)
    print("\n--- Test 2: Zero Trade Size ---")
    zero_cost = calculate_trading_cost_from_sr(
        symbol, 0.0, price, volatility, multiplier, sr_cost, capital, fx_rate
    )
    assert zero_cost == 0.0, "Zero trade size should have zero cost"
    print("✓ Zero trade size cost correct")
    
    # Test 3: Zero SR cost (no cost)
    print("\n--- Test 3: Zero SR Cost ---")
    zero_sr_cost = calculate_trading_cost_from_sr(
        symbol, trade_size, price, volatility, multiplier, 0.0, capital, fx_rate
    )
    assert zero_sr_cost == 0.0, "Zero SR cost should result in zero cost"
    print("✓ Zero SR cost correct")

def test_lookahead_bias_prevention():
    """
    Test that signals don't use future information.
    """
    print("\n" + "=" * 60)
    print("UNIT TEST 4: LOOKAHEAD BIAS PREVENTION")
    print("=" * 60)
    
    # Create test data with a known future spike
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.Series(100 + np.random.normal(0, 1, 100), index=dates)
    
    # Add a big spike at day 80
    prices.iloc[80:] += 20
    
    print(f"Test setup: Price spike at day 80")
    print(f"Price before spike (day 79): {prices.iloc[79]:.2f}")
    print(f"Price after spike (day 80): {prices.iloc[80]:.2f}")
    
    # Calculate breakout signals
    breakout_signals = calculate_breakout_signals_multi_period(prices, [10])
    
    # Check that signal at day 79 doesn't know about day 80 spike
    signal_day_79 = breakout_signals['breakout_10d'].iloc[79]
    signal_day_80 = breakout_signals['breakout_10d'].iloc[80]
    signal_day_81 = breakout_signals['breakout_10d'].iloc[81]
    
    print(f"\nSignal analysis:")
    print(f"Signal on day 79 (before spike): {signal_day_79}")
    print(f"Signal on day 80 (spike day): {signal_day_80}")
    print(f"Signal on day 81 (after spike): {signal_day_81}")
    
    # The signal on day 79 should not be influenced by day 80 spike
    # Signal on day 80 should trigger because it sees the current price vs past rolling high
    assert signal_day_80 == 1, "Signal should trigger on breakout day"
    print("✓ Lookahead bias prevention verified")

def test_strategy_integration():
    """
    Test the complete strategy with a small dataset to verify end-to-end functionality.
    """
    print("\n" + "=" * 60)
    print("UNIT TEST 5: STRATEGY INTEGRATION")
    print("=" * 60)
    
    try:
        print("Testing strategy with small capital and short date range...")
        
        # Run a quick test backtest
        test_results = backtest_breakout_cts_strategy(
            data_dir='Data',
            capital=50000,  # Small capital for testing
            risk_target=0.2,
            breakout_periods=[10, 25],  # Fewer periods for speed
            signal_combination_method='average',
            weight_method='equal',  # Simpler weighting for testing
            start_date='2023-01-01',
            end_date='2023-06-01'  # Short period for quick test
        )
        
        if test_results:
            # Verify result structure
            assert 'portfolio_data' in test_results, "Missing portfolio_data in results"
            assert 'performance' in test_results, "Missing performance in results"
            assert 'config' in test_results, "Missing config in results"
            
            # Verify performance metrics exist
            perf = test_results['performance']
            required_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown_pct', 'annualized_return']
            for metric in required_metrics:
                assert metric in perf, f"Missing performance metric: {metric}"
                assert not np.isnan(perf[metric]), f"Performance metric {metric} is NaN"
            
            # Verify portfolio data structure
            portfolio_df = test_results['portfolio_data']
            required_columns = ['portfolio_return', 'equity_eod', 'num_active_instruments', 'avg_breakout_signal']
            for col in required_columns:
                assert col in portfolio_df.columns, f"Missing portfolio column: {col}"
            
            # Verify no negative equity
            assert (portfolio_df['equity_eod'] > 0).all(), "Found negative equity values"
            
            # Verify reasonable signal ranges
            signal_stats = portfolio_df['avg_breakout_signal'].describe()
            assert signal_stats['min'] >= 0, "Breakout signals should be >= 0"
            assert signal_stats['max'] <= 1, "Breakout signals should be <= 1"
            
            print(f"✓ Integration test passed")
            print(f"  Returns: {perf['total_return']:.2%}")
            print(f"  Sharpe: {perf['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {perf['max_drawdown_pct']:.1f}%")
            print(f"  Avg Signal: {signal_stats['mean']:.3f}")
            
        else:
            raise ValueError("Strategy returned no results")
            
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def run_all_tests():
    """
    Run all unit tests and validation checks.
    """
    print("\n" + "=" * 80)
    print("RUNNING COMPREHENSIVE UNIT TESTS FOR BREAKOUT CTS STRATEGY")
    print("=" * 80)
    
    test_functions = [
        test_breakout_signal_calculation,
        test_position_sizing_logic,
        test_trading_cost_calculation,
        test_lookahead_bias_prevention,
        test_strategy_integration
    ]
    
    passed_tests = 0
    failed_tests = []
    
    for test_func in test_functions:
        try:
            test_func()
            passed_tests += 1
            print(f"✅ {test_func.__name__} PASSED")
        except Exception as e:
            failed_tests.append((test_func.__name__, str(e)))
            print(f"❌ {test_func.__name__} FAILED: {e}")
    
    print(f"\n" + "=" * 80)
    print(f"TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(test_functions)}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for test_name, error in failed_tests:
            print(f"  - {test_name}: {error}")
        raise AssertionError(f"{len(failed_tests)} tests failed")
    else:
        print(f"\n🎉 ALL TESTS PASSED! Strategy implementation verified.")

# ===========================================
# CONFIGURATION - MODIFY THESE AS NEEDED
# ===========================================
CAPITAL = 5000000               # Starting capital (reduced from 50M for reasonable test)
START_DATE = '2015-01-01'       # Backtest start date or None
END_DATE = '2020-01-01'         # Backtest end date or None
RISK_TARGET = 0.15              # 15% annual risk target (reduced from 20%)
WEIGHT_METHOD = 'equal'         # 'equal', 'vol_inverse', 'handcrafted'
BREAKOUT_PERIODS = [20, 40, 60] # Simpler breakout periods (3 instead of 5)
SIGNAL_COMBINATION_METHOD = 'average'  # Simple average instead of weighted

def main():
    """
    Test Breakout Cross-Timeframe Strategy implementation with comprehensive analysis.
    """
    try:
        print(f"\n" + "=" * 60)
        print("RUNNING BREAKOUT CROSS-TIMEFRAME STRATEGY (CTS)")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Capital: ${CAPITAL:,}")
        print(f"  Date Range: {START_DATE} to {END_DATE}")
        print(f"  Risk Target: {RISK_TARGET:.1%}")
        print(f"  Weight Method: {WEIGHT_METHOD}")
        print(f"  Breakout Periods: {BREAKOUT_PERIODS}")
        print(f"  Signal Combination: {SIGNAL_COMBINATION_METHOD}")
        print("=" * 60)
        
        # Run backtest
        results = backtest_breakout_cts_strategy(
            data_dir='Data',
            capital=CAPITAL,
            risk_target=RISK_TARGET,
            short_span=32,
            long_years=10,
            min_vol_floor=0.05,
            breakout_periods=BREAKOUT_PERIODS,
            signal_combination_method=SIGNAL_COMBINATION_METHOD,
            weight_method=WEIGHT_METHOD,
            common_hypothetical_SR=0.3,
            annual_turnover_T=7.0,
            start_date=START_DATE,
            end_date=END_DATE
        )
        
        # Analyze results
        if results:
            analyze_breakout_cts_results(results)
            plot_breakout_cts_equity_curve(results)
            
            # Compare with baseline Strategy 4
            print(f"\n" + "=" * 60)
            print("COMPARISON WITH BASELINE STRATEGY 4")
            print("=" * 60)
            
            try:
                # Run Strategy 4 for comparison
                from rob_port.chapter4 import backtest_multi_instrument_strategy
                
                baseline_results = backtest_multi_instrument_strategy(
                    data_dir='Data',
                    capital=CAPITAL,
                    risk_target=RISK_TARGET,
                    weight_method=WEIGHT_METHOD,
                    start_date=START_DATE,
                    end_date=END_DATE
                )
                
                if baseline_results:
                    baseline_perf = baseline_results['performance']
                    breakout_perf = results['performance']
                    
                    print(f"\nPerformance Comparison:")
                    print(f"{'Metric':<25} {'Strategy 4':<15} {'Breakout CTS':<15} {'Difference':<12}")
                    print("-" * 75)
                    print(f"{'Total Return':<25} {baseline_perf['total_return']:.2%} {breakout_perf['total_return']:.2%} {breakout_perf['total_return'] - baseline_perf['total_return']:.2%}")
                    print(f"{'Annualized Return':<25} {baseline_perf['annualized_return']:.2%} {breakout_perf['annualized_return']:.2%} {breakout_perf['annualized_return'] - baseline_perf['annualized_return']:.2%}")
                    print(f"{'Volatility':<25} {baseline_perf['annualized_volatility']:.2%} {breakout_perf['annualized_volatility']:.2%} {breakout_perf['annualized_volatility'] - baseline_perf['annualized_volatility']:.2%}")
                    print(f"{'Sharpe Ratio':<25} {baseline_perf['sharpe_ratio']:.3f} {breakout_perf['sharpe_ratio']:.3f} {breakout_perf['sharpe_ratio'] - baseline_perf['sharpe_ratio']:.3f}")
                    print(f"{'Max Drawdown':<25} {baseline_perf['max_drawdown_pct']:.1f}% {breakout_perf['max_drawdown_pct']:.1f}% {breakout_perf['max_drawdown_pct'] - baseline_perf['max_drawdown_pct']:.1f}%")
                else:
                    print("Could not run Strategy 4 comparison")
                    
            except Exception as e:
                print(f"Error running Strategy 4 comparison: {e}")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    
    return results

if __name__ == "__main__":
    # Check if we want to run tests only
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test-only':
        print("Running unit tests only...")
        run_all_tests()
    else:
        main() 