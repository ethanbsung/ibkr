from chapter5 import *
from chapter4 import *
from chapter3 import *
from chapter2 import *
from chapter1 import *
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')

#####   STRATEGY 6: SLOW TREND FOLLOWING, LONG AND SHORT   #####

def calculate_trend_signal_long_short(prices: pd.Series, fast_span: int = 64, slow_span: int = 256) -> pd.Series:
    """
    Calculate long/short trend signal from EWMAC.
    
    From book:
        Go long if: EWMAC(64,256) > 0
        Go short if: EWMAC(64,256) < 0
    
    Parameters:
        prices (pd.Series): Price series.
        fast_span (int): Fast EWMA span.
        slow_span (int): Slow EWMA span.
    
    Returns:
        pd.Series: Trend signal (+1 = long, -1 = short).
    """
    ewmac = calculate_ewma_trend(prices, fast_span, slow_span)
    
    # Go long if EWMAC > 0, short if EWMAC < 0
    trend_signal = np.where(ewmac > 0, 1, -1)
    
    return pd.Series(trend_signal, index=prices.index)

def calculate_strategy6_position_size(symbol, capital, weight, idm, price, volatility, 
                                    multiplier, trend_signal, risk_target=0.2, fx_rate=1.0):
    """
    Calculate position size for Strategy 6 with long/short trend filter.
    
    From book: "N_t = (Sign(trend_t) × Capital × IDM × Weight_t × τ) ÷ (Multiplier_t × Price_t × FX_t × σ_t)"
    
    Where if N > 0 we go long (in an uptrend), otherwise with N < 0 we would be short (in a downtrend).
    
    Parameters:
        symbol (str): Instrument symbol.
        capital (float): Total portfolio capital.
        weight (float): Weight allocated to this instrument.
        idm (float): Instrument Diversification Multiplier.
        price (float): Current price.
        volatility (float): Annualized volatility forecast.
        multiplier (float): Contract multiplier.
        trend_signal (float): Trend signal (+1 = long, -1 = short).
        risk_target (float): Target risk fraction.
        fx_rate (float): FX rate for currency conversion.
    
    Returns:
        float: Number of contracts for this instrument (positive = long, negative = short).
    """
    # Ensure trend_signal is valid (+1 or -1)
    if not (trend_signal == 1 or trend_signal == -1):
        return 0 # Take no position if signal is not explicitly long or short
    
    if np.isnan(volatility) or volatility <= 0:
        return 0
    
    # Use EXACT same formula as Strategy 4, but include trend signal
    # N = (Sign(trend) × Capital × IDM × Weight × τ) ÷ (Multiplier × Price × FX × σ%)
    numerator = trend_signal * capital * idm * weight * risk_target
    denominator = multiplier * price * fx_rate * volatility
    
    position_size = numerator / denominator
    
    # Protect against infinite or extremely large position sizes
    if np.isinf(position_size) or abs(position_size) > 100000:
        return 0
    
    # Round to nearest integer since you can only hold whole contracts
    return round(position_size)

def backtest_long_short_trend_strategy(data_dir='Data', capital=50000000, risk_target=0.2,
                                     short_span=32, long_years=10, min_vol_floor=0.05,
                                     trend_fast_span=64, trend_slow_span=256,
                                     weight_method='handcrafted',
                                     common_hypothetical_SR=0.3, annual_turnover_T=7.0,
                                     start_date=None, end_date=None):
    """
    Backtest Strategy 6: Long/short trend following multi-instrument portfolio with daily dynamic rebalancing.
    
    Implementation follows book exactly: "Trade a portfolio of one or more instruments, 
    each with positions scaled for a variable risk estimate. Hold a long position when 
    they have been in a long uptrend, and a short position in a downtrend."
    
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
    print("STRATEGY 6: SLOW TREND FOLLOWING, LONG AND SHORT")
    print("=" * 60)
    
    # Load all instrument data using the same function as chapter 4/5
    all_instruments_specs_df = load_instrument_data()
    raw_instrument_data = load_all_instrument_data(data_dir)
    
    if not raw_instrument_data:
        raise ValueError("No instrument data loaded successfully")
    
    print(f"\nPortfolio Configuration:")
    print(f"  Instruments initially loaded: {len(raw_instrument_data)}")
    print(f"  Capital: ${capital:,.0f}")
    print(f"  Risk Target: {risk_target:.1%}")
    print(f"  Weight Method: {weight_method}")
    print(f"  Trend Filter: EWMA({trend_fast_span},{trend_slow_span}) Long/Short")
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

        # Calculate blended volatility (same as Strategy 4/5)
        blended_vol_series = calculate_blended_volatility(
            raw_returns_for_vol, short_span=short_span, long_years=long_years, min_vol_floor=min_vol_floor
        )
        # Shift to prevent lookahead bias - forecast for day T uses data up to T-1
        df['vol_forecast'] = blended_vol_series.shift(1).reindex(df.index).ffill().fillna(min_vol_floor)
        
        # Calculate long/short trend signal using EWMAC (no lookahead bias)
        trend_signal_series = calculate_trend_signal_long_short(df['Last'], trend_fast_span, trend_slow_span)
        # Shift to prevent lookahead bias - trend signal for day T uses data up to T-1
        # Use forward fill instead of fillna(0) to ensure we always have +1 or -1
        df['trend_signal'] = trend_signal_series.shift(1).reindex(df.index).ffill()
        
        # For the very first day, if trend_signal is NaN, use the first valid trend signal
        if df['trend_signal'].isna().any():
            first_valid_trend = trend_signal_series.dropna().iloc[0] if len(trend_signal_series.dropna()) > 0 else 0 # Default to 0 (neutral)
            df['trend_signal'] = df['trend_signal'].fillna(first_valid_trend)
        
        # Ensure critical data is present
        df.dropna(subset=['Last', 'vol_forecast', 'daily_price_change_pct'], inplace=True)
        if df.empty:
            print(f"Skipping {symbol}: Empty after dropping NaNs in critical columns.")
            continue

        processed_instrument_data[symbol] = df

    if not processed_instrument_data:
        raise ValueError("No instruments remaining after preprocessing and volatility calculation.")
    
    print(f"  Instruments after preprocessing: {len(processed_instrument_data)}")

    # Determine common date range for backtest (same logic as chapter 4/5)
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

    # Initialize portfolio tracking (same structure as chapter 4/5)
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
                      'num_active_instruments': 0, 'intended_long_signals': 0, 'intended_short_signals': 0}
            for symbol_k in processed_instrument_data.keys(): 
                record[f'{symbol_k}_contracts'] = 0.0
                record[f'{symbol_k}_trend'] = 0.0 # Default placeholder to 0 (neutral)
            portfolio_daily_records.append(record)
            continue
        
        previous_trading_date = trading_days_range[idx-1]
        capital_at_start_of_day = current_portfolio_equity
        daily_total_pnl = 0.0
        current_day_positions_and_trends = {} # Store contracts and actual trend used for sizing
        
        # Metric: Count intended signals for ALL processed instruments
        daily_intended_long_signals_metric = 0
        daily_intended_short_signals_metric = 0
        for symbol_proc, df_proc in processed_instrument_data.items():
            intended_signal = 1.0 # Default to long if no specific signal found for the day
            if current_date in df_proc.index:
                signal_from_df = df_proc.loc[current_date, 'trend_signal']
                if pd.notna(signal_from_df) and abs(abs(signal_from_df) - 1) < 1e-6:
                    intended_signal = signal_from_df
            
            if intended_signal > 0.5: daily_intended_long_signals_metric += 1
            elif intended_signal < -0.5: daily_intended_short_signals_metric += 1
                
        num_instruments_with_actual_position = 0
        effective_data_cutoff_date = previous_trading_date

        # Reweighting logic (remains the same)
        current_iteration_eligible_instruments = {s for s, df_full in processed_instrument_data.items() 
                                                if not df_full[df_full.index <= effective_data_cutoff_date].empty and 
                                                   len(df_full[df_full.index <= effective_data_cutoff_date]) > max(short_span, trend_slow_span)}
        perform_reweight = (idx == 1) or (len(current_iteration_eligible_instruments) > len(known_eligible_instruments))
        if perform_reweight:
            known_eligible_instruments = current_iteration_eligible_instruments.copy()
            data_for_reweighting = {s_el: processed_instrument_data[s_el][processed_instrument_data[s_el].index <= effective_data_cutoff_date]
                                    for s_el in known_eligible_instruments if not processed_instrument_data[s_el][processed_instrument_data[s_el].index <= effective_data_cutoff_date].empty}
            if data_for_reweighting:
                weights = calculate_instrument_weights(data_for_reweighting, weight_method, all_instruments_specs_df, common_hypothetical_SR, annual_turnover_T, risk_target)
                idm = calculate_idm_from_count(sum(1 for w_val in weights.values() if w_val > 1e-6))

        # Calculate positions and P&L for each instrument
        for symbol, df_instrument in processed_instrument_data.items():
            instrument_multiplier = get_instrument_specs(symbol, all_instruments_specs_df)['multiplier']
            instrument_weight = weights.get(symbol, 0.0)
            num_contracts = 0.0
            instrument_pnl_today = 0.0
            actual_trend_used_for_sizing = 0.0 # Default to neutral if trade cannot be made due to missing data

            if instrument_weight > 1e-6:
                try:
                    price_for_sizing = df_instrument.loc[previous_trading_date, 'Last']
                    vol_for_sizing = df_instrument.loc[current_date, 'vol_forecast']
                    trend_signal_for_positioning = df_instrument.loc[current_date, 'trend_signal']
                    actual_trend_used_for_sizing = trend_signal_for_positioning # Capture for record

                    price_at_start_of_trading = df_instrument.loc[previous_trading_date, 'Last']
                    price_at_end_of_trading = df_instrument.loc[current_date, 'Last']
                    
                    if not (pd.isna(price_for_sizing) or pd.isna(vol_for_sizing) or pd.isna(trend_signal_for_positioning) or 
                            pd.isna(price_at_start_of_trading) or pd.isna(price_at_end_of_trading)):
                        
                        vol_for_sizing = max(vol_for_sizing, min_vol_floor)
                        # The calculate_strategy6_position_size function now handles invalid trend signals by returning 0.
                        # No need to default trend_signal_for_positioning to 1.0 here anymore.
                        # if abs(abs(trend_signal_for_positioning) - 1) > 1e-6: # Ensure strictly +1 or -1
                        #     trend_signal_for_positioning = 1.0 # Default to long if somehow invalid
                        #     actual_trend_used_for_sizing = trend_signal_for_positioning
                            
                        num_contracts = calculate_strategy6_position_size(
                            symbol=symbol, capital=capital_at_start_of_day, weight=instrument_weight, 
                            idm=idm, price=price_for_sizing, volatility=vol_for_sizing, 
                            multiplier=instrument_multiplier, trend_signal=trend_signal_for_positioning, 
                            risk_target=risk_target
                        )
                        instrument_pnl_today = num_contracts * instrument_multiplier * (price_at_end_of_trading - price_at_start_of_trading)
                        if abs(num_contracts) > 0.01:
                            num_instruments_with_actual_position += 1
                except KeyError:
                    # Data missing for trade execution, num_contracts & pnl remain 0.
                    # actual_trend_used_for_sizing keeps its default of 0.0 for record-keeping if this specific instrument's current_date data was missing for sizing.
                    # If current_date was present in df_instrument but other data (e.g. prev_day) was missing, actual_trend_used_for_sizing would have been set from df_instrument.
                    # For more robust record, we can try to fetch the intended signal again if KeyError happened before trend_signal_for_positioning was read.
                    if current_date in df_instrument.index:
                         current_day_signal = df_instrument.loc[current_date, 'trend_signal']
                         if pd.notna(current_day_signal) and abs(abs(current_day_signal)-1) < 1e-6: # Check if it's a valid +/-1 signal
                             actual_trend_used_for_sizing = current_day_signal
                         elif pd.notna(current_day_signal) and current_day_signal == 0: # Check if it's a valid 0 signal
                              actual_trend_used_for_sizing = 0.0
                    # else it remains 0.0 (default for this catch block)
                    pass 
            
            current_day_positions_and_trends[symbol] = {'contracts': num_contracts, 'trend': actual_trend_used_for_sizing}
            daily_total_pnl += instrument_pnl_today

        portfolio_daily_percentage_return = daily_total_pnl / capital_at_start_of_day if capital_at_start_of_day > 0 else 0.0
        current_portfolio_equity = capital_at_start_of_day * (1 + portfolio_daily_percentage_return)

        record = {'date': current_date, 'total_pnl': daily_total_pnl, 
                  'portfolio_return': portfolio_daily_percentage_return, 
                  'equity_sod': capital_at_start_of_day, 'equity_eod': current_portfolio_equity,
                  'num_active_instruments': num_instruments_with_actual_position,
                  'intended_long_signals': daily_intended_long_signals_metric, 
                  'intended_short_signals': daily_intended_short_signals_metric}
        
        for s_rec, data_rec in current_day_positions_and_trends.items(): 
            record[f'{s_rec}_contracts'] = data_rec['contracts']
            record[f'{s_rec}_trend'] = data_rec['trend']
        
        # Ensure all processed instruments have entries in the record for contracts and trend
        for s_proc in processed_instrument_data.keys():
            if f'{s_proc}_contracts' not in record:
                record[f'{s_proc}_contracts'] = 0.0 # Default if not in current_day_positions_and_trends
            if f'{s_proc}_trend' not in record:
                trend_val_fill = 0.0 # Default to 0 (neutral)
                if current_date in processed_instrument_data[s_proc].index:
                    sig = processed_instrument_data[s_proc].loc[current_date, 'trend_signal']
                    if pd.notna(sig) and (abs(abs(sig)-1) < 1e-6 or sig == 0): # Allow valid +/-1 or 0
                        trend_val_fill = sig
                record[f'{s_proc}_trend'] = trend_val_fill
                
        portfolio_daily_records.append(record)

    # Post-loop processing
    if not portfolio_daily_records: raise ValueError("No daily records generated.")
    portfolio_df = pd.DataFrame(portfolio_daily_records).set_index('date')
    
    account_curve = build_account_curve(portfolio_df['portfolio_return'], capital)
    performance = calculate_comprehensive_performance(account_curve, portfolio_df['portfolio_return'])
    
    performance['num_instruments'] = len(processed_instrument_data)
    performance['idm'] = idm
    performance['avg_active_instruments'] = portfolio_df['num_active_instruments'].mean()
    performance['avg_long_signals'] = portfolio_df['intended_long_signals'].mean() 
    performance['avg_short_signals'] = portfolio_df['intended_short_signals'].mean()
    performance['weight_method'] = weight_method
    performance['backtest_start'] = trading_days_range.min()
    performance['backtest_end'] = trading_days_range.max()
    performance['trend_fast_span'] = trend_fast_span
    performance['trend_slow_span'] = trend_slow_span
    
    instrument_stats = {}
    for symbol in processed_instrument_data.keys():
        pos_col = f'{symbol}_contracts'
        trend_col = f'{symbol}_trend' # This now reflects the trend used/intended for the day
        if pos_col in portfolio_df.columns and trend_col in portfolio_df.columns:
            inst_positions = portfolio_df[pos_col][portfolio_df[pos_col] != 0]
            # Use the recorded daily trend for consistency with what was intended/used
            inst_trends_recorded = portfolio_df[trend_col] 
            
            # Filter inst_trends_recorded for days when a position was actually held for some stats, or use all for others
            inst_trends_when_active = inst_trends_recorded[portfolio_df[pos_col] != 0]

            instrument_stats[symbol] = {
                'avg_position': inst_positions.mean() if not inst_positions.empty else 0.0,
                'weight': weights.get(symbol, 0.0),
                'active_days': len(inst_positions),
                # Avg intended trend signal over all days for this instrument:
                'avg_intended_trend_signal': inst_trends_recorded.mean() if not inst_trends_recorded.empty else 0.0,
                # Percent time long/short based on intended signals over all days for this instrument:
                'percent_time_long_intended': (inst_trends_recorded > 0.5).mean() if not inst_trends_recorded.empty else 0.0,
                'percent_time_short_intended': (inst_trends_recorded < -0.5).mean() if not inst_trends_recorded.empty else 0.0,
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
            'trend_fast_span': trend_fast_span, 'trend_slow_span': trend_slow_span,
            'weight_method': weight_method, 'common_hypothetical_SR': common_hypothetical_SR,
            'annual_turnover_T': annual_turnover_T,
            'backtest_start': trading_days_range.min(), 'backtest_end': trading_days_range.max()
        }
    }

def analyze_long_short_results(results):
    """
    Analyze and display comprehensive long/short trend following results.
    
    Parameters:
        results (dict): Results from backtest_long_short_trend_strategy.
    """
    performance = results['performance']
    instrument_stats = results['instrument_stats']
    config = results['config']
    
    print("\n" + "=" * 60)
    print("LONG/SHORT TREND FOLLOWING PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Overall performance
    print(f"\n--- Overall Portfolio Performance ---")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Annualized Return: {performance['annualized_return']:.2%}")
    print(f"Annualized Volatility: {performance['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {performance['max_drawdown_pct']:.1f}%")
    print(f"Skewness: {performance['skewness']:.3f}")
    
    # Long/short characteristics
    print(f"\n--- Long/Short Trend Following Characteristics ---")
    print(f"Average Active Instruments: {performance['avg_active_instruments']:.1f}")
    print(f"Average Long Signals: {performance['avg_long_signals']:.1f}")
    print(f"Average Short Signals: {performance['avg_short_signals']:.1f}")
    total_signals = performance['avg_long_signals'] + performance['avg_short_signals']
    print(f"Percent Time Long: {(performance['avg_long_signals'] / performance['num_instruments']):.1%}")
    print(f"Percent Time Short: {(performance['avg_short_signals'] / performance['num_instruments']):.1%}")
    print(f"Percent Time in Market: {(total_signals / performance['num_instruments']):.1%}")
    print(f"Trend Filter: EWMA({config['trend_fast_span']},{config['trend_slow_span']}) Long/Short")
    
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
    
    print(f"{'Symbol':<8} {'Weight':<8} {'Avg Pos':<10} {'%Long':<8} {'%Short':<8} {'Days':<6}")
    print("-" * 65)
    
    for symbol, stats in sorted_instruments[:10]:
        print(f"{symbol:<8} {stats['weight']:<8.3f} {stats['avg_position']:<10.2f} "
              f"{stats['percent_time_long_intended']:<8.1%} {stats['percent_time_short_intended']:<8.1%} {stats['active_days']:<6}")
    
    # Show instruments with highest long/short activity
    print(f"\n--- Top 10 Most Active Long/Short Followers (by Days Active) ---")
    sorted_by_activity = sorted(
        instrument_stats.items(), 
        key=lambda x: x[1]['active_days'], 
        reverse=True
    )
    
    print(f"{'Symbol':<8} {'Days':<6} {'%Long':<8} {'%Short':<8} {'Weight':<8} {'Avg Pos':<10}")
    print("-" * 65)
    
    for symbol, stats in sorted_by_activity[:10]:
        print(f"{symbol:<8} {stats['active_days']:<6} {stats['percent_time_long_intended']:<8.1%} "
              f"{stats['percent_time_short_intended']:<8.1%} {stats['weight']:<8.3f} {stats['avg_position']:<10.2f}")
    
    # Summary of long/short efficiency
    total_active_days = sum(stats['active_days'] for stats in instrument_stats.values())
    avg_long_percentage = sum(stats['percent_time_long_intended'] for stats in instrument_stats.values()) / len(instrument_stats)
    avg_short_percentage = sum(stats['percent_time_short_intended'] for stats in instrument_stats.values()) / len(instrument_stats)
    
    print(f"\n--- Long/Short Trend Following Summary ---")
    print(f"Total instrument-days with positions: {total_active_days:,}")
    print(f"Average % time long across all instruments: {avg_long_percentage:.1%}")
    print(f"Average % time short across all instruments: {avg_short_percentage:.1%}")
    print(f"Instruments with >25% time long: {sum(1 for stats in instrument_stats.values() if stats['percent_time_long_intended'] > 0.25)}")
    print(f"Instruments with >25% time short: {sum(1 for stats in instrument_stats.values() if stats['percent_time_short_intended'] > 0.25)}")
    print(f"Instruments with any activity: {len(instrument_stats)}")

def plot_strategy4_vs_strategy6_equity_curves(strategy4_results, strategy6_results, save_path='results/strategy4_vs_strategy6_equity_curves.png'):
    """
    Plot only the equity curves for Strategy 4 vs Strategy 6 comparison.
    
    Parameters:
        strategy4_results (dict): Results from Strategy 4 backtest.
        strategy6_results (dict): Results from Strategy 6 backtest.
        save_path (str): Path to save the plot.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        s4_config = strategy4_results['config']
        s6_config = strategy6_results['config']
        s4_perf = strategy4_results['performance']
        s6_perf = strategy6_results['performance']
        
        # Build equity curves
        s4_equity = build_account_curve(strategy4_results['portfolio_data']['portfolio_return'], s4_config['capital'])
        s6_equity = build_account_curve(strategy6_results['portfolio_data']['portfolio_return'], s6_config['capital'])
        
        plt.figure(figsize=(14, 8))
        
        # Plot equity curves
        plt.plot(s4_equity.index, s4_equity.values/1e6, 'b-', linewidth=2.5, 
                label=f'Strategy 4: Multi-Instrument (SR: {s4_perf["sharpe_ratio"]:.3f})')
        plt.plot(s6_equity.index, s6_equity.values/1e6, 'r-', linewidth=2.5, 
                label=f'Strategy 6: Long/Short Trend Following (SR: {s6_perf["sharpe_ratio"]:.3f})')
        
        plt.title('Equity Curve Comparison: Multi-Instrument vs Long/Short Trend Following', 
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
        
        print(f"\n✅ Strategy 4 vs Strategy 6 equity curves saved to: {save_path}")
        
    except Exception as e:
        print(f"Error plotting equity curves: {e}")
        import traceback
        traceback.print_exc()

def compare_all_strategies():
    """
    Compare Strategy 4 (no trend filter) vs Strategy 5 (long only) vs Strategy 6 (long/short).
    """
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON: STRATEGY 4 vs 5 vs 6")
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
        
        if strategy4_results and strategy5_results and strategy6_results:
            s4_perf = strategy4_results['performance']
            s5_perf = strategy5_results['performance']
            s6_perf = strategy6_results['performance']
            
            print(f"\n{'Strategy':<15} {'Ann. Return':<12} {'Volatility':<12} {'Sharpe':<8} {'Max DD':<8} {'Time in Market':<15}")
            print("-" * 95)
            
            print(f"{'Strategy 4':<15} {s4_perf['annualized_return']:<12.2%} "
                  f"{s4_perf['annualized_volatility']:<12.2%} "
                  f"{s4_perf['sharpe_ratio']:<8.3f} "
                  f"{s4_perf['max_drawdown_pct']:<8.1f}% "
                  f"{'100.0%':<15}")
            
            s5_time_in_market = (s5_perf['avg_long_signals'] / s5_perf['num_instruments']) * 100
            print(f"{'Strategy 5':<15} {s5_perf['annualized_return']:<12.2%} "
                  f"{s5_perf['annualized_volatility']:<12.2%} "
                  f"{s5_perf['sharpe_ratio']:<8.3f} "
                  f"{s5_perf['max_drawdown_pct']:<8.1f}% "
                  f"{s5_time_in_market:<15.1f}%")
            
            s6_time_in_market = ((s6_perf['avg_long_signals'] + s6_perf['avg_short_signals']) / s6_perf['num_instruments']) * 100
            print(f"{'Strategy 6':<15} {s6_perf['annualized_return']:<12.2%} "
                  f"{s6_perf['annualized_volatility']:<12.2%} "
                  f"{s6_perf['sharpe_ratio']:<8.3f} "
                  f"{s6_perf['max_drawdown_pct']:<8.1f}% "
                  f"{s6_time_in_market:<15.1f}%")
            
            print(f"\n--- Strategy 6 vs Strategy 5 Analysis ---")
            return_diff = s6_perf['annualized_return'] - s5_perf['annualized_return']
            vol_diff = s6_perf['annualized_volatility'] - s5_perf['annualized_volatility']
            sharpe_diff = s6_perf['sharpe_ratio'] - s5_perf['sharpe_ratio']
            dd_diff = s6_perf['max_drawdown_pct'] - s5_perf['max_drawdown_pct']
            
            print(f"Return Difference: {return_diff:+.2%}")
            print(f"Volatility Difference: {vol_diff:+.2%}")
            print(f"Sharpe Difference: {sharpe_diff:+.3f}")
            print(f"Max Drawdown Difference: {dd_diff:+.1f}%")
            print(f"Time in Market Difference: {s6_time_in_market - s5_time_in_market:+.1f}%")
            
            s6_long_pct = (s6_perf['avg_long_signals'] / s6_perf['num_instruments']) * 100
            s6_short_pct = (s6_perf['avg_short_signals'] / s6_perf['num_instruments']) * 100
            print(f"\nStrategy 6 Position Breakdown:")
            print(f"  Time Long: {s6_long_pct:.1f}%")
            print(f"  Time Short: {s6_short_pct:.1f}%")
            print(f"  Time Out: {100 - s6_time_in_market:.1f}%")
            
            return {
                'strategy4': strategy4_results,
                'strategy5': strategy5_results,
                'strategy6': strategy6_results
            }
        
    except Exception as e:
        print(f"Error in strategy comparison: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Test Strategy 6 implementation.
    """
    print("=" * 60)
    print("TESTING STRATEGY 6: LONG/SHORT TREND FOLLOWING")
    print("=" * 60)
    
    try:
        # Run Strategy 6 backtest
        results = backtest_long_short_trend_strategy(
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
        
        # Analyze results
        analyze_long_short_results(results)
        
        # Compare all strategies
        comparison = compare_all_strategies()
        
        # Plot Strategy 4 vs Strategy 6 equity curves
        if comparison and comparison['strategy4'] and comparison['strategy6']:
            plot_strategy4_vs_strategy6_equity_curves(comparison['strategy4'], comparison['strategy6'])
        
        print(f"\nStrategy 6 backtest completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error in Strategy 6 backtest: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
