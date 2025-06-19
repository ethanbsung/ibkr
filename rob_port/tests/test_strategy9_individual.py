#!/usr/bin/env python3
"""
Test Strategy 9: Multiple Trend Following Rules on Individual Instruments

This module provides comprehensive testing of Strategy 9 (Chapter 9) on individual instruments,
allowing for detailed analysis of how multiple trend following rules perform across different
markets and timeframes.

Key Features:
- Individual instrument backtesting with Strategy 9
- Detailed forecast analysis by instrument
- Performance comparison across instruments
- Visualization of results per instrument
- Analysis of individual EWMAC filter performance
- Individual instrument equity curves and statistics

Following book implementation exactly with no lookahead bias.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def backtest_strategy9_single_instrument(symbol: str, data_dir: str = 'Data', 
                                       capital: float = 1000000, risk_target: float = 0.2,
                                       short_span: int = 32, long_years: int = 10, 
                                       min_vol_floor: float = 0.05,
                                       forecast_combination: str = 'five_filters',
                                       buffer_fraction: float = 0.1,
                                       start_date: Optional[str] = None, 
                                       end_date: Optional[str] = None,
                                       debug_forecasts: bool = False) -> Dict:
    """
    Backtest Strategy 9 on a single instrument with detailed analysis.
    
    Implementation follows book exactly: Use multiple EWMAC trend filters,
    combine them with weights and FDM, then scale positions accordingly.
    
    Parameters:
        symbol (str): Instrument symbol (e.g., 'mes', 'cl', 'ng').
        data_dir (str): Directory containing price data files.
        capital (float): Initial capital for backtesting.
        risk_target (float): Target risk fraction.
        short_span (int): EWMA span for short-run volatility.
        long_years (int): Years for long-run volatility average.
        min_vol_floor (float): Minimum volatility floor.
        forecast_combination (str): Which forecast combination to use.
        buffer_fraction (float): Buffer fraction for trading.
        start_date (str): Start date for backtest (YYYY-MM-DD).
        end_date (str): End date for backtest (YYYY-MM-DD).
        debug_forecasts (bool): Whether to print detailed forecast debugging.
    
    Returns:
        dict: Comprehensive single instrument backtest results.
    """
    print("=" * 80)
    print(f"STRATEGY 9: SINGLE INSTRUMENT TEST - {symbol.upper()}")
    print("=" * 80)
    
    # Load instrument specifications and data
    instruments_file = os.path.join('..', data_dir, 'instruments.csv')
    if not os.path.exists(instruments_file):
        instruments_file = os.path.join(data_dir, 'instruments.csv')
    
    all_instruments_specs_df = load_instrument_data(instruments_file)
    try:
        specs = get_instrument_specs(symbol, all_instruments_specs_df)
        multiplier = specs['multiplier']
        print(f"Instrument: {symbol.upper()}")
        print(f"Multiplier: {multiplier}")
    except:
        raise ValueError(f"Could not find specifications for instrument: {symbol}")
    
    # Load price data for the specific instrument (try both cases)
    file_path = os.path.join('..', data_dir, f"{symbol.lower()}_daily_data.csv")
    if not os.path.exists(file_path):
        file_path = os.path.join(data_dir, f"{symbol.lower()}_daily_data.csv")
    if not os.path.exists(file_path):
        file_path = os.path.join('..', data_dir, f"{symbol}_daily_data.csv")
    if not os.path.exists(file_path):
        file_path = os.path.join(data_dir, f"{symbol}_daily_data.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found for {symbol}. Tried: {symbol.lower()}_daily_data.csv and {symbol}_daily_data.csv")
    
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    if 'Last' not in df.columns:
        raise ValueError(f"'Last' column not found in data for {symbol}")
    
    if 'Time' not in df.columns:
        raise ValueError(f"'Time' column not found in data for {symbol}")
    
    # Set Time as index and parse as dates
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    
    print(f"Raw data loaded: {len(df)} rows from {df.index.min().date()} to {df.index.max().date()}")
    
    # Get trend filter and forecast configurations
    filter_config = get_trend_filter_configs()
    forecast_configs = get_forecast_weights_and_fdm()
    selected_config = forecast_configs[forecast_combination]
    
    print(f"\nStrategy Configuration:")
    print(f"  Capital: ${capital:,.0f}")
    print(f"  Risk Target: {risk_target:.1%}")
    print(f"  Forecast Combination: {forecast_combination}")
    print(f"  Trend Filters: {', '.join(selected_config['filters'])}")
    print(f"  Forecast Weights: {selected_config['weights']}")
    print(f"  FDM: {selected_config['fdm']}")
    print(f"  Buffer Fraction: {buffer_fraction}")
    
    # Preprocess data
    df['daily_price_change_pct'] = df['Last'].pct_change()
    raw_returns_for_vol = df['daily_price_change_pct'].dropna()
    
    if len(raw_returns_for_vol) < 300:  # Need sufficient data for slowest trend filter
        raise ValueError(f"Insufficient data for {symbol}: {len(raw_returns_for_vol)} days (need at least 300)")
    
    # Calculate blended volatility (no lookahead bias)
    blended_vol_series = calculate_blended_volatility(
        raw_returns_for_vol, short_span=short_span, long_years=long_years, min_vol_floor=min_vol_floor
    )
    df['vol_forecast'] = blended_vol_series.shift(1).reindex(df.index).ffill().fillna(min_vol_floor)
    
    # Calculate individual EWMAC forecasts for detailed analysis
    individual_forecasts = {}
    for filter_name in selected_config['filters']:
        if filter_name in filter_config:
            config = filter_config[filter_name]
            
            raw_forecast = calculate_fast_raw_forecast(
                df['Last'], 
                config['fast_span'], 
                config['slow_span'],
                short_span,
                long_years,
                min_vol_floor
            )
            
            scaled_forecast = raw_forecast * config['forecast_scalar']
            capped_forecast = np.clip(scaled_forecast, -20.0, 20.0)
            
            # Shift to prevent lookahead bias
            individual_forecasts[filter_name] = capped_forecast.shift(1).reindex(df.index).fillna(0)
    
    # Calculate combined forecast using multiple trend filters (no lookahead bias)
    combined_forecast_series = calculate_multiple_trend_forecasts(
        df['Last'], filter_config, selected_config, 20.0, short_span, long_years, min_vol_floor
    )
    df['combined_forecast'] = combined_forecast_series.shift(1).reindex(df.index).fillna(0)
    
    # Add individual forecasts to dataframe for analysis
    for filter_name, forecast_series in individual_forecasts.items():
        df[f'forecast_{filter_name.lower()}'] = forecast_series
    
    # Remove NaN rows
    df.dropna(subset=['Last', 'vol_forecast', 'daily_price_change_pct'], inplace=True)
    
    if df.empty:
        raise ValueError(f"No valid data remaining for {symbol} after preprocessing")
    
    # Determine backtest period
    backtest_start_dt = pd.to_datetime(start_date) if start_date else df.index.min()
    backtest_end_dt = pd.to_datetime(end_date) if end_date else df.index.max()
    
    backtest_start_dt = max(backtest_start_dt, df.index.min())
    backtest_end_dt = min(backtest_end_dt, df.index.max())
    
    if backtest_start_dt >= backtest_end_dt:
        raise ValueError(f"Invalid backtest period: {backtest_start_dt} to {backtest_end_dt}")
    
    # Filter data to backtest period
    backtest_df = df[(df.index >= backtest_start_dt) & (df.index <= backtest_end_dt)].copy()
    
    print(f"\nBacktest Period:")
    print(f"  Start: {backtest_start_dt.date()}")
    print(f"  End: {backtest_end_dt.date()}")
    print(f"  Duration: {len(backtest_df)} trading days")
    
    # Initialize tracking variables
    portfolio_equity = capital
    daily_records = []
    current_position = 0.0
    
    # Main backtesting loop
    for idx, (current_date, row) in enumerate(backtest_df.iterrows()):
        if idx == 0:
            # First day setup
            record = {
                'date': current_date,
                'price': row['Last'],
                'position': 0.0,
                'optimal_position': 0.0,
                'combined_forecast': row['combined_forecast'],
                'vol_forecast': row['vol_forecast'],
                'daily_pnl': 0.0,
                'portfolio_return': 0.0,
                'equity': portfolio_equity,
                'trade_size': 0.0,
                'trades': 0
            }
            
            # Add individual forecasts
            for filter_name in selected_config['filters']:
                col_name = f'forecast_{filter_name.lower()}'
                if col_name in backtest_df.columns:
                    record[filter_name] = row[col_name]
                else:
                    record[filter_name] = 0.0
            
            daily_records.append(record)
            continue
        
        # Get previous day data for position sizing
        prev_idx = idx - 1
        prev_date = backtest_df.index[prev_idx]
        prev_row = backtest_df.iloc[prev_idx]
        
        # Position sizing based on previous day's close and current forecasts
        price_for_sizing = prev_row['Last']
        vol_for_sizing = row['vol_forecast']
        forecast_for_sizing = row['combined_forecast']
        
        # Calculate optimal position size (single instrument has weight=1.0, IDM=1.0)
        if (pd.notna(price_for_sizing) and pd.notna(vol_for_sizing) and 
            pd.notna(forecast_for_sizing) and vol_for_sizing > 0):
            
            vol_for_sizing = max(vol_for_sizing, min_vol_floor)
            
            optimal_position = calculate_strategy9_position_size(
                symbol=symbol, capital=portfolio_equity, weight=1.0, 
                idm=1.0, price=price_for_sizing, volatility=vol_for_sizing, 
                multiplier=multiplier, combined_forecast=forecast_for_sizing, 
                risk_target=risk_target
            )
            
            # Apply buffering
            buffer_width = calculate_buffer_width(
                symbol, portfolio_equity, 1.0, 1.0, 
                price_for_sizing, vol_for_sizing, multiplier, 
                risk_target, 1.0, buffer_fraction
            )
            
            new_position, trade_size = calculate_buffered_position(
                optimal_position, current_position, buffer_width
            )
        else:
            optimal_position = 0.0
            new_position = current_position
            trade_size = 0.0
        
        # Calculate P&L based on position held during the day
        price_start = prev_row['Last']
        price_end = row['Last']
        
        daily_pnl = current_position * multiplier * (price_end - price_start)
        daily_return = daily_pnl / portfolio_equity if portfolio_equity > 0 else 0.0
        
        # Update equity and position for next iteration
        portfolio_equity *= (1 + daily_return)
        current_position = new_position
        
        # Record daily results
        record = {
            'date': current_date,
            'price': row['Last'],
            'position': current_position,
            'optimal_position': optimal_position,
            'combined_forecast': forecast_for_sizing,
            'vol_forecast': vol_for_sizing,
            'daily_pnl': daily_pnl,
            'portfolio_return': daily_return,
            'equity': portfolio_equity,
            'trade_size': trade_size,
            'trades': 1 if abs(trade_size) > 0.01 else 0
        }
        
        # Add individual forecasts
        for filter_name in selected_config['filters']:
            col_name = f'forecast_{filter_name.lower()}'
            if col_name in backtest_df.columns:
                record[filter_name] = row[col_name]
            else:
                record[filter_name] = 0.0
        
        daily_records.append(record)
        
        # Debug output for first few days if requested
        if debug_forecasts and idx <= 5:
            print(f"\nDay {idx} ({current_date.date()}):")
            print(f"  Price: ${price_end:.2f}")
            print(f"  Combined Forecast: {forecast_for_sizing:.2f}")
            print(f"  Volatility: {vol_for_sizing:.3f}")
            print(f"  Optimal Position: {optimal_position:.2f}")
            print(f"  Current Position: {current_position:.2f}")
            print(f"  Trade Size: {trade_size:.2f}")
            print(f"  Daily P&L: ${daily_pnl:.2f}")
    
    # Convert to DataFrame and calculate performance
    results_df = pd.DataFrame(daily_records)
    results_df.set_index('date', inplace=True)
    
    # Calculate comprehensive performance metrics
    returns_series = results_df['portfolio_return']
    equity_curve = build_account_curve(returns_series, capital)
    performance = calculate_comprehensive_performance(equity_curve, returns_series)
    
    # Calculate individual forecast statistics
    forecast_stats = {}
    for filter_name in selected_config['filters']:
        if filter_name in results_df.columns:
            forecasts = results_df[filter_name]
            forecast_stats[filter_name] = {
                'mean': forecasts.mean(),
                'std': forecasts.std(),
                'mean_abs': forecasts.abs().mean(),
                'max': forecasts.max(),
                'min': forecasts.min(),
                'correlation_with_combined': forecasts.corr(results_df['combined_forecast']),
                'correlation_with_returns': forecasts.corr(returns_series)
            }
    
    # Calculate trading statistics
    trading_stats = {
        'total_trades': results_df['trades'].sum(),
        'avg_daily_trades': results_df['trades'].mean(),
        'trading_days': len(results_df[results_df['trades'] > 0]),
        'avg_position': results_df['position'].mean(),
        'avg_abs_position': results_df['position'].abs().mean(),
        'max_position': results_df['position'].max(),
        'min_position': results_df['position'].min(),
        'position_changes': (results_df['position'].diff() != 0).sum()
    }
    
    # Add instrument-specific metrics to performance
    performance.update({
        'symbol': symbol,
        'multiplier': multiplier,
        'forecast_combination': forecast_combination,
        'capital': capital,
        'risk_target': risk_target,
        'buffer_fraction': buffer_fraction,
        'backtest_start': backtest_start_dt,
        'backtest_end': backtest_end_dt,
        'total_trades': trading_stats['total_trades'],
        'avg_daily_trades': trading_stats['avg_daily_trades'],
        'avg_combined_forecast': results_df['combined_forecast'].mean(),
        'avg_abs_combined_forecast': results_df['combined_forecast'].abs().mean(),
        'avg_volatility': results_df['vol_forecast'].mean()
    })
    
    print(f"\n--- {symbol.upper()} BACKTEST COMPLETED ---")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Annualized Return: {performance['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {performance['max_drawdown_pct']:.1f}%")
    print(f"Total Trades: {trading_stats['total_trades']:,}")
    print(f"Average Combined Forecast: {results_df['combined_forecast'].mean():.2f}")
    
    return {
        'results_df': results_df,
        'performance': performance,
        'forecast_stats': forecast_stats,
        'trading_stats': trading_stats,
        'config': {
            'symbol': symbol,
            'capital': capital,
            'risk_target': risk_target,
            'forecast_combination': forecast_combination,
            'selected_filters': selected_config['filters'],
            'forecast_weights': selected_config['weights'],
            'fdm': selected_config['fdm'],
            'buffer_fraction': buffer_fraction,
            'backtest_start': backtest_start_dt,
            'backtest_end': backtest_end_dt
        }
    }

def test_multiple_instruments(instruments: List[str], data_dir: str = 'Data',
                            capital: float = 1000000, risk_target: float = 0.2,
                            forecast_combination: str = 'five_filters',
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Dict:
    """
    Test Strategy 9 on multiple individual instruments and compare results.
    
    Parameters:
        instruments (List[str]): List of instrument symbols to test.
        data_dir (str): Directory containing price data files.
        capital (float): Initial capital for each instrument test.
        risk_target (float): Target risk fraction.
        forecast_combination (str): Which forecast combination to use.
        start_date (str): Start date for backtest (YYYY-MM-DD).
        end_date (str): End date for backtest (YYYY-MM-DD).
    
    Returns:
        dict: Combined results for all instruments.
    """
    print("=" * 80)
    print(f"STRATEGY 9: MULTIPLE INSTRUMENT COMPARISON")
    print("=" * 80)
    print(f"Testing {len(instruments)} instruments: {', '.join([s.upper() for s in instruments])}")
    
    all_results = {}
    successful_tests = 0
    failed_tests = 0
    
    for symbol in instruments:
        try:
            print(f"\n--- Testing {symbol.upper()} ---")
            result = backtest_strategy9_single_instrument(
                symbol=symbol.upper(),
                data_dir=data_dir,
                capital=capital,
                risk_target=risk_target,
                forecast_combination=forecast_combination,
                start_date=start_date,
                end_date=end_date,
                debug_forecasts=False
            )
            all_results[symbol] = result
            successful_tests += 1
            
        except Exception as e:
            print(f"❌ Failed to test {symbol.upper()}: {e}")
            failed_tests += 1
            continue
    
    print(f"\n--- TESTING SUMMARY ---")
    print(f"Successful tests: {successful_tests}")
    print(f"Failed tests: {failed_tests}")
    print(f"Success rate: {successful_tests/(successful_tests+failed_tests)*100:.1f}%")
    
    if successful_tests > 0:
        # Create performance comparison
        comparison_data = []
        for symbol, result in all_results.items():
            perf = result['performance']
            comparison_data.append({
                'Symbol': symbol.upper(),
                'Total Return': perf['total_return'],
                'Ann. Return': perf['annualized_return'],
                'Volatility': perf['annualized_volatility'],
                'Sharpe Ratio': perf['sharpe_ratio'],
                'Max Drawdown': perf['max_drawdown_pct'],
                'Total Trades': perf['total_trades'],
                'Avg Forecast': perf['avg_combined_forecast'],
                'Avg |Forecast|': perf['avg_abs_combined_forecast']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Sharpe Ratio', ascending=False)
        
        print(f"\n--- PERFORMANCE RANKING (by Sharpe Ratio) ---")
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        return {
            'individual_results': all_results,
            'comparison_df': comparison_df,
            'summary': {
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'best_performer': comparison_df.iloc[0]['Symbol'] if len(comparison_df) > 0 else None,
                'avg_sharpe': comparison_df['Sharpe Ratio'].mean(),
                'avg_return': comparison_df['Ann. Return'].mean()
            }
        }
    else:
        return {'individual_results': {}, 'comparison_df': pd.DataFrame(), 'summary': {}}

def plot_individual_instrument_analysis(results: Dict, save_path: Optional[str] = None):
    """
    Create comprehensive plots for individual instrument Strategy 9 analysis.
    
    Parameters:
        results (dict): Results from backtest_strategy9_single_instrument.
        save_path (str): Optional path to save the plot.
    """
    symbol = results['config']['symbol']
    results_df = results['results_df']
    performance = results['performance']
    forecast_stats = results['forecast_stats']
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Strategy 9: {symbol.upper()} - Multiple Trend Following Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 1. Equity Curve
    ax1 = plt.subplot(4, 3, 1)
    equity_curve = build_account_curve(results_df['portfolio_return'], 
                                     results['config']['capital'])
    ax1.plot(equity_curve.index, equity_curve.values/1e6, 'darkblue', linewidth=2)
    ax1.set_title(f'{symbol.upper()} Equity Curve')
    ax1.set_ylabel('Portfolio Value ($M)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Price and Position
    ax2 = plt.subplot(4, 3, 2)
    ax2_twin = ax2.twinx()
    ax2.plot(results_df.index, results_df['price'], 'black', linewidth=1, label='Price')
    ax2_twin.plot(results_df.index, results_df['position'], 'red', linewidth=1, label='Position')
    ax2.set_title('Price vs Position')
    ax2.set_ylabel('Price', color='black')
    ax2_twin.set_ylabel('Position (Contracts)', color='red')
    ax2.grid(True, alpha=0.3)
    
    # 3. Combined Forecast
    ax3 = plt.subplot(4, 3, 3)
    ax3.plot(results_df.index, results_df['combined_forecast'], 'green', linewidth=1)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Cap')
    ax3.axhline(y=-20, color='red', linestyle='--', alpha=0.5)
    ax3.set_title('Combined Forecast')
    ax3.set_ylabel('Forecast Value')
    ax3.grid(True, alpha=0.3)
    
    # 4. Individual EWMAC Forecasts
    ax4 = plt.subplot(4, 3, 4)
    selected_filters = results['config']['selected_filters']
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    
    for i, filter_name in enumerate(selected_filters):
        if filter_name in results_df.columns:
            color = colors[i % len(colors)]
            ax4.plot(results_df.index, results_df[filter_name], 
                    color=color, linewidth=1, alpha=0.7, label=filter_name)
    
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title('Individual EWMAC Forecasts')
    ax4.set_ylabel('Forecast Value')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Volatility Forecast
    ax5 = plt.subplot(4, 3, 5)
    ax5.plot(results_df.index, results_df['vol_forecast'] * 100, 'orange', linewidth=1)
    ax5.set_title('Volatility Forecast')
    ax5.set_ylabel('Volatility (%)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Daily Returns
    ax6 = plt.subplot(4, 3, 6)
    returns = results_df['portfolio_return'] * 100
    ax6.plot(returns.index, returns.values, 'gray', linewidth=0.5, alpha=0.7)
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax6.set_title('Daily Returns')
    ax6.set_ylabel('Return (%)')
    ax6.grid(True, alpha=0.3)
    
    # 7. Drawdown
    ax7 = plt.subplot(4, 3, 7)
    drawdown_stats = calculate_maximum_drawdown(equity_curve)
    drawdown_series = drawdown_stats['drawdown_series'] * 100
    ax7.fill_between(drawdown_series.index, drawdown_series.values, 0, 
                     color='red', alpha=0.3)
    ax7.plot(drawdown_series.index, drawdown_series.values, 'red', linewidth=1)
    ax7.set_title('Drawdown')
    ax7.set_ylabel('Drawdown (%)')
    ax7.grid(True, alpha=0.3)
    
    # 8. Forecast Statistics Bar Chart
    ax8 = plt.subplot(4, 3, 8)
    filter_names = list(forecast_stats.keys())
    mean_abs_forecasts = [forecast_stats[f]['mean_abs'] for f in filter_names]
    
    bars = ax8.bar(range(len(filter_names)), mean_abs_forecasts, 
                   color=colors[:len(filter_names)])
    ax8.set_title('Mean Absolute Forecast by Filter')
    ax8.set_ylabel('Mean |Forecast|')
    ax8.set_xticks(range(len(filter_names)))
    ax8.set_xticklabels(filter_names, rotation=45)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Forecast Correlations Heatmap
    ax9 = plt.subplot(4, 3, 9)
    corr_matrix = []
    for f1 in filter_names:
        row = []
        for f2 in filter_names:
            if f1 in results_df.columns and f2 in results_df.columns:
                corr = results_df[f1].corr(results_df[f2])
                row.append(corr)
            else:
                row.append(0)
        corr_matrix.append(row)
    
    im = ax9.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax9.set_title('Forecast Correlations')
    ax9.set_xticks(range(len(filter_names)))
    ax9.set_yticks(range(len(filter_names)))
    ax9.set_xticklabels(filter_names, rotation=45)
    ax9.set_yticklabels(filter_names)
    plt.colorbar(im, ax=ax9)
    
    # 10. Position Size Distribution
    ax10 = plt.subplot(4, 3, 10)
    positions = results_df['position'][results_df['position'] != 0]
    ax10.hist(positions, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax10.set_title('Position Size Distribution')
    ax10.set_xlabel('Position (Contracts)')
    ax10.set_ylabel('Frequency')
    ax10.grid(True, alpha=0.3)
    
    # 11. Trading Activity
    ax11 = plt.subplot(4, 3, 11)
    # Rolling 30-day trade count
    rolling_trades = results_df['trades'].rolling(30).sum()
    ax11.plot(rolling_trades.index, rolling_trades.values, 'purple', linewidth=1)
    ax11.set_title('Trading Activity (30-day Rolling)')
    ax11.set_ylabel('Trades per 30 Days')
    ax11.grid(True, alpha=0.3)
    
    # 12. Performance Statistics Text
    ax12 = plt.subplot(4, 3, 12)
    ax12.axis('off')
    
    stats_text = f"""Performance Statistics:
    
Total Return: {performance['total_return']:.1%}
Annualized Return: {performance['annualized_return']:.1%}
Volatility: {performance['annualized_volatility']:.1%}
Sharpe Ratio: {performance['sharpe_ratio']:.3f}
Max Drawdown: {performance['max_drawdown_pct']:.1f}%
Skewness: {performance['skewness']:.3f}

Trading Statistics:
Total Trades: {performance['total_trades']:,}
Avg Daily Trades: {performance['avg_daily_trades']:.2f}
Avg Combined Forecast: {performance['avg_combined_forecast']:.2f}
Avg |Combined Forecast|: {performance['avg_abs_combined_forecast']:.2f}

Backtest Period:
Start: {performance['backtest_start'].strftime('%Y-%m-%d')}
End: {performance['backtest_end'].strftime('%Y-%m-%d')}"""
    
    ax12.text(0.05, 0.95, stats_text, transform=ax12.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    # Format dates on x-axis for time series plots
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax11]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Individual instrument analysis plot saved to: {save_path}")
    
    return fig

def plot_multi_instrument_comparison(results: Dict, save_path: Optional[str] = None):
    """
    Create comparison plots for multiple instruments tested with Strategy 9.
    
    Parameters:
        results (dict): Results from test_multiple_instruments.
        save_path (str): Optional path to save the plot.
    """
    individual_results = results['individual_results']
    comparison_df = results['comparison_df']
    
    if len(individual_results) == 0:
        print("No results to plot")
        return None
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Strategy 9: Multiple Instruments Comparison', fontsize=16, fontweight='bold')
    
    # 1. Equity Curves Comparison
    ax1 = plt.subplot(3, 3, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(individual_results)))
    
    for i, (symbol, result) in enumerate(individual_results.items()):
        results_df = result['results_df']
        capital = result['config']['capital']
        equity_curve = build_account_curve(results_df['portfolio_return'], capital)
        
        # Normalize to starting value of 100
        normalized_equity = (equity_curve / equity_curve.iloc[0]) * 100
        ax1.plot(normalized_equity.index, normalized_equity.values, 
                color=colors[i], linewidth=2, label=symbol.upper())
    
    ax1.set_title('Normalized Equity Curves')
    ax1.set_ylabel('Normalized Value (Start = 100)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Returns Comparison
    ax2 = plt.subplot(3, 3, 2)
    symbols = comparison_df['Symbol'].tolist()
    returns = comparison_df['Ann. Return'].tolist()
    bars = ax2.bar(symbols, [r*100 for r in returns], color=colors[:len(symbols)])
    ax2.set_title('Annualized Returns')
    ax2.set_ylabel('Return (%)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, returns):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Sharpe Ratios Comparison
    ax3 = plt.subplot(3, 3, 3)
    sharpe_ratios = comparison_df['Sharpe Ratio'].tolist()
    bars = ax3.bar(symbols, sharpe_ratios, color=colors[:len(symbols)])
    ax3.set_title('Sharpe Ratios')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, sharpe_ratios):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Volatility Comparison
    ax4 = plt.subplot(3, 3, 4)
    volatilities = comparison_df['Volatility'].tolist()
    bars = ax4.bar(symbols, [v*100 for v in volatilities], color=colors[:len(symbols)])
    ax4.set_title('Annualized Volatility')
    ax4.set_ylabel('Volatility (%)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Max Drawdown Comparison
    ax5 = plt.subplot(3, 3, 5)
    drawdowns = comparison_df['Max Drawdown'].tolist()
    bars = ax5.bar(symbols, drawdowns, color=colors[:len(symbols)])
    ax5.set_title('Maximum Drawdown')
    ax5.set_ylabel('Max Drawdown (%)')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Trading Activity Comparison
    ax6 = plt.subplot(3, 3, 6)
    trades = comparison_df['Total Trades'].tolist()
    bars = ax6.bar(symbols, trades, color=colors[:len(symbols)])
    ax6.set_title('Total Trades')
    ax6.set_ylabel('Number of Trades')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Average Forecast Comparison
    ax7 = plt.subplot(3, 3, 7)
    avg_forecasts = comparison_df['Avg Forecast'].tolist()
    bars = ax7.bar(symbols, avg_forecasts, color=colors[:len(symbols)])
    ax7.set_title('Average Combined Forecast')
    ax7.set_ylabel('Avg Forecast')
    ax7.tick_params(axis='x', rotation=45)
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 8. Risk-Return Scatter
    ax8 = plt.subplot(3, 3, 8)
    for i, (symbol, return_val, vol_val, sharpe_val) in enumerate(zip(symbols, returns, volatilities, sharpe_ratios)):
        ax8.scatter(vol_val*100, return_val*100, s=100, color=colors[i], label=symbol)
        ax8.annotate(symbol, (vol_val*100, return_val*100), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    ax8.set_title('Risk-Return Profile')
    ax8.set_xlabel('Volatility (%)')
    ax8.set_ylabel('Return (%)')
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary Statistics Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary = results['summary']
    stats_text = f"""Multi-Instrument Summary:
    
Total Instruments Tested: {summary.get('successful_tests', 0)}
Failed Tests: {summary.get('failed_tests', 0)}

Best Performer: {summary.get('best_performer', 'N/A')}
Average Sharpe Ratio: {summary.get('avg_sharpe', 0):.3f}
Average Ann. Return: {summary.get('avg_return', 0)*100:.1f}%

Top 3 by Sharpe:
{comparison_df.head(3)[['Symbol', 'Sharpe Ratio']].to_string(index=False)}"""
    
    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-instrument comparison plot saved to: {save_path}")
    
    return fig

def main():
    """
    Main function to demonstrate Strategy 9 individual instrument testing.
    """
    print("=" * 80)
    print("STRATEGY 9: INDIVIDUAL INSTRUMENT TESTING")
    print("=" * 80)
    
    # Configuration
    capital = 1000000
    risk_target = 0.2
    forecast_combination = 'five_filters'
    
    # Test a few major instruments individually
    test_instruments = ['MES', 'QM', 'QG', 'ZB', 'MGC', 'SI']
    
    print(f"Testing Strategy 9 on individual instruments:")
    print(f"Instruments: {', '.join([s.upper() for s in test_instruments])}")
    print(f"Capital per instrument: ${capital:,}")
    print(f"Risk target: {risk_target:.1%}")
    print(f"Forecast combination: {forecast_combination}")
    
    try:
        # Test single instrument (MES as example)
        print(f"\n--- DETAILED SINGLE INSTRUMENT TEST: MES ---")
        single_result = backtest_strategy9_single_instrument(
            symbol='MES',
            capital=capital,
            risk_target=risk_target,
            forecast_combination=forecast_combination,
            debug_forecasts=True
        )
        
        # Plot detailed analysis for single instrument
        os.makedirs('results', exist_ok=True)
        plot_individual_instrument_analysis(
            single_result, 
            save_path='results/strategy9_mes_individual_analysis.png'
        )
        
        # Test multiple instruments
        print(f"\n--- MULTIPLE INSTRUMENT COMPARISON ---")
        multi_results = test_multiple_instruments(
            instruments=test_instruments,
            capital=capital,
            risk_target=risk_target,
            forecast_combination=forecast_combination
        )
        
        # Plot multi-instrument comparison
        if len(multi_results['individual_results']) > 0:
            plot_multi_instrument_comparison(
                multi_results,
                save_path='results/strategy9_multi_instrument_comparison.png'
            )
        
        print(f"\n✅ Strategy 9 individual instrument testing completed!")
        print(f"Results saved to results/ directory")
        
        return {
            'single_result': single_result,
            'multi_results': multi_results
        }
        
    except Exception as e:
        print(f"❌ Error in Strategy 9 individual testing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()