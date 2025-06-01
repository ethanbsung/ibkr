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
                                     forecast_config: dict, cap: float = 20.0) -> pd.Series:
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
    
    Returns:
        pd.Series: Combined capped forecast.
    """
    individual_forecasts = {}
    
    # Calculate individual forecasts for each filter
    for filter_name in forecast_config['filters']:
        if filter_name in filter_config:
            config = filter_config[filter_name]
            
            # Calculate raw forecast
            raw_forecast = calculate_fast_raw_forecast(
                prices, 
                config['fast_span'], 
                config['slow_span']
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
                                   short_span=32, long_years=10, 
                                   forecast_combination='five_filters',
                                   buffer_fraction=0.1,
                                   weight_method='handcrafted',
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
        forecast_combination (str): Which forecast combination to use.
        buffer_fraction (float): Buffer fraction for trading.
        weight_method (str): Method for calculating instrument weights.
        start_date (str): Start date for backtest (YYYY-MM-DD).
        end_date (str): End date for backtest (YYYY-MM-DD).
        debug_forecasts (bool): Whether to print forecast debug info.
    
    Returns:
        dict: Comprehensive backtest results.
    """
    print("=" * 60)
    print("STRATEGY 9: MULTIPLE TREND FOLLOWING RULES")
    print("=" * 60)
    
    # Load all instrument data
    instrument_data = load_all_instrument_data(data_dir)
    
    if len(instrument_data) == 0:
        raise ValueError("No instrument data loaded successfully")
    
    # Load instrument specifications
    instruments_df = load_instrument_data()
    
    # Get trend filter and forecast configurations
    filter_config = get_trend_filter_configs()
    forecast_configs = get_forecast_weights_and_fdm()
    selected_config = forecast_configs[forecast_combination]
    
    print(f"\nPortfolio Configuration:")
    print(f"  Instruments: {len(instrument_data)}")
    print(f"  Capital: ${capital:,.0f}")
    print(f"  Risk Target: {risk_target:.1%}")
    print(f"  Weight Method: {weight_method}")
    print(f"  Forecast Combination: {forecast_combination}")
    print(f"  Trend Filters: {', '.join(selected_config['filters'])}")
    print(f"  Forecast Weights: {selected_config['weights']}")
    print(f"  FDM: {selected_config['fdm']}")
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
    
    # Process each instrument and calculate volatility forecasts + combined trend forecasts + buffering
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
        
        if len(df_filtered) < 300:  # Need sufficient data for slowest trend filter (256 + buffer)
            continue
        
        # Calculate blended volatility forecast (same as previous strategies)
        df_filtered['blended_vol'] = calculate_blended_volatility(
            df_filtered['returns'], short_span=short_span, long_years=long_years
        )
        
        # Calculate combined forecast using multiple trend filters
        df_filtered['combined_forecast'] = calculate_multiple_trend_forecasts(
            df_filtered['Last'], filter_config, selected_config
        )
        
        # Debug first instrument forecasts if requested
        if debug_forecasts and symbol == list(instrument_data.keys())[0]:
            print(f"\n=== FORECAST DEBUG FOR {symbol} ===")
            print(f"Sample Combined Forecasts: {df_filtered['combined_forecast'].dropna()[:10].values}")
            print(f"Average Combined Forecast: {df_filtered['combined_forecast'].mean():.3f}")
            print(f"Average Absolute Forecast: {df_filtered['combined_forecast'].abs().mean():.3f}")
        
        # Calculate position sizes with combined forecast scaling and buffering
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
                prev_forecast = df_filtered['combined_forecast'].iloc[i-1]
                
                if (np.isnan(prev_vol) or prev_vol <= 0 or np.isnan(prev_forecast)):
                    new_position = current_position
                    trade_size = 0
                    optimal_pos = 0
                    buffer_width = 0
                else:
                    # Calculate optimal position
                    optimal_pos = calculate_strategy9_position_size(
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
                portfolio_df.loc[date, f'{symbol}_forecast'] = df.loc[date, 'combined_forecast']
                
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
    print(f"Average combined forecast: {portfolio_df['avg_forecast'].mean():.2f}")
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
    performance['forecast_combination'] = forecast_combination
    performance['selected_filters'] = selected_config['filters']
    performance['forecast_weights'] = selected_config['weights']
    performance['fdm'] = selected_config['fdm']
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
            'forecast_combination': forecast_combination,
            'selected_filters': selected_config['filters'],
            'forecast_weights': selected_config['weights'],
            'fdm': selected_config['fdm'],
            'buffer_fraction': buffer_fraction,
            'weight_method': weight_method,
            'backtest_start': backtest_start,
            'backtest_end': backtest_end
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
            equity_curve = build_account_curve(portfolio_data['strategy_returns'], 100)
            
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
    print(f"\n--- Multiple Trend Following Characteristics ---")
    print(f"Average Active Instruments: {performance['avg_active_instruments']:.1f}")
    print(f"Average Combined Forecast: {performance['avg_forecast']:.2f}")
    print(f"Average Absolute Forecast: {performance['avg_abs_forecast']:.2f}")
    print(f"Forecast Combination: {config['forecast_combination']}")
    print(f"Trend Filters: {', '.join(config['selected_filters'])}")
    print(f"Forecast Weights: {config['forecast_weights']}")
    print(f"FDM: {config['fdm']}")
    
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

def main():
    """
    Test Strategy 9 implementation and compare all strategies.
    """
    print("=" * 60)
    print("TESTING STRATEGY 9: MULTIPLE TREND FOLLOWING RULES")
    print("=" * 60)
    
    try:
        # Test different forecast combinations
        forecast_combinations = ['five_filters', 'four_filters', 'three_filters']
        
        print("Testing different forecast combinations...")
        
        for combination in forecast_combinations:
            print(f"\n--- Testing {combination.replace('_', ' ').title()} ---")
            
            results = backtest_multiple_trend_strategy(
                data_dir='Data',
                capital=50000000,
                risk_target=0.2,
                forecast_combination=combination,
                weight_method='handcrafted',
                debug_forecasts=(combination == 'five_filters')  # Debug first one only
            )
            
            # Quick performance summary
            perf = results['performance']
            print(f"  Annualized Return: {perf['annualized_return']:.2%}")
            print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
            print(f"  Average Daily Trades: {perf['avg_daily_trades']:.1f}")
            
            if combination == 'five_filters':
                # Analyze detailed results for the default combination
                analyze_multiple_trend_results(results)
        
        # Compare all strategies with plotting
        print(f"\n--- Running Complete Strategy Comparison ---")
        comparison = compare_all_strategies()
        
        print(f"\nStrategy 9 testing completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error in Strategy 9 testing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 