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

#####   STRATEGY 4: MULTI-INSTRUMENT VARIABLE RISK PORTFOLIO   #####

def load_all_instrument_data(data_dir='Data'):
    """
    Load all available instrument data from CSV files.
    
    Parameters:
        data_dir (str): Directory containing the data files.
    
    Returns:
        dict: Dictionary with symbol as key and DataFrame as value.
    """
    # Load instrument specs
    instruments_df = load_instrument_data()
    
    instrument_data = {}
    failed_loads = []
    
    for _, row in instruments_df.iterrows():
        symbol = row['Symbol']
        filename = f"{symbol.lower()}_daily_data.csv"
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, parse_dates=['Time'])
                df.set_index('Time', inplace=True)
                df = df.dropna()
                
                # Calculate returns
                df['returns'] = df['Last'].pct_change()
                df = df.dropna()
                
                # Only include if we have sufficient data
                if len(df) > 252:  # At least 1 year of data
                    instrument_data[symbol] = df
                else:
                    pass  # Skip without printing
                    
            except Exception as e:
                failed_loads.append((symbol, str(e)))
        else:
            pass  # Skip without printing
    
    print(f"Successfully loaded {len(instrument_data)} instruments")
    return instrument_data

def calculate_idm_from_count(num_instruments):
    """
    Calculate Instrument Diversification Multiplier based on number of instruments.
    
    This is a simplified version. The book references Table 16 and Appendix B
    for more sophisticated calculations.
    
    Parameters:
        num_instruments (int): Number of instruments in portfolio.
    
    Returns:
        float: IDM value.
    """
    # Simple approximation: IDM increases with sqrt of number of instruments
    # but with diminishing returns. Cap at reasonable level.
    if num_instruments <= 1:
        return 1.0
    elif num_instruments <= 5:
        return 1.5
    elif num_instruments <= 10:
        return 2.0
    elif num_instruments <= 20:
        return 2.5
    elif num_instruments <= 50:
        return 3.0
    else:
        return 3.5

def calculate_instrument_weights(instrument_data, method='equal', instruments_df=None):
    """
    Calculate weights for each instrument in the portfolio.
    
    Parameters:
        instrument_data (dict): Dictionary of instrument DataFrames.
        method (str): Weighting method ('equal', 'vol_inverse', 'handcrafted').
        instruments_df (pd.DataFrame): Instrument specifications for handcrafted method.
    
    Returns:
        dict: Dictionary of weights for each instrument.
    """
    symbols = list(instrument_data.keys())
    num_instruments = len(symbols)
    
    if method == 'equal':
        # Equal weights
        weight = 1.0 / num_instruments
        return {symbol: weight for symbol in symbols}
    
    elif method == 'vol_inverse':
        # Inverse volatility weights
        vols = {}
        for symbol, df in instrument_data.items():
            if len(df) > 252:
                vol = df['returns'].std() * np.sqrt(business_days_per_year)
                vols[symbol] = vol
        
        # Calculate inverse volatility weights
        inv_vols = {symbol: 1/vol for symbol, vol in vols.items()}
        total_inv_vol = sum(inv_vols.values())
        
        return {symbol: inv_vol/total_inv_vol for symbol, inv_vol in inv_vols.items()}
    
    elif method == 'handcrafted':
        # Sophisticated handcrafted weighting algorithm
        return calculate_handcrafted_weights(instrument_data, instruments_df)
    
    else:
        # Default to equal weights
        weight = 1.0 / num_instruments
        return {symbol: weight for symbol in symbols}

def calculate_handcrafted_weights(instrument_data, instruments_df):
    """
    Calculate handcrafted instrument weights based on multiple factors.
    
    This algorithm considers:
    1. Risk-adjusted costs (lower is better)
    2. Volatility scaling (for risk parity)
    3. Performance characteristics (Sharpe ratio, skewness)
    4. Diversification benefits across asset classes
    
    Parameters:
        instrument_data (dict): Dictionary of instrument DataFrames.
        instruments_df (pd.DataFrame): Instrument specifications.
    
    Returns:
        dict: Dictionary of optimized weights.
    """
    print(f"\n--- Calculating Handcrafted Weights ---")
    
    # Asset class groupings for diversification
    asset_classes = {
        'bonds': ['ZT', 'Z3N', 'ZF', 'ZN', 'TN', 'TWE', 'ZB', 'YE', 'GBS', 'GBM', 'GBL', 'GBX', 'BTS', 'BTP', 'FBON'],
        'equities': ['MYM', 'MNQ', 'RSV', 'M2K', 'MES', 'CAC40', 'DAX', 'SMI', 'DJ600', 'ESTX50', 'SXAP', 'SXPP', 'SXDP', 'SXIP', 'SX8P', 'SXTP', 'SX6P', 'XINA50', 'SSG', 'TWN'],
        'fx': ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'RP', 'RY', 'UC', 'SIR', 'SND'],
        'commodities': ['ALI', 'HG', 'MGC', 'SCI', 'PA', 'PL', 'SI', 'QM', 'HH', 'RB', 'QG', 'HO', 'AIGCI'],
        'agriculture': ['CSC', 'ZC', 'GF', 'HE', 'LE', 'ZO', 'KE', 'ZR', 'ZS', 'ZM', 'ZL', 'ZW'],
        'volatility': ['VIX', 'V2TX'],
        'crypto': ['MBT', 'ETHUSDRR']
    }
    
    # Initialize weight calculation components
    symbol_scores = {}
    
    for symbol, df in instrument_data.items():
        if len(df) < 252:  # Need minimum data
            continue
            
        try:
            # Get instrument specs
            try:
                specs = get_instrument_specs(symbol, instruments_df)
                sr_cost = specs.get('sr_cost', 0.01)  # Default cost if missing
                if pd.isna(sr_cost) or sr_cost <= 0:
                    sr_cost = 0.01  # Default for invalid values
            except:
                sr_cost = 0.01  # Default for missing instruments
            
            # Calculate key metrics
            returns = df['returns'].dropna()
            
            # 1. Volatility (for inverse vol weighting component)
            vol = returns.std() * np.sqrt(business_days_per_year)
            
            # 2. Performance metrics
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(business_days_per_year) if returns.std() > 0 else 0
            skewness = returns.skew() if len(returns) > 10 else 0
            
            # 3. Cost efficiency (lower SR cost is better)
            cost_efficiency = 1.0 / (sr_cost + 0.001)  # Add small constant to avoid division by zero
            
            # 4. Risk-adjusted performance
            risk_adj_performance = sharpe_ratio * 0.5 + max(0, skewness) * 0.1  # Prefer positive skew
            
            # 5. Volatility scaling factor (inverse vol component)
            vol_scaling = 1.0 / vol if vol > 0.01 else 0
            
            # Combined score (higher is better)
            base_score = (
                cost_efficiency * 0.3 +           # 30% weight on cost efficiency
                vol_scaling * 0.3 +               # 30% weight on volatility scaling
                (risk_adj_performance + 1) * 0.2 + # 20% weight on performance (add 1 to handle negatives)
                1.0 * 0.2                         # 20% base allocation
            )
            
            symbol_scores[symbol] = {
                'base_score': base_score,
                'vol': vol,
                'sharpe': sharpe_ratio,
                'skewness': skewness,
                'sr_cost': sr_cost,
                'cost_efficiency': cost_efficiency,
                'asset_class': None
            }
            
            # Determine asset class
            for ac, symbols in asset_classes.items():
                if symbol in symbols:
                    symbol_scores[symbol]['asset_class'] = ac
                    break
            if symbol_scores[symbol]['asset_class'] is None:
                symbol_scores[symbol]['asset_class'] = 'other'
                
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
    
    # Apply diversification adjustments
    # Target allocation by asset class (these sum to 1.0)
    target_allocations = {
        'bonds': 0.25,
        'equities': 0.30,
        'fx': 0.15,
        'commodities': 0.15,
        'agriculture': 0.08,
        'volatility': 0.02,
        'crypto': 0.03,
        'other': 0.02
    }
    
    # Calculate weights within each asset class
    final_weights = {}
    
    for asset_class, target_allocation in target_allocations.items():
        # Get instruments in this asset class
        class_instruments = {k: v for k, v in symbol_scores.items() 
                           if v['asset_class'] == asset_class}
        
        if not class_instruments:
            continue
        
        # Calculate relative weights within asset class based on scores
        total_class_score = sum(inst['base_score'] for inst in class_instruments.values())
        
        if total_class_score > 0:
            for symbol, data in class_instruments.items():
                class_weight = data['base_score'] / total_class_score
                final_weights[symbol] = class_weight * target_allocation
        
        # Reduced output - only print summary
    
    # Normalize weights to sum to 1.0
    total_weight = sum(final_weights.values())
    if total_weight > 0:
        final_weights = {k: v/total_weight for k, v in final_weights.items()}
    
    # Display top allocations
    sorted_weights = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 Weighted Instruments:")
    print(f"{'Symbol':<8} {'Weight':<8} {'AssetClass':<12} {'Sharpe':<8} {'Vol':<8} {'Cost':<8}")
    print("-" * 65)
    
    for symbol, weight in sorted_weights[:10]:
        data = symbol_scores[symbol]
        print(f"{symbol:<8} {weight:<8.3f} {data['asset_class']:<12} {data['sharpe']:<8.3f} "
              f"{data['vol']:<8.2%} {data['sr_cost']:<8.4f}")
    
    return final_weights

def calculate_portfolio_position_size(symbol, capital, weight, idm, price, volatility, 
                                    multiplier, risk_target=0.2, fx_rate=1.0):
    """
    Calculate position size for an instrument in a multi-instrument portfolio.
    
    Formula from Strategy 4:
        N = (Capital × IDM × Weight × τ) ÷ (Multiplier × Price × FX × σ%)
    
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
    
    Returns:
        float: Number of contracts for this instrument.
    """
    if np.isnan(volatility) or volatility <= 0:
        return 0
    
    numerator = capital * idm * weight * risk_target
    denominator = multiplier * price * fx_rate * volatility
    
    position_size = numerator / denominator
    
    # Protect against infinite or extremely large position sizes
    if np.isinf(position_size) or position_size > 100000:
        return 0
    
    return position_size

def backtest_multi_instrument_strategy(data_dir='Data', capital=50000000, risk_target=0.2,
                                     short_span=32, long_years=10, weight_method='equal',
                                     start_date=None, end_date=None):
    """
    Backtest Strategy 4: Multi-instrument portfolio with variable risk scaling.
    
    Parameters:
        data_dir (str): Directory containing price data files.
        capital (float): Initial capital.
        risk_target (float): Target risk fraction.
        short_span (int): EWMA span for short-run volatility.
        long_years (int): Years for long-run volatility average.
        weight_method (str): Method for calculating instrument weights.
        start_date (str): Start date for backtest (YYYY-MM-DD).
        end_date (str): End date for backtest (YYYY-MM-DD).
    
    Returns:
        dict: Comprehensive backtest results.
    """
    print("=" * 60)
    print("STRATEGY 4: MULTI-INSTRUMENT VARIABLE RISK PORTFOLIO")
    print("=" * 60)
    
    # Load all instrument data
    instrument_data = load_all_instrument_data(data_dir)
    
    if len(instrument_data) == 0:
        raise ValueError("No instrument data loaded successfully")
    
    # Load instrument specifications
    instruments_df = load_instrument_data()
    
    print(f"\nPortfolio Configuration:")
    print(f"  Instruments: {len(instrument_data)}")
    print(f"  Capital: ${capital:,.0f}")
    print(f"  Risk Target: {risk_target:.1%}")
    print(f"  Weight Method: {weight_method}")
    
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
    
    # Process each instrument and calculate volatility forecasts
    processed_data = {}
    
    for symbol, df in instrument_data.items():
        # Get instrument specs
        try:
            specs = get_instrument_specs(symbol, instruments_df)
            multiplier = specs['multiplier']
        except:
            continue
        
        # Filter data to backtest period
        df_filtered = df[(df.index >= backtest_start) & (df.index <= backtest_end)].copy()
        
        if len(df_filtered) < 50:  # Need minimum data
            continue
        
        # Calculate blended volatility forecast
        df_filtered['blended_vol'] = calculate_blended_volatility(
            df_filtered['returns'], short_span=short_span, long_years=long_years
        )
        
        # Calculate position sizes
        positions = []
        for i in range(len(df_filtered)):
            if i == 0:
                positions.append(0)  # No position on first day
            else:
                prev_price = df_filtered['Last'].iloc[i-1]
                prev_vol = df_filtered['blended_vol'].iloc[i-1]
                
                if np.isnan(prev_vol) or prev_vol <= 0:
                    position = 0
                else:
                    position = calculate_portfolio_position_size(
                        symbol, capital, weights[symbol], idm, 
                        prev_price, prev_vol, multiplier, risk_target
                    )
                positions.append(position)
        
        df_filtered['position'] = positions
        df_filtered['position_lag'] = df_filtered['position'].shift(1)
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
    
    print(f"\nCombining portfolio...")
    print(f"Successfully processed {len(processed_data)} instruments")
    
    # Create portfolio DataFrame with full date range
    portfolio_df = pd.DataFrame(index=full_date_range)
    portfolio_df['total_pnl'] = 0.0
    portfolio_df['num_active_instruments'] = 0
    
    # Aggregate P&L across all instruments for each day
    for symbol, df in processed_data.items():
        # Reindex to match portfolio dates, forward fill for missing business days
        symbol_data = df.reindex(full_date_range, method='ffill')
        
        # Only include P&L where we actually have data (not forward filled)
        actual_dates = df.index.intersection(full_date_range)
        
        # Initialize columns if they don't exist
        portfolio_df[f'{symbol}_position'] = 0.0
        portfolio_df[f'{symbol}_pnl'] = 0.0
        
        # Add P&L only for dates where we have actual data
        for date in actual_dates:
            if date in symbol_data.index and not pd.isna(symbol_data.loc[date, 'instrument_pnl']):
                portfolio_df.loc[date, 'total_pnl'] += symbol_data.loc[date, 'instrument_pnl']
                portfolio_df.loc[date, f'{symbol}_pnl'] = symbol_data.loc[date, 'instrument_pnl']
                portfolio_df.loc[date, f'{symbol}_position'] = symbol_data.loc[date, 'position_lag']
                
                if abs(symbol_data.loc[date, 'position_lag']) > 0.01:
                    portfolio_df.loc[date, 'num_active_instruments'] += 1
    
    # Calculate portfolio returns
    portfolio_df['strategy_returns'] = portfolio_df['total_pnl'] / capital
    
    # Remove rows with no activity (weekends, holidays)
    portfolio_df = portfolio_df[portfolio_df.index.weekday < 5]  # Business days only
    portfolio_df = portfolio_df.dropna(subset=['strategy_returns'])
    
    print(f"Final portfolio data: {len(portfolio_df)} observations")
    print(f"Average active instruments: {portfolio_df['num_active_instruments'].mean():.1f}")
    
    # Calculate performance metrics
    account_curve = build_account_curve(portfolio_df['strategy_returns'], capital)
    performance = calculate_comprehensive_performance(account_curve, portfolio_df['strategy_returns'])
    
    # Add portfolio-specific metrics
    performance['num_instruments'] = len(processed_data)
    performance['idm'] = idm
    performance['avg_active_instruments'] = portfolio_df['num_active_instruments'].mean()
    performance['weight_method'] = weight_method
    performance['backtest_start'] = backtest_start
    performance['backtest_end'] = backtest_end
    
    # Calculate per-instrument statistics
    instrument_stats = {}
    for symbol in processed_data.keys():
        pnl_col = f'{symbol}_pnl'
        pos_col = f'{symbol}_position'
        
        if pnl_col in portfolio_df.columns:
            # Get only non-zero P&L periods for this instrument
            inst_pnl = portfolio_df[pnl_col][portfolio_df[pnl_col] != 0]
            
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
                    'total_pnl': inst_pnl.sum()
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
            'weight_method': weight_method,
            'backtest_start': backtest_start,
            'backtest_end': backtest_end
        }
    }

def plot_strategy4_equity_curve(results, save_path='results/strategy4_equity_curve.png'):
    """
    Plot Strategy 4 equity curve and save to file.
    
    Parameters:
        results (dict): Results from backtest_multi_instrument_strategy.
        save_path (str): Path to save the plot.
    """
    try:
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Get portfolio data
        portfolio_df = results['portfolio_data']
        config = results['config']
        performance = results['performance']
        
        # Build equity curve
        equity_curve = build_account_curve(portfolio_df['strategy_returns'], config['capital'])
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(equity_curve.index, equity_curve.values, 'b-', linewidth=1.5, label='Strategy 4: Multi-Instrument Portfolio')
        plt.title('Strategy 4: Multi-Instrument Portfolio Equity Curve', fontsize=14, fontweight='bold')
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format y-axis for millions
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        # Plot drawdown
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
        
        # Format x-axis dates for both subplots
        for ax in plt.gcf().get_axes():
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add performance metrics as text
        textstr = f'''Performance Summary:
Total Return: {performance['total_return']:.1%}
Annualized Return: {performance['annualized_return']:.1%}
Volatility: {performance['annualized_volatility']:.1%}
Sharpe Ratio: {performance['sharpe_ratio']:.3f}
Max Drawdown: {performance['max_drawdown_pct']:.1f}%
Instruments: {performance['num_instruments']}
Period: {config['backtest_start'].strftime('%Y-%m-%d')} to {config['backtest_end'].strftime('%Y-%m-%d')}'''
        
        plt.figtext(0.02, 0.02, textstr, fontsize=9, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)  # Make room for performance text
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Strategy 4 equity curve saved to: {save_path}")
        
    except Exception as e:
        print(f"Error plotting equity curve: {e}")
        import traceback
        traceback.print_exc()

def analyze_portfolio_results(results):
    """
    Analyze and display comprehensive portfolio results.
    
    Parameters:
        results (dict): Results from backtest_multi_instrument_strategy.
    """
    performance = results['performance']
    instrument_stats = results['instrument_stats']
    config = results['config']
    
    print("\n" + "=" * 60)
    print("PORTFOLIO PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Overall performance
    print(f"\n--- Overall Portfolio Performance ---")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Annualized Return: {performance['annualized_return']:.2%}")
    print(f"Annualized Volatility: {performance['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {performance['max_drawdown_pct']:.1f}%")
    print(f"Skewness: {performance['skewness']:.3f}")
    if 'kurtosis' in performance:
        print(f"Kurtosis: {performance['kurtosis']:.3f}")
    
    # Portfolio characteristics
    print(f"\n--- Portfolio Characteristics ---")
    print(f"Number of Instruments: {performance['num_instruments']}")
    print(f"IDM: {performance['idm']:.2f}")
    print(f"Average Active Instruments: {performance['avg_active_instruments']:.1f}")
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
    
    print(f"{'Symbol':<8} {'Weight':<8} {'Return':<10} {'Sharpe':<8} {'Vol':<8} {'MaxDD':<8} {'TotalPnL':<12} {'Days':<6}")
    print("-" * 85)
    
    for symbol, stats in sorted_instruments[:10]:
        print(f"{symbol:<8} {stats['weight']:<8.3f} {stats['total_return']:<10.2%} "
              f"{stats['sharpe_ratio']:<8.3f} {stats['volatility']:<8.2%} "
              f"{stats['max_drawdown']:<8.1f}% ${stats['total_pnl']:<11,.0f} {stats['active_days']:<6}")
    
    # Worst performing instruments
    print(f"\n--- Bottom 5 Performing Instruments (by Total P&L) ---")
    print(f"{'Symbol':<8} {'Weight':<8} {'Return':<10} {'Sharpe':<8} {'Vol':<8} {'MaxDD':<8} {'TotalPnL':<12} {'Days':<6}")
    print("-" * 85)
    
    for symbol, stats in sorted_instruments[-5:]:
        print(f"{symbol:<8} {stats['weight']:<8.3f} {stats['total_return']:<10.2%} "
              f"{stats['sharpe_ratio']:<8.3f} {stats['volatility']:<8.2%} "
              f"{stats['max_drawdown']:<8.1f}% ${stats['total_pnl']:<11,.0f} {stats['active_days']:<6}")

#####   UNIT TESTS   #####

def test_position_size_calculation():
    """Test position size calculation function."""
    print("\n=== Testing Position Size Calculation ===")
    
    # Test normal case
    position = calculate_portfolio_position_size(
        symbol='MES', capital=50000000, weight=0.02, idm=2.5, 
        price=4500, volatility=0.16, multiplier=5, risk_target=0.2
    )
    expected = (50000000 * 2.5 * 0.02 * 0.2) / (5 * 4500 * 1.0 * 0.16)
    print(f"Normal case: position={position:.2f}, expected={expected:.2f}")
    assert abs(position - expected) < 0.01, f"Position calculation failed: {position} != {expected}"
    
    # Test zero volatility
    position_zero = calculate_portfolio_position_size(
        'MES', 50000000, 0.02, 2.5, 4500, 0.0, 5, 0.2
    )
    print(f"Zero volatility: position={position_zero}")
    assert position_zero == 0, "Zero volatility should return 0 position"
    
    # Test NaN volatility
    position_nan = calculate_portfolio_position_size(
        'MES', 50000000, 0.02, 2.5, 4500, np.nan, 5, 0.2
    )
    print(f"NaN volatility: position={position_nan}")
    assert position_nan == 0, "NaN volatility should return 0 position"
    
    print("✓ Position size calculation tests passed")

def test_idm_calculation():
    """Test IDM calculation function."""
    print("\n=== Testing IDM Calculation ===")
    
    test_cases = [
        (1, 1.0),
        (3, 1.5),
        (8, 2.0),
        (15, 2.5),
        (30, 3.0),
        (100, 3.5)
    ]
    
    for num_instruments, expected_idm in test_cases:
        calculated_idm = calculate_idm_from_count(num_instruments)
        print(f"Instruments: {num_instruments}, IDM: {calculated_idm} (expected: {expected_idm})")
        assert calculated_idm == expected_idm, f"IDM calculation failed for {num_instruments} instruments"
    
    print("✓ IDM calculation tests passed")

def test_instrument_weights():
    """Test instrument weighting function."""
    print("\n=== Testing Instrument Weights ===")
    
    # Create sample data
    sample_data = {
        'A': pd.DataFrame({'returns': np.random.normal(0, 0.01, 100)}),
        'B': pd.DataFrame({'returns': np.random.normal(0, 0.02, 100)}),
        'C': pd.DataFrame({'returns': np.random.normal(0, 0.015, 100)})
    }
    
    # Test equal weights
    equal_weights = calculate_instrument_weights(sample_data, 'equal')
    expected_weight = 1.0 / 3
    print(f"Equal weights: {equal_weights}")
    
    for symbol, weight in equal_weights.items():
        assert abs(weight - expected_weight) < 0.001, f"Equal weight calculation failed for {symbol}"
    
    # Test that weights sum to 1
    total_weight = sum(equal_weights.values())
    print(f"Total weight: {total_weight}")
    assert abs(total_weight - 1.0) < 0.001, "Weights should sum to 1"
    
    print("✓ Instrument weights tests passed")

def test_data_loading():
    """Test data loading function."""
    print("\n=== Testing Data Loading ===")
    
    # Test loading with actual data directory
    try:
        instrument_data = load_all_instrument_data('Data')
        print(f"Loaded {len(instrument_data)} instruments")
        
        # Check that each loaded instrument has required columns
        for symbol, df in instrument_data.items():
            required_columns = ['Last', 'returns']
            for col in required_columns:
                assert col in df.columns, f"Missing column {col} in {symbol}"
            
            # Check data quality
            assert len(df) > 0, f"Empty data for {symbol}"
            assert not df['Last'].isna().all(), f"All NaN prices for {symbol}"
            
        print("✓ Data loading tests passed")
        return instrument_data
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return {}

def run_unit_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)
    
    try:
        test_position_size_calculation()
        test_idm_calculation()
        test_instrument_weights()
        instrument_data = test_data_loading()
        
        print("\n" + "=" * 60)
        print("ALL UNIT TESTS PASSED ✓")
        print("=" * 60)
        
        return True, instrument_data
        
    except Exception as e:
        print(f"\n✗ Unit test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def compare_weighting_methods():
    """
    Compare different weighting methods.
    """
    print("\n" + "=" * 80)
    print("COMPARING WEIGHTING METHODS")
    print("=" * 80)
    
    methods = ['equal', 'handcrafted']
    results_by_method = {}
    
    for method in methods:
        print(f"\n{'='*20} TESTING {method.upper()} WEIGHTING {'='*20}")
        
        try:
            results = backtest_multi_instrument_strategy(
                data_dir='Data',
                capital=50000000,
                risk_target=0.2,
                weight_method=method
            )
            results_by_method[method] = results
            
            # Brief summary
            perf = results['performance']
            print(f"\n{method.upper()} SUMMARY:")
            print(f"  Total Return: {perf['total_return']:.2%}")
            print(f"  Annualized Return: {perf['annualized_return']:.2%}")
            print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {perf['max_drawdown_pct']:.1f}%")
            
        except Exception as e:
            print(f"Error with {method} method: {e}")
            results_by_method[method] = None
    
    # Detailed comparison
    if len(results_by_method) >= 2:
        print(f"\n" + "=" * 80)
        print("WEIGHTING METHOD COMPARISON")
        print("=" * 80)
        
        print(f"{'Metric':<25} {'Equal':<15} {'Handcrafted':<15} {'Difference':<15}")
        print("-" * 80)
        
        equal_result = results_by_method.get('equal', {})
        handcrafted_result = results_by_method.get('handcrafted', {})
        
        equal_perf = equal_result.get('performance', {}) if equal_result else {}
        handcrafted_perf = handcrafted_result.get('performance', {}) if handcrafted_result else {}
        
        if equal_perf and handcrafted_perf:
            metrics = [
                ('Total Return', 'total_return', '%'),
                ('Annualized Return', 'annualized_return', '%'),
                ('Volatility', 'annualized_volatility', '%'),
                ('Sharpe Ratio', 'sharpe_ratio', ''),
                ('Max Drawdown', 'max_drawdown_pct', '%')
            ]
            
            for name, key, unit in metrics:
                equal_val = equal_perf.get(key, 0)
                handcrafted_val = handcrafted_perf.get(key, 0)
                diff = handcrafted_val - equal_val
                
                if unit == '%':
                    print(f"{name:<25} {equal_val:<15.2%} {handcrafted_val:<15.2%} {diff:<15.2%}")
                else:
                    print(f"{name:<25} {equal_val:<15.3f} {handcrafted_val:<15.3f} {diff:<15.3f}")
    
    return results_by_method

def main():
    """
    Test Strategy 4 implementation with unit tests.
    """
    # First run unit tests
    tests_passed, instrument_data = run_unit_tests()
    
    if not tests_passed:
        print("Unit tests failed. Stopping execution.")
        return
    
    try:
        # Compare weighting methods
        results_by_method = compare_weighting_methods()
        
        # Use handcrafted results for detailed analysis
        if 'handcrafted' in results_by_method and results_by_method['handcrafted']:
            results = results_by_method['handcrafted']
            
            print(f"\n" + "=" * 60)
            print("DETAILED HANDCRAFTED STRATEGY ANALYSIS")
            print("=" * 60)
            
            # Analyze results
            analyze_portfolio_results(results)
            
            # Plot equity curve
            plot_strategy4_equity_curve(results)
        
        # Compare with single instrument strategy (Chapter 3)
        print(f"\n" + "=" * 60)
        print("COMPARISON WITH SINGLE INSTRUMENT STRATEGY")
        print("=" * 60)
        
        try:
            # Run MES-only strategy for comparison using same date range
            backtest_start = results['config']['backtest_start']
            backtest_end = results['config']['backtest_end']
            
            # Load MES data for the same period
            mes_df = pd.read_csv('Data/mes_daily_data.csv', parse_dates=['Time'])
            mes_df.set_index('Time', inplace=True)
            mes_df = mes_df[(mes_df.index >= backtest_start) & (mes_df.index <= backtest_end)]
            
            # Save temporary file for comparison
            mes_df.to_csv('Data/mes_temp_comparison.csv')
            
            mes_results = backtest_variable_risk_strategy(
                'Data/mes_temp_comparison.csv', 
                capital=50000000, 
                risk_target=0.2
            )
            
            # Clean up temp file
            os.remove('Data/mes_temp_comparison.csv')
            
            mes_perf = mes_results['performance']
            multi_perf = results['performance']
            
            print(f"\nPerformance Comparison (Same Time Period):")
            print(f"{'Metric':<25} {'MES Only':<12} {'Multi-Inst':<12} {'Difference':<12}")
            print("-" * 65)
            print(f"{'Total Return':<25} {mes_perf['total_return']:<12.2%} {multi_perf['total_return']:<12.2%} "
                  f"{multi_perf['total_return'] - mes_perf['total_return']:<12.2%}")
            print(f"{'Annualized Return':<25} {mes_perf['annualized_return']:<12.2%} {multi_perf['annualized_return']:<12.2%} "
                  f"{multi_perf['annualized_return'] - mes_perf['annualized_return']:<12.2%}")
            print(f"{'Volatility':<25} {mes_perf['annualized_volatility']:<12.2%} {multi_perf['annualized_volatility']:<12.2%} "
                  f"{multi_perf['annualized_volatility'] - mes_perf['annualized_volatility']:<12.2%}")
            print(f"{'Sharpe Ratio':<25} {mes_perf['sharpe_ratio']:<12.3f} {multi_perf['sharpe_ratio']:<12.3f} "
                  f"{multi_perf['sharpe_ratio'] - mes_perf['sharpe_ratio']:<12.3f}")
            print(f"{'Max Drawdown':<25} {mes_perf['max_drawdown_pct']:<12.1f}% {multi_perf['max_drawdown_pct']:<12.1f}% "
                  f"{multi_perf['max_drawdown_pct'] - mes_perf['max_drawdown_pct']:<12.1f}%")
            
        except Exception as e:
            print(f"Could not run comparison with MES: {e}")
        
        print(f"\nStrategy 4 backtest completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error in Strategy 4 backtest: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
