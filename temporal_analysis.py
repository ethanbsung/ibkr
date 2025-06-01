import pandas as pd
import numpy as np
import sys
import os
sys.path.append('rob_port')
from chapter9 import *

def analyze_temporal_performance():
    """Analyze performance breakdown by time periods."""
    print('=== TEMPORAL PERFORMANCE ANALYSIS ===')
    
    # Load the latest results
    try:
        cached_results = get_cached_strategy_results()
        
        if 'strategy7' not in cached_results:
            print("Need Strategy 7 results...")
            return
            
        # Try to load Strategy 8 directly
        strategy8_config = {
            'capital': 50000000,
            'risk_target': 0.2,
            'weight_method': 'handcrafted'
        }
        
        strategy8_results = load_strategy_results('strategy8', strategy8_config)
        if strategy8_results is None:
            print("Need Strategy 8 results...")
            return
            
        s7_data = cached_results['strategy7']['portfolio_data']
        s8_data = strategy8_results['portfolio_data']
        
        # Analyze by time periods
        periods = [
            ('2000-2005', '2000-01-01', '2005-12-31'),
            ('2006-2010', '2006-01-01', '2010-12-31'), 
            ('2011-2015', '2011-01-01', '2015-12-31'),
            ('2016-2020', '2016-01-01', '2020-12-31'),
            ('2021-2024', '2021-01-01', '2024-12-31')
        ]
        
        print('\nStrategy 7 (Slow Trends) Performance by Period:')
        print('Period        Ann.Return  Sharpe   MaxDD   Years')
        print('-' * 50)
        
        for period_name, start, end in periods:
            period_data = s7_data[(s7_data.index >= start) & (s7_data.index <= end)]
            if len(period_data) > 20:
                returns = period_data['strategy_returns'].dropna()
                if len(returns) > 0:
                    ann_ret = returns.mean() * 256
                    sharpe = returns.mean() / returns.std() * np.sqrt(256) if returns.std() > 0 else 0
                    cum_ret = (1 + returns).cumprod()
                    max_dd = ((cum_ret / cum_ret.cummax()) - 1).min() * 100
                    years = len(returns) / 256
                    print(f'{period_name:<12} {ann_ret:>8.1%} {sharpe:>7.2f} {max_dd:>7.1f}% {years:>6.1f}')
        
        print('\nStrategy 8 (Fast + Buffering) Performance by Period:')
        print('Period        Ann.Return  Sharpe   MaxDD   Years')
        print('-' * 50)
        
        for period_name, start, end in periods:
            period_data = s8_data[(s8_data.index >= start) & (s8_data.index <= end)]
            if len(period_data) > 20:
                returns = period_data['strategy_returns'].dropna()
                if len(returns) > 0:
                    ann_ret = returns.mean() * 256
                    sharpe = returns.mean() / returns.std() * np.sqrt(256) if returns.std() > 0 else 0
                    cum_ret = (1 + returns).cumprod()
                    max_dd = ((cum_ret / cum_ret.cummax()) - 1).min() * 100
                    years = len(returns) / 256
                    print(f'{period_name:<12} {ann_ret:>8.1%} {sharpe:>7.2f} {max_dd:>7.1f}% {years:>6.1f}')
        
        # Look at recent performance specifically
        recent_data_s7 = s7_data[s7_data.index >= '2020-01-01']
        recent_data_s8 = s8_data[s8_data.index >= '2020-01-01']
        
        print('\n=== RECENT PERFORMANCE (2020+) ===')
        if len(recent_data_s7) > 0:
            recent_ret_s7 = recent_data_s7['strategy_returns'].dropna()
            recent_ret_s8 = recent_data_s8['strategy_returns'].dropna()
            
            print(f'Strategy 7 Recent: {recent_ret_s7.mean() * 256:.1%} annual, {recent_ret_s7.mean() / recent_ret_s7.std() * np.sqrt(256):.2f} Sharpe')
            print(f'Strategy 8 Recent: {recent_ret_s8.mean() * 256:.1%} annual, {recent_ret_s8.mean() / recent_ret_s8.std() * np.sqrt(256):.2f} Sharpe')
            
        # Analyze market regime changes
        print('\n=== MARKET ENVIRONMENT ANALYSIS ===')
        
        # Look at average volatility by period
        print('\nAverage Market Volatility by Period:')
        for period_name, start, end in periods:
            period_data = s7_data[(s7_data.index >= start) & (s7_data.index <= end)]
            if len(period_data) > 20:
                # Calculate realized volatility of returns
                returns = period_data['strategy_returns'].dropna()
                if len(returns) > 0:
                    market_vol = returns.std() * np.sqrt(256)
                    print(f'{period_name}: {market_vol:.1%} annualized volatility')
        
        # Check trend persistence - how often do trends last?
        print('\n=== TREND PERSISTENCE ANALYSIS ===')
        print('Note: Declining trend persistence could explain strategy decay')
        
        # Calculate rolling correlations to see if trends are becoming less persistent
        for period_name, start, end in periods:
            period_data = s7_data[(s7_data.index >= start) & (s7_data.index <= end)]
            if len(period_data) > 50:
                returns = period_data['strategy_returns'].dropna()
                if len(returns) > 30:
                    # Look at autocorrelation of returns (trend persistence)
                    autocorr_1day = returns.autocorr(lag=1)
                    autocorr_5day = returns.autocorr(lag=5)
                    print(f'{period_name}: 1-day autocorr: {autocorr_1day:.3f}, 5-day autocorr: {autocorr_5day:.3f}')
    
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_temporal_performance() 