#!/usr/bin/env python3
"""
Comprehensive Strategy Test with Bug Fixes

This script tests all strategies (6-8) with the P&L calculation fixes applied.
"""

import sys
sys.path.append('rob_port')
from chapter8 import *
from chapter7 import *
from chapter6 import *
from chapter5 import *
from chapter4 import *

def test_all_strategies():
    """
    Test all strategies with a common configuration to compare performance.
    """
    print("=" * 80)
    print("COMPREHENSIVE STRATEGY TEST WITH BUG FIXES")
    print("=" * 80)
    
    # Common configuration - use full available data period
    config = {
        'data_dir': 'Data',
        'capital': 50000000,
        'risk_target': 0.2,
        'short_span': 32,
        'long_years': 10,
        'min_vol_floor': 0.05,
        'weight_method': 'handcrafted',
        'common_hypothetical_SR': 0.3,
        'annual_turnover_T': 7.0,
        # Remove date constraints to use full available data period
        # 'start_date': '2023-01-01',
        # 'end_date': '2023-12-31'
    }
    
    print(f"Test Configuration:")
    print(f"  Period: Full available data (instruments phase in as available)")
    print(f"  Capital: ${config['capital']:,}")
    print(f"  Risk Target: {config['risk_target']:.1%}")
    print(f"  Weight Method: {config['weight_method']}")
    
    results = {}
    
    try:
        # Strategy 4 (baseline - no trend filter)
        print(f"\n{'-'*60}")
        print("RUNNING STRATEGY 4 (No Trend Filter)")
        print(f"{'-'*60}")
        
        results['strategy4'] = backtest_multi_instrument_strategy(**config)
        
        s4_perf = results['strategy4']['performance']
        print(f"Strategy 4 Results:")
        print(f"  Annualized Return: {s4_perf['annualized_return']:.2%}")
        print(f"  Sharpe Ratio: {s4_perf['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {s4_perf['max_drawdown_pct']:.1f}%")
        
    except Exception as e:
        print(f"Strategy 4 failed: {e}")
        results['strategy4'] = None
    
    try:
        # Strategy 5 (trend following, long only)
        print(f"\n{'-'*60}")
        print("RUNNING STRATEGY 5 (Long Only Trend Following)")
        print(f"{'-'*60}")
        
        s5_config = config.copy()
        s5_config.update({
            'trend_fast_span': 64,
            'trend_slow_span': 256
        })
        
        results['strategy5'] = backtest_trend_following_strategy(**s5_config)
        
        s5_perf = results['strategy5']['performance']
        print(f"Strategy 5 Results:")
        print(f"  Annualized Return: {s5_perf['annualized_return']:.2%}")
        print(f"  Sharpe Ratio: {s5_perf['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {s5_perf['max_drawdown_pct']:.1f}%")
        print(f"  Avg Long Signals: {s5_perf['avg_long_signals']:.1f}")
        
    except Exception as e:
        print(f"Strategy 5 failed: {e}")
        results['strategy5'] = None
    
    try:
        # Strategy 6 (long/short trend following)
        print(f"\n{'-'*60}")
        print("RUNNING STRATEGY 6 (Long/Short Trend Following)")
        print(f"{'-'*60}")
        
        s6_config = config.copy()
        s6_config.update({
            'trend_fast_span': 64,
            'trend_slow_span': 256
        })
        
        results['strategy6'] = backtest_long_short_trend_strategy(**s6_config)
        
        s6_perf = results['strategy6']['performance']
        print(f"Strategy 6 Results:")
        print(f"  Annualized Return: {s6_perf['annualized_return']:.2%}")
        print(f"  Sharpe Ratio: {s6_perf['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {s6_perf['max_drawdown_pct']:.1f}%")
        print(f"  Avg Long Signals: {s6_perf['avg_long_signals']:.1f}")
        print(f"  Avg Short Signals: {s6_perf['avg_short_signals']:.1f}")
        
    except Exception as e:
        print(f"Strategy 6 failed: {e}")
        results['strategy6'] = None
    
    try:
        # Strategy 7 (slow forecasts)
        print(f"\n{'-'*60}")
        print("RUNNING STRATEGY 7 (Slow Forecasts)")
        print(f"{'-'*60}")
        
        s7_config = config.copy()
        s7_config.update({
            'trend_fast_span': 64,
            'trend_slow_span': 256,
            'forecast_scalar': 1.9,
            'forecast_cap': 20.0
        })
        
        results['strategy7'] = backtest_forecast_trend_strategy(**s7_config)
        
        s7_perf = results['strategy7']['performance']
        print(f"Strategy 7 Results:")
        print(f"  Annualized Return: {s7_perf['annualized_return']:.2%}")
        print(f"  Sharpe Ratio: {s7_perf['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {s7_perf['max_drawdown_pct']:.1f}%")
        print(f"  Avg Forecast: {s7_perf['avg_forecast']:.2f}")
        print(f"  Avg Abs Forecast: {s7_perf['avg_abs_forecast']:.2f}")
        
    except Exception as e:
        print(f"Strategy 7 failed: {e}")
        results['strategy7'] = None
    
    try:
        # Strategy 8 (fast forecasts with buffering) - NOW WITH BUG FIX
        print(f"\n{'-'*60}")
        print("RUNNING STRATEGY 8 (Fast Forecasts + Buffering) - WITH BUG FIX")
        print(f"{'-'*60}")
        
        s8_config = config.copy()
        s8_config.update({
            'trend_fast_span': 16,
            'trend_slow_span': 64,
            'forecast_scalar': 4.1,
            'forecast_cap': 20.0,
            'buffer_fraction': 0.1
        })
        
        results['strategy8'] = backtest_fast_trend_strategy_with_buffering(**s8_config)
        
        s8_perf = results['strategy8']['performance']
        print(f"Strategy 8 Results:")
        print(f"  Annualized Return: {s8_perf['annualized_return']:.2%}")
        print(f"  Sharpe Ratio: {s8_perf['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {s8_perf['max_drawdown_pct']:.1f}%")
        print(f"  Avg Forecast: {s8_perf['avg_forecast']:.2f}")
        print(f"  Avg Abs Forecast: {s8_perf['avg_abs_forecast']:.2f}")
        print(f"  Avg Daily Trades: {s8_perf['avg_daily_trades']:.1f}")
        
    except Exception as e:
        print(f"Strategy 8 failed: {e}")
        results['strategy8'] = None
    
    # Comparison table
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON TABLE")
    print(f"{'='*80}")
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) > 0:
        print(f"{'Strategy':<12} {'Ann. Return':<12} {'Sharpe':<8} {'Max DD':<8} {'Special Feature':<20}")
        print("-" * 70)
        
        for strategy, result in valid_results.items():
            perf = result['performance']
            
            if strategy == 'strategy4':
                special = "Always Long"
            elif strategy == 'strategy5':
                special = f"Long Only ({perf['avg_long_signals']:.0f} signals)"
            elif strategy == 'strategy6':
                special = f"Long/Short ({perf['avg_long_signals']:.0f}L/{perf['avg_short_signals']:.0f}S)"
            elif strategy == 'strategy7':
                special = f"Slow Fcst ({perf['avg_abs_forecast']:.1f})"
            elif strategy == 'strategy8':
                special = f"Fast+Buff ({perf['avg_daily_trades']:.1f} trades/day)"
            else:
                special = "Unknown"
            
            print(f"{strategy:<12} {perf['annualized_return']:<12.2%} "
                  f"{perf['sharpe_ratio']:<8.3f} {perf['max_drawdown_pct']:<8.1f}% {special:<20}")
        
        # Performance ranking
        print(f"\n--- Performance Ranking (by Sharpe Ratio) ---")
        sorted_by_sharpe = sorted(valid_results.items(), 
                                key=lambda x: x[1]['performance']['sharpe_ratio'], 
                                reverse=True)
        
        for i, (strategy, result) in enumerate(sorted_by_sharpe, 1):
            perf = result['performance']
            print(f"{i}. {strategy}: {perf['sharpe_ratio']:.3f} Sharpe, "
                  f"{perf['annualized_return']:.2%} return")
    
    return results

if __name__ == "__main__":
    results = test_all_strategies()
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TEST COMPLETE")
    print(f"{'='*80}") 