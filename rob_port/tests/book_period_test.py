import pandas as pd
import numpy as np
import sys
import os
sys.path.append('rob_port')
from chapter9 import *

def test_book_time_period():
    """Test all strategies using book's time period (2000-2019) only."""
    print('=' * 80)
    print('TESTING WITH BOOK\'S TIME PERIOD: 2000-2019')
    print('=' * 80)
    
    # Book's time period
    start_date = '2000-01-01'
    end_date = '2019-12-31'
    
    print(f"Testing period: {start_date} to {end_date}")
    print("This matches the apparent book period from equity curve charts.\n")
    
    # Standard config
    config = {
        'capital': 50000000,
        'risk_target': 0.2,
        'weight_method': 'handcrafted',
        'start_date': start_date,
        'end_date': end_date
    }
    
    results = {}
    
    try:
        # Strategy 4 (Long only)
        print("Running Strategy 4 (Long only) for book period...")
        results['strategy4'] = backtest_multi_instrument_strategy(
            data_dir='Data', **config
        )
        
        # Strategy 7 (Slow trend)
        print("Running Strategy 7 (Slow trend) for book period...")
        results['strategy7'] = backtest_forecast_trend_strategy(
            data_dir='Data', **config
        )
        
        # Strategy 8 (Fast trend)
        print("Running Strategy 8 (Fast trend) for book period...")
        results['strategy8'] = backtest_fast_trend_strategy_with_buffering(
            data_dir='Data', **config
        )
        
        # Strategy 9 (Multiple trend)
        print("Running Strategy 9 (Multiple trend) for book period...")
        config['forecast_combination'] = 'five_filters'
        results['strategy9'] = backtest_multiple_trend_strategy(
            data_dir='Data', **config
        )
        
        # Compare with book's Table 39 results
        print("\n" + "=" * 80)
        print("BOOK PERIOD RESULTS vs TABLE 39")
        print("=" * 80)
        
        book_performance = {
            'Strategy 4 (Long only)': {'return': 15.4, 'sharpe': 0.85},
            'Strategy 7 (Slow trend)': {'return': 21.5, 'sharpe': 0.96},
            'Strategy 8 (Fast trend)': {'return': 24.1, 'sharpe': 1.06},
            'Strategy 9 (Multiple trend)': {'return': 25.2, 'sharpe': 1.14}
        }
        
        strategy_names = {
            'strategy4': 'Strategy 4 (Long only)',
            'strategy7': 'Strategy 7 (Slow trend)',
            'strategy8': 'Strategy 8 (Fast trend)',
            'strategy9': 'Strategy 9 (Multiple trend)'
        }
        
        print(f"{'Strategy':<25} {'Book Return':<12} {'2000-19 Return':<15} {'Difference':<12} {'Book Sharpe':<12} {'2000-19 Sharpe':<15} {'Difference':<12}")
        print("-" * 120)
        
        for strategy_key, result in results.items():
            if result:
                strategy_name = strategy_names[strategy_key]
                our_perf = result['performance']
                book_perf = book_performance[strategy_name]
                
                our_return = our_perf['annualized_return'] * 100
                our_sharpe = our_perf['sharpe_ratio']
                
                book_return = book_perf['return']
                book_sharpe = book_perf['sharpe']
                
                ret_diff = our_return - book_return
                sharpe_diff = our_sharpe - book_sharpe
                
                print(f"{strategy_name:<25} {book_return:<12.1f}% {our_return:<15.1f}% {ret_diff:<12.1f}% {book_sharpe:<12.2f} {our_sharpe:<15.2f} {sharpe_diff:<12.2f}")
        
        # Analyze the gaps
        print(f"\n=== GAP ANALYSIS ===")
        
        major_gaps = []
        implementation_issues = []
        
        for strategy_key, result in results.items():
            if result:
                strategy_name = strategy_names[strategy_key]
                our_perf = result['performance']
                book_perf = book_performance[strategy_name]
                
                ret_gap = (our_perf['annualized_return'] * 100) - book_perf['return']
                sharpe_gap = our_perf['sharpe_ratio'] - book_perf['sharpe']
                
                if abs(ret_gap) > 5:  # More than 5% difference
                    major_gaps.append(f"{strategy_name}: {ret_gap:.1f}% return gap")
                
                if abs(sharpe_gap) > 0.3:  # More than 0.3 Sharpe difference
                    implementation_issues.append(f"{strategy_name}: {sharpe_gap:.2f} Sharpe gap")
        
        if major_gaps:
            print("🚨 MAJOR PERFORMANCE GAPS (even with book's time period):")
            for gap in major_gaps:
                print(f"   {gap}")
        
        if implementation_issues:
            print("\n🔴 IMPLEMENTATION ISSUES CONFIRMED:")
            for issue in implementation_issues:
                print(f"   {issue}")
            print("\n   These gaps are too large for data differences alone.")
            print("   Fundamental implementation errors are likely present.")
        
        # Time period impact analysis
        print(f"\n=== TIME PERIOD IMPACT ANALYSIS ===")
        
        # Load our full period results for comparison
        cached_results = get_cached_strategy_results()
        
        if cached_results:
            print(f"{'Strategy':<25} {'Full Period':<12} {'2000-19 Only':<12} {'Time Impact':<12}")
            print("-" * 65)
            
            for strategy_key, book_result in results.items():
                if book_result and strategy_key in cached_results:
                    full_perf = cached_results[strategy_key]['performance']
                    book_perf = book_result['performance']
                    
                    full_return = full_perf['annualized_return'] * 100
                    book_return = book_perf['annualized_return'] * 100
                    time_impact = book_return - full_return
                    
                    strategy_name = strategy_names[strategy_key].split('(')[0].strip()
                    print(f"{strategy_name:<25} {full_return:<12.1f}% {book_return:<12.1f}% {time_impact:<12.1f}%")
        
        # Final assessment
        print(f"\n=== FINAL ASSESSMENT ===")
        
        if major_gaps or implementation_issues:
            print("🔴 CONCLUSION: Implementation errors confirmed")
            print("   Even using the book's exact time period (2000-2019),")
            print("   we still have massive performance gaps.")
            print("   This rules out regime change as the primary cause.")
            
            print(f"\n   ROOT CAUSES TO INVESTIGATE:")
            print("   1. Forecast calculation methodology")
            print("   2. Position sizing implementation")
            print("   3. Portfolio construction (weights, IDM)")
            print("   4. Data preprocessing differences")
            print("   5. Trading cost assumptions")
            
        else:
            print("✅ CONCLUSION: Time period effect was the main issue")
            print("   Results are much closer to book when using 2000-2019 period.")
            print("   The 2020-2024 period significantly degraded performance.")
        
        return results
        
    except Exception as e:
        print(f"Error in book period test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_book_time_period() 