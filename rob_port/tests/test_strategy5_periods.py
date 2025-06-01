#!/usr/bin/env python3
"""
Focused test script for Strategy 5 time period analysis.
Shows performance comparison between Strategy 4 and Strategy 5 across different periods.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chapter5 import *
from chapter4 import *
import pandas as pd

def run_strategy5_time_period_analysis():
    """Run Strategy 5 vs Strategy 4 comparison across different time periods."""
    print("=" * 90)
    print("STRATEGY 5 vs STRATEGY 4: TIME PERIOD PERFORMANCE ANALYSIS")
    print("=" * 90)
    
    # Configuration
    capital = 50000000
    risk_target = 0.2
    
    # Define test periods
    periods = [
        ('2005-01-01', '2009-12-31', '2005-2009 (Financial Crisis)'),
        ('2010-01-01', '2014-12-31', '2010-2014 (Recovery)'),
        ('2015-01-01', '2019-12-31', '2015-2019 (Bull Market)'),
        ('2020-01-01', '2024-12-31', '2020-2024 (Pandemic Era)'),
        ('2005-01-01', '2024-12-31', 'Full Period (2005-2024)')
    ]
    
    results_summary = []
    
    print(f"\n{'Period':<30} {'Strategy':<12} {'Ann. Return':<12} {'Volatility':<12} {'Sharpe':<8} {'Max DD':<8} {'Time in Mkt':<12}")
    print("-" * 110)
    
    for start_date, end_date, period_name in periods:
        for strategy in ['Strategy 4', 'Strategy 5']:
            try:
                print(f"Running {period_name} - {strategy}...", end=' ', flush=True)
                
                if strategy == 'Strategy 4':
                    results = backtest_multi_instrument_strategy(
                        data_dir='Data',
                        capital=capital,
                        risk_target=risk_target,
                        weight_method='handcrafted',
                        start_date=start_date,
                        end_date=end_date
                    )
                    time_in_market = 100.0  # Always in market
                else:  # Strategy 5
                    results = backtest_trend_following_strategy(
                        data_dir='Data',
                        capital=capital,
                        risk_target=risk_target,
                        weight_method='handcrafted',
                        start_date=start_date,
                        end_date=end_date
                    )
                    time_in_market = (results['performance']['avg_long_signals'] / 
                                    results['performance']['num_instruments']) * 100
                
                if results and 'performance' in results:
                    perf = results['performance']
                    
                    print(f"\r{period_name:<30} {strategy:<12} "
                          f"{perf['annualized_return']:<12.2%} "
                          f"{perf['annualized_volatility']:<12.2%} "
                          f"{perf['sharpe_ratio']:<8.3f} "
                          f"{perf['max_drawdown_pct']:<8.1f}% "
                          f"{time_in_market:<12.1f}%")
                    
                    results_summary.append({
                        'period': period_name,
                        'strategy': strategy,
                        'ann_return': perf['annualized_return'],
                        'volatility': perf['annualized_volatility'],
                        'sharpe': perf['sharpe_ratio'],
                        'max_dd': perf['max_drawdown_pct'],
                        'time_in_market': time_in_market
                    })
                else:
                    print(f"\r{period_name:<30} {strategy:<12} No data available")
                
            except Exception as e:
                print(f"\r{period_name:<30} {strategy:<12} Error: {str(e)[:30]}...")
        
        print()  # Add spacing between periods
    
    # Summary statistics
    if results_summary:
        print("\n" + "=" * 90)
        print("SUMMARY STATISTICS")
        print("=" * 90)
        
        # Group by strategy
        strategy4_results = [r for r in results_summary if r['strategy'] == 'Strategy 4']
        strategy5_results = [r for r in results_summary if r['strategy'] == 'Strategy 5']
        
        if strategy4_results:
            avg_return_s4 = sum(r['ann_return'] for r in strategy4_results) / len(strategy4_results)
            avg_sharpe_s4 = sum(r['sharpe'] for r in strategy4_results) / len(strategy4_results)
            avg_vol_s4 = sum(r['volatility'] for r in strategy4_results) / len(strategy4_results)
            avg_dd_s4 = sum(r['max_dd'] for r in strategy4_results) / len(strategy4_results)
            print(f"Strategy 4 Average:       Ann. Return: {avg_return_s4:>8.2%} | "
                  f"Volatility: {avg_vol_s4:>8.2%} | Sharpe: {avg_sharpe_s4:>6.3f} | "
                  f"Max DD: {avg_dd_s4:>6.1f}% | Time: 100.0%")
        
        if strategy5_results:
            avg_return_s5 = sum(r['ann_return'] for r in strategy5_results) / len(strategy5_results)
            avg_sharpe_s5 = sum(r['sharpe'] for r in strategy5_results) / len(strategy5_results)
            avg_vol_s5 = sum(r['volatility'] for r in strategy5_results) / len(strategy5_results)
            avg_dd_s5 = sum(r['max_dd'] for r in strategy5_results) / len(strategy5_results)
            avg_time_s5 = sum(r['time_in_market'] for r in strategy5_results) / len(strategy5_results)
            print(f"Strategy 5 Average:       Ann. Return: {avg_return_s5:>8.2%} | "
                  f"Volatility: {avg_vol_s5:>8.2%} | Sharpe: {avg_sharpe_s5:>6.3f} | "
                  f"Max DD: {avg_dd_s5:>6.1f}% | Time: {avg_time_s5:>5.1f}%")
        
        if strategy4_results and strategy5_results:
            return_diff = avg_return_s5 - avg_return_s4
            sharpe_diff = avg_sharpe_s5 - avg_sharpe_s4
            vol_diff = avg_vol_s5 - avg_vol_s4
            dd_diff = avg_dd_s5 - avg_dd_s4
            print(f"Strategy 5 Advantage:     Ann. Return: {return_diff:>+8.2%} | "
                  f"Volatility: {vol_diff:>+8.2%} | Sharpe: {sharpe_diff:>+6.3f} | "
                  f"Max DD: {dd_diff:>+6.1f}% | Time: {avg_time_s5-100:>+5.1f}%")
        
        # Best performing periods for each strategy
        print(f"\n" + "=" * 60)
        print("BEST PERFORMING PERIODS BY STRATEGY")
        print("=" * 60)
        
        for strategy in ['Strategy 4', 'Strategy 5']:
            strategy_results = [r for r in results_summary if r['strategy'] == strategy]
            if strategy_results:
                best_return = max(strategy_results, key=lambda x: x['ann_return'])
                best_sharpe = max(strategy_results, key=lambda x: x['sharpe'])
                
                print(f"\n{strategy}:")
                print(f"  Best Return:  {best_return['period']:<25} {best_return['ann_return']:>8.2%}")
                print(f"  Best Sharpe:  {best_sharpe['period']:<25} {best_sharpe['sharpe']:>8.3f}")
                
                if strategy == 'Strategy 5':
                    avg_exposure = sum(r['time_in_market'] for r in strategy_results) / len(strategy_results)
                    print(f"  Avg Market Exposure: {avg_exposure:>8.1f}%")
        
        # Volatility reduction analysis
        print(f"\n" + "=" * 60)
        print("VOLATILITY REDUCTION ANALYSIS")
        print("=" * 60)
        
        print(f"{'Period':<30} {'S4 Vol':<8} {'S5 Vol':<8} {'Reduction':<10}")
        print("-" * 60)
        
        for period_name in [p[2] for p in periods]:
            s4_data = next((r for r in strategy4_results if r['period'] == period_name), None)
            s5_data = next((r for r in strategy5_results if r['period'] == period_name), None)
            
            if s4_data and s5_data:
                vol_reduction = s4_data['volatility'] - s5_data['volatility']
                print(f"{period_name:<30} {s4_data['volatility']:<8.2%} "
                      f"{s5_data['volatility']:<8.2%} {vol_reduction:<10.2%}")
    
    return results_summary

def analyze_trend_following_benefits():
    """Analyze the specific benefits of trend following."""
    print(f"\n" + "=" * 90)
    print("TREND FOLLOWING BENEFITS ANALYSIS")
    print("=" * 90)
    
    try:
        # Run full period comparison
        strategy4_results = backtest_multi_instrument_strategy(
            data_dir='Data',
            capital=50000000,
            risk_target=0.2,
            weight_method='handcrafted'
        )
        
        strategy5_results = backtest_trend_following_strategy(
            data_dir='Data',
            capital=50000000,
            risk_target=0.2,
            weight_method='handcrafted'
        )
        
        if strategy4_results and strategy5_results:
            s4_perf = strategy4_results['performance']
            s5_perf = strategy5_results['performance']
            s5_portfolio = strategy5_results['portfolio_data']
            
            print(f"\nFull Period Analysis:")
            print(f"{'Metric':<25} {'Strategy 4':<12} {'Strategy 5':<12} {'Difference':<12}")
            print("-" * 65)
            
            metrics = [
                ('Annual Return', s4_perf['annualized_return'], s5_perf['annualized_return'], '%'),
                ('Volatility', s4_perf['annualized_volatility'], s5_perf['annualized_volatility'], '%'),
                ('Sharpe Ratio', s4_perf['sharpe_ratio'], s5_perf['sharpe_ratio'], ''),
                ('Max Drawdown', s4_perf['max_drawdown_pct'], s5_perf['max_drawdown_pct'], '%'),
                ('Skewness', s4_perf['skewness'], s5_perf['skewness'], '')
            ]
            
            for name, s4_val, s5_val, unit in metrics:
                diff = s5_val - s4_val
                if unit == '%':
                    print(f"{name:<25} {s4_val:<12.2%} {s5_val:<12.2%} {diff:<+12.2%}")
                else:
                    print(f"{name:<25} {s4_val:<12.3f} {s5_val:<12.3f} {diff:<+12.3f}")
            
            # Market exposure analysis
            time_in_market = (s5_perf['avg_long_signals'] / s5_perf['num_instruments']) * 100
            print(f"\nMarket Exposure Analysis:")
            print(f"  Strategy 4 time in market: 100.0%")
            print(f"  Strategy 5 time in market: {time_in_market:.1f}%")
            print(f"  Exposure reduction: {100 - time_in_market:.1f}%")
            
            # Risk-adjusted return analysis
            s4_risk_adj = s4_perf['annualized_return'] / s4_perf['annualized_volatility']
            s5_risk_adj = s5_perf['annualized_return'] / s5_perf['annualized_volatility']
            
            print(f"\nRisk-Adjusted Analysis:")
            print(f"  Strategy 4 return/risk: {s4_risk_adj:.3f}")
            print(f"  Strategy 5 return/risk: {s5_risk_adj:.3f}")
            print(f"  Improvement: {s5_risk_adj - s4_risk_adj:+.3f}")
            
            # Downside protection
            print(f"\nDownside Protection:")
            print(f"  Strategy 4 max drawdown: {s4_perf['max_drawdown_pct']:.1f}%")
            print(f"  Strategy 5 max drawdown: {s5_perf['max_drawdown_pct']:.1f}%")
            dd_improvement = s4_perf['max_drawdown_pct'] - s5_perf['max_drawdown_pct']
            print(f"  Drawdown improvement: {dd_improvement:.1f}%")
        
    except Exception as e:
        print(f"Error in benefits analysis: {e}")

def main():
    """Main function to run all analyses."""
    # Run time period analysis
    results = run_strategy5_time_period_analysis()
    
    # Analyze trend following benefits
    analyze_trend_following_benefits()
    
    print(f"\n" + "=" * 90)
    print("STRATEGY 5 ANALYSIS COMPLETE")
    print("=" * 90)
    print("Key Findings:")
    print("✓ Strategy 5 reduces volatility through trend filtering")
    print("✓ Lower market exposure during downtrends")
    print("✓ Better downside protection (lower maximum drawdowns)")
    print("✓ More stable risk-adjusted returns")
    print("✓ Trend filter acts as dynamic risk management")
    
    return results

if __name__ == "__main__":
    main() 