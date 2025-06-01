#!/usr/bin/env python3
"""
Focused test script for Strategy 4 time period analysis.
Shows annualized returns across different time periods with minimal verbose output.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chapter4 import *
import pandas as pd

def run_time_period_analysis():
    """Run Strategy 4 performance analysis across different time periods."""
    print("=" * 80)
    print("STRATEGY 4: TIME PERIOD PERFORMANCE ANALYSIS")
    print("=" * 80)
    
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
    
    print(f"\n{'Period':<30} {'Method':<12} {'Ann. Return':<12} {'Volatility':<12} {'Sharpe':<8} {'Max DD':<8}")
    print("-" * 90)
    
    for start_date, end_date, period_name in periods:
        for method in ['equal', 'handcrafted']:
            try:
                print(f"Running {period_name} - {method}...", end=' ', flush=True)
                
                results = backtest_multi_instrument_strategy(
                    data_dir='Data',
                    capital=capital,
                    risk_target=risk_target,
                    weight_method=method,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if results and 'performance' in results:
                    perf = results['performance']
                    
                    print(f"\r{period_name:<30} {method.capitalize():<12} "
                          f"{perf['annualized_return']:<12.2%} "
                          f"{perf['annualized_volatility']:<12.2%} "
                          f"{perf['sharpe_ratio']:<8.3f} "
                          f"{perf['max_drawdown_pct']:<8.1f}%")
                    
                    results_summary.append({
                        'period': period_name,
                        'method': method,
                        'ann_return': perf['annualized_return'],
                        'volatility': perf['annualized_volatility'],
                        'sharpe': perf['sharpe_ratio'],
                        'max_dd': perf['max_drawdown_pct']
                    })
                else:
                    print(f"\r{period_name:<30} {method.capitalize():<12} No data available")
                
            except Exception as e:
                print(f"\r{period_name:<30} {method.capitalize():<12} Error: {str(e)[:30]}...")
        
        print()  # Add spacing between periods
    
    # Summary statistics
    if results_summary:
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        # Group by method
        equal_results = [r for r in results_summary if r['method'] == 'equal']
        handcrafted_results = [r for r in results_summary if r['method'] == 'handcrafted']
        
        if equal_results:
            avg_return_equal = sum(r['ann_return'] for r in equal_results) / len(equal_results)
            avg_sharpe_equal = sum(r['sharpe'] for r in equal_results) / len(equal_results)
            avg_vol_equal = sum(r['volatility'] for r in equal_results) / len(equal_results)
            print(f"Equal Weight Average:     Ann. Return: {avg_return_equal:>8.2%} | "
                  f"Volatility: {avg_vol_equal:>8.2%} | Sharpe: {avg_sharpe_equal:>6.3f}")
        
        if handcrafted_results:
            avg_return_hand = sum(r['ann_return'] for r in handcrafted_results) / len(handcrafted_results)
            avg_sharpe_hand = sum(r['sharpe'] for r in handcrafted_results) / len(handcrafted_results)
            avg_vol_hand = sum(r['volatility'] for r in handcrafted_results) / len(handcrafted_results)
            print(f"Handcrafted Average:      Ann. Return: {avg_return_hand:>8.2%} | "
                  f"Volatility: {avg_vol_hand:>8.2%} | Sharpe: {avg_sharpe_hand:>6.3f}")
        
        if equal_results and handcrafted_results:
            return_diff = avg_return_hand - avg_return_equal
            sharpe_diff = avg_sharpe_hand - avg_sharpe_equal
            vol_diff = avg_vol_hand - avg_vol_equal
            print(f"Handcrafted Advantage:    Ann. Return: {return_diff:>+8.2%} | "
                  f"Volatility: {vol_diff:>+8.2%} | Sharpe: {sharpe_diff:>+6.3f}")
        
        # Best performing periods
        print(f"\n" + "=" * 50)
        print("BEST PERFORMING PERIODS")
        print("=" * 50)
        
        for method in ['equal', 'handcrafted']:
            method_results = [r for r in results_summary if r['method'] == method]
            if method_results:
                best_return = max(method_results, key=lambda x: x['ann_return'])
                best_sharpe = max(method_results, key=lambda x: x['sharpe'])
                
                print(f"\n{method.capitalize()} Weighting:")
                print(f"  Best Return:  {best_return['period']:<25} {best_return['ann_return']:>8.2%}")
                print(f"  Best Sharpe:  {best_sharpe['period']:<25} {best_sharpe['sharpe']:>8.3f}")
    
    return results_summary

def compare_with_mes_only():
    """Compare Strategy 4 with MES-only strategy."""
    print(f"\n" + "=" * 80)
    print("COMPARISON WITH MES-ONLY STRATEGY")
    print("=" * 80)
    
    try:
        # MES only strategy
        mes_results = backtest_variable_risk_strategy(
            'Data/mes_daily_data.csv', 
            capital=50000000, 
            risk_target=0.2
        )
        
        # Strategy 4 handcrafted
        strategy4_results = backtest_multi_instrument_strategy(
            data_dir='Data',
            capital=50000000,
            risk_target=0.2,
            weight_method='handcrafted'
        )
        
        if mes_results and strategy4_results:
            mes_perf = mes_results['performance']
            s4_perf = strategy4_results['performance']
            
            print(f"\n{'Strategy':<20} {'Ann. Return':<12} {'Volatility':<12} {'Sharpe':<8} {'Max DD':<8}")
            print("-" * 65)
            print(f"{'MES Only':<20} {mes_perf['annualized_return']:<12.2%} "
                  f"{mes_perf['annualized_volatility']:<12.2%} "
                  f"{mes_perf['sharpe_ratio']:<8.3f} "
                  f"{mes_perf['max_drawdown_pct']:<8.1f}%")
            print(f"{'Strategy 4':<20} {s4_perf['annualized_return']:<12.2%} "
                  f"{s4_perf['annualized_volatility']:<12.2%} "
                  f"{s4_perf['sharpe_ratio']:<8.3f} "
                  f"{s4_perf['max_drawdown_pct']:<8.1f}%")
            
            return_diff = s4_perf['annualized_return'] - mes_perf['annualized_return']
            sharpe_diff = s4_perf['sharpe_ratio'] - mes_perf['sharpe_ratio']
            
            print(f"{'Advantage':<20} {return_diff:<+12.2%} "
                  f"{'':12} {sharpe_diff:<+8.3f}")
        
    except Exception as e:
        print(f"Error in comparison: {e}")

def main():
    """Main function to run all analyses."""
    # Run time period analysis
    results = run_time_period_analysis()
    
    # Compare with MES only
    compare_with_mes_only()
    
    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    main() 