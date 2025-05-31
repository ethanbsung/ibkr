#!/usr/bin/env python3
"""
Test different cost multipliers to find optimal dynamic optimization settings.
"""

from chapter4 import *
import warnings
warnings.filterwarnings('ignore')

def test_cost_multipliers():
    """Test different cost multipliers to find optimal settings."""
    print("=" * 70)
    print("TESTING DIFFERENT COST MULTIPLIERS")
    print("=" * 70)
    
    # Load a simple 3-instrument portfolio
    instruments_df = load_instrument_data('../Data/instruments.csv')
    test_symbols = ['MES', 'MYM', 'MNQ']
    data = load_instrument_data_files(test_symbols, start_date='2023-01-01', end_date='2024-01-01')
    portfolio_weights = {symbol: 1.0/3.0 for symbol in data.keys()}
    
    # Test different cost multipliers
    cost_multipliers = [0, 5, 10, 20, 50, 100]
    results = []
    
    for cost_mult in cost_multipliers:
        print(f"\n--- Testing Cost Multiplier: {cost_mult} ---")
        
        try:
            # Test dynamic optimization with this cost multiplier
            result = backtest_portfolio_with_dynamic_optimization(
                data, portfolio_weights, instruments_df,
                capital=100000,
                risk_target=0.2,
                cost_multiplier=cost_mult,
                rebalance_frequency='weekly',
                use_buffering=True,
                buffer_fraction=0.1
            )
            
            annual_return = result['annual_return']
            sharpe_ratio = result['sharpe_ratio']
            tracking_error = result.get('avg_tracking_error', 0)
            turnover = result.get('annual_turnover', 0)
            
            results.append({
                'cost_multiplier': cost_mult,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'tracking_error': tracking_error,
                'turnover': turnover
            })
            
            print(f"  Annual Return: {annual_return:.2%}")
            print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
            print(f"  Tracking Error: {tracking_error:.6f}")
            print(f"  Annual Turnover: {turnover:.1f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'cost_multiplier': cost_mult,
                'annual_return': 0,
                'sharpe_ratio': 0,
                'tracking_error': 0,
                'turnover': 0
            })
    
    # Compare with static optimization
    print(f"\n--- Static Optimization Baseline ---")
    try:
        static_result = backtest_portfolio_with_individual_data(
            data, portfolio_weights, instruments_df,
            capital=100000,
            risk_target=0.2
        )
        static_return = static_result['annual_return']
        static_sharpe = static_result['sharpe_ratio']
        
        print(f"  Annual Return: {static_return:.2%}")
        print(f"  Sharpe Ratio: {static_sharpe:.3f}")
        print(f"  Turnover: ~0 (buy and hold)")
        
    except Exception as e:
        print(f"  Error: {e}")
        static_return, static_sharpe = 0, 0
    
    # Summary table
    print("\n" + "=" * 70)
    print("COST MULTIPLIER COMPARISON")
    print("=" * 70)
    print(f"{'Cost Mult':<10} {'Return':<8} {'Sharpe':<8} {'Track Err':<10} {'Turnover':<10}")
    print("-" * 50)
    
    for r in results:
        print(f"{r['cost_multiplier']:<10} {r['annual_return']:<7.2%} {r['sharpe_ratio']:<7.3f} "
              f"{r['tracking_error']:<9.6f} {r['turnover']:<9.1f}")
    
    print(f"{'Static':<10} {static_return:<7.2%} {static_sharpe:<7.3f} {'N/A':<9} {'0':<9}")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if results:
        best_return = max(results, key=lambda x: x['annual_return'])
        best_sharpe = max(results, key=lambda x: x['sharpe_ratio'])
        
        print(f"Best Return: Cost Multiplier {best_return['cost_multiplier']} "
              f"({best_return['annual_return']:.2%})")
        print(f"Best Sharpe: Cost Multiplier {best_sharpe['cost_multiplier']} "
              f"({best_sharpe['sharpe_ratio']:.3f})")
        
        # Find optimal (where return plateaus)
        if len(results) > 1:
            # Look for the point where returns stop improving significantly
            optimal = None
            for i in range(1, len(results)):
                current = results[i]
                previous = results[i-1]
                
                # If return difference < 0.5% and turnover is significantly lower
                if (abs(current['annual_return'] - previous['annual_return']) < 0.005 and 
                    current['turnover'] < previous['turnover'] * 0.8):
                    optimal = current
                    break
            
            if optimal:
                print(f"Recommended: Cost Multiplier {optimal['cost_multiplier']} "
                      f"(good return {optimal['annual_return']:.2%}, lower turnover {optimal['turnover']:.1f})")
            else:
                print("Recommended: Use lowest cost multiplier for best performance")

if __name__ == "__main__":
    test_cost_multipliers() 