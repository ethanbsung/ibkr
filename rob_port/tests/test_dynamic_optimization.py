#!/usr/bin/env python3
"""
Test script for dynamic optimization functionality.
This demonstrates Chapter 25's dynamic optimization in action.
"""

from chapter4 import *
from dynamic_optimization import *
import numpy as np
import pandas as pd

def test_dynamic_optimization_simple():
    """
    Test dynamic optimization with a simple 3-instrument portfolio.
    """
    print("=" * 70)
    print("DYNAMIC OPTIMIZATION SIMPLE TEST")
    print("=" * 70)
    
    # Load data
    print("Loading instruments and data...")
    instruments_df = load_instrument_data()
    
    # Use a simple 3-instrument portfolio for testing
    test_symbols = ['MES', 'MYM', 'MNQ']  # 3 equity indices
    
    # Load data for these instruments
    data = load_instrument_data_files(test_symbols, start_date='2020-01-01', end_date='2024-01-01')
    
    if len(data) < 3:
        print(f"Error: Only loaded {len(data)} instruments, need 3 for test")
        return
    
    print(f"Loaded data for: {list(data.keys())}")
    
    # Create a simple equal-weight portfolio
    portfolio_weights = {symbol: 1.0/3.0 for symbol in data.keys()}
    
    print(f"\nTesting Portfolio:")
    for symbol, weight in portfolio_weights.items():
        print(f"  {symbol}: {weight:.1%}")
    
    # Test static optimization first
    print(f"\n----- Static Optimization Backtest -----")
    static_result = backtest_portfolio_with_individual_data(
        portfolio_weights, data, instruments_df, capital=100000, risk_target=0.2
    )
    
    if 'error' in static_result:
        print(f"Static optimization error: {static_result['error']}")
    else:
        static_perf = static_result['performance']
        print(f"Static Results:")
        print(f"  Annual Return: {static_perf['annualized_return']:.2%}")
        print(f"  Sharpe Ratio: {static_perf['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {static_perf['max_drawdown_pct']:.1f}%")
        print(f"  IDM: {static_result['idm']:.2f}")
    
    # Test dynamic optimization
    print(f"\n----- Dynamic Optimization Backtest -----")
    dynamic_result = backtest_portfolio_with_dynamic_optimization(
        portfolio_weights, data, instruments_df, capital=100000, risk_target=0.2,
        cost_multiplier=50, use_buffering=True, rebalance_frequency='weekly'
    )
    
    if 'error' in dynamic_result:
        print(f"Dynamic optimization error: {dynamic_result['error']}")
    else:
        dynamic_perf = dynamic_result['performance']
        print(f"Dynamic Results:")
        print(f"  Annual Return: {dynamic_perf['annualized_return']:.2%}")
        print(f"  Sharpe Ratio: {dynamic_perf['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {dynamic_perf['max_drawdown_pct']:.1f}%")
        print(f"  Average Tracking Error: {dynamic_perf.get('avg_tracking_error', 0):.6f}")
        print(f"  Annual Turnover: {dynamic_perf.get('annual_turnover', 0):.1f} contracts")
        print(f"  Average Adjustment Factor: {dynamic_perf.get('avg_adjustment_factor', 1):.3f}")
    
    # Comparison
    if 'error' not in static_result and 'error' not in dynamic_result:
        print(f"\n----- Performance Comparison -----")
        print(f"{'Metric':<20} {'Static':<12} {'Dynamic':<12} {'Improvement':<12}")
        print("-" * 60)
        
        s_ret = static_perf['annualized_return']
        d_ret = dynamic_perf['annualized_return']
        s_sharpe = static_perf['sharpe_ratio']
        d_sharpe = dynamic_perf['sharpe_ratio']
        s_dd = static_perf['max_drawdown_pct']
        d_dd = dynamic_perf['max_drawdown_pct']
        
        print(f"{'Annual Return':<20} {s_ret:<12.1%} {d_ret:<12.1%} {d_ret-s_ret:+.1%}")
        print(f"{'Sharpe Ratio':<20} {s_sharpe:<12.3f} {d_sharpe:<12.3f} {d_sharpe-s_sharpe:+.3f}")
        print(f"{'Max Drawdown':<20} {s_dd:<12.1f}% {d_dd:<12.1f}% {s_dd-d_dd:+.1f}pp")
        
        turnover = dynamic_perf.get('annual_turnover', 0)
        print(f"{'Annual Turnover':<20} {'0':<12} {turnover:<12.0f} {'-':<12}")
    
    return static_result, dynamic_result

def test_dynamic_optimization_components():
    """
    Test individual components of dynamic optimization.
    """
    print("\n" + "=" * 70)
    print("DYNAMIC OPTIMIZATION COMPONENTS TEST")
    print("=" * 70)
    
    # Test with simulated data matching the book example
    capital = 500000
    instruments = ['US_5Y', 'US_10Y', 'SP500']
    
    # Simulate instrument data
    instruments_data = {
        'US_5Y': {
            'price': 110.0,
            'volatility': 0.052,
            'specs': {'multiplier': 1000, 'sr_cost': 0.00167}
        },
        'US_10Y': {
            'price': 120.0,
            'volatility': 0.082,
            'specs': {'multiplier': 1000, 'sr_cost': 0.00160}
        },
        'SP500': {
            'price': 4500.0,
            'volatility': 0.171,
            'specs': {'multiplier': 5, 'sr_cost': 0.00028}
        }
    }
    
    portfolio_weights = {'US_5Y': 0.33, 'US_10Y': 0.33, 'SP500': 0.34}
    current_positions = {'US_5Y': 0, 'US_10Y': 0, 'SP500': 0}
    
    # Create synthetic returns matrix for covariance calculation
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    
    # Generate correlated returns
    correlation_matrix = np.array([
        [1.0, 0.9, -0.1],
        [0.9, 1.0, -0.1],
        [-0.1, -0.1, 1.0]
    ])
    
    volatilities = np.array([0.052, 0.082, 0.171]) / np.sqrt(252)  # Daily volatilities
    
    # Generate multivariate normal returns
    mean_returns = np.zeros(3)
    cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
    returns_data = np.random.multivariate_normal(mean_returns, cov_matrix, len(dates))
    
    returns_matrix = pd.DataFrame(returns_data, index=dates, columns=instruments)
    
    print(f"Testing with simulated data:")
    print(f"  Capital: ${capital:,.0f}")
    print(f"  Instruments: {instruments}")
    print(f"  Portfolio weights: {portfolio_weights}")
    
    # Test dynamic optimization calculation
    print(f"\n----- Testing Dynamic Optimization Calculation -----")
    try:
        optimization_result = calculate_dynamic_portfolio_positions(
            instruments_data, capital, current_positions, 
            portfolio_weights, returns_matrix,
            risk_target=0.2, cost_multiplier=50, use_buffering=True
        )
        
        print(f"Optimization Results:")
        print(f"  Optimal Positions: {optimization_result.get('positions', {})}")
        print(f"  Optimal Unrounded: {optimization_result.get('optimal_unrounded', {})}")
        print(f"  IDM: {optimization_result.get('idm', 1.0):.2f}")
        print(f"  Tracking Error: {optimization_result.get('tracking_error', 0):.6f}")
        print(f"  Adjustment Factor: {optimization_result.get('adjustment_factor', 1.0):.3f}")
        print(f"  Buffer: {optimization_result.get('buffer', 0):.6f}")
        
        if 'error' in optimization_result:
            print(f"  Error: {optimization_result['error']}")
    
    except Exception as e:
        print(f"Error in dynamic optimization: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Run all dynamic optimization tests.
    """
    try:
        # Test with real data
        static_result, dynamic_result = test_dynamic_optimization_simple()
        
        # Test components with simulated data
        test_dynamic_optimization_components()
        
        print(f"\n" + "=" * 70)
        print("DYNAMIC OPTIMIZATION TEST COMPLETE")
        print("=" * 70)
        
        print(f"\nKey Takeaways:")
        print(f"1. Dynamic optimization successfully calculates optimal integer positions")
        print(f"2. The greedy algorithm finds solutions that minimize tracking error")
        print(f"3. Buffering reduces unnecessary turnover when tracking error is small")
        print(f"4. Cost penalties ensure transaction costs are considered in optimization")
        print(f"5. The system gracefully falls back to simple sizing if optimization fails")
        
        print(f"\nUsage in your strategies:")
        print(f"- Use backtest_portfolio_with_dynamic_optimization() instead of backtest_portfolio_with_individual_data()")
        print(f"- Set rebalance_frequency to 'weekly' or 'monthly' for practical implementation")
        print(f"- Enable buffering (use_buffering=True) to reduce turnover")
        print(f"- Adjust cost_multiplier (50-100) based on your transaction costs")
        print(f"- Monitor avg_tracking_error and annual_turnover metrics")
        
    except Exception as e:
        print(f"Error in testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 