#!/usr/bin/env python3
"""
Quick test for Strategy 9 to check if P&L bug fix was applied
"""

import sys
sys.path.append('rob_port')
from chapter9 import backtest_multiple_trend_strategy

def test_strategy9():
    print("=" * 80)
    print("STRATEGY 9 QUICK TEST")
    print("=" * 80)
    
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
        # Use full available data period
        # 'start_date': '2023-01-01',
        # 'end_date': '2023-12-31',
        # Strategy 9 specific parameters
        'forecast_combination': 'five_filters',
        'buffer_fraction': 0.1
    }
    
    print(f"Testing Strategy 9 with multiple trend rules...")
    print(f"Forecast Combination: {config['forecast_combination']}")
    print(f"Buffer Fraction: {config['buffer_fraction']}")
    
    try:
        result = backtest_multiple_trend_strategy(**config)
        perf = result['performance']
        
        print(f"\nStrategy 9 Results:")
        print(f"  Annualized Return: {perf['annualized_return']:.2%}")
        print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {perf['max_drawdown_pct']:.1f}%")
        print(f"  Avg Combined Forecast: {perf['avg_combined_forecast']:.2f}")
        print(f"  Avg Daily Trades: {perf['avg_daily_trades']:.1f}")
        
        return result
        
    except Exception as e:
        print(f"Strategy 9 failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_strategy9() 