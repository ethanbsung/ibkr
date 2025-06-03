#!/usr/bin/env python3
"""
Test script to demonstrate minimum capital filtering in Strategy 4.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chapter4 import (
    backtest_multi_instrument_strategy,
    test_minimum_capital_calculation,
    test_capital_filtering,
    load_all_instrument_data,
    load_instrument_data,
    load_fx_data,
    get_instrument_currency_mapping,
    filter_instruments_by_capital
)

def test_capital_filtering_focused():
    """
    Test minimum capital filtering with different capital levels.
    """
    print("=" * 80)
    print("MINIMUM CAPITAL FILTERING DEMONSTRATION")
    print("=" * 80)
    
    # Run unit tests first
    print("\n=== Running Unit Tests ===")
    test_minimum_capital_calculation()
    test_capital_filtering()
    
    # Test with real data and different capital levels
    print("\n=== Testing with Real Data ===")
    
    try:
        # Load required data
        print("Loading instrument data...")
        instrument_data = load_all_instrument_data('Data')
        instruments_df = load_instrument_data()
        fx_data = load_fx_data('Data')
        currency_mapping = get_instrument_currency_mapping()
        
        if not instrument_data:
            print("No instrument data loaded. Check Data directory.")
            return
        
        print(f"Loaded {len(instrument_data)} instruments")
        
        # Test different capital levels
        capital_levels = [100000, 500000, 1000000, 5000000]  # $100k to $5M
        
        for capital in capital_levels:
            print(f"\n{'='*20} TESTING ${capital:,.0f} CAPITAL {'='*20}")
            
            # Apply filtering
            filtered_instruments = filter_instruments_by_capital(
                instrument_data, instruments_df, fx_data, currency_mapping,
                capital, risk_target=0.2, assumed_num_instruments=10
            )
            
            original_count = len(instrument_data)
            filtered_count = len(filtered_instruments)
            
            print(f"\nSUMMARY:")
            print(f"  Original Instruments: {original_count}")
            print(f"  Eligible Instruments: {filtered_count}")
            print(f"  Filtered Out: {original_count - filtered_count}")
            print(f"  Eligibility Rate: {filtered_count/original_count:.1%}")
            
    except Exception as e:
        print(f"Error in focused testing: {e}")
        import traceback
        traceback.print_exc()

def test_strategy_with_different_capitals():
    """
    Run strategy backtests with different capital levels.
    """
    print("\n" + "=" * 80)
    print("STRATEGY BACKTESTS WITH DIFFERENT CAPITAL LEVELS")
    print("=" * 80)
    
    capital_levels = [100000, 500000, 1000000]  # Test smaller amounts
    
    for capital in capital_levels:
        print(f"\n{'='*20} BACKTEST WITH ${capital:,.0f} CAPITAL {'='*20}")
        
        try:
            results = backtest_multi_instrument_strategy(
                data_dir='Data',
                capital=capital,
                risk_target=0.2,
                weight_method='equal',  # Use equal for simplicity
                common_hypothetical_SR=0.3,
                annual_turnover_T=7.0,
                start_date='2020-01-01',  # Shorter period for faster testing
                end_date='2021-12-31'
            )
            
            if results and results.get('performance'):
                perf = results['performance']
                num_instruments = results['performance'].get('num_instruments', 0)
                weights = results.get('weights', {})
                
                print(f"\nRESULTS:")
                print(f"  Instruments Used: {num_instruments}")
                print(f"  Total Return: {perf['total_return']:.2%}")
                print(f"  Annualized Return: {perf['annualized_return']:.2%}")
                print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
                print(f"  Max Drawdown: {perf['max_drawdown_pct']:.1f}%")
                
                # Show top 5 instruments by weight
                sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                print(f"  Top 5 Instruments:")
                for symbol, weight in sorted_weights[:5]:
                    print(f"    {symbol}: {weight:.3f}")
                    
            else:
                print(f"  No valid strategy results (likely all instruments filtered out)")
                
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    test_capital_filtering_focused()
    test_strategy_with_different_capitals()
    print("\nMinimum capital filtering tests completed!") 