#!/usr/bin/env python3
"""
Quick test of optimized selection strategy.
"""

from chapter4 import *
import warnings
warnings.filterwarnings('ignore')

def test_optimized_selection():
    """Test optimized selection strategy only."""
    print("=" * 60)
    print("OPTIMIZED SELECTION STRATEGY TEST")
    print("=" * 60)
    
    # Load data
    instruments_df = load_instrument_data('../Data/instruments.csv')
    available_instruments = get_available_instruments(instruments_df)
    suitable_instruments = select_instruments_by_criteria(
        instruments_df, available_instruments, 1000000
    )
    
    print(f"Available instruments: {len(available_instruments)}")
    print(f"Suitable instruments: {len(suitable_instruments)}")
    
    # Load data for suitable instruments
    data = load_instrument_data_files(suitable_instruments)
    
    print(f"Successfully loaded data for {len(data)} instruments")
    
    # Asset class breakdown
    asset_classes = create_asset_class_groups(instruments_df, list(data.keys()))
    print(f"\nAsset class distribution:")
    for asset_class, symbols in asset_classes.items():
        print(f"  {asset_class}: {len(symbols)} instruments")
    
    # Test different cost multipliers for dynamic optimization
    cost_multipliers = [5, 10, 20, 50]
    print(f"\n" + "=" * 60)
    print("TESTING DIFFERENT COST MULTIPLIERS")
    print("=" * 60)
    
    for cost_mult in cost_multipliers:
        print(f"\n--- Cost Multiplier: {cost_mult} ---")
        
        # Run optimized selection with dynamic optimization
        optimized_selection_instruments = select_instruments_optimized_selection(
            data, instruments_df, max_instruments=15
        )
        
        print(f"Selected {len(optimized_selection_instruments)} instruments for optimization")
        
        # Get portfolio weights (equal weight)
        portfolio_weights = {symbol: 1.0/len(optimized_selection_instruments) 
                           for symbol in optimized_selection_instruments}
        
        # Backtest with dynamic optimization
        result = backtest_portfolio_with_dynamic_optimization(
            data, portfolio_weights, instruments_df,
            capital=1000000,
            risk_target=0.2,
            cost_multiplier=cost_mult,
            rebalance_frequency='weekly',
            use_buffering=True,
            buffer_fraction=0.1
        )
        
        print(f"Annual Return: {result['annual_return']:.2%}")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {result['max_drawdown']:.1%}")
        if 'annual_turnover' in result:
            print(f"Annual Turnover: {result['annual_turnover']:.1f}")
        if 'avg_tracking_error' in result:
            print(f"Avg Tracking Error: {result['avg_tracking_error']:.6f}")
    
    # Compare with static optimization baseline
    print(f"\n--- Static Optimization Baseline ---")
    optimized_selection_instruments = select_instruments_optimized_selection(
        data, instruments_df, max_instruments=15
    )
    portfolio_weights = {symbol: 1.0/len(optimized_selection_instruments) 
                       for symbol in optimized_selection_instruments}
    
    static_result = backtest_portfolio_with_individual_data(
        data, portfolio_weights, instruments_df,
        capital=1000000,
        risk_target=0.2
    )
    
    print(f"Annual Return: {static_result['annual_return']:.2%}")
    print(f"Sharpe Ratio: {static_result['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {static_result['max_drawdown']:.1%}")
    print(f"Annual Turnover: ~0 (buy and hold)")
    
    print(f"\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("Dynamic optimization should perform as well as or better than static")
    print("with appropriate cost multiplier tuning.")

if __name__ == "__main__":
    test_optimized_selection() 