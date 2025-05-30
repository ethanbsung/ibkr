#!/usr/bin/env python3
"""
Simple test to identify the dynamic optimization backtest issue.
"""

from chapter4 import *
from dynamic_optimization import *
import warnings
warnings.filterwarnings('ignore')

def test_dynamic_backtest_simple():
    """Test dynamic backtest with minimal data to find the issue."""
    print("Testing dynamic optimization backtest with minimal data...")
    
    # Load just 2 instruments
    instruments_df = load_instrument_data()
    available_instruments = get_available_instruments(instruments_df)
    test_symbols = available_instruments[:2]
    
    print(f"Testing with: {test_symbols}")
    
    # Load recent data only (last 100 days)
    data = load_instrument_data_files(test_symbols, start_date='2024-06-01', end_date='2024-12-01')
    
    if not data:
        print("No data loaded!")
        return
    
    print(f"Loaded data for: {list(data.keys())}")
    for symbol, df in data.items():
        print(f"  {symbol}: {len(df)} rows, price range: ${df['Last'].min():.2f} - ${df['Last'].max():.2f}")
    
    # Simple equal weights
    portfolio_weights = {symbol: 0.5 for symbol in data.keys()}
    capital = 1000000
    
    # Run a short dynamic optimization backtest
    result = backtest_portfolio_with_dynamic_optimization(
        portfolio_weights, data, instruments_df, capital,
        start_date='2024-06-01', end_date='2024-08-01',  # Just 2 months
        cost_multiplier=1,  # Low cost
        use_buffering=False,  # No buffering
        rebalance_frequency='weekly'  # Weekly rebalancing
    )
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"Backtest completed successfully!")
    perf = result['performance']
    print(f"Annual Return: {perf['annualized_return']:.2%}")
    print(f"Volatility: {perf['annualized_volatility']:.2%}")
    print(f"Sharpe: {perf['sharpe_ratio']:.3f}")
    
    # Check position sizes and notional exposure
    final_positions = result['final_positions']
    print(f"\nPosition Analysis:")
    total_notional = 0
    for symbol in final_positions:
        pos = final_positions[symbol]
        if pos != 0:
            specs = get_instrument_specs(symbol, instruments_df)
            last_price = data[symbol]['Last'].iloc[-1]
            notional = abs(pos) * specs['multiplier'] * last_price
            total_notional += notional
            leverage_ratio = notional / capital
            print(f"  {symbol}: {pos} contracts, ${notional:,.0f} notional, {leverage_ratio:.1f}x leverage")
    
    total_leverage = total_notional / capital
    print(f"  Total leverage: {total_leverage:.1f}x")
    
    # Check the data for any obvious issues
    df = result['data']
    print(f"\nReturns stats:")
    print(f"  Min: {df['portfolio_returns'].min():.6f}")
    print(f"  Max: {df['portfolio_returns'].max():.6f}")
    print(f"  Mean: {df['portfolio_returns'].mean():.6f}")
    print(f"  Std: {df['portfolio_returns'].std():.6f}")
    print(f"  NaN count: {df['portfolio_returns'].isna().sum()}")
    print(f"  Inf count: {np.isinf(df['portfolio_returns']).sum()}")
    
    print(f"\nFirst few returns:")
    print(df['portfolio_returns'].head(10))
    
    print(f"\nLast few returns:")
    print(df['portfolio_returns'].tail(10))
    
    print(f"\nExtreme returns:")
    extreme_mask = np.abs(df['portfolio_returns']) > 0.05  # > 5% daily
    if extreme_mask.any():
        extreme_returns = df[extreme_mask]
        print(extreme_returns[['portfolio_returns'] + [col for col in df.columns if 'position_' in col]])
    else:
        print("No extreme returns found")

if __name__ == "__main__":
    test_dynamic_backtest_simple() 