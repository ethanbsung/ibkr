#!/usr/bin/env python3
"""
Debug script to examine what's happening in the real backtest.
"""

from chapter4 import *
from dynamic_optimization import *
import warnings
warnings.filterwarnings('ignore')

def debug_real_backtest_positions():
    """Debug the position sizing in the real backtest."""
    print("=" * 70)
    print("DEBUGGING REAL BACKTEST POSITION SIZING")
    print("=" * 70)
    
    # Load the same data as the main backtest
    instruments_df = load_instrument_data()
    available_instruments = get_available_instruments(instruments_df)
    suitable_instruments = select_instruments_by_criteria(instruments_df, available_instruments, 1000000)
    data = load_instrument_data_files(suitable_instruments[:5])  # Just test with 5 instruments
    
    print(f"Testing with {len(data)} instruments: {list(data.keys())}")
    
    # Create simple equal weights like in the main test
    portfolio_weights = {symbol: 1.0/len(data) for symbol in data.keys()}
    capital = 1000000
    
    print(f"Portfolio weights: {portfolio_weights}")
    print(f"Capital: ${capital:,}")
    
    # Get recent data for a single day calculation
    returns_matrix = create_returns_matrix(data)
    if returns_matrix.empty:
        print("No returns data available!")
        return
    
    # Get a recent date for testing
    test_date = returns_matrix.index[-100]  # 100 days from end
    print(f"Testing position calculation for date: {test_date}")
    
    # Prepare instruments data for dynamic optimization
    instruments_data = {}
    for symbol in portfolio_weights.keys():
        if test_date in data[symbol].index:
            try:
                specs = get_instrument_specs(symbol, instruments_df)
                price = data[symbol].loc[test_date, 'Last']
                
                # Get volatility
                symbol_returns = data[symbol]['returns']
                vol_series = calculate_blended_volatility(symbol_returns)
                if test_date in vol_series.index:
                    volatility = vol_series.loc[test_date]
                else:
                    volatility = vol_series.iloc[-1]
                
                if not pd.isna(price) and not pd.isna(volatility) and volatility > 0:
                    instruments_data[symbol] = {
                        'price': price,
                        'volatility': volatility,
                        'specs': specs
                    }
                    
                    print(f"{symbol}: Price=${price:.2f}, Vol={volatility:.1%}, SR_cost={specs['sr_cost']:.5f}")
                    
            except Exception as e:
                print(f"Error with {symbol}: {e}")
    
    if not instruments_data:
        print("No valid instruments data!")
        return
    
    # Calculate optimal positions using dynamic optimization approach
    current_positions = {symbol: 0 for symbol in instruments_data.keys()}
    recent_returns = returns_matrix.loc[:test_date].tail(252)
    
    print(f"\n=== CALCULATING OPTIMAL POSITIONS ===")
    
    try:
        result = calculate_dynamic_portfolio_positions(
            instruments_data, capital, current_positions, 
            portfolio_weights, recent_returns,
            risk_target=0.2, cost_multiplier=50, use_buffering=False
        )
        
        optimal_positions = result['optimal_unrounded']
        weight_per_contract = result['weight_per_contract']
        
        print(f"IDM: {result['idm']:.3f}")
        print(f"\nOptimal Unrounded Positions:")
        for symbol, pos in optimal_positions.items():
            wpc = weight_per_contract[symbol]
            optimal_weight = pos * wpc
            print(f"  {symbol}: {pos:.3f} contracts (weight: {optimal_weight:.3%})")
        
        print(f"\nOptimized Integer Positions:")
        final_positions = result['positions']
        for symbol, pos in final_positions.items():
            wpc = weight_per_contract[symbol]
            actual_weight = pos * wpc
            print(f"  {symbol}: {pos} contracts (weight: {actual_weight:.3%})")
            
        # Compare with simple position sizing
        print(f"\n=== COMPARISON WITH SIMPLE POSITION SIZING ===")
        for symbol in instruments_data.keys():
            data_item = instruments_data[symbol]
            simple_pos = calculate_position_size_with_idm(
                capital, portfolio_weights[symbol], result['idm'], 
                data_item['specs']['multiplier'], data_item['price'], 
                1.0, data_item['volatility'], 0.2
            )
            print(f"  {symbol}: Simple={simple_pos:.3f}, Dynamic={optimal_positions[symbol]:.3f}")
            
    except Exception as e:
        print(f"Error in dynamic optimization: {e}")
        import traceback
        traceback.print_exc()

def debug_cost_calculation():
    """Debug the cost calculation differences."""
    print("\n" + "=" * 70)
    print("DEBUGGING COST CALCULATION")
    print("=" * 70)
    
    # Load a few instruments
    instruments_df = load_instrument_data()
    available_instruments = get_available_instruments(instruments_df)
    test_symbols = available_instruments[:3]
    
    capital = 1000000
    
    for symbol in test_symbols:
        try:
            specs = get_instrument_specs(symbol, instruments_df)
            
            # Assume typical values
            price = 100.0
            volatility = 0.15
            multiplier = specs['multiplier']
            sr_cost = specs['sr_cost']
            
            # Calculate weight per contract
            wpc = calculate_weight_per_contract(multiplier, price, 1.0, capital)
            
            # Method 1: Current implementation (from dynamic_optimization.py)
            cost_pct = sr_cost * volatility
            cost_weight_current = cost_pct / wpc if wpc != 0 else float('inf')
            
            # Method 2: Book's formula with estimated currency cost
            notional_exposure = multiplier * price
            currency_cost_estimate = sr_cost * notional_exposure
            cost_weight_book = (currency_cost_estimate / capital) / wpc if wpc != 0 else float('inf')
            
            print(f"{symbol}:")
            print(f"  SR_cost: {sr_cost:.5f}")
            print(f"  Multiplier: {multiplier}")
            print(f"  Weight per contract: {wpc:.5f}")
            print(f"  Current method cost: {cost_weight_current:.8f}")
            print(f"  Book method cost: {cost_weight_book:.8f}")
            print(f"  Ratio: {cost_weight_book/cost_weight_current:.2f}x")
            print()
            
        except Exception as e:
            print(f"Error with {symbol}: {e}")

if __name__ == "__main__":
    debug_real_backtest_positions()
    debug_cost_calculation() 