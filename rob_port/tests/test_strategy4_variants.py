#!/usr/bin/env python3
"""
Test Strategy 4 with different capital levels and compare approaches.
"""

from chapter4 import *
from chapter3 import *
from chapter2 import *
from chapter1 import *
import pandas as pd

def test_strategy4_capital_levels():
    """Test Strategy 4 with different capital levels."""
    
    print("STRATEGY 4: CAPITAL LEVEL COMPARISON")
    print("=" * 60)
    
    # Load instruments data
    instruments_df = load_instrument_data()
    
    # Test different capital levels
    capital_levels = [
        1000000,    # $1M
        5000000,    # $5M  
        10000000,   # $10M
        25000000,   # $25M
        50000000    # $50M
    ]
    
    risk_target = 0.20
    
    results_summary = []
    
    for capital in capital_levels:
        print(f"\n----- Testing with ${capital:,.0f} Capital -----")
        
        try:
            # Implement Strategy 4
            results = implement_strategy4(
                instruments_df, capital, risk_target, max_instruments=15
            )
            
            if results:
                num_instruments = len(results['selected_instruments'])
                idm = results['idm']
                portfolio_sr = results['portfolio_sr']
                total_notional = results['total_notional']
                leverage = total_notional / capital
                
                results_summary.append({
                    'Capital': capital,
                    'Instruments': num_instruments,
                    'IDM': idm,
                    'Portfolio_SR': portfolio_sr,
                    'Leverage': leverage,
                    'Notional': total_notional
                })
                
                print(f"  Instruments: {num_instruments}")
                print(f"  IDM: {idm:.2f}")
                print(f"  Portfolio SR: {portfolio_sr:.4f}")
                print(f"  Leverage: {leverage:.2f}x")
                print(f"  Total Notional: ${total_notional:,.0f}")
                
            else:
                print("  Failed to implement strategy")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    # Create summary DataFrame
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        
        print(f"\n----- Capital Level Summary -----")
        print(summary_df.to_string(index=False, float_format='%.2f'))

def compare_individual_vs_portfolio():
    """Compare individual instrument trading vs portfolio approach."""
    
    print("\n" + "=" * 60)
    print("INDIVIDUAL VS PORTFOLIO COMPARISON")
    print("=" * 60)
    
    instruments_df = load_instrument_data()
    capital = 10000000  # $10M
    risk_target = 0.20
    
    # Get some example instruments
    test_instruments = ['MES', 'MNQ', 'MYM', 'ZN', 'HG']
    
    print(f"\nCapital: ${capital:,.0f}")
    print(f"Risk Target: {risk_target:.1%}")
    
    # Individual instrument approach
    print(f"\n----- Individual Instrument Analysis -----")
    
    individual_positions = {}
    individual_sharpes = {}
    
    for symbol in test_instruments:
        try:
            specs = get_instrument_specs(symbol, instruments_df)
            
            # Use example price and volatility
            if symbol == 'ZN':
                price, vol = 110, 0.08
            elif symbol == 'HG':
                price, vol = 4.0, 0.25
            else:
                price, vol = 4000, 0.20
            
            # Calculate individual position
            position = calculate_variable_position_size(
                capital, specs['multiplier'], price, vol, risk_target
            )
            
            # Calculate expected Sharpe (from chapter4)
            annual_sr = calculate_instrument_annual_sr(symbol, instruments_df, price, vol)
            
            individual_positions[symbol] = position
            individual_sharpes[symbol] = annual_sr
            
            print(f"{symbol:>6}: Position {position:>8.2f} | SR {annual_sr:>6.4f} | "
                  f"SR Cost {specs['sr_cost']:>8.6f}")
            
        except Exception as e:
            print(f"{symbol:>6}: Error - {e}")
    
    # Portfolio approach
    print(f"\n----- Portfolio Approach -----")
    
    try:
        portfolio_results = implement_strategy4(
            instruments_df, capital, risk_target, max_instruments=10
        )
        
        if portfolio_results:
            print(f"Selected Instruments: {len(portfolio_results['selected_instruments'])}")
            print(f"IDM: {portfolio_results['idm']:.2f}")
            print(f"Portfolio SR: {portfolio_results['portfolio_sr']:.4f}")
            
            # Show positions for our test instruments
            print(f"\nPositions in Portfolio:")
            for symbol in test_instruments:
                if symbol in portfolio_results['position_sizes']:
                    portfolio_pos = portfolio_results['position_sizes'][symbol]
                    individual_pos = individual_positions.get(symbol, 0)
                    ratio = portfolio_pos / individual_pos if individual_pos > 0 else 0
                    
                    print(f"{symbol:>6}: Individual {individual_pos:>8.2f} | "
                          f"Portfolio {portfolio_pos:>8.2f} | Ratio {ratio:>6.2f}x")
                else:
                    print(f"{symbol:>6}: Not selected for portfolio")
                    
    except Exception as e:
        print(f"Portfolio error: {e}")

def demonstrate_idm_effects():
    """Demonstrate the effects of IDM on position sizing."""
    
    print("\n" + "=" * 60)
    print("IDM EFFECTS DEMONSTRATION")  
    print("=" * 60)
    
    instruments_df = load_instrument_data()
    capital = 50000000
    risk_target = 0.20
    
    # Example instrument
    symbol = 'MES'
    specs = get_instrument_specs(symbol, instruments_df)
    price = 4500
    volatility = 0.16
    weight = 0.10  # 10% allocation
    
    print(f"Example: {symbol} ({specs['name']})")
    print(f"Capital: ${capital:,.0f}")
    print(f"Price: ${price}")
    print(f"Volatility: {volatility:.1%}")
    print(f"Weight: {weight:.1%}")
    print(f"Risk Target: {risk_target:.1%}")
    
    print(f"\n{'Instruments':<12} {'IDM':<6} {'Position':<10} {'Leverage Effect':<15}")
    print("-" * 50)
    
    for num_instruments in [1, 2, 5, 10, 15, 20, 30]:
        idm = get_idm_for_instruments(num_instruments)
        
        # Calculate position with IDM
        position = calculate_strategy4_position_size(
            capital, symbol, price, volatility, weight, idm, 
            instruments_df, risk_target
        )
        
        # Calculate what position would be without IDM
        position_no_idm = calculate_strategy4_position_size(
            capital, symbol, price, volatility, weight, 1.0,
            instruments_df, risk_target
        )
        
        leverage_effect = position / position_no_idm if position_no_idm > 0 else 0
        
        print(f"{num_instruments:<12} {idm:<6.2f} {position:<10.2f} {leverage_effect:<15.2f}x")

def risk_target_analysis():
    """Analyze appropriate risk targets for different portfolio sizes."""
    
    print("\n" + "=" * 60)
    print("RISK TARGET ANALYSIS")
    print("=" * 60)
    
    print("Book recommendations:")
    print("- Single instrument: 10% risk target")
    print("- 2-6 instruments: 10-20% interpolated")
    print("- 7+ instruments with good diversification: up to 25%")
    print("- Institutional portfolios: potentially higher")
    
    instruments_df = load_instrument_data()
    capital = 50000000
    
    portfolio_sizes = [1, 3, 5, 10, 15, 20]
    risk_targets = [0.10, 0.15, 0.20, 0.25, 0.30]
    
    print(f"\nIDM scaling effects:")
    print(f"{'Portfolio Size':<15} {'IDM':<6} {'Recommended Risk Target':<25}")
    print("-" * 50)
    
    for size in portfolio_sizes:
        idm = get_idm_for_instruments(size)
        
        if size == 1:
            recommended = "10%"
        elif size <= 6:
            recommended = "10-20% (interpolated)"
        elif size <= 15:
            recommended = "20-25%"
        else:
            recommended = "25%+"
            
        print(f"{size:<15} {idm:<6.2f} {recommended:<25}")

def main():
    """Run all Strategy 4 tests."""
    
    # Test 1: Different capital levels
    test_strategy4_capital_levels()
    
    # Test 2: Individual vs portfolio comparison  
    compare_individual_vs_portfolio()
    
    # Test 3: IDM effects demonstration
    demonstrate_idm_effects()
    
    # Test 4: Risk target analysis
    risk_target_analysis()
    
    print(f"\n" + "=" * 60)
    print("STRATEGY 4 TESTING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main() 