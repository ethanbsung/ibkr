#!/usr/bin/env python3
"""
Debug script to verify dynamic optimization implementation against book examples.
"""

from chapter4 import *
from dynamic_optimization import *
import warnings
warnings.filterwarnings('ignore')

def test_book_example_exactly():
    """Test the exact example from the book to verify implementation."""
    print("=" * 70)
    print("TESTING BOOK EXAMPLE EXACTLY")
    print("=" * 70)
    
    # Book's exact example data (Chapter 25)
    capital = 500000
    instruments = ['US_5Y', 'US_10Y', 'SP500']
    
    # Book's data
    portfolio_weights = {'US_5Y': 0.33, 'US_10Y': 0.33, 'SP500': 0.34}
    prices = {'US_5Y': 110, 'US_10Y': 115, 'SP500': 4500}
    multipliers = {'US_5Y': 1000, 'US_10Y': 1000, 'SP500': 5}
    volatilities = {'US_5Y': 0.052, 'US_10Y': 0.082, 'SP500': 0.171}
    
    # Book's correlation matrix
    correlation_data = [
        [1.0, 0.9, -0.1],
        [0.9, 1.0, -0.1], 
        [-0.1, -0.1, 1.0]
    ]
    correlation_matrix = pd.DataFrame(correlation_data, index=instruments, columns=instruments)
    
    print("=== BOOK'S INPUT DATA ===")
    print(f"Capital: ${capital:,}")
    print(f"Portfolio weights: {portfolio_weights}")
    print(f"Prices: {prices}")
    print(f"Multipliers: {multipliers}")
    print(f"Volatilities: {volatilities}")
    print(f"Correlation matrix:")
    print(correlation_matrix)
    
    # Step 1: Calculate IDM
    idm = calculate_idm_from_correlations(portfolio_weights, correlation_matrix)
    print(f"\n=== STEP 1: IDM CALCULATION ===")
    print(f"IDM: {idm:.3f}")
    print(f"Book expected: ~1.5")
    
    # Step 2: Calculate optimal unrounded positions
    print(f"\n=== STEP 2: OPTIMAL UNROUNDED POSITIONS ===")
    risk_target = 0.20
    combined_forecast = 1.0  # Neutral forecast
    
    optimal_positions = {}
    weight_per_contract = {}
    
    for symbol in instruments:
        # Optimal position calculation
        optimal_pos = calculate_optimal_unrounded_position(
            capital, combined_forecast, idm, portfolio_weights[symbol], 
            volatilities[symbol], multipliers[symbol], prices[symbol], 1.0, risk_target
        )
        optimal_positions[symbol] = optimal_pos
        
        # Weight per contract
        wpc = calculate_weight_per_contract(multipliers[symbol], prices[symbol], 1.0, capital)
        weight_per_contract[symbol] = wpc
        
        print(f"{symbol}: Optimal={optimal_pos:.2f}, WPC={wpc:.4f}")
    
    print(f"Note: These optimal positions are calculated from the given inputs and may differ from")
    print(f"the arbitrary example values shown in Table 114 of the book.")
    
    # Step 3: Create covariance matrix
    print(f"\n=== STEP 3: COVARIANCE MATRIX ===")
    vol_series = pd.Series(volatilities)
    vol_matrix = np.outer(vol_series.values, vol_series.values)
    covariance_matrix = pd.DataFrame(vol_matrix * correlation_matrix.values, 
                                   index=instruments, columns=instruments)
    
    print("Covariance matrix:")
    print(covariance_matrix.round(6))
    
    # Step 4: Test tracking error with zero positions
    print(f"\n=== STEP 4: TRACKING ERROR (ZERO POSITIONS) ===")
    current_positions = {'US_5Y': 0, 'US_10Y': 0, 'SP500': 0}
    
    # Calculate tracking error weights
    tracking_weights = {}
    for symbol in instruments:
        current_weight = current_positions[symbol] * weight_per_contract[symbol]
        optimal_weight = optimal_positions[symbol] * weight_per_contract[symbol]
        tracking_weights[symbol] = optimal_weight - current_weight
    
    tracking_weights_series = pd.Series(tracking_weights)
    tracking_error_std = calculate_tracking_error_std(tracking_weights_series, covariance_matrix)
    
    print(f"Tracking error weights: {tracking_weights}")
    print(f"Tracking error std: {tracking_error_std:.6f}")
    print(f"Book expected: ~0.0267")
    
    # Step 5: Test costs calculation
    print(f"\n=== STEP 5: COST CALCULATION ===")
    
    # Use book's exact cost values from the example (Table on page 8)
    currency_costs = {'US_5Y': 5.50, 'US_10Y': 11.50, 'SP500': 0.375}  # Actual currency costs from book
    
    cost_weight_terms = {}
    for symbol in instruments:
        # Cost per trade in weight terms using book's exact formula
        # Cost in weight terms = (Currency cost ÷ Capital) ÷ Weight per contract
        currency_cost = currency_costs[symbol]
        cost_weight = (currency_cost / capital) / weight_per_contract[symbol] if weight_per_contract[symbol] != 0 else 0
        cost_weight_terms[symbol] = cost_weight
        
        print(f"{symbol}: Currency_cost=${currency_cost:.2f}, Cost_weight={cost_weight:.8f}")
    
    # Step 6: Test greedy algorithm
    print(f"\n=== STEP 6: GREEDY ALGORITHM ===")
    
    initial_positions = {'US_5Y': 0, 'US_10Y': 0, 'SP500': 0}
    
    # Try different cost multipliers
    for cost_mult in [0, 10, 50, 100]:
        print(f"\n--- Cost Multiplier: {cost_mult} ---")
        
        optimized_positions = run_greedy_algorithm(
            optimal_positions, initial_positions, weight_per_contract,
            covariance_matrix, cost_weight_terms, cost_mult
        )
        
        final_tracking_error = calculate_solution_tracking_error(
            optimized_positions, optimal_positions, weight_per_contract,
            covariance_matrix, cost_weight_terms, cost_mult
        )
        
        print(f"Optimized positions: {optimized_positions}")
        print(f"Final tracking error: {final_tracking_error:.6f}")
        
        # Test manually what happens if we add one more SP500 contract
        test_positions = optimized_positions.copy()
        test_positions['SP500'] = test_positions.get('SP500', 0) + 1
        test_tracking_error = calculate_solution_tracking_error(
            test_positions, optimal_positions, weight_per_contract,
            covariance_matrix, cost_weight_terms, cost_mult
        )
        print(f"If we add 1 more SP500: positions={test_positions}, tracking_error={test_tracking_error:.6f}")
    
    print(f"Book's final optimized positions (Table 123): US_5Y=0, US_10Y=1, SP500=2")
    
    return {
        'idm': idm,
        'optimal_positions': optimal_positions,
        'weight_per_contract': weight_per_contract,
        'covariance_matrix': covariance_matrix,
        'cost_weight_terms': cost_weight_terms
    }

def test_cost_multiplier_sensitivity():
    """Test how different cost multipliers affect performance."""
    print("\n" + "=" * 70)
    print("COST MULTIPLIER SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    # Load real data for testing
    instruments_df = load_instrument_data()
    test_symbols = ['MES', 'MYM', 'MNQ']
    data = load_instrument_data_files(test_symbols, start_date='2023-01-01', end_date='2024-01-01')
    
    portfolio_weights = {symbol: 1.0/3.0 for symbol in data.keys()}
    capital = 500000
    
    # Test different cost multipliers
    cost_multipliers = [0, 1, 5, 10, 25, 50, 100, 200]
    
    results = []
    
    for cost_mult in cost_multipliers:
        print(f"\n--- Testing Cost Multiplier: {cost_mult} ---")
        
        # Backtest with this cost multiplier
        result = backtest_portfolio_with_dynamic_optimization(
            portfolio_weights, data, instruments_df, capital=capital,
            cost_multiplier=cost_mult, rebalance_frequency='weekly'
        )
        
        if 'error' not in result:
            perf = result['performance']
            results.append({
                'cost_multiplier': cost_mult,
                'annual_return': perf['annualized_return'],
                'sharpe_ratio': perf['sharpe_ratio'],
                'max_drawdown': perf['max_drawdown_pct'],
                'annual_turnover': perf.get('annual_turnover', 0)
            })
            
            print(f"  Annual Return: {perf['annualized_return']:.2%}")
            print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
            print(f"  Annual Turnover: {perf.get('annual_turnover', 0):.1f}")
        else:
            print(f"  Error: {result['error']}")
    
    # Display results table
    if results:
        print(f"\n=== COST MULTIPLIER SENSITIVITY RESULTS ===")
        print(f"{'Cost Mult':<10} {'Return':<8} {'Sharpe':<8} {'MaxDD':<8} {'Turnover':<10}")
        print("-" * 50)
        
        for r in results:
            print(f"{r['cost_multiplier']:<10} {r['annual_return']:<8.1%} {r['sharpe_ratio']:<8.3f} "
                  f"{r['max_drawdown']:<8.1f}% {r['annual_turnover']:<10.0f}")
    
    return results

def test_buffering_effectiveness():
    """Test how buffering affects trading frequency and performance."""
    print("\n" + "=" * 70)
    print("BUFFERING EFFECTIVENESS TEST")
    print("=" * 70)
    
    # Load real data
    instruments_df = load_instrument_data()
    test_symbols = ['MES', 'MYM', 'MNQ']
    data = load_instrument_data_files(test_symbols, start_date='2023-01-01', end_date='2024-01-01')
    
    portfolio_weights = {symbol: 1.0/3.0 for symbol in data.keys()}
    capital = 500000
    
    # Test with and without buffering
    buffer_settings = [
        {'use_buffering': False, 'buffer_fraction': 0.0},
        {'use_buffering': True, 'buffer_fraction': 0.01},
        {'use_buffering': True, 'buffer_fraction': 0.05},
        {'use_buffering': True, 'buffer_fraction': 0.10}
    ]
    
    for setting in buffer_settings:
        buffer_desc = f"No Buffering" if not setting['use_buffering'] else f"Buffer {setting['buffer_fraction']:.1%}"
        print(f"\n--- {buffer_desc} ---")
        
        result = backtest_portfolio_with_dynamic_optimization(
            portfolio_weights, data, instruments_df, capital=capital,
            cost_multiplier=50, rebalance_frequency='daily',
            use_buffering=setting['use_buffering'],
            buffer_fraction=setting['buffer_fraction']
        )
        
        if 'error' not in result:
            perf = result['performance']
            print(f"  Annual Return: {perf['annualized_return']:.2%}")
            print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
            print(f"  Annual Turnover: {perf.get('annual_turnover', 0):.1f}")
            print(f"  Avg Tracking Error: {perf.get('avg_tracking_error', 0):.6f}")
            print(f"  Avg Adjustment Factor: {perf.get('avg_adjustment_factor', 1):.3f}")
        else:
            print(f"  Error: {result['error']}")

def main():
    """Run all debug tests."""
    print("DYNAMIC OPTIMIZATION DEBUGGING")
    print("Verifying implementation against book examples...")
    
    # Test 1: Book example verification
    book_results = test_book_example_exactly()
    
    # Test 2: Cost multiplier sensitivity
    cost_results = test_cost_multiplier_sensitivity()
    
    # Test 3: Buffering effectiveness
    test_buffering_effectiveness()
    
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    print("\n1. Check if IDM calculation matches book (~1.5 for 3-instrument example)")
    print("2. Verify optimal positions match book (US_5Y≈0.4, US_10Y≈0.9, SP500≈3.1)")
    print("3. Test tracking error with zero positions (~0.0267)")
    print("4. Optimal cost multiplier should be where performance plateaus")
    print("5. If all components match book but performance doesn't, check:")
    print("   - Rebalancing frequency (daily vs weekly vs monthly)")
    print("   - Cost calculation method")
    print("   - Data quality and alignment issues")

if __name__ == "__main__":
    main() 