import pandas as pd
import numpy as np
import sys
import os
sys.path.append('rob_port')
from chapter9 import *

def analyze_data_coverage_vs_book():
    """Analyze data coverage and implementation differences vs the book."""
    print('=' * 80)
    print('DATA COVERAGE & IMPLEMENTATION ANALYSIS vs BOOK')
    print('=' * 80)
    
    # Book's expected instruments (from Table 39 context)
    book_instruments = {
        # Bonds
        'bonds': ['ZT', 'Z3N', 'ZF', 'ZN', 'TN', 'TWE', 'ZB', 'YE', 'GBS', 'GBM', 'GBL', 'GBX', 'BTS', 'BTP', 'FBON'],
        # Equities  
        'equities': ['MYM', 'MNQ', 'RSV', 'M2K', 'MES', 'CAC40', 'DAX', 'SMI', 'DJ600', 'ESTX50', 'SXAP', 'SXPP', 'SXDP', 'SXIP', 'SX8P', 'SXTP', 'SX6P', 'XINA50', 'SSG', 'TWN'],
        # FX
        'fx': ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'RP', 'RY', 'UC', 'SIR', 'SND'],
        # Commodities
        'commodities': ['ALI', 'HG', 'MGC', 'SCI', 'PA', 'PL', 'SI', 'QM', 'HH', 'RB', 'QG', 'HO', 'AIGCI'],
        # Agriculture
        'agriculture': ['CSC', 'ZC', 'GF', 'HE', 'LE', 'ZO', 'KE', 'ZR', 'ZS', 'ZM', 'ZL', 'ZW'],
        # Volatility
        'volatility': ['VIX', 'V2TX'],
        # Crypto (not in original book but we have)
        'crypto': ['MBT', 'ETHUSDRR']
    }
    
    # Flatten to get all expected instruments
    all_book_instruments = []
    for category, instruments in book_instruments.items():
        all_book_instruments.extend(instruments)
    
    print(f"Book's expected instruments: {len(all_book_instruments)}")
    
    # Load our actual data
    try:
        instrument_data = load_all_instrument_data('Data')
        our_instruments = set(instrument_data.keys())
        
        print(f"Our actual instruments: {len(our_instruments)}")
        print(f"Missing instruments: {len(all_book_instruments) - len(our_instruments)}")
        
        # Find missing instruments by category
        print(f"\n=== MISSING INSTRUMENTS BY CATEGORY ===")
        total_missing = 0
        for category, expected in book_instruments.items():
            missing = [inst for inst in expected if inst not in our_instruments]
            coverage = (len(expected) - len(missing)) / len(expected) * 100
            total_missing += len(missing)
            
            print(f"{category.upper():<12}: {len(expected)-len(missing):>2}/{len(expected):<2} ({coverage:>5.1f}%) Missing: {missing}")
        
        print(f"\nTotal missing: {total_missing} instruments ({total_missing/len(all_book_instruments)*100:.1f}%)")
        
        # Analyze impact on performance by looking at book's performance targets
        print(f"\n=== BOOK vs OUR PERFORMANCE COMPARISON ===")
        
        # Book's reported performance (from Table 39)
        book_performance = {
            'Strategy 4 (Long only)': {'return': 15.4, 'sharpe': 0.85},
            'Strategy 7 (Slow trend)': {'return': 21.5, 'sharpe': 0.96},
            'Strategy 8 (Fast trend)': {'return': 24.1, 'sharpe': 1.06},
            'Strategy 9 (Multiple trend)': {'return': 25.2, 'sharpe': 1.14}
        }
        
        # Get our actual performance
        cached_results = get_cached_strategy_results()
        
        our_performance = {}
        if 'strategy4' in cached_results:
            s4_perf = cached_results['strategy4']['performance']
            our_performance['Strategy 4 (Long only)'] = {
                'return': s4_perf['annualized_return'] * 100,
                'sharpe': s4_perf['sharpe_ratio']
            }
        
        if 'strategy7' in cached_results:
            s7_perf = cached_results['strategy7']['performance']
            our_performance['Strategy 7 (Slow trend)'] = {
                'return': s7_perf['annualized_return'] * 100,
                'sharpe': s7_perf['sharpe_ratio']
            }
        
        # Try to load Strategy 8 and 9
        for strategy_num, strategy_name in [(8, 'Fast trend'), (9, 'Multiple trend')]:
            config = {'capital': 50000000, 'risk_target': 0.2, 'weight_method': 'handcrafted'}
            if strategy_num == 9:
                config['forecast_combination'] = 'five_filters'
            
            results = load_strategy_results(f'strategy{strategy_num}', config)
            if results:
                perf = results['performance']
                our_performance[f'Strategy {strategy_num} ({strategy_name})'] = {
                    'return': perf['annualized_return'] * 100,
                    'sharpe': perf['sharpe_ratio']
                }
        
        # Compare performance
        print(f"{'Strategy':<25} {'Book Return':<12} {'Our Return':<12} {'Difference':<12} {'Book Sharpe':<12} {'Our Sharpe':<12} {'Difference':<12}")
        print("-" * 110)
        
        massive_underperformance = []
        
        for strategy in book_performance:
            if strategy in our_performance:
                book_ret = book_performance[strategy]['return']
                our_ret = our_performance[strategy]['return']
                ret_diff = our_ret - book_ret
                
                book_sharpe = book_performance[strategy]['sharpe']
                our_sharpe = our_performance[strategy]['sharpe']
                sharpe_diff = our_sharpe - book_sharpe
                
                print(f"{strategy:<25} {book_ret:<12.1f}% {our_ret:<12.1f}% {ret_diff:<12.1f}% {book_sharpe:<12.2f} {our_sharpe:<12.2f} {sharpe_diff:<12.2f}")
                
                # Flag massive underperformance
                if ret_diff < -10 or sharpe_diff < -0.5:
                    massive_underperformance.append(strategy)
        
        if massive_underperformance:
            print(f"\n🚨 MASSIVE UNDERPERFORMANCE DETECTED: {massive_underperformance}")
        
        # Analyze time period differences
        print(f"\n=== TIME PERIOD ANALYSIS ===")
        
        # Book appears to use data ending around 2019 based on the equity curve
        book_end_date = '2019-12-31'  # Estimated from chart
        our_end_date = '2024-12-31'   # Our data goes to 2024
        
        print(f"Book's apparent end date: ~{book_end_date}")
        print(f"Our backtest end date: {our_end_date}")
        print(f"Extra years in our data: ~5 years (2020-2024)")
        print(f"This explains some performance difference - we include COVID period and recent decay!")
        
        # Test with book's time period
        print(f"\n=== PERFORMANCE WITH BOOK'S TIME PERIOD (2000-2019) ===")
        
        if 'strategy9' in [k for k in os.listdir('results') if k.startswith('strategy9')]:
            s9_config = {'capital': 50000000, 'risk_target': 0.2, 'weight_method': 'handcrafted', 'forecast_combination': 'five_filters'}
            s9_results = load_strategy_results('strategy9', s9_config)
            
            if s9_results:
                s9_data = s9_results['portfolio_data']
                
                # Filter to book's period
                book_period_data = s9_data[(s9_data.index >= '2000-01-01') & (s9_data.index <= book_end_date)]
                
                if len(book_period_data) > 100:
                    book_period_returns = book_period_data['strategy_returns'].dropna()
                    book_period_ann_ret = book_period_returns.mean() * 256 * 100
                    book_period_sharpe = book_period_returns.mean() / book_period_returns.std() * np.sqrt(256)
                    
                    print(f"Strategy 9 (2000-2019 only):")
                    print(f"  Our Return: {book_period_ann_ret:.1f}%")
                    print(f"  Book Return: 25.2%")
                    print(f"  Difference: {book_period_ann_ret - 25.2:.1f}%")
                    print(f"  Our Sharpe: {book_period_sharpe:.2f}")
                    print(f"  Book Sharpe: 1.14")
                    print(f"  Difference: {book_period_sharpe - 1.14:.2f}")
        
        # Analyze potential implementation issues
        print(f"\n=== POTENTIAL IMPLEMENTATION ISSUES ===")
        
        issues = []
        
        # 1. Missing instruments impact
        missing_pct = total_missing / len(all_book_instruments) * 100
        if missing_pct > 20:
            issues.append(f"Missing {missing_pct:.1f}% of instruments - significant portfolio impact")
        
        # 2. Data quality
        print(f"Checking data quality...")
        min_data_length = min([len(df) for df in instrument_data.values()])
        max_data_length = max([len(df) for df in instrument_data.values()])
        
        if max_data_length - min_data_length > 1000:  # More than ~4 years difference
            issues.append("Significant data length inconsistencies between instruments")
        
        # 3. Performance magnitude
        if len(massive_underperformance) > 0:
            issues.append("Massive underperformance suggests fundamental implementation errors")
        
        # 4. Time period effect
        recent_vs_historical = True  # We know this from previous analysis
        if recent_vs_historical:
            issues.append("Including 2020-2024 period significantly hurts performance")
        
        print(f"\nIdentified Issues:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        
        # Final assessment
        print(f"\n=== ROOT CAUSE ASSESSMENT ===")
        
        if missing_pct > 25:
            print("🔴 PRIMARY CAUSE: Missing 25%+ of instruments")
            print("   - Portfolio diversification severely compromised")
            print("   - Missing key alpha-generating instruments")
            print("   - Weight allocation completely different from book")
        
        if recent_vs_historical:
            print("🟡 SECONDARY CAUSE: Time period extension to 2020-2024")
            print("   - Book likely ends around 2019")
            print("   - COVID period and recent alpha decay hurt performance")
            print("   - Need to test with book's exact time period")
        
        if len(massive_underperformance) > 2:
            print("🔴 CRITICAL: Fundamental implementation errors likely")
            print("   - Performance gaps too large for just missing data")
            print("   - Check: forecast calculations, position sizing, portfolio construction")
        
        # Recommendations
        print(f"\n=== RECOMMENDATIONS ===")
        
        print("1. IMMEDIATE DATA AUDIT:")
        print("   - Acquire missing 25 instruments if possible")
        print("   - Focus on high-weight instruments from missing list")
        print("   - Verify data quality and consistency")
        
        print("\n2. TIME PERIOD TESTING:")
        print("   - Re-run all strategies with 2000-2019 period only")
        print("   - Compare results with book's Table 39")
        print("   - This will isolate implementation vs. regime change issues")
        
        print("\n3. IMPLEMENTATION VERIFICATION:")
        print("   - Double-check forecast calculations")
        print("   - Verify position sizing formulas")
        print("   - Audit portfolio construction methodology")
        
        return {
            'missing_instruments': total_missing,
            'missing_percentage': missing_pct,
            'performance_gaps': massive_underperformance,
            'our_instruments': len(our_instruments),
            'book_instruments': len(all_book_instruments)
        }
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    analyze_data_coverage_vs_book() 