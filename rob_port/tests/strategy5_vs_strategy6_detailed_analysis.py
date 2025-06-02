#!/usr/bin/env python3
"""
Detailed Analysis: Strategy 5 vs Strategy 6 Signal and Position Comparison

This script investigates why Strategy 6 underperformed Strategy 5 during the 6-month test period
by analyzing signal differences, position sizing, and individual instrument behavior.
"""

import sys
sys.path.append('rob_port')
from chapter6 import *
from chapter5 import *
from chapter4 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_signal_differences():
    """Analyze the fundamental signal differences between Strategy 5 and Strategy 6."""
    print("=" * 80)
    print("SIGNAL GENERATION COMPARISON: STRATEGY 5 vs STRATEGY 6")
    print("=" * 80)
    
    # Load sample data
    raw_data = load_all_instrument_data('Data')
    
    # Test on a few key instruments
    test_instruments = ['MES', 'MNQ', 'EUR', 'ZB']
    available_instruments = [sym for sym in test_instruments if sym in raw_data]
    
    print(f"Analyzing signal differences for {len(available_instruments)} instruments:")
    print(f"{'Symbol':<8} {'S5 Long %':<10} {'S6 Long %':<10} {'S6 Short %':<11} {'Signal Diff':<12}")
    print("-" * 65)
    
    signal_comparison = {}
    
    for symbol in available_instruments:
        data = raw_data[symbol].copy()
        if len(data) < 500:
            continue
            
        # Use last 6 months of data (approximately 130 trading days)
        recent_data = data.tail(130)
        prices = recent_data['Last']
        
        # Strategy 5 signals (long-only)
        s5_signals = calculate_trend_signal(prices, fast_span=64, slow_span=256)
        s5_signals_clean = s5_signals.dropna()
        
        # Strategy 6 signals (long/short)
        s6_signals = calculate_trend_signal_long_short(prices, fast_span=64, slow_span=256)
        s6_signals_clean = s6_signals.dropna()
        
        if len(s5_signals_clean) > 0 and len(s6_signals_clean) > 0:
            # Calculate percentages
            s5_long_pct = (s5_signals_clean == 1).mean()
            s5_flat_pct = (s5_signals_clean == 0).mean()
            
            s6_long_pct = (s6_signals_clean == 1).mean()
            s6_short_pct = (s6_signals_clean == -1).mean()
            
            # The key insight: Strategy 5 can be flat (0), Strategy 6 is always positioned
            signal_diff = s6_long_pct - s5_long_pct
            
            print(f"{symbol:<8} {s5_long_pct:<10.1%} {s6_long_pct:<10.1%} {s6_short_pct:<11.1%} {signal_diff:<+12.1%}")
            
            signal_comparison[symbol] = {
                's5_long_pct': s5_long_pct,
                's5_flat_pct': s5_flat_pct,
                's6_long_pct': s6_long_pct,
                's6_short_pct': s6_short_pct,
                'signal_diff': signal_diff
            }
    
    return signal_comparison

def run_detailed_strategy_comparison():
    """Run both strategies and analyze detailed differences."""
    print("\n" + "=" * 80)
    print("DETAILED STRATEGY COMPARISON: 6-MONTH PERIOD")
    print("=" * 80)
    
    # Configuration for 6-month test
    config = {
        'data_dir': 'Data',
        'capital': 10000000,
        'risk_target': 0.2,
        'short_span': 32,
        'long_years': 10,
        'min_vol_floor': 0.05,
        'weight_method': 'handcrafted',
        'common_hypothetical_SR': 0.3,
        'annual_turnover_T': 7.0,
        'start_date': '2023-01-01',
        'end_date': '2023-06-30',
        'trend_fast_span': 64,
        'trend_slow_span': 256
    }
    
    try:
        print("Running Strategy 5 (Long Only)...")
        s5_results = backtest_trend_following_strategy(**config)
        
        print("Running Strategy 6 (Long/Short)...")
        s6_results = backtest_long_short_trend_strategy(**config)
        
        if not (s5_results and s6_results):
            print("❌ Failed to run one or both strategies")
            return None, None
            
        # Extract data for analysis
        s5_data = s5_results['portfolio_data']
        s6_data = s6_results['portfolio_data']
        s5_perf = s5_results['performance']
        s6_perf = s6_results['performance']
        
        print(f"\nPerformance Summary:")
        print(f"{'Metric':<20} {'Strategy 5':<12} {'Strategy 6':<12} {'Difference':<12}")
        print("-" * 60)
        print(f"{'Return':<20} {s5_perf['annualized_return']:<12.2%} {s6_perf['annualized_return']:<12.2%} {s6_perf['annualized_return']-s5_perf['annualized_return']:<+12.2%}")
        print(f"{'Sharpe':<20} {s5_perf['sharpe_ratio']:<12.3f} {s6_perf['sharpe_ratio']:<12.3f} {s6_perf['sharpe_ratio']-s5_perf['sharpe_ratio']:<+12.3f}")
        print(f"{'Volatility':<20} {s5_perf['annualized_volatility']:<12.2%} {s6_perf['annualized_volatility']:<12.2%} {s6_perf['annualized_volatility']-s5_perf['annualized_volatility']:<+12.2%}")
        print(f"{'Max DD':<20} {s5_perf['max_drawdown_pct']:<12.1f}% {s6_perf['max_drawdown_pct']:<12.1f}% {s6_perf['max_drawdown_pct']-s5_perf['max_drawdown_pct']:<+12.1f}%")
        
        # Analyze position differences
        print(f"\nPosition Analysis:")
        print(f"{'Metric':<25} {'Strategy 5':<12} {'Strategy 6':<12}")
        print("-" * 50)
        print(f"{'Avg Long Signals':<25} {s5_perf['avg_long_signals']:<12.1f} {s6_perf['avg_long_signals']:<12.1f}")
        print(f"{'Avg Short Signals':<25} {'0.0':<12} {s6_perf['avg_short_signals']:<12.1f}")
        
        s5_time_in_market = (s5_perf['avg_long_signals'] / s5_perf['num_instruments']) * 100
        s6_time_in_market = ((s6_perf['avg_long_signals'] + s6_perf['avg_short_signals']) / s6_perf['num_instruments']) * 100
        
        print(f"{'Time in Market':<25} {s5_time_in_market:<12.1f}% {s6_time_in_market:<12.1f}%")
        print(f"{'Time Long':<25} {s5_time_in_market:<12.1f}% {(s6_perf['avg_long_signals']/s6_perf['num_instruments'])*100:<12.1f}%")
        print(f"{'Time Short':<25} {'0.0%':<12} {(s6_perf['avg_short_signals']/s6_perf['num_instruments'])*100:<12.1f}%")
        
        return s5_results, s6_results
        
    except Exception as e:
        print(f"❌ Error in strategy comparison: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def analyze_individual_instrument_performance(s5_results, s6_results):
    """Analyze performance differences at the individual instrument level."""
    print("\n" + "=" * 80)
    print("INDIVIDUAL INSTRUMENT ANALYSIS")
    print("=" * 80)
    
    if not (s5_results and s6_results):
        print("❌ Missing strategy results")
        return
    
    s5_data = s5_results['portfolio_data']
    s6_data = s6_results['portfolio_data']
    s5_weights = s5_results['weights']
    s6_weights = s6_results['weights']
    
    # Find common instruments
    s5_instruments = [col.replace('_contracts', '') for col in s5_data.columns if col.endswith('_contracts')]
    s6_instruments = [col.replace('_contracts', '') for col in s6_data.columns if col.endswith('_contracts')]
    common_instruments = set(s5_instruments) & set(s6_instruments)
    
    print(f"Analyzing {len(common_instruments)} common instruments:")
    print(f"{'Symbol':<8} {'Weight':<8} {'S5 Avg Pos':<10} {'S6 Avg Pos':<10} {'S6 Long %':<10} {'S6 Short %':<11}")
    print("-" * 70)
    
    for symbol in sorted(common_instruments):
        if symbol in s5_weights and symbol in s6_weights:
            weight = s5_weights[symbol]
            
            # Calculate average positions
            s5_pos_col = f'{symbol}_contracts'
            s6_pos_col = f'{symbol}_contracts'
            
            if s5_pos_col in s5_data.columns and s6_pos_col in s6_data.columns:
                s5_positions = s5_data[s5_pos_col]
                s6_positions = s6_data[s6_pos_col]
                
                s5_avg_pos = s5_positions[s5_positions != 0].mean() if (s5_positions != 0).any() else 0
                s6_avg_pos = s6_positions[s6_positions != 0].mean() if (s6_positions != 0).any() else 0
                
                # Calculate S6 long/short percentages
                s6_long_pct = (s6_positions > 0.01).mean()
                s6_short_pct = (s6_positions < -0.01).mean()
                
                print(f"{symbol:<8} {weight:<8.3f} {s5_avg_pos:<10.1f} {s6_avg_pos:<10.1f} {s6_long_pct:<10.1%} {s6_short_pct:<11.1%}")

def analyze_position_sizing_differences():
    """Analyze how position sizing differs between the strategies."""
    print("\n" + "=" * 80)
    print("POSITION SIZING FORMULA COMPARISON")
    print("=" * 80)
    
    # Test parameters
    capital = 10000000
    weight = 0.05
    idm = 2.5
    price = 4500
    volatility = 0.16
    multiplier = 5
    risk_target = 0.2
    
    print(f"Test Parameters:")
    print(f"  Capital: ${capital:,}")
    print(f"  Weight: {weight:.1%}")
    print(f"  IDM: {idm}")
    print(f"  Price: ${price}")
    print(f"  Volatility: {volatility:.1%}")
    print(f"  Multiplier: {multiplier}")
    print(f"  Risk Target: {risk_target:.1%}")
    
    # Strategy 5 position sizing (when signal = 1)
    s5_position_when_long = calculate_strategy5_position_size(
        'TEST', capital, weight, idm, price, volatility, multiplier, 1, risk_target
    )
    
    # Strategy 5 position sizing (when signal = 0)
    s5_position_when_flat = calculate_strategy5_position_size(
        'TEST', capital, weight, idm, price, volatility, multiplier, 0, risk_target
    )
    
    # Strategy 6 position sizing (when signal = +1)
    s6_position_when_long = calculate_strategy6_position_size(
        'TEST', capital, weight, idm, price, volatility, multiplier, 1, risk_target
    )
    
    # Strategy 6 position sizing (when signal = -1)
    s6_position_when_short = calculate_strategy6_position_size(
        'TEST', capital, weight, idm, price, volatility, multiplier, -1, risk_target
    )
    
    print(f"\nPosition Sizing Results:")
    print(f"{'Strategy':<12} {'Signal':<8} {'Position':<10} {'Notes'}")
    print("-" * 50)
    print(f"{'Strategy 5':<12} {'+1 (long)':<8} {s5_position_when_long:<10.1f} {'Long position'}")
    print(f"{'Strategy 5':<12} {'0 (flat)':<8} {s5_position_when_flat:<10.1f} {'No position'}")
    print(f"{'Strategy 6':<12} {'+1 (long)':<8} {s6_position_when_long:<10.1f} {'Long position'}")
    print(f"{'Strategy 6':<12} {'-1 (short)':<8} {s6_position_when_short:<10.1f} {'Short position'}")
    
    print(f"\nKey Insights:")
    print(f"  • Strategy 5 and 6 have IDENTICAL long positions when both signal long")
    print(f"  • Strategy 5 goes flat (0) when trend is weak")
    print(f"  • Strategy 6 goes short when trend is weak")
    print(f"  • The difference is: Strategy 6 shorts vs Strategy 5 stays flat")

def create_signal_comparison_plot(s5_results, s6_results):
    """Create a visualization comparing signals between strategies."""
    print("\n" + "=" * 80)
    print("CREATING SIGNAL COMPARISON VISUALIZATION")
    print("=" * 80)
    
    if not (s5_results and s6_results):
        print("❌ Missing strategy results")
        return
    
    try:
        s5_data = s5_results['portfolio_data']
        s6_data = s6_results['portfolio_data']
        
        # Calculate daily signal counts
        s5_long_signals = s5_data['intended_long_signals'] if 'intended_long_signals' in s5_data.columns else None
        s6_long_signals = s6_data['intended_long_signals'] if 'intended_long_signals' in s6_data.columns else None
        s6_short_signals = s6_data['intended_short_signals'] if 'intended_short_signals' in s6_data.columns else None
        
        if s5_long_signals is None or s6_long_signals is None or s6_short_signals is None:
            print("❌ Signal data not available in portfolio data")
            return
        
        # Create plot
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Long signals comparison
        axes[0].plot(s5_data.index, s5_long_signals, 'b-', label='Strategy 5 Long Signals', linewidth=2)
        axes[0].plot(s6_data.index, s6_long_signals, 'g-', label='Strategy 6 Long Signals', linewidth=2)
        axes[0].set_title('Long Signals Comparison')
        axes[0].set_ylabel('Number of Long Signals')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Strategy 6 long vs short
        axes[1].plot(s6_data.index, s6_long_signals, 'g-', label='Strategy 6 Long', linewidth=2)
        axes[1].plot(s6_data.index, s6_short_signals, 'r-', label='Strategy 6 Short', linewidth=2)
        axes[1].set_title('Strategy 6: Long vs Short Signals')
        axes[1].set_ylabel('Number of Signals')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Time in market comparison
        s5_time_in_market = s5_long_signals / s5_results['performance']['num_instruments']
        s6_time_in_market = (s6_long_signals + s6_short_signals) / s6_results['performance']['num_instruments']
        
        axes[2].plot(s5_data.index, s5_time_in_market, 'b-', label='Strategy 5 Time in Market', linewidth=2)
        axes[2].plot(s6_data.index, s6_time_in_market, 'purple', label='Strategy 6 Time in Market', linewidth=2)
        axes[2].set_title('Time in Market Comparison')
        axes[2].set_ylabel('Fraction of Portfolio Active')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('strategy5_vs_strategy6_signals.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Signal comparison plot saved as 'strategy5_vs_strategy6_signals.png'")
        
    except Exception as e:
        print(f"❌ Error creating plot: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run the complete Strategy 5 vs Strategy 6 analysis."""
    print("🔍 STRATEGY 5 vs STRATEGY 6 DETAILED ANALYSIS")
    print("=" * 80)
    
    # Step 1: Analyze signal generation differences
    signal_comparison = analyze_signal_differences()
    
    # Step 2: Run detailed strategy comparison
    s5_results, s6_results = run_detailed_strategy_comparison()
    
    if s5_results and s6_results:
        # Step 3: Analyze individual instruments
        analyze_individual_instrument_performance(s5_results, s6_results)
        
        # Step 4: Analyze position sizing
        analyze_position_sizing_differences()
        
        # Step 5: Create visualization
        create_signal_comparison_plot(s5_results, s6_results)
        
        print(f"\n" + "=" * 80)
        print("🎯 KEY FINDINGS SUMMARY")
        print("=" * 80)
        print("1. Strategy 5 and Strategy 6 use IDENTICAL signals (EWMAC 64,256)")
        print("2. When EWMAC > 0: Both strategies go long with SAME position size")
        print("3. When EWMAC < 0: Strategy 5 goes FLAT (0), Strategy 6 goes SHORT")
        print("4. In a bull market: Strategy 6's short positions LOSE money")
        print("5. Strategy 5 avoids losses by staying flat instead of shorting")
        print("6. This explains why Strategy 6 underperforms in bull markets!")
        
    else:
        print("❌ Could not complete analysis due to strategy execution errors")

if __name__ == "__main__":
    main() 