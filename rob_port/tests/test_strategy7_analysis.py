#!/usr/bin/env python3
"""
Strategy 7 Analysis: Comprehensive comparison of all strategies
showing the evolution from basic portfolio to sophisticated forecast-based trading.
"""

import sys
import os
sys.path.append('rob_port')

from chapter7 import *
from chapter6 import *
from chapter5 import *
from chapter4 import *
import pandas as pd
import numpy as np

def analyze_strategy_evolution():
    """
    Analyze the evolution from Strategy 4 to Strategy 7.
    """
    print("=" * 80)
    print("STRATEGY EVOLUTION ANALYSIS: FROM BASIC PORTFOLIO TO FORECAST TRADING")
    print("=" * 80)
    
    # Test parameters
    capital = 50000000
    risk_target = 0.2
    
    strategies = {}
    
    try:
        print("\n🔄 Running Strategy 4: Basic Multi-Instrument Portfolio...")
        strategies['Strategy 4'] = backtest_multi_instrument_strategy(
            data_dir='Data', capital=capital, risk_target=risk_target, weight_method='handcrafted'
        )
        
        print("\n🔄 Running Strategy 5: Long-Only Trend Following...")
        strategies['Strategy 5'] = backtest_trend_following_strategy(
            data_dir='Data', capital=capital, risk_target=risk_target, weight_method='handcrafted'
        )
        
        print("\n🔄 Running Strategy 6: Long/Short Trend Following...")
        strategies['Strategy 6'] = backtest_long_short_trend_strategy(
            data_dir='Data', capital=capital, risk_target=risk_target, weight_method='handcrafted'
        )
        
        print("\n🔄 Running Strategy 7: Forecast-Based Trend Following...")
        strategies['Strategy 7'] = backtest_forecast_trend_strategy(
            data_dir='Data', capital=capital, risk_target=risk_target, weight_method='handcrafted'
        )
        
        # Create comprehensive comparison
        print("\n" + "=" * 80)
        print("COMPREHENSIVE STRATEGY COMPARISON")
        print("=" * 80)
        
        # Performance comparison table
        print(f"\n{'Strategy':<12} {'Return':<8} {'Vol':<8} {'Sharpe':<8} {'MaxDD':<8} {'Skew':<8} {'Key Feature':<25}")
        print("-" * 95)
        
        for name, results in strategies.items():
            if results:
                perf = results['performance']
                feature_map = {
                    'Strategy 4': 'Always Long (100% exposure)',
                    'Strategy 5': 'Long/Flat (38% exposure)',
                    'Strategy 6': 'Long/Short (69% exposure)',
                    'Strategy 7': 'Variable Forecast Scaling'
                }
                
                print(f"{name:<12} {perf['annualized_return']:<8.2%} {perf['annualized_volatility']:<8.2%} "
                      f"{perf['sharpe_ratio']:<8.3f} {perf['max_drawdown_pct']:<8.1f}% "
                      f"{perf['skewness']:<8.3f} {feature_map[name]:<25}")
        
        # Strategy 7 specific analysis
        if 'Strategy 7' in strategies and strategies['Strategy 7']:
            s7_results = strategies['Strategy 7']
            s7_perf = s7_results['performance']
            
            print(f"\n" + "=" * 60)
            print("STRATEGY 7 FORECAST ANALYSIS")
            print("=" * 60)
            
            print(f"\n📊 Forecast Characteristics:")
            print(f"  Average Forecast: {s7_perf['avg_forecast']:.2f}")
            print(f"  Average Absolute Forecast: {s7_perf['avg_abs_forecast']:.2f}")
            print(f"  Forecast Scalar: {s7_results['config']['forecast_scalar']}")
            print(f"  Forecast Cap: ±{s7_results['config']['forecast_cap']}")
            
            # Analyze forecast distribution
            portfolio_data = s7_results['portfolio_data']
            forecast_cols = [col for col in portfolio_data.columns if col.endswith('_forecast')]
            
            if forecast_cols:
                all_forecasts = []
                for col in forecast_cols:
                    forecasts = portfolio_data[col].dropna()
                    all_forecasts.extend(forecasts.tolist())
                
                all_forecasts = pd.Series(all_forecasts)
                
                print(f"\n📈 Forecast Distribution:")
                print(f"  Min Forecast: {all_forecasts.min():.2f}")
                print(f"  Max Forecast: {all_forecasts.max():.2f}")
                print(f"  Std Forecast: {all_forecasts.std():.2f}")
                print(f"  % Positive: {(all_forecasts > 0).mean():.1%}")
                print(f"  % Negative: {(all_forecasts < 0).mean():.1%}")
                print(f"  % Near Zero (±1): {(all_forecasts.abs() <= 1).mean():.1%}")
                print(f"  % Strong (±10): {(all_forecasts.abs() >= 10).mean():.1%}")
                print(f"  % Capped (±20): {(all_forecasts.abs() >= 19.9).mean():.1%}")
        
        # Evolution benefits analysis
        print(f"\n" + "=" * 60)
        print("STRATEGY EVOLUTION BENEFITS")
        print("=" * 60)
        
        if all(name in strategies and strategies[name] for name in ['Strategy 4', 'Strategy 5', 'Strategy 6', 'Strategy 7']):
            s4_perf = strategies['Strategy 4']['performance']
            s5_perf = strategies['Strategy 5']['performance']
            s6_perf = strategies['Strategy 6']['performance']
            s7_perf = strategies['Strategy 7']['performance']
            
            print(f"\n🎯 Strategy 5 vs Strategy 4 (Adding Trend Filter):")
            print(f"  Return Change: {s5_perf['annualized_return'] - s4_perf['annualized_return']:+.2%}")
            print(f"  Volatility Change: {s5_perf['annualized_volatility'] - s4_perf['annualized_volatility']:+.2%}")
            print(f"  Sharpe Change: {s5_perf['sharpe_ratio'] - s4_perf['sharpe_ratio']:+.3f}")
            print(f"  Max DD Change: {s5_perf['max_drawdown_pct'] - s4_perf['max_drawdown_pct']:+.1f}%")
            
            print(f"\n🎯 Strategy 6 vs Strategy 5 (Adding Short Positions):")
            print(f"  Return Change: {s6_perf['annualized_return'] - s5_perf['annualized_return']:+.2%}")
            print(f"  Volatility Change: {s6_perf['annualized_volatility'] - s5_perf['annualized_volatility']:+.2%}")
            print(f"  Sharpe Change: {s6_perf['sharpe_ratio'] - s5_perf['sharpe_ratio']:+.3f}")
            print(f"  Max DD Change: {s6_perf['max_drawdown_pct'] - s5_perf['max_drawdown_pct']:+.1f}%")
            
            print(f"\n🎯 Strategy 7 vs Strategy 6 (Adding Forecast Scaling):")
            print(f"  Return Change: {s7_perf['annualized_return'] - s6_perf['annualized_return']:+.2%}")
            print(f"  Volatility Change: {s7_perf['annualized_volatility'] - s6_perf['annualized_volatility']:+.2%}")
            print(f"  Sharpe Change: {s7_perf['sharpe_ratio'] - s6_perf['sharpe_ratio']:+.3f}")
            print(f"  Max DD Change: {s7_perf['max_drawdown_pct'] - s6_perf['max_drawdown_pct']:+.1f}%")
            
            print(f"\n🏆 Overall Evolution (Strategy 7 vs Strategy 4):")
            print(f"  Return Improvement: {s7_perf['annualized_return'] - s4_perf['annualized_return']:+.2%}")
            print(f"  Volatility Change: {s7_perf['annualized_volatility'] - s4_perf['annualized_volatility']:+.2%}")
            print(f"  Sharpe Improvement: {s7_perf['sharpe_ratio'] - s4_perf['sharpe_ratio']:+.3f}")
            print(f"  Max DD Change: {s7_perf['max_drawdown_pct'] - s4_perf['max_drawdown_pct']:+.1f}%")
        
        # Key insights
        print(f"\n" + "=" * 60)
        print("KEY INSIGHTS")
        print("=" * 60)
        
        print(f"\n💡 Strategy Evolution Insights:")
        print(f"  • Strategy 4: Basic diversified portfolio - always long")
        print(f"  • Strategy 5: Adds trend filtering - reduces exposure, improves risk-adjusted returns")
        print(f"  • Strategy 6: Adds short positions - increases market exposure, captures downtrends")
        print(f"  • Strategy 7: Adds forecast scaling - variable position sizing based on trend strength")
        
        print(f"\n🔍 Forecast Scaling Benefits:")
        print(f"  • Provides nuanced position sizing (not just binary long/short)")
        print(f"  • Scales positions based on trend conviction")
        print(f"  • Allows for gradual position adjustments")
        print(f"  • Caps extreme positions to manage risk")
        
        print(f"\n⚖️ Risk Management Evolution:")
        print(f"  • Strategy 4: Constant exposure, high drawdowns")
        print(f"  • Strategy 5: Reduced exposure, much lower drawdowns")
        print(f"  • Strategy 6: Full exposure but directional flexibility")
        print(f"  • Strategy 7: Variable exposure based on conviction")
        
        return strategies
        
    except Exception as e:
        print(f"Error in strategy analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_time_periods():
    """
    Analyze Strategy 7 performance across different time periods.
    """
    print("\n" + "=" * 80)
    print("STRATEGY 7 TIME PERIOD ANALYSIS")
    print("=" * 80)
    
    periods = [
        ('2000-2004', 'Dot-Com Crash Era'),
        ('2005-2009', 'Financial Crisis Era'),
        ('2010-2014', 'Recovery Era'),
        ('2015-2019', 'Low Volatility Era'),
        ('2020-2024', 'Pandemic Era')
    ]
    
    capital = 50000000
    risk_target = 0.2
    
    print(f"\n{'Period':<12} {'Description':<20} {'Return':<8} {'Vol':<8} {'Sharpe':<8} {'MaxDD':<8} {'AvgFcst':<8}")
    print("-" * 85)
    
    for period_range, description in periods:
        start_year, end_year = period_range.split('-')
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        
        try:
            results = backtest_forecast_trend_strategy(
                data_dir='Data',
                capital=capital,
                risk_target=risk_target,
                weight_method='handcrafted',
                start_date=start_date,
                end_date=end_date
            )
            
            if results:
                perf = results['performance']
                avg_forecast = perf.get('avg_forecast', 0)
                
                print(f"{period_range:<12} {description:<20} {perf['annualized_return']:<8.2%} "
                      f"{perf['annualized_volatility']:<8.2%} {perf['sharpe_ratio']:<8.3f} "
                      f"{perf['max_drawdown_pct']:<8.1f}% {avg_forecast:<8.2f}")
            
        except Exception as e:
            print(f"{period_range:<12} {description:<20} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8}")

def main():
    """
    Run comprehensive Strategy 7 analysis.
    """
    print("🚀 Starting Strategy 7 Comprehensive Analysis...")
    
    # Main evolution analysis
    strategies = analyze_strategy_evolution()
    
    # Time period analysis
    analyze_time_periods()
    
    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETE ✅")
    print("=" * 80)
    
    if strategies and 'Strategy 7' in strategies:
        s7_perf = strategies['Strategy 7']['performance']
        print(f"\n🏆 Strategy 7 Final Results:")
        print(f"  📈 Annualized Return: {s7_perf['annualized_return']:.2%}")
        print(f"  📊 Volatility: {s7_perf['annualized_volatility']:.2%}")
        print(f"  ⭐ Sharpe Ratio: {s7_perf['sharpe_ratio']:.3f}")
        print(f"  📉 Max Drawdown: {s7_perf['max_drawdown_pct']:.1f}%")
        print(f"  🎯 Average Forecast: {s7_perf['avg_forecast']:.2f}")
        
        print(f"\n💡 Key Achievement:")
        print(f"  Strategy 7 demonstrates sophisticated trend following with variable")
        print(f"  position sizing based on forecast strength, exactly as described in")
        print(f"  Robert Carver's book. The forecast scaling provides nuanced position")
        print(f"  management that adapts to trend conviction levels.")

if __name__ == "__main__":
    main() 