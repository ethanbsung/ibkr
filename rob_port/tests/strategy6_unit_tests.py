#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Strategy 6 (Long/Short Trend Following)

This module tests all components of Strategy 6 to identify performance issues.
"""

import sys
sys.path.append('rob_port')
from chapter6 import *
from chapter5 import *
from chapter4 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def test_trend_signal_calculation():
    """Test that trend signals are calculated correctly."""
    print("=" * 60)
    print("TEST 1: TREND SIGNAL CALCULATION")
    print("=" * 60)
    
    # Create test price series with known trend
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    
    # Uptrend: prices going from 100 to 200
    uptrend_prices = pd.Series(np.linspace(100, 200, 300), index=dates)
    
    # Downtrend: prices going from 200 to 100  
    downtrend_prices = pd.Series(np.linspace(200, 100, 300), index=dates)
    
    # Sideways: oscillating around 150
    sideways_prices = pd.Series(150 + 10 * np.sin(np.linspace(0, 10*np.pi, 300)), index=dates)
    
    # Test uptrend
    uptrend_signals = calculate_trend_signal_long_short(uptrend_prices, fast_span=64, slow_span=256)
    uptrend_signals_clean = uptrend_signals.dropna()
    
    # Test downtrend
    downtrend_signals = calculate_trend_signal_long_short(downtrend_prices, fast_span=64, slow_span=256)
    downtrend_signals_clean = downtrend_signals.dropna()
    
    # Test sideways
    sideways_signals = calculate_trend_signal_long_short(sideways_prices, fast_span=64, slow_span=256)
    sideways_signals_clean = sideways_signals.dropna()
    
    print(f"Uptrend Analysis:")
    print(f"  Total signals: {len(uptrend_signals_clean)}")
    print(f"  Long signals: {(uptrend_signals_clean == 1).sum()}")
    print(f"  Short signals: {(uptrend_signals_clean == -1).sum()}")
    print(f"  % Long: {(uptrend_signals_clean == 1).mean():.1%}")
    print(f"  Expected: Should be mostly long (>70%)")
    print(f"  ✅ PASS" if (uptrend_signals_clean == 1).mean() > 0.7 else "❌ FAIL")
    
    print(f"\nDowntrend Analysis:")
    print(f"  Total signals: {len(downtrend_signals_clean)}")
    print(f"  Long signals: {(downtrend_signals_clean == 1).sum()}")
    print(f"  Short signals: {(downtrend_signals_clean == -1).sum()}")
    print(f"  % Short: {(downtrend_signals_clean == -1).mean():.1%}")
    print(f"  Expected: Should be mostly short (>70%)")
    print(f"  ✅ PASS" if (downtrend_signals_clean == -1).mean() > 0.7 else "❌ FAIL")
    
    print(f"\nSideways Analysis:")
    print(f"  Total signals: {len(sideways_signals_clean)}")
    print(f"  Long signals: {(sideways_signals_clean == 1).sum()}")
    print(f"  Short signals: {(sideways_signals_clean == -1).sum()}")
    print(f"  % Long: {(sideways_signals_clean == 1).mean():.1%}")
    print(f"  % Short: {(sideways_signals_clean == -1).mean():.1%}")
    print(f"  Expected: Should be roughly balanced (40-60% each)")
    long_pct = (sideways_signals_clean == 1).mean()
    print(f"  ✅ PASS" if 0.4 <= long_pct <= 0.6 else "❌ FAIL")

def test_position_sizing():
    """Test that position sizing works correctly for long and short signals."""
    print("\n" + "=" * 60)
    print("TEST 2: POSITION SIZING")
    print("=" * 60)
    
    # Test parameters
    symbol = "TEST"
    capital = 1000000
    weight = 0.1
    idm = 2.5
    price = 100
    volatility = 0.2
    multiplier = 1
    risk_target = 0.2
    
    # Test long signal
    long_position = calculate_strategy6_position_size(
        symbol, capital, weight, idm, price, volatility, 
        multiplier, trend_signal=1, risk_target=risk_target
    )
    
    # Test short signal
    short_position = calculate_strategy6_position_size(
        symbol, capital, weight, idm, price, volatility, 
        multiplier, trend_signal=-1, risk_target=risk_target
    )
    
    # Test neutral/invalid signal
    neutral_position = calculate_strategy6_position_size(
        symbol, capital, weight, idm, price, volatility, 
        multiplier, trend_signal=0, risk_target=risk_target
    )
    
    print(f"Position Sizing Test:")
    print(f"  Long signal (+1): {long_position:.2f} contracts")
    print(f"  Short signal (-1): {short_position:.2f} contracts")
    print(f"  Neutral signal (0): {neutral_position:.2f} contracts")
    
    # Expected: long and short should be equal magnitude but opposite sign
    expected_magnitude = abs(long_position)
    print(f"\nExpected Results:")
    print(f"  Long position should be positive: ✅ PASS" if long_position > 0 else "❌ FAIL")
    print(f"  Short position should be negative: ✅ PASS" if short_position < 0 else "❌ FAIL")
    print(f"  Magnitudes should be equal: ✅ PASS" if abs(abs(long_position) - abs(short_position)) < 0.01 else "❌ FAIL")
    print(f"  Neutral should be zero: ✅ PASS" if abs(neutral_position) < 0.01 else "❌ FAIL")
    
    # Test the actual formula
    expected_position = (capital * idm * weight * risk_target) / (multiplier * price * volatility)
    print(f"\nFormula Verification:")
    print(f"  Expected magnitude: {expected_position:.2f}")
    print(f"  Actual long: {long_position:.2f}")
    print(f"  Actual short: {short_position:.2f}")
    print(f"  Formula correct: ✅ PASS" if abs(abs(long_position) - expected_position) < 0.01 else "❌ FAIL")

def test_pnl_calculation():
    """Test P&L calculation for long and short positions."""
    print("\n" + "=" * 60)
    print("TEST 3: P&L CALCULATION")
    print("=" * 60)
    
    # Test scenarios
    multiplier = 1
    contracts_long = 10  # Long 10 contracts
    contracts_short = -10  # Short 10 contracts
    
    # Price movements
    price_start = 100
    price_up = 110  # +10 price move
    price_down = 90  # -10 price move
    
    # Long position P&L tests
    long_pnl_up = contracts_long * multiplier * (price_up - price_start)
    long_pnl_down = contracts_long * multiplier * (price_down - price_start)
    
    # Short position P&L tests  
    short_pnl_up = contracts_short * multiplier * (price_up - price_start)
    short_pnl_down = contracts_short * multiplier * (price_down - price_start)
    
    print(f"P&L Calculation Test:")
    print(f"  Position: {contracts_long} contracts (long)")
    print(f"  Price up ({price_start} → {price_up}): P&L = {long_pnl_up:.2f}")
    print(f"  Price down ({price_start} → {price_down}): P&L = {long_pnl_down:.2f}")
    print(f"  Expected: Long profits when price rises, loses when price falls")
    print(f"  ✅ PASS" if long_pnl_up > 0 and long_pnl_down < 0 else "❌ FAIL")
    
    print(f"\n  Position: {contracts_short} contracts (short)")
    print(f"  Price up ({price_start} → {price_up}): P&L = {short_pnl_up:.2f}")
    print(f"  Price down ({price_start} → {price_down}): P&L = {short_pnl_down:.2f}")
    print(f"  Expected: Short loses when price rises, profits when price falls")
    print(f"  ✅ PASS" if short_pnl_up < 0 and short_pnl_down > 0 else "❌ FAIL")

def test_real_data_signals():
    """Test trend signals on real market data."""
    print("\n" + "=" * 60)
    print("TEST 4: REAL DATA SIGNAL ANALYSIS")
    print("=" * 60)
    
    # Load a sample instrument
    try:
        raw_data = load_all_instrument_data('Data')
        if not raw_data:
            print("❌ FAIL: No data loaded")
            return
            
        # Use MES as test instrument
        if 'MES' not in raw_data:
            print("❌ FAIL: MES data not found")
            return
            
        mes_data = raw_data['MES'].copy()
        if 'Last' not in mes_data.columns:
            print("❌ FAIL: Last column not found")
            return
            
        # Calculate signals
        signals = calculate_trend_signal_long_short(mes_data['Last'], fast_span=64, slow_span=256)
        signals_clean = signals.dropna()
        
        if len(signals_clean) == 0:
            print("❌ FAIL: No valid signals generated")
            return
            
        print(f"MES Signal Analysis:")
        print(f"  Total periods: {len(mes_data)}")
        print(f"  Valid signals: {len(signals_clean)}")
        print(f"  Long signals: {(signals_clean == 1).sum()}")
        print(f"  Short signals: {(signals_clean == -1).sum()}")
        print(f"  % Long: {(signals_clean == 1).mean():.1%}")
        print(f"  % Short: {(signals_clean == -1).mean():.1%}")
        
        # Check for reasonable distribution
        long_pct = (signals_clean == 1).mean()
        if 0.3 <= long_pct <= 0.7:
            print(f"  Signal distribution: ✅ PASS (reasonable balance)")
        else:
            print(f"  Signal distribution: ⚠️  WARNING (heavily skewed)")
            
        # Check recent signals (last 100 days)
        recent_signals = signals_clean.tail(100)
        recent_long_pct = (recent_signals == 1).mean()
        print(f"\nRecent 100 days:")
        print(f"  % Long: {recent_long_pct:.1%}")
        print(f"  % Short: {(1-recent_long_pct):.1%}")
        
    except Exception as e:
        print(f"❌ FAIL: Error in real data test: {e}")

def test_strategy5_vs_strategy6_comparison():
    """Compare Strategy 5 and 6 on a small sample to identify differences."""
    print("\n" + "=" * 60)
    print("TEST 5: STRATEGY 5 vs 6 COMPARISON")
    print("=" * 60)
    
    # Run both strategies on a limited dataset for comparison
    config = {
        'data_dir': 'Data',
        'capital': 10000000,  # Smaller for testing
        'risk_target': 0.2,
        'short_span': 32,
        'long_years': 10,
        'min_vol_floor': 0.05,
        'weight_method': 'handcrafted',
        'common_hypothetical_SR': 0.3,
        'annual_turnover_T': 7.0,
        'start_date': '2023-01-01',
        'end_date': '2023-06-30',  # Just 6 months for testing
        'trend_fast_span': 64,
        'trend_slow_span': 256
    }
    
    try:
        print("Running Strategy 5 (Long Only)...")
        s5_results = backtest_trend_following_strategy(**config)
        
        print("Running Strategy 6 (Long/Short)...")
        s6_results = backtest_long_short_trend_strategy(**config)
        
        if s5_results and s6_results:
            s5_perf = s5_results['performance']
            s6_perf = s6_results['performance']
            
            print(f"\nPerformance Comparison (6 months):")
            print(f"  Strategy 5 (Long Only):")
            print(f"    Return: {s5_perf['annualized_return']:.2%}")
            print(f"    Sharpe: {s5_perf['sharpe_ratio']:.3f}")
            print(f"    Max DD: {s5_perf['max_drawdown_pct']:.1f}%")
            print(f"    Avg Long Signals: {s5_perf['avg_long_signals']:.1f}")
            
            print(f"  Strategy 6 (Long/Short):")
            print(f"    Return: {s6_perf['annualized_return']:.2%}")
            print(f"    Sharpe: {s6_perf['sharpe_ratio']:.3f}")
            print(f"    Max DD: {s6_perf['max_drawdown_pct']:.1f}%")
            print(f"    Avg Long Signals: {s6_perf['avg_long_signals']:.1f}")
            print(f"    Avg Short Signals: {s6_perf['avg_short_signals']:.1f}")
            
            # Analyze the differences
            return_diff = s6_perf['annualized_return'] - s5_perf['annualized_return']
            sharpe_diff = s6_perf['sharpe_ratio'] - s5_perf['sharpe_ratio']
            
            print(f"\n  Differences (Strategy 6 - Strategy 5):")
            print(f"    Return difference: {return_diff:+.2%}")
            print(f"    Sharpe difference: {sharpe_diff:+.3f}")
            
            if sharpe_diff < -0.1:
                print(f"    ⚠️  WARNING: Strategy 6 significantly underperforms Strategy 5")
            else:
                print(f"    ✅ PASS: Reasonable performance difference")
                
            # Check position distributions
            s5_data = s5_results['portfolio_data']
            s6_data = s6_results['portfolio_data']
            
            print(f"\n  Portfolio Analysis:")
            print(f"    Strategy 5 avg return: {s5_data['portfolio_return'].mean():.6f}")
            print(f"    Strategy 6 avg return: {s6_data['portfolio_return'].mean():.6f}")
            print(f"    Strategy 5 return std: {s5_data['portfolio_return'].std():.6f}")
            print(f"    Strategy 6 return std: {s6_data['portfolio_return'].std():.6f}")
            
    except Exception as e:
        print(f"❌ FAIL: Error in comparison test: {e}")
        import traceback
        traceback.print_exc()

def test_single_instrument_deep_dive():
    """Deep dive analysis of a single instrument to understand behavior."""
    print("\n" + "=" * 60)
    print("TEST 6: SINGLE INSTRUMENT DEEP DIVE")
    print("=" * 60)
    
    try:
        # Load MES data
        raw_data = load_all_instrument_data('Data')
        if 'MES' not in raw_data:
            print("❌ FAIL: MES data not found")
            return
            
        mes_data = raw_data['MES'].copy()
        
        # Add required columns
        mes_data['daily_price_change_pct'] = mes_data['Last'].pct_change()
        
        # Calculate volatility
        returns = mes_data['daily_price_change_pct'].dropna()
        blended_vol = calculate_blended_volatility(returns, short_span=32, long_years=10, min_vol_floor=0.05)
        mes_data['vol_forecast'] = blended_vol.shift(1).reindex(mes_data.index).ffill().fillna(0.05)
        
        # Calculate trend signals
        trend_signals = calculate_trend_signal_long_short(mes_data['Last'], fast_span=64, slow_span=256)
        mes_data['trend_signal'] = trend_signals.shift(1).reindex(mes_data.index).ffill()
        
        # Remove NaNs
        mes_data = mes_data.dropna(subset=['Last', 'vol_forecast', 'daily_price_change_pct', 'trend_signal'])
        
        if len(mes_data) < 100:
            print("❌ FAIL: Insufficient data after cleaning")
            return
            
        # Analyze recent period
        recent_data = mes_data.tail(100)
        
        print(f"MES Deep Dive Analysis (last 100 days):")
        print(f"  Data points: {len(recent_data)}")
        print(f"  Price range: {recent_data['Last'].min():.2f} - {recent_data['Last'].max():.2f}")
        print(f"  Avg volatility: {recent_data['vol_forecast'].mean():.4f}")
        print(f"  Volatility at floor: {(recent_data['vol_forecast'] <= 0.051).sum()}")
        
        # Signal analysis
        signals = recent_data['trend_signal']
        print(f"\n  Signal Distribution:")
        print(f"    Long signals: {(signals == 1).sum()}")
        print(f"    Short signals: {(signals == -1).sum()}")
        print(f"    Other signals: {((signals != 1) & (signals != -1)).sum()}")
        
        # Simulate position sizing
        capital = 1000000
        weight = 0.05  # 5% weight
        idm = 2.5
        risk_target = 0.2
        multiplier = 5  # MES multiplier
        
        positions = []
        pnls = []
        
        for i in range(1, min(50, len(recent_data))):  # Test 50 days
            row = recent_data.iloc[i]
            prev_row = recent_data.iloc[i-1]
            
            position = calculate_strategy6_position_size(
                'MES', capital, weight, idm, prev_row['Last'], 
                row['vol_forecast'], multiplier, row['trend_signal'], risk_target
            )
            
            pnl = position * multiplier * (row['Last'] - prev_row['Last'])
            
            positions.append(position)
            pnls.append(pnl)
        
        print(f"\n  Position Analysis (50 days):")
        print(f"    Avg position: {np.mean(positions):.2f}")
        print(f"    Position range: {min(positions):.2f} to {max(positions):.2f}")
        print(f"    Long positions: {sum(1 for p in positions if p > 0.01)}")
        print(f"    Short positions: {sum(1 for p in positions if p < -0.01)}")
        print(f"    Zero positions: {sum(1 for p in positions if abs(p) <= 0.01)}")
        
        print(f"\n  P&L Analysis (50 days):")
        print(f"    Total P&L: ${sum(pnls):,.2f}")
        print(f"    Avg daily P&L: ${np.mean(pnls):,.2f}")
        print(f"    P&L std: ${np.std(pnls):,.2f}")
        print(f"    Positive days: {sum(1 for p in pnls if p > 0)}")
        print(f"    Negative days: {sum(1 for p in pnls if p < 0)}")
        
    except Exception as e:
        print(f"❌ FAIL: Error in deep dive: {e}")
        import traceback
        traceback.print_exc()

def run_all_tests():
    """Run all Strategy 6 unit tests."""
    print("🔍 STRATEGY 6 COMPREHENSIVE UNIT TESTS")
    print("=" * 80)
    
    test_trend_signal_calculation()
    test_position_sizing()
    test_pnl_calculation()
    test_real_data_signals()
    test_strategy5_vs_strategy6_comparison()
    test_single_instrument_deep_dive()
    
    print("\n" + "=" * 80)
    print("🏁 ALL TESTS COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    run_all_tests() 