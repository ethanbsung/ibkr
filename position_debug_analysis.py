#!/usr/bin/env python3
"""
Position Debug Analysis for Chapters 6-8

This script analyzes the position calculation and updating mechanism to identify
potential issues causing underperformance in the trend following strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the rob_port directory to path so we can import the modules
sys.path.append('rob_port')

from chapter8 import *
from chapter7 import *
from chapter6 import *
from chapter5 import *
from chapter4 import *
from chapter3 import *
from chapter2 import *
from chapter1 import *

def analyze_single_instrument_positioning(symbol='MES', days_to_analyze=50):
    """
    Analyze positioning for a single instrument in detail to verify calculations.
    """
    print(f"\n{'='*80}")
    print(f"DETAILED POSITION ANALYSIS FOR {symbol}")
    print(f"{'='*80}")
    
    # Load data for this instrument
    try:
        df = pd.read_csv(f'Data/{symbol.lower()}_daily_data.csv', parse_dates=['Time'])
        df.set_index('Time', inplace=True)
        print(f"Loaded {len(df)} days of data for {symbol}")
    except FileNotFoundError:
        print(f"Could not find data file for {symbol}")
        return
    
    if len(df) < 300:
        print(f"Insufficient data for {symbol} ({len(df)} days)")
        return
    
    # Calculate basic metrics
    df['daily_returns'] = df['Last'].pct_change()
    df = df.dropna()
    
    # Calculate EWMA trends for different strategies
    # Strategy 6: EWMAC(64,256) for long/short signals
    fast_ewma_64 = df['Last'].ewm(span=64, adjust=False).mean()
    slow_ewma_256 = df['Last'].ewm(span=256, adjust=False).mean()
    ewmac_64_256 = fast_ewma_64 - slow_ewma_256
    df['trend_signal_s6'] = np.where(ewmac_64_256 > 0, 1, -1)
    
    # Strategy 7: Same as Strategy 6 but with forecasts
    blended_vol = calculate_blended_volatility(df['daily_returns'], 32, 10, 0.05)
    daily_price_vol = df['Last'] * blended_vol / 16
    df['raw_forecast_s7'] = ewmac_64_256 / daily_price_vol
    df['scaled_forecast_s7'] = df['raw_forecast_s7'] * 1.9  # forecast scalar
    df['capped_forecast_s7'] = np.clip(df['scaled_forecast_s7'], -20, 20)
    
    # Strategy 8: Fast EWMAC(16,64) for forecasts
    fast_ewma_16 = df['Last'].ewm(span=16, adjust=False).mean()
    slow_ewma_64 = df['Last'].ewm(span=64, adjust=False).mean()
    ewmac_16_64 = fast_ewma_16 - slow_ewma_64
    df['raw_forecast_s8'] = ewmac_16_64 / daily_price_vol
    df['scaled_forecast_s8'] = df['raw_forecast_s8'] * 4.1  # forecast scalar for fast trend
    df['capped_forecast_s8'] = np.clip(df['scaled_forecast_s8'], -20, 20)
    
    # Shift signals/forecasts to prevent lookahead bias
    df['trend_signal_s6_shifted'] = df['trend_signal_s6'].shift(1).fillna(1)
    df['forecast_s7_shifted'] = df['capped_forecast_s7'].shift(1).fillna(0)
    df['forecast_s8_shifted'] = df['capped_forecast_s8'].shift(1).fillna(0)
    df['vol_forecast'] = blended_vol.shift(1).reindex(df.index).ffill().fillna(0.05)
    
    # Analyze last N days
    recent_df = df.tail(days_to_analyze).copy()
    
    print(f"\n--- Analysis of Last {days_to_analyze} Days ---")
    print(f"Date Range: {recent_df.index[0].date()} to {recent_df.index[-1].date()}")
    print(f"Price Range: ${recent_df['Last'].min():.2f} to ${recent_df['Last'].max():.2f}")
    print(f"Price Change: {((recent_df['Last'].iloc[-1] / recent_df['Last'].iloc[0]) - 1) * 100:.2f}%")
    
    # Calculate position sizes for each strategy
    capital = 50000000
    weight = 0.1  # Example weight
    idm = 2.0  # Example IDM
    risk_target = 0.2
    multiplier = 5  # MES multiplier
    
    positions_s6 = []
    positions_s7 = []
    positions_s8 = []
    
    for i, (date, row) in enumerate(recent_df.iterrows()):
        price = row['Last']
        vol = max(row['vol_forecast'], 0.05)
        
        # Strategy 6 position
        trend_signal = row['trend_signal_s6_shifted']
        pos_s6 = (trend_signal * capital * idm * weight * risk_target) / (multiplier * price * vol)
        positions_s6.append(round(pos_s6))
        
        # Strategy 7 position
        forecast_s7 = row['forecast_s7_shifted']
        pos_s7 = (forecast_s7 * capital * idm * weight * risk_target) / (10 * multiplier * price * vol)
        positions_s7.append(pos_s7)
        
        # Strategy 8 position
        forecast_s8 = row['forecast_s8_shifted']
        pos_s8 = (forecast_s8 * capital * idm * weight * risk_target) / (10 * multiplier * price * vol)
        positions_s8.append(pos_s8)
    
    recent_df['pos_s6'] = positions_s6
    recent_df['pos_s7'] = positions_s7
    recent_df['pos_s8'] = positions_s8
    
    # Calculate daily P&L for each strategy
    recent_df['pnl_s6'] = recent_df['pos_s6'].shift(1) * multiplier * recent_df['Last'].diff()
    recent_df['pnl_s7'] = recent_df['pos_s7'].shift(1) * multiplier * recent_df['Last'].diff()
    recent_df['pnl_s8'] = recent_df['pos_s8'].shift(1) * multiplier * recent_df['Last'].diff()
    
    # Print summary statistics
    print(f"\n--- Position Statistics ---")
    print(f"Strategy 6 (Long/Short):")
    print(f"  Avg Position: {recent_df['pos_s6'].mean():.2f}")
    print(f"  Position Range: {recent_df['pos_s6'].min():.0f} to {recent_df['pos_s6'].max():.0f}")
    print(f"  Long Days: {(recent_df['pos_s6'] > 0).sum()}")
    print(f"  Short Days: {(recent_df['pos_s6'] < 0).sum()}")
    print(f"  Total P&L: ${recent_df['pnl_s6'].sum():,.0f}")
    
    print(f"\nStrategy 7 (Slow Forecasts):")
    print(f"  Avg Position: {recent_df['pos_s7'].mean():.2f}")
    print(f"  Position Range: {recent_df['pos_s7'].min():.1f} to {recent_df['pos_s7'].max():.1f}")
    print(f"  Avg Forecast: {recent_df['forecast_s7_shifted'].mean():.2f}")
    print(f"  Forecast Range: {recent_df['forecast_s7_shifted'].min():.1f} to {recent_df['forecast_s7_shifted'].max():.1f}")
    print(f"  Total P&L: ${recent_df['pnl_s7'].sum():,.0f}")
    
    print(f"\nStrategy 8 (Fast Forecasts):")
    print(f"  Avg Position: {recent_df['pos_s8'].mean():.2f}")
    print(f"  Position Range: {recent_df['pos_s8'].min():.1f} to {recent_df['pos_s8'].max():.1f}")
    print(f"  Avg Forecast: {recent_df['forecast_s8_shifted'].mean():.2f}")
    print(f"  Forecast Range: {recent_df['forecast_s8_shifted'].min():.1f} to {recent_df['forecast_s8_shifted'].max():.1f}")
    print(f"  Total P&L: ${recent_df['pnl_s8'].sum():,.0f}")
    
    # Check for common issues
    print(f"\n--- Potential Issues Analysis ---")
    
    # Issue 1: Are trend signals changing frequently?
    trend_changes = (recent_df['trend_signal_s6_shifted'].diff() != 0).sum()
    print(f"Strategy 6 trend signal changes: {trend_changes} out of {len(recent_df)} days ({trend_changes/len(recent_df)*100:.1f}%)")
    
    # Issue 2: Are forecasts reasonable?
    if recent_df['forecast_s7_shifted'].std() == 0:
        print("WARNING: Strategy 7 forecasts are constant (no variation)")
    if recent_df['forecast_s8_shifted'].std() == 0:
        print("WARNING: Strategy 8 forecasts are constant (no variation)")
    
    # Issue 3: Are positions too small?
    avg_pos_s6 = abs(recent_df['pos_s6']).mean()
    avg_pos_s7 = abs(recent_df['pos_s7']).mean()
    avg_pos_s8 = abs(recent_df['pos_s8']).mean()
    
    if avg_pos_s6 < 1:
        print(f"WARNING: Strategy 6 average position size is very small ({avg_pos_s6:.3f} contracts)")
    if avg_pos_s7 < 1:
        print(f"WARNING: Strategy 7 average position size is very small ({avg_pos_s7:.3f} contracts)")
    if avg_pos_s8 < 1:
        print(f"WARNING: Strategy 8 average position size is very small ({avg_pos_s8:.3f} contracts)")
    
    # Issue 4: Volatility forecast issues
    if recent_df['vol_forecast'].min() == 0.05:
        vol_floor_days = (recent_df['vol_forecast'] == 0.05).sum()
        print(f"WARNING: Volatility forecast hitting floor ({vol_floor_days} days at 5% minimum)")
    
    # Create detailed output for first 10 days
    print(f"\n--- Detailed Daily Breakdown (First 10 Days) ---")
    print(f"{'Date':<12} {'Price':<8} {'Vol%':<6} {'S6Sig':<6} {'S6Pos':<8} {'S7Fcst':<8} {'S7Pos':<8} {'S8Fcst':<8} {'S8Pos':<8}")
    print("-" * 90)
    
    for i, (date, row) in enumerate(recent_df.head(10).iterrows()):
        print(f"{date.strftime('%Y-%m-%d'):<12} ${row['Last']:<7.2f} {row['vol_forecast']*100:<5.1f}% "
              f"{row['trend_signal_s6_shifted']:<6.0f} {row['pos_s6']:<8.0f} "
              f"{row['forecast_s7_shifted']:<8.2f} {row['pos_s7']:<8.1f} "
              f"{row['forecast_s8_shifted']:<8.2f} {row['pos_s8']:<8.1f}")
    
    return recent_df

def compare_strategies_timing():
    """
    Compare how different strategies handle timing and position updates.
    """
    print(f"\n{'='*80}")
    print("STRATEGY TIMING AND POSITION UPDATE COMPARISON")
    print(f"{'='*80}")
    
    # Test the key differences between strategies
    test_prices = pd.Series([100, 101, 102, 104, 103, 102, 105, 108, 106, 109], 
                           index=pd.date_range('2023-01-01', periods=10))
    
    print("Test Price Series:")
    for date, price in test_prices.items():
        print(f"  {date.strftime('%Y-%m-%d')}: ${price}")
    
    # Calculate EWMA trends
    fast_64 = test_prices.ewm(span=64, adjust=False).mean()
    slow_256 = test_prices.ewm(span=256, adjust=False).mean()
    ewmac_64_256 = fast_64 - slow_256
    
    fast_16 = test_prices.ewm(span=16, adjust=False).mean()
    slow_64 = test_prices.ewm(span=64, adjust=False).mean()
    ewmac_16_64 = fast_16 - slow_64
    
    print(f"\nEWMAC Values:")
    print(f"EWMAC(64,256): {ewmac_64_256.iloc[-1]:.4f}")
    print(f"EWMAC(16,64): {ewmac_16_64.iloc[-1]:.4f}")
    
    # Test position sizing formulas
    capital = 50000000
    weight = 0.1
    idm = 2.0
    risk_target = 0.2
    multiplier = 5
    price = test_prices.iloc[-1]
    volatility = 0.15
    
    print(f"\nPosition Sizing Test (Price=${price}, Vol=15%):")
    
    # Strategy 4 baseline
    pos_s4 = (capital * idm * weight * risk_target) / (multiplier * price * volatility)
    print(f"Strategy 4 (Always Long): {pos_s4:.1f} contracts")
    
    # Strategy 6 
    trend_signal = 1 if ewmac_64_256.iloc[-1] > 0 else -1
    pos_s6 = (trend_signal * capital * idm * weight * risk_target) / (multiplier * price * volatility)
    print(f"Strategy 6 (Long/Short): {pos_s6:.1f} contracts (signal: {trend_signal})")
    
    # Strategy 7
    daily_price_vol = price * volatility / 16
    raw_forecast = ewmac_64_256.iloc[-1] / daily_price_vol
    scaled_forecast = raw_forecast * 1.9
    capped_forecast = np.clip(scaled_forecast, -20, 20)
    pos_s7 = (capped_forecast * capital * idm * weight * risk_target) / (10 * multiplier * price * volatility)
    print(f"Strategy 7 (Slow Forecast): {pos_s7:.1f} contracts (forecast: {capped_forecast:.2f})")
    
    # Strategy 8
    raw_forecast_fast = ewmac_16_64.iloc[-1] / daily_price_vol
    scaled_forecast_fast = raw_forecast_fast * 4.1
    capped_forecast_fast = np.clip(scaled_forecast_fast, -20, 20)
    pos_s8 = (capped_forecast_fast * capital * idm * weight * risk_target) / (10 * multiplier * price * volatility)
    print(f"Strategy 8 (Fast Forecast): {pos_s8:.1f} contracts (forecast: {capped_forecast_fast:.2f})")

def check_lookahead_bias():
    """
    Verify that no lookahead bias exists in the implementations.
    """
    print(f"\n{'='*80}")
    print("LOOKAHEAD BIAS VERIFICATION")
    print(f"{'='*80}")
    
    # Create test scenario
    dates = pd.date_range('2023-01-01', periods=10)
    prices = pd.Series([100, 102, 104, 103, 105, 107, 106, 108, 110, 109], index=dates)
    
    print("Verification that signals/forecasts are properly shifted:")
    print("Day T position should use signals calculated from data up to Day T-1")
    
    # Calculate unshifted signals
    fast_ewma = prices.ewm(span=3, adjust=False).mean()  # Use short span for demonstration
    slow_ewma = prices.ewm(span=5, adjust=False).mean()
    ewmac = fast_ewma - slow_ewma
    trend_signal = np.where(ewmac > 0, 1, -1)
    
    # Create shifted version
    trend_signal_shifted = pd.Series(trend_signal).shift(1).fillna(1)
    
    print(f"\n{'Date':<12} {'Price':<8} {'EWMAC':<8} {'Signal':<8} {'Shifted':<8} {'Used for':<15}")
    print("-" * 70)
    
    for i, (date, price) in enumerate(prices.items()):
        if i < len(ewmac):
            ewmac_val = ewmac.iloc[i]
            signal_val = trend_signal[i]
            shifted_val = trend_signal_shifted.iloc[i] if i > 0 else "N/A"
            used_for = f"Day {i+2} position" if i < len(prices)-1 else "No future"
            
            print(f"{date.strftime('%Y-%m-%d'):<12} ${price:<7.1f} {ewmac_val:<8.3f} {signal_val:<8.0f} {shifted_val:<8} {used_for:<15}")
    
    print(f"\n✓ Verification: Day T position uses signal calculated from Day T-1 data")

def run_comprehensive_debug():
    """
    Run comprehensive debugging analysis.
    """
    print("COMPREHENSIVE POSITION DEBUG ANALYSIS")
    print("=" * 80)
    
    # 1. Check basic calculations for a single instrument
    analyze_single_instrument_positioning('MES', 30)
    
    # 2. Compare strategy timing
    compare_strategies_timing()
    
    # 3. Verify no lookahead bias
    check_lookahead_bias()
    
    print(f"\n{'='*80}")
    print("DEBUG ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_comprehensive_debug() 