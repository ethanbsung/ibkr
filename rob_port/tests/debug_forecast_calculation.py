#!/usr/bin/env python3
"""
Debug Forecast Calculation Issues

This script analyzes why forecasts are constantly hitting the maximum cap in Strategy 7.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add the rob_port directory to path so we can import the modules
sys.path.append('rob_port')

from chapter7 import *
from chapter6 import *
from chapter5 import *
from chapter4 import *
from chapter3 import *
from chapter2 import *
from chapter1 import *

def debug_forecast_calculation(symbol='MES', days_to_analyze=100):
    """
    Debug the forecast calculation pipeline to understand why forecasts are maxed out.
    """
    print(f"\n{'='*80}")
    print(f"DEBUGGING FORECAST CALCULATION FOR {symbol}")
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
    
    # Take recent data for analysis
    recent_df = df.tail(days_to_analyze).copy()
    print(f"\nAnalyzing last {days_to_analyze} days:")
    print(f"Date Range: {recent_df.index[0].date()} to {recent_df.index[-1].date()}")
    print(f"Price Range: ${recent_df['Last'].min():.2f} to ${recent_df['Last'].max():.2f}")
    
    # Step 1: Calculate EWMAC
    fast_ewma_64 = recent_df['Last'].ewm(span=64, adjust=False).mean()
    slow_ewma_256 = recent_df['Last'].ewm(span=256, adjust=False).mean()
    ewmac_64_256 = fast_ewma_64 - slow_ewma_256
    
    print(f"\n--- EWMAC Analysis ---")
    print(f"EWMAC Range: {ewmac_64_256.min():.4f} to {ewmac_64_256.max():.4f}")
    print(f"EWMAC Mean: {ewmac_64_256.mean():.4f}")
    print(f"EWMAC Std: {ewmac_64_256.std():.4f}")
    
    # Step 2: Calculate volatility
    blended_vol = calculate_blended_volatility(recent_df['daily_returns'], 32, 10, 0.05)
    
    print(f"\n--- Volatility Analysis ---")
    print(f"Blended Vol Range: {blended_vol.min():.4f} to {blended_vol.max():.4f}")
    print(f"Blended Vol Mean: {blended_vol.mean():.4f}")
    print(f"Days at floor (5%): {(blended_vol == 0.05).sum()}")
    print(f"% Days at floor: {(blended_vol == 0.05).mean()*100:.1f}%")
    
    # Step 3: Calculate daily price volatility
    daily_price_vol = recent_df['Last'] * blended_vol / 16
    
    print(f"\n--- Daily Price Volatility Analysis ---")
    print(f"Daily Price Vol Range: {daily_price_vol.min():.4f} to {daily_price_vol.max():.4f}")
    print(f"Daily Price Vol Mean: {daily_price_vol.mean():.4f}")
    
    # Step 4: Calculate raw forecast
    raw_forecast = ewmac_64_256 / daily_price_vol
    raw_forecast = raw_forecast.replace([np.inf, -np.inf], 0).fillna(0)
    
    print(f"\n--- Raw Forecast Analysis ---")
    print(f"Raw Forecast Range: {raw_forecast.min():.4f} to {raw_forecast.max():.4f}")
    print(f"Raw Forecast Mean: {raw_forecast.mean():.4f}")
    print(f"Raw Forecast Std: {raw_forecast.std():.4f}")
    
    # Step 5: Calculate scaled forecast
    forecast_scalar = 1.9
    scaled_forecast = raw_forecast * forecast_scalar
    
    print(f"\n--- Scaled Forecast Analysis ---")
    print(f"Scaled Forecast Range: {scaled_forecast.min():.4f} to {scaled_forecast.max():.4f}")
    print(f"Scaled Forecast Mean: {scaled_forecast.mean():.4f}")
    print(f"Values > 20: {(scaled_forecast > 20).sum()}")
    print(f"Values < -20: {(scaled_forecast < -20).sum()}")
    print(f"% Values hitting cap: {((scaled_forecast > 20) | (scaled_forecast < -20)).mean()*100:.1f}%")
    
    # Step 6: Calculate capped forecast
    capped_forecast = np.clip(scaled_forecast, -20, 20)
    
    print(f"\n--- Capped Forecast Analysis ---")
    print(f"Capped Forecast Range: {capped_forecast.min():.4f} to {capped_forecast.max():.4f}")
    print(f"Capped Forecast Mean: {capped_forecast.mean():.4f}")
    print(f"Days at +20 cap: {(capped_forecast == 20).sum()}")
    print(f"Days at -20 cap: {(capped_forecast == -20).sum()}")
    print(f"% Days at any cap: {((capped_forecast == 20) | (capped_forecast == -20)).mean()*100:.1f}%")
    
    # Identify the problem
    print(f"\n--- Problem Identification ---")
    
    # Check if volatility floor is too low
    if (blended_vol == 0.05).mean() > 0.8:
        print("❌ PROBLEM: Volatility is hitting the floor too often (>80% of days)")
        print("   This makes daily_price_vol very small, causing raw_forecast to be huge")
        print("   SOLUTION: Consider raising min_vol_floor or fixing volatility calculation")
    
    # Check if forecast scalar is too high
    if ((scaled_forecast > 20) | (scaled_forecast < -20)).mean() > 0.5:
        print("❌ PROBLEM: Forecast scalar (1.9) is too high, causing frequent capping")
        print("   SOLUTION: Consider reducing forecast scalar")
    
    # Check if division by 16 is too aggressive
    avg_price = recent_df['Last'].mean()
    avg_vol = blended_vol.mean()
    avg_daily_price_vol = avg_price * avg_vol / 16
    avg_ewmac = abs(ewmac_64_256).mean()
    expected_raw_forecast = avg_ewmac / avg_daily_price_vol
    
    print(f"\n--- Expected Values Check ---")
    print(f"Average Price: ${avg_price:.2f}")
    print(f"Average Vol: {avg_vol:.4f}")
    print(f"Average Daily Price Vol: {avg_daily_price_vol:.4f}")
    print(f"Average |EWMAC|: {avg_ewmac:.4f}")
    print(f"Expected Raw Forecast: {expected_raw_forecast:.2f}")
    print(f"Expected Scaled Forecast: {expected_raw_forecast * forecast_scalar:.2f}")
    
    if expected_raw_forecast * forecast_scalar > 20:
        print("❌ PROBLEM: Expected scaled forecast exceeds cap consistently")
    
    # Show detailed breakdown for first 10 days
    print(f"\n--- Detailed Daily Breakdown (First 10 Days) ---")
    print(f"{'Date':<12} {'Price':<8} {'EWMAC':<8} {'Vol%':<6} {'DPVol':<8} {'Raw':<8} {'Scaled':<8} {'Capped':<8}")
    print("-" * 90)
    
    for i in range(min(10, len(recent_df))):
        date = recent_df.index[i]
        price = recent_df['Last'].iloc[i]
        ewmac = ewmac_64_256.iloc[i]
        vol = blended_vol.iloc[i]
        dpvol = daily_price_vol.iloc[i]
        raw = raw_forecast.iloc[i]
        scaled = scaled_forecast.iloc[i]
        capped = capped_forecast.iloc[i]
        
        print(f"{date.strftime('%Y-%m-%d'):<12} ${price:<7.2f} {ewmac:<8.3f} {vol*100:<5.1f}% "
              f"{dpvol:<8.3f} {raw:<8.1f} {scaled:<8.1f} {capped:<8.1f}")
    
    return {
        'recent_df': recent_df,
        'ewmac': ewmac_64_256,
        'blended_vol': blended_vol,
        'daily_price_vol': daily_price_vol,
        'raw_forecast': raw_forecast,
        'scaled_forecast': scaled_forecast,
        'capped_forecast': capped_forecast
    }

def test_different_parameters():
    """
    Test different parameter combinations to see what works better.
    """
    print(f"\n{'='*80}")
    print("TESTING DIFFERENT FORECAST PARAMETERS")
    print(f"{'='*80}")
    
    # Load MES data
    try:
        df = pd.read_csv('Data/mes_daily_data.csv', parse_dates=['Time'])
        df.set_index('Time', inplace=True)
        df['daily_returns'] = df['Last'].pct_change()
        df = df.dropna()
        recent_df = df.tail(50).copy()
    except FileNotFoundError:
        print("Could not find MES data")
        return
    
    # Calculate base components
    fast_ewma = recent_df['Last'].ewm(span=64, adjust=False).mean()
    slow_ewma = recent_df['Last'].ewm(span=256, adjust=False).mean()
    ewmac = fast_ewma - slow_ewma
    
    # Test different parameters
    test_cases = [
        {"vol_floor": 0.05, "scalar": 1.9, "divisor": 16, "name": "Current"},
        {"vol_floor": 0.10, "scalar": 1.9, "divisor": 16, "name": "Higher Vol Floor"},
        {"vol_floor": 0.05, "scalar": 1.0, "divisor": 16, "name": "Lower Scalar"},
        {"vol_floor": 0.05, "scalar": 1.9, "divisor": 32, "name": "Higher Divisor"},
        {"vol_floor": 0.10, "scalar": 1.0, "divisor": 32, "name": "Conservative"},
    ]
    
    print(f"{'Case':<20} {'% at Cap':<10} {'Avg Forecast':<12} {'Forecast Range':<20}")
    print("-" * 70)
    
    for case in test_cases:
        # Calculate volatility with this floor
        blended_vol = calculate_blended_volatility(recent_df['daily_returns'], 32, 10, case["vol_floor"])
        
        # Calculate daily price volatility with this divisor
        daily_price_vol = recent_df['Last'] * blended_vol / case["divisor"]
        
        # Calculate forecast with this scalar
        raw_forecast = ewmac / daily_price_vol
        raw_forecast = raw_forecast.replace([np.inf, -np.inf], 0).fillna(0)
        scaled_forecast = raw_forecast * case["scalar"]
        capped_forecast = np.clip(scaled_forecast, -20, 20)
        
        # Calculate metrics
        pct_at_cap = ((capped_forecast == 20) | (capped_forecast == -20)).mean() * 100
        avg_forecast = capped_forecast.mean()
        forecast_range = f"{capped_forecast.min():.1f} to {capped_forecast.max():.1f}"
        
        print(f"{case['name']:<20} {pct_at_cap:<10.1f}% {avg_forecast:<12.2f} {forecast_range:<20}")

def main():
    """
    Run comprehensive forecast debugging.
    """
    print("COMPREHENSIVE FORECAST DEBUG ANALYSIS")
    print("=" * 80)
    
    # Debug the forecast calculation
    debug_data = debug_forecast_calculation('MES', 50)
    
    # Test different parameters
    test_different_parameters()
    
    print(f"\n{'='*80}")
    print("FORECAST DEBUG ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 