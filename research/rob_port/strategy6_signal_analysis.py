#!/usr/bin/env python3
"""
Detailed Analysis of Strategy 6 Signal Generation

This script investigates why Strategy 6 generates heavily biased long signals.
"""

import sys
sys.path.append('rob_port')
from chapter6 import *
from chapter5 import *
from chapter4 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_ewmac_calculation():
    """Analyze the EWMAC calculation in detail."""
    print("=" * 80)
    print("EWMAC SIGNAL GENERATION ANALYSIS")
    print("=" * 80)
    
    # Load MES data
    raw_data = load_all_instrument_data('Data')
    mes_data = raw_data['MES'].copy()
    
    print(f"MES Data Analysis:")
    print(f"  Total periods: {len(mes_data)}")
    print(f"  Date range: {mes_data.index.min()} to {mes_data.index.max()}")
    print(f"  Price range: {mes_data['Last'].min():.2f} to {mes_data['Last'].max():.2f}")
    
    # Focus on recent period
    recent_data = mes_data.tail(1000)  # Last 1000 days
    prices = recent_data['Last']
    
    # Calculate EWMAC components step by step
    fast_span = 64
    slow_span = 256
    
    # Step 1: Calculate EMAs
    fast_ema = prices.ewm(span=fast_span).mean()
    slow_ema = prices.ewm(span=slow_span).mean()
    
    # Step 2: Calculate crossover
    ewmac_raw = fast_ema - slow_ema
    
    # Step 3: Apply sign function for signals
    signals = np.where(ewmac_raw > 0, 1, -1)
    signals_series = pd.Series(signals, index=prices.index)
    
    print(f"\nEWMAC Breakdown (last 1000 days):")
    print(f"  Fast EMA (64): {fast_ema.tail(1).iloc[0]:.2f}")
    print(f"  Slow EMA (256): {slow_ema.tail(1).iloc[0]:.2f}")
    print(f"  EWMAC (Fast - Slow): {ewmac_raw.tail(1).iloc[0]:.2f}")
    print(f"  Current signal: {signals_series.tail(1).iloc[0]}")
    
    # Analyze signal distribution
    long_signals = (signals_series == 1).sum()
    short_signals = (signals_series == -1).sum()
    
    print(f"\nSignal Distribution:")
    print(f"  Long signals: {long_signals} ({long_signals/len(signals_series):.1%})")
    print(f"  Short signals: {short_signals} ({short_signals/len(signals_series):.1%})")
    
    # Check if this is due to overall market trend
    price_start = prices.iloc[0]
    price_end = prices.iloc[-1]
    total_return = (price_end - price_start) / price_start
    
    print(f"\nMarket Context:")
    print(f"  Starting price: {price_start:.2f}")
    print(f"  Ending price: {price_end:.2f}")
    print(f"  Total return: {total_return:.1%}")
    print(f"  This explains the long bias!" if total_return > 0.1 else "  Long bias not explained by trend")
    
    # Analyze different time periods
    periods = [
        ('Recent 250 days', -250),
        ('Recent 500 days', -500),
        ('Recent 750 days', -750),
        ('All 1000 days', -1000)
    ]
    
    print(f"\nSignal Distribution by Period:")
    for name, days in periods:
        period_signals = signals_series.iloc[days:]
        long_pct = (period_signals == 1).mean()
        print(f"  {name}: {long_pct:.1%} long")
    
    return {
        'prices': prices,
        'fast_ema': fast_ema,
        'slow_ema': slow_ema,
        'ewmac_raw': ewmac_raw,
        'signals': signals_series
    }

def test_alternative_signal_methods():
    """Test alternative signal generation methods."""
    print("\n" + "=" * 80)
    print("ALTERNATIVE SIGNAL METHODS")
    print("=" * 80)
    
    # Load data
    raw_data = load_all_instrument_data('Data')
    mes_data = raw_data['MES'].copy()
    recent_data = mes_data.tail(1000)
    prices = recent_data['Last']
    
    # Method 1: Current EWMAC
    current_signals = calculate_trend_signal_long_short(prices, 64, 256)
    
    # Method 2: Different span combinations
    alt_signals_1 = calculate_trend_signal_long_short(prices, 32, 128)  # Faster
    alt_signals_2 = calculate_trend_signal_long_short(prices, 16, 64)   # Much faster
    
    # Method 3: Add neutral zone (dead band)
    def calculate_neutral_zone_signals(prices, fast_span=64, slow_span=256, neutral_threshold=0.01):
        ewmac = calculate_ewma_trend(prices, fast_span, slow_span)
        # Normalize by price to get percentage crossover
        normalized_ewmac = ewmac / prices
        
        signals = np.where(normalized_ewmac > neutral_threshold, 1,
                          np.where(normalized_ewmac < -neutral_threshold, -1, 0))
        return pd.Series(signals, index=prices.index)
    
    neutral_signals = calculate_neutral_zone_signals(prices)
    
    # Compare distributions
    methods = [
        ('Current EWMAC(64,256)', current_signals),
        ('Faster EWMAC(32,128)', alt_signals_1),
        ('Much Faster EWMAC(16,64)', alt_signals_2),
        ('Neutral Zone EWMAC', neutral_signals)
    ]
    
    print(f"Signal Distribution Comparison:")
    print(f"{'Method':<25} {'Long %':<8} {'Short %':<8} {'Neutral %':<10}")
    print("-" * 60)
    
    for name, signals in methods:
        signals_clean = signals.dropna()
        if len(signals_clean) > 0:
            long_pct = (signals_clean == 1).mean()
            short_pct = (signals_clean == -1).mean()
            neutral_pct = (signals_clean == 0).mean() if 0 in signals_clean.values else 0
            print(f"{name:<25} {long_pct:<8.1%} {short_pct:<8.1%} {neutral_pct:<10.1%}")

def analyze_market_regime():
    """Analyze different market regimes and their impact on signals."""
    print("\n" + "=" * 80)
    print("MARKET REGIME ANALYSIS")
    print("=" * 80)
    
    # Load multiple instruments
    raw_data = load_all_instrument_data('Data')
    
    instruments_to_test = ['MES', 'MNQ', 'DAX', 'EUR', 'ZB', 'HG']
    available_instruments = [sym for sym in instruments_to_test if sym in raw_data]
    
    print(f"Analyzing {len(available_instruments)} instruments:")
    print(f"{'Symbol':<8} {'Long %':<8} {'Short %':<8} {'Recent Trend':<12}")
    print("-" * 50)
    
    for symbol in available_instruments:
        data = raw_data[symbol].copy()
        if len(data) < 500:
            continue
            
        # Calculate signals
        prices = data['Last'].tail(1000)  # Last 1000 days
        signals = calculate_trend_signal_long_short(prices, 64, 256)
        signals_clean = signals.dropna()
        
        if len(signals_clean) > 0:
            long_pct = (signals_clean == 1).mean()
            short_pct = (signals_clean == -1).mean()
            
            # Calculate recent trend
            recent_return = (prices.iloc[-1] - prices.iloc[-250]) / prices.iloc[-250]
            trend_desc = "Bullish" if recent_return > 0.1 else "Bearish" if recent_return < -0.1 else "Sideways"
            
            print(f"{symbol:<8} {long_pct:<8.1%} {short_pct:<8.1%} {trend_desc:<12}")

def plot_signal_analysis():
    """Create visualizations of signal behavior."""
    print("\n" + "=" * 80)
    print("CREATING SIGNAL VISUALIZATION")
    print("=" * 80)
    
    try:
        # Get EWMAC analysis results
        results = analyze_ewmac_calculation()
        
        # Create plot
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Price and EMAs
        recent_data = results['prices'].tail(500)
        fast_ema = results['fast_ema'].tail(500)
        slow_ema = results['slow_ema'].tail(500)
        
        axes[0].plot(recent_data.index, recent_data.values, 'k-', label='MES Price', linewidth=1)
        axes[0].plot(fast_ema.index, fast_ema.values, 'b-', label='Fast EMA (64)', linewidth=1.5)
        axes[0].plot(slow_ema.index, slow_ema.values, 'r-', label='Slow EMA (256)', linewidth=1.5)
        axes[0].set_title('MES Price and EMAs (Last 500 Days)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: EWMAC crossover
        ewmac = results['ewmac_raw'].tail(500)
        axes[1].plot(ewmac.index, ewmac.values, 'g-', label='EWMAC (Fast - Slow)', linewidth=1)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].fill_between(ewmac.index, ewmac.values, 0, 
                            where=(ewmac.values > 0), alpha=0.3, color='green', label='Long Signal')
        axes[1].fill_between(ewmac.index, ewmac.values, 0, 
                            where=(ewmac.values < 0), alpha=0.3, color='red', label='Short Signal')
        axes[1].set_title('EWMAC Crossover Signal')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Signal distribution over time
        signals = results['signals'].tail(500)
        signal_rolling = signals.rolling(window=30).mean()  # 30-day rolling average
        
        axes[2].plot(signal_rolling.index, signal_rolling.values, 'purple', linewidth=2)
        axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[2].axhline(y=0.5, color='green', linestyle=':', alpha=0.5, label='50% Long')
        axes[2].axhline(y=-0.5, color='red', linestyle=':', alpha=0.5, label='50% Short')
        axes[2].set_title('30-Day Rolling Average of Signals (1=Long, -1=Short)')
        axes[2].set_ylabel('Signal Strength')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('strategy6_signal_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Signal analysis plot saved as 'strategy6_signal_analysis.png'")
        
    except Exception as e:
        print(f"❌ Error creating plot: {e}")

def run_signal_analysis():
    """Run complete signal analysis."""
    analyze_ewmac_calculation()
    test_alternative_signal_methods()
    analyze_market_regime()
    plot_signal_analysis()

if __name__ == "__main__":
    run_signal_analysis() 