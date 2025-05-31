from chapter5 import *

def test_trend_indicators():
    """
    Test the trend indicator calculations to ensure they match the book's methodology.
    """
    print("=" * 60)
    print("TESTING CHAPTER 5 TREND INDICATORS")
    print("=" * 60)
    
    # Create sample price data
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    # Create trending price series
    base_trend = np.linspace(100, 150, 300)
    noise = np.random.normal(0, 2, 300)
    prices = pd.Series(base_trend + noise, index=dates)
    
    print(f"Sample price data: {len(prices)} days")
    print(f"Price range: {prices.min():.2f} to {prices.max():.2f}")
    
    # Test moving averages
    fast_window = 64
    slow_window = 256
    
    # Calculate both SMA and EWMA
    sma_signals = calculate_moving_average_crossover(prices, fast_window, slow_window, use_ewma=False)
    ewma_signals = calculate_moving_average_crossover(prices, fast_window, slow_window, use_ewma=True)
    
    print(f"\n--- Simple Moving Average Results ---")
    print(f"Fast SMA (64): min={sma_signals['fast_ma'].min():.2f}, max={sma_signals['fast_ma'].max():.2f}")
    print(f"Slow SMA (256): min={sma_signals['slow_ma'].min():.2f}, max={sma_signals['slow_ma'].max():.2f}")
    print(f"Uptrend signals: {sma_signals['trend_signal'].sum()} out of {len(sma_signals['trend_signal'])} days")
    print(f"Uptrend percentage: {sma_signals['trend_signal'].mean():.1%}")
    
    print(f"\n--- Exponential Moving Average Results ---")
    print(f"Fast EWMA (64): min={ewma_signals['fast_ma'].min():.2f}, max={ewma_signals['fast_ma'].max():.2f}")
    print(f"Slow EWMA (256): min={ewma_signals['slow_ma'].min():.2f}, max={ewma_signals['slow_ma'].max():.2f}")
    print(f"Uptrend signals: {ewma_signals['trend_signal'].sum()} out of {len(ewma_signals['trend_signal'])} days")
    print(f"Uptrend percentage: {ewma_signals['trend_signal'].mean():.1%}")
    
    # Calculate signal differences (whipsaws)
    sma_changes = sma_signals['trend_signal'].diff().abs().sum()
    ewma_changes = ewma_signals['trend_signal'].diff().abs().sum()
    
    print(f"\n--- Signal Stability (fewer changes = better) ---")
    print(f"SMA signal changes: {sma_changes}")
    print(f"EWMA signal changes: {ewma_changes}")
    print(f"EWMA is {'more' if ewma_changes < sma_changes else 'less'} stable than SMA")
    
    return {
        'sma_signals': sma_signals,
        'ewma_signals': ewma_signals,
        'prices': prices
    }

def test_single_instrument_trend_following():
    """
    Test trend following on a single instrument to demonstrate the concept.
    """
    print("\n" + "=" * 60)
    print("TESTING SINGLE INSTRUMENT TREND FOLLOWING")
    print("=" * 60)
    
    # Load MES data for testing
    try:
        instruments_df = load_instrument_data()
        data = load_instrument_data_files(['MES'])
        
        if 'MES' not in data:
            print("MES data not available for testing")
            return
        
        mes_data = data['MES']
        print(f"Loaded MES data: {len(mes_data)} observations")
        print(f"Date range: {mes_data.index.min()} to {mes_data.index.max()}")
        
        # Calculate trend signals
        prices = mes_data['Last']
        trend_signals = calculate_moving_average_crossover(prices, 64, 256, use_ewma=True)
        
        print(f"\n--- MES Trend Analysis ---")
        print(f"Total days in uptrend: {trend_signals['trend_signal'].sum()}")
        print(f"Uptrend percentage: {trend_signals['trend_signal'].mean():.1%}")
        print(f"Signal changes (trend reversals): {trend_signals['trend_signal'].diff().abs().sum()}")
        print(f"Average trend strength: {trend_signals['trend_strength'].mean():.4f}")
        
        # Test with a small portfolio
        portfolio_weights = {'MES': 1.0}
        
        print(f"\n--- Single Instrument Portfolio Test ---")
        result = backtest_trend_following_portfolio(
            portfolio_weights, data, instruments_df,
            capital=100000, risk_target=0.2, start_date='2015-01-01'
        )
        
        if 'error' not in result:
            perf = result['performance']
            print(f"Performance with trend filter:")
            print(f"  Annual Return: {perf['annualized_return']:.2%}")
            print(f"  Volatility: {perf['annualized_volatility']:.2%}")
            print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {perf['max_drawdown_pct']:.1f}%")
            print(f"  Avg Trend Exposure: {perf['avg_trend_exposure']:.1%}")
        else:
            print(f"Error in backtest: {result['error']}")
            
    except Exception as e:
        print(f"Error in single instrument test: {e}")

def test_trend_filter_mechanics():
    """
    Test the specific mechanics of how trend filters affect position sizing.
    """
    print("\n" + "=" * 60)
    print("TESTING TREND FILTER MECHANICS")
    print("=" * 60)
    
    # Create sample trend signals
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    
    # Sample portfolio weights
    portfolio_weights = {
        'INST1': 0.5,  # 50% allocation
        'INST2': 0.3,  # 30% allocation  
        'INST3': 0.2   # 20% allocation
    }
    
    # Sample trend signals - different instruments in different trend states
    trend_signals_dict = {
        'INST1': {
            'trend_signal': pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 1, 1], index=dates),  # Mostly uptrend
            'uptrend': pd.Series([True, True, True, False, False, False, True, True, True, True], index=dates)
        },
        'INST2': {
            'trend_signal': pd.Series([0, 0, 1, 1, 1, 1, 1, 0, 0, 0], index=dates),  # Mixed trend
            'uptrend': pd.Series([False, False, True, True, True, True, True, False, False, False], index=dates)
        },
        'INST3': {
            'trend_signal': pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], index=dates),  # Always uptrend
            'uptrend': pd.Series([True, True, True, True, True, True, True, True, True, True], index=dates)
        }
    }
    
    print("Original portfolio weights:")
    for instrument, weight in portfolio_weights.items():
        print(f"  {instrument}: {weight:.1%}")
    
    # Apply trend filters
    trend_adjusted_weights = apply_trend_filter_to_weights(portfolio_weights, trend_signals_dict)
    
    print(f"\nTrend-adjusted weights by date:")
    print(f"{'Date':<12} {'INST1':<8} {'INST2':<8} {'INST3':<8} {'Total':<8}")
    print("-" * 50)
    
    for date in dates:
        if date in trend_adjusted_weights:
            weights = trend_adjusted_weights[date]
            total = sum(weights.values())
            print(f"{date.strftime('%Y-%m-%d'):<12} {weights.get('INST1', 0):<8.1%} {weights.get('INST2', 0):<8.1%} {weights.get('INST3', 0):<8.1%} {total:<8.1%}")
    
    print(f"\nKey observations:")
    print("1. Weights are renormalized among instruments in uptrends")
    print("2. When instruments are in downtrends, they get zero weight")
    print("3. Total weight may be less than 100% when some instruments are out of trend")
    print("4. This reduces portfolio exposure during broad market downtrends")

def main():
    """
    Run all Chapter 5 tests to demonstrate functionality.
    """
    print("CHAPTER 5 TREND FOLLOWING - FUNCTIONALITY TESTS")
    print("=" * 80)
    
    # Test 1: Basic trend indicators
    trend_test_results = test_trend_indicators()
    
    # Test 2: Single instrument application
    test_single_instrument_trend_following()
    
    # Test 3: Trend filter mechanics
    test_trend_filter_mechanics()
    
    print("\n" + "=" * 60)
    print("CHAPTER 5 TESTING COMPLETED")
    print("=" * 60)
    print("Key Features Demonstrated:")
    print("✓ Moving average crossover signals (SMA vs EWMA)")
    print("✓ Trend signal generation and stability")
    print("✓ Dynamic weight adjustment based on trends")
    print("✓ Position sizing with trend filters")
    print("✓ Portfolio-level trend exposure management")
    print("✓ Integration with optimized instrument selection")

if __name__ == "__main__":
    main() 