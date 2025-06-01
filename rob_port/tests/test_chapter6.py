import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import pandas as pd
from chapter6 import *

class TestChapter6Strategy(unittest.TestCase):
    """Test Strategy 6: Long/Short Trend Following implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.capital = 50000000
        self.risk_target = 0.2
        
        # Create sample price data for testing
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        np.random.seed(42)  # For reproducible tests
        
        # Trending up price series
        self.trending_up_prices = pd.Series(
            100 + np.cumsum(np.random.normal(0.05, 1, len(dates))),
            index=dates
        )
        
        # Trending down price series  
        self.trending_down_prices = pd.Series(
            100 + np.cumsum(np.random.normal(-0.05, 1, len(dates))),
            index=dates
        )
        
        # Sideways price series
        self.sideways_prices = pd.Series(
            100 + np.cumsum(np.random.normal(0, 1, len(dates))),
            index=dates
        )
    
    def test_trend_signal_long_short(self):
        """Test long/short trend signal calculation."""
        print("\n=== Testing Long/Short Trend Signal ===")
        
        # Test trending up series
        up_signal = calculate_trend_signal_long_short(self.trending_up_prices)
        
        # Should be mostly positive (long signals)
        long_signals = (up_signal == 1).sum()
        short_signals = (up_signal == -1).sum()
        
        print(f"Trending up - Long signals: {long_signals}, Short signals: {short_signals}")
        self.assertGreater(long_signals, short_signals, "Trending up should have more long signals")
        
        # Test trending down series
        down_signal = calculate_trend_signal_long_short(self.trending_down_prices)
        
        long_signals_down = (down_signal == 1).sum()
        short_signals_down = (down_signal == -1).sum()
        
        print(f"Trending down - Long signals: {long_signals_down}, Short signals: {short_signals_down}")
        self.assertGreater(short_signals_down, long_signals_down, "Trending down should have more short signals")
        
        # Test that signals are only +1 or -1
        unique_signals = np.unique(up_signal)
        expected_signals = [-1, 1]
        self.assertTrue(all(sig in expected_signals for sig in unique_signals),
                       "Signals should only be +1 or -1")
        
        print("✓ Long/short trend signal tests passed")
    
    def test_strategy6_position_sizing(self):
        """Test Strategy 6 position sizing with long/short signals."""
        print("\n=== Testing Strategy 6 Position Sizing ===")
        
        # Test parameters
        symbol = 'TEST'
        weight = 0.02
        idm = 2.5
        price = 4500
        volatility = 0.16
        multiplier = 5
        
        # Test long signal (+1)
        long_position = calculate_strategy6_position_size(
            symbol, self.capital, weight, idm, price, volatility,
            multiplier, 1, self.risk_target
        )
        
        # Test short signal (-1)
        short_position = calculate_strategy6_position_size(
            symbol, self.capital, weight, idm, price, volatility,
            multiplier, -1, self.risk_target
        )
        
        print(f"Long position: {long_position:.2f}")
        print(f"Short position: {short_position:.2f}")
        
        # Positions should be equal magnitude but opposite sign
        self.assertAlmostEqual(abs(long_position), abs(short_position), places=2,
                              msg="Long and short positions should have equal magnitude")
        self.assertGreater(long_position, 0, "Long signal should produce positive position")
        self.assertLess(short_position, 0, "Short signal should produce negative position")
        
        # Test zero volatility
        zero_vol_position = calculate_strategy6_position_size(
            symbol, self.capital, weight, idm, price, 0.0,
            multiplier, 1, self.risk_target
        )
        self.assertEqual(zero_vol_position, 0, "Zero volatility should return zero position")
        
        # Test NaN inputs
        nan_position = calculate_strategy6_position_size(
            symbol, self.capital, weight, idm, price, np.nan,
            multiplier, 1, self.risk_target
        )
        self.assertEqual(nan_position, 0, "NaN volatility should return zero position")
        
        print("✓ Strategy 6 position sizing tests passed")
    
    def test_long_short_vs_long_only_signals(self):
        """Test that long/short signals differ appropriately from long-only signals."""
        print("\n=== Testing Long/Short vs Long-Only Signals ===")
        
        # Calculate both signal types
        long_only_signal = calculate_trend_signal(self.trending_down_prices)
        long_short_signal = calculate_trend_signal_long_short(self.trending_down_prices)
        
        # Long-only should have 0s and 1s
        long_only_unique = np.unique(long_only_signal)
        self.assertTrue(all(sig in [0, 1] for sig in long_only_unique),
                       "Long-only signals should be 0 or 1")
        
        # Long/short should have -1s and 1s
        long_short_unique = np.unique(long_short_signal)
        self.assertTrue(all(sig in [-1, 1] for sig in long_short_unique),
                       "Long/short signals should be -1 or +1")
        
        # Where long-only is 0, long/short should be -1
        zero_positions = (long_only_signal == 0)
        short_positions = (long_short_signal == -1)
        
        # They should be mostly aligned (allowing for some differences due to initialization)
        alignment_ratio = (zero_positions == short_positions).mean()
        print(f"Signal alignment ratio: {alignment_ratio:.3f}")
        self.assertGreater(alignment_ratio, 0.95, "Signals should be mostly aligned")
        
        print("✓ Long/short vs long-only signal comparison passed")
    
    def test_strategy6_backtest_basic(self):
        """Test basic Strategy 6 backtest functionality."""
        print("\n=== Testing Strategy 6 Backtest ===")
        
        try:
            # Run a limited backtest
            results = backtest_long_short_trend_strategy(
                data_dir='Data',
                capital=self.capital,
                risk_target=self.risk_target,
                weight_method='handcrafted',
                start_date='2020-01-01',
                end_date='2023-12-31'
            )
            
            # Check basic structure
            required_keys = ['portfolio_data', 'performance', 'instrument_stats', 'weights', 'config']
            for key in required_keys:
                self.assertIn(key, results, f"Results should contain {key}")
            
            # Check performance metrics
            performance = results['performance']
            required_metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 
                              'max_drawdown_pct', 'avg_long_signals', 'avg_short_signals']
            for metric in required_metrics:
                self.assertIn(metric, performance, f"Performance should contain {metric}")
            
            # Check that we have both long and short signals
            self.assertGreater(performance['avg_long_signals'], 0, "Should have long signals")
            self.assertGreater(performance['avg_short_signals'], 0, "Should have short signals")
            
            # Check time in market
            total_signals = performance['avg_long_signals'] + performance['avg_short_signals']
            time_in_market = total_signals / performance['num_instruments']
            self.assertGreater(time_in_market, 0.5, "Should be in market majority of time")
            self.assertLess(time_in_market, 1.0, "Should not be in market 100% of time")
            
            print(f"✓ Strategy 6 backtest completed successfully")
            print(f"  Annual Return: {performance['annualized_return']:.2%}")
            print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
            print(f"  Long Signals: {performance['avg_long_signals']:.1f}")
            print(f"  Short Signals: {performance['avg_short_signals']:.1f}")
            print(f"  Time in Market: {time_in_market:.1%}")
            
        except Exception as e:
            self.fail(f"Strategy 6 backtest failed: {e}")
    
    def test_strategy_comparison(self):
        """Test comparison between strategies 4, 5, and 6."""
        print("\n=== Testing Strategy Comparison ===")
        
        try:
            # This is a lighter test - just check that comparison runs
            # Full comparison is tested in main()
            
            # Test that we can calculate different signal types
            prices = self.trending_up_prices[:100]  # Smaller dataset for speed
            
            # Strategy 5 signal (long-only)
            s5_signal = calculate_trend_signal(prices)
            
            # Strategy 6 signal (long/short)
            s6_signal = calculate_trend_signal_long_short(prices)
            
            # Strategy 6 should always have a position (long or short)
            self.assertTrue(all(abs(sig) == 1 for sig in s6_signal),
                           "Strategy 6 should always have position")
            
            # Strategy 5 can be flat (0)
            self.assertTrue(any(sig == 0 for sig in s5_signal),
                           "Strategy 5 should have some flat periods")
            
            print("✓ Strategy comparison logic tests passed")
            
        except Exception as e:
            self.fail(f"Strategy comparison test failed: {e}")
    
    def test_performance_metrics(self):
        """Test that performance metrics are calculated correctly."""
        print("\n=== Testing Performance Metrics ===")
        
        # Create simple test data with proper date index
        dates = pd.date_range('2020-01-01', periods=250, freq='D')
        test_returns = pd.Series([0.01, -0.005, 0.015, -0.01, 0.008] * 50, index=dates)  # 250 days
        test_capital = 1000000
        
        account_curve = build_account_curve(test_returns, test_capital)
        performance = calculate_comprehensive_performance(account_curve, test_returns)
        
        # Check that all required metrics exist
        required_metrics = [
            'total_return', 'annualized_return', 'annualized_volatility',
            'sharpe_ratio', 'max_drawdown_pct', 'skewness'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, performance, f"Should have {metric}")
            self.assertIsNotNone(performance[metric], f"{metric} should not be None")
        
        # Basic sanity checks
        self.assertIsInstance(performance['total_return'], float)
        self.assertIsInstance(performance['sharpe_ratio'], float)
        self.assertLessEqual(performance['max_drawdown_pct'], 0, "Max drawdown should be negative")
        
        print("✓ Performance metrics tests passed")

def run_strategy6_tests():
    """Run all Strategy 6 tests."""
    print("=" * 60)
    print("RUNNING STRATEGY 6 UNIT TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestChapter6Strategy)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n" + "=" * 60)
        print("ALL STRATEGY 6 TESTS PASSED ✓")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print("SOME STRATEGY 6 TESTS FAILED ✗")
        print("=" * 60)
        return False

if __name__ == "__main__":
    run_strategy6_tests() 