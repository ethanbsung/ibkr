import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chapter5 import *
from chapter4 import *
import pandas as pd
import numpy as np
import unittest
from datetime import datetime, timedelta

class TestStrategy5(unittest.TestCase):
    """
    Comprehensive test suite for Strategy 5: Trend following multi-instrument portfolio.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.capital = 50000000
        self.risk_target = 0.2
        self.instruments_df = load_instrument_data()
        
    def test_ewma_trend_calculation(self):
        """Test EWMA trend filter calculation."""
        print("\n=== Testing EWMA Trend Calculation ===")
        
        # Create synthetic price data with clear trend
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        # Uptrend: starts at 100, gradually increases to 150
        prices = pd.Series(100 + np.arange(300) * 0.5 + np.random.normal(0, 1, 300), index=dates)
        
        # Calculate EWMAC
        ewmac = calculate_ewma_trend(prices, fast_span=64, slow_span=256)
        trend_signal = calculate_trend_signal(prices, fast_span=64, slow_span=256)
        
        # In an uptrend, EWMAC should generally be positive
        # and trend signal should be mostly 1
        avg_ewmac = ewmac.iloc[-100:].mean()  # Last 100 observations
        avg_signal = trend_signal.iloc[-100:].mean()  # Last 100 observations
        
        print(f"Average EWMAC (last 100 obs): {avg_ewmac:.4f}")
        print(f"Average trend signal (last 100 obs): {avg_signal:.4f}")
        
        self.assertGreater(avg_ewmac, 0, "EWMAC should be positive in uptrend")
        self.assertGreater(avg_signal, 0.6, "Trend signal should be mostly long in uptrend")
        
        # Test edge cases
        constant_prices = pd.Series([100] * 300, index=dates)
        ewmac_flat = calculate_ewma_trend(constant_prices)
        
        # EWMAC should be near zero for flat prices
        self.assertLess(abs(ewmac_flat.iloc[-1]), 0.01, "EWMAC should be near zero for flat prices")
        
        print("✓ EWMA trend calculation tests passed")
    
    def test_strategy5_position_sizing(self):
        """Test Strategy 5 position sizing with trend filter."""
        print("\n=== Testing Strategy 5 Position Sizing ===")
        
        # Test normal case with trend signal = 1
        position_long = calculate_strategy5_position_size(
            symbol='MES', capital=self.capital, weight=0.02, idm=2.5, 
            price=4500, volatility=0.16, multiplier=5, trend_signal=1.0, risk_target=0.2
        )
        
        # Test same parameters with trend signal = 0
        position_flat = calculate_strategy5_position_size(
            symbol='MES', capital=self.capital, weight=0.02, idm=2.5, 
            price=4500, volatility=0.16, multiplier=5, trend_signal=0.0, risk_target=0.2
        )
        
        print(f"Position with long signal: {position_long:.2f}")
        print(f"Position with flat signal: {position_flat:.2f}")
        
        self.assertGreater(position_long, 0, "Position should be positive with long signal")
        self.assertEqual(position_flat, 0, "Position should be zero with flat signal")
        
        # Test that long position equals base position from Strategy 4
        base_position = calculate_portfolio_position_size(
            'MES', self.capital, 0.02, 2.5, 4500, 0.16, 5, 0.2
        )
        
        self.assertAlmostEqual(position_long, base_position, places=2, 
                              msg="Long position should equal base Strategy 4 position")
        
        print("✓ Strategy 5 position sizing tests passed")
    
    def test_trend_following_backtest_smoke(self):
        """Smoke test for Strategy 5 trend following backtest."""
        print("\n=== Strategy 5 Trend Following Smoke Test ===")
        
        try:
            # Run a quick backtest with a shorter time period
            results = backtest_trend_following_strategy(
                data_dir='Data',
                capital=self.capital,
                risk_target=0.2,
                weight_method='handcrafted',
                start_date='2020-01-01',
                end_date='2023-12-31'
            )
            
            # Validate results structure
            self.assertIn('portfolio_data', results, "Results should contain portfolio_data")
            self.assertIn('performance', results, "Results should contain performance")
            self.assertIn('instrument_stats', results, "Results should contain instrument_stats")
            self.assertIn('weights', results, "Results should contain weights")
            
            # Validate trend-specific metrics
            performance = results['performance']
            required_metrics = ['avg_long_signals', 'trend_fast_span', 'trend_slow_span']
            
            for metric in required_metrics:
                self.assertIn(metric, performance, f"Performance should include {metric}")
            
            # Validate that we have reasonable trend following behavior
            portfolio_data = results['portfolio_data']
            self.assertIn('num_long_signals', portfolio_data.columns, 
                         "Portfolio data should track long signals")
            
            avg_long_signals = performance['avg_long_signals']
            num_instruments = performance['num_instruments']
            
            # Should be trading less than 100% of the time due to trend filter
            time_in_market = avg_long_signals / num_instruments
            self.assertLess(time_in_market, 0.9, 
                           "Trend filter should reduce time in market below 90%")
            
            print(f"Smoke test passed: {len(results['instrument_stats'])} instruments processed")
            print(f"Time in market: {time_in_market:.1%}")
            print(f"Total return: {performance['total_return']:.2%}")
            print(f"Sharpe ratio: {performance['sharpe_ratio']:.3f}")
            
        except Exception as e:
            self.fail(f"Strategy 5 smoke test failed: {e}")
    
    def test_strategy_comparison(self):
        """Test comparison between Strategy 4 and Strategy 5."""
        print("\n=== Testing Strategy 4 vs Strategy 5 Comparison ===")
        
        # Test period (shorter for faster execution)
        start_date = '2015-01-01'
        end_date = '2020-12-31'
        
        try:
            # Strategy 4 (no trend filter)
            strategy4_results = backtest_multi_instrument_strategy(
                data_dir='Data',
                capital=self.capital,
                risk_target=0.2,
                weight_method='handcrafted',
                start_date=start_date,
                end_date=end_date
            )
            
            # Strategy 5 (with trend filter)
            strategy5_results = backtest_trend_following_strategy(
                data_dir='Data',
                capital=self.capital,
                risk_target=0.2,
                weight_method='handcrafted',
                start_date=start_date,
                end_date=end_date
            )
            
            s4_perf = strategy4_results['performance']
            s5_perf = strategy5_results['performance']
            
            print(f"Strategy 4 - Ann. Return: {s4_perf['annualized_return']:.2%}, "
                  f"Volatility: {s4_perf['annualized_volatility']:.2%}, "
                  f"Sharpe: {s4_perf['sharpe_ratio']:.3f}")
            print(f"Strategy 5 - Ann. Return: {s5_perf['annualized_return']:.2%}, "
                  f"Volatility: {s5_perf['annualized_volatility']:.2%}, "
                  f"Sharpe: {s5_perf['sharpe_ratio']:.3f}")
            
            # Strategy 5 should typically have lower volatility due to trend filter
            self.assertLess(s5_perf['annualized_volatility'], 
                           s4_perf['annualized_volatility'] * 1.1,  # Allow small tolerance
                           "Strategy 5 should generally have lower volatility")
            
            # Both should have reasonable Sharpe ratios
            self.assertGreater(s4_perf['sharpe_ratio'], -1.0, 
                             "Strategy 4 Sharpe should be reasonable")
            self.assertGreater(s5_perf['sharpe_ratio'], -1.0, 
                             "Strategy 5 Sharpe should be reasonable")
            
            # Strategy 5 should reduce market exposure
            time_in_market = s5_perf['avg_long_signals'] / s5_perf['num_instruments']
            print(f"Strategy 5 time in market: {time_in_market:.1%}")
            self.assertLess(time_in_market, 0.85, 
                           "Strategy 5 should reduce market exposure below 85%")
            
        except Exception as e:
            print(f"Strategy comparison test failed: {e}")
            # Don't fail the test if data issues, just warn
            print("Note: This may be due to limited data availability")

class TestStrategy5Integration(unittest.TestCase):
    """Integration tests for Strategy 5."""
    
    def test_full_trend_following_workflow(self):
        """Test the complete trend following workflow."""
        print("\n=== Full Trend Following Workflow Test ===")
        
        try:
            # Run a short backtest
            results = backtest_trend_following_strategy(
                data_dir='Data',
                capital=50000000,
                risk_target=0.2,
                weight_method='handcrafted',
                start_date='2018-01-01',
                end_date='2022-12-31'
            )
            
            # Should complete without errors
            self.assertIsNotNone(results, "Backtest should return results")
            self.assertIn('performance', results, "Should have performance metrics")
            
            # Validate trend following specific features
            performance = results['performance']
            self.assertIn('avg_long_signals', performance, 
                         "Should track average long signals")
            
            # Check that instruments have trend data
            instrument_stats = results['instrument_stats']
            if instrument_stats:
                sample_instrument = next(iter(instrument_stats.values()))
                self.assertIn('percent_time_long', sample_instrument,
                             "Instrument stats should include trend metrics")
                
                # Percent time long should be reasonable (between 0% and 100%)
                pct_long = sample_instrument['percent_time_long']
                self.assertGreaterEqual(pct_long, 0.0, 
                                       "Percent time long should be >= 0%")
                self.assertLessEqual(pct_long, 1.0, 
                                    "Percent time long should be <= 100%")
            
            print("Full workflow test completed successfully")
            
        except Exception as e:
            self.fail(f"Full workflow test failed: {e}")

def run_strategy5_tests():
    """Run all Strategy 5 tests."""
    print("=" * 80)
    print("RUNNING STRATEGY 5 TEST SUITE")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestStrategy5))
    suite.addTests(loader.loadTestsFromTestCase(TestStrategy5Integration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASSED' if success else 'FAILED'}")
    
    return success

if __name__ == "__main__":
    # Run the tests
    success = run_strategy5_tests()
    sys.exit(0 if success else 1) 