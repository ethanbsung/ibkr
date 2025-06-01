import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import pandas as pd
from chapter7 import *

class TestChapter7Strategy(unittest.TestCase):
    """Test Strategy 7: Forecast Trend Following implementation."""
    
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

    def test_raw_forecast_calculation(self):
        """Test raw forecast calculation."""
        print("\n=== Testing Raw Forecast Calculation ===")
        
        # Test with trending up prices
        raw_forecast = calculate_raw_forecast(self.trending_up_prices)
        
        # Should have positive values for uptrend
        positive_forecasts = (raw_forecast > 0).sum()
        total_forecasts = len(raw_forecast.dropna())
        
        print(f"Uptrend: {positive_forecasts}/{total_forecasts} positive forecasts")
        self.assertGreater(positive_forecasts / total_forecasts, 0.4, 
                          "Uptrend should have mostly positive forecasts")
        
        # Test with trending down prices
        raw_forecast_down = calculate_raw_forecast(self.trending_down_prices)
        negative_forecasts = (raw_forecast_down < 0).sum()
        total_forecasts_down = len(raw_forecast_down.dropna())
        
        print(f"Downtrend: {negative_forecasts}/{total_forecasts_down} negative forecasts")
        self.assertGreater(negative_forecasts / total_forecasts_down, 0.4,
                          "Downtrend should have mostly negative forecasts")
        
        print("✓ Raw forecast calculation tests passed")

    def test_scaled_forecast_calculation(self):
        """Test scaled forecast calculation."""
        print("\n=== Testing Scaled Forecast Calculation ===")
        
        # Create test raw forecast
        raw_forecast = pd.Series([1.0, -2.0, 3.0, -0.5, 0.0])
        
        # Test with default scalar (1.9)
        scaled_forecast = calculate_scaled_forecast(raw_forecast)
        expected = raw_forecast * 1.9
        
        pd.testing.assert_series_equal(scaled_forecast, expected)
        print(f"Scaled forecast: {scaled_forecast.tolist()}")
        
        # Test with custom scalar
        custom_scalar = 2.5
        scaled_custom = calculate_scaled_forecast(raw_forecast, custom_scalar)
        expected_custom = raw_forecast * custom_scalar
        
        pd.testing.assert_series_equal(scaled_custom, expected_custom)
        print(f"Custom scaled forecast: {scaled_custom.tolist()}")
        
        print("✓ Scaled forecast calculation tests passed")

    def test_capped_forecast_calculation(self):
        """Test capped forecast calculation."""
        print("\n=== Testing Capped Forecast Calculation ===")
        
        # Create test scaled forecast with extreme values
        scaled_forecast = pd.Series([25.0, -30.0, 15.0, -10.0, 0.0])
        
        # Test with default cap (20.0)
        capped_forecast = calculate_capped_forecast(scaled_forecast)
        expected = pd.Series([20.0, -20.0, 15.0, -10.0, 0.0])
        
        pd.testing.assert_series_equal(capped_forecast, expected)
        print(f"Capped forecast: {capped_forecast.tolist()}")
        
        # Test with custom cap
        custom_cap = 10.0
        capped_custom = calculate_capped_forecast(scaled_forecast, custom_cap)
        expected_custom = pd.Series([10.0, -10.0, 10.0, -10.0, 0.0])
        
        pd.testing.assert_series_equal(capped_custom, expected_custom)
        print(f"Custom capped forecast: {capped_custom.tolist()}")
        
        print("✓ Capped forecast calculation tests passed")

    def test_complete_forecast_pipeline(self):
        """Test complete forecast calculation pipeline."""
        print("\n=== Testing Complete Forecast Pipeline ===")
        
        # Test with trending up prices
        forecast = calculate_forecast_for_instrument(self.trending_up_prices)
        
        # Check that forecast is within expected bounds
        self.assertTrue(forecast.min() >= -20.0, "Forecast should be >= -20")
        self.assertTrue(forecast.max() <= 20.0, "Forecast should be <= 20")
        
        # Check that we have reasonable number of non-zero forecasts
        non_zero_forecasts = (forecast.abs() > 0.1).sum()
        total_forecasts = len(forecast.dropna())
        
        print(f"Non-zero forecasts: {non_zero_forecasts}/{total_forecasts}")
        self.assertGreater(non_zero_forecasts / total_forecasts, 0.5,
                          "Should have reasonable number of non-zero forecasts")
        
        print("✓ Complete forecast pipeline tests passed")

    def test_strategy7_position_sizing(self):
        """Test Strategy 7 position sizing calculation."""
        print("\n=== Testing Strategy 7 Position Sizing ===")
        
        # Test normal case with positive forecast
        position = calculate_strategy7_position_size(
            symbol='MES', capital=50000000, weight=0.02, idm=2.5, 
            price=4500, volatility=0.16, multiplier=5, forecast=10.0, risk_target=0.2
        )
        
        # Calculate expected position
        expected = (10.0 * 50000000 * 2.5 * 0.02 * 0.2) / (10 * 5 * 4500 * 1.0 * 0.16)
        print(f"Positive forecast position: {position:.2f}, expected: {expected:.2f}")
        self.assertAlmostEqual(position, expected, places=2)
        
        # Test with negative forecast (short position)
        position_short = calculate_strategy7_position_size(
            'MES', 50000000, 0.02, 2.5, 4500, 0.16, 5, -10.0, 0.2
        )
        expected_short = (-10.0 * 50000000 * 2.5 * 0.02 * 0.2) / (10 * 5 * 4500 * 1.0 * 0.16)
        print(f"Negative forecast position: {position_short:.2f}, expected: {expected_short:.2f}")
        self.assertAlmostEqual(position_short, expected_short, places=2)
        
        # Test with zero forecast
        position_zero = calculate_strategy7_position_size(
            'MES', 50000000, 0.02, 2.5, 4500, 0.16, 5, 0.0, 0.2
        )
        print(f"Zero forecast position: {position_zero}")
        self.assertEqual(position_zero, 0.0)
        
        # Test with zero volatility
        position_zero_vol = calculate_strategy7_position_size(
            'MES', 50000000, 0.02, 2.5, 4500, 0.0, 5, 10.0, 0.2
        )
        print(f"Zero volatility position: {position_zero_vol}")
        self.assertEqual(position_zero_vol, 0.0)
        
        print("✓ Strategy 7 position sizing tests passed")

    def test_forecast_scaling_vs_binary_signals(self):
        """Test that forecast scaling provides more nuanced position sizing than binary signals."""
        print("\n=== Testing Forecast vs Binary Signal Comparison ===")
        
        # Test parameters
        symbol = 'MES'
        capital = 50000000
        weight = 0.02
        idm = 2.5
        price = 4500
        volatility = 0.16
        multiplier = 5
        risk_target = 0.2
        
        # Test different forecast strengths
        forecasts = [1.0, 5.0, 10.0, 15.0, 20.0]
        positions = []
        
        for forecast in forecasts:
            position = calculate_strategy7_position_size(
                symbol, capital, weight, idm, price, volatility, 
                multiplier, forecast, risk_target
            )
            positions.append(position)
            print(f"Forecast {forecast:4.1f} -> Position {position:8.2f}")
        
        # Positions should increase with forecast strength
        for i in range(1, len(positions)):
            self.assertGreater(positions[i], positions[i-1],
                             f"Position should increase with forecast strength")
        
        # Compare with binary signal (Strategy 6 equivalent)
        binary_position = calculate_strategy6_position_size(
            symbol, capital, weight, idm, price, volatility, 
            multiplier, 1.0, risk_target  # Binary signal = 1
        )
        
        # Forecast of 10 should give similar position to binary signal
        forecast_10_position = positions[2]  # forecast = 10.0
        ratio = forecast_10_position / binary_position
        print(f"Forecast 10 vs Binary ratio: {ratio:.2f}")
        self.assertAlmostEqual(ratio, 1.0, delta=0.1,
                             msg="Forecast of 10 should be similar to binary signal")
        
        print("✓ Forecast scaling vs binary signal tests passed")

    def test_backtest_functionality(self):
        """Test that the backtest runs without errors."""
        print("\n=== Testing Backtest Functionality ===")
        
        try:
            # Run a short backtest
            results = backtest_forecast_trend_strategy(
                data_dir='Data',
                capital=self.capital,
                risk_target=0.2,
                weight_method='handcrafted',
                start_date='2020-01-01',
                end_date='2023-12-31'
            )
            
            # Check that results contain expected keys
            required_keys = ['portfolio_data', 'performance', 'instrument_stats', 'config']
            for key in required_keys:
                self.assertIn(key, results, f"Results should contain {key}")
            
            # Check performance metrics
            performance = results['performance']
            self.assertIsInstance(performance['annualized_return'], float)
            self.assertIsInstance(performance['sharpe_ratio'], float)
            self.assertIsInstance(performance['max_drawdown_pct'], float)
            
            # Check forecast-specific metrics
            self.assertIn('avg_forecast', performance)
            self.assertIn('avg_abs_forecast', performance)
            
            print(f"Backtest completed successfully")
            print(f"  Annualized Return: {performance['annualized_return']:.2%}")
            print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
            print(f"  Average Forecast: {performance['avg_forecast']:.2f}")
            print(f"  Average Absolute Forecast: {performance['avg_abs_forecast']:.2f}")
            
            print("✓ Backtest functionality tests passed")
            
        except Exception as e:
            self.fail(f"Backtest failed with error: {e}")

    def test_performance_metrics(self):
        """Test that performance metrics are calculated correctly."""
        print("\n=== Testing Performance Metrics ===")
        
        # Create simple test data with proper date index
        dates = pd.date_range('2020-01-01', periods=250, freq='D')
        test_returns = pd.Series([0.01, -0.005, 0.015, -0.01, 0.008] * 50, index=dates)  # 250 days
        test_capital = 1000000
        
        account_curve = build_account_curve(test_returns, test_capital)
        performance = calculate_comprehensive_performance(account_curve, test_returns)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'total_return', 'annualized_return', 'annualized_volatility',
            'sharpe_ratio', 'max_drawdown_pct', 'skewness'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, performance, f"Performance should include {metric}")
            self.assertIsInstance(performance[metric], (int, float), 
                                f"{metric} should be numeric")
        
        print("✓ Performance metrics tests passed")

def run_tests():
    """Run all Strategy 7 tests."""
    print("=" * 60)
    print("RUNNING STRATEGY 7 UNIT TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestChapter7Strategy)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("ALL STRATEGY 7 TESTS PASSED ✓")
    else:
        print("SOME STRATEGY 7 TESTS FAILED ✗")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    print("=" * 60)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_tests() 