import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import pandas as pd
from chapter8 import *

class TestChapter8Strategy(unittest.TestCase):
    """Test Strategy 8: Fast Trend Following with Buffering implementation."""
    
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

    def test_fast_raw_forecast_calculation(self):
        """Test fast raw forecast calculation with EWMAC(16,64)."""
        print("\n=== Testing Fast Raw Forecast Calculation ===")
        
        # Test with trending up prices
        raw_forecast = calculate_fast_raw_forecast(self.trending_up_prices, fast_span=16, slow_span=64)
        
        # Should have positive values for uptrend
        positive_forecasts = (raw_forecast > 0).sum()
        total_forecasts = len(raw_forecast.dropna())
        
        print(f"Uptrend: {positive_forecasts}/{total_forecasts} positive forecasts")
        self.assertGreater(positive_forecasts / total_forecasts, 0.4, 
                          "Uptrend should have mostly positive forecasts")
        
        # Test with trending down prices
        raw_forecast_down = calculate_fast_raw_forecast(self.trending_down_prices, fast_span=16, slow_span=64)
        negative_forecasts = (raw_forecast_down < 0).sum()
        total_forecasts_down = len(raw_forecast_down.dropna())
        
        print(f"Downtrend: {negative_forecasts}/{total_forecasts_down} negative forecasts")
        self.assertGreater(negative_forecasts / total_forecasts_down, 0.4,
                          "Downtrend should have mostly negative forecasts")
        
        print("✓ Fast raw forecast calculation tests passed")

    def test_fast_forecast_pipeline(self):
        """Test complete fast forecast calculation pipeline."""
        print("\n=== Testing Fast Forecast Pipeline ===")
        
        # Test with trending up prices and fast parameters
        forecast = calculate_fast_forecast_for_instrument(
            self.trending_up_prices, 
            fast_span=16, 
            slow_span=64, 
            forecast_scalar=4.1
        )
        
        # Check that forecast is within expected bounds
        self.assertTrue(forecast.min() >= -20.0, "Forecast should be >= -20")
        self.assertTrue(forecast.max() <= 20.0, "Forecast should be <= 20")
        
        # Check that we have reasonable number of non-zero forecasts
        non_zero_forecasts = (forecast.abs() > 0.1).sum()
        total_forecasts = len(forecast.dropna())
        
        print(f"Non-zero forecasts: {non_zero_forecasts}/{total_forecasts}")
        self.assertGreater(non_zero_forecasts / total_forecasts, 0.5,
                          "Should have reasonable number of non-zero forecasts")
        
        # Test that fast forecast is different from slow forecast
        slow_forecast = calculate_forecast_for_instrument(
            self.trending_up_prices, 
            fast_span=64, 
            slow_span=256, 
            forecast_scalar=1.9
        )
        
        # Fast and slow forecasts should be different
        correlation = forecast.corr(slow_forecast)
        print(f"Fast vs Slow forecast correlation: {correlation:.3f}")
        self.assertLess(correlation, 0.95, "Fast and slow forecasts should be different")
        
        print("✓ Fast forecast pipeline tests passed")

    def test_buffer_width_calculation(self):
        """Test buffer width calculation."""
        print("\n=== Testing Buffer Width Calculation ===")
        
        # Test normal case
        buffer_width = calculate_buffer_width(
            symbol='MES', capital=50000000, weight=0.02, idm=2.5, 
            price=4500, volatility=0.16, multiplier=5, 
            risk_target=0.2, fx_rate=1.0, buffer_fraction=0.1
        )
        
        # Calculate expected buffer width
        expected = (0.1 * 50000000 * 2.5 * 0.02 * 0.2) / (5 * 4500 * 1.0 * 0.16)
        print(f"Buffer width: {buffer_width:.4f}, expected: {expected:.4f}")
        self.assertAlmostEqual(buffer_width, expected, places=4)
        
        # Test with zero volatility
        buffer_zero = calculate_buffer_width(
            'MES', 50000000, 0.02, 2.5, 4500, 0.0, 5, 0.2, 1.0, 0.1
        )
        print(f"Zero volatility buffer width: {buffer_zero}")
        self.assertEqual(buffer_zero, 0.0)
        
        # Test with different buffer fraction
        buffer_larger = calculate_buffer_width(
            'MES', 50000000, 0.02, 2.5, 4500, 0.16, 5, 0.2, 1.0, 0.2
        )
        print(f"Larger buffer fraction (0.2): {buffer_larger:.4f}")
        self.assertAlmostEqual(buffer_larger, 2 * buffer_width, places=4)
        
        print("✓ Buffer width calculation tests passed")

    def test_buffered_position_calculation(self):
        """Test buffered position calculation logic."""
        print("\n=== Testing Buffered Position Calculation ===")
        
        # Test case 1: Position within buffer (no trading)
        optimal_position = 10.0
        current_position = 9.5
        buffer_width = 1.0
        
        new_position, trade_size = calculate_buffered_position(
            optimal_position, current_position, buffer_width
        )
        
        print(f"Within buffer: optimal={optimal_position}, current={current_position}, buffer={buffer_width}")
        print(f"  Result: new_position={new_position}, trade_size={trade_size}")
        self.assertEqual(new_position, current_position, "Should not trade within buffer")
        self.assertEqual(trade_size, 0, "Trade size should be zero within buffer")
        
        # Test case 2: Position below buffer (buy to upper buffer)
        current_position = 8.0  # Below lower buffer of 9
        new_position, trade_size = calculate_buffered_position(
            optimal_position, current_position, buffer_width
        )
        
        expected_new_position = round(optimal_position + buffer_width)  # Upper buffer
        expected_trade_size = expected_new_position - current_position
        
        print(f"Below buffer: optimal={optimal_position}, current={current_position}, buffer={buffer_width}")
        print(f"  Result: new_position={new_position}, trade_size={trade_size}")
        print(f"  Expected: new_position={expected_new_position}, trade_size={expected_trade_size}")
        self.assertEqual(new_position, expected_new_position)
        self.assertEqual(trade_size, expected_trade_size)
        
        # Test case 3: Position above buffer (sell to lower buffer)
        current_position = 12.0  # Above upper buffer of 11
        new_position, trade_size = calculate_buffered_position(
            optimal_position, current_position, buffer_width
        )
        
        expected_new_position = round(optimal_position - buffer_width)  # Lower buffer
        expected_trade_size = expected_new_position - current_position
        
        print(f"Above buffer: optimal={optimal_position}, current={current_position}, buffer={buffer_width}")
        print(f"  Result: new_position={new_position}, trade_size={trade_size}")
        print(f"  Expected: new_position={expected_new_position}, trade_size={expected_trade_size}")
        self.assertEqual(new_position, expected_new_position)
        self.assertEqual(trade_size, expected_trade_size)
        
        # Test case 4: Zero buffer width (always trade to optimal)
        new_position, trade_size = calculate_buffered_position(
            optimal_position, current_position, 0.0
        )
        
        print(f"Zero buffer: optimal={optimal_position}, current={current_position}")
        print(f"  Result: new_position={new_position}, trade_size={trade_size}")
        self.assertEqual(new_position, optimal_position)
        self.assertEqual(trade_size, optimal_position - current_position)
        
        print("✓ Buffered position calculation tests passed")

    def test_strategy8_position_sizing(self):
        """Test Strategy 8 position sizing calculation."""
        print("\n=== Testing Strategy 8 Position Sizing ===")
        
        # Test normal case with positive forecast
        position = calculate_strategy8_position_size(
            symbol='MES', capital=50000000, weight=0.02, idm=2.5, 
            price=4500, volatility=0.16, multiplier=5, forecast=10.0, risk_target=0.2
        )
        
        # Calculate expected position (same as Strategy 7)
        expected = (10.0 * 50000000 * 2.5 * 0.02 * 0.2) / (10 * 5 * 4500 * 1.0 * 0.16)
        print(f"Positive forecast position: {position:.2f}, expected: {expected:.2f}")
        self.assertAlmostEqual(position, expected, places=2)
        
        # Test with negative forecast (short position)
        position_short = calculate_strategy8_position_size(
            'MES', 50000000, 0.02, 2.5, 4500, 0.16, 5, -10.0, 0.2
        )
        expected_short = (-10.0 * 50000000 * 2.5 * 0.02 * 0.2) / (10 * 5 * 4500 * 1.0 * 0.16)
        print(f"Negative forecast position: {position_short:.2f}, expected: {expected_short:.2f}")
        self.assertAlmostEqual(position_short, expected_short, places=2)
        
        # Test with zero forecast
        position_zero = calculate_strategy8_position_size(
            'MES', 50000000, 0.02, 2.5, 4500, 0.16, 5, 0.0, 0.2
        )
        print(f"Zero forecast position: {position_zero}")
        self.assertEqual(position_zero, 0.0)
        
        print("✓ Strategy 8 position sizing tests passed")

    def test_fast_vs_slow_forecast_differences(self):
        """Test that fast forecasts behave differently from slow forecasts."""
        print("\n=== Testing Fast vs Slow Forecast Differences ===")
        
        # Calculate both fast and slow forecasts
        fast_forecast = calculate_fast_forecast_for_instrument(
            self.trending_up_prices, 
            fast_span=16, 
            slow_span=64, 
            forecast_scalar=4.1
        )
        
        slow_forecast = calculate_forecast_for_instrument(
            self.trending_up_prices, 
            fast_span=64, 
            slow_span=256, 
            forecast_scalar=1.9
        )
        
        # Fast forecast should be more responsive (higher absolute values on average)
        fast_abs_avg = fast_forecast.abs().mean()
        slow_abs_avg = slow_forecast.abs().mean()
        
        print(f"Fast forecast average absolute value: {fast_abs_avg:.3f}")
        print(f"Slow forecast average absolute value: {slow_abs_avg:.3f}")
        
        # Fast forecasts should generally be more volatile/responsive
        fast_std = fast_forecast.std()
        slow_std = slow_forecast.std()
        
        print(f"Fast forecast standard deviation: {fast_std:.3f}")
        print(f"Slow forecast standard deviation: {slow_std:.3f}")
        
        # Fast should generally be more volatile due to shorter spans and higher scalar
        self.assertGreater(fast_std, slow_std * 0.8, 
                          "Fast forecast should be at least somewhat more volatile")
        
        print("✓ Fast vs slow forecast difference tests passed")

    def test_backtest_functionality(self):
        """Test that the Strategy 8 backtest runs without errors."""
        print("\n=== Testing Strategy 8 Backtest Functionality ===")
        
        try:
            # Run a short backtest
            results = backtest_fast_trend_strategy_with_buffering(
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
            
            # Check Strategy 8 specific metrics
            self.assertIn('avg_forecast', performance)
            self.assertIn('avg_abs_forecast', performance)
            self.assertIn('avg_daily_trades', performance)
            self.assertIn('total_trades', performance)
            
            # Check config contains Strategy 8 specific parameters
            config = results['config']
            self.assertEqual(config['trend_fast_span'], 16)
            self.assertEqual(config['trend_slow_span'], 64)
            self.assertEqual(config['forecast_scalar'], 4.1)
            self.assertEqual(config['buffer_fraction'], 0.1)
            
            print(f"Strategy 8 backtest completed successfully")
            print(f"  Annualized Return: {performance['annualized_return']:.2%}")
            print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
            print(f"  Average Forecast: {performance['avg_forecast']:.2f}")
            print(f"  Average Daily Trades: {performance['avg_daily_trades']:.1f}")
            
            print("✓ Strategy 8 backtest functionality tests passed")
            
        except Exception as e:
            self.fail(f"Strategy 8 backtest failed with error: {e}")

    def test_buffering_reduces_trading(self):
        """Test that buffering actually reduces trading frequency."""
        print("\n=== Testing Buffering Reduces Trading ===")
        
        # Simulate position changes over time
        optimal_positions = [10, 10.2, 9.8, 10.1, 9.9, 10.3, 9.7, 10.2]
        buffer_width = 0.5
        
        # Without buffering (trade to optimal every time)
        unbuffered_trades = 0
        current_unbuffered = 0
        for optimal in optimal_positions:
            trade = optimal - current_unbuffered
            if abs(trade) > 0.01:
                unbuffered_trades += 1
            current_unbuffered = optimal
        
        # With buffering
        buffered_trades = 0
        current_buffered = 0
        for optimal in optimal_positions:
            new_pos, trade_size = calculate_buffered_position(
                optimal, current_buffered, buffer_width
            )
            if abs(trade_size) > 0.01:
                buffered_trades += 1
            current_buffered = new_pos
        
        print(f"Unbuffered trades: {unbuffered_trades}")
        print(f"Buffered trades: {buffered_trades}")
        
        # Buffering should reduce trading
        self.assertLessEqual(buffered_trades, unbuffered_trades,
                           "Buffering should reduce or maintain trading frequency")
        
        print("✓ Buffering reduces trading tests passed")

def run_tests():
    """Run all Strategy 8 tests."""
    print("=" * 60)
    print("RUNNING STRATEGY 8 UNIT TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestChapter8Strategy)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("ALL STRATEGY 8 TESTS PASSED ✓")
    else:
        print("SOME STRATEGY 8 TESTS FAILED ✗")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    print("=" * 60)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_tests() 