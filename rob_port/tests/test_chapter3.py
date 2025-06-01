import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chapter3 import *
from chapter4 import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class Chapter3TestSuite:
    """
    Comprehensive test suite for Chapter 3 variable risk scaling strategy.
    Tests against book's expected results and validates across different time periods.
    """
    
    def __init__(self, capital=50000000):
        self.capital = capital
        self.instruments_df = load_instrument_data()
        self.book_expected_results = {
            'strategy_two': {
                'mean_annual_return': 0.121,
                'annual_costs': -0.0004,
                'average_drawdown': -0.169,
                'standard_deviation': 0.250,
                'sharpe_ratio': 0.48,
                'skew': -0.47,
                'lower_tail': 2.21,
                'upper_tail': 1.79
            },
            'strategy_three': {
                'mean_annual_return': 0.122,
                'annual_costs': -0.0006,
                'average_drawdown': -0.188,
                'standard_deviation': 0.228,
                'sharpe_ratio': 0.54,
                'skew': -0.68,
                'lower_tail': 1.76,
                'upper_tail': 1.21
            }
        }
        
    def test_mes_single_instrument(self, start_date='2000-01-01', end_date='2024-12-31'):
        """
        Test MES futures with variable risk scaling (Strategy 2: Fixed risk estimate).
        """
        print(f"\n{'='*80}")
        print(f"TESTING MES FUTURES - STRATEGY 2 (Fixed Risk Estimate)")
        print(f"Period: {start_date} to {end_date}")
        print(f"{'='*80}")
        
        try:
            # Load MES data
            mes_data = pd.read_csv('Data/mes_daily_data.csv', parse_dates=['Time'])
            mes_data.set_index('Time', inplace=True)
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            mes_data = mes_data[(mes_data.index >= start_dt) & (mes_data.index <= end_dt)]
            
            if mes_data.empty:
                print(f"No data available for period {start_date} to {end_date}")
                return None
                
            print(f"Data period: {mes_data.index[0].date()} to {mes_data.index[-1].date()}")
            print(f"Total trading days: {len(mes_data)}")
            
            # Calculate returns
            mes_data['returns'] = mes_data['Last'].pct_change()
            mes_data = mes_data.dropna()
            
            # Strategy 2: Fixed risk estimate (use long-term average volatility)
            long_term_vol = mes_data['returns'].std() * np.sqrt(business_days_per_year)
            print(f"Long-term volatility estimate: {long_term_vol:.1%}")
            
            # Get MES specs
            mes_specs = get_instrument_specs('MES', self.instruments_df)
            multiplier = mes_specs['multiplier']
            
            # Calculate fixed position size
            avg_price = mes_data['Last'].mean()
            fixed_position = calculate_variable_position_size(
                self.capital, multiplier, avg_price, long_term_vol, 0.2
            )
            
            print(f"Average price: ${avg_price:.2f}")
            print(f"Fixed position size: {fixed_position:.2f} contracts")
            print(f"Multiplier: {multiplier}")
            
            # Calculate strategy returns with fixed position
            mes_data['position'] = fixed_position
            mes_data['notional_exposure'] = mes_data['position'] * multiplier * mes_data['Last'].shift(1)
            mes_data['strategy_pnl'] = mes_data['position'] * multiplier * mes_data['returns'] * mes_data['Last'].shift(1)
            mes_data['strategy_returns'] = mes_data['strategy_pnl'] / self.capital
            
            # Remove NaN values
            mes_data = mes_data.dropna()
            
            if mes_data.empty:
                print("No valid data after processing")
                return None
            
            # Calculate performance metrics
            performance = calculate_comprehensive_performance(
                build_account_curve(mes_data['strategy_returns'], self.capital),
                mes_data['strategy_returns']
            )
            
            # Display results
            self._display_strategy_results(performance, 'strategy_two', 'MES Fixed Risk')
            
            # Calculate costs
            annual_cost = calculate_annual_risk_adjusted_cost(
                avg_price, multiplier, mes_specs.get('commission', 0.62), 
                0.25, long_term_vol, 4, 6
            )
            print(f"Annual risk-adjusted cost: {annual_cost:.6f} SR units")
            
            return {
                'performance': performance,
                'data': mes_data,
                'annual_cost': annual_cost,
                'position_size': fixed_position,
                'volatility': long_term_vol
            }
            
        except Exception as e:
            print(f"Error in MES test: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_mes_variable_risk(self, start_date='2000-01-01', end_date='2024-12-31'):
        """
        Test MES futures with variable risk scaling (Strategy 3: Variable risk estimate).
        """
        print(f"\n{'='*80}")
        print(f"TESTING MES FUTURES - STRATEGY 3 (Variable Risk Estimate)")
        print(f"Period: {start_date} to {end_date}")
        print(f"{'='*80}")
        
        try:
            # Use the existing backtest function
            results = backtest_variable_risk_strategy(
                'Data/mes_daily_data.csv', 
                self.capital, 
                risk_target=0.2,
                short_span=32,
                long_years=10
            )
            
            if 'error' in results:
                print(f"Error in backtest: {results['error']}")
                return None
            
            # Filter by date range if needed
            data = results['data']
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            data = data[(data.index >= start_dt) & (data.index <= end_dt)]
            
            if data.empty:
                print(f"No data available for period {start_date} to {end_date}")
                return None
            
            print(f"Data period: {data.index[0].date()} to {data.index[-1].date()}")
            print(f"Total trading days: {len(data)}")
            
            # Recalculate performance for filtered period
            performance = calculate_comprehensive_performance(
                build_account_curve(data['strategy_returns'], self.capital),
                data['strategy_returns']
            )
            
            # Display results
            self._display_strategy_results(performance, 'strategy_three', 'MES Variable Risk')
            
            # Additional variable risk metrics
            print(f"\nVariable Risk Metrics:")
            print(f"  Average Position: {data['position'].mean():.2f} contracts")
            print(f"  Max Position: {data['position'].max():.2f} contracts")
            print(f"  Min Position: {data['position'].min():.2f} contracts")
            print(f"  Position Std Dev: {data['position'].std():.2f}")
            print(f"  Average Volatility: {data['blended_vol'].mean():.1%}")
            print(f"  Min Volatility: {data['blended_vol'].min():.1%}")
            print(f"  Max Volatility: {data['blended_vol'].max():.1%}")
            
            return {
                'performance': performance,
                'data': data,
                'position_stats': {
                    'mean': data['position'].mean(),
                    'max': data['position'].max(),
                    'min': data['position'].min(),
                    'std': data['position'].std()
                },
                'volatility_stats': {
                    'mean': data['blended_vol'].mean(),
                    'min': data['blended_vol'].min(),
                    'max': data['blended_vol'].max()
                }
            }
            
        except Exception as e:
            print(f"Error in MES variable risk test: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_jumbo_portfolio(self, start_date='2000-01-01', end_date='2024-12-31', max_instruments=50):
        """
        Test jumbo portfolio with variable risk scaling.
        """
        print(f"\n{'='*80}")
        print(f"TESTING JUMBO PORTFOLIO - Variable Risk Scaling")
        print(f"Period: {start_date} to {end_date}")
        print(f"Max Instruments: {max_instruments}")
        print(f"{'='*80}")
        
        try:
            # Get available instruments
            available_instruments = get_available_instruments(self.instruments_df)
            print(f"Available instruments: {len(available_instruments)}")
            
            # Load data for available instruments
            data = load_instrument_data_files(available_instruments[:max_instruments])
            print(f"Loaded data for: {len(data)} instruments")
            
            if len(data) == 0:
                print("No data loaded for jumbo portfolio")
                return None
            
            # Create jumbo portfolio
            jumbo_weights = create_jumbo_portfolio(self.instruments_df, data, max_instruments=max_instruments)
            
            if not jumbo_weights:
                print("Could not create jumbo portfolio")
                return None
            
            print(f"Jumbo portfolio instruments: {len(jumbo_weights)}")
            print(f"Instruments: {list(jumbo_weights.keys())}")
            
            # Backtest the jumbo portfolio
            results = backtest_portfolio_with_individual_data(
                jumbo_weights, data, self.instruments_df, 
                capital=self.capital, risk_target=0.2,
                start_date=start_date, end_date=end_date
            )
            
            if 'error' in results:
                print(f"Error in jumbo portfolio backtest: {results['error']}")
                return None
            
            performance = results['performance']
            
            # Display results
            print(f"\nJumbo Portfolio Performance:")
            print(f"  Annual Return: {performance['annualized_return']:.2%}")
            print(f"  Volatility: {performance['annualized_volatility']:.2%}")
            print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {performance['max_drawdown_pct']:.1f}%")
            print(f"  Skewness: {performance['skewness']:.3f}")
            print(f"  IDM: {results.get('idm', 'N/A'):.2f}")
            print(f"  Number of Instruments: {performance['num_instruments']}")
            
            return {
                'performance': performance,
                'weights': jumbo_weights,
                'idm': results.get('idm', 1.0),
                'num_instruments': performance['num_instruments']
            }
            
        except Exception as e:
            print(f"Error in jumbo portfolio test: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_different_time_periods(self):
        """
        Test strategies across different time periods to validate consistency.
        """
        print(f"\n{'='*80}")
        print(f"TESTING DIFFERENT TIME PERIODS")
        print(f"{'='*80}")
        
        # Define test periods
        periods = [
            ('2000-01-01', '2005-12-31', 'Early 2000s'),
            ('2006-01-01', '2010-12-31', '2006-2010 (Crisis)'),
            ('2011-01-01', '2015-12-31', '2011-2015'),
            ('2016-01-01', '2020-12-31', '2016-2020'),
            ('2021-01-01', '2024-12-31', '2021-2024'),
            ('2000-01-01', '2024-12-31', 'Full Period')
        ]
        
        results_summary = []
        
        for start_date, end_date, period_name in periods:
            print(f"\n--- Testing Period: {period_name} ---")
            
            # Test MES fixed risk
            mes_fixed = self.test_mes_single_instrument(start_date, end_date)
            
            # Test MES variable risk
            mes_variable = self.test_mes_variable_risk(start_date, end_date)
            
            if mes_fixed and mes_variable:
                results_summary.append({
                    'period': period_name,
                    'start_date': start_date,
                    'end_date': end_date,
                    'mes_fixed_return': mes_fixed['performance']['annualized_return'],
                    'mes_fixed_vol': mes_fixed['performance']['annualized_volatility'],
                    'mes_fixed_sharpe': mes_fixed['performance']['sharpe_ratio'],
                    'mes_variable_return': mes_variable['performance']['annualized_return'],
                    'mes_variable_vol': mes_variable['performance']['annualized_volatility'],
                    'mes_variable_sharpe': mes_variable['performance']['sharpe_ratio']
                })
        
        # Display summary
        if results_summary:
            self._display_period_summary(results_summary)
        
        return results_summary
    
    def _display_strategy_results(self, performance, strategy_key, strategy_name):
        """Display strategy results compared to book expectations."""
        print(f"\n{strategy_name} Results:")
        print(f"  Annual Return: {performance['annualized_return']:.2%}")
        print(f"  Volatility: {performance['annualized_volatility']:.2%}")
        print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {performance['max_drawdown_pct']:.1f}%")
        print(f"  Skewness: {performance['skewness']:.3f}")
        
        # Compare to book if available
        if strategy_key in self.book_expected_results:
            expected = self.book_expected_results[strategy_key]
            print(f"\nComparison to Book:")
            print(f"  Return - Expected: {expected['mean_annual_return']:.1%}, Actual: {performance['annualized_return']:.1%}")
            print(f"  Volatility - Expected: {expected['standard_deviation']:.1%}, Actual: {performance['annualized_volatility']:.1%}")
            print(f"  Sharpe - Expected: {expected['sharpe_ratio']:.2f}, Actual: {performance['sharpe_ratio']:.2f}")
            print(f"  Skew - Expected: {expected['skew']:.2f}, Actual: {performance['skewness']:.2f}")
            
            # Calculate differences
            return_diff = abs(performance['annualized_return'] - expected['mean_annual_return'])
            vol_diff = abs(performance['annualized_volatility'] - expected['standard_deviation'])
            sharpe_diff = abs(performance['sharpe_ratio'] - expected['sharpe_ratio'])
            
            print(f"\nDifferences:")
            print(f"  Return difference: {return_diff:.1%}")
            print(f"  Volatility difference: {vol_diff:.1%}")
            print(f"  Sharpe difference: {sharpe_diff:.2f}")
            
            # Validation flags
            return_ok = return_diff < 0.05  # Within 5%
            vol_ok = vol_diff < 0.05       # Within 5%
            sharpe_ok = sharpe_diff < 0.2   # Within 0.2
            
            print(f"\nValidation:")
            print(f"  Return within 5%: {'✓' if return_ok else '✗'}")
            print(f"  Volatility within 5%: {'✓' if vol_ok else '✗'}")
            print(f"  Sharpe within 0.2: {'✓' if sharpe_ok else '✗'}")
            print(f"  Overall: {'✓ PASS' if all([return_ok, vol_ok, sharpe_ok]) else '✗ FAIL'}")
    
    def _display_period_summary(self, results_summary):
        """Display summary of results across different periods."""
        print(f"\n{'='*80}")
        print(f"SUMMARY ACROSS TIME PERIODS")
        print(f"{'='*80}")
        
        df = pd.DataFrame(results_summary)
        
        print(f"\nMES Fixed Risk Strategy:")
        print(f"  Average Return: {df['mes_fixed_return'].mean():.1%}")
        print(f"  Average Volatility: {df['mes_fixed_vol'].mean():.1%}")
        print(f"  Average Sharpe: {df['mes_fixed_sharpe'].mean():.2f}")
        print(f"  Return Range: {df['mes_fixed_return'].min():.1%} to {df['mes_fixed_return'].max():.1%}")
        
        print(f"\nMES Variable Risk Strategy:")
        print(f"  Average Return: {df['mes_variable_return'].mean():.1%}")
        print(f"  Average Volatility: {df['mes_variable_vol'].mean():.1%}")
        print(f"  Average Sharpe: {df['mes_variable_sharpe'].mean():.2f}")
        print(f"  Return Range: {df['mes_variable_return'].min():.1%} to {df['mes_variable_return'].max():.1%}")
        
        print(f"\nPeriod-by-Period Results:")
        for _, row in df.iterrows():
            print(f"  {row['period']:<20} | Fixed: {row['mes_fixed_return']:>6.1%} ({row['mes_fixed_sharpe']:>4.2f}) | Variable: {row['mes_variable_return']:>6.1%} ({row['mes_variable_sharpe']:>4.2f})")
    
    def run_comprehensive_tests(self):
        """
        Run all tests in the comprehensive test suite.
        """
        print(f"{'='*80}")
        print(f"CHAPTER 3 COMPREHENSIVE TEST SUITE")
        print(f"Capital: ${self.capital:,.0f}")
        print(f"{'='*80}")
        
        # Test 1: MES Fixed Risk (Strategy 2)
        mes_fixed_results = self.test_mes_single_instrument()
        
        # Test 2: MES Variable Risk (Strategy 3)
        mes_variable_results = self.test_mes_variable_risk()
        
        # Test 3: Jumbo Portfolio
        jumbo_results = self.test_jumbo_portfolio()
        
        # Test 4: Different Time Periods
        period_results = self.test_different_time_periods()
        
        # Final Summary
        print(f"\n{'='*80}")
        print(f"FINAL TEST SUMMARY")
        print(f"{'='*80}")
        
        if mes_fixed_results:
            print(f"✓ MES Fixed Risk Strategy completed successfully")
        else:
            print(f"✗ MES Fixed Risk Strategy failed")
            
        if mes_variable_results:
            print(f"✓ MES Variable Risk Strategy completed successfully")
        else:
            print(f"✗ MES Variable Risk Strategy failed")
            
        if jumbo_results:
            print(f"✓ Jumbo Portfolio Strategy completed successfully")
        else:
            print(f"✗ Jumbo Portfolio Strategy failed")
            
        if period_results:
            print(f"✓ Time Period Analysis completed successfully")
        else:
            print(f"✗ Time Period Analysis failed")
        
        return {
            'mes_fixed': mes_fixed_results,
            'mes_variable': mes_variable_results,
            'jumbo': jumbo_results,
            'periods': period_results
        }

def main():
    """
    Run the comprehensive Chapter 3 test suite.
    """
    test_suite = Chapter3TestSuite(capital=50000000)
    results = test_suite.run_comprehensive_tests()
    return results

if __name__ == "__main__":
    main() 