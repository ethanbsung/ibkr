#!/usr/bin/env python3

import sys
sys.path.append('.')

from chapter4 import *
import warnings
warnings.filterwarnings('ignore')

def quick_test():
    """Quick test of different cost multipliers."""
    print("=" * 60)
    print("COST MULTIPLIER TEST")
    print("=" * 60)
    
    # Manual data setup
    instruments_df = load_instrument_data('../Data/instruments.csv')
    
    # Load just a few known instruments
    test_symbols = ['MES', 'MYM', 'MNQ', 'FDAX', 'FGBL']  # Mix of equity/bond futures
    data = {}
    
    for symbol in test_symbols:
        try:
            df = load_single_instrument_data(symbol)
            if df is not None and len(df) > 500:  # Need decent amount of data
                data[symbol] = df
                print(f"Loaded {symbol}: {len(df)} rows")
        except:
            pass
    
    if len(data) < 3:
        print("Not enough instruments loaded!")
        return
    
    print(f"\nUsing {len(data)} instruments: {list(data.keys())}")
    
    # Equal weight portfolio
    portfolio_weights = {symbol: 1.0/len(data) for symbol in data.keys()}
    
    # Test cost multipliers
    cost_multipliers = [0, 5, 10, 20, 50]
    
    print(f"\n{'Cost':<6} {'Return':<8} {'Sharpe':<8} {'Turnover':<10}")
    print("-" * 35)
    
    for cost_mult in cost_multipliers:
        try:
            result = backtest_portfolio_with_dynamic_optimization(
                data, portfolio_weights, instruments_df,
                capital=1000000,
                risk_target=0.2,
                cost_multiplier=cost_mult,
                rebalance_frequency='weekly'
            )
            
            ret = result.get('annual_return', 0)
            sharpe = result.get('sharpe_ratio', 0)
            turnover = result.get('annual_turnover', 0)
            
            print(f"{cost_mult:<6} {ret:<7.2%} {sharpe:<7.3f} {turnover:<9.1f}")
            
        except Exception as e:
            print(f"{cost_mult:<6} ERROR: {str(e)[:30]}")
    
    # Static baseline
    try:
        static_result = backtest_portfolio_with_individual_data(
            data, portfolio_weights, instruments_df,
            capital=1000000,
            risk_target=0.2
        )
        ret = static_result.get('annual_return', 0)
        sharpe = static_result.get('sharpe_ratio', 0)
        print(f"{'Static':<6} {ret:<7.2%} {sharpe:<7.3f} {'0':<9}")
    except Exception as e:
        print(f"Static ERROR: {e}")

if __name__ == "__main__":
    quick_test() 