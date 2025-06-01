#!/usr/bin/env python3
"""
Test and debug the instrument selection implementation
"""

import sys
sys.path.append('rob_port')

from rob_port.instr_select import *
from chapter1 import load_instrument_data
import pandas as pd
import numpy as np

def test_minimum_capital_check():
    """Test which instruments meet minimum capital requirements"""
    print("=" * 80)
    print("TESTING MINIMUM CAPITAL REQUIREMENTS")
    print("=" * 80)
    
    instruments_df = load_instrument_data()
    instrument_config = create_instrument_config_from_instruments_df(instruments_df)
    
    capital = 1000000
    risk_target = 0.2
    approx_IDM = 2.5
    approx_number_of_instruments = 5
    approx_initial_weight = 1.0 / approx_number_of_instruments
    
    print(f"Testing with capital: ${capital:,.0f}")
    print(f"Approx weight: {approx_initial_weight:.3f}")
    print(f"Approx IDM: {approx_IDM}")
    
    passed_instruments = []
    failed_instruments = []
    
    for instrument_code in instrument_config.index[:20]:  # Test first 20
        try:
            okay = minimum_capital_okay_for_instrument(
                instrument_code=instrument_code,
                instrument_config=instrument_config,
                capital=capital,
                weight=approx_initial_weight,
                idm=approx_IDM,
                risk_target=risk_target
            )
            
            config = instrument_config.loc[instrument_code]
            min_cap = minimum_capital_for_sub_strategy(
                fx=config.fx_rate,
                idm=approx_IDM,
                weight=approx_initial_weight,
                instrument_risk_ann_perc=config.ann_std,
                price=config.price,
                multiplier=config.multiplier,
                risk_target=risk_target
            )
            
            if okay:
                passed_instruments.append((instrument_code, min_cap))
                print(f"✓ {instrument_code}: Min capital ${min_cap:,.0f}")
            else:
                failed_instruments.append((instrument_code, min_cap))
                print(f"✗ {instrument_code}: Min capital ${min_cap:,.0f} (FAILED)")
                
        except Exception as e:
            print(f"✗ {instrument_code}: ERROR - {e}")
            
    print(f"\nPassed: {len(passed_instruments)} instruments")
    print(f"Failed: {len(failed_instruments)} instruments")
    
    return passed_instruments

def test_cost_calculation():
    """Test risk-adjusted cost calculations"""
    print("\n" + "=" * 80)
    print("TESTING RISK-ADJUSTED COSTS")
    print("=" * 80)
    
    instruments_df = load_instrument_data()
    instrument_config = create_instrument_config_from_instruments_df(instruments_df)
    position_turnover = 5
    
    costs = []
    
    for instrument_code in instrument_config.index[:10]:
        try:
            cost = risk_adjusted_cost_for_instrument(
                instrument_code=instrument_code,
                instrument_config=instrument_config,
                position_turnover=position_turnover
            )
            costs.append((instrument_code, cost))
            print(f"{instrument_code}: Cost {cost:.4f}")
            
        except Exception as e:
            print(f"{instrument_code}: ERROR - {e}")
    
    # Sort by cost
    costs.sort(key=lambda x: x[1])
    print(f"\nCheapest instruments:")
    for inst, cost in costs[:5]:
        print(f"  {inst}: {cost:.4f}")
    
    return costs

def test_correlation_matrix():
    """Test correlation matrix creation"""
    print("\n" + "=" * 80)
    print("TESTING CORRELATION MATRIX")
    print("=" * 80)
    
    # Test with a few instruments that have data
    test_instruments = ['ES', 'NQ', 'YM']  # Common US equity futures
    
    print(f"Testing correlation matrix for: {test_instruments}")
    
    try:
        corr_matrix = create_correlation_matrix_from_data(test_instruments)
        print(f"Correlation matrix shape: {corr_matrix.values.shape}")
        print("Correlation matrix:")
        print(corr_matrix.as_pd())
        
        # Test portfolio weights calculation
        portfolio_weights = calculate_portfolio_weights(test_instruments, corr_matrix)
        print(f"\nPortfolio weights: {dict(portfolio_weights)}")
        
        # Test IDM calculation
        idm = calculate_idm(portfolio_weights, corr_matrix)
        print(f"IDM: {idm:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")

def test_sr_calculation():
    """Test Sharpe ratio calculation for instruments"""
    print("\n" + "=" * 80)
    print("TESTING SHARPE RATIO CALCULATION")
    print("=" * 80)
    
    instruments_df = load_instrument_data()
    instrument_config = create_instrument_config_from_instruments_df(instruments_df)
    
    # Get some instruments that passed minimum capital
    passed_instruments = test_minimum_capital_check()
    
    if not passed_instruments:
        print("No instruments passed minimum capital test!")
        return
    
    # Test SR calculation for first few instruments
    test_instruments = [inst[0] for inst in passed_instruments[:3]]
    
    capital = 1000000
    risk_target = 0.2
    pre_cost_SR = 0.4
    position_turnover = 5
    
    for i, instruments in enumerate([[test_instruments[0]], test_instruments[:2], test_instruments]):
        if len(instruments) > len(test_instruments):
            break
            
        print(f"\nTesting SR for {len(instruments)} instruments: {instruments}")
        
        try:
            correlation_matrix = create_correlation_matrix_from_data(instruments)
            
            sr = calculate_SR_for_selected_instruments(
                selected_instruments=instruments,
                pre_cost_SR=pre_cost_SR,
                instrument_config=instrument_config,
                position_turnover=position_turnover,
                correlation_matrix=correlation_matrix,
                capital=capital,
                risk_target=risk_target
            )
            
            print(f"SR: {sr:.4f}")
            
            if sr < -999999:
                print("Failed minimum capital check!")
            
        except Exception as e:
            print(f"Error: {e}")

def test_full_selection_debug():
    """Test full selection with debug output"""
    print("\n" + "=" * 80)
    print("TESTING FULL SELECTION WITH DEBUG")
    print("=" * 80)
    
    instruments_df = load_instrument_data()
    
    # Test with higher capital to see if we get more instruments
    for capital in [10000000, 50000000]:
        print(f"\n--- Testing with capital: ${capital:,.0f} ---")
        
        try:
            results = implement_carver_static_instrument_selection(
                instruments_df=instruments_df,
                capital=capital,
                risk_target=0.2,
                pre_cost_SR=0.4,
                position_turnover=5,
                approx_number_of_instruments=5,
                approx_IDM=2.5
            )
            
            if results and len(results['selected_instruments']) > 1:
                print(f"Success! Selected {len(results['selected_instruments'])} instruments")
                break
            else:
                print(f"Only selected {len(results.get('selected_instruments', []))} instruments")
                
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Run all tests"""
    print("INSTRUMENT SELECTION TESTING AND DEBUGGING")
    print("=" * 80)
    
    # Test each component
    test_minimum_capital_check()
    test_cost_calculation()
    test_correlation_matrix()
    test_sr_calculation()
    test_full_selection_debug()

if __name__ == "__main__":
    main() 