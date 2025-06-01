#!/usr/bin/env python3
"""
Summary test of the instrument selection implementation
"""

import sys
sys.path.append('rob_port')
from rob_port.instr_select import implement_carver_static_instrument_selection
from chapter1 import load_instrument_data

def main():
    print('=== CARVER STATIC INSTRUMENT SELECTION SUMMARY ===')
    instruments_df = load_instrument_data()

    for capital in [1000000, 5000000]:
        print(f'\nCapital: ${capital:,.0f}')
        results = implement_carver_static_instrument_selection(
            instruments_df=instruments_df,
            capital=capital,
            risk_target=0.2,
            pre_cost_SR=0.4,
            position_turnover=5,
            approx_number_of_instruments=5,
            approx_IDM=2.5
        )
        
        if results:
            print(f'Selected instruments: {len(results["selected_instruments"])}')
            print(f'Final SR: {results["final_SR"]:.4f}')
            print(f'IDM: {results["idm"]:.2f}')
            print(f'Top instruments: {results["selected_instruments"][:5]}')

if __name__ == "__main__":
    main() 