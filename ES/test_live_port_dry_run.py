#!/usr/bin/env python
"""
Dry run test for live_port.py position sizing and order execution logic.
Tests multiple scenarios without connecting to IBKR or placing real orders.
"""

import json
import os
import sys
from datetime import datetime
from collections import defaultdict
import logging

# Add the current directory to path to import from live_port
sys.path.append('.')

# Import functions from live_port
from live_port import (
    allocation_percentages, contract_specs, risk_multiplier,
    ibs_entry_threshold, ibs_exit_threshold,
    williams_period, wr_buy_threshold, wr_sell_threshold,
    calculate_position_size, compute_ibs, compute_williams_r,
    load_portfolio_state, save_portfolio_state
)

# Mock data structures
class MockBar:
    def __init__(self, date, open_price, high, low, close):
        self.date = date
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close

class MockContract:
    def __init__(self, symbol, conId):
        self.symbol = symbol
        self.conId = conId

class MockPositionRouter:
    """Mock version of PositionRouter for dry run testing"""
    def __init__(self, symbol, initial_state=None):
        self.symbol = symbol
        self.strategy_lots = defaultdict(int)
        self.orders_placed = []
        
        if initial_state:
            self.strategy_lots.update(initial_state)
    
    def sync(self, desired_dict):
        """Mock sync that records what orders would be placed"""
        logger = logging.getLogger()
        
        current_total = sum(self.strategy_lots.values())
        desired_total = sum(desired_dict.values())
        net_change = desired_total - current_total
        
        logger.info(f"Router {self.symbol}: Current virtual lots: {dict(self.strategy_lots)}")
        logger.info(f"Router {self.symbol}: Desired lots: {desired_dict}")
        logger.info(f"Router {self.symbol}: Current total: {current_total}")
        logger.info(f"Router {self.symbol}: Desired total: {desired_total}")
        logger.info(f"Router {self.symbol}: Net change needed: {net_change}")
        
        if net_change == 0:
            logger.info(f"Router {self.symbol}: No trade needed")
        else:
            action = 'BUY' if net_change > 0 else 'SELL'
            qty = abs(net_change)
            logger.info(f"Router {self.symbol}: Would place {action} order for {qty} contracts")
            
            # Record the order that would be placed
            self.orders_placed.append({
                'action': action,
                'quantity': qty,
                'symbol': self.symbol
            })
        
        # Update virtual lots (optimistic)
        self.strategy_lots.update(desired_dict)
        return net_change != 0

def create_test_scenarios():
    """Create various test scenarios"""
    scenarios = []
    
    # Scenario 1: Starting fresh (no positions)
    scenarios.append({
        'name': 'Fresh Start - No Positions',
        'current_equity': 50000,
        'portfolio_state': {
            'positions': {strategy: {'in_position': False, 'position': None} 
                         for strategy in allocation_percentages},
            'current_equity': 50000
        },
        'market_data': {
            'ES': {'price': 6000, 'high': 6050, 'low': 5950, 'ibs': 0.05, 'williams_r': -95},  # IBS entry, Williams entry
            'YM': {'price': 42000, 'high': 42100, 'low': 41900, 'ibs': 0.95, 'williams_r': -10}, # IBS exit, Williams exit
            'GC': {'price': 3400, 'high': 3450, 'low': 3350, 'ibs': 0.5, 'williams_r': -50},   # No signals
            'NQ': {'price': 22000, 'high': 22100, 'low': 21900, 'ibs': 0.08, 'williams_r': -85} # IBS entry, Williams entry
        }
    })
    
    # Scenario 2: Current state (positions exist, need rebalancing)
    scenarios.append({
        'name': 'Current State - Rebalancing Needed',
        'current_equity': 32169.72,
        'portfolio_state': {
            'positions': {
                'IBS_ES': {'in_position': True, 'position': {'contracts': 2, 'entry_price': 6028.25}},
                'IBS_YM': {'in_position': True, 'position': {'contracts': 2, 'entry_price': 42175.0}},
                'IBS_GC': {'in_position': True, 'position': {'contracts': 2, 'entry_price': 3376.2}},
                'IBS_NQ': {'in_position': False, 'position': None},
                'Williams_ES': {'in_position': False, 'position': None},
                'Williams_YM': {'in_position': False, 'position': None},
                'Williams_GC': {'in_position': False, 'position': None},
                'Williams_NQ': {'in_position': False, 'position': None}
            },
            'current_equity': 32169.72
        },
        'market_data': {
            'ES': {'price': 6037.5, 'high': 6055.5, 'low': 5941.0, 'ibs': 0.843, 'williams_r': -13.79}, # Hold IBS, Williams entry
            'YM': {'price': 42549.0, 'high': 42748.0, 'low': 41847.0, 'ibs': 0.779, 'williams_r': -40},  # Hold IBS, no Williams
            'GC': {'price': 3404.6, 'high': 3476.0, 'low': 3401.4, 'ibs': 0.043, 'williams_r': -60},    # Hold IBS, no Williams
            'NQ': {'price': 21939.75, 'high': 21996.5, 'low': 21505.0, 'ibs': 0.885, 'williams_r': -20} # No IBS, no Williams
        }
    })
    
    # Scenario 3: Exit signals
    scenarios.append({
        'name': 'Exit Signals - Reduce Positions',
        'current_equity': 45000,
        'portfolio_state': {
            'positions': {
                'IBS_ES': {'in_position': True, 'position': {'contracts': 2, 'entry_price': 6000}},
                'IBS_YM': {'in_position': True, 'position': {'contracts': 1, 'entry_price': 42000}},
                'IBS_GC': {'in_position': False, 'position': None},
                'IBS_NQ': {'in_position': True, 'position': {'contracts': 1, 'entry_price': 22000}},
                'Williams_ES': {'in_position': True, 'position': {'contracts': 1, 'entry_price': 6000}},
                'Williams_YM': {'in_position': False, 'position': None},
                'Williams_GC': {'in_position': True, 'position': {'contracts': 2, 'entry_price': 3400}},
                'Williams_NQ': {'in_position': False, 'position': None}
            },
            'current_equity': 45000
        },
        'market_data': {
            'ES': {'price': 6100, 'high': 6150, 'low': 6050, 'ibs': 0.95, 'williams_r': -5},   # Exit both
            'YM': {'price': 42200, 'high': 42300, 'low': 42100, 'ibs': 0.5, 'williams_r': -50}, # Hold IBS, no Williams
            'GC': {'price': 3450, 'high': 3500, 'low': 3400, 'ibs': 0.08, 'williams_r': -10},  # IBS entry, Williams exit
            'NQ': {'price': 22200, 'high': 22300, 'low': 22100, 'ibs': 0.92, 'williams_r': -85} # IBS exit, Williams entry
        }
    })
    
    return scenarios

def run_scenario_test(scenario):
    """Run a single scenario test"""
    logger = logging.getLogger()
    logger.info(f"\n{'='*60}")
    logger.info(f"TESTING SCENARIO: {scenario['name']}")
    logger.info(f"{'='*60}")
    
    current_equity = scenario['current_equity']
    state = scenario['portfolio_state']
    market_data = scenario['market_data']
    
    logger.info(f"Current Equity: ${current_equity:,.2f}")
    
    # Initialize routers with current state
    routers = {}
    desired_positions = {}
    
    for symbol in ['ES', 'YM', 'GC', 'NQ']:
        # Initialize router with current positions
        initial_router_state = {}
        for strategy in [f'IBS_{symbol}', f'Williams_{symbol}']:
            if state['positions'][strategy]['in_position']:
                initial_router_state[strategy] = state['positions'][strategy]['position']['contracts']
            else:
                initial_router_state[strategy] = 0
        
        routers[symbol] = MockPositionRouter(symbol, initial_router_state)
        desired_positions[symbol] = {}
    
    # Process IBS strategies
    logger.info(f"\n--- Processing IBS Strategies ---")
    for symbol in ['ES', 'YM', 'GC', 'NQ']:
        strategy_key = f'IBS_{symbol}'
        multiplier = contract_specs[symbol]['multiplier']
        
        # Get market data
        price = market_data[symbol]['price']
        ibs = market_data[symbol]['ibs']
        
        logger.info(f"\n{strategy_key}:")
        logger.info(f"  Price: {price}, IBS: {ibs:.3f}")
        
        # Calculate target contracts
        target_contracts = calculate_position_size(
            current_equity,
            allocation_percentages[strategy_key],
            price,
            multiplier
        )
        
        allocation_pct = allocation_percentages[strategy_key]
        target_dollar = current_equity * allocation_pct * risk_multiplier
        logger.info(f"  Allocation: {allocation_pct*100:.0f}% * {risk_multiplier}x = ${target_dollar:,.0f} target")
        logger.info(f"  Target contracts: {target_contracts}")
        
        # Get current position state
        strategy_state = state['positions'][strategy_key]
        
        # Execute IBS logic
        if strategy_state['in_position']:
            if ibs > ibs_exit_threshold:
                desired_qty = 0
                logger.info(f"  IBS EXIT signal (IBS: {ibs:.3f} > {ibs_exit_threshold})")
            else:
                desired_qty = target_contracts  # Use target contracts for rebalancing
                current_contracts = strategy_state['position']['contracts']
                if desired_qty != current_contracts:
                    logger.info(f"  IBS REBALANCING: {current_contracts} → {desired_qty} contracts (IBS: {ibs:.3f})")
                else:
                    logger.info(f"  HOLDING position (IBS: {ibs:.3f}, exit threshold: {ibs_exit_threshold})")
        else:
            if ibs < ibs_entry_threshold:
                desired_qty = target_contracts
                logger.info(f"  IBS ENTRY signal (IBS: {ibs:.3f} < {ibs_entry_threshold})")
            else:
                desired_qty = 0
                logger.info(f"  NO ENTRY signal (IBS: {ibs:.3f}, entry threshold: {ibs_entry_threshold})")
        
        desired_positions[symbol][strategy_key] = desired_qty
        logger.info(f"  Desired contracts: {desired_qty}")
    
    # Process Williams strategies
    logger.info(f"\n--- Processing Williams Strategies ---")
    for symbol in ['ES', 'YM', 'GC', 'NQ']:
        strategy_key = f'Williams_{symbol}'
        multiplier = contract_specs[symbol]['multiplier']
        
        # Get market data
        price = market_data[symbol]['price']
        williams_r = market_data[symbol]['williams_r']
        
        logger.info(f"\n{strategy_key}:")
        logger.info(f"  Price: {price}, Williams %R: {williams_r:.2f}")
        
        # Calculate target contracts
        target_contracts = calculate_position_size(
            current_equity,
            allocation_percentages[strategy_key],
            price,
            multiplier
        )
        
        allocation_pct = allocation_percentages[strategy_key]
        target_dollar = current_equity * allocation_pct * risk_multiplier
        logger.info(f"  Allocation: {allocation_pct*100:.0f}% * {risk_multiplier}x = ${target_dollar:,.0f} target")
        logger.info(f"  Target contracts: {target_contracts}")
        
        # Get current position state
        williams_state = state['positions'][strategy_key]
        
        # Execute Williams logic
        if williams_state['in_position']:
            # Check exit conditions
            exit_signal = False
            if williams_r > wr_sell_threshold:
                exit_signal = True
                logger.info(f"  Williams EXIT signal: Williams %R > {wr_sell_threshold}")
            
            if exit_signal:
                desired_qty = 0
            else:
                desired_qty = target_contracts  # Use target contracts for rebalancing
                current_contracts = williams_state['position']['contracts']
                if desired_qty != current_contracts:
                    logger.info(f"  Williams REBALANCING: {current_contracts} → {desired_qty} contracts (Williams %R: {williams_r:.2f})")
                else:
                    logger.info(f"  HOLDING position (Williams %R: {williams_r:.2f})")
        else:
            if williams_r < wr_buy_threshold:
                desired_qty = target_contracts
                logger.info(f"  Williams ENTRY signal (Williams %R: {williams_r:.2f} < {wr_buy_threshold})")
            else:
                desired_qty = 0
                logger.info(f"  NO ENTRY signal (Williams %R: {williams_r:.2f}, threshold: {wr_buy_threshold})")
        
        desired_positions[symbol][strategy_key] = desired_qty
        logger.info(f"  Desired contracts: {desired_qty}")
    
    # Execute router sync for all instruments
    logger.info(f"\n--- Router Execution ---")
    total_orders = []
    
    for symbol in ['ES', 'YM', 'GC', 'NQ']:
        logger.info(f"\n{symbol} Router Sync:")
        order_placed = routers[symbol].sync(desired_positions[symbol])
        if order_placed:
            total_orders.extend(routers[symbol].orders_placed)
    
    # Summary
    logger.info(f"\n--- SCENARIO SUMMARY ---")
    logger.info(f"Total orders that would be placed: {len(total_orders)}")
    for order in total_orders:
        logger.info(f"  {order['action']} {order['quantity']} {order['symbol']} contracts")
    
    # Final position summary
    logger.info(f"\nFinal Virtual Positions:")
    for symbol in ['ES', 'YM', 'GC', 'NQ']:
        logger.info(f"  {symbol}: {dict(routers[symbol].strategy_lots)}")
    
    return total_orders

def main():
    """Run all dry run tests"""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    
    logger.info("Starting Live Port Dry Run Tests")
    logger.info(f"Risk Multiplier: {risk_multiplier}x")
    logger.info(f"IBS Thresholds: Entry < {ibs_entry_threshold}, Exit > {ibs_exit_threshold}")
    logger.info(f"Williams Thresholds: Entry < {wr_buy_threshold}, Exit > {wr_sell_threshold}")
    
    scenarios = create_test_scenarios()
    all_results = []
    
    for scenario in scenarios:
        orders = run_scenario_test(scenario)
        all_results.append({
            'scenario': scenario['name'],
            'orders': orders
        })
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("OVERALL TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    for result in all_results:
        logger.info(f"\n{result['scenario']}:")
        if result['orders']:
            for order in result['orders']:
                logger.info(f"  → {order['action']} {order['quantity']} {order['symbol']}")
        else:
            logger.info("  → No orders")
    
    logger.info(f"\nDry run tests completed successfully!")

if __name__ == '__main__':
    main() 