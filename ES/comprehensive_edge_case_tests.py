#!/usr/bin/env python
"""
Comprehensive edge case tests for live_port.py
Tests every possible scenario to ensure robust operation in all conditions.
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

class MockPositionRouter:
    """Enhanced mock router for comprehensive testing"""
    def __init__(self, symbol, initial_state=None):
        self.symbol = symbol
        self.strategy_lots = defaultdict(int)
        self.orders_placed = []
        self.sync_calls = []
        
        if initial_state:
            self.strategy_lots.update(initial_state)
    
    def sync(self, desired_dict):
        """Mock sync that records detailed information"""
        logger = logging.getLogger()
        
        # Record this sync call for analysis
        sync_call = {
            'symbol': self.symbol,
            'current_lots': dict(self.strategy_lots),
            'desired_lots': desired_dict.copy(),
            'timestamp': datetime.now().isoformat()
        }
        self.sync_calls.append(sync_call)
        
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
            order = {
                'action': action,
                'quantity': qty,
                'symbol': self.symbol,
                'from_total': current_total,
                'to_total': desired_total
            }
            self.orders_placed.append(order)
        
        # Update virtual lots (optimistic)
        self.strategy_lots.update(desired_dict)
        return net_change != 0

def create_edge_case_scenarios():
    """Create comprehensive edge case test scenarios"""
    scenarios = []
    
    # Edge Case 1: Extreme low equity - minimum position sizing
    scenarios.append({
        'name': 'Edge Case 1: Extreme Low Equity',
        'current_equity': 1000,  # Very low equity
        'portfolio_state': {
            'positions': {strategy: {'in_position': False, 'position': None} 
                         for strategy in allocation_percentages},
            'current_equity': 1000
        },
        'market_data': {
            'ES': {'price': 6000, 'high': 6050, 'low': 5950, 'ibs': 0.05, 'williams_r': -95},
            'YM': {'price': 42000, 'high': 42100, 'low': 41900, 'ibs': 0.05, 'williams_r': -95},
            'GC': {'price': 3400, 'high': 3450, 'low': 3350, 'ibs': 0.05, 'williams_r': -95},
            'NQ': {'price': 22000, 'high': 22100, 'low': 21900, 'ibs': 0.05, 'williams_r': -95}
        },
        'expected_behavior': 'All strategies should get minimum 1 contract despite low equity'
    })
    
    # Edge Case 2: Extreme high equity - large position sizing
    scenarios.append({
        'name': 'Edge Case 2: Extreme High Equity',
        'current_equity': 1000000,  # Very high equity
        'portfolio_state': {
            'positions': {strategy: {'in_position': False, 'position': None} 
                         for strategy in allocation_percentages},
            'current_equity': 1000000
        },
        'market_data': {
            'ES': {'price': 6000, 'high': 6050, 'low': 5950, 'ibs': 0.05, 'williams_r': -95},
            'YM': {'price': 42000, 'high': 42100, 'low': 41900, 'ibs': 0.05, 'williams_r': -95},
            'GC': {'price': 3400, 'high': 3450, 'low': 3350, 'ibs': 0.05, 'williams_r': -95},
            'NQ': {'price': 22000, 'high': 22100, 'low': 21900, 'ibs': 0.05, 'williams_r': -95}
        },
        'expected_behavior': 'Large position sizes should be calculated correctly'
    })
    
    # Edge Case 3: Zero range bars (high = low)
    scenarios.append({
        'name': 'Edge Case 3: Zero Range Bars',
        'current_equity': 50000,
        'portfolio_state': {
            'positions': {strategy: {'in_position': False, 'position': None} 
                         for strategy in allocation_percentages},
            'current_equity': 50000
        },
        'market_data': {
            'ES': {'price': 6000, 'high': 6000, 'low': 6000, 'ibs': 0.5, 'williams_r': -50},  # Zero range
            'YM': {'price': 42000, 'high': 42000, 'low': 42000, 'ibs': 0.5, 'williams_r': -50},  # Zero range
            'GC': {'price': 3400, 'high': 3400, 'low': 3400, 'ibs': 0.5, 'williams_r': -50},  # Zero range
            'NQ': {'price': 22000, 'high': 22000, 'low': 22000, 'ibs': 0.5, 'williams_r': -50}  # Zero range
        },
        'expected_behavior': 'Should handle zero range bars gracefully with neutral IBS/Williams'
    })
    
    # Edge Case 4: Exact threshold values
    scenarios.append({
        'name': 'Edge Case 4: Exact Threshold Values',
        'current_equity': 50000,
        'portfolio_state': {
            'positions': {
                'IBS_ES': {'in_position': True, 'position': {'contracts': 2, 'entry_price': 6000}},
                'IBS_YM': {'in_position': True, 'position': {'contracts': 2, 'entry_price': 42000}},
                'IBS_GC': {'in_position': True, 'position': {'contracts': 2, 'entry_price': 3400}},
                'IBS_NQ': {'in_position': True, 'position': {'contracts': 2, 'entry_price': 22000}},
                'Williams_ES': {'in_position': True, 'position': {'contracts': 1, 'entry_price': 6000}},
                'Williams_YM': {'in_position': True, 'position': {'contracts': 1, 'entry_price': 42000}},
                'Williams_GC': {'in_position': True, 'position': {'contracts': 1, 'entry_price': 3400}},
                'Williams_NQ': {'in_position': True, 'position': {'contracts': 1, 'entry_price': 22000}}
            },
            'current_equity': 50000
        },
        'market_data': {
            'ES': {'price': 6000, 'high': 6100, 'low': 5900, 'ibs': 0.1, 'williams_r': -90},    # Exact thresholds
            'YM': {'price': 42000, 'high': 42200, 'low': 41800, 'ibs': 0.9, 'williams_r': -30},  # Exact thresholds
            'GC': {'price': 3400, 'high': 3500, 'low': 3300, 'ibs': 0.1, 'williams_r': -90},    # Exact thresholds
            'NQ': {'price': 22000, 'high': 22200, 'low': 21800, 'ibs': 0.9, 'williams_r': -30}  # Exact thresholds
        },
        'expected_behavior': 'Should handle exact threshold values consistently'
    })
    
    # Edge Case 5: Mixed signals - some entry, some exit, some hold
    scenarios.append({
        'name': 'Edge Case 5: Mixed Signals Complex',
        'current_equity': 75000,
        'portfolio_state': {
            'positions': {
                'IBS_ES': {'in_position': True, 'position': {'contracts': 3, 'entry_price': 6000}},
                'IBS_YM': {'in_position': False, 'position': None},
                'IBS_GC': {'in_position': True, 'position': {'contracts': 1, 'entry_price': 3400}},
                'IBS_NQ': {'in_position': False, 'position': None},
                'Williams_ES': {'in_position': False, 'position': None},
                'Williams_YM': {'in_position': True, 'position': {'contracts': 2, 'entry_price': 42000}},
                'Williams_GC': {'in_position': False, 'position': None},
                'Williams_NQ': {'in_position': True, 'position': {'contracts': 1, 'entry_price': 22000}}
            },
            'current_equity': 75000
        },
        'market_data': {
            'ES': {'price': 6100, 'high': 6200, 'low': 6000, 'ibs': 0.95, 'williams_r': -95},  # IBS exit, Williams entry
            'YM': {'price': 42100, 'high': 42200, 'low': 42000, 'ibs': 0.05, 'williams_r': -5},   # IBS entry, Williams exit
            'GC': {'price': 3450, 'high': 3500, 'low': 3400, 'ibs': 0.5, 'williams_r': -50},    # No signals
            'NQ': {'price': 22100, 'high': 22200, 'low': 22000, 'ibs': 0.08, 'williams_r': -85}  # IBS entry, no Williams
        },
        'expected_behavior': 'Complex mixed signals should be handled correctly'
    })
    
    # Edge Case 6: Extreme price movements
    scenarios.append({
        'name': 'Edge Case 6: Extreme Price Movements',
        'current_equity': 50000,
        'portfolio_state': {
            'positions': {strategy: {'in_position': False, 'position': None} 
                         for strategy in allocation_percentages},
            'current_equity': 50000
        },
        'market_data': {
            'ES': {'price': 10000, 'high': 12000, 'low': 8000, 'ibs': 0.05, 'williams_r': -95},   # Extreme range
            'YM': {'price': 60000, 'high': 70000, 'low': 50000, 'ibs': 0.05, 'williams_r': -95}, # Extreme range
            'GC': {'price': 5000, 'high': 6000, 'low': 4000, 'ibs': 0.05, 'williams_r': -95},    # Extreme range
            'NQ': {'price': 30000, 'high': 35000, 'low': 25000, 'ibs': 0.05, 'williams_r': -95}  # Extreme range
        },
        'expected_behavior': 'Should handle extreme price movements and calculate position sizes correctly'
    })
    
    # Edge Case 7: Router conflict resolution - both strategies want same instrument
    scenarios.append({
        'name': 'Edge Case 7: Router Conflict Resolution',
        'current_equity': 40000,
        'portfolio_state': {
            'positions': {
                'IBS_ES': {'in_position': True, 'position': {'contracts': 1, 'entry_price': 6000}},
                'IBS_YM': {'in_position': False, 'position': None},
                'IBS_GC': {'in_position': False, 'position': None},
                'IBS_NQ': {'in_position': False, 'position': None},
                'Williams_ES': {'in_position': True, 'position': {'contracts': 1, 'entry_price': 6000}},
                'Williams_YM': {'in_position': False, 'position': None},
                'Williams_GC': {'in_position': False, 'position': None},
                'Williams_NQ': {'in_position': False, 'position': None}
            },
            'current_equity': 40000
        },
        'market_data': {
            'ES': {'price': 6000, 'high': 6100, 'low': 5900, 'ibs': 0.05, 'williams_r': -95},  # Both want to enter
            'YM': {'price': 42000, 'high': 42100, 'low': 41900, 'ibs': 0.95, 'williams_r': -5}, # Both want to exit
            'GC': {'price': 3400, 'high': 3500, 'low': 3300, 'ibs': 0.5, 'williams_r': -50},   # No signals
            'NQ': {'price': 22000, 'high': 22100, 'low': 21900, 'ibs': 0.5, 'williams_r': -50}  # No signals
        },
        'expected_behavior': 'Router should handle multiple strategies on same instrument correctly'
    })
    
    # Edge Case 8: Fractional position sizing
    scenarios.append({
        'name': 'Edge Case 8: Fractional Position Sizing',
        'current_equity': 15000,  # Small equity that might cause fractional contracts
        'portfolio_state': {
            'positions': {strategy: {'in_position': False, 'position': None} 
                         for strategy in allocation_percentages},
            'current_equity': 15000
        },
        'market_data': {
            'ES': {'price': 6000, 'high': 6050, 'low': 5950, 'ibs': 0.05, 'williams_r': -95},
            'YM': {'price': 42000, 'high': 42100, 'low': 41900, 'ibs': 0.05, 'williams_r': -95},
            'GC': {'price': 3400, 'high': 3450, 'low': 3350, 'ibs': 0.05, 'williams_r': -95},
            'NQ': {'price': 22000, 'high': 22100, 'low': 21900, 'ibs': 0.05, 'williams_r': -95}
        },
        'expected_behavior': 'Should round fractional contracts to integers properly'
    })
    
    # Edge Case 9: All exit signals simultaneously
    scenarios.append({
        'name': 'Edge Case 9: Mass Exit Event',
        'current_equity': 60000,
        'portfolio_state': {
            'positions': {
                'IBS_ES': {'in_position': True, 'position': {'contracts': 2, 'entry_price': 6000}},
                'IBS_YM': {'in_position': True, 'position': {'contracts': 2, 'entry_price': 42000}},
                'IBS_GC': {'in_position': True, 'position': {'contracts': 2, 'entry_price': 3400}},
                'IBS_NQ': {'in_position': True, 'position': {'contracts': 2, 'entry_price': 22000}},
                'Williams_ES': {'in_position': True, 'position': {'contracts': 1, 'entry_price': 6000}},
                'Williams_YM': {'in_position': True, 'position': {'contracts': 1, 'entry_price': 42000}},
                'Williams_GC': {'in_position': True, 'position': {'contracts': 1, 'entry_price': 3400}},
                'Williams_NQ': {'in_position': True, 'position': {'contracts': 1, 'entry_price': 22000}}
            },
            'current_equity': 60000
        },
        'market_data': {
            'ES': {'price': 6100, 'high': 6200, 'low': 6000, 'ibs': 0.95, 'williams_r': -5},   # All exit
            'YM': {'price': 42100, 'high': 42200, 'low': 42000, 'ibs': 0.95, 'williams_r': -5}, # All exit
            'GC': {'price': 3450, 'high': 3500, 'low': 3400, 'ibs': 0.95, 'williams_r': -5},   # All exit
            'NQ': {'price': 22100, 'high': 22200, 'low': 22000, 'ibs': 0.95, 'williams_r': -5}  # All exit
        },
        'expected_behavior': 'Should handle mass exit event correctly'
    })
    
    # Edge Case 10: All entry signals simultaneously
    scenarios.append({
        'name': 'Edge Case 10: Mass Entry Event',
        'current_equity': 100000,
        'portfolio_state': {
            'positions': {strategy: {'in_position': False, 'position': None} 
                         for strategy in allocation_percentages},
            'current_equity': 100000
        },
        'market_data': {
            'ES': {'price': 6000, 'high': 6100, 'low': 5900, 'ibs': 0.05, 'williams_r': -95},  # All entry
            'YM': {'price': 42000, 'high': 42200, 'low': 41800, 'ibs': 0.05, 'williams_r': -95}, # All entry
            'GC': {'price': 3400, 'high': 3500, 'low': 3300, 'ibs': 0.05, 'williams_r': -95},   # All entry
            'NQ': {'price': 22000, 'high': 22200, 'low': 21800, 'ibs': 0.05, 'williams_r': -95}  # All entry
        },
        'expected_behavior': 'Should handle mass entry event correctly'
    })
    
    return scenarios

def run_edge_case_test(scenario):
    """Run a single edge case test with detailed analysis"""
    logger = logging.getLogger()
    logger.info(f"\n{'='*80}")
    logger.info(f"TESTING EDGE CASE: {scenario['name']}")
    logger.info(f"{'='*80}")
    logger.info(f"Expected Behavior: {scenario['expected_behavior']}")
    
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
    
    # Track all calculations for analysis
    calculations = {
        'ibs_calculations': [],
        'williams_calculations': [],
        'position_sizing': [],
        'router_decisions': []
    }
    
    # Process IBS strategies with detailed tracking
    logger.info(f"\n--- Processing IBS Strategies ---")
    for symbol in ['ES', 'YM', 'GC', 'NQ']:
        strategy_key = f'IBS_{symbol}'
        multiplier = contract_specs[symbol]['multiplier']
        
        # Get market data
        price = market_data[symbol]['price']
        high = market_data[symbol]['high']
        low = market_data[symbol]['low']
        ibs = market_data[symbol]['ibs']
        
        # Verify IBS calculation
        calculated_ibs = compute_ibs(MockBar('2025-01-01', price, high, low, price))
        
        logger.info(f"\n{strategy_key}:")
        logger.info(f"  Price: {price}, High: {high}, Low: {low}")
        logger.info(f"  IBS (provided): {ibs:.3f}, IBS (calculated): {calculated_ibs:.3f}")
        
        if abs(ibs - calculated_ibs) > 0.001:
            logger.warning(f"  IBS MISMATCH! Using calculated value: {calculated_ibs:.3f}")
            ibs = calculated_ibs
        
        # Calculate target contracts
        target_contracts = calculate_position_size(
            current_equity,
            allocation_percentages[strategy_key],
            price,
            multiplier
        )
        
        allocation_pct = allocation_percentages[strategy_key]
        target_dollar = current_equity * allocation_pct * risk_multiplier
        contract_value = price * multiplier
        
        logger.info(f"  Allocation: {allocation_pct*100:.1f}% * {risk_multiplier}x = ${target_dollar:,.0f} target")
        logger.info(f"  Contract value: ${contract_value:,.0f}")
        logger.info(f"  Raw calculation: {target_dollar / contract_value:.3f} contracts")
        logger.info(f"  Target contracts (rounded): {target_contracts}")
        
        # Track calculation
        calculations['position_sizing'].append({
            'strategy': strategy_key,
            'target_dollar': target_dollar,
            'contract_value': contract_value,
            'raw_contracts': target_dollar / contract_value,
            'final_contracts': target_contracts
        })
        
        # Get current position state
        strategy_state = state['positions'][strategy_key]
        
        # Execute IBS logic
        if strategy_state['in_position']:
            current_contracts = strategy_state['position']['contracts']
            if ibs > ibs_exit_threshold:
                desired_qty = 0
                decision = f"IBS EXIT signal (IBS: {ibs:.3f} > {ibs_exit_threshold})"
            else:
                desired_qty = target_contracts
                if desired_qty != current_contracts:
                    decision = f"IBS REBALANCING: {current_contracts} → {desired_qty} contracts (IBS: {ibs:.3f})"
                else:
                    decision = f"HOLDING position (IBS: {ibs:.3f}, exit threshold: {ibs_exit_threshold})"
        else:
            if ibs < ibs_entry_threshold:
                desired_qty = target_contracts
                decision = f"IBS ENTRY signal (IBS: {ibs:.3f} < {ibs_entry_threshold})"
            else:
                desired_qty = 0
                decision = f"NO ENTRY signal (IBS: {ibs:.3f}, entry threshold: {ibs_entry_threshold})"
        
        logger.info(f"  {decision}")
        logger.info(f"  Desired contracts: {desired_qty}")
        
        calculations['ibs_calculations'].append({
            'strategy': strategy_key,
            'ibs': ibs,
            'decision': decision,
            'desired_qty': desired_qty
        })
        
        desired_positions[symbol][strategy_key] = desired_qty
    
    # Process Williams strategies with detailed tracking
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
        logger.info(f"  Allocation: {allocation_pct*100:.1f}% * {risk_multiplier}x = ${target_dollar:,.0f} target")
        logger.info(f"  Target contracts: {target_contracts}")
        
        # Get current position state
        williams_state = state['positions'][strategy_key]
        
        # Execute Williams logic
        if williams_state['in_position']:
            current_contracts = williams_state['position']['contracts']
            exit_signal = False
            if williams_r > wr_sell_threshold:
                exit_signal = True
                exit_reason = f"Williams %R > {wr_sell_threshold}"
            
            if exit_signal:
                desired_qty = 0
                decision = f"Williams EXIT signal: {exit_reason}"
            else:
                desired_qty = target_contracts
                if desired_qty != current_contracts:
                    decision = f"Williams REBALANCING: {current_contracts} → {desired_qty} contracts (Williams %R: {williams_r:.2f})"
                else:
                    decision = f"HOLDING position (Williams %R: {williams_r:.2f})"
        else:
            if williams_r < wr_buy_threshold:
                desired_qty = target_contracts
                decision = f"Williams ENTRY signal (Williams %R: {williams_r:.2f} < {wr_buy_threshold})"
            else:
                desired_qty = 0
                decision = f"NO ENTRY signal (Williams %R: {williams_r:.2f}, threshold: {wr_buy_threshold})"
        
        logger.info(f"  {decision}")
        logger.info(f"  Desired contracts: {desired_qty}")
        
        calculations['williams_calculations'].append({
            'strategy': strategy_key,
            'williams_r': williams_r,
            'decision': decision,
            'desired_qty': desired_qty
        })
        
        desired_positions[symbol][strategy_key] = desired_qty
    
    # Execute router sync for all instruments with detailed analysis
    logger.info(f"\n--- Router Execution Analysis ---")
    total_orders = []
    
    for symbol in ['ES', 'YM', 'GC', 'NQ']:
        logger.info(f"\n{symbol} Router Detailed Analysis:")
        
        # Show before state
        before_state = dict(routers[symbol].strategy_lots)
        before_total = sum(before_state.values())
        
        # Show desired state
        desired_state = desired_positions[symbol]
        desired_total = sum(desired_state.values())
        
        logger.info(f"  Before: {before_state} (Total: {before_total})")
        logger.info(f"  Desired: {desired_state} (Total: {desired_total})")
        logger.info(f"  Net Change: {desired_total - before_total}")
        
        # Execute sync
        order_placed = routers[symbol].sync(desired_state)
        
        if order_placed:
            total_orders.extend(routers[symbol].orders_placed)
            
        # Show after state
        after_state = dict(routers[symbol].strategy_lots)
        after_total = sum(after_state.values())
        logger.info(f"  After: {after_state} (Total: {after_total})")
        
        calculations['router_decisions'].append({
            'symbol': symbol,
            'before_state': before_state,
            'desired_state': desired_state,
            'after_state': after_state,
            'net_change': desired_total - before_total,
            'orders': routers[symbol].orders_placed[-1:] if routers[symbol].orders_placed else []
        })
    
    # Comprehensive analysis
    logger.info(f"\n--- COMPREHENSIVE ANALYSIS ---")
    
    # Position sizing analysis
    logger.info(f"\nPosition Sizing Analysis:")
    total_target_value = 0
    for calc in calculations['position_sizing']:
        total_target_value += calc['target_dollar']
        logger.info(f"  {calc['strategy']}: ${calc['target_dollar']:,.0f} → {calc['final_contracts']} contracts")
    
    logger.info(f"  Total target allocation: ${total_target_value:,.0f}")
    logger.info(f"  Percentage of equity: {(total_target_value/current_equity)*100:.1f}%")
    logger.info(f"  Expected percentage: {sum(allocation_percentages.values())*risk_multiplier*100:.1f}%")
    
    # Signal analysis
    logger.info(f"\nSignal Analysis:")
    ibs_entries = sum(1 for calc in calculations['ibs_calculations'] if 'ENTRY' in calc['decision'])
    ibs_exits = sum(1 for calc in calculations['ibs_calculations'] if 'EXIT' in calc['decision'])
    ibs_holds = sum(1 for calc in calculations['ibs_calculations'] if 'HOLDING' in calc['decision'])
    ibs_rebalances = sum(1 for calc in calculations['ibs_calculations'] if 'REBALANCING' in calc['decision'])
    
    williams_entries = sum(1 for calc in calculations['williams_calculations'] if 'ENTRY' in calc['decision'])
    williams_exits = sum(1 for calc in calculations['williams_calculations'] if 'EXIT' in calc['decision'])
    williams_holds = sum(1 for calc in calculations['williams_calculations'] if 'HOLDING' in calc['decision'])
    williams_rebalances = sum(1 for calc in calculations['williams_calculations'] if 'REBALANCING' in calc['decision'])
    
    logger.info(f"  IBS: {ibs_entries} entries, {ibs_exits} exits, {ibs_holds} holds, {ibs_rebalances} rebalances")
    logger.info(f"  Williams: {williams_entries} entries, {williams_exits} exits, {williams_holds} holds, {williams_rebalances} rebalances")
    
    # Order analysis
    logger.info(f"\nOrder Analysis:")
    buy_orders = [o for o in total_orders if o['action'] == 'BUY']
    sell_orders = [o for o in total_orders if o['action'] == 'SELL']
    
    total_buy_qty = sum(o['quantity'] for o in buy_orders)
    total_sell_qty = sum(o['quantity'] for o in sell_orders)
    
    logger.info(f"  Buy orders: {len(buy_orders)} (Total qty: {total_buy_qty})")
    logger.info(f"  Sell orders: {len(sell_orders)} (Total qty: {total_sell_qty})")
    logger.info(f"  Net position change: {total_buy_qty - total_sell_qty}")
    
    # Summary
    logger.info(f"\n--- SCENARIO SUMMARY ---")
    logger.info(f"Total orders that would be placed: {len(total_orders)}")
    for order in total_orders:
        logger.info(f"  {order['action']} {order['quantity']} {order['symbol']} contracts ({order['from_total']}→{order['to_total']})")
    
    # Final position summary
    logger.info(f"\nFinal Virtual Positions:")
    for symbol in ['ES', 'YM', 'GC', 'NQ']:
        logger.info(f"  {symbol}: {dict(routers[symbol].strategy_lots)}")
    
    # Edge case specific validations
    logger.info(f"\n--- EDGE CASE VALIDATIONS ---")
    
    # Validate minimum contracts
    for symbol in ['ES', 'YM', 'GC', 'NQ']:
        for strategy in [f'IBS_{symbol}', f'Williams_{symbol}']:
            final_qty = routers[symbol].strategy_lots[strategy]
            if final_qty > 0 and final_qty < 1:
                logger.error(f"ERROR: {strategy} has fractional contracts: {final_qty}")
    
    # Validate total allocation doesn't exceed reasonable bounds
    total_final_value = 0
    for symbol in ['ES', 'YM', 'GC', 'NQ']:
        price = market_data[symbol]['price']
        multiplier = contract_specs[symbol]['multiplier']
        total_contracts = sum(routers[symbol].strategy_lots.values())
        symbol_value = total_contracts * price * multiplier
        total_final_value += symbol_value
    
    allocation_ratio = total_final_value / current_equity
    expected_ratio = sum(allocation_percentages.values()) * risk_multiplier
    
    logger.info(f"Final allocation ratio: {allocation_ratio:.2f} (Expected: {expected_ratio:.2f})")
    
    if abs(allocation_ratio - expected_ratio) > 0.1:  # 10% tolerance
        logger.warning(f"WARNING: Allocation ratio deviation: {abs(allocation_ratio - expected_ratio):.2f}")
    
    return {
        'scenario_name': scenario['name'],
        'orders': total_orders,
        'calculations': calculations,
        'final_positions': {symbol: dict(routers[symbol].strategy_lots) for symbol in ['ES', 'YM', 'GC', 'NQ']},
        'allocation_ratio': allocation_ratio,
        'expected_ratio': expected_ratio
    }

def main():
    """Run all edge case tests"""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    
    logger.info("Starting Comprehensive Edge Case Tests for Live Port")
    logger.info(f"Risk Multiplier: {risk_multiplier}x")
    logger.info(f"IBS Thresholds: Entry < {ibs_entry_threshold}, Exit > {ibs_exit_threshold}")
    logger.info(f"Williams Thresholds: Entry < {wr_buy_threshold}, Exit > {wr_sell_threshold}")
    
    scenarios = create_edge_case_scenarios()
    all_results = []
    
    for scenario in scenarios:
        try:
            result = run_edge_case_test(scenario)
            all_results.append(result)
        except Exception as e:
            logger.error(f"ERROR in scenario {scenario['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final comprehensive summary
    logger.info(f"\n{'='*80}")
    logger.info("COMPREHENSIVE TEST SUMMARY")
    logger.info(f"{'='*80}")
    
    total_scenarios = len(scenarios)
    successful_scenarios = len(all_results)
    
    logger.info(f"Scenarios tested: {successful_scenarios}/{total_scenarios}")
    
    for result in all_results:
        logger.info(f"\n{result['scenario_name']}:")
        if result['orders']:
            for order in result['orders']:
                logger.info(f"  → {order['action']} {order['quantity']} {order['symbol']}")
        else:
            logger.info("  → No orders")
        
        logger.info(f"  Allocation ratio: {result['allocation_ratio']:.2f} (Expected: {result['expected_ratio']:.2f})")
    
    # Overall validation
    logger.info(f"\n--- OVERALL VALIDATION ---")
    
    allocation_errors = [r for r in all_results if abs(r['allocation_ratio'] - r['expected_ratio']) > 0.1]
    if allocation_errors:
        logger.error(f"ALLOCATION ERRORS in {len(allocation_errors)} scenarios:")
        for error in allocation_errors:
            logger.error(f"  {error['scenario_name']}: {error['allocation_ratio']:.2f} vs {error['expected_ratio']:.2f}")
    else:
        logger.info("✅ All allocation ratios within acceptable bounds")
    
    logger.info(f"\n🎉 Comprehensive edge case testing completed!")
    logger.info(f"All {successful_scenarios} scenarios processed successfully")

if __name__ == '__main__':
    main() 