#!/usr/bin/env python3
"""
Script to update README.md with live trading metrics from account snapshots
"""
import json
import os
import re
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict, List, Optional

def load_position_data() -> List[Dict]:
    """Load current position data from portfolio state"""
    try:
        # Try to load from portfolio state (from live trading system)
        with open('portfolio_state.json', 'r') as f:
            portfolio_state = json.load(f)
        
        positions = []
        if 'positions' in portfolio_state:
            for strategy, pos_data in portfolio_state['positions'].items():
                if pos_data.get('in_position', False) and pos_data.get('position'):
                    pos = pos_data['position']
                    
                    # Map strategy names to proper symbols
                    symbol_map = {
                        'IBS_ES': 'MES', 'Williams_ES': 'MES',
                        'IBS_YM': 'MYM', 'Williams_YM': 'MYM', 
                        'IBS_GC': 'MGC', 'Williams_GC': 'MGC',
                        'IBS_NQ': 'MNQ', 'Williams_NQ': 'MNQ',
                        'IBS_ZQ': 'ZQ', 'Williams_ZQ': 'ZQ'
                    }
                    
                    symbol = symbol_map.get(strategy, strategy.split('_')[-1])
                    
                    # Format entry time
                    entry_time = pos.get('entry_time', 'N/A')
                    if isinstance(entry_time, str) and 'T' in entry_time:
                        try:
                            # Parse ISO format and get just the date
                            from datetime import datetime
                            dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                            entry_time = dt.strftime('%Y-%m-%d')
                        except:
                            entry_time = entry_time.split('T')[0]  # Fallback
                    
                    # Calculate position metrics
                    contracts = pos.get('contracts', 0)
                    entry_price = pos.get('entry_price', 0)
                    
                    # Contract multipliers for notional value calculation
                    multipliers = {
                        'MES': 5, 'MYM': 0.5, 'MGC': 10, 'MNQ': 2, 'ZQ': 2000
                    }
                    multiplier = multipliers.get(symbol, 1)
                    notional_value = contracts * entry_price * multiplier
                    
                    positions.append({
                        'strategy': strategy,
                        'symbol': symbol,
                        'side': 'Long' if contracts > 0 else 'Short',
                        'contracts': abs(contracts),
                        'entry_price': entry_price,
                        'entry_time': entry_time,
                        'notional_value': notional_value,
                        'multiplier': multiplier
                    })
        
        return positions
        
    except FileNotFoundError:
        print("Portfolio state file not found")
        return []
    except Exception as e:
        print(f"Error loading position data: {e}")
        return []

def load_trading_metrics() -> Optional[Dict]:
    """Load the latest trading metrics from JSON file"""
    try:
        with open('account_snapshots/daily_snapshots.json', 'r') as f:
            snapshots = json.load(f)
        
        if not snapshots:
            return None
            
        # Get the latest snapshot
        latest = sorted(snapshots, key=lambda x: x['date'])[-1]
        
        # Calculate historical performance if we have enough data
        metrics = {
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M UTC'),
            'account_value': latest['net_liquidation'],
            'unrealized_pnl': latest['unrealized_pnl'],
            'realized_pnl': latest.get('cumulative_realized_pnl', latest['realized_pnl']),  # Use cumulative if available
            'total_pnl': latest['unrealized_pnl'] + latest.get('cumulative_realized_pnl', latest['realized_pnl']),
            'total_pnl_pct': 0,  # Will calculate if we have historical data
            'trading_days': len(snapshots),
            'sharpe_ratio': 0,  # Will calculate if enough data
            'max_drawdown_pct': 0,  # Will calculate if enough data
            'win_rate': 0,  # Placeholder for now
            'total_trades': 0  # Placeholder for now
        }
        
        # Calculate performance over time if we have multiple snapshots
        if len(snapshots) > 1:
            # Use the first snapshot to determine initial capital
            first_snapshot = sorted(snapshots, key=lambda x: x['date'])[0]
            
            # For proper initial capital calculation:
            # If first snapshot has no P&L, use net_liquidation as initial capital
            # Otherwise, calculate initial capital by removing P&L from first snapshot
            if first_snapshot['unrealized_pnl'] == 0 and first_snapshot['realized_pnl'] == 0:
                initial_capital = first_snapshot['net_liquidation']
            else:
                # Back out the initial capital from first snapshot
                initial_capital = first_snapshot['net_liquidation'] - first_snapshot['unrealized_pnl'] - first_snapshot['realized_pnl']
            
            # Calculate total P&L across all time - special handling for existing data
            current_account_value = latest['net_liquidation']
            
            # For existing data without cumulative tracking, calculate cumulative realized P&L
            # by looking for account value changes that aren't explained by unrealized P&L changes
            if 'cumulative_realized_pnl' not in latest:
                cumulative_realized_pnl = 0
                
                # Look through snapshots to estimate cumulative realized P&L
                for i in range(len(snapshots)):
                    daily_realized = snapshots[i]['realized_pnl']
                    
                    # If there's a daily realized P&L, add it to cumulative
                    if daily_realized != 0:
                        cumulative_realized_pnl += daily_realized
                
                # Also check for unexplained account value changes (closed positions not tracked)
                # This is a heuristic: if account value changed more than unrealized P&L change suggests
                if len(snapshots) >= 2:
                    for i in range(1, len(snapshots)):
                        prev_snapshot = snapshots[i-1]
                        curr_snapshot = snapshots[i]
                        
                        # Expected account value based on previous value and unrealized P&L change
                        prev_base_value = prev_snapshot['net_liquidation'] - prev_snapshot['unrealized_pnl']
                        expected_account_value = prev_base_value + curr_snapshot['unrealized_pnl']
                        actual_account_value = curr_snapshot['net_liquidation']
                        
                        # If there's a significant difference not explained by recorded realized P&L
                        diff = actual_account_value - expected_account_value
                        if abs(diff) > 1 and curr_snapshot['realized_pnl'] == 0:
                            cumulative_realized_pnl += diff
                
                # Update the realized P&L to use our calculated cumulative
                metrics['realized_pnl'] = cumulative_realized_pnl
            else:
                cumulative_realized_pnl = latest['cumulative_realized_pnl']
            
            total_account_change = current_account_value - initial_capital
            
            # Use account value change as the true total P&L measure, but also show components
            metrics['total_pnl'] = total_account_change
            
            # Also store component-based total for comparison
            component_total_pnl = latest['unrealized_pnl'] + cumulative_realized_pnl
            
            if initial_capital > 0:
                metrics['total_pnl_pct'] = (total_account_change / initial_capital) * 100
                
                # Calculate advanced risk metrics if we have enough data
                if len(snapshots) >= 7:  # Need at least a week of data
                    daily_returns = []
                    daily_values = []
                    
                    for i in range(1, len(snapshots)):
                        prev_val = snapshots[i-1]['net_liquidation']
                        curr_val = snapshots[i]['net_liquidation']
                        if prev_val > 0:
                            daily_return = (curr_val - prev_val) / prev_val
                            daily_returns.append(daily_return)
                            daily_values.append(curr_val)
                    
                    if daily_returns:
                        import numpy as np
                        
                        # Calculate Sharpe Ratio (annualized)
                        mean_return = np.mean(daily_returns)
                        std_return = np.std(daily_returns)
                        if std_return > 0:
                            metrics['sharpe_ratio'] = (mean_return / std_return) * np.sqrt(252)  # Annualized
                        
                        # Calculate Maximum Drawdown from peak to trough
                        peak = initial_capital
                        max_dd = 0
                        for value in [initial_capital] + daily_values:
                            if value > peak:
                                peak = value
                            drawdown = (peak - value) / peak
                            max_dd = max(max_dd, drawdown)
                        metrics['max_drawdown_pct'] = max_dd * 100
                
                # Calculate period returns
                periods = {'1_week': 7, '1_month': 30, '3_months': 90}
                for period_name, days_back in periods.items():
                    period_snapshot = get_snapshot_near_date(snapshots, latest['date'], days_back)
                    if period_snapshot:
                        period_return = (latest['net_liquidation'] - period_snapshot['net_liquidation']) / period_snapshot['net_liquidation'] * 100
                        metrics[f'{period_name}_return'] = period_return
        else:
            # If only one snapshot, we can't calculate meaningful returns
            # Use the P&L values as reported but note uncertainty
            metrics['total_pnl'] = latest['unrealized_pnl'] + latest['realized_pnl']
            # Don't calculate percentage without baseline
        
        return metrics
        
    except FileNotFoundError:
        print("Trading metrics file not found")
        return None
    except Exception as e:
        print(f"Error loading trading metrics: {e}")
        return None

def get_snapshot_near_date(snapshots: List[Dict], current_date: str, days_back: int) -> Optional[Dict]:
    """Find snapshot closest to the target date"""
    target_date = datetime.strptime(current_date, '%Y-%m-%d') - timedelta(days=days_back)
    
    closest_snapshot = None
    min_diff = float('inf')
    
    for snapshot in snapshots:
        snapshot_date = datetime.strptime(snapshot['date'], '%Y-%m-%d')
        diff = abs((snapshot_date - target_date).days)
        if diff < min_diff:
            min_diff = diff
            closest_snapshot = snapshot
    
    return closest_snapshot if min_diff <= 7 else None  # Within a week

def format_currency(value: float) -> str:
    """Format currency values with appropriate styling"""
    if value >= 0:
        return f"${value:,.2f}"
    else:
        return f"-${abs(value):,.2f}"

def format_percentage(value: float) -> str:
    """Format percentage values with appropriate styling"""
    if value >= 0:
        return f"+{value:.2f}%"
    else:
        return f"{value:.2f}%"

def generate_dynamic_badges(metrics: Dict) -> str:
    """Generate dynamic badges for README header"""
    
    # Format values for badges (URL encoding)
    account_value = f"${metrics['account_value']:,.0f}"
    total_pnl = metrics['total_pnl']
    
    # Determine colors based on P&L
    pnl_color = "brightgreen" if total_pnl >= 0 else "red"
    pnl_value = f"${abs(total_pnl):,.0f}" if total_pnl >= 0 else f"-${abs(total_pnl):,.0f}"
    
    # Calculate return percentage badge - always show if we have P&L data
    return_badge = ""
    if metrics.get('total_pnl_pct', 0) != 0:
        return_pct = metrics['total_pnl_pct']
        return_color = "brightgreen" if return_pct >= 0 else "red"
        return_value = f"+{return_pct:.1f}%" if return_pct >= 0 else f"{return_pct:.1f}%"
        return_badge = f"![Return](https://img.shields.io/badge/Return-{urllib.parse.quote(return_value)}-{return_color})\n"
    
    # Generate trading status badge
    status_color = "brightgreen" if total_pnl >= 0 else "yellow"
    status_text = "PAPER"  # Always use PAPER for this educational system
    
    # Advanced metrics badges
    sharpe_badge = ""
    if metrics.get('sharpe_ratio', 0) > 0:
        sharpe_value = f"{metrics['sharpe_ratio']:.2f}"
        sharpe_color = "brightgreen" if metrics['sharpe_ratio'] > 1.5 else "green" if metrics['sharpe_ratio'] > 1.0 else "yellow"
        sharpe_badge = f"![Sharpe](https://img.shields.io/badge/Sharpe-{sharpe_value}-{sharpe_color})\n"
    
    drawdown_badge = ""
    if metrics.get('max_drawdown_pct', 0) > 0:
        dd_value = f"{metrics['max_drawdown_pct']:.1f}%"
        # More conservative colors since end-of-day data underestimates true drawdown
        dd_color = "green" if metrics['max_drawdown_pct'] < 2 else "yellow" if metrics['max_drawdown_pct'] < 5 else "red"
        drawdown_badge = f"![Max DD](https://img.shields.io/badge/Max_DD-{urllib.parse.quote(dd_value)}-{dd_color})\n"
    
    # Return just the dynamic badges - no license badge duplication
    badge_lines = [
        f"![Account Value](https://img.shields.io/badge/Account-{urllib.parse.quote(account_value)}-blue)",
        f"![P&L](https://img.shields.io/badge/P&L-{urllib.parse.quote(pnl_value)}-{pnl_color})"
    ]
    
    # Add return badge if available
    if return_badge.strip():
        badge_lines.append(return_badge.strip())
    
    # Add sharpe badge if available
    if sharpe_badge.strip():
        badge_lines.append(sharpe_badge.strip())
    
    # Add drawdown badge if available  
    if drawdown_badge.strip():
        badge_lines.append(drawdown_badge.strip())
    
    # Add status and last updated badges
    badge_lines.extend([
        f"![Status](https://img.shields.io/badge/Trading-{status_text}-{status_color})",
        f"![Last Updated](https://img.shields.io/badge/Last_Updated-{urllib.parse.quote(metrics['last_updated'].split()[0])}-lightgrey)"
    ])
    
    badges = '\n'.join(badge_lines)
    
    return badges

def generate_metrics_section(metrics: Dict, positions: List[Dict]) -> str:
    """Generate the live metrics section for README"""
    
    pnl_emoji = "📈" if metrics['total_pnl'] >= 0 else "📉"
    
    section = f"""
## 📊 Paper Trading Performance

> **Last Updated:** {metrics['last_updated']} | **Trading Days:** {metrics['trading_days']}

### Current Paper Account Status
| Metric | Value |
|--------|-------|
| **Account Value** | {format_currency(metrics['account_value'])} |
| **Total P&L** | {pnl_emoji} {format_currency(metrics['total_pnl'])} |
| **Unrealized P&L** | {format_currency(metrics['unrealized_pnl'])} |
| **Realized P&L** | {format_currency(metrics['realized_pnl'])} |
"""

    # Always show total return if we have it
    if metrics.get('total_pnl_pct', 0) != 0:
        section += f"| **Total Return** | {format_percentage(metrics['total_pnl_pct'])} |\n"
    
    # Add advanced risk metrics if available
    if metrics.get('sharpe_ratio', 0) > 0:
        section += f"| **Sharpe Ratio** | {metrics['sharpe_ratio']:.2f} |\n"
    if metrics.get('max_drawdown_pct', 0) > 0:
        section += f"| **Max Drawdown** | {metrics['max_drawdown_pct']:.1f}%* |\n"
    
    # Add current positions with enhanced detail
    if positions:
        section += f"\n### Current Positions\n"
        section += f"| Strategy | Symbol | Side | Contracts | Entry Price | Notional Value | Entry Date |\n"
        section += f"|----------|--------|------|-----------|-------------|----------------|------------|\n"
        
        total_notional = 0
        for pos in positions:
            section += f"| **{pos['strategy']}** | {pos['symbol']} | {pos['side']} | {pos['contracts']} | ${pos['entry_price']:.2f} | {format_currency(pos['notional_value'])} | {pos['entry_time']} |\n"
            total_notional += pos['notional_value']
        
        # Add position summary
        leverage = total_notional / metrics['account_value'] if metrics['account_value'] > 0 else 0
        section += f"\n**Portfolio Summary:** {len(positions)} positions, {format_currency(total_notional)} total notional, {leverage:.1f}x leverage\n"
    else:
        section += f"\n### Current Positions\n"
        section += f"*No positions currently open - waiting for entry signals*\n"
    
    # Add portfolio risk metrics with enhanced detail
    if positions:
        section += f"\n### Portfolio Risk Metrics\n"
        section += f"| Metric | Value | Notes |\n"
        section += f"|--------|-------|-------|\n"
        
        # Calculate risk metrics
        total_notional = sum(pos['notional_value'] for pos in positions)
        leverage = total_notional / metrics['account_value'] if metrics['account_value'] > 0 else 0
        
        section += f"| **Total Notional** | {format_currency(total_notional)} | Combined exposure |\n"
        section += f"| **Gross Leverage** | {leverage:.2f}x | Account value multiple |\n"
        section += f"| **Positions Count** | {len(positions)} | Active strategies |\n"
        
        # Calculate position concentration
        if total_notional > 0:
            largest_position = max(positions, key=lambda p: p['notional_value'])
            concentration = (largest_position['notional_value'] / total_notional) * 100
            section += f"| **Largest Position** | {concentration:.1f}% | {largest_position['symbol']} ({largest_position['strategy']}) |\n"
            
            # Add strategy breakdown
            strategy_exposure = {}
            for pos in positions:
                strategy_type = pos['strategy'].split('_')[0]  # IBS or Williams
                if strategy_type not in strategy_exposure:
                    strategy_exposure[strategy_type] = 0
                strategy_exposure[strategy_type] += pos['notional_value']
            
            for strategy_type, exposure in strategy_exposure.items():
                pct = (exposure / total_notional) * 100
                section += f"| **{strategy_type} Exposure** | {pct:.1f}% | {format_currency(exposure)} notional |\n"
    
    # Add period returns if available
    if any(key.endswith('_return') for key in metrics.keys()):
        section += f"\n### Recent Performance\n"
        section += f"| Period | Return |\n"
        section += f"|--------|--------|\n"
        
        if '1_week_return' in metrics:
            section += f"| **1 Week** | {format_percentage(metrics['1_week_return'])} |\n"
        if '1_month_return' in metrics:
            section += f"| **1 Month** | {format_percentage(metrics['1_month_return'])} |\n"
        if '3_months_return' in metrics:
            section += f"| **3 Months** | {format_percentage(metrics['3_months_return'])} |\n"
    
    section += f"\n*📝 Metrics automatically updated via GitHub Actions from paper trading IBKR account*\n"
    
    # Add disclaimer for max drawdown if it's being shown
    if metrics.get('max_drawdown_pct', 0) > 0:
        section += f"\n*⚠️ Max Drawdown calculated from end-of-day data only - may underestimate true intraday drawdown*\n"
    
    return section

def update_readme_with_metrics(metrics_section: str, badges: str) -> bool:
    """Update README.md with the new metrics section and dynamic badges"""
    try:
        with open('README.md', 'r') as f:
            content = f.read()
        
        # More robust badge replacement - find the License badge and replace everything after it until the description
        license_badge = '[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)'
        description_start = 'A comprehensive algorithmic trading system'
        
        license_pos = content.find(license_badge)
        desc_pos = content.find(description_start)
        
        if license_pos != -1 and desc_pos != -1:
            # Replace everything between the license badge and the description with our new badges
            before_badges = content[:license_pos + len(license_badge)]
            after_badges = content[desc_pos:]
            content = before_badges + '\n\n' + badges + '\n\n' + after_badges
        
        # Clean up the Paper Trading Performance section more robustly
        # Look for both live and paper trading sections for compatibility
        paper_trading_start = content.find('## 📊 Paper Trading Performance')
        live_trading_start = content.find('## 📊 Live Trading Performance')
        
        trading_start = paper_trading_start if paper_trading_start != -1 else live_trading_start
        
        if trading_start != -1:
            # Find the next major section (Trading Strategies)
            next_section = content.find('## 📊 Trading Strategies', trading_start + 1)
            if next_section == -1:
                # If Trading Strategies not found, look for Technical Architecture
                next_section = content.find('## 🔧 Technical Architecture', trading_start + 1)
            
            if next_section != -1:
                # Replace the entire Trading Performance section
                before_section = content[:trading_start]
                after_section = content[next_section:]
                content = before_section + metrics_section.strip() + '\n\n' + after_section
            else:
                # If no next section found, replace to end of file
                before_section = content[:trading_start]
                content = before_section + metrics_section.strip()
        else:
            # If no existing Trading Performance section, add after Key Features
            key_features_pattern = '## 🚀 Key Features'
            key_features_pos = content.find(key_features_pattern)
            if key_features_pos != -1:
                # Find the next section after Key Features
                next_section_patterns = ['## 📊 Trading Strategies', '## 🔧 Technical Architecture']
                next_section_pos = len(content)
                
                for pattern in next_section_patterns:
                    pos = content.find(pattern, key_features_pos + 1)
                    if pos != -1:
                        next_section_pos = min(next_section_pos, pos)
                
                if next_section_pos < len(content):
                    before_insert = content[:next_section_pos]
                    after_insert = content[next_section_pos:]
                    content = before_insert + '\n' + metrics_section.strip() + '\n\n' + after_insert
        
        # Write the updated content
        with open('README.md', 'w') as f:
            f.write(content)
        
        print("README.md updated successfully with live trading metrics and dynamic badges")
        return True
        
    except Exception as e:
        print(f"Error updating README: {e}")
        return False

def main():
    """Main function to update README with trading metrics"""
    print("Updating README with paper trading metrics, positions, and dynamic badges...")
    
    # Load trading metrics
    metrics = load_trading_metrics()
    if not metrics:
        print("No trading metrics available")
        return
    
    # Load position data
    positions = load_position_data()
    print(f"Loaded {len(positions)} open positions")
    
    # Generate dynamic badges
    badges = generate_dynamic_badges(metrics)
    
    # Generate metrics section
    metrics_section = generate_metrics_section(metrics, positions)
    
    # Update README
    success = update_readme_with_metrics(metrics_section, badges)
    
    if success:
        print("✅ README updated successfully with paper trading dynamic badges and positions")
    else:
        print("❌ Failed to update README")

if __name__ == "__main__":
    main() 