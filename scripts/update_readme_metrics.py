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
                    
                    positions.append({
                        'strategy': strategy,
                        'symbol': symbol,
                        'side': 'Long' if pos.get('contracts', 0) > 0 else 'Short',
                        'contracts': abs(pos.get('contracts', 0)),
                        'entry_price': pos.get('entry_price', 0),
                        'entry_time': entry_time
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
            'realized_pnl': latest['realized_pnl'],
            'total_pnl': latest['unrealized_pnl'] + latest['realized_pnl'],
            'total_pnl_pct': 0,  # Will calculate if we have historical data
            'trading_days': len(snapshots),
            'sharpe_ratio': 0,  # Will calculate if enough data
            'max_drawdown_pct': 0,  # Will calculate if enough data
            'win_rate': 0,  # Placeholder for now
            'total_trades': 0  # Placeholder for now
        }
        
        # Calculate performance over time if we have multiple snapshots
        if len(snapshots) > 1:
            # Calculate returns from first snapshot (initial capital)
            first_snapshot = sorted(snapshots, key=lambda x: x['date'])[0]
            initial_value = first_snapshot['net_liquidation'] - first_snapshot['unrealized_pnl'] - first_snapshot['realized_pnl']
            
            if initial_value > 0:
                total_return = (latest['net_liquidation'] - initial_value) / initial_value * 100
                metrics['total_pnl_pct'] = total_return
                
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
                        
                        # Calculate Maximum Drawdown
                        peak = daily_values[0]
                        max_dd = 0
                        for value in daily_values:
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
            # If only one snapshot, try to estimate return from P&L
            total_pnl = latest['unrealized_pnl'] + latest['realized_pnl']
            estimated_initial = latest['net_liquidation'] - total_pnl
            if estimated_initial > 0:
                metrics['total_pnl_pct'] = (total_pnl / estimated_initial) * 100
        
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
    status_text = "LIVE" if metrics['trading_days'] > 0 else "PAPER"
    
    # Advanced metrics badges
    sharpe_badge = ""
    if metrics.get('sharpe_ratio', 0) > 0:
        sharpe_value = f"{metrics['sharpe_ratio']:.2f}"
        sharpe_color = "brightgreen" if metrics['sharpe_ratio'] > 1.5 else "green" if metrics['sharpe_ratio'] > 1.0 else "yellow"
        sharpe_badge = f"![Sharpe](https://img.shields.io/badge/Sharpe-{sharpe_value}-{sharpe_color})\n"
    
    drawdown_badge = ""
    if metrics.get('max_drawdown_pct', 0) > 0:
        dd_value = f"{metrics['max_drawdown_pct']:.1f}%25"
        # More conservative colors since end-of-day data underestimates true drawdown
        dd_color = "green" if metrics['max_drawdown_pct'] < 2 else "yellow" if metrics['max_drawdown_pct'] < 5 else "red"
        drawdown_badge = f"![Max DD](https://img.shields.io/badge/Max_DD-{urllib.parse.quote(dd_value)}-{dd_color})\n"
    
    badges = f"""[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Account Value](https://img.shields.io/badge/Account-{urllib.parse.quote(account_value)}-blue)
![P&L](https://img.shields.io/badge/P&L-{urllib.parse.quote(pnl_value)}-{pnl_color})
{return_badge}{sharpe_badge}{drawdown_badge}![Status](https://img.shields.io/badge/Trading-{status_text}-{status_color})
![Last Updated](https://img.shields.io/badge/Updated-{urllib.parse.quote(metrics['last_updated'].split()[0])}-lightgrey)"""
    
    return badges

def generate_metrics_section(metrics: Dict, positions: List[Dict]) -> str:
    """Generate the live metrics section for README"""
    
    pnl_emoji = "📈" if metrics['total_pnl'] >= 0 else "📉"
    
    section = f"""
## 📊 Live Trading Performance

> **Last Updated:** {metrics['last_updated']} | **Trading Days:** {metrics['trading_days']}

### Current Account Status
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
    
    # Add current positions
    if positions:
        section += f"\n### Current Positions\n"
        section += f"| Strategy | Symbol | Side | Contracts | Entry Price | Entry Date |\n"
        section += f"|----------|--------|------|-----------|-------------|------------|\n"
        
        for pos in positions:
            section += f"| **{pos['strategy']}** | {pos['symbol']} | {pos['side']} | {pos['contracts']} | ${pos['entry_price']:.2f} | {pos['entry_time']} |\n"
    else:
        section += f"\n### Current Positions\n"
        section += f"*No positions currently open - waiting for entry signals*\n"
    
    # Add portfolio risk metrics
    if positions:
        # Calculate portfolio exposure
        total_notional = 0
        contract_multipliers = {
            'MES': 5, 'MYM': 0.5, 'MGC': 10, 'MNQ': 2, 'ZQ': 2000
        }
        
        section += f"\n### Portfolio Risk Metrics\n"
        section += f"| Metric | Value |\n"
        section += f"|--------|-------|\n"
        
        for pos in positions:
            multiplier = contract_multipliers.get(pos['symbol'], 1)
            notional = pos['contracts'] * pos['entry_price'] * multiplier
            total_notional += notional
        
        leverage = total_notional / metrics['account_value'] if metrics['account_value'] > 0 else 0
        section += f"| **Total Notional** | {format_currency(total_notional)} |\n"
        section += f"| **Gross Leverage** | {leverage:.1f}x |\n"
        section += f"| **Risk per Position** | {100/len(positions):.1f}% avg allocation |\n"
        
        # Calculate position concentration
        largest_position = max(positions, key=lambda p: contract_multipliers.get(p['symbol'], 1) * p['contracts'] * p['entry_price'])
        largest_notional = contract_multipliers.get(largest_position['symbol'], 1) * largest_position['contracts'] * largest_position['entry_price']
        concentration = (largest_notional / total_notional) * 100 if total_notional > 0 else 0
        section += f"| **Largest Position** | {concentration:.1f}% ({largest_position['symbol']}) |\n"
    
    # Add period returns if available
    if '1_week_return' in metrics:
        section += f"\n### Recent Performance\n"
        section += f"| Period | Return |\n"
        section += f"|--------|--------|\n"
        
        if '1_week_return' in metrics:
            section += f"| **1 Week** | {format_percentage(metrics['1_week_return'])} |\n"
        if '1_month_return' in metrics:
            section += f"| **1 Month** | {format_percentage(metrics['1_month_return'])} |\n"
        if '3_months_return' in metrics:
            section += f"| **3 Months** | {format_percentage(metrics['3_months_return'])} |\n"
    
    section += f"\n*📝 Metrics automatically updated via GitHub Actions from live IBKR account*\n"
    
    # Add disclaimer for max drawdown if it's being shown
    if metrics.get('max_drawdown_pct', 0) > 0:
        section += f"\n*⚠️ Max Drawdown calculated from end-of-day data only - may underestimate true intraday drawdown*\n"
    
    return section

def update_readme_with_metrics(metrics_section: str, badges: str) -> bool:
    """Update README.md with the new metrics section and dynamic badges"""
    try:
        with open('README.md', 'r') as f:
            content = f.read()
        
        # Replace all badges (static + dynamic) in one go
        # Pattern to match from Python badge to end of badges
        badge_section_pattern = r'([![]Python[^\n]*\n[![]Interactive Brokers[^\n]*\n)([![]License[^\n]*\n\n(!\[[^\]]+\][^\n]*\n)*)?'
        
        if re.search(badge_section_pattern, content, re.DOTALL):
            # Replace the entire badge section
            badge_replacement = r'\1' + badges + r'\n'
            content = re.sub(badge_section_pattern, badge_replacement, content, flags=re.DOTALL)
        else:
            # Fallback: just add after Interactive Brokers badge
            ib_pattern = r'([![]Interactive Brokers[^\n]*\n)'
            if re.search(ib_pattern, content):
                badge_replacement = r'\1' + badges + r'\n'
                content = re.sub(ib_pattern, badge_replacement, content)
        
        # Update live metrics section
        pattern = r'## 📊 Live Trading Performance.*?(?=##|\Z)'
        
        if re.search(pattern, content, re.DOTALL):
            # Replace existing section
            content = re.sub(pattern, metrics_section.strip(), content, flags=re.DOTALL)
        else:
            # Add new section after the Key Features section
            insert_pattern = r'(## 🚀 Key Features.*?)(\n## 📊 Trading Strategies)'
            replacement = r'\1\n' + metrics_section.strip() + r'\2'
            content = re.sub(insert_pattern, replacement, content, flags=re.DOTALL)
        
        new_content = content
        
        with open('README.md', 'w') as f:
            f.write(new_content)
        
        print("README.md updated successfully with live trading metrics and dynamic badges")
        return True
        
    except Exception as e:
        print(f"Error updating README: {e}")
        return False

def main():
    """Main function to update README with trading metrics"""
    print("Updating README with live trading metrics, positions, and dynamic badges...")
    
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
        print("✅ README updated successfully with dynamic badges and positions")
    else:
        print("❌ Failed to update README")

if __name__ == "__main__":
    main() 