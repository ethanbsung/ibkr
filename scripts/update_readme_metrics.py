#!/usr/bin/env python3
"""
Script to update README.md with live trading metrics from account snapshots
"""
import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

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
            'trading_days': len(snapshots)
        }
        
        # Calculate performance over time if we have multiple snapshots
        if len(snapshots) > 1:
            # Calculate returns from first snapshot
            first_snapshot = sorted(snapshots, key=lambda x: x['date'])[0]
            initial_value = first_snapshot['net_liquidation'] - first_snapshot['unrealized_pnl'] - first_snapshot['realized_pnl']
            
            if initial_value > 0:
                total_return = (latest['net_liquidation'] - initial_value) / initial_value * 100
                metrics['total_pnl_pct'] = total_return
                
                # Calculate period returns
                periods = {'1_week': 7, '1_month': 30, '3_months': 90}
                for period_name, days_back in periods.items():
                    period_snapshot = get_snapshot_near_date(snapshots, latest['date'], days_back)
                    if period_snapshot:
                        period_return = (latest['net_liquidation'] - period_snapshot['net_liquidation']) / period_snapshot['net_liquidation'] * 100
                        metrics[f'{period_name}_return'] = period_return
        
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

def generate_metrics_section(metrics: Dict) -> str:
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

    if metrics['total_pnl_pct'] != 0:
        section += f"| **Total Return** | {format_percentage(metrics['total_pnl_pct'])} |\n"
    
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
    
    return section

def update_readme_with_metrics(metrics_section: str) -> bool:
    """Update README.md with the new metrics section"""
    try:
        with open('README.md', 'r') as f:
            content = f.read()
        
        # Pattern to match the existing live metrics section
        pattern = r'## 📊 Live Trading Performance.*?(?=##|\Z)'
        
        if re.search(pattern, content, re.DOTALL):
            # Replace existing section
            new_content = re.sub(pattern, metrics_section.strip(), content, flags=re.DOTALL)
        else:
            # Add new section after the Key Features section
            insert_pattern = r'(## 🚀 Key Features.*?)(\n## 📊 Trading Strategies)'
            replacement = r'\1\n' + metrics_section.strip() + r'\2'
            new_content = re.sub(insert_pattern, replacement, content, flags=re.DOTALL)
        
        with open('README.md', 'w') as f:
            f.write(new_content)
        
        print("README.md updated successfully with live trading metrics")
        return True
        
    except Exception as e:
        print(f"Error updating README: {e}")
        return False

def main():
    """Main function to update README with trading metrics"""
    print("Updating README with live trading metrics...")
    
    # Load trading metrics
    metrics = load_trading_metrics()
    if not metrics:
        print("No trading metrics available")
        return
    
    # Generate metrics section
    metrics_section = generate_metrics_section(metrics)
    
    # Update README
    success = update_readme_with_metrics(metrics_section)
    
    if success:
        print("✅ README updated successfully")
    else:
        print("❌ Failed to update README")

if __name__ == "__main__":
    main() 