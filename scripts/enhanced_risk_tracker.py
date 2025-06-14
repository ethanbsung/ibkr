#!/usr/bin/env python3
"""
Enhanced risk tracking system for accurate Sharpe ratio and Maximum Drawdown calculations
"""

import json
import time
from datetime import datetime, timedelta
import pytz
from ib_insync import *
import pandas as pd
import numpy as np

class EnhancedRiskTracker:
    def __init__(self):
        self.ib = None
        self.data_file = 'account_snapshots/enhanced_risk_data.json'
        self.market_tz = pytz.timezone('US/Eastern')
        
    def connect_to_ibkr(self):
        """Connect to IBKR"""
        self.ib = IB()
        try:
            self.ib.connect('127.0.0.1', 4002, clientId=3)
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def is_market_hours(self):
        """Check if currently in market hours"""
        now = datetime.now(self.market_tz)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Check if weekday and within market hours
        return (now.weekday() < 5 and market_open <= now <= market_close)
    
    def get_current_account_value(self):
        """Get real-time account value"""
        try:
            account_values = self.ib.accountValues()
            for av in account_values:
                if av.tag == 'NetLiquidation' and av.currency == 'USD':
                    return float(av.value)
        except:
            pass
        return None
    
    def track_intraday_performance(self):
        """Track account value during market hours for true max drawdown"""
        if not self.is_market_hours():
            print("Market closed - no intraday tracking needed")
            return
        
        print("Starting intraday performance tracking...")
        intraday_values = []
        
        while self.is_market_hours():
            current_value = self.get_current_account_value()
            if current_value:
                timestamp = datetime.now(self.market_tz).isoformat()
                intraday_values.append({
                    'timestamp': timestamp,
                    'account_value': current_value
                })
                print(f"{timestamp}: ${current_value:,.2f}")
            
            time.sleep(300)  # Check every 5 minutes
        
        # Save intraday data
        self.save_intraday_data(intraday_values)
        
    def save_intraday_data(self, intraday_values):
        """Save intraday tracking data"""
        try:
            # Load existing data
            data = self.load_risk_data()
            
            # Add today's intraday data
            today = datetime.now(self.market_tz).strftime('%Y-%m-%d')
            data['intraday_tracking'][today] = intraday_values
            
            # Calculate true max drawdown for today
            if intraday_values:
                values = [v['account_value'] for v in intraday_values]
                peak = values[0]
                max_dd_today = 0
                
                for value in values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    max_dd_today = max(max_dd_today, drawdown)
                
                data['daily_max_drawdowns'][today] = {
                    'max_drawdown_pct': max_dd_today * 100,
                    'peak_value': peak,
                    'trough_value': min(values),
                    'daily_range': max(values) - min(values)
                }
            
            self.save_risk_data(data)
            
        except Exception as e:
            print(f"Error saving intraday data: {e}")
    
    def capture_daily_close(self):
        """Capture precise post-market data for Sharpe ratio"""
        try:
            post_market_time = datetime.now(self.market_tz).replace(hour=17, minute=0, second=0, microsecond=0)
            now = datetime.now(self.market_tz)
            
            # Only capture if within 5 minutes of 5 PM ET
            if abs((now - post_market_time).total_seconds()) < 300:
                account_value = self.get_current_account_value()
                if account_value:
                    data = self.load_risk_data()
                    today = now.strftime('%Y-%m-%d')
                    
                    data['daily_closes'][today] = {
                        'date': today,
                        'account_value': account_value,
                        'timestamp': now.isoformat(),
                        'post_market': True
                    }
                    
                    self.save_risk_data(data)
                    print(f"Post-market capture (5 PM ET): ${account_value:,.2f}")
                    return True
            
        except Exception as e:
            print(f"Error capturing daily close: {e}")
        
        return False
    
    def load_risk_data(self):
        """Load existing risk tracking data"""
        try:
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'daily_closes': {},
                'intraday_tracking': {},
                'daily_max_drawdowns': {},
                'metadata': {
                    'created': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat()
                }
            }
    
    def save_risk_data(self, data):
        """Save risk tracking data"""
        import os
        os.makedirs('account_snapshots', exist_ok=True)
        
        data['metadata']['last_updated'] = datetime.now().isoformat()
        
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def calculate_accurate_sharpe_ratio(self, days=30):
        """Calculate Sharpe ratio from consistent daily close data"""
        data = self.load_risk_data()
        daily_closes = data.get('daily_closes', {})
        
        if len(daily_closes) < 7:
            return None, "Need at least 7 days of daily close data"
        
        # Sort by date and calculate daily returns
        sorted_dates = sorted(daily_closes.keys())[-days:]
        values = [daily_closes[date]['account_value'] for date in sorted_dates]
        
        returns = []
        for i in range(1, len(values)):
            daily_return = (values[i] - values[i-1]) / values[i-1]
            returns.append(daily_return)
        
        if not returns:
            return None, "Insufficient return data"
        
        # Calculate annualized Sharpe ratio
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return None, "Zero volatility - cannot calculate Sharpe"
        
        # Assume 0% risk-free rate, annualize with sqrt(252)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
        
        return sharpe_ratio, f"Based on {len(returns)} daily returns"
    
    def calculate_true_max_drawdown(self):
        """Calculate maximum drawdown using intraday data when available"""
        data = self.load_risk_data()
        
        # Option 1: Use intraday data if available
        daily_max_dds = data.get('daily_max_drawdowns', {})
        if daily_max_dds:
            max_intraday_dd = max(dd['max_drawdown_pct'] for dd in daily_max_dds.values())
            return max_intraday_dd, "Based on intraday tracking"
        
        # Option 2: Fallback to daily close data
        daily_closes = data.get('daily_closes', {})
        if len(daily_closes) < 2:
            return None, "Need at least 2 days of data"
        
        sorted_dates = sorted(daily_closes.keys())
        values = [daily_closes[date]['account_value'] for date in sorted_dates]
        
        peak = values[0]
        max_drawdown = 0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown * 100, "Based on daily close values (may underestimate true max DD)"

def main():
    """Main function for enhanced risk tracking"""
    tracker = EnhancedRiskTracker()
    
    if not tracker.connect_to_ibkr():
        print("Failed to connect to IBKR")
        return
    
    print("Enhanced Risk Tracker Started")
    
    # Capture post-market data if it's close to 5 PM ET
    if tracker.capture_daily_close():
        print("Daily post-market data captured")
    
    # Calculate current metrics
    sharpe, sharpe_note = tracker.calculate_accurate_sharpe_ratio()
    max_dd, dd_note = tracker.calculate_true_max_drawdown()
    
    print(f"\nCurrent Risk Metrics:")
    print(f"Sharpe Ratio: {sharpe:.3f if sharpe else 'N/A'} ({sharpe_note})")
    print(f"Max Drawdown: {max_dd:.2f}% ({dd_note})" if max_dd else f"Max Drawdown: N/A ({dd_note})")
    
    # Start intraday tracking if market is open
    if tracker.is_market_hours():
        tracker.track_intraday_performance()
    
    tracker.ib.disconnect()

if __name__ == "__main__":
    main() 