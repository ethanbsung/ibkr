from ib_insync import *
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
import os
import json

# Contract multipliers from live_port.py for proper avgCost display
contract_multipliers = {
    'MES': 5,      # Micro E-mini S&P 500
    'MYM': 0.50,   # Micro Dow 
    'MGC': 10,     # Micro Gold
    'MNQ': 2       # Micro Nasdaq
}

def connect_to_ibkr():
    """Establish connection to IB Gateway"""
    ib = IB()
    try:
        # Connect to IB Gateway on localhost:4002 (paper trading) or localhost:4001 (live)
        # Note: IB Gateway uses different ports than TWS
        ib.connect('127.0.0.1', 4002, clientId=2)
        print("Successfully connected to IB Gateway")
        return ib
    except Exception as e:
        print(f"Error connecting to IB Gateway: {e}")
        print("Make sure IB Gateway is running and API is enabled")
        print("IB Gateway paper trading port: 4002, live trading port: 4001")
        return None

def get_account_summary(ib):
    """Fetch account summary including balance"""
    try:
        account = ib.managedAccounts()[0]  
        summary = ib.accountSummary(account)
        
        # Convert to DataFrame for better readability
        df = pd.DataFrame(summary)
        # Check the actual number of columns and set appropriate column names
        if not df.empty:
            if len(df.columns) == 5:
                df.columns = ['Account', 'Tag', 'Value', 'Currency', 'Model Code']
            elif len(df.columns) == 4:
                df.columns = ['Tag', 'Value', 'Currency', 'Account']
            else:
                print(f"Unexpected number of columns in account summary: {len(df.columns)}")
        return df
    except Exception as e:
        print(f"Error fetching account summary: {e}")
        return pd.DataFrame()

def get_key_account_values(ib):
    """Get key account values in a clean format"""
    try:
        account = ib.managedAccounts()[0]
        
        # Request specific account values we care about
        key_tags = ['NetLiquidation', 'TotalCashValue', 'UnrealizedPnL', 'RealizedPnL', 
                   'AvailableFunds', 'BuyingPower', 'GrossPositionValue']
        
        account_values = {}
        account_data = ib.accountValues(account)
        
        # Create a lookup dictionary from account values
        account_lookup = {av.tag: av for av in account_data}
        
        for tag in key_tags:
            if tag in account_lookup:
                av = account_lookup[tag]
                try:
                    value = float(av.value)
                    account_values[tag] = f"{value:,.2f} {av.currency}"
                except:
                    account_values[tag] = f"{av.value} {av.currency}"
            else:
                account_values[tag] = "N/A"
        
        return account_values
    except Exception as e:
        print(f"Error fetching key account values: {e}")
        return {}

def get_current_positions(ib):
    """Fetch current positions"""
    try:
        positions = ib.positions()
        if not positions:
            return pd.DataFrame()
        
        position_data = []
        for pos in positions:
            # Get the multiplier for this contract symbol
            multiplier = contract_multipliers.get(pos.contract.symbol, 1)
            
            # Divide avgCost by multiplier for futures contracts
            adjusted_avg_cost = pos.avgCost / multiplier if pos.contract.secType == 'FUT' and multiplier != 1 else pos.avgCost
            
            position_data.append({
                'Symbol': pos.contract.symbol,
                'SecType': pos.contract.secType,
                'Exchange': pos.contract.exchange,
                'Position': pos.position,
                'AvgCost': adjusted_avg_cost,
                'Account': pos.account
            })
        
        return pd.DataFrame(position_data)
    except Exception as e:
        print(f"Error fetching positions: {e}")
        return pd.DataFrame()

def get_portfolio_pnl(ib):
    """Get PnL for each position using portfolio data"""
    try:
        portfolio = ib.portfolio()
        if not portfolio:
            return pd.DataFrame()
        
        portfolio_data = []
        for item in portfolio:
            # Get the multiplier for this contract symbol
            multiplier = contract_multipliers.get(item.contract.symbol, 1)
            
            # Divide avgCost by multiplier for futures contracts
            adjusted_avg_cost = item.averageCost / multiplier if item.contract.secType == 'FUT' and multiplier != 1 else item.averageCost
            
            portfolio_data.append({
                'Symbol': item.contract.symbol,
                'SecType': item.contract.secType,
                'Position': item.position,
                'MarketPrice': item.marketPrice,
                'MarketValue': item.marketValue,
                'AvgCost': adjusted_avg_cost,
                'UnrealizedPnL': item.unrealizedPNL,
                'RealizedPnL': item.realizedPNL,
                'Account': item.account
            })
        
        return pd.DataFrame(portfolio_data)
    except Exception as e:
        print(f"Error fetching portfolio PnL: {e}")
        return pd.DataFrame()

def get_total_pnl_summary(ib):
    """Get total account PnL in both $ and %"""
    try:
        account = ib.managedAccounts()[0]
        
        # Subscribe to PnL updates
        pnl_data = ib.reqPnL(account)
        time.sleep(2)  # Wait for data to arrive
        
        # Get account values for percentage calculation
        account_values = ib.accountValues(account)
        account_lookup = {av.tag: av for av in account_values}
        
        net_liquidation = 0
        unrealized_pnl = 0
        realized_pnl = 0
        
        if 'NetLiquidation' in account_lookup:
            net_liquidation = float(account_lookup['NetLiquidation'].value)
        
        if pnl_data:
            if hasattr(pnl_data, 'unrealizedPnL') and not pd.isna(pnl_data.unrealizedPnL):
                unrealized_pnl = pnl_data.unrealizedPnL
            if hasattr(pnl_data, 'realizedPnL') and not pd.isna(pnl_data.realizedPnL):
                realized_pnl = pnl_data.realizedPnL
        
        # If PnL subscription didn't work, try account values
        if unrealized_pnl == 0 and 'UnrealizedPnL' in account_lookup:
            try:
                unrealized_pnl = float(account_lookup['UnrealizedPnL'].value)
            except:
                pass
                
        if realized_pnl == 0 and 'RealizedPnL' in account_lookup:
            try:
                realized_pnl = float(account_lookup['RealizedPnL'].value)
            except:
                pass
        
        total_pnl = unrealized_pnl + realized_pnl
        
        # Calculate percentage
        pnl_percentage = 0
        if net_liquidation > 0:
            pnl_percentage = (total_pnl / net_liquidation) * 100
        
        return {
            'Total_PnL_USD': f"${total_pnl:,.2f}",
            'Total_PnL_Percent': f"{pnl_percentage:.2f}%",
            'Unrealized_PnL': f"${unrealized_pnl:,.2f}",
            'Realized_PnL': f"${realized_pnl:,.2f}",
            'Net_Liquidation': f"${net_liquidation:,.2f}"
        }
        
    except Exception as e:
        print(f"Error fetching PnL summary: {e}")
        return {}

def save_daily_snapshot(ib):
    """Save daily account snapshot for historical tracking"""
    try:
        account_values = ib.accountValues()
        account_lookup = {av.tag: av for av in account_values}
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Get portfolio data for more detailed P&L calculation
        portfolio = ib.portfolio()
        
        # Calculate total realized P&L from all positions
        total_realized_pnl = 0
        total_unrealized_pnl = 0
        
        for item in portfolio:
            if hasattr(item, 'realizedPNL') and item.realizedPNL:
                total_realized_pnl += item.realizedPNL
            if hasattr(item, 'unrealizedPNL') and item.unrealizedPNL:
                total_unrealized_pnl += item.unrealizedPNL
        
        # Use account values as primary source, portfolio as backup
        net_liquidation = float(account_lookup.get('NetLiquidation', {}).value or 0)
        unrealized_pnl = float(account_lookup.get('UnrealizedPnL', {}).value or total_unrealized_pnl or 0)
        
        # For realized P&L, try to get cumulative value
        # IBKR sometimes resets daily, so we'll track cumulative ourselves
        account_realized_pnl = float(account_lookup.get('RealizedPnL', {}).value or 0)
        
        # Load existing snapshots to calculate cumulative realized P&L
        snapshots_file = 'account_snapshots/daily_snapshots.json'
        existing_snapshots = []
        if os.path.exists(snapshots_file):
            with open(snapshots_file, 'r') as f:
                existing_snapshots = json.load(f)
        
        # Calculate cumulative realized P&L if we have historical data
        cumulative_realized_pnl = account_realized_pnl
        if existing_snapshots:
            # Check if we have yesterday's data
            yesterday_snapshots = [s for s in existing_snapshots if s['date'] < today]
            if yesterday_snapshots:
                last_snapshot = sorted(yesterday_snapshots, key=lambda x: x['date'])[-1]
                
                # If current account shows 0 realized P&L but we had some before,
                # it means IBKR reset the counter, so we maintain our cumulative tracking
                if account_realized_pnl == 0 and last_snapshot.get('cumulative_realized_pnl', 0) != 0:
                    # Check if account value changed in a way that suggests realized gains
                    expected_account_value = last_snapshot['net_liquidation'] - last_snapshot['unrealized_pnl'] + unrealized_pnl
                    account_value_diff = net_liquidation - expected_account_value
                    
                    # If there's a difference, it's likely realized P&L
                    if abs(account_value_diff) > 0.01:  # More than 1 cent difference
                        cumulative_realized_pnl = last_snapshot.get('cumulative_realized_pnl', 0) + account_value_diff
                    else:
                        cumulative_realized_pnl = last_snapshot.get('cumulative_realized_pnl', 0)
                else:
                    # Normal case: add to existing cumulative
                    cumulative_realized_pnl = last_snapshot.get('cumulative_realized_pnl', 0) + account_realized_pnl
        
        snapshot = {
            'date': today,
            'net_liquidation': net_liquidation,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': account_realized_pnl,  # Daily realized P&L from IBKR
            'cumulative_realized_pnl': cumulative_realized_pnl,  # Our cumulative tracking
            'total_cash': float(account_lookup.get('TotalCashValue', {}).value or 0),
            'gross_position_value': float(account_lookup.get('GrossPositionValue', {}).value or 0)
        }
        
        # Create snapshots directory if it doesn't exist
        os.makedirs('account_snapshots', exist_ok=True)
        
        # Add today's snapshot (replace if already exists)
        existing_snapshots = [s for s in existing_snapshots if s['date'] != today]
        existing_snapshots.append(snapshot)
        
        # Keep only last 400 days of data
        existing_snapshots = sorted(existing_snapshots, key=lambda x: x['date'])[-400:]
        
        # Save updated snapshots
        with open(snapshots_file, 'w') as f:
            json.dump(existing_snapshots, f, indent=2)
        
        print(f"Daily snapshot saved for {today}")
        print(f"  Net Liquidation: ${net_liquidation:,.2f}")
        print(f"  Unrealized P&L: ${unrealized_pnl:,.2f}")
        print(f"  Daily Realized P&L: ${account_realized_pnl:,.2f}")
        print(f"  Cumulative Realized P&L: ${cumulative_realized_pnl:,.2f}")
        return True
        
    except Exception as e:
        print(f"Error saving daily snapshot: {e}")
        return False

def get_historical_pnl_estimates(ib):
    """Calculate historical PnL performance over different periods"""
    try:
        # Save today's snapshot first
        save_daily_snapshot(ib)
        
        # Load historical snapshots
        snapshots_file = 'account_snapshots/daily_snapshots.json'
        if not os.path.exists(snapshots_file):
            return get_placeholder_historical_data()
        
        with open(snapshots_file, 'r') as f:
            snapshots = json.load(f)
        
        if len(snapshots) < 2:
            return get_placeholder_historical_data()
        
        # Sort by date
        snapshots = sorted(snapshots, key=lambda x: x['date'])
        current_snapshot = snapshots[-1]
        current_value = current_snapshot['net_liquidation']
        
        periods = {
            '1_Week': 7,
            '1_Month': 30,
            '3_Months': 90,
            '6_Months': 180,
            '1_Year': 365
        }
        
        historical_estimates = {}
        current_date = datetime.now()
        
        for period_name, days in periods.items():
            target_date = (current_date - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Find closest historical snapshot
            closest_snapshot = None
            min_diff = float('inf')
            
            for snapshot in snapshots:
                date_diff = abs((datetime.strptime(snapshot['date'], '%Y-%m-%d') - 
                               datetime.strptime(target_date, '%Y-%m-%d')).days)
                if date_diff < min_diff:
                    min_diff = date_diff
                    closest_snapshot = snapshot
            
            if closest_snapshot and min_diff <= 7:  # Within a week of target date
                historical_value = closest_snapshot['net_liquidation']
                pnl_dollar = current_value - historical_value
                pnl_percent = (pnl_dollar / historical_value * 100) if historical_value > 0 else 0
                
                historical_estimates[period_name] = {
                    'PnL_USD': f"${pnl_dollar:,.2f}",
                    'PnL_Percent': f"{pnl_percent:.2f}%",
                    'Days_Back': min_diff,
                    'From_Date': closest_snapshot['date']
                }
            else:
                historical_estimates[period_name] = {
                    'PnL_USD': "N/A - Insufficient historical data",
                    'PnL_Percent': "N/A - Insufficient historical data",
                    'Note': f"Need {days} days of data"
                }
        
        return historical_estimates
        
    except Exception as e:
        print(f"Error calculating historical PnL: {e}")
        return get_placeholder_historical_data()

def get_placeholder_historical_data():
    """Return placeholder historical data when no snapshots available"""
    periods = ['1_Week', '1_Month', '3_Months', '6_Months', '1_Year']
    
    historical_estimates = {}
    for period in periods:
        historical_estimates[period] = {
            'PnL_USD': "N/A - Start tracking to get historical data",
            'PnL_Percent': "N/A - Start tracking to get historical data",
            'Note': f"Run this script daily to build {period} history"
        }
    
    return historical_estimates

def display_account_info(ib):
    """Display comprehensive account information"""
    print("=" * 80)
    print("INTERACTIVE BROKERS ACCOUNT INFORMATION")
    print("=" * 80)
    
    # Key Account Values
    print("\n📊 KEY ACCOUNT VALUES:")
    print("-" * 40)
    key_values = get_key_account_values(ib)
    for key, value in key_values.items():
        print(f"{key:<20}: {value}")
    
    # Total PnL Summary
    print("\n💰 PROFIT & LOSS SUMMARY:")
    print("-" * 40)
    pnl_summary = get_total_pnl_summary(ib)
    for key, value in pnl_summary.items():
        if key == 'Total_PnL_USD':
            print(f"{'Total PnL ($)':<20}: {value}")
        elif key == 'Total_PnL_Percent':
            print(f"{'Total PnL (%)':<20}: {value}")
        elif 'Unrealized' in key:
            print(f"{'Unrealized PnL':<20}: {value}")
        elif 'Realized' in key:
            print(f"{'Realized PnL':<20}: {value}")
        elif 'Net_Liquidation' in key:
            print(f"{'Account Value':<20}: {value}")
    
    # Historical PnL (placeholder)
    print("\n📈 HISTORICAL PERFORMANCE:")
    print("-" * 40)
    historical_pnl = get_historical_pnl_estimates(ib)
    for period, data in historical_pnl.items():
        print(f"\n{period.replace('_', ' ')}:")
        if isinstance(data, dict):
            for key, value in data.items():
                if key == 'PnL_USD':
                    print(f"  PnL ($): {value}")
                elif key == 'PnL_Percent':
                    print(f"  PnL (%): {value}")
                elif key == 'Days_Back':
                    print(f"  Days back: {value}")
                elif key == 'From_Date':
                    print(f"  From date: {value}")
                elif key == 'Note':
                    print(f"  Note: {value}")
    
    # Current Positions
    print("\n📋 CURRENT POSITIONS:")
    print("-" * 40)
    positions_df = get_current_positions(ib)
    if not positions_df.empty:
        print(positions_df.to_string(index=False))
    else:
        print("No positions found")
    
    # Portfolio PnL
    print("\n💹 POSITION PnL DETAILS:")
    print("-" * 40)
    portfolio_df = get_portfolio_pnl(ib)
    if not portfolio_df.empty:
        # Create a new DataFrame with formatted values to avoid pandas warnings
        formatted_data = []
        for index, row in portfolio_df.iterrows():
            formatted_row = {
                'Symbol': row['Symbol'],
                'SecType': row['SecType'],
                'Position': f"{row['Position']:.0f}" if pd.notna(row['Position']) else "N/A",
                'MarketPrice': f"{row['MarketPrice']:.2f}" if pd.notna(row['MarketPrice']) else "N/A",
                'MarketValue': f"{row['MarketValue']:,.2f}" if pd.notna(row['MarketValue']) else "N/A",
                'AvgCost': f"{row['AvgCost']:.2f}" if pd.notna(row['AvgCost']) else "N/A",
                'UnrealizedPnL': f"{row['UnrealizedPnL']:,.2f}" if pd.notna(row['UnrealizedPnL']) else "N/A",
                'RealizedPnL': f"{row['RealizedPnL']:,.2f}" if pd.notna(row['RealizedPnL']) else "N/A",
                'Account': row['Account']
            }
            formatted_data.append(formatted_row)
        
        formatted_df = pd.DataFrame(formatted_data)
        print(formatted_df.to_string(index=False))
    else:
        print("No portfolio items found")
    
    print("\n" + "=" * 80)

def main():
    """Main function to connect and fetch account information"""
    ib = connect_to_ibkr()
    
    if ib is None:
        print("Failed to connect to IB Gateway")
        return
    
    try:
        # Wait for connection to stabilize
        print("Waiting for connection to stabilize...")
        time.sleep(3)
        
        # Display all account information
        display_account_info(ib)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("\nDisconnected from IB Gateway")

if __name__ == "__main__":
    main()
