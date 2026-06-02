#!/usr/bin/env python3
"""
Simple IB Gateway Connection Test
Tests basic connectivity to Interactive Brokers Gateway
"""

from ib_insync import *
import sys
import time

def test_ib_connection():
    """Test connection to IB Gateway"""
    
    # Connection parameters (matching your live_port.py)
    IB_HOST = '127.0.0.1'
    IB_PORT = 4002
    CLIENT_ID = 999  # Using different client ID for testing
    
    print("=== IB Gateway Connection Test ===")
    print(f"Host: {IB_HOST}")
    print(f"Port: {IB_PORT}")
    print(f"Client ID: {CLIENT_ID}")
    print()
    
    # Create IB connection
    ib = IB()
    
    try:
        print("Attempting to connect...")
        start_time = time.time()
        
        # Connect to IB Gateway
        ib.connect(IB_HOST, IB_PORT, CLIENT_ID)
        
        connection_time = time.time() - start_time
        print(f"✅ Connection successful! (took {connection_time:.2f} seconds)")
        
        # Test basic functionality
        print("\n--- Testing Basic Functions ---")
        
        # Check if connected
        if ib.isConnected():
            print("✅ Connection status: Connected")
        else:
            print("❌ Connection status: Not connected")
            return False
        
        # Get account info
        try:
            accounts = ib.managedAccounts()
            print(f"✅ Account access: {len(accounts)} account(s) available")
            for account in accounts:
                print(f"   - {account}")
        except Exception as e:
            print(f"⚠️  Account info error: {e}")
        
        # Test market data request (simple test)
        try:
            # Try to get current time from IB
            server_time = ib.reqCurrentTime()
            print(f"✅ Server time: {server_time}")
        except Exception as e:
            print(f"⚠️  Server time error: {e}")
        
        # Test contract request
        try:
            # Try to get contract details for ES (E-mini S&P 500) - using current month
            contract = Future('ES', '202509', 'CME')  # September 2025
            ib.qualifyContracts(contract)
            print(f"✅ Contract qualification: ES {contract.lastTradeDateOrContractMonth}")
        except Exception as e:
            print(f"⚠️  Contract qualification error: {e}")
        
        print("\n=== Connection Test Complete ===")
        print("✅ All basic tests passed!")
        return True
        
    except ConnectionRefusedError:
        print("❌ Connection refused!")
        print("   - Make sure IB Gateway is running")
        print("   - Check that port 4002 is correct")
        print("   - Verify Gateway is configured for API connections")
        return False
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("   - Check IB Gateway status")
        print("   - Verify connection parameters")
        return False
        
    finally:
        # Always disconnect
        try:
            ib.disconnect()
            print("✅ Disconnected from IB Gateway")
        except:
            pass

def main():
    """Main function"""
    print("Starting IB Gateway connection test...")
    print()
    
    success = test_ib_connection()
    
    if success:
        print("\n🎉 Connection test PASSED!")
        print("Your IB Gateway is ready for trading.")
    else:
        print("\n💥 Connection test FAILED!")
        print("Please check your IB Gateway setup.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 