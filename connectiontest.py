from ib_insync import IB

# --- Configuration Parameters ---
IB_HOST = '127.0.0.1'    # IB Gateway/TWS host
IB_PORT = 4000           # IB Gateway/TWS paper trading port
CLIENT_ID = 2            # Unique client ID (ensure it's unique)

# --- Connect to IBKR ---
ib = IB()
print("Connecting to IBKR...")
try:
    ib.connect(host=IB_HOST, port=IB_PORT, clientId=CLIENT_ID, timeout=30)  # Increased timeout
    print("Connected.")
    server_version = ib.serverVersion()
    print(f"IBKR Server Version: {server_version}")
except Exception as e:
    print(f"API connection failed: {e}")
finally:
    if ib.isConnected():
        ib.disconnect()
        print("Disconnected from IBKR.")