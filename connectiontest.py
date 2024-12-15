from ib_insync import*
# --- Configuration Parameters ---
IB_HOST = '127.0.0.1'  # IB Gateway/TWS host
IB_PORT = 4002         # IB Gateway paper trading API port
CLIENT_ID = 1          # Ensure a unique client ID

# --- Connect to IBKR ---
ib = IB()
print("Connecting to IBKR...")
try:
    ib.connect(host=IB_HOST, port=IB_PORT, clientId=CLIENT_ID, timeout=61)
    print("Connected to IBKR.")
    server_version = ib.serverVersion()
    print(f"IBKR Server Version: {server_version}")
except Exception as e:
    print(f"API connection failed: {e}")
finally:
    if ib.isConnected():
        ib.disconnect()
        print("Disconnected from IBKR.")