from ib_insync import IB
from ib_insync import *

# Configuration Parameters
IB_HOST = '127.0.0.1'
IB_PORT = 4002
CLIENT_ID = 1

# Connect to IBKR
ib = IB()
print("Connecting to IBKR...")
ib.connect(host=IB_HOST, port=IB_PORT, clientId=CLIENT_ID)
print("Connected.")

# Fetch IBKR server version as a simple test
server_version = ib.serverVersion()
print(f"IBKR Server Version: {server_version}")

# Disconnect
ib.disconnect()
print("Disconnected from IBKR.")