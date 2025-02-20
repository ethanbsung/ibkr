import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pytz

# --- Configuration ---
INITIAL_CAPITAL = 5000
CSV_FILE = "Data/es_1m_data.csv"
START_DATE = pd.Timestamp('2024-06-01', tz='UTC')
END_DATE   = pd.Timestamp('2024-12-31 23:59:59', tz='UTC')

# --- Load CSV Data ---
try:
    df = pd.read_csv(CSV_FILE, parse_dates=['Time'])
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit(1)

# Set the 'Time' column as the index.
df.set_index('Time', inplace=True)
df.index = pd.to_datetime(df.index)

# Ensure the index is timezone-aware; if not, localize to UTC.
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')

# Filter the data to only include dates from June 2024 to December 2024.
df = df[(df.index >= START_DATE) & (df.index <= END_DATE)]

# Rename the 'Last' column to 'close' for consistency.
if 'Last' in df.columns:
    df.rename(columns={'Last': 'close'}, inplace=True)
else:
    print("Error: CSV file does not contain a 'Last' column to be used as the close price.")
    exit(1)

# --- Compute the Equity Curve ---
# Calculate percentage change and cumulative returns.
df['pct_change'] = df['close'].pct_change().fillna(0)
df['cum_return'] = (1 + df['pct_change']).cumprod()
df['equity'] = INITIAL_CAPITAL * df['cum_return']

# --- Plot the Equity Curve ---
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['equity'], label='Equity Curve', color='blue')
plt.title("Equity Curve from CSV Data (June-Dec 2024)")
plt.xlabel("Time")
plt.ylabel("Equity ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()