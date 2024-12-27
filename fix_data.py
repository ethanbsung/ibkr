import pandas as pd

# Load your 1-minute data and 30-minute data
df_1m = pd.read_csv("es_1m_data.csv")
df_30m = pd.read_csv("es_1h_data.csv")

# Align the column names and order
required_columns = ['Time', 'Symbol', 'Open', 'High', 'Low', 'Last', 'Change', '%Chg', 'Volume', 'Open Int']

# Strip column names to remove leading/trailing spaces
df_1m.columns = df_1m.columns.str.strip()
df_30m.columns = df_30m.columns.str.strip()

# Check if any columns are missing in the 30-minute data
missing_columns = set(required_columns) - set(df_30m.columns)
if missing_columns:
    print(f"Missing columns in 30-minute data: {missing_columns}")
    # Add missing columns with default values (e.g., 0 or NaN)
    for col in missing_columns:
        df_30m[col] = 0  # Default value can be adjusted if needed

# Reorder 30-minute columns to match the 1-minute data
df_30m = df_30m[required_columns]

# Save the aligned 30-minute data (optional)
df_30m.to_csv("es_1h_aligned.csv", index=False)

# Verify alignment
print("1-Minute Data Columns:", df_1m.columns.tolist())
print("30-Minute Data Columns (Aligned):", df_30m.columns.tolist())