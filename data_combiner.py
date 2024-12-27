import pandas as pd

# Load the earlier and later CSV files
earlier_file = "esh25_intraday-nearby-1min_historical-data-download-12-26-2024-72.csv"  # Replace with your earlier file name
later_file = "es_1m_later.csv"  # Replace with your later file name

# Read the CSV files into dataframes
earlier_data = pd.read_csv(earlier_file)
later_data = pd.read_csv(later_file)

# Ensure the data is sorted by date for proper alignment
earlier_data.sort_values(by="Time", inplace=True)
later_data.sort_values(by="Time", inplace=True)

# Find the first timestamp in the later dataset
overlap_time = later_data["Time"].iloc[0]

# Drop rows in the earlier dataset that overlap with the later dataset
earlier_data = earlier_data[earlier_data["Time"] < overlap_time]

# Locate the overlapping row in both datasets for adjustment
earlier_overlap_row = earlier_data.tail(1)
later_overlap_row = later_data.head(1)

# Check if overlap exists
if earlier_overlap_row.empty or later_overlap_row.empty:
    raise ValueError("No overlapping timestamp found between the datasets!")

# Calculate the adjustment factor based on the 'Last' price
adjustment_factor = later_overlap_row["Last"].values[0] - earlier_overlap_row["Last"].values[0]
print(f"Adjustment Factor: {adjustment_factor}")

# Apply the adjustment to the earlier dataset
for column in ["Open", "High", "Low", "Last"]:
    earlier_data[column] += adjustment_factor

# Combine the adjusted earlier data and later data
combined_data = pd.concat([earlier_data, later_data])

# Ensure the combined data is sorted by time
combined_data.sort_values(by="Time", inplace=True)

# Save the combined dataset to overwrite the later file
combined_data.to_csv(later_file, index=False)
print(f"Combined data saved to {later_file}.")