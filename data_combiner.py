import pandas as pd
import numpy as np
import shutil
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the earlier and later CSV files
earlier_file = "ngf25_intraday-nearby-5min_historical-data-download-12-28-2024-44.csv"  # Replace with your earlier file name
later_file = "ng_5m_data.csv"  # Replace with your later file name


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
logging.info(f"Adjustment Factor: {adjustment_factor}")

# Apply the adjustment to the earlier dataset
for column in ["Open", "High", "Low", "Last"]:
    if column in earlier_data.columns:
        earlier_data[column] += adjustment_factor
    else:
        logging.warning(f"Column '{column}' not found in earlier_data.")

# Combine the adjusted earlier data and later data
combined_data = pd.concat([earlier_data, later_data])

# Ensure the combined data is sorted by time
combined_data.sort_values(by="Time", inplace=True)

# Define the columns to round
columns_to_round = ["Open", "High", "Low", "Last"]

# Round the specified columns to 2 decimal places
combined_data[columns_to_round] = combined_data[columns_to_round].round(2)

# Handle the 'Volume' column
# Step 1: Ensure 'Volume' is numeric
combined_data['Volume'] = pd.to_numeric(combined_data['Volume'], errors='coerce')

# Step 2: Check for non-finite values
non_finite_mask = ~np.isfinite(combined_data['Volume'])
if non_finite_mask.any():
    logging.info(f"Found {non_finite_mask.sum()} non-finite 'Volume' values. Handling them by setting to 0.")
    # Option 1: Fill non-finite values with 0
    combined_data.loc[non_finite_mask, 'Volume'] = 0
    # Option 2: Alternatively, you could drop these rows
    # combined_data = combined_data[~non_finite_mask]

# Step 3: Optionally, fill any remaining NaN values (if any)
combined_data['Volume'] = combined_data['Volume'].fillna(0)

# Step 4: Convert 'Volume' to integer
combined_data['Volume'] = combined_data['Volume'].astype(int)

# (Optional) If there are other numeric columns that should be integers, handle them similarly

# Save the combined dataset to overwrite the later file with two decimal precision
combined_data.to_csv(later_file, index=False, float_format='%.2f')
logging.info(f"Combined data saved to {later_file}.")

# (Optional) Validate the saved file
# test_data = pd.read_csv(later_file)
# logging.info("Sample of the combined data:")
# logging.info(test_data.head())
# logging.info("Data types of the combined data:")
# logging.info(test_data.dtypes)