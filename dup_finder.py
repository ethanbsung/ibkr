import pandas as pd
import os

# List of CSV files to check
csv_files = ['combined_es_1min_futures_data.csv']

# Function to load, check, and remove duplicates
def remove_duplicates(csv_file):
    try:
        # Load CSV
        df = pd.read_csv(
            csv_file,
            parse_dates=['date'],
            date_format="%Y-%m-%d %H:%M:%S%z"
        )
        df.set_index('date', inplace=True)

        # Find duplicates
        duplicates = df[df.index.duplicated(keep=False)]

        if not duplicates.empty:
            print(f"\nDuplicate indices found and removed in {csv_file}:")
            print(duplicates.sort_index())

            # Remove duplicates (keeping the first occurrence)
            df_cleaned = df[~df.index.duplicated(keep='first')]

            # Save cleaned file with a "_cleaned" suffix
            cleaned_file = os.path.splitext(csv_file)[0] + '_cleaned.csv'
            df_cleaned.to_csv(cleaned_file)
            print(f"Cleaned file saved as {cleaned_file}.")
        else:
            print(f"No duplicates found in {csv_file}.")

    except FileNotFoundError:
        print(f"File not found: {csv_file}")
    except pd.errors.EmptyDataError:
        print(f"No data: {csv_file} is empty.")
    except Exception as e:
        print(f"An error occurred while processing {csv_file}: {e}")

# Check and clean all files
for csv_file in csv_files:
    remove_duplicates(csv_file)