#!/usr/bin/env python3
"""
Simple example of using the Databento ES downloader.

Before running this script:
1. Set your DATABENTO_API_KEY in the .env file
2. Install dependencies: pip install -r requirements_databento.txt
"""

from databento_es_downloader import DatabentoESDownloader
from datetime import datetime, timedelta

def simple_download_example():
    """Simple example of downloading ES OHLCV-1s data."""
    
    # Initialize downloader
    downloader = DatabentoESDownloader()
    
    # Download data for yesterday (adjust as needed)
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    today = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Downloading ES data from {yesterday} to {today}")
    
    # Download and save to CSV
    df = downloader.download_es_ohlcv_1s(
        start_date=yesterday,
        end_date=today,
        save_to_csv=True
    )
    
    if not df.empty:
        print(f"Successfully downloaded {len(df)} rows")
        print("\nFirst few rows:")
        print(df.head())
    else:
        print("No data was downloaded. Check date range and market hours.")

if __name__ == "__main__":
    simple_download_example() 