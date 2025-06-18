#!/usr/bin/env python3
"""
Databento ES OHLCV-1s Data Downloader

This script downloads ES (E-mini S&P 500) futures OHLCV-1s data from Databento.
Requires a valid Databento API key stored in .env file.

Usage:
    python databento_es_downloader.py

Configuration:
    - Set your DATABENTO_API_KEY in the .env file
    - Modify the date range and symbols as needed
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import databento as db

# Load environment variables from .env file
load_dotenv()

class DatabentoESDownloader:
    def __init__(self):
        """Initialize the Databento client with API key from environment."""
        self.api_key = os.getenv('DATABENTO_API_KEY')
        if not self.api_key or self.api_key == 'your_databento_api_key_here':
            raise ValueError(
                "Please set your DATABENTO_API_KEY in the .env file. "
                "Get your API key from https://databento.com/portal/api-keys"
            )
        
        # Initialize the Historical client
        self.client = db.Historical(self.api_key)
        print(f"✓ Databento client initialized successfully")
    
    def download_es_ohlcv_1s(self, 
                            symbols=None, 
                            start_date=None, 
                            end_date=None, 
                            save_to_csv=True,
                            output_dir='Data'):
        """
        Download ES OHLCV-1s data from Databento.
        
        Args:
            symbols (list): List of ES symbols. If None, uses continuous contract.
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format  
            save_to_csv (bool): Whether to save data to CSV file
            output_dir (str): Directory to save CSV files
            
        Returns:
            pd.DataFrame: Downloaded OHLCV data
        """
        
        # Default parameters
        if symbols is None:
            # Use continuous contract for ES (front month)
            symbols = ['ES.n.0']  # ES continuous front month
            stype_in = 'continuous'
        else:
            stype_in = 'raw_symbol'
            
        if start_date is None:
            # Default to last trading day
            start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
        if end_date is None:
            # Default to today
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"📊 Downloading ES OHLCV-1s data...")
        print(f"   Symbols: {symbols}")
        print(f"   Date range: {start_date} to {end_date}")
        print(f"   Dataset: GLBX.MDP3 (CME Globex)")
        
        try:
            # Request OHLCV-1s data from Databento
            data = self.client.timeseries.get_range(
                dataset='GLBX.MDP3',  # CME Globex dataset for ES futures
                schema='ohlcv-1s',    # 1-second OHLCV bars
                symbols=symbols,
                stype_in=stype_in,
                start=start_date,
                end=end_date,
            )
            
            # Convert to pandas DataFrame
            df = data.to_df()
            
            if df.empty:
                print("⚠️  No data returned. Check your date range and market hours.")
                return df
            
            print(f"✓ Downloaded {len(df):,} rows of OHLCV-1s data")
            print(f"   Time range: {df.index[0]} to {df.index[-1]}")
            print(f"   Columns: {list(df.columns)}")
            
            # Display sample data
            print("\n📈 Sample data (first 5 rows):")
            print(df.head())
            
            # Display data statistics
            print(f"\n📊 Data Statistics:")
            print(f"   Open price range: ${df['open'].min():.2f} - ${df['open'].max():.2f}")
            print(f"   Volume range: {df['volume'].min():,} - {df['volume'].max():,}")
            print(f"   Total volume: {df['volume'].sum():,}")
            
            # Save to CSV if requested
            if save_to_csv:
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate filename with date range
                filename = f"es_ohlcv_1s_{start_date}_to_{end_date}.csv"
                filepath = os.path.join(output_dir, filename)
                
                # Save to CSV
                df.to_csv(filepath)
                print(f"💾 Data saved to: {filepath}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error downloading data: {str(e)}")
            print(f"   Check your API key and date range")
            raise
    
    def download_specific_contract(self, contract_symbol, start_date, end_date):
        """
        Download data for a specific ES contract (e.g., 'ESH4' for March 2024).
        
        Args:
            contract_symbol (str): Specific contract symbol (e.g., 'ESH4', 'ESM4')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: Downloaded OHLCV data
        """
        return self.download_es_ohlcv_1s(
            symbols=[contract_symbol],
            start_date=start_date,
            end_date=end_date
        )
    
    def get_cost_estimate(self, symbols, start_date, end_date):
        """
        Get cost estimate for the data request before downloading.
        
        Args:
            symbols (list): List of symbols
            start_date (str): Start date 
            end_date (str): End date
            
        Returns:
            dict: Cost information
        """
        try:
            cost_info = self.client.metadata.get_cost(
                dataset='GLBX.MDP3',
                schema='ohlcv-1s',
                symbols=symbols,
                stype_in='continuous' if symbols == ['ES.n.0'] else 'raw_symbol',
                start=start_date,
                end=end_date,
            )
            
            print(f"💰 Cost estimate: ${cost_info}")
            return cost_info
            
        except Exception as e:
            print(f"⚠️  Could not get cost estimate: {str(e)}")
            return None

def main():
    """Main function to demonstrate usage."""
    try:
        # Initialize the downloader
        downloader = DatabentoESDownloader()
        
        # Example 1: Download recent ES data (continuous contract)
        print("=" * 60)
        print("Example 1: Download ES continuous contract data")
        print("=" * 60)
        
        # Download last 2 days of data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        
        # Get cost estimate first (optional)
        downloader.get_cost_estimate(['ES.n.0'], start_date, end_date)
        
        # Download the data
        df = downloader.download_es_ohlcv_1s(
            start_date=start_date,
            end_date=end_date
        )
        
        # Example 2: Download specific contract
        print("\n" + "=" * 60)
        print("Example 2: Download specific ES contract (if available)")
        print("=" * 60)
        
        # Note: You'll need to adjust the contract symbol based on current available contracts
        # This is just an example - check Databento for current contract symbols
        # df_specific = downloader.download_specific_contract('ESH4', '2024-03-01', '2024-03-02')
        
        print("\n✓ Download completed successfully!")
        print("\nNote: Remember to:")
        print("1. Check the Data/ directory for your CSV files")
        print("2. Verify the data quality and completeness")
        print("3. Be aware of Databento API costs for large data requests")
        
    except Exception as e:
        print(f"\n❌ Error in main execution: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Check your DATABENTO_API_KEY in the .env file")
        print("2. Ensure you have databento library installed: pip install databento")
        print("3. Verify your API key is valid and has sufficient credits")
        print("4. Check if the requested date range includes trading days")

if __name__ == "__main__":
    main() 