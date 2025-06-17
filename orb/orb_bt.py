import pandas as pd
import numpy as np


'''
ORB does well on inside days, small range days, similar to RBR (rally base rally) and DBD (drop base drop)

Places buy and sell stops and uses one as entry and other as stop loss

Places orders 1 "stretch" above the high and 1 "stretch" below the low

NR7: After any day that has a daily range less than the previous 6 days


'''


def calculate_stretch(data, period=10):
    """
    Calculate the stretch for each day based on the previous 10 days.
    
    The stretch is determined by looking at the previous ten days and averaging 
    the sum of the differences between the open for each day and the closest 
    extreme to the open on each day.
    
    Parameters:
    data (pd.DataFrame): DataFrame with columns ['Open', 'High', 'Low']
    period (int): Number of previous days to look at (default: 10)
    
    Returns:
    pd.Series: Series with stretch values for each day
    """
    
    # Calculate the distance from open to high and open to low
    open_to_high = data['High'] - data['Open']
    open_to_low = data['Open'] - data['Low']
    
    # Find the closest extreme (minimum distance from open)
    closest_extreme_distance = np.minimum(open_to_high, open_to_low)
    
    # Calculate rolling sum of closest extreme distances over the specified period
    rolling_sum = closest_extreme_distance.rolling(window=period, min_periods=1).sum()
    
    # Calculate the stretch as the average (rolling sum divided by actual window size)
    # For periods with less than 'period' days, use the actual number of days available
    actual_window_size = closest_extreme_distance.rolling(window=period, min_periods=1).count()
    stretch = rolling_sum / actual_window_size
    
    return stretch

def load_and_calculate_stretch(csv_file, period=10):
    """
    Load data from CSV file and calculate stretch values.
    
    Parameters:
    csv_file (str): Path to the CSV file
    period (int): Number of previous days to look at (default: 10)
    
    Returns:
    pd.DataFrame: DataFrame with original data plus stretch column
    """
    
    # Load the data
    data = pd.read_csv(csv_file)
    
    # Convert Time column to datetime if it exists
    if 'Time' in data.columns:
        data['Time'] = pd.to_datetime(data['Time'])
        data.set_index('Time', inplace=True)
    
    # Calculate stretch
    data['Stretch'] = calculate_stretch(data, period)
    
    return data

# Example usage with MES data
if __name__ == "__main__":
    # Load MES data and calculate stretch
    mes_data = load_and_calculate_stretch('Data/mes_daily_data.csv')
    
    # Display first 20 rows with stretch calculation
    print("MES Data with Stretch Calculation:")
    print("=" * 80)
    print(mes_data[['Open', 'High', 'Low', 'Last', 'Stretch']].head(20))
    
    # Display some statistics
    print(f"\nStretch Statistics:")
    print(f"Mean Stretch: {mes_data['Stretch'].mean():.2f}")
    print(f"Std Stretch: {mes_data['Stretch'].std():.2f}")
    print(f"Min Stretch: {mes_data['Stretch'].min():.2f}")
    print(f"Max Stretch: {mes_data['Stretch'].max():.2f}")
