import pandas as pd
from datetime import datetime
import os
import calendar
import re

def load_energy_production_data(file_path):
    """
    Load energy production data from CSV file
    
    Args:
        file_path: Path to the energy production data file
        
    Returns:
        DataFrame with energy production data
    """
    print(f"Loading energy production data from {file_path}...")
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check column names and rename if needed
        if 'Date/Time' in df.columns and 'Energy Produced (Wh)' in df.columns:
            df.rename(columns={'Date/Time': 'DATE', 'Energy Produced (Wh)': 'ENERGY_WH'}, inplace=True)
        
        # Convert date to datetime
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        # Clean energy values (remove commas and convert to numeric)
        df['ENERGY_WH'] = df['ENERGY_WH'].astype(str).apply(lambda x: re.sub(r'[^\d.]', '', x))
        df['ENERGY_WH'] = pd.to_numeric(df['ENERGY_WH'], errors='coerce')
        
        # Add month and year columns
        df['MONTH'] = df['DATE'].dt.month
        df['YEAR'] = df['DATE'].dt.year
        
        print(f"Loaded {len(df)} energy production records from {df['DATE'].min()} to {df['DATE'].max()}")
        return df
        
    except Exception as e:
        print(f"Error loading energy production data: {e}")
        return None

def calculate_monthly_averages(df):
    """
    Calculate monthly average temperatures from weather data
    
    Args:
        df: DataFrame with weather data
        
    Returns:
        DataFrame with monthly average temperatures
    """
    if df is None or df.empty:
        return None
    
    # Extract month and year
    df['MONTH'] = df['DATE'].dt.month
    df['YEAR'] = df['DATE'].dt.year
    
    # Check which temperature columns are available
    temp_columns = []
    for col in ['TAVG', 'TMAX', 'TMIN', 'TOBS']:
        if col in df.columns and not df[col].isna().all():
            temp_columns.append(col)
    
    if not temp_columns:
        print("No temperature data available")
        return None
    
    print(f"Available temperature columns: {temp_columns}")
    
    # Calculate monthly averages for each temperature column
    monthly_data = df.groupby(['YEAR', 'MONTH'])[temp_columns].mean().reset_index()
    
    # Create a human-readable month name
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
        7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    monthly_data['MONTH_NAME'] = monthly_data['MONTH'].map(month_names)
    
    # Sort by year and month
    monthly_data.sort_values(['YEAR', 'MONTH'], inplace=True)
    
    return monthly_data

def load_temperature_data(station_id="USW00094855", data_file="historic_data.csv"):
    """
    Load temperature data for a specific station
    
    Args:
        station_id: ID of the weather station
        data_file: Name of the data file to load from the data/raw directory
        
    Returns:
        DataFrame with monthly temperature data
    """
    # Define a function to load and process weather data
    def load_weather_data_from_raw(station_id, data_file):
        file_path = os.path.join('data', 'raw', data_file)
        print(f"Loading weather data for station {station_id} from {file_path}...")
        
        try:
            # Try to use chunking for large files
            chunksize = 100000  # Adjust based on available memory
            chunks = []
            
            for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
                # Filter for station in each chunk
                station_chunk = chunk[chunk['STATION'] == station_id]
                if not station_chunk.empty:
                    chunks.append(station_chunk)
            
            if not chunks:
                print(f"No data found for station {station_id} in {file_path}")
                return None
            
            # Combine chunks
            station_data = pd.concat(chunks, ignore_index=True)
            print(f"Found {len(station_data)} records for station {station_id}")
            
            # Convert DATE to datetime
            station_data['DATE'] = pd.to_datetime(station_data['DATE'])
            
            # Process temperature columns
            # If TAVG is not available but TOBS is, use TOBS as TAVG
            if ('TAVG' not in station_data.columns or station_data['TAVG'].isna().all()) and 'TOBS' in station_data.columns:
                if not station_data['TOBS'].isna().all():
                    print("Using TOBS (observed temperature) as TAVG (average temperature)")
                    station_data['TAVG'] = station_data['TOBS']
            
            # If neither TAVG nor TOBS is available, but TMAX and TMIN are, calculate TAVG as their average
            if ('TAVG' not in station_data.columns or station_data['TAVG'].isna().all()) and 'TMAX' in station_data.columns and 'TMIN' in station_data.columns:
                tmax_na = station_data['TMAX'].isna().all()
                tmin_na = station_data['TMIN'].isna().all()
                if not (tmax_na or tmin_na):
                    print("Calculating TAVG as average of TMAX and TMIN")
                    station_data['TAVG'] = (station_data['TMAX'] + station_data['TMIN']) / 2
            
            return station_data
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    # Load the weather data from the raw directory
    print(f"Loading temperature data for station {station_id}...")
    station_data = load_weather_data_from_raw(station_id, data_file)
    
    if station_data is None or station_data.empty:
        print(f"No temperature data available for station {station_id}")
        return None
    
    # Calculate monthly averages
    monthly_data = calculate_monthly_averages(station_data)
    
    if monthly_data is None or monthly_data.empty:
        print(f"No monthly temperature data available for station {station_id}")
        return None
    
    return monthly_data

def calculate_monthly_energy_production(energy_df):
    """
    Calculate the total energy production per month
    
    Args:
        energy_df: DataFrame with energy production data
        
    Returns:
        DataFrame with monthly energy production
    """
    # Group by year and month
    monthly_energy = energy_df.groupby(['YEAR', 'MONTH'])['ENERGY_WH'].sum().reset_index()
    
    # Add month name
    month_names = {i: calendar.month_name[i] for i in range(1, 13)}
    monthly_energy['MONTH_NAME'] = monthly_energy['MONTH'].map(month_names)
    
    # Convert Wh to kWh for better readability
    monthly_energy['ENERGY_KWH'] = monthly_energy['ENERGY_WH'] / 1000
    
    return monthly_energy

def merge_energy_and_temperature(energy_df, temp_df):
    """
    Merge energy production and temperature data
    
    Args:
        energy_df: DataFrame with monthly energy production
        temp_df: DataFrame with monthly temperature data
        
    Returns:
        DataFrame with merged data
    """
    # Make sure we have both datasets
    if energy_df is None or temp_df is None:
        print("Cannot merge datasets: One or both datasets are missing")
        return None
    
    # Merge on YEAR and MONTH
    merged_df = pd.merge(
        energy_df,
        temp_df[['YEAR', 'MONTH', 'TAVG', 'TMAX', 'TMIN']],
        on=['YEAR', 'MONTH'],
        how='inner'
    )
    
    # Calculate energy to temperature ratios
    merged_df['ENERGY_TAVG_RATIO'] = merged_df['ENERGY_KWH'] / merged_df['TAVG']
    merged_df['ENERGY_TMAX_RATIO'] = merged_df['ENERGY_KWH'] / merged_df['TMAX']
    merged_df['ENERGY_TMIN_RATIO'] = merged_df['ENERGY_KWH'] / merged_df['TMIN']
    
    # Sort by date
    merged_df.sort_values(['YEAR', 'MONTH'], inplace=True)
    
    return merged_df

def save_processed_data(df, filename, index=False):
    """
    Save processed data to the processed directory
    
    Args:
        df: DataFrame to save
        filename: Name of the CSV file to save
        index: Whether to save the DataFrame index
    """
    # Ensure the processed directory exists
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Create the full file path
    file_path = os.path.join(processed_dir, filename)
    
    # Save the data
    df.to_csv(file_path, index=index)
    print(f"Saved processed data to {file_path}")
    
    return file_path

def main():
    """
    Main function to process energy and temperature data
    """
    # File paths
    energy_file = os.path.join('data', 'raw', '223752_site_energy_production_report.csv')
    weather_file = 'historic_data.csv'
    
    # Create processed directory
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load energy production data
    energy_data = load_energy_production_data(energy_file)
    
    if energy_data is None:
        print("Cannot proceed without energy production data")
        return
    
    # Save raw energy data in a cleaned format
    save_processed_data(energy_data, 'energy_data_cleaned.csv')
    
    # Calculate monthly energy production
    monthly_energy = calculate_monthly_energy_production(energy_data)
    
    # Save monthly energy data
    save_processed_data(monthly_energy, 'monthly_energy.csv')
    
    # Load temperature data from the nearest weather station
    station_id = "USW00094855"  # OSHKOSH WITTMAN REGIONAL AIRPORT
    temp_data = load_temperature_data(station_id, weather_file)
    
    if temp_data is not None:
        # Save monthly temperature data
        save_processed_data(temp_data, 'monthly_temperature.csv')
    
    # Merge datasets
    merged_data = merge_energy_and_temperature(monthly_energy, temp_data)
    
    if merged_data is not None:
        # Save the merged dataset
        save_processed_data(merged_data, 'energy_temperature_merged.csv')
        
        # Basic summary
        print("\n=== Processing complete ===")
        print(f"Analysis period: {merged_data['YEAR'].min()}-{merged_data['MONTH'].min()} to {merged_data['YEAR'].max()}-{merged_data['MONTH'].max()}")
        print(f"Total data points: {len(merged_data)}")
        print(f"Saved merged data to {os.path.join(processed_dir, 'energy_temperature_merged.csv')}")
    else:
        print("Failed to merge energy and temperature data")

if __name__ == "__main__":
    main() 