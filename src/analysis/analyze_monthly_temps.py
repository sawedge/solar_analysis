import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def analyze_monthly_temperatures(station_id="US1WIWN0031", use_historic=True):
    """
    Analyze average temperature by month for a specific weather station.
    
    Args:
        station_id: The station ID to analyze
        use_historic: Whether to use historic_data.csv (True) or cached_weather_data.csv (False)
    
    Returns:
        DataFrame with monthly average temperatures
    """
    # Get the path to the data file
    if use_historic:
        data_file = os.path.join('data', 'historic_data.csv')
    else:
        data_file = os.path.join('data', 'cached_weather_data.csv')
    
    # Load the weather data
    print(f"Loading weather data from {data_file}...")
    try:
        # Try to use chunking for large files
        chunksize = 100000  # Adjust based on available memory
        chunks = []
        
        for chunk in pd.read_csv(data_file, chunksize=chunksize, low_memory=False):
            # Filter for station in each chunk
            station_chunk = chunk[chunk['STATION'] == station_id]
            if not station_chunk.empty:
                chunks.append(station_chunk)
        
        if not chunks:
            print(f"No data found for station {station_id} in {data_file}")
            
            # Try the alternative file if the current one has no data
            if use_historic:
                print("Trying cached_weather_data.csv instead...")
                return analyze_monthly_temperatures(station_id, use_historic=False)
            return None
        
        # Combine chunks
        station_data = pd.concat(chunks, ignore_index=True)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        if use_historic:
            print("Trying cached_weather_data.csv instead...")
            return analyze_monthly_temperatures(station_id, use_historic=False)
        return None
    
    # Convert DATE to datetime
    station_data['DATE'] = pd.to_datetime(station_data['DATE'])
    
    print(f"Found {len(station_data)} records for station {station_id}")
    
    # Extract month and year from the date
    station_data['MONTH'] = station_data['DATE'].dt.month
    station_data['YEAR'] = station_data['DATE'].dt.year
    
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
    
    # Check which temperature columns are available
    temp_columns = []
    for col in ['TAVG', 'TMAX', 'TMIN', 'TOBS']:
        if col in station_data.columns and not station_data[col].isna().all():
            temp_columns.append(col)
    
    if not temp_columns:
        print("No temperature data available")
        return None
    
    print(f"Available temperature columns: {temp_columns}")
    
    # Create a dataframe for monthly averages
    monthly_avg = pd.DataFrame()
    
    # Group by year and month, then calculate average temperatures
    for col in temp_columns:
        # Skip columns with all NaN values
        if station_data[col].isna().all():
            continue
            
        # Calculate monthly averages
        temp_avg = station_data.groupby(['YEAR', 'MONTH'])[col].mean().reset_index()
        temp_avg.rename(columns={col: f'AVG_{col}'}, inplace=True)
        
        if monthly_avg.empty:
            monthly_avg = temp_avg
        else:
            monthly_avg = pd.merge(monthly_avg, temp_avg, on=['YEAR', 'MONTH'], how='outer')
    
    # Create a human-readable month name
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
        7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    monthly_avg['MONTH_NAME'] = monthly_avg['MONTH'].map(month_names)
    
    # Sort by year and month
    monthly_avg.sort_values(['YEAR', 'MONTH'], inplace=True)
    
    return monthly_avg

def generate_monthly_temperature_report(df, station_id):
    """
    Generate a report of monthly average temperatures.
    
    Args:
        df: DataFrame with monthly average temperatures
        station_id: The station ID being analyzed
    """
    if df is None or df.empty:
        print("No data available for report")
        return
    
    print("\n===== Monthly Temperature Report =====")
    print(f"Station ID: {station_id}")
    print(f"Data range: {df['YEAR'].min()}-{df['MONTH'].min()} to {df['YEAR'].max()}-{df['MONTH'].max()}")
    print("\nMonthly Average Temperatures (°F):")
    
    # Format for display
    display_df = df[['YEAR', 'MONTH_NAME']].copy()
    
    # Add the temperature columns
    for col in df.columns:
        if col.startswith('AVG_'):
            display_df[col.replace('AVG_', '')] = df[col].round(1)
    
    # Print as a formatted table
    print(display_df.to_string(index=False))
    
    # Calculate overall monthly averages (across years)
    if len(df['YEAR'].unique()) > 1 or len(df['MONTH'].unique()) > 1:
        print("\nOverall Monthly Averages (across all years):")
        monthly_means = df.groupby('MONTH_NAME').mean(numeric_only=True).reset_index()
        
        # Sort by month order
        month_order = {name: idx for idx, name in enumerate(
            ['January', 'February', 'March', 'April', 'May', 'June',
             'July', 'August', 'September', 'October', 'November', 'December']
        )}
        monthly_means['month_idx'] = monthly_means['MONTH_NAME'].map(month_order)
        monthly_means.sort_values('month_idx', inplace=True)
        monthly_means.drop('month_idx', axis=1, inplace=True)
        
        # Format for display
        display_overall = monthly_means[['MONTH_NAME']].copy()
        
        # Add the temperature columns
        for col in monthly_means.columns:
            if col.startswith('AVG_'):
                display_overall[col.replace('AVG_', '')] = monthly_means[col].round(1)
        
        print(display_overall.to_string(index=False))
    
    # If only one month is available, display a note
    else:
        print("\nNote: Data is only available for one month. More data is needed for a full monthly analysis.")

def plot_monthly_temperatures(df, station_id, save_path=None):
    """
    Create a plot of monthly average temperatures.
    
    Args:
        df: DataFrame with monthly average temperatures
        station_id: The station ID being analyzed
        save_path: Path to save the plot image, if None, the plot will be displayed
    """
    if df is None or df.empty:
        print("No data available for plotting")
        return
    
    # Check if we have multiple months to plot
    if len(df) <= 1:
        print("Not enough data for a meaningful plot")
        return
    
    # Create a figure and axis
    plt.figure(figsize=(12, 6))
    
    # Plot temperature data
    colors = {
        'AVG_TAVG': 'green', 
        'AVG_TMAX': 'red', 
        'AVG_TMIN': 'blue',
        'AVG_TOBS': 'purple'
    }
    
    # Create date strings for x-axis
    df['date_str'] = df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str)
    
    # Plot each temperature column
    for col in df.columns:
        if col.startswith('AVG_'):
            plt.plot(
                df['date_str'], 
                df[col], 
                marker='o', 
                linestyle='-', 
                color=colors.get(col, 'gray'),
                label=col.replace('AVG_', '')
            )
    
    # Add labels and title
    plt.xlabel('Year-Month')
    plt.ylabel('Temperature (°F)')
    plt.title(f'Monthly Average Temperatures for Station {station_id}')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    # Specify the station ID
    station_id = "US1WIWN0031"  # MENASHA 0.3 W, WI US
    
    # Analyze temperature data
    monthly_avg = analyze_monthly_temperatures(station_id)
    
    # Generate report
    generate_monthly_temperature_report(monthly_avg, station_id)
    
    # Create plot
    if monthly_avg is not None and len(monthly_avg) > 1:
        plot_filepath = os.path.join('data', f'{station_id}_monthly_temps.png')
        plot_monthly_temperatures(monthly_avg, station_id, save_path=plot_filepath)
    
    # If another station is needed for comparison or if primary station has no data
    backup_station = "USW00094855"  # OSHKOSH WITTMAN REGIONAL AIRPORT
    if monthly_avg is None or len(monthly_avg) <= 1:
        print(f"\nTrying backup station {backup_station}...")
        backup_avg = analyze_monthly_temperatures(backup_station)
        if backup_avg is not None and not backup_avg.empty:
            generate_monthly_temperature_report(backup_avg, backup_station)
            if len(backup_avg) > 1:
                backup_plot_filepath = os.path.join('data', f'{backup_station}_monthly_temps.png')
                plot_monthly_temperatures(backup_avg, backup_station, save_path=backup_plot_filepath)

if __name__ == "__main__":
    main() 