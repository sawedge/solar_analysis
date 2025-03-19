import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def load_weather_data(station_id, data_file="historic_data.csv"):
    """
    Load and filter weather data for a specific station
    
    Args:
        station_id: Weather station ID
        data_file: Name of the data file to load from the data directory
        
    Returns:
        DataFrame with weather data for the specified station
    """
    file_path = os.path.join('data', data_file)
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
        process_temperature_columns(station_data)
        
        return station_data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def process_temperature_columns(df):
    """
    Process temperature columns in the DataFrame:
    - Calculate TAVG if not available
    - Convert temperature units if needed
    
    Args:
        df: DataFrame with weather data
    """
    # If TAVG is not available but TOBS is, use TOBS as TAVG
    if ('TAVG' not in df.columns or df['TAVG'].isna().all()) and 'TOBS' in df.columns:
        if not df['TOBS'].isna().all():
            print("Using TOBS (observed temperature) as TAVG (average temperature)")
            df['TAVG'] = df['TOBS']
    
    # If neither TAVG nor TOBS is available, but TMAX and TMIN are, calculate TAVG as their average
    if ('TAVG' not in df.columns or df['TAVG'].isna().all()) and 'TMAX' in df.columns and 'TMIN' in df.columns:
        tmax_na = df['TMAX'].isna().all()
        tmin_na = df['TMIN'].isna().all()
        if not (tmax_na or tmin_na):
            print("Calculating TAVG as average of TMAX and TMIN")
            df['TAVG'] = (df['TMAX'] + df['TMIN']) / 2

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

def create_pivot_table(monthly_data, value_column='TAVG'):
    """
    Create a pivot table with months as rows and years as columns
    
    Args:
        monthly_data: DataFrame with monthly average temperatures
        value_column: Column to use for the pivot table values
        
    Returns:
        Pivot table DataFrame
    """
    if monthly_data is None or monthly_data.empty:
        return None
    
    if value_column not in monthly_data.columns:
        print(f"Column {value_column} not found in data")
        return None
    
    # Create pivot table
    pivot = monthly_data.pivot(index='MONTH_NAME', columns='YEAR', values=value_column)
    
    # Sort rows by month order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
    pivot = pivot.reindex(month_order)
    
    # Calculate row means (average temperature by month across years)
    pivot['AVG'] = pivot.mean(axis=1, numeric_only=True)
    
    return pivot

def generate_temperature_summary(station_id, temp_column='TAVG'):
    """
    Generate a monthly temperature summary for a station
    
    Args:
        station_id: Weather station ID
        temp_column: Temperature column to analyze (TAVG, TMAX, TMIN)
        
    Returns:
        Pivot table with monthly temperature data
    """
    # Load weather data
    station_data = load_weather_data(station_id)
    
    if station_data is None or station_data.empty:
        print(f"No data available for station {station_id}")
        return None
    
    # Calculate monthly averages
    monthly_data = calculate_monthly_averages(station_data)
    
    if monthly_data is None or monthly_data.empty:
        print(f"No monthly data available for station {station_id}")
        return None
    
    # Create pivot table
    pivot_data = create_pivot_table(monthly_data, temp_column)
    
    # Return the processed data
    return pivot_data

def plot_monthly_temperature_trends(pivot_data, station_id, temp_type="average", save_path=None):
    """
    Create a plot of monthly temperature trends
    
    Args:
        pivot_data: Pivot table with monthly temperature data
        station_id: Weather station ID
        temp_type: Type of temperature (average, maximum, minimum)
        save_path: Path to save the plot image
    """
    if pivot_data is None or pivot_data.empty:
        print("No data available for plotting")
        return
    
    plt.figure(figsize=(14, 8))
    
    # Plot the AVG column (average across years)
    plt.plot(pivot_data.index, pivot_data['AVG'], 'k-', linewidth=3, label='Average across all years')
    
    # Plot data for each year
    years = [col for col in pivot_data.columns if col != 'AVG']
    for year in years:
        # Skip years with incomplete data
        if pivot_data[year].isna().sum() > 6:  # If more than half the months are missing
            continue
        plt.plot(pivot_data.index, pivot_data[year], 'o-', alpha=0.7, label=str(year))
    
    # Add labels and title
    plt.xlabel('Month')
    plt.ylabel('Temperature (°F)')
    plt.title(f'Monthly {temp_type.capitalize()} Temperatures for Station {station_id}')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def format_temperature_table(pivot_data, station_name, temp_type="Average"):
    """
    Format the pivot table for display
    
    Args:
        pivot_data: Pivot table with monthly temperature data
        station_name: Name of the weather station
        temp_type: Type of temperature (Average, Maximum, Minimum)
        
    Returns:
        Formatted string with the temperature table
    """
    if pivot_data is None or pivot_data.empty:
        return "No data available for table"
    
    # Round values for display
    formatted_data = pivot_data.round(1)
    
    # Create header
    header = f"\n{temp_type} Monthly Temperatures (°F) for {station_name}\n"
    header += "=" * 80 + "\n"
    
    # Generate table as string
    table_str = formatted_data.to_string()
    
    return header + table_str

def main():
    # Define stations
    stations = {
        "USW00094855": "OSHKOSH WITTMAN REGIONAL AIRPORT, WI US",
        "US1WIWN0031": "MENASHA 0.3 W, WI US"
    }
    
    # Select primary station for analysis
    primary_station_id = "USW00094855"  # Use Oshkosh data which is more complete
    station_name = stations[primary_station_id]
    
    # Generate temperature summaries
    print("\n=== Analyzing Average Temperatures (TAVG) ===")
    avg_temps = generate_temperature_summary(primary_station_id, 'TAVG')
    print(format_temperature_table(avg_temps, station_name, "Average"))
    
    print("\n=== Analyzing Maximum Temperatures (TMAX) ===")
    max_temps = generate_temperature_summary(primary_station_id, 'TMAX')
    print(format_temperature_table(max_temps, station_name, "Maximum"))
    
    print("\n=== Analyzing Minimum Temperatures (TMIN) ===")
    min_temps = generate_temperature_summary(primary_station_id, 'TMIN')
    print(format_temperature_table(min_temps, station_name, "Minimum"))
    
    # Create plots
    plots_dir = os.path.join('data', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_monthly_temperature_trends(
        avg_temps, 
        primary_station_id, 
        "average", 
        save_path=os.path.join(plots_dir, f'{primary_station_id}_avg_temps.png')
    )
    
    plot_monthly_temperature_trends(
        max_temps, 
        primary_station_id, 
        "maximum", 
        save_path=os.path.join(plots_dir, f'{primary_station_id}_max_temps.png')
    )
    
    plot_monthly_temperature_trends(
        min_temps, 
        primary_station_id, 
        "minimum", 
        save_path=os.path.join(plots_dir, f'{primary_station_id}_min_temps.png')
    )

if __name__ == "__main__":
    main() 