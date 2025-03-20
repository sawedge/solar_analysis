import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def load_temperature_data(station_id="USW00094855", data_file="historic_data.csv"):
    """
    Load temperature data for a specific station using the monthly_temperature_analysis.py logic
    
    Args:
        station_id: ID of the weather station
        data_file: Name of the data file to load from the data/raw directory
        
    Returns:
        DataFrame with monthly temperature data (pivot table)
    """
    # We'll use the functions from monthly_temperature_analysis.py
    from src.analysis.monthly_temperature_analysis import calculate_monthly_averages
    
    # Define a modified version of load_weather_data to use our new data path
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
            
            # Process temperature columns (copied from monthly_temperature_analysis.py)
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
    
    # Return the monthly data (not the pivot table)
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

def plot_energy_temp_ratio(df, save_path=None):
    """
    Create plots of energy production, temperature, and their ratio over time
    
    Args:
        df: DataFrame with merged energy and temperature data
        save_path: Directory to save plots
    """
    if df is None or df.empty:
        print("No data available for plotting")
        return
    
    # Create date strings for x-axis
    df['DATE_STR'] = df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    
    # Plot 1: Energy Production
    axes[0].plot(df['DATE_STR'], df['ENERGY_KWH'], 'o-', color='blue')
    axes[0].set_title('Monthly Energy Production (kWh)')
    axes[0].set_ylabel('Energy (kWh)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Average Temperature
    axes[1].plot(df['DATE_STR'], df['TAVG'], 'o-', color='red')
    axes[1].set_title('Monthly Average Temperature (°F)')
    axes[1].set_ylabel('Temperature (°F)')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Energy/Temp Ratio
    axes[2].plot(df['DATE_STR'], df['ENERGY_TAVG_RATIO'], 'o-', color='purple')
    axes[2].set_title('Energy Production to Average Temperature Ratio (kWh/°F)')
    axes[2].set_ylabel('Ratio (kWh/°F)')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Energy vs Temperature scatter plot
    axes[3].scatter(df['TAVG'], df['ENERGY_KWH'], alpha=0.7)
    axes[3].set_title('Energy Production vs. Average Temperature')
    axes[3].set_xlabel('Temperature (°F)')
    axes[3].set_ylabel('Energy (kWh)')
    axes[3].grid(True, alpha=0.3)
    
    # Add trend line to scatter plot
    if len(df) > 1:
        z = np.polyfit(df['TAVG'], df['ENERGY_KWH'], 1)
        p = np.poly1d(z)
        axes[3].plot(
            [df['TAVG'].min(), df['TAVG'].max()], 
            [p(df['TAVG'].min()), p(df['TAVG'].max())], 
            "r--", 
            alpha=0.7
        )
        # Add correlation coefficient
        corr = df['TAVG'].corr(df['ENERGY_KWH'])
        axes[3].annotate(
            f"Correlation: {corr:.2f}", 
            xy=(0.05, 0.95), 
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    # Format x-axis for date plots
    for ax in axes[:3]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save or display the plots
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def analyze_seasonal_patterns(df):
    """
    Analyze seasonal patterns in energy production and temperature ratio
    
    Args:
        df: DataFrame with merged energy and temperature data
        
    Returns:
        DataFrame with seasonal analysis
    """
    if df is None or df.empty:
        print("No data available for seasonal analysis")
        return None
    
    # Define seasons (meteorological seasons in Northern Hemisphere)
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    df['SEASON'] = df['MONTH'].map(season_map)
    
    # Group by season
    seasonal_analysis = df.groupby('SEASON').agg({
        'ENERGY_KWH': 'mean',
        'TAVG': 'mean',
        'ENERGY_TAVG_RATIO': 'mean'
    }).reset_index()
    
    # Sort by season
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    seasonal_analysis['SEASON_ORDER'] = seasonal_analysis['SEASON'].map(
        {season: i for i, season in enumerate(season_order)}
    )
    seasonal_analysis.sort_values('SEASON_ORDER', inplace=True)
    seasonal_analysis.drop('SEASON_ORDER', axis=1, inplace=True)
    
    # Round values for display
    seasonal_analysis = seasonal_analysis.round(2)
    
    return seasonal_analysis

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
    # File paths
    energy_file = os.path.join('data', 'raw', '223752_site_energy_production_report.csv')
    weather_file = 'historic_data.csv'
    
    # Create output directories
    plots_dir = os.path.join('data', 'plots')
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(plots_dir, exist_ok=True)
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
    
    # Load temperature data from the nearest weather station with good data
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
        
        # Print summary statistics
        print("\n=== Energy Production to Temperature Ratio Analysis ===")
        print(f"Analysis period: {merged_data['YEAR'].min()}-{merged_data['MONTH'].min()} to {merged_data['YEAR'].max()}-{merged_data['MONTH'].max()}")
        print(f"Total energy production: {merged_data['ENERGY_KWH'].sum():.2f} kWh")
        print(f"Average monthly energy production: {merged_data['ENERGY_KWH'].mean():.2f} kWh")
        print(f"Average energy/temperature ratio: {merged_data['ENERGY_TAVG_RATIO'].mean():.2f} kWh/°F")
        
        # Create a pivot table by year and month
        print("\n=== Monthly Energy to Temperature Ratio ===")
        pivot = merged_data.pivot_table(
            index='MONTH_NAME',
            columns='YEAR',
            values='ENERGY_TAVG_RATIO',
            aggfunc='mean'
        )
        
        # Reorder months
        month_order = [calendar.month_name[i] for i in range(1, 13)]
        pivot = pivot.reindex(month_order)
        
        # Calculate row means
        pivot['AVG'] = pivot.mean(axis=1)
        
        # Print pivot table
        print(pivot.round(2))
        
        # Save pivot table
        pivot_with_reset = pivot.reset_index()
        save_processed_data(pivot_with_reset, 'monthly_energy_temp_ratio_pivot.csv')
        
        # Analyze seasonal patterns
        seasonal_analysis = analyze_seasonal_patterns(merged_data)
        print("\n=== Seasonal Analysis ===")
        print(seasonal_analysis)
        
        # Save seasonal analysis
        save_processed_data(seasonal_analysis, 'seasonal_analysis.csv')
        
        # Create plots
        plot_path = os.path.join(plots_dir, 'energy_temp_ratio.png')
        plot_energy_temp_ratio(merged_data, plot_path)
        
        # Additional plot: Monthly ratio by year
        plt.figure(figsize=(14, 8))
        
        # Get years sorted
        years = sorted(merged_data['YEAR'].unique())
        
        # Create month-based data for each year
        for year in years:
            year_data = merged_data[merged_data['YEAR'] == year]
            plt.plot(year_data['MONTH'], year_data['ENERGY_TAVG_RATIO'], 'o-', label=str(year))
        
        plt.xlabel('Month')
        plt.ylabel('Energy/Temperature Ratio (kWh/°F)')
        plt.title('Monthly Energy to Temperature Ratio by Year')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(range(1, 13), [calendar.month_abbr[i] for i in range(1, 13)])
        plt.tight_layout()
        
        monthly_ratio_path = os.path.join(plots_dir, 'monthly_energy_temp_ratio.png')
        plt.savefig(monthly_ratio_path)
        print(f"Plot saved to {monthly_ratio_path}")
        
    else:
        print("Failed to merge energy and temperature data")

if __name__ == "__main__":
    main() 