import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def create_energy_temp_timeline(data_file=None, save_path=None):
    """
    Create a dual-axis line graph showing energy production and temperature over time
    
    Args:
        data_file: Path to the merged energy and temperature data file
        save_path: Path to save the output plot
    """
    # Default file path if none provided
    if data_file is None:
        data_file = os.path.join('data', 'processed', 'energy_temperature_merged.csv')
    
    # Load the data
    print(f"Loading data from {data_file}")
    try:
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} data points")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Check required columns
    required_cols = ['YEAR', 'MONTH', 'ENERGY_KWH', 'TAVG']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Data is missing required columns: {missing_cols}")
        return
    
    # Sort data by date
    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
    df = df.sort_values('DATE')
    
    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot energy production on the first y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Energy Production (kWh)', color=color)
    ax1.plot(df['DATE'], df['ENERGY_KWH'], color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create a second y-axis for temperature
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Temperature (°F)', color=color)
    ax2.plot(df['DATE'], df['TAVG'], color=color, marker='s')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Format x-axis to show dates properly
    fig.autofmt_xdate()
    
    # Set title and add grid
    plt.title('Monthly Solar Production and Temperature')
    ax1.grid(True, alpha=0.3)
    
    # Save or display the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    return

def main():
    """Main function to create the timeline graph"""
    # Create output directory for plots
    plots_dir = os.path.join('data', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate line graph
    data_file = os.path.join('data', 'processed', 'energy_temperature_merged.csv')
    save_path = os.path.join(plots_dir, 'energy_temperature_timeline.png')
    create_energy_temp_timeline(data_file, save_path)

if __name__ == "__main__":
    main()