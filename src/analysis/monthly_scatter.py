import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def create_monthly_scatter(data_file=None, save_path=None):
    """
    Create a simple scatter plot of temperature vs energy produced
    
    Args:
        data_file: Path to the merged data file (energy + temperature)
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
    required_cols = ['TAVG', 'ENERGY_KWH']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Data is missing required columns: {missing_cols}")
        return
    
    # Create figure and axis
    plt.figure(figsize=(10, 8))
    
    # Create a simple scatter plot with single color
    plt.scatter(
        df['TAVG'],
        df['ENERGY_KWH'],
        color='blue',
        s=100,  # Marker size
        alpha=0.7
    )
    
    # Add labels and title
    plt.xlabel('Average Temperature (°F)')
    plt.ylabel('Energy Production (kWh)')
    plt.title('Monthly Solar Production vs. Average Temperature')
    plt.grid(True, alpha=0.3)
    
    # Add a trend line
    if len(df) > 1:
        z = np.polyfit(df['TAVG'], df['ENERGY_KWH'], 1)
        p = np.poly1d(z)
        plt.plot(
            [df['TAVG'].min(), df['TAVG'].max()],
            [p(df['TAVG'].min()), p(df['TAVG'].max())],
            "r--",
            alpha=0.7
        )
        
        # Add correlation coefficient
        corr = df['TAVG'].corr(df['ENERGY_KWH'])
        plt.annotate(
            f"Correlation: {corr:.2f}",
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
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
    """Main function to create the scatter plot"""
    # Create output directory for plots
    plots_dir = os.path.join('data', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate scatter plot
    data_file = os.path.join('data', 'processed', 'energy_temperature_merged.csv')
    save_path = os.path.join(plots_dir, 'monthly_temp_vs_energy.png')
    create_monthly_scatter(data_file, save_path)

if __name__ == "__main__":
    main()