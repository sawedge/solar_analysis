import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def create_energy_line_graph(data_file=None, save_path=None):
    """
    Create a simple line graph of energy production over time
    
    Args:
        data_file: Path to the monthly energy data file
        save_path: Path to save the output plot
    """
    # Default file path if none provided
    if data_file is None:
        data_file = os.path.join('data', 'processed', 'monthly_energy.csv')
    
    # Load the data
    print(f"Loading data from {data_file}")
    try:
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} data points")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Check required columns
    required_cols = ['YEAR', 'MONTH', 'ENERGY_KWH']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Data is missing required columns: {missing_cols}")
        return
    
    # Sort data by date
    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
    df = df.sort_values('DATE')
    
    # Create figure and axis
    plt.figure(figsize=(12, 6))
    
    # Create a simple line graph
    plt.plot(
        df['DATE'],
        df['ENERGY_KWH'],
        'b-',
        linewidth=2,
        marker='o',
        markersize=8
    )
    
    # Format x-axis to show dates properly
    plt.gcf().autofmt_xdate()
    
    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Energy Production (kWh)')
    plt.title('Monthly Solar Energy Production')
    plt.grid(True, alpha=0.3)
    
    # Add average line
    avg = df['ENERGY_KWH'].mean()
    plt.axhline(y=avg, color='r', linestyle='--', alpha=0.7)
    plt.text(
        df['DATE'].iloc[0], 
        avg * 1.05, 
        f'Average: {avg:.1f} kWh',
        color='r'
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
    """Main function to create the line graph"""
    # Create output directory for plots
    plots_dir = os.path.join('data', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate line graph
    data_file = os.path.join('data', 'processed', 'monthly_energy.csv')
    save_path = os.path.join(plots_dir, 'monthly_energy_production.png')
    create_energy_line_graph(data_file, save_path)

if __name__ == "__main__":
    main()