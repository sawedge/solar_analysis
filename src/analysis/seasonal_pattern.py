import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import calendar

def create_seasonal_pattern_plot(data_file=None, save_path=None):
    """
    Create a plot showing the seasonal pattern of temperature vs energy production
    by connecting months in sequence to show the annual cycle
    
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
    required_cols = ['MONTH', 'MONTH_NAME', 'TAVG', 'ENERGY_KWH']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Data is missing required columns: {missing_cols}")
        return
    
    # Create figure and axis
    plt.figure(figsize=(12, 9))
    
    # Calculate average values by month if we have multiple years
    if 'YEAR' in df.columns and len(df['YEAR'].unique()) > 1:
        # Group by month and calculate means
        monthly_avg = df.groupby('MONTH').agg({
            'TAVG': 'mean',
            'ENERGY_KWH': 'mean',
            'MONTH_NAME': 'first'  # Keep month name
        }).reset_index()
        
        # Sort by month
        monthly_avg = monthly_avg.sort_values('MONTH')
        
        # Plot individual year data as faded scatter
        for year in sorted(df['YEAR'].unique()):
            year_data = df[df['YEAR'] == year].sort_values('MONTH')
            plt.plot(
                year_data['TAVG'], year_data['ENERGY_KWH'], 
                'o-', alpha=0.3, label=f'Year {year}',
                linewidth=1
            )
        
        # Plot the average cycle with bigger markers and thicker line
        plt.plot(
            monthly_avg['TAVG'], monthly_avg['ENERGY_KWH'], 
            'o-', color='red', linewidth=2.5, 
            markersize=10, label='Monthly Average'
        )
        
        # Add month labels to the average cycle
        for i, row in monthly_avg.iterrows():
            plt.annotate(
                row['MONTH_NAME'][:3],
                (row['TAVG'], row['ENERGY_KWH']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold'
            )
    else:
        # Just one year or no year column - sort by month and connect the dots
        sorted_df = df.sort_values('MONTH')
        
        plt.plot(
            sorted_df['TAVG'], sorted_df['ENERGY_KWH'], 
            'o-', linewidth=2, markersize=10
        )
        
        # Add month labels
        for i, row in sorted_df.iterrows():
            plt.annotate(
                row['MONTH_NAME'][:3],
                (row['TAVG'], row['ENERGY_KWH']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10
            )
    
    # Add direction arrows to show progression of months
    if len(df) > 1:
        # Get data points in order
        if 'YEAR' in df.columns and len(df['YEAR'].unique()) > 1:
            points = monthly_avg[['TAVG', 'ENERGY_KWH']].values
        else:
            points = df.sort_values('MONTH')[['TAVG', 'ENERGY_KWH']].values
        
        # Add arrows to show direction of time
        for i in range(len(points) - 1):
            # Calculate arrow position (middle of line segment)
            arrow_pos_x = (points[i][0] + points[i+1][0]) / 2
            arrow_pos_y = (points[i][1] + points[i+1][1]) / 2
            
            # Calculate arrow direction
            dx = points[i+1][0] - points[i][0]
            dy = points[i+1][1] - points[i][1]
            
            # Normalize and scale
            magnitude = np.sqrt(dx**2 + dy**2)
            if magnitude > 0:
                dx = dx / magnitude
                dy = dy / magnitude
                
                plt.arrow(
                    arrow_pos_x, arrow_pos_y, dx, dy,
                    head_width=0.5, head_length=0.8, 
                    fc='black', ec='black', alpha=0.7
                )
    
    # Add labels and title
    plt.xlabel('Average Temperature (°F)')
    plt.ylabel('Energy Production (kWh)')
    plt.title('Seasonal Pattern: Solar Production vs. Temperature')
    plt.grid(True, alpha=0.3)
    
    # Add legend if we have multiple years
    if 'YEAR' in df.columns and len(df['YEAR'].unique()) > 1:
        plt.legend(loc='best')
    
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
    """Main function to create the seasonal pattern plot"""
    # Create output directory for plots
    plots_dir = os.path.join('data', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate plot
    data_file = os.path.join('data', 'processed', 'energy_temperature_merged.csv')
    save_path = os.path.join(plots_dir, 'seasonal_pattern.png')
    create_seasonal_pattern_plot(data_file, save_path)

if __name__ == "__main__":
    main()