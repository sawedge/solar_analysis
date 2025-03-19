import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from energy_temp_ratio_analysis import (
    load_energy_production_data,
    calculate_monthly_energy_production,
    load_temperature_data,
    merge_energy_and_temperature
)

def create_scatter_plot(merged_data, save_path=None):
    """
    Create a scatter plot with average temperature on X axis and solar production on Y axis.
    Each point represents one month.
    
    Args:
        merged_data: DataFrame with merged energy and temperature data
        save_path: Path to save the plot image
    """
    if merged_data is None or merged_data.empty:
        print("No data available for plotting")
        return
    
    # Remove July and August 2013 as outliers
    filtered_data = merged_data.copy()
    outlier_condition = ((filtered_data['YEAR'] == 2013) & 
                         (filtered_data['MONTH'].isin([7, 8])))
    
    # Print information about removed data
    removed_data = filtered_data[outlier_condition].copy()
    if not removed_data.empty:
        print(f"Removed {len(removed_data)} data points as outliers:")
        for _, row in removed_data.iterrows():
            print(f"  - {row['MONTH_NAME']} {row['YEAR']}: Temp={row['TAVG']:.1f}°F, Energy={row['ENERGY_KWH']:.1f} kWh")
    
    # Filter out the outliers
    filtered_data = filtered_data[~outlier_condition]
    
    # Create figure and axis with extra space for the legend
    plt.figure(figsize=(14, 8))
    
    # Adjust the subplot to make room for the legend
    plt.subplots_adjust(right=0.85)
    
    # Create point labels (month and year)
    point_labels = [f"{row['MONTH_NAME']} {row['YEAR']}" for _, row in filtered_data.iterrows()]
    
    # Create scatter plot with discrete colors for each month
    from scipy.spatial import ConvexHull
    import calendar
    import matplotlib.patches as mpatches
    
    # Define month names
    month_names = {i: calendar.month_name[i] for i in range(1, 13)}
    
    # Using the provided discrete color palette
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable
    
    # Original palette provided, with one additional color added for December
    # Added '#023858' as the 12th color which complements the blue end of the spectrum
    palette = [
        "#377eb8",  # January
        "#546d9e",  # February
        "#715d84",  # March
        "#8e4c6a",  # April
        "#aa3b50",  # May
        "#c72b36",  # June
        "#e41a1c",  # July
        "#c12e3b",  # August
        "#9f425a",  # September
        "#7c567a",  # October
        "#5a6a99",  # November
        "#377eb8",  # December
    ]

    
    # Map month numbers to colors
    month_colors = {month: palette[month-1] for month in range(1, 13)}
    
    # Create scatter plots by month for legend
    legend_handles = []
    
    # Plot each month's data with its own color
    for month, group in filtered_data.groupby('MONTH'):
        scatter = plt.scatter(
            group['TAVG'], 
            group['ENERGY_KWH'],
            s=100,  # Marker size
            alpha=0.7,
            color=month_colors[month],
            label=month_names[month]
        )
        # Only add to legend if we have data points
        if not group.empty:
            legend_handles.append(
                mpatches.Patch(color=month_colors[month], label=month_names[month])
            )
    
    # Add trend line
    if len(filtered_data) > 1:
        z = np.polyfit(filtered_data['TAVG'], filtered_data['ENERGY_KWH'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(filtered_data['TAVG'].min(), filtered_data['TAVG'].max(), 100)
        plt.plot(x_range, p(x_range), "r--", alpha=0.7)
        
        # Add correlation coefficient
        corr = filtered_data['TAVG'].corr(filtered_data['ENERGY_KWH'])
        plt.annotate(
            f"Correlation: {corr:.2f}", 
            xy=(0.05, 0.95), 
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    # Define offsets for month labels - adjust these values to move labels
    # Format: {month_number: (x_offset, y_offset)}
    # Positive x moves right, negative x moves left
    # Positive y moves up, negative y moves down
    label_offsets = {
        1: (0, -50),    # January
        2: (0, 30),    # February
        3: (-5, 0),    # March
        4: (0, 30),    # April
        5: (-5, 0),    # May
        6: (-3, 20),    # June
        7: (0, 30),    # July
        8: (5, 0),    # August
        9: (5, -30),    # September
        10: (5, -20),   # October
        11: (5, -20),   # November
        12: (3, -17)    # December
    }
    
    # Draw dotted lines around each month and add month labels
    month_groups = filtered_data.groupby('MONTH')
    
    for month, group in month_groups:
        # Get the offsets for this month (default to (0,0) if not specified)
        x_offset, y_offset = label_offsets.get(month, (0, 0))
        
        if len(group) >= 3:  # Need at least 3 points for a convex hull
            # Get x and y coordinates
            points = np.vstack([group['TAVG'], group['ENERGY_KWH']]).T
            
            # Calculate convex hull
            hull = ConvexHull(points)
            
            # Get hull vertices and plot
            hull_vertices = hull.vertices.tolist() + [hull.vertices[0]]  # Close the loop
            plt.plot(
                points[hull_vertices, 0], 
                points[hull_vertices, 1], 
                'k:', 
                linewidth=1.5,
                color=month_colors[month]
            )
            
            # Add month label at the centroid of the hull with offsets
            centroid_x = np.mean(points[hull.vertices, 0])
            centroid_y = np.mean(points[hull.vertices, 1])
            
            plt.text(
                centroid_x + x_offset, centroid_y + y_offset, 
                month_names[month],
                fontsize=12, 
                fontweight='bold',
                ha='center', 
                va='center',
                bbox=dict(
                    facecolor='white', 
                    alpha=0.7, 
                    edgecolor=month_colors[month], 
                    boxstyle='round,pad=0.3'
                )
            )
        elif len(group) == 2:  # Draw a line for just 2 points
            plt.plot(
                group['TAVG'], 
                group['ENERGY_KWH'], 
                'k:', 
                linewidth=1.5,
                color=month_colors[month]
            )
            
            # Add label in the middle of the line with offsets
            centroid_x = np.mean(group['TAVG'])
            centroid_y = np.mean(group['ENERGY_KWH'])
            
            plt.text(
                centroid_x + x_offset, centroid_y + y_offset,
                month_names[month],
                fontsize=12,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(
                    facecolor='white',
                    alpha=0.7,
                    edgecolor=month_colors[month],
                    boxstyle='round,pad=0.3'
                )
            )
        elif len(group) == 1:  # Just label the single point with offsets
            # Default offset of 100 for single points (can be adjusted with y_offset)
            default_y_offset = 100
            
            plt.text(
                group['TAVG'].iloc[0] + x_offset, 
                group['ENERGY_KWH'].iloc[0] + default_y_offset + y_offset,
                month_names[month],
                fontsize=12,
                fontweight='bold',
                ha='center',
                va='bottom',
                bbox=dict(
                    facecolor='white',
                    alpha=0.7, 
                    edgecolor=month_colors[month],
                    boxstyle='round,pad=0.3'
                )
            )
    
    # Add discrete legend to the right side of the plot outside the axes
    # Sort the legend handles by month number
    legend_handles_sorted = sorted(legend_handles, 
                                  key=lambda x: list(month_names.values()).index(x.get_label()) + 1)
    
    plt.legend(
        handles=legend_handles_sorted, 
        title="Month", 
        loc="center left", 
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        borderaxespad=0.
    )
    
    # Add labels and title
    plt.xlabel('Average Temperature (°F)')
    plt.ylabel('Energy Production (kWh)')
    plt.title('Monthly Solar Production vs. Average Temperature')
    plt.grid(True, alpha=0.3)
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    # File paths
    energy_file = os.path.join('data', '223752_site_energy_production_report.csv')
    
    # Create output directory for plots
    plots_dir = os.path.join('data', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load and process data
    print("Loading and processing data...")
    energy_data = load_energy_production_data(energy_file)
    
    if energy_data is None:
        print("Cannot proceed without energy production data")
        return
    
    # Calculate monthly energy production
    monthly_energy = calculate_monthly_energy_production(energy_data)
    
    # Load temperature data from the nearest weather station
    station_id = "USW00094855"  # OSHKOSH WITTMAN REGIONAL AIRPORT
    temp_data = load_temperature_data(station_id)
    
    # Merge datasets
    merged_data = merge_energy_and_temperature(monthly_energy, temp_data)
    
    if merged_data is not None:
        # Print basic statistics
        print("\n=== Temperature vs. Solar Production Analysis ===")
        print(f"Analysis period: {merged_data['YEAR'].min()}-{merged_data['MONTH'].min()} to {merged_data['YEAR'].max()}-{merged_data['MONTH'].max()}")
        print(f"Number of data points: {len(merged_data)}")
        print(f"Temperature range: {merged_data['TAVG'].min():.1f}°F to {merged_data['TAVG'].max():.1f}°F")
        print(f"Production range: {merged_data['ENERGY_KWH'].min():.1f} kWh to {merged_data['ENERGY_KWH'].max():.1f} kWh")
        
        # Create and save scatter plot
        plot_path = os.path.join(plots_dir, 'temp_vs_solar_scatter.png')
        create_scatter_plot(merged_data, plot_path)
    else:
        print("Failed to merge energy and temperature data")

if __name__ == "__main__":
    main()