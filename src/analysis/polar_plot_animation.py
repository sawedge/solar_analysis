#!/usr/bin/env python
"""
Script to create an animated GIF of a polar plot showing monthly solar production.
Each frame represents a new month where:
- The current month is black (fully opaque)
- Previous months fade to light grey (20% opacity) over 15 months but remain visible
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import calendar
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib import cm
import imageio
from tqdm import tqdm
import tempfile

def load_site_data(file_path):
    """
    Load the site energy production data from CSV file.
    
    Args:
        file_path: Path to the CSV file containing energy production data
        
    Returns:
        DataFrame with formatted data having DATE and AC_POWER columns
    """
    print(f"Loading data from {file_path}...")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Rename columns to match expected format
    df.rename(columns={
        'Date/Time': 'DATE',
        'Energy Produced (Wh)': 'AC_POWER'
    }, inplace=True)
    
    # Convert DATE to datetime
    df['DATE'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y')
    
    # Remove commas from AC_POWER and convert to numeric
    df['AC_POWER'] = df['AC_POWER'].astype(str).str.replace(',', '').astype(float)
    
    print(f"Loaded data with {len(df)} records spanning from {df['DATE'].min().date()} to {df['DATE'].max().date()}")
    
    return df

def prepare_monthly_data(df):
    """
    Prepare monthly average data for plotting.
    
    Args:
        df: DataFrame with DATE and AC_POWER columns
        
    Returns:
        DataFrame with monthly averages
    """
    # Create a copy of the dataframe
    df_copy = df.copy()
    
    # Ensure DATE column is datetime type
    if 'DATE' in df_copy.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_copy['DATE']):
            df_copy['DATE'] = pd.to_datetime(df_copy['DATE'])
    
    # Extract year and month components
    df_copy['month'] = df_copy['DATE'].dt.month
    df_copy['year'] = df_copy['DATE'].dt.year
    
    # Create a year-month column for easier chronological ordering
    df_copy['year_month'] = df_copy['year'].astype(str) + '-' + df_copy['month'].astype(str).str.zfill(2)
    
    # Group by year and month to get average production
    monthly_avg = df_copy.groupby(['year', 'month', 'year_month'])['AC_POWER'].mean().reset_index()
    
    # Sort chronologically
    monthly_avg = monthly_avg.sort_values(['year', 'month'])
    
    return monthly_avg

def create_polar_frame(monthly_data, current_idx, fade_months=15, fig=None, ax=None):
    """
    Create a single frame of the polar plot animation.
    
    Args:
        monthly_data: DataFrame with prepared monthly average data
        current_idx: Index of the current month to display
        fade_months: Number of months to include in the fade effect
        fig: Matplotlib figure object (created if None)
        ax: Matplotlib axes object (created if None)
        
    Returns:
        fig, ax: The figure and axes objects
    """
    if fig is None or ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, polar=True)
    else:
        ax.clear()
    
    # Define the angles for each month (in radians)
    # We start with January at the top (90 degrees or π/2)
    # and move clockwise (negative angle direction in polar coordinates)
    angles = np.array([
        0,               # January (top)
        np.pi/6,         # February
        2*np.pi/6,       # March
        3*np.pi/6,       # April
        4*np.pi/6,       # May
        5*np.pi/6,       # June
        6*np.pi/6,       # July
        7*np.pi/6,       # August
        8*np.pi/6,       # September
        9*np.pi/6,       # October
        10*np.pi/6,      # November
        11*np.pi/6       # December
    ])
    
    # Set the plot direction to clockwise
    ax.set_theta_direction(-1)
    
    # Set January (0 angle) at the top
    ax.set_theta_zero_location("N")
    
    # Set the labels for each month
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(angles)
    ax.set_xticklabels(month_names)
    
    # Calculate the maximum value across all data for consistent y-axis
    max_val = monthly_data['AC_POWER'].max() * 1.1  # Add 10% margin
    ax.set_ylim(0, max_val)
    
    # Set radial ticks
    ax.set_rticks(np.linspace(0, max_val, 5))
    
    # Draw historical points and connect them with lines in chronological order
    if current_idx > 0:
        # For each pair of consecutive months in the historical data
        for i in range(current_idx - 1):
            current_month = monthly_data.iloc[i]
            next_month = monthly_data.iloc[i + 1]
            
            # Get angles and values for both months
            current_angle = angles[current_month['month'] - 1]
            next_angle = angles[next_month['month'] - 1]
            
            current_value = current_month['AC_POWER']
            next_value = next_month['AC_POWER']
            
            # Draw a line between these consecutive months
            ax.plot([current_angle, next_angle], 
                    [current_value, next_value], 
                    '-', linewidth=1, alpha=0.2,
                    color='grey', zorder=1)
            
        # Plot all historical points
        for idx in range(current_idx):
            month_data = monthly_data.iloc[idx]
            month_num = month_data['month']
            angle = angles[month_num - 1]
            value = month_data['AC_POWER']
            
            # Plot the point in light grey
            ax.plot([angle], [value], 'o', 
                    markersize=5, alpha=0.2,
                    color='grey', zorder=1)
    
    # Now draw the fade effect for the most recent months
    # Loop through each visible month (current and previous fade_months)
    for i in range(min(fade_months, current_idx) + 1):
        month_idx = current_idx - i
        
        if month_idx < 0:
            continue
            
        this_month = monthly_data.iloc[month_idx]
        
        # If this is the current month, use black
        if i == 0:
            color = (0, 0, 0, 1.0)  # Pure black for current month
            marker_size = 10
            line_width = 2.5
            zorder = 10  # Put current month on top
        else:
            # Calculate opacity and color based on how many months ago
            # Transition from black to grey as we go back in time
            # Ensure a minimum visibility with higher opacity for more recent months
            opacity = max(0.3, 1.0 - (i / fade_months))
            grey_value = min(0.8, 0.3 + (i / fade_months) * 0.5)
            color = (grey_value, grey_value, grey_value, opacity)
            marker_size = max(6, 10 - i * 0.2)
            line_width = max(1.5, 2.5 - i * 0.1)
            zorder = 10 - i  # Ensure newer months are drawn on top
            
        # Get the month number (1-12) and the corresponding angle
        month_num = this_month['month']
        angle = angles[month_num - 1]  # Adjust for 0-indexing
        
        # Plot this month's data point
        ax.plot([angle], [this_month['AC_POWER']], 'o', 
                markersize=marker_size, color=color, zorder=zorder)
        
        # If we have enough points, draw lines between them
        if i > 0:
            # Get the previous month's data that we've already plotted
            prev_month_idx = current_idx - (i - 1)
            if prev_month_idx >= 0:
                prev_month = monthly_data.iloc[prev_month_idx]
                prev_month_num = prev_month['month']
                prev_angle = angles[prev_month_num - 1]
                
                # Get color for the previous month
                if i == 1:  # Connection to current month
                    prev_color = (0, 0, 0, 1.0)
                else:
                    prev_opacity = max(0.3, 1.0 - ((i-1) / fade_months))
                    prev_grey = min(0.8, 0.3 + ((i-1) / fade_months) * 0.5)
                    prev_color = (prev_grey, prev_grey, prev_grey, prev_opacity)
                
                # Calculate a color between this month and previous month
                avg_opacity = (color[3] + prev_color[3]) / 2
                avg_grey = (color[0] + prev_color[0]) / 2
                line_color = (avg_grey, avg_grey, avg_grey, avg_opacity)
                
                # Draw a line from the previous month to this month
                ax.plot([prev_angle, angle], 
                       [prev_month['AC_POWER'], this_month['AC_POWER']], 
                       '-', linewidth=line_width, color=line_color, zorder=zorder-0.5)
    
    # Set the title showing the current month and year
    current_month = monthly_data.iloc[current_idx]
    month_name = calendar.month_name[current_month['month']]
    ax.set_title(f'Solar Production: {month_name} {current_month["year"]}', size=15)
    
    # Add a text to explain the visualization
    plt.figtext(0.5, 0.01, 
                'Current month in black, previous months fade to light grey over 15 months',
                ha='center', fontsize=10, bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})
    
    return fig, ax

def create_animation(df, output_path, fade_months=15, fps=4):
    """
    Create an animated GIF of the polar plot.
    
    Args:
        df: DataFrame with DATE and AC_POWER columns
        output_path: Path to save the output GIF file
        fade_months: Number of months to include in the fade effect
        fps: Frames per second in the output GIF
    """
    # Prepare the monthly data
    monthly_data = prepare_monthly_data(df)
    
    # Create a temporary directory to store individual frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create each frame of the animation
        filenames = []
        
        # Setup the figure once and reuse it for all frames
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Create a progress bar
        for i in tqdm(range(len(monthly_data)), desc="Creating animation frames"):
            # Create the frame
            fig, ax = create_polar_frame(monthly_data, i, fade_months, fig, ax)
            
            # Save the frame
            frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
            fig.savefig(frame_path, dpi=100, bbox_inches='tight')
            filenames.append(frame_path)
        
        plt.close(fig)
        
        # Create the GIF
        print("Creating GIF...")
        with imageio.get_writer(output_path, mode='I', duration=1000/fps) as writer:
            for filename in tqdm(filenames, desc="Adding frames to GIF"):
                image = imageio.v2.imread(filename)
                writer.append_data(image)
    
    print(f"Animation saved to: {output_path}")

def main():
    """Load actual site data and create the animated polar plot."""
    # Path to the site energy production report
    data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '223752_site_energy_production_report.csv')
    
    # Load the data
    site_data = load_site_data(data_file)
    
    # Define output path
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports', 'visualizations')
    os.makedirs(reports_dir, exist_ok=True)
    output_path = os.path.join(reports_dir, f'site_polar_animation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.gif')
    
    # Create the animation
    print("Creating polar plot animation...")
    create_animation(site_data, output_path, fade_months=15, fps=4)
    
    print("Done!")

if __name__ == "__main__":
    main() 