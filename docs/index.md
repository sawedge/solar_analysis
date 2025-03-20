---
layout: default
title: Solar Energy & Temperature - An Unexpected Pattern
---

# Solar Energy & Temperature: An Unexpected Pattern

## Introduction: A Simple Dataset

It began with a simple email from a friend who had installed solar panels on his Wisconsin home. "I've been collecting this data for years," he wrote, "maybe you can find something interesting in it?"

The attached CSV file was underwhelming at first glance - just two columns:

```
Date/Time, Energy Produced (Wh)
2013-07-28, 12545
2013-07-29, 13267
...
```

![Raw data sample](/assets/images/raw_data_sample.png)

By itself, this was too basic to draw meaningful conclusions. The seasonal patterns were obvious - more sun in summer, less in winter - but I wondered if there might be deeper insights hiding in the relationship between weather and energy production.

## Enhancing the Dataset: NOAA Weather Integration

To add context, I pulled historical weather data from NOAA for the nearest weather station to my friend's home - the Oshkosh Wittman Regional Airport. The weather dataset provided a wealth of variables:

- Daily temperature averages, minimums, and maximums
- Precipitation measurements
- Wind speed and direction
- Cloud cover estimates

```python
def fetch_noaa_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch weather data from NOAA's API for the given date range.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with weather data
    """
    # Code to fetch weather data from NOAA
```

By merging these datasets based on date, I created a rich foundation for analysis.

## First Visualizations: Time Series Patterns

Initial time series plots confirmed the expected seasonal variation in both solar production and temperature:

![Time series of solar production](/assets/images/time_series_plot.png)

The correlation seemed straightforward - solar production peaks in summer when temperatures are highest. But was the relationship really that simple?

## Heat Mapping: Visualizing the Annual Cycle

To better understand patterns across years, I created a heatmap showing daily production values organized by month and year:

![Heatmap of solar production](/assets/images/heatmap_plot.png)

The heatmap revealed consistent patterns year-over-year, with interesting anomalies during extreme weather events. But I wanted to see this data in a more dynamic way.

## Animation: Solar Production in Polar Coordinates

Converting the time series to polar coordinates (with each year forming a complete circle) created a compelling animation of how production varied through the seasons:

![Polar plot animation](/assets/images/polar_animation.gif)

```python
def create_polar_animation(df, output_path):
    """
    Create a polar plot animation of solar production over time.
    
    Args:
        df: DataFrame with solar production data
        output_path: Path to save the animation
    """
    # Animation code
```

The animation beautifully illustrated the yearly cycles, but I was still curious about the direct relationship between temperature and production.

## Temperature vs. Production: The Initial Scatter Plot

Plotting average temperature against energy production showed a positive correlation - hardly surprising:

![Basic scatter plot](/assets/images/basic_scatter.png)

The correlation coefficient of approximately 0.65 confirmed a moderately strong relationship. However, this overall correlation was masking something far more interesting.

## The Monthly Pattern Emerges

When I color-coded the data points by month and drew boundaries around each month's cluster, an unexpected pattern emerged:

![Scatter plot with monthly clusters](/assets/images/monthly_clusters.png)

Instead of points randomly distributed along the correlation line, they formed distinct clusters by month - and these clusters didn't follow the overall trend!

## The Revelation: A Counterclockwise Loop

The most fascinating discovery was that when viewed by month, the temperature-production relationship forms a counterclockwise loop through the year:

![Final visualization showing the loop pattern](/assets/images/temp_vs_solar_scatter.png)

For the same temperature, solar panels produce significantly more energy in spring than in fall. For example, at 60°F:
- In March/April: ~500 kWh/month
- In September/October: ~350 kWh/month

That's nearly 30% difference at the same temperature!

## The Solar Analemma Connection

This pattern reminds me of the solar analemma - the figure-eight pattern traced by the sun when observed at the same time each day throughout a year. While not exactly the same phenomenon, both demonstrate how the Earth's tilt and orbital characteristics create cyclical patterns that aren't intuitive at first glance.

![Solar analemma](/assets/images/analemma.jpg)

## Conclusion: Why This Matters

This counterclockwise pattern has practical implications for solar energy planning:
- Spring months deliver unexpectedly high production relative to temperature
- Fall months underperform compared to their temperature counterparts
- Solar production forecasting models should account for season, not just temperature

What began as a simple two-column dataset revealed a fascinating pattern that connects astronomy, meteorology, and renewable energy. The analysis continues, but the looping pattern remains one of the most elegant findings in this ongoing exploration.

---

*This analysis was conducted using Python with pandas, matplotlib, and seaborn. Full code available in the [GitHub repository](https://github.com/sawedge/solar_analysis).*