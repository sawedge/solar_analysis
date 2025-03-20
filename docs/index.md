---
layout: default
title: Solar Panel Data - An Unexpected Pattern
---

# Solar Panel Data - An Unexpected Pattern

## Introduction: A Simple Dataset

10 years of solar panel data. Two columns. 

```
Date/Time, Energy Produced (Wh)
2013-07-28, 12545
2013-07-29, 13267
...
```
I mean, alright... we can make a line chart. 

By itself, this was too basic to draw meaningful conclusions. The seasonal patterns were obvious - more sun in summer, less in winter - but I wondered where we could take it.

## Enhancing the Dataset: NOAA Weather 

To add context, I pulled historical weather data from NOAA for a nearby weather station. The weather dataset provided a wealth of variables:

- Daily temperature averages, minimums, and maximums
- Precipitation measurements
- Wind speed and direction
- Cloud cover estimates

We ignored everything except temperature. 

## Time Series 

Initial time series plots confirmed the expected seasonal variation in both solar production and temperature:

![Time series of solar production](/data/plots/monthly_energy_production.png)

We can also view both energy production and temperature on the same timeline:

![Energy and temperature timeline](/data/plots/energy_temperature_timeline.png)

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

![Basic scatter plot](/data/plots/monthly_temp_vs_energy.png)

The correlation coefficient of approximately 0.65 confirmed a moderately strong relationship. However, this overall correlation was masking something far more interesting.

## The Monthly Pattern Emerges

When I color-coded the data points by month and drew boundaries around each month's cluster, an unexpected pattern emerged:

![Scatter plot with monthly clusters](/assets/images/monthly_clusters.png)

Instead of points randomly distributed along the correlation line, they formed distinct clusters by month - and these clusters didn't follow the overall trend!

## The Revelation: A Counterclockwise Loop

The most fascinating discovery was that when viewed by month, the temperature-production relationship forms a counterclockwise loop through the year:

![Final visualization showing the loop pattern](/data/plots/seasonal_pattern.png)

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

*This analysis was conducted using Python with pandas, matplotlib, and numpy. All visualizations are generated directly from the processed data in our pipeline. Full code available in the [GitHub repository](https://github.com/sawedge/solar_analysis).*