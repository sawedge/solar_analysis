# Solar Analysis Project

This project analyzes the relationship between temperature and solar panel energy production for a residential solar installation in Wisconsin. The analysis explores how ambient temperature affects solar energy production efficiency throughout the year.

## Data Sources

This project uses two primary datasets:

1. **Solar Production Data** (`223752_site_energy_production_report.csv`):
   - Contains daily energy production data from a residential solar installation
   - Format: Date/Time, Energy Produced (Wh)
   - Origin: Exported from the solar panel monitoring system

2. **Weather Data** (`historic_data.csv`):
   - Contains historical weather data from NOAA for the Oshkosh Wittman Regional Airport (Station ID: USW00094855)
   - Format: DATE, STATION, TAVG, TMAX, TMIN, PRCP, etc.
   - Origin: National Oceanic and Atmospheric Administration (NOAA) Climate Data records
   - Note: This file was manually obtained and is required for analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/sawedge/solar_analysis.git
cd solar_analysis

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run main analysis
python -m src.main

# Generate temperature vs solar production scatter plot
python -m src.analysis.temp_vs_solar_scatter
```

## Key Visualizations

1. **Temperature vs. Solar Production Scatter Plot**:
   - Visualizes the relationship between average temperature and solar energy production
   - Each point represents one month of data
   - Groups points by month to show seasonal patterns

2. **Energy/Temperature Ratio Analysis**:
   - Analyzes how efficiently solar panels convert sunlight to electricity at different temperatures
   - Identifies optimal temperature ranges for solar production

## Project Structure

- `src/data_processing/`: Data loading and processing modules
- `src/analysis/`: Data analysis modules
- `src/visualization/`: Plotting and visualization modules
- `data/`: Raw and processed data files
- `reports/`: Generated reports and analysis outputs
- `tests/`: Unit tests
- `docs/`: Documentation including data dictionaries

## License

MIT License
