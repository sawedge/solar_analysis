# Solar Analysis Project

This project analyzes the relationship between temperature and solar panel energy production.

## Installation

```bash
# Clone the repository
git clone [repository-url]
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

## Project Structure

- `src/data_processing/`: Data loading and processing modules
- `src/analysis/`: Data analysis modules
- `src/visualization/`: Plotting and visualization modules
- `data/`: Raw and processed data files
- `reports/`: Generated reports and analysis outputs
- `tests/`: Unit tests

## License

MIT License
