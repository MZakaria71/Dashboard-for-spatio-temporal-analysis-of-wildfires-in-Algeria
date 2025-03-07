# Historical Wildfire Analysis Dashboard

The **Wildfire Analysis Dashboard** is an open source, interactive web application that analyses wildfire data in Algeria from 2001 to 2020. It processes raster datasets for land cover and burn dates, computes province-level statistics, and visualises the results through an interactive Plotly Dash dashboard.

---

## Features

- **Interactive Map:**  
  Displays province boundaries of Algeria along with wildfire burned area data.  
  Hover over a province to view its name and statistical details.

- **Time-Series Charts:**  
  Visualise trends in burned area across different land cover categories (Forest, Cropland, Shrubland) over the years.

- **Daily Burn Patterns:**  
  Analyse the frequency of burned pixels per day throughout the year.

- **Automated Data Validation & Processing:**  
  Checks for the existence of required datasets and processes them to compute aggregated statistics.

---

## Requirements

- Python 3.7 or higher
- [Dash](https://dash.plotly.com/)
- [Plotly](https://plotly.com/python/)
- [Rasterio](https://rasterio.readthedocs.io/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [GeoPandas](https://geopandas.org/)
- [Rasterstats](https://pythonhosted.org/rasterstats/)

You will also need the following datasets, which should be placed in the corresponding directories:

- **Administrative Boundaries:**  
  - `data/Dz_adm1.shp` along with its companion files (`.shx`, `.dbf`, `.prj`).

- **Land Cover Data:**  
  - Files following the template:  
    `data/LandCover_Exports/LandCover_Summer_{year}_95Percent_Confidence.tif`  
    for each year from 2001 to 2020.

- **Burn Date Data:**  
  - Files following the template:  
    `data/BurnDate_Exports/BurnDate_Summer_{year}_95Percent_Confidence.tif`  
    for each year from 2001 to 2020.

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository

Author
Z.M
Date: 2025-03-03
