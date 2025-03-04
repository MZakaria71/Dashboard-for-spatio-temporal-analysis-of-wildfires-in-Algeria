#!/usr/bin/env python
"""
Wildfire Analysis Dashboard
---------------------------

This application analyses wildfire data in Algeria (2001-2020) by processing
land cover and burn date data, then displays the results via an interactive
dashboard built with Plotly Dash.

Tasks performed:
    1. Validate that the required data files exist.
    2. Process global land cover data and burn date data.
    3. Compute province-specific statistics.
    4. Build an interactive dashboard to visualise wildfire-related metrics.

Requirements:
    - dash
    - plotly
    - rasterio
    - numpy
    - pandas
    - geopandas
    - rasterstats

Author: Z.M
Date: [2025-03-03]
"""

import os
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats

# ----------------------- Constants and File Paths -------------------------
PROVINCES_PATH = 'data/Dz_adm1.shp'
LANDCOVER_TEMPLATE = 'data/LandCover_Exports/LandCover_Summer_{year}_95Percent_Confidence.tif'
BURNDATE_TEMPLATE = 'data/BurnDate_Exports/BurnDate_Summer_{year}_95Percent_Confidence.tif'
START_YEAR = 2001
END_YEAR = 2020  # inclusive
PIXEL_AREA = 250 * 250 / 1e6  # km² per pixel

# ----------------------- Debug: Check Shapefile Components -------------------------
print("DEBUG: Checking existence of shapefile components...")
for ext in ['shp', 'shx', 'dbf', 'prj']:
    path = f"data/Dz_adm1.{ext}"
    print(f"DEBUG: {path} exists? {os.path.exists(path)}")

# Load administrative boundaries
try:
    provinces = gpd.read_file(PROVINCES_PATH)
    print("DEBUG: Provinces DataFrame loaded with shape:", provinces.shape)
except Exception as e:
    print("ERROR: Failed to load provinces shapefile.", e)
    raise

# ----------------------- File Validation -------------------------
def validate_files():
    """
    Validate that all required land cover and burn date files exist.
    Raises:
        FileNotFoundError: If one or more required files are missing.
    """
    missing = []
    for year in range(START_YEAR, END_YEAR + 1):
        lc_path = LANDCOVER_TEMPLATE.format(year=year)
        bd_path = BURNDATE_TEMPLATE.format(year=year)
        if not os.path.exists(lc_path):
            missing.append(lc_path)
        if not os.path.exists(bd_path):
            missing.append(bd_path)
    if missing:
        print("DEBUG: Missing files:", missing)
        raise FileNotFoundError(f"Missing {len(missing)} file(s):\n" + "\n".join(missing))
    else:
        print("DEBUG: All required raster files exist.")

validate_files()

# ----------------------- Global Data Processing -------------------------
def process_landcover_burns() -> pd.DataFrame:
    """
    Process global land cover data for each year by reclassifying pixel values into
    aggregated land cover groups and computing burned area.
    Returns:
        pd.DataFrame: Annual data with burned area (km²) for 'Forest', 'Cropland', and 'Shrubland'.
    """
    annual_data = []
    reclass_map = {
        10: "Cropland", 20: "Cropland", 30: "Cropland", 40: "Cropland",
        50: "Forest", 60: "Forest", 70: "Forest", 80: "Forest",
        90: "Forest", 100: "Forest", 110: "Forest",
        120: "Shrubland", 130: "Shrubland", 140: "Shrubland", 150: "Shrubland",
        170: "Forest", 180: "Shrubland"
    }
    for year in range(START_YEAR, END_YEAR + 1):
        file_path = LANDCOVER_TEMPLATE.format(year=year)
        with rasterio.open(file_path) as src:
            data = src.read(1)
            valid_mask = data != 0
        class_counts = np.bincount(data[valid_mask].flatten(), minlength=181)
        record = {"Year": year, "Forest": 0.0, "Cropland": 0.0, "Shrubland": 0.0}
        for code, group in reclass_map.items():
            count = class_counts[code] if code < len(class_counts) else 0
            record[group] += count * PIXEL_AREA
        annual_data.append(record)
        print(f"DEBUG: Processed land cover for year {year}: {record}")
    df = pd.DataFrame(annual_data)
    print("DEBUG: Global land cover DataFrame shape:", df.shape)
    return df

def process_burndates() -> pd.DataFrame:
    """
    Process global burn date data by extracting the day-of-year values from burn date rasters.
    Returns:
        pd.DataFrame: DataFrame containing 'day_of_year' and 'year' columns.
    """
    all_dates = []
    for year in range(START_YEAR, END_YEAR + 1):
        file_path = BURNDATE_TEMPLATE.format(year=year)
        with rasterio.open(file_path) as src:
            data = src.read(1)
            dates = data[data > 0].flatten()
        df = pd.DataFrame({'day_of_year': dates})
        df['year'] = year
        all_dates.append(df)
        print(f"DEBUG: Processed burn dates for year {year}, count: {len(dates)}")
    combined = pd.concat(all_dates, ignore_index=True)
    print("DEBUG: Global burn date DataFrame shape:", combined.shape)
    return combined

print("DEBUG: Processing global land cover data...")
landcover_df = process_landcover_burns()
print("DEBUG: Processing global burn date data...")
burndate_df = process_burndates()

# ----------------------- Province-Specific Data Functions -------------------------
def get_province_landcover(province_geom) -> pd.DataFrame:
    """
    For a given province geometry, compute annual burned area per aggregated land cover group.
    Args:
        province_geom: Geometry of the province.
    Returns:
        pd.DataFrame: Annual data for the province with burned area for each land cover group.
    """
    reclass_map = {
        10: "Cropland", 20: "Cropland", 30: "Cropland", 40: "Cropland",
        50: "Forest", 60: "Forest", 70: "Forest", 80: "Forest",
        90: "Forest", 100: "Forest", 110: "Forest",
        120: "Shrubland", 130: "Shrubland", 140: "Shrubland", 150: "Shrubland",
        170: "Forest", 180: "Shrubland"
    }
    records = []
    for year in range(START_YEAR, END_YEAR + 1):
        file_path = LANDCOVER_TEMPLATE.format(year=year)
        with rasterio.open(file_path) as src:
            # Reproject province geometry to raster CRS
            geom_proj = gpd.GeoSeries([province_geom], crs=provinces.crs).to_crs(src.crs).iloc[0]
            try:
                out_image, _ = rasterio.mask.mask(src, [geom_proj], crop=True)
            except Exception as e:
                print(f"DEBUG: Masking failed for year {year}: {e}")
                continue
            data = out_image[0]
            valid_mask = data != 0
        record = {"Year": year, "Forest": 0.0, "Cropland": 0.0, "Shrubland": 0.0}
        if np.any(valid_mask):
            class_counts = np.bincount(data[valid_mask].flatten(), minlength=181)
        else:
            class_counts = np.zeros(181, dtype=int)
        for code, group in reclass_map.items():
            count = class_counts[code] if code < len(class_counts) else 0
            record[group] += count * PIXEL_AREA
        records.append(record)
        print(f"DEBUG: Province land cover for year {year}: {record}")
    df = pd.DataFrame(records) if records else pd.DataFrame(columns=["Year", "Forest", "Cropland", "Shrubland"])
    print("DEBUG: Province land cover DataFrame shape:", df.shape)
    return df

def get_province_burndates(province_geom) -> pd.DataFrame:
    """
    For a given province geometry, extract burn date pixel values by year.
    Args:
        province_geom: Geometry of the province.
    Returns:
        pd.DataFrame: DataFrame with burn dates (day_of_year) and corresponding year.
    """
    frames = []
    for year in range(START_YEAR, END_YEAR + 1):
        file_path = BURNDATE_TEMPLATE.format(year=year)
        with rasterio.open(file_path) as src:
            geom_proj = gpd.GeoSeries([province_geom], crs=provinces.crs).to_crs(src.crs).iloc[0]
            try:
                out_image, _ = rasterio.mask.mask(src, [geom_proj], crop=True)
            except Exception as e:
                print(f"DEBUG: Masking burn dates failed for year {year}: {e}")
                continue
            data = out_image[0]
            valid = data > 0
            if np.any(valid):
                dates = data[valid].flatten()
                df = pd.DataFrame({'day_of_year': dates})
                df['year'] = year
                frames.append(df)
                print(f"DEBUG: Province burn dates for year {year}, count: {len(dates)}")
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['day_of_year', 'year'])
    print("DEBUG: Province burn date DataFrame shape:", combined.shape)
    return combined

# ----------------------- Dashboard Layout -------------------------
app = dash.Dash(__name__)
app.title = "Algeria Wildfire Analysis (2001-2020)"

app.layout = html.Div([
    html.H1("Algeria Wildfire Analysis (2001-2020)",
            style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    # Year Dropdown: "Total" (aggregated) or a specific year.
    html.Div([
        html.Label("Select Year for Map:"),
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': 'Total', 'value': 'total'}] +
                    [{'label': str(year), 'value': year} for year in range(START_YEAR, END_YEAR + 1)],
            value='total',
            clearable=False,
            style={'width': '200px'}
        )
    ], style={'textAlign': 'center', 'padding': '10px'}),
    
    # Global map display
    dcc.Graph(id='province-map'),
    html.Div(id='map-status', style={'textAlign': 'center', 'color': '#7f8c8d', 'padding': '10px'}),
    
    html.Hr(),
    
    # Charts: Land cover analysis and daily burn pattern.
    html.Div([
        html.Div([
            dcc.Graph(id='landcover-time-series'),
            dcc.Checklist(
                id='landcover-selector',
                options=[{'label': cat, 'value': cat} for cat in ['Forest', 'Cropland', 'Shrubland']],
                value=['Forest', 'Cropland', 'Shrubland'],
                inline=True,
                style={'padding': '10px'}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        html.Div([
            dcc.Graph(id='daily-burn-pattern')
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ])
])

# ----------------------- Callbacks -------------------------
@app.callback(
    [Output('province-map', 'figure'),
     Output('map-status', 'children')],
    [Input('year-dropdown', 'value')]
)
def update_map(selected_year):
    """
    Update the map based on the selected year. Aggregates data if "total" is selected.
    Args:
        selected_year: Selected year from the dropdown or 'total'.
    Returns:
        tuple: (Map figure, status message)
    """
    print("DEBUG: update_map callback triggered with selected_year =", selected_year)
    if selected_year == 'total':
        province_stats = {i: 0 for i in range(len(provinces))}
        for year in range(START_YEAR, END_YEAR + 1):
            file_path = BURNDATE_TEMPLATE.format(year=year)
            if not os.path.exists(file_path):
                print(f"DEBUG: File not found for year {year}: {file_path}")
                continue
            with rasterio.open(file_path) as src:
                raster_crs = src.crs
                affine = src.transform
                provinces_reproj = provinces.to_crs(raster_crs)
                burned_mask = src.read(1) > 0
            stats = zonal_stats(
                provinces_reproj,
                burned_mask,
                affine=affine,
                stats=['sum'],
                nodata=0,
                all_touched=True,
                geojson_out=True
            )
            for i, stat in enumerate(stats):
                burned_val = stat['properties'].get('sum', 0)
                province_stats[i] += burned_val if burned_val is not None else 0
            print(f"DEBUG: Year {year} stats computed.")
        gdf_stats = provinces.copy()
        gdf_stats['burned_area'] = [province_stats[i] for i in range(len(provinces))]
        gdf_stats['burned_area'] = gdf_stats['burned_area'] * (250**2) / 1e6
        map_title = "Total Burned Area (2001-2020)"
        print("DEBUG: Global aggregated stats computed.")
    else:
        file_path = BURNDATE_TEMPLATE.format(year=selected_year)
        if not os.path.exists(file_path):
            return go.Figure(), "Data not available for selected year."
        with rasterio.open(file_path) as src:
            raster_crs = src.crs
            affine = src.transform
            provinces_reproj = provinces.to_crs(raster_crs)
            burned_mask = src.read(1) > 0
        stats = zonal_stats(
            provinces_reproj,
            burned_mask,
            affine=affine,
            stats=['sum'],
            nodata=0,
            all_touched=True,
            geojson_out=True
        )
        gdf_stats = gpd.GeoDataFrame.from_features(stats)
        gdf_stats.crs = raster_crs
        gdf_stats = gdf_stats.to_crs(provinces.crs)
        gdf_stats['burned_area'] = gdf_stats['sum'].fillna(0) * (250**2) / 1e6
        map_title = f"Burned Area in {selected_year}"
        print(f"DEBUG: Stats computed for year {selected_year}.")

    print("DEBUG: gdf_stats shape:", gdf_stats.shape)
    
    # Convert the entire GeoDataFrame to a GeoJSON object.
    geojson_data = gdf_stats.__geo_interface__
    
    # Create a choropleth mapbox with a white background (no basemap tiles)
    fig = px.choropleth_mapbox(
        gdf_stats,
        geojson=geojson_data,
        locations=gdf_stats.index,
        color='burned_area',
        mapbox_style="white-bg",
        zoom=4.5,
        center={"lat": 36, "lon": 3},
        opacity=0.8,
        labels={'burned_area': 'Burned Area (km²)'},
        color_continuous_scale='YlOrRd'
    )
    # Remove any mapbox layers to ensure no basemap is shown
    fig.update_layout(mapbox=dict(layers=[]))
    # Update the trace so that the fill is fully transparent and only boundaries are visible
    fig.update_traces(marker_opacity=0, marker_line_width=2, marker_line_color="black")
    
    return fig, f"Map displaying {map_title}."


@app.callback(
    [Output('landcover-time-series', 'figure'),
     Output('daily-burn-pattern', 'figure')],
    [Input('province-map', 'clickData'),
     Input('landcover-selector', 'value')]
)
def update_charts(clickData, selected_categories):
    """
    Update the land cover and daily burn pattern charts based on the selected province.
    If no province is selected, global data is displayed.
    Args:
        clickData: Data from the map click event.
        selected_categories: List of land cover groups to display.
    Returns:
        tuple: (Land cover time series figure, Daily burn pattern figure)
    """
    print("DEBUG: update_charts callback triggered.")
    if clickData is None:
        df_lc = landcover_df.copy()
        df_bd = burndate_df.copy()
        title_suffix = " (All Provinces)"
        print("DEBUG: Using global data for charts.")
    else:
        province_index = clickData['points'][0]['location']
        selected_province = provinces.iloc[int(province_index)]
        province_name = selected_province.get('ADM1_EN', f"Province {province_index}")
        title_suffix = f" ({province_name})"
        df_lc = get_province_landcover(selected_province.geometry)
        df_bd = get_province_burndates(selected_province.geometry)
        print(f"DEBUG: Using data for province: {province_name}")
    
    # Land Cover Bar Plot
    if not df_lc.empty:
        df_melted = df_lc.melt(id_vars='Year', value_vars=['Forest', 'Cropland', 'Shrubland'],
                               var_name='Category', value_name='Burned Area')
        df_melted = df_melted[df_melted['Category'].isin(selected_categories)]
        fig_lc = px.bar(df_melted, x='Year', y='Burned Area', color='Category', barmode='group',
                        labels={'Burned Area': 'Burned Area (km²)', 'Category': 'Land Cover Group'},
                        template='plotly_white',
                        title="Land Cover Burned Area" + title_suffix)
        print("DEBUG: Land cover chart generated, data shape:", df_melted.shape)
    else:
        fig_lc = go.Figure()
        fig_lc.update_layout(title="No land cover data available" + title_suffix)
        print("DEBUG: Land cover DataFrame is empty.")
    
    # Daily Burn Pattern Bar Plot
    if not df_bd.empty:
        daily_counts = df_bd.groupby('day_of_year').size().reset_index(name='counts')
        fig_bd = go.Figure()
        fig_bd.add_trace(go.Bar(
            x=daily_counts['day_of_year'],
            y=daily_counts['counts'],
            marker_color='#e74c3c',
            name='Burn Frequency'
        ))
        fig_bd.update_layout(
            title="Daily Burn Frequency" + title_suffix,
            xaxis_title='Day of Year',
            yaxis_title='Number of Burned Pixels (250m)',
            hovermode='x unified',
            template='plotly_white',
            showlegend=False
        )
        print("DEBUG: Daily burn pattern chart generated, data shape:", daily_counts.shape)
    else:
        fig_bd = go.Figure()
        fig_bd.update_layout(title="No burn date data available" + title_suffix)
        print("DEBUG: Burn date DataFrame is empty.")
    
    return fig_lc, fig_bd

# ----------------------- Main Entry Point -------------------------
def main():
    """
    Main entry point for the Dash application.
    """
    port = int(os.environ.get("PORT", 8051))
    print("DEBUG: Starting app on port", port)
    app.run_server(debug=False, host='0.0.0.0', port=port)

if __name__ == '__main__':
    main()
