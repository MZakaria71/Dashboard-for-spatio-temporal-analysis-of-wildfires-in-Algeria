{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "335bf220-cc37-4318-b0ce-0b1a53c3744a",
   "metadata": {},
   "outputs": [],
   "source": [
    "#!/usr/bin/env python\n",
    "\"\"\"\n",
    "Wildfire Analysis Dashboard\n",
    "---------------------------\n",
    "\n",
    "This application analyses wildfire data in Algeria (2001-2020) by processing\n",
    "land cover and burn date data, then displays the results via an interactive\n",
    "dashboard built with Plotly Dash.\n",
    "\n",
    "Tasks performed:\n",
    "    1. Validate that the required data files exist.\n",
    "    2. Process global land cover data and burn date data.\n",
    "    3. Compute province-specific statistics.\n",
    "    4. Build an interactive dashboard to visualise wildfire-related metrics.\n",
    "\n",
    "Requirements:\n",
    "    - dash\n",
    "    - plotly\n",
    "    - rasterio\n",
    "    - numpy\n",
    "    - pandas\n",
    "    - geopandas\n",
    "    - rasterstats\n",
    "\n",
    "Author: [Your Name]\n",
    "Date: [YYYY-MM-DD]\n",
    "License: [Appropriate License]\n",
    "\"\"\"\n",
    "\n",
    "import os\n",
    "import dash\n",
    "from dash import dcc, html, Input, Output\n",
    "import plotly.express as px\n",
    "import plotly.graph_objects as go\n",
    "import rasterio\n",
    "import rasterio.mask\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import geopandas as gpd\n",
    "from rasterstats import zonal_stats\n",
    "\n",
    "# ----------------------- Constants and File Paths -------------------------\n",
    "PROVINCES_PATH = 'data/Dz_adm1.shp'\n",
    "LANDCOVER_TEMPLATE = 'data\\LandCover_Exports\\LandCover_Summer_{year}_95Percent_Confidence.tif'\n",
    "BURNDATE_TEMPLATE = 'data\\BurnDate_Exports\\BurnDate_Summer_{year}_95Percent_Confidence.tif'\n",
    "START_YEAR = 2001\n",
    "END_YEAR = 2020  # inclusive\n",
    "PIXEL_AREA = 250 * 250 / 1e6  # km² per pixel\n",
    "\n",
    "# Load administrative boundaries\n",
    "provinces = gpd.read_file(PROVINCES_PATH)\n",
    "\n",
    "# ----------------------- File Validation -------------------------\n",
    "def validate_files():\n",
    "    \"\"\"\n",
    "    Validate that all required land cover and burn date files exist.\n",
    "\n",
    "    Raises:\n",
    "        FileNotFoundError: If one or more required files are missing.\n",
    "    \"\"\"\n",
    "    missing = []\n",
    "    for year in range(START_YEAR, END_YEAR + 1):\n",
    "        lc_path = LANDCOVER_TEMPLATE.format(year=year)\n",
    "        bd_path = BURNDATE_TEMPLATE.format(year=year)\n",
    "        if not os.path.exists(lc_path):\n",
    "            missing.append(lc_path)\n",
    "        if not os.path.exists(bd_path):\n",
    "            missing.append(bd_path)\n",
    "    if missing:\n",
    "        raise FileNotFoundError(f\"Missing {len(missing)} file(s):\\n\" + \"\\n\".join(missing))\n",
    "\n",
    "validate_files()\n",
    "\n",
    "# ----------------------- Global Data Processing -------------------------\n",
    "def process_landcover_burns() -> pd.DataFrame:\n",
    "    \"\"\"\n",
    "    Process global land cover data for each year by reclassifying pixel values into\n",
    "    aggregated land cover groups and computing burned area.\n",
    "\n",
    "    Returns:\n",
    "        pd.DataFrame: Annual data with burned area (km²) for 'Forest', 'Cropland', and 'Shrubland'.\n",
    "    \"\"\"\n",
    "    annual_data = []\n",
    "    reclass_map = {\n",
    "        10: \"Cropland\", 20: \"Cropland\", 30: \"Cropland\", 40: \"Cropland\",\n",
    "        50: \"Forest\", 60: \"Forest\", 70: \"Forest\", 80: \"Forest\",\n",
    "        90: \"Forest\", 100: \"Forest\", 110: \"Forest\",\n",
    "        120: \"Shrubland\", 130: \"Shrubland\", 140: \"Shrubland\", 150: \"Shrubland\",\n",
    "        170: \"Forest\", 180: \"Shrubland\"\n",
    "    }\n",
    "    for year in range(START_YEAR, END_YEAR + 1):\n",
    "        file_path = LANDCOVER_TEMPLATE.format(year=year)\n",
    "        with rasterio.open(file_path) as src:\n",
    "            data = src.read(1)\n",
    "            valid_mask = data != 0\n",
    "        class_counts = np.bincount(data[valid_mask].flatten(), minlength=181)\n",
    "        record = {\"Year\": year, \"Forest\": 0.0, \"Cropland\": 0.0, \"Shrubland\": 0.0}\n",
    "        for code, group in reclass_map.items():\n",
    "            count = class_counts[code] if code < len(class_counts) else 0\n",
    "            record[group] += count * PIXEL_AREA\n",
    "        annual_data.append(record)\n",
    "    return pd.DataFrame(annual_data)\n",
    "\n",
    "def process_burndates() -> pd.DataFrame:\n",
    "    \"\"\"\n",
    "    Process global burn date data by extracting the day-of-year values from burn date rasters.\n",
    "\n",
    "    Returns:\n",
    "        pd.DataFrame: DataFrame containing 'day_of_year' and 'year' columns.\n",
    "    \"\"\"\n",
    "    all_dates = []\n",
    "    for year in range(START_YEAR, END_YEAR + 1):\n",
    "        file_path = BURNDATE_TEMPLATE.format(year=year)\n",
    "        with rasterio.open(file_path) as src:\n",
    "            data = src.read(1)\n",
    "            dates = data[data > 0].flatten()\n",
    "        df = pd.DataFrame({'day_of_year': dates})\n",
    "        df['year'] = year\n",
    "        all_dates.append(df)\n",
    "    return pd.concat(all_dates, ignore_index=True)\n",
    "\n",
    "print(\"Processing global land cover data...\")\n",
    "landcover_df = process_landcover_burns()\n",
    "print(\"Processing global burn date data...\")\n",
    "burndate_df = process_burndates()\n",
    "\n",
    "# ----------------------- Province-Specific Data Functions -------------------------\n",
    "def get_province_landcover(province_geom) -> pd.DataFrame:\n",
    "    \"\"\"\n",
    "    For a given province geometry, compute annual burned area per aggregated land cover group.\n",
    "\n",
    "    Args:\n",
    "        province_geom: Geometry of the province.\n",
    "\n",
    "    Returns:\n",
    "        pd.DataFrame: Annual data for the province with burned area for each land cover group.\n",
    "    \"\"\"\n",
    "    reclass_map = {\n",
    "        10: \"Cropland\", 20: \"Cropland\", 30: \"Cropland\", 40: \"Cropland\",\n",
    "        50: \"Forest\", 60: \"Forest\", 70: \"Forest\", 80: \"Forest\",\n",
    "        90: \"Forest\", 100: \"Forest\", 110: \"Forest\",\n",
    "        120: \"Shrubland\", 130: \"Shrubland\", 140: \"Shrubland\", 150: \"Shrubland\",\n",
    "        170: \"Forest\", 180: \"Shrubland\"\n",
    "    }\n",
    "    records = []\n",
    "    for year in range(START_YEAR, END_YEAR + 1):\n",
    "        file_path = LANDCOVER_TEMPLATE.format(year=year)\n",
    "        with rasterio.open(file_path) as src:\n",
    "            # Reproject province geometry to raster CRS\n",
    "            geom_proj = gpd.GeoSeries([province_geom], crs=provinces.crs).to_crs(src.crs).iloc[0]\n",
    "            try:\n",
    "                out_image, _ = rasterio.mask.mask(src, [geom_proj], crop=True)\n",
    "            except Exception:\n",
    "                continue\n",
    "            data = out_image[0]\n",
    "            valid_mask = data != 0\n",
    "        record = {\"Year\": year, \"Forest\": 0.0, \"Cropland\": 0.0, \"Shrubland\": 0.0}\n",
    "        if np.any(valid_mask):\n",
    "            class_counts = np.bincount(data[valid_mask].flatten(), minlength=181)\n",
    "        else:\n",
    "            class_counts = np.zeros(181, dtype=int)\n",
    "        for code, group in reclass_map.items():\n",
    "            count = class_counts[code] if code < len(class_counts) else 0\n",
    "            record[group] += count * PIXEL_AREA\n",
    "        records.append(record)\n",
    "    return pd.DataFrame(records) if records else pd.DataFrame(columns=[\"Year\", \"Forest\", \"Cropland\", \"Shrubland\"])\n",
    "\n",
    "def get_province_burndates(province_geom) -> pd.DataFrame:\n",
    "    \"\"\"\n",
    "    For a given province geometry, extract burn date pixel values by year.\n",
    "\n",
    "    Args:\n",
    "        province_geom: Geometry of the province.\n",
    "\n",
    "    Returns:\n",
    "        pd.DataFrame: DataFrame with burn dates (day_of_year) and corresponding year.\n",
    "    \"\"\"\n",
    "    frames = []\n",
    "    for year in range(START_YEAR, END_YEAR + 1):\n",
    "        file_path = BURNDATE_TEMPLATE.format(year=year)\n",
    "        with rasterio.open(file_path) as src:\n",
    "            geom_proj = gpd.GeoSeries([province_geom], crs=provinces.crs).to_crs(src.crs).iloc[0]\n",
    "            try:\n",
    "                out_image, _ = rasterio.mask.mask(src, [geom_proj], crop=True)\n",
    "            except Exception:\n",
    "                continue\n",
    "            data = out_image[0]\n",
    "            valid = data > 0\n",
    "            if np.any(valid):\n",
    "                dates = data[valid].flatten()\n",
    "                df = pd.DataFrame({'day_of_year': dates})\n",
    "                df['year'] = year\n",
    "                frames.append(df)\n",
    "    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['day_of_year', 'year'])\n",
    "\n",
    "# ----------------------- Dashboard Layout -------------------------\n",
    "app = dash.Dash(__name__)\n",
    "app.title = \"Algeria Wildfire Analysis (2001-2020)\"\n",
    "\n",
    "app.layout = html.Div([\n",
    "    html.H1(\"Algeria Wildfire Analysis (2001-2020)\",\n",
    "            style={'textAlign': 'center', 'color': '#2c3e50'}),\n",
    "    \n",
    "    # Year Dropdown: \"Total\" (aggregated) or a specific year.\n",
    "    html.Div([\n",
    "        html.Label(\"Select Year for Map:\"),\n",
    "        dcc.Dropdown(\n",
    "            id='year-dropdown',\n",
    "            options=[{'label': 'Total', 'value': 'total'}] +\n",
    "                    [{'label': str(year), 'value': year} for year in range(START_YEAR, END_YEAR + 1)],\n",
    "            value='total',\n",
    "            clearable=False,\n",
    "            style={'width': '200px'}\n",
    "        )\n",
    "    ], style={'textAlign': 'center', 'padding': '10px'}),\n",
    "    \n",
    "    # Global map display\n",
    "    dcc.Graph(id='province-map'),\n",
    "    html.Div(id='map-status', style={'textAlign': 'center', 'color': '#7f8c8d', 'padding': '10px'}),\n",
    "    \n",
    "    html.Hr(),\n",
    "    \n",
    "    # Charts: Land cover analysis and daily burn pattern.\n",
    "    html.Div([\n",
    "        html.Div([\n",
    "            dcc.Graph(id='landcover-time-series'),\n",
    "            dcc.Checklist(\n",
    "                id='landcover-selector',\n",
    "                options=[{'label': cat, 'value': cat} for cat in ['Forest', 'Cropland', 'Shrubland']],\n",
    "                value=['Forest', 'Cropland', 'Shrubland'],\n",
    "                inline=True,\n",
    "                style={'padding': '10px'}\n",
    "            )\n",
    "        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),\n",
    "        \n",
    "        html.Div([\n",
    "            dcc.Graph(id='daily-burn-pattern')\n",
    "        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})\n",
    "    ])\n",
    "])\n",
    "\n",
    "# ----------------------- Callbacks -------------------------\n",
    "@app.callback(\n",
    "    [Output('province-map', 'figure'),\n",
    "     Output('map-status', 'children')],\n",
    "    [Input('year-dropdown', 'value')]\n",
    ")\n",
    "def update_map(selected_year):\n",
    "    \"\"\"\n",
    "    Update the map based on the selected year. Aggregates data if \"total\" is selected.\n",
    "\n",
    "    Args:\n",
    "        selected_year: Selected year from the dropdown or 'total'.\n",
    "\n",
    "    Returns:\n",
    "        tuple: (Map figure, status message)\n",
    "    \"\"\"\n",
    "    if selected_year == 'total':\n",
    "        province_stats = {i: 0 for i in range(len(provinces))}\n",
    "        for year in range(START_YEAR, END_YEAR + 1):\n",
    "            file_path = BURNDATE_TEMPLATE.format(year=year)\n",
    "            if not os.path.exists(file_path):\n",
    "                continue\n",
    "            with rasterio.open(file_path) as src:\n",
    "                raster_crs = src.crs\n",
    "                affine = src.transform\n",
    "                provinces_reproj = provinces.to_crs(raster_crs)\n",
    "                burned_mask = src.read(1) > 0\n",
    "            stats = zonal_stats(\n",
    "                provinces_reproj,\n",
    "                burned_mask,\n",
    "                affine=affine,\n",
    "                stats=['sum'],\n",
    "                nodata=0,\n",
    "                all_touched=True,\n",
    "                geojson_out=True\n",
    "            )\n",
    "            for i, stat in enumerate(stats):\n",
    "                burned_val = stat['properties'].get('sum', 0)\n",
    "                province_stats[i] += burned_val if burned_val is not None else 0\n",
    "        gdf_stats = provinces.copy()\n",
    "        gdf_stats['burned_area'] = [province_stats[i] for i in range(len(provinces))]\n",
    "        gdf_stats['burned_area'] = gdf_stats['burned_area'] * (250**2) / 1e6\n",
    "        map_title = \"Total Burned Area (2001-2020)\"\n",
    "    else:\n",
    "        file_path = BURNDATE_TEMPLATE.format(year=selected_year)\n",
    "        if not os.path.exists(file_path):\n",
    "            return go.Figure(), \"Data not available for selected year.\"\n",
    "        with rasterio.open(file_path) as src:\n",
    "            raster_crs = src.crs\n",
    "            affine = src.transform\n",
    "            provinces_reproj = provinces.to_crs(raster_crs)\n",
    "            burned_mask = src.read(1) > 0\n",
    "        stats = zonal_stats(\n",
    "            provinces_reproj,\n",
    "            burned_mask,\n",
    "            affine=affine,\n",
    "            stats=['sum'],\n",
    "            nodata=0,\n",
    "            all_touched=True,\n",
    "            geojson_out=True\n",
    "        )\n",
    "        gdf_stats = gpd.GeoDataFrame.from_features(stats)\n",
    "        gdf_stats.crs = raster_crs\n",
    "        gdf_stats = gdf_stats.to_crs(provinces.crs)\n",
    "        gdf_stats['burned_area'] = gdf_stats['sum'].fillna(0) * (250**2) / 1e6\n",
    "        map_title = f\"Burned Area in {selected_year}\"\n",
    "    \n",
    "    fig = px.choropleth_mapbox(\n",
    "        gdf_stats,\n",
    "        geojson=gdf_stats.geometry,\n",
    "        locations=gdf_stats.index,\n",
    "        color='burned_area',\n",
    "        mapbox_style=\"carto-positron\",\n",
    "        zoom=4.5,\n",
    "        center={\"lat\": 36, \"lon\": 3},\n",
    "        opacity=0.8,\n",
    "        labels={'burned_area': 'Burned Area (km²)'},\n",
    "        color_continuous_scale='YlOrRd',\n",
    "        hover_data={'ADM1_EN': True, 'burned_area': ':.2f'}\n",
    "    )\n",
    "    fig.update_layout(margin={\"r\": 0, \"t\": 0, \"l\": 0, \"b\": 0},\n",
    "                      title={'text': map_title, 'x': 0.5})\n",
    "    return fig, f\"Map displaying {map_title}.\"\n",
    "\n",
    "@app.callback(\n",
    "    [Output('landcover-time-series', 'figure'),\n",
    "     Output('daily-burn-pattern', 'figure')],\n",
    "    [Input('province-map', 'clickData'),\n",
    "     Input('landcover-selector', 'value')]\n",
    ")\n",
    "def update_charts(clickData, selected_categories):\n",
    "    \"\"\"\n",
    "    Update the land cover and daily burn pattern charts based on the selected province.\n",
    "    If no province is selected, global data is displayed.\n",
    "\n",
    "    Args:\n",
    "        clickData: Data from the map click event.\n",
    "        selected_categories: List of land cover groups to display.\n",
    "\n",
    "    Returns:\n",
    "        tuple: (Land cover time series figure, Daily burn pattern figure)\n",
    "    \"\"\"\n",
    "    if clickData is None:\n",
    "        df_lc = landcover_df.copy()\n",
    "        df_bd = burndate_df.copy()\n",
    "        title_suffix = \" (All Provinces)\"\n",
    "    else:\n",
    "        province_index = clickData['points'][0]['location']\n",
    "        selected_province = provinces.iloc[int(province_index)]\n",
    "        province_name = selected_province.get('ADM1_EN', f\"Province {province_index}\")\n",
    "        title_suffix = f\" ({province_name})\"\n",
    "        df_lc = get_province_landcover(selected_province.geometry)\n",
    "        df_bd = get_province_burndates(selected_province.geometry)\n",
    "    \n",
    "    # Land Cover Bar Plot\n",
    "    if not df_lc.empty:\n",
    "        df_melted = df_lc.melt(id_vars='Year', value_vars=['Forest', 'Cropland', 'Shrubland'],\n",
    "                               var_name='Category', value_name='Burned Area')\n",
    "        df_melted = df_melted[df_melted['Category'].isin(selected_categories)]\n",
    "        fig_lc = px.bar(df_melted, x='Year', y='Burned Area', color='Category', barmode='group',\n",
    "                        labels={'Burned Area': 'Burned Area (km²)', 'Category': 'Land Cover Group'},\n",
    "                        template='plotly_white',\n",
    "                        title=\"Land Cover Burned Area\" + title_suffix)\n",
    "    else:\n",
    "        fig_lc = go.Figure()\n",
    "        fig_lc.update_layout(title=\"No land cover data available\" + title_suffix)\n",
    "    \n",
    "    # Daily Burn Pattern Bar Plot\n",
    "    if not df_bd.empty:\n",
    "        daily_counts = df_bd.groupby('day_of_year').size().reset_index(name='counts')\n",
    "        fig_bd = go.Figure()\n",
    "        fig_bd.add_trace(go.Bar(\n",
    "            x=daily_counts['day_of_year'],\n",
    "            y=daily_counts['counts'],\n",
    "            marker_color='#e74c3c',\n",
    "            name='Burn Frequency'\n",
    "        ))\n",
    "        fig_bd.update_layout(\n",
    "            title=\"Daily Burn Frequency\" + title_suffix,\n",
    "            xaxis_title='Day of Year',\n",
    "            yaxis_title='Number of Burned Pixels (250m)',\n",
    "            hovermode='x unified',\n",
    "            template='plotly_white',\n",
    "            showlegend=False\n",
    "        )\n",
    "    else:\n",
    "        fig_bd = go.Figure()\n",
    "        fig_bd.update_layout(title=\"No burn date data available\" + title_suffix)\n",
    "    \n",
    "    return fig_lc, fig_bd\n",
    "\n",
    "# ----------------------- Main Entry Point -------------------------\n",
    "def main():\n",
    "    \"\"\"\n",
    "    Main entry point for the Dash application.\n",
    "    \"\"\"\n",
    "    port = int(os.environ.get(\"PORT\", 8051))\n",
    "    app.run_server(debug=False, host='0.0.0.0', port=port)\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    main()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
