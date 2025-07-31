#!/usr/bin/env python
"""
Wildfire Analysis Dashboard - Streamlit Version
-----------------------------------------------

This application analyses wildfire data in Algeria (2001-2020) by processing
land cover and burn date data, then displays the results via an interactive
dashboard built with Streamlit.

Requirements:
    - streamlit
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
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats
from pathlib import Path

# ----------------------- Page Configuration -------------------------
st.set_page_config(
    page_title="Algeria Wildfire Analysis",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------- Constants and File Paths -------------------------
PROVINCES_PATH = Path(r"C:\Users\lenovo legion\Documents\ArcGIS data\DZ\frontiere\dza_admbnda_adm1_unhcr_20200120.shp")
LANDCOVER_TEMPLATE = r"C:\Users\Lenovo legion\Desktop\Datasets\LandCover_Exports\LandCover_Summer_{year}_95Percent_Confidence.tif"
BURNDATE_TEMPLATE = r"C:\Users\Lenovo legion\Desktop\Datasets\LandCover_Exports\BurnDate_Summer_{year}_95Percent_Confidence.tif"

START_YEAR = 2001
END_YEAR = 2020  # inclusive
PIXEL_AREA = 250 * 250 / 1e6  # km² per pixel

# ----------------------- Cached Data Loading Functions -------------------------
@st.cache_data
def load_provinces():
    """Load and cache administrative boundaries"""
    try:
        provinces = gpd.read_file(PROVINCES_PATH)
        st.success(f"✅ Provinces DataFrame loaded with shape: {provinces.shape}")
        
        if provinces.crs is None or provinces.crs.to_string().lower() != "epsg:4326":
            provinces = provinces.to_crs(epsg=4326)
            st.info("🔄 Provinces reprojected to EPSG:4326")
        
        return provinces
    except Exception as e:
        st.error(f"❌ Failed to load provinces shapefile: {e}")
        return None

@st.cache_data
def validate_files():
    """Validate that all required files exist"""
    missing = []
    for year in range(START_YEAR, END_YEAR + 1):
        lc_path = LANDCOVER_TEMPLATE.format(year=year)
        bd_path = BURNDATE_TEMPLATE.format(year=year)
        if not os.path.exists(lc_path):
            missing.append(lc_path)
        if not os.path.exists(bd_path):
            missing.append(bd_path)
    
    if missing:
        st.error(f"❌ Missing {len(missing)} file(s):")
        for file in missing:
            st.write(f"- {file}")
        return False
    else:
        st.success("✅ All required raster files exist")
        return True

@st.cache_data
def process_landcover_burns():
    """Process global land cover data"""
    annual_data = []
    reclass_map = {
        10: "Cropland", 20: "Cropland", 30: "Cropland", 40: "Cropland",
        50: "Forest", 60: "Forest", 70: "Forest", 80: "Forest",
        90: "Forest", 100: "Forest", 110: "Forest",
        120: "Shrubland", 130: "Shrubland", 140: "Shrubland", 150: "Shrubland",
        170: "Forest", 180: "Shrubland"
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, year in enumerate(range(START_YEAR, END_YEAR + 1)):
        progress_bar.progress((i + 1) / (END_YEAR - START_YEAR + 1))
        status_text.text(f'Processing land cover for year {year}...')
        
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
    
    progress_bar.empty()
    status_text.empty()
    
    df = pd.DataFrame(annual_data)
    st.success(f"✅ Global land cover processing complete. DataFrame shape: {df.shape}")
    return df

@st.cache_data
def process_burndates():
    """Process global burn date data"""
    all_dates = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, year in enumerate(range(START_YEAR, END_YEAR + 1)):
        progress_bar.progress((i + 1) / (END_YEAR - START_YEAR + 1))
        status_text.text(f'Processing burn dates for year {year}...')
        
        file_path = BURNDATE_TEMPLATE.format(year=year)
        with rasterio.open(file_path) as src:
            data = src.read(1)
            dates = data[data > 0].flatten()
        
        df = pd.DataFrame({'day_of_year': dates})
        df['year'] = year
        all_dates.append(df)
    
    progress_bar.empty()
    status_text.empty()
    
    combined = pd.concat(all_dates, ignore_index=True)
    st.success(f"✅ Global burn date processing complete. DataFrame shape: {combined.shape}")
    return combined

# ----------------------- Province-Specific Data Functions -------------------------
def get_province_landcover(province_geom):
    """Compute annual burned area per land cover group for a province"""
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
            geom_proj = gpd.GeoSeries([province_geom], crs="EPSG:4326").to_crs(src.crs).iloc[0]
            try:
                out_image, _ = rasterio.mask.mask(src, [geom_proj], crop=True)
            except Exception as e:
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
    
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["Year", "Forest", "Cropland", "Shrubland"])

def get_province_burndates(province_geom):
    """Extract burn date pixel values for a province"""
    frames = []
    for year in range(START_YEAR, END_YEAR + 1):
        file_path = BURNDATE_TEMPLATE.format(year=year)
        with rasterio.open(file_path) as src:
            geom_proj = gpd.GeoSeries([province_geom], crs="EPSG:4326").to_crs(src.crs).iloc[0]
            try:
                out_image, _ = rasterio.mask.mask(src, [geom_proj], crop=True)
            except Exception as e:
                continue
            
            data = out_image[0]
            valid = data > 0
            
            if np.any(valid):
                dates = data[valid].flatten()
                df = pd.DataFrame({'day_of_year': dates})
                df['year'] = year
                frames.append(df)
    
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['day_of_year', 'year'])

# ----------------------- Chart Creation Functions -------------------------
def create_map(provinces, selected_year):
    """Create the choropleth map"""
    if selected_year == 'Total':
        province_stats = {i: 0 for i in range(len(provinces))}
        
        for year in range(START_YEAR, END_YEAR + 1):
            file_path = BURNDATE_TEMPLATE.format(year=year)
            if not os.path.exists(file_path):
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
        
        gdf_stats = provinces.copy()
        gdf_stats['burned_area'] = [province_stats[i] for i in range(len(provinces))]
        gdf_stats['burned_area'] = gdf_stats['burned_area'] * (250**2) / 1e6
        map_title = "Total Burned Area (2001-2020)"
        
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
    
    # Ensure EPSG:4326
    if gdf_stats.crs.to_string().lower() != "epsg:4326":
        gdf_stats = gdf_stats.to_crs(epsg=4326)
    
    # Create choropleth map
    geojson_data = gdf_stats.__geo_interface__
    
    fig = px.choropleth_mapbox(
        gdf_stats,
        geojson=geojson_data,
        locations=gdf_stats.index,
        color='burned_area',
        mapbox_style="white-bg",
        opacity=0.7,
        color_continuous_scale='YlOrRd',
        title=map_title,
        labels={'burned_area': 'Burned Area (km²)'}
    )
    
    # Center the map
    centroid = gdf_stats.geometry.unary_union.centroid
    fig.update_layout(
        mapbox=dict(
            center={"lat": centroid.y, "lon": centroid.x},
            zoom=4.5
        ),
        margin={"r":0,"t":30,"l":0,"b":0}
    )
    
    fig.update_traces(marker_opacity=0.7, marker_line_width=1, marker_line_color="black")
    
    return fig, f"Map displaying {map_title}"

def create_landcover_chart(df_lc, selected_categories, title_suffix):
    """Create land cover time series chart"""
    if df_lc.empty:
        fig = go.Figure()
        fig.update_layout(title="No land cover data available" + title_suffix)
        return fig
    
    df_melted = df_lc.melt(
        id_vars='Year', 
        value_vars=['Forest', 'Cropland', 'Shrubland'],
        var_name='Category', 
        value_name='Burned Area'
    )
    df_melted = df_melted[df_melted['Category'].isin(selected_categories)]
    
    fig = px.bar(
        df_melted, 
        x='Year', 
        y='Burned Area', 
        color='Category', 
        barmode='group',
        labels={'Burned Area': 'Burned Area (km²)', 'Category': 'Land Cover Group'},
        template='plotly_white',
        title="Land Cover Burned Area" + title_suffix
    )
    
    return fig

def create_burn_pattern_chart(df_bd, title_suffix):
    """Create daily burn pattern chart"""
    if df_bd.empty:
        fig = go.Figure()
        fig.update_layout(title="No burn date data available" + title_suffix)
        return fig
    
    daily_counts = df_bd.groupby('day_of_year').size().reset_index(name='counts')
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily_counts['day_of_year'],
        y=daily_counts['counts'],
        marker_color='#e74c3c',
        name='Burn Frequency'
    ))
    
    fig.update_layout(
        title="Daily Burn Frequency" + title_suffix,
        xaxis_title='Day of Year',
        yaxis_title='Number of Burned Pixels (250m)',
        template='plotly_white',
        showlegend=False
    )
    
    return fig

# ----------------------- Main Application -------------------------
def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🔥 Algeria Wildfire Analysis (2001-2020)")
    st.markdown("---")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Load and validate data
    with st.sidebar:
        st.header("🎛️ Controls")
        
        if st.button("🔄 Load/Refresh Data"):
            st.session_state.data_loaded = False
    
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            # Load provinces
            provinces = load_provinces()
            if provinces is None:
                st.stop()
            
            # Validate files
            if not validate_files():
                st.stop()
            
            # Process global data
            st.subheader("🔄 Processing Global Data")
            landcover_df = process_landcover_burns()
            burndate_df = process_burndates()
            
            # Store in session state
            st.session_state.provinces = provinces
            st.session_state.landcover_df = landcover_df
            st.session_state.burndate_df = burndate_df
            st.session_state.data_loaded = True
            
            st.success("✅ All data loaded successfully!")
    
    # Get data from session state
    provinces = st.session_state.provinces
    landcover_df = st.session_state.landcover_df
    burndate_df = st.session_state.burndate_df
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("---")
        
        # Year selection
        year_options = ['Total'] + list(range(START_YEAR, END_YEAR + 1))
        selected_year = st.selectbox(
            "📅 Select Year for Map:",
            year_options,
            index=0
        )
        
        # Province selection
        province_names = ['All Provinces'] + sorted(provinces['ADM1_EN'].tolist())
        selected_province_name = st.selectbox(
            "🗺️ Select Province for Analysis:",
            province_names,
            index=0
        )
        
        # Land cover categories
        landcover_categories = st.multiselect(
            "🌳 Land Cover Categories:",
            ['Forest', 'Cropland', 'Shrubland'],
            default=['Forest', 'Cropland', 'Shrubland']
        )
        
        st.markdown("---")
        st.markdown("### 📊 Data Info")
        st.info(f"**Years:** {START_YEAR}-{END_YEAR}\n\n**Provinces:** {len(provinces)}\n\n**Pixel Size:** 250m")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🗺️ Provincial Burned Area Map")
        
        with st.spinner("Creating map..."):
            map_fig, map_status = create_map(provinces, selected_year)
        
        st.plotly_chart(map_fig, use_container_width=True)
        st.info(map_status)
    
    with col2:
        st.subheader("📊 Land Cover Analysis")
        
        # Get data based on selection
        if selected_province_name == 'All Provinces':
            df_lc = landcover_df.copy()
            title_suffix = " (All Provinces)"
        else:
            with st.spinner(f"Processing data for {selected_province_name}..."):
                province_row = provinces[provinces['ADM1_EN'] == selected_province_name].iloc[0]
                df_lc = get_province_landcover(province_row.geometry)
                title_suffix = f" ({selected_province_name})"
        
        lc_fig = create_landcover_chart(df_lc, landcover_categories, title_suffix)
        st.plotly_chart(lc_fig, use_container_width=True)
    
    # Daily burn pattern (full width)
    st.subheader("📈 Daily Burn Pattern")
    
    if selected_province_name == 'All Provinces':
        df_bd = burndate_df.copy()
        title_suffix = " (All Provinces)"
    else:
        with st.spinner(f"Processing burn dates for {selected_province_name}..."):
            province_row = provinces[provinces['ADM1_EN'] == selected_province_name].iloc[0]
            df_bd = get_province_burndates(province_row.geometry)
            title_suffix = f" ({selected_province_name})"
    
    bd_fig = create_burn_pattern_chart(df_bd, title_suffix)
    st.plotly_chart(bd_fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**Algeria Wildfire Analysis Dashboard** | Built with Streamlit 🚀")

if __name__ == '__main__':
    main()
