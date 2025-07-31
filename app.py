#!/usr/bin/env python3
"""
Optimised Wildfire Analysis Dashboard for Algeria (2001-2020) - Streamlit Version
----------------------------------------------------------------------------------
This Streamlit version maintains the performance optimizations from the original:
  - Using pathlib for path handling
  - Caching heavy computations with st.cache_data and st.cache_resource
  - Parallelising raster processing with ThreadPoolExecutor
  - Adding logging instead of print statements
  - Fixed province click handling and statistics display

Author: [Your Name]
Date:   [YYYY-MM-DD]
License: [Appropriate License]
"""
import logging
from pathlib import Path
import pickle
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import rasterio
from rasterio.mask import mask
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from rasterstats import zonal_stats

# ----------------------- Streamlit Page Configuration -----------------------
st.set_page_config(
    page_title="Algeria Wildfire Analysis",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------- Logging Configuration -----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# ----------------------- Constants & Paths -----------------------
# Convert string paths to Path objects
PROVINCES_PATH = 'data/Dz_adm1.shp'
LANDCOVER_TEMPLATE = 'data/LandCover_Exports/LandCover_Summer_{year}_95Percent_Confidence.tif'
BURNDATE_TEMPLATE = 'data/BurnDate_Exports/BurnDate_Summer_{year}_95Percent_Confidence.tif'

START_YEAR, END_YEAR = 2001, 2020  # inclusive
PIXEL_AREA = 250 * 250 / 1e6  # km² per pixel
CATEGORIES = ["Forest", "Cropland", "Shrubland"]

# Mapping raster class codes to aggregated categories
RECLASS_MAP = {
    10: "Cropland", 20: "Cropland", 30: "Cropland", 40: "Cropland",
    50: "Forest", 60: "Forest", 70: "Forest", 80: "Forest",
    90: "Forest", 100: "Forest", 110: "Forest", 170: "Forest",
    120: "Shrubland", 130: "Shrubland", 140: "Shrubland", 150: "Shrubland", 180: "Shrubland"
}
# Precompute a LUT for fast reclassification
MAX_CODE = max(RECLASS_MAP.keys())
lut = np.full(MAX_CODE+1, -1, dtype=int)
for code, cat in RECLASS_MAP.items():
    lut[code] = CATEGORIES.index(cat)

# ----------------------- Utility Functions -----------------------
@st.cache_resource
def load_provinces() -> gpd.GeoDataFrame:
    """Load administrative boundaries once."""
    logger.info("Loading provinces shapefile...")
    if not PROVINCES_PATH.exists():
        st.error(f"Provinces shapefile not found at: {PROVINCES_PATH}")
        return gpd.GeoDataFrame()
    
    gdf = gpd.read_file(PROVINCES_PATH)
    
    # Check which column contains province names
    possible_columns = ['ADM1_EN', 'ADM0_EN', 'NAME_1', 'NAME', 'ADMIN1']
    province_col = None
    
    for col in possible_columns:
        if col in gdf.columns:
            province_col = col
            break
    
    if province_col is None:
        st.warning(f"Available columns: {list(gdf.columns)}")
        st.error("Could not find province name column. Please check your shapefile.")
        return gpd.GeoDataFrame()
    
    # Standardize the column name
    if province_col != 'ADM1_EN':
        gdf['ADM1_EN'] = gdf[province_col]
    
    logger.info(f"Loaded {len(gdf)} provinces using column '{province_col}'")
    return gdf

@st.cache_data
def provinces_in_crs(crs) -> gpd.GeoDataFrame:
    """Reproject provinces to given CRS, cached."""
    provinces = load_provinces()
    if provinces.empty:
        return provinces
    return provinces.to_crs(crs)

def windowed_mask(src, geom) -> np.ndarray:
    """Read only the window covering the geometry and mask."""
    try:
        out_image, _ = mask(src, [geom], crop=True)
        return out_image[0]
    except Exception as e:
        logger.warning(f"Masking failed: {e}")
        return np.array([])

def safe_lut_lookup(data, lut):
    """Safely apply LUT lookup with bounds checking."""
    # Ensure data is within bounds of LUT
    valid_indices = (data >= 0) & (data < len(lut))
    result = np.full_like(data, -1, dtype=int)
    result[valid_indices] = lut[data[valid_indices]]
    return result

# ----------------------- Global Data Processing -----------------------

def _process_lc_year(year: int) -> dict:
    """Helper for parallel landcover processing of one year."""
    path = Path(LANDCOVER_TEMPLATE.format(year=year))
    if not path.exists():
        logger.warning(f"Landcover file missing for {year}: {path}")
        return None
    try:
        with rasterio.open(path) as src:
            data = src.read(1)
        
        # Only process valid pixels (non-zero values)
        valid_mask = data > 0
        if not np.any(valid_mask):
            logger.warning(f"No valid data found for year {year}")
            return {"Year": year, **{cat: 0.0 for cat in CATEGORIES}}
        
        valid_data = data[valid_mask]
        classes = safe_lut_lookup(valid_data, lut)
        
        # Only count pixels that map to valid categories
        valid_classes = classes[classes >= 0]
        if len(valid_classes) == 0:
            logger.warning(f"No valid land cover classes found for year {year}")
            return {"Year": year, **{cat: 0.0 for cat in CATEGORIES}}
        
        counts = np.bincount(valid_classes, minlength=len(CATEGORIES))
        areas = counts * PIXEL_AREA
        
        result = {"Year": year, **{cat: areas[i] for i, cat in enumerate(CATEGORIES)}}
        logger.info(f"Year {year}: Total area = {sum(areas):.2f} km²")
        return result
        
    except Exception as e:
        logger.error(f"Error processing landcover for year {year}: {e}")
        return None

@st.cache_data
def compute_global_landcover() -> pd.DataFrame:
    """Compute or load cached global landcover burned areas."""
    cache_file = BASE_DIR / "cache_global_landcover.pkl"
    if cache_file.exists():
        logger.info("Loading cached global landcover data...")
        try:
            return pickle.load(cache_file.open('rb'))
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    logger.info("Processing global landcover data in parallel...")
    records = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_process_lc_year, yr): yr for yr in range(START_YEAR, END_YEAR+1)}
        for fut in as_completed(futures):
            rec = fut.result()
            if rec:
                records.append(rec)
    
    if not records:
        st.warning("No landcover data found!")
        return pd.DataFrame(columns=["Year"] + CATEGORIES)
    
    df = pd.DataFrame(sorted(records, key=lambda x: x['Year']))
    
    # Try to cache the result
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(df, cache_file.open('wb'))
    except Exception as e:
        logger.warning(f"Failed to cache data: {e}")
    
    return df

@st.cache_data
def compute_global_burndates() -> pd.DataFrame:
    """Compute or load cached global burn dates."""
    cache_file = BASE_DIR / "cache_global_burndates.pkl"
    if cache_file.exists():
        logger.info("Loading cached global burn date data...")
        try:
            return pickle.load(cache_file.open('rb'))
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    logger.info("Processing global burn date data...")
    frames = []
    for year in range(START_YEAR, END_YEAR+1):
        path = Path(BURNDATE_TEMPLATE.format(year=year))
        if not path.exists():
            logger.warning(f"BurnDate file missing for {year}: {path}")
            continue
        try:
            with rasterio.open(path) as src:
                data = src.read(1)
            days = data[data > 0].flatten()
            df = pd.DataFrame({'day_of_year': days, 'year': year})
            frames.append(df)
        except Exception as e:
            logger.error(f"Error processing burn dates for year {year}: {e}")
    
    if not frames:
        st.warning("No burn date data found!")
        return pd.DataFrame(columns=['day_of_year', 'year'])
    
    result = pd.concat(frames, ignore_index=True)
    
    # Try to cache the result
    try:
        pickle.dump(result, cache_file.open('wb'))
    except Exception as e:
        logger.warning(f"Failed to cache data: {e}")
    
    return result

# ----------------------- Province‐Specific Functions -----------------------

@st.cache_data
def process_province_landcover(prov_idx: int) -> pd.DataFrame:
    """Compute annual burned areas per landcover for one province."""
    provinces = load_provinces()
    if provinces.empty or prov_idx >= len(provinces):
        return pd.DataFrame(columns=["Year"]+CATEGORIES)
    
    geom = provinces.iloc[prov_idx].geometry
    province_name = provinces.iloc[prov_idx]['ADM1_EN']
    logger.info(f"Processing landcover for province: {province_name}")
    
    records = []
    for year in range(START_YEAR, END_YEAR+1):
        path = Path(LANDCOVER_TEMPLATE.format(year=year))
        if not path.exists():
            logger.warning(f"Landcover file missing for {year}: {path}")
            continue
        try:
            with rasterio.open(path) as src:
                geom_proj = gpd.GeoSeries([geom], crs=provinces.crs).to_crs(src.crs).iloc[0]
                data = windowed_mask(src, geom_proj)
            
            if data.size == 0:
                logger.warning(f"No data extracted for {province_name} in year {year}")
                continue
            
            # Only process valid pixels (non-zero values)
            valid_mask = data > 0
            if not np.any(valid_mask):
                logger.warning(f"No valid pixels for {province_name} in year {year}")
                continue
            
            valid_data = data[valid_mask]
            classes = safe_lut_lookup(valid_data, lut)
            
            # Only count pixels that map to valid categories
            valid_classes = classes[classes >= 0]
            if len(valid_classes) == 0:
                logger.warning(f"No valid land cover classes for {province_name} in year {year}")
                continue
            
            counts = np.bincount(valid_classes, minlength=len(CATEGORIES))
            areas = counts * PIXEL_AREA
            
            record = {"Year": year, **{cat: areas[i] for i, cat in enumerate(CATEGORIES)}}
            total_area = sum(areas)
            if total_area > 0:
                logger.info(f"{province_name} {year}: Total area = {total_area:.2f} km²")
                records.append(record)
            
        except Exception as e:
            logger.error(f"Error processing province landcover for {province_name} year {year}: {e}")
    
    result_df = pd.DataFrame(records) if records else pd.DataFrame(columns=["Year"]+CATEGORIES)
    logger.info(f"Province {province_name}: Processed {len(records)} years of data")
    return result_df

@st.cache_data
def process_province_burndates(prov_idx: int) -> pd.DataFrame:
    """Extract burn dates for one province."""
    provinces = load_provinces()
    if provinces.empty or prov_idx >= len(provinces):
        return pd.DataFrame(columns=['day_of_year','year'])
    
    geom = provinces.iloc[prov_idx].geometry
    frames = []
    for year in range(START_YEAR, END_YEAR+1):
        path = Path(BURNDATE_TEMPLATE.format(year=year))
        if not path.exists():
            continue
        try:
            with rasterio.open(path) as src:
                geom_proj = gpd.GeoSeries([geom], crs=provinces.crs).to_crs(src.crs).iloc[0]
                data = windowed_mask(src, geom_proj)
            if data.size == 0:
                continue
            days = data[data > 0].flatten()
            frames.append(pd.DataFrame({'day_of_year': days, 'year': year}))
        except Exception as e:
            logger.error(f"Error processing province burn dates for year {year}: {e}")
    
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['day_of_year','year'])

@st.cache_data
def get_province_statistics(prov_idx: int, selected_year) -> dict:
    """Get statistics for a specific province and year."""
    provinces = load_provinces()
    if provinces.empty or prov_idx >= len(provinces):
        return {}
    
    province_name = provinces.iloc[prov_idx]['ADM1_EN']
    
    if selected_year == 'total':
        # Get all years data for this province
        df_lc = process_province_landcover(prov_idx)
        if not df_lc.empty:
            stats = {
                'province_name': province_name,
                'total_burned_area': df_lc[CATEGORIES].sum().sum(),
                'forest_area': df_lc['Forest'].sum(),
                'cropland_area': df_lc['Cropland'].sum(),
                'shrubland_area': df_lc['Shrubland'].sum(),
                'years_with_data': len(df_lc)
            }
        else:
            stats = {
                'province_name': province_name,
                'total_burned_area': 0,
                'forest_area': 0,
                'cropland_area': 0,
                'shrubland_area': 0,
                'years_with_data': 0
            }
    else:
        # Get specific year data for this province
        df_lc = process_province_landcover(prov_idx)
        year_data = df_lc[df_lc['Year'] == selected_year] if not df_lc.empty else pd.DataFrame()
        
        if not year_data.empty:
            stats = {
                'province_name': province_name,
                'total_burned_area': year_data[CATEGORIES].sum().sum(),
                'forest_area': year_data['Forest'].iloc[0],
                'cropland_area': year_data['Cropland'].iloc[0],
                'shrubland_area': year_data['Shrubland'].iloc[0],
                'year': selected_year
            }
        else:
            stats = {
                'province_name': province_name,
                'total_burned_area': 0,
                'forest_area': 0,
                'cropland_area': 0,
                'shrubland_area': 0,
                'year': selected_year
            }
    
    return stats

# ----------------------- Map Generation Function -----------------------
@st.cache_data
def generate_map(selected_year):
    """Generate the choropleth map for the selected year."""
    provinces = load_provinces()
    if provinces.empty:
        return None, "Provinces data not available"
    
    if selected_year == 'total':
        # aggregate by province across all years
        stats = {i: 0 for i in range(len(provinces))}
        for year in range(START_YEAR, END_YEAR+1):
            path = Path(BURNDATE_TEMPLATE.format(year=year))
            if not path.exists():
                continue
            try:
                with rasterio.open(path) as src:
                    arr = src.read(1) > 0
                    provinces_proj = provinces_in_crs(src.crs)
                    if provinces_proj.empty:
                        continue
                    zs = zonal_stats(provinces_proj, arr, affine=src.transform,
                                     stats=['sum'], nodata=0, all_touched=True)
                for i, z in enumerate(zs):
                    stats[i] += z.get('sum', 0) or 0
            except Exception as e:
                logger.error(f"Error processing map data for year {year}: {e}")
        
        gdf = provinces.copy()
        gdf['burned_area'] = [stats[i] * (250**2) / 1e6 for i in range(len(provinces))]
        title = "Total Burned Area (2001–2020)"
    else:
        path = Path(BURNDATE_TEMPLATE.format(year=selected_year))
        if not path.exists():
            return None, f"No data for {selected_year}"
        try:
            with rasterio.open(path) as src:
                arr = src.read(1) > 0
                provinces_proj = provinces_in_crs(src.crs)
                if provinces_proj.empty:
                    return None, "Provinces projection failed"
                zs = zonal_stats(provinces_proj, arr, affine=src.transform,
                                 stats=['sum'], nodata=0, all_touched=True, geojson_out=True)
            gdf = gpd.GeoDataFrame.from_features(zs)
            gdf.crs = src.crs
            gdf = gdf.to_crs(provinces.crs)
            gdf['burned_area'] = gdf['sum'].fillna(0) * (250**2) / 1e6
            title = f"Burned Area in {selected_year}"
        except Exception as e:
            logger.error(f"Error generating map: {e}")
            return None, f"Error generating map for {selected_year}"

    try:
        # Add province names to hover data
        hover_template = '<b>%{customdata[0]}</b><br>Burned Area: %{z:.2f} km²<extra></extra>'
        
        fig = px.choropleth_mapbox(
            gdf,
            geojson=gdf.geometry,
            locations=gdf.index,
            color='burned_area',
            mapbox_style='carto-positron',
            zoom=4.5,
            center={'lat': 36, 'lon': 3},
            opacity=0.8,
            labels={'burned_area': 'Burned Area (km²)'},
            color_continuous_scale='YlOrRd',
            custom_data=['ADM1_EN'] if 'ADM1_EN' in gdf.columns else [gdf.index]
        )
        
        # Update hover template
        fig.update_traces(hovertemplate=hover_template)
        
        fig.update_layout(
            margin={'r':0,'t':30,'l':0,'b':0}, 
            title={'text': title, 'x':0.5}
        )
        return fig, title
    except Exception as e:
        logger.error(f"Error creating map figure: {e}")
        return None, f"Error creating map visualization"

def debug_data_files():
    """Debug function to check data files and their contents."""
    st.sidebar.subheader("🔧 Debug Information")
    
    if st.sidebar.button("Check Data Files"):
        st.sidebar.write("**File Check Results:**")
        
        # Check a few sample files
        sample_years = [2001, 2010, 2020]
        for year in sample_years:
            lc_path = Path(LANDCOVER_TEMPLATE.format(year=year))
            bd_path = Path(BURNDATE_TEMPLATE.format(year=year))
            
            st.sidebar.write(f"**Year {year}:**")
            st.sidebar.write(f"LC exists: {lc_path.exists()}")
            st.sidebar.write(f"BD exists: {bd_path.exists()}")
            
            if lc_path.exists():
                try:
                    with rasterio.open(lc_path) as src:
                        data = src.read(1)
                        unique_vals = np.unique(data)
                        non_zero = np.sum(data > 0)
                        st.sidebar.write(f"LC non-zero pixels: {non_zero}")
                        st.sidebar.write(f"LC unique values: {len(unique_vals)}")
                        
                        # Check if we have expected land cover values
                        expected_values = list(RECLASS_MAP.keys())
                        found_values = [v for v in unique_vals if v in expected_values]
                        st.sidebar.write(f"Expected LC values found: {len(found_values)}")
                        
                except Exception as e:
                    st.sidebar.write(f"Error reading LC: {e}")
            
            if bd_path.exists():
                try:
                    with rasterio.open(bd_path) as src:
                        data = src.read(1)
                        burn_pixels = np.sum(data > 0)
                        st.sidebar.write(f"BD burn pixels: {burn_pixels}")
                except Exception as e:
                    st.sidebar.write(f"Error reading BD: {e}")
            
            st.sidebar.write("---")

# ----------------------- Streamlit App Layout -----------------------

def main():
    st.title("🔥 Algeria Wildfire Analysis (2001-2020)")
    
    # Check if data directories exist
    if not BASE_DIR.exists():
        st.error(f"Data directory not found: {BASE_DIR}")
        return
    
    if not PROVINCES_PATH.exists():
        st.error(f"Provinces shapefile not found: {PROVINCES_PATH}")
        return
    
    # Load global data
    with st.spinner("Loading global data..."):
        landcover_df = compute_global_landcover()
        burndate_df = compute_global_burndates()
        provinces = load_provinces()
    
    if provinces.empty:
        st.error("Could not load provinces data.")
        return
    
    # Initialize session state for clicked province
    if 'clicked_province_idx' not in st.session_state:
        st.session_state.clicked_province_idx = None
    
    # Display province information
    st.info(f"📍 Loaded {len(provinces)} provinces from ADM1 boundary data")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Add debug function
    debug_data_files()
    
    # Year selection
    year_options = ['total'] + list(range(START_YEAR, END_YEAR+1))
    year_labels = ['Total (2001-2020)'] + [str(y) for y in range(START_YEAR, END_YEAR+1)]
    selected_year = st.sidebar.selectbox(
        "Select Year:",
        options=year_options,
        format_func=lambda x: year_labels[year_options.index(x)],
        index=0
    )
    
    # Province selection
    province_options = ['All Provinces'] + list(provinces['ADM1_EN'].values)
    selected_province = st.sidebar.selectbox(
        "Select Province:",
        options=province_options,
        index=0
    )
    
    # Update clicked province index based on dropdown selection
    if selected_province != 'All Provinces':
        prov_idx = provinces[provinces['ADM1_EN'] == selected_province].index[0]
        st.session_state.clicked_province_idx = prov_idx
    else:
        st.session_state.clicked_province_idx = None
    
    # Landcover category selection
    st.sidebar.subheader("Landcover Categories")
    selected_categories = []
    for category in CATEGORIES:
        if st.sidebar.checkbox(category, value=True):
            selected_categories.append(category)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Geographic Distribution")
        with st.spinner("Generating map..."):
            fig_map, map_title = generate_map(selected_year)
            if fig_map:
                # Add click instructions
                st.info("💡 Use the dropdown in the sidebar to select a province for detailed analysis")
                
                # Display the map
                st.plotly_chart(fig_map, use_container_width=True, key="map")
            else:
                st.error(map_title)
    
    with col2:
        st.subheader("Statistics")
        
        # Determine which statistics to show
        if st.session_state.clicked_province_idx is not None:
            # Show clicked province statistics
            prov_stats = get_province_statistics(st.session_state.clicked_province_idx, selected_year)
            if prov_stats:
                st.markdown(f"**📍 {prov_stats['province_name']}**")
                
                if selected_year == 'total':
                    st.metric("Total Burned Area", f"{prov_stats['total_burned_area']:.1f} km²")
                    st.metric("Years with Data", f"{prov_stats['years_with_data']}")
                else:
                    st.metric(f"Total Burned Area ({selected_year})", f"{prov_stats['total_burned_area']:.1f} km²")
                
                # Show breakdown by category
                st.metric("Forest", f"{prov_stats['forest_area']:.1f} km²")
                st.metric("Cropland", f"{prov_stats['cropland_area']:.1f} km²")
                st.metric("Shrubland", f"{prov_stats['shrubland_area']:.1f} km²")
                
                # Button to clear selection
                if st.button("🌍 Show Global Statistics"):
                    st.session_state.clicked_province_idx = None
                    st.rerun()
            else:
                st.info("No data available for selected province")
        else:
            # Show global statistics
            st.markdown("**🌍 Global Statistics**")
            if selected_year == 'total':
                if not landcover_df.empty:
                    total_area = landcover_df[CATEGORIES].sum().sum()
                    st.metric("Total Burned Area", f"{total_area:.1f} km²")
                    
                    # Show breakdown by category
                    for cat in CATEGORIES:
                        cat_total = landcover_df[cat].sum()
                        st.metric(f"{cat}", f"{cat_total:.1f} km²")
                else:
                    st.info("No landcover data available")
            else:
                if not landcover_df.empty:
                    year_data = landcover_df[landcover_df['Year'] == selected_year]
                    if not year_data.empty:
                        total_area = year_data[CATEGORIES].sum().sum()
                        st.metric(f"Total Burned Area ({selected_year})", f"{total_area:.1f} km²")
                        
                        for cat in CATEGORIES:
                            cat_area = year_data[cat].iloc[0] if not year_data[cat].empty else 0
                            st.metric(f"{cat}", f"{cat_area:.1f} km²")
                    else:
                        st.info(f"No data available for {selected_year}")
                else:
                    st.info("No landcover data available")
    
    # Charts section
    st.subheader("Detailed Analysis")
    
    # Determine which province data to use
    if st.session_state.clicked_province_idx is not None or selected_province != 'All Provinces':
        if st.session_state.clicked_province_idx is not None:
            prov_idx = st.session_state.clicked_province_idx
            province_name = provinces.iloc[prov_idx]['ADM1_EN']
        else:
            prov_idx = provinces[provinces['ADM1_EN'] == selected_province].index[0]
            province_name = selected_province
        
        with st.spinner(f"Processing data for {province_name}..."):
            df_lc = process_province_landcover(prov_idx)
            df_bd = process_province_burndates(prov_idx)
        chart_suffix = f" ({province_name})"
    else:
        df_lc = landcover_df.copy()
        df_bd = burndate_df.copy()
        chart_suffix = " (All Provinces)"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Landcover Burned Area{chart_suffix}")
        if not df_lc.empty and selected_categories:
            dfm = df_lc.melt(id_vars='Year', value_vars=CATEGORIES,
                             var_name='Category', value_name='Burned Area')
            dfm = dfm[dfm['Category'].isin(selected_categories)]
            fig_lc = px.bar(dfm, x='Year', y='Burned Area', color='Category', 
                           barmode='group', title=f'Landcover Burned Area{chart_suffix}')
            fig_lc.update_layout(xaxis_title="Year", yaxis_title="Burned Area (km²)")
            st.plotly_chart(fig_lc, use_container_width=True)
        else:
            st.info("No landcover data available or no categories selected.")
    
    with col2:
        st.subheader(f"Daily Burn Frequency{chart_suffix}")
        if not df_bd.empty:
            cnt = df_bd.groupby('day_of_year').size().reset_index(name='count')
            fig_bd = go.Figure(go.Bar(x=cnt['day_of_year'], y=cnt['count']))
            fig_bd.update_layout(
                title=f'Daily Burn Frequency{chart_suffix}',
                xaxis_title='Day of Year', 
                yaxis_title='Pixel Count'
            )
            st.plotly_chart(fig_bd, use_container_width=True)
        else:
            st.info("No burn date data available.")
    
    # Provincial summary table
    if not provinces.empty:
        st.subheader("📋 Provincial Summary")
        with st.expander("View all provinces"):
            # Create a summary table with statistics
            province_summary = []
            for idx, row in provinces.iterrows():
                prov_stats = get_province_statistics(idx, selected_year)
                province_summary.append({
                    'Province': row['ADM1_EN'],
                    'Total Burned Area (km²)': f"{prov_stats.get('total_burned_area', 0):.1f}",
                    'Forest (km²)': f"{prov_stats.get('forest_area', 0):.1f}",
                    'Cropland (km²)': f"{prov_stats.get('cropland_area', 0):.1f}",
                    'Shrubland (km²)': f"{prov_stats.get('shrubland_area', 0):.1f}"
                })
            
            summary_df = pd.DataFrame(province_summary)
            st.dataframe(summary_df, use_container_width=True)
    
    # Additional information
    with st.expander("ℹ️ About this Dashboard"):
        st.markdown(f"""
        This dashboard analyzes wildfire data for Algeria from 2001 to 2020 using:
        - **Landcover data**: Categorized into Forest, Cropland, and Shrubland
        - **Burn date data**: Shows when fires occurred during each year
        - **Spatial analysis**: Province-level breakdown using ADM1 boundaries
        - **Administrative level**: {len(provinces)} provinces loaded
        
        **How to use:**
        1. Select a year or view total data across all years
        2. Choose a specific province from the dropdown in the sidebar
        3. Filter landcover categories of interest
        4. View province-specific statistics and charts
        5. Explore the interactive visualizations
        
        **Data sources**: Satellite imagery processed to 95% confidence level  
        **Spatial resolution**: 250m pixels  
        **Administrative boundaries**: ADM1 (Province level)
        
        **Features:**
        - Select provinces from the dropdown to see specific statistics
        - Use the map to visualize burned area distribution
        - Statistics update automatically based on your selection
        - Global view available when "All Provinces" is selected
        - Detailed charts show temporal patterns and land cover breakdown
        """)

if __name__ == '__main__':
    main()
