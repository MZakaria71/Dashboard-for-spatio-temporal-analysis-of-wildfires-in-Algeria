#!/usr/bin/env python3
"""
Algeria Wildfire Analysis — Streamlit App (2001–2020)
-----------------------------------------------------

Requirements (Python ≥3.9):
    streamlit, plotly, rasterio, numpy, pandas, geopandas, rasterstats

Author: Z.Matougui
Date: 2025-08-08
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
from rasterstats import zonal_stats

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------- Page Configuration -------------------------
st.set_page_config(
    page_title="Algeria Wildfire Analysis",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------- Constants & Defaults -----------------------
DEFAULT_DATA_DIR = Path("data")
PROVINCES_PATH_DEFAULT = DEFAULT_DATA_DIR / "Dz_adm1.shp"
LANDCOVER_TEMPLATE_DEFAULT = (
    DEFAULT_DATA_DIR / "LandCover_Exports" / "LandCover_Summer_{year}_95Percent_Confidence.tif"
)
BURNDATE_TEMPLATE_DEFAULT = (
    DEFAULT_DATA_DIR / "BurnDate_Exports" / "BurnDate_Summer_{year}_95Percent_Confidence.tif"
)

START_YEAR_DEFAULT = 2001
END_YEAR_DEFAULT = 2020  # inclusive

RECLASS_MAP: Dict[int, str] = {
    10: "Cropland", 20: "Cropland", 30: "Cropland", 40: "Cropland",
    50: "Forest", 60: "Forest", 70: "Forest", 80: "Forest",
    90: "Forest", 100: "Forest", 110: "Forest",
    120: "Shrubland", 130: "Shrubland", 140: "Shrubland", 150: "Shrubland",
    170: "Forest", 180: "Shrubland",
}

LANDCOVER_GROUPS = ["Forest", "Cropland", "Shrubland"]

# ----------------------- Utility Functions -------------------------

def _pixel_area_km2_from_transform(transform: rasterio.Affine) -> float:
    """Compute pixel area in km² using the affine transform (north-up assumption)."""
    return abs(transform.a) * abs(transform.e) / 1e6


def _safe_crs_to_epsg_str(crs) -> str:
    try:
        if crs is None:
            return ""
        return crs.to_string().lower()
    except Exception:
        return ""


def _province_geom_in_raster_crs(geometry, raster_crs) -> gpd.GeoSeries:
    """Reproject a geometry to match the raster's CRS."""
    # Create a GeoSeries with the geometry and EPSG:4326 CRS
    geom_series = gpd.GeoSeries([geometry], crs="EPSG:4326")
    # Reproject to the raster's CRS
    return geom_series.to_crs(raster_crs)


@st.cache_data(show_spinner=False)
def load_provinces(path: Path) -> Optional[gpd.GeoDataFrame]:
    """Load administrative boundaries and ensure EPSG:4326 with an ADM1_EN column."""
    try:
        gdf = gpd.read_file(path)
        if gdf.crs is None or _safe_crs_to_epsg_str(gdf.crs) != "epsg:4326":
            gdf = gdf.to_crs(epsg=4326)
        name_col = None
        for cand in ("ADM1_EN", "NAME_1", "name", "adm1_en"):
            if cand in gdf.columns:
                name_col = cand
                break
        if name_col is None:
            gdf["ADM1_EN"] = [f"Province {i}" for i in range(len(gdf))]
        elif name_col != "ADM1_EN":
            gdf = gdf.rename(columns={name_col: "ADM1_EN"})
        return gdf[["ADM1_EN", "geometry"]].copy()
    except Exception as e:
        st.error(f"Failed to load provinces: {e}")
        return None


# ----------------------- Core Processing ---------------------------
@st.cache_data(show_spinner=False)
def process_landcover_burns(lc_template: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Aggregate national land-cover area (km²) by group per year."""
    records: List[Dict[str, float]] = []
    progress = st.progress(0.0)
    status = st.empty()

    n_years = end_year - start_year + 1
    for i, year in enumerate(range(start_year, end_year + 1)):
        status.text(f"Reading land cover for {year}…")
        path = Path(lc_template.format(year=year))
        if not path.exists():
            st.warning(f"File not found: {path}")
            records.append({"Year": year, "Forest": 0.0, "Cropland": 0.0, "Shrubland": 0.0})
            continue
        try:
            with rasterio.open(path) as src:
                data = src.read(1)
                valid = data != 0
                area_km2 = _pixel_area_km2_from_transform(src.transform)
            counts = np.bincount(data[valid].ravel(), minlength=max(RECLASS_MAP.keys()) + 1)
            rec = {"Year": year, "Forest": 0.0, "Cropland": 0.0, "Shrubland": 0.0}
            for code, group in RECLASS_MAP.items():
                if code < len(counts):
                    rec[group] += counts[code] * area_km2
            records.append(rec)
        except Exception as e:
            st.error(f"Error processing {path}: {e}")
            records.append({"Year": year, "Forest": 0.0, "Cropland": 0.0, "Shrubland": 0.0})
        progress.progress((i + 1) / n_years)

    progress.empty(); status.empty()
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def process_burndates(bd_template: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Collect burn dates nationally across years (pixel DOY values > 0)."""
    frames: List[pd.DataFrame] = []
    progress = st.progress(0.0)
    status = st.empty()

    n_years = end_year - start_year + 1
    for i, year in enumerate(range(start_year, end_year + 1)):
        status.text(f"Reading burn dates for {year}…")
        path = Path(bd_template.format(year=year))
        if not path.exists():
            st.warning(f"File not found: {path}")
            continue
        try:
            with rasterio.open(path) as src:
                data = src.read(1)
                valid = data > 0
                area_km2 = _pixel_area_km2_from_transform(src.transform)
            if np.any(valid):
                df = pd.DataFrame({"day_of_year": data[valid].ravel()})
                df["year"] = year
                df["pixel_area_km2"] = area_km2
                frames.append(df)
        except Exception as e:
            st.error(f"Error processing {path}: {e}")
        progress.progress((i + 1) / n_years)

    progress.empty(); status.empty()
    if not frames:
        return pd.DataFrame(columns=["day_of_year", "year", "pixel_area_km2"])
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def get_province_landcover(_geom, lc_template: str, start_year: int, end_year: int) -> pd.DataFrame:
    geom = _geom
    recs: List[Dict[str, float]] = []
    for year in range(start_year, end_year + 1):
        path = Path(lc_template.format(year=year))
        if not path.exists():
            recs.append({"Year": year, "Forest": 0.0, "Cropland": 0.0, "Shrubland": 0.0})
            continue
        try:
            with rasterio.open(path) as src:
                geom_proj = _province_geom_in_raster_crs(geom, src.crs).iloc[0]
                try:
                    out, _ = rasterio.mask.mask(src, [geom_proj], crop=True)
                except Exception:
                    recs.append({"Year": year, "Forest": 0.0, "Cropland": 0.0, "Shrubland": 0.0})
                    continue
                data = out[0]
                valid = data != 0
                area_km2 = _pixel_area_km2_from_transform(src.transform)
            counts = np.bincount(data[valid].ravel(), minlength=max(RECLASS_MAP.keys()) + 1) if np.any(valid) else np.zeros(max(RECLASS_MAP.keys()) + 1, dtype=int)
            rec = {"Year": year, "Forest": 0.0, "Cropland": 0.0, "Shrubland": 0.0}
            for code, group in RECLASS_MAP.items():
                if code < len(counts):
                    rec[group] += counts[code] * area_km2
            recs.append(rec)
        except Exception as e:
            st.error(f"Error processing province data for {year}: {e}")
            recs.append({"Year": year, "Forest": 0.0, "Cropland": 0.0, "Shrubland": 0.0})
    return pd.DataFrame(recs)


@st.cache_data(show_spinner=False)
def get_province_burndates(_geom, bd_template: str, start_year: int, end_year: int) -> pd.DataFrame:
    geom = _geom
    frames: List[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        path = Path(bd_template.format(year=year))
        if not path.exists():
            continue
        try:
            with rasterio.open(path) as src:
                geom_proj = _province_geom_in_raster_crs(geom, src.crs).iloc[0]
                try:
                    out, _ = rasterio.mask.mask(src, [geom_proj], crop=True)
                except Exception:
                    continue
                data = out[0]
                valid = data > 0
                area_km2 = _pixel_area_km2_from_transform(src.transform)
            if np.any(valid):
                df = pd.DataFrame({"day_of_year": data[valid].ravel()})
                df["year"] = year
                df["pixel_area_km2"] = area_km2
                frames.append(df)
        except Exception as e:
            st.error(f"Error processing burn dates for province in {year}: {e}")
    if not frames:
        return pd.DataFrame(columns=["day_of_year", "year", "pixel_area_km2"])
    return pd.concat(frames, ignore_index=True)


# ----------------------- Visualization Helpers ---------------------

def create_map(provinces: gpd.GeoDataFrame, selected_year: int | str, bd_template: str, start_year: int, end_year: int) -> Tuple[go.Figure, str]:
    """Choropleth of burned area (km²) by province for a year or total."""
    if selected_year == "Total":
        province_sums = np.zeros(len(provinces), dtype=float)
        for year in range(start_year, end_year + 1):
            path = Path(bd_template.format(year=year))
            if not path.exists():
                continue
            try:
                with rasterio.open(path) as src:
                    affine = src.transform
                    area_km2 = _pixel_area_km2_from_transform(affine)
                    provinces_reproj = provinces.to_crs(src.crs)
                    burned_mask = src.read(1) > 0
                stats = zonal_stats(
                    provinces_reproj, burned_mask, affine=affine,
                    stats=["sum"], nodata=0, all_touched=True, geojson_out=True,
                )
                for i, stat in enumerate(stats):
                    s = stat["properties"].get("sum", 0) or 0
                    province_sums[i] += s * area_km2
            except Exception as e:
                st.error(f"Error processing map data for {year}: {e}")
                continue
        gdf_stats = provinces.copy()
        gdf_stats["burned_area_km2"] = province_sums
        title = f"Total Burned Area ({start_year}–{end_year})"
    else:
        path = Path(bd_template.format(year=selected_year))
        if not path.exists():
            return go.Figure(), "Data not available for selected year."
        try:
            with rasterio.open(path) as src:
                affine = src.transform
                area_km2 = _pixel_area_km2_from_transform(affine)
                provinces_reproj = provinces.to_crs(src.crs)
                burned_mask = src.read(1) > 0
            stats = zonal_stats(
                provinces_reproj, burned_mask, affine=affine,
                stats=["sum"], nodata=0, all_touched=True, geojson_out=True,
            )
            gdf_stats = gpd.GeoDataFrame.from_features(stats)
            gdf_stats.crs = provinces_reproj.crs
            gdf_stats = gdf_stats.to_crs(provinces.crs)
            gdf_stats["burned_area_km2"] = gdf_stats["sum"].fillna(0) * area_km2
            gdf_stats["ADM1_EN"] = provinces["ADM1_EN"].values
            title = f"Burned Area in {selected_year}"
        except Exception as e:
            st.error(f"Error creating map: {e}")
            return go.Figure(), "Error creating map visualization."

    if _safe_crs_to_epsg_str(gdf_stats.crs) != "epsg:4326":
        gdf_stats = gdf_stats.to_crs(epsg=4326)

    fig = px.choropleth_mapbox(
        gdf_stats,
        geojson=gdf_stats.__geo_interface__,
        locations="ADM1_EN",
        featureidkey="properties.ADM1_EN",
        color="burned_area_km2",
        hover_name="ADM1_EN",
        hover_data={"burned_area_km2": ":.2f"},
        mapbox_style="carto-positron",
        opacity=0.75,
        color_continuous_scale="YlOrRd",
        labels={"burned_area_km2": "Burned area (km²)"},
        title=title,
    )
    fig.update_layout(mapbox=dict(center=dict(lat=32.0, lon=2.5), zoom=4.2), margin=dict(r=0, t=40, l=0, b=0))
    return fig, f"Map displaying {title}"


def create_landcover_chart(df: pd.DataFrame, categories: List[str], suffix: str) -> go.Figure:
    if df.empty:
        fig = go.Figure(); fig.update_layout(title=f"No land-cover data{suffix}")
    else:
        melted = df.melt("Year", value_vars=[c for c in LANDCOVER_GROUPS if c in df.columns], var_name="Category", value_name="Area (km²)")
        melted = melted[melted["Category"].isin(categories)]
        fig = px.bar(melted, x="Year", y="Area (km²)", color="Category", barmode="group", template="plotly_white", title=f"Land-Cover Area{suffix}")
    return fig


def create_burn_pattern_chart(df_bd: pd.DataFrame, suffix: str) -> go.Figure:
    if df_bd.empty:
        fig = go.Figure(); fig.update_layout(title=f"No burn-date data{suffix}")
        return fig
    daily = df_bd.groupby("day_of_year", as_index=False).size()
    fig = go.Figure()
    fig.add_bar(x=daily["day_of_year"], y=daily["size"], name="Burn frequency")
    fig.update_layout(title=f"Daily Burn Frequency{suffix}", xaxis_title="Day of year", yaxis_title="Burned pixels (count)", template="plotly_white", showlegend=False)
    return fig


def create_monthly_seasonality(df_bd: pd.DataFrame, suffix: str) -> go.Figure:
    if df_bd.empty:
        return go.Figure()
    doy = df_bd["day_of_year"].astype(int).clip(lower=1, upper=366)
    bins = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 366]
    labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month = pd.cut(doy, bins=bins, labels=labels, include_lowest=True, right=True)
    m = month.value_counts().sort_index().reset_index(); m.columns = ["Month","count"]
    return px.bar(m, x="Month", y="count", title=f"Monthly Seasonality{suffix}", template="plotly_white")


def create_cumulative_curve(df_bd: pd.DataFrame, suffix: str) -> go.Figure:
    if df_bd.empty:
        return go.Figure()
    daily = df_bd.groupby("day_of_year", as_index=False).size().sort_values("day_of_year")
    daily["cum"] = daily["size"].cumsum()
    return px.line(daily, x="day_of_year", y="cum", title=f"Cumulative Burned-Pixel Count{suffix}", template="plotly_white", labels={"day_of_year":"Day of year","cum":"Cumulative count"})


# ----------------------- Main App ----------------------------------

def main() -> None:
    st.title("🔥 Algeria Wildfire Analysis (2001–2020)")
    st.markdown("---")

    # Use defaults (no data-path controls in UI)
    provinces_path = PROVINCES_PATH_DEFAULT
    lc_template = str(LANDCOVER_TEMPLATE_DEFAULT)
    bd_template = str(BURNDATE_TEMPLATE_DEFAULT)
    start_year = START_YEAR_DEFAULT
    end_year = END_YEAR_DEFAULT

    # Sidebar controls (dropdown province selection)
    with st.sidebar:
        st.header("🎛️ Controls")
        year_options: List[int | str] = ["Total"] + list(range(start_year, end_year + 1))
        selected_year = st.selectbox("📅 Map year", year_options, index=0, key="map_year")

    provinces = load_provinces(provinces_path)
    if provinces is None or provinces.empty:
        st.error("Failed to load province boundaries. Please check the data directory.")
        st.stop()

    # Province dropdown
    with st.sidebar:
        province_names = ["All Provinces"] + sorted(provinces["ADM1_EN"].tolist())
        selected_province = st.selectbox("🗺️ Province", province_names, index=0, key="province_select")
        categories = st.multiselect("🌳 Land-cover groups", LANDCOVER_GROUPS, default=LANDCOVER_GROUPS, key="lc_groups")

    # Compute / load national aggregates
    with st.spinner("Processing national aggregates…"):
        lc_df = process_landcover_burns(lc_template, start_year, end_year)
        bd_df = process_burndates(bd_template, start_year, end_year)

    # KPI cards
    kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    if not bd_df.empty:
        total_area_km2_mean = float((bd_df["pixel_area_km2"].fillna(bd_df["pixel_area_km2"].median())).mean())
        approx_total_burned_km2 = len(bd_df) * total_area_km2_mean
        by_year = bd_df.groupby("year").size()
        peak_year = int(by_year.idxmax())
        peak_count = int(by_year.max())
    else:
        approx_total_burned_km2 = 0.0
        peak_year = start_year
        peak_count = 0
    with kcol1:
        st.metric("Approx. burned area (km²)", f"{approx_total_burned_km2:,.0f}")
    with kcol2:
        st.metric("Peak burn year", f"{peak_year}")
    with kcol3:
        st.metric("Pixels (records)", f"{len(bd_df):,}")
    with kcol4:
        st.metric("Peak pixels in a year", f"{peak_count:,}")

    # Map
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🗺️ Provincial Burned Area Map")
        with st.spinner("Building map…"):
            fig_map, map_status = create_map(provinces, selected_year, bd_template, start_year, end_year)
        st.plotly_chart(fig_map, use_container_width=True)
        st.info(map_status)

    # Land-cover analysis
    with col2:
        st.subheader("📊 Land-Cover Analysis")
        if selected_province == "All Provinces":
            lc_sel = lc_df
            suffix = " (All Provinces)"
        else:
            with st.spinner(f"Computing land-cover for {selected_province}…"):
                geom = provinces.loc[provinces["ADM1_EN"] == selected_province, "geometry"].iloc[0]
                lc_sel = get_province_landcover(geom, lc_template, start_year, end_year)
            suffix = f" ({selected_province})"
        fig_lc = create_landcover_chart(lc_sel, categories, suffix)
        st.plotly_chart(fig_lc, use_container_width=True)

    # Burn patterns
    st.subheader("📈 Burn Patterns")
    if selected_province == "All Provinces":
        bd_sel = bd_df
        suffix = " (All Provinces)"
    else:
        with st.spinner(f"Extracting burn dates for {selected_province}…"):
            geom = provinces.loc[provinces["ADM1_EN"] == selected_province, "geometry"].iloc[0]
            bd_sel = get_province_burndates(geom, bd_template, start_year, end_year)
        suffix = f" ({selected_province})"

    st.plotly_chart(create_burn_pattern_chart(bd_sel, suffix), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(create_monthly_seasonality(bd_sel, suffix), use_container_width=True)
    with c2:
        st.plotly_chart(create_cumulative_curve(bd_sel, suffix), use_container_width=True)

    # Export buttons
    with st.sidebar:
        st.markdown("---")
        st.caption("Export")
        st.download_button(
            "⬇️ National land-cover CSV",
            lc_df.to_csv(index=False).encode("utf-8"),
            file_name="national_landcover_km2.csv",
            mime="text/csv",
        )
        st.download_button(
            "⬇️ National burn-dates CSV",
            bd_df.to_csv(index=False).encode("utf-8"),
            file_name="national_burndates.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.caption("Algeria Wildfire Analysis — built with Streamlit")


if __name__ == "__main__":
    main()
