#!/usr/bin/env python3
"""
Algeria Wildfire Analysis — Streamlit App (2001–2020)
------------------------------------------------------
Rewritten to use pre-aggregated Parquet files instead of raw GeoTIFF rasters.
Dashboard startup: < 1 second (previously: minutes).

Data source: MODIS MCD64A1 (burned area) + MCD12Q1 (land cover) via Google Earth Engine
Spatial resolution: ADM1 (wilaya) and ADM2 (commune) level

Author: Z.Matougui
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Algeria Wildfire Analysis",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")

BURN_COLS   = ["burned_forest_km2", "burned_shrubland_km2",
               "burned_cropland_km2", "burned_other_km2", "burned_total_km2"]
BURN_LABELS = {
    "burned_forest_km2":    "Forest",
    "burned_shrubland_km2": "Shrubland",
    "burned_cropland_km2":  "Cropland",
    "burned_other_km2":     "Other",
}
BURN_TYPE_COLS = list(BURN_LABELS.keys())   # excludes burned_total_km2

LC_COLS   = ["forest_km2", "shrubland_km2", "cropland_km2", "other_km2", "total_km2"]
LC_LABELS = {
    "forest_km2":    "Forest",
    "shrubland_km2": "Shrubland",
    "cropland_km2":  "Cropland",
    "other_km2":     "Other",
}
LC_TYPE_COLS = list(LC_LABELS.keys())   # excludes total_km2

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May",  6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

SEASON_ORDER   = ["Spring", "Summer", "Autumn", "Winter"]
SEASON_COLOURS = {
    "Spring": "#52B788",
    "Summer": "#E63946",
    "Autumn": "#F4A261",
    "Winter": "#90E0EF",
}
COVER_COLOURS = {
    "Forest":    "#2D6A4F",
    "Shrubland": "#95D5B2",
    "Cropland":  "#F4A261",
    "Other":     "#ADB5BD",
}


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    b1  = pd.read_parquet(DATA_DIR / "burned_area_adm1.parquet")
    b2  = pd.read_parquet(DATA_DIR / "burned_area_adm2.parquet")
    lc1 = pd.read_parquet(DATA_DIR / "landcover_adm1.parquet")
    lc2 = pd.read_parquet(DATA_DIR / "landcover_adm2.parquet")
    with open(DATA_DIR / "admin_hierarchy.json", encoding="utf-8") as f:
        hierarchy = json.load(f)
    return b1, b2, lc1, lc2, hierarchy


@st.cache_data(show_spinner=False)
def load_provinces() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(DATA_DIR / "Dz_adm1.shp")
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf[["ADM1_EN", "geometry"]].copy()


# ── Filtering helpers ─────────────────────────────────────────────────────────
def filter_burn(
    b1: pd.DataFrame, b2: pd.DataFrame,
    wilaya: str, commune: str,
    yr_min: int, yr_max: int,
) -> pd.DataFrame:
    """Return burned-area DataFrame filtered to the selected scope and year range."""
    grp_cols_adm2 = ["ADM1_CODE", "ADM1_NAME", "ADM2_CODE", "ADM2_NAME",
                     "year", "month", "season"]
    grp_cols_adm1 = ["ADM1_CODE", "ADM1_NAME", "year", "month", "season"]
    grp_cols_nat  = ["year", "month", "season"]

    if commune != "All":
        df = b2[b2["ADM2_NAME"] == commune].copy()
    elif wilaya != "All Wilayas":
        df = (
            b2[b2["ADM1_NAME"] == wilaya]
            .groupby(grp_cols_adm1, observed=True)[BURN_COLS]
            .sum().reset_index()
        )
    else:
        df = (
            b1.groupby(grp_cols_nat, observed=True)[BURN_COLS]
            .sum().reset_index()
        )

    return df[(df["year"] >= yr_min) & (df["year"] <= yr_max)].copy()


def filter_lc(
    lc1: pd.DataFrame, lc2: pd.DataFrame,
    wilaya: str, commune: str,
    yr_min: int, yr_max: int,
) -> pd.DataFrame:
    """Return land-cover DataFrame filtered to the selected scope and year range."""
    if commune != "All":
        df = lc2[lc2["ADM2_NAME"] == commune].copy()
    elif wilaya != "All Wilayas":
        df = (
            lc2[lc2["ADM1_NAME"] == wilaya]
            .groupby(["ADM1_CODE", "ADM1_NAME", "year"], observed=True)[LC_COLS]
            .sum().reset_index()
        )
    else:
        df = lc1.groupby("year", observed=True)[LC_COLS].sum().reset_index()

    return df[(df["year"] >= yr_min) & (df["year"] <= yr_max)].copy()


# ── Chart helpers ─────────────────────────────────────────────────────────────
def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, x=0.5, y=0.5,
                       xref="paper", yref="paper", font_size=14)
    fig.update_layout(template="plotly_white")
    return fig


def chart_annual_bar(df: pd.DataFrame, suffix: str) -> go.Figure:
    annual = df.groupby("year")["burned_total_km2"].sum().reset_index()
    if annual.empty:
        return _empty_fig("No data")
    fig = px.bar(
        annual, x="year", y="burned_total_km2",
        title=f"Annual Burned Area{suffix}",
        labels={"year": "Year", "burned_total_km2": "Burned area (km²)"},
        template="plotly_white",
        color_discrete_sequence=["#E25822"],
    )
    fig.update_layout(showlegend=False, xaxis=dict(dtick=2))
    return fig


def chart_trend_line(df: pd.DataFrame, suffix: str) -> go.Figure:
    annual = df.groupby("year")["burned_total_km2"].sum().reset_index()
    if annual.empty:
        return _empty_fig("No data")
    fig = px.line(
        annual, x="year", y="burned_total_km2",
        title=f"Burned Area Trend{suffix}",
        labels={"year": "Year", "burned_total_km2": "Burned area (km²)"},
        template="plotly_white", markers=True,
    )
    fig.update_traces(line_color="#B5000A", marker_color="#E25822")
    fig.update_layout(xaxis=dict(dtick=2))
    return fig


def chart_monthly(df: pd.DataFrame, suffix: str) -> go.Figure:
    monthly = (
        df.groupby("month")["burned_total_km2"].sum()
        .reindex(range(1, 13), fill_value=0).reset_index()
    )
    monthly["month_name"] = monthly["month"].map(MONTH_NAMES)
    if monthly.empty:
        return _empty_fig("No data")
    fig = px.bar(
        monthly, x="month_name", y="burned_total_km2",
        title=f"Monthly Fire Seasonality{suffix}",
        labels={"month_name": "Month", "burned_total_km2": "Burned area (km²)"},
        template="plotly_white",
        color_discrete_sequence=["#FF6B35"],
    )
    fig.update_layout(showlegend=False)
    return fig


def chart_seasonal(df: pd.DataFrame, suffix: str) -> go.Figure:
    seasonal = (
        df.groupby("season", observed=True)["burned_total_km2"].sum()
        .reindex(SEASON_ORDER).fillna(0).reset_index()
    )
    fig = px.bar(
        seasonal, x="season", y="burned_total_km2",
        title=f"Fire Activity by Season{suffix}",
        labels={"season": "Season", "burned_total_km2": "Burned area (km²)"},
        template="plotly_white",
        color="season",
        color_discrete_map=SEASON_COLOURS,
    )
    fig.update_layout(showlegend=False)
    return fig


def chart_burn_by_type(df: pd.DataFrame, categories: List[str], suffix: str) -> go.Figure:
    cols = [c for c in BURN_TYPE_COLS if BURN_LABELS[c] in categories]
    if not cols:
        return _empty_fig("Select at least one land-cover type")
    annual = df.groupby("year")[cols].sum().reset_index()
    melted = annual.melt("year", value_vars=cols,
                         var_name="_col", value_name="Burned area (km²)")
    melted["Land cover"] = melted["_col"].map(BURN_LABELS)
    fig = px.bar(
        melted, x="year", y="Burned area (km²)", color="Land cover",
        barmode="stack",
        title=f"Burned Area by Cover Type{suffix}",
        template="plotly_white",
        color_discrete_map=COVER_COLOURS,
    )
    fig.update_layout(xaxis=dict(dtick=2))
    return fig


def chart_lc_composition(df: pd.DataFrame, categories: List[str], suffix: str) -> go.Figure:
    cols = [c for c in LC_TYPE_COLS if LC_LABELS[c] in categories]
    if df.empty or not cols:
        return _empty_fig("No land-cover data")
    annual = df.groupby("year")[cols].sum().reset_index()
    melted = annual.melt("year", value_vars=cols,
                         var_name="_col", value_name="Area (km²)")
    melted["Land cover"] = melted["_col"].map(LC_LABELS)
    fig = px.bar(
        melted, x="year", y="Area (km²)", color="Land cover",
        barmode="stack",
        title=f"Land Cover Composition{suffix}",
        template="plotly_white",
        color_discrete_map=COVER_COLOURS,
    )
    fig.update_layout(xaxis=dict(dtick=2))
    return fig


def build_choropleth(
    provinces: gpd.GeoDataFrame, b1: pd.DataFrame,
    yr_min: int, yr_max: int,
) -> go.Figure:
    """Choropleth of total burned area per wilaya over the selected year range."""
    totals = (
        b1[(b1["year"] >= yr_min) & (b1["year"] <= yr_max)]
        .groupby("ADM1_NAME", observed=True)["burned_total_km2"].sum()
        .reset_index()
    )
    gdf = provinces.merge(totals, left_on="ADM1_EN", right_on="ADM1_NAME", how="left")
    gdf["burned_total_km2"] = gdf["burned_total_km2"].fillna(0)

    fig = px.choropleth_mapbox(
        gdf,
        geojson=gdf.__geo_interface__,
        locations="ADM1_EN",
        featureidkey="properties.ADM1_EN",
        color="burned_total_km2",
        hover_name="ADM1_EN",
        hover_data={"burned_total_km2": ":.1f"},
        mapbox_style="carto-positron",
        opacity=0.75,
        color_continuous_scale="YlOrRd",
        labels={"burned_total_km2": "Burned (km²)"},
        title=f"Total Burned Area by Wilaya ({yr_min}–{yr_max})",
    )
    fig.update_layout(
        mapbox=dict(center=dict(lat=28.0, lon=2.5), zoom=3.8),
        margin=dict(r=0, t=40, l=0, b=0),
        coloraxis_colorbar=dict(title="km²", len=0.6),
    )
    return fig


# ── Main app ──────────────────────────────────────────────────────────────────
def main() -> None:
    st.title("🔥 Algeria Wildfire Analysis (2001–2020)")
    st.caption(
        "**Author:** Z.Matougui · "
        "**Data:** MODIS MCD64A1 (burned area) + MCD12Q1 (land cover) via Google Earth Engine"
    )
    st.markdown("---")

    # Load all data (instant — Parquet cached on first run)
    with st.spinner("Loading data…"):
        b1, b2, lc1, lc2, hierarchy = load_data()
        provinces = load_provinces()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🎛️ Controls")

        yr_min, yr_max = st.slider(
            "📅 Year range", 2001, 2020, (2001, 2020), key="year_range"
        )

        wilaya_names = ["All Wilayas"] + [w["name"] for w in hierarchy]
        selected_wilaya = st.selectbox(
            "🗺️ Wilaya", wilaya_names, index=0, key="wilaya_sel"
        )

        # Commune list is dynamic based on selected wilaya
        if selected_wilaya != "All Wilayas":
            w_entry = next((w for w in hierarchy if w["name"] == selected_wilaya), None)
            commune_options = ["All"] + ([c["name"] for c in w_entry["communes"]] if w_entry else [])
        else:
            commune_options = ["All"]

        selected_commune = st.selectbox(
            "🏘️ Commune",
            commune_options,
            index=0,
            key="commune_sel",
            disabled=(selected_wilaya == "All Wilayas"),
        )

        categories = st.multiselect(
            "🌳 Land-cover types",
            ["Forest", "Shrubland", "Cropland", "Other"],
            default=["Forest", "Shrubland", "Cropland"],
            key="lc_cats",
        )

        st.markdown("---")

    # ── Filter ────────────────────────────────────────────────────────────────
    df_burn = filter_burn(b1, b2, selected_wilaya, selected_commune, yr_min, yr_max)
    df_lc   = filter_lc(lc1, lc2, selected_wilaya, selected_commune, yr_min, yr_max)

    if selected_commune != "All":
        scope_label = f"{selected_commune} ({selected_wilaya})"
    elif selected_wilaya != "All Wilayas":
        scope_label = selected_wilaya
    else:
        scope_label = "Algeria"
    title_suffix = f" — {scope_label}"

    # ── KPI row ───────────────────────────────────────────────────────────────
    total_burned = float(df_burn["burned_total_km2"].sum())
    annual_totals = df_burn.groupby("year")["burned_total_km2"].sum()
    peak_year  = int(annual_totals.idxmax()) if not annual_totals.empty else "—"
    peak_km2   = float(annual_totals.max()) if not annual_totals.empty else 0.0
    monthly_totals = df_burn.groupby("month")["burned_total_km2"].sum()
    peak_month = MONTH_NAMES.get(int(monthly_totals.idxmax()), "—") if not monthly_totals.empty else "—"

    if selected_wilaya == "All Wilayas":
        n_communes = int(b2["ADM2_NAME"].nunique())
    else:
        n_communes = int(b2.loc[b2["ADM1_NAME"] == selected_wilaya, "ADM2_NAME"].nunique())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔥 Total burned area", f"{total_burned:,.0f} km²")
    k2.metric("📅 Peak year", str(peak_year), f"{peak_km2:,.0f} km²")
    k3.metric("📆 Peak fire month", peak_month)
    k4.metric("🏘️ Communes in view", f"{n_communes:,}")

    st.markdown("---")

    # ── Row 1: Map + Annual bar ───────────────────────────────────────────────
    col_map, col_bar = st.columns([1.3, 1])

    with col_map:
        st.subheader("🗺️ Wilaya Burned Area Map")
        fig_map = build_choropleth(provinces, b1, yr_min, yr_max)
        st.plotly_chart(fig_map, use_container_width=True)

    with col_bar:
        st.subheader("📊 Annual Burned Area")
        st.plotly_chart(chart_annual_bar(df_burn, title_suffix), use_container_width=True)
        st.plotly_chart(chart_trend_line(df_burn, title_suffix), use_container_width=True)

    # ── Row 2: Seasonality + Land cover ──────────────────────────────────────
    st.markdown("---")
    col_season, col_cover = st.columns(2)

    with col_season:
        st.subheader("📆 Fire Seasonality")
        st.plotly_chart(chart_monthly(df_burn, title_suffix), use_container_width=True)
        st.plotly_chart(chart_seasonal(df_burn, title_suffix), use_container_width=True)

    with col_cover:
        st.subheader("🌳 Land Cover Analysis")
        st.plotly_chart(chart_burn_by_type(df_burn, categories, title_suffix),
                        use_container_width=True)
        st.plotly_chart(chart_lc_composition(df_lc, categories, title_suffix),
                        use_container_width=True)

    # ── Sidebar exports ───────────────────────────────────────────────────────
    with st.sidebar:
        st.caption("Export")
        st.download_button(
            "⬇️ Burned area CSV",
            df_burn.to_csv(index=False).encode("utf-8"),
            file_name="burned_area_selection.csv",
            mime="text/csv",
        )
        st.download_button(
            "⬇️ Land cover CSV",
            df_lc.to_csv(index=False).encode("utf-8"),
            file_name="landcover_selection.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.caption(
        "Algeria Wildfire Analysis · Built with Streamlit + Plotly · "
        "Data: NASA MODIS via Google Earth Engine"
    )


if __name__ == "__main__":
    main()
