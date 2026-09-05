#!/usr/bin/env python3
"""
Algeria Wildfire Analysis — Streamlit App
------------------------------------------------------
The displayed period is derived from the data, not hardcoded: it spans the
union of the burned-area record (MCD64A1) and the ignition record (FIRMS).
Rewritten to use pre-aggregated Parquet files instead of raw GeoTIFF rasters.
Dashboard startup: < 1 second (previously: minutes).

Data source: MODIS MCD64A1 (burned area) + MCD12Q1 (land cover) via Google Earth Engine
Spatial resolution: ADM1 (wilaya) and ADM2 (commune) level

Author: Zakaria Matougui, researcher at Territory Planning Research Centre
        (CRAT), Algeria
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Algeria Wildfire Analysis",
    page_icon="🔥",
    layout="wide",
    # "auto", not "expanded": on a phone the sidebar is a full-width
    # drawer, so forcing it open buries the dashboard behind Controls.
    initial_sidebar_state="auto",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
IGNITION_FILE = DATA_DIR / "ignitions.parquet"
DAILY_FILE = DATA_DIR / "event_daily.parquet"   # per-event growth curves
WEATHER_FILE = DATA_DIR / "fire_weather.parquet"   # ERA5, by fetch_weather.py

# Algeria's fire season. August dominates, but June through September is where
# essentially all of the burned area sits, and a season-window comparison is
# less noisy than a single month.
FIRE_SEASON = (6, 7, 8, 9)
# FAO GAUL 2015 wilaya boundaries, simplified to ~500 m by gee_export.js. Same
# source as the burned-area tables, so the 48 wilayas match the data exactly —
# unlike the old Dz_adm1.shp, which used the post-2019 58-wilaya scheme and left
# ten wilayas permanently unmatched.
PROVINCE_FILE = DATA_DIR / "gaul_adm1.geojson"

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

ALL_COVERS = ["Forest", "Shrubland", "Cropland", "Other"]

# Column holding the burned area for the user's current land-cover selection.
SEL_COL = "burned_sel_km2"

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

NODATA_FILL = "#E9ECEF"
NODATA_LINE = "#ADB5BD"
HIGHLIGHT   = "#1D3557"
PARTIAL_FILL = "#C9CCD1"   # incomplete final year

ALL_WILAYAS = "All Wilayas"
ALL_COMMUNES = "All"

# ── Streamlit compatibility ───────────────────────────────────────────────────
# `use_container_width` is deprecated from Streamlit 1.49 in favour of
# `width="stretch"`. Pick the right keyword at import time so the app runs on
# both old and new releases.
def _streamlit_version() -> Tuple[int, int]:
    parts = st.__version__.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return (0, 0)


STRETCH = ({"width": "stretch"} if _streamlit_version() >= (1, 49)
           else {"use_container_width": True})


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


def _polygonal(geom: dict) -> Optional[dict]:
    """Reduce any GeoJSON geometry to its polygonal parts.

    Simplifying in Earth Engine leaves degenerate slivers behind: 40 of the 48
    wilayas come back as a GeometryCollection mixing the real polygon with stray
    Points and LineStrings, which Plotly will not draw. Keep the polygons.
    """
    kind = geom.get("type")
    if kind in ("Polygon", "MultiPolygon"):
        return geom
    if kind == "GeometryCollection":
        parts: List = []
        for sub in geom.get("geometries", []):
            got = _polygonal(sub)
            if got is None:
                continue
            if got["type"] == "Polygon":
                parts.append(got["coordinates"])
            else:
                parts.extend(got["coordinates"])
        return {"type": "MultiPolygon", "coordinates": parts} if parts else None
    return None


def _coords_bounds(geom: dict) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []

    def walk(node) -> None:
        if node and isinstance(node[0], (int, float)):
            xs.append(node[0])
            ys.append(node[1])
        else:
            for child in node:
                walk(child)

    walk(geom["coordinates"])
    return min(xs), min(ys), max(xs), max(ys)


def _merge_bounds(all_bounds) -> Tuple[float, float, float, float]:
    xs0, ys0, xs1, ys1 = zip(*all_bounds)
    return min(xs0), min(ys0), max(xs1), max(ys1)


@st.cache_data(show_spinner=False)
def load_provinces() -> dict:
    """Wilaya boundaries as plain GeoJSON — no geopandas, no shapefile.

    Returns the cleaned FeatureCollection plus per-wilaya bounding boxes, which
    is everything the map needs.
    """
    with open(PROVINCE_FILE, encoding="utf-8") as f:
        raw = json.load(f)

    features, bounds = [], {}
    for feat in raw["features"]:
        geom = _polygonal(feat["geometry"])
        if geom is None:
            continue
        name = str(feat["properties"]["ADM1_NAME"])
        features.append({
            "type": "Feature",
            "id": name,
            "properties": {"ADM1_NAME": name},
            "geometry": geom,
        })
        bounds[name] = _coords_bounds(geom)

    return {
        "geojson": {"type": "FeatureCollection", "features": features},
        "names": [f["properties"]["ADM1_NAME"] for f in features],
        "bounds": bounds,
        "total_bounds": _merge_bounds(list(bounds.values())),
        "geoms": {f["properties"]["ADM1_NAME"]: f["geometry"] for f in features},
    }


@st.cache_data(show_spinner=False)
def load_ignitions() -> Optional[pd.DataFrame]:
    """FIRMS-derived ignition points, or None if the file has not been built."""
    if not IGNITION_FILE.exists():
        return None
    return pd.read_parquet(IGNITION_FILE)


@st.cache_data(show_spinner=False)
def partial_final_year() -> Optional[Tuple[int, int]]:
    """Detect an incomplete final year in the burned-area record.

    MCD64A1 trails real time by a couple of months, so a fresh export usually
    ends mid-year — but the GEE script emits a row for every month regardless,
    filling absent scenes with zeros. A zero month is therefore ambiguous: it
    can mean "no fires" or "no data".

    The record disambiguates itself. Some months are non-zero in *every*
    complete year (July and August, the core of Algeria's fire season). If one
    of those is empty in the final year, the data stops before it.

    Returns (year, last_covered_month), or None when the final year is complete.
    """
    b1, _, _, _, _ = load_data()
    monthly = (
        b1.groupby(["year", "month"])["burned_total_km2"].sum()
        .unstack(fill_value=0.0)
    )
    if len(monthly.index) < 5:            # too short to calibrate against
        return None

    last = int(monthly.index.max())
    prior = monthly.loc[monthly.index < last]
    always_active = [int(m) for m in monthly.columns if (prior[m] > 0).all()]
    empty_now = [m for m in always_active if monthly.loc[last, m] == 0]
    if not always_active or not empty_now:
        return None
    return last, int(min(empty_now)) - 1


@st.cache_data(show_spinner=False)
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise once per unique frame instead of on every rerun."""
    return df.to_csv(index=False).encode("utf-8")


# ── Land-cover selection helpers ──────────────────────────────────────────────
def selected_burn_cols(categories: Sequence[str]) -> List[str]:
    """Burned-area columns matching the user's land-cover selection."""
    return [c for c in BURN_TYPE_COLS if BURN_LABELS[c] in categories]


def with_selected_total(df: pd.DataFrame, categories: Sequence[str]) -> pd.DataFrame:
    """Attach `SEL_COL`: burned area restricted to the selected cover types.

    Every KPI and chart reads this column, so the land-cover filter applies
    consistently across the whole dashboard instead of only two charts.
    """
    cols = selected_burn_cols(categories)
    out = df.copy()
    out[SEL_COL] = out[cols].sum(axis=1) if cols else 0.0
    return out


def cover_scope_label(categories: Sequence[str]) -> str:
    if not categories:
        return "no cover type selected"
    if set(categories) == set(ALL_COVERS):
        return "all cover types"
    return ", ".join(c for c in ALL_COVERS if c in categories)


# ── Filtering helpers ─────────────────────────────────────────────────────────
def filter_burn(
    b1: pd.DataFrame, b2: pd.DataFrame,
    wilaya: str, commune_code: Optional[int],
    yr_min: int, yr_max: int,
) -> pd.DataFrame:
    """Return burned-area DataFrame filtered to the selected scope and year range.

    Communes are matched on ADM2_CODE, not name: 40 commune names are shared by
    more than one wilaya, so name matching silently merged unrelated communes.
    """
    grp_cols_adm1 = ["ADM1_CODE", "ADM1_NAME", "year", "month", "season"]
    grp_cols_nat  = ["year", "month", "season"]

    if commune_code is not None:
        df = b2[b2["ADM2_CODE"] == commune_code].copy()
    elif wilaya != ALL_WILAYAS:
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
    wilaya: str, commune_code: Optional[int],
    yr_min: int, yr_max: int,
) -> pd.DataFrame:
    """Return land-cover DataFrame filtered to the selected scope and year range."""
    if commune_code is not None:
        df = lc2[lc2["ADM2_CODE"] == commune_code].copy()
    elif wilaya != ALL_WILAYAS:
        df = (
            lc2[lc2["ADM1_NAME"] == wilaya]
            .groupby(["ADM1_CODE", "ADM1_NAME", "year"], observed=True)[LC_COLS]
            .sum().reset_index()
        )
    else:
        df = lc1.groupby("year", observed=True)[LC_COLS].sum().reset_index()

    return df[(df["year"] >= yr_min) & (df["year"] <= yr_max)].copy()


@st.cache_data(show_spinner=False)
def count_communes_with_fire(
    wilaya: str, commune_code: Optional[int],
    yr_min: int, yr_max: int, categories: Tuple[str, ...],
) -> int:
    """Distinct communes that actually recorded fire in the current selection.

    Counted on ADM2_CODE — counting distinct ADM2_NAME undercounts by 45,
    because 40 commune names are reused across wilayas.
    """
    _, b2, _, _, _ = load_data()
    cols = selected_burn_cols(categories)
    if not cols:
        return 0

    df = b2[(b2["year"] >= yr_min) & (b2["year"] <= yr_max)]
    if commune_code is not None:
        df = df[df["ADM2_CODE"] == commune_code]
    elif wilaya != ALL_WILAYAS:
        df = df[df["ADM1_NAME"] == wilaya]

    burned = df[cols].sum(axis=1)
    return int(df.loc[burned > 0, "ADM2_CODE"].nunique())


# ── Chart helpers ─────────────────────────────────────────────────────────────
def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, x=0.5, y=0.5,
                       xref="paper", yref="paper", font_size=14)
    fig.update_layout(template="plotly_white")
    return fig


def chart_annual_bar(
    df: pd.DataFrame, suffix: str, partial_year: Optional[int] = None,
) -> go.Figure:
    annual = df.groupby("year")[SEL_COL].sum().reset_index()
    if annual.empty:
        return _empty_fig("No data")
    fig = px.bar(
        annual, x="year", y=SEL_COL,
        title=f"Annual Burned Area{suffix}",
        labels={"year": "Year", SEL_COL: "Burned area (km²)"},
        template="plotly_white",
        color_discrete_sequence=["#E25822"],
    )
    # Grey the incomplete year so its short bar is not read as a real decline.
    if partial_year is not None and (annual["year"] == partial_year).any():
        fig.update_traces(marker_color=[
            PARTIAL_FILL if y == partial_year else "#E25822"
            for y in annual["year"]
        ])
    fig.update_layout(showlegend=False, xaxis=dict(dtick=2))
    return fig


def chart_trend_line(df: pd.DataFrame, suffix: str) -> go.Figure:
    annual = df.groupby("year")[SEL_COL].sum().reset_index()
    if annual.empty:
        return _empty_fig("No data")
    fig = px.line(
        annual, x="year", y=SEL_COL,
        title=f"Burned Area Trend{suffix}",
        labels={"year": "Year", SEL_COL: "Burned area (km²)"},
        template="plotly_white", markers=True,
    )
    fig.update_traces(line_color="#B5000A", marker_color="#E25822")
    fig.update_layout(xaxis=dict(dtick=2))
    return fig


def chart_monthly(df: pd.DataFrame, suffix: str) -> go.Figure:
    if df.empty:
        return _empty_fig("No data")
    monthly = (
        df.groupby("month")[SEL_COL].sum()
        .reindex(range(1, 13), fill_value=0).reset_index()
    )
    monthly["month_name"] = monthly["month"].map(MONTH_NAMES)
    fig = px.bar(
        monthly, x="month_name", y=SEL_COL,
        title=f"Monthly Fire Seasonality{suffix}",
        labels={"month_name": "Month", SEL_COL: "Burned area (km²)"},
        template="plotly_white",
        color_discrete_sequence=["#FF6B35"],
    )
    fig.update_layout(showlegend=False)
    return fig


def chart_seasonal(df: pd.DataFrame, suffix: str) -> go.Figure:
    if df.empty:
        return _empty_fig("No data")
    seasonal = (
        df.groupby("season", observed=True)[SEL_COL].sum()
        .reindex(SEASON_ORDER).fillna(0).reset_index()
    )
    fig = px.bar(
        seasonal, x="season", y=SEL_COL,
        title=f"Fire Activity by Season{suffix}",
        labels={"season": "Season", SEL_COL: "Burned area (km²)"},
        template="plotly_white",
        color="season",
        color_discrete_map=SEASON_COLOURS,
    )
    fig.update_layout(showlegend=False)
    return fig


def chart_burn_by_type(df: pd.DataFrame, categories: List[str], suffix: str) -> go.Figure:
    cols = selected_burn_cols(categories)
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


# ── Map ───────────────────────────────────────────────────────────────────────
def _view_from_bounds(bounds) -> Tuple[dict, float]:
    """Centre + zoom that frame the given (minx, miny, maxx, maxy) bounds."""
    minx, miny, maxx, maxy = bounds
    span = max(maxx - minx, maxy - miny, 0.05)
    zoom = min(9.0, max(3.5, math.log2(360.0 / span) - 0.3))
    return dict(lat=(miny + maxy) / 2, lon=(minx + maxx) / 2), zoom


def _outline_coords(geom: dict) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Exterior ring coordinates, with None separators between parts."""
    lons: List[Optional[float]] = []
    lats: List[Optional[float]] = []
    polys = (geom["coordinates"] if geom["type"] == "MultiPolygon"
             else [geom["coordinates"]])
    for poly in polys:
        if not poly:
            continue
        for lon, lat in poly[0]:          # exterior ring
            lons.append(lon)
            lats.append(lat)
        lons.append(None)
        lats.append(None)
    return lons, lats


# A wilaya holding only a few km² of the selected cover types yields a
# meaningless rate — one fire divided by almost nothing puts it at the top of
# the map. Below this it is shown as no data instead.
MIN_BURNABLE_KM2 = 50.0

MAP_METRICS = {
    "Share of burnable land": "rate",
    "Total burned area": "total",
    "Years with fire": "recurrence",
}

METRIC_META = {
    "total": dict(
        bar="km²", fmt=":,.1f", title="Burned Area by Wilaya",
        label="Burned area (km²)",
    ),
    "rate": dict(
        bar="%/yr", fmt=":.2f", title="Annual Burn Rate by Wilaya",
        label="% of burnable land per year",
    ),
    "recurrence": dict(
        bar="years", fmt=":.0f", title="Years with Fire by Wilaya",
        label="Years with any fire",
    ),
}


@st.cache_data(show_spinner=False)
def wilaya_metric(
    yr_min: int, yr_max: int, categories: Tuple[str, ...], metric: str,
) -> pd.DataFrame:
    """One value per wilaya for the chosen map metric.

    `total` ranks wilayas largely by their size — Tlemcen and Sidi Bel Abbes
    sit in its top ten at ~12% of their burnable land, while Blida is tenth on
    area and first once normalised. `rate` divides by the land that could
    actually burn, so it measures fire regime rather than geography.
    """
    burn_cols = selected_burn_cols(categories)
    if not burn_cols:
        return pd.DataFrame(columns=["ADM1_NAME", "value"])

    b1, _, lc1, _, _ = load_data()
    window = b1[(b1["year"] >= yr_min) & (b1["year"] <= yr_max)].copy()
    window["_sel"] = window[burn_cols].sum(axis=1)

    if metric == "recurrence":
        yearly = window.groupby(["ADM1_NAME", "year"], observed=True)["_sel"].sum()
        val = (yearly > 0).groupby("ADM1_NAME", observed=True).sum()
        return val.rename("value").reset_index()

    if metric == "rate":
        # A partial final year would add its fires to the numerator while
        # counting as a whole year in the denominator, deflating every rate.
        # Drop it from both rather than quietly averaging it in.
        partial = partial_final_year()
        if partial and yr_min <= partial[0] <= yr_max:
            window = window[window["year"] != partial[0]]
        n_years = int(window["year"].nunique())
        if not n_years:
            return pd.DataFrame(columns=["ADM1_NAME", "value"])

        lc_cols = [c for c in LC_TYPE_COLS if LC_LABELS[c] in categories]
        lcw = lc1[(lc1["year"] >= yr_min) & (lc1["year"] <= yr_max)]
        if lcw.empty or not lc_cols:
            return pd.DataFrame(columns=["ADM1_NAME", "value"])

        burned = window.groupby("ADM1_NAME", observed=True)["_sel"].sum()
        burnable = (lcw.assign(_b=lcw[lc_cols].sum(axis=1))
                       .groupby("ADM1_NAME", observed=True)["_b"].mean())
        out = pd.concat([burned.rename("burned"),
                         burnable.rename("burnable")], axis=1).dropna()
        out = out[out["burnable"] >= MIN_BURNABLE_KM2]
        out["value"] = 100.0 * out["burned"] / (out["burnable"] * n_years)
        return out.reset_index()[["ADM1_NAME", "value"]]

    total = window.groupby("ADM1_NAME", observed=True)["_sel"].sum()
    return total.rename("value").reset_index()


@st.cache_data(show_spinner=False)
def build_choropleth(
    yr_min: int, yr_max: int, wilaya: str, categories: Tuple[str, ...],
    metric: str = "rate",
) -> go.Figure:
    """Choropleth of the chosen metric per wilaya for the current selection.

    Cached on its scalar arguments: the geometry is re-serialised only when the
    selection actually changes, not on every widget interaction.
    """
    cols = selected_burn_cols(categories)
    if not cols:
        return _empty_fig("Select at least one land-cover type")

    b1, _, _, _, _ = load_data()
    prov = load_provinces()
    meta = METRIC_META[metric]

    totals = wilaya_metric(yr_min, yr_max, categories, metric)
    if not totals.empty:
        totals["ADM1_NAME"] = totals["ADM1_NAME"].astype(str)

    frame = pd.DataFrame({"ADM1_NAME": prov["names"]}).merge(
        totals, on="ADM1_NAME", how="left"
    ) if not totals.empty else pd.DataFrame(
        {"ADM1_NAME": prov["names"], "value": float("nan")})
    known = frame[frame["value"].notna()]
    unknown = frame[frame["value"].isna()]

    # Nothing to colour: the selected years lie outside the burned-area record.
    # An all-grey map with no explanation looks like a rendering failure.
    if known.empty:
        extra = ("<br><sub>Rates exclude the incomplete final year</sub>"
                 if metric == "rate" else "")
        return _empty_fig(
            f"No burned-area data for {yr_min}–{yr_max}<br>"
            f"<sub>MCD64A1 covers {int(b1['year'].min())}–{int(b1['year'].max())}"
            f"</sub>{extra}"
        )

    fig = px.choropleth_map(
        known,
        geojson=prov["geojson"],
        locations="ADM1_NAME",
        featureidkey="properties.ADM1_NAME",
        color="value",
        hover_name="ADM1_NAME",
        hover_data={"value": meta["fmt"]},
        map_style="carto-positron",
        opacity=0.75,
        color_continuous_scale="YlOrRd",
        labels={"value": meta["label"]},
        title=f"{meta['title']} ({yr_min}–{yr_max})",
    )

    # Boundaries and data now come from the same GAUL 2015 release, so this is
    # normally empty. Kept as a guard against a future boundary/data mismatch.
    if not unknown.empty:
        fig.add_trace(go.Choroplethmap(
            geojson=prov["geojson"],
            locations=unknown["ADM1_NAME"],
            featureidkey="properties.ADM1_NAME",
            z=[0] * len(unknown),
            colorscale=[[0, NODATA_FILL], [1, NODATA_FILL]],
            showscale=False,
            marker=dict(opacity=0.6, line=dict(width=0.5, color=NODATA_LINE)),
            hovertemplate="<b>%{location}</b><br>No data<extra></extra>",
            name="No data",
        ))

    # Frame the selected wilaya and outline it, so the map tracks the sidebar.
    if wilaya in prov["bounds"]:
        center, zoom = _view_from_bounds(prov["bounds"][wilaya])
        lons, lats = _outline_coords(prov["geoms"][wilaya])
        fig.add_trace(go.Scattermap(
            lon=lons, lat=lats, mode="lines",
            line=dict(width=2.5, color=HIGHLIGHT),
            hoverinfo="skip", showlegend=False,
        ))
    else:
        center, zoom = _view_from_bounds(prov["total_bounds"])

    fig.update_layout(
        map=dict(center=center, zoom=zoom),
        margin=dict(r=0, t=40, l=0, b=0),
        # Overlay the colourbar on the map. With a zero right margin there is no
        # gutter for it to sit in, so anchoring it outside gets it clipped.
        coloraxis_colorbar=dict(
            title=meta["bar"], len=0.7, thickness=12,
            x=0.98, xanchor="right", y=0.5, yanchor="middle",
            bgcolor="rgba(255,255,255,0.75)", outlinewidth=0,
        ),
        showlegend=False,
    )
    return fig


@st.cache_data(show_spinner=False)
def chart_recurrence(
    yr_min: int, yr_max: int, categories: Tuple[str, ...],
    wilaya: str, suffix: str, top_n: int = 15,
) -> go.Figure:
    """Communes that burn most often — a fuel-management list, not a total.

    Cumulative area answers "where was the most lost". How often a place burns
    is a different question, and the one that identifies somewhere the fire
    regime keeps returning to. Across all cover types Ain Zitoun burned in 22
    of the record's 26 years; restricted to forest, shrubland and cropland the
    list is led by Messelmoun and Ouled Hellal at 20.
    """
    cols = selected_burn_cols(categories)
    if not cols:
        return _empty_fig("Select at least one land-cover type")

    _, b2, _, _, _ = load_data()
    window = b2[(b2["year"] >= yr_min) & (b2["year"] <= yr_max)].copy()
    if wilaya != ALL_WILAYAS:
        window = window[window["ADM1_NAME"] == wilaya]
    if window.empty:
        return _empty_fig("No burned-area data in this selection")

    window["_sel"] = window[cols].sum(axis=1)
    yearly = window.groupby(
        ["ADM1_NAME", "ADM2_NAME", "year"], observed=True)["_sel"].sum()
    years = (yearly > 0).groupby(
        ["ADM1_NAME", "ADM2_NAME"], observed=True).sum().rename("years")
    top = years[years > 0].sort_values(ascending=False).head(top_n).reset_index()
    if top.empty:
        return _empty_fig("No commune burned in this selection")

    top["label"] = top["ADM2_NAME"].astype(str) + ", " + top["ADM1_NAME"].astype(str)
    span = yr_max - yr_min + 1
    fig = px.bar(
        top.iloc[::-1], x="years", y="label", orientation="h",
        title=f"Most Frequently Burning Communes{suffix}",
        labels={"years": f"Years with fire (of {span})", "label": ""},
        template="plotly_white", color_discrete_sequence=["#9D0208"],
    )
    fig.update_layout(showlegend=False, margin=dict(l=0, r=10, t=40, b=0))
    return fig


# ── Ignition analysis ─────────────────────────────────────────────────────────
# Ignitions are derived from NASA FIRMS active-fire detections by
# prepare_ignitions.py: detections are clustered in space and time into fire
# events, and the earliest detection of each event becomes its ignition point.
# MCD64A1 alone cannot answer "where do fires start" — it is a monthly burned
# -area mask with no ignition location.
def filter_ignitions(
    ign: pd.DataFrame, wilaya: str, commune_code: Optional[int],
    yr_min: int, yr_max: int,
) -> pd.DataFrame:
    df = ign[(ign["year"] >= yr_min) & (ign["year"] <= yr_max)]
    # ADM2_CODE is -1 when the ignitions were built without the GAUL commune
    # boundaries. Fall back to the wilaya rather than returning nothing.
    resolved = (df["ADM2_CODE"] > 0).any()
    if commune_code is not None and resolved:
        df = df[df["ADM2_CODE"] == commune_code]
    elif wilaya != ALL_WILAYAS:
        df = df[df["ADM1_NAME"] == wilaya]
    return df.copy()


def provisional_years(ign: Optional[pd.DataFrame]) -> List[int]:
    """Years whose ignitions include near-real-time data.

    NRT rows have no `type` field, so they are never screened for static
    industrial sources, and they are not reprocessed the way archive rows are.
    Mixing them into a year silently makes that year non-comparable, so the
    affected years are marked rather than blended.
    """
    if ign is None or ign.empty or "source" not in ign.columns:
        return []
    return sorted(int(y) for y in ign.loc[ign["source"] == "nrt", "year"].unique())


def communes_resolved(ign: Optional[pd.DataFrame]) -> bool:
    return ign is not None and not ign.empty and bool((ign["ADM2_CODE"] > 0).any())


def _ignition_hover(df: pd.DataFrame) -> Tuple[Sequence, str]:
    """Per-point tooltip text: ignition date first, then place, size, sensor.

    The strings are assembled here rather than inside the hovertemplate because
    the wording has to bend to the data — a one-overpass fire is not "0 days",
    an unresolved commune must not print an empty line, and NRT points have to
    admit they are provisional.
    """
    date = df["date"].dt.strftime("%d %b %Y")

    place = df["ADM1_NAME"].astype(str)
    has_commune = (df["ADM2_CODE"] > 0) & df["ADM2_NAME"].notna()
    place = place.mask(has_commune, df["ADM2_NAME"].astype(str) + ", " + place)

    n = df["n_detections"].astype(int)
    days = df["duration_days"].fillna(0).round().astype(int)
    size = n.astype(str) + " detection" + n.gt(1).map({True: "s", False: ""})
    # duration_days is a span: 0 means the fire was seen on a single overpass.
    size = size.mask(days > 0, size + " over " + (days + 1).astype(str) + " days")

    frp = df["frp_max_mw"]
    frp_txt = ("peak " + frp.map(lambda v: f"{v:,.0f}") + " MW FRP").where(
        frp.notna() & (frp > 0), "FRP not reported")

    sensor = df["instrument"].astype(str) + " " + df["satellite"].astype(str)
    sensor = sensor.mask(df["source"].astype(str) == "nrt",
                         sensor + " · NRT, provisional")

    template = (
        # Constant text belongs in the template, not in customdata: the
        # template ships once, customdata ships per point.
        "<b>Ignited %{customdata[0]}</b><br>"
        "%{customdata[1]}<br>"
        "%{customdata[2]} · %{customdata[3]}<br>"
        "<span style='font-size:0.85em'>%{customdata[4]}</span>"
        "<extra></extra>"
    )
    cd = pd.DataFrame({"date": date, "place": place, "size": size,
                       "frp": frp_txt, "sensor": sensor})
    return cd.to_numpy(), template


def _selected_ignition(event, df: pd.DataFrame) -> Optional[str]:
    """Detail line for a tapped point — the touch equivalent of the tooltip.

    Plotly binds map hover to mousemove on the MapLibre canvas, and a
    touchscreen never fires one, so on a phone the dots are inert. Selection
    does fire on tap, so the same fields are rendered under the map instead.
    """
    try:
        points = event["selection"]["points"]
    except (TypeError, KeyError, IndexError):
        return None
    if not points:
        return None

    fields = points[-1].get("customdata")          # last tap wins
    if not fields:
        # Older Streamlit builds omit customdata from the event; the trace is
        # plotted straight from df, so the point index still resolves.
        idx = points[-1].get("point_index", points[-1].get("point_number"))
        if idx is None or not 0 <= int(idx) < len(df):
            return None
        fields = _ignition_hover(df.iloc[[int(idx)]])[0][0]

    fields = list(fields)
    if len(fields) < 5:
        return None
    date, place, size, frp, sensor = fields[:5]
    return f"**Ignited {date}** · {place} · {size} · {frp} · {sensor}"


def map_ignitions(
    df: pd.DataFrame, bounds, suffix: str, mode: str = "points",
) -> go.Figure:
    """Where fires start — a wilaya choropleth cannot show ignition corridors.

    Two mutually exclusive renderings, because they cannot be stacked: 22,000
    markers drawn over the heatmap at national zoom cover it completely, and
    the heatmap is a smoothed raster whose hover can only report a bin's
    intensity, never a fire. Points carry the per-event detail; the heatmap
    reads better as a pure density surface.
    """
    if df.empty:
        return _empty_fig("No ignitions in this selection")
    center, zoom = _view_from_bounds(bounds)

    if mode == "heatmap":
        fig = px.density_map(
            df, lat="lat", lon="lon", radius=10,
            center=center, zoom=zoom, map_style="carto-positron",
            color_continuous_scale="Inferno",
            title=f"Ignition Density{suffix}",
        )
        fig.update_layout(
            margin=dict(r=0, t=40, l=0, b=0),
            coloraxis_colorbar=dict(
                title="Ignitions", len=0.7, thickness=12,
                x=0.98, xanchor="right", y=0.5, yanchor="middle",
                bgcolor="rgba(255,255,255,0.75)", outlinewidth=0,
            ),
        )
        return fig

    customdata, template = _ignition_hover(df)
    # Marker area tracks detection count so large fires read as large, clamped
    # so a 408-detection event cannot swallow its neighbours. Semi-transparent
    # so that overlapping dots still pile up into a visible density.
    n = df["n_detections"].clip(lower=1, upper=40)
    # Everything here is serialised per point and shipped to the browser.
    # Plotly base64-encodes numeric numpy arrays but writes plain JSON for
    # Python lists, so keep these float32 arrays — .tolist() would roughly
    # triple the cost of the coordinate and size channels.
    fig = go.Figure(go.Scattermap(
        lat=df["lat"], lon=df["lon"], mode="markers",
        marker=dict(size=(2.5 + 1.2 * (n ** 0.5)).astype("float32"),
                    color="#D00000", opacity=0.55),
        customdata=customdata, hovertemplate=template,
        name="", showlegend=False,
        hoverlabel=dict(bgcolor="white", bordercolor="#9D0208",
                        font=dict(color="#1B1B1F")),
    ))
    fig.update_layout(
        title=f"Ignitions{suffix}", showlegend=False,
        margin=dict(r=0, t=40, l=0, b=0),
        map=dict(style="carto-positron", center=center, zoom=zoom),
    )
    return fig


def chart_ign_annual(
    df: pd.DataFrame, suffix: str, provisional: Optional[List[int]] = None,
) -> go.Figure:
    if df.empty:
        return _empty_fig("No ignitions in this selection")
    annual = df.groupby("year").size().reset_index(name="ignitions")
    fig = px.bar(
        annual, x="year", y="ignitions",
        title=f"Ignitions per Year{suffix}",
        labels={"year": "Year", "ignitions": "Ignitions"},
        template="plotly_white", color_discrete_sequence=["#9D0208"],
    )
    if provisional:
        fig.update_traces(marker_color=[
            PARTIAL_FILL if y in provisional else "#9D0208"
            for y in annual["year"]
        ])
    fig.update_layout(showlegend=False, xaxis=dict(dtick=2))
    return fig


def chart_ign_seasonality(df: pd.DataFrame, suffix: str) -> go.Figure:
    if df.empty:
        return _empty_fig("No ignitions in this selection")
    monthly = (
        df.groupby("month").size().reindex(range(1, 13), fill_value=0)
        .reset_index(name="ignitions")
    )
    monthly["month_name"] = monthly["month"].map(MONTH_NAMES)
    fig = px.bar(
        monthly, x="month_name", y="ignitions",
        title=f"Ignition Seasonality{suffix}",
        labels={"month_name": "Month", "ignitions": "Ignitions"},
        template="plotly_white", color_discrete_sequence=["#DC2F02"],
    )
    fig.update_layout(showlegend=False)
    return fig


def chart_ign_season_timing(
    df: pd.DataFrame, suffix: str, last_full_year: Optional[int] = None,
) -> go.Figure:
    """First, median and last ignition day per year — does the season lengthen?

    A partial final year (the FIRMS archive typically ends mid-year) would drag
    every percentile down and fake a trend, so it is excluded.
    """
    note = ""
    if last_full_year is not None and not df.empty:
        dropped = sorted(df.loc[df["year"] > last_full_year, "year"].unique())
        if dropped:
            df = df[df["year"] <= last_full_year]
            note = f"  ·  {dropped[0]}+ excluded (incomplete year)"

    if df.empty or df["year"].nunique() < 3:
        return _empty_fig("Need at least 3 complete years of ignitions")
    suffix = f"{suffix}{note}"
    # Percentiles are robust to a single stray winter ignition.
    g = df.groupby("year")["doy"]
    timing = pd.DataFrame({
        "Season start (5th pct)":  g.quantile(0.05),
        "Season midpoint (median)": g.median(),
        "Season end (95th pct)":   g.quantile(0.95),
    }).reset_index()
    melted = timing.melt("year", var_name="Measure", value_name="Day of year")
    fig = px.line(
        melted, x="year", y="Day of year", color="Measure", markers=True,
        title=f"Fire Season Timing{suffix}",
        labels={"year": "Year"}, template="plotly_white",
        color_discrete_sequence=["#52B788", "#E25822", "#6A040F"],
    )
    fig.update_layout(xaxis=dict(dtick=2),
                      legend=dict(orientation="h", y=-0.25, title=None))
    return fig


def chart_ign_size_dist(df: pd.DataFrame, suffix: str) -> go.Figure:
    """Fire size is heavy-tailed: a few events dominate the burned total."""
    if df.empty or "extent_km2" not in df.columns:
        return _empty_fig("No ignitions in this selection")
    sizes = df.loc[df["extent_km2"] > 0, "extent_km2"]
    if sizes.empty:
        return _empty_fig("No sized events in this selection")
    fig = px.histogram(
        sizes, x="extent_km2", nbins=40, log_y=True,
        title=f"Fire Event Size Distribution{suffix}",
        labels={"extent_km2": "Detected extent (km²)"},
        template="plotly_white", color_discrete_sequence=["#6A040F"],
    )
    fig.update_layout(showlegend=False, yaxis_title="Events (log scale)")
    return fig


def chart_burned_per_ignition(
    df_burn: pd.DataFrame, df_ign: pd.DataFrame, suffix: str,
) -> go.Figure:
    """km² burned per ignition — separates many small starts from few big ones."""
    if df_ign.empty:
        return _empty_fig("No ignitions in this selection")
    burned = df_burn.groupby("year")[SEL_COL].sum()
    counts = df_ign.groupby("year").size()
    joined = pd.concat([burned.rename("km2"), counts.rename("n")], axis=1).dropna()
    joined = joined[joined["n"] > 0]
    if joined.empty:
        return _empty_fig("No overlapping years between burned area and ignitions")
    joined["per_ignition"] = joined["km2"] / joined["n"]
    joined = joined.reset_index()
    fig = px.bar(
        joined, x="year", y="per_ignition",
        title=f"Burned Area per Ignition{suffix}",
        labels={"year": "Year", "per_ignition": "km² per ignition"},
        template="plotly_white", color_discrete_sequence=["#BC6C25"],
    )
    fig.update_layout(showlegend=False, xaxis=dict(dtick=2))
    return fig


# ── Fire weather ──────────────────────────────────────────────────────────────
# Counts alone invite the wrong conclusion: 2021 had far fewer ignitions than
# 2020 and burned Kabylie to the ground. This layer supplies the other half of
# the question — whether a season's weather made fire easy.
@st.cache_data(show_spinner=False)
def load_weather() -> Optional[pd.DataFrame]:
    if not WEATHER_FILE.exists():
        return None
    return pd.read_parquet(WEATHER_FILE)


def weather_seasons(
    w: pd.DataFrame, wilaya: str, yr_min: int, yr_max: int,
) -> pd.DataFrame:
    """Fire-season weather per year, averaged over the wilayas in scope.

    Averaged rather than summed: a national total would just count wilayas.
    And when no wilaya is chosen, only those sampled at an ignition centroid
    are averaged — the Saharan units bake at 45 C with 10% humidity every
    summer and have nothing to burn, so including them would swamp the signal
    with weather that never starts a fire.
    """
    season = w[w["month"].isin(FIRE_SEASON)
               & (w["year"] >= yr_min) & (w["year"] <= yr_max)]
    if wilaya != ALL_WILAYAS:
        season = season[season["ADM1_NAME"] == wilaya]
    elif "sample_basis" in season.columns:
        season = season[season["sample_basis"].astype(str) == "ignition centroid"]
    if season.empty:
        return season

    per_wilaya = season.groupby(["ADM1_NAME", "year"], observed=True).agg(
        fire_weather_days=("fire_weather_days", "sum"),
        t_max_c=("t_max_c", "mean"),
        rh_min_pct=("rh_min_pct", "mean"),
        wind_max_kmh=("wind_max_kmh", "mean"),
        precip_mm=("precip_mm", "sum"),
    ).reset_index()
    return per_wilaya.groupby("year", observed=True).mean(
        numeric_only=True).reset_index()


def chart_fire_weather_days(seasons: pd.DataFrame, suffix: str) -> go.Figure:
    """Fire-weather days per season against the period's own normal."""
    if seasons.empty:
        return _empty_fig("No weather data in this selection")
    normal = float(seasons["fire_weather_days"].median())
    colours = ["#9D0208" if v >= normal else "#F48C06"
               for v in seasons["fire_weather_days"]]
    fig = px.bar(
        seasons, x="year", y="fire_weather_days",
        title=f"Fire-Weather Days per Season{suffix}",
        labels={"year": "Year", "fire_weather_days": "Days (Jun–Sep)"},
        template="plotly_white",
    )
    fig.update_traces(marker_color=colours)
    fig.add_hline(y=normal, line_dash="dash", line_color="#495057",
                  annotation_text=f"median {normal:.0f}",
                  annotation_position="top left")
    fig.update_layout(showlegend=False, xaxis=dict(dtick=2))
    return fig


def chart_weather_vs_fire(
    seasons: pd.DataFrame, ign: Optional[pd.DataFrame],
    wilaya: str, suffix: str,
) -> go.Figure:
    """Fire-weather days against ignitions, one point per year.

    The point of the whole layer: a year sitting far above the cloud burned
    despite ordinary weather, which points at ignition pressure; one far to
    the right burned in conditions that would have carried any spark.
    """
    if seasons.empty or ign is None or ign.empty:
        return _empty_fig("Needs both weather and ignition data")

    fires = ign[ign["month"].isin(FIRE_SEASON)]
    if wilaya != ALL_WILAYAS:
        fires = fires[fires["ADM1_NAME"] == wilaya]
    counts = fires.groupby("year").size().rename("ignitions").reset_index()

    merged = seasons.merge(counts, on="year", how="inner")
    if merged.empty:
        return _empty_fig("No overlapping years")

    fig = px.scatter(
        merged, x="fire_weather_days", y="ignitions", text="year",
        title=f"Weather against Ignitions{suffix}",
        labels={"fire_weather_days": "Fire-weather days (Jun–Sep)",
                "ignitions": "Ignitions (Jun–Sep)"},
        template="plotly_white",
    )
    fig.update_traces(
        marker=dict(size=11, color="#9D0208", opacity=0.8),
        textposition="top center", textfont=dict(size=9, color="#495057"),
        hovertemplate="<b>%{text}</b><br>%{x:.0f} fire-weather days"
                      "<br>%{y:,} ignitions<extra></extra>",
    )
    fig.update_layout(showlegend=False)
    return fig


def chart_weather_drivers(seasons: pd.DataFrame, suffix: str) -> go.Figure:
    """The four drivers as deviations from their own period mean.

    Plotted as anomalies because the raw units share no scale — 34 °C, 22%,
    25 km/h and 12 mm cannot sit on one axis and be read.
    """
    if seasons.empty or len(seasons) < 2:
        return _empty_fig("Not enough years to show anomalies")
    fields = {
        "t_max_c": "Max temperature",
        "rh_min_pct": "Min humidity",
        "wind_max_kmh": "Max wind",
        "precip_mm": "Rainfall",
    }
    fig = go.Figure()
    for col, label in fields.items():
        series = seasons[col]
        sd = float(series.std())
        if not sd:
            continue
        fig.add_trace(go.Scatter(
            x=seasons["year"], y=(series - series.mean()) / sd,
            mode="lines+markers", name=label,
            hovertemplate=f"{label}: %{{y:+.2f}} sd<extra></extra>",
        ))
    fig.add_hline(y=0, line_color="#ADB5BD", line_width=1)
    fig.update_layout(
        title=f"Fire-Season Drivers{suffix}", template="plotly_white",
        yaxis_title="Standard deviations from the period mean",
        xaxis_title="Year", xaxis=dict(dtick=2), hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=0, yanchor="bottom"),
    )
    return fig


def render_weather_section(
    wilaya: str, yr_min: int, yr_max: int, suffix: str,
) -> None:
    st.subheader("🌡️ Fire Weather")

    w = load_weather()
    if w is None:
        st.info(
            "**No fire-weather data yet.** This panel needs "
            "`data/fire_weather.parquet`.\n\n"
            "Build it with `python fetch_weather.py` — it pulls ERA5 daily "
            "reanalysis from the Open-Meteo archive, which needs no API key."
        )
        return

    seasons = weather_seasons(w, wilaya, yr_min, yr_max)
    if seasons.empty:
        covered = set(w["ADM1_NAME"].astype(str))
        if wilaya != ALL_WILAYAS and wilaya not in covered:
            st.info(
                f"**{wilaya} is not in the fire-weather layer.** It is fetched "
                f"only for the wilayas that actually record fires, since the "
                f"rest have no fire weather worth the request. Add them with "
                f"`python fetch_weather.py --all-wilayas`."
            )
        else:
            st.warning("No weather data for this year range.")
        return

    latest = seasons.iloc[-1]
    normal = seasons["fire_weather_days"].median()
    hottest = seasons.loc[seasons["t_max_c"].idxmax()]
    driest = seasons.loc[seasons["precip_mm"].idxmin()]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"🔥 Fire-weather days {int(latest['year'])}",
              f"{latest['fire_weather_days']:.0f}",
              f"{latest['fire_weather_days'] - normal:+.0f} vs median",
              delta_color="inverse")
    k2.metric("📊 Season median", f"{normal:.0f} days")
    k3.metric("🌡️ Hottest season", f"{int(hottest['year'])}",
              f"{hottest['t_max_c']:.1f} °C mean daily max", delta_color="off")
    k4.metric("🏜️ Driest season", f"{int(driest['year'])}",
              f"{driest['precip_mm']:.0f} mm total", delta_color="off")

    st.caption(
        f"A **fire-weather day** is a day whose maximum temperature reached "
        f"30 °C, minimum humidity fell to 30% or below, and maximum wind "
        f"reached 20 km/h — an explicit rule, not a named index, so the four "
        f"drivers below can be read against it. Season is June–September."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_fire_weather_days(seasons, suffix),
                        key="wx_days", **STRETCH)
    with c2:
        st.plotly_chart(
            chart_weather_vs_fire(seasons, load_ignitions(), wilaya, suffix),
            key="wx_vs_fire", **STRETCH,
        )
    st.caption(
        "A year high on the ignition axis but ordinary on the weather axis "
        "burned because something started fires, not because conditions were "
        "exceptional. A year far to the right faced conditions that would "
        "have carried almost any spark."
    )

    st.plotly_chart(chart_weather_drivers(seasons, suffix),
                    key="wx_drivers", **STRETCH)

    basis = w[w["ADM1_NAME"] == wilaya]["sample_basis"] if wilaya != ALL_WILAYAS \
        else pd.Series(dtype=object)
    where = (f"sampled at {basis.iloc[0]}" if len(basis)
             else "averaged over the wilayas that actually record fires")
    st.caption(
        f"ERA5 reanalysis (~11 km), {where}. Each wilaya is sampled at the "
        f"mean position of its own ignitions rather than its geometric centre, "
        f"so the series describes conditions where that wilaya burns rather "
        f"than an average over land that never does. Rebuild with "
        f"`python fetch_weather.py`."
    )


# ── Fire event catalogue ──────────────────────────────────────────────────────
# Every other panel aggregates events into counts and totals, which answers
# "how much" and never "which fire". The August 2021 Kabylie disaster is in
# this dataset and was unreachable through the UI until this section existed.
EVENT_RANKS = {
    "Detected extent": ("extent_km2", "%.0f km²"),
    "Cumulative FRP":     ("frp_sum_mw",    "%,.0f MW"),
    "Peak FRP":           ("frp_max_mw",    "%,.0f MW"),
    "Detections":         ("n_detections",  "%d"),
    "Duration":           ("duration_days", "%.1f days"),
}


def event_table(df: pd.DataFrame, rank_by: str, top_n: int) -> pd.DataFrame:
    """The top `top_n` events by the chosen measure, ready to display."""
    col, _ = EVENT_RANKS[rank_by]
    top = df.nlargest(top_n, col).copy()
    out = pd.DataFrame({
        "Date": top["date"].dt.date,
        "Commune": top["ADM2_NAME"].astype(str),
        "Wilaya": top["ADM1_NAME"].astype(str),
        # Cast before rounding: these are float32 on disk, and rounding one
        # in place leaves values like 28.200001 in the table.
        "Extent (km²)": top["extent_km2"].astype("float64").round(0),
        "Detections": top["n_detections"],
        "Days": top["duration_days"].astype("float64").round(1),
        "Peak FRP (MW)": top["frp_max_mw"].astype("float64").round(0),
        "Total FRP (MW)": top["frp_sum_mw"].astype("float64").round(0),
        "Source": top["source"].astype(str),
    })
    # Keep the original rows alongside, so a selected display row can be
    # traced back to its ignition without re-deriving anything.
    out.attrs["rows"] = top.reset_index(drop=True)
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_daily() -> Optional[pd.DataFrame]:
    """Per-event daily progression, written alongside the ignitions."""
    if not DAILY_FILE.exists():
        return None
    return pd.read_parquet(DAILY_FILE)


def chart_growth(daily: pd.DataFrame, row: pd.Series) -> go.Figure:
    """How one fire developed: extent gained per day against total reached.

    New cells rather than detections, because re-detecting yesterday's ground
    adds radiative power but no area. The two together separate a fire that
    kept spreading from one that burned hard in place.
    """
    d = daily[daily["ignition_id"] == row["ignition_id"]].sort_values("day")
    if d.empty:
        return _empty_fig("No daily record for this event")
    d = d.copy()
    d["cumulative"] = d["new_cells"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["day"], y=d["new_cells"], name="New extent that day",
        marker_color="#F48C06",
        hovertemplate="%{x|%d %b}<br>%{y} km² newly detected<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d["day"], y=d["cumulative"], name="Cumulative extent",
        mode="lines+markers", line=dict(color="#9D0208", width=2.5),
        hovertemplate="%{x|%d %b}<br>%{y} km² total<extra></extra>",
    ))
    single = len(d) == 1
    fig.update_layout(
        title="Growth" + (" — seen on a single day" if single else ""),
        template="plotly_white", margin=dict(l=0, r=10, t=40, b=0),
        yaxis_title="km² (1 km cells)", xaxis_title="",
        legend=dict(orientation="h", y=1.12, x=0, yanchor="bottom"),
        bargap=0.35,
    )
    return fig


def map_single_event(row: pd.Series, context: pd.DataFrame) -> go.Figure:
    """One fire in place, with that year's other ignitions around it."""
    others = context[context["ignition_id"] != row["ignition_id"]]
    fig = go.Figure()
    if not others.empty:
        fig.add_trace(go.Scattermap(
            lat=others["lat"], lon=others["lon"], mode="markers",
            marker=dict(size=6, color="#9AA0A6", opacity=0.55),
            hoverinfo="skip", showlegend=False, name="",
        ))
    fig.add_trace(go.Scattermap(
        lat=[row["lat"]], lon=[row["lon"]], mode="markers",
        marker=dict(size=20, color="#D00000", opacity=0.9),
        hovertemplate=(f"<b>{row['date']:%d %b %Y}</b><br>"
                       f"{row['ADM2_NAME']}, {row['ADM1_NAME']}<extra></extra>"),
        showlegend=False, name="",
    ))
    fig.update_layout(
        margin=dict(r=0, t=30, l=0, b=0), showlegend=False,
        title=f"{row['ADM2_NAME']}, {row['ADM1_NAME']} — {row['date']:%d %b %Y}",
        map=dict(style="carto-positron",
                 center=dict(lat=float(row["lat"]), lon=float(row["lon"])),
                 zoom=8.5),
    )
    return fig


def render_event_catalogue(
    ign: Optional[pd.DataFrame], wilaya: str, commune_code: Optional[int],
    yr_min: int, yr_max: int, suffix: str,
) -> None:
    st.subheader("📇 Fire Event Catalogue")

    if ign is None or ign.empty:
        st.info("**No ignition data yet.** This panel needs "
                "`data/ignitions.parquet` — run `prepare_ignitions.py`.")
        return

    df = filter_ignitions(ign, wilaya, commune_code, yr_min, yr_max)
    if df.empty:
        st.warning("No fire events in this selection. Widen the year range or "
                   "clear the wilaya filter.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        rank_by = st.radio(
            "Rank by", list(EVENT_RANKS), horizontal=True,
            key="event_rank", label_visibility="collapsed",
        )
    with c2:
        top_n = st.selectbox("Show", [25, 50, 100, 250], index=1,
                             key="event_top_n", label_visibility="collapsed")

    table = event_table(df, rank_by, int(top_n))
    rows = table.attrs["rows"]

    st.caption(
        f"The {len(table)} largest of **{len(df):,}** events{suffix}, by "
        f"**{rank_by.lower()}**. Click a row to place it on the map."
    )
    event = st.dataframe(
        table, key="event_table", hide_index=True,
        on_select="rerun", selection_mode="single-row", **STRETCH,
    )

    picked = None
    try:
        chosen = event["selection"]["rows"]
        if chosen:
            picked = rows.iloc[int(chosen[0])]
    except (TypeError, KeyError, IndexError, ValueError):
        picked = None

    if picked is None:
        st.caption("No row selected — pick one to see where it burned.")
    else:
        same_year = ign[(ign["year"] == int(picked["year"]))
                        & (ign["ADM1_NAME"] == picked["ADM1_NAME"])]
        m1, m2 = st.columns([1.3, 1])
        with m1:
            st.plotly_chart(map_single_event(picked, same_year),
                            key="event_map", **STRETCH)
            st.caption("Grey points are the other ignitions recorded in that "
                       "wilaya that year.")
        with m2:
            rank_col, _ = EVENT_RANKS[rank_by]
            better = int((df[rank_col] > picked[rank_col]).sum()) + 1
            e1, e2 = st.columns(2)
            e1.metric("🔥 Detected extent",
                      f"{picked['extent_km2']:,.0f} km²")
            e2.metric("🛰️ Detections", f"{int(picked['n_detections']):,}")
            e3, e4 = st.columns(2)
            e3.metric("⚡ Peak FRP", f"{picked['frp_max_mw']:,.0f} MW")
            e4.metric("⏱️ Duration", f"{picked['duration_days']:.1f} days")
            daily = load_daily()
            if daily is not None:
                st.plotly_chart(chart_growth(daily, picked),
                                key="event_growth", **STRETCH)
            st.caption(
                f"Ranked **#{better:,}** of {len(df):,} events in this "
                f"selection by {rank_by.lower()}. First detected "
                f"{picked['date']:%d %B %Y} by {picked['instrument']} "
                f"{picked['satellite']}."
            )
            if str(picked["source"]) == "nrt":
                st.caption("⚠️ Near-real-time: never screened for static "
                           "industrial sources, and not reprocessed.")

    st.caption(
        "**Detected extent counts distinct 1 km ground cells**, so a patch "
        "seen on five overpasses counts once. It is a lower bound on burned "
        "area, not a measurement: ground that burns between overpasses, under "
        "cloud, or below a 1 km MODIS pixel is missed entirely. "
        "**Cumulative FRP sums instantaneous readings** across detections, so "
        "it rewards fires that happened to be seen more often — compare it "
        "against detection count rather than reading it as released energy."
    )


def render_ignition_section(
    ign: Optional[pd.DataFrame], df_burn: pd.DataFrame,
    wilaya: str, commune_code: Optional[int],
    yr_min: int, yr_max: int, suffix: str,
) -> None:
    st.subheader("🎯 Ignition Analysis")

    if ign is None:
        st.info(
            "**No ignition data yet.** This panel needs `data/ignitions.parquet`.\n\n"
            "1. Download the Algeria active-fire archive (MODIS `MCD14ML`, and "
            "optionally VIIRS 375 m) as CSV from "
            "[NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/download/).\n"
            "2. Run section 6 of `gee_export.js` to export `gaul_adm2.geojson`.\n"
            "3. Run `python prepare_ignitions.py`.",
            icon="🎯",
        )
        return

    df = filter_ignitions(ign, wilaya, commune_code, yr_min, yr_max)
    prov = load_provinces()
    bounds = prov["bounds"].get(wilaya, prov["total_bounds"])

    n_ign = len(df)
    years_span = max(yr_max - yr_min + 1, 1)
    # Median duration is 0 for most events — a MODIS fire is usually seen on a
    # single overpass — so it says nothing. Detections per fire is the useful
    # size proxy: it is what separates August 2021 (8.1) from August 2020 (3.2).
    det_per_fire = float(df["n_detections"].mean()) if n_ign else 0.0

    # Burned area and ignitions can cover different periods. Compute the ratio
    # only over years where both exist, otherwise it silently reads low.
    burn_years = set(df_burn.loc[df_burn[SEL_COL] > 0, "year"].unique())
    both = df[df["year"].isin(burn_years)]
    burned_total = float(df_burn[SEL_COL].sum())
    ratio = f"{burned_total / len(both):,.2f} km²" if len(both) else "—"

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("🎯 Ignitions", f"{n_ign:,}")
    i2.metric("📈 Ignitions per year", f"{n_ign / years_span:,.0f}")
    i3.metric("🛰️ Detections per fire", f"{det_per_fire:.2f}",
              help="Mean MODIS/VIIRS detections per fire event — a proxy for "
                   "fire size. Higher means fewer, larger fires.")
    i4.metric("🔥 Burned per ignition", ratio,
              help="Averaged over years where both burned-area and ignition "
                   "data exist.")

    provisional = provisional_years(ign)
    shown = [y for y in provisional if yr_min <= y <= yr_max]
    if shown:
        span = f"{shown[0]}" if len(shown) == 1 else f"{shown[0]}–{shown[-1]}"
        st.caption(
            f"⚠️ **{span} is provisional** — it includes near-real-time "
            f"detections, which carry no `type` field and are therefore never "
            f"screened for static industrial sources. Treat the count as "
            f"indicative and exclude it from trend fits. Its bar is greyed below."
        )

    if not communes_resolved(ign):
        st.caption(
            "Ignitions were matched to wilayas only — run section 6 of "
            "`gee_export.js` and rebuild to enable commune-level filtering."
        )

    c1, c2 = st.columns([1.3, 1])
    with c1:
        map_mode = st.radio(
            "Map style", ["Individual ignitions", "Density heatmap"],
            horizontal=True, key="ign_map_mode", label_visibility="collapsed",
            help="**Individual ignitions** draws one dot per fire event at its "
                 "first detection — hover a dot for its ignition date, place, "
                 "size and sensor; dot size tracks detection count. "
                 "**Density heatmap** smooths them into a continuous surface, "
                 "which reads better at national scale but cannot be hovered.",
        )
        points_mode = not map_mode.startswith("Density")
        fig = map_ignitions(df, bounds, suffix,
                            "points" if points_mode else "heatmap")
        if points_mode:
            # The detail line sits above the map, not below: MapLibre draws its
            # basemap attribution outside Plotly's margin box, where on a narrow
            # screen it overlaps whatever follows the chart. Above also keeps
            # the map from shifting under your finger when a tap selects.
            detail = st.empty()
            # on_select is what makes the map work on a touchscreen: a tap
            # cannot raise a hover tooltip, but it does select a point.
            event = st.plotly_chart(
                fig, key="ign_map_points", on_select="rerun",
                selection_mode="points", **STRETCH,
            )
            detail.caption(
                _selected_ignition(event, df)
                or "Tap a dot for its ignition details — with a mouse, hovering "
                   "one shows the same thing."
            )
        else:
            st.plotly_chart(fig, key="ign_map_heat", **STRETCH)
    with c2:
        st.plotly_chart(chart_ign_annual(df, suffix, provisional),
                        key="ign_annual", **STRETCH)
        st.plotly_chart(chart_ign_seasonality(df, suffix),
                        key="ign_seasonality", **STRETCH)

    # Work out the last year the archive actually covers through the fire
    # season, from the unfiltered record so a quiet wilaya cannot skew it.
    last_year = int(ign["year"].max())
    tail_doy = int(ign.loc[ign["year"] == last_year, "doy"].max())
    last_full_year = last_year - (1 if tail_doy < 300 else 0)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            chart_ign_season_timing(df, suffix, last_full_year),
            key="ign_season_timing", **STRETCH,
        )
    with c4:
        st.plotly_chart(chart_burned_per_ignition(df_burn, df, suffix),
                        key="ign_burned_per", **STRETCH)

    st.plotly_chart(chart_ign_size_dist(df, suffix), key="ign_size_dist",
                    **STRETCH)

    st.caption(
        "Ignitions are derived from NASA FIRMS active-fire detections, clustered "
        "in space and time into fire events; the earliest detection of each event "
        "is its ignition. Detected extent is a lower bound — it counts distinct "
        "1 km cells flagged at an overpass, so it is not the true burned area. "
        "Ignition counts are **not** filtered by land cover: FIRMS detections "
        "carry no land-cover attribute."
    )


# ── Main app ──────────────────────────────────────────────────────────────────
def main() -> None:
    # Load all data (instant — Parquet cached on first run)
    with st.spinner("Loading data…"):
        b1, b2, lc1, lc2, hierarchy = load_data()

    burn_yr_min, burn_yr_max = int(b1["year"].min()), int(b1["year"].max())

    # The FIRMS ignition record usually runs years past the MCD64A1 burned-area
    # record. Span the union so the later years — including the August 2021
    # Kabylie fires — are reachable, rather than clipping to the shorter series.
    partial = partial_final_year()
    _ign_years = load_ignitions()
    data_yr_min, data_yr_max = burn_yr_min, burn_yr_max
    if _ign_years is not None and not _ign_years.empty:
        data_yr_min = min(data_yr_min, int(_ign_years["year"].min()))
        data_yr_max = max(data_yr_max, int(_ign_years["year"].max()))

    st.title(f"🔥 Algeria Wildfire Analysis ({data_yr_min}–{data_yr_max})")
    st.caption(
        "**Author:** Zakaria Matougui, researcher at Territory Planning "
        "Research Centre (CRAT), Algeria · "
        "**Data:** MODIS MCD64A1 (burned area) + MCD12Q1 (land cover)"
    )
    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🎛️ Controls")

        yr_min, yr_max = st.slider(
            "📅 Year range", data_yr_min, data_yr_max,
            (data_yr_min, data_yr_max), key="year_range",
        )

        wilaya_names = [ALL_WILAYAS] + [w["name"] for w in hierarchy]
        selected_wilaya = st.selectbox(
            "🗺️ Wilaya", wilaya_names, index=0, key="wilaya_sel"
        )

        # Commune list is dynamic based on selected wilaya. Options are ADM2
        # codes, not names — commune names are not unique across Algeria.
        commune_labels = {ALL_COMMUNES: ALL_COMMUNES}
        commune_options: List = [ALL_COMMUNES]
        if selected_wilaya != ALL_WILAYAS:
            w_entry = next((w for w in hierarchy if w["name"] == selected_wilaya), None)
            for c in (w_entry["communes"] if w_entry else []):
                commune_options.append(c["code"])
                commune_labels[c["code"]] = c["name"]

        selected_commune = st.selectbox(
            "🏘️ Commune",
            commune_options,
            index=0,
            key="commune_sel",
            format_func=lambda v: commune_labels.get(v, str(v)),
            disabled=(selected_wilaya == ALL_WILAYAS),
        )
        commune_code = None if selected_commune == ALL_COMMUNES else int(selected_commune)
        commune_name = commune_labels.get(selected_commune, ALL_COMMUNES)

        categories = st.multiselect(
            "🌳 Land-cover types",
            ALL_COVERS,
            default=["Forest", "Shrubland", "Cropland"],
            key="lc_cats",
        )

        st.markdown("---")

    cat_key = tuple(c for c in ALL_COVERS if c in categories)

    # ── Filter ────────────────────────────────────────────────────────────────
    df_burn = with_selected_total(
        filter_burn(b1, b2, selected_wilaya, commune_code, yr_min, yr_max), categories
    )
    df_lc = filter_lc(lc1, lc2, selected_wilaya, commune_code, yr_min, yr_max)

    if commune_code is not None:
        scope_label = f"{commune_name} ({selected_wilaya})"
    elif selected_wilaya != ALL_WILAYAS:
        scope_label = selected_wilaya
    else:
        scope_label = "Algeria"
    title_suffix = f" — {scope_label}"

    # ── KPI row ───────────────────────────────────────────────────────────────
    total_burned = float(df_burn[SEL_COL].sum())
    annual_totals = df_burn.groupby("year")[SEL_COL].sum()
    annual_totals = annual_totals[annual_totals > 0]
    has_fire = not annual_totals.empty

    peak_year = int(annual_totals.idxmax()) if has_fire else "—"
    peak_km2  = float(annual_totals.max()) if has_fire else 0.0

    monthly_totals = df_burn.groupby("month")[SEL_COL].sum()
    monthly_totals = monthly_totals[monthly_totals > 0]
    peak_month = (MONTH_NAMES.get(int(monthly_totals.idxmax()), "—")
                  if not monthly_totals.empty else "—")

    n_communes = count_communes_with_fire(
        selected_wilaya, commune_code, yr_min, yr_max, cat_key
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔥 Burned area", f"{total_burned:,.0f} km²")
    k2.metric("📅 Peak year", str(peak_year),
              f"{peak_km2:,.0f} km²" if has_fire else None,
              delta_color="off")
    k3.metric("📆 Peak fire month", peak_month)
    k4.metric("🏘️ Communes with fire", f"{n_communes:,}")

    st.caption(
        f"Scope: **{scope_label}** · {yr_min}–{yr_max} · "
        f"cover types: **{cover_scope_label(categories)}**"
    )

    st.markdown("---")

    # A segmented control rather than st.tabs: Streamlit builds every tab's
    # body, hidden ones included, and a Plotly map first painted into a
    # zero-width container falls back to Plotly's default 700 px width and
    # never re-fits. On a phone the map was then drawn wider than its column,
    # leaving the fires outside the visible slice. Only the chosen section is
    # built here, which also stops shipping both sections' charts every rerun.
    # A radio rather than st.segmented_control: AppTest cannot drive a
    # segmented control on this Streamlit line — every run after one exists
    # raises inside its own widget-state serialisation — which would leave the
    # whole ignition half untestable.
    BURN_VIEW, IGN_VIEW = "🔥 Burned area", "🎯 Ignitions"
    EVENT_VIEW, WX_VIEW = "📇 Fire events", "🌡️ Fire weather"
    view = st.radio(
        "Section", [BURN_VIEW, IGN_VIEW, EVENT_VIEW, WX_VIEW], horizontal=True,
        key="main_view", label_visibility="collapsed",
    )

    if view == BURN_VIEW:
        if yr_max > burn_yr_max:
            st.caption(
                f"⚠️ Burned-area data (MODIS MCD64A1) ends in **{burn_yr_max}**. "
                f"Years {burn_yr_max + 1}–{yr_max} appear under **Ignitions** only."
            )
        if partial and yr_min <= partial[0] <= yr_max:
            st.caption(
                f"⚠️ **{partial[0]} is incomplete** — MCD64A1 covers it only "
                f"through **{MONTH_NAMES[partial[1]]} {partial[0]}**, so its "
                f"annual total is not comparable with earlier years and should "
                f"be excluded from any trend fit. Its bar is greyed below."
            )

        # ── Row 1: Map + Annual bar ──────────────────────────────────────────
        col_map, col_bar = st.columns([1.3, 1])

        with col_map:
            st.subheader("🗺️ Wilaya Fire Map")
            metric_label = st.radio(
                "Map metric", list(MAP_METRICS), horizontal=True,
                key="burn_metric", label_visibility="collapsed",
                help="**Share of burnable land** divides by the forest, "
                     "shrubland and cropland each wilaya actually holds, so it "
                     "measures fire regime rather than size. **Total burned "
                     "area** is the raw km² — useful, but it ranks wilayas "
                     "largely by how big they are. **Years with fire** counts "
                     "how often a wilaya burns at all.",
            )
            metric = MAP_METRICS[metric_label]
            st.plotly_chart(
                build_choropleth(yr_min, yr_max, selected_wilaya, cat_key,
                                 metric),
                key="burn_map", **STRETCH,
            )
            if metric == "rate":
                st.caption(
                    "Cumulative burned area divided by the wilaya's own "
                    f"burnable land, per year. Wilayas with under "
                    f"{MIN_BURNABLE_KM2:.0f} km² of the selected cover types "
                    "are shown as no data — the ratio is meaningless there. "
                    "An incomplete final year is excluded from both sides."
                )
            elif metric == "total":
                st.caption(
                    "Raw km². A large wilaya burns more because it is large: "
                    "Tlemcen and Sidi Bel Abbes rank high here but sit near "
                    "12% of their burnable land, while Blida is tenth on area "
                    "and first on share."
                )

        with col_bar:
            st.subheader("📊 Annual Burned Area")
            st.plotly_chart(
                chart_annual_bar(df_burn, title_suffix,
                                 partial[0] if partial else None),
                key="burn_annual", **STRETCH)
            st.plotly_chart(chart_trend_line(df_burn, title_suffix),
                            key="burn_trend", **STRETCH)

        # ── Row 2: Recurrence ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔁 Fire Recurrence")
        st.plotly_chart(
            chart_recurrence(yr_min, yr_max, cat_key, selected_wilaya,
                             title_suffix),
            key="burn_recurrence", **STRETCH,
        )
        st.caption(
            "How often a place burns, not how much it lost. Totals point at "
            "the single worst season; recurrence points at where the fire "
            "regime keeps returning, which is what fuel management acts on."
        )

        # ── Row 3: Seasonality + Land cover ─────────────────────────────────
        st.markdown("---")
        col_season, col_cover = st.columns(2)

        with col_season:
            st.subheader("📆 Fire Seasonality")
            st.plotly_chart(chart_monthly(df_burn, title_suffix),
                            key="burn_monthly", **STRETCH)
            st.plotly_chart(chart_seasonal(df_burn, title_suffix),
                            key="burn_seasonal", **STRETCH)

        with col_cover:
            st.subheader("🌳 Land Cover Analysis")
            st.plotly_chart(chart_burn_by_type(df_burn, categories, title_suffix),
                            key="burn_by_type", **STRETCH)
            st.plotly_chart(chart_lc_composition(df_lc, categories, title_suffix),
                            key="lc_composition", **STRETCH)

    elif view == IGN_VIEW:
        render_ignition_section(
            load_ignitions(), df_burn, selected_wilaya, commune_code,
            yr_min, yr_max, title_suffix,
        )

    elif view == EVENT_VIEW:
        render_event_catalogue(
            load_ignitions(), selected_wilaya, commune_code,
            yr_min, yr_max, title_suffix,
        )

    else:
        render_weather_section(selected_wilaya, yr_min, yr_max, title_suffix)

    # ── Data notes ────────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("ℹ️ Data notes and caveats"):
        st.markdown(
            f"""
- **Burned area** — MODIS **MCD64A1** (500 m). Fires smaller than roughly one
  500 m pixel are not detected, so totals are a **lower bound**, and small
  agricultural or urban-edge fires are systematically under-represented.
- **Land cover** — MODIS **MCD12Q1**, reported per year and intersected with the
  burned-area mask.
- **Administrative units** — burned area is aggregated on the **pre-2019
  48-wilaya** scheme. The boundary file uses the current **58-wilaya** scheme, so
  the 10 wilayas created in 2019 are shown as *no data* (grey); their fires are
  counted within their former parent wilayas.
- **Communes** are identified by `ADM2_CODE`. 40 commune names are shared by more
  than one wilaya, so name-based selection is not reliable.
- **Coverage** — {data_yr_min}–{data_yr_max}. The August **2021** Kabylie fires
  fall outside this window.
- **Ignitions** — NASA FIRMS active-fire detections (MODIS `MCD14ML` 1 km and/or
  VIIRS 375 m), clustered in space and time (1.5 km / 5 days) into fire events;
  the earliest detection of each event is taken as its ignition.
- **Ignition filtering** — Algeria's Saharan oil and gas fields are the single
  largest source of thermal anomalies in the archive, so four filters are
  applied: FIRMS `type` must be *presumed vegetation fire* (this alone removes
  about **64%** of detections), low-confidence detections are dropped, grid
  cells burning in more than 24 distinct months are treated as static sources,
  and events lasting over 30 days are discarded. Finally a **vegetation mask**
  removes ignitions in administrative units with under 5% burnable cover — the
  hyper-arid south has nothing to burn, so detections there are industrial or
  surface artefacts rather than wildfires.
- Note that `acq_time` reflects fixed satellite overpasses (Terra ~10:30/22:30,
  Aqua and VIIRS ~13:30/01:30 local), so time of day carries **no** information
  about how a fire started.
            """
        )

    # ── Sidebar exports ───────────────────────────────────────────────────────
    with st.sidebar:
        st.caption("Export")
        st.download_button(
            "⬇️ Burned area CSV",
            to_csv_bytes(df_burn),
            file_name="burned_area_selection.csv",
            mime="text/csv",
        )
        st.download_button(
            "⬇️ Land cover CSV",
            to_csv_bytes(df_lc),
            file_name="landcover_selection.csv",
            mime="text/csv",
        )

    st.caption(
        "Algeria Wildfire Analysis · Built with Streamlit + Plotly · "
        "Data: NASA MODIS"
    )


if __name__ == "__main__":
    main()
