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

Author: Z.Matougui
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


@st.cache_data(show_spinner=False)
def build_choropleth(
    yr_min: int, yr_max: int, wilaya: str, categories: Tuple[str, ...],
) -> go.Figure:
    """Choropleth of burned area per wilaya for the current selection.

    Cached on its scalar arguments: the geometry is re-serialised only when the
    selection actually changes, not on every widget interaction.
    """
    cols = selected_burn_cols(categories)
    if not cols:
        return _empty_fig("Select at least one land-cover type")

    b1, _, _, _, _ = load_data()
    prov = load_provinces()

    window = b1[(b1["year"] >= yr_min) & (b1["year"] <= yr_max)]
    totals = (
        window.assign(**{SEL_COL: window[cols].sum(axis=1)})
        .groupby("ADM1_NAME", observed=True)[SEL_COL].sum()
        .reset_index()
    )
    totals["ADM1_NAME"] = totals["ADM1_NAME"].astype(str)

    frame = pd.DataFrame({"ADM1_NAME": prov["names"]}).merge(
        totals, on="ADM1_NAME", how="left"
    )
    known = frame[frame[SEL_COL].notna()]
    unknown = frame[frame[SEL_COL].isna()]

    # Nothing to colour: the selected years lie outside the burned-area record.
    # An all-grey map with no explanation looks like a rendering failure.
    if known.empty:
        return _empty_fig(
            f"No burned-area data for {yr_min}–{yr_max}<br>"
            f"<sub>MCD64A1 covers {int(b1['year'].min())}–{int(b1['year'].max())}</sub>"
        )

    fig = px.choropleth_map(
        known,
        geojson=prov["geojson"],
        locations="ADM1_NAME",
        featureidkey="properties.ADM1_NAME",
        color=SEL_COL,
        hover_name="ADM1_NAME",
        hover_data={SEL_COL: ":.1f"},
        map_style="carto-positron",
        opacity=0.75,
        color_continuous_scale="YlOrRd",
        labels={SEL_COL: "Burned (km²)"},
        title=f"Burned Area by Wilaya ({yr_min}–{yr_max})",
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
            title="km²", len=0.7, thickness=12,
            x=0.98, xanchor="right", y=0.5, yanchor="middle",
            bgcolor="rgba(255,255,255,0.75)", outlinewidth=0,
        ),
        showlegend=False,
    )
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
    if df.empty or "footprint_km2" not in df.columns:
        return _empty_fig("No ignitions in this selection")
    sizes = df.loc[df["footprint_km2"] > 0, "footprint_km2"]
    if sizes.empty:
        return _empty_fig("No sized events in this selection")
    fig = px.histogram(
        sizes, x="footprint_km2", nbins=40, log_y=True,
        title=f"Fire Event Size Distribution{suffix}",
        labels={"footprint_km2": "Detected footprint (km²)"},
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
        "is its ignition. Detected footprint is a lower bound — it counts only "
        "pixels flagged at an overpass, so it is not the true burned area. "
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
        "**Author:** Z.Matougui · "
        "**Data:** MODIS MCD64A1 (burned area) + MCD12Q1 (land cover) via Google Earth Engine"
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
    view = st.radio(
        "Section", [BURN_VIEW, IGN_VIEW], horizontal=True,
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
            st.subheader("🗺️ Wilaya Burned Area Map")
            st.plotly_chart(
                build_choropleth(yr_min, yr_max, selected_wilaya, cat_key),
                key="burn_map", **STRETCH,
            )

        with col_bar:
            st.subheader("📊 Annual Burned Area")
            st.plotly_chart(
                chart_annual_bar(df_burn, title_suffix,
                                 partial[0] if partial else None),
                key="burn_annual", **STRETCH)
            st.plotly_chart(chart_trend_line(df_burn, title_suffix),
                            key="burn_trend", **STRETCH)

        # ── Row 2: Seasonality + Land cover ─────────────────────────────────
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

    else:
        render_ignition_section(
            load_ignitions(), df_burn, selected_wilaya, commune_code,
            yr_min, yr_max, title_suffix,
        )

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
        "Data: NASA MODIS via Google Earth Engine"
    )


if __name__ == "__main__":
    main()
