"""
prepare_ignitions.py
────────────────────
Derives wildfire **ignition points** from NASA FIRMS active-fire detections
and writes them as Parquet for the Streamlit dashboard.

FIRMS gives one row per satellite *detection*, not per fire. A single fire
produces many detections across several overpasses and days. This script
groups detections into fire **events** by space-time clustering, then takes
the earliest detection of each event as its ignition.

Pipeline
    1. Load FIRMS CSV(s)                     (MODIS MCD14ML and/or VIIRS)
    2. Quality filter                        (type, confidence)
    3. Persistent-source removal             (gas flares — critical in Algeria)
    4. Space-time clustering                 (union-find over a rolling window)
    5. Ignition = earliest detection per cluster
    6. Spatial join to FAO GAUL 2015 ADM1/ADM2
    7. Write data/ignitions.parquet

Expected input (place next to this script):
    fire_archive_M-C61_*.csv     MODIS  MCD14ML   2001–2020
    fire_archive_SV-C2_*.csv     VIIRS  S-NPP 375 m (optional, 2012+)

    Download: https://firms.modaps.eosdis.nasa.gov/download/
      Region "Algeria" (or a bounding box), full archive, CSV format.

Boundaries (must match the burned-area tables — FAO GAUL 2015):
    data/gaul_adm2.geojson       exported by section 6 of gee_export.js

Output → data/
    ignitions.parquet            one row per ignition

Usage:
    pip install pandas pyarrow geopandas shapely numpy scipy
    python prepare_ignitions.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Windows consoles default to cp1252 and cannot encode this script's output.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# ── paths ──────────────────────────────────────────────────────────────────
INPUT_DIR  = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

BOUNDARY_FILE  = OUTPUT_DIR / "gaul_adm2.geojson"   # FAO GAUL 2015 ADM1 + ADM2
OUTPUT_FILE    = OUTPUT_DIR / "ignitions.parquet"

# FIRMS downloads land as a zipped shapefile or CSV; look in both the script
# directory and data/.
SEARCH_DIRS = [INPUT_DIR, OUTPUT_DIR]

# ── which FIRMS products to use ────────────────────────────────────────────
#   M-C61   MODIS Terra + Aqua, 1 km       2000-11 → present
#   SV-C2   VIIRS S-NPP, 375 m             2012-01 → present
#   J1V-C2  VIIRS NOAA-20, 375 m           2018-01 → present
#   J2V-C2  VIIRS NOAA-21, 375 m           2023-01 → present
#
# IMPORTANT — do not mix sensors for trend analysis. Each new satellite adds
# detection capability, so a combined record contains a large *artificial*
# upward step every time one comes online (2012, 2018, 2023). An ignition-count
# trend computed on the mix measures the satellite constellation, not Algeria's
# fire regime. MODIS alone is homogeneous back to 2000 and shares a sensor
# family with the MCD64A1 burned-area product, so it is the default.
PRODUCTS = ("M-C61",)

# Near-real-time rows extend coverage to today but skip the quality control
# applied to the archive: no `type` field, so no static-source screening, and
# no post-hoc reprocessing. Every ignition carries a `source` column recording
# which it came from, and the dashboard marks affected years as provisional.
INCLUDE_NRT = True

# ── tuning ─────────────────────────────────────────────────────────────────
# Space-time clustering. Two detections belong to the same fire event if they
# are within EPS_KM of each other and within EPS_DAYS in time. Defaults follow
# common practice for MODIS 1 km active fire (e.g. Global Fire Atlas uses a
# 1-pixel / multi-day flood fill); tighten EPS_KM to ~0.75 for VIIRS-only runs.
EPS_KM   = 1.5
EPS_DAYS = 5

# Quality filters
MODIS_MIN_CONFIDENCE = 30           # 0–100; <30 is "low confidence"
VIIRS_DROP_CONFIDENCE = {"l"}       # VIIRS uses l / n / h

# Persistent-source (gas flare) removal. Algeria's Saharan oil and gas fields
# — Hassi Messaoud, Hassi R'Mel, In Amenas — flare continuously and generate
# year-round FIRMS detections that are not wildfires. Any 0.01° cell active in
# more than FLARE_MIN_MONTHS distinct year-months is treated as a static
# source and dropped. Set to 0 to disable.
FLARE_CELL_DEG    = 0.01
FLARE_MIN_MONTHS  = 24

# Second pass on the same problem, applied after clustering. A flare that
# wanders across neighbouring grid cells survives the cell test above but still
# produces an "event" lasting months. No Algerian vegetation fire burns for 30
# days: in this archive every event above that threshold sits in the Saharan
# hydrocarbon belt (Ouargla, Illizi, El Oued) at a median latitude of 31.7,
# against 36.4 for everything else. Set to 0 to disable.
MAX_EVENT_DAYS = 30

# Vegetation mask. The Saharan south has essentially nothing to burn, so any
# thermal anomaly there is industrial, agricultural or a desert surface
# artefact — not a wildfire. Rather than cut on latitude, which is arbitrary and
# would discard genuinely vegetated southern pockets, drop ignitions in
# administrative units whose burnable cover (forest + shrubland + cropland from
# MCD12Q1) falls below this share of their area.
#
# The threshold is not sensitive: in this dataset eight wilayas sit at 0.0-0.6%
# burnable and the next one up is at 8.4%, so anything between 1 and 8 removes
# exactly the same units. Ouargla (0.1%) and Illizi (0.4%) alone hold 4,007
# "ignitions" — Hassi Messaoud and the In Amenas gas field.
MIN_BURNABLE_PCT = 5.0

SEASON_MAP = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring",  4: "Spring", 5: "Spring",
              6: "Summer",  7: "Summer", 8: "Summer",
              9: "Autumn", 10: "Autumn", 11: "Autumn"}

# Algeria is UTC+1 all year (no daylight saving).
UTC_OFFSET_HOURS = 1


# ── 1. load ────────────────────────────────────────────────────────────────
def _wanted(member: str) -> bool:
    """Does this file belong to a product we asked for?"""
    stem = Path(member).name
    if not stem.startswith(("fire_archive_", "fire_nrt_")):
        return False
    if stem.startswith("fire_nrt_") and not INCLUDE_NRT:
        return False
    return any(f"_{p}_" in stem for p in PRODUCTS)


def _normalise(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """FIRMS ships CSV in lower case and shapefiles in upper case."""
    df = df.drop(columns=[c for c in ("geometry",) if c in df.columns])
    df.columns = [c.strip().lower() for c in df.columns]
    # VIIRS names its brightness bands differently from MODIS.
    if "bright_ti4" in df.columns:
        df = df.rename(columns={"bright_ti4": "brightness",
                                "bright_ti5": "bright_t31"})
    if "instrument" not in df.columns:
        df["instrument"] = "VIIRS" if "bright_ti4" in df.columns else "MODIS"
    df["product"] = label
    return df


def load_firms() -> pd.DataFrame:
    """Read every requested FIRMS file — zipped shapefile, .shp or .csv."""
    import geopandas as gpd

    frames, seen = [], []

    for d in SEARCH_DIRS:
        # a) zipped shapefile downloads, read in place via the zip:// handler
        for z in sorted(glob.glob(str(d / "DL_FIRE_*.zip"))):
            import zipfile
            with zipfile.ZipFile(z) as zf:
                members = [m for m in zf.namelist()
                           if m.lower().endswith(".shp") and _wanted(m)]
            for m in members:
                g = gpd.read_file(f"zip://{z}!{m}")
                label = Path(m).stem
                frames.append(_normalise(pd.DataFrame(g), label))
                seen.append((label, len(g)))

        # b) loose shapefiles / CSVs
        for pat, reader in (("fire_*.shp", gpd.read_file),
                            ("fire_*.csv", pd.read_csv)):
            for f in sorted(glob.glob(str(d / pat))):
                if not _wanted(f):
                    continue
                obj = reader(f)
                label = Path(f).stem
                frames.append(_normalise(pd.DataFrame(obj), label))
                seen.append((label, len(obj)))

    # Downloading the section-7 export and forgetting the flag would silently
    # do nothing, so say when NRT files are present but being skipped.
    if not INCLUDE_NRT:
        import zipfile
        skipped = []
        for d in SEARCH_DIRS:
            for f in sorted(glob.glob(str(d / "fire_nrt_*.csv"))
                            + glob.glob(str(d / "fire_nrt_*.shp"))):
                if any(f"_{p}_" in Path(f).stem for p in PRODUCTS):
                    skipped.append(Path(f).name)
            # NRT rows usually arrive as members inside the archive download.
            for z in sorted(glob.glob(str(d / "DL_FIRE_*.zip"))):
                try:
                    with zipfile.ZipFile(z) as zf:
                        for m in zf.namelist():
                            stem = Path(m).name
                            if (m.lower().endswith(".shp")
                                    and stem.startswith("fire_nrt_")
                                    and any(f"_{p}_" in stem for p in PRODUCTS)):
                                skipped.append(f"{Path(z).name}!{stem}")
                except zipfile.BadZipFile:
                    continue
        if skipped:
            print(f"  note: {len(skipped)} near-real-time file(s) present but "
                  f"skipped (INCLUDE_NRT is False):")
            for name in skipped:
                print(f"        {name}")
            print("        set INCLUDE_NRT = True to extend the record with them")

    if not frames:
        sys.exit(
            f"No FIRMS files found for products {PRODUCTS} in "
            f"{[str(d) for d in SEARCH_DIRS]}.\n"
            "Download the Algeria archive from "
            "https://firms.modaps.eosdis.nasa.gov/download/ (shapefile or CSV) "
            "and leave the DL_FIRE_*.zip in data/."
        )

    for label, n in seen:
        print(f"  read {label:45s} {n:>9,} detections")

    if len({lbl.split('_')[2] for lbl, _ in seen}) > 1:
        print("\n  !! WARNING: more than one satellite product loaded.\n"
              "     Detection capability jumps each time a new satellite comes\n"
              "     online, which creates an artificial upward trend in\n"
              "     ignition counts. Use a single product for trend analysis.\n")

    out = pd.concat(frames, ignore_index=True)
    print(f"  total {'':45s} {len(out):>9,} detections")
    return out


# ── 2. quality filter ──────────────────────────────────────────────────────
def quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)

    # `type` 0 = presumed vegetation fire; 1 = volcano, 2 = other static land
    # source, 3 = offshore. Only 0 is a wildfire candidate.
    #
    # Not every source carries the field: FIRMS near-real-time files and the
    # MOD14A1/MYD14A1 export from gee_export.js section 7 omit it. Concatenating
    # them with archive rows leaves those cells NaN, and a bare `== 0` test would
    # silently discard every one of them. Keep rows that never had the field and
    # say so, rather than dropping data the user just went and fetched.
    if "type" in df.columns:
        kind = pd.to_numeric(df["type"], errors="coerce")
        unscreened = int(kind.isna().sum())
        df = df[kind.isna() | (kind == 0)]
        print(f"  type==0 (vegetation fire)      : dropped {n0 - len(df):>7,}")
        if unscreened:
            print(f"    {unscreened:,} rows carry no `type` field (NRT / GEE "
                  f"sources) - kept, but not screened for static sources")

    n1 = len(df)
    conf = df["confidence"]
    if conf.dtype == object:
        keep = ~conf.astype(str).str.strip().str.lower().isin(VIIRS_DROP_CONFIDENCE)
    else:
        keep = conf >= MODIS_MIN_CONFIDENCE
    df = df[keep]
    print(f"  confidence threshold           : dropped {n1 - len(df):>7,}")

    return df.copy()


def drop_persistent_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Remove gas flares and other static hot spots."""
    if not FLARE_MIN_MONTHS:
        return df

    cell = (
        (df["latitude"] / FLARE_CELL_DEG).round().astype("int32").astype(str)
        + "_"
        + (df["longitude"] / FLARE_CELL_DEG).round().astype("int32").astype(str)
    )
    ym = df["acq_dt"].dt.year * 12 + df["acq_dt"].dt.month
    months_active = pd.DataFrame({"cell": cell, "ym": ym}).groupby("cell")["ym"].nunique()

    flares = set(months_active[months_active > FLARE_MIN_MONTHS].index)
    keep = ~cell.isin(flares)
    print(f"  persistent sources             : dropped {(~keep).sum():>7,} "
          f"detections in {len(flares):,} cells")
    return df[keep].copy()


# ── 3. space-time clustering ───────────────────────────────────────────────
class _Union:
    """Union-find with path compression."""

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, i: int) -> int:
        p = self.parent
        root = i
        while p[root] != root:
            root = p[root]
        while p[i] != root:          # path compression
            p[i], i = root, p[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cluster_events(df: pd.DataFrame) -> pd.DataFrame:
    """Assign an `event_id` by space-time flood fill.

    Detections are sorted by time; for each day we build a KD-tree over the
    following EPS_DAYS window and link everything within EPS_KM. That keeps the
    neighbour search local instead of O(n^2) over the whole 20-year archive.
    """
    from scipy.spatial import cKDTree

    df = df.sort_values("acq_dt", kind="mergesort").reset_index(drop=True)

    # Local equirectangular projection — accurate well below 1% at these scales.
    lat0 = float(df["latitude"].mean())
    x = df["longitude"].to_numpy() * 111.320 * np.cos(np.radians(lat0))
    y = df["latitude"].to_numpy() * 110.574
    xy = np.column_stack([x, y])

    day = (df["acq_dt"].dt.normalize() - df["acq_dt"].dt.normalize().min()).dt.days
    day = day.to_numpy()

    uf = _Union(len(df))
    order = np.argsort(day, kind="mergesort")
    # start index of each day value, for slicing the rolling window
    unique_days, first_idx = np.unique(day[order], return_index=True)
    bounds = dict(zip(unique_days.tolist(), first_idx.tolist()))
    ends = list(first_idx[1:]) + [len(order)]
    day_end = dict(zip(unique_days.tolist(), ends))

    for d in unique_days:
        lo = bounds[d]
        # window covers days [d, d + EPS_DAYS]
        hi_day = max((u for u in unique_days if u <= d + EPS_DAYS), default=d)
        hi = day_end[hi_day]
        win = order[lo:hi]
        if len(win) < 2:
            continue
        tree = cKDTree(xy[win])
        pairs = tree.query_pairs(EPS_KM, output_type="ndarray")
        if len(pairs):
            a = win[pairs[:, 0]]
            b = win[pairs[:, 1]]
            ok = np.abs(day[a] - day[b]) <= EPS_DAYS
            for i, j in zip(a[ok], b[ok]):
                uf.union(int(i), int(j))

    roots = np.array([uf.find(i) for i in range(len(df))], dtype=np.int64)
    df["event_id"] = pd.factorize(roots)[0].astype("int32")
    print(f"  clustered {len(df):,} detections into {df['event_id'].nunique():,} events")
    return df


# ── 4. ignitions ───────────────────────────────────────────────────────────
def derive_ignitions(df: pd.DataFrame) -> pd.DataFrame:
    """Earliest detection of each event, plus event-level attributes."""
    # Pixel footprint from the FIRMS along-scan / along-track dimensions.
    if {"scan", "track"} <= set(df.columns):
        df["px_km2"] = df["scan"] * df["track"]
    else:
        df["px_km2"] = np.where(df["instrument"].str.upper().str.startswith("V"),
                                0.375 ** 2, 1.0)

    events = df.groupby("event_id", sort=False).agg(
        n_detections=("event_id", "size"),
        first_dt=("acq_dt", "min"),
        last_dt=("acq_dt", "max"),
        frp_max_mw=("frp", "max"),
        frp_sum_mw=("frp", "sum"),
        footprint_km2=("px_km2", "sum"),
    )
    events["duration_days"] = (
        (events["last_dt"] - events["first_dt"]).dt.total_seconds() / 86400.0
    )

    if MAX_EVENT_DAYS:
        too_long = events["duration_days"] > MAX_EVENT_DAYS
        if too_long.any():
            print(f"  residual static sources          : dropped "
                  f"{int(too_long.sum()):>5,} events lasting > {MAX_EVENT_DAYS} days")
            events = events[~too_long]
            df = df[df["event_id"].isin(events.index)]

    # The ignition is the earliest detection; ties broken by strongest FRP.
    idx = (
        df.sort_values(["event_id", "acq_dt", "frp"], ascending=[True, True, False])
        .groupby("event_id", sort=False)
        .head(1)
        .set_index("event_id")
    )

    ign = idx.join(events, how="left").reset_index()
    ign = ign.rename(columns={"latitude": "lat", "longitude": "lon"})

    # Provenance: archive rows are quality-controlled and screened for static
    # sources; near-real-time rows are neither. Keeping this per ignition lets
    # the dashboard mark the affected years instead of silently blending them.
    ign["source"] = (
        ign["product"].astype(str)
        .str.startswith("fire_archive_")
        .map({True: "archive", False: "nrt"})
    )

    local = ign["acq_dt"] + pd.Timedelta(hours=UTC_OFFSET_HOURS)
    ign["date"]       = local.dt.date.astype("datetime64[ns]")
    ign["year"]       = local.dt.year.astype("int16")
    ign["month"]      = local.dt.month.astype("int8")
    ign["doy"]        = local.dt.dayofyear.astype("int16")
    ign["hour_local"] = local.dt.hour.astype("int8")
    ign["season"]     = ign["month"].map(SEASON_MAP).astype("category")
    return ign


# ── 5. admin join ──────────────────────────────────────────────────────────
def join_admin(ign: pd.DataFrame) -> pd.DataFrame:
    """Point-in-polygon join to FAO GAUL 2015 ADM1 + ADM2.

    GAUL 2015 is exactly what the burned-area tables are aggregated on, so the
    codes join directly. There is deliberately no fallback boundary file: any
    other source uses the post-2019 58-wilaya scheme, which would place
    ignitions in wilayas the burned-area data does not contain and silently
    break the dashboard's wilaya filter. Better to stop and ask for the export.
    """
    import geopandas as gpd

    if not BOUNDARY_FILE.exists():
        sys.exit(
            f"Missing {BOUNDARY_FILE}.\n"
            "Run section 6 of gee_export.js in the Earth Engine Code Editor to "
            "export the FAO GAUL 2015 Algeria boundaries, then put "
            "gaul_adm2.geojson in data/ and re-run this script."
        )

    adm = gpd.read_file(BOUNDARY_FILE)
    cols = ["ADM1_CODE", "ADM1_NAME", "ADM2_CODE", "ADM2_NAME"]
    level = "GAUL 2015 ADM1 + ADM2"

    print(f"  joining against {level}")
    adm = adm[cols + ["geometry"]]
    if adm.crs is None or adm.crs.to_epsg() != 4326:
        adm = adm.to_crs(epsg=4326)

    pts = gpd.GeoDataFrame(
        ign, geometry=gpd.points_from_xy(ign["lon"], ign["lat"]), crs="EPSG:4326"
    )
    joined = gpd.sjoin(pts, adm, how="left", predicate="within").drop(
        columns=["geometry", "index_right"]
    )

    outside = joined["ADM1_NAME"].isna().sum()
    if outside:
        print(f"  {outside:,} ignitions fell outside the boundary file - dropped")
        joined = joined[joined["ADM1_NAME"].notna()].copy()

    # Reconcile wilaya names against the burned-area tables so the dashboard's
    # wilaya filter matches. Anything unmatched is reported, never silently kept.
    burn = pd.read_parquet(OUTPUT_DIR / "burned_area_adm1.parquet",
                           columns=["ADM1_CODE", "ADM1_NAME"]).drop_duplicates()
    known = set(burn["ADM1_NAME"].astype(str))
    joined["ADM1_NAME"] = joined["ADM1_NAME"].astype(str)
    unmatched = ~joined["ADM1_NAME"].isin(known)
    if unmatched.any():
        names = sorted(joined.loc[unmatched, "ADM1_NAME"].unique())
        print(f"  {unmatched.sum():,} ignitions in {len(names)} wilayas absent "
              f"from the burned-area tables (post-2019 units): {names[:6]}"
              f"{' …' if len(names) > 6 else ''}")
        print("    They are kept, but the dashboard's wilaya filter cannot "
              "select them.")

    if "ADM1_CODE" not in joined.columns:
        code_of = dict(zip(burn["ADM1_NAME"].astype(str), burn["ADM1_CODE"]))
        joined["ADM1_CODE"] = joined["ADM1_NAME"].map(code_of).fillna(-1)
    if "ADM2_CODE" not in joined.columns:
        joined["ADM2_CODE"] = -1
        joined["ADM2_NAME"] = "(not resolved)"

    joined["ADM1_CODE"] = joined["ADM1_CODE"].astype("int32")
    joined["ADM2_CODE"] = joined["ADM2_CODE"].astype("int32")
    joined["ADM1_NAME"] = joined["ADM1_NAME"].astype("category")
    joined["ADM2_NAME"] = joined["ADM2_NAME"].astype("category")
    return joined


def drop_non_vegetated(ign: pd.DataFrame) -> pd.DataFrame:
    """Remove ignitions where there is no burnable vegetation.

    Uses commune-level land cover when the ignitions carry a resolved
    ADM2_CODE, and falls back to wilaya level otherwise. Commune level is much
    sharper: a large Saharan wilaya can hold a small irrigated pocket that a
    wilaya-level average hides.
    """
    if not MIN_BURNABLE_PCT:
        return ign

    use_adm2 = (ign["ADM2_CODE"] > 0).any()
    src = "landcover_adm2.parquet" if use_adm2 else "landcover_adm1.parquet"
    key = "ADM2_CODE" if use_adm2 else "ADM1_NAME"

    lc = pd.read_parquet(OUTPUT_DIR / src)
    veg = lc.groupby(key, observed=True)[
        ["forest_km2", "shrubland_km2", "cropland_km2", "total_km2"]
    ].mean()
    burnable = (veg["forest_km2"] + veg["shrubland_km2"] + veg["cropland_km2"])
    pct = 100.0 * burnable / veg["total_km2"].replace(0, np.nan)

    keep_units = set(pct[pct >= MIN_BURNABLE_PCT].index)
    keep = ign[key].isin(keep_units)

    dropped = ign[~keep]
    if len(dropped):
        by_unit = dropped["ADM1_NAME"].value_counts()
        print(f"  non-vegetated units ({key[:4]} level, "
              f"< {MIN_BURNABLE_PCT:g}% burnable cover):")
        print(f"    dropped {len(dropped):,} ignitions "
              f"({len(dropped) / len(ign) * 100:.1f}%)")
        for name, n in by_unit[by_unit > 0].head(10).items():
            print(f"      {str(name):<16} {n:>6,}")
    return ign[keep].copy()


# ── main ───────────────────────────────────────────────────────────────────
OUT_COLS = [
    "ignition_id", "lat", "lon", "date", "year", "month", "doy", "hour_local",
    "season", "ADM1_CODE", "ADM1_NAME", "ADM2_CODE", "ADM2_NAME",
    "instrument", "satellite", "confidence_raw", "daynight",
    "frp_mw", "frp_max_mw", "frp_sum_mw",
    "n_detections", "duration_days", "footprint_km2", "source",
]


def main() -> None:
    print("1. Loading FIRMS detections")
    df = load_firms()

    # CSV gives acq_date as a string, shapefile as a real date; acq_time is
    # HHMM UTC in both, sometimes as an int that has lost its leading zero.
    day = pd.to_datetime(df["acq_date"], errors="coerce").dt.normalize()
    hhmm = df["acq_time"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(4)
    minutes = (
        pd.to_numeric(hhmm.str[:2], errors="coerce") * 60
        + pd.to_numeric(hhmm.str[2:], errors="coerce")
    )
    df["acq_dt"] = day + pd.to_timedelta(minutes, unit="m")

    bad = df["acq_dt"].isna().sum()
    if bad:
        print(f"  {bad:,} rows with unparseable timestamps dropped")
    df = df[df["acq_dt"].notna()].copy()

    print("\n2. Quality filtering")
    df = quality_filter(df)
    df = drop_persistent_sources(df)
    print(f"  kept                           : {len(df):>7,} detections")

    print("\n3. Space-time clustering "
          f"(eps = {EPS_KM} km / {EPS_DAYS} days)")
    df = cluster_events(df)

    print("\n4. Deriving ignitions")
    ign = derive_ignitions(df)
    print(f"  {len(ign):,} ignitions")

    print("\n5. Joining FAO GAUL 2015 admin units")
    ign = join_admin(ign)

    print("\n6. Vegetation mask")
    ign = drop_non_vegetated(ign)

    ign = ign.sort_values("acq_dt").reset_index(drop=True)
    ign["ignition_id"] = np.arange(1, len(ign) + 1, dtype="int32")
    ign = ign.rename(columns={"frp": "frp_mw", "confidence": "confidence_raw"})
    ign["confidence_raw"] = ign["confidence_raw"].astype(str)
    for c in ("instrument", "satellite", "daynight"):
        if c not in ign.columns:
            ign[c] = "?"
    ign[["instrument", "satellite", "daynight", "source"]] = ign[
        ["instrument", "satellite", "daynight", "source"]
    ].astype("category")
    for c in ("frp_mw", "frp_max_mw", "frp_sum_mw", "duration_days",
              "footprint_km2", "lat", "lon"):
        ign[c] = ign[c].astype("float32")
    ign["n_detections"] = ign["n_detections"].astype("int32")

    out = ign[[c for c in OUT_COLS if c in ign.columns]]
    out.to_parquet(OUTPUT_FILE, index=False)

    print(f"\nOK  {OUTPUT_FILE}  ({OUTPUT_FILE.stat().st_size / 1e6:.2f} MB)")
    print(f"  {len(out):,} ignitions | {out['year'].min()}-{out['year'].max()} | "
          f"{out['ADM1_NAME'].nunique()} wilayas")
    print("\n  Ignitions per year:")
    print(out.groupby("year").size().to_string())


if __name__ == "__main__":
    main()
