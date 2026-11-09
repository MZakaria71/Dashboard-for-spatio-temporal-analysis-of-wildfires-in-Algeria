#!/usr/bin/env python3
"""
prepare_spread.py
─────────────────
Turns the geostationary fire pixels written by fetch_geostationary.py into
fire events that can be followed slot by slot: where a fire started, when to
the nearest 10 or 15 minutes, how fast and in which direction its front
moved, and how much energy it released.

What the FIRMS pipeline cannot do
    prepare_ignitions.py clusters MODIS / VIIRS overpasses, so its event_daily
    table has one row per fire per day and its ignition time is the first
    overpass to catch the fire, hours late in the worst case. Here the same
    question is asked of a sensor that looks every 10 minutes (MTG) or 15
    (MSG), which is what makes spread measurable at all.

Pipeline
    1. Load data/geo_detections.parquet (and the slot log, to tell a slot with
       no fire from a slot that was never read)
    2. Cells: geostationary pixels sit on a fixed grid, so one burning patch
       comes back at identical coordinates every slot. Each distinct position
       is a cell, joined once to its GAUL commune.
    3. Screening, with the thresholds prepare_ignitions.py uses, so the two
       records are filtered alike: cells outside Algeria, in communes with
       under MIN_BURNABLE_PCT burnable cover (the Saharan gas flares), or
       burning in more than FLARE_MIN_MONTHS distinct months are dropped.
    4. Space-time clustering: a cell's detections are cut into runs wherever
       it goes unseen for EPS_HOURS; runs within EPS_KM of each other whose
       time spans come within EPS_HOURS are one fire. Done per platform: MSG
       and MTG are separate records and are never mixed.
    5. Per event, per slot: fire pixels, FRP, newly reached cells, footprint,
       distance of the front from the origin and its rate of advance.
    6. Per event: origin, start, duration, peak FRP, fire radiative energy,
       biomass consumed, fastest spread, heading, and the FIRMS ignition it
       matches with how many hours earlier the geostationary record saw it.

Output -> data/
    geo_events.parquet             one row per event
    geo_event_slots.parquet        one row per event per slot
    geo_event_cells.parquet        one row per event per pixel (arrival time)
    geo_event_detections.parquet   one row per fire pixel per slot

Usage
    pip install pandas pyarrow numpy scipy geopandas shapely
    python prepare_spread.py

What the numbers mean, and do not
    * Footprint is the summed area of pixels flagged as burning. A pixel is
      flagged whether one hectare or all of it burns, so this is the envelope
      the fire stayed inside, not burned area; MCD64A1 remains the burned-area
      record.
    * Spread rate is the advance of the farthest burning pixel from the
      origin over the past ROS_WINDOW_MIN minutes. The front can only move in
      steps of one pixel, so rates come quantised: about 1.2 km for MTG and
      3.6 km for MSG over the fire belt. Read the slower values as "under one
      pixel per hour" rather than as measurements. The peak of that series
      (max_ros_kmh) is the noisier number; mean_ros_kmh, the farthest reach
      divided by the time taken to reach it, averages the steps out and is
      the one to compare fires on.
    * Fire radiative energy integrates FRP over the slots in which the fire was
      seen. Thick smoke and cloud hide a fire, and SEVIRI misses fires below
      roughly 40 to 50 MW altogether, so it is a lower bound. Biomass uses the
      0.368 kg per MJ combustion factor of Wooster et al. (2005).
    * Times are slot starts in UTC. The scan reaches northern Algeria some
      minutes into the slot, the same offset in every slot, so durations and
      rates are unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from fetch_geostationary import PLATFORMS
from prepare_ignitions import FLARE_MIN_MONTHS, MIN_BURNABLE_PCT, UTC_OFFSET_HOURS

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

DATA_DIR = Path(__file__).parent / "data"
DETECTIONS_FILE = DATA_DIR / "geo_detections.parquet"
SLOTS_FILE = DATA_DIR / "geo_slots.parquet"
# The simplified communes are committed, unlike the full-resolution export
# prepare_ignitions.py wants; at ~90 m they are far finer than any
# geostationary pixel, so nothing is lost by using them here.
COMMUNE_FILE = DATA_DIR / "gaul_adm2_simple.geojson"
LANDCOVER_FILE = DATA_DIR / "landcover_adm2.parquet"
IGNITION_FILE = DATA_DIR / "ignitions.parquet"

OUT_EVENTS = DATA_DIR / "geo_events.parquet"
OUT_SLOTS = DATA_DIR / "geo_event_slots.parquet"
OUT_CELLS = DATA_DIR / "geo_event_cells.parquet"
OUT_DETECTIONS = DATA_DIR / "geo_event_detections.parquet"

# ── tuning ─────────────────────────────────────────────────────────────────
# Neighbour distance, about one and a half pixels over the fire belt: adjacent
# pixels link, diagonal ones too, and a one-pixel hole does not split a fire.
EPS_KM: Dict[str, float] = {"MTG": 3.0, "MSG": 6.0}

# How long a fire may go unseen and still be the same fire. Smoke from the
# fire itself, passing cloud and the night-time drop below the detection
# threshold all open gaps of a few hours inside one event.
EPS_HOURS = 6.0

# Seen in fewer slots than this, an event is too short to have a spread to
# measure, and isolated single-slot flags are where false alarms concentrate.
MIN_FIRE_SLOTS = 3

# Same reasoning as prepare_ignitions.py: no Algerian wildfire burns a month.
MAX_EVENT_DAYS = 30

# The window the spread rate is measured over. Shorter is noisier, because the
# front moves in whole pixels.
ROS_WINDOW_MIN = 60

# A FIRMS ignition is the same fire if it lies within LINK_KM of the
# geostationary origin and was first seen within LINK_HOURS of it.
LINK_KM = 10.0
LINK_HOURS = 24.0

# Biomass consumed per unit fire radiative energy (Wooster et al. 2005, J.
# Geophys. Res. 110, D24311), kg per MJ.
BIOMASS_KG_PER_MJ = 0.368

COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


# ── geometry ───────────────────────────────────────────────────────────────
def project_km(lat, lon) -> np.ndarray:
    """Per-point equirectangular km, as prepare_ignitions.cluster_events does."""
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    return np.column_stack([lon * 111.320 * np.cos(np.radians(lat)),
                            lat * 110.574])


def offsets_km(lat, lon, lat0: float, lon0: float):
    """East and north offsets from one origin, in km."""
    dx = (np.asarray(lon) - lon0) * 111.320 * np.cos(np.radians(lat0))
    dy = (np.asarray(lat) - lat0) * 110.574
    return dx, dy


def nominal_pixel_km2(lat, lon, ssp_km: float, sat_lon: float = 0.0) -> np.ndarray:
    """Ground area of a geostationary pixel, for files that carry none.

    A pixel's footprint grows with the square of the slant range and with the
    obliquity of the view: area ~ ssp^2 (r / h)^2 / cos(VZA). Over Algiers that
    turns SEVIRI's 3 km nadir sampling into about 13 km2.
    """
    R, RS, H = 6378.137, 42164.0, 35786.0
    lat = np.radians(np.asarray(lat, dtype="float64"))
    dlon = np.radians(np.asarray(lon, dtype="float64") - sat_lon)
    cosg = np.cos(lat) * np.cos(dlon)
    r = np.sqrt(R ** 2 + RS ** 2 - 2 * R * RS * cosg)
    sin_vza = np.clip(RS * np.sqrt(1 - cosg ** 2) / r, 0, 0.999)
    return ssp_km ** 2 * (r / H) ** 2 / np.sqrt(1 - sin_vza ** 2)


def compass(deg: float) -> Optional[str]:
    if deg is None or not np.isfinite(deg):
        return None
    return COMPASS[int(((deg % 360) + 11.25) // 22.5) % 16]


# ── 1. load ────────────────────────────────────────────────────────────────
def load() -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if not DETECTIONS_FILE.exists():
        sys.exit(f"{DETECTIONS_FILE} not found. Run fetch_geostationary.py first.")
    det = pd.read_parquet(DETECTIONS_FILE)
    det["time"] = pd.to_datetime(det["time"])
    det["platform"] = det["platform"].astype(str)
    unknown = sorted(set(det["platform"]) - set(PLATFORMS))
    if unknown:
        sys.exit(f"Unknown platform(s) {unknown} in {DETECTIONS_FILE}.")

    # Fill missing pixel areas from the viewing geometry.
    det["pixel_km2"] = det["pixel_km2"].astype("float64")
    for plat, spec in PLATFORMS.items():
        gap = (det["platform"] == plat) & det["pixel_km2"].isna()
        if gap.any():
            det.loc[gap, "pixel_km2"] = nominal_pixel_km2(
                det.loc[gap, "lat"], det.loc[gap, "lon"], spec["ssp_km"])

    slots = None
    if SLOTS_FILE.exists():
        slots = pd.read_parquet(SLOTS_FILE)
        slots["time"] = pd.to_datetime(slots["time"])
        slots["platform"] = slots["platform"].astype(str)

    for plat, d in det.groupby("platform"):
        print(f"  {plat}: {len(d):>9,} fire pixels  "
              f"{d['time'].min():%Y-%m-%d %H:%M} -> {d['time'].max():%Y-%m-%d %H:%M}")
    if slots is None:
        print("  no slot log: a slot without fire cannot be told from one never "
              "read, so observation coverage is left blank")
    return det, slots


# ── 2-3. cells and screening ───────────────────────────────────────────────
def build_cells(det: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct pixel position, with its commune."""
    import geopandas as gpd

    det["_la"] = det["lat"].round(4)
    det["_lo"] = det["lon"].round(4)
    cells = (det.groupby(["platform", "_la", "_lo"], sort=False)
                .agg(lat=("lat", "mean"), lon=("lon", "mean"),
                     pixel_km2=("pixel_km2", "median"))
                .reset_index())
    cells["cell"] = np.arange(len(cells), dtype="int64")

    adm = gpd.read_file(COMMUNE_FILE)[
        ["ADM1_CODE", "ADM1_NAME", "ADM2_CODE", "ADM2_NAME", "geometry"]]
    pts = gpd.GeoDataFrame(
        cells[["cell"]], geometry=gpd.points_from_xy(cells["lon"], cells["lat"]),
        crs="EPSG:4326")
    if adm.crs is None:
        adm = adm.set_crs(epsg=4326)
    joined = gpd.sjoin(pts, adm.to_crs(epsg=4326), how="left", predicate="within")
    joined = joined.drop_duplicates("cell").set_index("cell")
    for c in ("ADM1_CODE", "ADM1_NAME", "ADM2_CODE", "ADM2_NAME"):
        cells[c] = cells["cell"].map(joined[c])

    det_cell = det.merge(cells[["platform", "_la", "_lo", "cell"]],
                         on=["platform", "_la", "_lo"], how="left")
    det["cell"] = det_cell["cell"].to_numpy()
    det.drop(columns=["_la", "_lo"], inplace=True)
    return cells


def screen(det: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    n0 = len(det)

    outside = cells["ADM2_CODE"].isna()
    keep_cells = set(cells.loc[~outside, "cell"])
    det = det[det["cell"].isin(keep_cells)]
    print(f"  outside Algeria's communes      : dropped {n0 - len(det):>8,}")

    lc = pd.read_parquet(LANDCOVER_FILE)
    veg = lc.groupby("ADM2_CODE", observed=True)[
        ["forest_km2", "shrubland_km2", "cropland_km2", "total_km2"]].mean()
    pct = 100 * (veg["forest_km2"] + veg["shrubland_km2"] + veg["cropland_km2"]) \
        / veg["total_km2"].replace(0, np.nan)
    burnable = set(pct[pct >= MIN_BURNABLE_PCT].index)
    code = det["cell"].map(cells.set_index("cell")["ADM2_CODE"])
    n1 = len(det)
    det = det[code.isin(burnable)]
    print(f"  under {MIN_BURNABLE_PCT:g}% burnable cover        : "
          f"dropped {n1 - len(det):>8,}")

    if FLARE_MIN_MONTHS:
        ym = det["time"].dt.year * 12 + det["time"].dt.month
        months = ym.groupby(det["cell"]).nunique()
        static = set(months[months > FLARE_MIN_MONTHS].index)
        n2 = len(det)
        det = det[~det["cell"].isin(static)]
        print(f"  persistent sources              : dropped {n2 - len(det):>8,} "
              f"in {len(static):,} cells")
    return det.copy()


# ── 4. clustering ──────────────────────────────────────────────────────────
def cell_runs(det: pd.DataFrame) -> pd.DataFrame:
    """Cut each cell's detections wherever it goes unseen for EPS_HOURS."""
    det.sort_values(["cell", "time"], inplace=True, kind="mergesort")
    gap = det.groupby("cell")["time"].diff() > pd.Timedelta(hours=EPS_HOURS)
    new_cell = det["cell"].ne(det["cell"].shift())
    det["run"] = (new_cell | gap).cumsum().astype("int64") - 1
    return det.groupby("run").agg(
        platform=("platform", "first"), lat=("lat", "first"),
        lon=("lon", "first"), t0=("time", "min"), t1=("time", "max"),
    ).reset_index()


def link_runs(runs: pd.DataFrame, eps_km: float, eps_h: float) -> np.ndarray:
    """Event label per run: connected components of the space-time links.

    Runs are taken in start order, a couple of days at a time; each batch is
    searched against everything that starts before its latest run ends plus
    eps_h, which is every run it could link to. That keeps the neighbour search
    local, as cluster_events does for FIRMS.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    runs = runs.sort_values("t0", kind="mergesort").reset_index(drop=True)
    n = len(runs)
    origin = runs["t0"].min()
    t0 = ((runs["t0"] - origin).dt.total_seconds() / 3600).to_numpy()
    t1 = ((runs["t1"] - origin).dt.total_seconds() / 3600).to_numpy()
    xy = project_km(runs["lat"], runs["lon"])

    step_h = 48.0
    src, dst = [], []
    i = 0
    while i < n:
        j = max(int(np.searchsorted(t0, t0[i] + step_h, side="left")), i + 1)
        hi = float(t1[i:j].max()) + eps_h
        k = int(np.searchsorted(t0, hi, side="right"))
        idx = np.arange(i, k)
        if len(idx) > 1:
            pairs = cKDTree(xy[idx]).query_pairs(eps_km, output_type="ndarray")
            if len(pairs):
                a = idx[pairs.min(axis=1)]
                b = idx[pairs.max(axis=1)]
                # a starts no later than b, so the gap between their spans is
                # b's start minus a's end (negative when they overlap).
                ok = (a < j) & (t0[b] - t1[a] <= eps_h)
                src.append(a[ok])
                dst.append(b[ok])
        i = j

    if src:
        a = np.concatenate(src)
        b = np.concatenate(dst)
    else:
        a = b = np.array([], dtype="int64")
    graph = coo_matrix((np.ones(len(a), dtype="int8"), (a, b)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return pd.Series(labels, index=runs["run"].to_numpy())


def cluster(det: pd.DataFrame) -> pd.DataFrame:
    runs = cell_runs(det)
    labels = []
    offset = 0
    for plat, r in runs.groupby("platform"):
        lab = link_runs(r, EPS_KM[plat], EPS_HOURS)
        labels.append(lab + offset)
        offset += int(lab.max()) + 1
    det["event"] = det["run"].map(pd.concat(labels)).astype("int64")
    print(f"  {len(det):,} fire pixels in {det['event'].nunique():,} events")
    return det


# ── 5-6. per-event metrics ─────────────────────────────────────────────────
def event_metrics(d: pd.DataFrame, cells: pd.DataFrame, slot_min: int,
                  observed: Optional[set]):
    """Slot table, cell table and summary row for one event."""
    slot = pd.Timedelta(minutes=slot_min)
    t_first, t_last = d["time"].min(), d["time"].max()

    # Origin: the FRP-weighted centre of what the first slot saw.
    first = d[d["time"] == t_first]
    w = first["frp_mw"].clip(lower=1e-3)
    lat0 = float(np.average(first["lat"], weights=w))
    lon0 = float(np.average(first["lon"], weights=w))

    # Cells: arrival time, residence, and distance from the origin.
    c = d.groupby("cell").agg(
        lat=("lat", "first"), lon=("lon", "first"),
        pixel_km2=("pixel_km2", "first"),
        first_time=("time", "min"), last_time=("time", "max"),
        n_slots=("time", "nunique"), frp_max_mw=("frp_mw", "max"),
    ).reset_index()
    dx, dy = offsets_km(c["lat"], c["lon"], lat0, lon0)
    c["dist_km"] = np.hypot(dx, dy)
    c["bearing"] = np.degrees(np.arctan2(dx, dy)) % 360

    # Slot timeline from first to last detection, so quiet slots inside the
    # event are rows too.
    grid = pd.date_range(t_first, t_last, freq=slot)
    times = pd.DatetimeIndex(sorted(set(grid) | set(d["time"])))
    per = d.groupby("time").agg(n_pixels=("cell", "size"),
                                frp_mw=("frp_mw", "sum"))
    s = per.reindex(times, fill_value=0).astype("float64")
    s.index.name = "time"
    s["observed"] = ([t in observed for t in times] if observed is not None
                     else True)
    # A slot never read is unknown, not zero.
    s.loc[~s["observed"].astype(bool) & (s["n_pixels"] == 0),
          ["n_pixels", "frp_mw"]] = np.nan

    new = c.groupby("first_time").agg(new_cells=("cell", "size"),
                                      new_km2=("pixel_km2", "sum"),
                                      reach=("dist_km", "max"))
    new = new.reindex(times, fill_value=0)
    s["new_cells"] = new["new_cells"].astype("int32")
    s["footprint_km2"] = new["new_km2"].cumsum()
    s["front_km"] = new["reach"].cummax()

    # Advance of the front over the trailing window. Before the window reaches
    # back past ignition, measure from the extent first detected, so the
    # initial pixel does not count as spread.
    window = pd.Timedelta(minutes=ROS_WINDOW_MIN)
    front = s["front_km"]
    back = front.reindex(times - window, method="ffill").to_numpy()
    back = np.where(np.isnan(back), front.iloc[0], back)
    s["ros_kmh"] = (front.to_numpy() - back) / (window.total_seconds() / 3600)

    # Energy: each slot's FRP held for the length of the slot.
    fre_mj = float(np.nansum(s["frp_mw"]) * slot.total_seconds())

    peak_at = s["frp_mw"].idxmax()
    far = c.loc[c["dist_km"].idxmax()]
    reach_h = (far["first_time"] - t_first).total_seconds() / 3600
    pixel_km = float(np.sqrt(c["pixel_km2"].median()))
    spread = float(far["dist_km"]) >= 0.5 * pixel_km
    heading = float(far["bearing"]) if spread else np.nan

    # Where it started: the commune of the brightest first-slot pixel.
    seed = first.loc[first["frp_mw"].idxmax(), "cell"]
    seat = cells.set_index("cell").loc[seed]

    n_obs = int(s["observed"].sum()) if observed is not None else np.nan
    summary = {
        "start_utc": t_first, "end_utc": t_last,
        "duration_h": (t_last - t_first + slot).total_seconds() / 3600,
        "lat": lat0, "lon": lon0,
        "ADM1_CODE": seat["ADM1_CODE"], "ADM1_NAME": seat["ADM1_NAME"],
        "ADM2_CODE": seat["ADM2_CODE"], "ADM2_NAME": seat["ADM2_NAME"],
        "n_cells": len(c), "footprint_km2": float(c["pixel_km2"].sum()),
        "n_slots_fire": int((s["n_pixels"] > 0).sum()),
        "n_slots_obs": n_obs,
        "peak_frp_mw": float(s["frp_mw"].max()), "peak_time_utc": peak_at,
        "fre_tj": fre_mj / 1e6,
        "biomass_t": fre_mj * BIOMASS_KG_PER_MJ / 1000,
        "max_front_km": float(s["front_km"].max()),
        "max_ros_kmh": float(np.nanmax(s["ros_kmh"])) if spread else 0.0,
        "mean_ros_kmh": (float(far["dist_km"]) / reach_h
                         if spread and reach_h > 0 else 0.0),
        "heading_deg": heading, "heading": compass(heading),
    }
    return s.reset_index(), c, summary


def link_firms(events: pd.DataFrame) -> pd.DataFrame:
    """Match each event to the FIRMS ignition of the same fire, if any."""
    events["firms_ignition_id"] = pd.array([pd.NA] * len(events), dtype="Int32")
    events["firms_lag_h"] = np.nan
    if not IGNITION_FILE.exists() or events.empty:
        return events
    from scipy.spatial import cKDTree

    ign = pd.read_parquet(IGNITION_FILE, columns=[
        "ignition_id", "lat", "lon", "date", "hour_local"])
    # ignitions.parquet keeps the local hour, not the minute: take the middle
    # of that hour, so the lag is good to about half an hour either way.
    ign["t_utc"] = (pd.to_datetime(ign["date"])
                    + pd.to_timedelta(ign["hour_local"].astype(float) + 0.5, unit="h")
                    - pd.Timedelta(hours=UTC_OFFSET_HOURS))
    lo = events["start_utc"].min() - pd.Timedelta(hours=LINK_HOURS)
    hi = events["start_utc"].max() + pd.Timedelta(hours=LINK_HOURS)
    ign = ign[(ign["t_utc"] >= lo) & (ign["t_utc"] <= hi)].reset_index(drop=True)
    if ign.empty:
        return events

    tree = cKDTree(project_km(ign["lat"], ign["lon"]))
    hits = tree.query_ball_point(project_km(events["lat"], events["lon"]), LINK_KM)
    for row, cand in zip(events.index, hits):
        if not cand:
            continue
        lag = ((ign.loc[cand, "t_utc"] - events.at[row, "start_utc"])
               .dt.total_seconds() / 3600)
        lag = lag[lag.abs() <= LINK_HOURS]
        if lag.empty:
            continue
        best = lag.abs().idxmin()
        events.at[row, "firms_ignition_id"] = int(ign.at[best, "ignition_id"])
        events.at[row, "firms_lag_h"] = float(lag[best])
    return events


def build(det: pd.DataFrame, cells: pd.DataFrame,
          slots: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    size = det.groupby("event").agg(
        n_slots=("time", "nunique"), t0=("time", "min"), t1=("time", "max"))
    long_ = (size["t1"] - size["t0"]) > pd.Timedelta(days=MAX_EVENT_DAYS)
    short = size["n_slots"] < MIN_FIRE_SLOTS
    print(f"  seen in under {MIN_FIRE_SLOTS} slots          : "
          f"dropped {int(short.sum()):>8,} events")
    print(f"  lasting over {MAX_EVENT_DAYS} days           : "
          f"dropped {int(long_.sum()):>8,} events")
    keep = size.index[~short & ~long_]
    det = det[det["event"].isin(keep)]

    observed: Dict[str, set] = {}
    if slots is not None:
        for plat, s in slots.groupby("platform"):
            observed[plat] = set(s["time"])

    order = size.loc[keep].sort_values("t0").index
    groups = det.groupby("event", sort=False)
    ev_rows, slot_rows, cell_rows = [], [], []
    for new_id, ev in enumerate(order, 1):
        d = groups.get_group(ev)
        plat = d["platform"].iat[0]
        s, c, summary = event_metrics(
            d, cells, PLATFORMS[plat]["slot_min"], observed.get(plat))
        s.insert(0, "event_id", new_id)
        c.insert(0, "event_id", new_id)
        slot_rows.append(s)
        cell_rows.append(c)
        ev_rows.append({"event_id": new_id, "platform": plat,
                        "slot_min": PLATFORMS[plat]["slot_min"], **summary})
        if new_id % 500 == 0:
            print(f"  {new_id:,} / {len(order):,} events measured")

    if not ev_rows:
        return {}

    events = pd.DataFrame(ev_rows)
    local = events["start_utc"] + pd.Timedelta(hours=UTC_OFFSET_HOURS)
    events["year"] = local.dt.year.astype("int16")
    events["month"] = local.dt.month.astype("int8")
    events = link_firms(events)

    remap = pd.Series(np.arange(1, len(order) + 1), index=order)
    dets = det.assign(event_id=det["event"].map(remap).astype("int32"))[
        ["event_id", "time", "lat", "lon", "frp_mw"]]
    return {
        "events": events,
        "slots": pd.concat(slot_rows, ignore_index=True),
        "cells": pd.concat(cell_rows, ignore_index=True),
        "detections": dets.sort_values(["event_id", "time"]).reset_index(drop=True),
    }


def write(out: Dict[str, pd.DataFrame]) -> None:
    ev = out["events"]
    ev["event_id"] = ev["event_id"].astype("int32")
    for c in ("ADM1_CODE", "ADM2_CODE"):
        ev[c] = ev[c].astype("int32")
    for c in ("ADM1_NAME", "ADM2_NAME", "platform", "heading"):
        ev[c] = ev[c].astype("category")
    for c in ("lat", "lon"):
        ev[c] = ev[c].astype("float64")
    for c in ("duration_h", "footprint_km2", "peak_frp_mw", "fre_tj",
              "biomass_t", "max_front_km", "max_ros_kmh", "mean_ros_kmh",
              "heading_deg",
              "firms_lag_h", "n_slots_obs"):
        ev[c] = ev[c].astype("float32")
    for c in ("n_cells", "n_slots_fire"):
        ev[c] = ev[c].astype("int32")
    ev["slot_min"] = ev["slot_min"].astype("int8")

    sl = out["slots"]
    sl["event_id"] = sl["event_id"].astype("int32")
    sl["observed"] = sl["observed"].astype(bool)
    for c in ("n_pixels", "frp_mw", "footprint_km2", "front_km", "ros_kmh"):
        sl[c] = sl[c].astype("float32")

    ce = out["cells"].drop(columns=["cell", "bearing"])
    ce["event_id"] = ce["event_id"].astype("int32")
    ce["n_slots"] = ce["n_slots"].astype("int32")
    for c in ("pixel_km2", "frp_max_mw", "dist_km"):
        ce[c] = ce[c].astype("float32")

    de = out["detections"]
    de["frp_mw"] = de["frp_mw"].astype("float32")

    for frame, path in ((ev, OUT_EVENTS), (sl, OUT_SLOTS),
                        (ce, OUT_CELLS), (de, OUT_DETECTIONS)):
        frame.to_parquet(path, index=False)
        print(f"OK  {path}  ({len(frame):,} rows, "
              f"{path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    print("1. Loading geostationary fire pixels")
    det, slots = load()

    print("\n2. Cells and their communes")
    cells = build_cells(det)
    print(f"  {len(cells):,} distinct pixel positions")

    print("\n3. Screening")
    det = screen(det, cells)
    if det.empty:
        sys.exit("Nothing left over Algeria's vegetated communes.")

    eps = ", ".join(f"{k} {v:g} km" for k, v in EPS_KM.items())
    print(f"\n4. Space-time clustering (eps = {eps} / {EPS_HOURS:g} h)")
    det = cluster(det)

    print("\n5. Measuring each event")
    out = build(det, cells, slots)
    if not out:
        sys.exit("No event survived the filters.")

    ev = out["events"]
    print(f"\n  {len(ev):,} events")
    for plat, e in ev.groupby("platform", observed=True):
        linked = e["firms_lag_h"].notna()
        line = (f"  {plat}: {len(e):,} events, fastest front "
                f"{e['max_ros_kmh'].max():.1f} km/h")
        if linked.any():
            line += (f"; {int(linked.sum()):,} matched to FIRMS, seen a median "
                     f"{e.loc[linked, 'firms_lag_h'].median():+.1f} h earlier")
        print(line)

    print()
    write(out)


if __name__ == "__main__":
    main()
