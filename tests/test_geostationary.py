"""
Tests for the geostationary layer: fetch_geostationary.py reading LSA SAF
FRP-PIXEL list files, and prepare_spread.py turning them into fire spread.

Everything here is synthetic, built so the right answer is known in advance:
a fire whose front runs north-east at exactly 2 km/h has to come back as one
event heading NE at about 2 km/h. The administrative and land-cover files are
the real ones in data/, so the screening is tested against Algeria as it is.

    pip install pytest h5py scipy geopandas shapely
    python -m pytest tests
"""

from __future__ import annotations

import bz2
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("scipy")
pytest.importorskip("geopandas")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fetch_geostationary as fg  # noqa: E402
import prepare_spread as ps  # noqa: E402

T0 = datetime(2026, 7, 15, 11, 0)          # UTC, 12:00 in Algeria
ORIGIN = (36.62, 4.15)                      # Tizi Ouzou, well vegetated
FLARE = (31.70, 6.05)                       # Hassi Messaoud gas field
NIGERIA = (9.50, 7.50)                      # on the disc, not in Algeria
ROS_KMH = 2.0
STEP_DEG = 0.0125                           # ~1.1-1.4 km, an MTG-like grid


def write_list_file(path: Path, lat, lon, frp, convention: str = "lsa") -> None:
    """An FRP-PIXEL list product in one of the two storage conventions."""
    lat, lon, frp = (np.asarray(v, dtype="float64") for v in (lat, lon, frp))
    with h5py.File(path, "w") as f:
        if convention == "lsa":            # HDF5: integers / SCALING_FACTOR
            for name, values, sf in (("LATITUDE", lat, 100.0),
                                     ("LONGITUDE", lon, 100.0),
                                     ("FRP", frp, 10.0)):
                ds = f.create_dataset(name, data=np.rint(values * sf).astype("int32"))
                ds.attrs["SCALING_FACTOR"] = sf
                ds.attrs["OFFSET"] = 0.0
            f.create_dataset("PIXEL_SIZE", data=np.full(len(lat), 1.5))
        else:                              # NetCDF4 / CF: scale_factor
            grp = f.create_group("fire")
            grp.create_dataset("latitude", data=lat.astype("float32"))
            grp.create_dataset("longitude", data=lon.astype("float32"))
            ds = grp.create_dataset("frp", data=np.rint(frp * 10).astype("int16"))
            ds.attrs["scale_factor"] = 0.1
            ds.attrs["_FillValue"] = np.int16(-1)


def file_bytes(tmp_path: Path, **kw) -> bytes:
    p = tmp_path / "tmp.h5"
    write_list_file(p, **kw)
    return p.read_bytes()


# ── reading ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("convention", ["lsa", "cf"])
def test_both_storage_conventions_read_to_physical_units(tmp_path, convention):
    raw = file_bytes(tmp_path, lat=[36.5, 9.5], lon=[4.25, 7.5],
                     frp=[120.3, 55.0], convention=convention)
    df = fg.read_list_product(bz2.compress(raw), "x.bz2")
    assert len(df) == 2
    assert df["lat"].tolist() == pytest.approx([36.5, 9.5], abs=0.01)
    assert df["frp_mw"].tolist() == pytest.approx([120.3, 55.0], abs=0.1)
    assert len(fg.clip(df)) == 1           # Nigeria is not in Algeria


def test_fill_values_and_zero_frp_are_dropped(tmp_path):
    raw = file_bytes(tmp_path, lat=[36.5, 36.6], lon=[4.2, 4.3],
                     frp=[-0.1, 0.0], convention="cf")
    assert fg.read_list_product(raw, "x").empty


def test_wrong_scaling_is_refused_not_written(tmp_path):
    p = tmp_path / "bad.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("LATITUDE", data=np.array([3650], dtype="int32"))
        f.create_dataset("LONGITUDE", data=np.array([425], dtype="int32"))
        f.create_dataset("FRP", data=np.array([100.0]))
    with pytest.raises(fg.FetchError, match="out of range"):
        fg.read_list_product(p.read_bytes(), "bad")


def test_missing_dataset_names_what_the_file_holds(tmp_path):
    p = tmp_path / "odd.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("LAT_DEG", data=np.array([36.5]))
    with pytest.raises(fg.FetchError, match="lat_deg"):
        fg.read_list_product(p.read_bytes(), "odd")


def test_file_selection_keeps_list_products_only():
    names = [
        "HDF5_LSASAF_MSG_FRP-PIXEL-ListProduct_MSG-Disk_202108101200.bz2",
        "HDF5_LSASAF_MSG_FRP-PIXEL-QualityProduct_MSG-Disk_202108101200.bz2",
        "HDF5_LSASAF_MSG_FRP-PIXEL-ListProduct_MSG-Disk_202108101200.bz2.md5",
        "index.html",
    ]
    assert fg.select_files(names) == names[:1]
    assert fg.slot_time(names[0]) == datetime(2021, 8, 10, 12, 0)
    assert fg.slot_time("no_stamp_here.nc") is None


# ── a synthetic fire, end to end ───────────────────────────────────────────
def spreading_fire(start: datetime, hours: float = 5.0, residence_h: float = 1.0):
    """Pixels of a fire running NE at ROS_KMH, on an MTG-like 10 min grid.

    A cell ignites when the front, a line perpendicular to the NE axis moving
    at ROS_KMH, reaches it, and burns for residence_h. The strip is 3 km wide.
    """
    lat0, lon0 = ORIGIN
    grid = np.arange(-12, 13) * STEP_DEG
    la, lo = np.meshgrid(lat0 + grid, lon0 + grid, indexing="ij")
    la, lo = la.ravel(), lo.ravel()
    dx, dy = ps.offsets_km(la, lo, lat0, lon0)
    along = (dx + dy) / np.sqrt(2)
    across = (dx - dy) / np.sqrt(2)
    inside = (along >= -0.8) & (np.abs(across) <= 1.5)
    arrival = np.where(inside, np.maximum(along, 0) / ROS_KMH, np.inf)

    slots = {}
    for k in range(int(hours * 6) + 1):
        t_h = k / 6
        burning = (arrival <= t_h) & (arrival + residence_h >= t_h)
        slots[start + timedelta(minutes=10 * k)] = (
            la[burning], lo[burning], np.full(burning.sum(), 25.0))
    return slots


def write_folder(folder: Path, fires: list) -> None:
    """One list file per slot, merging every fire burning in it."""
    merged: dict = {}
    for slots in fires:
        for t, (la, lo, frp) in slots.items():
            prev = merged.get(t, ([], [], []))
            merged[t] = tuple(np.concatenate([p, n]) for p, n in zip(prev, (la, lo, frp)))
    t_all = min(merged), max(merged)
    t = t_all[0]
    while t <= t_all[1]:
        la, lo, frp = merged.get(t, ([], [], []))
        # Every slot also carries a fire far away on the disc.
        la = np.append(la, NIGERIA[0])
        lo = np.append(lo, NIGERIA[1])
        frp = np.append(frp, 300.0)
        name = f"NETCDF4_LSASAF_MTG_FRP-PIXEL-ListProduct_MTG-FD_{t:%Y%m%d%H%M}.nc"
        write_list_file(folder / name, la, lo, frp, convention="cf")
        t += timedelta(minutes=10)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """Point both scripts at a scratch data folder and run them."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(fg, "DATA_DIR", data)
    monkeypatch.setattr(fg, "DETECTIONS_FILE", data / "geo_detections.parquet")
    monkeypatch.setattr(fg, "SLOTS_FILE", data / "geo_slots.parquet")
    monkeypatch.setattr(ps, "DETECTIONS_FILE", data / "geo_detections.parquet")
    monkeypatch.setattr(ps, "SLOTS_FILE", data / "geo_slots.parquet")
    for name in ("OUT_EVENTS", "OUT_SLOTS", "OUT_CELLS", "OUT_DETECTIONS"):
        monkeypatch.setattr(ps, name, data / getattr(ps, name).name)

    # A FIRMS ignition two hours after the geostationary start, 3 km away.
    local = T0 + timedelta(hours=2 + ps.UTC_OFFSET_HOURS)
    pd.DataFrame({
        "ignition_id": np.array([77], dtype="int32"),
        "lat": np.array([ORIGIN[0] + 0.02], dtype="float32"),
        "lon": np.array([ORIGIN[1] + 0.02], dtype="float32"),
        "date": [pd.Timestamp(local.date())],
        "hour_local": np.array([local.hour], dtype="int8"),
    }).to_parquet(data / "ignitions.parquet")
    monkeypatch.setattr(ps, "IGNITION_FILE", data / "ignitions.parquet")

    raw = tmp_path / "raw"
    raw.mkdir()
    flare = {T0 + timedelta(minutes=10 * k): (np.array([FLARE[0]]),
                                              np.array([FLARE[1]]),
                                              np.array([40.0]))
             for k in range(40)}
    write_folder(raw, [
        spreading_fire(T0),
        spreading_fire(T0 + timedelta(days=3)),     # same place, new fire
        flare,
    ])
    fg.run_local("MTG", raw, fg.ALGERIA_BBOX, refresh=False)

    det, slots = ps.load()
    cells = ps.build_cells(det)
    det = ps.screen(det, cells)
    det = ps.cluster(det)
    out = ps.build(det, cells, slots)
    ps.write(out)
    return out, raw


def test_fire_is_found_once_per_episode_and_flare_is_screened(pipeline):
    out, _ = pipeline
    ev = out["events"]
    assert len(ev) == 2                       # two fires, no flare
    assert set(ev["ADM1_NAME"].astype(str)) == {"Tizi Ouzou"}
    assert (ev["start_utc"].iloc[1] - ev["start_utc"].iloc[0]) == timedelta(days=3)
    assert ev["start_utc"].iloc[0] == T0


def test_spread_rate_and_heading_are_recovered(pipeline):
    out, _ = pipeline
    first = out["events"].iloc[0]
    assert first["heading"] in {"NE", "NNE", "ENE"}
    # The front moves in whole pixels, so allow about a pixel per hour of error.
    assert first["max_ros_kmh"] == pytest.approx(ROS_KMH, abs=1.4)
    assert first["max_front_km"] == pytest.approx(ROS_KMH * 5, abs=2.0)
    # Averaged over the whole run, the pixel steps wash out.
    assert first["mean_ros_kmh"] == pytest.approx(ROS_KMH, abs=0.3)

    slots = out["slots"][out["slots"]["event_id"] == first["event_id"]]
    assert slots["front_km"].is_monotonic_increasing
    assert slots["footprint_km2"].is_monotonic_increasing
    assert slots["observed"].all()
    mid = slots[slots["time"] >= T0 + timedelta(hours=2)]["ros_kmh"]
    assert mid.median() == pytest.approx(ROS_KMH, abs=1.0)


def test_energy_is_frp_times_time(pipeline):
    out, _ = pipeline
    first = out["events"].iloc[0]
    det = out["detections"][out["detections"]["event_id"] == first["event_id"]]
    expected_tj = det["frp_mw"].sum() * 600 / 1e6
    assert first["fre_tj"] == pytest.approx(expected_tj, rel=1e-4)
    assert first["biomass_t"] == pytest.approx(
        expected_tj * 1e6 * ps.BIOMASS_KG_PER_MJ / 1000, rel=1e-4)


def test_firms_match_reports_how_much_earlier_meteosat_saw_it(pipeline):
    out, _ = pipeline
    ev = out["events"]
    assert int(ev["firms_ignition_id"].iloc[0]) == 77
    # Two hours later, read from the middle of the FIRMS hour.
    assert ev["firms_lag_h"].iloc[0] == pytest.approx(2.5, abs=0.01)
    assert pd.isna(ev["firms_ignition_id"].iloc[1])


def test_rerun_skips_logged_slots_and_does_not_duplicate(pipeline):
    out, raw = pipeline
    before = pd.read_parquet(fg.DETECTIONS_FILE)
    fg.run_local("MTG", raw, fg.ALGERIA_BBOX, refresh=False)
    fg.run_local("MTG", raw, fg.ALGERIA_BBOX, refresh=True)
    after = pd.read_parquet(fg.DETECTIONS_FILE)
    assert len(after) == len(before)
    slots = pd.read_parquet(fg.SLOTS_FILE)
    assert not slots.duplicated(["platform", "time"]).any()


def test_unread_slots_are_unknown_not_zero(pipeline, monkeypatch):
    out, _ = pipeline
    first = out["events"].iloc[0]
    det = out["detections"][out["detections"]["event_id"] == first["event_id"]]
    # Pretend the slot log lost an hour in the middle of the fire, and that the
    # fire was not seen in it.
    hole = {T0 + timedelta(minutes=10 * k) for k in range(12, 18)}
    seen = det[~det["time"].isin(hole)].assign(
        pixel_km2=1.5, platform="MTG", cell=lambda d: d.groupby(
            ["lat", "lon"]).ngroup())
    cells = seen.groupby("cell").first().reset_index().assign(
        ADM1_CODE=1, ADM1_NAME="x", ADM2_CODE=1, ADM2_NAME="y")
    observed = {T0 + timedelta(minutes=10 * k) for k in range(31)} - hole
    s, _, summary = ps.event_metrics(seen, cells, 10, observed)
    gap = s[s["time"].isin(hole)]
    assert (~gap["observed"]).all()
    assert gap["frp_mw"].isna().all()
    assert summary["n_slots_obs"] == 31 - len(hole)
