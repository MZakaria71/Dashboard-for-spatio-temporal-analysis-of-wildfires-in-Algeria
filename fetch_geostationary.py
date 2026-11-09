#!/usr/bin/env python3
"""
fetch_geostationary.py
──────────────────────
Pull geostationary active-fire detections over Algeria from the Meteosat
satellites, so a fire can be followed every 10 or 15 minutes instead of at the
four or so daily overpasses of MODIS and VIIRS.

Why geostationary
    A polar orbiter sees northern Algeria at fixed hours (Terra about 10:30 and
    22:30, Aqua and VIIRS about 13:30 and 01:30 local). Between two passes a
    wind-driven fire in the Tell can cross a whole commune unseen, so FIRMS can
    say where a fire was, but not how it got there. Meteosat sits over 0 degrees
    longitude and images the whole of Algeria continuously:

        MTG-I  (Meteosat-12, FCI)      every 10 min   ~1 km at nadir
        MSG    (Meteosat-8 to 11, SEVIRI) every 15 min   ~3 km at nadir

    Over the fire belt (36 to 37 N) the view is oblique, so pixels are larger
    than at nadir: about 1.5 km2 for FCI and 13 km2 for SEVIRI. That is coarse
    for burned area but plenty to time a fire and follow its front.

Source
    The fire products derived from those two imagers by EUMETSAT's Land Surface
    Analysis Satellite Application Facility (LSA SAF), free with a registered
    account:

        MSG FRP-PIXEL   LSA-502   operational, multi-year archive
        MTG FRP-PIXEL   LSA-509   demonstration since 2025, back-processed to
                                  1 January 2025

    Both are "list products": one file per repeat cycle holding every pixel the
    algorithm flagged as fire over the full disc, with its position, its fire
    radiative power (FRP, MW) and an uncertainty. This script keeps the rows
    that fall over Algeria.

    The LSA SAF product is used rather than running a detection algorithm on
    the Level 1 radiances: it is validated, it handles the cloud, sun-glint and
    hot-desert false alarms that a home-made threshold gets wrong over the
    Sahara, and it ships FRP in physical units.

Credentials
    Register (free) at https://lsa-saf.eumetsat.int and put the data-service
    login in the environment. Like fetch_firms.py, this never prints them.

        PowerShell   $env:LSASAF_USER = "you"; $env:LSASAF_PASSWORD = "..."
        bash         export LSASAF_USER=you LSASAF_PASSWORD=...

Usage
    python fetch_geostationary.py --platform MTG --start 2025-07-20 --end 2025-07-24
    python fetch_geostationary.py --platform MSG --start 2021-08-09 --end 2021-08-13
    python fetch_geostationary.py --platform MTG --days 2      # near real time
    python fetch_geostationary.py --platform MTG --days 1 --dry-run
    python fetch_geostationary.py --platform MSG --local D:/lsasaf/frp

    --local reads files you already have (downloaded through the LSA SAF web
    interface, received over EUMETCast, copied from a colleague), so nothing
    here depends on the remote layout. Compressed .bz2 / .gz files are fine.

    Then build the spread tables the dashboard reads:

        python prepare_spread.py

    Re-running over a period already fetched is safe and cheap: every slot
    processed is logged in data/geo_slots.parquet and skipped next time, and
    detections are de-duplicated on (platform, slot, position).

    A full disc list product can reach a few MB in the African burning
    seasons, and MTG writes 144 of them a day. A week of MTG is therefore a
    download of the order of a gigabyte to keep a few thousand Algerian rows.
    Fetch the episodes you want to study, not whole years.

Remote layout
    The data service could not be reached from where this script was written,
    so the default folders in PLATFORMS below are unverified. If a dry run
    reports that a day folder is missing, open the data service in a browser,
    find the folder that holds one day of FRP-PIXEL list files, and pass the
    part above the YYYY/MM/DD levels as --base-url. File selection does not
    depend on exact names: any file in the day folder with a 12-digit
    YYYYMMDDHHMM timestamp in its name is taken, quality files are skipped, and
    when "ListProduct" files are present only those are kept.
"""

from __future__ import annotations

import argparse
import base64
import bz2
import gzip
import io
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

DATA_DIR = Path(__file__).parent / "data"
DETECTIONS_FILE = DATA_DIR / "geo_detections.parquet"
SLOTS_FILE = DATA_DIR / "geo_slots.parquet"

# west, south, east, north. The same loose box as fetch_firms.py: the
# administrative clip happens in prepare_spread.py, against the GAUL communes.
ALGERIA_BBOX = (-9.0, 18.5, 12.5, 37.5)

SERVICE = "https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS"

# slot_min is the repeat cycle, ssp_km the sampling distance at the
# sub-satellite point (used by prepare_spread.py when a file carries no pixel
# area). See "Remote layout" in the docstring before trusting base_url.
PLATFORMS: Dict[str, dict] = {
    "MTG": {
        "label": "MTG-I FCI, LSA SAF MTG FRP-PIXEL (LSA-509)",
        "slot_min": 10,
        "ssp_km": 1.0,
        "base_url": f"{SERVICE}/MTG/MTFRPPIXEL/NETCDF",
    },
    "MSG": {
        "label": "MSG SEVIRI, LSA SAF MSG FRP-PIXEL (LSA-502)",
        "slot_min": 15,
        "ssp_km": 3.0,
        "base_url": f"{SERVICE}/MSG/FRP-PIXEL/HDF5",
    },
}

# Dataset names to look for, lower case, first match wins. LSA SAF's HDF5
# files use upper-case names (LATITUDE, FRP, ...) and its NetCDF files lower
# case, so the lookup ignores case and group paths.
FIELDS: Dict[str, Tuple[str, ...]] = {
    "lat":        ("latitude", "lat"),
    "lon":        ("longitude", "lon"),
    "frp_mw":     ("frp",),
    "frp_unc_mw": ("frp_uncertainty", "frp_unc", "frp_error"),
    "confidence": ("fire_confidence", "confidence"),
    "pixel_km2":  ("pixel_size", "pixel_area"),
}
REQUIRED = ("lat", "lon", "frp_mw")

STAMP = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(?!\d)")
SKIP_SUFFIXES = (".md5", ".xml", ".txt", ".html", ".htm", ".png", ".jpg",
                 ".json", ".sha256", ".tmp", ".part")

PAUSE_S = 0.2          # between file requests: be a polite client
MAX_RETRIES = 4
SAVE_EVERY = 144       # write progress to disk every N slots (one MTG day)


class FetchError(RuntimeError):
    pass


# ── HTTP ───────────────────────────────────────────────────────────────────
def _auth_header() -> str:
    user = os.environ.get("LSASAF_USER", "").strip()
    password = os.environ.get("LSASAF_PASSWORD", "")
    if not user or not password:
        sys.exit(
            "LSASAF_USER and LSASAF_PASSWORD are not set.\n"
            "Register (free) at https://lsa-saf.eumetsat.int, then set both in "
            "the environment (see the top of this script). To read files you "
            "already have, use --local DIR instead."
        )
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return f"Basic {token}"


def _get(url: str, auth: str) -> Optional[bytes]:
    """GET with retries on transient failures. None on 404."""
    req = urllib.request.Request(url, headers={
        "Authorization": auth, "User-Agent": "wildfire-dashboard/1.0"})
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code in (401, 403):
                raise FetchError(
                    f"The data service refused the login (HTTP {exc.code}). "
                    "Check LSASAF_USER / LSASAF_PASSWORD, and that the account "
                    "has been activated for data downloads.") from None
            if exc.code < 500 or attempt == MAX_RETRIES:
                raise FetchError(f"HTTP {exc.code} ({exc.reason}) for {url}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == MAX_RETRIES:
                raise FetchError(f"{type(exc).__name__}: {exc} for {url}") from None
        time.sleep(2 ** attempt)
    return None


def list_day(base_url: str, day: date, auth: str) -> Optional[List[str]]:
    """File names in one day folder, or None if the folder does not exist."""
    url = f"{base_url.rstrip('/')}/{day:%Y/%m/%d}/"
    raw = _get(url, auth)
    if raw is None:
        return None
    html = raw.decode("utf-8", errors="replace")
    names = []
    for href in re.findall(r'href="([^"?#]+)"', html):
        name = urllib.parse.unquote(href.rstrip("/").split("/")[-1])
        if name and not href.endswith("/"):
            names.append(name)
    return sorted(set(names))


# ── file selection ─────────────────────────────────────────────────────────
def slot_time(name: str) -> Optional[datetime]:
    """The repeat-cycle start encoded in the file name, UTC."""
    m = STAMP.search(Path(name).name)
    if not m:
        return None
    try:
        return datetime(*(int(g) for g in m.groups()))
    except ValueError:
        return None


def select_files(names: Iterable[str]) -> List[str]:
    """The fire list files among everything a folder holds."""
    keep = [n for n in names
            if slot_time(n) is not None
            and not n.lower().endswith(SKIP_SUFFIXES)
            and "quality" not in n.lower()]
    listed = [n for n in keep if "listproduct" in n.lower()]
    return listed or keep


# ── reading ────────────────────────────────────────────────────────────────
def _decompress(raw: bytes) -> bytes:
    if raw[:3] == b"BZh":
        return bz2.decompress(raw)
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)
    return raw


def _scalar(value) -> Optional[float]:
    try:
        arr = np.asarray(value).ravel()
        return float(arr[0]) if arr.size else None
    except (TypeError, ValueError):
        return None


def _physical(ds) -> np.ndarray:
    """Dataset values in physical units, whichever convention the file uses.

    LSA SAF HDF5 stores integers with a SCALING_FACTOR to divide by (and an
    OFFSET, zero in practice); CF NetCDF stores scale_factor / add_offset to
    multiply and add. A file with neither is taken as already physical.
    """
    values = np.asarray(ds[()]).astype("float64").ravel()
    attrs = {str(k).lower(): v for k, v in ds.attrs.items()}

    for key in ("_fillvalue", "missing_value"):
        fill = _scalar(attrs[key]) if key in attrs else None
        if fill is not None:
            values[values == fill] = np.nan

    lsa = _scalar(attrs.get("scaling_factor"))
    if lsa not in (None, 0.0):
        values = (values - (_scalar(attrs.get("offset")) or 0.0)) / lsa
    else:
        scale = _scalar(attrs.get("scale_factor"))
        offset = _scalar(attrs.get("add_offset"))
        if scale is not None:
            values = values * scale
        if offset is not None:
            values = values + offset
    return values


def read_list_product(raw: bytes, name: str) -> pd.DataFrame:
    """Every fire pixel in one FRP-PIXEL list file (HDF5 or NetCDF4)."""
    try:
        import h5py
    except ImportError:
        sys.exit("h5py is needed to read the FRP-PIXEL files: pip install h5py")

    data = _decompress(raw)
    if data[:4] == b"CDF\x01" or data[:4] == b"CDF\x02":
        raise FetchError(f"{name}: classic NetCDF3, not HDF5-based NetCDF4. "
                         "Convert it with `nccopy -k nc4` first.")

    with h5py.File(io.BytesIO(data), "r") as h5:
        found: Dict[str, object] = {}

        def visit(path, obj):
            if isinstance(obj, h5py.Dataset):
                found.setdefault(path.split("/")[-1].lower(), obj)

        h5.visititems(visit)

        cols: Dict[str, np.ndarray] = {}
        for field, candidates in FIELDS.items():
            hit = next((found[c] for c in candidates if c in found), None)
            if hit is not None:
                cols[field] = _physical(hit)

        missing = [f for f in REQUIRED if f not in cols]
        if missing:
            raise FetchError(
                f"{name}: no dataset for {missing}. The file holds: "
                f"{sorted(found)[:30]}. Add the right name to FIELDS.")

    n = len(cols["lat"])
    for field, arr in list(cols.items()):
        if len(arr) != n:
            # A per-file scalar or a mis-matched optional field: drop it rather
            # than broadcast something that is not per pixel.
            if field in REQUIRED:
                raise FetchError(f"{name}: {field} has {len(arr)} values, "
                                 f"latitude has {n}")
            del cols[field]

    df = pd.DataFrame(cols)
    df = df[np.isfinite(df["lat"]) & np.isfinite(df["lon"])
            & np.isfinite(df["frp_mw"]) & (df["frp_mw"] > 0)]

    # Guard against a scaling convention this reader got wrong: a factor of 100
    # off puts every latitude outside the globe, and silently writing that would
    # place Algerian fires in the ocean.
    if len(df) and (df["lat"].abs().max() > 90 or df["lon"].abs().max() > 180):
        raise FetchError(f"{name}: coordinates out of range after scaling "
                         f"(lat up to {df['lat'].abs().max():.1f}). Check the "
                         "SCALING_FACTOR / scale_factor attributes.")
    if "pixel_km2" in df and len(df) and df["pixel_km2"].median() > 1e4:
        df["pixel_km2"] = df["pixel_km2"] / 1e6          # stored in m2
    return df


def clip(df: pd.DataFrame, bbox=ALGERIA_BBOX) -> pd.DataFrame:
    w, s, e, n = bbox
    return df[(df["lon"] >= w) & (df["lon"] <= e)
              & (df["lat"] >= s) & (df["lat"] <= n)]


# ── storage ────────────────────────────────────────────────────────────────
DET_COLS = ["platform", "time", "lat", "lon", "frp_mw", "frp_unc_mw",
            "confidence", "pixel_km2"]


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    for c in DET_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[DET_COLS].copy()
    df["platform"] = df["platform"].astype(str)
    df["time"] = pd.to_datetime(df["time"])
    for c in ("lat", "lon"):
        df[c] = df[c].astype("float64")
    for c in ("frp_mw", "frp_unc_mw", "confidence", "pixel_km2"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df


def save(new_det: List[pd.DataFrame], new_slots: List[dict]) -> Tuple[int, int]:
    """Merge into the files on disk. Returns (detections, slots) now stored."""
    DATA_DIR.mkdir(exist_ok=True)

    frames = [f for f in new_det if len(f)]
    if DETECTIONS_FILE.exists():
        frames.insert(0, pd.read_parquet(DETECTIONS_FILE))
    if frames:
        det = _tidy(pd.concat(frames, ignore_index=True))
        key = det.assign(_la=det["lat"].round(4), _lo=det["lon"].round(4))
        det = det[~key.duplicated(["platform", "time", "_la", "_lo"], keep="last")]
        det = det.sort_values(["platform", "time"]).reset_index(drop=True)
        det.to_parquet(DETECTIONS_FILE, index=False)
        n_det = len(det)
    else:
        n_det = 0

    slots = pd.DataFrame(new_slots,
                         columns=["platform", "time", "n_detections", "file"])
    if SLOTS_FILE.exists():
        slots = pd.concat([pd.read_parquet(SLOTS_FILE), slots], ignore_index=True)
    slots["time"] = pd.to_datetime(slots["time"])
    slots["n_detections"] = slots["n_detections"].astype("int32")
    slots = (slots.drop_duplicates(["platform", "time"], keep="last")
                  .sort_values(["platform", "time"]).reset_index(drop=True))
    slots.to_parquet(SLOTS_FILE, index=False)
    return n_det, len(slots)


def done_slots(platform: str) -> set:
    if not SLOTS_FILE.exists():
        return set()
    s = pd.read_parquet(SLOTS_FILE)
    return set(pd.to_datetime(s.loc[s["platform"] == platform, "time"]))


# ── drivers ────────────────────────────────────────────────────────────────
def _process(platform: str, name: str, raw: bytes, bbox) -> Tuple[pd.DataFrame, dict]:
    t = slot_time(name)
    det = clip(read_list_product(raw, name), bbox)
    det = det.assign(platform=platform, time=t)
    return det, {"platform": platform, "time": t,
                 "n_detections": len(det), "file": Path(name).name}


def run_remote(platform: str, base_url: str, start: date, end: date,
               bbox, dry_run: bool, refresh: bool) -> None:
    auth = _auth_header()
    have = set() if refresh else done_slots(platform)
    det_buf: List[pd.DataFrame] = []
    slot_buf: List[dict] = []
    n_files = n_rows = 0
    lo = datetime.combine(start, datetime.min.time())
    hi = datetime.combine(end + timedelta(days=1), datetime.min.time())

    day = start
    try:
        while day <= end:
            names = list_day(base_url, day, auth)
            if names is None:
                print(f"  {day}  no folder at {base_url}/{day:%Y/%m/%d}/")
                if day == start:
                    print("        If every day says this, the remote layout "
                          "differs from the default:\n        see 'Remote "
                          "layout' at the top of this script and pass "
                          "--base-url.")
                day += timedelta(days=1)
                continue

            files = [n for n in select_files(names) if lo <= slot_time(n) < hi]
            todo = [n for n in files if slot_time(n) not in have]
            print(f"  {day}  {len(files):>4} slots listed, {len(todo):>4} to fetch")
            if dry_run:
                for n in todo[:3]:
                    print(f"        {n}")
                day += timedelta(days=1)
                continue

            folder = f"{base_url.rstrip('/')}/{day:%Y/%m/%d}"
            for name in todo:
                raw = _get(f"{folder}/{urllib.parse.quote(name)}", auth)
                if raw is None:
                    print(f"        vanished before download: {name}")
                    continue
                try:
                    det, slot = _process(platform, name, raw, bbox)
                except FetchError as exc:
                    print(f"        skipped: {exc}")
                    continue
                det_buf.append(det)
                slot_buf.append(slot)
                n_files += 1
                n_rows += len(det)
                if len(slot_buf) >= SAVE_EVERY:
                    save(det_buf, slot_buf)
                    det_buf, slot_buf = [], []
                time.sleep(PAUSE_S)
            day += timedelta(days=1)
    finally:
        # An interrupted run (Ctrl+C, a dropped connection) keeps what it got:
        # the slots it logged are skipped on the next run.
        if slot_buf:
            save(det_buf, slot_buf)
            det_buf, slot_buf = [], []

    if dry_run:
        print("\nDry run: nothing downloaded.")
        return
    n_det, n_slots = save(det_buf, slot_buf)
    print(f"\n{n_files:,} slots read, {n_rows:,} fire pixels over Algeria.")
    print(f"OK  {DETECTIONS_FILE}  ({n_det:,} detections)")
    print(f"OK  {SLOTS_FILE}  ({n_slots:,} slots logged)")


def run_local(platform: str, folder: Path, bbox, refresh: bool) -> None:
    if not folder.exists():
        sys.exit(f"{folder} does not exist.")
    have = set() if refresh else done_slots(platform)
    paths = sorted(p for p in folder.rglob("*") if p.is_file())
    names = set(select_files(p.name for p in paths))
    paths = [p for p in paths if p.name in names and slot_time(p.name) not in have]
    if not paths:
        print(f"No new FRP-PIXEL list files under {folder}.")
        return

    det_buf: List[pd.DataFrame] = []
    slot_buf: List[dict] = []
    n_rows = 0
    for i, p in enumerate(paths, 1):
        try:
            det, slot = _process(platform, p.name, p.read_bytes(), bbox)
        except FetchError as exc:
            print(f"  skipped: {exc}")
            continue
        det_buf.append(det)
        slot_buf.append(slot)
        n_rows += len(det)
        if i % SAVE_EVERY == 0:
            save(det_buf, slot_buf)
            det_buf, slot_buf = [], []
            print(f"  {i:,} / {len(paths):,} files")
    n_det, n_slots = save(det_buf, slot_buf)
    print(f"\n{len(paths):,} files read, {n_rows:,} fire pixels over Algeria.")
    print(f"OK  {DETECTIONS_FILE}  ({n_det:,} detections)")
    print(f"OK  {SLOTS_FILE}  ({n_slots:,} slots logged)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--platform", choices=sorted(PLATFORMS), required=True,
                    help="MTG (FCI, 10 min, 2025 on) or MSG (SEVIRI, 15 min)")
    ap.add_argument("--start", type=date.fromisoformat, help="first day, UTC")
    ap.add_argument("--end", type=date.fromisoformat,
                    help="last day, UTC (default: today)")
    ap.add_argument("--days", type=int, help="the last N days, ending today")
    ap.add_argument("--local", type=Path,
                    help="read list files already on disk instead of fetching")
    ap.add_argument("--base-url", help="override the data-service folder")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be fetched, download nothing")
    ap.add_argument("--refresh", action="store_true",
                    help="re-read slots already logged in geo_slots.parquet")
    args = ap.parse_args()

    spec = PLATFORMS[args.platform]
    print(f"{spec['label']}: one slot every {spec['slot_min']} min")

    if args.local:
        run_local(args.platform, args.local, ALGERIA_BBOX, args.refresh)
        return

    end = args.end or date.today()
    if args.days:
        start = end - timedelta(days=args.days - 1)
    elif args.start:
        start = args.start
    else:
        sys.exit("Give --start (and optionally --end), --days N, or --local DIR.")
    if start > end:
        sys.exit(f"--start {start} is after --end {end}.")

    base_url = args.base_url or spec["base_url"]
    print(f"{start} -> {end}  from {base_url}\n")
    try:
        run_remote(args.platform, base_url, start, end, ALGERIA_BBOX,
                   args.dry_run, args.refresh)
    except FetchError as exc:
        sys.exit(f"\n{exc}")


if __name__ == "__main__":
    main()
