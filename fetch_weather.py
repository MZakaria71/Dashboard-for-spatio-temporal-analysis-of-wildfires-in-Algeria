#!/usr/bin/env python3
"""
fetch_weather.py
────────────────
Builds the fire-weather layer: daily ERA5 reanalysis per wilaya, aggregated to
months, written as data/fire_weather.parquet.

Why this exists
    The dashboard can say 2021 had 541 ignitions against 2020's 1,403, which
    invites exactly the wrong conclusion. It cannot say whether a season was
    extreme *weather* or extreme *ignition pressure*. This is the missing half.

Source
    ERA5 daily aggregates via the Open-Meteo archive API. Free, no key, no
    account. ERA5 is the same reanalysis Copernicus serves; Open-Meteo simply
    hosts it with a plain HTTP interface, which keeps this reproducible with
    one command instead of a manual export.

Two choices worth knowing about
    1. WHERE each wilaya is sampled. ERA5 is ~11 km, and a wilaya can be
       enormous — averaging Tamanrasset's weather over its whole area
       describes empty desert, not the places that burn. So each wilaya is
       sampled at the mean position of its own recorded ignitions. Wilayas
       with too few ignitions to place a point fall back to a representative
       point inside the polygon.

    2. WHAT counts as a fire-weather day. Rather than reproduce a named index
       from memory and risk a wrong coefficient, the rule is explicit and
       stated in the dashboard: daily maximum temperature at or above
       FWD_TEMP_C, daily minimum relative humidity at or below FWD_RH_PCT, and
       daily maximum wind at or above FWD_WIND_KMH. The four raw drivers are
       stored alongside it, so anyone can apply a different rule.

    By default only the wilayas that actually record fires are fetched — the
    others have no fire weather worth a request against a volume-capped free
    API, and the dashboard's national aggregate already excludes them.

Usage
    python fetch_weather.py                 # 2001 to today, fire wilayas
    python fetch_weather.py --all-wilayas   # include the Saharan units too
    python fetch_weather.py --start 2015-01-01
    python fetch_weather.py --limit 3       # a quick trial run

    Open-Meteo's free tier caps by data volume, so a full run can exceed its
    hourly limit partway through. Each wilaya is saved as it arrives; re-run
    the same command and it resumes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

DATA_DIR = Path(__file__).parent / "data"
IGNITION_FILE = DATA_DIR / "ignitions.parquet"
BOUNDARY_FILE = DATA_DIR / "gaul_adm1.geojson"
OUTPUT_FILE = DATA_DIR / "fire_weather.parquet"

API = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = [
    "temperature_2m_max",
    "relative_humidity_2m_min",
    "wind_speed_10m_max",
    "precipitation_sum",
]
TIMEZONE = "Africa/Algiers"

# An explicit rule, not a named index — see the module docstring.
FWD_TEMP_C = 30.0
FWD_RH_PCT = 30.0
FWD_WIND_KMH = 20.0

# Below this many ignitions, the mean position is not a meaningful "where the
# fires are" and the polygon's own representative point is used instead.
MIN_IGNITIONS_FOR_CENTROID = 20

PAUSE_S = 2.0          # be a polite client of a free service
MAX_RETRIES = 4
RETRY_WAIT_S = 65      # the minutely limit clears in about a minute


class RateLimited(RuntimeError):
    """The hourly cap — not worth waiting out inside one run."""


def sampling_points(limit: int | None, fire_only: bool = True) -> pd.DataFrame:
    """One lat/lon per wilaya: where that wilaya's fires actually happen."""
    if not IGNITION_FILE.exists():
        sys.exit(f"{IGNITION_FILE} not found — run prepare_ignitions.py first.")
    ign = pd.read_parquet(IGNITION_FILE)

    counts = ign.groupby("ADM1_NAME", observed=True).agg(
        lat=("lat", "mean"), lon=("lon", "mean"), n_ignitions=("lat", "size"))
    well_sampled = counts[counts["n_ignitions"] >= MIN_IGNITIONS_FOR_CENTROID]

    chosen = {
        name: dict(lat=r["lat"], lon=r["lon"],
                   n_ignitions=int(r["n_ignitions"]), basis="ignition centroid")
        for name, r in well_sampled.iterrows()
    }

    if BOUNDARY_FILE.exists():
        import geopandas as gpd
        adm = gpd.read_file(BOUNDARY_FILE)
        # Every wilaya the dashboard can draw gets a row, so the weather layer
        # joins cleanly; those without enough fires to locate keep the polygon.
        for name, geom in zip(adm["ADM1_NAME"].astype(str), adm.geometry):
            if name in chosen:
                continue
            p = geom.representative_point()
            chosen[name] = dict(lat=p.y, lon=p.x,
                                n_ignitions=int(counts["n_ignitions"].get(name, 0)),
                                basis="polygon centre")
    elif len(chosen) < len(counts):
        print(f"  note: {len(counts) - len(chosen)} wilaya(s) have under "
              f"{MIN_IGNITIONS_FOR_CENTROID} ignitions and no boundary file "
              f"to fall back on; they are skipped")

    pts = (pd.DataFrame.from_dict(chosen, orient="index")
             .rename_axis("ADM1_NAME").reset_index())

    if fire_only:
        # The default. The dashboard's national aggregate already averages
        # only these, and a wilaya with no recorded fires has no fire weather
        # worth a request against a volume-capped free API.
        skipped = int((pts["basis"] != "ignition centroid").sum())
        pts = pts[pts["basis"] == "ignition centroid"]
        if skipped:
            print(f"  skipping {skipped} wilaya(s) with under "
                  f"{MIN_IGNITIONS_FOR_CENTROID} recorded ignitions "
                  f"(pass --all-wilayas to include them)")

    # Busiest first, not alphabetical. Open-Meteo's free tier will not serve
    # the whole record in one window, so a run is usually interrupted — and
    # what it managed to fetch should be the wilayas that carry the fires.
    # Alphabetical order stopped at Guelma and left out Skikda, Tizi Ouzou,
    # Jijel and Medea: four of the six busiest, and the Kabylie fire with them.
    pts = pts.sort_values("n_ignitions", ascending=False).reset_index(drop=True)
    return pts.head(limit) if limit else pts


def fetch_daily(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """Daily ERA5 for one point."""
    q = urllib.parse.urlencode({
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARS), "timezone": TIMEZONE,
    })
    # Open-Meteo's free tier is rate-limited per minute and weights a request
    # by how much data it returns, so a 26-year daily pull is expensive and a
    # 429 partway through a 48-wilaya run is normal rather than exceptional.
    # Wait it out instead of discarding the wilayas already fetched.
    payload = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=180) as r:
                payload = json.load(r)
            break
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            hourly = "hourly" in detail.lower()
            # One wait for a minutely limit, then give up on this run. An
            # escalating ladder here was spending a quarter of each hourly
            # window sleeping, which is budget that could have been fetching:
            # repeated minutely rejections mean the hourly budget is nearly
            # gone anyway, and the caller resumes from disk.
            if exc.code == 429 and not hourly and attempt == 0:
                print(f"      minutely limit; waiting {RETRY_WAIT_S}s once")
                time.sleep(RETRY_WAIT_S)
                continue
            if exc.code == 429:
                # The hourly cap will not clear inside any sane retry loop.
                # Everything fetched so far is already on disk, so stop
                # cleanly and let the next run pick up where this one left off.
                raise RateLimited(detail)
            raise SystemExit(f"Open-Meteo returned HTTP {exc.code}. {detail}")
        except urllib.error.URLError as exc:
            # DNS and connection blips are transient. Retry briefly, then hand
            # back a resumable stop rather than a hard exit — a moment of bad
            # networking should not discard an hour of fetching.
            if attempt < MAX_RETRIES - 1:
                print(f"      {exc.reason}; retrying in 20s")
                time.sleep(20)
                continue
            raise RateLimited(f"network unreachable: {exc.reason}")
    if payload is None:
        raise SystemExit("Open-Meteo did not return data after retries.")

    if "daily" not in payload:
        raise SystemExit(f"Unexpected response: {str(payload)[:300]}")
    d = pd.DataFrame(payload["daily"])
    d["time"] = pd.to_datetime(d["time"])
    return d


def monthly(d: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one wilaya's daily series to months."""
    d = d.rename(columns={
        "temperature_2m_max": "t_max_c",
        "relative_humidity_2m_min": "rh_min_pct",
        "wind_speed_10m_max": "wind_max_kmh",
        "precipitation_sum": "precip_mm",
    })
    d["fire_weather_day"] = (
        (d["t_max_c"] >= FWD_TEMP_C)
        & (d["rh_min_pct"] <= FWD_RH_PCT)
        & (d["wind_max_kmh"] >= FWD_WIND_KMH)
    )
    d["year"] = d["time"].dt.year
    d["month"] = d["time"].dt.month
    out = d.groupby(["year", "month"]).agg(
        t_max_c=("t_max_c", "mean"),
        rh_min_pct=("rh_min_pct", "mean"),
        wind_max_kmh=("wind_max_kmh", "mean"),
        precip_mm=("precip_mm", "sum"),
        fire_weather_days=("fire_weather_day", "sum"),
        n_days=("time", "size"),
    ).reset_index()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build the fire-weather layer.")
    p.add_argument("--start", default="2001-01-01", metavar="YYYY-MM-DD",
                   help="first day to fetch (default 2001-01-01, matching the "
                        "burned-area record)")
    p.add_argument("--end", default=None, metavar="YYYY-MM-DD",
                   help="last day (default: today)")
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N wilayas — for a trial run")
    p.add_argument("--refresh", action="store_true",
                   help="refetch every wilaya instead of resuming")
    p.add_argument("--all-wilayas", action="store_true",
                   help="include wilayas with no recorded fires (they are "
                        "skipped by default — see sampling_points)")
    args = p.parse_args()

    start = pd.Timestamp(args.start).date()
    end = pd.Timestamp(args.end).date() if args.end else date.today()
    if start > end:
        sys.exit(f"--start {start} is after --end {end}.")

    pts = sampling_points(args.limit, fire_only=not args.all_wilayas)
    print(f"Fetching ERA5 daily weather for {len(pts)} wilayas, "
          f"{start} .. {end}")
    by_basis = pts["basis"].value_counts().to_dict()
    print("  sampling point: " + ", ".join(f"{v} at {k}" for k, v in by_basis.items()))

    # Resume. Open-Meteo's hourly cap stops a full run partway through, and a
    # 26-year pull per wilaya is too expensive to repeat for the sake of tidy
    # code: keep what is already fetched and only ask for the rest.
    frames: list[pd.DataFrame] = []
    done: set[str] = set()
    if OUTPUT_FILE.exists() and not args.refresh:
        prior = pd.read_parquet(OUTPUT_FILE)
        same_span = (int(prior["year"].min()) <= start.year
                     and int(prior["year"].max()) >= end.year - 1)
        if same_span:
            frames.append(prior)
            done = set(prior["ADM1_NAME"].astype(str))
            print(f"  resuming: {len(done)} wilaya(s) already on disk")
        else:
            print("  existing file covers a different span — refetching all")

    def save() -> Optional[pd.DataFrame]:
        if not frames:
            return None
        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(subset=["ADM1_NAME", "year", "month"],
                                  keep="last")
        for c in ("t_max_c", "rh_min_pct", "wind_max_kmh", "precip_mm"):
            out[c] = out[c].astype("float32")
        for c in ("year", "month", "fire_weather_days", "n_days"):
            out[c] = out[c].astype("int16")
        out["ADM1_NAME"] = out["ADM1_NAME"].astype("category")
        out["sample_basis"] = out["sample_basis"].astype("category")
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUTPUT_FILE, index=False)
        return out

    t0 = time.time()
    todo = [r for _, r in pts.iterrows() if str(r["ADM1_NAME"]) not in done]
    for i, row in enumerate(todo, 1):
        try:
            daily = fetch_daily(float(row["lat"]), float(row["lon"]), start, end)
        except RateLimited as exc:
            save()
            print(f"\n  Open-Meteo's hourly limit reached after "
                  f"{len(done) + i - 1} of {len(pts)} wilayas.")
            print(f"  {exc}")
            print(f"\n  Progress is saved. Re-run the same command in an hour "
                  f"and it will\n  continue from {row['ADM1_NAME']}.")
            sys.exit(2)
        m = monthly(daily)
        m.insert(0, "ADM1_NAME", str(row["ADM1_NAME"]))
        m["sample_lat"] = float(row["lat"])
        m["sample_lon"] = float(row["lon"])
        m["sample_basis"] = str(row["basis"])
        frames.append(m)
        print(f"  [{len(done) + i:>2}/{len(pts)}] {row['ADM1_NAME']:<20} "
              f"{len(daily):,} days -> {len(m)} months", flush=True)
        time.sleep(PAUSE_S)

    out = save()
    if out is None:
        sys.exit("Nothing fetched and nothing on disk — no data written.")

    print(f"\nOK  {OUTPUT_FILE}  ({OUTPUT_FILE.stat().st_size / 1e6:.2f} MB) "
          f"in {time.time() - t0:.0f}s")
    print(f"  {len(out):,} wilaya-months | {out['year'].min()}-{out['year'].max()}"
          f" | {out['ADM1_NAME'].nunique()} wilayas")
    print(f"\n  A fire-weather day is Tmax >= {FWD_TEMP_C:.0f} C, "
          f"RHmin <= {FWD_RH_PCT:.0f} %, wind >= {FWD_WIND_KMH:.0f} km/h.")
    season = out[out["month"].isin([6, 7, 8, 9])]
    worst = (season.groupby("year")["fire_weather_days"].sum()
                   .sort_values(ascending=False).head(5))
    print("\n  Most fire-weather days nationally, Jun-Sep:")
    for y, n in worst.items():
        print(f"    {y}  {n:,}")


if __name__ == "__main__":
    main()
