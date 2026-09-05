#!/usr/bin/env python3
"""
fetch_firms.py
──────────────
Pull recent NASA FIRMS active-fire detections from the FIRMS API and write
them where prepare_ignitions.py will find them.

The FIRMS "Download" page only publishes its quality-controlled archive
roughly three months behind real time, and ships the gap as a near-real-time
member inside the same zip. That member ages the moment it is downloaded, so
keeping the ignition record current otherwise means re-requesting the whole
archive by hand. This does the same job in one command.

Overlapping fetches are safe. prepare_ignitions.py de-duplicates detections on
(satellite, timestamp, position), so re-running this over a period already
covered adds nothing rather than double-counting it.

    MAP KEY
    -------
    Free, issued instantly, and NOT a secret in the password sense — but it is
    yours and rate-limited, so keep it out of shell history and out of the
    repo. Get one at https://firms.modaps.eosdis.nasa.gov/api/map_key/ and put
    it in the environment:

        PowerShell   $env:FIRMS_MAP_KEY = "your-key"
        bash         export FIRMS_MAP_KEY=your-key

    This script never prints the key, and redacts it from any error it raises.

    USAGE
    -----
        python fetch_firms.py                     # the last 10 days
        python fetch_firms.py --days 30           # the last 30 days
        python fetch_firms.py --since 2026-08-20  # from a date to today
        python fetch_firms.py --source VIIRS_SNPP_NRT

    Then rebuild:

        python prepare_ignitions.py

    The FIRMS API serves at most 10 days per request, so longer windows are
    fetched in chunks. Requests are also rate-limited per key (the limit and
    your current usage are printed before anything is fetched).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd

# Windows consoles default to cp1252 and cannot encode this script's output.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

OUTPUT_DIR = Path(__file__).parent / "data"

API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
KEY_STATUS = "https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/"

# west,south,east,north — Algeria plus a small margin. prepare_ignitions.py
# clips properly against the GAUL boundaries, so a loose box costs nothing but
# a few rows.
ALGERIA_BBOX = "-9.0,18.5,12.5,37.5"

MAX_DAYS_PER_REQUEST = 10          # hard API limit

# FIRMS API source names, and the product code prepare_ignitions.py knows them
# by. The output filename carries the product code so _wanted() matches it
# without any change to the loader.
SOURCES = {
    "MODIS_NRT":        "M-C61",
    "MODIS_SP":         "M-C61",
    "VIIRS_SNPP_NRT":   "SV-C2",
    "VIIRS_SNPP_SP":    "SV-C2",
    "VIIRS_NOAA20_NRT": "J1V-C2",
    "VIIRS_NOAA21_NRT": "J2V-C2",
}


class FirmsError(RuntimeError):
    pass


def _redact(text: str, key: str) -> str:
    return text.replace(key, "<MAP_KEY>") if key else text


def _get(url: str, key: str) -> str:
    """GET a URL, keeping the map key out of anything that reaches the user."""
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise FirmsError(
            f"FIRMS returned HTTP {exc.code} ({exc.reason}). "
            f"URL: {_redact(url, key)}"
        ) from None
    except urllib.error.URLError as exc:
        raise FirmsError(f"Could not reach FIRMS: {exc.reason}") from None


def check_key(key: str) -> None:
    """Fail early on a bad key, and show how much quota is left."""
    try:
        raw = _get(f"{KEY_STATUS}?MAP_KEY={key}", key)
    except FirmsError as exc:
        # FIRMS answers a bad key with a bare 403 on this endpoint.
        if "HTTP 401" in str(exc) or "HTTP 403" in str(exc):
            raise FirmsError(
                "FIRMS rejected the map key in FIRMS_MAP_KEY (HTTP 403).\n"
                "  Check it at https://firms.modaps.eosdis.nasa.gov/api/map_key/"
                " — keys expire if unused."
            ) from None
        raise
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        raise FirmsError(
            "FIRMS did not return a readable key status. Response began:\n"
            f"  {_redact(raw[:200], key)}"
        ) from None

    if isinstance(info, dict) and info.get("error"):
        raise FirmsError(f"FIRMS rejected the map key: {info['error']}")

    # Field names have changed before, so report whatever came back rather
    # than assuming a shape — minus the key itself.
    shown = {k: v for k, v in info.items() if "key" not in k.lower()}
    if shown:
        print("  key status: " + ", ".join(f"{k}={v}" for k, v in shown.items()))


def fetch_window(key: str, source: str, bbox: str,
                 start: date, days: int) -> pd.DataFrame:
    """One API call: `days` days of detections beginning on `start`."""
    url = f"{API}/{key}/{source}/{bbox}/{days}/{start:%Y-%m-%d}"
    body = _get(url, key)

    # A rejected request comes back as plain prose with a 200, not an HTTP
    # error, so sniff for the CSV header rather than trusting the status.
    head = body.lstrip()[:400]
    if "latitude" not in head.lower():
        raise FirmsError(
            "FIRMS returned no CSV. It said:\n  "
            + _redact(head.strip() or "(empty response)", key)
        )

    df = pd.read_csv(StringIO(body))
    print(f"  {start:%Y-%m-%d} +{days:>2}d  {len(df):>7,} detections")
    return df


def daterange_chunks(start: date, end: date, size: int):
    """Split [start, end] into windows of at most `size` days."""
    cur = start
    while cur <= end:
        days = min(size, (end - cur).days + 1)
        yield cur, days
        cur += timedelta(days=days)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fetch recent FIRMS active-fire detections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--days", type=int, default=10,
                   help="how many days back from today (default 10)")
    g.add_argument("--since", metavar="YYYY-MM-DD",
                   help="fetch from this date to today instead")
    p.add_argument("--source", default="MODIS_NRT", choices=sorted(SOURCES),
                   help="FIRMS product (default MODIS_NRT, matching "
                        "PRODUCTS in prepare_ignitions.py)")
    p.add_argument("--bbox", default=ALGERIA_BBOX,
                   help="west,south,east,north (default: Algeria)")
    p.add_argument("--out", type=Path, default=None,
                   help="output CSV (default: data/fire_nrt_<PRODUCT>_api_"
                        "<start>_<end>.csv)")
    args = p.parse_args()

    import os
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not key:
        sys.exit(
            "FIRMS_MAP_KEY is not set.\n"
            "  Get a free key at https://firms.modaps.eosdis.nasa.gov/api/map_key/\n"
            "  then, in this shell:\n"
            '    PowerShell   $env:FIRMS_MAP_KEY = "your-key"\n'
            "    bash         export FIRMS_MAP_KEY=your-key\n"
            "  Passing it as an argument would leave it in your shell history, "
            "so this script only reads the environment."
        )

    today = date.today()
    if args.since:
        try:
            start = datetime.strptime(args.since, "%Y-%m-%d").date()
        except ValueError:
            sys.exit(f"--since must be YYYY-MM-DD, got {args.since!r}")
        if start > today:
            sys.exit(f"--since {start} is in the future.")
    else:
        if args.days < 1:
            sys.exit("--days must be at least 1.")
        start = today - timedelta(days=args.days - 1)

    product = SOURCES[args.source]
    span = (today - start).days + 1
    n_requests = len(list(daterange_chunks(start, today, MAX_DAYS_PER_REQUEST)))

    print(f"Fetching {args.source} ({product}) for {start} .. {today} "
          f"— {span} days in {n_requests} request(s)")
    try:
        check_key(key)
        frames = [fetch_window(key, args.source, args.bbox, s, d)
                  for s, d in daterange_chunks(start, today, MAX_DAYS_PER_REQUEST)]
    except FirmsError as exc:
        sys.exit(f"\n{exc}")

    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        sys.exit("\nFIRMS returned no detections for that window and area. "
                 "Nothing written.")

    # Same identity rule prepare_ignitions.py uses, applied here so the file
    # is clean on its own: consecutive windows can repeat a boundary day.
    before = len(df)
    df = df.drop_duplicates(
        subset=[c for c in ("satellite", "acq_date", "acq_time",
                            "latitude", "longitude") if c in df.columns])
    if before != len(df):
        print(f"  dropped {before - len(df):,} repeats across request windows")

    covered = sorted(pd.to_datetime(df["acq_date"]).dt.date.unique())
    out = args.out or (
        OUTPUT_DIR / f"fire_nrt_{product}_api_"
                     f"{covered[0]:%Y%m%d}_{covered[-1]:%Y%m%d}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nOK  {out}")
    print(f"  {len(df):,} detections | {covered[0]} .. {covered[-1]}")

    missing = [d for d in pd.date_range(start, today, freq="D").date
               if d not in set(covered)]
    if missing:
        shown = ", ".join(str(d) for d in missing[:8])
        more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        print(f"  note: {len(missing)} requested day(s) came back empty: "
              f"{shown}{more}")
        print("        Usually genuine — a quiet day has no detections — but a "
              "run of them\n        near today can mean the product has not "
              "published yet.")

    older = sorted(OUTPUT_DIR.glob(f"fire_nrt_{product}_api_*.csv"))
    older = [o for o in older if o.resolve() != out.resolve()]
    if older:
        print(f"\n  {len(older)} earlier fetch(es) for {product} are still in "
              f"data/. They are\n  de-duplicated on load, so leaving them is "
              f"harmless; delete them if you\n  want a tidy folder:")
        for o in older:
            print(f"    {o.name}")

    print("\nNext:  python prepare_ignitions.py")


if __name__ == "__main__":
    main()
