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

    The FIRMS API serves at most 5 days per request, so longer windows are
    fetched in chunks. Requests are also rate-limited per key (the limit and
    your current usage are printed before anything is fetched).

    The near-real-time sources are a rolling window — MODIS_NRT holds roughly
    four months — so this checks what FIRMS actually covers before fetching
    and clamps the window to it. For anything older, MODIS_SP is the archive
    equivalent and picks up exactly where MODIS_NRT stops; for a bulk backfill
    the zip from the Download page beats 5 days per request.
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
AVAILABILITY = "https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv"

# west,south,east,north — Algeria plus a small margin. prepare_ignitions.py
# clips properly against the GAUL boundaries, so a loose box costs nothing but
# a few rows.
ALGERIA_BBOX = "-9.0,18.5,12.5,37.5"

# Hard API limit. FIRMS rejects anything else with
# "Invalid day range. Expects [1..5]" — the number is not documented next to
# the endpoint, so if it changes the error text now says so plainly.
MAX_DAYS_PER_REQUEST = 5

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
        # HTTPError is itself a response: FIRMS explains a 400 in the body, and
        # throwing that away turns a precise complaint into a bare status code.
        try:
            detail = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            detail = ""
        parts = [f"FIRMS returned HTTP {exc.code} ({exc.reason})."]
        if detail:
            parts.append("  It said: " + _redact(detail[:600], key))
        parts.append("  URL: " + _redact(url, key))
        raise FirmsError("\n".join(parts)) from None
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


def fetch_availability(key: str) -> dict[str, tuple[date, date]]:
    """What each FIRMS source covers, as {source: (min_date, max_date)}.

    Worth one call per run. The near-real-time sources are a rolling window —
    MODIS_NRT holds only about four months — so a --since older than that
    silently returns nothing useful, and asking first turns that into a
    sentence instead of an empty file.
    """
    body = _get(f"{AVAILABILITY}/{key}/ALL", key)
    table = pd.read_csv(StringIO(body))
    out: dict[str, tuple[date, date]] = {}
    for _, r in table.iterrows():
        try:
            out[str(r["data_id"])] = (
                pd.to_datetime(r["min_date"]).date(),
                pd.to_datetime(r["max_date"]).date(),
            )
        except (KeyError, ValueError, TypeError):
            continue
    return out


def show_availability(key: str) -> None:
    """Print what FIRMS says it holds — the answer to most rejected requests."""
    try:
        body = _get(f"{AVAILABILITY}/{key}/ALL", key)
    except FirmsError as exc:
        print(f"\n  (could not read data availability: {exc})")
        return
    print("\n  FIRMS reports these sources and date ranges:")
    for line in body.strip().splitlines()[:25]:
        print("    " + line)


def diagnose(key: str, source: str, bbox: str, start: date) -> None:
    """Narrow down which part of a rejected request FIRMS objected to.

    A 400 names no field, and the area endpoint has four of them. Each probe
    below changes exactly one thing, so whichever one starts working is the
    answer.
    """
    probes = [
        ("same request, no date (most recent days instead)",
         f"{API}/{key}/{source}/{bbox}/1"),
        ("no date, small box over northern Algeria",
         f"{API}/{key}/{source}/2,35,8,37/1"),
        ("your date, small box",
         f"{API}/{key}/{source}/2,35,8,37/1/{start:%Y-%m-%d}"),
        ("VIIRS instead of MODIS, no date",
         f"{API}/{key}/VIIRS_SNPP_NRT/{bbox}/1"),
    ]
    print("\n  Narrowing it down:")
    for label, url in probes:
        try:
            body = _get(url, key)
            ok = "latitude" in body.lstrip()[:400].lower()
            rows = max(len(body.strip().splitlines()) - 1, 0)
            print(f"    {'OK  ' if ok else 'odd '} {label}"
                  + (f"  ({rows:,} rows)" if ok
                     else "  -> " + _redact(body.strip()[:120], key)))
        except FirmsError as exc:
            first = str(exc).splitlines()
            said = next((l.strip() for l in first if l.strip().startswith("It said:")),
                        first[0])
            print(f"    FAIL {label}  -> {said}")


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
                   help="how many days back from today (default 10). FIRMS "
                        "serves 5 days per request, so this is chunked.")
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
    end = today
    try:
        check_key(key)
        cover = fetch_availability(key)
    except FirmsError as exc:
        sys.exit(f"\n{exc}")

    if args.source in cover:
        lo, hi = cover[args.source]
        print(f"  {args.source} holds {lo} .. {hi}")
        if start < lo:
            # The NRT sources are a rolling window, so this is routine rather
            # than an error — say what is being skipped and where it lives.
            older = [s for s, p in SOURCES.items()
                     if p == product and s != args.source and s in cover
                     and cover[s][0] <= start]
            print(f"  !! {start} predates it; starting at {lo} instead.")
            if older:
                print(f"     Earlier days are in {', '.join(older)} — but for a "
                      f"bulk backfill\n     the archive zip from the Download "
                      f"page is far cheaper than\n     {MAX_DAYS_PER_REQUEST}"
                      f" days per request.")
            start = lo
        if end > hi:
            print(f"  !! {hi} is the latest it holds; stopping there.")
            end = hi
        if start > end:
            sys.exit(f"\nNothing to fetch: {args.source} covers {lo} .. {hi}.")

    span = (end - start).days + 1
    n_requests = len(list(daterange_chunks(start, end, MAX_DAYS_PER_REQUEST)))
    print(f"Fetching {args.source} ({product}) for {start} .. {end} "
          f"— {span} days in {n_requests} request(s)")
    try:
        frames = [fetch_window(key, args.source, args.bbox, s, d)
                  for s, d in daterange_chunks(start, end, MAX_DAYS_PER_REQUEST)]
    except FirmsError as exc:
        print(f"\n{exc}")
        if "HTTP 400" in str(exc):
            print("\n  A 400 means FIRMS parsed the request and rejected it —"
                  " the source name, the\n  bounding box, the day range or the"
                  " date. Working out which:")
            diagnose(key, args.source, args.bbox, start)
            show_availability(key)
        sys.exit(1)

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

    missing = [d for d in pd.date_range(start, end, freq="D").date
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
