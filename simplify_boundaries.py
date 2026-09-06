#!/usr/bin/env python3
"""
simplify_boundaries.py
──────────────────────
Turns the full-resolution GAUL commune boundaries into a file small enough to
commit and to hand to a browser.

Why this exists
    data/gaul_adm2.geojson is 8.1 MB — 1,541 communes at 202k coordinates,
    exported at full precision by section 6 of gee_export.js. It is gitignored,
    so a clone has no commune geometry and the dashboard could only ever draw
    wilayas. Plotly also ships the whole FeatureCollection to the client on
    every render, and 8 MB of that is not something to put on a phone.

    Two things make it small. Douglas-Peucker at 0.001 degrees (~90 m here)
    drops the vertices that carry no visible detail at national scale, and
    rounding to four decimals (~11 m) throws away the sixteen digits GEE writes
    for coordinates that were never that accurate. Together: 8.1 MB -> 2.2 MB,
    which gzips to about 0.55 MB over the wire.

    Simplification runs per polygon, so a border shared by two communes can be
    generalised slightly differently on each side. At the zoom this map is read
    at the resulting slivers are sub-pixel; the alternative is a topology-aware
    pass (the topojson package) and another dependency for something nobody can
    see.

Usage
    python simplify_boundaries.py
    python simplify_boundaries.py --tolerance 0.002    # smaller, coarser

    Needs shapely. Only has to be re-run when the GAUL export changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shapely.geometry import mapping, shape
from shapely.ops import transform

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

HERE = Path(__file__).parent
SOURCE = HERE / "data" / "gaul_adm2.geojson"
OUTPUT = HERE / "data" / "gaul_adm2_simple.geojson"

# ~90 m at Algeria's latitudes. Communes on the coast are a few km across, so
# this generalises their outline without changing which one you are pointing at.
DEFAULT_TOLERANCE = 0.001
# ~11 m. Below the tolerance above, so it costs nothing the simplify kept.
DECIMALS = 4

KEEP_PROPS = ("ADM1_CODE", "ADM1_NAME", "ADM2_CODE", "ADM2_NAME")


def count_coords(geom) -> int:
    def walk(node) -> int:
        if not isinstance(node, (list, tuple)):
            return 0
        if node and isinstance(node[0], (int, float)):
            return 1
        return sum(walk(child) for child in node)

    if geom.get("type") == "GeometryCollection":
        return sum(count_coords(g) for g in geom.get("geometries", []))
    return walk(geom.get("coordinates", []))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Shrink the GAUL commune boundaries for the dashboard.")
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                   help=f"Douglas-Peucker tolerance in degrees "
                        f"(default {DEFAULT_TOLERANCE})")
    p.add_argument("--decimals", type=int, default=DECIMALS,
                   help=f"coordinate decimal places (default {DECIMALS})")
    args = p.parse_args()

    if not SOURCE.exists():
        sys.exit(f"{SOURCE} not found — export it from section 6 of "
                 f"gee_export.js first.")

    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    print(f"Read {SOURCE.name}: {len(raw['features']):,} features, "
          f"{SOURCE.stat().st_size / 1e6:.2f} MB")

    round_pt = lambda x, y, z=None: (round(x, args.decimals),
                                     round(y, args.decimals))

    features, dropped, repaired, coords_in, coords_out = [], [], 0, 0, 0
    for feat in raw["features"]:
        coords_in += count_coords(feat["geometry"])
        geom = shape(feat["geometry"])
        if args.tolerance:
            geom = geom.simplify(args.tolerance, preserve_topology=True)
        geom = transform(round_pt, geom)
        # Rounding snaps neighbouring vertices onto each other, which can leave
        # a ring of one repeated point or a hairline self-intersection. Both are
        # invisible in a file and fatal in a renderer — MapLibre builds the
        # whole trace or none of it, so nine collapsed rings out of 1,711 blank
        # the entire map. buffer(0) rebuilds the polygon from its own boundary,
        # dropping degenerate rings and resolving the crossings.
        if not geom.is_valid:
            geom = geom.buffer(0)
            repaired += 1
        # A commune small enough to vanish at this tolerance would draw as
        # nothing anyway; record it rather than shipping an empty feature.
        if geom.is_empty:
            dropped.append(feat["properties"].get("ADM2_NAME", "?"))
            continue
        out = mapping(geom)
        coords_out += count_coords(out)
        features.append({
            "type": "Feature",
            "properties": {k: feat["properties"][k] for k in KEEP_PROPS
                           if k in feat["properties"]},
            "geometry": out,
        })

    fc = {"type": "FeatureCollection", "features": features}
    OUTPUT.write_text(json.dumps(fc, separators=(",", ":")), encoding="utf-8")

    print(f"\nOK  {OUTPUT}  ({OUTPUT.stat().st_size / 1e6:.2f} MB)")
    print(f"  {len(features):,} features | {coords_in:,} -> {coords_out:,} "
          f"coordinates ({100 * coords_out / coords_in:.0f}% kept)")
    if repaired:
        print(f"  repaired {repaired} geometries left invalid by rounding")
    if dropped:
        print(f"  dropped {len(dropped)} empty after simplify: "
              f"{', '.join(dropped[:10])}")


if __name__ == "__main__":
    main()
