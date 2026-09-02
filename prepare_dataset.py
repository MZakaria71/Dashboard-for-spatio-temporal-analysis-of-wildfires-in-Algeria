"""
prepare_dataset.py
──────────────────
Converts the two CSVs downloaded from GEE into Parquet files
optimised for the Streamlit dashboard.

GEE exports at ADM2 (commune) level. This script also derives the ADM1
(wilaya) tables by groupby aggregation — no extra GEE computation needed.

Expected input (place in the same folder as this script):
    burned_area_adm2_month.csv
    landcover_adm2_year.csv

Output → data/
    burned_area_adm2.parquet   commune × year × month  (~4 MB)
    burned_area_adm1.parquet   wilaya  × year × month  (~0.3 MB)
    landcover_adm2.parquet     commune × year           (~0.3 MB)
    landcover_adm1.parquet     wilaya  × year           (~0.05 MB)
    admin_hierarchy.json       ADM1 → [ADM2] map for UI

Usage:
    pip install pandas pyarrow
    python prepare_dataset.py
"""

import glob
import json
import sys
from pathlib import Path

import pandas as pd

# Windows consoles default to cp1252, which cannot encode the arrows, ellipses
# and superscripts used in this script's progress output — printing one raises
# UnicodeEncodeError and kills the run after the work is already done.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):                     # non-reconfigurable stream
    pass

# ── paths ──────────────────────────────────────────────────────────────────
INPUT_DIR  = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

# Re-exporting from GEE usually leaves several generations of the same table
# side by side — the browser appends "(1)", "(2)" … to repeat downloads, so the
# newest export is often NOT the one with the plain name. Picking by filename
# silently reprocesses stale data, so pick by what the file actually contains:
# the widest year coverage wins, and every candidate is printed.
SEARCH_DIRS = [INPUT_DIR, OUTPUT_DIR]


def pick_csv(stem: str) -> Path:
    """Choose the export covering the most years, reporting all candidates."""
    paths = []
    for d in SEARCH_DIRS:
        paths.extend(glob.glob(str(d / f"{stem}*.csv")))
    paths = sorted(set(paths))
    if not paths:
        sys.exit(f"No {stem}*.csv found in {[str(d) for d in SEARCH_DIRS]}")

    scored = []
    for p in paths:
        try:
            years = pd.read_csv(p, usecols=["year"])["year"]
            scored.append((years.nunique(), int(years.min()), int(years.max()),
                           len(years), p))
        except Exception as exc:                      # unreadable / wrong schema
            print(f"    skipping {Path(p).name}: {exc}")

    if not scored:
        sys.exit(f"No readable {stem}*.csv found.")

    scored.sort(reverse=True)
    print(f"  {stem}:")
    for n_years, y0, y1, rows, p in scored:
        mark = "->" if p == scored[0][4] else "  "
        print(f"    {mark} {Path(p).name:<34} {y0}-{y1}  "
              f"{n_years:>2} years  {rows:>8,} rows")
    return Path(scored[0][4])


print("Selecting input CSVs (widest year coverage wins):")
BURNED_CSV    = pick_csv("burned_area_adm2_month")
LANDCOVER_CSV = pick_csv("landcover_adm2_year")
print()

BURN_COLS = [
    "burned_forest_km2", "burned_shrubland_km2",
    "burned_cropland_km2", "burned_other_km2", "burned_total_km2"
]
LC_COLS = ["forest_km2", "shrubland_km2", "cropland_km2", "other_km2", "total_km2"]

SEASON_MAP = {
    12: "Winter", 1: "Winter",  2: "Winter",
    3:  "Spring", 4: "Spring",  5: "Spring",
    6:  "Summer", 7: "Summer",  8: "Summer",
    9:  "Autumn", 10: "Autumn", 11: "Autumn",
}

GEE_SYSTEM_COLS = {"system:index", ".geo"}


def drop_gee_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c in GEE_SYSTEM_COLS])


def cast_admin_cols(df: pd.DataFrame, levels: list[str]) -> pd.DataFrame:
    for lvl in levels:
        df[f"ADM{lvl}_CODE"] = df[f"ADM{lvl}_CODE"].astype("int32")
        df[f"ADM{lvl}_NAME"] = df[f"ADM{lvl}_NAME"].astype("category")
    return df


def add_lc_pct(df: pd.DataFrame) -> pd.DataFrame:
    for cls in ["forest", "shrubland", "cropland", "other"]:
        col = f"{cls}_km2"
        if col in df.columns:
            df[f"{cls}_pct"] = (
                (df[col] / df["total_km2"].replace(0, float("nan"))) * 100
            ).round(2).astype("float32")
    return df


# ── 1. BURNED AREA — ADM2 ─────────────────────────────────────────────────
print("Loading burned area CSV (ADM2)…")
b2 = pd.read_csv(BURNED_CSV)
b2 = drop_gee_cols(b2)
b2 = cast_admin_cols(b2, ["1", "2"])
b2["year"]   = b2["year"].astype("int16")
b2["month"]  = b2["month"].astype("int8")
b2[BURN_COLS] = b2[BURN_COLS].fillna(0).astype("float32")
b2["season"] = b2["month"].map(SEASON_MAP).astype("category")

print(f"  Rows: {len(b2):,}   Columns: {list(b2.columns)}")
b2.to_parquet(OUTPUT_DIR / "burned_area_adm2.parquet", index=False)
print(f"  → saved burned_area_adm2.parquet")

# ── 2. BURNED AREA — ADM1 (derived by aggregation) ────────────────────────
print("\nDeriving burned area ADM1 from ADM2…")
b1 = (
    b2.groupby(["ADM1_CODE", "ADM1_NAME", "year", "month", "season"], observed=True)[BURN_COLS]
    .sum()
    .reset_index()
)
# Restore compact dtypes after groupby
b1[BURN_COLS] = b1[BURN_COLS].astype("float32")
b1["ADM1_NAME"] = b1["ADM1_NAME"].astype("category")
b1["season"]    = b1["season"].astype("category")

print(f"  Rows: {len(b1):,}   (expected wilayas × 240)")
b1.to_parquet(OUTPUT_DIR / "burned_area_adm1.parquet", index=False)
print(f"  → saved burned_area_adm1.parquet")

# ── 3. LAND COVER — ADM2 ─────────────────────────────────────────────────
print("\nLoading land cover CSV (ADM2)…")
lc2 = pd.read_csv(LANDCOVER_CSV)
lc2 = drop_gee_cols(lc2)
lc2 = cast_admin_cols(lc2, ["1", "2"])
lc2["year"]   = lc2["year"].astype("int16")
lc2[LC_COLS]  = lc2[LC_COLS].fillna(0).astype("float32")
lc2 = add_lc_pct(lc2)

print(f"  Rows: {len(lc2):,}   Columns: {list(lc2.columns)}")
lc2.to_parquet(OUTPUT_DIR / "landcover_adm2.parquet", index=False)
print(f"  → saved landcover_adm2.parquet")

# ── 4. LAND COVER — ADM1 (derived by aggregation) ─────────────────────────
print("\nDeriving land cover ADM1 from ADM2…")
lc1 = (
    lc2.groupby(["ADM1_CODE", "ADM1_NAME", "year"], observed=True)[LC_COLS]
    .sum()
    .reset_index()
)
lc1[LC_COLS]    = lc1[LC_COLS].astype("float32")
lc1["ADM1_NAME"] = lc1["ADM1_NAME"].astype("category")
lc1 = add_lc_pct(lc1)

print(f"  Rows: {len(lc1):,}   (expected wilayas × 20)")
lc1.to_parquet(OUTPUT_DIR / "landcover_adm1.parquet", index=False)
print(f"  → saved landcover_adm1.parquet")

# ── 5. ADMIN HIERARCHY JSON — for UI dropdowns ────────────────────────────
print("\nBuilding admin hierarchy…")

# Unique ADM1 → list of ADM2s (sorted alphabetically)
hierarchy = {}
adm_pairs = (
    b2[["ADM1_CODE", "ADM1_NAME", "ADM2_CODE", "ADM2_NAME"]]
    .drop_duplicates()
    .sort_values(["ADM1_NAME", "ADM2_NAME"])
)

for _, row in adm_pairs.iterrows():
    key = str(row["ADM1_CODE"])
    if key not in hierarchy:
        hierarchy[key] = {
            "code": int(row["ADM1_CODE"]),
            "name": str(row["ADM1_NAME"]),
            "communes": []
        }
    hierarchy[key]["communes"].append({
        "code": int(row["ADM2_CODE"]),
        "name": str(row["ADM2_NAME"])
    })

hierarchy_list = sorted(hierarchy.values(), key=lambda x: x["name"])

with open(OUTPUT_DIR / "admin_hierarchy.json", "w", encoding="utf-8") as f:
    json.dump(hierarchy_list, f, ensure_ascii=False, indent=2)

n_wilayas  = len(hierarchy_list)
n_communes = sum(len(w["communes"]) for w in hierarchy_list)
print(f"  → saved admin_hierarchy.json  ({n_wilayas} wilayas, {n_communes} communes)")

# ── 6. VALIDATION ─────────────────────────────────────────────────────────
print("\n── Validation ────────────────────────────────────────────────────────")

national = (
    b1.groupby("year")["burned_total_km2"].sum()
    .rename("national_km2")
    .reset_index()
)
print("\nNational burned area by year (km²) — ADM1 aggregated:")
print(national.to_string(index=False))

monthly = b1.groupby("month")["burned_total_km2"].sum()
print(f"\nPeak fire month: {monthly.idxmax()}  ({monthly.max():.0f} km² "
      f"total {int(b1['year'].min())}-{int(b1['year'].max())})")

top_wilayas = (
    b1.groupby("ADM1_NAME", observed=True)["burned_total_km2"]
    .sum().sort_values(ascending=False).head(5)
)
print("\nTop 5 most affected wilayas:")
print(top_wilayas.to_string())

top_communes = (
    b2.groupby(["ADM1_NAME", "ADM2_NAME"], observed=True)["burned_total_km2"]
    .sum().sort_values(ascending=False).head(10)
)
print("\nTop 10 most affected communes:")
print(top_communes.to_string())

print("\n✓ Dataset preparation complete.")
print(f"\nFiles in {OUTPUT_DIR}:")
for p in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {p.name:<35} {p.stat().st_size / 1024:>7.1f} KB")
