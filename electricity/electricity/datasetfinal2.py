import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIG
# ============================================================
base = Path(__file__).resolve().parent

wards_path = base / "wards" / "zwe_polbnda_adm3_250k_cso.shp"
adm1_path = base / "ZWE_adm" / "ZWE_adm1.shp"
adm2_path = base / "ZWE_adm" / "ZWE_adm2.shp"

viirs_dir = base / "tif"
worldpop_dir = base / "WorldPop"
ghsl_dir = base / "GHSL"

output_csv = base / "final_dataset2.csv"

# If WorldPop is people-per-pixel use "count"
# If it is already density use "density"
WORLDPOP_MODE = "count"

# ============================================================
# HELPERS
# ============================================================
def extract_year_from_filename(filename):
    m = re.search(r"(19|20)\d{2}", filename)
    return int(m.group(0)) if m else None

def build_raster_lookup(folder):
    lookup = {}
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    for f in folder.iterdir():
        if f.suffix.lower() == ".tif":
            year = extract_year_from_filename(f.name)
            if year is not None:
                lookup[year] = str(f)

    return dict(sorted(lookup.items()))

def ensure_crs(gdf, fallback="EPSG:4326"):
    if gdf.crs is None:
        gdf = gdf.set_crs(fallback)
    return gdf

def get_metric_crs():
    # Zimbabwe
    return "EPSG:32735"

def assign_admin_by_largest_overlap(wards_gdf, admin_gdf, admin_id_col, output_col):
    """
    Assign each ward to the admin polygon with the largest intersection area.
    Fallback to nearest admin polygon if no intersection is found.
    """
    metric_crs = get_metric_crs()

    wards_m = wards_gdf[["ward_id", "geometry"]].copy().to_crs(metric_crs)
    admin_m = admin_gdf[[admin_id_col, "geometry"]].copy().to_crs(metric_crs)

    # First pass: candidates by intersects
    candidates = gpd.sjoin(
        wards_m,
        admin_m,
        how="left",
        predicate="intersects"
    )

    if len(candidates) > 0:
        # Bring in admin geometry explicitly for overlap computation
        admin_geom = admin_m.rename(columns={"geometry": "admin_geometry"})
        candidates = candidates.merge(
            admin_geom[[admin_id_col, "admin_geometry"]],
            on=admin_id_col,
            how="left"
        )

        # Compute overlap area
        candidates["overlap_area"] = candidates.apply(
            lambda r: r.geometry.intersection(r.admin_geometry).area
            if pd.notna(r[admin_id_col]) else 0.0,
            axis=1
        )

        # Keep best match per ward
        candidates = candidates.sort_values(
            ["ward_id", "overlap_area"],
            ascending=[True, False]
        )
        best = candidates.drop_duplicates(subset=["ward_id"])[["ward_id", admin_id_col]]
        best = best.rename(columns={admin_id_col: output_col})
    else:
        best = wards_m[["ward_id"]].copy()
        best[output_col] = np.nan

    wards_out = wards_gdf.merge(best, on="ward_id", how="left")

    # Fallback: nearest admin for any still missing
    missing_mask = wards_out[output_col].isna()
    if missing_mask.any():
        missing = wards_out.loc[missing_mask, ["ward_id", "geometry"]].copy().to_crs(metric_crs)

        nearest = gpd.sjoin_nearest(
            missing,
            admin_m[[admin_id_col, "geometry"]],
            how="left",
            distance_col="dist_to_admin"
        )[[ "ward_id", admin_id_col ]]

        nearest = nearest.drop_duplicates(subset=["ward_id"])
        nearest = nearest.rename(columns={admin_id_col: output_col})

        wards_out = wards_out.drop(columns=[output_col])
        wards_out = wards_out.merge(
            pd.concat([
                best.rename(columns={output_col: output_col}),
                nearest
            ]).drop_duplicates(subset=["ward_id"], keep="last"),
            on="ward_id",
            how="left"
        )

    return wards_out

def zonal_stat(wards_gdf, raster_path, stat="mean"):
    values = []

    with rasterio.open(raster_path) as src:
        wards_proj = wards_gdf.to_crs(src.crs)
        nodata = src.nodata

        for geom in wards_proj.geometry:
            try:
                out_image, _ = mask(src, [geom], crop=True, filled=False)
                arr = out_image[0]

                if hasattr(arr, "compressed"):
                    arr = arr.compressed()
                else:
                    arr = np.asarray(arr).ravel()

                if nodata is not None:
                    arr = arr[arr != nodata]

                arr = arr[np.isfinite(arr)]

                if arr.size == 0:
                    values.append(np.nan)
                    continue

                if stat == "mean":
                    values.append(float(arr.mean()))
                elif stat == "sum":
                    values.append(float(arr.sum()))
                elif stat == "median":
                    values.append(float(np.median(arr)))
                else:
                    raise ValueError(f"Unsupported stat: {stat}")

            except Exception:
                values.append(np.nan)

    return values

# ============================================================
# LOAD SHAPEFILES
# ============================================================
wards = gpd.read_file(wards_path)
adm1 = gpd.read_file(adm1_path)
adm2 = gpd.read_file(adm2_path)

wards = ensure_crs(wards)
adm1 = ensure_crs(adm1)
adm2 = ensure_crs(adm2)

target_crs = wards.crs
adm1 = adm1.to_crs(target_crs)
adm2 = adm2.to_crs(target_crs)

# ============================================================
# PREPARE WARDS
# ============================================================
if "ZIMWARDSID" not in wards.columns:
    raise ValueError("ZIMWARDSID not found in ward shapefile")

wards = wards[["ZIMWARDSID", "geometry"]].copy()
wards = wards.rename(columns={"ZIMWARDSID": "ward_id"})
wards["ward_id"] = wards["ward_id"].astype(str)

# one row per ward geometry
wards = wards.drop_duplicates(subset=["ward_id"]).reset_index(drop=True)

# fix invalid geometries if needed
wards["geometry"] = wards.buffer(0)
adm1["geometry"] = adm1.buffer(0)
adm2["geometry"] = adm2.buffer(0)

# ============================================================
# ASSIGN ADMINS
# ============================================================
if "ID_1" not in adm1.columns:
    raise ValueError("ID_1 not found in province shapefile")
if "ID_2" not in adm2.columns:
    raise ValueError("ID_2 not found in district shapefile")

wards = assign_admin_by_largest_overlap(wards, adm2, "ID_2", "district_id")
wards = assign_admin_by_largest_overlap(wards, adm1, "ID_1", "province_id")

if wards["district_id"].isna().any():
    missing = wards.loc[wards["district_id"].isna(), "ward_id"].tolist()[:10]
    raise ValueError(f"Some wards are still missing district assignment. Examples: {missing}")

if wards["province_id"].isna().any():
    missing = wards.loc[wards["province_id"].isna(), "ward_id"].tolist()[:10]
    raise ValueError(f"Some wards are still missing province assignment. Examples: {missing}")

# ============================================================
# WARD AREA
# ============================================================
metric_crs = get_metric_crs()
wards_metric = wards.to_crs(metric_crs)
wards["ward_area_km2"] = wards_metric.geometry.area / 1_000_000.0

# ============================================================
# RASTER LOOKUPS
# ============================================================
viirs_lookup = build_raster_lookup(viirs_dir)
worldpop_lookup = build_raster_lookup(worldpop_dir)
ghsl_lookup = build_raster_lookup(ghsl_dir)

common_years = sorted(set(viirs_lookup) & set(worldpop_lookup) & set(ghsl_lookup))
if not common_years:
    raise ValueError("No common years across VIIRS, WorldPop, and GHSL")

print("Common years:", common_years)

# ============================================================
# EXTRACT DATA
# ============================================================
records = []

for year in common_years:
    print(f"Processing year {year}...")

    radiance_vals = zonal_stat(wards, viirs_lookup[year], stat="mean")
    built_vals = zonal_stat(wards, ghsl_lookup[year], stat="mean")

    if WORLDPOP_MODE == "count":
        pop_total_vals = zonal_stat(wards, worldpop_lookup[year], stat="sum")
        population_density_vals = (
            np.array(pop_total_vals, dtype=float) /
            np.array(wards["ward_area_km2"], dtype=float)
        )
    elif WORLDPOP_MODE == "density":
        population_density_vals = zonal_stat(wards, worldpop_lookup[year], stat="mean")
    else:
        raise ValueError("WORLDPOP_MODE must be 'count' or 'density'")

    tmp = wards[["ward_id", "district_id", "province_id", "ward_area_km2"]].copy()
    tmp["year"] = year
    tmp["radiance"] = radiance_vals
    tmp["population_density"] = population_density_vals
    tmp["built_area"] = built_vals

    records.append(tmp)

df = pd.concat(records, ignore_index=True)

# ============================================================
# CLEAN
# ============================================================
df = (
    df.groupby(["ward_id", "district_id", "province_id", "ward_area_km2", "year"], as_index=False)
      .agg({
          "radiance": "mean",
          "population_density": "mean",
          "built_area": "mean"
      })
)

for col in ["radiance", "population_density", "built_area"]:
    df[col] = df.groupby("year")[col].transform(lambda s: s.fillna(s.median()))

df["radiance"] = pd.to_numeric(df["radiance"], errors="coerce").clip(lower=0)
df["population_density"] = pd.to_numeric(df["population_density"], errors="coerce").clip(lower=0)
df["built_area"] = pd.to_numeric(df["built_area"], errors="coerce").clip(lower=0)

# ============================================================
# URBAN FLAG
# ============================================================
parts = []
for year, sub in df.groupby("year"):
    sub = sub.copy()

    built_thr = sub["built_area"].quantile(0.70)
    pop_thr = sub["population_density"].quantile(0.70)
    rad_thr = sub["radiance"].quantile(0.75)

    sub["urban"] = (
        (
            (sub["built_area"] >= built_thr) &
            (sub["population_density"] >= pop_thr)
        ) |
        (sub["radiance"] >= rad_thr)
    ).astype(int)

    parts.append(sub)

df = pd.concat(parts, ignore_index=True)

# ============================================================
# FINAL
# ============================================================
if df.duplicated(["ward_id", "year"]).sum() > 0:
    raise ValueError("Duplicate ward-year rows still exist")

df["ward_id"] = df["ward_id"].astype(str)
df["district_id"] = df["district_id"].astype(str)
df["province_id"] = df["province_id"].astype(str)
df["year"] = df["year"].astype(int)
df["urban"] = df["urban"].astype(int)

df = df[
    [
        "ward_id",
        "district_id",
        "province_id",
        "year",
        "radiance",
        "population_density",
        "built_area",
        "urban",
    ]
].sort_values(["year", "province_id", "district_id", "ward_id"]).reset_index(drop=True)

print("\nRows:", len(df))
print("Unique wards:", df["ward_id"].nunique())
print("Duplicates ward-year:", df.duplicated(["ward_id", "year"]).sum())
print("\nUrban share by year:")
print(df.groupby("year")["urban"].mean())

df.to_csv(output_csv, index=False)
print(f"\nSaved: {output_csv}")