# ===============================
# IMPORT PACKAGES
# ===============================
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from pathlib import Path
import numpy as np
import pandas as pd

# ===============================
# PATH SETUP
# ===============================
base_dir = Path.cwd()
tif_dir = base_dir / "tif"

country_path = base_dir / "ZWE_adm" / "ZWE_adm0.shp"
province_path = base_dir / "ZWE_adm" / "ZWE_adm1.shp"
ward_path = base_dir / "wards" / "zwe_polbnda_adm3_250k_cso.shp"

output_dir = base_dir / "comparison2"
output_dir.mkdir(exist_ok=True)

# ===============================
# FILE PATHS (AUTO-DETECT)
# ===============================
rasters = {}

for file in tif_dir.glob("*.tif"):
    stem = file.stem

    # extract 4-digit year from filename
    parts = stem.split("_")
    year = None
    for p in parts:
        if p.isdigit() and len(p) == 4:
            year = p
            break

    if year is not None:
        rasters[year] = file

# keep only the comparison years you want
wanted_years = ["2015", "2019", "2024"]
rasters = {y: rasters[y] for y in wanted_years if y in rasters}

print("Loaded rasters:")
for y, p in rasters.items():
    print(y, "->", p)

if len(rasters) != len(wanted_years):
    missing = [y for y in wanted_years if y not in rasters]
    raise FileNotFoundError(f"Missing comparison raster(s) for year(s): {missing}")

# preserve order
rasters = {y: rasters[y] for y in wanted_years}

# ===============================
# LOAD BOUNDARIES
# ===============================
gdf_country = gpd.read_file(country_path)
gdf_province = gpd.read_file(province_path)
gdf_ward = gpd.read_file(ward_path)

if gdf_country.crs is None:
    gdf_country = gdf_country.set_crs("EPSG:4326")
if gdf_province.crs is None:
    gdf_province = gdf_province.set_crs("EPSG:4326")
if gdf_ward.crs is None:
    gdf_ward = gdf_ward.set_crs("EPSG:4326")

# fix invalid geometries if present
gdf_country["geometry"] = gdf_country.buffer(0)
gdf_province["geometry"] = gdf_province.buffer(0)
gdf_ward["geometry"] = gdf_ward.buffer(0)

# ===============================
# CHECK FILES
# ===============================
for year, raster_path in rasters.items():
    if not raster_path.exists():
        raise FileNotFoundError(f"{raster_path} NOT FOUND.")

# ===============================
# READ ALL RASTERS FIRST
# Use a common normalization range across years
# ===============================
raster_data = {}
all_positive = []

for year, raster_path in rasters.items():
    with rasterio.open(raster_path) as src:
        country_proj = gdf_country.to_crs(src.crs)
        province_proj = gdf_province.to_crs(src.crs)
        ward_proj = gdf_ward.to_crs(src.crs)

        data = src.read(1).astype("float32")

        # Handle nodata and non-positive as no electricity
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan

        positive_vals = data[np.isfinite(data) & (data > 0)]
        if positive_vals.size > 0:
            all_positive.append(positive_vals)

        raster_data[year] = {
            "data": data,
            "bounds": src.bounds,
            "country_proj": country_proj,
            "province_proj": province_proj,
            "ward_proj": ward_proj
        }

if len(all_positive) == 0:
    raise ValueError("No positive raster values found in the supplied TIFF files.")

all_positive = np.concatenate(all_positive)

# Global scaling across all years for fair comparison
global_vmin = np.nanpercentile(all_positive, 2)
global_vmax = np.nanpercentile(all_positive, 98)

# Safety
if global_vmax <= global_vmin:
    global_vmax = np.nanmax(all_positive)

# ===============================
# CLASSIFICATION
# Categories styled like your image
# 0 = No Electricity  -> black
# 1 = Very low access -> blue
# 2 = Low access      -> green
# 3 = Moderate access -> yellow
# 4 = High access     -> orange/red
# ===============================
category_labels = [
    "No Electricity",
    "Very low access",
    "Low access",
    "Moderate access",
    "High access"
]

category_colors = [
    "#000000",  # black
    "#1f77ff",  # blue
    "#7CFC00",  # light green
    "#FFD700",  # yellow/gold
    "#FF4500"   # orange-red
]

cmap = ListedColormap(category_colors)
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

def classify_raster(data, vmin, vmax):
    """
    Convert raster to 5 classes:
    0 No Electricity
    1 Very low access
    2 Low access
    3 Moderate access
    4 High access
    """
    classified = np.zeros(data.shape, dtype=np.int16)

    valid_mask = np.isfinite(data) & (data > 0)
    if not np.any(valid_mask):
        return classified

    clipped = np.clip(data[valid_mask], vmin, vmax)
    scaled = (clipped - vmin) / (vmax - vmin + 1e-12)

    cats = np.zeros_like(scaled, dtype=np.int16)
    cats[(scaled > 0.00) & (scaled <= 0.15)] = 1
    cats[(scaled > 0.15) & (scaled <= 0.35)] = 2
    cats[(scaled > 0.35) & (scaled <= 0.65)] = 3
    cats[(scaled > 0.65)] = 4

    classified[valid_mask] = cats
    return classified

# ===============================
# CREATE TABLE OF CATEGORY COUNTS
# ===============================
for year, info in raster_data.items():
    data = info["data"]
    classified = classify_raster(data, global_vmin, global_vmax)
    raster_data[year]["classified"] = classified

    unique, counts = np.unique(classified, return_counts=True)
    count_map = dict(zip(unique, counts))

    year_counts = {
        "No Electricity": int(count_map.get(0, 0)),
        "Very low access": int(count_map.get(1, 0)),
        "Low access": int(count_map.get(2, 0)),
        "Moderate access": int(count_map.get(3, 0)),
        "High access": int(count_map.get(4, 0))
    }

    raster_data[year]["category_counts"] = year_counts

# Build comparison table
comparison_rows = []
for label in category_labels:
    row = {"Category": label}
    for year in rasters.keys():
        row[year] = raster_data[year]["category_counts"][label]
    comparison_rows.append(row)

# Total row
total_row = {"Category": "Total"}
for year in rasters.keys():
    total_row[year] = sum(raster_data[year]["category_counts"].values())
comparison_rows.append(total_row)

comparison_df = pd.DataFrame(comparison_rows)

# Save CSV
comparison_csv_path = output_dir / "comparison_table.csv"
comparison_df.to_csv(comparison_csv_path, index=False)

# ===============================
# LEGEND HANDLES
# ===============================
legend_handles = [
    Patch(facecolor=color, edgecolor="black", label=label)
    for color, label in zip(category_colors, category_labels)
]

# ===============================
# PROVINCE LABEL SETUP
# ===============================
province_name_col = None
for col in ["NAME_1", "name", "province", "PROVINCE"]:
    if col in gdf_province.columns:
        province_name_col = col
        break

# ===============================
# SAVE INDIVIDUAL MAPS
# ===============================
for year, info in raster_data.items():
    classified = info["classified"]
    bounds = info["bounds"]
    country_proj = info["country_proj"]
    province_proj = info["province_proj"]
    ward_proj = info["ward_proj"]

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("black")

    ax.imshow(
        classified,
        extent=extent,
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest"
    )

    if year == "2024":
        # 2024: keep blue raster look like 2020,
        # but use boundaries like 2015/2019
        ward_proj.boundary.plot(
            ax=ax,
            edgecolor="#1f1f1f",
            linewidth=0.22,
            alpha=0.80,
            zorder=3
        )
        province_proj.boundary.plot(
            ax=ax,
            edgecolor="white",
            linewidth=1.2,
            alpha=1.0,
            zorder=4
        )
        country_proj.boundary.plot(
            ax=ax,
            edgecolor="white",
            linewidth=1.4,
            alpha=1.0,
            zorder=5
        )

        if province_name_col is not None:
            provinces_cent = province_proj.to_crs(epsg=3857).copy()
            provinces_cent["centroid"] = provinces_cent.geometry.centroid

            province_lab = province_proj.copy()
            province_lab["centroid"] = provinces_cent["centroid"].to_crs(province_proj.crs)

            for _, row in province_lab.iterrows():
                ax.text(
                    row["centroid"].x,
                    row["centroid"].y,
                    str(row[province_name_col]),
                    color="#cfcfcf",
                    fontsize=8,
                    ha="center",
                    va="center",
                    zorder=6
                )
    else:
        # 2015 and 2019 unchanged
        ward_proj.boundary.plot(
            ax=ax,
            edgecolor="#1f1f1f",
            linewidth=0.22,
            alpha=0.80,
            zorder=3
        )
        province_proj.boundary.plot(
            ax=ax,
            edgecolor="white",
            linewidth=1.2,
            alpha=1.0,
            zorder=4
        )
        country_proj.boundary.plot(
            ax=ax,
            edgecolor="white",
            linewidth=1.4,
            alpha=1.0,
            zorder=5
        )

        if province_name_col is not None:
            provinces_cent = province_proj.to_crs(epsg=3857).copy()
            provinces_cent["centroid"] = provinces_cent.geometry.centroid

            province_lab = province_proj.copy()
            province_lab["centroid"] = provinces_cent["centroid"].to_crs(province_proj.crs)

            for _, row in province_lab.iterrows():
                ax.text(
                    row["centroid"].x,
                    row["centroid"].y,
                    str(row[province_name_col]),
                    color="#cfcfcf",
                    fontsize=8,
                    ha="center",
                    va="center",
                    zorder=6
                )

    ax.set_title(f"Zimbabwe {year}", fontsize=13, fontweight="bold")
    ax.axis("off")

    ax.legend(
        handles=legend_handles,
        title="Legend",
        loc="lower left",
        fontsize=8,
        title_fontsize=9,
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / f"comparison_map_{year}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )
    plt.show()

# ===============================
# COMBINED FIGURE (3 maps + legend only)
# ===============================
fig = plt.figure(figsize=(14, 8), facecolor="white")
gs = fig.add_gridspec(
    nrows=2,
    ncols=3,
    height_ratios=[1.0, 0.28],
    width_ratios=[1.0, 1.0, 1.0],
    hspace=0.08,
    wspace=0.12
)

years = list(rasters.keys())

# Top row: 3 maps
for i, year in enumerate(years):
    ax = fig.add_subplot(gs[0, i])

    classified = raster_data[year]["classified"]
    bounds = raster_data[year]["bounds"]
    country_proj = raster_data[year]["country_proj"]
    province_proj = raster_data[year]["province_proj"]
    ward_proj = raster_data[year]["ward_proj"]

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    ax.set_facecolor("black")

    ax.imshow(
        classified,
        extent=extent,
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest"
    )

    if year == "2024":
        # TRUE 2020 STYLE

        ward_proj.boundary.plot(
            ax=ax,
            edgecolor="white",
            linewidth=0.10,
            alpha=0.30,
            zorder=3
        )

        province_proj.boundary.plot(
            ax=ax,
            edgecolor="black",
            linewidth=1.0,
            zorder=4
        )

        country_proj.boundary.plot(
            ax=ax,
            edgecolor="black",
            linewidth=1.2,
            zorder=5
        )

        if province_name_col is not None:
            provinces_cent = province_proj.to_crs(epsg=3857).copy()
            provinces_cent["centroid"] = provinces_cent.geometry.centroid

            province_lab = province_proj.copy()
            province_lab["centroid"] = provinces_cent["centroid"].to_crs(province_proj.crs)

            for _, row in province_lab.iterrows():
                ax.text(
                    row["centroid"].x,
                    row["centroid"].y,
                    str(row[province_name_col]),
                    color="black",  # key difference
                    fontsize=8,
                    ha="center",
                    va="center",
                    zorder=6
                )

        if province_name_col is not None:
            provinces_cent = province_proj.to_crs(epsg=3857).copy()
            provinces_cent["centroid"] = provinces_cent.geometry.centroid

            province_lab = province_proj.copy()
            province_lab["centroid"] = provinces_cent["centroid"].to_crs(province_proj.crs)

            for _, row in province_lab.iterrows():
                ax.text(
                    row["centroid"].x,
                    row["centroid"].y,
                    str(row[province_name_col]),
                    color="#cfcfcf",
                    fontsize=7,
                    ha="center",
                    va="center",
                    zorder=6
                )
    else:
        # 2015 and 2019 unchanged
        ward_proj.boundary.plot(
            ax=ax,
            edgecolor="#1f1f1f",
            linewidth=0.18,
            alpha=0.75,
            zorder=3
        )
        province_proj.boundary.plot(
            ax=ax,
            edgecolor="white",
            linewidth=1.1,
            zorder=4
        )
        country_proj.boundary.plot(
            ax=ax,
            edgecolor="white",
            linewidth=1.3,
            zorder=5
        )

        if province_name_col is not None:
            provinces_cent = province_proj.to_crs(epsg=3857).copy()
            provinces_cent["centroid"] = provinces_cent.geometry.centroid

            province_lab = province_proj.copy()
            province_lab["centroid"] = provinces_cent["centroid"].to_crs(province_proj.crs)

            for _, row in province_lab.iterrows():
                ax.text(
                    row["centroid"].x,
                    row["centroid"].y,
                    str(row[province_name_col]),
                    color="#cfcfcf",
                    fontsize=7,
                    ha="center",
                    va="center",
                    zorder=6
                )

    ax.set_title(year, fontsize=12, fontweight="bold")
    ax.axis("off")

# Bottom row: legend only, centered
ax_leg = fig.add_subplot(gs[1, :])
ax_leg.axis("off")
ax_leg.legend(
    handles=legend_handles,
    title="Legend",
    loc="center",
    ncol=len(legend_handles),
    fontsize=9,
    title_fontsize=10,
    frameon=True
)

plt.tight_layout()
plt.savefig(
    output_dir / "comparison_combined_figure.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)
plt.show()

print(f"Done. Outputs saved in: {output_dir}")
print(f"CSV table saved to: {comparison_csv_path}")