# ===============================
# IMPORT PACKAGES
# ===============================
import geopandas as gpd
import pathlib
import matplotlib.pyplot as plt

# ===============================
# PATHS
# ===============================
base_dir = pathlib.Path(".")

zwe_adm_dir = base_dir / "ZWE_adm"
wards_dir = base_dir / "wards"

# ===============================
# LOAD DATA
# ===============================
gdf_country = gpd.read_file(zwe_adm_dir / "ZWE_adm0.shp")
gdf_prov    = gpd.read_file(zwe_adm_dir / "ZWE_adm1.shp")
gdf_dist    = gpd.read_file(zwe_adm_dir / "ZWE_adm2.shp")
gdf_wards   = gpd.read_file(wards_dir / "zwe_polbnda_adm3_250k_cso.shp")

# ===============================
# FIX CRS (ONE SOURCE OF TRUTH)
# ===============================
target_crs = gdf_country.crs

def fix_crs(gdf, target_crs):
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)  # assume WGS84
    return gdf.to_crs(target_crs)

gdf_prov  = fix_crs(gdf_prov, target_crs)
gdf_dist  = fix_crs(gdf_dist, target_crs)
gdf_wards = fix_crs(gdf_wards, target_crs)

# ===============================
# CLIP WARDS (ONLY HEAVY LAYER)
# ===============================
gdf_wards = gpd.clip(gdf_wards, gdf_country)

# ===============================
# PLOT (HIERARCHY MATTERS)
# ===============================
fig, ax = plt.subplots(figsize=(14, 10))

# 1. Wards (most detailed → bottom layer)
gdf_wards.plot(
    ax=ax,
    facecolor='none',
    edgecolor='lightgray',
    linewidth=0.2
)

# 2. Districts
gdf_dist.plot(
    ax=ax,
    facecolor='none',
    edgecolor='gray',
    linewidth=0.5
)

# 3. Provinces
gdf_prov.plot(
    ax=ax,
    facecolor='none',
    edgecolor='black',
    linewidth=1.2
)

# 4. Country boundary (strongest emphasis)
gdf_country.plot(
    ax=ax,
    facecolor='none',
    edgecolor='black',
    linewidth=2.0
)

# ===============================
# LABELS (PROVINCES ONLY - CLEAN)
# ===============================
for _, row in gdf_prov.iterrows():
    centroid = row.geometry.centroid
    ax.text(
        centroid.x, centroid.y,
        row["NAME_1"],
        fontsize=9,
        ha='center',
        weight='bold'
    )

# ===============================
# FINAL FORMATTING
# ===============================
ax.set_title(
    "Zimbabwe Administrative Boundaries\n(Wards, Districts, Provinces)",
    fontsize=14,
    fontweight='bold'
)

ax.axis('off')

plt.tight_layout()
plt.show()

# ===============================
# OUTPUT CHECK
# ===============================
print("Total wards:", len(gdf_wards))
print("CRS:", gdf_wards.crs)