# ===============================
# IMPORT PACKAGES
# ===============================
import geopandas as gpd
import pathlib

# ===============================
# PATHS
# ===============================
base_dir = pathlib.Path(".")

zwe_adm_dir = base_dir / "ZWE_adm"

# ===============================
# LOAD DATA
# ===============================
gdf_country = gpd.read_file(zwe_adm_dir / "ZWE_adm0.shp")
gdf_prov    = gpd.read_file(zwe_adm_dir / "ZWE_adm1.shp")
gdf_dist    = gpd.read_file(zwe_adm_dir / "ZWE_adm2.shp")

# ===============================
# FIX CRS (CONSISTENCY)
# ===============================
target_crs = gdf_country.crs

def fix_crs(gdf, target_crs):
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    return gdf.to_crs(target_crs)

gdf_prov = fix_crs(gdf_prov, target_crs)
gdf_dist = fix_crs(gdf_dist, target_crs)

# ===============================
# PROVINCES: ID + NAME
# ===============================
print("\n--- PROVINCE IDS AND NAMES ---")

prov_id_col = "ID_1"
prov_name_col = "NAME_1"

print(gdf_prov[[prov_id_col, prov_name_col]].head())

# ===============================
# DISTRICTS: ID + NAME
# ===============================
print("\n--- DISTRICT IDS AND NAMES ---")

dist_id_col = "ID_2"
dist_name_col = "NAME_2"

print(gdf_dist[[dist_id_col, dist_name_col]].head())

# ===============================
# BASIC VALIDATION (DON’T SKIP)
# ===============================
print("\n--- VALIDATION ---")

print("Unique Province IDs:", gdf_prov[prov_id_col].nunique())
print("Total Provinces:", len(gdf_prov))

print("Unique District IDs:", gdf_dist[dist_id_col].nunique())
print("Total Districts:", len(gdf_dist))