#!/usr/bin/env python3
import os, sys, glob
import geopandas as gpd

# ----------  folders ----------
ROOT      = os.path.join(os.path.dirname(__file__), "..")
FOOT_DIR  = os.path.join(ROOT, "outputs", "footprint")
OUT_DIR   = os.path.join(FOOT_DIR, "shp")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------  build file list ----------
files = sorted(glob.glob(os.path.join(FOOT_DIR, "*.geojson")))
if not files:
    print("[ERROR] No GeoJSON files found in outputs/footprint/")
    sys.exit(1)

print("Select GeoJSON file to convert to Shapefile:")
for idx, path in enumerate(files):
    print(f"[{idx}] {os.path.basename(path)}")
choice = input("Enter index: ").strip()
if not choice.isdigit() or int(choice) not in range(len(files)):
    print("[ERROR] Invalid selection.")
    sys.exit(1)

geojson_path = files[int(choice)]

# ----------  convert ----------
base_name = os.path.splitext(os.path.basename(geojson_path))[0]
shp_path  = os.path.join(OUT_DIR, base_name + ".shp")

print(f"Converting  : {geojson_path}")
print(f"Shapefile   : {shp_path}")

gdf = gpd.read_file(geojson_path)
gdf.to_file(shp_path, driver='ESRI Shapefile')
print("[SUCCESS] Shapefile written.")