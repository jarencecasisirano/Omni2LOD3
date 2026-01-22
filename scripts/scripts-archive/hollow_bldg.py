#!/usr/bin/env python3
import os, sys, glob, laspy, numpy as np, open3d as o3d
from shapely.geometry import Polygon, Point
import geopandas as gpd

ROOT      = os.path.join(os.path.dirname(__file__), "..")
IN_DIR    = os.path.join(ROOT, "outputs", "complete_las")
FOOT_DIR  = os.path.join(ROOT, "outputs", "footprint", "shp")
OUT_DIR   = os.path.join(ROOT, "outputs", "complete_las")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------  picker ----------
def pick_file(folder, ext, prompt):
    files = sorted(glob.glob(os.path.join(folder, f"*.{ext}")))
    if not files:
        print(f"[ERROR] No *.{ext} files in {folder}"); sys.exit(1)
    print(f"\n{prompt}:")
    for idx, path in enumerate(files):
        print(f"[{idx}] {os.path.basename(path)}")
    choice = input("Enter index: ").strip()
    if not choice.isdigit() or int(choice) not in range(len(files)):
        print("[ERROR] Invalid selection."); sys.exit(1)
    return files[int(choice)]

las_path  = pick_file(IN_DIR,  "las", "Select complete LAS (with façade)")
foot_path = pick_file(FOOT_DIR, "shp", "Select matching footprint")

# ----------  load ----------
las  = laspy.read(las_path)
foot = gpd.read_file(foot_path)
poly = foot.geometry[0]
if not isinstance(poly, Polygon):
    print("[ERROR] Shapefile must contain exactly one polygon."); sys.exit(1)

mask      = las.classification == 6
xyz       = np.vstack((las.x[mask], las.y[mask], las.z[mask])).T
rgb       = np.vstack((las.red[mask], las.green[mask], las.blue[mask])).T

# ----------  keep only points within 25 cm of footprint EDGE ----------
buf_poly = poly.buffer(0.25)          # 25 cm buffer around footprint
edge_mask = np.array([buf_poly.contains(Point(x, y)) for x, y in xyz[:, :2]])

# ----------  keep façade (vertical) points ----------
# any point whose Z is **not** within 25 cm of the roof-top → façade
roof_top = xyz[:, 2].max()
facade_mask = xyz[:, 2] < (roof_top - 0.25)

# ----------  final mask ----------
keep_mask = edge_mask | facade_mask   # edge OR vertical
xyz_out   = xyz[keep_mask]
rgb_out   = rgb[keep_mask]

# ----------  write ----------
base      = os.path.splitext(os.path.basename(las_path))[0]
out_path  = os.path.join(OUT_DIR, base + "_hollowed.las")

header = laspy.LasHeader(point_format=las.header.point_format.id,
                         version=str(las.header.version))
header.scales  = las.header.scales
header.offsets = las.header.offsets
header.vlrs    = las.header.vlrs

new_las = laspy.LasData(header)
new_las.x = xyz_out[:, 0]
new_las.y = xyz_out[:, 1]
new_las.z = xyz_out[:, 2]
new_las.red, new_las.green, new_las.blue = rgb_out.T
new_las.classification = np.full(len(xyz_out), 6, dtype=np.uint8)
new_las.write(out_path)

print(f"[SUCCESS] Hollowed façade + roof shell saved to: {out_path}")
print(f"          Original pts: {len(xyz)}  Shell pts: {len(xyz_out)}")

# quick 3-D look
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_out)
pcd.colors = o3d.utility.Vector3dVector(np.tile([1, 1, 0], (len(xyz_out), 1)))
print("Visualising hollow shell – close window when done")
o3d.visualization.draw_geometries([pcd], window_name="Hollowed façade + roof")