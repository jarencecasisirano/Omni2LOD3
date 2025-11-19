#!/usr/bin/env python3
import os, sys, glob, laspy, numpy as np, geopandas as gpd
from shapely.geometry import Polygon
from shapely import contains_xy

# ----------  folders ----------
ROOT        = os.path.join(os.path.dirname(__file__), "..")
BLDG_DIR    = os.path.join(ROOT, "outputs", "building_classification")
FOOT_DIR    = os.path.join(ROOT, "outputs", "footprint", "shp")
OUT_DIR     = os.path.join(ROOT, "outputs", "complete_las")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------  picker helpers ----------
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

# ----------  pick inputs ----------
las_path   = pick_file(BLDG_DIR, "las", "Select building-classified LAS")
foot_path  = pick_file(FOOT_DIR, "shp", "Select matching footprint shapefile")

# ----------  load data ----------
las  = laspy.read(las_path)
foot = gpd.read_file(foot_path)
if len(foot) != 1:
    print("[ERROR] Shapefile must contain exactly one polygon."); sys.exit(1)
poly = foot.geometry[0]
if not isinstance(poly, Polygon):
    print("[ERROR] Geometry is not a single Polygon."); sys.exit(1)

# ----------  roof points + meta ----------
roof_mask = las.classification == 6
xyz_r   = np.vstack((las.x[roof_mask], las.y[roof_mask], las.z[roof_mask])).T
rgb_r   = np.vstack((las.red[roof_mask], las.green[roof_mask], las.blue[roof_mask])).T
max_z   = float(xyz_r[:, 2].max())
min_z   = float(xyz_r[:, 2].min())
spacing = 0.5          # 50 cm grid

# ----------  pure-Numpy height grid (25 cm) ----------
from shapely import contains_xy

res        = 0.25
xmin, ymin, xmax, ymax = poly.bounds
x_edge     = np.arange(xmin, xmax + res, res)
y_edge     = np.arange(ymin, ymax + res, res)
x_centres  = x_edge[:-1] + res / 2          # pixel centres
y_centres  = y_edge[:-1] + res / 2
xx, yy     = np.meshgrid(x_centres, y_centres)
grid_xy    = np.column_stack((xx.ravel(), yy.ravel()))

# burn max-Z per pixel
height_grid = np.full(grid_xy.shape[0], np.nan, dtype=np.float32)
for x, y, z in zip(xyz_r[:, 0], xyz_r[:, 1], xyz_r[:, 2]):
    # find nearest pixel
    c = int((x - xmin) / res); r = int((y - ymin) / res)
    if 0 <= c < len(x_centres) and 0 <= r < len(y_centres):
        idx = r * len(x_centres) + c
        height_grid[idx] = np.nanmax([height_grid[idx], z])

height_grid = np.nan_to_num(height_grid, nan=min_z)   # fill empty pixels

# ----------  variable-height façade ----------
facade_xyz, facade_rgb, facade_cls = [], [], []
for (x, y), z_top in zip(grid_xy, height_grid):
    if not contains_xy(poly, x, y):
        continue
    z_vals = np.arange(min_z, z_top + res, res)
    for z in z_vals:
        facade_xyz.append([x, y, z])
        facade_rgb.append([65000, 65000, 65000])
        facade_cls.append(6)

facade_xyz = np.array(facade_xyz, dtype=np.float64)
facade_rgb = np.array(facade_rgb, dtype=np.uint16)
facade_cls = np.array(facade_cls, dtype=np.uint8)

# ----------  merge roof + facade ----------
all_xyz = np.vstack((xyz_r, facade_xyz))
all_rgb = np.vstack((rgb_r, facade_rgb))
all_cls = np.hstack((las.classification[roof_mask], facade_cls))

# ----------  write LAS ----------
out_name = os.path.splitext(os.path.basename(las_path))[0] + "_with_facade.las"
out_path = os.path.join(OUT_DIR, out_name)

header = laspy.LasHeader(point_format=las.header.point_format.id,
                         version=str(las.header.version))   # <-- here
header.scales  = las.header.scales
header.offsets = las.header.offsets
header.vlrs    = las.header.vlrs

new_las = laspy.LasData(header)
new_las.x = all_xyz[:, 0]
new_las.y = all_xyz[:, 1]
new_las.z = all_xyz[:, 2]
new_las.red, new_las.green, new_las.blue = all_rgb.T
new_las.classification = all_cls
new_las.write(out_path)

print(f"[SUCCESS] Roof + façade points saved to: {out_path}")
print(f"          Roof pts: {len(xyz_r)}  Façade pts: {len(facade_xyz)}  Total: {len(all_xyz)}")

# ----------  quick 3-D check ----------
import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_xyz)
# colour: roof = red, façade = white
colors = np.empty((len(all_xyz), 3))
colors[:len(xyz_r)]      = [1, 0, 0]   # roof
colors[len(xyz_r):]      = [0.9, 0.9, 0.9]  # façade
pcd.colors = o3d.utility.Vector3dVector(colors)

print("Visualising roof (red) + façade (white) – close window when done")
o3d.visualization.draw_geometries([pcd],
                                  window_name="Roof + Façade points",
                                  width=900, height=700)