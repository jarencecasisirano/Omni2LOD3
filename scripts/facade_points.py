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

# ----------  generate facade grid INSIDE polygon ----------
xmin, ymin, xmax, ymax = poly.bounds
x_grid = np.arange(xmin, xmax + spacing, spacing)
y_grid = np.arange(ymin, ymax + spacing, spacing)
xx, yy = np.meshgrid(x_grid, y_grid)
grid_xy = np.column_stack((xx.ravel(), yy.ravel()))

# keep only points inside polygon
inside_mask = contains_xy(poly, grid_xy[:, 0], grid_xy[:, 1])
grid_xy = grid_xy[inside_mask]

# vertical sampling every 50 cm from ground to roof top
z_vals = np.arange(min_z, max_z + spacing, spacing)
n_facade = len(grid_xy) * len(z_vals)
facade_xyz = np.empty((n_facade, 3))
facade_rgb = np.full((n_facade, 3), 65535, dtype=np.uint16)  # white-ish
facade_cls = np.full(n_facade, 6, dtype=np.uint8)            # building class

idx = 0
for x, y in grid_xy:
    for z in z_vals:
        facade_xyz[idx] = [x, y, z]
        idx += 1

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