#!/usr/bin/env python3
"""
Facade-points – perimeter-only edition
Creates a thin façade skin (8-point voxels) only on the footprint edge.
Everything else (colouring, LAS merge, visualisation) is identical to the
original script.
"""
import os, sys, glob, laspy, numpy as np, geopandas as gpd
from shapely.geometry import Polygon
from shapely import contains_xy
from scipy.ndimage import binary_erosion          # NEW: for edge detection

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
res     = 0.25          # 25 cm grid cell

# ----------  pure-Numpy height grid ----------
xmin, ymin, xmax, ymax = poly.bounds
x_edge     = np.arange(xmin, xmax + res, res)
y_edge     = np.arange(ymin, ymax + res, res)
x_centres  = x_edge[:-1] + res / 2
y_centres  = y_edge[:-1] + res / 2
xx, yy     = np.meshgrid(x_centres, y_centres)
grid_xy    = np.column_stack((xx.ravel(), yy.ravel()))

# burn max-Z per pixel
height_grid = np.full(grid_xy.shape[0], np.nan, dtype=np.float32)
for x, y, z in zip(xyz_r[:, 0], xyz_r[:, 1], xyz_r[:, 2]):
    c = int((x - xmin) / res); r = int((y - ymin) / res)
    if 0 <= c < len(x_centres) and 0 <= r < len(y_centres):
        idx = r * len(x_centres) + c
        height_grid[idx] = np.nanmax([height_grid[idx], z])
height_grid = np.nan_to_num(height_grid, nan=min_z)

# ----------  footprint mask (inside vs outside) ----------
inside = np.array([contains_xy(poly, x, y) for x, y in grid_xy], dtype=bool)

# reshape to 2-D for neighbour logic
h_grid  = height_grid.reshape(len(y_centres), len(x_centres))
in_grid = inside.reshape(len(y_centres), len(x_centres))

# ----------  EDGE MASK (inside but neighbour outside) ----------
kernel = np.array([[0,1,0],
                   [1,0,1],
                   [0,1,0]], dtype=bool)   # 4-neighbour
edge_mask = in_grid & ~binary_erosion(in_grid, structure=kernel, border_value=1)

# ----------  façade voxels – ONLY ON EDGES ----------
facade_xyz, facade_rgb, facade_cls = [], [], []

# iterate only over edge cells that also have roof data
rows, cols = np.where(edge_mask & (~np.isnan(h_grid)))
for r, c in zip(rows, cols):
    x = x_centres[c]
    y = y_centres[r]
    z_top = h_grid[r, c]
    z_vals = np.arange(min_z, z_top + res, res)
    for z in z_vals:
        cube = np.array([
            [x - res/2, y - res/2, z - res/2],
            [x + res/2, y - res/2, z - res/2],
            [x + res/2, y + res/2, z - res/2],
            [x - res/2, y + res/2, z - res/2],
            [x - res/2, y - res/2, z + res/2],
            [x + res/2, y - res/2, z + res/2],
            [x + res/2, y + res/2, z + res/2],
            [x - res/2, y + res/2, z + res/2]
        ])
        facade_xyz.append(cube)
        facade_rgb.extend([[65000, 65000, 65000]] * 8)
        facade_cls.extend([6] * 8)

facade_xyz = np.vstack(facade_xyz) if facade_xyz else np.empty((0, 3))
facade_rgb = np.array(facade_rgb, dtype=np.uint16)
facade_cls = np.array(facade_cls, dtype=np.uint8)

# ----------  merge ORIGINAL + façade ----------
all_xyz = np.vstack((las.x, las.y, las.z)).T
all_rgb = np.vstack((las.red, las.green, las.blue)).T
all_cls = las.classification.copy()

all_xyz = np.vstack((all_xyz, facade_xyz))
all_rgb = np.vstack((all_rgb, facade_rgb))
all_cls = np.hstack((all_cls, facade_cls))

# ----------  paint classification colours into RGB ----------
new_rgb = np.empty((len(all_cls), 3), dtype=np.uint16)
new_rgb[:] = [60000, 60000, 60000]        # default white
new_rgb[all_cls == 1] = [30000, 30000, 30000]   # unclassified grey
new_rgb[all_cls == 2] = [40000, 20000, 0]       # ground brown
new_rgb[all_cls == 6] = [50000, 0, 0]           # building red
new_rgb[len(las.points):] = [0, 0, 50000]       # façade blue
all_rgb = new_rgb

# ----------  write LAS ----------
out_name = os.path.splitext(os.path.basename(las_path))[0] + "_with_facade.las"
out_path = os.path.join(OUT_DIR, out_name)

header = laspy.LasHeader(point_format=las.header.point_format.id,
                         version=str(las.header.version))
header.scales  = las.header.scales
header.offsets = las.header.offsets
header.vlrs    = list(las.header.vlrs)

new_las = laspy.LasData(header)
new_las.x = all_xyz[:, 0]
new_las.y = all_xyz[:, 1]
new_las.z = all_xyz[:, 2]
new_las.red, new_las.green, new_las.blue = all_rgb.T
new_las.classification = all_cls
new_las.write(out_path)

print(f"[SUCCESS] Full point cloud (ground kept) + façade saved to: {out_path}")
print(f"          Original pts: {len(las.points)}  Façade pts: {len(facade_xyz)}  Total: {len(all_xyz)}")

# ----------  quick 3-D check ----------
import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_xyz)

colors = np.empty((len(all_xyz), 3))
# ----- original points -----
orig_cls = all_cls[:len(las.points)]
colors_orig = np.empty((len(orig_cls), 3))
colors_orig[:] = [0.9, 0.9, 0.9]
colors_orig[orig_cls == 1] = [0.5, 0.5, 0.5]
colors_orig[orig_cls == 2] = [0.6, 0.3, 0.0]
colors_orig[orig_cls == 6] = [1.0, 0.0, 0.0]
colors[:len(las.points)] = colors_orig
# ----- façade points -----
colors[len(las.points):] = [0.0, 0.0, 1.0]
pcd.colors = o3d.utility.Vector3dVector(colors)

print("Visualising – close window when done")
o3d.visualization.draw_geometries([pcd],
                                  window_name="Roof(red) Façade(blue) Ground(brown)",
                                  width=900, height=700)