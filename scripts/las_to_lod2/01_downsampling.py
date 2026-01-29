#!/usr/bin/env python3
import sys
import os
import time
import numpy as np
import laspy
import open3d as o3d
from sklearn.neighbors import KDTree

# ============================================================
# Path setup for utils
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from utils.loading import create_bar

start_time = time.time()

# ------------------------------------------------------------------
# 1. Arguments (supplied by main.py)
# ------------------------------------------------------------------
if len(sys.argv) < 4:
    print("Usage: python 01_downsampling.py <input_las> <output_las> <voxel_size>")
    sys.exit(1)

INPUT_LAS  = sys.argv[1]
OUTPUT_LAS = sys.argv[2]
VOXEL_SIZE = float(sys.argv[3])

# ------------------------------------------------------------------
# 2. Ensure output folder exists
# ------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)

# ------------------------------------------------------------------
# 3. Load LAS (preserve header + CRS)
# ------------------------------------------------------------------
print(f"-> Loading LAS: {INPUT_LAS}")
las = laspy.read(INPUT_LAS)

print(f"Input LAS Version: {las.header.version}")
print(f"Input LAS Point Format: {las.header.point_format.id}")

try:
    print(f"Input CRS: {las.header.parse_crs()}")
except Exception:
    print("Input CRS: None / could not parse")

points = np.vstack((las.x, las.y, las.z)).T
classes = las.classification

# ------------------------------------------------------------------
# 4. Voxel downsampling (Open3D)
# ------------------------------------------------------------------
print(f"-> Performing voxel downsampling (voxel = {VOXEL_SIZE}) ...")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

pcd_down = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
down_points = np.asarray(pcd_down.points)

print(f"-> Reduced from {len(points):,} to {len(down_points):,} points.")

# ------------------------------------------------------------------
# 5. Interpolate classifications (nearest neighbour)
# ------------------------------------------------------------------
print("-> Interpolating classifications (nearest neighbor)...")

tree = KDTree(points, leaf_size=40)
indices = np.zeros(len(down_points), dtype=int)

bar = create_bar("Processing points", len(down_points))

for i, pt in enumerate(down_points):
    _, idx = tree.query([pt], k=1)
    indices[i] = idx[0][0]
    bar.next()

bar.finish()

down_classes = classes[indices]

# ------------------------------------------------------------------
# 6. Save downsampled LAS (CORRECT header handling for laspy 2.x)
# ------------------------------------------------------------------
print("-> Saving downsampled LAS...")

# Create new header matching input
header = laspy.LasHeader(
    point_format=las.header.point_format,
    version=las.header.version
)

# Copy scales + offsets
header.scales = las.header.scales
header.offsets = las.header.offsets

# Copy CRS safely
try:
    header.parse_crs(las.header.parse_crs())
except Exception:
    pass

# Set new bounds
header.mins = down_points.min(axis=0)
header.maxs = down_points.max(axis=0)

# Create output LAS
out_las = laspy.LasData(header)
out_las.x = down_points[:, 0]
out_las.y = down_points[:, 1]
out_las.z = down_points[:, 2]
out_las.classification = down_classes

out_las.write(OUTPUT_LAS)

print(f"-> Downsampled LAS saved to: {OUTPUT_LAS}")

end_time = time.time()
print(f"=== Done! Downsampling finished in {end_time - start_time:.2f} seconds ===")
