#!/usr/bin/env python3
import sys, os, laspy, numpy as np, open3d as o3d
from sklearn.neighbors import KDTree
from progress.bar import ChargingBar
import time

start_time = time.time()

# ------------------------------------------------------------------
# 1.  Hardcoded paths
# ------------------------------------------------------------------
INPUT_LAS   = r"C:\Projects\Omni2LOD3\data\01_point_cloud\NIMBB 112025.las"
OUTPUT_LAS  = r"C:\Projects\Omni2LOD3\outputs\00_archive\01_test_downsampled.las"
VOXEL_SIZE  = 0.5

# ------------------------------------------------------------------
# 2.  Ensure output folder exists
# ------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)

# ------------------------------------------------------------------
# 3.  Load LAS
# ------------------------------------------------------------------
las = laspy.read(INPUT_LAS)
print(f"Input LAS Version: {las.header.version}")
print(f"Input LAS Point Format: {las.header.point_format.id}")
points  = np.vstack((las.x, las.y, las.z)).T
classes = las.classification

# ------------------------------------------------------------------
# 4.  Voxel downsampling
# ------------------------------------------------------------------
print("-> Performing voxel downsampling...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
down_points = np.asarray(pcd.points)
print(f"-> Reduced from {len(points)} to {len(down_points)} points.")

# ------------------------------------------------------------------
# 5.  Interpolate classifications (nearest neighbour)
# ------------------------------------------------------------------
print("-> Interpolating classifications from original point cloud...")
tree = KDTree(points)
indices = np.zeros(len(down_points), dtype=int)
bar = ChargingBar("Processing points", max=len(down_points), suffix="%(percent)d%%")
for i in range(len(down_points)):
    _, idx = tree.query([down_points[i]], k=1)
    indices[i] = idx[0][0]
    bar.next()
bar.finish()
down_classes = classes[indices]

# ------------------------------------------------------------------
# 6.  Save down-sampled LAS  (copy VLRs as-is, no CRS parsing)
# ------------------------------------------------------------------
print("-> Saving downsampled LAS...")
header = laspy.LasHeader(point_format=2, version="1.2")
header.scales   = las.header.scales
header.offsets  = las.header.offsets
header.mins     = down_points.min(axis=0)
header.maxs     = down_points.max(axis=0)
header.vlrs     = las.header.vlrs  # Copy all VLRs (including CRS)

out_las = laspy.LasData(header)
out_las.x = down_points[:, 0]
out_las.y = down_points[:, 1]
out_las.z = down_points[:, 2]
out_las.classification = down_classes
out_las.write(OUTPUT_LAS)
print(f"-> Downsampled LAS saved to: {OUTPUT_LAS}")

end_time = time.time()
print(f"=== Done! Downsampling finished in {end_time - start_time:.2f} seconds ===")