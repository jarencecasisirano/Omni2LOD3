#!/usr/bin/env python3
"""
Recreate Pipeline B's EXACT process using Pipeline A's input file
This will help identify if there's something about the processing order
"""
import sys
import os
import laspy
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree

print("="*60)
print("EXACT PIPELINE B REPLICATION TEST")
print("="*60)

# Use Pipeline A's raw input file
INPUT_LAS = r"C:\Projects\Omni2LOD3\data\01_point_cloud\NIMBB 112025.las"
OUTPUT_DOWNSAMPLE = r"C:\Projects\Omni2LOD3\outputs\00_archive\02_test_B_process_downsampled.las"
OUTPUT_CLIPPED = r"C:\Projects\Omni2LOD3\outputs\00_archive\02_test_B_process_clipped.las"
VOXEL_SIZE = 0.5

os.makedirs(os.path.dirname(OUTPUT_DOWNSAMPLE), exist_ok=True)

# ================================================================
# STEP 1: DOWNSAMPLE (exact copy of Pipeline B's code)
# ================================================================
print("\n[STEP 1: DOWNSAMPLING]")
print(f"Input: {INPUT_LAS}")

las = laspy.read(INPUT_LAS)
print(f"Input LAS Version: {las.header.version}")
print(f"Input LAS Point Format: {las.header.point_format.id}")

points = np.vstack((las.x, las.y, las.z)).T
classes = las.classification

print(f"-> Performing voxel downsampling...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
down_points = np.asarray(pcd.points)
print(f"-> Reduced from {len(points)} to {len(down_points)} points.")

print("-> Interpolating classifications from original point cloud...")
tree = KDTree(points)
indices = np.zeros(len(down_points), dtype=int)
for i in range(len(down_points)):
    _, idx = tree.query([down_points[i]], k=1)
    indices[i] = idx[0][0]
    if i % 10000 == 0:
        print(f"  Progress: {i}/{len(down_points)}")

down_classes = classes[indices]

print("-> Saving downsampled LAS...")
header = laspy.LasHeader(point_format=2, version="1.2")
header.scales = las.header.scales
header.offsets = las.header.offsets
header.mins = down_points.min(axis=0)
header.maxs = down_points.max(axis=0)
header.vlrs = las.header.vlrs  # Copy all VLRs (including CRS)

out_las = laspy.LasData(header)
out_las.x = down_points[:, 0]
out_las.y = down_points[:, 1]
out_las.z = down_points[:, 2]
out_las.classification = down_classes
out_las.write(OUTPUT_DOWNSAMPLE)
print(f"-> Downsampled LAS saved to: {OUTPUT_DOWNSAMPLE}")

# ================================================================
# STEP 2: CLIP Z (exact copy of Pipeline B's code)
# ================================================================
print("\n[STEP 2: CLIPPING Z OUTLIERS]")
print(f"Reading: {OUTPUT_DOWNSAMPLE}")

las = laspy.read(OUTPUT_DOWNSAMPLE)
z = las.z
mask = (z >= -50) & (z <= 200)

print(f"Original points: {len(z)}")
print(f"Kept points: {np.sum(mask)}")
print(f"Removed points: {len(z) - np.sum(mask)}")

clipped_las = laspy.create(
    point_format=las.header.point_format,
    file_version=las.header.version
)
clipped_las.header = las.header

for dim in las.point_format.dimension_names:
    setattr(
        clipped_las,
        dim,
        getattr(las, dim)[mask]
    )

clipped_las.write(OUTPUT_CLIPPED)
print(f"Saved: {OUTPUT_CLIPPED}")

# ================================================================
# VERIFICATION
# ================================================================
print("\n" + "="*60)
print("VERIFICATION")
print("="*60)

print("\nNow try this file in CityForge:")
print(f"  {OUTPUT_CLIPPED}")
print("\nThis file was created using EXACT Pipeline B code.")
print("If this works: Pipeline A's code has a subtle bug")
print("If this fails: The issue is with the input file or environment")

# Quick inspection
verify = laspy.read(OUTPUT_CLIPPED)
print(f"\nFile stats:")
print(f"  Points: {len(verify.points):,}")
print(f"  Format: {verify.header.point_format.id}")
print(f"  Version: {verify.header.version}")
print(f"  VLRs: {len(verify.header.vlrs)}")
unique_classes = np.unique(verify.classification)
print(f"  Classes: {dict(zip(*np.unique(verify.classification, return_counts=True)))}")