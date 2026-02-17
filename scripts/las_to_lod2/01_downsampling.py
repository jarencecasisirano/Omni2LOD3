# 01_downsampling.py
import sys
import time
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.loading import create_bar


def _parse_args(argv):
    if len(argv) < 4:
        print("Usage: python 01_downsampling.py <input_las> <output_las> <voxel_size>")
        sys.exit(1)

    input_las = Path(argv[1])
    output_las = Path(argv[2])
    voxel_size = float(argv[3])
    return input_las, output_las, voxel_size


def main():
    start_time = time.time()

    # 1. Arguments (supplied by main.py) ----------------------------
    input_las, output_las, voxel_size = _parse_args(sys.argv)

    # 2. Ensure output folder exists --------------------------------
    output_las.parent.mkdir(parents=True, exist_ok=True)

    # 3. Load LAS ----------------------------------------------------
    las = laspy.read(str(input_las))
    points = np.vstack((las.x, las.y, las.z)).T
    classes = las.classification

    # 3. Voxel downsampling --------------------------------------------
    print(f"\t-> Performing voxel downsampling (voxel size = {voxel_size})...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    down_points = np.asarray(pcd.points)
    print(f"\t-> Reduced from {len(points):,} to {len(down_points):,} points.")

    # 5. Interpolate classifications (nearest neighbour) -------------
    print("\t-> Interpolating classifications from original point cloud...")
    tree = KDTree(points)
    indices = np.zeros(len(down_points), dtype=int)
    bar = create_bar("\t\tProcessing points", len(down_points))
    for i in range(len(down_points)):
        _, idx = tree.query([down_points[i]], k=1)
        indices[i] = idx[0][0]
        bar.next()
    bar.finish()
    down_classes = classes[indices]

    # 6. Save down-sampled LAS (copy VLRs as-is, no CRS parsing) ----
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
    out_las.write(str(output_las))
    print(f"\t-> Downsampled LAS saved in folder: {output_las.parent}")

    end_time = time.time()
    print(f"=== Done! Downsampling finished in {end_time - start_time:.2f} seconds ===")


if __name__ == "__main__":
    main()
