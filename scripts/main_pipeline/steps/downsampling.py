import time
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree

from utils.loading import create_bar


def run(input_las, output_las, voxel_size, progress_callback=None):
    """
    Pipeline-safe downsampling module.
    """

    start_time = time.time()

    input_las = Path(input_las)
    output_las = Path(output_las)

    output_las.parent.mkdir(parents=True, exist_ok=True)

    # Load LAS
    las = laspy.read(str(input_las))
    points = np.vstack((las.x, las.y, las.z)).T
    classes = las.classification

    # Downsampling
    print(f"-> Voxel downsampling (size={voxel_size})")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    down_points = np.asarray(pcd.points)

    print(f"-> Reduced: {len(points):,} → {len(down_points):,}")

    # Classification transfer
    print("-> Transferring classifications...")

    tree = KDTree(points)

    indices = np.zeros(len(down_points), dtype=int)

    bar = create_bar("Processing", len(down_points))

    for i in range(len(down_points)):
        _, idx = tree.query([down_points[i]], k=1)
        indices[i] = idx[0][0]
        bar.next()

        # GUI hook (future use)
        if progress_callback:
            progress_callback(i / len(down_points))

    bar.finish()

    down_classes = classes[indices]

    # Save LAS
    header = laspy.LasHeader(point_format=2, version="1.2")
    header.scales = las.header.scales
    header.offsets = las.header.offsets
    header.mins = down_points.min(axis=0)
    header.maxs = down_points.max(axis=0)
    header.vlrs = las.header.vlrs

    out_las = laspy.LasData(header)
    out_las.x, out_las.y, out_las.z = down_points.T
    out_las.classification = down_classes
    out_las.write(str(output_las))

    print(f"Saved: {output_las}")

    return {
        "input": str(input_las),
        "output": str(output_las),
        "original_points": len(points),
        "downsampled_points": len(down_points),
        "time_sec": time.time() - start_time
    }


# CLI wrapper (kept thin)
def _cli():
    import sys

    if len(sys.argv) < 4:
        print("Usage: python 01_downsampling.py <input> <output> <voxel>")
        sys.exit(1)

    run(
        sys.argv[1],
        sys.argv[2],
        float(sys.argv[3])
    )


if __name__ == "__main__":
    _cli()