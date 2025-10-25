import sys
import laspy
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import time
from progress.bar import ChargingBar

start_time = time.time()

if len(sys.argv) < 4:
    print("Usage: python voxel_downsampling.py <input_las> <output_las> <voxel_size>")
    sys.exit(1)

INPUT_LAS = sys.argv[1]
OUTPUT_LAS = sys.argv[2]
VOXEL_SIZE = float(sys.argv[3])

# Load LAS
las = laspy.read(INPUT_LAS)
print(f"Input LAS Version: {las.header.version}")
print(f"Input LAS Point Format: {las.header.point_format.id}")
points = np.vstack((las.x, las.y, las.z)).T
classes = las.classification

# Voxel downsampling
print("-> Performing voxel downsampling...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
down_points = np.asarray(pcd.points)
print(f"-> Reduced from {len(points)} to {len(down_points)} points.")

# Interpolate classifications
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

# Save output
print("-> Saving downsampled LAS...")
header = laspy.LasHeader(point_format=2, version="1.2")  # Force LAS 1.2, point format 2
header.scales = las.header.scales
header.offsets = las.header.offsets
header.mins = [np.min(down_points[:, 0]), np.min(down_points[:, 1]), np.min(down_points[:, 2])]
header.maxs = [np.max(down_points[:, 0]), np.max(down_points[:, 1]), np.max(down_points[:, 2])]
if las.header.parse_crs() is not None:
    header.add_crs(las.header.parse_crs())
else:
    header.vlrs = []

out_las = laspy.LasData(header)
out_las.x = down_points[:, 0]
out_las.y = down_points[:, 1]
out_las.z = down_points[:, 2]
out_las.classification = down_classes
out_las.write(OUTPUT_LAS)
print(f"-> Downsampled LAS saved to: {OUTPUT_LAS}")

end_time = time.time()
print(f"=== Done! Downsampling finished in {end_time - start_time:.2f} seconds ===")