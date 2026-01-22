import pdal
import json
import laspy
import open3d as o3d
import numpy as np
import os
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
import sys
from progress.bar import ChargingBar
import time
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

start_time = time.time()
print("=== Starting classification pipeline (ground, noise, unclassified) ===")

if len(sys.argv) < 3:
    print("Usage: python classify_points.py <input_las> <output_las>")
    sys.exit(1)

INPUT_LAS = sys.argv[1]
OUTPUT_LAS = sys.argv[2]
VIS_OUTPUT_PNG = OUTPUT_LAS.replace(".las", "_vis.png")

# -----------------------------
# STEP 1: PDAL for ground and noise classification (with laspy Z normalization)
# -----------------------------
print("STEP 1: Running PDAL for ground and noise classification (with Z normalization)...")

# Create a normalized temporary LAS (only if needed)
base_out_dir = os.path.dirname(OUTPUT_LAS)
temp_norm_path = os.path.join(base_out_dir, os.path.basename(OUTPUT_LAS).replace(".las", "_znorm_temp.las"))
use_temp = False

try:
    print(f"-> Reading input LAS to check Z range: {INPUT_LAS}")
    las_in = laspy.read(INPUT_LAS)
    z_min_before = float(np.min(las_in.z))
    z_max_before = float(np.max(las_in.z))
    print(f"   Z before normalization: min={z_min_before:.2f}, max={z_max_before:.2f}")

    # Normalize in memory (shift so lowest Z == 0). Do it always to keep behavior consistent.
    if True:
        print("-> Normalizing Z (shifting so minimum Z -> 0) and writing temporary LAS for PDAL...")
        las_in.z = las_in.z - z_min_before

        # write the normalized temporary LAS
        las_in.write(temp_norm_path)
        use_temp = True

        # report new range
        las_check = laspy.read(temp_norm_path)
        z_min_after = float(np.min(las_check.z))
        z_max_after = float(np.max(las_check.z))
        print(f"   Z after normalization: min={z_min_after:.2f}, max={z_max_after:.2f}")
    else:
        # if you ever want to skip normalization, set use_temp=False and use INPUT_LAS directly
        use_temp = False

except Exception as e:
    print("[ERROR] Failed to normalize with laspy:", e)
    # fallback: try to run PDAL on original file
    use_temp = False

# Choose which input to pass to PDAL
pdal_input = temp_norm_path if use_temp else INPUT_LAS

# Build PDAL pipeline (run on pdal_input)
pipeline_json = {
    "pipeline": [
        pdal_input,
        {
            "type": "filters.assign",
            "value": "Classification = 0"
        },
        {
            "type": "filters.elm",
            "cell": 10,
            "class": 7,
            "threshold": 3
        },
        {
            "type": "filters.outlier",
            "method": "statistical",
            "mean_k": 8,
            "multiplier": 3,
            "class": 7
        },
        {
            "type": "filters.csf",
            "resolution": 0.5,
            "threshold": 0.5,
            "hdiff": 0.3,
            "rigidness": 3,
            "smooth": True,
            "iterations": 500,
            "ignore": "Classification[7:7]"
        },
        {
            "type": "writers.las",
            "filename": OUTPUT_LAS,
            "major_version": 1,
            "minor_version": 2,
            "dataformat_id": 2  # Force point format 2
        }
    ]
}

try:
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    pipeline.execute()
    print("-> Ground and noise classification complete and written to LAS.")
except Exception as e:
    print("[ERROR] PDAL pipeline failed:", e)
    # Clean up temp file if it exists, then re-raise so caller sees failure
    if use_temp and os.path.exists(temp_norm_path):
        try:
            os.remove(temp_norm_path)
        except Exception:
            pass
    raise

# remove temporary normalized file (we no longer need it)
if use_temp and os.path.exists(temp_norm_path):
    try:
        os.remove(temp_norm_path)
        print(f"-> Removed temporary normalized LAS: {temp_norm_path}")
    except Exception as e:
        print(f"[WARNING] Could not delete temp file {temp_norm_path}: {e}")


# STEP 2: Load LAS file
print("STEP 2: Loading LAS file...")
las = laspy.read(OUTPUT_LAS)
points = np.vstack((las.x, las.y, las.z)).T
classes = las.classification
print(f"-> Loaded {len(points)} points")

# STEP 3: Detect noise using DBSCAN
print("STEP 3: Detecting noise with DBSCAN...")
unclassified_mask = classes == 0
unclassified_points = points[unclassified_mask]
unclassified_indices = np.where(unclassified_mask)[0]

if len(unclassified_points) == 0:
    print("[WARNING] No unclassified points found for noise detection.")
else:
    print("STEP 3.1: Applying statistical outlier removal...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(unclassified_points)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    unclassified_points = np.asarray(pcd.points)
    unclassified_indices = unclassified_indices[ind]
    print(f"-> Retained {len(unclassified_points)} unclassified points after outlier removal")

    points_2d = unclassified_points[:, :2]
    db = DBSCAN(eps=2.0, min_samples=50).fit(points_2d)
    labels = db.labels_
    noise_mask = labels == -1
    noise_indices = unclassified_indices[noise_mask]
    classes[noise_indices] = 7
    print(f"-> Detected and classified {np.sum(noise_mask)} points as noise.")

# STEP 4: Save output LAS
print("STEP 4: Saving output LAS...")
new_header = laspy.LasHeader(point_format=2, version="1.2")  # Force LAS 1.2, point format 2
new_header.scales = las.header.scales
new_header.offsets = las.header.offsets
new_header.mins = [np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])]
new_header.maxs = [np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])]
if las.header.parse_crs() is not None:
    new_header.add_crs(las.header.parse_crs())
else:
    new_header.vlrs = []

new_las = laspy.LasData(new_header)
new_las.x = points[:, 0]
new_las.y = points[:, 1]
new_las.z = points[:, 2]
new_las.classification = classes
new_las.write(OUTPUT_LAS)
print(f"-> Output saved to: {OUTPUT_LAS}")

# Classification summary
print("-> Classification counts:")
total_points = len(classes)
unique_classes, class_counts = np.unique(classes, return_counts=True)
for uc, count in zip(unique_classes, class_counts):
    percentage = (count / total_points) * 100
    print(f"   Class {uc}: {count} points ({percentage:.2f}%)")

# STEP 5: Visualize with Open3D
print("STEP 5: Visualizing...")
def visualize_by_class(path):
    las = laspy.read(path)
    points = np.vstack((las.x, las.y, las.z)).T
    classes = las.classification
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    color_map = {
        0: [0.5, 0.5, 0.5],  # Unclassified - gray
        2: [0.59, 0.29, 0.0],  # Ground - brown
        7: [1.0, 1.0, 0.0],    # Noise - yellow
    }
    colors = np.array([color_map.get(c, [0.7, 0.7, 0.7]) for c in classes])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(VIS_OUTPUT_PNG)
    vis.run()
    vis.destroy_window()
    print(f"-> Saved visualization screenshot to: {VIS_OUTPUT_PNG}")

visualize_by_class(OUTPUT_LAS)

# STEP 6: Remove noise points and save final output
print("STEP 6: Removing noise points...")
non_noise_mask = classes != 7
points = points[non_noise_mask]
classes = classes[non_noise_mask]

new_header = laspy.LasHeader(point_format=2, version="1.2")  # Force LAS 1.2, point format 2
new_header.scales = las.header.scales
new_header.offsets = las.header.offsets
new_header.mins = [np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])]
new_header.maxs = [np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])]
if las.header.parse_crs() is not None:
    new_header.add_crs(las.header.parse_crs())
else:
    new_header.vlrs = []

final_las = laspy.LasData(new_header)
final_las.x = points[:, 0]
final_las.y = points[:, 1]
final_las.z = points[:, 2]
final_las.classification = classes
final_las.write(OUTPUT_LAS.replace("_vis", "_final"))
print(f"-> Final output without noise saved to: {OUTPUT_LAS.replace('_vis', '_final')}")

end_time = time.time()
elapsed = end_time - start_time
print(f"=== Done! Point classification finished in {elapsed:.2f} seconds ===")