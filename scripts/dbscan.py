import laspy
import numpy as np
import open3d as o3d
import sys
import os
import time
from sklearn.cluster import DBSCAN
from progress.bar import ChargingBar
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

start_time = time.time()
print("=== Starting DBSCAN clustering for unclassified points ===")

# === Input/Output Setup ===
if len(sys.argv) < 3:
    print("Usage: python dbscan.py <input_las> <output_las>")
    sys.exit(1)

INPUT_LAS = sys.argv[1]
OUTPUT_LAS = sys.argv[2]
VIS_OUTPUT_PNG = os.path.splitext(OUTPUT_LAS)[0] + "_vis.png"

if not os.path.exists(INPUT_LAS):
    print(f"[ERROR] Input file not found: {INPUT_LAS}")
    sys.exit(1)

OUTPUT_DIR = os.path.dirname(OUTPUT_LAS)
if OUTPUT_DIR and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Step 1: Load and filter unclassified points ===
print("STEP 1: Reading LAS file and filtering unclassified points...")
las = laspy.read(INPUT_LAS)
points = np.vstack((las.x, las.y, las.z)).T
classification = las.classification

# Try Class 0 first, fall back to Class 1 if no Class 0
unclassified_mask = classification == 0
if np.sum(unclassified_mask) == 0:
    print("[WARNING] No Class 0 (unclassified) points found. Trying Class 1...")
    unclassified_mask = classification == 1
    if np.sum(unclassified_mask) == 0:
        print("[ERROR] No unclassified (Class 0) or potential unclassified (Class 1) points found.")
        sys.exit(1)

unclassified_points = points[unclassified_mask]
unclassified_indices = np.where(unclassified_mask)[0]

print(f"-> Loaded {len(unclassified_points)} unclassified points out of {len(points)} total points")

# === Step 1.1: Statistical Outlier Removal ===
print("STEP 1.1: Applying statistical outlier removal...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(unclassified_points)
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
unclassified_points = np.asarray(pcd.points)
unclassified_indices = unclassified_indices[ind]

if len(unclassified_points) == 0:
    print("[ERROR] All unclassified points removed during outlier filtering.")
    sys.exit(1)

print(f"-> Retained {len(unclassified_points)} unclassified points after outlier removal")

# === Step 2: Apply DBSCAN clustering ===
print("STEP 2: Running DBSCAN clustering...")
points_2d = unclassified_points[:, :2]
db = DBSCAN(eps=1.3, min_samples=100).fit(points_2d)  # Increased eps and min_samples
labels = db.labels_

# Filter out small clusters and limit to 22 clusters (to stay within 10–31)
label_counts = np.bincount(labels[labels != -1])
valid_labels = np.where(label_counts >= 200)[0]  # Increased threshold to reduce clusters
valid_mask = np.isin(labels, valid_labels)
labels[~valid_mask] = -1  # Mark small clusters and noise as -1

# Limit to 22 clusters to stay within classification range 10–31
unique_labels = np.unique(labels[labels != -1])
if len(unique_labels) > 22:
    print(f"[WARNING] Found {len(unique_labels)} clusters, limiting to 22 to stay within valid classification range (10–31).")
    # Sort clusters by size and keep top 22
    label_sizes = [(label, label_counts[label]) for label in unique_labels]
    label_sizes.sort(key=lambda x: x[1], reverse=True)  # Sort by count descending
    valid_labels = [label for label, _ in label_sizes[:22]]
    valid_mask = np.isin(labels, valid_labels)
    labels[~valid_mask] = -1  # Mark excess clusters as noise
    unique_labels = valid_labels

print(f"-> Found {len(unique_labels)} potential building clusters after filtering.")

if len(unique_labels) > 0:
    for cluster_id, label in enumerate(unique_labels):
        cluster_mask = labels == label
        cluster_indices = unclassified_indices[cluster_mask]
        classification[cluster_indices] = 10 + cluster_id  # Assign classes 10–31

# Mark noise points (labels == -1) as class 7
noise_mask = labels == -1
noise_indices = unclassified_indices[noise_mask]
classification[noise_indices] = 7  # Assign noise to class 7

# === Step 3: Visualize clustered point cloud ===
print("STEP 3: Visualizing clustered point cloud...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Assign colors
color_map = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(unique_labels), 20)))
np.random.shuffle(color_map)

# Define a mapping of tab20 RGB colors to common names
tab20_colors = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
color_name_map = {
    tuple(tab20_colors[0][:3]): "Blue",
    tuple(tab20_colors[1][:3]): "Light Blue",
    tuple(tab20_colors[2][:3]): "Orange",
    tuple(tab20_colors[3][:3]): "Light Orange",
    tuple(tab20_colors[4][:3]): "Green",
    tuple(tab20_colors[5][:3]): "Light Green",
    tuple(tab20_colors[6][:3]): "Red",
    tuple(tab20_colors[7][:3]): "Light Red",
    tuple(tab20_colors[8][:3]): "Purple",
    tuple(tab20_colors[9][:3]): "Light Purple",
    tuple(tab20_colors[10][:3]): "Brown",
    tuple(tab20_colors[11][:3]): "Light Brown",
    tuple(tab20_colors[12][:3]): "Pink",
    tuple(tab20_colors[13][:3]): "Light Pink",
    tuple(tab20_colors[14][:3]): "Gray",
    tuple(tab20_colors[15][:3]): "Light Gray",
    tuple(tab20_colors[16][:3]): "Olive",
    tuple(tab20_colors[17][:3]): "Light Olive",
    tuple(tab20_colors[18][:3]): "Cyan",
    tuple(tab20_colors[19][:3]): "Light Cyan",
}

colors = np.zeros((len(points), 3))
for i, cls in enumerate(classification):
    if cls >= 10:  # Cluster classes
        cluster_idx = cls - 10
        if cluster_idx < len(color_map):
            colors[i] = color_map[cluster_idx][:3]
        else:
            colors[i] = [0.7, 0.7, 0.7]  # Default gray for excess clusters
    elif cls == 7:  # Noise - yellow
        colors[i] = [1.0, 1.0, 0.0]
    elif cls == 2:  # Ground - brown
        colors[i] = [0.59, 0.29, 0.0]
    elif cls == 0:  # Unclassified - gray
        colors[i] = [0.5, 0.5, 0.5]
    elif cls == 1:  # Potential unclassified - light gray
        colors[i] = [0.7, 0.7, 0.7]
    else:
        colors[i] = [0.7, 0.7, 0.7]  # Default gray

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

# === Step 4: Remove noise and save output ===
print("STEP 4: Removing noise points...")
non_noise_mask = classification != 7
points = points[non_noise_mask]
classification = classification[non_noise_mask]

# Save output LAS with retained CRS and geometry
print("STEP 5: Saving output LAS...")
new_header = laspy.LasHeader(point_format=2, version="1.2")  # Force LAS 1.2, point format 2
new_header.scales = las.header.scales
new_header.offsets = las.header.offsets
new_header.mins = [np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])]
new_header.maxs = [np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])]
new_header.global_encoding = las.header.global_encoding
if las.header.parse_crs() is not None:
    new_header.add_crs(las.header.parse_crs())
else:
    new_header.vlrs = []

new_las = laspy.LasData(new_header)
new_las.x = points[:, 0]
new_las.y = points[:, 1]
new_las.z = points[:, 2]
new_las.classification = classification
new_las.write(OUTPUT_LAS)
print(f"-> Output saved to: {OUTPUT_LAS}")

# Classification summary for top 3 clusters
print("-> Top 3 clusters by point count:")
total_points = len(classification)
unique_classes, class_counts = np.unique(classification, return_counts=True)

# Filter clusters (classes >= 10) and sort by count
cluster_classes = unique_classes[unique_classes >= 10]
cluster_counts = class_counts[unique_classes >= 10]
if len(cluster_classes) > 0:
    sorted_indices = np.argsort(cluster_counts)[::-1]
    top_3_indices = sorted_indices[:3]
    for idx in top_3_indices:
        uc = cluster_classes[idx]
        count = cluster_counts[idx]
        percentage = (count / total_points) * 100
        color_idx = uc - 10
        if color_idx < len(color_map):
            rgb = tuple(color_map[color_idx][:3])
            color_name = color_name_map.get(rgb, "Unknown Color")
        else:
            color_name = "No color assigned"
        print(f"   Class {uc}: {count} points ({percentage:.2f}%) - {color_name}")
else:
    print("   No clusters found (classes >= 10).")

# === DONE ===
end_time = time.time()
elapsed = end_time - start_time
print(f"=== Done! Clustering finished in {elapsed:.2f} seconds ===")