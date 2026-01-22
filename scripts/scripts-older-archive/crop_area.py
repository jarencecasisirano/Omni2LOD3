import laspy
import numpy as np
import open3d as o3d
import os

# Path to the DBSCAN output LAS file
INPUT_LAS = r"D:\Projects\Thesis\outputs\building_classification\dji_mbb_downsampled_0_5_clustered.las"
OUTPUT_LAS = r"D:\Projects\Thesis\outputs\building_classification\cropped_cluster.las"

# Ensure output directory exists
OUTPUT_DIR = os.path.dirname(OUTPUT_LAS)
if OUTPUT_DIR and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Read the LAS file
print("Reading LAS file...")
las = laspy.read(INPUT_LAS)
points = np.vstack((las.x, las.y, las.z)).T
classification = las.classification
total_points = len(classification)

# Step 2: Identify top 3 clusters (classes >= 10)
unique_classes, class_counts = np.unique(classification, return_counts=True)
cluster_classes = unique_classes[unique_classes >= 10]
cluster_counts = class_counts[unique_classes >= 10]

if len(cluster_classes) == 0:
    print("No DBSCAN clusters (classes >= 10) found.")
    exit(1)

# Sort clusters by count and select top 3
sorted_indices = np.argsort(cluster_counts)[::-1]
top_3_indices = sorted_indices[:min(3, len(cluster_classes))]
top_3_classes = cluster_classes[top_3_indices]
top_3_counts = cluster_counts[top_3_indices]

# Assign basic colors: Red, Yellow, Blue
color_map = {
    top_3_classes[0]: ([1.0, 0.0, 0.0], "Red"),  # First cluster: Red
    top_3_classes[1] if len(top_3_classes) > 1 else -1: ([1.0, 1.0, 0.0], "Yellow"),  # Second: Yellow
    top_3_classes[2] if len(top_3_classes) > 2 else -1: ([0.0, 0.0, 1.0], "Blue"),  # Third: Blue
}

# Print top 3 clusters
print("Top 3 clusters by point count:")
for i, (cls, count) in enumerate(zip(top_3_classes, top_3_counts)):
    percentage = (count / total_points) * 100
    color_name = color_map[cls][1] if cls in color_map else "Unknown"
    print(f"   Class {cls}: {count} points ({percentage:.2f}%) - {color_name}")

# Step 3: Visualize top 3 clusters and ground points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
colors = np.zeros((len(points), 3))

for i, cls in enumerate(classification):
    if cls in top_3_classes:
        colors[i] = color_map[cls][0]  # Assign Red, Yellow, or Blue
    elif cls == 2:  # Ground points
        colors[i] = [0.59, 0.29, 0.0]  # Brown
    else:
        colors[i] = [0.5, 0.5, 0.5]  # Gray for others (including unclassified, noise)

pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd], window_name="Top 3 Clusters and Ground Points")

# Step 4: Prompt user for color selection
valid_colors = [color_map[cls][1] for cls in top_3_classes if cls in color_map]
print(f"\nAvailable colors: {', '.join(valid_colors)}")
while True:
    selected_color = input("Enter the color of the cluster to keep (Red, Yellow, Blue): ").capitalize()
    if selected_color in valid_colors:
        break
    print(f"Invalid color. Please choose from: {', '.join(valid_colors)}")

# Find the class corresponding to the selected color
selected_class = None
for cls, (rgb, name) in color_map.items():
    if name == selected_color and cls != -1:
        selected_class = cls
        break

if selected_class is None:
    print("Error: Selected color not associated with a valid cluster.")
    exit(1)

# Step 5: Compute 3D bounding box for the selected cluster
cluster_mask = classification == selected_class
cluster_points = points[cluster_mask]
min_xyz = np.min(cluster_points, axis=0)  # Min X, Y, Z
max_xyz = np.max(cluster_points, axis=0)  # Max X, Y, Z
buffer = 5.0  # Buffer in meters around the cluster (applied to all dimensions)
min_xyz -= buffer
max_xyz += buffer

# Filter points within the 3D bounding box (ALL points, regardless of class)
crop_mask = (
    (points[:, 0] >= min_xyz[0]) & (points[:, 0] <= max_xyz[0]) &  # X bounds
    (points[:, 1] >= min_xyz[1]) & (points[:, 1] <= max_xyz[1]) &  # Y bounds
    (points[:, 2] >= min_xyz[2]) & (points[:, 2] <= max_xyz[2])    # Z bounds
)
cropped_points = points[crop_mask]
cropped_classification = classification[crop_mask]

# Step 6: Save cropped LAS file with preserved metadata
print("Saving cropped LAS file...")
new_header = laspy.LasHeader(point_format=las.header.point_format.id, version=las.header.version)
new_header.scales = las.header.scales
new_header.offsets = las.header.offsets
new_header.mins = np.min(cropped_points, axis=0)  # Update mins for cropped points
new_header.maxs = np.max(cropped_points, axis=0)  # Update maxs for cropped points
new_header.global_encoding = las.header.global_encoding

if las.header.parse_crs() is not None:
    new_header.add_crs(las.header.parse_crs())

new_header.vlrs = list(las.header.vlrs)
new_las = laspy.LasData(new_header)
new_las.x = cropped_points[:, 0]
new_las.y = cropped_points[:, 1]
new_las.z = cropped_points[:, 2]
new_las.classification = cropped_classification

# Copy additional fields if present (e.g., intensity, return_number)
for dimension in las.point_format.dimensions:
    if dimension.name not in ['X', 'Y', 'Z', 'classification']:
        if hasattr(las, dimension.name):
            setattr(new_las, dimension.name, getattr(las, dimension.name)[crop_mask])

new_las.write(OUTPUT_LAS)
print(f"Output saved to: {OUTPUT_LAS}")

# Summary of cropped file
unique_classes, class_counts = np.unique(cropped_classification, return_counts=True)
print("Cropped file classification counts:")
for uc, count in zip(unique_classes, class_counts):
    percentage = (count / len(cropped_classification)) * 100
    print(f"   Class {uc}: {count} points ({percentage:.2f}%)")