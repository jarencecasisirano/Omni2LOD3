import laspy
import numpy as np
import json
import os

# Paths
INPUT_LAS = r"D:\Projects\Thesis\outputs\ground_classification\NIMBB 111725_group1_densified_point_cloud_downsampled_0_2_ground_classified.las"
GEOJSON_PATH = r"D:\Projects\Thesis\data\bounding_box\NIMBB_bounding_box1.geojson"
OUTPUT_LAS = r"D:\Projects\Thesis\outputs\building_classification\NIMBB 111725_02.las"

# Ensure output directory exists
OUTPUT_DIR = os.path.dirname(OUTPUT_LAS)
if OUTPUT_DIR and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Read the LAS file
print("Reading LAS file...")
las = laspy.read(INPUT_LAS)
points = np.vstack((las.x, las.y, las.z)).T
classification = las.classification

# Step 2: Read the GeoJSON file
print("Reading GeoJSON bounding area...")
try:
    with open(GEOJSON_PATH, 'r') as f:
        geojson_data = json.load(f)
    if not geojson_data["features"] or not geojson_data["features"][0]["geometry"]["coordinates"]:
        raise ValueError("GeoJSON contains no valid polygon coordinates.")
    # Extract the first polygon's coordinates from MultiPolygon
    coordinates = geojson_data["features"][0]["geometry"]["coordinates"]
    if not coordinates or not coordinates[0]:  # Check if the first polygon exists
        raise ValueError("No valid polygon found in MultiPolygon.")
    # Flatten the coordinate list and remove the closing point
    polygon_coords = coordinates[0][0]  # Access the first ring
    polygon = np.array(polygon_coords[:-1])  # Exclude the closing point
    print(f"Extracted {len(polygon)} points from GeoJSON: {polygon}")  # Debug output
    if len(polygon) < 3:
        raise ValueError(f"Polygon must have at least 3 points, got {len(polygon)}.")
except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
    print(f"Error processing GeoJSON: {e}. Please check the file or recreate it in QGIS.")
    exit(1)

# Step 3: Check points inside the polygon (using 2D coordinates, ignoring Z)
print("Checking points inside the bounding area...")
def point_in_polygon(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

building_mask = np.array([point_in_polygon(p[0], p[1], polygon) for p in points[:, :2]])
non_ground_mask = classification != 2
building_points_mask = building_mask & non_ground_mask
classification[building_points_mask] = 6  # Reclassify non-ground points inside polygon as building (class 6)

# Step 4: Save updated LAS file
print("Saving updated LAS file...")
new_header = laspy.LasHeader(point_format=las.header.point_format.id, version=las.header.version)
new_header.scales = las.header.scales
new_header.offsets = las.header.offsets
new_header.mins = np.min(points, axis=0)  # Use original mins
new_header.maxs = np.max(points, axis=0)  # Use original maxs
new_header.global_encoding = las.header.global_encoding

if las.header.parse_crs() is not None:
    new_header.add_crs(las.header.parse_crs())

new_header.vlrs = list(las.header.vlrs)
new_las = laspy.LasData(new_header)
new_las.x = points[:, 0]
new_las.y = points[:, 1]
new_las.z = points[:, 2]
new_las.classification = classification

# Copy additional fields if present
for dimension in las.header.point_format.dimensions:
    if dimension.name not in ['X', 'Y', 'Z', 'classification']:
        if hasattr(las, dimension.name):
            setattr(new_las, dimension.name, getattr(las, dimension.name))

new_las.write(OUTPUT_LAS)
print(f"Output saved to: {OUTPUT_LAS}")

# Step 5: Summary of reclassified points
print("Updated classification counts:")
unique_classes, class_counts = np.unique(classification, return_counts=True)
for uc, count in zip(unique_classes, class_counts):
    percentage = (count / len(classification)) * 100
    print(f"   Class {uc}: {count} points ({percentage:.2f}%)")