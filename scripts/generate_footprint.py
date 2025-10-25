import sys
import laspy
import numpy as np
from sklearn.cluster import DBSCAN
import alphashape
import geopandas as gpd
from shapely.geometry import Polygon
import open3d as o3d

def main(input_las, output_shp, alpha=0.0, eps=1.0, min_samples=10):
    # Read LAS file
    las = laspy.read(input_las)
    
    # Filter building points (class 6)
    building_mask = las.classification == 6
    if np.sum(building_mask) == 0:
        print("No building points found (class 6).")
        sys.exit(1)
    
    xyz = np.vstack((las.x[building_mask], las.y[building_mask], las.z[building_mask])).T
    
    # Compute Z midpoint for bottom half
    min_z, max_z = np.min(xyz[:, 2]), np.max(xyz[:, 2])
    midpoint_z = min_z + (max_z - min_z) / 2
    bottom_mask = xyz[:, 2] <= midpoint_z
    bottom_xy = xyz[bottom_mask, :2]  # Project to XY
    
    if len(bottom_xy) < 10:
        print("Insufficient points in bottom half.")
        sys.exit(1)
    
    # Remove outliers with DBSCAN
    db = DBSCAN(eps=eps, min_samples=int(min_samples)).fit(bottom_xy)
    labels = db.labels_
    
    # Identify largest cluster
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        print("No valid clusters found after outlier removal.")
        sys.exit(1)
    largest_label = unique_labels[np.argmax(counts)]
    core_xy = bottom_xy[labels == largest_label]
    
    if len(core_xy) < 3:
        print("Insufficient points in largest cluster for polygon.")
        sys.exit(1)
    
    # Compute concave hull
    hull = alphashape.alphashape(core_xy, alpha)
    
    if not isinstance(hull, Polygon):
        print("Computed hull is not a valid Polygon.")
        sys.exit(1)
    
    # Save as shapefile
    gdf = gpd.GeoDataFrame(geometry=[hull], crs="EPSG:32651")
    gdf.to_file(output_shp, driver='ESRI Shapefile')
    print(f"Footprint shapefile saved to: {output_shp}")
    
    # Visualization
    # Create point cloud for building points (red)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (len(xyz), 1)))  # Red RGB
    
    # Create line set for footprint (yellow)
    hull_coords = np.array(hull.exterior.coords)[:-1]  # Exclude last point (repeated)
    z_base = min_z - 0.5  # Slightly below min Z
    z_top = z_base + 1.0  # Small extrusion for visibility
    points_3d = []
    lines = []
    for i, coord in enumerate(hull_coords):
        points_3d.append([coord[0], coord[1], z_base])
        points_3d.append([coord[0], coord[1], z_top])
        if i < len(hull_coords) - 1:
            lines.append([2 * i, 2 * (i + 1)])  # Base line
            lines.append([2 * i + 1, 2 * (i + 1) + 1])  # Top line
            lines.append([2 * i, 2 * i + 1])  # Vertical line
        else:
            lines.append([2 * i, 0])  # Close base loop
            lines.append([2 * i + 1, 1])  # Close top loop
            lines.append([2 * i, 2 * i + 1])  # Last vertical line
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_3d)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile([1.0, 1.0, 0.0], (len(lines), 1)))  # Yellow RGB
    
    # Visualize
    print("Visualizing building points (red) and footprint (yellow)...")
    o3d.visualization.draw_geometries([pcd, line_set], window_name="Building Points and Footprint")

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 6:
        print("Usage: python generate_footprint.py <input_las> <output_shp> [alpha] [eps] [min_samples]")
        sys.exit(1)
    
    # Default parameters
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    eps = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    min_samples = float(sys.argv[5]) if len(sys.argv) > 5 else 10
    
    main(sys.argv[1], sys.argv[2], alpha, eps, min_samples)