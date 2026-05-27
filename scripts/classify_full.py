import laspy
import numpy as np
import open3d as o3d
from scipy.interpolate import griddata
import argparse
import os
import sys

# Disable Open3D printing
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def classify_las(input_path, output_path):
    print(f"Loading {input_path}...")
    las = laspy.read(input_path)
    
    # Extract coordinates
    points_xyz = np.vstack((las.x, las.y, las.z)).T
    
    # Initialize classification array (0 = Unclassified)
    # If file already has classification, preserve it? The prompt implies re-classification.
    # We will reset to 0 for simplicity or just overwrite.
    classification = np.zeros(len(points_xyz), dtype=np.uint8)
    
    # ---------------------------------------------------------
    # 1. Noise Removal (Statistical Outlier Removal)
    # ---------------------------------------------------------
    print("Step 1: Removing Noise (Statistical Outlier Removal)...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    
    # remove_statistical_outlier returns [cleaned_pcd, inlier_indices]
    # We want to find outliers to mark them as Class 7
    _, inlier_indices_vec = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    inlier_mask = np.zeros(len(points_xyz), dtype=bool)
    inlier_mask[inlier_indices_vec] = True
    
    # Mark outliers as Noise (7)
    classification[~inlier_mask] = 7
    
    # Work only with inliers for the next steps
    non_noise_indices = np.where(inlier_mask)[0]
    non_noise_xyz = points_xyz[non_noise_indices]
    
    # ---------------------------------------------------------
    # 2. Ground Classification (Grid-Based Min Filter)
    # ---------------------------------------------------------
    print("Step 2: Classifying Ground (Grid-based Min Filter)...")
    
    # Grid parameters
    grid_size = 2.0 # meters
    
    # Find mins in grid
    x_min, y_min, _ = np.min(non_noise_xyz, axis=0)
    x_max, y_max, _ = np.max(non_noise_xyz, axis=0)
    
    width = int(np.ceil((x_max - x_min) / grid_size))
    height = int(np.ceil((y_max - y_min) / grid_size))
    
    # Create grid indices
    x_idx = ((non_noise_xyz[:, 0] - x_min) / grid_size).astype(int)
    y_idx = ((non_noise_xyz[:, 1] - y_min) / grid_size).astype(int)
    
    # Hash/Map grid cell to Min Z
    grid_min_z = {}
    for i in range(len(non_noise_xyz)):
        key = (x_idx[i], y_idx[i])
        z = non_noise_xyz[i, 2]
        if key not in grid_min_z or z < grid_min_z[key]:
            grid_min_z[key] = z
            
    # Extract Ground Control Points (GCPs) - Center of cells with min Z
    gcp_xy = []
    gcp_z = []
    
    for (gx, gy), z in grid_min_z.items():
        cx = x_min + (gx + 0.5) * grid_size
        cy = y_min + (gy + 0.5) * grid_size
        gcp_xy.append([cx, cy])
        gcp_z.append(z)
        
    gcp_xy = np.array(gcp_xy)
    gcp_z = np.array(gcp_z)
    
    # Interpolate Ground Surface
    points_xy = non_noise_xyz[:, 0:2]
    # methods: 'nearest', 'linear', 'cubic'. 'linear' is fast and good enough.
    ground_z_interp = griddata(gcp_xy, gcp_z, points_xy, method='nearest') 
    
    # Height Above Ground
    hag = non_noise_xyz[:, 2] - ground_z_interp
    
    # Threshold for Ground
    GROUND_THRESHOLD = 0.3 # meters
    ground_mask_local = hag < GROUND_THRESHOLD
    
    # Update Classification
    # We need to map local non_noise indices back to global classification array
    classification[non_noise_indices[ground_mask_local]] = 2
    
    # ---------------------------------------------------------
    # 3. Classify Low Vegetation (Low Non-Ground)
    # ---------------------------------------------------------
    print("Step 3: Classifying Low Vegetation...")
    low_veg_mask_local = (hag >= GROUND_THRESHOLD) & (hag < 2.0)
    classification[non_noise_indices[low_veg_mask_local]] = 3
    
    # ---------------------------------------------------------
    # 4. Classify High Points (Buildings vs High Vegetation)
    # ---------------------------------------------------------
    print("Step 4: Classifying High Objects (Building vs High Veg)...")
    
    # Filter points that are candidates (HAG > 2.0)
    high_obj_mask_local = hag >= 2.0
    high_obj_indices_global = non_noise_indices[high_obj_mask_local]
    
    if len(high_obj_indices_global) > 0:
        high_xyz = points_xyz[high_obj_indices_global]
        
        # Use Open3D for Plane Segmentation (RANSAC)
        # Buildings are composed of Planes (Roofs, Facades)
        # Vegetation is scattered
        
        high_pcd = o3d.geometry.PointCloud()
        high_pcd.points = o3d.utility.Vector3dVector(high_xyz)
        
        building_mask_high = np.zeros(len(high_xyz), dtype=bool)
        
        # Iterative RANSAC to find planes
        remaining_pcd = high_pcd
        # Track indices relative to 'high_xyz'
        current_indices = np.arange(len(high_xyz))
        
        MIN_PLANE_SIZE = 100 # Minimum points to call it a plane
        
        iteration = 0
        while len(current_indices) > MIN_PLANE_SIZE and iteration < 50:
            # Segment plane
            plane_model, inliers_rel = remaining_pcd.segment_plane(distance_threshold=0.2,
                                                                 ransac_n=3,
                                                                 num_iterations=100)
            
            if len(inliers_rel) < MIN_PLANE_SIZE:
                break
                
            # Mark these as Building
            # 'inliers_rel' are indices into 'remaining_pcd' / 'current_indices'
            original_indices = current_indices[inliers_rel]
            building_mask_high[original_indices] = True
            
            # Remove inliers for next round
            remaining_pcd = remaining_pcd.select_by_index(inliers_rel, invert=True)
            current_indices = np.delete(current_indices, inliers_rel)
            
            iteration += 1
            if iteration % 10 == 0:
                print(f"  - Found {iteration} planes so far...")

        # Assign Classes
        # In High Points:
        #   Building Mask -> Class 6 (Building)
        #   Rest -> Class 5 (High Veg)
        
        classification[high_obj_indices_global[building_mask_high]] = 6
        classification[high_obj_indices_global[~building_mask_high]] = 5
        
    # ---------------------------------------------------------
    # 5. Save Output
    # ---------------------------------------------------------
    print(f"Saving to {output_path}...")
    
    new_header = laspy.LasHeader(point_format=las.header.point_format.id, version=las.header.version)
    new_header.scales = las.header.scales
    new_header.offsets = las.header.offsets
    new_header.mins = las.header.mins
    new_header.maxs = las.header.maxs
    
    if las.header.parse_crs() is not None:
        new_header.add_crs(las.header.parse_crs())
        
    out_las = laspy.LasData(new_header)
    out_las.x = las.x
    out_las.y = las.y
    out_las.z = las.z
    out_las.classification = classification
    
    # Preserve other fields if possible, or just minimal
    # (Simplified for now)
    
    out_las.write(output_path)
    
    # Summary
    classes, counts = np.unique(classification, return_counts=True)
    print("\nClassification Results:")
    for c, count in zip(classes, counts):
        name = {2: "Ground", 3: "Low Veg", 5: "High Veg", 6: "Building", 7: "Noise", 0: "Unclassed"}.get(c, str(c))
        print(f"  Class {c} ({name}): {count} points")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify LAS using Open3D/Laspy")
    parser.add_argument("input_file", help="Input LAS file")
    parser.add_argument("output_file", help="Output LAS file")
    args = parser.parse_args()
    
    classify_las(args.input_file, args.output_file)
