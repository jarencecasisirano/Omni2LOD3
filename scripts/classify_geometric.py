import open3d as o3d
import numpy as np
import laspy
import os
import argparse
import glob
from tqdm import tqdm

# Desired Colors (RGB 0-255)
COLOR_MAP = {
    'vegetation': [0, 255, 0],       # Green
    'ground': [165, 42, 42],         # Brown
    'building': [255, 0, 0],         # Red
    'unclassified': [0, 0, 0]        # Black
}

def classify_geometric(file_path, output_path):
    """
    Unsupervised geometric classification using:
    1. RANSAC plane detection for ground
    2. Normal orientation + color heuristics for building vs vegetation
    """
    print(f"Processing {os.path.basename(file_path)}...")
    
    # Load LAS
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Load original colors if available
    if hasattr(las, 'red'):
        colors = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        colors = np.zeros((len(points), 3))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Initialize classification
    # 0: Unclassified, 1: Ground, 2: Building, 3: Vegetation
    classification = np.zeros(len(points), dtype=np.int32)
    new_colors = np.zeros((len(points), 3))
    
    # Step 1: Ground Detection (RANSAC plane fitting)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.3,
                                             ransac_n=3,
                                             num_iterations=1000)
    
    # Check if plane is horizontal
    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    
    is_horizontal = abs(normal[2]) > 0.7  # Z-component dominant
    
    if is_horizontal and len(inliers) > 0:
        classification[inliers] = 1  # Ground
        new_colors[inliers] = np.array(COLOR_MAP['ground']) / 255.0
        
        # Get non-ground points
        non_ground_pcd = pcd.select_by_index(inliers, invert=True)
        non_ground_indices = np.delete(np.arange(len(points)), inliers)
    else:
        non_ground_pcd = pcd
        non_ground_indices = np.arange(len(points))
    
    # Step 2: Building vs Vegetation classification
    if len(non_ground_pcd.points) > 0:
        # Estimate normals
        non_ground_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        
        ng_normals = np.asarray(non_ground_pcd.normals)
        ng_colors = np.asarray(non_ground_pcd.colors)
        
        for i in range(len(non_ground_pcd.points)):
            original_idx = non_ground_indices[i]
            
            # Normal Z-component (vertical surfaces have low |nz|)
            nz = abs(ng_normals[i][2])
            
            # Color analysis
            r, g, b = ng_colors[i]
            is_green = (g > r * 1.1) and (g > b * 1.1)  # Green dominant
            
            # Geometric: vertical surface = building wall
            is_vertical = nz < 0.3
            
            if is_green:
                classification[original_idx] = 3  # Vegetation
                new_colors[original_idx] = np.array(COLOR_MAP['vegetation']) / 255.0
            elif is_vertical:
                classification[original_idx] = 2  # Building
                new_colors[original_idx] = np.array(COLOR_MAP['building']) / 255.0
            else:
                # Likely building roof or other
                classification[original_idx] = 2  # Default to building
                new_colors[original_idx] = np.array(COLOR_MAP['building']) / 255.0
    
    # Save results
    new_las = laspy.LasData(las.header)
    new_las.x = las.x
    new_las.y = las.y
    new_las.z = las.z
    
    # Update colors
    new_las.red = (new_colors[:, 0] * 65535).astype(np.uint16)
    new_las.green = (new_colors[:, 1] * 65535).astype(np.uint16)
    new_las.blue = (new_colors[:, 2] * 65535).astype(np.uint16)
    
    # Update classification (LAS standard: 2=Ground, 6=Building, 4=Med Vegetation)
    las_class_map = {0: 0, 1: 2, 2: 6, 3: 4}
    final_classes = np.zeros_like(classification, dtype=np.uint8)
    for k, v in las_class_map.items():
        final_classes[classification == k] = v
    new_las.classification = final_classes
    
    new_las.write(output_path)
    print(f"Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Geometric-based unsupervised classification")
    parser.add_argument("--input_dir", type=str, default="outputs/03_pointclouds", 
                       help="Input directory")
    parser.add_argument("--output_dir", type=str, default="outputs/05_geometric", 
                       help="Output directory")
    args = parser.parse_args()
    
    # Find all LAS files
    files = glob.glob(os.path.join(args.input_dir, "**", "*.las"), recursive=True)
    
    if not files:
        print(f"No .las files found in {args.input_dir}")
        return
    
    print(f"Found {len(files)} files to process")
    
    for f in tqdm(files):
        # Maintain directory structure
        rel_path = os.path.relpath(os.path.dirname(f), args.input_dir)
        out_folder = args.output_dir if rel_path == "." else os.path.join(args.output_dir, rel_path)
        os.makedirs(out_folder, exist_ok=True)
        
        out_name = os.path.basename(f)
        out_path = os.path.join(out_folder, out_name)
        
        classify_geometric(f, out_path)
    
    print(f"\nAll files processed and saved to {args.output_dir}")

if __name__ == "__main__":
    main()
