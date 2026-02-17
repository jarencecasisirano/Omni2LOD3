import open3d as o3d
import numpy as np
import laspy
import os
import argparse
import glob
from tqdm import tqdm
from CSF import CSF, VecInt

def filter_buildings(file_path, output_path, nb_neighbors=20, std_ratio=2.0,
                    cloth_resolution=0.5, rigidness=3, class_threshold=0.5):
    """
    Multi-step filtration to isolate buildings:
    1. Statistical Outlier Removal (SOR) to denoise
    2. Cloth Simulation Filter (CSF) to remove ground
    
    Parameters:
    -----------
    file_path : str
        Path to input LAS file
    output_path : str
        Path to output LAS file
    nb_neighbors : int
        Number of neighbors for SOR (default: 20)
    std_ratio : float
        Standard deviation ratio for SOR (default: 2.0)
    cloth_resolution : float
        CSF cloth grid resolution (default: 0.5)
    rigidness : int
        CSF cloth stiffness 1=steep, 2=relief, 3=flat (default: 3)
    class_threshold : float
        CSF classification threshold (default: 0.5)
    """
    print(f"Processing {os.path.basename(file_path)}...")
    
    # Load LAS file
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float64)
    
    # Load colors if available
    if hasattr(las, 'red'):
        colors = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0
    else:
        colors = None
    
    print(f"  Original points: {len(points):,}")
    
    # Step 1: Statistical Outlier Removal (Denoising)
    print("  Step 1: Denoising with Statistical Outlier Removal...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Remove statistical outliers
    pcd_filtered, inliers = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    
    points_denoised = np.asarray(pcd_filtered.points)
    if colors is not None:
        colors_denoised = np.asarray(pcd_filtered.colors)
    else:
        colors_denoised = None
    
    print(f"  After denoising: {len(points_denoised):,} points ({len(points) - len(points_denoised):,} removed)")
    
    # Step 2: Ground Removal with CSF
    print("  Step 2: Ground removal with Cloth Simulation Filter...")
    
    # CSF requires specific format: list of [x, y, z]
    csf = CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = cloth_resolution
    csf.params.rigidness = rigidness
    csf.params.time_step = 0.65
    csf.params.class_threshold = class_threshold
    csf.params.interations = 500
    
    # Set the point cloud
    csf.setPointCloud(points_denoised.tolist())
    
    # Execute ground filtering
    # Use CSF VecInt wrapper for C++ compatibility
    ground_indices = VecInt()
    non_ground_indices = VecInt()
    csf.do_filtering(ground_indices, non_ground_indices)
    
    # Convert to numpy arrays for indexing
    ground_indices = np.array(ground_indices)
    non_ground_indices = np.array(non_ground_indices)
    
    print(f"  Ground points: {len(ground_indices):,}")
    print(f"  Non-ground (buildings/vegetation): {len(non_ground_indices):,}")
    
    # Extract non-ground points (buildings)
    points_buildings = points_denoised[non_ground_indices]
    if colors_denoised is not None:
        colors_buildings = colors_denoised[non_ground_indices]
    else:
        colors_buildings = np.zeros((len(points_buildings), 3))
    
    # Save results
    print(f"  Saving {len(points_buildings):,} points to {output_path}...")
    
    # Create new LAS with filtered points
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = las.header.scales
    header.offsets = las.header.offsets
    
    new_las = laspy.LasData(header)
    
    # Set coordinates
    new_las.x = points_buildings[:, 0]
    new_las.y = points_buildings[:, 1]
    new_las.z = points_buildings[:, 2]
    
    # Set colors
    new_las.red = (colors_buildings[:, 0] * 65535).astype(np.uint16)
    new_las.green = (colors_buildings[:, 1] * 65535).astype(np.uint16)
    new_las.blue = (colors_buildings[:, 2] * 65535).astype(np.uint16)
    
    # Set classification (6 = Building as default for non-ground)
    new_las.classification = np.full(len(points_buildings), 6, dtype=np.uint8)
    
    new_las.write(output_path)
    print(f"  Saved successfully!")

def main():
    parser = argparse.ArgumentParser(
        description="Building isolation pipeline: Denoise + Ground removal"
    )
    parser.add_argument("--input_dir", type=str, default="outputs/03_pointclouds",
                       help="Input directory containing LAS files")
    parser.add_argument("--output_dir", type=str, default="outputs/06_buildings",
                       help="Output directory for filtered files")
    parser.add_argument("--nb_neighbors", type=int, default=20,
                       help="Number of neighbors for SOR (default: 20)")
    parser.add_argument("--std_ratio", type=float, default=2.0,
                       help="Standard deviation ratio for SOR (default: 2.0)")
    parser.add_argument("--cloth_resolution", type=float, default=0.5,
                       help="CSF cloth grid resolution (default: 0.5)")
    parser.add_argument("--rigidness", type=int, default=3,
                       help="CSF rigidness: 1=steep, 2=relief, 3=flat (default: 3)")
    parser.add_argument("--class_threshold", type=float, default=0.5,
                       help="CSF classification threshold (default: 0.5)")
    
    args = parser.parse_args()
    
    # Find all LAS files
    files = glob.glob(os.path.join(args.input_dir, "**", "*.las"), recursive=True)
    
    if not files:
        print(f"No .las files found in {args.input_dir}")
        return
    
    print(f"Found {len(files)} files to process\n")
    print("=" * 60)
    
    for f in tqdm(files, desc="Overall Progress"):
        # Maintain directory structure
        rel_path = os.path.relpath(os.path.dirname(f), args.input_dir)
        out_folder = args.output_dir if rel_path == "." else os.path.join(args.output_dir, rel_path)
        os.makedirs(out_folder, exist_ok=True)
        
        out_name = os.path.basename(f)
        out_path = os.path.join(out_folder, out_name)
        
        try:
            filter_buildings(
                f, out_path,
                nb_neighbors=args.nb_neighbors,
                std_ratio=args.std_ratio,
                cloth_resolution=args.cloth_resolution,
                rigidness=args.rigidness,
                class_threshold=args.class_threshold
            )
        except Exception as e:
            print(f"  ERROR processing {f}: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print(f"All files processed! Output saved to {args.output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
