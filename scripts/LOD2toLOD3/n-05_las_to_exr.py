import os
import glob
import argparse
import numpy as np
import laspy
import OpenEXR
import Imath

def create_exr_from_las(las_path, out_dir, resolution=0.05):
    """
    Converts a .las file into a 2D Orthographic EXR image by automatically 
    finding the optimal viewing angle using PCA on the X-Y plane.
    """
    print(f"Processing: {os.path.basename(las_path)}")
    
    # 1. Read LAS data
    las = laspy.read(las_path)
    
    x = las.x
    y = las.y
    z = las.z
    
    # Normalize RGB to 0.0 - 1.0
    r = las.red / 65535.0
    g = las.green / 65535.0
    b = las.blue / 65535.0

    # 2. Find the Optimal Camera Angle (PCA on X-Y plane)
    # We want to look straight at the facade while keeping Z pointing UP.
    xy_points = np.vstack((x, y)).T
    xy_centroid = np.mean(xy_points, axis=0)
    xy_centered = xy_points - xy_centroid
    
    # Calculate covariance matrix and eigenvectors to find the dominant axis
    cov_matrix = np.cov(xy_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by variance descending. 
    # The largest eigenvector is the facade's width. The smallest is its depth.
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sort_idx]
    
    axis_width = eigenvectors[:, 0]  # The new "Camera X" (Left/Right)
    axis_depth = eigenvectors[:, 1]  # The new "Camera Y" (Forward/Backward)

    # 3. Project points into the Camera's Local Coordinate System
    # This transforms the 3D space so the facade is perfectly flat to the camera
    local_x = np.dot(xy_centered, axis_width)
    local_depth = np.dot(xy_centered, axis_depth)
    
    # 4. Determine Image Dimensions based on the local projection
    min_x, max_x = np.min(local_x), np.max(local_x)
    min_z, max_z = np.min(z), np.max(z)
    
    width = int(np.ceil((max_x - min_x) / resolution)) + 1
    height = int(np.ceil((max_z - min_z) / resolution)) + 1
    
    # 5. Calculate 2D pixel coordinates (u, v)
    u = np.floor((local_x - min_x) / resolution).astype(int)
    v = np.floor((max_z - z) / resolution).astype(int) # Flip Z so top is Y=0

    # 6. Handle Occlusion (Depth Sorting using local_depth)
    # Sort by the new camera depth so the points closest to the camera overwrite the ones behind
    sort_idx = np.argsort(-local_depth)
    
    u = u[sort_idx]
    v = v[sort_idx]
    # CRITICAL: We map the ORIGINAL XYZ values to the pixels, not the local ones.
    # This ensures the image retains true geographic data for reconversion later.
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    z_sorted = z[sort_idx]
    r_sorted = r[sort_idx]
    g_sorted = g[sort_idx]
    b_sorted = b[sort_idx]

    # 7. Initialize Image Channels with NaN
    img_x = np.full((height, width), np.nan, dtype=np.float32)
    img_y = np.full((height, width), np.nan, dtype=np.float32)
    img_z = np.full((height, width), np.nan, dtype=np.float32)
    img_r = np.full((height, width), np.nan, dtype=np.float32)
    img_g = np.full((height, width), np.nan, dtype=np.float32)
    img_b = np.full((height, width), np.nan, dtype=np.float32)

    # 8. Populate the image arrays
    img_x[v, u] = x_sorted
    img_y[v, u] = y_sorted
    img_z[v, u] = z_sorted
    img_r[v, u] = r_sorted
    img_g[v, u] = g_sorted
    img_b[v, u] = b_sorted

    # 9. Write to EXR
    filename = os.path.splitext(os.path.basename(las_path))[0] + ".exr"
    out_path = os.path.join(out_dir, filename)
    
    header = OpenEXR.Header(width, height)
    FLOAT = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    
    header['channels'] = {
        'X': FLOAT, 'Y': FLOAT, 'Z': FLOAT, 
        'R': FLOAT, 'G': FLOAT, 'B': FLOAT
    }
    
    out = OpenEXR.OutputFile(out_path, header)
    
    out.writePixels({
        'X': img_x.tobytes(),
        'Y': img_y.tobytes(),
        'Z': img_z.tobytes(),
        'R': img_r.tobytes(),
        'G': img_g.tobytes(),
        'B': img_b.tobytes()
    })
    out.close()
    
    print(f"Saved: {out_path} ({width}x{height} pixels)")

def main():
    parser = argparse.ArgumentParser(description="Convert LAS point clouds to EXR images with auto-alignment.")
    parser.add_argument(
        '-i', '--input', 
        type=str, 
        default="outputs/06_aligned_p2p/NIMBB-2",
        help="Path to the folder containing .las files."
    )
    args = parser.parse_args()
    
    input_dir = args.input
    
    if not os.path.exists(input_dir):
        print(f"Error: The directory '{input_dir}' does not exist.")
        return
        
    folder_name = os.path.basename(os.path.normpath(input_dir))
    
    output_base_dir = "outputs/07_exr_image"
    output_dir = os.path.join(output_base_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    las_files = glob.glob(os.path.join(input_dir, "*.las"))
    
    if not las_files:
        print(f"No .las files found in {input_dir}")
        return
        
    print(f"Found {len(las_files)} files. Outputting to: {output_dir}")
    
    RESOLUTION_METERS = 0.10
    
    for las_file in las_files:
        create_exr_from_las(las_file, output_dir, resolution=RESOLUTION_METERS)
        
    print("All processing complete.")

if __name__ == "__main__":
    main()