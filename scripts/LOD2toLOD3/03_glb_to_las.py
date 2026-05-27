import argparse
import os
import glob
import trimesh
import numpy as np
import laspy
from tqdm import tqdm

def process_file(file_path, output_dir, n_samples=1000000):
    """
    Converts a single .glb file to .las point cloud.
    """
    try:
        filename = os.path.basename(file_path)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}.las")
        
        print(f"Processing: {filename}")

        # Load the mesh
        # Don't force mesh, so we can Inspect the scene
        mesh = trimesh.load(file_path)

        if isinstance(mesh, trimesh.Scene):
            # Inspect scene for PointClouds and Meshes
            point_clouds = [g for g in mesh.geometry.values() if isinstance(g, trimesh.points.PointCloud)]
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            
            final_points = []
            final_colors = []
            
            # Handle Point Clouds
            for pc in point_clouds:
                final_points.append(pc.vertices)
                if hasattr(pc, 'colors') and len(pc.colors) > 0:
                    final_colors.append(pc.colors)
                else:
                    # Default white
                    final_colors.append(np.ones((len(pc.vertices), 4), dtype=np.uint8) * 255)

            # Handle Meshes (Sample them)
            if meshes:
                combined_mesh = trimesh.util.concatenate(meshes)
                if not combined_mesh.is_empty:
                    # Sample points
                    points, face_indices = trimesh.sample.sample_surface(combined_mesh, n_samples)
                    final_points.append(points)
                    
                    # Colors for sampled points
                    if hasattr(combined_mesh.visual, 'to_color'):
                        try:
                            combined_mesh.visual = combined_mesh.visual.to_color()
                        except:
                            pass
                            
                    if not hasattr(combined_mesh.visual, 'vertex_colors'):
                        combined_mesh.visual.vertex_colors = np.ones((len(combined_mesh.vertices), 4), dtype=np.uint8) * 255

                    triangles = combined_mesh.vertices[combined_mesh.faces[face_indices]]
                    v0 = triangles[:, 0, :]
                    v1 = triangles[:, 1, :]
                    v2 = triangles[:, 2, :]
                    p = points
                    v0v1 = v1 - v0
                    v0v2 = v2 - v0
                    v0p = p - v0
                    d00 = np.einsum('ij,ij->i', v0v1, v0v1)
                    d01 = np.einsum('ij,ij->i', v0v1, v0v2)
                    d11 = np.einsum('ij,ij->i', v0v2, v0v2)
                    d20 = np.einsum('ij,ij->i', v0p, v0v1)
                    d21 = np.einsum('ij,ij->i', v0p, v0v2)
                    denom = d00 * d11 - d01 * d01
                    denom[denom == 0] = 1.0
                    v = (d11 * d20 - d01 * d21) / denom
                    w = (d00 * d21 - d01 * d20) / denom
                    u = 1.0 - v - w
                    
                    faces = combined_mesh.faces[face_indices]
                    if hasattr(combined_mesh.visual, 'vertex_colors') and len(combined_mesh.visual.vertex_colors) > 0:
                        face_colors = combined_mesh.visual.vertex_colors[faces]
                    else:
                        face_colors = np.ones((len(faces), 3, 4), dtype=np.uint8) * 255
                    
                    c0 = face_colors[:, 0, :]
                    c1 = face_colors[:, 1, :]
                    c2 = face_colors[:, 2, :]
                    point_colors = (u[:, None] * c0 + v[:, None] * c1 + w[:, None] * c2).astype(np.uint8)
                    final_colors.append(point_colors)

            if not final_points:
                print(f"Skipping {filename}: No relevant geometry (Mesh or PointCloud) found")
                return

            all_points = np.vstack(final_points)
            all_colors = np.vstack(final_colors)
            
        elif isinstance(mesh, trimesh.Trimesh):
             # Logic for single mesh (same as above but simplified)
             # Reuse the mesh logic... actually let's just treat it as a list of 1 mesh
             pass # Logic implied if we restructure code but for now let's copy-paste or assume Scene handles most
             
             # Sample points
             points, face_indices = trimesh.sample.sample_surface(mesh, n_samples)
             all_points = points
             
             # (Color logic repeated... redundant but functional)
             if hasattr(mesh.visual, 'to_color'):
                try:
                    mesh.visual = mesh.visual.to_color()
                except:
                    pass
             if not hasattr(mesh.visual, 'vertex_colors'):
                 mesh.visual.vertex_colors = np.ones((len(mesh.vertices), 4), dtype=np.uint8) * 255
                 
             triangles = mesh.vertices[mesh.faces[face_indices]]
             v0 = triangles[:, 0, :]
             v1 = triangles[:, 1, :]
             v2 = triangles[:, 2, :]
             p = points
             v0v1 = v1 - v0
             v0v2 = v2 - v0
             v0p = p - v0
             d00 = np.einsum('ij,ij->i', v0v1, v0v1)
             d01 = np.einsum('ij,ij->i', v0v1, v0v2)
             d11 = np.einsum('ij,ij->i', v0v2, v0v2)
             d20 = np.einsum('ij,ij->i', v0p, v0v1)
             d21 = np.einsum('ij,ij->i', v0p, v0v2)
             denom = d00 * d11 - d01 * d01
             denom[denom == 0] = 1.0
             v = (d11 * d20 - d01 * d21) / denom
             w = (d00 * d21 - d01 * d20) / denom
             u = 1.0 - v - w
             faces = mesh.faces[face_indices]
             if hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) > 0:
                face_colors = mesh.visual.vertex_colors[faces]
             else:
                face_colors = np.ones((len(faces), 3, 4), dtype=np.uint8) * 255
             c0 = face_colors[:, 0, :]
             c1 = face_colors[:, 1, :]
             c2 = face_colors[:, 2, :]
             all_colors = (u[:, None] * c0 + v[:, None] * c1 + w[:, None] * c2).astype(np.uint8)
             
        else:
             print(f"Skipping {filename}: Unknown geometry type {type(mesh)}")
             return

        # Write LAS
        if len(all_points) == 0:
             print(f"Skipping {filename}: 0 points")
             return
             
        # Create LAS header
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = np.array([0.001, 0.001, 0.001]) # mm precision
        header.offsets = np.min(all_points, axis=0) # offset by min to maintain precision
        
        las = laspy.LasData(header)
        las.x = all_points[:, 0]
        las.y = all_points[:, 1]
        las.z = all_points[:, 2]
        
        # LAS requires 16-bit color
        # Ensure colors are (N, 3) or (N, 4)
        if all_colors.shape[1] == 4:
            all_colors = all_colors[:, :3]
            
        las.red = all_colors[:, 0].astype(np.uint16) * 256
        las.green = all_colors[:, 1].astype(np.uint16) * 256
        las.blue = all_colors[:, 2].astype(np.uint16) * 256
        
        las.write(output_path)
        print(f"Saved to: {output_path}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert GLB files to LAS point clouds.")
    parser.add_argument("--file", type=str, help="Specific GLB file to process (filename only)")
    parser.add_argument("--input_dir", type=str, default="outputs/02_glb_file", help="Input directory containing GLB files")
    parser.add_argument("--output_dir", type=str, default="outputs/03_pointclouds", help="Output directory for LAS files")
    parser.add_argument("--samples", type=int, default=1000000, help="Number of points to sample per file")
    
    args = parser.parse_args()
    
    files_to_process = []
    
    if args.file:
        # If a specific file is requested, look for it recursively if not found directly
        file_path = os.path.join(args.input_dir, args.file)
        if os.path.exists(file_path):
             files_to_process.append(file_path)
        else:
            # Try to find it recursively
            found_files = glob.glob(os.path.join(args.input_dir, "**", args.file), recursive=True)
            if found_files:
                files_to_process.extend(found_files)
            else:
                print(f"Error: File {args.file} not found in {args.input_dir}")
                return
    else:
        # Interactive selection
        if not os.path.exists(args.input_dir):
             print(f"Error: Input directory {args.input_dir} does not exist.")
             return

        # Get subdirectories
        subdirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
        subdirs.sort()
        
        print(f"\nFound {len(subdirs)} subdirectories in {args.input_dir}:")
        print("0: [PROCESS ALL FILES RECURSIVELY]")
        print("1: [ROOT ONLY] (Files in top-level directory)")
        for i, subdir in enumerate(subdirs):
            print(f"{i+2}: {subdir}")
            
        try:
            selection = input("\nSelect directory to process (enter number): ")
            selection = int(selection)
        except ValueError:
            print("Invalid selection. Exiting.")
            return

        if selection == 0:
            print("Processing ALL files recursively...")
            files_to_process = glob.glob(os.path.join(args.input_dir, "**", "*.glb"), recursive=True)
        elif selection == 1:
            print("Processing ROOT files only...")
            files_to_process = glob.glob(os.path.join(args.input_dir, "*.glb"))
        elif selection >= 2 and selection < len(subdirs) + 2:
            target_subdir = subdirs[selection - 2]
            print(f"Processing directory: {target_subdir}")
            # Process files in that subdirectory (and its subdirectories if any)
            target_path = os.path.join(args.input_dir, target_subdir)
            files_to_process = glob.glob(os.path.join(target_path, "**", "*.glb"), recursive=True)
        else:
            print("Invalid selection number. Exiting.")
            return

    if not files_to_process:
        print(f"No .glb files found to process.")
        return

    print(f"Found {len(files_to_process)} files to process.")
    
    # Process files
    for file_path in tqdm(files_to_process):
        # Determine relative path to maintain structure in output
        rel_path = os.path.relpath(os.path.dirname(file_path), args.input_dir)
        if rel_path == ".":
            current_output_dir = args.output_dir
        else:
            current_output_dir = os.path.join(args.output_dir, rel_path)
            
        os.makedirs(current_output_dir, exist_ok=True)
        process_file(file_path, current_output_dir, args.samples)

if __name__ == "__main__":
    main()
