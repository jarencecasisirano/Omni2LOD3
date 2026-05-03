import os
import json
import numpy as np
import laspy
import open3d as o3d
from sklearn.cluster import DBSCAN
import argparse

def select_file_from_dir(prompt_msg, directory, extension):
    """Lists files in a directory and prompts the user to select one via terminal."""
    print(f"\n--- {prompt_msg} ---")
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        exit()

    files = [f for f in os.listdir(directory) if f.lower().endswith(extension)]
    
    if not files:
        print(f"Error: No '{extension}' files found in '{directory}'.")
        exit()

    for i, file_name in enumerate(files):
        print(f"[{i + 1}] {file_name}")

    while True:
        try:
            choice = input(f"Select a file (1-{len(files)}): ").strip()
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(files):
                selected_file = files[choice_idx]
                return os.path.join(directory, selected_file)
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(files)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def select_files():
    """Handles terminal selection for LAS and CityJSON files."""
    las_dir = "outputs/11A_facade_curve"
    json_dir = "outputs/15_curve_openings_json"

    las_path = select_file_from_dir("Select the LAS file", las_dir, ".las")
    print(f"Selected: {las_path}")

    json_path = select_file_from_dir("Select the CityJSON file", json_dir, ".json")
    print(f"Selected: {json_path}\n")

    return las_path, json_path

def create_mesh(points, max_points=10000):
    """Creates a mesh using BPA for large clouds, and Convex Hull for small ones."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    num_points = len(pcd.points)

    if num_points > max_points:
        ratio = max_points / num_points
        print(f"      Downsampling from {num_points} to ~{max_points} points...")
        pcd = pcd.random_down_sample(ratio)
        num_points = len(pcd.points)
        
    if num_points < 4: 
        print("      Not enough points for meshing (< 4). Skipping.")
        return None

    if num_points < 500:
        print(f"      Small cluster ({num_points} pts). Using Convex Hull strategy...")
        try:
            mesh, _ = pcd.compute_convex_hull()
            mesh.compute_vertex_normals()
            return mesh
        except Exception as e:
            print(f"      Convex Hull failed: {e}")
            return None

    try:
        print(f"      Attempting Ball Pivoting Algorithm ({num_points} pts)...")
        
        nn_k = min(30, num_points - 1)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=nn_k))
        
        tangent_k = min(100, num_points - 1)
        pcd.orient_normals_consistent_tangent_plane(tangent_k)
        
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances) if distances else 0.05
        avg_dist = max(avg_dist, 0.01) 
        
        radii = [avg_dist, avg_dist * 2.0, avg_dist * 4.0]
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
            
        if len(mesh.triangles) == 0:
             print("      BPA resulted in 0 faces. Falling back to Convex Hull...")
             mesh, _ = pcd.compute_convex_hull()
             
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        return mesh
        
    except Exception as e:
        print(f"      BPA Failed ({e}). Falling back to Convex Hull...")
        try:
            mesh, _ = pcd.compute_convex_hull()
            mesh.compute_vertex_normals()
            return mesh
        except:
            return None

def append_mesh_to_cityjson(cj_data, mesh, cluster_id, parent_building_id=None):
    """Converts an Open3D mesh to CityJSON geometry and appends it to the dataset."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    if len(triangles) == 0:
        return False
    
    # 1. Extract transform parameters to compress coordinates
    scale = cj_data.get("transform", {}).get("scale", [1.0, 1.0, 1.0])
    translate = cj_data.get("transform", {}).get("translate", [0.0, 0.0, 0.0])
    
    # 2. Track the starting index for our new vertices
    start_idx = len(cj_data["vertices"])
    
    # 3. Transform and append new vertices to the global array
    new_vertices = []
    for v in vertices:
        v_int = [
            int(round((v[0] - translate[0]) / scale[0])),
            int(round((v[1] - translate[1]) / scale[1])),
            int(round((v[2] - translate[2]) / scale[2]))
        ]
        new_vertices.append(v_int)
    cj_data["vertices"].extend(new_vertices)
    
    # 4. Generate the nested boundary arrays for a MultiSurface
    # A Triangle is a Polygon with 1 ring. MultiSurface is an array of Polygons.
    boundaries = []
    for tri in triangles:
        # CityJSON ring format: [v0, v1, v2] (No need to close the ring manually)
        ring = [
            int(tri[0]) + start_idx, 
            int(tri[1]) + start_idx, 
            int(tri[2]) + start_idx
        ]
        boundaries.append([ring])
        
    # 5. Create the BuildingInstallation CityObject
    inst_id = f"BuildingInstallation_Cluster_{cluster_id}"
    installation_obj = {
        "type": "BuildingInstallation",
        "attributes": {
            "description": f"Mesh generated via BPA/Hull from cluster {cluster_id}"
        },
        "geometry": [{
            "type": "MultiSurface",
            "lod": "2.2",
            "boundaries": boundaries
        }]
    }
    
    # 6. Link to parent building and add to CityObjects dictionary
    if parent_building_id:
        installation_obj["parents"] = [parent_building_id]
        parent_obj = cj_data["CityObjects"][parent_building_id]
        if "children" not in parent_obj:
            parent_obj["children"] = []
        parent_obj["children"].append(inst_id)
        
    cj_data["CityObjects"][inst_id] = installation_obj
    return True

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Append Convex Hull BuildingInstallations to a CityJSON file."
    )
    parser.add_argument("--las", metavar="PATH", help="Path to input .las point cloud file.")
    parser.add_argument("--json", metavar="PATH", help="Path to input .json CityJSON file.")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1 & 2. File Selection
    las_path = args.las
    json_path = args.json
    
    if not las_path or not json_path:
        las_path, json_path = select_files()
    
    # 3. Read LAS and run DBSCAN
    print("Reading point cloud...")
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    print(f"Running DBSCAN on {len(points)} points... (eps=0.5, min_samples=20)")
    clustering = DBSCAN(eps=0.5, min_samples=20).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels)
    valid_clusters = [lbl for lbl in unique_labels if lbl != -1]
    
    print(f"Found {len(valid_clusters)} valid clusters.")

    # Parse CityJSON file
    print(f"Parsing CityJSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        cj_data = json.load(f)
        
    if "CityObjects" not in cj_data:
        raise ValueError("Invalid CityJSON: 'CityObjects' missing.")

    # Find the primary Building to attach the installations to
    building_id = None
    for obj_id, obj in cj_data["CityObjects"].items():
        if obj.get("type") == "Building":
            building_id = obj_id
            break
            
    if building_id is None:
        print("Warning: No Building found in CityJSON. Installations will be created without parents.")

    # 4, 5, 6. Iterate clusters, Mesh, and Append
    appended_count = 0
    
    for label in valid_clusters:
        cluster_points = points[labels == label]
        print(f"\nProcessing Cluster {label} ({len(cluster_points)} points)...")
        
        # Generate Mesh
        mesh = create_mesh(cluster_points, max_points=20000)
        
        if mesh is None or len(mesh.triangles) == 0:
            print(f"  → Mesh generation failed or resulted in 0 faces. Skipping.")
            continue
            
        print(f"  → Successfully generated mesh with {len(mesh.triangles)} faces.")
        
        # Convert to CityJSON dictionaries
        success = append_mesh_to_cityjson(cj_data, mesh, label, parent_building_id=building_id)
        
        if success:
            appended_count += 1

    # 7. Save the modified CityJSON
    output_dir = "outputs/16_final_LOD3_json"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(json_path)
    output_filename = os.path.join(output_dir, base_name.replace(".json", "_meshed.city.json"))
    print(f"\nSaving modified CityJSON to: {output_filename}")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(cj_data, f, separators=(',', ':'))
        
    print(f"Success! Appended {appended_count} meshed clusters to the file.")

if __name__ == "__main__":
    main()