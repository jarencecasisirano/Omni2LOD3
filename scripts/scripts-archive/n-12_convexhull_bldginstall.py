import argparse
import os
import numpy as np
import laspy
import open3d as o3d
from sklearn.cluster import DBSCAN
from lxml import etree

# CityGML 2.0 Namespaces
NSMAP = {
    'core': 'http://www.opengis.net/citygml/2.0',
    'bldg': 'http://www.opengis.net/citygml/building/2.0',
    'gml': 'http://www.opengis.net/gml'
}

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
    """Handles terminal selection for LAS and GML files."""
    las_dir = "outputs/10_facade_features_cleaned"
    gml_dir = "outputs/13_openings_gml"

    las_path = select_file_from_dir("Select the LAS file", las_dir, ".las")
    print(f"Selected: {las_path}")

    gml_path = select_file_from_dir("Select the GML file", gml_dir, ".gml")
    print(f"Selected: {gml_path}\n")

    return las_path, gml_path

def create_convex_hull_mesh(points, max_points=5000):
    """Creates a watertight mesh using strictly the Convex Hull algorithm."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    num_points = len(pcd.points)

    # 1. Downsample if too large 
    # (Convex hull handles large sets well, but this keeps processing snappy)
    if num_points > max_points:
        ratio = max_points / num_points
        print(f"      Downsampling from {num_points} to ~{max_points} points...")
        pcd = pcd.random_down_sample(ratio)
        num_points = len(pcd.points)
        
    # 2. Check for minimum required points for a 3D volume
    if num_points < 4:
        print("      Not enough points for a 3D convex hull (< 4). Skipping.")
        return None
        
    try:
        print(f"      Generating Convex Hull ({num_points} pts)...")
        # compute_convex_hull returns a tuple: (TriangleMesh, list of point indices)
        mesh, _ = pcd.compute_convex_hull()
        mesh.compute_vertex_normals()
            
        # Clean up any potential artifacts
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        return mesh
        
    except Exception as e:
        print(f"      Convex Hull computation failed: {e}")
        return None

def create_gml_from_mesh(mesh, cluster_id):
    """Generates a bldg:BuildingInstallation containing the mesh triangles."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    if len(triangles) == 0:
        return None
        
    installation = etree.Element(f"{{{NSMAP['bldg']}}}BuildingInstallation")
    
    desc = etree.SubElement(installation, f"{{{NSMAP['gml']}}}description")
    desc.text = f"Mesh generated via Convex Hull from cluster {cluster_id} ({len(triangles)} faces)"
    
    lod2_geom = etree.SubElement(installation, f"{{{NSMAP['bldg']}}}lod2Geometry")
    multi_surface = etree.SubElement(lod2_geom, f"{{{NSMAP['gml']}}}MultiSurface")
    
    # Convert every triangle into a GML polygon
    for tri in triangles:
        v0, v1, v2 = vertices[tri]
        # Close the ring by appending the first vertex at the end
        coords = [v0, v1, v2, v0] 
        
        surface_member = etree.SubElement(multi_surface, f"{{{NSMAP['gml']}}}surfaceMember")
        polygon = etree.Element(f"{{{NSMAP['gml']}}}Polygon")
        exterior = etree.SubElement(polygon, f"{{{NSMAP['gml']}}}exterior")
        linear_ring = etree.SubElement(exterior, f"{{{NSMAP['gml']}}}LinearRing")
        pos_list = etree.SubElement(linear_ring, f"{{{NSMAP['gml']}}}posList", srsDimension="3")
        
        pos_list_str = " ".join([f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in coords])
        pos_list.text = pos_list_str
        
        surface_member.append(polygon)
        
    return installation

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Append Convex Hull BuildingInstallations to a CityGML file."
    )
    parser.add_argument(
        "--las",
        metavar="PATH",
        help="Path to the input .las point cloud file. "
             "If omitted, an interactive prompt is shown.",
    )
    parser.add_argument(
        "--gml",
        metavar="PATH",
        help="Path to the input .gml CityGML file. "
             "If omitted, an interactive prompt is shown.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Path for the output .gml file. "
             "Defaults to <input_gml>_convex_hulls.gml when omitted.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1 & 2. File Selection (argparse takes priority; fall back to interactive)
    if args.las:
        if not os.path.isfile(args.las):
            print(f"Error: LAS file not found: {args.las}")
            exit(1)
        las_path = args.las
        print(f"LAS file (from argument): {las_path}")
    else:
        las_path = None  # resolved below via select_files

    if args.gml:
        if not os.path.isfile(args.gml):
            print(f"Error: GML file not found: {args.gml}")
            exit(1)
        gml_path = args.gml
        print(f"GML file (from argument): {gml_path}")
    else:
        gml_path = None  # resolved below via select_files

    if las_path is None or gml_path is None:
        sel_las, sel_gml = select_files()
        if las_path is None:
            las_path = sel_las
        if gml_path is None:
            gml_path = sel_gml
    
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

    # Parse GML file
    print(f"Parsing GML: {gml_path}")
    tree = etree.parse(gml_path)
    root_gml = tree.getroot()
    
    building = root_gml.find('.//bldg:Building', namespaces=NSMAP)
    if building is None:
        raise ValueError("No <bldg:Building> element found in the provided GML file.")

    # 4, 5, 6. Iterate clusters, Mesh, and Append
    appended_count = 0
    total_faces = 0
    
    for label in valid_clusters:
        cluster_points = points[labels == label]
        print(f"\nProcessing Cluster {label} ({len(cluster_points)} points)...")
        
        # Generate Mesh using only Convex Hull
        mesh = create_convex_hull_mesh(cluster_points, max_points=5000)
        
        if mesh is None or len(mesh.triangles) == 0:
            print(f"  → Mesh generation failed or resulted in 0 faces. Skipping.")
            continue
            
        print(f"  → Successfully generated convex hull with {len(mesh.triangles)} faces.")
        
        # Convert to GML
        installation_xml = create_gml_from_mesh(mesh, label)
        
        if installation_xml is not None:
            outer_installation = etree.Element(f"{{{NSMAP['bldg']}}}outerBuildingInstallation")
            outer_installation.append(installation_xml)
            building.append(outer_installation)
            
            appended_count += 1
            total_faces += len(mesh.triangles)

    # 7. Save the modified GML
    if args.output:
        output_filename = args.output
    else:
        output_filename = gml_path.replace(".gml", "_convex_hulls.gml")
    print(f"\nSaving modified GML to: {output_filename}")
    
    tree.write(output_filename, pretty_print=True, xml_declaration=True, encoding="utf-8")
    print(f"Success! Appended {appended_count} convex hull clusters ({total_faces} total polygons) to the file.")

if __name__ == "__main__":
    main()