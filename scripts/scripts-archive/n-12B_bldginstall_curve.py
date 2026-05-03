import os
import numpy as np
import laspy
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
    """Handles the terminal selection for both LAS and GML files."""
    las_dir = "outputs/11A_facade_curve"
    gml_dir = "outputs/13_openings_gml"

    las_path = select_file_from_dir("Select the LAS file", las_dir, ".las")
    print(f"Selected: {las_path}")

    gml_path = select_file_from_dir("Select the GML file", gml_dir, ".gml")
    print(f"Selected: {gml_path}\n")

    return las_path, gml_path

def parse_wall_surfaces_from_gml(tree):
    """Extracts WallSurfaces from the GML and calculates Newell's normals and 2D origins."""
    root = tree.getroot()
    surfaces = []
    
    for ws in root.findall('.//bldg:WallSurface', namespaces=NSMAP):
        for poly in ws.findall('.//gml:Polygon', namespaces=NSMAP):
            pos_el = poly.find('.//gml:exterior//gml:posList', namespaces=NSMAP)
            if pos_el is None or not pos_el.text:
                continue

            vals = list(map(float, pos_el.text.split()))
            if len(vals) < 9:
                continue

            coords = np.array(vals, dtype=np.float64).reshape(-1, 3)
            if len(coords) < 3:
                continue

            # Newell's method for polygon normal
            n = np.zeros(3)
            for i in range(len(coords)):
                curr = coords[i]
                nxt  = coords[(i + 1) % len(coords)]
                n[0] += (curr[1] - nxt[1]) * (curr[2] + nxt[2])
                n[1] += (curr[2] - nxt[2]) * (curr[0] + nxt[0])
                n[2] += (curr[0] - nxt[0]) * (curr[1] + nxt[1])

            nh  = n[:2]
            mag = np.linalg.norm(nh)
            if mag < 1e-6:
                continue   # skip horizontal slabs

            normal_2d = nh / mag
            centroid  = coords.mean(axis=0)

            surfaces.append({
                'idx':       len(surfaces),
                'coords':    coords,
                'normal_2d': normal_2d,
                'origin_2d': centroid[:2].copy(),
                'z_min':     float(coords[:, 2].min()),
                'z_max':     float(coords[:, 2].max()),
                'xy_min':    coords[:, :2].min(axis=0),
                'xy_max':    coords[:, :2].max(axis=0),
            })
            
    return surfaces

def split_by_wall(points, wall_surfaces):
    """Assigns each point in the point cloud to the nearest WallSurface normal plane."""
    if not wall_surfaces:
        return []

    pts_xy   = points[:, :2]
    n_walls  = len(wall_surfaces)

    # Compute orthogonal distance from each point to each wall's 2D plane
    dist_matrix = np.empty((len(points), n_walls), dtype=np.float64)
    for j, ws in enumerate(wall_surfaces):
        dist_matrix[:, j] = (pts_xy - ws['origin_2d']) @ ws['normal_2d']

    # Assign point to the wall with the minimum absolute distance
    assignments = np.argmin(np.abs(dist_matrix), axis=1)

    sub_clusters = []
    for j, ws in enumerate(wall_surfaces):
        mask = assignments == j
        if not mask.any():
            continue
        sub_pts = points[mask]
        sub_clusters.append({
            'wall_idx': j,
            'points': sub_pts
        })

    return sub_clusters

def create_gml_polygon(coords):
    """Creates a gml:Polygon XML element from a list of 3D coordinates."""
    if not np.array_equal(coords[0], coords[-1]):
        coords.append(coords[0])
        
    pos_list_str = " ".join([f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in coords])
    
    polygon = etree.Element(f"{{{NSMAP['gml']}}}Polygon")
    exterior = etree.SubElement(polygon, f"{{{NSMAP['gml']}}}exterior")
    linear_ring = etree.SubElement(exterior, f"{{{NSMAP['gml']}}}LinearRing")
    pos_list = etree.SubElement(linear_ring, f"{{{NSMAP['gml']}}}posList", srsDimension="3")
    pos_list.text = pos_list_str
    
    return polygon

def create_building_installation(min_pt, max_pt, cluster_id, wall_idx):
    """Generates a bldg:BuildingInstallation containing the 3D bounding box."""
    x1, y1, z1 = min_pt
    x2, y2, z2 = max_pt
    
    v0, v1, v2, v3 = [x1,y1,z1], [x2,y1,z1], [x2,y2,z1], [x1,y2,z1] 
    v4, v5, v6, v7 = [x1,y1,z2], [x2,y1,z2], [x2,y2,z2], [x1,y2,z2] 
    
    faces = [
        [v3, v2, v1, v0], # Bottom
        [v4, v5, v6, v7], # Top
        [v0, v1, v5, v4], # Front
        [v2, v3, v7, v6], # Back
        [v3, v0, v4, v7], # Left
        [v1, v2, v6, v5]  # Right
    ]

    installation = etree.Element(f"{{{NSMAP['bldg']}}}BuildingInstallation")
    
    desc = etree.SubElement(installation, f"{{{NSMAP['gml']}}}description")
    desc.text = f"Slab generated from point cloud cluster {cluster_id} on Wall {wall_idx}"
    
    lod2_geom = etree.SubElement(installation, f"{{{NSMAP['bldg']}}}lod2Geometry")
    multi_surface = etree.SubElement(lod2_geom, f"{{{NSMAP['gml']}}}MultiSurface")
    
    for face in faces:
        surface_member = etree.SubElement(multi_surface, f"{{{NSMAP['gml']}}}surfaceMember")
        polygon = create_gml_polygon(face)
        surface_member.append(polygon)
        
    return installation

def main():
    las_path, gml_path = select_files()
    
    print("Reading point cloud...")
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    print(f"Parsing GML: {gml_path}")
    tree = etree.parse(gml_path)
    root_gml = tree.getroot()
    
    building = root_gml.find('.//bldg:Building', namespaces=NSMAP)
    if building is None:
        raise ValueError("No <bldg:Building> element found in the provided GML file.")

    print("Extracting WallSurfaces...")
    wall_surfaces = parse_wall_surfaces_from_gml(tree)
    print(f"Found {len(wall_surfaces)} valid WallSurfaces.")

    print("Splitting point cloud by nearest WallSurface...")
    sub_clusters = split_by_wall(points, wall_surfaces)
    print(f"Generated {len(sub_clusters)} wall-aligned point sub-clusters.")

    appended_count = 0
    
    # Process each wall's point cluster individually
    for sc in sub_clusters:
        sc_pts = sc['points']
        w_idx = sc['wall_idx']
        
        if len(sc_pts) < 20:
            print(f"Skipping Wall {w_idx}: Insufficient points ({len(sc_pts)}) for DBSCAN.")
            continue
            
        print(f"Running DBSCAN on Wall {w_idx} sub-cluster ({len(sc_pts)} points)...")
        clustering = DBSCAN(eps=0.5, min_samples=20).fit(sc_pts)
        labels = clustering.labels_
        unique_labels = set(labels)
        
        valid_clusters = [lbl for lbl in unique_labels if lbl != -1]
        print(f"  → Found {len(valid_clusters)} valid DBSCAN clusters.")

        for label in valid_clusters:
            cluster_points = sc_pts[labels == label]
            min_pt = cluster_points.min(axis=0)
            max_pt = cluster_points.max(axis=0)
            
            installation_xml = create_building_installation(min_pt, max_pt, label, w_idx)
            
            outer_installation = etree.Element(f"{{{NSMAP['bldg']}}}outerBuildingInstallation")
            outer_installation.append(installation_xml)
            building.append(outer_installation)
            
            appended_count += 1

    # Save the modified GML
    output_filename = gml_path.replace(".gml", "_with_curved_slabs.gml")
    print(f"\nSaving modified GML to: {output_filename}")
    
    tree.write(output_filename, pretty_print=True, xml_declaration=True, encoding="utf-8")
    print(f"Success! Appended {appended_count} bounding box slabs.")

if __name__ == "__main__":
    main()