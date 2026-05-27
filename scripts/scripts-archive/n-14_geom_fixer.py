import os
import numpy as np
import trimesh
from lxml import etree
from shapely.geometry import Polygon
from itertools import combinations

# ==========================================
# FILE PATHS
# ==========================================
INPUT_FILE = "outputs/13_openings_gml/Trial12_with_slabs_meshed.gml"
OUTPUT_DIR = "outputs/final"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Trial1.gml")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# namespaces typically found in CityGML (adjust based on your specific GML version)
NSMAP = {
    'core': 'http://www.opengis.net/citygml/2.0',
    'bldg': 'http://www.opengis.net/citygml/building/2.0',
    'gml': 'http://www.opengis.net/gml'
}

def extract_vertices_from_gml(gml_polygon):
    """
    Extracts a list of 3D coordinates from a GML Polygon node.
    (Placeholder: Requires adaptation to your specific GML coordinate formatting).
    """
    pos_list = gml_polygon.xpath('.//gml:posList', namespaces=NSMAP)
    if not pos_list:
        return []
    
    # Convert 'x y z x y z' string into an Nx3 numpy array
    coords = np.array([float(v) for v in pos_list[0].text.split()])
    return coords.reshape(-1, 3)

def resolve_planar_overlaps(features):
    """
    Task 1: Finds overlapping planar features (Windows, Doors, Walls)
    and removes the feature with the smaller area.
    """
    to_delete = set()
    
    # Compare every feature against every other feature
    for feat_a, feat_b in combinations(features, 2):
        if feat_a['id'] in to_delete or feat_b['id'] in to_delete:
            continue
            
        poly_a = feat_a['geometry'] # Shapely polygon
        poly_b = feat_b['geometry'] # Shapely polygon
        
        # Check if they intersect and share the same space
        if poly_a.intersects(poly_b):
            intersection_area = poly_a.intersection(poly_b).area
            
            # If the overlap is significant (not just touching edges)
            if intersection_area > 0.01: 
                # Delete the one with the smaller total area
                if poly_a.area < poly_b.area:
                    to_delete.add(feat_a['id'])
                else:
                    to_delete.add(feat_b['id'])
                    
    return [f for f in features if f['id'] not in to_delete]

def hollow_out_walls(walls, installations):
    """
    Task 2: Uses 3D Boolean operations to cut holes in walls where 
    Building Installations intersect them.
    """
    modified_walls = []
    
    for wall in walls:
        wall_mesh = wall['mesh'] # Trimesh object
        
        for inst in installations:
            inst_mesh = inst['mesh'] # Trimesh object
            
            # Check for bounding box intersection first to save computation
            if trimesh.collision.intersects_bbox(wall_mesh, inst_mesh):
                
                # Perform Boolean Difference: Wall - Installation
                # This turns the intersecting part of the installation into a hole in the wall
                try:
                    # Note: engine='blender' or 'scad' is heavily recommended for robust booleans in trimesh
                    wall_mesh = trimesh.boolean.difference([wall_mesh, inst_mesh], engine='blender')
                except Exception as e:
                    print(f"Boolean operation failed on wall {wall['id']}: {e}")
                    
        wall['mesh'] = wall_mesh
        modified_walls.append(wall)
        
    return modified_walls

def process_citygml(input_path, output_path):
    print(f"Parsing {input_path}...")
    tree = etree.parse(input_path)
    root = tree.getroot()

    # ---------------------------------------------------------
    # 1. Gather Features (Abstract logic to be adapted to your XML)
    # ---------------------------------------------------------
    # You will need to extract your Windows, Doors, and Walls here,
    # convert their GML coordinates to Shapely Polygons (for Task 1)
    # and Trimesh Objects (for Task 2).
    
    planar_features = []       # Populate with dicts: {'id': element, 'geometry': Shapely Polygon}
    walls = []                 # Populate with dicts: {'id': element, 'mesh': Trimesh object}
    building_installations = [] # Populate with dicts: {'id': element, 'mesh': Trimesh object}

    print("Resolving planar overlaps (Windows/Doors/Walls)...")
    # resolved_planar = resolve_planar_overlaps(planar_features)
    
    # Remove the deleted planar features from the XML tree
    # for feature in planar_features:
    #     if feature not in resolved_planar:
    #         feature['element'].getparent().remove(feature['element'])

    print("Processing volumetric installations (hollowing walls)...")
    # modified_walls = hollow_out_walls(walls, building_installations)
    
    # ---------------------------------------------------------
    # 2. Reconstruct Geometry in XML
    # ---------------------------------------------------------
    # Once `modified_walls` are calculated, you must extract the vertices 
    # from the new Trimesh objects and overwrite the <gml:posList> nodes 
    # in the XML tree so the new geometry is saved.

    print(f"Saving resolved GML to {output_path}...")
    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding='utf-8')
    print("Process complete!")

if __name__ == "__main__":
    if os.path.exists(INPUT_FILE):
        process_citygml(INPUT_FILE, OUTPUT_FILE)
    else:
        print(f"Error: Could not find input file at {INPUT_FILE}")