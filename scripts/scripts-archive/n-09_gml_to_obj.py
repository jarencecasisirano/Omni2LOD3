#!/usr/bin/env python3
"""
Convert CityGML (.gml) files to Wavefront OBJ (.obj).

Reads all .gml files from outputs/00_gml_wall_merged, extracts every
gml:Polygon (GroundSurface, RoofSurface, WallSurface), and writes
triangulated meshes as .obj files to outputs/01_lod2_obj.

Usage:
    conda activate lidar-test
    python scripts/LOD2toLOD3/09_gml_to_obj.py
"""

import os
import sys
import glob
import re
from xml.etree import ElementTree as ET


# =====================================================================
# Namespace map for CityGML
# =====================================================================
NS = {
    'gml':  'http://www.opengis.net/gml',
    'bldg': 'http://www.opengis.net/citygml/building/2.0',
    'core': 'http://www.opengis.net/citygml/2.0',
}

INPUT_DIR  = "outputs/00_gml_wall_merged"
OUTPUT_DIR = "outputs/01_lod2_obj"


# =====================================================================
# Parse gml:posList into list of (x, y, z) tuples
# =====================================================================
def parse_poslist(text, srs_dim=3):
    """Parse a gml:posList string into a list of coordinate tuples."""
    vals = [float(v) for v in text.strip().split()]
    coords = []
    for i in range(0, len(vals), srs_dim):
        coords.append(tuple(vals[i:i+srs_dim]))
    # Remove duplicate closing vertex if polygon is closed
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return coords


# =====================================================================
# Extract all polygons from a GML file
# =====================================================================
def extract_polygons(gml_path):
    """
    Parse a CityGML file and return a list of polygons.
    Each polygon is a list of (x, y, z) coordinate tuples.
    """
    tree = ET.parse(gml_path)
    root = tree.getroot()

    polygons = []

    # Find ALL gml:Polygon elements regardless of nesting
    for poly_elem in root.iter('{http://www.opengis.net/gml}Polygon'):
        # Look for exterior ring
        exterior = poly_elem.find(
            'gml:exterior/gml:LinearRing/gml:posList', NS)
        if exterior is None:
            # Try without namespace prefix (some files use default ns)
            exterior = poly_elem.find(
                '{http://www.opengis.net/gml}exterior/'
                '{http://www.opengis.net/gml}LinearRing/'
                '{http://www.opengis.net/gml}posList')
        if exterior is not None and exterior.text:
            coords = parse_poslist(exterior.text)
            if len(coords) >= 3:
                polygons.append(coords)

    return polygons


# =====================================================================
# Fan triangulation for convex/near-convex polygons
# =====================================================================
def triangulate_fan(polygon):
    """
    Simple fan triangulation from vertex 0.
    Returns list of (i, j, k) index triples (0-based within the polygon).
    """
    tris = []
    for i in range(1, len(polygon) - 1):
        tris.append((0, i, i + 1))
    return tris


# =====================================================================
# Write OBJ file
# =====================================================================
def write_obj(polygons, output_path):
    """Write polygons to a Wavefront OBJ file with triangulation."""
    vertex_offset = 0
    vertices = []
    faces = []

    for poly in polygons:
        # Add vertices
        for x, y, z in poly:
            vertices.append((x, y, z))

        # Triangulate and add faces (OBJ is 1-indexed)
        tris = triangulate_fan(poly)
        for i, j, k in tris:
            faces.append((
                vertex_offset + i + 1,
                vertex_offset + j + 1,
                vertex_offset + k + 1
            ))
        vertex_offset += len(poly)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(f"# CityGML to OBJ conversion\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n\n")

        for x, y, z in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

        f.write("\n")

        for i, j, k in faces:
            f.write(f"f {i} {j} {k}\n")

    return len(vertices), len(faces)


# =====================================================================
# Main
# =====================================================================
def main():
    # Find GML files
    pattern = os.path.join(INPUT_DIR, "*.gml")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No .gml files found in {INPUT_DIR}")
        sys.exit(1)

    print(f"Found {len(files)} GML file(s) in {INPUT_DIR}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for gml_path in files:
        filename = os.path.basename(gml_path)
        name_no_ext = os.path.splitext(filename)[0]
        print(f"\n  Processing: {filename}")

        # Extract polygons
        polygons = extract_polygons(gml_path)
        print(f"    Polygons found: {len(polygons)}")

        if not polygons:
            print(f"    WARNING: No polygons found, skipping.")
            continue

        # Count total vertices
        total_verts = sum(len(p) for p in polygons)
        print(f"    Total vertices: {total_verts}")

        # Write OBJ
        obj_name = f"{name_no_ext}.obj"
        obj_path = os.path.join(OUTPUT_DIR, obj_name)
        n_verts, n_faces = write_obj(polygons, obj_path)
        print(f"    Output: {obj_path}")
        print(f"    OBJ stats: {n_verts} vertices, {n_faces} triangles")

    print(f"\n{'='*60}")
    print(f"Done! OBJ files saved to: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
