#!/usr/bin/env python3
"""
Convert CityGML (.gml) files to LAS point clouds.

Reads all .gml files from outputs/00_gml_wall_merged, extracts every
surface polygon (GroundSurface, RoofSurface, WallSurface), samples
points on the triangulated polygons, and writes one .las per input
file into outputs/08_lod2_points.
"""

import os
import glob
import numpy as np
import open3d as o3d
import laspy
from lxml import etree

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_DIR  = 'outputs/00_gml_wall_merged'
OUTPUT_DIR = 'outputs/08_lod2_points'
DENSITY    = 0.05   # metres — approx spacing between sampled points

NAMESPACES = {
    'gml':  'http://www.opengis.net/gml',
    'bldg': 'http://www.opengis.net/citygml/building/2.0',
    'core': 'http://www.opengis.net/citygml/2.0',
}

# Surface types to extract
SURFACE_TYPES = [
    'bldg:WallSurface',
    'bldg:RoofSurface',
    'bldg:GroundSurface',
]


# ---------------------------------------------------------------------------
# GML parsing
# ---------------------------------------------------------------------------

def parse_all_surfaces(gml_file: str):
    """
    Parse a CityGML file and return a list of (surface_type, polygon_id, Nx3 coords)
    for every polygon found in any of the SURFACE_TYPES.
    """
    root = etree.parse(gml_file).getroot()
    surfaces = []

    for stype in SURFACE_TYPES:
        for surf_elem in root.xpath(f'//{stype}', namespaces=NAMESPACES):
            for polygon in surf_elem.xpath('.//gml:Polygon', namespaces=NAMESPACES):
                poly_id = polygon.get('{http://www.opengis.net/gml}id', 'unknown')
                for pos_list in polygon.xpath('.//gml:posList', namespaces=NAMESPACES):
                    text = (pos_list.text or '').strip()
                    if not text:
                        continue
                    coords = np.array(list(map(float, text.split())),
                                      dtype=np.float64).reshape(-1, 3)
                    if len(coords) >= 3:
                        surfaces.append((stype, poly_id, coords))

    return surfaces


# ---------------------------------------------------------------------------
# Point sampling
# ---------------------------------------------------------------------------

def sample_polygon(coords: np.ndarray, density: float = DENSITY) -> np.ndarray:
    """
    Fan-triangulate a polygon and uniformly sample points on it.
    Returns an Nx3 numpy array of sampled XYZ points.
    """
    n = len(coords)
    if n < 3:
        return np.empty((0, 3), dtype=np.float64)

    # Fan triangulation from vertex 0
    triangles = [[0, i, i + 1] for i in range(1, n - 1)]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(coords)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))

    area  = mesh.get_surface_area()
    n_pts = max(100, int(area / (density ** 2)))

    pcd = mesh.sample_points_uniformly(number_of_points=n_pts)
    return np.asarray(pcd.points)


# ---------------------------------------------------------------------------
# LAS writing
# ---------------------------------------------------------------------------

def write_las(points: np.ndarray, output_path: str):
    """
    Write an Nx3 array of XYZ points to a LAS 1.4, point format 0 file.
    Coordinates are stored with millimetre precision.
    """
    header = laspy.LasHeader(point_format=0, version="1.4")
    header.scales  = [0.001, 0.001, 0.001]
    header.offsets = [
        np.floor(points[:, 0].min()),
        np.floor(points[:, 1].min()),
        np.floor(points[:, 2].min()),
    ]

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.write(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    gml_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.gml')))
    if not gml_files:
        print(f"No .gml files found in {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Found {len(gml_files)} GML file(s) in {INPUT_DIR}\n")

    for gml_path in gml_files:
        basename = os.path.splitext(os.path.basename(gml_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{basename}.las")

        print(f"Processing: {os.path.basename(gml_path)}")
        surfaces = parse_all_surfaces(gml_path)
        print(f"  Extracted {len(surfaces)} polygon(s)")

        # Collect sampled points from every polygon
        all_points = []
        counts = {}
        for stype, poly_id, coords in surfaces:
            pts = sample_polygon(coords)
            all_points.append(pts)
            label = stype.split(':')[1]
            counts[label] = counts.get(label, 0) + 1

        for label, cnt in sorted(counts.items()):
            print(f"    {label}: {cnt} polygon(s)")

        if not all_points:
            print("  ⚠  No points sampled — skipping")
            continue

        merged = np.vstack(all_points)
        print(f"  Total points: {len(merged):,}")
        print(f"  X range: {merged[:,0].min():.3f} – {merged[:,0].max():.3f}")
        print(f"  Y range: {merged[:,1].min():.3f} – {merged[:,1].max():.3f}")
        print(f"  Z range: {merged[:,2].min():.3f} – {merged[:,2].max():.3f}")

        write_las(merged, out_path)
        print(f"  ✓ Saved: {out_path}\n")

    print("Done.")


if __name__ == '__main__':
    main()
