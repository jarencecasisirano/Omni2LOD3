#!/usr/bin/env python3
"""
Convert CityGML (.gml) RoofSurfaces + upper WallSurfaces to LAS point clouds.

For each .gml file in outputs/00_gml_wall_merged:
  1. Sample the *entire* RoofSurface polygons.
  2. Find WallSurface polygons that share vertices with RoofSurfaces.
  3. Clip those wall polygons so that only the top 3 metres (below the
     shared roof edge) are kept, using the Sutherland-Hodgman algorithm
     against a horizontal Z-threshold plane.
  4. Write the combined point cloud to outputs/08_roof_wall_points.
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
INPUT_DIR       = 'outputs/00_gml_wall_merged'
OUTPUT_DIR      = 'outputs/08_roof_wall_points'
DENSITY         = 0.05    # metres — approx spacing between sampled points
WALL_DROP       = 1.0     # metres below roof edge to keep
VERTEX_TOL_XY   = 0.01    # metres — XY tolerance for shared-vertex matching

NAMESPACES = {
    'gml':  'http://www.opengis.net/gml',
    'bldg': 'http://www.opengis.net/citygml/building/2.0',
    'core': 'http://www.opengis.net/citygml/2.0',
}


# ---------------------------------------------------------------------------
# GML parsing helpers
# ---------------------------------------------------------------------------

def _extract_polygons(root, surface_xpath: str):
    """Return list of (polygon_id, Nx3 coords) for all Polygons under *surface_xpath*."""
    polys = []
    for surf_elem in root.xpath(f'//{surface_xpath}', namespaces=NAMESPACES):
        for polygon in surf_elem.xpath('.//gml:Polygon', namespaces=NAMESPACES):
            poly_id = polygon.get('{http://www.opengis.net/gml}id', 'unknown')
            for pos_list in polygon.xpath('.//gml:posList', namespaces=NAMESPACES):
                text = (pos_list.text or '').strip()
                if not text:
                    continue
                coords = np.array(list(map(float, text.split())),
                                  dtype=np.float64).reshape(-1, 3)
                if len(coords) >= 3:
                    polys.append((poly_id, coords))
    return polys


def parse_surfaces(gml_file: str):
    """Return (roof_polys, wall_polys) each as list of (id, Nx3 coords)."""
    root = etree.parse(gml_file).getroot()
    roofs = _extract_polygons(root, 'bldg:RoofSurface')
    walls = _extract_polygons(root, 'bldg:WallSurface')
    return roofs, walls


# ---------------------------------------------------------------------------
# Vertex matching
# ---------------------------------------------------------------------------

def _build_roof_vertex_set(roof_polys, tol=VERTEX_TOL_XY):
    """
    Build a set of quantised (X, Y) tuples from all roof polygon vertices
    so we can quickly test whether a wall vertex touches a roof.
    """
    q = 1.0 / tol  # quantisation factor
    roof_xy = set()
    for _, coords in roof_polys:
        for pt in coords:
            roof_xy.add((round(pt[0] * q), round(pt[1] * q)))
    return roof_xy, q


def wall_shares_vertices_with_roof(wall_coords, roof_xy_set, q):
    """
    Return the list of *wall* vertex indices that also appear in
    the roof vertex set.  Empty list → no match.
    """
    shared = []
    for i, pt in enumerate(wall_coords):
        key = (round(pt[0] * q), round(pt[1] * q))
        if key in roof_xy_set:
            shared.append(i)
    return shared


# ---------------------------------------------------------------------------
# Polygon Z-clipping (Sutherland-Hodgman, keep z >= z_min)
# ---------------------------------------------------------------------------

def clip_polygon_z_min(coords: np.ndarray, z_min: float) -> np.ndarray:
    """
    Clip a 3-D polygon against the half-space z >= z_min.
    Returns the clipped polygon vertices (may be empty).
    """
    output = list(coords)
    if len(output) == 0:
        return np.empty((0, 3), dtype=np.float64)

    clipped = []
    n = len(output)
    for i in range(n):
        curr = output[i]
        nxt  = output[(i + 1) % n]
        c_in = curr[2] >= z_min
        n_in = nxt[2]  >= z_min

        if c_in and n_in:
            clipped.append(nxt)
        elif c_in and not n_in:
            # exiting
            t = (z_min - curr[2]) / (nxt[2] - curr[2]) if nxt[2] != curr[2] else 0.0
            inter = curr + t * (nxt - curr)
            clipped.append(inter)
        elif not c_in and n_in:
            # entering
            t = (z_min - curr[2]) / (nxt[2] - curr[2]) if nxt[2] != curr[2] else 0.0
            inter = curr + t * (nxt - curr)
            clipped.append(inter)
            clipped.append(nxt)
        # else both outside → skip

    if len(clipped) < 3:
        return np.empty((0, 3), dtype=np.float64)
    return np.array(clipped, dtype=np.float64)


# ---------------------------------------------------------------------------
# Point sampling
# ---------------------------------------------------------------------------

def sample_polygon(coords: np.ndarray, density: float = DENSITY) -> np.ndarray:
    """Fan-triangulate a polygon and uniformly sample points."""
    n = len(coords)
    if n < 3:
        return np.empty((0, 3), dtype=np.float64)

    triangles = [[0, i, i + 1] for i in range(1, n - 1)]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(coords)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))

    area  = mesh.get_surface_area()
    n_pts = max(100, int(area / (density ** 2)))
    pcd   = mesh.sample_points_uniformly(number_of_points=n_pts)
    return np.asarray(pcd.points)


# ---------------------------------------------------------------------------
# LAS writing
# ---------------------------------------------------------------------------

def write_las(points: np.ndarray, output_path: str):
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
        roof_polys, wall_polys = parse_surfaces(gml_path)
        print(f"  Roofs: {len(roof_polys)} polygon(s),  Walls: {len(wall_polys)} polygon(s)")

        # ---- 1. Sample full roof surfaces --------------------------------
        all_points = []
        for _, coords in roof_polys:
            pts = sample_polygon(coords)
            if len(pts):
                all_points.append(pts)
        print(f"  Roof points sampled: {sum(len(p) for p in all_points):,}")

        # ---- 2. Build roof vertex lookup ---------------------------------
        roof_xy, q = _build_roof_vertex_set(roof_polys)

        # ---- 3. Find walls sharing vertices with roofs, clip & sample ----
        matched_walls  = 0
        wall_pts_total = 0
        for wall_id, wall_coords in wall_polys:
            shared_idx = wall_shares_vertices_with_roof(wall_coords, roof_xy, q)
            if not shared_idx:
                continue
            matched_walls += 1

            # Determine the Z-threshold: highest shared vertex minus WALL_DROP
            shared_z = wall_coords[shared_idx, 2]
            z_min = shared_z.max() - WALL_DROP

            # Clip the wall polygon to keep only z >= z_min
            clipped = clip_polygon_z_min(wall_coords, z_min)
            if len(clipped) < 3:
                continue

            pts = sample_polygon(clipped)
            if len(pts):
                all_points.append(pts)
                wall_pts_total += len(pts)

        print(f"  Matched walls: {matched_walls}  →  wall points: {wall_pts_total:,}")

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
