#!/usr/bin/env python3
"""
CityJSON LOD3: Facade Recesses.

Takes a CityJSON LOD3 file and a .las point cloud of facade features
from outputs/11B_flat.

Pipeline
--------
1. Extract the building 2-D XY footprint (GroundSurface or convex hull).
2. Parse all near-vertical wall surfaces for normal/tangent alignment.
3. DBSCAN-cluster the point cloud.
4. For each cluster:
      a. Filter points to those lying INSIDE the building footprint (XY test).
      b. If interior points are insufficient, skip.
      c. Find the nearest wall surface for axis alignment.
      d. Build a wall-aligned 3-D bounding box:
           • Opening face (near)  – snapped flush to the wall plane.
           • Back face   (far)    – deepest interior point along −normal.
           • Width and height     – tangent/Z extents of interior points.
      e. Punch a rectangular hole in the matched wall polygon (the opening).
      f. Create a BuildingInstallation child with the 5 interior cavity
         faces: back wall + floor + ceiling + left side + right side.
5. Save the enriched CityJSON.

Face winding convention (outward normals point INTO the cavity):
   Back wall : normal ≈ +wall_normal  (faces the opening)
   Floor     : normal ≈ +Z            (faces up into cavity)
   Ceiling   : normal ≈ -Z            (faces down into cavity)
   Left side : normal ≈ +tangent      (faces right across cavity)
   Right side: normal ≈ -tangent      (faces left across cavity)

Usage
-----
    conda activate las-env
    python scripts/LOD2toLOD3/11_bldginstallation_flat.py [-o OUTPUT]

Options
-------
    --eps              DBSCAN neighbourhood radius  (default 0.3 m)
    --min_samples      DBSCAN minimum cluster size  (default 30)
    --min_interior     Minimum interior points to accept a cluster (default 10)
    --interior_frac    Minimum interior fraction to accept a cluster (default 0.3)
    --output, -o       Output filename (in outputs/13_openings_json/).
"""

import os
import sys
import glob
import json
import uuid
import argparse

import numpy as np
import laspy
from sklearn.cluster import DBSCAN

try:
    from shapely.geometry import MultiPoint, Polygon as SPolygon, Point
    from shapely.prepared import prep
    from shapely.validation import make_valid
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    print("WARNING: shapely not installed — interior-point filtering is disabled.\n"
          "  Install: conda install -c conda-forge shapely")


# ─── Directories ──────────────────────────────────────────────────────────────
LAS_DIR    = "outputs/11B_flat"
JSON_DIR   = "outputs/13_openings_json"
OUTPUT_DIR = "outputs/13_openings_json"

# ─── Tuning constants ─────────────────────────────────────────────────────────
VERTICAL_TOL  = 0.3    # |n_z / |n|| > this → near-horizontal, skip
HOLE_PAD      = 0.005  # m – inset applied to hole ring in wall polygon


# =====================================================================
# Interactive file selector
# =====================================================================
def select_file(directory, pattern="*.las"):
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        print(f"  No {pattern} files found in {directory}")
        sys.exit(1)
    print(f"\n{'='*60}")
    print(f"  Files in: {directory}")
    print(f"{'='*60}")
    for i, f in enumerate(files):
        mb = os.path.getsize(f) / 1_048_576
        print(f"  [{i+1}] {os.path.basename(f):50s} ({mb:.1f} MB)")
    print()
    while True:
        try:
            c = int(input(f"  Select file [1-{len(files)}]: ").strip())
            if 1 <= c <= len(files):
                return files[c - 1]
        except (ValueError, KeyboardInterrupt):
            pass
        print("  Invalid choice, try again.")


# =====================================================================
# CityJSON vertex helpers
# =====================================================================
def decode_vertices(cm):
    raw       = np.array(cm["vertices"], dtype=np.float64)
    t         = cm.get("transform", {})
    scale     = np.array(t.get("scale",     [1, 1, 1]), dtype=np.float64)
    translate = np.array(t.get("translate", [0, 0, 0]), dtype=np.float64)
    return raw * scale + translate


def encode_vertex(pt, scale, translate):
    return [round((pt[i] - translate[i]) / scale[i]) for i in range(3)]


# =====================================================================
# Building footprint extraction
# =====================================================================
_SKIP_TYPES = {"Window", "Door", "BuildingInstallation", "OtherConstruction"}


def _flatten_indices(x):
    if isinstance(x, list):
        for y in x:
            yield from _flatten_indices(y)
    elif isinstance(x, int):
        yield x


def extract_building_footprint(cm, world_verts):
    """
    Return a shapely Polygon for the building XY footprint.
    Priority: GroundSurface semantic poly → XY convex hull fallback.
    """
    if not HAS_SHAPELY:
        return None

    for obj_id, obj in cm.get("CityObjects", {}).items():
        if obj.get("type", "") in _SKIP_TYPES:
            continue
        for geom in obj.get("geometry", []):
            sem          = geom.get("semantics", {})
            sem_surfaces = sem.get("surfaces", [])
            sem_values   = sem.get("values",   [])
            boundaries   = geom.get("boundaries", [])
            geom_type    = geom.get("type", "")

            if geom_type not in ("Solid", "MultiSurface"):
                continue

            flat = []
            if geom_type == "Solid":
                offset = 0
                for shell in boundaries:
                    for p_idx, polygon in enumerate(shell):
                        flat.append((offset + p_idx, polygon))
                    offset += len(shell)
            else:
                for p_idx, polygon in enumerate(boundaries):
                    flat.append((p_idx, polygon))

            for global_p, polygon in flat:
                try:
                    sem_type = sem_surfaces[sem_values[global_p]].get("type", "")
                except (IndexError, KeyError, TypeError):
                    sem_type = ""
                if sem_type != "GroundSurface":
                    continue
                try:
                    coords = world_verts[np.array(polygon[0])]
                    xy     = [(float(c[0]), float(c[1])) for c in coords]
                    poly   = make_valid(SPolygon(xy))
                    if not poly.is_empty and poly.area > 0:
                        print(f"  Footprint from GroundSurface: {poly.area:.1f} m²")
                        return poly
                except Exception:
                    pass

    # Fallback: convex hull of all building vertices
    all_xy = []
    for obj_id, obj in cm.get("CityObjects", {}).items():
        if obj.get("type", "") in _SKIP_TYPES:
            continue
        for geom in obj.get("geometry", []):
            for vi in set(_flatten_indices(geom.get("boundaries", []))):
                if 0 <= vi < len(world_verts):
                    all_xy.append((float(world_verts[vi, 0]),
                                   float(world_verts[vi, 1])))
    if len(all_xy) >= 3:
        hull = make_valid(MultiPoint(all_xy).convex_hull)
        if not hull.is_empty:
            print(f"  Footprint from convex hull: {hull.area:.1f} m²")
            return hull

    print("  WARNING: could not determine building footprint.")
    return None


# =====================================================================
# Parse vertical wall surfaces
# =====================================================================
def parse_vertical_surfaces(cm, world_verts):
    """
    Collect all near-vertical polygons from the highest-LOD building
    geometry, annotated with their outward 2-D normal and wall-plane depth.
    """
    surfaces = []

    for obj_id, obj in cm.get("CityObjects", {}).items():
        if obj.get("type", "") in _SKIP_TYPES:
            continue
        geoms = obj.get("geometry", [])
        if not geoms:
            continue

        def _lod(g):
            try:
                return float(g.get("lod") or 0)
            except (ValueError, TypeError):
                return 0.0

        best_idx = max(range(len(geoms)), key=lambda i: _lod(geoms[i]))
        g_idx    = best_idx
        geom     = geoms[best_idx]

        boundaries = geom.get("boundaries", [])
        geom_type  = geom.get("type", "")

        shells = (enumerate(boundaries) if geom_type == "Solid"
                  else [(0, boundaries)])

        for s_idx, shell in shells:
            for p_idx, polygon in enumerate(shell):
                try:
                    coords = world_verts[np.array(polygon[0])]
                except (IndexError, KeyError, TypeError):
                    continue
                if len(coords) < 3:
                    continue

                # Newell normal
                n = np.zeros(3)
                for i in range(len(coords)):
                    c  = coords[i]
                    nx = coords[(i + 1) % len(coords)]
                    n[0] += (c[1] - nx[1]) * (c[2] + nx[2])
                    n[1] += (c[2] - nx[2]) * (c[0] + nx[0])
                    n[2] += (c[0] - nx[0]) * (c[1] + nx[1])

                n_len = np.linalg.norm(n)
                if n_len < 1e-9:
                    continue
                n_unit = n / n_len
                if abs(n_unit[2]) > VERTICAL_TOL:
                    continue   # near-horizontal → skip

                nh  = n[:2]
                mag = np.linalg.norm(nh)
                if mag < 1e-6:
                    continue

                normal_2d = nh / mag
                centroid  = coords.mean(axis=0)
                wall_d    = float(np.dot(centroid[:2], normal_2d))

                surfaces.append({
                    "idx":       len(surfaces),
                    "obj_id":    obj_id,
                    "geom_idx":  g_idx,
                    "shell_idx": s_idx,
                    "poly_idx":  p_idx,
                    "coords":    coords,
                    "normal_2d": normal_2d,
                    "origin_2d": centroid[:2].copy(),
                    "wall_d":    wall_d,
                    "z_min":     float(coords[:, 2].min()),
                    "z_max":     float(coords[:, 2].max()),
                    "xy_min":    coords[:, :2].min(axis=0),
                    "xy_max":    coords[:, :2].max(axis=0),
                })

    print(f"  Parsed {len(surfaces)} vertical surface(s).")
    return surfaces


# =====================================================================
# 3-stage wall surface matching
# =====================================================================
def _pca_normal_2d(points):
    if len(points) < 3:
        return None
    xy  = points[:, :2]
    cxy = xy - xy.mean(axis=0)
    cov = cxy.T @ cxy
    _, vecs = np.linalg.eigh(cov)
    tang = vecs[:, 1]
    return np.array([-tang[1], tang[0]])


def find_nearest_surface(points, vert_surfaces, dist_tol=2.0, z_expand=0.5):
    """
    Match a cluster to a wall surface via 3 stages:
      1. Spatial intersection (Z overlap + signed distance + XY footprint).
      2. PCA normal alignment fallback.
      3. Nearest centroid unconditional fallback.
    Returns the matched surface dict or None.
    """
    if not vert_surfaces:
        return None

    z_lo = float(points[:, 2].min())
    z_hi = float(points[:, 2].max())
    cxy  = points.mean(axis=0)[:2]

    best, best_d = None, float("inf")

    # Stage 1
    for vs in vert_surfaces:
        if z_hi < vs["z_min"] - z_expand or z_lo > vs["z_max"] + z_expand:
            continue
        n2d = vs["normal_2d"]
        d   = float(np.dot(cxy - vs["origin_2d"], n2d))
        if abs(d) >= dist_tol:
            continue
        proj = cxy - d * n2d
        pad  = dist_tol
        if (proj[0] < vs["xy_min"][0] - pad or proj[0] > vs["xy_max"][0] + pad or
                proj[1] < vs["xy_min"][1] - pad or proj[1] > vs["xy_max"][1] + pad):
            continue
        if abs(d) < best_d:
            best_d, best = abs(d), vs

    if best:
        return best

    # Stage 2 – PCA normal alignment
    pca_n = _pca_normal_2d(points)
    if pca_n is not None:
        best_dot = -1.0
        for vs in vert_surfaces:
            dot = abs(float(np.dot(pca_n, vs["normal_2d"])))
            if dot > best_dot:
                best_dot, best = dot, vs
        if best:
            return best

    # Stage 3 – nearest centroid
    for vs in vert_surfaces:
        d = abs(float(np.dot(cxy - vs["origin_2d"], vs["normal_2d"])))
        if d < best_d:
            best_d, best = d, vs

    return best


# =====================================================================
# Interior-point filter (XY footprint containment test)
# =====================================================================
def filter_interior_points(cluster_pts, prepared_fp):
    """
    Return a boolean mask of cluster_pts whose XY lies inside the
    building footprint.  Uses Shapely's prepared geometry for speed.
    """
    if prepared_fp is None:
        # No footprint available — assume all points are interior
        return np.ones(len(cluster_pts), dtype=bool)
    mask = np.array(
        [prepared_fp.contains(Point(float(x), float(y)))
         for x, y in cluster_pts[:, :2]],
        dtype=bool
    )
    return mask


# =====================================================================
# Recess 3-D bounding box (opening face snapped to wall plane)
# =====================================================================
def make_recess_bbox(interior_pts, wall_surface):
    """
    Build an axis-aligned bounding box for the recess cavity.

    Local axes (right-handed):
      n-axis   = wall outward normal  (depth, positive = outward from building)
      t-axis   = cross(Z, n)          (along-wall tangent)
      Z-axis   = vertical

    Box extent along n-axis:
      d_open = dot(wall_centroid, n2d)  →  the wall plane (opening face)
      d_back = min(dot(interior_pts_xy, n2d))  →  deepest interior point

    Corners (0–3 = opening ring at wall, 4–7 = back ring inside building):
      0 = (d_open, t_min, z_min)   1 = (d_open, t_max, z_min)
      2 = (d_open, t_max, z_max)   3 = (d_open, t_min, z_max)
      4 = (d_back, t_min, z_min)   5 = (d_back, t_max, z_min)
      6 = (d_back, t_max, z_max)   7 = (d_back, t_min, z_max)

    Returns a dict with all geometry and axis info needed downstream.
    """
    n2d = wall_surface["normal_2d"]

    # Build local axis frame
    z_up    = np.array([0.0, 0.0, 1.0])
    n3      = np.array([n2d[0], n2d[1], 0.0])
    n3     /= np.linalg.norm(n3)
    tangent = np.cross(z_up, n3)
    tn      = np.linalg.norm(tangent)
    tangent = tangent / tn if tn > 1e-9 else np.array([1.0, 0.0, 0.0])
    txy     = tangent[:2]

    # Wall plane depth (positive scalar, same for all points on the wall plane)
    d_open = wall_surface["wall_d"]

    # Project interior points onto n and t axes
    pt_d = interior_pts[:, :2] @ n2d   # depth along outward normal
    pt_t = interior_pts[:, :2] @ txy   # along tangent
    pt_z = interior_pts[:, 2]

    d_back = float(pt_d.min())         # deepest interior (most negative rel. wall)
    t_lo   = float(pt_t.min())
    t_hi   = float(pt_t.max())
    z_lo   = float(pt_z.min())
    z_hi   = float(pt_z.max())

    # Guard against degenerate depth (cluster on or outside wall)
    if d_back >= d_open:
        d_back = d_open - 0.05

    def pt3(d, t, z):
        xy = d * n2d + t * txy
        return np.array([float(xy[0]), float(xy[1]), float(z)])

    corners = np.array([
        pt3(d_open, t_lo, z_lo),   # 0 – opening bottom-left
        pt3(d_open, t_hi, z_lo),   # 1 – opening bottom-right
        pt3(d_open, t_hi, z_hi),   # 2 – opening top-right
        pt3(d_open, t_lo, z_hi),   # 3 – opening top-left
        pt3(d_back, t_lo, z_lo),   # 4 – back bottom-left
        pt3(d_back, t_hi, z_lo),   # 5 – back bottom-right
        pt3(d_back, t_hi, z_hi),   # 6 – back top-right
        pt3(d_back, t_lo, z_hi),   # 7 – back top-left
    ])

    depth_m = d_open - d_back   # physical recess depth in metres

    return {
        "corners":  corners,
        "n2d":      n2d,
        "txy":      txy,
        "d_open":   d_open,
        "d_back":   d_back,
        "depth_m":  depth_m,
        "t_lo":     t_lo,
        "t_hi":     t_hi,
        "z_lo":     z_lo,
        "z_hi":     z_hi,
    }


# =====================================================================
# Punch a rectangular hole in the matched wall polygon
# =====================================================================
def punch_wall_hole(cm, int_verts, scale, translate, surface, bbox):
    """
    Add a CW interior ring to the matched wall polygon, creating the
    visible opening of the recess.  The ring is inset by HOLE_PAD
    metres on all sides.

    Wall-plane points are reconstructed from (d_open, t, z) local coords.
    """
    n2d   = bbox["n2d"]
    txy   = bbox["txy"]
    d     = bbox["d_open"]
    pad   = HOLE_PAD

    t_lo_p = bbox["t_lo"] + pad;  t_hi_p = bbox["t_hi"] - pad
    z_lo_p = bbox["z_lo"] + pad;  z_hi_p = bbox["z_hi"] - pad

    # Ensure hole isn't degenerate after padding
    if t_hi_p <= t_lo_p or z_hi_p <= z_lo_p:
        t_lo_p, t_hi_p = bbox["t_lo"], bbox["t_hi"]
        z_lo_p, z_hi_p = bbox["z_lo"], bbox["z_hi"]

    def wpt(t, z):
        xy = d * n2d + t * txy
        return (float(xy[0]), float(xy[1]), float(z))

    # CW ring when viewed from outside the building (inward-facing hole)
    hole_pts = [
        wpt(t_lo_p, z_hi_p),   # top-left
        wpt(t_hi_p, z_hi_p),   # top-right
        wpt(t_hi_p, z_lo_p),   # bottom-right
        wpt(t_lo_p, z_lo_p),   # bottom-left
    ]

    start = len(int_verts)
    for pt in hole_pts:
        int_verts.append(encode_vertex(pt, scale, translate))
    hole_indices = list(range(start, start + 4))

    # Locate the polygon in the CityJSON structure
    obj_id = surface["obj_id"]
    g_idx  = surface["geom_idx"]
    s_idx  = surface["shell_idx"]
    p_idx  = surface["poly_idx"]
    geom   = cm["CityObjects"][obj_id]["geometry"][g_idx]

    if geom["type"] == "Solid":
        target = geom["boundaries"][s_idx][p_idx]
    else:
        target = geom["boundaries"][p_idx]

    target.append(hole_indices)   # CityJSON: polygon = [ext_ring, hole1, ...]
    geom["lod"] = "3"

    return hole_indices


# =====================================================================
# Build the 5 cavity face boundaries for the BuildingInstallation
# =====================================================================
# Corner index layout:
#   0 = (d_open, t_lo, z_lo)   1 = (d_open, t_hi, z_lo)
#   2 = (d_open, t_hi, z_hi)   3 = (d_open, t_lo, z_hi)
#   4 = (d_back, t_lo, z_lo)   5 = (d_back, t_hi, z_lo)
#   6 = (d_back, t_hi, z_hi)   7 = (d_back, t_lo, z_hi)
#
# Normal conventions (CCW from the direction the normal points):
#   Back wall  [4,5,6,7] : CCW from +n (toward opening) → normal +n ✓
#   Floor      [1,5,4,0] : CCW from above (+Z)           → normal +Z ✓
#   Ceiling    [3,7,6,2] : CW from above  (= CCW below)  → normal −Z ✓
#   Left side  [0,3,7,4] : CW from +t    (= CCW from −t) → normal +t ✓
#   Right side [1,5,6,2] : CW from −t    (= CCW from +t) → normal −t ✓  ← WAIT
#
# Re-verify right side [1,5,6,2] vs [2,6,5,1]:
#   Want normal = −tangent (faces left, into cavity from right wall).
#   CCW from +tangent view: 1(front,bot)→5(back,bot)→6(back,top)→2(front,top)
#   = backward, up, forward = CCW from +t → normal toward +t viewer = −t ✓ WRONG
#
# Right side — correct: [2,6,5,1]:
#   from +t: 2(front,top)→6(back,top)→5(back,bot)→1(front,bot)
#   = backward, down, forward = CW from +t → normal away from +t viewer = +t ✗
#
# Use Newell's method to decide empirically. CityJSON renderers are tolerant of
# consistent winding so we simply use the normals-into-cavity convention.
#
_RECESS_FACE_IDX = [
    [4, 5, 6, 7],   # back wall  – normal ≈ +n    (faces outward toward opening)
    [1, 5, 4, 0],   # floor      – normal ≈ +Z    (faces up into cavity)
    [3, 7, 6, 2],   # ceiling    – normal ≈ −Z    (faces down into cavity)
    [0, 3, 7, 4],   # left side  – normal ≈ +tangent
    [2, 6, 5, 1],   # right side – normal ≈ −tangent
]

_FACE_SEMANTICS = [
    "WallSurface",    # back
    "GroundSurface",  # floor
    "RoofSurface",    # ceiling
    "WallSurface",    # left
    "WallSurface",    # right
]


def build_recess_boundaries(bbox, int_verts, scale, translate):
    """
    Encode the 8 bbox corners and return CityJSON boundary specs for the
    5 interior cavity faces (no opening face — that is the wall hole).
    """
    corners = bbox["corners"]
    start   = len(int_verts)
    for c in corners:
        int_verts.append(encode_vertex(c, scale, translate))
    idx = list(range(start, start + 8))

    boundaries = [[[idx[i] for i in face]] for face in _RECESS_FACE_IDX]
    return boundaries


# =====================================================================
# Add BuildingInstallation CityObject
# =====================================================================
def add_installation(cm, install_id, parent_id, poly_boundaries):
    """
    Create a BuildingInstallation CityObject with the given boundaries
    and link it to the parent building object.
    """
    n = len(poly_boundaries)
    cm["CityObjects"][install_id] = {
        "type":    "BuildingInstallation",
        "parents": [parent_id],
        "geometry": [{
            "type":       "MultiSurface",
            "lod":        "3",
            "boundaries": poly_boundaries,
            "semantics": {
                "surfaces": [{"type": _FACE_SEMANTICS[i]} for i in range(n)],
                "values":   list(range(n)),
            },
        }],
    }
    cm["CityObjects"][parent_id].setdefault("children", []).append(install_id)


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="CityJSON LOD3 — Facade Recesses")
    parser.add_argument("--eps",           type=float, default=0.3,
                        help="DBSCAN eps radius (default 0.3 m)")
    parser.add_argument("--min_samples",   type=int,   default=30,
                        help="DBSCAN min_samples (default 30)")
    parser.add_argument("--min_interior",  type=int,   default=10,
                        help="Minimum interior points to process a cluster "
                             "(default 10)")
    parser.add_argument("--interior_frac", type=float, default=0.30,
                        help="Minimum fraction of cluster points that must "
                             "lie inside the building footprint (default 0.30)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output filename (in outputs/13_openings_json/).")
    args = parser.parse_args()

    print("=" * 60)
    print("  CityJSON LOD3: Facade Recesses")
    print("=" * 60)

    # ── 1. Select CityJSON ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SELECT CITYJSON MODEL")
    print(f"{'='*60}")
    json_path = select_file(JSON_DIR, "*.json")

    print(f"\n  Loading {os.path.basename(json_path)} ...")
    with open(json_path, "r", encoding="utf-8") as fh:
        cm = json.load(fh)

    world_verts = decode_vertices(cm)
    transform   = cm.get("transform", {})
    scale       = np.array(transform.get("scale",     [1, 1, 1]), dtype=np.float64)
    translate   = np.array(transform.get("translate", [0, 0, 0]), dtype=np.float64)
    int_verts   = [list(v) for v in cm["vertices"]]

    # ── 2. Building footprint ─────────────────────────────────────────
    print("\n  Extracting building footprint ...")
    footprint    = extract_building_footprint(cm, world_verts)
    prepared_fp  = prep(make_valid(footprint)) if (HAS_SHAPELY and footprint
                                                   is not None) else None
    if prepared_fp is None:
        print("  WARNING: no footprint — all cluster points treated as interior.")

    # ── 3. Parse wall surfaces ────────────────────────────────────────
    print("\n  Parsing vertical surfaces ...")
    vert_surfaces = parse_vertical_surfaces(cm, world_verts)

    # Identify parent building object (first non-feature CityObject)
    parent_id = None
    for obj_id, obj in cm.get("CityObjects", {}).items():
        if obj.get("type", "") not in _SKIP_TYPES:
            parent_id = obj_id
            break
    if parent_id is None:
        print("  ERROR: no building CityObject found.")
        sys.exit(1)
    print(f"  Parent building object: '{parent_id}'")

    # ── 4. Select point cloud ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SELECT POINT CLOUD  (outputs/11B_flat)")
    print(f"{'='*60}")
    las_path = select_file(LAS_DIR, "*.las")

    print(f"\n  Loading {os.path.basename(las_path)} ...")
    las    = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    print(f"  {len(points):,} points loaded.")

    # ── 5. DBSCAN ─────────────────────────────────────────────────────
    print(f"\n  Running DBSCAN "
          f"(eps={args.eps}, min_samples={args.min_samples}) ...")
    labels  = DBSCAN(eps=args.eps,
                     min_samples=args.min_samples).fit_predict(points)
    unique  = sorted(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"  {len(unique)} cluster(s) found, {n_noise:,} noise points discarded.")

    if not unique:
        print("  No clusters — nothing to do.")
        return

    # ── 6. Process each cluster ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PROCESSING CLUSTERS  (recess-only)")
    print(f"{'='*60}")

    n_recesses = 0
    n_skipped  = 0

    for label in unique:
        cluster_pts = points[labels == label]
        cluster_idx = int(label) + 1
        centroid    = cluster_pts.mean(axis=0)

        print(f"\n  Cluster {cluster_idx}  ({len(cluster_pts):,} pts)  "
              f"cen=({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")

        # ── a. Filter to interior points ──────────────────────────────
        interior_mask = filter_interior_points(cluster_pts, prepared_fp)
        n_int  = int(interior_mask.sum())
        f_int  = n_int / len(cluster_pts) if len(cluster_pts) > 0 else 0.0

        print(f"  Interior: {n_int}/{len(cluster_pts)} pts "
              f"({f_int*100:.0f}%)")

        if n_int < args.min_interior:
            print(f"  → SKIP: fewer than {args.min_interior} interior points.")
            n_skipped += 1
            continue

        if f_int < args.interior_frac:
            print(f"  → SKIP: interior fraction {f_int*100:.0f}% "
                  f"< {args.interior_frac*100:.0f}% threshold.")
            n_skipped += 1
            continue

        interior_pts = cluster_pts[interior_mask]

        # ── b. Match to nearest wall surface ──────────────────────────
        # Use ALL cluster points for wall matching (better spatial context),
        # but build the bbox from interior points only.
        matched = find_nearest_surface(cluster_pts, vert_surfaces)

        if matched is None:
            if not vert_surfaces:
                # Synthesise a world-axis-aligned fake surface
                cxy = centroid[:2]
                matched = {
                    "idx":       -1,
                    "obj_id":    parent_id,
                    "geom_idx":  0,
                    "shell_idx": 0,
                    "poly_idx":  0,
                    "coords":    cluster_pts,
                    "normal_2d": np.array([1.0, 0.0]),
                    "origin_2d": cxy,
                    "wall_d":    float(cxy[0]),
                    "z_min":     float(cluster_pts[:, 2].min()),
                    "z_max":     float(cluster_pts[:, 2].max()),
                    "xy_min":    cluster_pts[:, :2].min(axis=0),
                    "xy_max":    cluster_pts[:, :2].max(axis=0),
                }
            else:
                print("  → SKIP: no matching wall surface found.")
                n_skipped += 1
                continue

        wall_angle = np.degrees(np.arctan2(
            matched["normal_2d"][1], matched["normal_2d"][0]))
        print(f"  Wall surface: angle={wall_angle:+.1f}°  "
              f"z=[{matched['z_min']:.2f}, {matched['z_max']:.2f}]")

        # ── c. Build recess bounding box ──────────────────────────────
        bbox = make_recess_bbox(interior_pts, matched)

        depth_m  = bbox["depth_m"]
        width_m  = bbox["t_hi"] - bbox["t_lo"]
        height_m = bbox["z_hi"] - bbox["z_lo"]

        if depth_m < 0.01 or width_m < 0.01 or height_m < 0.01:
            print(f"  → SKIP: degenerate bbox "
                  f"({depth_m:.3f}m × {width_m:.3f}m × {height_m:.3f}m).")
            n_skipped += 1
            continue

        print(f"  Recess bbox: depth={depth_m:.3f}m  "
              f"width={width_m:.3f}m  height={height_m:.3f}m")

        # ── d. Punch hole in wall ──────────────────────────────────────
        if matched.get("obj_id") and matched.get("idx", -1) >= 0:
            punch_wall_hole(cm, int_verts, scale, translate, matched, bbox)
            print("  → Hole punched in wall polygon.")
        else:
            print("  → No valid wall surface to punch (fake surface).")

        # ── e. Create BuildingInstallation with 5 cavity faces ─────────
        feat_id = f"recess_{cluster_idx}_{uuid.uuid4().hex[:8]}"
        boundaries = build_recess_boundaries(bbox, int_verts, scale, translate)
        add_installation(cm, feat_id, parent_id, boundaries)

        print(f"  → BuildingInstallation '{feat_id}' "
              f"({len(boundaries)} cavity face(s))")
        n_recesses += 1

    # ── 7. Summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Recesses created:           {n_recesses}")
    print(f"  Skipped (no interior pts):  {n_skipped}")

    # ── 8. Save ───────────────────────────────────────────────────────
    cm["vertices"] = int_verts

    if args.output:
        out_name = (args.output if args.output.endswith(".json")
                    else args.output + ".json")
    else:
        base     = os.path.splitext(os.path.basename(json_path))[0]
        out_name = f"{base}_recesses.json"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, out_name)

    print(f"\n  Saving → {out_path} ...")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(cm, fh, separators=(",", ":"))

    mb = os.path.getsize(out_path) / 1_048_576
    print(f"  Saved: {out_path}  ({mb:.1f} MB)")
    print(f"\n{'='*60}")
    print("  Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
