#!/usr/bin/env python3
"""
CityJSON LOD3: Facade Building Installations — Recess / Relief Classifier.

Takes a CityJSON LOD3 file (e.g. from 10_trial_cityjson.py) and a .las
point cloud of non-window/door facade features from outputs/11A_flat.

Pipeline
--------
1. Extract the building's 2-D XY footprint (GroundSurface or convex hull).
2. DBSCAN-cluster the point cloud.
3. For each cluster:
      a. Build a wall-aligned 3-D bounding box (depth along the nearest
         vertical surface's outward normal, width along its tangent, height Z).
      b. Project the bbox to XY and compare it against the building footprint.

   RELIEF   – The bbox XY footprint lies mostly OUTSIDE the footprint
              (overlap fraction ≤ RECESS_THRESHOLD).
              → Append the full 3-D bbox as a BuildingInstallation.

   RECESS   – A significant part of the bbox XY footprint overlaps the
              building INTERIOR (overlap fraction > RECESS_THRESHOLD).
              → Clip away the interior portion with shapely.difference().
              → Extrude the remaining exterior 2-D polygon back to 3-D.
              → If the exterior is non-trivial, append it as a
                BuildingInstallation (the carved-away interior is the void).
              → If the bbox is entirely inside, no geometry is emitted.

4. Save the enriched CityJSON.

Requirements
------------
    conda install -c conda-forge shapely   (or pip install shapely)

Usage
-----
    conda activate las-env
    python scripts/LOD2toLOD3/11_bldginstallation_flat.py [-o OUTPUT]
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
    from shapely.geometry import MultiPoint, Polygon as SPolygon, MultiPolygon
    from shapely.validation import make_valid
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    print("WARNING: shapely not installed — recess clipping is disabled and all\n"
          "         features will be treated as reliefs.\n"
          "  Install: conda install -c conda-forge shapely")


# ─── Directories ──────────────────────────────────────────────────────────────
LAS_DIR    = "outputs/11B_flat"
JSON_DIR   = "outputs/13_openings_json"
OUTPUT_DIR = "outputs/13_openings_json"

# ─── Tuning constants ─────────────────────────────────────────────────────────
VERTICAL_TOL     = 0.3    # |n_z / |n|| > this → surface is near-horizontal, skip
RECESS_THRESHOLD = 0.40   # overlap fraction above which the cluster is a RECESS
MIN_EXTERIOR_M2  = 0.05   # m² – minimum exterior area to bother creating geometry


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
    """Recursively yield all integer vertex indices from a nested boundary list."""
    if isinstance(x, list):
        for y in x:
            yield from _flatten_indices(y)
    elif isinstance(x, int):
        yield x


def extract_building_footprint(cm, world_verts):
    """
    Return a shapely Polygon representing the building's XY footprint.

    Priority:
      1. GroundSurface semantic polygon from the CityJSON geometry.
      2. XY convex hull of all non-feature vertices (fallback).

    Returns None if shapely is not available.
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

            # Build flat list of (global_polygon_index, polygon)
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
                    ext_ring = polygon[0]
                    coords   = world_verts[np.array(ext_ring)]
                    xy       = [(float(c[0]), float(c[1])) for c in coords]
                    poly     = make_valid(SPolygon(xy))
                    if not poly.is_empty and poly.area > 0:
                        print(f"  Building footprint from GroundSurface: "
                              f"{poly.area:.1f} m²")
                        return poly
                except Exception:
                    pass

    # Fallback: convex hull of all building vertex XY coordinates
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
            print(f"  Building footprint from convex hull: {hull.area:.1f} m²")
            return hull

    print("  WARNING: could not determine building footprint.")
    return None


# =====================================================================
# Parse vertical surfaces (same as 10_trial_cityjson.py)
# =====================================================================
def parse_vertical_surfaces(cm, world_verts):
    """
    Return a list of dicts for all near-vertical polygons in the building
    geometry.  Skips Window, Door, BuildingInstallation child objects.
    Only processes the highest-LOD geometry per CityObject.
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
        g_idx, geom = best_idx, geoms[best_idx]

        boundaries = geom.get("boundaries", [])
        geom_type  = geom.get("type", "")

        if geom_type == "Solid":
            shell_iter = enumerate(boundaries)
        else:
            shell_iter = [(0, boundaries)]

        for s_idx, shell in shell_iter:
            for p_idx, polygon in enumerate(shell):
                try:
                    ext_ring = polygon[0]
                    coords   = world_verts[np.array(ext_ring)]
                except (IndexError, KeyError, TypeError):
                    continue

                if len(coords) < 3:
                    continue

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
                    continue

                nh  = n[:2]
                mag = np.linalg.norm(nh)
                if mag < 1e-6:
                    continue

                normal_2d = nh / mag
                centroid  = coords.mean(axis=0)

                surfaces.append({
                    "idx":       len(surfaces),
                    "obj_id":    obj_id,
                    "geom_idx":  g_idx,
                    "shell_idx": s_idx,
                    "poly_idx":  p_idx,
                    "coords":    coords,
                    "normal_2d": normal_2d,
                    "origin_2d": centroid[:2].copy(),
                    "z_min":     float(coords[:, 2].min()),
                    "z_max":     float(coords[:, 2].max()),
                    "xy_min":    coords[:, :2].min(axis=0),
                    "xy_max":    coords[:, :2].max(axis=0),
                })

    print(f"  Parsed {len(surfaces)} vertical surface(s).")
    return surfaces


# =====================================================================
# 3-stage nearest vertical surface matching
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
    3-stage matching: spatial intersection → PCA alignment → nearest centroid.
    Returns the matched surface dict or None.
    """
    if not vert_surfaces:
        return None

    z_lo = float(points[:, 2].min())
    z_hi = float(points[:, 2].max())
    cxy  = points.mean(axis=0)[:2]

    # Stage 1
    best, best_d = None, float("inf")
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

    # Stage 2 – PCA alignment
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
# Wall-aligned 3-D bounding box
# =====================================================================
def make_wall_aligned_bbox(points, wall_surface):
    """
    Build a wall-aligned 3-D bounding box for *points* using the wall's
    outward normal as the depth axis and its XY tangent as the width axis.

    Axes:
      axis0 = wall outward normal (depth direction)
      axis1 = cross(Z_up, axis0)  (along-wall tangent)
      axis2 = Z_up

    Returns (corners, z_lo, z_hi, n2d):
      corners – ndarray(8,3) world-space box corners ordered:
                0:(d_min,t_min,z_lo), 1:(d_max,t_min,z_lo),
                2:(d_max,t_max,z_lo), 3:(d_min,t_max,z_lo),
                4:(d_min,t_min,z_hi), 5:(d_max,t_min,z_hi),
                6:(d_max,t_max,z_hi), 7:(d_min,t_max,z_hi)
      z_lo, z_hi – vertical extents (for recess extrusion)
      n2d – 2D outward unit normal
    """
    n2d  = wall_surface["normal_2d"]
    n3   = np.array([n2d[0], n2d[1], 0.0])
    n3  /= np.linalg.norm(n3)

    z_up    = np.array([0.0, 0.0, 1.0])
    tangent = np.cross(z_up, n3)
    tn      = np.linalg.norm(tangent)
    tangent = tangent / tn if tn > 1e-9 else np.array([1.0, 0.0, 0.0])

    axes     = np.column_stack([n3, tangent, z_up])   # (3,3) right-handed
    centroid = points.mean(axis=0)
    proj     = (points - centroid) @ axes

    p_min, p_max = proj.min(axis=0), proj.max(axis=0)

    # 8 corners in local axes, back to world
    v_loc = np.array([
        [p_min[0], p_min[1], p_min[2]],
        [p_max[0], p_min[1], p_min[2]],
        [p_max[0], p_max[1], p_min[2]],
        [p_min[0], p_max[1], p_min[2]],
        [p_min[0], p_min[1], p_max[2]],
        [p_max[0], p_min[1], p_max[2]],
        [p_max[0], p_max[1], p_max[2]],
        [p_min[0], p_max[1], p_max[2]],
    ])
    corners = (v_loc @ axes.T) + centroid

    z_lo = float(corners[:, 2].min())
    z_hi = float(corners[:, 2].max())
    return corners, z_lo, z_hi, n2d


# Face winding indices into the 8 corners above (CCW from outside each face)
_BBOX_FACE_IDX = [
    [0, 3, 2, 1],   # bottom  (outward normal = −Z)
    [4, 5, 6, 7],   # top     (outward normal = +Z)
    [1, 2, 6, 5],   # far     (outward normal = +wall_normal, relief face)
    [0, 4, 7, 3],   # near    (outward normal = −wall_normal, building-facing)
    [0, 1, 5, 4],   # side A  (outward normal = −tangent)
    [3, 7, 6, 2],   # side B  (outward normal = +tangent)
]


def bbox_corners_to_boundaries(corners, int_verts, scale, translate):
    """
    Encode 8 bbox corners into int_verts and return a list of CityJSON
    polygon boundary specs (list of [[vertex_indices]]) for all 6 faces.
    """
    start = len(int_verts)
    for c in corners:
        int_verts.append(encode_vertex(c, scale, translate))
    idx = list(range(start, start + 8))
    return [[[idx[i] for i in face]] for face in _BBOX_FACE_IDX]


# =====================================================================
# Extrude a shapely 2-D polygon to CityJSON 3-D boundaries
# =====================================================================
def extrude_to_boundaries(poly_2d, z_lo, z_hi, int_verts, scale, translate):
    """
    Extrude a shapely Polygon (or MultiPolygon) from z_lo to z_hi.

    Returns a list of CityJSON polygon boundary specs
    [[v_indices]] — one per face (bottom, top, one side per edge).

    The exterior ring of each shapely polygon is assumed CCW from above,
    which gives correct outward normals for the extruded side walls.
    """
    if poly_2d is None or poly_2d.is_empty:
        return []

    # Handle MultiPolygon recursively
    if poly_2d.geom_type == "MultiPolygon":
        result = []
        for part in poly_2d.geoms:
            result.extend(
                extrude_to_boundaries(part, z_lo, z_hi, int_verts, scale, translate))
        return result

    if poly_2d.geom_type != "Polygon":
        return []

    ext_2d = list(poly_2d.exterior.coords)[:-1]   # CCW, no closing dup
    n = len(ext_2d)
    if n < 3:
        return []

    # Encode bottom ring (z_lo) and top ring (z_hi)
    bot_start = len(int_verts)
    for x, y in ext_2d:
        int_verts.append(encode_vertex((x, y, z_lo), scale, translate))
    bot = list(range(bot_start, bot_start + n))

    top_start = len(int_verts)
    for x, y in ext_2d:
        int_verts.append(encode_vertex((x, y, z_hi), scale, translate))
    top = list(range(top_start, top_start + n))

    boundaries = []

    # Bottom face: reverse CCW→gives normal pointing −Z ✓
    boundaries.append([list(reversed(bot))])

    # Top face: CCW from above → normal +Z ✓
    boundaries.append([top])

    # Side walls: bot[i]→bot[j]→top[j]→top[i] → outward normal ✓
    for i in range(n):
        j = (i + 1) % n
        boundaries.append([[bot[i], bot[j], top[j], top[i]]])

    return boundaries


# =====================================================================
# Classify and add BuildingInstallation to CityJSON
# =====================================================================
def add_installation(cm, int_verts, scale, translate,
                     install_id, parent_id, poly_boundaries):
    """
    Create a BuildingInstallation CityObject and link it to its parent.
    poly_boundaries is a list of CityJSON polygon specs [[vert_indices]].
    """
    cm["CityObjects"][install_id] = {
        "type":    "BuildingInstallation",
        "parents": [parent_id],
        "geometry": [{
            "type":       "MultiSurface",
            "lod":        "3",
            "boundaries": poly_boundaries,
            "semantics": {
                "surfaces": [{"type": "WallSurface"}] * len(poly_boundaries),
                "values":   list(range(len(poly_boundaries))),
            },
        }],
    }
    cm["CityObjects"][parent_id].setdefault("children", []).append(install_id)


# =====================================================================
# XY footprint of a set of 3-D world-space corners
# =====================================================================
def corners_xy_polygon(corners):
    """Return a shapely Polygon of the XY convex hull of the bbox corners."""
    xy  = [(float(c[0]), float(c[1])) for c in corners]
    pts = MultiPoint(xy)
    hull = pts.convex_hull
    if hull.geom_type != "Polygon":
        return None
    return hull


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="CityJSON LOD3 building installations (recess/relief)")
    parser.add_argument("--eps",         type=float, default=0.3,
                        help="DBSCAN eps radius (default 0.3 m)")
    parser.add_argument("--min_samples", type=int,   default=30,
                        help="DBSCAN min_samples (default 30)")
    parser.add_argument("--output", "-o", type=str,  default=None,
                        help="Output filename (inside outputs/13_openings_json/).")
    parser.add_argument("--recess_threshold", type=float, default=RECESS_THRESHOLD,
                        help=f"Overlap fraction for recess detection "
                             f"(default {RECESS_THRESHOLD})")
    args = parser.parse_args()

    print("=" * 60)
    print("  CityJSON LOD3: Building Installations")
    print("=" * 60)

    # ── 1. Select CityJSON ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SELECT CITYJSON MODEL")
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

    # ── 2. Extract building footprint ─────────────────────────────────
    print("\n  Extracting building footprint ...")
    footprint = extract_building_footprint(cm, world_verts)
    if footprint is None and HAS_SHAPELY:
        print("  WARNING: no footprint found — recess detection disabled.")

    # ── 3. Parse vertical surfaces for wall matching ───────────────────
    print("\n  Parsing vertical surfaces ...")
    vert_surfaces = parse_vertical_surfaces(cm, world_verts)

    # Find the parent building object (first non-feature CityObject)
    parent_id = None
    for obj_id, obj in cm.get("CityObjects", {}).items():
        if obj.get("type", "") not in _SKIP_TYPES:
            parent_id = obj_id
            break
    if parent_id is None:
        print("  ERROR: no building CityObject found.")
        sys.exit(1)
    print(f"  Parent building object: '{parent_id}'")

    # ── 4. Select point cloud ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SELECT POINT CLOUD  (outputs/11A_flat)")
    print(f"{'='*60}")
    las_path = select_file(LAS_DIR, "*.las")

    print(f"\n  Loading {os.path.basename(las_path)} ...")
    las    = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    print(f"  {len(points):,} points loaded.")

    # ── 5. DBSCAN ──────────────────────────────────────────────────────
    print(f"\n  Running DBSCAN (eps={args.eps}, min_samples={args.min_samples}) ...")
    labels  = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit_predict(points)
    unique  = sorted(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"  {len(unique)} cluster(s) found, {n_noise:,} noise points discarded.")

    if not unique:
        print("  No clusters — nothing to do.")
        return

    # ── 6. Classify and build geometry ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CLASSIFYING CLUSTERS")
    print(f"{'='*60}")

    n_relief = 0
    n_recess = 0
    n_skip   = 0

    for label in unique:
        cluster_pts = points[labels == label]
        cluster_idx = int(label) + 1
        centroid    = cluster_pts.mean(axis=0)

        print(f"\n  Cluster {cluster_idx}  ({len(cluster_pts):,} pts)  "
              f"centroid=({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")

        # Find nearest vertical surface for box alignment
        matched = find_nearest_surface(cluster_pts, vert_surfaces)
        if matched is None and not vert_surfaces:
            # No wall information: fallback to AABB aligned to world axes
            fake_ws = {
                "normal_2d": np.array([1.0, 0.0]),
                "origin_2d": centroid[:2].copy(),
                "z_min":     float(cluster_pts[:, 2].min()),
                "z_max":     float(cluster_pts[:, 2].max()),
                "xy_min":    cluster_pts[:, :2].min(axis=0),
                "xy_max":    cluster_pts[:, :2].max(axis=0),
                "coords":    cluster_pts,
            }
            matched = fake_ws

        corners, z_lo, z_hi, n2d = make_wall_aligned_bbox(cluster_pts, matched)

        # Wall angle for logging
        wall_angle = np.degrees(np.arctan2(n2d[1], n2d[0]))
        print(f"  Matched surface: angle={wall_angle:+.1f}°  "
              f"bbox z=[{z_lo:.2f}, {z_hi:.2f}]")

        # ── Classify via XY footprint intersection ─────────────────────
        feat_id = f"bldgInstall_{cluster_idx}_{uuid.uuid4().hex[:8]}"

        if HAS_SHAPELY and footprint is not None:
            bbox_xy = corners_xy_polygon(corners)
            if bbox_xy is None or bbox_xy.is_empty:
                print("  → degenerate XY footprint, skipping cluster.")
                n_skip += 1
                continue

            try:
                inter      = make_valid(footprint).intersection(make_valid(bbox_xy))
                inter_area = inter.area
                bbox_area  = bbox_xy.area
                overlap_f  = inter_area / bbox_area if bbox_area > 0 else 0.0
            except Exception as e:
                print(f"  → shapely error ({e}), treating as RELIEF.")
                overlap_f = 0.0

            if overlap_f > args.recess_threshold:
                # ── RECESS ── clip interior out, keep exterior shell ───
                print(f"  RECESS  (interior overlap {overlap_f*100:.0f}%)")
                try:
                    exterior_2d = make_valid(bbox_xy).difference(make_valid(footprint))
                except Exception:
                    exterior_2d = None

                if exterior_2d is None or exterior_2d.is_empty or \
                        exterior_2d.area < MIN_EXTERIOR_M2:
                    print("  → exterior portion too small or none — "
                          "pure interior void, no geometry emitted.")
                    n_skip += 1
                    continue

                boundaries = extrude_to_boundaries(
                    exterior_2d, z_lo, z_hi, int_verts, scale, translate)

                if not boundaries:
                    print("  → extrusion produced no boundary faces, skipping.")
                    n_skip += 1
                    continue

                add_installation(cm, int_verts, scale, translate,
                                 feat_id, parent_id, boundaries)
                print(f"  → BuildingInstallation '{feat_id}' "
                      f"(RECESS exterior, {len(boundaries)} face(s))")
                n_recess += 1

            else:
                # ── RELIEF ── full 3-D bbox as BuildingInstallation ────
                print(f"  RELIEF  (interior overlap {overlap_f*100:.0f}%)")
                boundaries = bbox_corners_to_boundaries(
                    corners, int_verts, scale, translate)
                add_installation(cm, int_verts, scale, translate,
                                 feat_id, parent_id, boundaries)
                print(f"  → BuildingInstallation '{feat_id}' "
                      f"(RELIEF, 6 faces)")
                n_relief += 1

        else:
            # shapely unavailable: treat everything as RELIEF
            print("  RELIEF  (shapely not available, no recess detection)")
            boundaries = bbox_corners_to_boundaries(
                corners, int_verts, scale, translate)
            add_installation(cm, int_verts, scale, translate,
                             feat_id, parent_id, boundaries)
            n_relief += 1

    # ── 7. Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Reliefs (full bbox):         {n_relief}")
    print(f"  Recesses (exterior shell):   {n_recess}")
    print(f"  Skipped (pure void / degen): {n_skip}")
    total = n_relief + n_recess
    print(f"  BuildingInstallation added:  {total}")

    # ── 8. Save ────────────────────────────────────────────────────────
    cm["vertices"] = int_verts

    if args.output:
        out_name = (args.output if args.output.endswith(".json")
                    else args.output + ".json")
    else:
        base     = os.path.splitext(os.path.basename(json_path))[0]
        out_name = f"{base}_installs.json"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, out_name)

    print(f"\n  Saving → {out_path} ...")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(cm, fh, separators=(",", ":"))

    mb = os.path.getsize(out_path) / 1_048_576
    print(f"  Saved: {out_path}  ({mb:.1f} MB)")
    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
