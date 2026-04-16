#!/usr/bin/env python3
"""
CityJSON LOD3: Facade Extrusions (BuildingInstallations).

Takes a CityJSON LOD3 file and a .las point cloud of facade features
from outputs/11B_flat.

Pipeline
--------
1. Extract the building 2-D XY footprint (GroundSurface → convex hull).
2. Parse all near-vertical wall surfaces for normal/tangent alignment.
3. DBSCAN-cluster the point cloud.
4. For each cluster:
      a. Filter points to those lying OUTSIDE the building footprint.
      b. If exterior points are insufficient, skip.
      c. Find the nearest wall surface for axis alignment.
      d. Build a wall-aligned 3-D bounding box that protrudes outward:
           • Near face  – snapped flush to the wall plane.
           • Far face   – outermost exterior point along +normal.
           • Width / height – tangent / Z extents of exterior points.
      e. Append a BuildingInstallation child with all 6 box faces.
5. Save the enriched CityJSON.

Usage
-----
    conda activate las-env
    python scripts/LOD2toLOD3/11_extrusions.py [-o OUTPUT]

Options
-------
    --eps              DBSCAN neighbourhood radius   (default 0.3 m)
    --min_samples      DBSCAN minimum cluster size   (default 30)
    --min_exterior     Minimum exterior pts to keep   (default 10)
    --exterior_frac    Minimum exterior fraction      (default 0.30)
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
    print("WARNING: shapely not installed — exterior filtering is disabled.\n"
          "  Install: conda install -c conda-forge shapely")


# ─── Directories ──────────────────────────────────────────────────────────────
LAS_DIR    = "outputs/11B_flat"
JSON_DIR   = "outputs/13_openings_json"
OUTPUT_DIR = "outputs/13_openings_json"

# ─── Tuning constants ─────────────────────────────────────────────────────────
VERTICAL_TOL = 0.3   # |n_z / |n|| > this → near-horizontal surface, skip


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
    Priority: GroundSurface semantic → XY convex hull fallback.
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
    geometry, annotated with outward 2-D normal and wall-plane depth.
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
# Exterior-point filter (XY footprint rejection test)
# =====================================================================
def filter_exterior_points(cluster_pts, prepared_fp):
    """
    Return a boolean mask: True for points whose XY lies OUTSIDE the
    building footprint.  If no footprint is available all points pass.
    """
    if prepared_fp is None:
        return np.ones(len(cluster_pts), dtype=bool)
    mask = np.array(
        [not prepared_fp.contains(Point(float(x), float(y)))
         for x, y in cluster_pts[:, :2]],
        dtype=bool
    )
    return mask


# =====================================================================
# Extrusion 3-D bounding box (near face snapped to wall, protrudes out)
# =====================================================================
def make_extrusion_bbox(exterior_pts, wall_surface):
    """
    Build a wall-aligned bounding box for an exterior extrusion cluster.

    Local axes:
      n-axis  = wall outward normal  (positive = outward from building)
      t-axis  = cross(Z, n)          (along-wall tangent)
      Z-axis  = vertical

    Box extent along n-axis:
      d_wall = dot(wall_centroid, n2d)  — near face, touching the wall
      d_out  = max(dot(pts_xy, n2d))   — far face, outermost exterior point

    Corners  (0–3 near ring at wall, 4–7 far ring outward):
      0 = (d_wall, t_lo, z_lo)   3 = (d_wall, t_hi, z_lo)
      1 = (d_out,  t_lo, z_lo)   2 = (d_out,  t_hi, z_lo)
      4 = (d_wall, t_lo, z_hi)   7 = (d_wall, t_hi, z_hi)
      5 = (d_out,  t_lo, z_hi)   6 = (d_out,  t_hi, z_hi)

    Returns a dict with corners and axis info.
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

    # Wall plane depth
    d_wall = wall_surface["wall_d"]

    # Project exterior points
    pt_d = exterior_pts[:, :2] @ n2d    # depth along outward normal
    pt_t = exterior_pts[:, :2] @ txy   # along tangent
    pt_z = exterior_pts[:, 2]

    d_out  = float(pt_d.max())          # outermost exterior point
    t_lo   = float(pt_t.min())
    t_hi   = float(pt_t.max())
    z_lo   = float(pt_z.min())
    z_hi   = float(pt_z.max())

    # Guard against degenerate depth (cluster at or behind wall)
    if d_out <= d_wall:
        d_out = d_wall + 0.05

    def pt3(d, t, z):
        xy = d * n2d + t * txy
        return np.array([float(xy[0]), float(xy[1]), float(z)])

    # 8 corners:
    # 0–3: near ring (at wall plane)  4–7: far ring (outward)
    corners = np.array([
        pt3(d_wall, t_lo, z_lo),   # 0 near bottom-left
        pt3(d_out,  t_lo, z_lo),   # 1 far  bottom-left
        pt3(d_out,  t_hi, z_lo),   # 2 far  bottom-right
        pt3(d_wall, t_hi, z_lo),   # 3 near bottom-right
        pt3(d_wall, t_lo, z_hi),   # 4 near top-left
        pt3(d_out,  t_lo, z_hi),   # 5 far  top-left
        pt3(d_out,  t_hi, z_hi),   # 6 far  top-right
        pt3(d_wall, t_hi, z_hi),   # 7 near top-right
    ])

    protrusion_m = d_out - d_wall
    width_m      = t_hi - t_lo
    height_m     = z_hi - z_lo

    return {
        "corners":      corners,
        "n2d":          n2d,
        "txy":          txy,
        "d_wall":       d_wall,
        "d_out":        d_out,
        "protrusion_m": protrusion_m,
        "t_lo":         t_lo,
        "t_hi":         t_hi,
        "z_lo":         z_lo,
        "z_hi":         z_hi,
        "width_m":      width_m,
        "height_m":     height_m,
    }


# =====================================================================
# Build all 6 face boundaries for a bbox extrusion
# =====================================================================
#
# Corner layout (n = outward normal axis, t = tangent axis, Z = vertical):
#   0 = (d_wall, t_lo, z_lo)   near-bottom-left
#   1 = (d_out,  t_lo, z_lo)   far-bottom-left
#   2 = (d_out,  t_hi, z_lo)   far-bottom-right
#   3 = (d_wall, t_hi, z_lo)   near-bottom-right
#   4 = (d_wall, t_lo, z_hi)   near-top-left
#   5 = (d_out,  t_lo, z_hi)   far-top-left
#   6 = (d_out,  t_hi, z_hi)   far-top-right
#   7 = (d_wall, t_hi, z_hi)   near-top-right
#
# Outward normals (verified with Newell / right-hand rule):
#   Bottom  [0,3,2,1]  → −Z
#   Top     [4,5,6,7]  → +Z
#   Far     [1,2,6,5]  → +n  (front face, away from building)
#   Near    [0,4,7,3]  → −n  (back face, against building wall)
#   Left    [0,1,5,4]  → −t
#   Right   [3,7,6,2]  → +t
#
_EXTRUSION_FACE_IDX = [
    [0, 3, 2, 1],   # bottom – normal −Z
    [4, 5, 6, 7],   # top    – normal +Z
    [1, 2, 6, 5],   # far    – normal +n (outward front face)
    [0, 4, 7, 3],   # near   – normal −n (against wall)
    [0, 1, 5, 4],   # left   – normal −t
    [3, 7, 6, 2],   # right  – normal +t
]

_FACE_SEMANTICS = [
    "GroundSurface",  # bottom
    "RoofSurface",    # top
    "WallSurface",    # far (front)
    "WallSurface",    # near (back, against building)
    "WallSurface",    # left
    "WallSurface",    # right
]


def build_extrusion_boundaries(bbox, int_verts, scale, translate):
    """
    Encode the 8 bbox corners into int_verts and return CityJSON
    polygon boundary specs for all 6 faces.
    """
    corners = bbox["corners"]
    start   = len(int_verts)
    for c in corners:
        int_verts.append(encode_vertex(c, scale, translate))
    idx = list(range(start, start + 8))
    return [[[idx[i] for i in face]] for face in _EXTRUSION_FACE_IDX]


# =====================================================================
# Add BuildingInstallation CityObject
# =====================================================================
def add_installation(cm, install_id, parent_id, poly_boundaries):
    """
    Create a BuildingInstallation CityObject and link it to its parent.
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
        description="CityJSON LOD3 — Facade Extrusions (BuildingInstallations)")
    parser.add_argument("--eps",           type=float, default=0.3,
                        help="DBSCAN eps radius (default 0.3 m)")
    parser.add_argument("--min_samples",   type=int,   default=30,
                        help="DBSCAN min_samples (default 30)")
    parser.add_argument("--min_exterior",  type=int,   default=10,
                        help="Minimum exterior points to keep a cluster "
                             "(default 10)")
    parser.add_argument("--exterior_frac", type=float, default=0.30,
                        help="Minimum fraction of cluster points that must "
                             "lie outside the building footprint (default 0.30)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output filename (in outputs/13_openings_json/).")
    args = parser.parse_args()

    print("=" * 60)
    print("  CityJSON LOD3: Facade Extrusions")
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
    footprint   = extract_building_footprint(cm, world_verts)
    prepared_fp = (prep(make_valid(footprint))
                   if HAS_SHAPELY and footprint is not None else None)
    if prepared_fp is None:
        print("  WARNING: no footprint — all cluster points treated as exterior.")

    # ── 3. Parse wall surfaces ────────────────────────────────────────
    print("\n  Parsing vertical surfaces ...")
    vert_surfaces = parse_vertical_surfaces(cm, world_verts)

    # ── Find parent building object ───────────────────────────────────
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
    print("  PROCESSING CLUSTERS  (exterior extrusions)")
    print(f"{'='*60}")

    n_extruded = 0
    n_skipped  = 0

    for label in unique:
        cluster_pts = points[labels == label]
        cluster_idx = int(label) + 1
        centroid    = cluster_pts.mean(axis=0)

        print(f"\n  Cluster {cluster_idx}  ({len(cluster_pts):,} pts)  "
              f"cen=({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")

        # ── a. Filter to exterior points ──────────────────────────────
        ext_mask = filter_exterior_points(cluster_pts, prepared_fp)
        n_ext    = int(ext_mask.sum())
        f_ext    = n_ext / len(cluster_pts) if len(cluster_pts) > 0 else 0.0

        print(f"  Exterior: {n_ext}/{len(cluster_pts)} pts ({f_ext*100:.0f}%)")

        if n_ext < args.min_exterior:
            print(f"  → SKIP: fewer than {args.min_exterior} exterior points.")
            n_skipped += 1
            continue

        if f_ext < args.exterior_frac:
            print(f"  → SKIP: exterior fraction {f_ext*100:.0f}% "
                  f"< {args.exterior_frac*100:.0f}% threshold.")
            n_skipped += 1
            continue

        exterior_pts = cluster_pts[ext_mask]

        # ── b. Match to nearest wall surface ──────────────────────────
        matched = find_nearest_surface(cluster_pts, vert_surfaces)

        if matched is None:
            if not vert_surfaces:
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

        # ── c. Build extrusion bounding box ───────────────────────────
        bbox = make_extrusion_bbox(exterior_pts, matched)

        p_m = bbox["protrusion_m"]
        w_m = bbox["width_m"]
        h_m = bbox["height_m"]

        if p_m < 0.01 or w_m < 0.01 or h_m < 0.01:
            print(f"  → SKIP: degenerate bbox "
                  f"({p_m:.3f}m × {w_m:.3f}m × {h_m:.3f}m).")
            n_skipped += 1
            continue

        print(f"  Extrusion bbox: protrusion={p_m:.3f}m  "
              f"width={w_m:.3f}m  height={h_m:.3f}m")

        # ── d. Create BuildingInstallation with 6 faces ───────────────
        feat_id    = f"extrusion_{cluster_idx}_{uuid.uuid4().hex[:8]}"
        boundaries = build_extrusion_boundaries(bbox, int_verts, scale, translate)
        add_installation(cm, feat_id, parent_id, boundaries)

        print(f"  → BuildingInstallation '{feat_id}' "
              f"({len(boundaries)} face(s))")
        n_extruded += 1

    # ── 7. Summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Extrusions created:          {n_extruded}")
    print(f"  Skipped (insufficient pts):  {n_skipped}")

    # ── 8. Save ───────────────────────────────────────────────────────
    cm["vertices"] = int_verts

    if args.output:
        out_name = (args.output if args.output.endswith(".json")
                    else args.output + ".json")
    else:
        base     = os.path.splitext(os.path.basename(json_path))[0]
        out_name = f"{base}_extrusions.json"

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
