#!/usr/bin/env python3
"""
Facade Features → CityJSON LOD3 Pipeline.

Mirrors the logic of 10_trial_openings.py but operates on CityJSON models.

Workflow
--------
1. User selects a CityJSON (.json) building from data/lod_2.
2. All polygons whose Newell normal has a small Z-component are treated as
   vertical surfaces (actual wall geometry), regardless of semantic label.
3. User selects one or more .las point clouds and assigns each a feature type
   (Window or Door).
4. DBSCAN clusters each cloud; each cluster's 3-D extents are projected as a
   flat rectangle onto the nearest vertical surface (3-stage spatial match →
   normal alignment → nearest-centroid fallback).
5. The projection is cut as a hole (interior ring) into the matched surface
   polygon and simultaneously created as a child Window / Door CityObject
   whose geometry is the same coplanar rectangle.
6. The surface's geometry LOD is upgraded to "3".
7. The modified CityJSON is saved to outputs/13_openings_json/<output>.

Usage
-----
    conda activate las-env
    python scripts/LOD2toLOD3/10_trial_cityjson.py [-o OUTPUT_NAME]

Options
-------
    --eps           DBSCAN neighbourhood radius           (default: 0.3 m)
    --min_samples   DBSCAN minimum cluster size           (default: 30)
    --output, -o    Output filename (saved in outputs/13_openings_json/).
                    Defaults to <source_basename>_LOD3.json.
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


# =====================================================================
# Directories
# =====================================================================
LAS_DIR    = "outputs/11B_flat"
JSON_DIR   = "outputs/00_json_wall_merged"
OUTPUT_DIR = "outputs/13_openings_json"

# Vertical-surface detection tolerance:
# surfaces whose |n_z / |n|| > VERTICAL_TOL are considered near-horizontal
# and are excluded from wall matching.
VERTICAL_TOL = 0.3   # ~17° from vertical

# Small inset applied to hole rings so they are strictly inside the polygon.
HOLE_PAD = 0.005   # metres

# Window / Door face is pushed this far along the outward wall normal so it
# sits slightly in front of the wall plane and avoids Z-fighting artefacts.
WINDOW_FACE_OFFSET = 0.01   # metres  (1 cm outward)


# =====================================================================
# Interactive file selector
# =====================================================================
def select_file(directory, pattern="*.las"):
    """List files in *directory* and let the user pick one."""
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        print(f"  No {pattern} files found in {directory}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Files in: {directory}")
    print(f"{'='*60}")
    for i, f in enumerate(files):
        size_mb = os.path.getsize(f) / 1_048_576
        print(f"  [{i+1}] {os.path.basename(f):50s} ({size_mb:.1f} MB)")
    print()

    while True:
        try:
            choice = int(input(f"  Select file [1-{len(files)}]: ").strip())
            if 1 <= choice <= len(files):
                return files[choice - 1]
        except (ValueError, KeyboardInterrupt):
            pass
        print("  Invalid choice, try again.")


# =====================================================================
# Interactive feature-type selector
# =====================================================================
FEATURE_TYPES = [
    ("window", "Window"),
    ("door",   "Door"),
]

def prompt_feature_type():
    """Ask the user which feature type the loaded cloud represents."""
    print("\n  CityJSON feature type for this point cloud:")
    for i, (name, cityjson_type) in enumerate(FEATURE_TYPES):
        print(f"    [{i+1}] {name:8s}  ({cityjson_type})")
    while True:
        try:
            idx = int(input(f"  Select [1-{len(FEATURE_TYPES)}]: ").strip()) - 1
            if 0 <= idx < len(FEATURE_TYPES):
                return FEATURE_TYPES[idx][0]
        except (ValueError, KeyboardInterrupt):
            pass
        print("  Invalid choice, try again.")


# =====================================================================
# CityJSON vertex helpers
# =====================================================================
def decode_vertices(cm):
    """
    Return an (N, 3) float64 array of real-world vertices from a CityJSON
    dict, applying the stored transform (scale × int_vertex + translate).
    """
    raw       = np.array(cm["vertices"], dtype=np.float64)
    t         = cm.get("transform", {})
    scale     = np.array(t.get("scale",     [1, 1, 1]), dtype=np.float64)
    translate = np.array(t.get("translate", [0, 0, 0]), dtype=np.float64)
    return raw * scale + translate


def encode_vertex(pt, scale, translate):
    """Convert a real-world 3-D point back to a CityJSON integer vertex list."""
    return [round((pt[i] - translate[i]) / scale[i]) for i in range(3)]


# =====================================================================
# Parse vertical surfaces from a CityJSON model
# =====================================================================
def parse_vertical_surfaces(cm, world_verts):
    """
    Walk every geometry in every CityObject and collect polygons whose
    outward normal is approximately horizontal (i.e. the surface is
    near-vertical), regardless of their semantic label.

    This is more robust than relying on WallSurface semantics, which may be
    missing or mislabelled in exported LOD2 models.

    Returns a list of dicts, one per qualifying polygon:

        {
            'idx':        int          – sequential index (used for logging)
            'obj_id':     str          – parent CityObject key
            'geom_idx':   int          – geometry index inside obj['geometry']
            'shell_idx':  int          – shell index (0 for non-Solid)
            'poly_idx':   int          – polygon index inside the shell
            'coords':     ndarray(M,3) – real-world exterior ring vertices
            'normal_2d':  ndarray(2,)  – XY outward unit normal
            'origin_2d':  ndarray(2,)  – ring centroid XY
            'z_min':      float
            'z_max':      float
            'xy_min':     ndarray(2,)
            'xy_max':     ndarray(2,)
        }
    """
    surfaces = []

    for obj_id, obj in cm.get("CityObjects", {}).items():
        # Skip child feature objects (Window, Door, BuildingInstallation, etc.)
        # that were created by a previous run — they have no wall geometry and
        # their (tiny) surfaces would become spurious wall candidates.
        obj_type = obj.get("type", "")
        if obj_type in ("Window", "Door", "BuildingInstallation",
                        "OtherConstruction"):
            continue

        geoms = obj.get("geometry", [])
        if not geoms:
            continue

        # Only process the HIGHEST-LOD geometry from each object so that
        # objects with multiple LOD representations (e.g. LOD1 + LOD2) do
        # not contribute duplicate surfaces that remain unholed after
        # processing and visually overlap the modified geometry.
        def _lod_val(g):
            try:
                return float(g.get("lod") or 0)
            except (ValueError, TypeError):
                return 0.0

        best_idx  = max(range(len(geoms)), key=lambda i: _lod_val(geoms[i]))
        g_idx_set = [(best_idx, geoms[best_idx])]

        for g_idx, geom in g_idx_set:
            boundaries = geom.get("boundaries", [])
            geom_type  = geom.get("type", "")

            if geom_type == "Solid":
                shell_iter = enumerate(boundaries)
            else:
                shell_iter = [(0, boundaries)]

            for s_idx, shell in shell_iter:
                for p_idx, polygon in enumerate(shell):
                    # Exterior ring = polygon[0]
                    try:
                        ext_ring = polygon[0]
                        coords   = world_verts[np.array(ext_ring)]   # (M, 3)
                    except (IndexError, KeyError, TypeError):
                        continue

                    if len(coords) < 3:
                        continue

                    # Newell's method → polygon normal
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
                    # Skip near-horizontal surfaces
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

    print(f"  Parsed {len(surfaces)} vertical surface(s) from CityJSON.")
    if not surfaces:
        print("  WARNING: no vertical surfaces found.")
    return surfaces


# =====================================================================
# DBSCAN clustering
# =====================================================================
def cluster_features(points, user_ftype, eps, min_samples):
    """
    Run DBSCAN on *points*.  Returns a list of dicts:
        { 'type': str, 'points': ndarray(N,3), 'n_points': int }
    sorted largest-first.
    """
    print(f"  Running DBSCAN ({len(points):,} pts, "
          f"eps={eps}, min_samples={min_samples}) ...")
    labels        = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    unique_labels = sorted(set(labels) - {-1})
    n_noise       = int((labels == -1).sum())
    print(f"    → {len(unique_labels)} clusters, {n_noise:,} noise points discarded")

    feats = []
    for cid in unique_labels:
        mask = labels == cid
        feats.append({
            "type":     user_ftype,
            "points":   points[mask],
            "n_points": int(mask.sum()),
        })
    feats.sort(key=lambda f: f["n_points"], reverse=True)
    return feats


# =====================================================================
# 3-stage wall matching  (mirrors find_wall_normal_for_cluster in .py)
# =====================================================================
def _pca_normal_2d(points):
    if len(points) < 3:
        return None
    xy  = points[:, :2]
    cxy = xy - xy.mean(axis=0)
    cov = cxy.T @ cxy
    _, eig_vecs = np.linalg.eigh(cov)
    tang = eig_vecs[:, 1]
    return np.array([-tang[1], tang[0]])


def find_matched_surface(points, vert_surfaces, dist_tol=2.0, z_expand=0.5):
    """
    3-stage matching — returns (matched_surface | None).

    Stage 1 – Spatial intersection (Z-overlap + signed centroid distance
              + XY footprint check).  Picks smallest |distance|.
    Stage 2 – PCA normal alignment fallback.
    Stage 3 – Nearest-centroid unconditional fallback.
    """
    if not vert_surfaces:
        return None

    cl_z_min = float(points[:, 2].min())
    cl_z_max = float(points[:, 2].max())
    centroid  = points.mean(axis=0)
    cxy       = centroid[:2]

    # Stage 1
    best_surf = None
    best_dist = float("inf")
    for vs in vert_surfaces:
        if cl_z_max < vs["z_min"] - z_expand or cl_z_min > vs["z_max"] + z_expand:
            continue
        n2d = vs["normal_2d"]
        d   = float(np.dot(cxy - vs["origin_2d"], n2d))
        if abs(d) >= dist_tol:
            continue
        proj_xy = cxy - d * n2d
        pad = dist_tol
        if (proj_xy[0] < vs["xy_min"][0] - pad or
                proj_xy[0] > vs["xy_max"][0] + pad or
                proj_xy[1] < vs["xy_min"][1] - pad or
                proj_xy[1] > vs["xy_max"][1] + pad):
            continue
        if abs(d) < best_dist:
            best_dist = abs(d)
            best_surf = vs

    if best_surf is not None:
        angle = np.degrees(np.arctan2(best_surf["normal_2d"][1],
                                      best_surf["normal_2d"][0]))
        print(f"[Stage1: dist={best_dist:.2f} m, angle={angle:+.1f}°]", end=" ")
        return best_surf

    # Stage 2 – PCA normal alignment
    pca_n = _pca_normal_2d(points)
    if pca_n is not None:
        best_dot = -1.0
        for vs in vert_surfaces:
            dot = abs(float(np.dot(pca_n, vs["normal_2d"])))
            if dot > best_dot:
                best_dot  = dot
                best_surf = vs
        if best_surf is not None:
            angle = np.degrees(np.arctan2(best_surf["normal_2d"][1],
                                          best_surf["normal_2d"][0]))
            print(f"[Stage2: align={best_dot:.3f}, angle={angle:+.1f}°]", end=" ")
            return best_surf

    # Stage 3 – nearest centroid
    for vs in vert_surfaces:
        d = abs(float(np.dot(cxy - vs["origin_2d"], vs["normal_2d"])))
        if d < best_dist:
            best_dist = d
            best_surf = vs
    if best_surf is not None:
        print("[Stage3: nearest-centroid fallback]", end=" ")
    return best_surf


# =====================================================================
# Coplanar rectangle projection onto a vertical surface
# =====================================================================
def compute_surface_projection(points, surface):
    """
    Project the cluster's lateral and vertical extent onto *surface*,
    producing a flat rectangle coplanar with the surface.

    Mirrors compute_wall_projection() from 10_trial_openings.py.

    Returns a dict:
        {
            'ring':       list of 5 (x,y,z) tuples – CCW closed rectangle
            'n2d':        ndarray(2,)
            'wall_depth': float
            'txy':        ndarray(2,)
        }
    or an empty dict if the projection is degenerate after clipping.
    """
    n2d        = surface["normal_2d"]
    wall_depth = float(np.dot(surface["origin_2d"], n2d))

    z_up    = np.array([0.0, 0.0, 1.0])
    wall_n3 = np.array([n2d[0], n2d[1], 0.0])
    tangent = np.cross(z_up, wall_n3)
    t_norm  = np.linalg.norm(tangent)
    if t_norm < 1e-9:
        tangent = np.array([1.0, 0.0, 0.0])
    else:
        tangent /= t_norm
    txy = tangent[:2]

    t_pts = points[:, :2] @ txy
    z_pts = points[:, 2]
    t_min_cl, t_max_cl = float(t_pts.min()), float(t_pts.max())
    z_min_cl, z_max_cl = float(z_pts.min()), float(z_pts.max())

    wall_t = surface["coords"][:, :2] @ txy
    t_min  = max(t_min_cl, float(wall_t.min()))
    t_max  = min(t_max_cl, float(wall_t.max()))
    z_min  = max(z_min_cl, surface["z_min"])
    z_max  = min(z_max_cl, surface["z_max"])

    if t_max <= t_min or z_max <= z_min:
        return {}

    def wpt(t, z):
        p_xy = wall_depth * n2d + t * txy
        return (float(p_xy[0]), float(p_xy[1]), float(z))

    ring = [
        wpt(t_min, z_min),
        wpt(t_max, z_min),
        wpt(t_max, z_max),
        wpt(t_min, z_max),
        wpt(t_min, z_min),   # close
    ]
    return {"ring": ring, "n2d": n2d, "wall_depth": wall_depth, "txy": txy}


# =====================================================================
# NMS deduplication of overlapping projected openings
# =====================================================================
def _proj_rect(proj):
    """
    Return (t_lo, t_hi, z_lo, z_hi) for a projection dict in the wall's
    local (tangent, Z) coordinate system.
    """
    txy  = proj["txy"]
    ring = proj["ring"][:-1]   # 4 open vertices
    t_vals = [float(np.array(pt[:2]) @ txy) for pt in ring]
    z_vals = [pt[2]             for pt in ring]
    return min(t_vals), max(t_vals), min(z_vals), max(z_vals)


def _rect_iou(a, b):
    """Intersection-over-Union of two (t_lo, t_hi, z_lo, z_hi) rectangles."""
    t_lo = max(a[0], b[0]);  t_hi = min(a[1], b[1])
    z_lo = max(a[2], b[2]);  z_hi = min(a[3], b[3])
    inter = max(0.0, t_hi - t_lo) * max(0.0, z_hi - z_lo)
    if inter == 0.0:
        return 0.0
    area_a = (a[1] - a[0]) * (a[3] - a[2])
    area_b = (b[1] - b[0]) * (b[3] - b[2])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def deduplicate_openings(openings, iou_threshold=0.3):
    """
    Greedy NMS on a list of (proj, ftype, feat_id) tuples.

    Steps
    -----
    1. For each opening extract its 2-D (t, z) rectangle from the proj dict.
    2. Sort by area — largest first, so big windows are kept preferentially.
    3. Greedily accept each opening unless it overlaps any already-accepted
       one by more than *iou_threshold* (default 0.3 = 30 %% overlap).

    Returns the filtered list in the same (proj, ftype, feat_id) format.
    """
    if len(openings) <= 1:
        return openings

    items = []
    for opening in openings:
        proj, ftype, feat_id = opening
        rect = _proj_rect(proj)
        area = (rect[1] - rect[0]) * (rect[3] - rect[2])
        items.append((area, rect, opening))

    # Sort largest-area first
    items.sort(key=lambda x: x[0], reverse=True)

    kept         = []
    kept_rects   = []
    for area, rect, opening in items:
        if not any(_rect_iou(rect, kr) > iou_threshold for kr in kept_rects):
            kept.append(opening)
            kept_rects.append(rect)

    return kept

def punch_hole_and_create_opening(cm, int_verts, scale, translate,
                                  surface, proj, ftype, feat_id):
    """
    1. Compute an inset hole ring from *proj* (4 open vertices).
    2. Encode the 4 ring points and add them to *int_verts*.
    3. Append the interior ring to the matched surface polygon in *cm*.
    4. Create a child Window / Door CityObject with the coplanar rectangle
       as its lod3MultiSurface (a single 4-vertex polygon, closed).
    5. Link child ↔ parent.
    6. Upgrade the matched surface's geometry LOD to "3".

    Returns the hole_indices list (for logging / testing).
    """
    ring  = proj["ring"]   # 5 points (closed), CCW
    n2d_h = proj["n2d"]
    txy_h = proj["txy"]
    wd_h  = proj["wall_depth"]
    pad   = HOLE_PAD

    body   = ring[:-1]   # 4 open vertices
    t_vals = [float(np.array(pt[:2]) @ txy_h) for pt in body]
    z_vals = [pt[2] for pt in body]
    t_lo   = min(t_vals) + pad
    t_hi   = max(t_vals) - pad
    z_lo   = min(z_vals) + pad
    z_hi   = max(z_vals) - pad

    if t_hi <= t_lo or z_hi <= z_lo:
        t_lo, t_hi = min(t_vals), max(t_vals)
        z_lo, z_hi = min(z_vals), max(z_vals)

    def hwpt(t, z):
        p_xy = wd_h * n2d_h + t * txy_h
        return (float(p_xy[0]), float(p_xy[1]), float(z))

    # CW interior ring (hole) when viewed from outside
    hole_pts = [hwpt(t_lo, z_hi), hwpt(t_hi, z_hi),
                hwpt(t_hi, z_lo), hwpt(t_lo, z_lo)]

    start_idx    = len(int_verts)
    for pt in hole_pts:
        int_verts.append(encode_vertex(pt, scale, translate))
    hole_indices = list(range(start_idx, start_idx + len(hole_pts)))

    # ── Punch hole into the matched polygon ──────────────────────────
    obj_id    = surface["obj_id"]
    g_idx     = surface["geom_idx"]
    s_idx     = surface["shell_idx"]
    p_idx     = surface["poly_idx"]

    geom      = cm["CityObjects"][obj_id]["geometry"][g_idx]
    geom_type = geom["type"]

    if geom_type == "Solid":
        target_polygon = geom["boundaries"][s_idx][p_idx]
    else:
        target_polygon = geom["boundaries"][p_idx]

    # Append interior ring (CityJSON: polygon = [ext, hole1, hole2, ...])
    target_polygon.append(hole_indices)

    # Upgrade the parent geometry LOD to 3
    geom["lod"] = "3"

    # ── Window / Door ring for the opening geometry ───────────────────
    # CCW ring (matches exterior convention): BL→BR→TR→TL + close
    # Push the window/door face outward by WINDOW_FACE_OFFSET so it sits
    # slightly in front of the wall plane, preventing Z-fighting artefacts.
    ox = float(n2d_h[0]) * WINDOW_FACE_OFFSET
    oy = float(n2d_h[1]) * WINDOW_FACE_OFFSET

    def hwpt_win(t, z):
        """Wall point offset outward for the opening face."""
        p_xy = wd_h * n2d_h + t * txy_h
        return (float(p_xy[0]) + ox, float(p_xy[1]) + oy, float(z))

    open_pts = [hwpt_win(t_lo, z_lo), hwpt_win(t_hi, z_lo),
                hwpt_win(t_hi, z_hi), hwpt_win(t_lo, z_hi)]
    open_start = len(int_verts)
    for pt in open_pts:
        int_verts.append(encode_vertex(pt, scale, translate))
    open_indices = list(range(open_start, open_start + len(open_pts)))

    # ── Create child CityObject ───────────────────────────────────────
    cityjson_type = "Window" if ftype == "window" else "Door"
    cm["CityObjects"][feat_id] = {
        "type":    cityjson_type,
        "parents": [obj_id],
        "geometry": [{
            "type":       "MultiSurface",
            "lod":        "3",
            "boundaries": [[open_indices]],
            "semantics": {
                "surfaces": [{"type": cityjson_type}],
                "values":   [0],
            },
        }],
    }

    # Link child ↔ parent
    parent_obj = cm["CityObjects"][obj_id]
    parent_obj.setdefault("children", []).append(feat_id)

    return hole_indices


# =====================================================================
# Main
# =====================================================================
def main():
    # =====================================================================
    # HPC HARDCODED INPUTS
    # =====================================================================
    HARDCODED_JSON_PATH = "/home/khalil.torneros/ICHEM/ichem_021726.json"
    HARDCODED_LAS_PATH  = "/home/khalil.torneros/ICHEM/ICHEM-window.las"
    HARDCODED_FEATURE_TYPE = "window"
    # =====================================================================

    parser = argparse.ArgumentParser(
        description="Facade Features → CityJSON LOD3")
    parser.add_argument("--eps",        type=float, default=3.0,
                        help="DBSCAN eps (neighbourhood radius, default 0.3)")
    parser.add_argument("--min_samples", type=int,  default=30,
                        help="DBSCAN min_samples (default 30)")
    parser.add_argument("--output", "-o", type=str,  default=None,
                        help="Output filename (saved in outputs/13_openings_json/). "
                             "Defaults to <source_basename>_LOD3.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  Facade Features → CityJSON LOD3")
    print("=" * 60)

    # ── 1. Select target CityJSON model ───────────────────────────────
    json_path = HARDCODED_JSON_PATH
    print(f"\n  [HPC MODE] Using hardcoded CityJSON: {json_path}")

    print(f"  Loading {os.path.basename(json_path)} ...")
    with open(json_path, "r", encoding="utf-8") as fh:
        cm = json.load(fh)

    world_verts = decode_vertices(cm)
    transform   = cm.get("transform", {})
    scale       = np.array(transform.get("scale",     [1, 1, 1]), dtype=np.float64)
    translate   = np.array(transform.get("translate", [0, 0, 0]), dtype=np.float64)
    int_verts   = [list(v) for v in cm["vertices"]]   # mutable working copy

    print(f"\n  Detecting vertical surfaces ...")
    vert_surfaces = parse_vertical_surfaces(cm, world_verts)
    if not vert_surfaces:
        print("  ERROR: No vertical surfaces found. Aborting.")
        sys.exit(1)

    # ── 2. Collect point cloud inputs ─────────────────────────────────
    las_inputs = [(HARDCODED_LAS_PATH, HARDCODED_FEATURE_TYPE)]

    print(f"\n  [HPC MODE] Using hardcoded point cloud:")
    for lp, ft in las_inputs:
        print(f"    {os.path.basename(lp):50s} → {ft}")

    # ── 3. DBSCAN + wall projection ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CLUSTERING + SURFACE PROJECTION  "
          f"(eps={args.eps}, min_samples={args.min_samples})")
    print(f"{'='*60}")

    # surface_opening_map[surf_idx] = {
    #     'surface': vert_surface_dict,
    #     'openings': [(proj_dict, ftype, feat_id), ...]
    # }
    surface_opening_map = {}
    summary             = {}
    feat_counter        = 0

    for pc_idx, (las_path, user_ftype) in enumerate(las_inputs):
        print(f"\n  [{pc_idx+1}/{len(las_inputs)}] "
              f"{os.path.basename(las_path)}  [{user_ftype}]")

        las    = laspy.read(las_path)
        points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
        print(f"    Points loaded: {len(points):,}")

        feats = cluster_features(points, user_ftype,
                                 eps=args.eps, min_samples=args.min_samples)
        if not feats:
            print("    No clusters found — skipping this cloud.")
            continue

        print(f"    Clusters found: {len(feats)}")

        for feat in feats:
            feat_counter += 1
            ftype  = feat["type"]
            n_pts  = feat["n_points"]
            print(f"\n    [{feat_counter}] {ftype:8s}  {n_pts:>8,} pts  ", end="")

            # 3-stage surface matching
            matched = find_matched_surface(feat["points"], vert_surfaces,
                                           dist_tol=2.0, z_expand=0.5)

            if matched is None:
                # No vertical surface at all — skip (rare)
                print("→ no surface found, skipped")
                continue

            # Try nearest-wall fallback if Stage 1/2 failed (matched still None
            # inside find_matched_surface already handles this via Stage 3).

            proj = compute_surface_projection(feat["points"], matched)
            if not proj:
                print("→ projection degenerate, skipped")
                continue

            sidx    = matched["idx"]
            feat_id = f"{ftype}_{feat_counter}_{uuid.uuid4().hex[:8]}"

            if sidx not in surface_opening_map:
                surface_opening_map[sidx] = {
                    "surface":  matched,
                    "openings": [],
                }
            surface_opening_map[sidx]["openings"].append(
                (proj, ftype, feat_id))

            summary[ftype] = summary.get(ftype, 0) + 1
            wall_angle = np.degrees(np.arctan2(
                matched["normal_2d"][1], matched["normal_2d"][0]))
            print(f"→ projected onto surface {sidx} "
                  f"(obj '{matched['obj_id']}', angle={wall_angle:+.1f}°)")

    if not surface_opening_map:
        print("\n  No openings created. Nothing to write.")
        return

    # ── 4. Apply holes + create opening CityObjects ────────────────────
    total_raw      = sum(len(d["openings"]) for d in surface_opening_map.values())
    print(f"\n  Deduplicating overlapping openings (IoU > 0.30) ...")
    n_suppressed = 0
    for data in surface_opening_map.values():
        raw      = data["openings"]
        deduped  = deduplicate_openings(raw, iou_threshold=0.30)
        dropped  = len(raw) - len(deduped)
        n_suppressed += dropped
        data["openings"] = deduped
    total_kept = total_raw - n_suppressed
    print(f"  {total_raw} raw → {total_kept} kept  "
          f"({n_suppressed} overlapping opening(s) suppressed)")

    print(f"\n  Applying {total_kept} opening(s) to "
          f"{len(surface_opening_map)} surface(s) ...")

    for sidx, data in surface_opening_map.items():
        surf     = data["surface"]
        n_before = len(data["openings"])
        for proj, ftype, feat_id in data["openings"]:
            punch_hole_and_create_opening(
                cm, int_verts, scale, translate,
                surf, proj, ftype, feat_id)
        print(f"  Surface {sidx} (obj '{surf['obj_id']}'): "
              f"{n_before} opening(s) punched.")

    # ── 5. Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FEATURE SUMMARY")
    print(f"{'='*60}")
    for ftype, count in sorted(summary.items()):
        cityjson = "Window" if ftype == "window" else "Door"
        print(f"    {ftype:10s} → {cityjson:10s}  × {count}")
    print(f"    Surfaces modified: {len(surface_opening_map)}")

    # ── 6. Save ────────────────────────────────────────────────────────
    cm["vertices"] = int_verts

    if args.output:
        out_name = (args.output if args.output.endswith(".json")
                    else args.output + ".json")
    else:
        src_base = os.path.splitext(os.path.basename(json_path))[0]
        out_name = f"{src_base}_LOD3.json"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, out_name)

    print(f"\n  Saving → {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(cm, fh, separators=(",", ":"))

    size_mb = os.path.getsize(output_path) / 1_048_576
    print(f"  ✓ Saved: {output_path}  ({size_mb:.1f} MB)")
    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()