#!/usr/bin/env python3
"""
Facade Features → CityJSON LOD3 Pipeline  (with curved-geometry tessellation).

Mirrors the logic of 11_curve_handling.py but operates on CityJSON (.json)
models instead of CityGML (.gml) files.

Each run selects one CityJSON model and one or more .las point clouds.  For
every point cloud the user manually chooses the CityGML feature type (Window,
Door, or BuildingInstallation).  All clouds are tessellated into curved slabs.
Windows and Doors punch coplanar holes into the matched LOD2 wall polygons and
are simultaneously created as child Window / Door CityObjects.
BuildingInstallations are appended as children without modifying the parent
wall geometry.

Usage:
    conda activate las-env
    python scripts/LOD2toLOD3/11_curve_handling_json.py [options]

Options:
    --n_slabs         Tessellation slabs per wall sub-cluster (default: 20)
    --slab_thickness  Half-depth of each slab in wall-normal direction, metres (default: 0.15)
    --min_pts_slab    Minimum points needed for a slab to be emitted (default: 5)
    --output, -o      Output JSON filename (default: <json_basename>_LOD3_curved.json)
"""

import os
import sys
import glob
import json
import uuid
import argparse

import numpy as np
import laspy


# =====================================================================
# Constants
# =====================================================================
INPUT_DIR  = "outputs/11A_facade_curve"
JSON_DIR   = "outputs/14_extrusions_json"
OUTPUT_DIR = "outputs/12_curve_json"

# Small inset applied to hole rings so they are strictly inside the polygon.
HOLE_PAD = 0.005   # metres

# Window / Door face is pushed this far along the outward wall normal so it
# sits slightly in front of the wall plane and avoids Z-fighting artefacts.
WINDOW_FACE_OFFSET = 0.01   # metres (1 cm outward)

# Vertical-surface detection tolerance:
# surfaces whose |n_z / |n|| > VERTICAL_TOL are considered near-horizontal
# and are excluded from wall matching.
VERTICAL_TOL = 0.3   # ~17° from vertical


# =====================================================================
# Interactive file selector
# =====================================================================
def select_file(directory, pattern="*.las"):
    """List files in directory and let user pick one."""
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
            choice = input(f"  Select file [1-{len(files)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]
        except (ValueError, KeyboardInterrupt):
            pass
        print("  Invalid choice, try again.")


# =====================================================================
# Interactive feature-type selector
# =====================================================================
FEATURE_TYPES = [
    ('window', 'Window'),
    ('door',   'Door'),
    ('other',  'BuildingInstallation'),
]

def prompt_feature_type():
    """Ask the user which CityJSON feature type the loaded point cloud represents."""
    print("\n  CityJSON feature type for this point cloud:")
    for i, (name, tag) in enumerate(FEATURE_TYPES):
        print(f"    [{i+1}] {name:8s}  ({tag})")
    while True:
        try:
            choice = input(f"  Select [1-{len(FEATURE_TYPES)}]: ").strip()
            idx = int(choice) - 1
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
    outward normal is approximately horizontal (i.e. near-vertical surface),
    regardless of their semantic label.

    Returns a list of dicts, one per qualifying polygon:
        {
            'idx':        int          – sequential index
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
        # Skip child feature objects that were created by a previous run
        obj_type = obj.get("type", "")
        if obj_type in ("Window", "Door", "BuildingInstallation",
                        "OtherConstruction"):
            continue

        geoms = obj.get("geometry", [])
        if not geoms:
            continue

        # Only process the HIGHEST-LOD geometry from each object
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
# Wall matching (3-stage, mirrors 11_curve_handling.py)
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


def _pca_wall_normal(points):
    n2d = _pca_normal_2d(points)
    if n2d is None:
        return None
    return np.array([n2d[0], n2d[1], 0.0])


def find_matched_surface(points, vert_surfaces, dist_tol=2.0, z_expand=0.5):
    """
    3-stage wall matching — returns (matched_surface | None).

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
        best_dot  = -1.0
        best_surf = None
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
    best_surf = None
    best_dist = float("inf")
    for vs in vert_surfaces:
        d = abs(float(np.dot(cxy - vs["origin_2d"], vs["normal_2d"])))
        if d < best_dist:
            best_dist = d
            best_surf = vs
    if best_surf is not None:
        print("[Stage3: nearest-centroid fallback]", end=" ")
    return best_surf


# =====================================================================
# Split points by closest vertical surface (pure geometry)
# =====================================================================
def split_by_surface(points, vert_surfaces, dist_tol=2.0, min_pts=5):
    """
    Assign every point to the closest vertical surface by signed distance,
    then discard any point whose absolute distance to its assigned surface
    exceeds *dist_tol* metres.  Sub-clusters with fewer than *min_pts*
    surviving points are also dropped.

    This prevents features from being created on surfaces that have no
    actual points nearby (i.e. the point cloud has no data in that area).
    """
    if not vert_surfaces:
        return []

    pts_xy  = points[:, :2]
    n_surfs = len(vert_surfaces)

    dist_matrix = np.full((len(points), n_surfs), np.inf, dtype=np.float64)
    pad = dist_tol

    for j, vs in enumerate(vert_surfaces):
        # Only consider points within the padded bounding box of the surface
        in_bounds = (
            (points[:, 0] >= vs["xy_min"][0] - pad) &
            (points[:, 0] <= vs["xy_max"][0] + pad) &
            (points[:, 1] >= vs["xy_min"][1] - pad) &
            (points[:, 1] <= vs["xy_max"][1] + pad) &
            (points[:, 2] >= vs["z_min"] - pad) &
            (points[:, 2] <= vs["z_max"] + pad)
        )
        if not in_bounds.any():
            continue
        
        pts_in_bounds = pts_xy[in_bounds]
        dist_matrix[in_bounds, j] = (pts_in_bounds - vs["origin_2d"]) @ vs["normal_2d"]

    assignments  = np.argmin(np.abs(dist_matrix), axis=1)
    min_abs_dist = np.abs(dist_matrix)[np.arange(len(points)), assignments]

    # Only keep points that are actually close to their assigned surface
    close_mask = min_abs_dist <= dist_tol

    sub_clusters = []
    for j, vs in enumerate(vert_surfaces):
        mask = (assignments == j) & close_mask
        if mask.sum() < min_pts:
            continue
        sub_pts = points[mask]
        wall_origin_depth = float(np.dot(vs["origin_2d"], vs["normal_2d"]))
        sub_clusters.append({
            "surface":           vs,
            "points":            sub_pts,
            "wall_origin_depth": wall_origin_depth,
        })

    return sub_clusters


# =====================================================================
# Wall Projection Helper for Holes
# =====================================================================
def compute_wall_projection(points, surface):
    """
    Project the cluster's lateral and vertical extent onto *surface*,
    producing a flat rectangle coplanar with the surface.

    Returns a dict:
        {
            'ring':       list of 5 (x,y,z) tuples – closed rectangle
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

    # wall_t = surface["coords"][:, :2] @ txy
    # t_min  = max(t_min_cl, float(wall_t.min()))
    # t_max  = min(t_max_cl, float(wall_t.max()))
    # z_min  = max(z_min_cl, surface["z_min"])
    # z_max  = min(z_max_cl, surface["z_max"])
    
    # if t_max <= t_min or z_max <= z_min:
    #     return {}

    #BINAGO
    wall_t = surface["coords"][:, :2] @ txy
    wall_t_min = float(wall_t.min())
    wall_t_max = float(wall_t.max())

    # Clamp the window to the wall's tangential extent.
    # If the point cloud overlaps the wall at all, clip to overlap.
    # If it doesn't overlap (points entirely outside), center it within the wall.
    t_min = max(t_min_cl, wall_t_min)
    t_max = min(t_max_cl, wall_t_max)
    if t_max <= t_min:
        # No overlap — place window at the point cloud center, clipped to wall width
        cl_center = (t_min_cl + t_max_cl) / 2.0
        cl_half   = (t_max_cl - t_min_cl) / 2.0
        t_min = max(cl_center - cl_half, wall_t_min)
        t_max = min(cl_center + cl_half, wall_t_max)

    z_min = max(z_min_cl, surface["z_min"])
    z_max = min(z_max_cl, surface["z_max"])
    if z_max <= z_min:
        z_min, z_max = z_min_cl, z_max_cl

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
# Tessellation — curved cluster → staircase of rectangular prisms
# =====================================================================
def tessellate_curved_cluster(points, wall_normal, n_slabs,
                               slab_thickness, min_pts_slab,
                               wall_depth=None, wall_coords=None):
    """
    Tessellate a point cloud cluster into a sequence of axis-aligned
    rectangular prisms ("slabs") arranged along the wall's tangent direction.

    Returns a list of prisms; each prism is a list of 6 polygon rings
    (one per face), where each ring is a list of (x, y, z) tuples (closed).
    """
    if len(points) < min_pts_slab:
        return []

    wn = np.asarray(wall_normal, dtype=np.float64)
    wn = wn / np.linalg.norm(wn)

    z_up    = np.array([0.0, 0.0, 1.0])
    tangent = np.cross(z_up, wn)
    t_norm  = np.linalg.norm(tangent)
    if t_norm < 1e-9:
        tangent = np.array([1.0, 0.0, 0.0])
    else:
        tangent = tangent / t_norm

    axes = np.column_stack([wn, tangent, z_up])
    projected = points @ axes

    pt_t_min = projected[:, 1].min()
    pt_t_max = projected[:, 1].max()

    if pt_t_max - pt_t_min < 1e-6:
        return []

    delta = (pt_t_max - pt_t_min) / n_slabs
    prism_list = []

    for i in range(n_slabs):
        t_lo = pt_t_min + i * delta
        t_hi = t_lo + delta

        if i < n_slabs - 1:
            mask = (projected[:, 1] >= t_lo) & (projected[:, 1] < t_hi)
        else:
            mask = (projected[:, 1] >= t_lo) & (projected[:, 1] <= t_hi)

        col_pts = projected[mask]
        if len(col_pts) < min_pts_slab:
            continue

        z_lo = float(col_pts[:, 2].min())
        z_hi = float(col_pts[:, 2].max())

        if z_hi - z_lo < 1e-4:
            continue

        if wall_depth is not None:
            depth_center = wall_depth
        else:
            depth_center = float(col_pts[:, 0].mean())

        depth_lo = depth_center - slab_thickness
        depth_hi = depth_center + slab_thickness

        box_t_lo = t_lo
        box_t_hi = t_hi

        if wall_coords is not None:
            proj_wall  = wall_coords @ axes
            wall_t_min = proj_wall[:, 1].min()
            wall_t_max = proj_wall[:, 1].max()

            safe_min = wall_t_min
            safe_max = wall_t_max

            if safe_max <= safe_min:
                mid = (wall_t_min + wall_t_max) / 2.0
                safe_min = mid - 0.01
                safe_max = mid + 0.01

            box_t_lo = max(box_t_lo, safe_min)
            box_t_hi = min(box_t_hi, safe_max)

            if box_t_hi <= box_t_lo:
                center   = (box_t_lo + box_t_hi) / 2.0
                box_t_lo = center - 0.05
                box_t_hi = center + 0.05

            box_t_lo = max(box_t_lo, safe_min)
            box_t_hi = min(box_t_hi, safe_max)

            if box_t_hi <= box_t_lo:
                center   = (box_t_lo + box_t_hi) / 2.0
                box_t_lo = center - 0.05
                box_t_hi = center + 0.05

        corners_local = np.array([
            [depth_lo, box_t_lo, z_lo], [depth_hi, box_t_lo, z_lo],
            [depth_hi, box_t_hi, z_lo], [depth_lo, box_t_hi, z_lo],
            [depth_lo, box_t_lo, z_hi], [depth_hi, box_t_lo, z_hi],
            [depth_hi, box_t_hi, z_hi], [depth_lo, box_t_hi, z_hi],
        ])

        corners_world = corners_local @ axes.T

        face_indices = [
            [0, 3, 2, 1], [4, 5, 6, 7],
            [0, 1, 5, 4], [3, 7, 6, 2],
            [0, 4, 7, 3], [1, 2, 6, 5],
        ]

        faces = []
        for fi in face_indices:
            ring = [tuple(corners_world[k]) for k in fi]
            ring.append(ring[0])
            faces.append(ring)

        prism_list.append(faces)

    return prism_list


# =====================================================================
# CityJSON geometry helpers
# =====================================================================
def _add_polygon_to_cityjson(int_verts, scale, translate, ring_world):
    """
    Encode a list of (x, y, z) real-world points as a CityJSON polygon
    (exterior ring only).  Returns integer index list.
    """
    indices = []
    for pt in ring_world[:-1]:   # strip closing duplicate
        int_verts.append(encode_vertex(pt, scale, translate))
        indices.append(len(int_verts) - 1)
    return indices


def punch_hole_and_create_opening(cm, int_verts, scale, translate,
                                  surface, proj, ftype, feat_id):
    """
    1. Compute an inset hole ring from *proj* (4 open vertices).
    2. Encode the 4 ring points and add them to *int_verts*.
    3. Append the interior ring to the matched surface polygon in *cm*.
    4. Create a child Window / Door CityObject with the coplanar rectangle
       as its lod3MultiSurface (a single 4-vertex polygon, closed).
    5. Link child ↔ parent.
    6. Upgrade the matched surface geometry LOD to "3".

    Returns the hole_indices list (for logging / testing).
    """
    ring  = proj["ring"]   # 5 points (closed)
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

    start_idx = len(int_verts)
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
    # Push the window/door face outward by WINDOW_FACE_OFFSET to avoid Z-fighting
    ox = float(n2d_h[0]) * WINDOW_FACE_OFFSET
    oy = float(n2d_h[1]) * WINDOW_FACE_OFFSET

    def hwpt_win(t, z):
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


def create_installation_from_faces(cm, int_verts, scale, translate,
                                   obj_id, feat_id, all_faces):
    """
    Encode all slab faces as a MultiSurface BuildingInstallation CityObject
    and link it to the parent building CityObject.

    *all_faces* is the flat list of polygon rings from tessellate_curved_cluster.
    """
    boundaries = []
    for ring_world in all_faces:
        indices = _add_polygon_to_cityjson(int_verts, scale, translate, ring_world)
        boundaries.append([indices])

    cm["CityObjects"][feat_id] = {
        "type":    "BuildingInstallation",
        "parents": [obj_id],
        "geometry": [{
            "type":       "MultiSurface",
            "lod":        "3",
            "boundaries": boundaries,
            "semantics": {
                "surfaces": [{"type": "WallSurface"}] * len(boundaries),
                "values":   list(range(len(boundaries))),
            },
        }],
    }

    parent_obj = cm["CityObjects"][obj_id]
    parent_obj.setdefault("children", []).append(feat_id)


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Facade Features → CityJSON LOD3 (curved geometry via tessellation)")
    parser.add_argument("--n_slabs",        type=int,   default=20,
                        help="Number of tessellation slabs per wall sub-cluster")
    parser.add_argument("--slab_thickness", type=float, default=0.15,
                        help="Half-depth of each slab in the wall-normal direction.")
    parser.add_argument("--min_pts_slab",   type=int,   default=5,
                        help="Minimum points required for a slab to be emitted")
    parser.add_argument("--output", "-o",   type=str,   default=None,
                        help="Output JSON filename")
    args = parser.parse_args()

    print("=" * 60)
    print("  Facade Features → CityJSON LOD3  [curved tessellation]")
    print("=" * 60)

    # ── 1. Select target CityJSON model ───────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SELECT TARGET CITYJSON MODEL")
    print(f"{'='*60}")
    json_path = select_file(JSON_DIR, "*.json")

    print(f"\n  Loading {os.path.basename(json_path)} ...")
    with open(json_path, "r", encoding="utf-8") as fh:
        cm = json.load(fh)

    world_verts = decode_vertices(cm)
    transform   = cm.get("transform", {})
    scale       = np.array(transform.get("scale",     [1, 1, 1]), dtype=np.float64)
    translate   = np.array(transform.get("translate", [0, 0, 0]), dtype=np.float64)
    int_verts   = [list(v) for v in cm["vertices"]]  # mutable working copy

    print(f"\n  Detecting vertical surfaces ...")
    vert_surfaces = parse_vertical_surfaces(cm, world_verts)
    if not vert_surfaces:
        print("  ERROR: No vertical surfaces found. Aborting.")
        sys.exit(1)

    # ── 2. Collect point cloud inputs ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SELECT POINT CLOUD INPUT(S)")
    print(f"{'='*60}")
    print("  You can add multiple point clouds; all will be processed together.")

    las_inputs = []
    while True:
        las_path     = select_file(INPUT_DIR, "*.las")
        feature_type = prompt_feature_type()
        las_inputs.append((las_path, feature_type))
        print(f"\n  ✓ Added: {os.path.basename(las_path):50s} → {feature_type}")
        again = input("\n  Add another point cloud? [y/N]: ").strip().lower()
        if again != "y":
            break

    print(f"\n  {len(las_inputs)} point cloud(s) selected:")
    for lp, ft in las_inputs:
        print(f"    {os.path.basename(lp):50s} → {ft}")

    # ── 3. Tessellation + wall projection ─────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TESSELLATION + WALL HOLE PROJECTION")
    print(f"  (n_slabs={args.n_slabs}, slab_thickness={args.slab_thickness})")
    print(f"{'='*60}")

    # surface_opening_map[surf_idx] = {
    #     'surface': vert_surface_dict,
    #     'openings': [(proj_dict, ftype, feat_id, all_faces), ...]
    # }
    surface_opening_map = {}
    fallback_feats      = []   # (ftype, feat_id, all_faces, obj_id) for installations
    summary             = {}
    feat_counter        = 0

    for pc_idx, (las_path, user_ftype) in enumerate(las_inputs):
        print(f"\n  [{pc_idx+1}/{len(las_inputs)}] {os.path.basename(las_path)}  [{user_ftype}]")

        las    = laspy.read(las_path)
        points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
        print(f"    Points loaded: {len(points):,}")

        if vert_surfaces:
            sub_clusters = split_by_surface(
                points, vert_surfaces,
                dist_tol=2.0,
                min_pts=args.min_pts_slab,
            )
            print(f"    → split into {len(sub_clusters)} wall sub-cluster(s) "
                  f"(within 2.0 m of surface, min {args.min_pts_slab} pts)")
        else:
            wall_n = _pca_wall_normal(points)
            if wall_n is None:
                continue
            sub_clusters = [{"surface": None, "points": points, "wall_origin_depth": None}]

        for sc_idx, sc in enumerate(sub_clusters):
            sc_pts     = sc["points"]
            wall_depth = sc["wall_origin_depth"]
            surf       = sc["surface"]

            if surf is not None:
                n2d    = surf["normal_2d"]
                wall_n = np.array([n2d[0], n2d[1], 0.0])
                angle  = np.degrees(np.arctan2(n2d[1], n2d[0]))
                print(f"    Wall {sc_idx+1}: {len(sc_pts):>7,} pts "
                      f"normal={angle:+.1f}° depth={wall_depth:.3f} ", end="")
            else:
                wall_n = _pca_wall_normal(sc_pts)
                if wall_n is None:
                    continue
                wall_depth = None
                print(f"    Wall {sc_idx+1}: {len(sc_pts):>7,} pts [PCA fallback] ", end="")

            if len(sc_pts) < args.min_pts_slab:
                print("→ too few points, skipped")
                continue

            wall_n_unit = wall_n / np.linalg.norm(wall_n)
            
            # Use 1 slab (whole cluster as one rectangular prism), matching
            # the original script's defaults for wall-splitting mode.
            prism_list = tessellate_curved_cluster(
                sc_pts, wall_normal=wall_n_unit, n_slabs=1,
                slab_thickness=args.slab_thickness, min_pts_slab=args.min_pts_slab,
                wall_depth=wall_depth,
                wall_coords=surf["coords"] if surf is not None else None,
            )

            if not prism_list:
                print("→ no slabs emitted, skipped")
                continue

            all_faces = [face for prism in prism_list for face in prism]
            print(f"→ {len(prism_list)} slab(s) × 6 = {len(all_faces)} polygons")

            feat_counter += 1
            feat_id = f"{user_ftype}_{feat_counter}_{uuid.uuid4().hex[:8]}"

            # Determine the parent building CityObject id
            obj_id = surf["obj_id"] if surf is not None else next(iter(cm["CityObjects"]))

            # ── If it's a hole-punching feature, project onto the wall
            if user_ftype in ("window", "door") and surf is not None:
                proj = compute_wall_projection(sc_pts, surf)
                if proj:
                    sidx = surf["idx"]
                    if sidx not in surface_opening_map:
                        surface_opening_map[sidx] = {"surface": surf, "openings": []}
                    surface_opening_map[sidx]["openings"].append(
                        (proj, user_ftype, feat_id, all_faces))
                    summary[user_ftype] = summary.get(user_ftype, 0) + 1
                    continue
                else:
                    print("    → projection degenerate, skipping")
                    continue

            # Fallback or BuildingInstallation
            fallback_feats.append((user_ftype, feat_id, all_faces, obj_id))
            summary[user_ftype] = summary.get(user_ftype, 0) + 1

    # ── 4. Apply holes + create Window/Door CityObjects ───────────────
    total_openings = sum(len(d["openings"]) for d in surface_opening_map.values())
    print(f"\n  Applying {total_openings} opening(s) to "
          f"{len(surface_opening_map)} surface(s) ...")

    for sidx, data in surface_opening_map.items():
        surf = data["surface"]
        for proj, ftype, feat_id, all_faces in data["openings"]:
            punch_hole_and_create_opening(
                cm, int_verts, scale, translate,
                surf, proj, ftype, feat_id)
        wall_angle = np.degrees(np.arctan2(
            surf["normal_2d"][1], surf["normal_2d"][0]))
        print(f"  Surface {sidx} (obj '{surf['obj_id']}', "
              f"angle={wall_angle:+.1f}°): "
              f"{len(data['openings'])} opening(s) applied.")

    # ── 5. Create BuildingInstallation CityObjects (fallback / other) ─
    for ftype, feat_id, all_faces, obj_id in fallback_feats:
        create_installation_from_faces(
            cm, int_verts, scale, translate,
            obj_id, feat_id, all_faces)
        print(f"  BuildingInstallation '{feat_id}' → parent '{obj_id}' "
              f"({len(all_faces)} faces)")

    # ── 6. Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FEATURE SUMMARY")
    print(f"{'='*60}")
    for ftype, count in sorted(summary.items()):
        cityjson_tag = {"window": "Window", "door": "Door"}.get(ftype, "BuildingInstallation")
        print(f"    {ftype:12s} → {cityjson_tag:22s}  × {count}")
    print(f"    Surfaces with openings: {len(surface_opening_map)}")
    print(f"    Installations / fallback: {len(fallback_feats)}")

    # ── 7. Save ────────────────────────────────────────────────────────
    cm["vertices"] = int_verts

    if args.output:
        out_name = args.output if args.output.endswith(".json") else args.output + ".json"
    else:
        src_base = os.path.splitext(os.path.basename(json_path))[0]
        out_name = f"{src_base}_LOD3_curved.json"

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
