#!/usr/bin/env python3
"""
Facade Features → CityGML LOD3 Pipeline.

Each run selects one GML model and one or more .las point clouds.  For every
point cloud the user manually chooses the CityGML feature type (Window, Door,
or BuildingInstallation).  All clouds are DBSCAN-clustered; each cluster is
matched to the nearest WallSurface from the LOD2 GML (using the same 3-stage
spatial-intersection → normal-alignment → PCA fallback as 11_curve_handling.py).
Bounding boxes are axis-aligned to their matched WallSurface normal, and the
resulting GML elements are attached to that wall.

Usage:
    conda activate lidar-test
    python scripts/LOD2toLOD3/10_features_to_gml.py [options]

Options:
    --eps             DBSCAN neighbourhood radius (default: 0.3)
    --min_samples     DBSCAN minimum cluster size (default: 30)
    --output, -o      Output GML filename (saved inside outputs/final/).
                      If omitted, defaults to <gml_basename>_LOD3.gml
"""

import os
import sys
import glob
import uuid
import argparse
import re
import xml.etree.ElementTree as ET

import numpy as np
import laspy
from sklearn.cluster import DBSCAN


# =====================================================================
# Constants
# =====================================================================
INPUT_DIR  = "outputs/11B_flat"
GML_DIR    = "data/lod_2"
OUTPUT_DIR = "outputs/13_openings_gml"


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
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  [{i+1}] {os.path.basename(f):45s} ({size_mb:.1f} MB)")
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
    ('window', 'bldg:Window'),
    ('door',   'bldg:Door'),
    ('other',  'bldg:BuildingInstallation'),
]

def prompt_feature_type():
    """Ask the user which CityGML feature type the loaded point cloud represents."""
    print("\n  CityGML feature type for this point cloud:")
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
# DBSCAN clustering
# =====================================================================
def cluster_features(points, user_ftype, eps, min_samples):
    """
    Run DBSCAN on *points* and label every resulting cluster with *user_ftype*
    (the feature type chosen interactively by the user).

    Returns list of dicts:
        { 'type': str, 'points': ndarray(N,3), 'n_points': int }
    """
    print(f"  Running DBSCAN ({len(points):,} pts, "
          f"eps={eps}, min_samples={min_samples}) ...")
    db = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_ids = db.fit_predict(points)

    unique_clusters = set(cluster_ids)
    unique_clusters.discard(-1)
    n_noise = int((cluster_ids == -1).sum())
    print(f"    → {len(unique_clusters)} clusters, "
          f"{n_noise:,} noise points discarded")

    features = []
    for cid in sorted(unique_clusters):
        cmask = cluster_ids == cid
        features.append({
            'type':     user_ftype,
            'points':   points[cmask],
            'n_points': int(cmask.sum()),
        })

    features.sort(key=lambda f: f['n_points'], reverse=True)
    return features


# =====================================================================
# Parse WallSurfaces from a LOD2 GML file  (ported from 11_curve_handling.py)
# =====================================================================
def parse_wall_surfaces_from_gml(gml_path):
    """
    Parse every WallSurface exterior polygon from a CityGML file.

    For each polygon the following dict is returned::

        {
          'coords':    ndarray(N, 3),   # raw ring vertices
          'normal_2d': ndarray(2,),     # horizontal unit normal (XY)
          'origin_2d': ndarray(2,),     # ring centroid projected to XY
          'z_min':     float,
          'z_max':     float,
          'xy_min':    ndarray(2,),     # AABB corners in XY
          'xy_max':    ndarray(2,),
        }

    Returns a list of such dicts (one per polygon found).
    """
    try:
        tree = ET.parse(gml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  WARNING: could not parse GML: {e}")
        return []

    ns_bldg = 'http://www.opengis.net/citygml/building/2.0'
    ns_gml  = 'http://www.opengis.net/gml'

    surfaces = []
    for ws in root.iter(f'{{{ns_bldg}}}WallSurface'):
        # Capture the WallSurface's own gml:id (used later to strip the
        # original LOD2 block from the output file).
        ws_gml_id = ws.get(f'{{{ns_gml}}}id') or ws.get('gml:id') or ''

        for poly in ws.iter(f'{{{ns_gml}}}Polygon'):
            exterior = poly.find(f'.//{{{ns_gml}}}exterior')
            if exterior is None:
                continue
            pos_el = exterior.find(f'.//{{{ns_gml}}}posList')
            if pos_el is None or not pos_el.text:
                continue

            vals = list(map(float, pos_el.text.split()))
            if len(vals) < 9:
                continue

            coords = np.array(vals, dtype=np.float64).reshape(-1, 3)
            if len(coords) < 3:
                continue

            # Newell's method → polygon normal
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
                continue   # horizontal slab — skip

            normal_2d = nh / mag
            centroid  = coords.mean(axis=0)

            surfaces.append({
                'idx':       len(surfaces),
                'gml_id':    ws_gml_id,          # original WallSurface id
                'coords':    coords,
                'normal_2d': normal_2d,
                'origin_2d': centroid[:2].copy(),
                'z_min':     float(coords[:, 2].min()),
                'z_max':     float(coords[:, 2].max()),
                'xy_min':    coords[:, :2].min(axis=0),
                'xy_max':    coords[:, :2].max(axis=0),
            })

    print(f"  Parsed {len(surfaces)} WallSurface polygon(s) from GML.")
    if not surfaces:
        print("  WARNING: no WallSurface polygons found.")
    return surfaces


def _pca_normal_2d(points):
    """
    Return the PCA-estimated wall normal as a 2-D unit vector, or None.
    Used by Stage 2 of find_wall_normal_for_cluster for dot-product matching.
    """
    if len(points) < 3:
        return None
    xy  = points[:, :2]
    cxy = xy - xy.mean(axis=0)
    cov = cxy.T @ cxy
    _, eig_vecs = np.linalg.eigh(cov)
    tang = eig_vecs[:, 1]
    return np.array([-tang[1], tang[0]])


def _pca_wall_normal(points):
    """
    Estimate the wall normal from the cluster's dominant horizontal spread
    (PCA in XY).  Returns a 3-D vector with Z = 0, or None if degenerate.
    """
    n2d = _pca_normal_2d(points)
    if n2d is None:
        return None
    return np.array([n2d[0], n2d[1], 0.0])


def find_nearest_wall(points, wall_surfaces):
    """
    Unconditional fallback: return the WallSurface whose plane is closest
    (in absolute signed-distance) to the cluster centroid, ignoring Z and
    XY bounding-box checks.  Used to force window/door clusters that failed
    spatial intersection onto *some* wall so they are always emitted as
    coplanar flat planes rather than 3-D bbox boxes.
    """
    if not wall_surfaces:
        return None
    cxy      = points.mean(axis=0)[:2]
    best_ws  = None
    best_d   = float('inf')
    for ws in wall_surfaces:
        d = abs(float(np.dot(cxy - ws['origin_2d'], ws['normal_2d'])))
        if d < best_d:
            best_d  = d
            best_ws = ws
    return best_ws


def find_wall_normal_for_cluster(points, wall_surfaces,
                                 dist_tol=2.0, z_expand=0.5):
    """
    Return the exact 2-D horizontal unit normal (as a 3-D vector with Z = 0)
    of the best-matching WallSurface for *points*, using three stages:

    Stage 1 — Spatial intersection:
      Checks Z overlap, signed centroid-to-plane distance < dist_tol, and
      XY footprint. Picks the wall with the smallest |distance|.

    Stage 2 — Normal alignment (coordinate-independent fallback):
      Estimates cluster's wall normal via PCA, then picks the WallSurface
      whose exact normal best aligns with that estimate.

    Stage 3 — Pure PCA (only when no wall surfaces exist at all).

    Also returns the matched WallSurface dict (or None).
    """
    if not wall_surfaces:
        print("[PCA – no wall surfaces]", end=" ")
        return _pca_wall_normal(points), None

    cl_z_min = float(points[:, 2].min())
    cl_z_max = float(points[:, 2].max())
    centroid  = points.mean(axis=0)
    cxy       = centroid[:2]

    # ── Stage 1: spatial intersection ───────────────────────────────
    best_normal = None
    best_wall   = None
    best_dist   = float('inf')

    for ws in wall_surfaces:
        if cl_z_max < ws['z_min'] - z_expand or cl_z_min > ws['z_max'] + z_expand:
            continue
        n2d = ws['normal_2d']
        d   = float(np.dot(cxy - ws['origin_2d'], n2d))
        if abs(d) >= dist_tol:
            continue
        proj_xy = cxy - d * n2d
        pad = dist_tol
        if (proj_xy[0] < ws['xy_min'][0] - pad or
                proj_xy[0] > ws['xy_max'][0] + pad or
                proj_xy[1] < ws['xy_min'][1] - pad or
                proj_xy[1] > ws['xy_max'][1] + pad):
            continue
        if abs(d) < best_dist:
            best_dist   = abs(d)
            best_normal = n2d
            best_wall   = ws

    if best_normal is not None:
        angle = np.degrees(np.arctan2(best_normal[1], best_normal[0]))
        print(f"[Stage1: dist={best_dist:.2f} m, angle={angle:+.1f}°]", end=" ")
        return np.array([best_normal[0], best_normal[1], 0.0]), best_wall

    # ── Stage 2: alignment with exact per-surface normals ────────────
    pca_n = _pca_normal_2d(points)
    if pca_n is not None:
        best_dot     = -1.0
        align_normal = None
        align_wall   = None
        for ws in wall_surfaces:
            dot = abs(float(np.dot(pca_n, ws['normal_2d'])))
            if dot > best_dot:
                best_dot     = dot
                align_normal = ws['normal_2d']
                align_wall   = ws
        if align_normal is not None:
            angle = np.degrees(np.arctan2(align_normal[1], align_normal[0]))
            print(f"[Stage2: align={best_dot:.3f}, angle={angle:+.1f}°]", end=" ")
            return np.array([align_normal[0], align_normal[1], 0.0]), align_wall

    # ── Stage 3: pure PCA fallback ───────────────────────────────────
    print("[Stage3: PCA fallback]", end=" ")
    return _pca_wall_normal(points), None


# =======# Small inset applied to every hole ring so it is strictly inside the wall polygon.
HOLE_PAD = 0.005   # metres


def compute_wall_projection(points, wall_surface):
    """
    Project the cluster’s lateral and vertical extent onto the matched
    WallSurface plane, producing a flat rectangle that lies exactly on the
    wall.  The rectangle is clipped to the wall polygon’s own tangent and Z
    extents so it never overflows the wall boundary.

    Returns a dict::

        {
          'ring':       list of 5 (x,y,z) tuples – CCW closed rectangle
                        on the wall plane (un-inset; used for the opening
                        geometry that fills the hole),
          'n2d':        ndarray(2,) – wall outward unit normal in XY,
          'wall_depth': float       – scalar depth: dot(origin_2d, n2d),
          'txy':        ndarray(2,) – along-wall unit tangent in XY,
        }

    Returns an empty dict if the projection is degenerate after clipping.
    """
    n2d        = wall_surface['normal_2d']            # 2-D outward unit normal
    wall_depth = float(np.dot(wall_surface['origin_2d'], n2d))  # scalar depth

    # Tangent (along-wall, in XY plane) perpendicular to the wall normal
    z_up    = np.array([0.0, 0.0, 1.0])
    wall_n3 = np.array([n2d[0], n2d[1], 0.0])
    tangent = np.cross(z_up, wall_n3)
    t_norm  = np.linalg.norm(tangent)
    if t_norm < 1e-9:
        tangent = np.array([1.0, 0.0, 0.0])
    else:
        tangent /= t_norm
    txy = tangent[:2]   # 2-D component (already in XY plane)

    # Project cluster points onto tangent and Z axes
    t_pts = points[:, :2] @ txy
    z_pts = points[:, 2]
    t_min_cl, t_max_cl = float(t_pts.min()), float(t_pts.max())
    z_min_cl, z_max_cl = float(z_pts.min()), float(z_pts.max())

    # Clip to the wall polygon’s own tangent and Z extents
    wall_t    = wall_surface['coords'][:, :2] @ txy
    wt_min    = float(wall_t.min())
    wt_max    = float(wall_t.max())
    wz_min    = wall_surface['z_min']
    wz_max    = wall_surface['z_max']

    t_min = max(t_min_cl, wt_min)
    t_max = min(t_max_cl, wt_max)
    z_min = max(z_min_cl, wz_min)
    z_max = min(z_max_cl, wz_max)

    if t_max <= t_min or z_max <= z_min:
        return {}   # degenerate after clipping

    def wpt(t, z):
        """A point ON the wall plane at (tangent=t, height=z)."""
        p_xy = wall_depth * n2d + t * txy
        return (float(p_xy[0]), float(p_xy[1]), float(z))

    # Counter-clockwise ring when viewed from the wall’s outward normal
    ring = [
        wpt(t_min, z_min),   # BL
        wpt(t_max, z_min),   # BR
        wpt(t_max, z_max),   # TR
        wpt(t_min, z_max),   # TL
        wpt(t_min, z_min),   # close
    ]
    return {'ring': ring, 'n2d': n2d, 'wall_depth': wall_depth, 'txy': txy}


# =====================================================================
# Build a LOD3 WallSurface fragment with holes + opening elements
# =====================================================================
def build_lod3_wall_fragment(wall_surface, openings, wall_frag_id):
    """
    Generate a CityGML <bldg:boundedBy> fragment for a WallSurface that
    has one or more openings cut through it.
    """
    frame = next((pd for pd, _, _ in openings if pd), None)

    raw_coords = wall_surface['coords']
    if not np.allclose(raw_coords[0], raw_coords[-1]):
        raw_coords = np.vstack([raw_coords, raw_coords[0]])

    if frame:
        n2d        = frame['n2d']
        wall_depth = frame['wall_depth']
        txy        = frame['txy']

        proj_pts = []
        for p in raw_coords:
            t_val = float(p[:2] @ txy)
            z_val = float(p[2])
            p_xy  = wall_depth * n2d + t_val * txy
            proj_pts.append((float(p_xy[0]), float(p_xy[1]), z_val))
        ext_poslist = ' '.join(f'{x:.6f} {y:.6f} {z:.6f}' for x, y, z in proj_pts)
    else:
        ext_poslist = ' '.join(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}' for p in raw_coords)

    interior_strs = []
    opening_strs = []

    for proj_dict, ftype, gml_id in openings:
        if not proj_dict:
            continue
        
        ring    = proj_dict['ring']
        n2d_h   = proj_dict['n2d']
        txy_h   = proj_dict['txy']
        wd_h    = proj_dict['wall_depth']
        pad     = HOLE_PAD

        body  = ring[:-1]
        t_vals = [float(pt[:2] @ txy_h) for pt in [(p[0], p[1], 0) for p in body]]
        z_vals = [pt[2] for pt in body]
        t_lo = min(t_vals) + pad
        t_hi = max(t_vals) - pad
        z_lo = min(z_vals) + pad
        z_hi = max(z_vals) - pad

        if t_hi <= t_lo or z_hi <= z_lo:
            t_lo, t_hi = min(t_vals), max(t_vals)
            z_lo, z_hi = min(z_vals), max(z_vals)

        def hwpt(t, z):
            p_xy = wd_h * n2d_h + t * txy_h
            return (float(p_xy[0]), float(p_xy[1]), float(z))

        # CW Hole Ring (for gml:interior)
        hole_body = [hwpt(t_lo, z_hi), hwpt(t_hi, z_hi), hwpt(t_hi, z_lo), hwpt(t_lo, z_lo)]
        hole_ring = hole_body + [hole_body[0]]
        hole_pl = ' '.join(f'{x:.6f} {y:.6f} {z:.6f}' for x, y, z in hole_ring)
        
        interior_strs.append(
            f'                  <gml:interior>\n'
            f'                    <gml:LinearRing>\n'
            f'                      <gml:posList srsDimension="3">{hole_pl}</gml:posList>\n'
            f'                    </gml:LinearRing>\n'
            f'                  </gml:interior>'
        )

        # CCW Window Ring (using the exact same padded bounds to eliminate Z-fighting)
        window_body = [hwpt(t_lo, z_lo), hwpt(t_hi, z_lo), hwpt(t_hi, z_hi), hwpt(t_lo, z_hi)]
        window_ring = window_body + [window_body[0]]
        window_pl = ' '.join(f'{x:.6f} {y:.6f} {z:.6f}' for x, y, z in window_ring)

        face_str = (
            f'              <gml:MultiSurface>\n'
            f'                <gml:surfaceMember>\n'
            f'                  <gml:Polygon gml:id="{gml_id}_face">\n'
            f'                    <gml:exterior>\n'
            f'                      <gml:LinearRing>\n'
            f'                        <gml:posList srsDimension="3">{window_pl}</gml:posList>\n'
            f'                      </gml:LinearRing>\n'
            f'                    </gml:exterior>\n'
            f'                  </gml:Polygon>\n'
            f'                </gml:surfaceMember>\n'
            f'              </gml:MultiSurface>'
        )
        
        tag = "Window" if ftype == "window" else "Door"
        opening_strs.append(
            f'        <bldg:opening>\n'
            f'          <bldg:{tag} gml:id="{gml_id}">\n'
            f'            <bldg:lod3MultiSurface>\n'
            f'{face_str}\n'
            f'            </bldg:lod3MultiSurface>\n'
            f'          </bldg:{tag}>\n'
            f'        </bldg:opening>'
        )

    interior_block = ('\n' + '\n'.join(interior_strs)) if interior_strs else ''
    openings_block = ('\n' + '\n'.join(opening_strs)) if opening_strs else ''

    wall_poly_str = (
        f'          <gml:MultiSurface>\n'
        f'            <gml:surfaceMember>\n'
        f'              <gml:Polygon gml:id="{wall_frag_id}_poly">\n'
        f'                <gml:exterior>\n'
        f'                  <gml:LinearRing>\n'
        f'                    <gml:posList srsDimension="3">{ext_poslist}</gml:posList>\n'
        f'                  </gml:LinearRing>\n'
        f'                </gml:exterior>'
        f'{interior_block}\n'
        f'              </gml:Polygon>\n'
        f'            </gml:surfaceMember>\n'
        f'          </gml:MultiSurface>'
    )

    return (
        f'    <bldg:boundedBy>\n'
        f'      <bldg:WallSurface gml:id="{wall_frag_id}">\n'
        f'        <bldg:lod3MultiSurface>\n'
        f'{wall_poly_str}\n'
        f'        </bldg:lod3MultiSurface>'
        f'{openings_block}\n'
        f'      </bldg:WallSurface>\n'
        f'    </bldg:boundedBy>'
    )

def create_bbox_polygons(points, wall_normal=None):
    """
    Create a 3D axis-aligned (to wall) Bounding Box from a point cluster.

    If *wall_normal* (a unit 3-D vector with Z=0) is given, the box is oriented
    so that two faces are parallel to the LOD2 WallSurface:
      axis0 = wall_normal          (depth direction, into/out of wall)
      axis1 = tangent along wall   (= cross(Z, wall_normal), normalised)
      axis2 = world Z              (vertical)

    If no wall_normal is given, PCA is used as fallback.

    Returns a list of 6 closed polygons (faces of the box).
    """
    if len(points) < 3:
        return []

    centroid = points.mean(axis=0)
    centered = points - centroid

    if wall_normal is not None:
        # ── Wall-aligned axes ──────────────────────────────────────────
        wn = np.asarray(wall_normal, dtype=np.float64)
        wn = wn / np.linalg.norm(wn)                      # depth axis

        z_up = np.array([0.0, 0.0, 1.0])
        tangent = np.cross(z_up, wn)                      # along-wall axis
        t_norm = np.linalg.norm(tangent)
        if t_norm < 1e-9:                                  # degenerate fallback
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent = tangent / t_norm

        # Orthonormal frame: [wall_normal, tangent, Z]
        axes = np.column_stack([wn, tangent, z_up])        # shape (3, 3)
    else:
        # ── PCA fallback ───────────────────────────────────────────────
        cov = np.cov(centered, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        idx = eig_vals.argsort()[::-1]
        axes = eig_vecs[:, idx]

    # Project points onto chosen axes
    projected = centered @ axes       # (N, 3)
    p_min = projected.min(axis=0)
    p_max = projected.max(axis=0)

    # 8 box corners in local frame
    v_proj = np.array([
        [p_min[0], p_min[1], p_min[2]],   # 0
        [p_max[0], p_min[1], p_min[2]],   # 1
        [p_max[0], p_max[1], p_min[2]],   # 2
        [p_min[0], p_max[1], p_min[2]],   # 3
        [p_min[0], p_min[1], p_max[2]],   # 4
        [p_max[0], p_min[1], p_max[2]],   # 5
        [p_max[0], p_max[1], p_max[2]],   # 6
        [p_min[0], p_max[1], p_max[2]],   # 7
    ])

    # Transform back to world space
    v_world = (v_proj @ axes.T) + centroid

    # 6 faces
    face_indices = [
        [0, 1, 2, 3],   # Bottom
        [4, 7, 6, 5],   # Top
        [0, 3, 7, 4],   # Wall-parallel face A (min depth)
        [1, 5, 6, 2],   # Wall-parallel face B (max depth)
        [0, 4, 5, 1],   # Side A
        [3, 2, 6, 7],   # Side B
    ]

    polygons = []
    for indices in face_indices:
        poly = [tuple(v_world[i]) for i in indices]
        poly.append(poly[0])    # close ring
        polygons.append(poly)

    return polygons


# =====================================================================
# CityGML XML generation (string-based for reliable namespace handling)
# =====================================================================
def _make_multi_surface(gml_id, polygons):
    """Build a <gml:MultiSurface> XML string from polygon list."""
    members = []
    for i, poly in enumerate(polygons):
        poslist = " ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in poly)
        members.append(
            f'              <gml:surfaceMember>\n'
            f'                <gml:Polygon gml:id="{gml_id}_poly_{i}">\n'
            f'                  <gml:exterior>\n'
            f'                    <gml:LinearRing>\n'
            f'                      <gml:posList srsDimension="3">'
            f'{poslist}</gml:posList>\n'
            f'                    </gml:LinearRing>\n'
            f'                  </gml:exterior>\n'
            f'                </gml:Polygon>\n'
            f'              </gml:surfaceMember>')
    return "\n".join(members)


def feature_to_gml(feature_type, gml_id, polygons):
    """
    Generate a CityGML XML fragment for one feature.

    Window / Door    → bldg:opening inside a bldg:WallSurface
    building_part    → bldg:consistsOfBuildingPart > bldg:BuildingPart
    """
    ms = _make_multi_surface(gml_id, polygons)

    if feature_type == "window":
        return (
            f'    <bldg:boundedBy>\n'
            f'      <bldg:WallSurface>\n'
            f'        <bldg:opening>\n'
            f'          <bldg:Window gml:id="{gml_id}">\n'
            f'            <bldg:lod3MultiSurface>\n'
            f'              <gml:MultiSurface>\n'
            f'{ms}\n'
            f'              </gml:MultiSurface>\n'
            f'            </bldg:lod3MultiSurface>\n'
            f'          </bldg:Window>\n'
            f'        </bldg:opening>\n'
            f'      </bldg:WallSurface>\n'
            f'    </bldg:boundedBy>')

    elif feature_type == "door":
        return (
            f'    <bldg:boundedBy>\n'
            f'      <bldg:WallSurface>\n'
            f'        <bldg:opening>\n'
            f'          <bldg:Door gml:id="{gml_id}">\n'
            f'            <bldg:lod3MultiSurface>\n'
            f'              <gml:MultiSurface>\n'
            f'{ms}\n'
            f'              </gml:MultiSurface>\n'
            f'            </bldg:lod3MultiSurface>\n'
            f'          </bldg:Door>\n'
            f'        </bldg:opening>\n'
            f'      </bldg:WallSurface>\n'
            f'    </bldg:boundedBy>')

    else:  # BuildingInstallation
        return (
            f'    <bldg:outerBuildingInstallation>\n'
            f'      <bldg:BuildingInstallation gml:id="{gml_id}">\n'
            f'        <bldg:lod3Geometry>\n'
            f'          <gml:MultiSurface>\n'
            f'{ms}\n'
            f'          </gml:MultiSurface>\n'
            f'        </bldg:lod3Geometry>\n'
            f'      </bldg:BuildingInstallation>\n'
            f'    </bldg:outerBuildingInstallation>')


# =====================================================================
# Geometric helpers for passthrough wall filtering
# =====================================================================
def _parse_pos_list(block):
    """
    Extract 3-D coordinates from the first <*:posList> element found in a
    raw GML text block.  Returns an ndarray(N, 3) or None if the block
    contains no usable posList.
    """
    m = re.search(
        r'<(?:[^\s:<>]+:)?posList[^>]*>([^<]+)</(?:[^\s:<>]+:)?posList>',
        block, re.DOTALL)
    if not m:
        return None
    try:
        vals = list(map(float, m.group(1).split()))
    except ValueError:
        return None
    if len(vals) < 9:
        return None
    return np.array(vals, dtype=np.float64).reshape(-1, 3)


def _wall_coplanar_with_any(coords, wall_opening_map,
                            normal_tol=0.15, depth_tol=0.5):
    """
    Return True if the polygon *coords* is geometrically co-planar with any
    wall in *wall_opening_map*, meaning it lies on the same facade plane and
    would visually overlap the LOD3 holed version that replaces it.

    Co-planarity is determined by two criteria:

    1. **Normal alignment**: the candidate wall's 2-D outward normal must be
       within ``normal_tol`` (in dot-product units ≈ cosθ) of the reference
       wall's normal.  Default 0.15 ≈ within ~25°.
    2. **Depth match**: the candidate wall's centroid, when projected onto the
       reference wall's normal, must agree with the reference wall's own depth
       to within ``depth_tol`` metres.  Default 0.5 m.

    Both criteria must be met for any single reference wall in the map.
    """
    # Newell's method → 2-D outward normal of the candidate wall
    n = np.zeros(3)
    nc = len(coords)
    for i in range(nc):
        c = coords[i]
        nx = coords[(i + 1) % nc]
        n[0] += (c[1] - nx[1]) * (c[2] + nx[2])
        n[1] += (c[2] - nx[2]) * (c[0] + nx[0])
        n[2] += (c[0] - nx[0]) * (c[1] + nx[1])
    nh  = n[:2]
    mag = np.linalg.norm(nh)
    if mag < 1e-6:
        return False   # horizontal face, not a competing wall
    n2d_cand = nh / mag

    centroid_xy = coords[:, :2].mean(axis=0)

    for data in wall_opening_map.values():
        ref    = data['wall']
        n2d_r  = ref['normal_2d']
        depth_r = float(ref['origin_2d'] @ n2d_r)

        # 1. Normal alignment (absolute dot product handles anti-parallel too)
        if abs(float(n2d_cand @ n2d_r)) < 1.0 - normal_tol:
            continue

        # 2. Depth agreement (project candidate centroid onto reference normal)
        depth_c = float(centroid_xy @ n2d_r)
        if abs(depth_c - depth_r) > depth_tol:
            continue

        return True   # co-planar with a replaced wall

    return False


# =====================================================================
# Collect passthrough surfaces from the original GML
# =====================================================================
def collect_passthrough_surfaces(gml_path, wall_opening_map):
    """
    Read the original GML file and return verbatim XML string fragments for
    every <*:boundedBy> block that must be preserved in the LOD3 output.

    This has been rewritten to STRICTLY filter by gml:id, permanently killing
    "zombie walls" by ensuring any wall marked for LOD3 replacement is never
    passed through.
    """
    with open(gml_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Extract the exact gml:ids of the walls we are replacing
    replaced_wall_ids = set()
    for data in wall_opening_map.values():
        orig_id = data['wall'].get('gml_id')
        if orig_id:
            replaced_wall_ids.add(orig_id)

    # Namespace-agnostic patterns
    block_pat = re.compile(
        r'[ \t]*'
        r'(<(?:[^\s:<>]+:)?boundedBy[^>]*>.*?</(?:[^\s:<>]+:)?boundedBy>)'
        r'[ \t]*',
        re.DOTALL,
    )
    roof_pat   = re.compile(r'<(?:[^\s:<>]+:)?RoofSurface[\s>]')
    wall_pat   = re.compile(r'<(?:[^\s:<>]+:)?WallSurface[\s>]')
    ground_pat = re.compile(r'<(?:[^\s:<>]+:)?GroundSurface[\s>]')

    # Pattern to extract any ID attribute flexibly (gml:id or id)
    id_pat = re.compile(r'(?:gml:)?id="([^"]+)"')

    passthrough = []
    n_roofs   = 0
    n_ground  = 0
    n_walls   = 0
    n_skipped = 0

    for m in block_pat.finditer(content):
        block = m.group(1).strip()

        # ── RoofSurface: always carry through ─────────────────────────
        if roof_pat.search(block):
            passthrough.append(block)
            n_roofs += 1
            continue

        # ── GroundSurface: always carry through ───────────────────────
        if ground_pat.search(block):
            passthrough.append(block)
            n_ground += 1
            continue

        # ── WallSurface: Strict ID-based filter ───────────────────────
        if wall_pat.search(block):
            if not wall_opening_map:
                # No openings at all — keep all walls
                passthrough.append(block)
                n_walls += 1
                continue

            # Extract all IDs present in this XML block
            block_ids = id_pat.findall(block)

            # Check if this block contains the ID of a wall we already replaced
            is_zombie = any(bid in replaced_wall_ids for bid in block_ids)

            if is_zombie:
                # Zombie wall detected! Skip it so it stays dead.
                n_skipped += 1
            else:
                # It's an unaffected wall, safe to pass through.
                passthrough.append(block)
                n_walls += 1

    print(f"  Passthrough: {n_roofs} RoofSurface(s), "
          f"{n_ground} GroundSurface(s), "
          f"{n_walls} unmodified WallSurface(s) "
          f"({n_skipped} replaced wall(s) cleanly killed).")
    return passthrough


# =====================================================================
# Append features to existing GML file  (pure-string, namespace-safe)
# =====================================================================
def append_to_gml(gml_path, gml_fragments, output_path, strip_wall_ids=None):
    """
    String-only implementation that preserves every namespace prefix exactly
    as written in the source file (no ElementTree re-serialisation, which would
    mangle all prefixes and make the file unreadable by CityGML viewers).

    Steps
    -----
    1. Read raw source GML text.
    2. Temporarily replace all <boundedBy> blocks with numbered placeholders
       so that the global-shell stripper cannot touch nested lod2MultiSurface
       geometry (which is the actual polygon data of each individual surface).
    3. Strip LOD2 global-level lod2Solid / lod2MultiSurface shells from the
       skeletal text (now safe because boundedBy content is protected).
    4. Restore the placeholders, filtering out replaced WallSurface blocks.
    5. Insert the new LOD3 fragments immediately before the last
       </bldg:Building> closing tag (using last, not first, to handle files
       that contain nested BuildingPart elements).
    """
    with open(gml_path, 'r', encoding='utf-8') as fh:
        content = fh.read()

    # ── 1. Pull out all <boundedBy> blocks → placeholders ─────────────
    bounded_full_pat = re.compile(
        r'<(?:[^\s:<>]+:)?boundedBy[^>]*>.*?</(?:[^\s:<>]+:)?boundedBy>',
        re.DOTALL)
    protected_blocks = []

    def _protect(m):
        idx = len(protected_blocks)
        protected_blocks.append(m.group(0))
        return f'__BB_{idx}__'

    content = bounded_full_pat.sub(_protect, content)

    # ── 2. Strip global LOD2 geometric shells (now safe) ──────────────
    shell_pat = re.compile(
        r'[ \t]*<(?:[^\s:<>]+:)?(?:lod2Solid|lod2MultiSurface)[^>]*>.*?'
        r'</(?:[^\s:<>]+:)?(?:lod2Solid|lod2MultiSurface)>[ \t]*\n?',
        re.DOTALL)
    n_shells = len(shell_pat.findall(content))
    content  = shell_pat.sub('', content)
    if n_shells:
        print(f"    Removed {n_shells} LOD2 global geometric shell block(s).")

    # ── 3. Restore placeholders, filtering replaced WallSurfaces ──────
    id_pat_local = re.compile(r'(?:gml:)?id="([^"]+)"')
    ws_pat_local = re.compile(r'<(?:[^\s:<>]+:)?WallSurface[\s>]')
    n_removed = 0

    def _restore(m):
        nonlocal n_removed
        block = protected_blocks[int(m.group(1))]
        if strip_wall_ids and ws_pat_local.search(block):
            ids_in_block = set(id_pat_local.findall(block))
            if ids_in_block & strip_wall_ids:
                n_removed += 1
                return ''      # replaced wall → delete
        return block           # keep everything else

    content = re.sub(r'__BB_(\d+)__', _restore, content)
    if n_removed:
        print(f"    Stripped {n_removed} original LOD2 WallSurface block(s).")

    # ── 4. Insert new fragments before the LAST </bldg:Building> tag ──
    # Using rfind-equivalent so nested BuildingPart closing tags are skipped.
    all_close = list(re.finditer(r'</(?:[^\s:<>]+:)?Building>', content))
    if not all_close:
        print("  ERROR: Could not locate closing Building tag in GML.")
        return False
    close_bldg   = all_close[-1]   # last occurrence = outermost Building
    insert_pos   = close_bldg.start()
    insertion    = "\n" + "\n".join(gml_fragments) + "\n"
    content      = content[:insert_pos] + insertion + content[insert_pos:]

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write(content)

    return True


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Convert facade features to CityGML LOD3")
    parser.add_argument("--eps", type=float, default=0.3,
                        help="DBSCAN eps (neighbourhood radius)")
    parser.add_argument("--min_samples", type=int, default=30,
                        help="DBSCAN min_samples")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output GML filename (saved inside outputs/final/). "
                             "If omitted, defaults to <gml_basename>_LOD3.gml")

    args = parser.parse_args()

    print("=" * 60)
    print("  Facade Features → CityGML LOD3")
    print("=" * 60)

    # ── 1. Select target GML model ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SELECT TARGET GML MODEL")
    print(f"{'='*60}")
    gml_path = select_file(GML_DIR, "*.gml")

    print(f"\n  Parsing WallSurfaces from {os.path.basename(gml_path)} ...")
    wall_surfaces = parse_wall_surfaces_from_gml(gml_path)

    # ── 2. Collect point cloud inputs ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SELECT POINT CLOUD INPUT(S)")
    print(f"{'='*60}")
    print("  You can add multiple point clouds; all will be combined into"
          " one output GML.")

    las_inputs = []   # list of (las_path, feature_type)
    while True:
        las_path     = select_file(INPUT_DIR, "*.las")
        feature_type = prompt_feature_type()
        las_inputs.append((las_path, feature_type))
        print(f"\n  \u2713 Added: {os.path.basename(las_path):45s} \u2192 {feature_type}")

        again = input("\n  Add another point cloud? [y/N]: ").strip().lower()
        if again != 'y':
            break

    print(f"\n  {len(las_inputs)} point cloud(s) selected:")
    for lp, ft in las_inputs:
        print(f"    {os.path.basename(lp):45s} \u2192 {ft}")

    # \u2500\u2500 3. DBSCAN clustering + wall projection \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    print(f"\n{'='*60}")
    print(f"  CLUSTERING + WALL PROJECTION  "
          f"(eps={args.eps}, min_samples={args.min_samples})")
    print(f"{'='*60}")

    # wall_opening_map[wall_idx] = {'wall': ws_dict,
    #                               'openings': [(ring, ftype, gml_id), ...]}
    wall_opening_map  = {}   # grouped by matched wall
    fallback_fragments = []  # installations / unmatched clusters (bbox fallback)
    summary           = {}
    feat_counter      = 0    # global index across all clouds

    for pc_idx, (las_path, user_ftype) in enumerate(las_inputs):
        print(f"\n  [{pc_idx+1}/{len(las_inputs)}] "
              f"{os.path.basename(las_path)}  [{user_ftype}]")

        las    = laspy.read(las_path)
        points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
        print(f"    Points loaded: {len(points):,}")

        features = cluster_features(points, user_ftype,
                                    eps=args.eps,
                                    min_samples=args.min_samples)
        if not features:
            print(f"    No clusters found — skipping this cloud.")
            continue

        print(f"    Clusters found: {len(features)}")

        for feat in features:
            feat_counter += 1
            ftype  = feat['type']
            n_pts  = feat['n_points']
            print(f"\n    [{feat_counter}] {ftype:12s}  {n_pts:>8,} pts  ", end="")

            # Find the nearest WallSurface via 3-stage spatial matching
            wall_n, matched_wall = find_wall_normal_for_cluster(
                feat['points'], wall_surfaces)

            gml_id = f"{ftype}_{feat_counter}_{uuid.uuid4().hex[:8]}"

            # ── Window / Door: ALWAYS project as flat plane on a wall ──
            if ftype in ('window', 'door'):
                # Prefer the spatially matched wall; fall back to nearest.
                wall_for_proj = matched_wall
                if wall_for_proj is None and wall_surfaces:
                    wall_for_proj = find_nearest_wall(feat['points'],
                                                      wall_surfaces)
                    if wall_for_proj is not None:
                        wa = np.degrees(np.arctan2(
                            wall_for_proj['normal_2d'][1],
                            wall_for_proj['normal_2d'][0]))
                        print(f"[nearest-wall fallback → wall "
                              f"{wall_for_proj['idx']}, "
                              f"angle={wa:+.1f}°] ", end="")

                if wall_for_proj is not None:
                    proj = compute_wall_projection(feat['points'],
                                                   wall_for_proj)
                    if proj:
                        widx = wall_for_proj['idx']
                        if widx not in wall_opening_map:
                            wall_opening_map[widx] = {
                                'wall': wall_for_proj, 'openings': []}
                        wall_opening_map[widx]['openings'].append(
                            (proj, ftype, gml_id))
                        summary[ftype] = summary.get(ftype, 0) + 1
                        wall_angle = np.degrees(np.arctan2(
                            wall_for_proj['normal_2d'][1],
                            wall_for_proj['normal_2d'][0]))
                        print(f"→ projected onto wall {widx} "
                              f"(angle={wall_angle:+.1f}°)")
                        continue   # skip bbox fallback
                    else:
                        print("[projection degenerate, bbox fallback] ",
                              end="")
                else:
                    print("[no wall found, bbox fallback] ", end="")

            # ── Installation / unmatched / degenerate projection: bbox ──
            polygons = create_bbox_polygons(feat['points'], wall_normal=wall_n)
            if not polygons:
                print("→ BBox failed, skipped")
                continue

            n_verts = sum(len(p) - 1 for p in polygons)
            if matched_wall is not None:
                wall_angle = np.degrees(np.arctan2(
                    matched_wall['normal_2d'][1], matched_wall['normal_2d'][0]))
                print(f"→ {len(polygons)} faces, {n_verts} verts "
                      f"(wall angle={wall_angle:+.1f}°)")
            else:
                print(f"→ {len(polygons)} faces, {n_verts} verts (PCA normal)")

            citygml_type = ftype if ftype in ('window', 'door') else 'installation'
            fallback_fragments.append(
                feature_to_gml(citygml_type, gml_id, polygons))
            summary[ftype] = summary.get(ftype, 0) + 1

    # ── Assemble final fragment list ────────────────────────────────
    # 1. LOD3 WallSurface fragments (wall polygon w/ holes + opening elements)
    gml_fragments = []
    strip_wall_ids = set()   # gml:ids of original LOD2 WallSurface blocks to remove
    for widx, data in wall_opening_map.items():
        wfid     = f"WallLOD3_{widx}_{uuid.uuid4().hex[:8]}"
        fragment = build_lod3_wall_fragment(data['wall'], data['openings'], wfid)
        gml_fragments.append(fragment)
        n_op = len(data['openings'])
        print(f"\n  Wall {widx}: emitted LOD3 WallSurface with "
              f"{n_op} opening(s).")
        # Mark original wall for removal
        orig_id = data['wall'].get('gml_id', '')
        if orig_id:
            strip_wall_ids.add(orig_id)

    # 2. Fallback fragments (installations, unmatched clusters).
    gml_fragments.extend(fallback_fragments)

    # 3. Passthrough surfaces (Roofs, GroundSurface, unmodified Walls) are
    #    preserved automatically by the string-based append_to_gml, which
    #    only strips the specific replaced walls from the original file text
    #    and leaves all other <boundedBy> blocks intact.

    if not gml_fragments:
        print("\n  No features created. Nothing to append.")
        return

    # ── 4. Summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FEATURE SUMMARY")
    print(f"{'='*60}")
    for ftype, count in sorted(summary.items()):
        citygml = {"window": "bldg:Window",
                   "door":   "bldg:Door"}.get(ftype, "bldg:BuildingInstallation")
        print(f"    {ftype:12s} → {citygml:30s} × {count}")
    n_wall_frags = len(wall_opening_map)
    n_fallback   = len(fallback_fragments)
    print(f"    {'LOD3 walls':12s}   {'(WallSurface+openings)':30s}   {n_wall_frags}")
    print(f"    {'bbox fallbk':12s}   {'(bbox / installation)':30s}   {n_fallback}")
    print(f"    {'Total frags':12s}   {' ':30s}   {len(gml_fragments)}")
    if strip_wall_ids:
        print(f"    Stripping {len(strip_wall_ids)} original LOD2 wall block(s).")

    # ── 5. Append and save ───────────────────────────────────────
    if args.output:
        out_name = args.output if args.output.endswith(".gml") else args.output + ".gml"
    else:
        gml_basename = os.path.splitext(os.path.basename(gml_path))[0]
        out_name     = f"{gml_basename}_LOD3.gml"
    output_path = os.path.join(OUTPUT_DIR, out_name)

    print(f"\n  Appending {len(gml_fragments)} LOD3 features "
          f"to {os.path.basename(gml_path)}...")

    success = append_to_gml(gml_path, gml_fragments, output_path,
                            strip_wall_ids=strip_wall_ids)

    if success:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  ✓ Saved: {output_path}  ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ Failed to create output.")

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
