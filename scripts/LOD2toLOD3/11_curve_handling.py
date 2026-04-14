#!/usr/bin/env python3
"""
Facade Features → CityGML LOD3 Pipeline  (with curved-geometry tessellation).

Each run selects one GML model and one or more .las point clouds.  For every
point cloud the user manually chooses the CityGML feature type (Window, Door,
or BuildingInstallation).  All clouds are tessellated into curved slabs. 
Windows and Doors punch coplanar holes into the matched LOD2 WallSurfaces, 
and the new fragments are appended to the file.

Usage:
    conda activate las-env
    python scripts/LOD2toLOD3/11_curve_handling.py [options]

Options:
    --n_slabs         Tessellation slabs per wall sub-cluster (default: 20)
    --slab_thickness  Half-depth of each slab in wall-normal direction, metres (default: 0.15)
    --min_pts_slab    Minimum points needed for a slab to be emitted (default: 5)
    --output, -o      Output GML filename (default: <gml_basename>_LOD3_curved.gml)
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


# =====================================================================
# Constants
# =====================================================================
INPUT_DIR  = "outputs/11A_facade_curve"
GML_DIR    = "data/lod_2"
OUTPUT_DIR = "outputs/12_curve_gml"
HOLE_PAD   = 0.005   # metres


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
# Wall-surface parsing and intersection-based normal selection
# =====================================================================
def parse_wall_surfaces_from_gml(gml_path):
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
        # Extract ID for regex stripping
        ws_gml_id = ws.get(f'{{{ns_gml}}}id') or ws.get('gml:id') or ws.get('id') or ''
        
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
                'gml_id':    ws_gml_id,
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


def find_wall_normal_for_cluster(points, wall_surfaces,
                                 dist_tol=2.0, z_expand=0.5):
    if not wall_surfaces:
        print("[PCA – no wall surfaces]", end=" ")
        return _pca_wall_normal(points)

    cl_z_min  = float(points[:, 2].min())
    cl_z_max  = float(points[:, 2].max())
    centroid  = points.mean(axis=0)
    cxy       = centroid[:2]

    best_normal = None
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

    if best_normal is not None:
        angle = np.degrees(np.arctan2(best_normal[1], best_normal[0]))
        print(f"[Stage1: dist={best_dist:.2f} m, angle={angle:+.1f}°]", end=" ")
        return np.array([best_normal[0], best_normal[1], 0.0])

    pca_n = _pca_normal_2d(points)
    if pca_n is not None:
        best_dot    = -1.0
        align_normal = None
        for ws in wall_surfaces:
            dot = abs(float(np.dot(pca_n, ws['normal_2d'])))
            if dot > best_dot:
                best_dot     = dot
                align_normal = ws['normal_2d']
        if align_normal is not None:
            angle = np.degrees(np.arctan2(align_normal[1], align_normal[0]))
            print(f"[Stage2: align={best_dot:.3f}, angle={angle:+.1f}°]", end=" ")
            return np.array([align_normal[0], align_normal[1], 0.0])

    print("[Stage3: PCA fallback]", end=" ")
    return _pca_wall_normal(points)


def _pca_normal_2d(points):
    if len(points) < 3: return None
    xy  = points[:, :2]
    cxy = xy - xy.mean(axis=0)
    cov = cxy.T @ cxy
    _, eig_vecs = np.linalg.eigh(cov)
    tang = eig_vecs[:, 1]
    return np.array([-tang[1], tang[0]])


def _pca_wall_normal(points):
    n2d = _pca_normal_2d(points)
    if n2d is None: return None
    return np.array([n2d[0], n2d[1], 0.0])


# =====================================================================
# Split points by closest WallSurface (pure geometry, no PCA)
# =====================================================================
def split_by_wall(points, wall_surfaces):
    if not wall_surfaces:
        return []

    pts_xy   = points[:, :2]
    n_walls  = len(wall_surfaces)

    dist_matrix = np.empty((len(points), n_walls), dtype=np.float64)
    for j, ws in enumerate(wall_surfaces):
        dist_matrix[:, j] = (pts_xy - ws['origin_2d']) @ ws['normal_2d']

    assignments = np.argmin(np.abs(dist_matrix), axis=1)

    sub_clusters = []
    for j, ws in enumerate(wall_surfaces):
        mask = assignments == j
        if not mask.any():
            continue
        sub_pts = points[mask]
        wall_origin_depth = float(np.dot(ws['origin_2d'], ws['normal_2d']))
        sub_clusters.append({
            'wall':              ws,
            'points':            sub_pts,
            'wall_origin_depth': wall_origin_depth,
        })

    return sub_clusters


# =====================================================================
# Wall Projection Helper for Holes
# =====================================================================
def compute_wall_projection(points, wall_surface):
    n2d        = wall_surface['normal_2d']
    wall_depth = float(np.dot(wall_surface['origin_2d'], n2d))

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
        return {}   

    def wpt(t, z):
        p_xy = wall_depth * n2d + t * txy
        return (float(p_xy[0]), float(p_xy[1]), float(z))

    ring = [
        wpt(t_min, z_min), wpt(t_max, z_min),
        wpt(t_max, z_max), wpt(t_min, z_max),
        wpt(t_min, z_min)
    ]
    return {'ring': ring, 'n2d': n2d, 'wall_depth': wall_depth, 'txy': txy}


# =====================================================================
# Tessellation — curved cluster → staircase of rectangular prisms
# =====================================================================
def tessellate_curved_cluster(points, wall_normal, n_slabs,
                              slab_thickness, min_pts_slab,
                              wall_depth=None, wall_coords=None):
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
            proj_wall = wall_coords @ axes
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
                 center = (box_t_lo + box_t_hi) / 2.0
                 box_t_lo = center - 0.05
                 box_t_hi = center + 0.05

            box_t_lo = max(box_t_lo, safe_min)
            box_t_hi = min(box_t_hi, safe_max)

            if box_t_hi <= box_t_lo:
                 center = (box_t_lo + box_t_hi) / 2.0
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
# CityGML XML Builders
# =====================================================================
def _make_multi_surface(gml_id, polygons):
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


def build_lod3_wall_fragment(wall_surface, openings, wall_frag_id):
    frame = next((pd for pd, _, _, _ in openings if pd), None)
    raw_coords = wall_surface['coords']
    if not np.allclose(raw_coords[0], raw_coords[-1]):
        raw_coords = np.vstack([raw_coords, raw_coords[0]])

    if frame:
        n2d, wall_depth, txy = frame['n2d'], frame['wall_depth'], frame['txy']
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
    for proj_dict, ftype, gml_id, all_faces in openings:
        if not proj_dict: continue
        ring = proj_dict['ring']
        n2d_h, txy_h, wd_h = proj_dict['n2d'], proj_dict['txy'], proj_dict['wall_depth']
        
        body  = ring[:-1]
        t_vals = [float(pt[:2] @ txy_h) for pt in [(p[0], p[1], 0) for p in body]]
        z_vals = [pt[2] for pt in body]
        
        t_lo, t_hi = min(t_vals) + HOLE_PAD, max(t_vals) - HOLE_PAD
        z_lo, z_hi = min(z_vals) + HOLE_PAD, max(z_vals) - HOLE_PAD

        if t_hi <= t_lo or z_hi <= z_lo:
            t_lo, t_hi = min(t_vals), max(t_vals)
            z_lo, z_hi = min(z_vals), max(z_vals)

        def hwpt(t, z):
            p_xy = wd_h * n2d_h + t * txy_h
            return (float(p_xy[0]), float(p_xy[1]), float(z))

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

    interior_block = ('\n' + '\n'.join(interior_strs)) if interior_strs else ''
    wall_poly_str = (
        f'          <gml:MultiSurface>\n'
        f'            <gml:surfaceMember>\n'
        f'              <gml:Polygon gml:id="{wall_frag_id}_poly">\n'
        f'                <gml:exterior>\n'
        f'                  <gml:LinearRing>\n'
        f'                    <gml:posList srsDimension="3">{ext_poslist}</gml:posList>\n'
        f'                  </gml:LinearRing>\n'
        f'                </gml:exterior>{interior_block}\n'
        f'              </gml:Polygon>\n'
        f'            </gml:surfaceMember>\n'
        f'          </gml:MultiSurface>'
    )

    opening_strs = []
    for proj_dict, ftype, gml_id, all_faces in openings:
        if not proj_dict: continue
        ms = _make_multi_surface(gml_id, all_faces)
        tag = 'Window' if ftype == 'window' else 'Door'
        op = (
            f'        <bldg:opening>\n'
            f'          <bldg:{tag} gml:id="{gml_id}">\n'
            f'            <bldg:lod3MultiSurface>\n'
            f'              <gml:MultiSurface>\n'
            f'{ms}\n'
            f'              </gml:MultiSurface>\n'
            f'            </bldg:lod3MultiSurface>\n'
            f'          </bldg:{tag}>\n'
            f'        </bldg:opening>'
        )
        opening_strs.append(op)

    openings_block = ('\n' + '\n'.join(opening_strs)) if opening_strs else ''
    return (
        f'    <bldg:boundedBy>\n'
        f'      <bldg:WallSurface gml:id="{wall_frag_id}">\n'
        f'        <bldg:lod3MultiSurface>\n'
        f'{wall_poly_str}\n'
        f'        </bldg:lod3MultiSurface>{openings_block}\n'
        f'      </bldg:WallSurface>\n'
        f'    </bldg:boundedBy>'
    )


def feature_to_gml(feature_type, gml_id, all_polygons):
    ms = _make_multi_surface(gml_id, all_polygons)
    if feature_type in ("window", "door"):
        tag = 'Window' if feature_type == 'window' else 'Door'
        return (
            f'    <bldg:boundedBy>\n'
            f'      <bldg:WallSurface>\n'
            f'        <bldg:opening>\n'
            f'          <bldg:{tag} gml:id="{gml_id}">\n'
            f'            <bldg:lod3MultiSurface>\n'
            f'              <gml:MultiSurface>\n{ms}\n'
            f'              </gml:MultiSurface>\n'
            f'            </bldg:lod3MultiSurface>\n'
            f'          </bldg:{tag}>\n'
            f'        </bldg:opening>\n'
            f'      </bldg:WallSurface>\n'
            f'    </bldg:boundedBy>')
    else: 
        return (
            f'    <bldg:outerBuildingInstallation>\n'
            f'      <bldg:BuildingInstallation gml:id="{gml_id}">\n'
            f'        <bldg:lod3Geometry>\n'
            f'          <gml:MultiSurface>\n{ms}\n'
            f'          </gml:MultiSurface>\n'
            f'        </bldg:lod3Geometry>\n'
            f'      </bldg:BuildingInstallation>\n'
            f'    </bldg:outerBuildingInstallation>')


# =====================================================================
# Passthrough Filters and Appending
# =====================================================================
def _parse_pos_list(block):
    m = re.search(r'<(?:[^\s:<>]+:)?posList[^>]*>([^<]+)</(?:[^\s:<>]+:)?posList>', block, re.DOTALL)
    if not m: return None
    try: vals = list(map(float, m.group(1).split()))
    except ValueError: return None
    if len(vals) < 9: return None
    return np.array(vals, dtype=np.float64).reshape(-1, 3)


def _wall_intersects_feature(coords, wall_opening_map):
    """
    Returns True ONLY if the wall's physical center and bounds 
    match a wall that received an opening.
    """
    centroid_xy = coords[:, :2].mean(axis=0)
    z_min = coords[:, 2].min()
    z_max = coords[:, 2].max()

    for data in wall_opening_map.values():
        ref = data['wall']
        # Match centroid and vertical bounds (0.1m tolerance)
        dist = np.linalg.norm(centroid_xy - ref['origin_2d'])
        if dist < 0.1 and abs(z_min - ref['z_min']) < 0.1 and abs(z_max - ref['z_max']) < 0.1:
            return True 
    return False


def collect_passthrough_surfaces(gml_path, wall_opening_map):
    with open(gml_path, 'r', encoding='utf-8') as f: content = f.read()
    block_pat = re.compile(r'[ \t]*(<(?:[^\s:<>]+:)?boundedBy[^>]*>.*?</(?:[^\s:<>]+:)?boundedBy>)[ \t]*', re.DOTALL)
    roof_pat = re.compile(r'<(?:[^\s:<>]+:)?RoofSurface[\s>]')
    wall_pat = re.compile(r'<(?:[^\s:<>]+:)?WallSurface[\s>]')

    passthrough, n_roofs, n_walls, n_skipped = [], 0, 0, 0

    for m in block_pat.finditer(content):
        block = m.group(1).strip()
        if roof_pat.search(block):
            passthrough.append(block)
            n_roofs += 1
            continue
        if wall_pat.search(block):
            if not wall_opening_map:
                passthrough.append(block)
                n_walls += 1
                continue
            coords = _parse_pos_list(block)
            if coords is None:
                passthrough.append(block)
                n_walls += 1
                continue
            # Use the new precise intersection check instead of coplanarity
            if _wall_intersects_feature(coords, wall_opening_map):
                n_skipped += 1
            else:
                passthrough.append(block)
                n_walls += 1

    print(f"  Passthrough: {n_roofs} Roof(s), {n_walls} unmodified Wall(s) ({n_skipped} replaced walls excluded).")
    return passthrough

def append_to_gml(gml_path, gml_fragments, output_path, strip_wall_ids=None):
    with open(gml_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if strip_wall_ids:
        for wid in strip_wall_ids:
            if not wid: continue
            pattern = (
                r'[ \t]*<(?:[^\s:<>]+:)?boundedBy[^>]*>\s*'
                r'<(?:[^\s:<>]+:)?WallSurface[^>]*' + re.escape(wid) +
                r'[^>]*>.*?</(?:[^\s:<>]+:)?WallSurface>\s*'
                r'</(?:[^\s:<>]+:)?boundedBy>[ \t]*(\r?\n)?'
            )
            content, n_removed = re.subn(pattern, '', content, flags=re.DOTALL)
            if n_removed:
                print(f"    Stripped {n_removed} original LOD2 block(s) for id='{wid}'.")

    ground_pattern = (
        r'[ \t]*<(?:[^\s:<>]+:)?boundedBy[^>]*>\s*'
        r'<(?:[^\s:<>]+:)?GroundSurface.*?'
        r'</(?:[^\s:<>]+:)?GroundSurface>\s*'
        r'</(?:[^\s:<>]+:)?boundedBy>[ \t]*(\r?\n)?'
    )
    extracted_grounds = []
    for match in re.finditer(ground_pattern, content, flags=re.DOTALL):
        extracted_grounds.append(match.group(0).strip())
        
    content, n_grounds = re.subn(ground_pattern, '', content, flags=re.DOTALL)
    if n_grounds:
        print(f"    Extracted {n_grounds} GroundSurface(s) for manual re-appending.")

    lod2_shell_patterns = [
        r'[ \t]*<(?:[^\s:<>]+:)?lod2Solid[^>]*>.*?</(?:[^\s:<>]+:)?lod2Solid>[ \t]*(\r?\n)?',
        r'[ \t]*<(?:[^\s:<>]+:)?lod2MultiSurface[^>]*>.*?</(?:[^\s:<>]+:)?lod2MultiSurface>[ \t]*(\r?\n)?',
    ]
    n_shells = 0
    for pat in lod2_shell_patterns:
        content, n = re.subn(pat, '', content, flags=re.DOTALL)
        n_shells += n
    if n_shells:
        print(f"    Removed {n_shells} LOD2 geometric shell block(s) (greedy).")

    marker = '</bldg:Building>'
    pos = content.rfind(marker)
    if pos == -1:
        m = re.search(r'</(?:[^\s:<>]+:)?Building>', content)
        if m: pos = m.start()
        else:
            print("  ERROR: Could not find closing Building tag in GML file.")
            return False

    if extracted_grounds:
        gml_fragments.extend(extracted_grounds)

    insertion = "\n" + "\n".join(gml_fragments) + "\n  "
    new_content = content[:pos] + insertion + content[pos:]

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    return True


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Facade Features → CityGML LOD3 (curved geometry via tessellation)")
    parser.add_argument("--n_slabs",        type=int,   default=20,
                        help="Number of tessellation slabs per wall sub-cluster")
    parser.add_argument("--slab_thickness", type=float, default=0.15,
                        help="Half-depth of each slab in the wall-normal direction.")
    parser.add_argument("--min_pts_slab",   type=int,   default=5,
                        help="Minimum points required for a slab to be emitted")
    parser.add_argument("--output", "-o",   type=str,   default=None,
                        help="Output GML filename")
    args = parser.parse_args()

    print("=" * 60)
    print("  Facade Features → CityGML LOD3  [curved tessellation]")
    print("=" * 60)

    gml_path = select_file(GML_DIR, "*.gml")
    print(f"\n  Parsing WallSurfaces from {os.path.basename(gml_path)} ...")
    wall_surfaces = parse_wall_surfaces_from_gml(gml_path)

    las_inputs = []
    while True:
        las_path     = select_file(INPUT_DIR, "*.las")
        feature_type = prompt_feature_type()
        las_inputs.append((las_path, feature_type))
        print(f"\n  ✓ Added: {os.path.basename(las_path):45s} → {feature_type}")

        again = input("\n  Add another point cloud? [y/N]: ").strip().lower()
        if again != 'y': break

    print(f"\n{'='*60}\n  TESSELLATION + WALL HOLE PROJECTION\n{'='*60}")

    wall_opening_map   = {}
    fallback_fragments = []
    summary            = {}

    for pc_idx, (las_path, user_ftype) in enumerate(las_inputs):
        print(f"\n  [{pc_idx+1}/{len(las_inputs)}] {os.path.basename(las_path)}  [{user_ftype}]")

        las    = laspy.read(las_path)
        points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
        print(f"    Points loaded: {len(points):,}")

        if wall_surfaces:
            sub_clusters = split_by_wall(points, wall_surfaces)
            print(f"    → split into {len(sub_clusters)} wall sub-cluster(s)")
        else:
            wall_n = _pca_wall_normal(points)
            if wall_n is None: continue
            sub_clusters = [{'wall': None, 'points': points, 'wall_origin_depth': None}]

        for sc_idx, sc in enumerate(sub_clusters):
            sc_pts, wall_depth = sc['points'], sc['wall_origin_depth']

            if sc['wall'] is not None:
                ws, n2d = sc['wall'], sc['wall']['normal_2d']
                wall_n = np.array([n2d[0], n2d[1], 0.0])
                angle  = np.degrees(np.arctan2(n2d[1], n2d[0]))
                print(f"    Wall {sc_idx+1}: {len(sc_pts):>7,} pts normal={angle:+.1f}° depth={wall_depth:.3f} ", end="")
            else:
                wall_n = _pca_wall_normal(sc_pts)
                if wall_n is None: continue
                wall_depth = None
                print(f"    Wall {sc_idx+1}: {len(sc_pts):>7,} pts [PCA fallback] ", end="")

            if len(sc_pts) < args.min_pts_slab:
                print("→ too few points, skipped")
                continue

            wall_n_unit = wall_n / np.linalg.norm(wall_n)

            # NOTE: We preserve your 1-slab call here, relying on args/defaults if you change it
            prism_list = tessellate_curved_cluster(
                sc_pts, wall_normal=wall_n_unit, n_slabs=1,
                slab_thickness=args.slab_thickness, min_pts_slab=args.min_pts_slab,
                wall_depth=wall_depth,
                wall_coords=ws['coords'] if sc['wall'] is not None else None
            )

            if not prism_list:
                print("→ no slabs emitted, skipped")
                continue

            all_faces = [face for prism in prism_list for face in prism]
            print(f"→ {len(prism_list)} slabs × 6 = {len(all_faces)} polygons")

            gml_id = f"{user_ftype}_{pc_idx+1}_w{sc_idx+1}_{uuid.uuid4().hex[:8]}"
            citygml_type = user_ftype if user_ftype in ("window", "door") else "installation"

            # ── If it's a hole-punching feature, project it onto the wall
            if citygml_type in ('window', 'door') and sc['wall'] is not None:
                proj = compute_wall_projection(sc_pts, sc['wall'])
                if proj:
                    widx = sc['wall']['idx']
                    if widx not in wall_opening_map:
                        wall_opening_map[widx] = {'wall': sc['wall'], 'openings': []}
                    wall_opening_map[widx]['openings'].append((proj, citygml_type, gml_id, all_faces))
                    continue
                else:
                    print("    → projection degenerate, fallback to pure BBox geometry")

            # Fallback or Installation
            fragment = feature_to_gml(citygml_type, gml_id, all_faces)
            fallback_fragments.append(fragment)

        summary[user_ftype] = summary.get(user_ftype, 0) + len(sub_clusters)

    # ── Assemble final fragment list
    gml_fragments = []
    strip_wall_ids = set()

    for widx, data in wall_opening_map.items():
        wfid = f"WallLOD3_{widx}_{uuid.uuid4().hex[:8]}"
        fragment = build_lod3_wall_fragment(data['wall'], data['openings'], wfid)
        gml_fragments.append(fragment)
        print(f"\n  Wall {widx}: emitted LOD3 WallSurface with {len(data['openings'])} opening(s).")
        orig_id = data['wall'].get('gml_id', '')
        if orig_id: strip_wall_ids.add(orig_id)

    gml_fragments.extend(fallback_fragments)

    print(f"\n  Collecting passthrough surfaces from {os.path.basename(gml_path)} ...")
    passthrough = collect_passthrough_surfaces(gml_path, wall_opening_map)
    gml_fragments.extend(passthrough)

    if not gml_fragments:
        print("\n  No features created. Nothing to append.")
        return

    # ── Summary & Append
    print(f"\n{'='*60}\n  FEATURE SUMMARY\n{'='*60}")
    for ftype, count in sorted(summary.items()):
        citygml = {"window": "bldg:Window", "door": "bldg:Door"}.get(ftype, "bldg:BuildingInstallation")
        print(f"    {ftype:12s} → {citygml:30s} × {count}")
    
    print(f"    {'LOD3 walls':12s}   {'(WallSurface+openings)':30s}   {len(wall_opening_map)}")
    print(f"    {'bbox fallbk':12s}   {'(bbox / installation)':30s}   {len(fallback_fragments)}")
    print(f"    {'passthrough':12s}   {'(roofs + plain walls)':30s}   {len(passthrough)}")

    if args.output: out_name = args.output if args.output.endswith(".gml") else args.output + ".gml"
    else: out_name = f"{os.path.splitext(os.path.basename(gml_path))[0]}_LOD3_curved.gml"
    output_path = os.path.join(OUTPUT_DIR, out_name)

    print(f"\n  Appending LOD3 features to {os.path.basename(gml_path)}...")
    success = append_to_gml(gml_path, gml_fragments, output_path, strip_wall_ids=strip_wall_ids)

    if success:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  ✓ Saved: {output_path}  ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ Failed to create output.")

if __name__ == "__main__":
    main()