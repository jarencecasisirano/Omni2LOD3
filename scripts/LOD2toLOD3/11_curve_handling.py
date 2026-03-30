#!/usr/bin/env python3
"""
Facade Features → CityGML LOD3 Pipeline  (with curved-geometry tessellation).

Points are sorted by semantic label (window / door / other), then each label
group is *tessellated* directly into a series of **axis-aligned rectangular
prisms** (slabs) snapped to the corresponding LOD2 WallSurface normal.  The
slabs form a staircase approximation of any curved profile (arched windows,
bay windows, rounded protrusions), similar in spirit to the planar-facet
decomposition used by the geoflow3d/geoflow-bundle project.

Usage:
    conda activate las-env
    python scripts/LOD2toLOD3/11_curve_handling.py [options]

Options:
    --n_slabs         Tessellation slabs per label group along wall tangent (default: 20)
    --slab_thickness  Half-depth of each slab in wall-normal direction, metres (default: 0.15)
    --min_pts_slab    Minimum points needed for a slab to be emitted (default: 5)
"""

import os
import sys
import glob
import uuid
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import laspy


# =====================================================================
# Constants
# =====================================================================
LABEL_MAP = {
    "other":   1,
    "wall":    2,
    "door":    3,
    "window":  4,
    "roof":    5,
    "ground":  6,
}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

INPUT_DIR  = "outputs/12_facade_curve"
GML_DIR    = "data/lod_2"
OUTPUT_DIR = "outputs/13_facade_curve_tesselated"


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
# Label grouping (no clustering — one feature per semantic class)
# =====================================================================
def group_by_label(points, labels):
    """
    Partition *points* into one feature per semantic label group.

    Label mapping:
      window (4)  → 'window'
      door   (3)  → 'door'
      everything else → 'other'

    Returns a list of dicts sorted by point count (descending):
        { 'type': str, 'points': ndarray(N,3), 'n_points': int }
    """
    def _label_group(code):
        name = LABEL_NAMES.get(int(code), 'other')
        return name if name in ('window', 'door') else 'other'

    label_groups = sorted(set(_label_group(c) for c in labels))
    print(f"\n  Label groups present: {label_groups}")

    features = []
    for group in label_groups:
        mask      = np.array([_label_group(c) == group for c in labels])
        group_pts = points[mask]
        if len(group_pts) == 0:
            continue
        print(f"  Group '{group}': {len(group_pts):,} pts")
        features.append({
            'type':     group,
            'points':   group_pts,
            'n_points': len(group_pts),
        })

    features.sort(key=lambda f: f['n_points'], reverse=True)
    return features


# =====================================================================
# Wall-surface parsing and intersection-based normal selection
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
    """
    Return the exact 2-D horizontal unit normal (as a 3-D vector with Z = 0)
    of the best-matching WallSurface for *points*, using two stages:

    Stage 1 — Spatial intersection (requires coordinate systems to match):
      Checks Z overlap, signed centroid-to-plane distance < dist_tol, and
      XY footprint. Picks the wall with the smallest |distance|.

    Stage 2 — Normal alignment (coordinate-independent fallback):
      Estimates the cluster's wall normal via PCA, then picks the
      WallSurface whose exact normal best aligns with that estimate.
      This always returns an exact LOD2 surface normal, never a PCA guess.

    Stage 3 — Pure PCA (only when no wall surfaces exist at all).
    """
    if not wall_surfaces:
        print("[PCA – no wall surfaces]", end=" ")
        return _pca_wall_normal(points)

    cl_z_min  = float(points[:, 2].min())
    cl_z_max  = float(points[:, 2].max())
    centroid  = points.mean(axis=0)
    cxy       = centroid[:2]

    # ── Stage 1: spatial intersection ───────────────────────────────
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

    # ── Stage 2: alignment with exact per-surface normals ────────────
    # Estimate cluster's wall-normal direction via PCA (coordinate-free).
    # Then pick the WallSurface whose exact normal best aligns with it.
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

    # ── Stage 3: pure PCA fallback ───────────────────────────────────
    print("[Stage3: PCA fallback]", end=" ")
    return _pca_wall_normal(points)


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


# =====================================================================
# Split points by closest WallSurface (pure geometry, no PCA)
# =====================================================================
def split_by_wall(points, wall_surfaces):
    """
    Assign every point in *points* to the closest WallSurface using the
    signed distance of each point's XY projection to each wall plane.

    Returns a list of dicts, one per wall that received ≥1 point::

        {
          'wall':              the wall_surfaces dict entry,
          'points':            ndarray(M, 3),
          'wall_origin_depth': float  — dot(wall_origin_xy, wall_n2d),
                               i.e. the depth of the wall plane in the
                               wall-normal projection frame.
        }
    """
    if not wall_surfaces:
        return []

    pts_xy   = points[:, :2]          # (N, 2)
    n_walls  = len(wall_surfaces)

    # Build (N, n_walls) signed-distance matrix in pure numpy
    dist_matrix = np.empty((len(points), n_walls), dtype=np.float64)
    for j, ws in enumerate(wall_surfaces):
        dist_matrix[:, j] = (pts_xy - ws['origin_2d']) @ ws['normal_2d']

    # Each point → wall with minimum |signed distance|
    assignments = np.argmin(np.abs(dist_matrix), axis=1)   # (N,)

    sub_clusters = []
    for j, ws in enumerate(wall_surfaces):
        mask = assignments == j
        if not mask.any():
            continue
        sub_pts = points[mask]
        # Wall-plane depth = dot(origin_2d, normal_2d)
        # Any point ON the wall plane satisfies dot(p_xy, n2d) == this value.
        wall_origin_depth = float(np.dot(ws['origin_2d'], ws['normal_2d']))
        sub_clusters.append({
            'wall':              ws,
            'points':            sub_pts,
            'wall_origin_depth': wall_origin_depth,
        })

    return sub_clusters


# =====================================================================
# Tessellation — curved cluster → staircase of rectangular prisms
# =====================================================================
def tessellate_curved_cluster(points, wall_normal, n_slabs,
                              slab_thickness, min_pts_slab,
                              wall_depth=None, wall_coords=None):
    if len(points) < min_pts_slab:
        return []

    # Build orthonormal frame
    wn = np.asarray(wall_normal, dtype=np.float64)
    wn = wn / np.linalg.norm(wn)

    z_up    = np.array([0.0, 0.0, 1.0])
    tangent = np.cross(z_up, wn)
    t_norm  = np.linalg.norm(tangent)
    if t_norm < 1e-9:
        tangent = np.array([1.0, 0.0, 0.0])
    else:
        tangent = tangent / t_norm

    axes = np.column_stack([wn, tangent, z_up])   # (3, 3)
    projected = points @ axes                     # (N, 3)

    # 1. Base bounds purely on the points
    pt_t_min = projected[:, 1].min()
    pt_t_max = projected[:, 1].max()

    if pt_t_max - pt_t_min < 1e-6:
        return []

    delta = (pt_t_max - pt_t_min) / n_slabs
    prism_list = []

    for i in range(n_slabs):
        t_lo = pt_t_min + i * delta
        t_hi = t_lo + delta

        # 2. Evaluate data using UNCLAMPED bounds (prevents deletion)
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

        # 3. GEOMETRY CLAMPING: Restrict physical box to the LOD2 Wall
        box_t_lo = t_lo
        box_t_hi = t_hi

        if wall_coords is not None:
            proj_wall = wall_coords @ axes
            wall_t_min = proj_wall[:, 1].min()
            wall_t_max = proj_wall[:, 1].max()

            # NO MARGIN: Allow bounding boxes to span exactly to the LOD2 wall joints.
            # This ensures continuous windows with no gaps, and prevents narrow 
            # wall segments from being mathematically swallowed/deleted.
            safe_min = wall_t_min
            safe_max = wall_t_max

            # Failsafe for extremely narrow LOD2 wall segments
            if safe_max <= safe_min:
                mid = (wall_t_min + wall_t_max) / 2.0
                safe_min = mid - 0.01
                safe_max = mid + 0.01

            # Clamp lateral width exactly to the wall boundaries
            box_t_lo = max(box_t_lo, safe_min)
            box_t_hi = min(box_t_hi, safe_max)

            # Failsafe to ensure positive width if points heavily overhung the corner
            if box_t_hi <= box_t_lo:
                 center = (box_t_lo + box_t_hi) / 2.0
                 box_t_lo = center - 0.05
                 box_t_hi = center + 0.05

            # Clamp lateral width
            box_t_lo = max(box_t_lo, safe_min)
            box_t_hi = min(box_t_hi, safe_max)

            # Failsafe to ensure positive width if points heavily overhung
            if box_t_hi <= box_t_lo:
                 center = (box_t_lo + box_t_hi) / 2.0
                 box_t_lo = center - 0.05
                 box_t_hi = center + 0.05

        # 4. Draw corners using the CLAMPED lateral bounds
        corners_local = np.array([
            [depth_lo, box_t_lo, z_lo],   # 0
            [depth_hi, box_t_lo, z_lo],   # 1
            [depth_hi, box_t_hi, z_lo],   # 2
            [depth_lo, box_t_hi, z_lo],   # 3
            [depth_lo, box_t_lo, z_hi],   # 4
            [depth_hi, box_t_lo, z_hi],   # 5
            [depth_hi, box_t_hi, z_hi],   # 6
            [depth_lo, box_t_hi, z_hi],   # 7
        ])

        corners_world = corners_local @ axes.T   # (8, 3)

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
# CityGML XML helpers
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


def feature_to_gml(feature_type, gml_id, all_polygons):
    """
    Generate a CityGML XML fragment for one feature.

    all_polygons is a flat list of polygon rings (each ring is a closed
    list of (x,y,z) tuples).  The rings come from all prism faces of a
    single tessellated cluster.

    Window / Door    → bldg:opening inside a bldg:WallSurface
    other            → bldg:outerBuildingInstallation
    """
    ms = _make_multi_surface(gml_id, all_polygons)

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
# Append features to existing GML file
# =====================================================================
def append_to_gml(gml_path, gml_fragments, output_path):
    """Insert GML fragments before the closing </bldg:Building> tag."""
    with open(gml_path, 'r', encoding='utf-8') as f:
        content = f.read()

    marker = '</bldg:Building>'
    pos = content.rfind(marker)
    if pos == -1:
        print("  ERROR: Could not find </bldg:Building> in GML file.")
        return False

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
                        help="Number of tessellation slabs per label group "
                             "along the wall tangent direction")
    parser.add_argument("--slab_thickness", type=float, default=0.15,
                        help="Half-depth of each slab in the wall-normal "
                             "direction (metres). The prism spans "
                             "[depth_mid - t, depth_mid + t].")
    parser.add_argument("--min_pts_slab",   type=int,   default=5,
                        help="Minimum points required for a slab to be emitted")
    args = parser.parse_args()

    print("=" * 60)
    print("  Facade Features → CityGML LOD3  [curved tessellation]")
    print("=" * 60)

    # ── 1. Select point cloud ──────────────────────────────────────
    las_path = select_file(INPUT_DIR, "*.las")
    print(f"\n  Loading: {os.path.basename(las_path)}")

    las = laspy.read(las_path)
    points          = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    classifications = np.array(las.classification, dtype=np.uint8)
    n_total         = len(points)
    print(f"  Total points: {n_total:,}")

    # Label breakdown
    print(f"\n  Label breakdown:")
    unique, counts = np.unique(classifications, return_counts=True)
    for code, count in zip(unique, counts):
        name = LABEL_NAMES.get(code, f"Unknown({code})")
        pct  = 100.0 * count / n_total
        print(f"    {name:20s}: {count:>10,} pts ({pct:5.1f}%)")

    # ── 2. Group by label (no clustering) ─────────────────────────
    print(f"\n{'='*60}")
    print(f"  LABEL GROUPING")
    print(f"{'='*60}")

    features = group_by_label(points, classifications)
    if not features:
        print("\n  No label groups found in point cloud.")
        return

    print(f"\n  Total label groups: {len(features)}")

    # ── 3. Select target GML model ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SELECT TARGET GML MODEL")
    print(f"{'='*60}")
    gml_path = select_file(GML_DIR, "*.gml")

    # Parse WallSurface geometry for intersection-based normal selection
    print(f"\n  Parsing WallSurfaces from {os.path.basename(gml_path)} ...")
    wall_surfaces = parse_wall_surfaces_from_gml(gml_path)

    # ── 4. Tessellation ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TESSELLATION  "
          f"(n_slabs={args.n_slabs}, slab_thickness=±{args.slab_thickness} m)")
    print(f"{'='*60}")

    gml_fragments = []
    summary       = {}

    for i, feat in enumerate(features):
        ftype  = feat['type']
        if ftype not in ('window', 'door'):
            ftype = 'other'

        n_pts = feat['n_points']
        pts   = feat['points']
        print(f"\n  [{i+1}/{len(features)}] {ftype:10s}  {n_pts:>8,} pts")

        # ── Split label group by WallSurface ────────────────────────────
        if wall_surfaces:
            sub_clusters = split_by_wall(pts, wall_surfaces)
            print(f"    → split into {len(sub_clusters)} wall sub-cluster(s)")
        else:
            # No GML surfaces: treat whole group as one cluster with PCA normal
            wall_n = _pca_wall_normal(pts)
            if wall_n is None:
                print("    → no wall surfaces and PCA failed, skipped")
                continue
            sub_clusters = [{
                'wall':              None,
                'points':            pts,
                'wall_origin_depth': None,
            }]

        for sc_idx, sc in enumerate(sub_clusters):
            sc_pts = sc['points']
            wall_depth = sc['wall_origin_depth']   # exact wall-plane depth

            if sc['wall'] is not None:
                ws   = sc['wall']
                n2d  = ws['normal_2d']
                wall_n = np.array([n2d[0], n2d[1], 0.0])
                angle  = np.degrees(np.arctan2(n2d[1], n2d[0]))
                print(f"    Wall {sc_idx+1}: {len(sc_pts):>7,} pts  "
                      f"normal={angle:+.1f}°  depth={wall_depth:.3f}  ", end="")
            else:
                wall_n = _pca_wall_normal(sc_pts) or wall_n
                wall_depth = None
                print(f"    Wall {sc_idx+1}: {len(sc_pts):>7,} pts  [PCA fallback]  ", end="")

            if len(sc_pts) < args.min_pts_slab:
                print("→ too few points, skipped")
                continue

            wall_n_unit = wall_n / np.linalg.norm(wall_n)

            prism_list = tessellate_curved_cluster(
                sc_pts,
                wall_normal    = wall_n_unit,
                n_slabs        = 1,
                slab_thickness = args.slab_thickness,
                min_pts_slab   = args.min_pts_slab,
                wall_depth     = wall_depth,
                wall_coords    = ws['coords'] if sc['wall'] is not None else None
            )

            if not prism_list:
                print("→ no slabs emitted, skipped")
                continue

            all_faces = [face for prism in prism_list for face in prism]
            print(f"→ {len(prism_list)} slabs × 6 = {len(all_faces)} polygons")

            gml_id       = f"{ftype}_{i+1}_w{sc_idx+1}_{uuid.uuid4().hex[:8]}"
            citygml_type = ftype if ftype in ("window", "door") else "installation"
            fragment     = feature_to_gml(citygml_type, gml_id, all_faces)
            gml_fragments.append(fragment)

        summary[ftype] = summary.get(ftype, 0) + len(sub_clusters)

    if not gml_fragments:
        print("\n  No tessellated features created. "
              "Try reducing --min_pts_slab or --n_slabs.")
        return

    # ── 5. Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FEATURE SUMMARY")
    print(f"{'='*60}")
    for ftype, count in sorted(summary.items()):
        citygml = {"window": "bldg:Window",
                   "door":   "bldg:Door"}.get(ftype, "bldg:BuildingInstallation")
        print(f"    {ftype:12s} → {citygml:30s} × {count}")
    print(f"    {'Total':12s}   {' ':30s}   {len(gml_fragments)}")

    # ── 6. Append and save ───────────────────────────────────────
    gml_basename = os.path.splitext(os.path.basename(gml_path))[0]
    out_name     = f"{gml_basename}_LOD3_curved.gml"
    output_path  = os.path.join(OUTPUT_DIR, out_name)

    print(f"\n  Appending {len(gml_fragments)} LOD3 features "
          f"to {os.path.basename(gml_path)} ...")

    success = append_to_gml(gml_path, gml_fragments, output_path)

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
