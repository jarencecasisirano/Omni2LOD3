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

INPUT_DIR  = "outputs/trials"
GML_DIR    = "outputs/trials"
OUTPUT_DIR = "outputs/trials"


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
# Parse wall normals from a LOD2 GML file
# =====================================================================
def parse_wall_normals_from_gml(gml_path, angle_tol_deg=5.0):
    """
    Extract horizontal (XY-plane) unit normals from every WallSurface polygon
    found inside a CityGML file, then cluster near-parallel normals and return
    only the *dominant* direction for each cluster.

    Returns a list of dominant 2-D unit vectors.
    """
    try:
        tree = ET.parse(gml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  WARNING: could not parse GML for wall normals: {e}")
        return []

    raw_normals = []

    for ws in root.iter('{http://www.opengis.net/citygml/building/2.0}WallSurface'):
        for poly in ws.iter('{http://www.opengis.net/gml}Polygon'):
            exterior = poly.find('.//{http://www.opengis.net/gml}exterior')
            if exterior is None:
                continue
            pos_el = exterior.find('.//{http://www.opengis.net/gml}posList')
            if pos_el is None or not pos_el.text:
                continue

            vals = list(map(float, pos_el.text.split()))
            if len(vals) < 9:
                continue

            pts = np.array(vals).reshape(-1, 3)
            if len(pts) < 3:
                continue

            # Newell's method for polygon normal
            n = np.zeros(3)
            for i in range(len(pts)):
                curr = pts[i]
                nxt  = pts[(i + 1) % len(pts)]
                n[0] += (curr[1] - nxt[1]) * (curr[2] + nxt[2])
                n[1] += (curr[2] - nxt[2]) * (curr[0] + nxt[0])
                n[2] += (curr[0] - nxt[0]) * (curr[1] + nxt[1])

            nh = n[:2]
            mag = np.linalg.norm(nh)
            if mag < 1e-6:
                continue
            raw_normals.append(nh / mag)

    if not raw_normals:
        print("  WARNING: no wall normals found in GML.")
        return []

    # ── Cluster near-parallel normals ──────────────────────────────
    tol_cos = np.cos(np.radians(angle_tol_deg))
    dominant = []
    raw_normals_arr = np.array(raw_normals)

    for nv in raw_normals_arr:
        matched = False
        for idx, (rep, cnt) in enumerate(dominant):
            if abs(float(np.dot(nv, rep))) >= tol_cos:
                new_rep = rep * cnt + nv * np.sign(float(np.dot(nv, rep)))
                nr = np.linalg.norm(new_rep)
                dominant[idx] = (new_rep / nr if nr > 1e-9 else rep, cnt + 1)
                matched = True
                break
        if not matched:
            dominant.append((nv, 1))

    dominant.sort(key=lambda x: x[1], reverse=True)
    dominant_normals = [d[0] for d in dominant]

    print(f"  Extracted {len(raw_normals)} raw wall normals "
          f"→ {len(dominant_normals)} dominant directions from GML.")
    for i, (nv, cnt) in enumerate(dominant):
        angle = np.degrees(np.arctan2(nv[1], nv[0]))
        print(f"    [{i+1}] angle={angle:+.1f}°  count={cnt}")
    return dominant_normals


def _best_wall_normal(points, wall_normals):
    """
    Given a point cluster and a list of candidate 2-D horizontal wall normals,
    return the wall normal (as a 3-D vector with Z=0) whose tangent direction
    best aligns with the cluster's dominant horizontal spread axis.
    """
    if not wall_normals:
        return None

    xy = points[:, :2]
    cxy = xy - xy.mean(axis=0)
    cov = cxy.T @ cxy
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    tangent_2d = eig_vecs[:, 1]   # largest eigenvalue → wall tangent

    cluster_normal_2d = np.array([-tangent_2d[1], tangent_2d[0]])

    best_n   = None
    best_dot = -1.0
    for wn in wall_normals:
        dot = abs(float(np.dot(cluster_normal_2d, wn)))
        if dot > best_dot:
            best_dot = dot
            best_n   = wn

    if best_n is None:
        return None

    return np.array([best_n[0], best_n[1], 0.0])




# =====================================================================
# Tessellation — curved cluster → staircase of rectangular prisms
# =====================================================================
def tessellate_curved_cluster(points, wall_normal, n_slabs,
                              slab_thickness, min_pts_slab):
    """
    Discretise a point group into a staircase of axis-aligned rectangular
    prisms, each snapped to the wall normal direction.

    Coordinate frame
    ----------------
      axis0 = wall_normal   (depth direction, into/out-of wall)
      axis1 = tangent        (along wall, horizontal)
      axis2 = world Z        (vertical)

    Algorithm
    ---------
    1. Project all points onto (depth, tangent, height) local frame.
    2. Divide the tangent range into n_slabs equal columns.
    3. For each column:
         a. Collect points in [t_lo, t_hi].
         b. depth_mid = mean depth of the points in THAT column.
            This varies per slab, creating the curved/staircase profile.
         c. depth_lo = depth_mid - slab_thickness
            depth_hi = depth_mid + slab_thickness
         d. z range = [z_min, z_max] of those points.
         e. Emit 6-face prism from (depth_lo, t_lo, z_lo) to
                                    (depth_hi, t_hi, z_hi).

    Returns
    -------
    list of prism_face_lists.
      Each entry is a list of 6 closed quad polygons (list of 5 tuples).
    """
    if len(points) < min_pts_slab:
        return []

    # Build orthonormal frame
    wn = np.asarray(wall_normal, dtype=np.float64)
    wn = wn / np.linalg.norm(wn)

    z_up   = np.array([0.0, 0.0, 1.0])
    tangent = np.cross(z_up, wn)
    t_norm  = np.linalg.norm(tangent)
    if t_norm < 1e-9:
        tangent = np.array([1.0, 0.0, 0.0])
    else:
        tangent = tangent / t_norm

    # axes columns: [wn, tangent, z_up]
    axes = np.column_stack([wn, tangent, z_up])   # (3, 3)

    # Project points
    projected = points @ axes        # (N, 3) in (depth, tang, height)

    t_min = projected[:, 1].min()
    t_max = projected[:, 1].max()
    if t_max - t_min < 1e-6:
        return []

    delta = (t_max - t_min) / n_slabs

    prism_list = []
    for i in range(n_slabs):
        t_lo = t_min + i * delta
        t_hi = t_lo + delta

        # Points in this slab column (inclusive at boundaries)
        if i < n_slabs - 1:
            mask = (projected[:, 1] >= t_lo) & (projected[:, 1] < t_hi)
        else:
            mask = (projected[:, 1] >= t_lo) & (projected[:, 1] <= t_hi)

        col_pts = projected[mask]
        if len(col_pts) < min_pts_slab:
            continue

        # Depth: use this column's own mean depth so each slab sits at its
        # actual position along the wall normal — this is what produces the
        # curved / staircase profile.
        depth_center = float(col_pts[:, 0].mean())
        depth_lo = depth_center - slab_thickness
        depth_hi = depth_center + slab_thickness

        z_lo = float(col_pts[:, 2].min())
        z_hi = float(col_pts[:, 2].max())

        if z_hi - z_lo < 1e-4:
            continue    # degenerate slab, skip

        # 8 corners in local frame (depth, tang, height)
        corners_local = np.array([
            [depth_lo, t_lo, z_lo],   # 0
            [depth_hi, t_lo, z_lo],   # 1
            [depth_hi, t_hi, z_lo],   # 2
            [depth_lo, t_hi, z_lo],   # 3
            [depth_lo, t_lo, z_hi],   # 4
            [depth_hi, t_lo, z_hi],   # 5
            [depth_hi, t_hi, z_hi],   # 6
            [depth_lo, t_hi, z_hi],   # 7
        ])

        # Back to world space:  world = local @ axes.T
        corners_world = corners_local @ axes.T   # (8, 3)

        # 6 faces (quad, CCW when viewed from outside)
        face_indices = [
            [0, 3, 2, 1],   # Bottom  (z_lo)
            [4, 5, 6, 7],   # Top     (z_hi)
            [0, 1, 5, 4],   # Front   (t_lo face)
            [3, 7, 6, 2],   # Back    (t_hi face)
            [0, 4, 7, 3],   # Left    (depth_lo, wall-parallel)
            [1, 2, 6, 5],   # Right   (depth_hi, wall-parallel)
        ]

        faces = []
        for fi in face_indices:
            ring = [tuple(corners_world[k]) for k in fi]
            ring.append(ring[0])    # close ring
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

    # Parse wall normals from GML
    print(f"\n  Parsing wall normals from {os.path.basename(gml_path)} ...")
    wall_normals = parse_wall_normals_from_gml(gml_path)

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

        n_pts  = feat['n_points']
        pts    = feat['points']
        print(f"\n  [{i+1}/{len(features)}] {ftype:10s}  {n_pts:>8,} pts  ", end="")

        # ── Wall normal ──────────────────────────────────────────
        wall_n = _best_wall_normal(pts, wall_normals)
        if wall_n is None:
            # Fallback: PCA-derived normal
            xy   = pts[:, :2]
            cxy  = xy - xy.mean(axis=0)
            cov  = cxy.T @ cxy
            _, evecs = np.linalg.eigh(cov)
            tang = evecs[:, 1]
            wall_n = np.array([-tang[1], tang[0], 0.0])

        wall_n_unit = wall_n / np.linalg.norm(wall_n)

        # ── Tessellate ───────────────────────────────────────────
        # Each slab uses its own per-column mean depth, so the depth
        # varies across slabs and follows the curved point-cloud profile.
        prism_list = tessellate_curved_cluster(
            pts,
            wall_normal    = wall_n_unit,
            n_slabs        = args.n_slabs,
            slab_thickness = args.slab_thickness,
            min_pts_slab   = args.min_pts_slab,
        )

        if not prism_list:
            print("→ no slabs emitted, skipped")
            continue

        # Flatten all prism faces into one MultiSurface per feature
        all_faces = [face for prism in prism_list for face in prism]
        n_faces   = len(all_faces)
        print(f"→ {len(prism_list)} slabs × 6 faces = {n_faces} polygons")

        # ── Convert to CityGML ───────────────────────────────────
        gml_id      = f"{ftype}_{i+1}_{uuid.uuid4().hex[:8]}"
        citygml_type = ftype if ftype in ("window", "door") else "installation"
        fragment    = feature_to_gml(citygml_type, gml_id, all_faces)
        gml_fragments.append(fragment)

        summary[ftype] = summary.get(ftype, 0) + 1

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
