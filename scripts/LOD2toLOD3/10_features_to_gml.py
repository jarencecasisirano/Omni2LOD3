#!/usr/bin/env python3
"""
Facade Features → CityGML LOD3 Pipeline.

Each run selects one GML model and one or more .las point clouds.  For every
point cloud the user manually chooses the CityGML feature type (Window, Door,
or BuildingInstallation).  All clouds are DBSCAN-clustered, bounding-boxed,
and combined into a single output GML file.

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
GML_DIR    = "outputs/12_curve_gml"
OUTPUT_DIR = "outputs/final"


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
# Parse wall normals from a LOD2 GML file
# =====================================================================
def parse_wall_normals_from_gml(gml_path, angle_tol_deg=5.0):
    """
    Extract horizontal (XY-plane) unit normals from every WallSurface polygon
    found inside a CityGML file, then cluster near-parallel normals and return
    only the *dominant* direction for each cluster.

    This avoids noisy slivers biasing the normal selection.  Normals that
    differ by less than *angle_tol_deg* are merged.

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

    # ── Cluster near-parallel normals and keep dominant directions ──
    # Two normals are "same direction" if the angle between them (or their
    # opposites) is < angle_tol_deg.
    tol_cos = np.cos(np.radians(angle_tol_deg))
    dominant = []  # list of (representative_normal, count)
    raw_normals_arr = np.array(raw_normals)

    for nv in raw_normals_arr:
        matched = False
        for idx, (rep, cnt) in enumerate(dominant):
            # Accept n or -n as same wall face
            if abs(float(np.dot(nv, rep))) >= tol_cos:
                # Running average (Welford-style direction update)
                new_rep = rep * cnt + nv * np.sign(float(np.dot(nv, rep)))
                nr = np.linalg.norm(new_rep)
                dominant[idx] = (new_rep / nr if nr > 1e-9 else rep, cnt + 1)
                matched = True
                break
        if not matched:
            dominant.append((nv, 1))

    # Sort by frequency, return the most common directions
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
    return the wall normal whose perpendicular direction best aligns with the
    cluster's dominant horizontal spread axis (i.e. the cluster lies *on* a
    wall, so the spread axis is the wall tangent and the normal is perpendicular
    to that).

    Returns a normalised 3-D vector (wall normal, with Z=0).
    """
    if not wall_normals:
        return None

    # PCA on XY to find the horizontal spread direction (tangent to the wall)
    xy = points[:, :2]
    cxy = xy - xy.mean(axis=0)
    cov = cxy.T @ cxy
    eig_vals, eig_vecs = np.linalg.eigh(cov)  # ascending order
    # Largest eigenvalue → first principal direction = wall tangent (along wall)
    tangent_2d = eig_vecs[:, 1]   # index 1 = largest

    # The wall normal is perpendicular to the tangent in XY
    cluster_normal_2d = np.array([-tangent_2d[1], tangent_2d[0]])

    # Pick the candidate wall normal most aligned with our estimate
    best_n = None
    best_dot = -1.0
    for wn in wall_normals:
        dot = abs(float(np.dot(cluster_normal_2d, wn)))
        if dot > best_dot:
            best_dot = dot
            best_n = wn

    if best_n is None:
        return None

    return np.array([best_n[0], best_n[1], 0.0])


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

    print(f"\n  Parsing wall normals from {os.path.basename(gml_path)} ...")
    wall_normals = parse_wall_normals_from_gml(gml_path)

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
        print(f"\n  ✓ Added: {os.path.basename(las_path):45s} → {feature_type}")

        again = input("\n  Add another point cloud? [y/N]: ").strip().lower()
        if again != 'y':
            break

    print(f"\n  {len(las_inputs)} point cloud(s) selected:")
    for lp, ft in las_inputs:
        print(f"    {os.path.basename(lp):45s} → {ft}")

    # ── 3. DBSCAN clustering + meshing ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CLUSTERING + MESHING  "
          f"(eps={args.eps}, min_samples={args.min_samples})")
    print(f"{'='*60}")

    gml_fragments = []
    summary       = {}
    feat_counter  = 0   # global index across all clouds

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

            wall_n   = _best_wall_normal(feat['points'], wall_normals)
            polygons = create_bbox_polygons(feat['points'], wall_normal=wall_n)

            if not polygons:
                print("→ BBox failed, skipped")
                continue

            n_verts = sum(len(p) - 1 for p in polygons)
            print(f"→ {len(polygons)} faces, {n_verts} vertices")

            gml_id       = f"{ftype}_{feat_counter}_{uuid.uuid4().hex[:8]}"
            citygml_type = ftype if ftype in ("window", "door") else "installation"
            fragment     = feature_to_gml(citygml_type, gml_id, polygons)
            gml_fragments.append(fragment)

            summary[ftype] = summary.get(ftype, 0) + 1

    if not gml_fragments:
        print("\n  No meshes created. Nothing to append.")
        return

    # ── 4. Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FEATURE SUMMARY")
    print(f"{'='*60}")
    for ftype, count in sorted(summary.items()):
        citygml = {"window": "bldg:Window",
                   "door":   "bldg:Door"}.get(ftype, "bldg:BuildingInstallation")
        print(f"    {ftype:12s} → {citygml:30s} × {count}")
    print(f"    {'Total':12s}   {' ':30s}   {len(gml_fragments)}")

    # ── 5. Append and save ───────────────────────────────────────
    if args.output:
        out_name = args.output if args.output.endswith(".gml") else args.output + ".gml"
    else:
        gml_basename = os.path.splitext(os.path.basename(gml_path))[0]
        out_name     = f"{gml_basename}_LOD3.gml"
    output_path = os.path.join(OUTPUT_DIR, out_name)

    print(f"\n  Appending {len(gml_fragments)} LOD3 features "
          f"to {os.path.basename(gml_path)}...")

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
