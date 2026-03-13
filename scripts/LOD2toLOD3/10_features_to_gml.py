#!/usr/bin/env python3
"""
Facade Features → CityGML LOD3 Pipeline.

Clusters facade feature points with DBSCAN, creates alpha-shape meshes,
converts them to CityGML LOD3 semantics (Window, Door, BuildingInstallation),
and appends them to an existing LOD2 GML model.

Usage:
    conda activate lidar-test
    python scripts/LOD2toLOD3/10_features_to_gml.py [options]

Options:
    --eps             DBSCAN neighbourhood radius (default: 0.3)
    --min_samples     DBSCAN minimum cluster size (default: 30)
    --poisson_depth   Poisson reconstruction depth (default: 6)
"""

import os
import sys
import glob
import uuid
import argparse

import numpy as np
import laspy
from sklearn.cluster import DBSCAN

from scipy.spatial import ConvexHull


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

INPUT_DIR  = "outputs/12_manual_planes_extracted"
GML_DIR    = "outputs/00_gml_wall_merged"
OUTPUT_DIR = "outputs/trial"


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
# DBSCAN clustering — spatial proximity, then majority-vote label
# =====================================================================
def cluster_features(points, labels, eps, min_samples):
    """
    Run DBSCAN on ALL points by spatial proximity (regardless of label).
    Each resulting cluster is assigned the majority label of its points.

    Returns list of dicts:
        { 'type': str, 'points': ndarray(N,3), 'n_points': int }
    """
    print(f"\n  Running DBSCAN on {len(points):,} points ...")
    db = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_ids = db.fit_predict(points)

    unique_clusters = set(cluster_ids)
    unique_clusters.discard(-1)
    n_noise = int((cluster_ids == -1).sum())
    print(f"    → {len(unique_clusters)} spatial clusters, "
          f"{n_noise:,} noise points discarded")

    features = []
    for cid in sorted(unique_clusters):
        cmask = cluster_ids == cid
        cluster_points = points[cmask]
        cluster_labels = labels[cmask]

        # Majority-vote label
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        majority_code = unique_labels[np.argmax(counts)]
        majority_name = LABEL_NAMES.get(majority_code, "other")

        features.append({
            'type':     majority_name,
            'points':   cluster_points,
            'n_points': int(cmask.sum()),
        })

    features.sort(key=lambda f: f['n_points'], reverse=True)
    return features


def create_bbox_polygons(points):
    """
    Create a 3D Oriented Bounding Box (OBB) from a point cluster.
    Uses PCA to find the principal axes.

    Returns a list of 6 closed polygons (faces of the box).
    """
    if len(points) < 3:
        return []

    # 1. PCA to find principal axes
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    
    # Sort eigenvalues in descending order and flip eigenvectors
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    # 2. Project points onto PCA axes to find extents
    projected = centered @ eig_vecs
    p_min = projected.min(axis=0)
    p_max = projected.max(axis=0)

    # 3. Define 8 box vertices in projected space
    # (x_min, y_min, z_min), (x_max, y_min, z_min), etc.
    v_proj = np.array([
        [p_min[0], p_min[1], p_min[2]], # 0
        [p_max[0], p_min[1], p_min[2]], # 1
        [p_max[0], p_max[1], p_min[2]], # 2
        [p_min[0], p_max[1], p_min[2]], # 3
        [p_min[0], p_min[1], p_max[2]], # 4
        [p_max[0], p_min[1], p_max[2]], # 5
        [p_max[0], p_max[1], p_max[2]], # 6
        [p_min[0], p_max[1], p_max[2]], # 7
    ])

    # 4. Transform vertices back to world space
    v_world = (v_proj @ eig_vecs.T) + centroid

    # 5. Define 6 faces (indices into v_world)
    # Bottom, Top, Front, Back, Left, Right
    face_indices = [
        [0, 1, 2, 3], # Bottom (min Z in local space)
        [4, 5, 6, 7], # Top (max Z)
        [0, 1, 5, 4], # Front
        [1, 2, 6, 5], # Right
        [2, 3, 7, 6], # Back
        [3, 0, 4, 7], # Left
    ]

    polygons = []
    for indices in face_indices:
        poly = [tuple(v_world[idx]) for idx in indices]
        poly.append(poly[0]) # Close ring
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

    Window / Door  → bldg:opening inside a bldg:WallSurface
    Other          → bldg:outerBuildingInstallation
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

    args = parser.parse_args()

    print("=" * 60)
    print("  Facade Features → CityGML LOD3")
    print("=" * 60)

    # ── 1. Select point cloud ──────────────────────────────────────
    las_path = select_file(INPUT_DIR, "*.las")
    las_fname = os.path.basename(las_path).lower()
    print(f"\n  Loading: {os.path.basename(las_path)}")

    # Infer default feature type from filename
    inferred_type = "other"
    if "window" in las_fname:
        inferred_type = "window"
    elif "door" in las_fname:
        inferred_type = "door"
    print(f"  Inferred feature type: {inferred_type}")

    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    classifications = np.array(las.classification, dtype=np.uint8)
    n_total = len(points)
    print(f"  Total points: {n_total:,}")

    # Label breakdown
    print(f"\n  Label breakdown:")
    unique, counts = np.unique(classifications, return_counts=True)
    for code, count in zip(unique, counts):
        name = LABEL_NAMES.get(code, f"Unknown({code})")
        pct = 100.0 * count / n_total
        print(f"    {name:20s}: {count:>10,} pts ({pct:5.1f}%)")

    # ── 2. DBSCAN clustering ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CLUSTERING  (eps={args.eps}, min_samples={args.min_samples})")
    print(f"{'='*60}")

    features = cluster_features(points, classifications,
                                eps=args.eps,
                                min_samples=args.min_samples)

    if not features:
        print("\n  No feature clusters found. Try adjusting --eps or --min_samples.")
        return

    print(f"\n  Total feature clusters: {len(features)}")

    # ── 3. Convex hull meshing ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  MESHING  (PCA + 2D Convex Hull)")
    print(f"{'='*60}")

    gml_fragments = []
    summary = {}

    for i, feat in enumerate(features):
        # Prefer filename-based type if majority is 'other' or 'wall'
        ftype = feat['type']
        if ftype in ("other", "wall") and inferred_type != "other":
            ftype = inferred_type
        
        n_pts  = feat['n_points']
        print(f"\n  [{i+1}/{len(features)}] {ftype:12s}  {n_pts:>8,} pts  ", end="")

        polygons = create_bbox_polygons(feat['points'])

        if not polygons:
            print("→ BBox failed, skipped")
            continue

        n_verts = sum(len(p) - 1 for p in polygons)  # minus closing vertex
        print(f"→ {len(polygons)} faces, {n_verts} vertices")

        # ── 4. Convert to CityGML ────────────────────────────────
        gml_id = f"{ftype}_{i+1}_{uuid.uuid4().hex[:8]}"
        citygml_type = ftype if ftype in ("window", "door") else "installation"
        fragment = feature_to_gml(citygml_type, gml_id, polygons)
        gml_fragments.append(fragment)

        summary[ftype] = summary.get(ftype, 0) + 1

    if not gml_fragments:
        print("\n  No meshes created. Nothing to append.")
        return

    # Summary
    print(f"\n{'='*60}")
    print(f"  FEATURE SUMMARY")
    print(f"{'='*60}")
    for ftype, count in sorted(summary.items()):
        citygml = {"window": "bldg:Window", "door": "bldg:Door"}.get(
            ftype, "bldg:BuildingInstallation")
        print(f"    {ftype:12s} → {citygml:30s} × {count}")
    print(f"    {'Total':12s}   {' ':30s}   {len(gml_fragments)}")

    # ── 5. Select target GML model ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SELECT TARGET GML MODEL")
    print(f"{'='*60}")
    gml_path = select_file(GML_DIR, "*.gml")

    # ── 6. Append and save ────────────────────────────────────────
    gml_basename = os.path.splitext(os.path.basename(gml_path))[0]
    out_name = f"{gml_basename}_LOD3.gml"
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
