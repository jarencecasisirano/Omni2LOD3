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

INPUT_DIR  = "outputs/10_facade_features"
GML_DIR    = "outputs/00_gml_wall_merged"
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


# =====================================================================
# Convex hull mesh via PCA plane projection (no Open3D — no segfaults)
# =====================================================================
def create_mesh(points):
    """
    Create a convex hull polygon from a point cluster.

    Projects points onto their best-fit plane (PCA), computes
    a 2D convex hull with scipy, and maps back to 3D.

    Returns a list containing one closed polygon
    (list of (x,y,z) tuples with first == last for GML).
    """
    if len(points) < 3:
        return []

    centroid = points.mean(axis=0)
    centered = points - centroid

    # PCA: eigenvectors of covariance, sorted ascending by eigenvalue.
    # The two LARGEST eigenvectors span the best-fit plane.
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    basis = eigenvectors[:, 1:]          # last 2 columns = plane basis

    # Project to 2D
    pts_2d = centered @ basis

    try:
        hull = ConvexHull(pts_2d)
    except Exception:
        return []

    # hull.vertices are in CCW order for 2D
    hull_3d = points[hull.vertices]

    poly = [tuple(p) for p in hull_3d]
    poly.append(poly[0])               # close ring

    return [poly]


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
    print(f"\n  Loading: {os.path.basename(las_path)}")

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
        ftype  = feat['type']
        n_pts  = feat['n_points']
        print(f"\n  [{i+1}/{len(features)}] {ftype:12s}  {n_pts:>8,} pts  ", end="")

        polygons = create_mesh(feat['points'])

        if not polygons:
            print("→ mesh failed, skipped")
            continue

        n_verts = sum(len(p) - 1 for p in polygons)  # minus closing vertex
        print(f"→ {len(polygons)} polygon(s), {n_verts} vertices")

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
