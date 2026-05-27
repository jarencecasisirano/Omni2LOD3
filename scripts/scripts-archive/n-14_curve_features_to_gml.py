#!/usr/bin/env python3
"""
Tessellated Curve Features → CityGML LOD3 Pipeline.

For each .las file in a user-selected folder (outputs/13_facade_curve_tesselated
or outputs/12_facade_curve), the script:
  1. Asks the user to select a feature type ONCE for all clusters / all files:
       w) Window              → bldg:Window  (inside bldg:WallSurface)
       d) Door                → bldg:Door    (inside bldg:WallSurface)
       i) BuildingInstallation→ bldg:outerBuildingInstallation
  2. Runs DBSCAN on each .las file to find spatial clusters.
  3. Creates a wall-aligned bounding box (6-face OBB) for every cluster.
  4. Assigns the chosen feature type to every cluster automatically.
  5. Appends all features to the chosen LOD2 .gml model.

Usage
-----
    conda activate las-env
    python scripts/LOD2toLOD3/14_curve_features_to_gml.py [options]

Options
-------
    --eps           DBSCAN neighbourhood radius, metres (default 0.3)
    --min_samples   DBSCAN minimum cluster size         (default 20)
    --output_dir    Output directory (default: outputs/14_curve_features_gml)
"""

import os
import sys
import glob
import uuid
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import laspy
from sklearn.cluster import DBSCAN


# ============================================================================
#  Constants
# ============================================================================
BASE_DIR   = Path(__file__).resolve().parents[2]   # project root

INPUT_BASE = BASE_DIR / "outputs" / "13_facade_curve_tesselated"
GML_DIR    = BASE_DIR / "outputs" / "00_gml_wall_merged"
OUTPUT_DIR = BASE_DIR / "outputs" / "14_curve_features_gml"

LABEL_MAP = {
    "other":  1,
    "wall":   2,
    "door":   3,
    "window": 4,
    "roof":   5,
    "ground": 6,
}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

# User assignment short-codes
ASSIGN_CODES = {
    "w": "window",
    "d": "door",
    "i": "installation",
    "s": "skip",
}


# ============================================================================
#  Interactive selectors
# ============================================================================

def select_folder(base_dir: Path) -> Path:
    """
    List immediate sub-folders inside *base_dir* and let the user choose one,
    OR accept base_dir itself if it contains .las files directly.
    """
    # Collect candidate directories: base_dir itself + its immediate children
    candidates = []

    # Check whether base_dir contains .las files directly
    if list(base_dir.glob("*.las")):
        candidates.append(base_dir)

    for p in sorted(base_dir.iterdir()):
        if p.is_dir() and list(p.glob("*.las")):
            candidates.append(p)

    if not candidates:
        print(f"  ERROR: No folders with .las files found under {base_dir}")
        sys.exit(1)

    if len(candidates) == 1:
        print(f"  Auto-selected folder: {candidates[0]}")
        return candidates[0]

    print(f"\n{'='*60}")
    print(f"  Select input folder  (must contain .las files)")
    print(f"{'='*60}")
    for i, p in enumerate(candidates):
        n = len(list(p.glob("*.las")))
        print(f"  [{i+1}] {p.relative_to(base_dir.parent)!s:55s}  ({n} .las file(s))")
    print()

    while True:
        try:
            choice = input(f"  Select folder [1-{len(candidates)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except (ValueError, KeyboardInterrupt):
            pass
        print("  Invalid choice, try again.")


def select_file(directory: Path, pattern: str = "*.gml") -> Path:
    """List files matching *pattern* in *directory* and let the user pick one."""
    files = sorted(directory.glob(pattern))
    if not files:
        print(f"  ERROR: No {pattern} files found in {directory}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Files in: {directory}")
    print(f"{'='*60}")
    for i, f in enumerate(files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  [{i+1}] {f.name:50s}  ({size_mb:.2f} MB)")
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


# ============================================================================
#  DBSCAN clustering
# ============================================================================

def run_dbscan(points: np.ndarray, eps: float, min_samples: int):
    """
    Apply DBSCAN directly to the 3-D point array.

    Returns
    -------
    list of dicts  { 'id': int, 'points': ndarray(N,3), 'n_points': int }
    sorted by point count descending (noise cluster -1 excluded).
    """
    print(f"  Running DBSCAN (eps={eps}, min_samples={min_samples}) "
          f"on {len(points):,} points …")

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_

    unique_ids = set(labels)
    unique_ids.discard(-1)
    n_noise = int((labels == -1).sum())

    print(f"  → {len(unique_ids)} cluster(s), {n_noise:,} noise points discarded")

    clusters = []
    for cid in sorted(unique_ids):
        mask = labels == cid
        cluster_pts = points[mask]
        clusters.append({
            "id":       cid,
            "points":   cluster_pts,
            "n_points": int(mask.sum()),
        })

    clusters.sort(key=lambda c: c["n_points"], reverse=True)
    return clusters


# ============================================================================
#  Wall-surface parsing and intersection-based normal selection
# ============================================================================

def parse_wall_surfaces_from_gml(gml_path: Path):
    """
    Parse every WallSurface exterior polygon from a CityGML file.

    For each polygon the following dict is returned::

        {
          'coords':    ndarray(N, 3),   # raw ring vertices (may repeat first)
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
        tree = ET.parse(str(gml_path))
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  WARNING: could not parse GML: {e}")
        return []

    ns_bldg = "http://www.opengis.net/citygml/building/2.0"
    ns_gml  = "http://www.opengis.net/gml"

    surfaces = []
    for ws in root.iter(f"{{{ns_bldg}}}WallSurface"):
        for poly in ws.iter(f"{{{ns_gml}}}Polygon"):
            exterior = poly.find(f".//{{{ns_gml}}}exterior")
            if exterior is None:
                continue
            pos_el = exterior.find(f".//{{{ns_gml}}}posList")
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
                continue   # horizontal polygon — skip

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


def find_wall_normal_for_cluster(
        points: np.ndarray,
        wall_surfaces,
        dist_tol: float = 2.0,
        z_expand: float = 0.5,
):
    """
    Find the WallSurface from *wall_surfaces* that spatially intersects the
    cluster and return its **exact** horizontal 2-D unit normal as a 3-D
    vector with Z = 0.

    Intersection criteria (all must hold):
      1. Z-range overlap  — cluster Z range overlaps the WallSurface Z range
                            (expanded by *z_expand* metres on each side).
      2. Signed-distance  — the cluster centroid's signed distance to the
                            wall plane is |d| < *dist_tol*.
      3. XY footprint      — the cluster centroid's XY projection onto the
                            wall plane lies within the WallSurface XY AABB
                            (expanded by *dist_tol*).

    Among candidates the one with the smallest |signed distance| wins.

    Falls back to a PCA-based normal when no WallSurface matches.
    """
    if not wall_surfaces:
        return _pca_wall_normal(points)

    cl_z_min = float(points[:, 2].min())
    cl_z_max = float(points[:, 2].max())
    centroid  = points.mean(axis=0)
    cxy       = centroid[:2]

    best_normal = None
    best_dist   = float('inf')

    for ws in wall_surfaces:
        # ── 1. Z overlap ───────────────────────────────────────────────
        ws_z_lo = ws['z_min'] - z_expand
        ws_z_hi = ws['z_max'] + z_expand
        if cl_z_max < ws_z_lo or cl_z_min > ws_z_hi:
            continue

        # ── 2. Signed distance of centroid to wall plane ───────────────
        n2d = ws['normal_2d']                          # unit vector
        d   = float(np.dot(cxy - ws['origin_2d'], n2d))
        if abs(d) >= dist_tol:
            continue

        # ── 3. XY footprint: project centroid onto the wall plane ──────
        proj_xy = cxy - d * n2d                        # onto wall plane
        pad = dist_tol
        if (proj_xy[0] < ws['xy_min'][0] - pad or
                proj_xy[0] > ws['xy_max'][0] + pad or
                proj_xy[1] < ws['xy_min'][1] - pad or
                proj_xy[1] > ws['xy_max'][1] + pad):
            continue

        # Candidate — prefer smallest absolute distance
        if abs(d) < best_dist:
            best_dist   = abs(d)
            best_normal = n2d

    if best_normal is not None:
        angle = np.degrees(np.arctan2(best_normal[1], best_normal[0]))
        print(f"[Stage1: dist={best_dist:.2f} m, angle={angle:+.1f}°]", end=" ")
        return np.array([best_normal[0], best_normal[1], 0.0])

    # ── Stage 2: alignment with exact per-surface normals ────────────────
    pca_n = _pca_normal_2d(points)
    if pca_n is not None:
        best_dot     = -1.0
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


def _pca_normal_2d(points: np.ndarray):
    """Return PCA wall-normal estimate as a 2-D unit vector, or None."""
    if len(points) < 3:
        return None
    xy  = points[:, :2]
    cxy = xy - xy.mean(axis=0)
    cov = cxy.T @ cxy
    _, eig_vecs = np.linalg.eigh(cov)
    tang = eig_vecs[:, 1]
    return np.array([-tang[1], tang[0]])


def _pca_wall_normal(points: np.ndarray):
    """
    Estimate the wall normal from the cluster's dominant horizontal spread
    (PCA in XY).  Returns a 3-D vector with Z = 0, or *None* if degenerate.
    """
    n2d = _pca_normal_2d(points)
    if n2d is None:
        return None
    return np.array([n2d[0], n2d[1], 0.0])


# ============================================================================
#  Bounding-box construction
# ============================================================================

def create_bbox_polygons(points: np.ndarray, wall_normal=None):
    """
    Build a wall-aligned oriented bounding box from *points*.

    If *wall_normal* (unit 3-D vector with Z=0) is given the box is oriented
    so that two faces are parallel to the wall.  Otherwise PCA is used.

    Returns a list of 6 closed quad polygons (each a list of 5 (x,y,z) tuples).
    """
    if len(points) < 3:
        return []

    centroid = points.mean(axis=0)
    centered = points - centroid

    if wall_normal is not None:
        wn      = np.asarray(wall_normal, dtype=np.float64)
        wn      = wn / np.linalg.norm(wn)
        z_up    = np.array([0.0, 0.0, 1.0])
        tangent = np.cross(z_up, wn)
        t_norm  = np.linalg.norm(tangent)
        tangent = tangent / t_norm if t_norm > 1e-9 else np.array([1.0, 0.0, 0.0])
        axes    = np.column_stack([wn, tangent, z_up])   # (3,3)
    else:
        cov = np.cov(centered, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        axes = eig_vecs[:, eig_vals.argsort()[::-1]]

    projected = centered @ axes     # (N, 3) in OBB frame
    p_min = projected.min(axis=0)
    p_max = projected.max(axis=0)

    # 8 corners in local OBB frame
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

    v_world = (v_proj @ axes.T) + centroid   # back to world space

    face_indices = [
        [0, 1, 2, 3],   # Bottom
        [4, 7, 6, 5],   # Top
        [0, 3, 7, 4],   # Wall-parallel A (min depth)
        [1, 5, 6, 2],   # Wall-parallel B (max depth)
        [0, 4, 5, 1],   # Side A
        [3, 2, 6, 7],   # Side B
    ]

    polygons = []
    for idxs in face_indices:
        ring = [tuple(v_world[k]) for k in idxs]
        ring.append(ring[0])   # close ring
        polygons.append(ring)

    return polygons


# ============================================================================
#  CityGML XML helpers
# ============================================================================

def _make_multi_surface(gml_id: str, polygons) -> str:
    """Build a <gml:MultiSurface> XML string from a list of polygon rings."""
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


def feature_to_gml(feature_type: str, gml_id: str, polygons) -> str:
    """
    Generate a CityGML XML fragment for one feature.

    'window'       → bldg:Window inside bldg:WallSurface / bldg:opening
    'door'         → bldg:Door   inside bldg:WallSurface / bldg:opening
    'installation' → bldg:outerBuildingInstallation
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

    else:  # installation
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


# ============================================================================
#  GML file I/O
# ============================================================================

def append_to_gml(gml_path: Path, gml_fragments, output_path: Path) -> bool:
    """Insert GML fragments before the closing </bldg:Building> tag."""
    content = gml_path.read_text(encoding="utf-8")

    marker = "</bldg:Building>"
    pos    = content.rfind(marker)
    if pos == -1:
        print("  ERROR: Could not find </bldg:Building> in the GML file.")
        return False

    insertion   = "\n" + "\n".join(gml_fragments) + "\n  "
    new_content = content[:pos] + insertion + content[pos:]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(new_content, encoding="utf-8")
    return True


# ============================================================================
#  Global feature-type selector  (asked once, applied to all clusters)
# ============================================================================

def prompt_global_assignment() -> str:
    """
    Ask the user to choose a single feature type that will be applied to
    every DBSCAN cluster across all .las files.

    Returns one of: 'window' | 'door' | 'installation'
    """
    print(f"\n{'='*60}")
    print(f"  Step 3: Select feature type for ALL clusters")
    print(f"{'='*60}")
    print(f"    w) Window               → bldg:Window")
    print(f"    d) Door                 → bldg:Door")
    print(f"    i) BuildingInstallation → bldg:outerBuildingInstallation")
    print()

    valid = {"w": "window", "d": "door", "i": "installation"}
    while True:
        try:
            code = input("  Feature type [w/d/i]: ").strip().lower()
            if code in valid:
                chosen = valid[code]
                type_label = {
                    "window":       "bldg:Window",
                    "door":         "bldg:Door",
                    "installation": "bldg:BuildingInstallation",
                }[chosen]
                print(f"  → All clusters will be tagged as: {type_label}")
                return chosen
        except KeyboardInterrupt:
            print("\n  Aborted.")
            sys.exit(0)
        print("  Invalid code — please enter w, d, or i.")


def _bbox_dimensions(points: np.ndarray, wall_normal=None):
    """Return (width_m, height_m, depth_m) of the OBB for display purposes."""
    if len(points) < 2:
        return (0.0, 0.0, 0.0)

    centroid = points.mean(axis=0)
    centered = points - centroid

    if wall_normal is not None:
        wn      = wall_normal / np.linalg.norm(wall_normal)
        z_up    = np.array([0.0, 0.0, 1.0])
        tangent = np.cross(z_up, wn)
        t_norm  = np.linalg.norm(tangent)
        tangent = tangent / t_norm if t_norm > 1e-9 else np.array([1.0, 0.0, 0.0])
        axes    = np.column_stack([wn, tangent, z_up])
    else:
        cov = np.cov(centered, rowvar=False)
        _, eig_vecs = np.linalg.eigh(cov)
        axes = eig_vecs[:, ::-1]

    proj  = centered @ axes
    spans = proj.max(axis=0) - proj.min(axis=0)
    # spans: [depth, width_along_wall, height]
    return float(spans[1]), float(spans[2]), float(spans[0])


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tessellated curve features → CityGML LOD3")
    parser.add_argument("--eps", type=float, default=0.3,
                        help="DBSCAN neighbourhood radius in metres (default 0.3)")
    parser.add_argument("--min_samples", type=int, default=20,
                        help="DBSCAN minimum cluster size (default 20)")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for the enriched .gml file")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    print("=" * 60)
    print("  14 · Curve Features → CityGML LOD3")
    print("=" * 60)

    # ── Step 1: select folder containing .las files ──────────────────
    print("\nStep 1: Select input folder (must contain .las files)")
    print(f"  (searching under {INPUT_BASE})")
    las_folder = select_folder(INPUT_BASE)
    las_files  = sorted(las_folder.glob("*.las"))
    print(f"\n  Folder : {las_folder}")
    print(f"  Files  : {len(las_files)} .las file(s)")

    # ── Step 2: select GML model ─────────────────────────────────────
    print("\nStep 2: Select target GML model")
    gml_path = select_file(GML_DIR, "*.gml")
    print(f"  GML    : {gml_path}")

    # Parse WallSurface geometry (coords + exact normals) for intersection tests
    print(f"\n  Parsing WallSurfaces from {gml_path.name} …")
    wall_surfaces = parse_wall_surfaces_from_gml(gml_path)

    # ── Step 3: global feature-type selection ─────────────────────────
    global_assignment = prompt_global_assignment()

    # ── Main processing loop ─────────────────────────────────────────
    all_gml_fragments = []
    global_summary    = {}   # feature_type → count

    for las_idx, las_path in enumerate(las_files):
        size_mb = las_path.stat().st_size / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"  File {las_idx+1}/{len(las_files)}: {las_path.name}  ({size_mb:.2f} MB)")
        print(f"{'='*60}")

        # Load point cloud
        las = laspy.read(str(las_path))
        points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
        n_total = len(points)
        print(f"  Total points: {n_total:,}")

        if n_total < args.min_samples:
            print(f"  WARNING: fewer points than min_samples={args.min_samples}, skipping file.")
            continue

        # ── Step 3: DBSCAN clustering ─────────────────────────────────
        print(f"\n  CLUSTERING")
        clusters = run_dbscan(points, eps=args.eps, min_samples=args.min_samples)

        if not clusters:
            print("  No clusters found — try adjusting --eps or --min_samples.")
            continue

        print(f"  Processing {len(clusters)} cluster(s) …")

        # ── Step 4: bounding box per cluster, apply global type ───────
        file_fragments = []
        file_summary   = {}

        for cl_idx, cluster in enumerate(clusters):
            pts   = cluster["points"]
            n_pts = cluster["n_points"]

            # Find the intersecting WallSurface and use its exact normal
            # so the bounding-box faces are coplanar with the LOD2 wall.
            cl_wall_n = find_wall_normal_for_cluster(pts, wall_surfaces)

            # Bounding box polygons
            polygons = create_bbox_polygons(pts, wall_normal=cl_wall_n)
            if not polygons:
                print(f"  Cluster {cl_idx+1}: degenerate (< 3 pts), skipped.")
                continue

            # Apply the globally selected feature type
            assignment = global_assignment
            dims = _bbox_dimensions(pts, wall_normal=cl_wall_n)
            w, h, d_ = dims
            type_label = {
                "window":       "bldg:Window",
                "door":         "bldg:Door",
                "installation": "bldg:BuildingInstallation",
            }[assignment]
            print(f"  Cluster {cl_idx+1}/{len(clusters)}  "
                  f"({n_pts:,} pts)  "
                  f"BBox W={w:.2f} H={h:.2f} D={d_:.2f} m  "
                  f"→ {type_label}")

            gml_id   = f"{assignment}_{las_idx+1}_{cl_idx+1}_{uuid.uuid4().hex[:8]}"
            fragment = feature_to_gml(assignment, gml_id, polygons)
            file_fragments.append(fragment)
            file_summary[assignment]  = file_summary.get(assignment, 0) + 1
            global_summary[assignment] = global_summary.get(assignment, 0) + 1

        # Per-file summary
        if file_fragments:
            print(f"\n  File summary:")
            for ftype, cnt in sorted(file_summary.items()):
                print(f"    {ftype:20s} × {cnt}")
            all_gml_fragments.extend(file_fragments)
        else:
            print("\n  No features assigned for this file.")

    # ── Final output ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  GLOBAL SUMMARY")
    print(f"{'='*60}")
    if not all_gml_fragments:
        print("  No features to append — nothing was saved.")
        return

    for ftype, cnt in sorted(global_summary.items()):
        type_label = {
            "window":       "bldg:Window",
            "door":         "bldg:Door",
            "installation": "bldg:BuildingInstallation",
        }.get(ftype, ftype)
        print(f"  {ftype:20s} → {type_label:30s} × {cnt}")
    print(f"  {'Total':20s}   {'':30s}   {len(all_gml_fragments)}")

    # Build output path
    gml_stem  = gml_path.stem
    out_name  = f"{gml_stem}_LOD3_curve_features.gml"
    out_path  = out_dir / out_name

    print(f"\n  Appending {len(all_gml_fragments)} feature(s) to {gml_path.name} …")
    success = append_to_gml(gml_path, all_gml_fragments, out_path)

    if success:
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved: {out_path}  ({size_mb:.2f} MB)")
    else:
        print(f"  ✗ Failed to write output.")

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
