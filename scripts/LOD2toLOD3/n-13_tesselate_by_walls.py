#!/usr/bin/env python3
"""
13_tesselate_by_walls.py — Segment a facade point cloud using wall-edge planes.

Pipeline
--------
1. User selects a .gml file from outputs/00_gml_wall_merged
2. User selects a .las file from outputs/12_facade_curve
3. WallSurfaces are extracted from the GML
4. Shared vertical edges between walls are identified
5. One vertical cutting plane is created per edge (plane is tangent to the edge)
6. The planes partition the entire point cloud into spatial segments
7. Each segment is saved as a separate .las file in outputs/13_facade_curve_tesselated

Usage
-----
    conda activate las-env
    python scripts/LOD2toLOD3/13_tesselate_by_walls.py

Dependencies
------------
    laspy, numpy, lxml (or stdlib xml.etree.ElementTree as fallback)
"""

import os
import sys
import glob
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import laspy

# ── Optional lxml for faster parsing ────────────────────────────────────────
try:
    from lxml import etree as _lxml_etree
    _HAS_LXML = True
except ImportError:
    _HAS_LXML = False

# ============================================================================
#  Constants
# ============================================================================
BASE_DIR   = Path(__file__).resolve().parents[2]   # project root
GML_DIR    = BASE_DIR / "outputs" / "00_gml_wall_merged"
LAS_DIR    = BASE_DIR / "outputs" / "12_facade_curve"
OUTPUT_DIR = BASE_DIR / "outputs" / "13_facade_curve_tesselated"

GML_NS = {
    "gml":  "http://www.opengis.net/gml",
    "bldg": "http://www.opengis.net/citygml/building/2.0",
    "core": "http://www.opengis.net/citygml/2.0",
}

# Edges whose endpoints differ in Z by less than this are considered horizontal
# (and thus NOT wall-edge separators).
VERTICAL_EDGE_MIN_DZ = 0.10   # metres

# Two edges are "the same" if both endpoints coincide within this tolerance
EDGE_MERGE_TOL = 0.05   # metres

# Minimum number of points to keep a segment (smaller ones are discarded)
MIN_SEGMENT_PTS = 5


# ============================================================================
#  Interactive file selector (console)
# ============================================================================

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
        print("  Invalid choice, please try again.")


# ============================================================================
#  GML Parsing
# ============================================================================

def _coords_from_poslist(text: str) -> Optional[np.ndarray]:
    """Parse a gml:posList text into an (N, 3) array, or None on failure."""
    vals = text.split()
    if len(vals) < 9:
        return None
    try:
        arr = np.array(vals, dtype=np.float64).reshape(-1, 3)
    except ValueError:
        return None
    return arr


def extract_wall_surfaces(gml_path: Path) -> List[np.ndarray]:
    """
    Parse a CityGML file and return one (N, 3) coordinate array per
    WallSurface polygon found.
    """
    print(f"\n  Parsing GML: {gml_path.name}")

    if _HAS_LXML:
        tree = _lxml_etree.parse(str(gml_path))
        root = tree.getroot()
        ws_elements = root.xpath("//bldg:WallSurface", namespaces=GML_NS)
    else:
        tree = ET.parse(str(gml_path))
        root = tree.getroot()
        ws_tag = f"{{{GML_NS['bldg']}}}WallSurface"
        ws_elements = root.iter(ws_tag)

    polygons: List[np.ndarray] = []

    for ws in ws_elements:
        if _HAS_LXML:
            pos_lists = ws.xpath(".//gml:posList", namespaces=GML_NS)
        else:
            pos_tag = f"{{{GML_NS['gml']}}}posList"
            pos_lists = list(ws.iter(pos_tag))

        for pl in pos_lists:
            text = (pl.text or "").strip()
            if not text:
                continue
            coords = _coords_from_poslist(text)
            if coords is not None and len(coords) >= 3:
                polygons.append(coords)

    print(f"  Found {len(polygons)} WallSurface polygon(s)")
    if not polygons:
        print("  WARNING: No WallSurface polygons found in GML.")
    return polygons


# ============================================================================
#  Edge Extraction
# ============================================================================

def polygon_edges(poly: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Return all edges of a closed polygon ring as (p0, p1) pairs.
    Strips the repeated closing vertex if present.
    """
    verts = poly
    if len(verts) > 1 and np.allclose(verts[0], verts[-1], atol=1e-6):
        verts = verts[:-1]
    edges = []
    n = len(verts)
    for i in range(n):
        edges.append((verts[i].copy(), verts[(i + 1) % n].copy()))
    return edges


def is_vertical_edge(p0: np.ndarray, p1: np.ndarray,
                     min_dz: float = VERTICAL_EDGE_MIN_DZ) -> bool:
    """
    Return True if the edge p0→p1 is predominantly vertical:
    - The vertical component (ΔZ) must exceed *min_dz*.
    - The horizontal displacement must be negligible relative to ΔZ
      (i.e. the edge is nearly a plumb line).
    """
    dz  = abs(p1[2] - p0[2])
    dxy = np.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
    if dz < min_dz:
        return False
    # Vertical means dz dominates — require dxy < dz (i.e. tilt < 45°)
    return dxy < dz


def edges_match(a0: np.ndarray, a1: np.ndarray,
                b0: np.ndarray, b1: np.ndarray,
                tol: float = EDGE_MERGE_TOL) -> bool:
    """True if edges (a0,a1) and (b0,b1) are the same segment (either direction)."""
    fwd = (np.linalg.norm(a0 - b0) < tol and np.linalg.norm(a1 - b1) < tol)
    rev = (np.linalg.norm(a0 - b1) < tol and np.linalg.norm(a1 - b0) < tol)
    return fwd or rev


def extract_vertical_wall_edges(
        polygons: List[np.ndarray],
        require_shared: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract all vertical edges from WallSurface polygons.

    Parameters
    ----------
    polygons       : list of (N,3) polygon arrays from extract_wall_surfaces().
    require_shared : if True, only return edges shared by ≥2 polygons.
                     Shared edges represent the junctions between adjacent walls.
                     If False, ALL vertical edges are returned (one per polygon
                     boundary), which gives one cutting plane per wall side.

    Returns
    -------
    List of (p0, p1) pairs representing unique vertical edges.
    """
    # Collect all vertical edges from all polygons
    all_vert: List[Tuple[np.ndarray, np.ndarray, int]] = []  # (p0, p1, polygon_idx)
    for poly_idx, poly in enumerate(polygons):
        for p0, p1 in polygon_edges(poly):
            if is_vertical_edge(p0, p1):
                all_vert.append((p0, p1, poly_idx))

    print(f"  Total vertical edge candidates: {len(all_vert)}")

    if not require_shared:
        # Deduplicate (same edge may appear in the same polygon twice due to ring)
        unique: List[Tuple[np.ndarray, np.ndarray]] = []
        for p0, p1, _ in all_vert:
            already = any(edges_match(p0, p1, u0, u1) for u0, u1 in unique)
            if not already:
                unique.append((p0, p1))
        return unique

    # require_shared=True: keep only edges seen in ≥2 different polygons
    shared: List[Tuple[np.ndarray, np.ndarray]] = []
    n = len(all_vert)
    used = [False] * n
    for i in range(n):
        if used[i]:
            continue
        p0i, p1i, idx_i = all_vert[i]
        partners = [j for j in range(i + 1, n)
                    if not used[j]
                    and all_vert[j][2] != idx_i
                    and edges_match(p0i, p1i, all_vert[j][0], all_vert[j][1])]
        if partners:
            shared.append((p0i, p1i))
            used[i] = True
            for j in partners:
                used[j] = True

    return shared


# ============================================================================
#  Vertical Plane Creation
# ============================================================================

class VerticalPlane:
    """
    An infinite vertical plane defined by:
      normal  — 2-D horizontal unit vector (XY-plane), stored as (nx, ny, 0)
      origin  — any point on the plane, used to compute signed distance

    The plane equation is:
        nx*(x - ox) + ny*(y - oy) = 0
    i.e.   signed_distance(p) = nx*(px - ox) + ny*(py - oy)
    """
    def __init__(self, p0: np.ndarray, p1: np.ndarray):
        # Edge direction (horizontal component only)
        d = p1[:2] - p0[:2]
        d_len = np.linalg.norm(d)
        if d_len < 1e-9:
            # Degenerate — edge is pure vertical (identical XY), use arbitrary normal
            self.normal = np.array([1.0, 0.0])
        else:
            t = d / d_len                    # tangent in XY
            self.normal = np.array([-t[1], t[0]])   # left-hand normal

        # Mid-point of the edge as origin
        self.origin = ((p0 + p1) / 2.0)[:2]

    def signed_distance(self, pts_xy: np.ndarray) -> np.ndarray:
        """
        Signed distance of each point to this plane.

        pts_xy : (N, 2) X-Y coordinates
        Returns: (N,) signed distances, positive on one side, negative on the other
        """
        return (pts_xy - self.origin[np.newaxis, :]) @ self.normal


def build_planes(edges: List[Tuple[np.ndarray, np.ndarray]]) -> List[VerticalPlane]:
    """Create one VerticalPlane per edge."""
    planes = [VerticalPlane(p0, p1) for p0, p1 in edges]
    return planes


# ============================================================================
#  Point Cloud Segmentation
# ============================================================================

def segment_point_cloud(
        pts: np.ndarray,
        planes: List[VerticalPlane],
) -> List[Tuple[int, np.ndarray]]:
    """
    Partition the point cloud into segments using the list of vertical planes.

    Each point is assigned a binary "side" vector (one bit per plane: +1 or -1)
    which is converted into a unique integer segment key.

    Returns
    -------
    List of (segment_id, mask_indices) pairs sorted by number of points desc.
    """
    n_pts    = len(pts)
    n_planes = len(planes)
    pts_xy   = pts[:, :2]

    if n_planes == 0:
        return [(0, np.arange(n_pts))]

    # Build a (N, n_planes) sign matrix: +1 / -1 per plane per point
    signs = np.ones((n_pts, n_planes), dtype=np.int8)
    for j, plane in enumerate(planes):
        d = plane.signed_distance(pts_xy)
        signs[:, j] = np.where(d >= 0, 1, -1)

    # Map each unique row (bit vector) to an integer segment id
    # We use a dict keyed on a tuple of the sign row
    seg_dict = {}
    row_keys  = [tuple(signs[i]) for i in range(n_pts)]

    for i, key in enumerate(row_keys):
        if key not in seg_dict:
            seg_dict[key] = []
        seg_dict[key].append(i)

    segments = []
    for seg_id, (key, indices) in enumerate(seg_dict.items()):
        segments.append((seg_id, np.array(indices, dtype=np.int64)))

    # Sort by point count descending
    segments.sort(key=lambda x: len(x[1]), reverse=True)
    return segments


# ============================================================================
#  LAS I/O
# ============================================================================

def load_las(las_path: Path) -> laspy.LasData:
    """Load and return a laspy.LasData object."""
    print(f"\n  Loading LAS: {las_path.name}")
    las = laspy.read(str(las_path))
    pts = np.vstack((las.x, las.y, las.z)).T
    print(f"  Total points: {len(pts):,}")
    return las


def save_segment_las(
        las_orig: laspy.LasData,
        indices: np.ndarray,
        output_path: Path,
):
    """Save a subset of points (identified by *indices*) to a new LAS file."""
    header = laspy.LasHeader(
        point_format=las_orig.header.point_format,
        version=las_orig.header.version,
    )

    pts_sub = np.vstack((
        las_orig.x[indices],
        las_orig.y[indices],
        las_orig.z[indices],
    )).T

    header.offsets = pts_sub.min(axis=0)
    header.scales  = np.array([0.001, 0.001, 0.001])

    new_las   = laspy.LasData(header=header)
    new_las.x = pts_sub[:, 0]
    new_las.y = pts_sub[:, 1]
    new_las.z = pts_sub[:, 2]

    # Preserve all extra dimensions (classification, color, intensity, etc.)
    for dim_name in las_orig.point_format.dimension_names:
        if dim_name in ("X", "Y", "Z"):
            continue
        try:
            src_data = getattr(las_orig, dim_name)
            if hasattr(src_data, "__len__") and len(src_data) == len(las_orig.x):
                setattr(new_las, dim_name, src_data[indices])
        except Exception:
            pass  # silently skip dimensions that can't be copied

    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_las.write(str(output_path))


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Segment a facade LAS point cloud using vertical planes "
            "derived from WallSurface edges in a CityGML file."
        )
    )
    parser.add_argument(
        "--require-shared", action="store_true", default=True,
        help=(
            "Only use edges shared by ≥2 wall polygons as cutting planes "
            "(junction edges between adjacent walls). "
            "Default: use ALL vertical edges from every wall polygon."
        ),
    )
    parser.add_argument(
        "--min-pts", type=int, default=MIN_SEGMENT_PTS,
        help=f"Minimum points in a segment to save it (default: {MIN_SEGMENT_PTS}).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  13 · Tessellate Facade Cloud by Wall Edges")
    print("=" * 60)

    # ── 1. File selection ────────────────────────────────────────────
    print("\nStep 1: Select GML file")
    gml_path = select_file(GML_DIR, "*.gml")
    print(f"  GML  : {gml_path.name}")

    print("\nStep 2: Select LAS file")
    las_path = select_file(LAS_DIR, "*.las")
    print(f"  LAS  : {las_path.name}")

    # ── 2. Extract WallSurfaces from GML ────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STEP 3 — Extract WallSurfaces")
    print(f"{'='*60}")
    polygons = extract_wall_surfaces(gml_path)
    if not polygons:
        print("  ERROR: No WallSurface polygons found. Aborting.")
        sys.exit(1)

    # ── 3. Extract vertical edges ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STEP 4 — Extract Vertical Edges")
    print(f"  (require_shared = {args.require_shared})")
    print(f"{'='*60}")
    edges = extract_vertical_wall_edges(polygons, require_shared=args.require_shared)
    n_edges = len(edges)
    print(f"  Unique vertical edges: {n_edges}")

    if n_edges == 0:
        print("  WARNING: No vertical edges found. Try --require-shared=False.")
        print("  Saving entire cloud as a single segment.")
        edges = []

    for i, (p0, p1) in enumerate(edges):
        dz  = abs(p1[2] - p0[2])
        dxy = np.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)
        xy_mid = ((p0 + p1) / 2.0)[:2]
        print(f"    Edge {i+1:3d}: "
              f"XY_mid=({xy_mid[0]:.2f},{xy_mid[1]:.2f})  "
              f"dZ={dz:.2f} m  dXY={dxy:.3f} m")

    # ── 4. Build vertical planes ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STEP 5 — Build Cutting Planes")
    print(f"{'='*60}")
    planes = build_planes(edges)
    print(f"  Planes created: {len(planes)}")
    for i, pl in enumerate(planes):
        print(f"    Plane {i+1:3d}: normal=({pl.normal[0]:+.4f}, {pl.normal[1]:+.4f})  "
              f"origin=({pl.origin[0]:.2f}, {pl.origin[1]:.2f})")

    # ── 5. Load point cloud ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STEP 5b — Load Point Cloud")
    print(f"{'='*60}")
    las = load_las(las_path)
    pts = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)

    # ── 6. Segment ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STEP 6 — Segment Point Cloud")
    print(f"{'='*60}")
    segments = segment_point_cloud(pts, planes)

    total_saved = 0
    skipped     = 0
    expected_segs = 2 ** max(len(planes), 0)
    print(f"  Points: {len(pts):,}")
    print(f"  Planes: {len(planes)}")
    print(f"  Maximum possible segments: {expected_segs}")
    print(f"  Occupied segments: {len(segments)}")

    # ── 7. Save segments ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STEP 7 — Save Segments")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}")

    stem = las_path.stem
    saved_files = []

    for seg_id, indices in segments:
        n_pts = len(indices)
        if n_pts < args.min_pts:
            skipped += 1
            continue

        out_name = f"{stem}_seg_{seg_id:04d}_{n_pts}pts.las"
        out_path = OUTPUT_DIR / out_name

        save_segment_las(las, indices, out_path)
        size_kb = out_path.stat().st_size / 1024
        print(f"  ✓ Segment {seg_id:4d}: {n_pts:8,} pts → {out_name}  ({size_kb:.0f} KB)")
        total_saved += 1
        saved_files.append(out_path)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  GML file         : {gml_path.name}")
    print(f"  LAS file         : {las_path.name}")
    print(f"  Wall polygons    : {len(polygons)}")
    print(f"  Cutting planes   : {len(planes)}")
    print(f"  Segments saved   : {total_saved}")
    print(f"  Segments skipped : {skipped}  (< {args.min_pts} pts)")
    print(f"  Output directory : {OUTPUT_DIR}")
    print(f"{'='*60}")
    print("  Done!\n")


if __name__ == "__main__":
    main()
