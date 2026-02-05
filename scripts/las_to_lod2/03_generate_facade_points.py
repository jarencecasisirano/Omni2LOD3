#!/usr/bin/env python3
"""
03_generate_facade_points.py - IMPROVED VERSION (SAFE ROOF + EDGE ONLY + FALLBACK + EDGE GAP FILL)

Base behavior (your working version):
- Two-pass per-polygon processing
  PASS A: estimate z_ground/z_roof per edge sample + record which samples were "supported"
  PASS B: generate facade columns; if a sample had no support, fill using robust median roof/ground
          from supported samples on the same polygon (or last-good fallback)

ADDED (what you asked, minimal change):
- PASS C (gap pass): after PASS B, detect remaining gaps along the edge samples (unsupported columns)
  and attempt to fill them using *nearest supported neighbor heights* (left/right along the edge list).
  If none exist, fall back to polygon median, then last-good.

Also:
- Progress bar using your utils.loading.create_bar (same import pattern as 01_downsampling.py)
- Conservative roof rules + sanity checks remain.

Why this fixes your “still missing edges”:
- Sometimes a section of edge samples all fail support (occlusion / sparse points / weird classification),
  so polygon median alone can still get skipped if sanity checks fail locally.
  The neighbor fill uses the closest valid height on the same polygon edge, which is usually what you want.
"""

import sys
import os
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep
import laspy

# ============================================================
# Path setup for utils (MATCH 01_downsampling.py)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.loading import create_bar  # <-- do not "try/except" this; you said it's working in your project

# =========================
# CLI ARGS
# =========================

if len(sys.argv) < 4:
    print("Usage:")
    print("  python 03_generate_facade_points.py <input_las> <footprint_shp> <output_las>")
    sys.exit(1)

LAS_INPUT = Path(sys.argv[1])
FOOTPRINT_SHP = Path(sys.argv[2])
LAS_OUTPUT = Path(sys.argv[3])
os.makedirs(LAS_OUTPUT.parent, exist_ok=True)

TARGET_EPSG = 32651

# =========================
# PARAMS
# =========================

EDGE_SAMPLE_DIST = 0.3
MIN_EDGE_SAMPLE_DIST = 0.15

Z_STEP = 0.25
MIN_FACADE_HEIGHT = 2.0

ROOF_PERCENTILE = 95.0
ROOF_SAFETY_MARGIN = 0.6
GROUND_PERCENTILE = 5.0

NEAR_EDGE_TOL = 1.5
BACKUP_SEARCH_RADIUS = 3.0

XY_JITTER = 0.08
Z_JITTER = 0.05

USE_BUILDING_CLASS_ONLY = False
BUILDING_CLASS = 6

# Safety / sanity
MAX_FACADE_HEIGHT = 80.0
MIN_LOCAL_SPAN = 1.0

# Edge-gap fill knobs
MIN_SUPPORT_POINTS = 8          # if fewer than this many nearby points, treat as "no support"
FILL_MIN_FACADE_HEIGHT = 2.0    # still require height span when filling
FILL_ROBUST_PCTL = 50.0         # median of per-sample ground/roof across the polygon

# PASS C knobs
GAP_MAX_RUN = 999999  # keep huge; we’ll try to fill any run
NEIGHBOR_SEARCH_LIMIT = 200  # max steps left/right to look for a supported neighbor

# =========================
# HELPERS
# =========================

def estimate_roof_z(z_values: np.ndarray, percentile: float) -> float:
    if len(z_values) == 0:
        return 0.0
    return float(np.percentile(z_values, percentile))

def sample_edge_adaptively(linestring, base_dist, min_dist):
    points = []
    length = linestring.length
    if length < min_dist:
        return [linestring.interpolate(0.5, normalized=True)]
    num_samples = max(2, int(np.ceil(length / base_dist)))
    for i in range(num_samples):
        frac = i / max(1, num_samples - 1)
        points.append(linestring.interpolate(frac, normalized=True))
    return points

def _filter_inside_prepped_polygon(prep_poly, X, Y, idx_candidates):
    if len(idx_candidates) == 0:
        return np.array([], dtype=int)
    inside = []
    for i in idx_candidates:
        if prep_poly.contains(Point(X[i], Y[i])):
            inside.append(i)
    return np.asarray(inside, dtype=int)

def find_roof_and_ground(X, Y, Z, cls, x0, y0, search_radius,
                         building_class, use_building_only,
                         prep_poly):
    """
    Returns: (z_ground, z_roof, n_support)
    n_support counts points used after radius+footprint(+optional class).
    """
    dx = X - x0
    dy = Y - y0
    dist2 = dx * dx + dy * dy

    # Primary radius
    rad_idx = np.where(dist2 <= (search_radius ** 2))[0]

    # Fallback radius
    if len(rad_idx) < 10:
        rad_idx = np.where(dist2 <= (BACKUP_SEARCH_RADIUS ** 2))[0]

    if len(rad_idx) < 5:
        return None, None, 0

    # Footprint-limited
    inside_idx = _filter_inside_prepped_polygon(prep_poly, X, Y, rad_idx)
    if len(inside_idx) < 5:
        return None, None, 0

    # Optional class filter
    if use_building_only:
        inside_idx = inside_idx[cls[inside_idx] == building_class]
        if len(inside_idx) < 5:
            return None, None, 0

    z_near = Z[inside_idx]

    local_min = float(z_near.min())
    local_max = float(z_near.max())
    if (local_max - local_min) < MIN_LOCAL_SPAN:
        return None, None, len(inside_idx)

    # Remove extreme spikes
    p99 = np.percentile(z_near, 99)
    z_near = z_near[z_near < p99]
    if len(z_near) < 5:
        return None, None, len(inside_idx)

    local_min = float(z_near.min())
    local_max = float(z_near.max())

    z_ground = float(np.percentile(z_near, GROUND_PERCENTILE))
    z_roof_raw = estimate_roof_z(z_near, ROOF_PERCENTILE)

    z_roof = float(z_roof_raw - ROOF_SAFETY_MARGIN)
    z_roof = min(z_roof, local_max - 0.2)

    if z_roof <= z_ground:
        return None, None, len(inside_idx)

    # Reject "ground jumped to roof"
    if (z_ground - local_min) > 0.75 * (local_max - local_min):
        return None, None, len(inside_idx)

    if (z_roof - z_ground) > MAX_FACADE_HEIGHT:
        return None, None, len(inside_idx)

    return z_ground, z_roof, len(inside_idx)

def _robust_polygon_fallback(z_ground_list, z_roof_list):
    """
    Robust per-polygon fallback from supported samples.
    """
    if len(z_ground_list) == 0 or len(z_roof_list) == 0:
        return None, None
    zg = float(np.percentile(np.asarray(z_ground_list, dtype=float), FILL_ROBUST_PCTL))
    zr = float(np.percentile(np.asarray(z_roof_list, dtype=float),   FILL_ROBUST_PCTL))
    if zr <= zg:
        return None, None
    if (zr - zg) > MAX_FACADE_HEIGHT:
        return None, None
    return zg, zr

def _valid_height_pair(zg, zr):
    if zg is None or zr is None:
        return False
    h = zr - zg
    if h < FILL_MIN_FACADE_HEIGHT:
        return False
    if h > MAX_FACADE_HEIGHT:
        return False
    return True

def _nearest_supported_neighbor(cols, idx, limit=NEIGHBOR_SEARCH_LIMIT):
    """
    Find nearest supported neighbor (left/right) in cols list.
    Returns (zg, zr) or (None, None).
    """
    n = len(cols)
    if n == 0:
        return None, None

    # expand outward
    for d in range(1, min(limit, n)):
        li = idx - d
        ri = idx + d
        if li >= 0 and cols[li]["supported"] and _valid_height_pair(cols[li]["z_ground"], cols[li]["z_roof"]):
            return cols[li]["z_ground"], cols[li]["z_roof"]
        if ri < n and cols[ri]["supported"] and _valid_height_pair(cols[ri]["z_ground"], cols[ri]["z_roof"]):
            return cols[ri]["z_ground"], cols[ri]["z_roof"]
    return None, None

def _emit_column_points(synthetic_xyz, x0, y0, z_ground, z_roof):
    z_vals = np.arange(z_ground + Z_STEP, z_roof, Z_STEP)
    for z in z_vals:
        synthetic_xyz.append((
            x0 + np.random.normal(0, XY_JITTER),
            y0 + np.random.normal(0, XY_JITTER),
            z  + np.random.normal(0, Z_JITTER)
        ))

# =========================
# MAIN
# =========================

def main():
    print("\n" + "="*70)
    print("IMPROVED FACADE POINT GENERATION (EDGE ONLY, SAFE ROOF + FALLBACK + EDGE GAP FILL)")
    print("="*70)
    print(f"Input LAS:      {LAS_INPUT}")
    print(f"Footprint SHP:  {FOOTPRINT_SHP}")
    print(f"Output LAS:     {LAS_OUTPUT}")

    # Load LAS
    print("\n-> Reading LAS...")
    las = laspy.read(LAS_INPUT)

    X = np.asarray(las.x)
    Y = np.asarray(las.y)
    Z = np.asarray(las.z)

    if hasattr(las, "classification"):
        cls = np.asarray(las.classification)
    else:
        cls = np.zeros(len(Z), dtype=np.uint8)

    print(f"   Total points: {len(Z):,}")

    # Load footprints
    print("\n-> Reading footprints...")
    gdf = gpd.read_file(FOOTPRINT_SHP)
    if gdf.crs is None or gdf.crs.to_epsg() != TARGET_EPSG:
        print(f"   Reprojecting to EPSG:{TARGET_EPSG}")
        gdf = gdf.to_crs(epsg=TARGET_EPSG)

    footprints = gdf.geometry.values
    print(f"   Footprints loaded: {len(footprints)}")

    # Pre-count edge samples for progress bar
    total_edges = 0
    poly_edge_cache = []  # store per-poly edge sample points so we don't re-sample twice

    for poly in footprints:
        if poly is None or poly.is_empty:
            poly_edge_cache.append([])
            continue

        boundaries = [poly.exterior]
        boundaries.extend(poly.interiors)

        edge_pts_all = []
        for boundary in boundaries:
            edge_pts_all.extend(sample_edge_adaptively(boundary, EDGE_SAMPLE_DIST, MIN_EDGE_SAMPLE_DIST))

        poly_edge_cache.append(edge_pts_all)
        total_edges += len(edge_pts_all)

    bar = create_bar("Generating facade edges", total_edges) if total_edges > 0 else None

    print("\n-> Generating facade points...")
    synthetic_xyz = []

    last_good_z_ground = None
    last_good_z_roof = None

    # =========================
    # Per-polygon processing
    # =========================
    for poly, edge_pts_all in zip(footprints, poly_edge_cache):
        if poly is None or poly.is_empty:
            continue
        if len(edge_pts_all) == 0:
            continue

        # buffered+prepared footprint for roof/ground estimation
        prep_poly = prep(poly.buffer(NEAR_EDGE_TOL))

        # -------------------------
        # PASS A: estimate columns
        # -------------------------
        cols = []
        good_grounds = []
        good_roofs = []

        for edge_pt in edge_pts_all:
            if bar is not None:
                bar.next()

            x0, y0 = edge_pt.x, edge_pt.y
            z_ground, z_roof, n_support = find_roof_and_ground(
                X, Y, Z, cls,
                x0, y0,
                NEAR_EDGE_TOL,
                BUILDING_CLASS,
                USE_BUILDING_CLASS_ONLY,
                prep_poly
            )

            supported = (z_ground is not None and z_roof is not None and n_support >= MIN_SUPPORT_POINTS)
            if supported:
                good_grounds.append(z_ground)
                good_roofs.append(z_roof)

            cols.append({
                "x0": x0, "y0": y0,
                "z_ground": z_ground, "z_roof": z_roof,
                "n_support": n_support,
                "supported": supported,
                "emitted": False,   # track if we actually generated points for this column
            })

        # Robust per-polygon fallback (median roof/ground across supported samples)
        poly_fallback_ground, poly_fallback_roof = _robust_polygon_fallback(good_grounds, good_roofs)

        # -------------------------
        # PASS B: generate columns (existing behavior)
        # -------------------------
        for c in cols:
            x0, y0 = c["x0"], c["y0"]
            z_ground, z_roof = c["z_ground"], c["z_roof"]

            if not c["supported"]:
                if poly_fallback_ground is not None and poly_fallback_roof is not None:
                    z_ground, z_roof = poly_fallback_ground, poly_fallback_roof
                elif last_good_z_ground is not None and last_good_z_roof is not None:
                    z_ground, z_roof = last_good_z_ground, last_good_z_roof
                else:
                    continue

            if not _valid_height_pair(z_ground, z_roof):
                continue

            last_good_z_ground = z_ground
            last_good_z_roof = z_roof

            _emit_column_points(synthetic_xyz, x0, y0, z_ground, z_roof)
            c["emitted"] = True

        # -------------------------
        # PASS C: edge gap pass (NEW)
        # If any columns still emitted==False, fill using nearest supported neighbor.
        # -------------------------
        any_gaps = any(not c["emitted"] for c in cols)
        if any_gaps:
            # First, mark which are "supported and valid"
            for i, c in enumerate(cols):
                if c["supported"] and _valid_height_pair(c["z_ground"], c["z_roof"]):
                    continue
                # treat unsupported or invalid as not supported for neighbor search
                c["supported"] = False

            for i, c in enumerate(cols):
                if c["emitted"]:
                    continue

                # nearest supported neighbor on this polygon edge list
                ng, nr = _nearest_supported_neighbor(cols, i)

                if ng is None or nr is None:
                    # fallback: polygon median
                    if poly_fallback_ground is not None and poly_fallback_roof is not None:
                        ng, nr = poly_fallback_ground, poly_fallback_roof
                    # fallback: last good
                    elif last_good_z_ground is not None and last_good_z_roof is not None:
                        ng, nr = last_good_z_ground, last_good_z_roof
                    else:
                        continue

                if not _valid_height_pair(ng, nr):
                    continue

                last_good_z_ground = ng
                last_good_z_roof = nr

                _emit_column_points(synthetic_xyz, c["x0"], c["y0"], ng, nr)
                c["emitted"] = True

    if bar is not None:
        bar.finish()

    synthetic_xyz = np.asarray(synthetic_xyz, dtype=float)

    print(f"\nSynthetic facade points generated: {len(synthetic_xyz):,}")

    if len(synthetic_xyz) == 0:
        print("\n[ERROR] No synthetic points generated!")
        print("Writing original LAS unchanged.")
        las.write(LAS_OUTPUT)
        return

    # Merge original + synthetic
    print("\n-> Merging original + synthetic points...")

    new_las = laspy.create(
        point_format=las.header.point_format,
        file_version=las.header.version
    )
    new_las.header = las.header  # preserve CRS/VLRs

    new_las.x = np.concatenate([X, synthetic_xyz[:, 0]])
    new_las.y = np.concatenate([Y, synthetic_xyz[:, 1]])
    new_las.z = np.concatenate([Z, synthetic_xyz[:, 2]])

    cls_synth = np.full(len(synthetic_xyz), BUILDING_CLASS, dtype=cls.dtype)
    new_las.classification = np.concatenate([cls, cls_synth])

    for dim in las.point_format.dimension_names:
        if dim in ["X", "Y", "Z", "classification"]:
            continue
        arr = getattr(las, dim)
        pad = np.zeros(len(synthetic_xyz), dtype=arr.dtype)
        setattr(new_las, dim, np.concatenate([arr, pad]))

    print(f"\n-> Writing output: {LAS_OUTPUT}")
    new_las.write(LAS_OUTPUT)

    print("\n" + "="*70)
    print("✓ DONE!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
