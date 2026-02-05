#!/usr/bin/env python3
"""
03_generate_facade_points.py - IMPROVED VERSION (SAFE ROOF + EDGE ONLY + FALLBACK)

Generates dense, uniform facade points along building footprint edges
with adaptive density, conservative roof height detection, and
neighbor-height fallback for missing facade segments.
"""

import sys
import os
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import laspy

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

# Ground-to-building override (semantic fix for Pix4D)
ENABLE_GROUND_OVERRIDE = True
GROUND_TO_BUILDING_Z_OFFSET = 3.0  # meters

# =========================
# FACADE GENERATION PARAMS
# =========================

# Edge sampling
EDGE_SAMPLE_DIST = 0.3
MIN_EDGE_SAMPLE_DIST = 0.15

# Vertical sampling
Z_STEP = 0.25
MIN_FACADE_HEIGHT = 2.0

# Roof detection (CONSERVATIVE)
ROOF_PERCENTILE = 95.0
ROOF_SAFETY_MARGIN = 0.6   # ⬅ LOWER THIS to reduce overshoot (try 0.3 if needed)
GROUND_PERCENTILE = 5.0

# Search radius for nearby LiDAR
NEAR_EDGE_TOL = 1.5
BACKUP_SEARCH_RADIUS = 3.0

# Jitter (noise for realism)
XY_JITTER = 0.08
Z_JITTER = 0.05

# Classification
USE_BUILDING_CLASS_ONLY = False
BUILDING_CLASS = 6

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

def find_roof_and_ground(X, Y, Z, cls, x0, y0, search_radius,
                         building_class, use_building_only):

    dx = X - x0
    dy = Y - y0
    dist2 = dx * dx + dy * dy

    # Primary search
    if use_building_only:
        near_mask = (dist2 <= search_radius**2) & (cls == building_class)
    else:
        near_mask = dist2 <= search_radius**2

    near_idx = np.where(near_mask)[0]

    # Fallback search
    if len(near_idx) < 10:
        r = BACKUP_SEARCH_RADIUS
        if use_building_only:
            near_mask = (dist2 <= r**2) & (cls == building_class)
        else:
            near_mask = dist2 <= r**2
        near_idx = np.where(near_mask)[0]

    if len(near_idx) < 5:
        return None, None, 0

    z_near = Z[near_idx]

    # Base + roof
    z_ground = np.percentile(z_near, GROUND_PERCENTILE)

    z_roof_raw = estimate_roof_z(z_near, ROOF_PERCENTILE)

    # ✅ Conservative roof cap (STOP BELOW NOISY ROOF)
    z_roof = z_roof_raw - ROOF_SAFETY_MARGIN

    return z_ground, z_roof, len(near_idx)

# =========================
# MAIN
# =========================

def main():
    print("\n" + "="*70)
    print("IMPROVED FACADE POINT GENERATION (EDGE ONLY, SAFE ROOF + FALLBACK)")
    print("="*70)
    print(f"Input LAS:      {LAS_INPUT}")
    print(f"Footprint SHP:  {FOOTPRINT_SHP}")
    print(f"Output LAS:     {LAS_OUTPUT}")

    # -------------------------
    # Load LAS
    # -------------------------
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

    # -------------------------
    # Global ground estimate
    # -------------------------
    if ENABLE_GROUND_OVERRIDE:
        ground_mask = cls == 2
        if ground_mask.sum() > 50:
            global_ground_z = np.percentile(Z[ground_mask], 50)
            print(f"Estimated global ground Z (median): {global_ground_z:.2f}")
        else:
            global_ground_z = np.percentile(Z, 5)
            print(f"Fallback global ground Z (P5): {global_ground_z:.2f}")
    else:
        global_ground_z = None

    # -------------------------
    # Building mask + override
    # -------------------------
    if USE_BUILDING_CLASS_ONLY:
        mask_building = cls == BUILDING_CLASS

        if ENABLE_GROUND_OVERRIDE and global_ground_z is not None:
            ground_mask = cls == 2
            high_ground_mask = ground_mask & (Z > global_ground_z + GROUND_TO_BUILDING_Z_OFFSET)

            n_override = high_ground_mask.sum()
            if n_override > 0:
                print(f"Overriding {n_override:,} ground points → BUILDING (for facade logic)")

            mask_building = mask_building | high_ground_mask

        print(f"Building-class (incl overrides): {mask_building.sum():,}")
    else:
        mask_building = np.ones_like(Z, dtype=bool)

    # -------------------------
    # Load footprints
    # -------------------------
    print("\n-> Reading footprints...")
    gdf = gpd.read_file(FOOTPRINT_SHP)

    if gdf.crs is None or gdf.crs.to_epsg() != TARGET_EPSG:
        print(f"   Reprojecting to EPSG:{TARGET_EPSG}")
        gdf = gdf.to_crs(epsg=TARGET_EPSG)

    footprints = gdf.geometry.values
    print(f"   Footprints loaded: {len(footprints)}")

    # -------------------------
    # Generate facade points
    # -------------------------
    print("\n-> Generating facade points...")
    synthetic_xyz = []

    # ⬇️ Neighbor fallback memory
    last_good_z_ground = None
    last_good_z_roof   = None

    for fp_idx, poly in enumerate(footprints):
        if poly is None or poly.is_empty:
            continue

        boundary = poly.exterior
        if boundary is None:
            continue

        edge_points = sample_edge_adaptively(
            boundary, EDGE_SAMPLE_DIST, MIN_EDGE_SAMPLE_DIST
        )

        for edge_pt in edge_points:
            x0, y0 = edge_pt.x, edge_pt.y

            z_ground, z_roof, n_nearby = find_roof_and_ground(
                X, Y, Z, cls, x0, y0,
                NEAR_EDGE_TOL,
                BUILDING_CLASS,
                USE_BUILDING_CLASS_ONLY
            )

            # -------------------------
            # Neighbor-height fallback
            # -------------------------
            if z_ground is None or z_roof is None:
                if last_good_z_ground is not None:
                    z_ground = last_good_z_ground
                    z_roof   = last_good_z_roof
                else:
                    continue

            facade_height = z_roof - z_ground

            if facade_height < MIN_FACADE_HEIGHT:
                if last_good_z_ground is not None:
                    z_ground = last_good_z_ground
                    z_roof   = last_good_z_roof
                    facade_height = z_roof - z_ground
                else:
                    continue

            # Save good heights for neighbors
            last_good_z_ground = z_ground
            last_good_z_roof   = z_roof

            # Vertical column (STOP before roof)
            z_vals = np.arange(z_ground + Z_STEP, z_roof, Z_STEP)

            for z in z_vals:
                xn = x0 + np.random.normal(0, XY_JITTER)
                yn = y0 + np.random.normal(0, XY_JITTER)
                zn = z  + np.random.normal(0, Z_JITTER)

                synthetic_xyz.append((xn, yn, zn))

    synthetic_xyz = np.array(synthetic_xyz)

    print(f"\nSynthetic facade points generated: {len(synthetic_xyz):,}")

    if len(synthetic_xyz) == 0:
        print("\n[ERROR] No synthetic points generated!")
        print("Writing original LAS unchanged.")
        las.write(LAS_OUTPUT)
        return

    # -------------------------
    # Merge original + synthetic (PRESERVE CRS)
    # -------------------------
    print("\n-> Merging original + synthetic points...")

    new_las = laspy.create(
        point_format=las.header.point_format,
        file_version=las.header.version
    )

    # CRITICAL: preserve full header (CRS, VLRs, etc.)
    new_las.header = las.header

    X_all = np.concatenate([X, synthetic_xyz[:, 0]])
    Y_all = np.concatenate([Y, synthetic_xyz[:, 1]])
    Z_all = np.concatenate([Z, synthetic_xyz[:, 2]])

    new_las.x = X_all
    new_las.y = Y_all
    new_las.z = Z_all

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

    print(f"\n   Original points:  {len(X):,}")
    print(f"   Synthetic points: {len(synthetic_xyz):,}")
    print(f"   Total points:     {len(X_all):,}")
    print(f"   Synthetic ratio:  {100*len(synthetic_xyz)/len(X_all):.1f}%")

    print("\n" + "="*70)
    print("✓ DONE!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
