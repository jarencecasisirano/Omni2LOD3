#!/usr/bin/env python3
"""
03_generate_facade_points.py

Generate synthetic facade points ON FOOTPRINT EDGES ONLY
and append to original LAS, preserving CRS + header metadata.
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

# =========================
# FACADE PARAMS
# =========================

ROOF_PERCENTILE = 98.0
Z_STEP = 0.5
MIN_FACADE_HEIGHT = 2.0

EDGE_SAMPLE_DIST = 0.75   # meters between facade columns
XY_JITTER = 0.15          # horizontal noise (meters)
Z_JITTER = 0.10           # vertical noise (meters)
NEAR_EDGE_TOL = 1.0       # search radius for nearby LiDAR (m)

USE_BUILDING_CLASS_ONLY = True
BUILDING_CLASS = 6

# =========================
# HELPERS
# =========================

def estimate_roof_z(z_values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(z_values, percentile))

def main():
    print("=== 03_generate_facade_points.py ===")
    print(f"Input LAS: {LAS_INPUT}")
    print(f"Footprint SHP: {FOOTPRINT_SHP}")
    print(f"Output LAS: {LAS_OUTPUT}")

    # -------------------------
    # Load LAS
    # -------------------------
    print("-> Reading LAS...")
    las = laspy.read(LAS_INPUT)

    X = np.asarray(las.x)
    Y = np.asarray(las.y)
    Z = np.asarray(las.z)

    if hasattr(las, "classification"):
        cls = np.asarray(las.classification)
    else:
        cls = np.zeros(len(Z), dtype=np.uint8)

    print(f"Total points: {len(Z):,}")

    if USE_BUILDING_CLASS_ONLY:
        mask_building = cls == BUILDING_CLASS
        print(f"Building-class points: {mask_building.sum():,}")
    else:
        mask_building = np.ones_like(Z, dtype=bool)

    # -------------------------
    # Load Footprints
    # -------------------------
    print("-> Reading footprints...")
    gdf = gpd.read_file(FOOTPRINT_SHP)

    if gdf.crs is None or gdf.crs.to_epsg() != TARGET_EPSG:
        print(f"-> Reprojecting footprints to EPSG:{TARGET_EPSG}")
        gdf = gdf.to_crs(epsg=TARGET_EPSG)

    footprints = gdf.geometry.values
    print(f"Number of footprints: {len(footprints)}")

    synthetic_xyz = []

    # -------------------------
    # Process footprints (EDGE ONLY)
    # -------------------------
    print("-> Generating facade points (edge-only)...")

    for i, poly in enumerate(footprints):
        if poly is None or poly.is_empty:
            continue

        boundary = poly.exterior
        if boundary is None:
            continue

        length = boundary.length
        sample_dists = np.arange(0, length, EDGE_SAMPLE_DIST)

        for d in sample_dists:
            p_edge = boundary.interpolate(d)
            x0, y0 = p_edge.x, p_edge.y

            # Nearby LiDAR for base + roof
            dx = X - x0
            dy = Y - y0
            dist2 = dx * dx + dy * dy

            near_mask = (dist2 <= NEAR_EDGE_TOL**2) & mask_building
            near_idx = np.where(near_mask)[0]

            if len(near_idx) < 10:
                continue

            z_near = Z[near_idx]
            z_base = np.percentile(z_near, 5)
            z_roof = estimate_roof_z(z_near, ROOF_PERCENTILE)

            facade_height = z_roof - z_base
            if facade_height < MIN_FACADE_HEIGHT:
                continue

            z_vals = np.arange(z_base + Z_STEP, z_roof, Z_STEP)

            for z in z_vals:
                xn = x0 + np.random.normal(0, XY_JITTER)
                yn = y0 + np.random.normal(0, XY_JITTER)
                zn = z  + np.random.normal(0, Z_JITTER)

                synthetic_xyz.append((xn, yn, zn))

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1} / {len(footprints)} footprints")

    synthetic_xyz = np.array(synthetic_xyz)

    print(f"Synthetic facade points generated: {len(synthetic_xyz):,}")

    if len(synthetic_xyz) == 0:
        print("WARNING: No synthetic points generated.")
        return

    # -------------------------
    # Merge + write LAS (CRS SAFE — like 02_clip_z.py)
    # -------------------------
    print("-> Merging original + synthetic points...")

    new_las = laspy.create(
        point_format=las.header.point_format,
        file_version=las.header.version
    )

    # CRITICAL: preserve full header (CRS, VLRs, scales, offsets)
    new_las.header = las.header

    # Merge coords
    X_all = np.concatenate([X, synthetic_xyz[:, 0]])
    Y_all = np.concatenate([Y, synthetic_xyz[:, 1]])
    Z_all = np.concatenate([Z, synthetic_xyz[:, 2]])

    new_las.x = X_all
    new_las.y = Y_all
    new_las.z = Z_all

    # Merge classification
    cls_synth = np.full(len(synthetic_xyz), BUILDING_CLASS, dtype=cls.dtype)
    new_las.classification = np.concatenate([cls, cls_synth])

    # Copy all other dimensions safely (like your clip script)
    for dim in las.point_format.dimension_names:
        if dim in ["X", "Y", "Z", "classification"]:
            continue
        arr = getattr(las, dim)
        pad = np.zeros(len(synthetic_xyz), dtype=arr.dtype)
        setattr(new_las, dim, np.concatenate([arr, pad]))

    print(f"-> Writing output LAS: {LAS_OUTPUT}")
    new_las.write(LAS_OUTPUT)

    print("=== DONE ===")

if __name__ == "__main__":
    main()
