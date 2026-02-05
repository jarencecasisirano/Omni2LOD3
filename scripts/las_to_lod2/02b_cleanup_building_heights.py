#!/usr/bin/env python3
"""
02b_cleanup_building_heights.py

FAST + SAFE local building-aware cleanup:
- Operates ONLY inside building footprints
- Forces building class inside footprints
- Robustly estimates roof height per footprint (buffered)
- Gently removes roof spike outliers (inside + small buffer)

Vectorized + scalable for millions of points.
LOD2 / LOD3 safe.
"""

import sys
import numpy as np
import geopandas as gpd
import laspy
from shapely.geometry import Point
from shapely.prepared import prep

# -------------------------
# PARAMS (SAFE)
# -------------------------

BUILDING_CLASS = 6

GROUND_PCTL = 5.0
ROOF_PCTL   = 99.0

ROOF_SAFETY_MARGIN = 0.2
MAX_ROOF_BUFFER   = 0.2

MIN_POINTS_IN_POLY = 50

FOOTPRINT_BUFFER = 2.0   # meters (roof spike cleanup zone)

# -------------------------
# CLI
# -------------------------

if len(sys.argv) < 4:
    print("Usage:")
    print("  python 02b_cleanup_building_heights.py <input_las> <footprint_shp> <output_las>")
    sys.exit(1)

LAS_INPUT     = sys.argv[1]
FOOTPRINT_SHP = sys.argv[2]
LAS_OUTPUT    = sys.argv[3]

print("\n" + "="*70)
print("FAST LOCAL BUILDING HEIGHT CLEANUP (INSIDE + BUFFER)")
print("="*70)
print(f"Input LAS: {LAS_INPUT}")
print(f"Footprint: {FOOTPRINT_SHP}")
print(f"Output:    {LAS_OUTPUT}")

# -------------------------
# Load LAS
# -------------------------

las = laspy.read(LAS_INPUT)

X = np.asarray(las.x)
Y = np.asarray(las.y)
Z = np.asarray(las.z)

if hasattr(las, "classification"):
    cls = np.asarray(las.classification)
else:
    cls = np.zeros(len(Z), dtype=np.uint8)

n_total = len(Z)
final_mask = np.ones(n_total, dtype=bool)

print(f"-> Total points: {n_total:,}")

# -------------------------
# Load footprints
# -------------------------

gdf = gpd.read_file(FOOTPRINT_SHP)
footprints = gdf.geometry.values

print(f"-> Footprints: {len(footprints)}")

removed_roof_spikes = 0
forced_building = 0
processed_points = 0

print("-> Processing footprints with buffer + prepared geometries")

# =========================
# PROCESS PER FOOTPRINT
# =========================

for poly_idx, poly in enumerate(footprints):
    if poly is None or poly.is_empty:
        continue

    # Clean geometry
    poly_real = poly.buffer(0)
    poly_buf  = poly_real.buffer(FOOTPRINT_BUFFER)

    prep_real = prep(poly_real)
    prep_buf  = prep(poly_buf)

    # ---- Fast bbox on BUFFERED footprint
    minx, miny, maxx, maxy = poly_buf.bounds

    bbox_mask = (
        (X >= minx) & (X <= maxx) &
        (Y >= miny) & (Y <= maxy)
    )

    idx_bbox = np.where(bbox_mask)[0]

    if len(idx_bbox) < MIN_POINTS_IN_POLY:
        continue

    inside_real = []
    inside_buf  = []

    # ---- Precise spatial test (robust)
    for i in idx_bbox:
        pt = Point(X[i], Y[i])

        if prep_real.covers(pt):
            inside_real.append(i)

        if prep_buf.covers(pt):
            inside_buf.append(i)

    if len(inside_buf) < MIN_POINTS_IN_POLY:
        continue

    inside_real = np.array(inside_real, dtype=int)
    inside_buf  = np.array(inside_buf, dtype=int)

    processed_points += len(inside_buf)

    # =========================
    # FORCE BUILDING CLASS (REAL footprint only)
    # =========================

    if len(inside_real) > 0:
        not_building = inside_real[cls[inside_real] != BUILDING_CLASS]
        if len(not_building) > 0:
            cls[not_building] = BUILDING_CLASS
            forced_building += len(not_building)

    # =========================
    # ROBUST ROOF ESTIMATION (BUFFERED)
    # =========================

    z_vals = Z[inside_buf]

    # Remove extreme junk before percentiles
    z_vals = z_vals[z_vals < np.percentile(z_vals, 99.5)]

    z_ground = np.percentile(z_vals, GROUND_PCTL)
    z_roof_raw = np.percentile(z_vals, ROOF_PCTL)

    z_roof = z_roof_raw - ROOF_SAFETY_MARGIN
    z_hard_cap = z_roof + MAX_ROOF_BUFFER

    # =========================
    # REMOVE ROOF SPIKES (BUFFERED)
    # =========================

    spike_mask = Z[inside_buf] > z_hard_cap
    n_spikes = spike_mask.sum()

    if n_spikes > 0:
        final_mask[inside_buf[spike_mask]] = False
        removed_roof_spikes += n_spikes

    if (poly_idx + 1) % 5 == 0:
        print(f"   Processed footprint {poly_idx+1}/{len(footprints)}")

# -------------------------
# Summary
# -------------------------

print("\n" + "-"*60)
print(f"Processed points (buffered):    {processed_points:,}")
print(f"Forced building class:          {forced_building:,}")
print(f"Removed roof spike points:      {removed_roof_spikes:,}")
print("-"*60)

# -------------------------
# Write output
# -------------------------

new_las = laspy.create(
    point_format=las.header.point_format,
    file_version=las.header.version
)

new_las.header = las.header

# --- Explicit core dimensions ---
new_las.x = X[final_mask]
new_las.y = Y[final_mask]
new_las.z = Z[final_mask]

# --- EXPLICIT classification (CRITICAL) ---
new_las.classification = cls[final_mask]

# --- Copy remaining dimensions safely ---
for dim in las.point_format.dimension_names:
    if dim in ["X", "Y", "Z", "classification"]:
        continue
    setattr(new_las, dim, getattr(las, dim)[final_mask])

new_las.write(LAS_OUTPUT)

print(f"\n-> Final points: {final_mask.sum():,}")
print(f"-> Saved: {LAS_OUTPUT}")
print("✓ Building height cleanup DONE")
print("="*70 + "\n")
