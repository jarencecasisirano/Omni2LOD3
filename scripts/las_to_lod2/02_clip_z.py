# 02_clip_z.py
"""
Z clipping optimized for NADIR DRONE imagery (ROOF-PRESERVING).

This script ONLY:
  1. Removes extreme global Z fliers (below ground / far above roof)
  2. Removes obvious vegetation (if classified)
  3. Removes ONLY extreme statistical fliers (very gentle MAD)

It DOES NOT:
  - Use isolation filters
  - Use IQR fences
  - Use local Z-diff pruning
  - Thin roof geometry

This preserves:
  - Roof ridges
  - Parapets
  - Roof edges
  - Small roof structures

Safe for LOD2 / LOD3 modeling.
"""

import sys
import os
import numpy as np
import laspy
import time

start_time = time.time()

# ------------------------------------------------------------------
# 1. Arguments
# ------------------------------------------------------------------
if len(sys.argv) < 3:
    print("Usage: python 02_clip_z.py <input_las> <output_las> [z_min] [z_max]")
    sys.exit(1)

INPUT_LAS  = sys.argv[1]
OUTPUT_LAS = sys.argv[2]

# Optional Z bounds (manual override)
Z_MIN = float(sys.argv[3]) if len(sys.argv) >= 4 else None
Z_MAX = float(sys.argv[4]) if len(sys.argv) >= 5 else None

# ------------------------------------------------------------------
# 2. DRONE-SPECIFIC CONFIG (SAFE)
# ------------------------------------------------------------------
# Conservative global percentiles
PERCENTILE_LOW  = 2.0
PERCENTILE_HIGH = 99.5

# Gentle MAD outlier threshold (EXTREME fliers only)
MAD_THRESHOLD = 10.0

# Classification filtering (optional)
REMOVE_VEG_CLASSES = False
VEG_CLASSES = [3, 4, 5]  # Low, Medium, High Vegetation

# ------------------------------------------------------------------
# 3. Ensure output folder exists
# ------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)

# ------------------------------------------------------------------
# 4. Load LAS
# ------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"DRONE NADIR - SAFE Z CLIPPING (ROOF PRESERVING)")
print(f"{'='*60}")
print(f"-> Reading: {INPUT_LAS}")

las = laspy.read(INPUT_LAS)

x = np.asarray(las.x)
y = np.asarray(las.y)
z = np.asarray(las.z)
n_total = len(z)

print(f"-> Total points: {n_total:,}")
print(f"-> Z range: [{z.min():.2f}, {z.max():.2f}]")

# ------------------------------------------------------------------
# 5. Classification filtering (remove vegetation ONLY)
# ------------------------------------------------------------------
mask_class = np.ones(n_total, dtype=bool)

if REMOVE_VEG_CLASSES and hasattr(las, "classification"):
    cls = np.asarray(las.classification)
    unique_classes = np.unique(cls)

    print(f"\n-> Classes present: {unique_classes}")

    for c in unique_classes:
        count = np.sum(cls == c)
        class_name = {
            1: "Unclassified",
            2: "Ground",
            3: "Low Veg",
            4: "Medium Veg",
            5: "High Veg",
            6: "Building"
        }.get(c, f"Class {c}")
        print(f"   {class_name}: {count:,} points")

    mask_veg = np.isin(cls, VEG_CLASSES)
    mask_class = ~mask_veg

    removed_veg = mask_veg.sum()
    if removed_veg > 0:
        print(f"-> Removed {removed_veg:,} vegetation points")
else:
    print("-> No vegetation filtering")

# ------------------------------------------------------------------
# 6. Conservative global Z bounds (percentiles)
# ------------------------------------------------------------------
z_valid = z[mask_class]

if len(z_valid) > 0:
    z_plow  = np.percentile(z_valid, PERCENTILE_LOW)
    z_phigh = np.percentile(z_valid, PERCENTILE_HIGH)

    # Tight but safe buffers
    auto_z_min = z_plow  - 2.0
    auto_z_max = z_phigh + 1.5

    zmin = Z_MIN if Z_MIN is not None else auto_z_min
    zmax = Z_MAX if Z_MAX is not None else auto_z_max

    print(f"\n-> Global Z bounds (conservative):")
    print(f"   P{PERCENTILE_LOW}:  {z_plow:.2f} → min: {zmin:.2f}")
    print(f"   P{PERCENTILE_HIGH}: {z_phigh:.2f} → max: {zmax:.2f}")
else:
    zmin = z.min()
    zmax = z.max()

mask_z = (z >= zmin) & (z <= zmax)

removed_bounds = n_total - (mask_class & mask_z).sum()
print(f"-> Removed {removed_bounds:,} extreme Z fliers")

# ------------------------------------------------------------------
# 7. VERY GENTLE MAD outlier removal (EXTREME ONLY)
# ------------------------------------------------------------------
print(f"\n-> Gentle MAD outlier detection (extreme fliers only)...")

current_mask = mask_class & mask_z
z_current = z[current_mask]

if len(z_current) > 100:
    z_median = np.median(z_current)
    mad = np.median(np.abs(z_current - z_median))

    if mad > 1e-6:
        z_score_mad = np.abs((z - z_median) / mad)
        mask_statistical = z_score_mad < MAD_THRESHOLD
    else:
        mask_statistical = np.ones(n_total, dtype=bool)

    removed_stat = current_mask.sum() - (current_mask & mask_statistical).sum()

    print(f"   Median Z = {z_median:.2f}")
    print(f"   MAD      = {mad:.4f}")
    print(f"   Threshold= {MAD_THRESHOLD}")
    print(f"   Removed {removed_stat:,} extreme statistical fliers")
else:
    mask_statistical = np.ones(n_total, dtype=bool)
    removed_stat = 0

# ------------------------------------------------------------------
# 8. Final mask (NO isolation, NO IQR)
# ------------------------------------------------------------------
final_mask = mask_class & mask_z & mask_statistical

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Original points:      {n_total:,}")
print(f"After class filter:   {mask_class.sum():,}")
print(f"After Z bounds:       {(mask_class & mask_z).sum():,}")
print(f"After MAD filter:     {final_mask.sum():,}")
print(f"{'='*60}")
print(f"TOTAL REMOVED:        {n_total - final_mask.sum():,} "
      f"({100*(n_total - final_mask.sum())/n_total:.2f}%)")
print(f"{'='*60}")

if final_mask.sum() == 0:
    print("\n[ERROR] All points filtered out!")
    sys.exit(1)

# ------------------------------------------------------------------
# 9. Write output (PRESERVE HEADER + CRS)
# ------------------------------------------------------------------
clipped_las = laspy.create(
    point_format=las.header.point_format,
    file_version=las.header.version
)

# IMPORTANT: preserve full header (CRS, scales, offsets, VLRs)
clipped_las.header = las.header

for dim in las.point_format.dimension_names:
    setattr(clipped_las, dim, getattr(las, dim)[final_mask])

clipped_las.write(OUTPUT_LAS)

z_final = clipped_las.z
print(f"\n-> Output Z range: [{z_final.min():.2f}, {z_final.max():.2f}]")
print(f"-> Saved: {OUTPUT_LAS}")

end_time = time.time()
print(f"\n{'='*60}")
print(f"✓ DONE! ({end_time - start_time:.2f} seconds)")
print(f"{'='*60}\n")
