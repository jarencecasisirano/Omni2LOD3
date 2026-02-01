#!/usr/bin/env python3
import sys
import os
import numpy as np
import laspy
import time

start_time = time.time()

# ------------------------------------------------------------------
# 1. Arguments (supplied by main.py)
# ------------------------------------------------------------------
if len(sys.argv) < 3:
    print("Usage: python 02_clip_z.py <input_las> <output_las> [z_min] [z_max]")
    sys.exit(1)

INPUT_LAS  = sys.argv[1]
OUTPUT_LAS = sys.argv[2]

# Optional Z bounds
Z_MIN = float(sys.argv[3]) if len(sys.argv) >= 4 else -50
Z_MAX = float(sys.argv[4]) if len(sys.argv) >= 5 else 200

# ------------------------------------------------------------------
# 2. Ensure output folder exists
# ------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)

# ------------------------------------------------------------------
# 3. Load and clip LAS
# ------------------------------------------------------------------
print(f"-> Reading: {INPUT_LAS}")
las = laspy.read(INPUT_LAS)

z = las.z
mask = (z >= Z_MIN) & (z <= Z_MAX)

print(f"Z min threshold: {Z_MIN}")
print(f"Z max threshold: {Z_MAX}")
print(f"Original points: {len(z):,}")
print(f"Kept points: {np.sum(mask):,}")
print(f"Removed points: {len(z) - np.sum(mask):,}")

# ------------------------------------------------------------------
# 4. Create clipped LAS (preserve format and header)
# ------------------------------------------------------------------
clipped_las = laspy.create(
    point_format=las.header.point_format,
    file_version=las.header.version
)

clipped_las.header = las.header

for dim in las.point_format.dimension_names:
    setattr(
        clipped_las,
        dim,
        getattr(las, dim)[mask]
    )

clipped_las.write(OUTPUT_LAS)
print(f"-> Saved: {OUTPUT_LAS}")

end_time = time.time()
print(f"=== Done! Z clipping finished in {end_time - start_time:.2f} seconds ===")