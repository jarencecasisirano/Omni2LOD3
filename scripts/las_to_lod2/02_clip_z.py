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
# 3. Load LAS
# ------------------------------------------------------------------
print(f"-> Loading LAS: {INPUT_LAS}")
las = laspy.read(INPUT_LAS)

print(f"Input LAS Version: {las.header.version}")
print(f"Input LAS Point Format: {las.header.point_format.id}")

try:
    print(f"Input CRS: {las.header.parse_crs()}")
except Exception:
    print("Input CRS: None / could not parse")

z = las.z
mask = (z >= Z_MIN) & (z <= Z_MAX)

print(f"Z min threshold: {Z_MIN}")
print(f"Z max threshold: {Z_MAX}")
print(f"Original points: {len(z):,}")
print(f"Kept points:     {np.sum(mask):,}")
print(f"Removed points:  {len(z) - np.sum(mask):,}")

# ------------------------------------------------------------------
# 4. Create new LAS header (laspy 2.x safe)
# ------------------------------------------------------------------
print("-> Creating clipped LAS header...")

header = laspy.LasHeader(
    point_format=las.header.point_format,
    version=las.header.version
)

# Copy scales + offsets
header.scales = las.header.scales
header.offsets = las.header.offsets

# Copy CRS safely
try:
    header.parse_crs(las.header.parse_crs())
except Exception:
    pass

# ------------------------------------------------------------------
# 5. Create output LAS + copy all dimensions
# ------------------------------------------------------------------
print("-> Writing clipped LAS...")

clipped_las = laspy.LasData(header)

for dim in las.point_format.dimension_names:
    data = getattr(las, dim)
    setattr(clipped_las, dim, data[mask])

# Update bounds
clipped_las.header.mins = np.array([
    clipped_las.x.min(),
    clipped_las.y.min(),
    clipped_las.z.min()
])

clipped_las.header.maxs = np.array([
    clipped_las.x.max(),
    clipped_las.y.max(),
    clipped_las.z.max()
])

clipped_las.write(OUTPUT_LAS)

print(f"-> Clipped LAS saved to: {OUTPUT_LAS}")

end_time = time.time()
print(f"=== Done! Z clipping finished in {end_time - start_time:.2f} seconds ===")
