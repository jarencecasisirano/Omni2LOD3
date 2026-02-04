# 02_clip_z.py
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

# Optional Z bounds (fallback)
Z_MIN = float(sys.argv[3]) if len(sys.argv) >= 4 else None
Z_MAX = float(sys.argv[4]) if len(sys.argv) >= 5 else None

# ------------------------------------------------------------------
# 2. Ensure output folder exists
# ------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)

# ------------------------------------------------------------------
# 3. Load LAS
# ------------------------------------------------------------------
print(f"-> Reading: {INPUT_LAS}")
las = laspy.read(INPUT_LAS)

z = las.z
n_total = len(z)

# ------------------------------------------------------------------
# 4. Automatic robust Z bounds (percentile-based)
# ------------------------------------------------------------------
z_p1  = np.percentile(z, 1)
z_p99 = np.percentile(z, 99)

auto_z_min = z_p1 - 5.0     # buffer
auto_z_max = z_p99 + 5.0

zmin = Z_MIN if Z_MIN is not None else auto_z_min
zmax = Z_MAX if Z_MAX is not None else auto_z_max

print(f"Auto Z min (p1):  {z_p1:.2f}")
print(f"Auto Z max (p99): {z_p99:.2f}")
print(f"Using Z min: {zmin:.2f}")
print(f"Using Z max: {zmax:.2f}")

# ------------------------------------------------------------------
# 5. Initial Z clipping
# ------------------------------------------------------------------
mask_z = (z >= zmin) & (z <= zmax)

# ------------------------------------------------------------------
# 6. Class-aware filtering (if available)
#    Prefer keeping ground + building
# ------------------------------------------------------------------
if hasattr(las, "classification"):
    cls = las.classification
    # Common ASPRS:
    # 2 = Ground, 6 = Building, 1 = Unclassified, 5 = High Vegetation
    keep_classes = np.isin(cls, [1, 2, 6])
    mask_class = keep_classes
else:
    mask_class = np.ones(n_total, dtype=bool)

# ------------------------------------------------------------------
# 7. Statistical Z outlier removal (MAD-based)
# ------------------------------------------------------------------
z_filt = z[mask_z & mask_class]

if len(z_filt) > 0:
    z_med = np.median(z_filt)
    mad = np.median(np.abs(z_filt - z_med)) + 1e-6
    z_score = np.abs((z - z_med) / mad)
    mask_stat = z_score < 6.0   # robust threshold
else:
    mask_stat = np.ones(n_total, dtype=bool)

# ------------------------------------------------------------------
# 8. Final mask
# ------------------------------------------------------------------
final_mask = mask_z & mask_class & mask_stat

print(f"Original points: {n_total:,}")
print(f"Kept points:     {np.sum(final_mask):,}")
print(f"Removed points:  {n_total - np.sum(final_mask):,}")

# ------------------------------------------------------------------
# 9. Write clipped LAS
# ------------------------------------------------------------------
clipped_las = laspy.create(
    point_format=las.header.point_format,
    file_version=las.header.version
)

clipped_las.header = las.header

for dim in las.point_format.dimension_names:
    setattr(clipped_las, dim, getattr(las, dim)[final_mask])

clipped_las.write(OUTPUT_LAS)
print(f"-> Saved: {OUTPUT_LAS}")

end_time = time.time()
print(f"=== Done! Smart Z clipping finished in {end_time - start_time:.2f} seconds ===")
