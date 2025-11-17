#!/usr/bin/env python3
"""
Clips extreme Z outliers from a LAS/LAZ file and writes the cleaned cloud
to scripts/outputs/clipped/<base_name>_clipped.las
Usage:
    python clip_outliers.py <input.las> [lower_percentile] [upper_percentile]
Defaults: lower = 0.5, upper = 99.5
"""
import os, sys, laspy, numpy as np

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python clip_outliers.py <input.las> [lower_pct] [upper_pct]")
    sys.exit(1)

INPUT_LAS  = sys.argv[1]
LOWER_PCT  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
UPPER_PCT  = float(sys.argv[3]) if len(sys.argv) > 3 else 99.5

# ------------------------------------------------------------------
# Build output path  (mirrors your “scripts/outputs/normalized” logic)
# ------------------------------------------------------------------
BASE = os.path.basename(INPUT_LAS).replace(".las", "").replace(".laz", "")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "clipped")
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_LAS = os.path.join(OUT_DIR, f"{BASE}_clipped.las")

# ------------------------------------------------------------------
# Work
# ------------------------------------------------------------------
print(f"[clip] reading  : {INPUT_LAS}")
las = laspy.read(INPUT_LAS)

z_low, z_high = np.percentile(las.z, [LOWER_PCT, UPPER_PCT])
keep = (las.z >= z_low) & (las.z <= z_high)
clipped = las[keep]

print(f"[clip] kept       : {len(clipped)} pts  (Z {z_low:.2f} – {z_high:.2f} m)")
print(f"[clip] writing    : {OUTPUT_LAS}")
clipped.write(OUTPUT_LAS)

print("[clip] done.")