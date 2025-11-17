#!/usr/bin/env python3
"""
normalize_localize_las.py

Loads a LAS/LAZ, recenters X/Y around the mean, normalizes Z so minimum Z -> 0,
and writes a new file to:
    D:\Projects\Thesis\outputs\normalized\<original_basename>_localized.las

Fixes earlier issue with header.copy() by using the existing header object.
"""

import laspy
import numpy as np
import os
import sys

def main():
    print("=== LAS Normalizer & Localizer ===")
    input_path = input("Enter the full path to your LAS/LAZ file: ").strip().strip('"')

    if not input_path:
        print("No input provided. Exiting.")
        return

    if not os.path.exists(input_path):
        print(f"❌ Error: File not found: {input_path}")
        return

    # fixed output directory requested by user
    output_dir = r"D:\Projects\Thesis\outputs\normalized"
    os.makedirs(output_dir, exist_ok=True)

    try:
        las = laspy.read(input_path)
    except Exception as e:
        print("❌ Failed to read LAS/LAZ file. Error:", e)
        return

    num_pts = len(las.points)
    print(f"✅ Loaded file with {num_pts:,} points.")

    # Convert to numpy arrays (these are float coordinate arrays)
    x = las.x
    y = las.y
    z = las.z

    # Compute shifts (center XY on mean, Z so min->0)
    mean_x, mean_y = float(np.mean(x)), float(np.mean(y))
    min_z = float(np.min(z))

    # Localize and normalize
    x_local = x - mean_x
    y_local = y - mean_y
    z_local = z - min_z

    # Prepare output header: use the same header object but adjust offsets/scales.
    # Note: modifying header will not break writing; this avoids using header.copy().
    header = las.header
    try:
        # set offsets to zero so coordinates are stored relative to origin
        header.offsets = (0.0, 0.0, 0.0)
    except Exception:
        # older/newer laspy versions might expect lists
        header.offsets = [0.0, 0.0, 0.0]

    # set reasonable scales for visualization precision (you can change if needed)
    try:
        header.scales = (0.01, 0.01, 0.01)
    except Exception:
        header.scales = [0.01, 0.01, 0.01]

    # Create a new LasData using this header
    try:
        las_out = laspy.LasData(header)
    except Exception as e:
        print("❌ Failed to create output LasData from header. Error:", e)
        return

    # Assign coordinates
    las_out.x = x_local
    las_out.y = y_local
    las_out.z = z_local

    # Copy all other point dimensions (intensity, classification, RGB, etc.)
    # We iterate through the input point format dimensions (safe across laspy versions)
    for dim in las.point_format.dimensions:
        name = dim.name
        if name in ("X", "Y", "Z"):
            continue
        try:
            las_out[name] = las[name]
        except Exception:
            # Some dimensions may be read-only or special — skip silently
            pass

    # Build output filename and write
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_localized.las")

    try:
        las_out.write(output_path)
    except Exception as e:
        print("❌ Failed to write output file. Error:", e)
        return

    # Print summary
    print("\n=== Normalization Complete ===")
    print(f"📂 Output saved to: {output_path}")
    print(f"X range: {float(x_local.min()):.3f} → {float(x_local.max()):.3f}")
    print(f"Y range: {float(y_local.min()):.3f} → {float(y_local.max()):.3f}")
    print(f"Z range: {float(z_local.min()):.3f} → {float(z_local.max()):.3f}")
    print(f"Shift applied: mean_x={mean_x:.3f}, mean_y={mean_y:.3f}, min_z={min_z:.3f}")
    print("\n✅ Your LAS is now centered near (0,0) and starts from Z=0.\n")

if __name__ == "__main__":
    main()
