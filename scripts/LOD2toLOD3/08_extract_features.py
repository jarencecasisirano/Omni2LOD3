#!/usr/bin/env python3
"""
Extract Facade Features (remove Wall points).

Loads a labelled point cloud from outputs/09_labelled, removes all points
classified as Wall (code 2), and saves the remaining facade features
(doors, windows, etc.) to outputs/10_facade_features.

Usage:
    conda activate lidar-test
    python scripts/LOD2toLOD3/08_extract_features.py
"""

import os
import sys
import glob

import numpy as np
import laspy


# Classification codes (must match 07_label_clusters.py)
LABEL_MAP = {
    "other":   1,
    "wall":    2,
    "door":    3,
    "window":  4,
    "roof":    5,
    "ground":  6,
}
LABEL_NAMES = {v: k.capitalize() for k, v in LABEL_MAP.items()}

INPUT_DIR = "outputs/09_labelled"
OUTPUT_DIR = "outputs/11_facade_features_classified"


def select_file(directory):
    """List LAS files in directory and let user pick one."""
    pattern = os.path.join(directory, "*.las")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No .las files found in {directory}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Labelled point clouds in: {directory}")
    print(f"{'='*60}")
    for i, f in enumerate(files):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  [{i+1}] {os.path.basename(f):40s}  ({size_mb:.1f} MB)")
    print()

    while True:
        try:
            choice = input(f"Select file [1-{len(files)}]: ").strip()
            if not choice: continue
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]
        except (ValueError, KeyboardInterrupt):
            pass
        print("  Invalid choice, try again.")


def main():
    # 1. Select file
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    else:
        file_path = select_file(INPUT_DIR)

    filename = os.path.basename(file_path)
    base_name = os.path.splitext(filename)[0]
    print(f"\n  Loading: {filename}")

    # 2. Load
    las = laspy.read(file_path)
    n_total = len(las.x)
    classifications = np.array(las.classification, dtype=np.uint8)
    print(f"  Total points: {n_total:,}")

    # 3. Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4. Process each unique class
    print(f"\n  Extracting classes to: {OUTPUT_DIR}")
    unique_codes, counts = np.unique(classifications, return_counts=True)
    
    for code, count in zip(unique_codes, counts):
        class_name = LABEL_NAMES.get(code, f"Class_{code}")
        print(f"    - {class_name:12s}: {count:>10,} pts", end="", flush=True)

        mask = classifications == code
        if not np.any(mask):
            print(" (Skipping, no points)")
            continue

        # Build output LAS
        header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
        header.scales = las.header.scales
        header.offsets = las.header.offsets

        new_las = laspy.LasData(header)
        new_las.x = np.array(las.x)[mask]
        new_las.y = np.array(las.y)[mask]
        new_las.z = np.array(las.z)[mask]

        # Preserve colors if they exist
        if hasattr(las, 'red'):
            new_las.red = np.array(las.red)[mask]
            new_las.green = np.array(las.green)[mask]
            new_las.blue = np.array(las.blue)[mask]
        
        new_las.classification = classifications[mask]

        # Save
        clean_name = base_name.replace("labelled_", "")
        out_filename = f"features_{clean_name}_{class_name.lower()}.las"
        output_path = os.path.join(OUTPUT_DIR, out_filename)
        
        new_las.write(output_path)
        print(f" -> {out_filename}")

    print(f"\n{'='*60}")
    print(f"  DONE: Extracted {len(unique_codes)} classes.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
