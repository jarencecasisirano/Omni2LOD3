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
OUTPUT_DIR = "outputs/10_facade_features"


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
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]
        except (ValueError, KeyboardInterrupt):
            pass
        print("  Invalid choice, try again.")


def main():
    # 1. Select file
    file_path = select_file(INPUT_DIR)
    filename = os.path.basename(file_path)
    print(f"\n  Loading: {filename}")

    # 2. Load
    las = laspy.read(file_path)
    n_total = len(las.x)
    classifications = np.array(las.classification, dtype=np.uint8)
    print(f"  Total points: {n_total:,}")

    # 3. Show current label breakdown
    print(f"\n  Label breakdown:")
    unique, counts = np.unique(classifications, return_counts=True)
    for code, count in zip(unique, counts):
        name = LABEL_NAMES.get(code, f"Unknown({code})")
        pct = 100.0 * count / n_total
        marker = "  ← REMOVING" if code == LABEL_MAP["wall"] else ""
        print(f"    {name:12s}: {count:>10,} pts ({pct:5.1f}%){marker}")

    # 4. Remove wall points
    wall_code = LABEL_MAP["wall"]
    keep_mask = classifications != wall_code
    n_kept = int(keep_mask.sum())
    n_removed = n_total - n_kept
    print(f"\n  Removing {n_removed:,} Wall points...")
    print(f"  Keeping  {n_kept:,} feature points")

    if n_kept == 0:
        print("  WARNING: No points remain after removing walls!")
        return

    # 5. Build output LAS
    header = laspy.LasHeader(point_format=2, version="1.2")
    header.scales = las.header.scales
    header.offsets = las.header.offsets

    new_las = laspy.LasData(header)
    new_las.x = np.array(las.x)[keep_mask]
    new_las.y = np.array(las.y)[keep_mask]
    new_las.z = np.array(las.z)[keep_mask]
    new_las.red = np.array(las.red)[keep_mask]
    new_las.green = np.array(las.green)[keep_mask]
    new_las.blue = np.array(las.blue)[keep_mask]
    new_las.classification = classifications[keep_mask]

    # 6. Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_name = filename.replace("labelled_", "features_")
    if out_name == filename:
        out_name = f"features_{filename}"
    output_path = os.path.join(OUTPUT_DIR, out_name)

    new_las.write(output_path)

    print(f"\n  Saved: {output_path}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  RESULT")
    print(f"{'='*60}")
    unique, counts = np.unique(classifications[keep_mask], return_counts=True)
    for code, count in zip(unique, counts):
        name = LABEL_NAMES.get(code, f"Unknown({code})")
        pct = 100.0 * count / n_kept
        print(f"    {name:12s}: {count:>10,} pts ({pct:5.1f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
