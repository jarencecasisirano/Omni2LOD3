#!/usr/bin/env python3
"""
merge_aligned_p2p.py — Merge aligned point clouds per subfolder

For every subfolder in outputs/06_aligned_p2p/:
  - Loads all .las files within it
  - Concatenates their points (and colours if present)
  - Saves the merged cloud to outputs/07_merged_las/<subfolder>.las

Usage
-----
    conda activate las-env
    python scripts/LOD2toLOD3/merge_aligned_p2p.py
"""

import sys
import numpy as np
import laspy
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parents[2]
INPUT_BASE  = BASE_DIR / "outputs" / "06_aligned_p2p"
OUTPUT_DIR  = BASE_DIR / "outputs" / "07_merged_las"


def merge_subfolder(subfolder: Path, output_dir: Path) -> bool:
    """
    Merge all .las files in `subfolder` into a single file.
    Returns True if successful, False if no files found.
    """
    las_files = sorted(subfolder.glob("*.las"))
    if not las_files:
        print(f"  [SKIP] {subfolder.name} — no .las files found")
        return False

    print(f"\n  Subfolder: {subfolder.name}  ({len(las_files)} files)")

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    has_color = True
    ref_las   = None   # keep first file for header reference

    for las_path in las_files:
        las = laspy.read(str(las_path))
        pts = np.vstack((las.x, las.y, las.z)).T
        all_points.append(pts)
        print(f"    {las_path.name}: {len(pts):,} pts")

        if has_color and hasattr(las, "red"):
            clr = np.vstack((las.red, las.green, las.blue)).T / 65535.0
            all_colors.append(clr)
        else:
            has_color = False

        if ref_las is None:
            ref_las = las

    merged_pts = np.vstack(all_points)

    # Build output LAS --------------------------------------------------------
    header          = laspy.LasHeader(
        point_format=ref_las.header.point_format,
        version=ref_las.header.version,
    )
    header.offsets  = merged_pts.min(0)
    header.scales   = np.array([0.001, 0.001, 0.001])

    new_las   = laspy.LasData(header)
    new_las.x = merged_pts[:, 0]
    new_las.y = merged_pts[:, 1]
    new_las.z = merged_pts[:, 2]

    if has_color and all_colors:
        merged_clr      = np.vstack(all_colors)
        new_las.red     = (merged_clr[:, 0] * 65535).astype(np.uint16)
        new_las.green   = (merged_clr[:, 1] * 65535).astype(np.uint16)
        new_las.blue    = (merged_clr[:, 2] * 65535).astype(np.uint16)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{subfolder.name}.las"
    new_las.write(str(out_path))
    print(f"    ✓ Saved: {out_path}  ({len(merged_pts):,} total points)")
    return True


def main():
    print("=" * 60)
    print("  Merge Aligned P2P Point Clouds")
    print("=" * 60)
    print(f"\n  Input : {INPUT_BASE}")
    print(f"  Output: {OUTPUT_DIR}")

    if not INPUT_BASE.exists():
        sys.exit(f"\nERROR: Input directory not found:\n  {INPUT_BASE}")

    subfolders = sorted([d for d in INPUT_BASE.iterdir() if d.is_dir()])
    if not subfolders:
        sys.exit(f"\nERROR: No subfolders found in:\n  {INPUT_BASE}")

    print(f"\n  Found {len(subfolders)} subfolder(s): "
          f"{', '.join(s.name for s in subfolders)}")

    merged_count = sum(merge_subfolder(sf, OUTPUT_DIR) for sf in subfolders)

    print(f"\n{'=' * 60}")
    print(f"  Done — {merged_count}/{len(subfolders)} subfolder(s) merged.")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
