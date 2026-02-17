# inspect_las.py
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io_helpers import choose_file, list_las_files
from utils.paths import (
    DATA_PC_DIR,
    OUT_DOWNSAMPLED,
    OUT_INFO,
    OUT_RECLASSIFIED,
    SCRIPT_VISUALIZE,
)


def choose_folder():
    """Let user select which folder category to inspect."""
    print("\nSelect folder:")
    print("\t[0] Raw Data")
    print("\t[1] Downsampled")
    print("\t[2] Reclassified")

    choice = input("Enter choice: ").strip()

    if choice == "0":
        return DATA_PC_DIR
    if choice == "1":
        return OUT_DOWNSAMPLED
    if choice == "2":
        return OUT_RECLASSIFIED

    print("[ERROR] Invalid selection.")
    return None

def inspect_las(las_path):
    print(f"Inspecting: {las_path}")
    las = laspy.read(las_path)
    z = las.z

    report_lines = []
    report_lines.append(f"\tFile: {las_path}")
    report_lines.append(f"\tNumber of points: {len(las.points)}")

    try:
        crs = las.header.parse_crs()
    except Exception:
        crs = None
    if crs is None:
        crs_text = "None"
    else:
        epsg = crs.to_epsg()
        crs_text = f"EPSG:{epsg}" if epsg is not None else str(crs)
    report_lines.append(f"\tCRS: {crs_text}")

    report_lines.append(f"\tPoint format: {las.header.point_format.id}")
    report_lines.append(f"\tLAS version: {las.header.version}")
    report_lines.append(f"\tScales:  {las.header.scales}")
    report_lines.append(f"\tZ min: {z.min():.3f}")
    report_lines.append(f"\tZ max: {z.max():.3f}")

    if hasattr(las, "classification"):
        classes, counts = np.unique(las.classification, return_counts=True)
        report_lines.append("\tClassification counts:")
        for c, n in zip(classes, counts):
            report_lines.append(f"\t  Class {int(c)}: {int(n)}")
    else:
        report_lines.append("No classification field")

    xmin, ymin, _ = las.x.min(), las.y.min(), z.min()
    xmax, ymax, _ = las.x.max(), las.y.max(), z.max()
    area = (xmax - xmin) * (ymax - ymin)
    density = len(z) / area if area > 0 else 0

    report_lines.append(f"\tXY extent: {xmax - xmin:.2f} x {ymax - ymin:.2f} m")
    report_lines.append(f"\tApprox point density: {density:.2f} pts/m2")

    for line in report_lines:
        print(line)

    os.makedirs(OUT_INFO, exist_ok=True)
    base = os.path.splitext(os.path.basename(las_path))[0]
    out_txt = os.path.join(OUT_INFO, f"{base}_info.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for line in report_lines:
            f.write(line + "\n")

    print(f"\t-> Saved LAS report to: {Path(out_txt).parent}")

def main():
    print("\n=== INSPECT LAS FILE ===")

    if len(sys.argv) >= 2 and os.path.exists(sys.argv[1]):
        picked = sys.argv[1]
    else:
        folder = choose_folder()
        if not folder:
            return
        files = list_las_files(folder)
        picked = choose_file(files, "Select LAS file to inspect:", indent_choices=True)
        if not picked:
            return

    inspect_las(picked)
    ans = input("\nWould you like to visualize this point cloud? [y/N]: ").strip().lower()
    if ans == "y":
        subprocess.run([sys.executable, SCRIPT_VISUALIZE, picked])


if __name__ == "__main__":
    main()
