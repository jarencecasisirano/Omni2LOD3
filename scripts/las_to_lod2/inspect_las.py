# inspect_las.py
import os
import sys
import glob
import laspy
import numpy as np
import subprocess
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

SCRIPT_VISUALIZE = os.path.join(PROJECT_ROOT, "scripts", "las_to_lod2", "visualize.py")
DATA_PC_DIR   = os.path.join(PROJECT_ROOT, "data", "01_point_cloud")

OUT_INFO = os.path.join(PROJECT_ROOT, "outputs", "00_las_info")
os.makedirs(OUT_INFO, exist_ok=True)

# Output folders
OUT_DOWNSAMPLED = os.path.join(PROJECT_ROOT, "outputs", "01_downsampled")
OUT_CLIPPED = os.path.join(PROJECT_ROOT, "outputs", "02_clipped")
OUT_COMPLETE = os.path.join(PROJECT_ROOT, "outputs", "03_complete_las")

def list_las_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.las")))
    return [f for f in files if not f.lower().endswith(".copc.las")]

def choose_file(files, prompt):
    if not files:
        print("[ERROR] No LAS files found.")
        return None
    print(f"\n{prompt}")
    for i, f in enumerate(files):
        print(f"[{i}] {os.path.basename(f)}")
    choice = input("Enter index: ").strip()
    if not choice.isdigit():
        print("[ERROR] Invalid selection.")
        return None
    idx = int(choice)
    if idx < 0 or idx >= len(files):
        print("[ERROR] Invalid selection.")
        return None
    return files[idx]

def choose_folder():
    """Let user select which folder category to inspect"""
    print("\n=== Select Folder Type ===")
    print("[0] Raw data")
    print("[1] Outputs")
    
    choice = input("Enter index: ").strip()
    
    if choice == "0":
        return DATA_PC_DIR
    elif choice == "1":
        return choose_output_subfolder()
    else:
        print("[ERROR] Invalid selection.")
        return None

def choose_output_subfolder():
    """Let user select which output subfolder to inspect"""
    print("\n=== Select Output Subfolder ===")
    print("[0] Downsampled")
    print("[1] Clipped")
    print("[2] Complete point cloud")
    
    choice = input("Enter index: ").strip()
    
    if choice == "0":
        return OUT_DOWNSAMPLED
    elif choice == "1":
        return OUT_CLIPPED
    elif choice == "2":
        return OUT_COMPLETE
    else:
        print("[ERROR] Invalid selection.")
        return None

def inspect_las(las_path):
    print("\n===================================================")
    print(f"Inspecting: {las_path}")
    print("===================================================")

    las = laspy.read(las_path)
    z = las.z

    report_lines = []
    report_lines.append(f"File: {las_path}")
    report_lines.append(f"Date: {datetime.now()}")
    report_lines.append(f"Number of points: {len(las.points)}")

    try:
        crs = las.header.parse_crs()
    except Exception:
        crs = None
    report_lines.append(f"CRS: {crs}")

    report_lines.append(f"Point format: {las.header.point_format.id}")
    report_lines.append(f"LAS version: {las.header.version}")
    report_lines.append(f"Scales:  {las.header.scales}")
    report_lines.append(f"Offsets: {las.header.offsets}")

    # Z stats
    report_lines.append(f"Z min: {z.min():.3f}")
    report_lines.append(f"Z max: {z.max():.3f}")
    report_lines.append(f"Z p1:  {np.percentile(z,1):.3f}")
    report_lines.append(f"Z p5:  {np.percentile(z,5):.3f}")
    report_lines.append(f"Z p50: {np.percentile(z,50):.3f}")
    report_lines.append(f"Z p95: {np.percentile(z,95):.3f}")
    report_lines.append(f"Z p99: {np.percentile(z,99):.3f}")

    # Class histogram
    if hasattr(las, "classification"):
        classes, counts = np.unique(las.classification, return_counts=True)
        report_lines.append("Classification counts:")
        for c, n in zip(classes, counts):
            report_lines.append(f"  Class {int(c)}: {int(n)}")
    else:
        report_lines.append("No classification field")

    # Bounding box + density
    xmin, ymin, zmin = las.x.min(), las.y.min(), z.min()
    xmax, ymax, zmax = las.x.max(), las.y.max(), z.max()
    area = (xmax - xmin) * (ymax - ymin)
    density = len(z) / area if area > 0 else 0

    report_lines.append(f"XY extent: {xmax-xmin:.2f} x {ymax-ymin:.2f} m")
    report_lines.append(f"Approx point density: {density:.2f} pts/m²")

    # Print to console
    for line in report_lines:
        print(line)

    # Save TXT report
    base = os.path.splitext(os.path.basename(las_path))[0]
    out_txt = os.path.join(OUT_INFO, f"{base}_inspect.txt")

    with open(out_txt, "w", encoding="utf-8") as f:
        for line in report_lines:
            f.write(line + "\n")

    print(f"\n-> Saved LAS report to: {out_txt}")

def main():
    print("\n=== Inspect LAS File ===")

    # NEW: if a LAS path is provided, inspect directly
    if len(sys.argv) >= 2 and os.path.exists(sys.argv[1]):
        inspect_las(sys.argv[1])
        ans = input("\nWould you like to visualize this point cloud? [y/N]: ").strip().lower()
        if ans == "y":
            subprocess.run([sys.executable, SCRIPT_VISUALIZE, sys.argv[1]])
        return

if __name__ == "__main__":
    main()