# inspect_las.py
import os
import sys
import glob
import laspy
import numpy as np
import subprocess


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

SCRIPT_VISUALIZE = os.path.join(PROJECT_ROOT, "scripts", "las_to_lod2", "visualize.py")
DATA_PC_DIR   = os.path.join(PROJECT_ROOT, "data", "01_point_cloud")
OUT_DOWNSAMPLED = os.path.join(PROJECT_ROOT, "outputs", "01_downsampled")
OUT_CLIPPED     = os.path.join(PROJECT_ROOT, "outputs", "02_clipped")
OUT_NORMALIZED  = os.path.join(PROJECT_ROOT, "outputs", "03_normalized")

FOLDER_MAP = {
    "1": ("Raw point clouds", DATA_PC_DIR),
    "2": ("Downsampled", OUT_DOWNSAMPLED),
    "3": ("Clipped", OUT_CLIPPED),
    "4": ("Normalized", OUT_NORMALIZED),
}

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

def inspect_las(las_path):
    if not os.path.exists(las_path):
        print(f"[ERROR] File not found: {las_path}")
        return

    print("\n===================================================")
    print(f"Inspecting: {las_path}")
    print("===================================================")

    las = laspy.read(las_path)

    print(f"Number of points: {len(las.points)}")

    try:
        crs = las.header.parse_crs()
    except Exception:
        crs = None
    print(f"CRS: {crs}")

    print(f"Point format: {las.header.point_format.id}")
    print(f"LAS version: {las.header.version}")
    print(f"Scales:  {las.header.scales}")
    print(f"Offsets: {las.header.offsets}")

    if hasattr(las, "classification"):
        classes, counts = np.unique(las.classification, return_counts=True)
        class_dict = dict(zip(classes.tolist(), counts.tolist()))
        print(f"Classes: {class_dict}")
    else:
        print("No classification field")

    print(f"Z min: {las.z.min():.3f}")
    print(f"Z max: {las.z.max():.3f}")

def main():
    print("\n=== Inspect LAS File ===")
    print("Select which dataset to inspect:")
    for k, (label, _) in FOLDER_MAP.items():
        print(f"[{k}] {label}")
    print("[Q] Quit")

    choice = input("Enter choice: ").strip().lower()
    if choice == "q":
        return

    if choice not in FOLDER_MAP:
        print("[ERROR] Invalid choice.")
        return

    label, folder = FOLDER_MAP[choice]
    files = list_las_files(folder)
    las_file = choose_file(files, f"Select LAS file from {label}:")
    if not las_file:
        return

    inspect_las(las_file)

    ans = input("\nWould you like to visualize this point cloud? [y/N]: ").strip().lower()
    if ans == "y":
        subprocess.run([
            sys.executable,
            SCRIPT_VISUALIZE,
            las_file
        ])

if __name__ == "__main__":
    main()
