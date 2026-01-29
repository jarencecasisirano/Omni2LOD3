#!/usr/bin/env python3
import os
import sys
import laspy
import numpy as np
import open3d as o3d
from pathlib import Path

# ============================================================
# Project paths (auto)
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ============================================================
# Color map for LAS classes
# ============================================================
CLASS_INFO = {
    2: {"name": "Ground",        "color": [0.55, 0.27, 0.07]},  # brown
    3: {"name": "Low Vegetation","color": [0.6, 0.9, 0.6]},    # light green
    5: {"name": "High Vegetation","color": [0.0, 0.6, 0.0]},   # dark green
    6: {"name": "Building",      "color": [1.0, 0.0, 0.0]},   # red
    11: {"name": "Road",         "color": [0.5, 0.5, 0.5]},   # gray
}

def print_class_legend(classes):
    print("\n=== Classification Legend ===")
    unique, counts = np.unique(classes, return_counts=True)

    for cls, cnt in zip(unique, counts):
        if cls in CLASS_INFO:
            name = CLASS_INFO[cls]["name"]
        else:
            name = "Unknown / Other"

        print(f"Class {cls:>2} | {name:<16} | Points: {cnt:,}")

    print("==============================")

# ============================================================
# CLI helpers (interactive mode)
# ============================================================

def choose_root_folder():
    print("\nSelect folder:")
    print("  [1] data")
    print("  [2] outputs")

    choice = input("Enter number: ").strip()
    if choice == "1":
        return DATA_DIR
    elif choice == "2":
        return OUTPUTS_DIR
    else:
        print("Invalid choice.")
        return choose_root_folder()

def choose_subfolder(root):
    subfolders = [p for p in root.iterdir() if p.is_dir()]

    if not subfolders:
        print("No subfolders found.")
        return root

    print(f"\nSubfolders in {root}:")
    for i, folder in enumerate(subfolders):
        print(f"  [{i}] {folder.name}")

    choice = input("Select subfolder (or press Enter to use this folder): ").strip()
    if choice == "":
        return root

    try:
        return subfolders[int(choice)]
    except (IndexError, ValueError):
        print("Invalid choice.")
        return choose_subfolder(root)

def choose_las_file(folder):
    las_files = sorted(folder.glob("*.las"))
    las_files = [f for f in las_files if not f.name.lower().endswith(".copc.las")]

    if not las_files:
        print("No .las files found in this folder.")
        return None

    print(f"\nLAS files in {folder}:")
    for i, f in enumerate(las_files):
        print(f"  [{i}] {f.name}")

    choice = input("Select file: ").strip()
    try:
        return las_files[int(choice)]
    except (IndexError, ValueError):
        print("Invalid choice.")
        return choose_las_file(folder)

# ============================================================
# Visualization core
# ============================================================
def visualize_las_classes(las_path, point_size=1.0):
    print(f"\nLoading {las_path}")
    las = laspy.read(las_path)

    xyz = np.vstack((las.x, las.y, las.z)).T
    classes = las.classification

    # Print legend + counts (for thesis / QA)
    print_class_legend(classes)

    colors = np.zeros((xyz.shape[0], 3))
    for cls, info in CLASS_INFO.items():
        colors[classes == cls] = info["color"]

    # Unknown classes = dark gray
    colors[colors.sum(axis=1) == 0] = [0.2, 0.2, 0.2]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # -------- Open3D Visualizer (for point size control) --------
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"LAS Classification: {las_path.name}",
        width=1280,
        height=800
    )

    vis.add_geometry(pcd)

    render_opt = vis.get_render_option()
    render_opt.point_size = float(point_size)   # ⭐ THIS CONTROLS POINT SIZE
    render_opt.background_color = np.array([0, 0, 0])  # black background (optional)

    vis.run()
    vis.destroy_window()


# ============================================================
# Main
# ============================================================

def main():
    # If a LAS path is provided, use it directly
    if len(sys.argv) >= 2:
        las_path = Path(sys.argv[1])
        if not las_path.exists():
            print(f"[ERROR] File not found: {las_path}")
            return
        visualize_las_classes(las_path)
        return

    # Otherwise, fall back to interactive browser
    root = choose_root_folder()
    folder = choose_subfolder(root)
    las_file = choose_las_file(folder)

    if las_file:
        visualize_las_classes(las_file)

if __name__ == "__main__":
    main()
