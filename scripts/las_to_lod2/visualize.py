# visualize.py
"""
CLI mode:
  python visualize.py <input_las>
"""
import sys
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io_helpers import choose_file, list_las_files
from utils.loading import create_bar
from utils.paths import OUT_DOWNSAMPLED, OUT_RECLASSIFIED, PROJECT_ROOT

# Color map for LAS classes ======================================

CLASS_INFO = {
    2: {"name": "Ground", "color": [0.55, 0.27, 0.07], "color_name": "Brown"},
    3: {"name": "Low Vegetation", "color": [0.6, 0.9, 0.6], "color_name": "Light Green"},
    5: {"name": "High Vegetation", "color": [0.0, 0.6, 0.0], "color_name": "Dark Green"},
    6: {"name": "Building", "color": [1.0, 0.0, 0.0], "color_name": "Red"},
    11: {"name": "Road", "color": [0.5, 0.5, 0.5], "color_name": "Gray"},
}


def print_class_legend(classes):
    print("\n=== Classification Legend ===")
    unique = np.unique(classes)

    for cls in unique:
        if cls in CLASS_INFO:
            name = CLASS_INFO[cls]["name"]
            color_name = CLASS_INFO[cls]["color_name"]
        else:
            name = "Unknown / Other"
            color_name = "Dark Gray"
        print(f"Class {cls:>2} | {name:<16} | Color: {color_name}")
    print("==============================")


def choose_root_folder():
    print("\nSelect folder:")
    print("  [1] data")
    print("  [2] outputs")

    choice = input("Enter choice: ").strip()
    if choice == "1":
        return Path(PROJECT_ROOT) / "data"
    if choice == "2":
        return Path(PROJECT_ROOT) / "outputs"

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
    picked = choose_file(
        list_las_files(str(folder)),
        f"LAS files in {folder}:",
        indent_choices=True,
    )
    return Path(picked) if picked else None


def visualize_las_classes(las_path, point_size=1.0):
    las_path = Path(las_path)
    bar = create_bar("        Preparing visualization", 1)
    las = laspy.read(str(las_path))

    xyz = np.vstack((las.x, las.y, las.z)).T
    classes = las.classification
    print_class_legend(classes)
    bar.next()
    bar.finish()

    colors = np.zeros((xyz.shape[0], 3))
    for cls, info in CLASS_INFO.items():
        colors[classes == cls] = info["color"]
    colors[colors.sum(axis=1) == 0] = [0.2, 0.2, 0.2]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"LAS Classification: {las_path.name}",
        width=1280,
        height=800,
    )
    vis.add_geometry(pcd)

    render_opt = vis.get_render_option()
    render_opt.point_size = float(point_size)
    render_opt.background_color = np.array([0, 0, 0])

    vis.run()
    vis.destroy_window()


def main():
    if len(sys.argv) >= 2:
        las_path = Path(sys.argv[1])
        if not las_path.exists():
            print(f"[ERROR] File not found: {las_path}")
            return
        visualize_las_classes(las_path)
        return

    root = choose_root_folder()
    folder = choose_subfolder(root)

    # Fast-path for common pipeline outputs.
    if folder == Path(PROJECT_ROOT) / "outputs":
        print("\nTip: choose [0] downsampled or [1] reclassified.")
        print("[0] 01_downsampled")
        print("[1] 02_reclassified")
        choice = input("Enter choice (or Enter to browse all): ").strip()
        if choice == "0":
            folder = Path(OUT_DOWNSAMPLED)
        elif choice == "1":
            folder = Path(OUT_RECLASSIFIED)

    las_file = choose_las_file(folder)
    if las_file:
        visualize_las_classes(las_file)


if __name__ == "__main__":
    main()
