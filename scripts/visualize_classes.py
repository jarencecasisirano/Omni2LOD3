import laspy
import numpy as np
import open3d as o3d
from pathlib import Path

# Root folders - dynamically determined from script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # scripts/ -> project root
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Color map for LAS classes
CLASS_COLORS = {
    2: [0.55, 0.27, 0.07],   # ground - brown
    3: [0.6, 0.9, 0.6],     # low veg - light green
    5: [0.0, 0.6, 0.0],     # high veg - dark green
    6: [1.0, 0.0, 0.0],     # building - red
    11: [0.5, 0.5, 0.5],    # road - gray
}

# ---------------------------
# CLI helpers
# ---------------------------

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

# ---------------------------
# Visualization
# ---------------------------

def visualize_las_classes(las_path):
    print(f"\nLoading {las_path}")
    las = laspy.read(las_path)

    xyz = np.vstack((las.x, las.y, las.z)).T
    classes = las.classification

    colors = np.zeros((xyz.shape[0], 3))
    for cls, color in CLASS_COLORS.items():
        colors[classes == cls] = color

    colors[colors.sum(axis=1) == 0] = [0.2, 0.2, 0.2]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"LAS Classification: {las_path.name}",
        point_show_normal=False
    )

# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    root = choose_root_folder()
    folder = choose_subfolder(root)
    las_file = choose_las_file(folder)

    if las_file:
        visualize_las_classes(las_file)
