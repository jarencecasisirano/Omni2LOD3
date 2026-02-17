# main.py
import subprocess
import sys

from pipeline.cityjson_steps import step_json_to_gml, step_validate_then_fix
from pipeline.point_cloud_steps import pick_visualize_las, step_assign_building_class, step_downsample
from utils.io_helpers import ensure_dirs
from utils.paths import SCRIPT_INSPECT, SCRIPT_VISUALIZE

def main():
    ensure_dirs()

    print("\n=== Omni2LOD3 ===")
    print("Which part of the pipeline would you like to start with?")
    print("[0] Inspect Point Cloud")
    print("[1] Voxel Downsample")
    print("[2] Reclassify Point Cloud")
    print("[3] Validate via Val3dity")
    print("[4] Convert CityJSON to CityGML 2.0")
    print("[V] Visualize point cloud")
    print("[Q] Quit")

    choice = input("Enter choice: ").strip().lower()

    if choice == "q":
        return

    if choice == "0":
        las_path = pick_visualize_las()
        if not las_path:
            return
        subprocess.run([sys.executable, SCRIPT_INSPECT, las_path])

    elif choice == "1":
        output_las = step_downsample()
        if output_las:
            step_assign_building_class(output_las)

    elif choice == "2":
        step_assign_building_class()

    elif choice == "3":
        step_validate_then_fix()

    elif choice == "4":
        step_json_to_gml()

    elif choice == "v":
        las_path = pick_visualize_las()
        if not las_path:
            return
        subprocess.run([sys.executable, SCRIPT_VISUALIZE, las_path])

    else:
        print("[ERROR] Invalid choice.")


if __name__ == "__main__":
    main()
