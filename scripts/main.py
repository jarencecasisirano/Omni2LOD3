import subprocess
import sys

from pipeline.cityjson_steps import step_json_to_gml, step_validate_then_fix
from pipeline.point_cloud_steps import pick_visualize_las, step_assign_building_class, step_downsample
from utils.data_organizer import organize_data_folders
from utils.io_helpers import ensure_dirs
from utils.paths import SCRIPT_INSPECT, SCRIPT_VISUALIZE
from utils.schema.profile import ensure_profile_for_validation_interactive

def main():
    ensure_dirs()
    org = organize_data_folders()
    if org["moved_files"] > 0 or org["copc_removed"] > 0:
        print(
            f"[INFO] Data organizer moved {org['moved_files']} file(s) into "
            f"{org['created_folders']} prefix folder(s)."
        )
        if org["copc_removed"] > 0:
            print(f"[INFO] Removed {org['copc_removed']} '*.copc.las' file(s).")

    print("\n=== OMNI2LOD3 ===")
    print("Which part of the pipeline would you like to start with?")
    print("\t[0] Inspect Point Cloud")
    print("\t[1] Voxel Downsample")
    print("\t[2] Reclassify Point Cloud")
    print("\t[3] Validate via Val3dity + cjio schema")
    print("\t[4] Convert CityJSON to CityGML 2.0")
    print("\t[V] Visualize point cloud")
    print("\t[Q] Quit")

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
        ensure_profile_for_validation_interactive()
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
