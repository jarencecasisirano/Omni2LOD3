import os
import subprocess
import sys

from utils.io_helpers import choose_file, list_las_files, strip_suffix
from utils.las_helpers import choose_matching_footprint_for_las, tell_user_digitize_footprint
from utils.paths import (
    DATA_JSON_DIR,
    DATA_PC_DIR,
    OUT_DOWNSAMPLED,
    OUT_RECLASSIFIED,
    SCRIPT_ASSIGN,
    SCRIPT_DOWN,
)

# ======================= POINT CLOUD PROCESSING =========================


def pick_visualize_las():
    print("\nSelect folder:")
    print("  [1] data")
    print("  [2] outputs")
    choice = input("Enter number: ").strip()

    if choice == "1":
        files = list_las_files(DATA_PC_DIR)
        return choose_file(files, "Select LAS file in data/01_point_cloud:")

    if choice == "2":
        print("\nSelect outputs subfolder:")
        print("[0] Downsampled")
        print("[1] Reclassified")
        sub = input("Enter index: ").strip()

        if sub == "0":
            folder = OUT_DOWNSAMPLED
        elif sub == "1":
            folder = OUT_RECLASSIFIED
        else:
            print("[ERROR] Invalid selection.")
            return None

        files = list_las_files(folder)
        return choose_file(files, f"Select LAS file in {os.path.basename(folder)}:")

    print("[ERROR] Invalid selection.")
    return None


def step_downsample():
    files = list_las_files(DATA_PC_DIR)
    input_las = choose_file(files, "Select raw point cloud to downsample:")
    if not input_las:
        return None

    base = os.path.splitext(os.path.basename(input_las))[0]
    voxel = input("Enter voxel size (e.g., 0.2): ").strip()
    try:
        voxel_float = float(voxel)
    except ValueError:
        print("[ERROR] Invalid voxel size.")
        return None

    voxel_str = str(voxel_float).replace(".", "")
    output_las = os.path.join(OUT_DOWNSAMPLED, f"{base}_{voxel_str}_downsampled.las")

    print("\n=== Running voxel downsampling ===")
    print(f"Input:  {input_las}")
    print(f"Output: {output_las}")

    result = subprocess.run([sys.executable, SCRIPT_DOWN, input_las, output_las, voxel])
    if result.returncode != 0:
        print("[ERROR] Downsampling failed.")
        return None

    return output_las


def step_assign_building_class(input_las=None):
    if input_las is None:
        files = list_las_files(OUT_DOWNSAMPLED)
        input_las = choose_file(files, "Select downsampled LAS to reassign by footprint:")
        if not input_las:
            return None

    base = os.path.splitext(os.path.basename(input_las))[0]
    base = strip_suffix(base, ["_downsampled"])
    output_las = os.path.join(OUT_RECLASSIFIED, f"{base}_reclassified.las")

    print("\n=== Running building class reassignment by footprint ===")
    print(f"Input:  {input_las}")
    print(f"Output: {output_las}")

    footprint = choose_matching_footprint_for_las(input_las, "reassignment")
    if not footprint:
        tell_user_digitize_footprint(input_las)
        return None

    result = subprocess.run([sys.executable, SCRIPT_ASSIGN, input_las, footprint, output_las])
    if result.returncode != 0:
        print("[ERROR] Building class reassignment failed.")
        return None

    if not os.path.exists(output_las):
        print(f"[ERROR] Reclassified LAS was not created: {output_las}")
        return None

    print("\n======================================================================")
    print("Point cloud is now ready for CityForge processing.")
    print("Next:")
    print(f"  1) Run CityForge using: {output_las}")
    print(f"  2) Save/export the resulting CityJSON into: {DATA_JSON_DIR}")
    print("  3) Then run menu option [3] (Validate + fix if invalid), then [4] (Convert to CityGML 2.0)")
    print("======================================================================\n")

    return output_las
