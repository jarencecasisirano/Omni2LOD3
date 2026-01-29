#!/usr/bin/env python3
import os
import sys
import glob
import subprocess

# ============================================================
# Project paths (AUTO — no hardcoded C:\ paths)
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_PC_DIR   = os.path.join(PROJECT_ROOT, "data", "01_point_cloud")
DATA_JSON_DIR = os.path.join(PROJECT_ROOT, "data", "02_json_model")

OUT_DOWNSAMPLED = os.path.join(PROJECT_ROOT, "outputs", "01_downsampled")
OUT_CLIPPED     = os.path.join(PROJECT_ROOT, "outputs", "02_clipped")
OUT_NORMALIZED  = os.path.join(PROJECT_ROOT, "outputs", "03_normalized")
OUT_LOD2_JSON   = os.path.join(PROJECT_ROOT, "outputs", "04_LOD2_json")
OUT_LOD2_GML    = os.path.join(PROJECT_ROOT, "outputs", "05_LOD2_gml")

SCRIPT_INSPECT = os.path.join(SCRIPT_DIR, "las_to_lod2", "inspect_las.py")
SCRIPT_DOWN = os.path.join(SCRIPT_DIR, "las_to_lod2", "01_downsampling.py")
SCRIPT_CLIP = os.path.join(SCRIPT_DIR, "las_to_lod2", "02_clip_z.py")
SCRIPT_NORM = os.path.join(SCRIPT_DIR, "las_to_lod2", "03_normalize.py")
SCRIPT_FIX  = os.path.join(SCRIPT_DIR, "las_to_lod2", "04_json_fix.py")
SCRIPT_GML  = os.path.join(SCRIPT_DIR, "las_to_lod2", "05_json_to_gml2.py")
SCRIPT_VISUALIZE = os.path.join(SCRIPT_DIR, "las_to_lod2", "visualize.py")

# ============================================================
# Utilities
# ============================================================

def ensure_dirs():
    for d in (
        OUT_DOWNSAMPLED,
        OUT_CLIPPED,
        OUT_NORMALIZED,
        OUT_LOD2_JSON,
        OUT_LOD2_GML,
    ):
        os.makedirs(d, exist_ok=True)

def list_las_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.las")))
    return [f for f in files if not f.lower().endswith(".copc.las")]

def choose_file(files, prompt):
    if not files:
        print(f"[ERROR] No LAS files found for: {prompt}")
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

def strip_suffix(name, suffixes):
    for s in suffixes:
        if name.endswith(s):
            return name[: -len(s)]
    return name

# ============================================================
# Pipeline steps
# ============================================================

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

    # Format voxel size for filename (0.5 -> 05, 0.2 -> 02, etc.)
    voxel_str = str(voxel_float).replace(".", "")
    output_las = os.path.join(OUT_DOWNSAMPLED, f"{base}_{voxel_str}_downsampled.las")

    print("\n=== Running voxel downsampling ===")
    print(f"Input:  {input_las}")
    print(f"Output: {output_las}")

    result = subprocess.run([
        sys.executable,
        SCRIPT_DOWN,
        input_las,
        output_las,
        voxel
    ])

    if result.returncode != 0:
        print("[ERROR] Downsampling failed.")
        return None

    return output_las

def step_clip(input_las=None):
    if input_las is None:
        files = list_las_files(OUT_DOWNSAMPLED)
        input_las = choose_file(files, "Select downsampled LAS to clip:")
        if not input_las:
            return None

    base = os.path.splitext(os.path.basename(input_las))[0]
    base = strip_suffix(base, ["_downsampled"])
    output_las = os.path.join(OUT_CLIPPED, f"{base}_clipped.las")

    print("\n=== Running Z clipping ===")
    print(f"Input:  {input_las}")
    print(f"Output: {output_las}")

    result = subprocess.run([
        sys.executable,
        SCRIPT_CLIP,
        input_las,
        output_las
    ])

    if result.returncode != 0:
        print("[ERROR] Clipping failed.")
        return None

    return output_las

def step_normalize(input_las=None):
    if input_las is None:
        files = list_las_files(OUT_CLIPPED)
        input_las = choose_file(files, "Select clipped LAS to normalize:")
        if not input_las:
            return None

    base = os.path.splitext(os.path.basename(input_las))[0]
    base = strip_suffix(base, ["_clipped"])
    output_las = os.path.join(OUT_NORMALIZED, f"{base}_normalized.las")

    print("\n=== Running normalization ===")
    print(f"Input:  {input_las}")
    print(f"Output: {output_las}")

    result = subprocess.run([
        sys.executable,
        SCRIPT_NORM,
        input_las,
        output_las
    ])

    if result.returncode != 0:
        print("[ERROR] Normalization failed.")
        return None

    print("\n=== Point cloud ready for CityForge ===")
    print("Please:")
    print("  1. Open QGIS / CityForge")
    print("  2. Digitize building footprint")
    print("  3. Generate LOD2 CityJSON model")
    print(f"  4. Save CityJSON to: {DATA_JSON_DIR}")

    return output_las

def step_fix_cityjson():
    json_files = sorted(glob.glob(os.path.join(DATA_JSON_DIR, "*.json")))
    if not json_files:
        print("[ERROR] No CityJSON files found in data/02_json_model")
        return None

    input_json = choose_file(json_files, "Select CityJSON file to fix:")
    if not input_json:
        return None

    base = os.path.splitext(os.path.basename(input_json))[0]
    output_json = os.path.join(OUT_LOD2_JSON, f"{base}_fixed.json")

    print("\n=== Running CityJSON fix ===")
    print(f"Input:  {input_json}")
    print(f"Output: {output_json}")

    result = subprocess.run([
        sys.executable,
        SCRIPT_FIX,
        input_json,
        output_json
    ])

    if result.returncode != 0:
        print("[ERROR] CityJSON fix failed.")
        return None

    return output_json

def step_json_to_gml():
    json_files = sorted(glob.glob(os.path.join(OUT_LOD2_JSON, "*.json")))
    if not json_files:
        print("[ERROR] No fixed CityJSON files found in outputs/04_LOD2_json")
        return None

    input_json = choose_file(json_files, "Select fixed CityJSON to convert to CityGML:")
    if not input_json:
        return None

    base = os.path.splitext(os.path.basename(input_json))[0]
    output_gml = os.path.join(OUT_LOD2_GML, f"{base}.gml")

    print("\n=== Converting CityJSON to CityGML 2.0 ===")
    print(f"Input:  {input_json}")
    print(f"Output: {output_gml}")

    result = subprocess.run([
        sys.executable,
        SCRIPT_GML,
        input_json,
        output_gml
    ])

    if result.returncode != 0:
        print("[ERROR] CityJSON to CityGML failed.")
        return None

    return output_gml

# ============================================================
# Main menu + auto pipeline
# ============================================================

def main():
    ensure_dirs()

    print("\n=== Omni2LOD3 ===")
    print("Which part of the pipeline would you like to start with?")
    print("[0] Inspect point cloud")
    print("[1] Voxel downsample")
    print("[2] Clip Z outliers")
    print("[3] Normalize point cloud")
    print("[4] Post-process CityJSON file")
    print("[5] Convert CityJSON to CityGML 2.0")
    print("[V] Visualize point cloud")
    print("[Q] Quit")

    choice = input("Enter choice: ").strip().lower()

    if choice == "q":
        return

    last_output = None

    if choice == "0":
        subprocess.run([sys.executable, SCRIPT_INSPECT])
        return

    elif choice == "1":
        last_output = step_downsample()
        if last_output:
            last_output = step_clip(last_output)
        if last_output:
            last_output = step_normalize(last_output)

    elif choice == "2":
        last_output = step_clip()
        if last_output:
            last_output = step_normalize(last_output)

    elif choice == "3":
        step_normalize()

    elif choice == "4":
        step_fix_cityjson()

    elif choice == "5":
        step_json_to_gml()

    elif choice == "v":
        subprocess.run([sys.executable, SCRIPT_VISUALIZE])
        return

    else:
        print("[ERROR] Invalid choice.")

if __name__ == "__main__":
    main()
