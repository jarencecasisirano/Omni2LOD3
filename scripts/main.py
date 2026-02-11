# main.py
import os
import sys
import glob
import json
import hashlib
import subprocess

# ============================================================
# Project paths (AUTO — no hardcoded C:\ paths)
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_PC_DIR   = os.path.join(PROJECT_ROOT, "data", "01_point_cloud")
DATA_SHP_DIR  = os.path.join(PROJECT_ROOT, "data", "02_footprint")
DATA_JSON_DIR = os.path.join(PROJECT_ROOT, "data", "03_json_model")

OUT_INFO       = os.path.join(PROJECT_ROOT, "outputs", "00_las_info")
OUT_DOWNSAMPLED= os.path.join(PROJECT_ROOT, "outputs", "01_downsampled")
OUT_CLIPPED    = os.path.join(PROJECT_ROOT, "outputs", "02_clipped")
OUT_COMPLETE   = os.path.join(PROJECT_ROOT, "outputs", "03_complete_las")
OUT_VAL3DITY   = os.path.join(PROJECT_ROOT, "outputs", "04_val3dity")
OUT_LOD2_JSON  = os.path.join(PROJECT_ROOT, "outputs", "05_LOD2_json")
OUT_LOD2_GML   = os.path.join(PROJECT_ROOT, "outputs", "06_LOD2_gml")

SCRIPT_CLEAN    = os.path.join(SCRIPT_DIR, "las_to_lod2", "02b_cleanup_building_heights.py")
SCRIPT_INSPECT  = os.path.join(SCRIPT_DIR, "las_to_lod2", "inspect_las.py")
SCRIPT_DOWN     = os.path.join(SCRIPT_DIR, "las_to_lod2", "01_downsampling.py")
SCRIPT_CLIP     = os.path.join(SCRIPT_DIR, "las_to_lod2", "02_clip_z.py")
SCRIPT_GEN      = os.path.join(SCRIPT_DIR, "las_to_lod2", "03_generate_facade_points.py")
SCRIPT_VALIDATE = os.path.join(SCRIPT_DIR, "las_to_lod2", "04_validate_val3dity.py")
SCRIPT_FIX      = os.path.join(SCRIPT_DIR, "las_to_lod2", "05_json_fix.py")
SCRIPT_GML      = os.path.join(SCRIPT_DIR, "las_to_lod2", "06_json_to_gml2.py")
SCRIPT_VISUALIZE= os.path.join(SCRIPT_DIR, "las_to_lod2", "visualize.py")


# ============================================================
# Utilities
# ============================================================

def ensure_dirs():
    for d in (
        OUT_INFO,
        OUT_DOWNSAMPLED,
        OUT_CLIPPED,
        OUT_COMPLETE,
        OUT_VAL3DITY,
        OUT_LOD2_JSON,
        OUT_LOD2_GML,
    ):
        os.makedirs(d, exist_ok=True)

def list_las_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.las")))
    return [f for f in files if not f.lower().endswith(".copc.las")]

def list_shp_files(folder):
    return sorted(glob.glob(os.path.join(folder, "*.shp")))

def list_json_files(folder):
    return sorted(glob.glob(os.path.join(folder, "*.json")))

def choose_file(files, prompt):
    if not files:
        print(f"[ERROR] No files found for: {prompt}")
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

def get_latest_cleaned_las():
    files = sorted(
        glob.glob(os.path.join(OUT_CLIPPED, "*_cleaned.las")),
        key=os.path.getmtime
    )
    return files[-1] if files else None

def extract_prefix(file_path):
    """
    Prefix heuristic: take first token before '_' from filename.
    NEC_112025_05_clipped_cleaned.las -> NEC
    nimbb_020626_fixed.json -> nimbb
    """
    base = os.path.splitext(os.path.basename(file_path))[0]
    if "_" in base:
        return base.split("_")[0]
    return base

def file_hash(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_val3dity_codes(report_json_path):
    if not os.path.exists(report_json_path):
        return []
    try:
        with open(report_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    codes = {}

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, int) and ("error" in k.lower() or "code" in k.lower()):
                    codes[v] = codes.get(v, 0) + 1
                else:
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    return sorted(codes.keys())

def detect_las_crs(las_path):
    """
    Best-effort CRS detection using laspy header.parse_crs().
    Returns str or None.
    """
    try:
        import laspy
        las = laspy.read(las_path)
        crs = las.header.parse_crs()
        return str(crs) if crs is not None else None
    except Exception:
        return None

def auto_select_footprint_for_las(las_path):
    """
    Try to pick a footprint shapefile automatically using LAS prefix.

    Rules:
    - Match prefix token (case-insensitive) against shapefile basename.
      e.g. prefix 'NEC' matches 'NEC_footprint_1.shp' or 'nec_fp.shp'
    - If exactly one match: return it (auto)
    - If multiple matches: prompt user among matches
    - If none: return None
    """
    shp_files = list_shp_files(DATA_SHP_DIR)
    if not shp_files:
        return None

    prefix = extract_prefix(las_path).lower()

    matches = []
    for shp in shp_files:
        name = os.path.basename(shp).lower()
        if name.startswith(prefix + "_") or name.startswith(prefix) or (prefix in name):
            matches.append(shp)

    # Prefer "startswith(prefix_...)" matches if they exist
    strict = [m for m in matches if os.path.basename(m).lower().startswith(prefix + "_")]
    if strict:
        matches = strict

    if len(matches) == 1:
        print(f"\n[Auto] Using footprint: {os.path.basename(matches[0])} (matched prefix '{prefix.upper()}')")
        return matches[0]

    if len(matches) > 1:
        return choose_file(matches, f"Multiple footprints match '{prefix.upper()}'. Select footprint SHP:")

    return None

def tell_user_digitize_footprint(las_path):
    crs = detect_las_crs(las_path)
    print("\n[WARNING] No matching footprint shapefile found.")
    print("Please digitize a footprint shapefile for this building and save it to:")
    print(f"  {DATA_SHP_DIR}")
    print("Make sure the shapefile CRS matches the LAS CRS.")
    print(f"Detected LAS CRS: {crs if crs else 'Unknown (could not parse CRS from LAS header)'}")


def pick_visualize_las():
    """
    New behavior:
    - If user picks 'data' -> immediately list LAS in data/01_point_cloud
    - If user picks 'outputs' -> ask which output subfolder (downsampled/clipped/complete)
    """
    print("\nSelect folder:")
    print("  [1] data")
    print("  [2] outputs")
    choice = input("Enter number: ").strip()

    if choice == "1":
        files = list_las_files(DATA_PC_DIR)
        return choose_file(files, "Select LAS file in data/01_point_cloud:")

    if choice == "2":
        print(f"\nSelect outputs subfolder:")
        print("[0] Downsampled")
        print("[1] Clipped")
        print("[2] Complete point cloud")
        sub = input("Enter index: ").strip()

        if sub == "0":
            folder = OUT_DOWNSAMPLED
        elif sub == "1":
            folder = OUT_CLIPPED
        elif sub == "2":
            folder = OUT_COMPLETE
        else:
            print("[ERROR] Invalid selection.")
            return None

        files = list_las_files(folder)
        return choose_file(files, f"Select LAS file in {os.path.basename(folder)}:")

    print("[ERROR] Invalid selection.")
    return None


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

    result = subprocess.run([sys.executable, SCRIPT_CLIP, input_las, output_las])
    if result.returncode != 0:
        print("[ERROR] Clipping failed.")
        return None

    # Run building-aware cleanup right after clip_z
    cleaned_las = os.path.join(OUT_CLIPPED, f"{base}_clipped_cleaned.las")

    print("\n=== Running building height cleanup ===")
    print(f"Input:  {output_las}")
    print(f"Output: {cleaned_las}")

    # Auto-pick footprint based on LAS prefix; fallback to prompt
    footprint = auto_select_footprint_for_las(output_las)
    if not footprint:
        # If no match, let user choose ANY if available, else instruct digitize
        all_shps = list_shp_files(DATA_SHP_DIR)
        if all_shps:
            footprint = choose_file(all_shps, "No obvious footprint match. Select footprint SHP for cleanup:")
        else:
            tell_user_digitize_footprint(output_las)
            return None

    result2 = subprocess.run([sys.executable, SCRIPT_CLEAN, output_las, footprint, cleaned_las])
    if result2.returncode != 0:
        print("[ERROR] Building cleanup failed.")
        return None

    if not os.path.exists(cleaned_las):
        print(f"[ERROR] Cleaned LAS was not created: {cleaned_las}")
        return None

    return cleaned_las

def step_generate(input_las=None, prompt_for_las=False):
    """
    Behavior:
    - If prompt_for_las=True, ALWAYS ask user to choose LAS from outputs/02_clipped (non-copc).
      This is for menu choice [3].
    - If prompt_for_las=False, we expect caller (choice [2] or [1]) to pass input_las.
      If it's missing, fall back to latest cleaned.
    """
    if prompt_for_las:
        las_files = list_las_files(OUT_CLIPPED)
        input_las = choose_file(las_files, "Select LAS in outputs/02_clipped to generate facades:")
        if not input_las:
            return None
    else:
        if input_las is None:
            input_las = get_latest_cleaned_las()
            if not input_las:
                print("[ERROR] No *_cleaned.las found in 02_clipped.")
                return None

    # Auto-pick footprint based on LAS prefix; fallback to prompt
    footprint_shp = auto_select_footprint_for_las(input_las)
    if not footprint_shp:
        all_shps = list_shp_files(DATA_SHP_DIR)
        if all_shps:
            footprint_shp = choose_file(all_shps, "No obvious footprint match. Select footprint SHP (used for facade):")
        else:
            tell_user_digitize_footprint(input_las)
            return None

    base = os.path.splitext(os.path.basename(input_las))[0]
    base = strip_suffix(base, ["_clipped_cleaned", "_clipped"])
    output_las = os.path.join(OUT_COMPLETE, f"{base}_facade.las")

    print("\n=== Generating facade points ===")
    print(f"Input LAS:      {input_las}")
    print(f"Footprint SHP:  {footprint_shp}")
    print(f"Output LAS:     {output_las}")

    result = subprocess.run([sys.executable, SCRIPT_GEN, input_las, footprint_shp, output_las])
    if result.returncode != 0:
        print("[ERROR] Facade generation failed.")
        return None

    # End-of-step guidance
    print("\n======================================================================")
    print("✅ Point cloud is now ready for CityForge processing.")
    print("Next:")
    print(f"  1) Run CityForge using: {output_las}")
    print(f"  2) Save/export the resulting CityJSON into: {DATA_JSON_DIR}")
    print("  3) Then run menu option [4] (Post-process CityJSON) and [5] (Convert to CityGML 2.0)")
    print("======================================================================\n")

    return output_las

def step_fix_cityjson():
    json_files = list_json_files(DATA_JSON_DIR)
    if not json_files:
        print(f"[ERROR] No CityJSON files found in: {DATA_JSON_DIR}")
        return None

    input_json = choose_file(json_files, "Select CityJSON file to fix:")
    if not input_json:
        return None

    base = os.path.splitext(os.path.basename(input_json))[0]
    output_json = os.path.join(OUT_LOD2_JSON, f"{base}_fixed.json")

    print("\n=== Running CityJSON fix ===")
    print(f"Input:  {input_json}")
    print(f"Output: {output_json}")

    result = subprocess.run([sys.executable, SCRIPT_FIX, input_json, output_json])
    if result.returncode != 0:
        print("[ERROR] CityJSON fix failed.")
        return None

    if not os.path.exists(output_json):
        print(f"[ERROR] Expected output not created: {output_json}")
        return None

    return output_json

def step_json_to_gml(input_json=None):
    if input_json is None:
        json_files = list_json_files(OUT_LOD2_JSON)
        if not json_files:
            print(f"[ERROR] No fixed CityJSON files found in: {OUT_LOD2_JSON}")
            return None

        input_json = choose_file(json_files, f"Select fixed CityJSON to convert to CityGML ({os.path.basename(OUT_LOD2_JSON)}):")
        if not input_json:
            return None

    base = os.path.splitext(os.path.basename(input_json))[0]
    output_gml = os.path.join(OUT_LOD2_GML, f"{base}.gml")

    print("\n=== Converting CityJSON to CityGML 2.0 ===")
    print(f"Input:  {input_json}")
    print(f"Output: {output_gml}")

    result = subprocess.run([sys.executable, SCRIPT_GML, input_json, output_gml])
    if result.returncode != 0:
        print("[ERROR] CityJSON to CityGML failed.")
        return None

    if not os.path.exists(output_gml):
        print(f"[ERROR] Expected output not created: {output_gml}")
        return None

    return output_gml


def step_validate_then_fix():
    MAX_FIX_PASSES = 5
    FIXABLE_CODES = {102, 902}

    json_files = list_json_files(DATA_JSON_DIR)
    if not json_files:
        print(f"[ERROR] No CityJSON files found in: {DATA_JSON_DIR}")
        return None

    input_json = choose_file(json_files, "Select CityJSON file to validate:")
    if not input_json:
        return None

    current_json = input_json
    base = os.path.splitext(os.path.basename(input_json))[0]
    output_json = os.path.join(OUT_LOD2_JSON, f"{base}_FIXED.json")

    for i in range(MAX_FIX_PASSES + 1):
        print("\n=== Running val3dity validation ===")
        result = subprocess.run([sys.executable, SCRIPT_VALIDATE, current_json])

        if result.returncode == 1:
            print("[ERROR] Validation failed.")
            return None

        if result.returncode == 0:
            print("[OK] CityJSON is valid.")
            step_json_to_gml(current_json)
            return current_json

        if result.returncode != 2:
            print("[ERROR] Unexpected validation exit code.")
            return None

        report_json = os.path.join(OUT_VAL3DITY, f"{Path(current_json).stem}_val3dity.json")
        codes = parse_val3dity_codes(report_json)
        if codes and not any(c in FIXABLE_CODES for c in codes):
            print(f"[WARN] No fixable error codes found: {codes}")
            return None

        if i == MAX_FIX_PASSES:
            print("[WARN] Reached max fix passes without a valid file.")
            return None

        print("[WARN] CityJSON is invalid. Running fix...")

        pre_hash = file_hash(current_json)

        print("\n=== Running CityJSON fix ===")
        print(f"Input:  {current_json}")
        print(f"Output: {output_json}")

        result_fix = subprocess.run([sys.executable, SCRIPT_FIX, current_json, output_json])
        if result_fix.returncode != 0:
            print("[ERROR] CityJSON fix failed.")
            return None

        if not os.path.exists(output_json):
            print(f"[ERROR] Expected output not created: {output_json}")
            return None

        post_hash = file_hash(output_json)
        if pre_hash == post_hash:
            print("[WARN] Fix produced no changes. Stopping.")
            return None

        current_json = output_json

    return None


# ============================================================
# Main menu
# ============================================================

def main():
    ensure_dirs()

    print("\n=== Omni2LOD3 ===")
    print("Which part of the pipeline would you like to start with?")
    print("[0] Inspect point cloud")
    print("[1] Voxel downsample")
    print("[2] Clip Z outliers")
    print("[3] Generate facade points")
    print("[4] Validate (then fix if invalid)")
    print("[5] Post-process CityJSON file")
    print("[6] Convert CityJSON to CityGML 2.0")
    print("[V] Visualize point cloud")
    print("[Q] Quit")

    choice = input("Enter choice: ").strip().lower()

    if choice == "q":
        return

    if choice == "0":
        # Let main.py choose LAS; then call inspect with filepath
        las_path = pick_visualize_las()  # reusing the improved picker
        if not las_path:
            return
        subprocess.run([sys.executable, SCRIPT_INSPECT, las_path])

    elif choice == "1":
        out = step_downsample()
        if out:
            out = step_clip(out)
        if out:
            step_generate(out, prompt_for_las=False)

    elif choice == "2":
        out = step_clip()
        if out:
            step_generate(out, prompt_for_las=False)

    elif choice == "3":
        step_generate(input_las=None, prompt_for_las=True)

    elif choice == "4":
        step_validate_then_fix()

    elif choice == "5":
        step_fix_cityjson()

    elif choice == "6":
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
