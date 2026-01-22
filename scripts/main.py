import os, subprocess, glob

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "..", "data")
CLIPPED_DIR   = os.path.join(BASE_DIR, "..", "outputs", "clipped")
DOWNSAMPLED_DIR = os.path.join(BASE_DIR, "..", "outputs", "downsampled")
GROUND_CLASSIFIED_DIR = os.path.join(BASE_DIR, "..", "outputs", "ground_classification")
BUILDING_CLASSIFIED_DIR = os.path.join(BASE_DIR, "..", "outputs", "building_classification")
COMPLETE_LAS_DIR      = os.path.join(BASE_DIR, "..", "outputs", "complete_las")
SEGMENTED_PLANES_DIR = os.path.join(BASE_DIR, "..", "outputs", "segmentation")
FOOTPRINT_DIR = os.path.join(BASE_DIR, "..", "outputs", "footprint")

for d in (DOWNSAMPLED_DIR, GROUND_CLASSIFIED_DIR, BUILDING_CLASSIFIED_DIR,
          COMPLETE_LAS_DIR, SEGMENTED_PLANES_DIR, FOOTPRINT_DIR):
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------------
#  ONE list that mixes raw + clipped + complete_las for down-sample picker
# ------------------------------------------------------------------
_downsample_candidates = []
for folder in (DATA_DIR, CLIPPED_DIR, COMPLETE_LAS_DIR):
    _downsample_candidates.extend(
        sorted(glob.glob(os.path.join(folder, "*.las")))
    )

# Menu options and associated scripts
MENU_OPTIONS = [
    ("VOXEL DOWNSAMPLE", "voxel_downsampling.py","Select raw/clipped LAS input for voxel_downsampling.py", _downsample_candidates),
    ("GROUND CLASSIFICATION", "classify_points.py", "Select downsampled LAS input for ground classification (edited classify_points.py)", DOWNSAMPLED_DIR),
    ("CLUSTER UNCLASSIFIED", "dbscan.py", "Select ground-classified LAS input for DBSCAN clustering", GROUND_CLASSIFIED_DIR),
    ("PLANE SEGMENTATION", "segment_planes.py", "Select building-classified LAS input for segment_planes.py", BUILDING_CLASSIFIED_DIR),
    ("PLANE CLASSIFICATION", "classify_planes.py", "Select segmented PLY input for classify_planes.py", SEGMENTED_PLANES_DIR),
    ("GENERATE FOOTPRINT", "generate_footprint.py", "Select building-classified OR complete LAS input for footprint generation",
     (BUILDING_CLASSIFIED_DIR, COMPLETE_LAS_DIR)),
]

def choose_file_from_folder(folder, prompt, extensions=[".las", ".geojson", ".ply"]):
    """Single-folder picker (legacy)."""
    files = sorted([f for f in glob.glob(os.path.join(folder, "*")) if os.path.splitext(f)[1].lower() in extensions])
    if not files:
        print(f"[ERROR] No files with extensions {extensions} found in {folder}")
        return None
    print(f"\n{prompt}:")
    for i, f in enumerate(files):
        print(f"[{i}] {os.path.basename(f)}")
    choice = input("Enter index of file: ").strip()
    if not choice.isdigit() or int(choice) not in range(len(files)):
        print("[ERROR] Invalid selection")
        return None
    return files[int(choice)]

def generate_output_path(script, input_path):
    base = os.path.basename(input_path)
    name, _ = os.path.splitext(base)
    # Clean up name by removing previous suffixes
    suffixes_to_remove = ["_downsampled_.*", "_ground_classified", "_building_classified", "_segmented"]
    for suffix in suffixes_to_remove:
        if suffix in name:
            name = name.split(suffix)[0]
    if script == "voxel_downsampling.py":
        return os.path.join(DOWNSAMPLED_DIR, f"{name}_downsampled_{{VOXEL}}.las")
    elif script == "classify_points.py":
        return os.path.join(GROUND_CLASSIFIED_DIR, f"{name}_ground_classified.las")
    elif script == "dbscan.py":
        return os.path.join(BUILDING_CLASSIFIED_DIR, f"{name}_clustered.las")
    elif script == "classify_building.py":
        return os.path.join(BUILDING_CLASSIFIED_DIR, f"{name}_building_classified.las")
    elif script == "segment_planes.py":
        return os.path.join(SEGMENTED_PLANES_DIR, f"{name}_segmented.ply")
    elif script == "classify_planes.py":
        return [
            os.path.join(SEGMENTED_PLANES_DIR, f"{name}_roofs.ply"),
            os.path.join(SEGMENTED_PLANES_DIR, f"{name}_walls.ply"),
        ]
    elif script == "generate_footprint.py":
        return os.path.join(FOOTPRINT_DIR, f"{name}_footprint.geojson")
    return None

def run_selected_step(option_index):
    label, script, prompt, folder = MENU_OPTIONS[option_index]
    if script == "voxel_downsampling.py":
        # ----------------------------------------------------------
        # build the same combined list on the fly
        # ----------------------------------------------------------
        candidates = []
        for d in (DATA_DIR, CLIPPED_DIR, COMPLETE_LAS_DIR):
            candidates.extend(glob.glob(os.path.join(d, "*.las")))
        candidates = sorted(candidates)
        if not candidates:
            print("[ERROR] No LAS/LAZ found in data/, outputs/clipped/, or outputs/complete_las/")
            return False
        print(f"\n{prompt}:")
        for idx, path in enumerate(candidates):
            folder_name = os.path.basename(os.path.dirname(path))
            print(f"[{idx}] {os.path.basename(path)}  ({folder_name})")
        try:
            choice = int(input("Enter index of file: ").strip())
            input_file = candidates[choice]
        except (ValueError, IndexError):
            print("[ERROR] Invalid selection")
            return False

        voxel_size = input("Enter voxel size (e.g., 0.2): ").strip()
        try:
            float(voxel_size)
        except ValueError:
            print("[ERROR] Invalid voxel size.")
            return False
        output_path = generate_output_path(script, input_file).replace("{VOXEL}", voxel_size.replace('.', '_'))
        print(f"\n=== Running: {script} ===")
        print(f"Input: {input_file}")
        print(f"Output: {output_path}")
        result = subprocess.run(["python", script, input_file, output_path, voxel_size])
        if result.returncode != 0:
            print(f"[ERROR] {script} failed.")
            return False
        print(f"[SUCCESS] {script} complete. Output: {output_path}\n")
        return True

    elif script == "segment_planes.py":
        input_file = choose_file_from_folder(folder, prompt, extensions=[".las", ".laz"])
        if not input_file:
            return False
        output_path = generate_output_path(script, input_file)
        print(f"\n=== Running: {script} ===")
        print(f"Input: {input_file}")
        print(f"Output: {output_path}")
        result = subprocess.run(["python", script, input_file, output_path], check=True)
        if result.returncode != 0:
            print(f"[ERROR] {script} failed with return code {result.returncode}.")
            return False
        print(f"[SUCCESS] {script} complete. Output: {output_path}\n")
        return True

    elif script == "classify_planes.py":
        input_file = choose_file_from_folder(folder, prompt, extensions=[".ply"])
        if not input_file:
            return False
        output_paths = generate_output_path(script, input_file)
        print(f"\n=== Running: {script} ===")
        print(f"Input: {input_file}")
        print(f"Outputs: {output_paths[0]}, {output_paths[1]}")
        result = subprocess.run(["python", script, input_file, output_paths[0], output_paths[1]], check=True)
        if result.returncode != 0:
            print(f"[ERROR] {script} failed with return code {result.returncode}.")
            return False
        print(f"[SUCCESS] {script} complete.\n")
        return True

    elif script == "generate_footprint.py":
        # ----------------------------------------------------------
        # NEW: allow picker to span two folders
        # ----------------------------------------------------------
        folders = folder if isinstance(folder, (list, tuple)) else [folder]
        candidates = []
        for d in folders:
            candidates.extend(glob.glob(os.path.join(d, "*.las")))
        candidates = sorted(set(candidates))          # remove duplicates
        if not candidates:
            print("[ERROR] No LAS/LAZ found in supplied folders.")
            return False
        print(f"\n{prompt}:")
        for idx, path in enumerate(candidates):
            print(f"[{idx}] {os.path.basename(path)}  ({os.path.basename(os.path.dirname(path))})")
        try:
            choice = int(input("Enter index of file: ").strip())
            input_file = candidates[choice]
        except (ValueError, IndexError):
            print("[ERROR] Invalid selection.")
            return False
        output_path = generate_output_path(script, input_file)
        print(f"\n=== Running: {script} ===")
        print(f"Input: {input_file}")
        print(f"Output: {output_path}")
        result = subprocess.run(["python", script, input_file, output_path])
        if result.returncode != 0:
            print(f"[ERROR] {script} failed with return code {result.returncode}.")
            return False
        print(f"[SUCCESS] {script} complete. Output: {output_path}\n")
        return True

    else:  # Generic for ground classification, building classification, and dbscan
        input_file = choose_file_from_folder(folder, prompt, extensions=[".las", ".laz"])
        if not input_file:
            return False
        output_path = generate_output_path(script, input_file)
        print(f"\n=== Running: {script} ===")
        print(f"Input: {input_file}")
        print(f"Output: {output_path}")
        result = subprocess.run(["python", script, input_file, output_path], check=True)
        if result.returncode != 0:
            print(f"[ERROR] {script} failed with return code {result.returncode}.")
            return False
        print(f"[SUCCESS] {script} complete. Output: {output_path}\n")
        return True

def main():
    print("What do you want to do?")
    for i, (label, _, _, _) in enumerate(MENU_OPTIONS):
        print(f"[{i}] {label}")
    choice = input("Enter index: ").strip()
    if not choice.isdigit() or int(choice) not in range(len(MENU_OPTIONS)):
        print("[ERROR] Invalid selection.")
        return
    run_selected_step(int(choice))

if __name__ == "__main__":
    main()