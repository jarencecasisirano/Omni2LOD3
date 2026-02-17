import glob
import hashlib
import os

from utils.paths import OUTPUT_DIRS

# ======================= UTILITIES =========================


def ensure_dirs():
    for directory in OUTPUT_DIRS:
        os.makedirs(directory, exist_ok=True)


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
    for i, file_path in enumerate(files):
        print(f"[{i}] {os.path.basename(file_path)}")

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
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def file_hash(path):
    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha1.update(chunk)
    return sha1.hexdigest()
