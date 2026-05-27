import glob
import hashlib
import os

from utils.paths import OUTPUT_DIRS

def ensure_dirs():
    for directory in OUTPUT_DIRS:
        os.makedirs(directory, exist_ok=True)


def list_las_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "**", "*.las"), recursive=True))
    return [f for f in files if not f.lower().endswith(".copc.las")]


def list_shp_files(folder):
    return sorted(glob.glob(os.path.join(folder, "**", "*.shp"), recursive=True))


def list_json_files(folder):
    return sorted(glob.glob(os.path.join(folder, "**", "*.json"), recursive=True))


def choose_file(files, prompt, indent_choices=True):
    if not files:
        print(f"[ERROR] No files found for: {prompt}")
        return None
    print(f"\n{prompt}")
    prefix = "\t" if indent_choices else ""
    for i, file_path in enumerate(files):
        print(f"{prefix}[{i}] {os.path.basename(file_path)}")

    choice = input("Enter choice: ").strip()
    if not choice.isdigit():
        print("[ERROR] Invalid selection.")
        return None
    idx = int(choice)
    if idx < 0 or idx >= len(files):
        print("[ERROR] Invalid selection.")
        return None
    return files[idx]


def choose_index(n, prompt, max_index=None, allowed_values=None):
    choice = input(prompt).strip()
    if not choice.isdigit():
        return None

    idx = int(choice)
    if allowed_values and idx in allowed_values:
        return idx

    upper = (n - 1) if max_index is None else max_index
    if idx < 0 or idx > upper:
        return None
    return idx


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
