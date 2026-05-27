import os
import shutil

from utils.paths import DATA_JSON_DIR, DATA_PC_DIR, DATA_SHP_DIR, OUT_INFO, OUTPUT_DIRS


def _extract_prefix(filename):
    stem, _ = os.path.splitext(filename)
    token = stem.split("_", 1)[0].strip()
    if not token:
        return None
    return token.upper()


def _should_skip_file(path):
    name = os.path.basename(path).lower()
    # Keep profile file at a stable location for schema defaults.
    if name == "schema_identity.json":
        return True
    return False


def _organize_single_folder(folder):
    moved = 0
    created_dirs = set()
    if not os.path.isdir(folder):
        return moved, 0

    for entry in os.scandir(folder):
        if not entry.is_file():
            continue
        if _should_skip_file(entry.path):
            continue
        prefix = _extract_prefix(entry.name)
        if not prefix:
            continue
        target_dir = os.path.join(folder, prefix)
        target_path = os.path.join(target_dir, entry.name)
        if os.path.abspath(entry.path) == os.path.abspath(target_path):
            continue
        os.makedirs(target_dir, exist_ok=True)
        shutil.move(entry.path, target_path)
        created_dirs.add(target_dir)
        moved += 1

    return moved, len(created_dirs)


def _remove_copc_in_folder(folder):
    removed = 0
    if not os.path.isdir(folder):
        return 0
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(".copc.las"):
                fpath = os.path.join(root, name)
                try:
                    os.remove(fpath)
                    removed += 1
                except OSError:
                    continue
    return removed


def organize_data_folders():
    data_folders = (DATA_PC_DIR, DATA_SHP_DIR, DATA_JSON_DIR)
    output_folders = tuple(d for d in OUTPUT_DIRS if d != OUT_INFO)
    folders = data_folders + output_folders

    total_moved = 0
    total_created = 0
    total_copc_removed = 0

    for folder in folders:
        moved, created = _organize_single_folder(folder)
        total_moved += moved
        total_created += created

    # Remove transient COPC products from both data and outputs trees.
    cleanup_roots = data_folders + OUTPUT_DIRS
    for root in cleanup_roots:
        total_copc_removed += _remove_copc_in_folder(root)

    return {
        "moved_files": total_moved,
        "created_folders": total_created,
        "copc_removed": total_copc_removed,
    }
