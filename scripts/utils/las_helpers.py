import os

from utils.io_helpers import choose_file, list_shp_files
from utils.paths import DATA_SHP_DIR

# ======================= LAS UTILITIES =========================

def extract_prefix(file_path):
    """
    Prefix: first token before '_' from filename.
    e.g. nimbb_020626_fixed.json -> nimbb
    """
    base = os.path.splitext(os.path.basename(file_path))[0]
    if "_" in base:
        return base.split("_")[0]
    return base


def detect_las_crs(las_path):
    """
    CRS detection using laspy header.parse_crs().
    """
    try:
        import laspy

        las = laspy.read(las_path)
        crs = las.header.parse_crs()
        return str(crs) if crs is not None else None
    except Exception:
        return None


def matching_footprints_for_las(las_path):
    """
    Return footprint shapefiles whose basename matches LAS prefix.
    """
    shp_files = list_shp_files(DATA_SHP_DIR)
    if not shp_files:
        return []

    prefix = extract_prefix(las_path).lower()

    matches = []
    for shp in shp_files:
        name = os.path.basename(shp).lower()
        if name.startswith(prefix + "_") or name.startswith(prefix) or (prefix in name):
            matches.append(shp)

    strict = [m for m in matches if os.path.basename(m).lower().startswith(prefix + "_")]
    if strict:
        matches = strict

    return matches


def choose_matching_footprint_for_las(las_path, purpose):
    matches = matching_footprints_for_las(las_path)
    prefix = extract_prefix(las_path).upper()
    if len(matches) == 1:
        print(
            f"\t-> Auto-selected footprint: {os.path.basename(matches[0])} "
            f"(matched prefix '{prefix}')"
        )
        return matches[0]
    if len(matches) > 1:
        prompt = f"Multiple footprints match '{prefix}'. Select footprint SHP for {purpose}:"
        return choose_file(matches, prompt)
    return None


def tell_user_digitize_footprint(las_path):
    crs = detect_las_crs(las_path)
    print("\n[WARNING] No matching footprint shapefile found.")
    print("Please digitize a footprint shapefile for this building and save it to:")
    print(f"  {DATA_SHP_DIR}")
    print("Make sure the shapefile CRS matches the LAS CRS.")
    print(f"Detected LAS CRS: {crs if crs else 'Unknown (could not parse CRS from LAS header)'}")
