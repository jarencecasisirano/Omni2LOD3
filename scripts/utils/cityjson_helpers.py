import os
from pathlib import Path

from utils.las_helpers import extract_prefix


def prefix_dir(base_dir, file_path):
    out_dir = Path(base_dir) / extract_prefix(str(file_path)).upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def schema_fixed_stem(source_path):
    base = Path(source_path).stem
    if base.endswith("_FIXED"):
        return f"{base[:-6]}_SCHEMA_FIXED"
    return f"{base}_SCHEMA_FIXED"


def gml_stem_from_json(source_path):
    base = Path(source_path).stem
    if base.endswith("_SCHEMA_FIXED"):
        return base[:-13]
    return base


def val3dity_report_json_path(out_val3dity_dir, source_json):
    report_dir = prefix_dir(out_val3dity_dir, source_json)
    return os.path.join(str(report_dir), f"{Path(source_json).stem}_val3dity.json")
