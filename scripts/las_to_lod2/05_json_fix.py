# 05_json_fix.py
"""
Fixes (report-driven):
- 902 EMPTY_PRIMITIVE: removes geometry entries with empty boundaries
- 102 CONSECUTIVE_POINTS_SAME: targeted cleanup only on val3dity-flagged faces

CLI mode:
  python 05_json_fix.py <input_json> <output_json> [--report <report_json>] [--tol 0.001]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io_helpers import choose_index, list_json_files
from utils.las_helpers import extract_prefix
from utils.paths import DATA_JSON_DIR, OUT_LOD2_JSON, OUT_VAL3DITY
from utils.val3dity.error_102 import apply_102_fix_from_report
from utils.val3dity.error_902 import apply_902_fix
from utils.val3dity.report import extract_error_codes, load_report_json


DEFAULT_INPUT_DIR = Path(DATA_JSON_DIR)
DEFAULT_OUTPUT_DIR = Path(OUT_LOD2_JSON)
DEFAULT_REPORT_DIR = Path(OUT_VAL3DITY)
DEFAULT_TOL = 0.001


def _report_snap_tol(report_json: dict):
    try:
        tol = report_json.get("parameters", {}).get("snap_tol", None)
        if tol is None:
            return None
        t = float(tol)
        if t > 0:
            return t
    except Exception:
        pass
    return None


def _default_report_path_for_input(input_path: Path):
    prefix_dir = DEFAULT_REPORT_DIR / extract_prefix(str(input_path)).upper()
    prefixed = prefix_dir / f"{input_path.stem}_val3dity.json"
    if prefixed.exists():
        return prefixed
    return DEFAULT_REPORT_DIR / f"{input_path.stem}_val3dity.json"


def _applied_codes_text(stats: dict) -> str:
    applied = []
    if stats.get("fix_902_enabled"):
        applied.append("902")
    if stats.get("fix_102_enabled"):
        applied.append("102")
    return ", ".join(applied) if applied else "none"


def fix_cityjson_file(input_path: Path, output_path: Path, report_json: dict, tol_override=None, target_code=None):
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    city_objects = data.get("CityObjects", {})
    if not isinstance(city_objects, dict):
        raise ValueError("Invalid CityJSON: CityObjects is not a dict")

    codes = extract_error_codes(report_json)
    fix_902_enabled = 902 in codes
    fix_102_enabled = 102 in codes

    if target_code is not None:
        fix_902_enabled = fix_902_enabled and target_code == 902
        fix_102_enabled = fix_102_enabled and target_code == 102

    objects_modified = 0
    geometries_removed = 0

    tol = tol_override if tol_override is not None else _report_snap_tol(report_json)
    if tol is None:
        tol = DEFAULT_TOL

    if fix_102_enabled:
        apply_102_fix_from_report(data, report_json, tol=tol)

    if fix_902_enabled:
        stats902 = apply_902_fix(data)
        objects_modified = stats902["objects_modified"]
        geometries_removed = stats902["geometries_removed"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    return {
        "objects_modified": objects_modified,
        "geometries_removed": geometries_removed,
        "fix_902_enabled": fix_902_enabled,
        "fix_102_enabled": fix_102_enabled,
        "selected_code": target_code,
        "tol_used": tol,
    }


def _parse_cli_options(args):
    report_path = None
    tol_override = None
    target_code = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--report":
            if i + 1 >= len(args):
                raise ValueError("Missing value for --report")
            report_path = Path(args[i + 1])
            i += 2
            continue
        if arg == "--tol":
            if i + 1 >= len(args):
                raise ValueError("Missing value for --tol")
            tol_override = float(args[i + 1])
            if tol_override <= 0:
                raise ValueError("--tol must be > 0")
            i += 2
            continue
        if arg == "--target-code":
            if i + 1 >= len(args):
                raise ValueError("Missing value for --target-code")
            target_code = int(args[i + 1])
            if target_code not in (102, 902):
                raise ValueError("--target-code must be one of: 102, 902")
            i += 2
            continue
        raise ValueError(f"Unknown argument: {arg}")

    return report_path, tol_override, target_code


def _run_one_file(in_path: Path, out_path: Path, report_path: Path, tol_override=None, target_code=None):
    if not in_path.exists():
        print(f"[ERROR] Input file not found: {in_path}")
        return None

    try:
        report_json = load_report_json(report_path)
    except Exception as e:
        print(f"[ERROR] Could not load val3dity report: {e}")
        return None

    try:
        return fix_cityjson_file(
            in_path,
            out_path,
            report_json=report_json,
            tol_override=tol_override,
            target_code=target_code,
        )
    except Exception as e:
        print(f"[ERROR] Fix failed: {e}")
        return None


def main():
    # CLI mode
    if len(sys.argv) >= 3:
        in_path = Path(sys.argv[1])
        out_path = Path(sys.argv[2])

        try:
            report_override, tol_override, target_code = _parse_cli_options(sys.argv[3:])
        except Exception as e:
            print(f"[ERROR] {e}")
            print("Usage: python 05_json_fix.py <input_json> <output_json> [--report <report_json>] [--tol 0.001]")
            sys.exit(1)

        report_path = report_override if report_override else _default_report_path_for_input(in_path)
        stats = _run_one_file(in_path, out_path, report_path, tol_override=tol_override, target_code=target_code)
        if stats is None:
            sys.exit(1)

        print(f"Output: {out_path}")
        print(f"Applied fix: {_applied_codes_text(stats)}")
        return

    # Interactive mode
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR

    print("\n=== RUNNING CITYJSON FIX (INTERACTIVE MODE) ===")

    files = [Path(p) for p in list_json_files(input_dir)]
    if not files:
        print(f"[ERROR] No JSON files found in: {input_dir}")
        sys.exit(1)

    print("\nSelect CityJSON file to fix:\n")
    for i, p in enumerate(files):
        print(f"\t[{i}] {p.name}")
    print(f"\t[{len(files)}] Process ALL files")
    print("\t[99] Exit\n")

    idx = choose_index(
        len(files),
        "Enter choice: ",
        max_index=len(files),
        allowed_values={99},
    )
    if idx is None:
        print("[ERROR] Invalid selection.")
        sys.exit(1)

    if idx == 99:
        return

    targets = files if idx == len(files) else [files[idx]]
    processed = 0

    for p in targets:
        out_path = output_dir / f"{p.stem}_FIXED.json"
        report_path = _default_report_path_for_input(p)

        print(f"\nInput:  {p}")

        stats = _run_one_file(p, out_path, report_path, tol_override=None)
        if stats is None:
            continue

        processed += 1
        print(f"Output: {out_path}")
        print(f"Applied fix: {_applied_codes_text(stats)}")

    if len(targets) > 1:
        print(f"\nFiles processed: {processed}/{len(targets)}")


if __name__ == "__main__":
    main()
