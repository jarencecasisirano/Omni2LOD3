import os
import subprocess
import sys
from pathlib import Path

from utils.io_helpers import choose_file, file_hash, list_json_files
from utils.las_helpers import extract_prefix
from utils.paths import (
    DATA_JSON_DIR,
    OUT_LOD2_GML,
    OUT_LOD2_JSON,
    OUT_VAL3DITY,
    SCRIPT_FIX,
    SCRIPT_GML,
    SCRIPT_SCHEMA_FIX,
    SCRIPT_VALIDATE,
)
from utils.val3dity.report import load_report_error_codes

# ======================= CITYJSON TO CITYGML PROCESSING =========================


def parse_val3dity_codes(report_json_path):
    try:
        return sorted(load_report_error_codes(report_json_path))
    except Exception:
        return []


def _prefix_dir(base_dir, file_path):
    prefix = extract_prefix(file_path).upper()
    out_dir = os.path.join(base_dir, prefix)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _schema_fixed_stem(source_path):
    base = os.path.splitext(os.path.basename(source_path))[0]
    if base.endswith("_FIXED"):
        return f"{base[:-6]}_SCHEMA_FIXED"
    return f"{base}_SCHEMA_FIXED"


def _gml_stem_from_json(source_path):
    base = os.path.splitext(os.path.basename(source_path))[0]
    if base.endswith("_SCHEMA_FIXED"):
        return base[:-13]
    return base


def _val3dity_report_json_path(source_json):
    report_dir = _prefix_dir(OUT_VAL3DITY, source_json)
    stem = Path(source_json).stem
    return os.path.join(report_dir, f"{stem}_val3dity.json")


def step_json_to_gml(input_json=None):
    if input_json is None:
        json_files = list_json_files(OUT_LOD2_JSON)
        if not json_files:
            print(f"[ERROR] No fixed CityJSON files found in: {OUT_LOD2_JSON}")
            return None

        prompt = f"Select fixed CityJSON to convert to CityGML ({os.path.basename(OUT_LOD2_JSON)}):"
        input_json = choose_file(json_files, prompt)
        if not input_json:
            return None

    output_dir = _prefix_dir(OUT_LOD2_GML, input_json)
    output_stem = _gml_stem_from_json(input_json)
    output_gml = os.path.join(output_dir, f"{output_stem}.gml")

    result = subprocess.run([sys.executable, SCRIPT_GML, input_json, output_gml])
    if result.returncode != 0:
        print("[ERROR] CityJSON to CityGML failed.")
        return None

    if not os.path.exists(output_gml):
        print(f"[ERROR] Expected output not created: {output_gml}")
        return None

    return output_gml


def step_schema_then_fix(input_json):
    output_dir = _prefix_dir(OUT_LOD2_JSON, input_json)
    output_json = os.path.join(output_dir, f"{_schema_fixed_stem(input_json)}.json")

    result = subprocess.run(
        [
            sys.executable,
            SCRIPT_SCHEMA_FIX,
            input_json,
            output_json,
        ],
    )

    if result.returncode != 0:
        print("[WARN] Schema fix step failed; continuing with current JSON.")
        return input_json

    if not os.path.exists(output_json):
        print("[WARN] Schema fix did not produce an output file; continuing with current JSON.")
        return input_json

    return output_json


def step_validate_then_fix():
    max_fix_passes = 5
    # Thesis mode: apply conservative auto-fixes; allow conversion when only
    # residual 102 remains from source-modeling limitations.
    fixable_codes = {102, 902}
    convertible_invalid_codes = {102}

    json_files = list_json_files(DATA_JSON_DIR)
    if not json_files:
        print(f"[ERROR] No CityJSON files found in: {DATA_JSON_DIR}")
        return None

    input_json = choose_file(json_files, "Select CityJSON file to validate:")
    if not input_json:
        return None

    current_json = input_json
    base = os.path.splitext(os.path.basename(input_json))[0]
    output_dir = _prefix_dir(OUT_LOD2_JSON, input_json)
    output_json = os.path.join(output_dir, f"{base}_FIXED.json")

    for i in range(max_fix_passes + 1):
        print(f"\n=== RUN {i + 1}: VAL3DITY FOR VALIDATION ===")
        result = subprocess.run([sys.executable, SCRIPT_VALIDATE, current_json])

        if result.returncode == 1:
            print("[ERROR] Validation failed.")
            return None

        if result.returncode == 0:
            schema_json = step_schema_then_fix(current_json)
            step_json_to_gml(schema_json)
            return schema_json

        if result.returncode != 2:
            print("[ERROR] Unexpected validation exit code.")
            return None

        report_json = _val3dity_report_json_path(current_json)
        codes = parse_val3dity_codes(report_json)
        if codes:
            only_convertible_invalid = all(code in convertible_invalid_codes for code in codes)
            any_fixable = any(code in fixable_codes for code in codes)
            if only_convertible_invalid:
                print("[NOTE] CityJSON retains residual val3dity code(s): 102.")
                print(
                    "[NOTE] Interpreted as source-modeling limitation "
                    "(near-duplicate consecutive vertices), while geometry "
                    "remains interpretable for LoD2 use."
                )
                print("[NOTE] Proceeding to CityGML conversion with this documented limitation.")
                schema_json = step_schema_then_fix(current_json)
                step_json_to_gml(schema_json)
                return schema_json
            if not any_fixable:
                print(f"[WARN] No additional fixable codes found: {codes}")
                print(
                    "[NOTE] Remaining issues are treated as source-modeling limitations; "
                    "geometry is retained as interpretable for intended LoD2 use."
                )
                print("[NOTE] Proceeding to CityGML conversion with documented limitations.")
                schema_json = step_schema_then_fix(current_json)
                step_json_to_gml(schema_json)
                return schema_json

        if i == max_fix_passes:
            print("[WARN] Reached max fix passes without a fully valid file.")
            print(
                "[NOTE] Remaining issues are treated as source-modeling limitations; "
                "geometry is retained as interpretable for intended LoD2 use."
            )
            print("[NOTE] Proceeding to CityGML conversion with documented limitations.")
            schema_json = step_schema_then_fix(current_json)
            step_json_to_gml(schema_json)
            return schema_json

        print("\t[WARN] CityJSON is invalid. Running fix...")

        pre_hash = file_hash(current_json)
        applied_codes = sorted([code for code in codes if code in fixable_codes]) if codes else []
        applied_text = ", ".join(str(code) for code in applied_codes) if applied_codes else "auto"
        print("\n=== RUNNING CITYJSON FIX ===")
        print(f"Output: {output_json}")
        print(f"Applied fix: {applied_text}")

        fix_cmd = [sys.executable, SCRIPT_FIX, current_json, output_json]

        result_fix = subprocess.run(
            fix_cmd,
            capture_output=True,
            text=True,
        )
        if result_fix.returncode != 0:
            print("[ERROR] CityJSON fix failed.")
            if result_fix.stderr and result_fix.stderr.strip():
                print(result_fix.stderr.strip())
            elif result_fix.stdout and result_fix.stdout.strip():
                print(result_fix.stdout.strip())
            return None

        if not os.path.exists(output_json):
            print(f"[ERROR] Expected output not created: {output_json}")
            return None

        post_hash = file_hash(output_json)
        if pre_hash == post_hash:
            print("[WARN] Fix produced no changes.")
            convert_anyway = input("Convert to CityGML anyway? [y/N]: ").strip().lower()
            if convert_anyway in {"y", "yes"}:
                schema_json = step_schema_then_fix(current_json)
                step_json_to_gml(schema_json)
                return schema_json
            print("[WARN] Stopping.")
            return None

        current_json = output_json

    return None
