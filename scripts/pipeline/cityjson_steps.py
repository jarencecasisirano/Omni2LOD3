import os
import subprocess
import sys
from pathlib import Path

from utils.io_helpers import choose_file, file_hash, list_json_files
from utils.paths import (
    DATA_JSON_DIR,
    OUT_LOD2_GML,
    OUT_LOD2_JSON,
    OUT_VAL3DITY,
    SCRIPT_FIX,
    SCRIPT_GML,
    SCRIPT_VALIDATE,
)
from utils.val3dity_report import load_report_error_codes

# ======================= CITYJSON TO CITYGML PROCESSING =========================


def parse_val3dity_codes(report_json_path):
    try:
        return sorted(load_report_error_codes(report_json_path))
    except Exception:
        return []


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

    base = os.path.splitext(os.path.basename(input_json))[0]
    output_gml = os.path.join(OUT_LOD2_GML, f"{base}.gml")

    result = subprocess.run([sys.executable, SCRIPT_GML, input_json, output_gml])
    if result.returncode != 0:
        print("[ERROR] CityJSON to CityGML failed.")
        return None

    if not os.path.exists(output_gml):
        print(f"[ERROR] Expected output not created: {output_gml}")
        return None

    return output_gml


def step_validate_then_fix():
    max_fix_passes = 5
    fixable_codes = {102, 204, 307, 902}

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

    for i in range(max_fix_passes + 1):
        print(f"\n=== RUN {i + 1}: VAL3DITY FOR VALIDATION ===")
        result = subprocess.run([sys.executable, SCRIPT_VALIDATE, current_json])

        if result.returncode == 1:
            print("[ERROR] Validation failed.")
            return None

        if result.returncode == 0:
            step_json_to_gml(current_json)
            return current_json

        if result.returncode != 2:
            print("[ERROR] Unexpected validation exit code.")
            return None

        report_json = os.path.join(OUT_VAL3DITY, f"{Path(current_json).stem}_val3dity.json")
        codes = parse_val3dity_codes(report_json)
        if codes and not any(code in fixable_codes for code in codes):
            print(f"[WARN] No fixable error codes found: {codes}")
            return None

        if i == max_fix_passes:
            print("[WARN] Reached max fix passes without a valid file.")
            return None

        print("\t[WARN] CityJSON is invalid. Running fix...")

        pre_hash = file_hash(current_json)
        applied_codes = sorted([code for code in codes if code in fixable_codes]) if codes else []
        applied_text = ", ".join(str(code) for code in applied_codes) if applied_codes else "auto"
        print("\n=== RUNNING CITYJSON FIX ===")
        print(f"Output: {output_json}")
        print(f"Applied fix: {applied_text}")

        result_fix = subprocess.run(
            [sys.executable, SCRIPT_FIX, current_json, output_json],
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
            print("[WARN] Fix produced no changes. Stopping.")
            return None

        current_json = output_json

    return None
