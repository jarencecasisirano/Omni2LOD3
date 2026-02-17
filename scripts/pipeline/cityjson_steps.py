import json
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

# ======================= CITYJSON TO CITYGML PROCESSING =========================


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
            for key, value in node.items():
                if isinstance(value, int) and ("error" in key.lower() or "code" in key.lower()):
                    codes[value] = codes.get(value, 0) + 1
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    return sorted(codes.keys())


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

        prompt = f"Select fixed CityJSON to convert to CityGML ({os.path.basename(OUT_LOD2_JSON)}):"
        input_json = choose_file(json_files, prompt)
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
    max_fix_passes = 5
    fixable_codes = {102, 204, 902}

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
        if codes and not any(code in fixable_codes for code in codes):
            print(f"[WARN] No fixable error codes found: {codes}")
            return None

        if i == max_fix_passes:
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
