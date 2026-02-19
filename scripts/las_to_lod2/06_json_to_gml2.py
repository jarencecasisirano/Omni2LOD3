# 06_json_to_gml2.py
"""
CLI mode (used by main.py):
    python 06_json_to_gml2.py <input_json> <output_gml>
"""

import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.cityjson_helpers import gml_stem_from_json, prefix_dir
from utils.io_helpers import choose_file, list_json_files
from utils.paths import OUT_LOD2_GML, OUT_LOD2_JSON, PROJECT_ROOT, TOOLS_DIR

PROJECT_ROOT = Path(PROJECT_ROOT)
DEFAULT_JSON_DIR = Path(OUT_LOD2_JSON)
DEFAULT_GML_DIR = Path(OUT_LOD2_GML)

ENV_BAT = os.environ.get("CITYGML_TOOLS_BAT", "").strip()

def _rel(pathlike):
    p = Path(pathlike)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(p)


def find_citygml_tools_bat() -> Path | None:
    if ENV_BAT:
        p = Path(ENV_BAT)
        if p.exists():
            return p

    tools_dir = Path(TOOLS_DIR)
    if tools_dir.exists():
        hits = list(tools_dir.rglob("citygml-tools.bat"))
        if hits:
            # pick newest (often safest if you have multiple versions)
            hits.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return hits[0]

    return None

def convert_to_citygml2(json_path: Path, output_gml: Path, tools_bat: Path) -> int:
    output_gml.parent.mkdir(parents=True, exist_ok=True)

    print("\n=== CONVERTING CITYJSON TO CITYGML 2.0 ===")
    print(f"Input:  {_rel(json_path)}")
    print(f"Tool:   {tools_bat}")

    cmd = [
        str(tools_bat),
        "from-cityjson",
        str(json_path),
        "-v", "2.0",
        "-o", str(output_gml),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("\n\t[ERROR] citygml-tools failed.")
            if result.stdout.strip():
                print("\n\t--- stdout ---")
                print(result.stdout)
            if result.stderr.strip():
                print("\n\t--- stderr ---")
                print(result.stderr)
            return result.returncode

        if not output_gml.exists():
            print("\n\t[ERROR] Tool reported success but output file was not created.")
            return 2

        print("\tSUCCESS! CityGML 2.0 saved:")
        print(f"\t{_rel(output_gml)}")
        return 0

    except FileNotFoundError:
        print(f"[ERROR] Tool not found: {tools_bat}")
        return 3

def main():
    tools_bat = find_citygml_tools_bat()
    if tools_bat is None:
        print("[\tERROR] Could not find citygml-tools.bat.")
        print("\tFix options:")
        print("\t  1) Put citygml-tools under: <PROJECT_ROOT>/tools/")
        print("\t  2) Or set env var CITYGML_TOOLS_BAT to the .bat path")
        sys.exit(1)

    if len(sys.argv) >= 3: # CLI mode
        json_in = Path(sys.argv[1]).resolve()
        gml_out = Path(sys.argv[2]).resolve()

        if not json_in.exists():
            print(f"[ERROR] Input JSON not found: {json_in}")
            sys.exit(1)

        rc = convert_to_citygml2(json_in, gml_out, tools_bat)
        sys.exit(rc)

    if not DEFAULT_JSON_DIR.exists():
        print(f"[ERROR] Folder not found: {DEFAULT_JSON_DIR}")
        sys.exit(1)

    json_files = [Path(p) for p in list_json_files(DEFAULT_JSON_DIR)]
    if not json_files:
        print(f"[ERROR] No .json files found in: {DEFAULT_JSON_DIR}")
        sys.exit(1)

    picked = choose_file([str(p) for p in json_files], "Select fixed CityJSON to convert to CityGML:")
    if not picked:
        sys.exit(1)
    picked = Path(picked)

    DEFAULT_GML_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = prefix_dir(DEFAULT_GML_DIR, picked)
    output_gml = out_dir / f"{gml_stem_from_json(picked)}.gml"

    rc = convert_to_citygml2(picked, output_gml, tools_bat)
    sys.exit(rc)

if __name__ == "__main__":
    main()
