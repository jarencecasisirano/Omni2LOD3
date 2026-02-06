#!/usr/bin/env python3
"""
05_json_to_gml2.py
Convert CityJSON -> CityGML 2.0 using citygml-tools (from-cityjson).

Pipeline-friendly:
- CLI mode (used by main.py):
    python 05_json_to_gml2.py <input_json> <output_gml>

- Interactive mode (no args):
    - lists JSON in outputs/04_LOD2_json
    - writes GML to outputs/05_LOD2_gml
"""

import os
import sys
import glob
import subprocess
from pathlib import Path

# ============================================================
# Project-relative paths (NO hardcoded C:\ paths)
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent                  # .../scripts/las_to_lod2
PROJECT_ROOT = SCRIPT_DIR.parent.parent                       # .../Omni2LOD3

DEFAULT_JSON_DIR = PROJECT_ROOT / "outputs" / "04_LOD2_json"
DEFAULT_GML_DIR  = PROJECT_ROOT / "outputs" / "05_LOD2_gml"

# Optional: allow user to override tool location via env var
ENV_BAT = os.environ.get("CITYGML_TOOLS_BAT", "").strip()

def find_citygml_tools_bat() -> Path | None:
    """
    Find citygml-tools.bat.
    Priority:
      1) env var CITYGML_TOOLS_BAT
      2) PROJECT_ROOT/tools/**/citygml-tools.bat
    """
    if ENV_BAT:
        p = Path(ENV_BAT)
        if p.exists():
            return p

    tools_dir = PROJECT_ROOT / "tools"
    if tools_dir.exists():
        hits = list(tools_dir.rglob("citygml-tools.bat"))
        if hits:
            # pick newest (often safest if you have multiple versions)
            hits.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return hits[0]

    return None

def list_json_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() == ".json"])
    return files

def choose_file(files: list[Path], prompt: str) -> Path | None:
    if not files:
        print(f"[ERROR] No files found for: {prompt}")
        return None
    print(f"\n{prompt}")
    for i, p in enumerate(files):
        print(f"[{i}] {p.name}")
    choice = input("Enter index: ").strip()
    if not choice.isdigit():
        print("[ERROR] Invalid selection.")
        return None
    idx = int(choice)
    if idx < 0 or idx >= len(files):
        print("[ERROR] Invalid selection.")
        return None
    return files[idx]

def convert_to_citygml2(json_path: Path, output_gml: Path, tools_bat: Path) -> int:
    output_gml.parent.mkdir(parents=True, exist_ok=True)

    print("\n=== Converting CityJSON to CityGML 2.0 ===")
    print(f"Input:   {json_path}")
    print(f"Output:  {output_gml}")
    print(f"Tool:    {tools_bat}")

    cmd = [
        str(tools_bat),
        "from-cityjson",
        str(json_path),
        "-v", "2.0",
        "-o", str(output_gml),
    ]

    try:
        # capture_output=True hides tool logs; set to False if you want live logs
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("\n[ERROR] citygml-tools failed.")
            if result.stdout.strip():
                print("\n--- stdout ---")
                print(result.stdout)
            if result.stderr.strip():
                print("\n--- stderr ---")
                print(result.stderr)
            return result.returncode

        if not output_gml.exists():
            print("\n[ERROR] Tool reported success but output file was not created.")
            return 2

        print("\n✓ SUCCESS! CityGML 2.0 saved:")
        print(f"  {output_gml}")
        return 0

    except FileNotFoundError:
        print(f"[ERROR] Tool not found: {tools_bat}")
        return 3

def main():
    tools_bat = find_citygml_tools_bat()
    if tools_bat is None:
        print("[ERROR] Could not find citygml-tools.bat.")
        print("Fix options:")
        print("  1) Put citygml-tools under: <PROJECT_ROOT>/tools/")
        print("  2) Or set env var CITYGML_TOOLS_BAT to the .bat path")
        sys.exit(1)

    # ------------------------------------------------------------
    # CLI mode for main.py
    # ------------------------------------------------------------
    if len(sys.argv) >= 3:
        json_in = Path(sys.argv[1]).resolve()
        gml_out = Path(sys.argv[2]).resolve()

        if not json_in.exists():
            print(f"[ERROR] Input JSON not found: {json_in}")
            sys.exit(1)

        rc = convert_to_citygml2(json_in, gml_out, tools_bat)
        sys.exit(rc)

    # ------------------------------------------------------------
    # Interactive mode (standalone)
    # ------------------------------------------------------------
    if not DEFAULT_JSON_DIR.exists():
        print(f"[ERROR] Folder not found: {DEFAULT_JSON_DIR}")
        sys.exit(1)

    json_files = list_json_files(DEFAULT_JSON_DIR)
    if not json_files:
        print(f"[ERROR] No .json files found in: {DEFAULT_JSON_DIR}")
        sys.exit(1)

    picked = choose_file(json_files, "Select fixed CityJSON to convert to CityGML:")
    if not picked:
        sys.exit(1)

    DEFAULT_GML_DIR.mkdir(parents=True, exist_ok=True)
    output_gml = DEFAULT_GML_DIR / f"{picked.stem}.gml"

    rc = convert_to_citygml2(picked, output_gml, tools_bat)
    sys.exit(rc)

if __name__ == "__main__":
    main()
