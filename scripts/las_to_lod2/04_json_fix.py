#!/usr/bin/env python3
"""
04_json_fix.py

LOD2 CityJSON Empty Geometry Fixer (PIPELINE-FRIENDLY)

Behavior:
- Default INPUT dir:  <project_root>/data/03_json_model
- Default OUTPUT dir: <project_root>/outputs/04_LOD2_json
- Lists JSON files in INPUT dir and asks user which one to process (or ALL)
- Writes: <stem>_FIXED.json into OUTPUT dir

Also supports non-interactive CLI:
  python 04_json_fix.py <input_json> <output_json>
"""

import os
import sys
import json
from pathlib import Path

# -------------------------
# Path defaults (relative to repo)
# -------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # scripts/
# If your repo layout is Omni2LOD3/scripts/las_to_lod2/04_json_fix.py
# then parent of scripts is Omni2LOD3/scripts, so project root is scripts/..
PROJECT_ROOT = PROJECT_ROOT.parent

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "03_json_model"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "04_LOD2_json"

# -------------------------
# Fix logic
# -------------------------

def _is_empty_boundaries(boundaries) -> bool:
    # boundaries expected to be list-like
    if boundaries is None:
        return True
    if not isinstance(boundaries, (list, tuple)):
        return True
    return len(boundaries) == 0

def _remove_empty_geometries(cityobj: dict) -> int:
    """
    Remove geometry entries where geometry['boundaries'] is empty.
    Returns count removed.
    """
    geoms = cityobj.get("geometry", None)
    if not geoms or not isinstance(geoms, list):
        return 0

    keep = []
    removed = 0
    for g in geoms:
        if not isinstance(g, dict):
            keep.append(g)
            continue
        boundaries = g.get("boundaries", None)
        if _is_empty_boundaries(boundaries):
            removed += 1
            continue
        keep.append(g)

    if removed > 0:
        cityobj["geometry"] = keep
    return removed

def fix_cityjson_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    """
    Returns (objects_modified, geometries_removed).
    """
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    city_objects = data.get("CityObjects", {})
    if not isinstance(city_objects, dict):
        raise ValueError("Invalid CityJSON: CityObjects is not a dict")

    objects_modified = 0
    geometries_removed = 0

    for obj_id, obj in city_objects.items():
        if not isinstance(obj, dict):
            continue
        removed = _remove_empty_geometries(obj)
        if removed > 0:
            objects_modified += 1
            geometries_removed += removed

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    return objects_modified, geometries_removed

# -------------------------
# UI helpers
# -------------------------

def list_json_files(folder: Path):
    if not folder.exists():
        return []
    return sorted(folder.glob("*.json"))

def choose_index(n: int, prompt: str) -> int | None:
    choice = input(prompt).strip()
    if not choice.isdigit():
        return None
    idx = int(choice)
    if idx < 0 or idx > n:
        return None
    return idx

# -------------------------
# Main
# -------------------------

def main():
    # ------------------------------------------------------------
    # CLI mode: called from main.py as: python 04_json_fix.py in out
    # ------------------------------------------------------------
    if len(sys.argv) >= 3:
        in_path = Path(sys.argv[1])
        out_path = Path(sys.argv[2])

        print("\n" + "=" * 70)
        print("🏗️  LOD2 CityJSON Geometry Fixer (CLI MODE)")
        print("=" * 70)
        print(f"Input:  {in_path}")
        print(f"Output: {out_path}")

        if not in_path.exists():
            print(f"[ERROR] Input file not found: {in_path}")
            sys.exit(1)

        try:
            obj_mod, geom_rm = fix_cityjson_file(in_path, out_path)
        except Exception as e:
            print(f"[ERROR] Fix failed: {e}")
            sys.exit(1)

        print(f"✅ Done. Objects modified: {obj_mod}, geometries removed: {geom_rm}")
        print("=" * 70 + "\n")
        return

    # ------------------------------------------------------------
    # Interactive folder mode (standalone use)
    # ------------------------------------------------------------
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR

    print("\n" + "=" * 70)
    print("🏗️  LOD2 CityJSON Geometry Fixer (INTERACTIVE MODE)")
    print("=" * 70)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")

    files = list_json_files(input_dir)
    if not files:
        print(f"[ERROR] No JSON files found in: {input_dir}")
        sys.exit(1)

    print(f"\n📁 Found {len(files)} JSON file(s):\n")
    for i, p in enumerate(files):
        size_kb = p.stat().st_size / 1024.0
        print(f"  [{i}] {p.name:<35} ({size_kb:,.1f} KB)")
    print(f"  [{len(files)}] Process ALL files")
    print("  [99] Exit\n")

    idx = choose_index(len(files), f"👉 Select file to fix [0-{len(files)}] (or 99): ")
    if idx is None:
        print("[ERROR] Invalid selection.")
        sys.exit(1)

    if idx == 99:
        return

    targets = files if idx == len(files) else [files[idx]]

    total_obj_mod = 0
    total_geom_rm = 0

    for p in targets:
        out_name = f"{p.stem}_FIXED.json"
        out_path = output_dir / out_name

        print(f"\n📄 Processing: {p.name}")
        try:
            obj_mod, geom_rm = fix_cityjson_file(p, out_path)
        except Exception as e:
            print(f"[ERROR] Failed on {p.name}: {e}")
            continue

        total_obj_mod += obj_mod
        total_geom_rm += geom_rm
        print(f"  ✅ Saved to: {out_path}")
        print(f"  📊 Summary: {obj_mod} objects modified, {geom_rm} geometries removed")

    print("\n" + "=" * 70)
    print("📊 PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Files processed:          {len(targets)}/{len(targets)}")
    print(f"Objects modified:         {total_obj_mod}")
    print(f"Empty geometries removed: {total_geom_rm}")
    print("=" * 70)
    print(f"📂 Check: {output_dir}")
    print("✨ Done!\n")

if __name__ == "__main__":
    main()
