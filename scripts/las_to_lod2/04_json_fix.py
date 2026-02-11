# 04_json_fix.py
"""
LOD2 CityJSON Geometry Fixer (PIPELINE-FRIENDLY)

Fixes only:
- 902 EMPTY_PRIMITIVE: removes geometry entries with empty boundaries []
- 102 CONSECUTIVE_POINTS_SAME: removes consecutive vertices that have identical XYZ coords
  (even if the vertex indices differ)

Behavior:
- Default INPUT dir:  <project_root>/data/03_json_model
- Default OUTPUT dir: <project_root>/outputs/04_LOD2_json
- Lists JSON files in INPUT dir and asks user which one to process (or ALL)
- Writes: <stem>_FIXED.json into OUTPUT dir

CLI mode:
  python 04_json_fix.py <input_json> <output_json>
"""

import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "03_json_model"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "04_LOD2_json"

EPS = 1e-12


def _is_empty_boundaries(boundaries) -> bool:
    if boundaries is None:
        return True
    if not isinstance(boundaries, (list, tuple)):
        return True
    return len(boundaries) == 0


def _is_ring(node):
    return isinstance(node, list) and len(node) >= 2 and all(isinstance(v, int) for v in node)


def _coord_key(vertices, vidx):
    try:
        v = vertices[vidx]
        return (float(v[0]), float(v[1]), float(v[2]))
    except Exception:
        return None


def _same_xyz(a, b):
    if a is None or b is None:
        return False
    return (abs(a[0] - b[0]) <= EPS) and (abs(a[1] - b[1]) <= EPS) and (abs(a[2] - b[2]) <= EPS)


def _clean_ring_by_xyz(ring, vertices):
    if not ring:
        return ring, 0

    cleaned = []
    fixed = 0
    prev_xyz = None

    for vidx in ring:
        xyz = _coord_key(vertices, vidx)
        if prev_xyz is not None and _same_xyz(xyz, prev_xyz):
            fixed += 1
            continue
        cleaned.append(vidx)
        prev_xyz = xyz

    return cleaned, fixed


def _clean_boundaries_recursive(boundaries, vertices):
    """
    Only fix 102 by removing consecutive same-XYZ points in rings.
    Does NOT drop rings/faces or perform other topology edits.
    """
    rings_fixed = 0

    def walk(node):
        nonlocal rings_fixed

        if _is_ring(node):
            cleaned, fixed = _clean_ring_by_xyz(node, vertices)
            rings_fixed += fixed
            return cleaned

        if isinstance(node, list):
            return [walk(child) for child in node]

        return node

    return walk(boundaries), rings_fixed


def _fix_cityobject_geometries(cityobj: dict, vertices):
    geoms = cityobj.get("geometry", None)
    if not geoms or not isinstance(geoms, list):
        return {
            "geometries_removed": 0,
            "rings_fixed": 0,
            "geometries_touched": 0,
        }

    keep = []
    geometries_removed = 0
    geometries_touched = 0
    rings_fixed_total = 0

    for g in geoms:
        if not isinstance(g, dict):
            keep.append(g)
            continue

        boundaries = g.get("boundaries", None)

        # 902: remove empty primitive
        if _is_empty_boundaries(boundaries):
            geometries_removed += 1
            continue

        newb, rf = _clean_boundaries_recursive(boundaries, vertices)
        g["boundaries"] = newb

        if rf > 0:
            geometries_touched += 1
            rings_fixed_total += rf

        # Keep 902 check after cleanup (top-level empty only)
        if _is_empty_boundaries(newb):
            geometries_removed += 1
            continue

        keep.append(g)

    if geometries_removed > 0 or geometries_touched > 0:
        cityobj["geometry"] = keep

    return {
        "geometries_removed": geometries_removed,
        "rings_fixed": rings_fixed_total,
        "geometries_touched": geometries_touched,
    }


def fix_cityjson_file(input_path: Path, output_path: Path):
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    city_objects = data.get("CityObjects", {})
    if not isinstance(city_objects, dict):
        raise ValueError("Invalid CityJSON: CityObjects is not a dict")

    vertices = data.get("vertices", None)
    if not isinstance(vertices, list) or len(vertices) == 0:
        raise ValueError("Invalid CityJSON: missing/empty 'vertices' array (needed for 102 fix)")

    objects_modified = 0
    geometries_removed = 0
    rings_fixed = 0

    for _, obj in city_objects.items():
        if not isinstance(obj, dict):
            continue

        stats = _fix_cityobject_geometries(obj, vertices)

        changed = stats["geometries_removed"] > 0 or stats["geometries_touched"] > 0
        if changed:
            objects_modified += 1
            geometries_removed += stats["geometries_removed"]
            rings_fixed += stats["rings_fixed"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    return {
        "objects_modified": objects_modified,
        "geometries_removed": geometries_removed,
        "rings_fixed": rings_fixed,
    }


def list_json_files(folder: Path):
    if not folder.exists():
        return []
    return sorted(folder.glob("*.json"))


def choose_index(n: int, prompt: str):
    choice = input(prompt).strip()
    if not choice.isdigit():
        return None
    idx = int(choice)
    if idx < 0 or idx > n:
        return None
    return idx


def main():
    # CLI mode
    if len(sys.argv) >= 3:
        in_path = Path(sys.argv[1])
        out_path = Path(sys.argv[2])

        print("\n" + "=" * 70)
        print("LOD2 CityJSON Geometry Fixer (CLI MODE)")
        print("=" * 70)
        print(f"Input:  {in_path}")
        print(f"Output: {out_path}")

        if not in_path.exists():
            print(f"[ERROR] Input file not found: {in_path}")
            sys.exit(1)

        try:
            stats = fix_cityjson_file(in_path, out_path)
        except Exception as e:
            print(f"[ERROR] Fix failed: {e}")
            sys.exit(1)

        print("Done.")
        print(f"  Objects modified:           {stats['objects_modified']}")
        print(f"  Empty geometries removed:   {stats['geometries_removed']}")
        print(f"  Vertex drops for 102 fixed: {stats['rings_fixed']}")
        print("=" * 70 + "\n")
        return

    # Interactive mode
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR

    print("\n" + "=" * 70)
    print("LOD2 CityJSON Geometry Fixer (INTERACTIVE MODE)")
    print("=" * 70)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")

    files = list_json_files(input_dir)
    if not files:
        print(f"[ERROR] No JSON files found in: {input_dir}")
        sys.exit(1)

    print(f"\nFound {len(files)} JSON file(s):\n")
    for i, p in enumerate(files):
        size_kb = p.stat().st_size / 1024.0
        print(f"  [{i}] {p.name:<35} ({size_kb:,.1f} KB)")
    print(f"  [{len(files)}] Process ALL files")
    print("  [99] Exit\n")

    idx = choose_index(len(files), f"Select file to fix [0-{len(files)}] (or 99): ")
    if idx is None:
        print("[ERROR] Invalid selection.")
        sys.exit(1)

    if idx == 99:
        return

    targets = files if idx == len(files) else [files[idx]]

    totals = {
        "objects_modified": 0,
        "geometries_removed": 0,
        "rings_fixed": 0,
    }

    for p in targets:
        out_name = f"{p.stem}_FIXED.json"
        out_path = output_dir / out_name

        print(f"\nProcessing: {p.name}")
        try:
            stats = fix_cityjson_file(p, out_path)
        except Exception as e:
            print(f"[ERROR] Failed on {p.name}: {e}")
            continue

        for k in totals:
            totals[k] += stats[k]

        print(f"  Saved to: {out_path}")
        print("  Summary:")
        print(f"     Objects modified:           {stats['objects_modified']}")
        print(f"     Empty geometries removed:   {stats['geometries_removed']}")
        print(f"     Vertex drops for 102 fixed: {stats['rings_fixed']}")

    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Files processed:            {len(targets)}/{len(targets)}")
    print(f"Objects modified:           {totals['objects_modified']}")
    print(f"Empty geometries removed:   {totals['geometries_removed']}")
    print(f"Vertex drops for 102 fixed: {totals['rings_fixed']}")
    print("=" * 70)
    print(f"Check: {output_dir}")
    print("Done.\n")


if __name__ == "__main__":
    main()
