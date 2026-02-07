# 04_json_fix.py

""""
LOD2 CityJSON Geometry Fixer (PIPELINE-FRIENDLY)

Fixes:
- 902 EMPTY_PRIMITIVE: removes geometry entries with empty boundaries []
- 102 CONSECUTIVE_POINTS_SAME: removes consecutive vertices that have identical XYZ coords
  (IMPORTANT: this can happen even if vertex indices differ)

Behavior:
- Default INPUT dir:  <project_root>/data/03_json_model
- Default OUTPUT dir: <project_root>/outputs/04_LOD2_json
- Lists JSON files in INPUT dir and asks user which one to process (or ALL)
- Writes: <stem>_FIXED.json into OUTPUT dir

Also supports non-interactive CLI:
  python 04_json_fix.py <input_json> <output_json>
"""

import sys
import json
from pathlib import Path

# -------------------------
# Path defaults (relative to repo)
# -------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PROJECT_ROOT.parent

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "03_json_model"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "04_LOD2_json"

# -------------------------
# Helpers
# -------------------------
EPS = 1e-12  # tiny tolerance for float equality


def _is_empty_boundaries(boundaries) -> bool:
    if boundaries is None:
        return True
    if not isinstance(boundaries, (list, tuple)):
        return True
    return len(boundaries) == 0


def _is_ring(node):
    return isinstance(node, list) and len(node) >= 2 and all(isinstance(v, int) for v in node)


def _coord_key(vertices, vidx):
    """Return a comparable XYZ key for a vertex index."""
    try:
        v = vertices[vidx]
        # CityJSON vertices are usually [x,y,z] floats
        x, y, z = float(v[0]), float(v[1]), float(v[2])
        return (x, y, z)
    except Exception:
        return None


def _same_xyz(a, b):
    if a is None or b is None:
        return False
    return (abs(a[0] - b[0]) <= EPS) and (abs(a[1] - b[1]) <= EPS) and (abs(a[2] - b[2]) <= EPS)


def _clean_ring_by_xyz(ring, vertices):
    """
    Remove consecutive points with the same XYZ coordinates (val3dity Error 102),
    even if their vertex indices differ.
    Returns: (cleaned_ring, fixed_count)
    """
    if not ring:
        return ring, 0

    cleaned = []
    fixed = 0

    prev_xyz = None
    for vidx in ring:
        xyz = _coord_key(vertices, vidx)
        if prev_xyz is not None and _same_xyz(xyz, prev_xyz):
            # consecutive same point in space -> drop this vertex
            fixed += 1
            continue
        cleaned.append(vidx)
        prev_xyz = xyz

    return cleaned, fixed


def _clean_boundaries_recursive(boundaries, vertices):
    """
    Recursively traverse CityJSON 'boundaries' and:
      - for rings (list[int]), remove consecutive same-XYZ vertices (Error 102)
      - drop degenerate rings (<3 unique vertex indices OR <3 unique XYZ)
      - drop empty containers created by dropping children
    Returns:
      (new_boundaries, rings_fixed, rings_dropped, faces_dropped)
    """
    rings_fixed = 0
    rings_dropped = 0
    faces_dropped = 0

    def ring_unique_xyz_count(r):
        keys = []
        for vi in r:
            k = _coord_key(vertices, vi)
            if k is not None:
                keys.append(k)
        # use exact keys; EPS already used in consecutive check
        return len(set(keys))

    def walk(node, depth=0):
        nonlocal rings_fixed, rings_dropped, faces_dropped

        if _is_ring(node):
            cleaned, fixed = _clean_ring_by_xyz(node, vertices)
            if fixed > 0:
                rings_fixed += fixed

            # Drop degenerate ring:
            # - fewer than 3 unique vertex indices OR fewer than 3 unique XYZ
            if len(set(cleaned)) < 3 or ring_unique_xyz_count(cleaned) < 3:
                rings_dropped += 1
                return None

            return cleaned

        if isinstance(node, list):
            new_list = []
            for child in node:
                cc = walk(child, depth + 1)
                if cc is None:
                    continue
                new_list.append(cc)

            if len(new_list) == 0:
                if depth > 0:
                    faces_dropped += 1
                return None

            return new_list

        return node

    newb = walk(boundaries, depth=0)
    if newb is None:
        newb = []
    return newb, rings_fixed, rings_dropped, faces_dropped


def _fix_cityobject_geometries(cityobj: dict, vertices):
    """
    For each geometry:
      - clean rings by XYZ (Error 102)
      - remove empties (Error 902)
    Returns stats dict.
    """
    geoms = cityobj.get("geometry", None)
    if not geoms or not isinstance(geoms, list):
        return {
            "geometries_removed": 0,
            "rings_fixed": 0,
            "rings_dropped": 0,
            "faces_dropped": 0,
            "geometries_touched": 0,
        }

    keep = []
    geometries_removed = 0
    geometries_touched = 0
    rings_fixed_total = 0
    rings_dropped_total = 0
    faces_dropped_total = 0

    for g in geoms:
        if not isinstance(g, dict):
            keep.append(g)
            continue

        boundaries = g.get("boundaries", None)

        # If already empty, remove via 902 logic
        if _is_empty_boundaries(boundaries):
            geometries_removed += 1
            continue

        newb, rf, rd, fd = _clean_boundaries_recursive(boundaries, vertices)
        g["boundaries"] = newb

        if rf or rd or fd:
            geometries_touched += 1
            rings_fixed_total += rf
            rings_dropped_total += rd
            faces_dropped_total += fd

        # After cleaning, might become empty -> remove (902)
        if _is_empty_boundaries(newb):
            geometries_removed += 1
            continue

        keep.append(g)

    if geometries_removed > 0 or geometries_touched > 0:
        cityobj["geometry"] = keep

    return {
        "geometries_removed": geometries_removed,
        "rings_fixed": rings_fixed_total,
        "rings_dropped": rings_dropped_total,
        "faces_dropped": faces_dropped_total,
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
    rings_dropped = 0
    faces_dropped = 0

    for obj_id, obj in city_objects.items():
        if not isinstance(obj, dict):
            continue

        stats = _fix_cityobject_geometries(obj, vertices)

        changed = stats["geometries_removed"] > 0 or stats["geometries_touched"] > 0
        if changed:
            objects_modified += 1
            geometries_removed += stats["geometries_removed"]
            rings_fixed += stats["rings_fixed"]
            rings_dropped += stats["rings_dropped"]
            faces_dropped += stats["faces_dropped"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    return {
        "objects_modified": objects_modified,
        "geometries_removed": geometries_removed,
        "rings_fixed": rings_fixed,
        "rings_dropped": rings_dropped,
        "faces_dropped": faces_dropped,
    }


# -------------------------
# UI helpers
# -------------------------
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


# -------------------------
# Main
# -------------------------
def main():
    # CLI mode
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
            stats = fix_cityjson_file(in_path, out_path)
        except Exception as e:
            print(f"[ERROR] Fix failed: {e}")
            sys.exit(1)

        print("✅ Done.")
        print(f"  Objects modified:           {stats['objects_modified']}")
        print(f"  Empty geometries removed:   {stats['geometries_removed']}")
        print(f"  Vertex drops for 102 fixed: {stats['rings_fixed']}")
        print(f"  Rings dropped (degenerate): {stats['rings_dropped']}")
        print(f"  Faces/surfaces dropped:     {stats['faces_dropped']}")
        print("=" * 70 + "\n")
        return

    # Interactive mode
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

    totals = {
        "objects_modified": 0,
        "geometries_removed": 0,
        "rings_fixed": 0,
        "rings_dropped": 0,
        "faces_dropped": 0,
    }

    for p in targets:
        out_name = f"{p.stem}_FIXED.json"
        out_path = output_dir / out_name

        print(f"\n📄 Processing: {p.name}")
        try:
            stats = fix_cityjson_file(p, out_path)
        except Exception as e:
            print(f"[ERROR] Failed on {p.name}: {e}")
            continue

        for k in totals:
            totals[k] += stats[k]

        print(f"  ✅ Saved to: {out_path}")
        print(f"  📊 Summary:")
        print(f"     Objects modified:           {stats['objects_modified']}")
        print(f"     Empty geometries removed:   {stats['geometries_removed']}")
        print(f"     Vertex drops for 102 fixed: {stats['rings_fixed']}")
        print(f"     Rings dropped (degenerate): {stats['rings_dropped']}")
        print(f"     Faces/surfaces dropped:     {stats['faces_dropped']}")

    print("\n" + "=" * 70)
    print("📊 PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Files processed:            {len(targets)}/{len(targets)}")
    print(f"Objects modified:           {totals['objects_modified']}")
    print(f"Empty geometries removed:   {totals['geometries_removed']}")
    print(f"Vertex drops for 102 fixed: {totals['rings_fixed']}")
    print(f"Rings dropped (degenerate): {totals['rings_dropped']}")
    print(f"Faces/surfaces dropped:     {totals['faces_dropped']}")
    print("=" * 70)
    print(f"📂 Check: {output_dir}")
    print("✨ Done!\n")


if __name__ == "__main__":
    main()
