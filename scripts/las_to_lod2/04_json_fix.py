# 04_json_fix.py
"""
LOD2 CityJSON Geometry Fixer (PIPELINE-FRIENDLY)

Fixes only:
- 902 EMPTY_PRIMITIVE: removes geometry entries with empty boundaries []

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
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs" / "03_val3dity"


def _is_empty_boundaries(boundaries) -> bool:
    if boundaries is None:
        return True
    if not isinstance(boundaries, (list, tuple)):
        return True
    return len(boundaries) == 0


def _extract_error_codes_from_node(node, out_codes: set):
    if isinstance(node, dict):
        for k, v in node.items():
            kl = str(k).lower()
            if "error" in kl or "code" in kl:
                if isinstance(v, int):
                    out_codes.add(v)
                elif isinstance(v, str) and v.strip().isdigit():
                    out_codes.add(int(v.strip()))
            _extract_error_codes_from_node(v, out_codes)
        return

    if isinstance(node, list):
        for item in node:
            _extract_error_codes_from_node(item, out_codes)


def _load_report_error_codes(report_json_path: Path):
    if not report_json_path.exists():
        raise FileNotFoundError(f"val3dity report not found: {report_json_path}")

    try:
        data = json.loads(report_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Could not parse val3dity report JSON: {e}") from e

    codes = set()
    _extract_error_codes_from_node(data, codes)
    return codes


def _default_report_path_for_input(input_path: Path):
    return DEFAULT_REPORT_DIR / f"{input_path.stem}_val3dity.json"


def _fix_cityobject_geometries(cityobj: dict):
    geoms = cityobj.get("geometry", None)
    if not geoms or not isinstance(geoms, list):
        return {
            "geometries_removed": 0,
            "geometries_touched": 0,
        }

    keep = []
    geometries_removed = 0

    for g in geoms:
        if not isinstance(g, dict):
            keep.append(g)
            continue

        boundaries = g.get("boundaries", None)

        # 902: remove empty primitive
        if _is_empty_boundaries(boundaries):
            geometries_removed += 1
            continue

        keep.append(g)

    geometries_touched = 1 if geometries_removed > 0 else 0
    if geometries_touched > 0:
        cityobj["geometry"] = keep

    return {
        "geometries_removed": geometries_removed,
        "geometries_touched": geometries_touched,
    }


def fix_cityjson_file(input_path: Path, output_path: Path, fix_902_enabled: bool):
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    city_objects = data.get("CityObjects", {})
    if not isinstance(city_objects, dict):
        raise ValueError("Invalid CityJSON: CityObjects is not a dict")

    objects_modified = 0
    geometries_removed = 0

    for _, obj in city_objects.items():
        if not isinstance(obj, dict):
            continue

        if fix_902_enabled:
            stats = _fix_cityobject_geometries(obj)
        else:
            stats = {"geometries_removed": 0, "geometries_touched": 0}

        changed = stats["geometries_removed"] > 0 or stats["geometries_touched"] > 0
        if changed:
            objects_modified += 1
            geometries_removed += stats["geometries_removed"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    return {
        "objects_modified": objects_modified,
        "geometries_removed": geometries_removed,
        "fix_902_enabled": fix_902_enabled,
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
        report_path = _default_report_path_for_input(in_path)

        if len(sys.argv) >= 5:
            if sys.argv[3] == "--report":
                report_path = Path(sys.argv[4])
            else:
                print("[ERROR] Unknown arguments.")
                print("Usage: python 04_json_fix.py <input_json> <output_json> [--report <report_json>]")
                sys.exit(1)
        elif len(sys.argv) == 4:
            print("[ERROR] Missing value for --report")
            print("Usage: python 04_json_fix.py <input_json> <output_json> [--report <report_json>]")
            sys.exit(1)

        print("\n" + "=" * 70)
        print("LOD2 CityJSON Geometry Fixer (CLI MODE)")
        print("=" * 70)
        print(f"Input:  {in_path}")
        print(f"Output: {out_path}")
        print(f"Report: {report_path}")

        if not in_path.exists():
            print(f"[ERROR] Input file not found: {in_path}")
            sys.exit(1)

        try:
            codes = _load_report_error_codes(report_path)
        except Exception as e:
            print(f"[ERROR] Could not load val3dity report: {e}")
            sys.exit(1)

        fix_902_enabled = 902 in codes

        try:
            stats = fix_cityjson_file(in_path, out_path, fix_902_enabled=fix_902_enabled)
        except Exception as e:
            print(f"[ERROR] Fix failed: {e}")
            sys.exit(1)

        codes_txt = ", ".join(str(c) for c in sorted(codes)) if codes else "none"
        print("Done.")
        print(f"  val3dity codes found:      {codes_txt}")
        print(f"  Applied fix 902:           {'yes' if fix_902_enabled else 'no'}")
        print(f"  Objects modified:         {stats['objects_modified']}")
        print(f"  Empty geometries removed: {stats['geometries_removed']}")
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
    print(f"Report directory: {DEFAULT_REPORT_DIR}")

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
    }

    for p in targets:
        out_name = f"{p.stem}_FIXED.json"
        out_path = output_dir / out_name
        report_path = _default_report_path_for_input(p)

        print(f"\nProcessing: {p.name}")
        print(f"  Using report: {report_path.name}")

        try:
            codes = _load_report_error_codes(report_path)
        except Exception as e:
            print(f"[ERROR] Missing/invalid report for {p.name}: {e}")
            continue

        fix_902_enabled = 902 in codes
        try:
            stats = fix_cityjson_file(p, out_path, fix_902_enabled=fix_902_enabled)
        except Exception as e:
            print(f"[ERROR] Failed on {p.name}: {e}")
            continue

        for k in totals:
            totals[k] += stats[k]

        print(f"  Saved to: {out_path}")
        print("  Summary:")
        codes_txt = ", ".join(str(c) for c in sorted(codes)) if codes else "none"
        print(f"     val3dity codes found:      {codes_txt}")
        print(f"     Applied fix 902:           {'yes' if fix_902_enabled else 'no'}")
        print(f"     Objects modified:         {stats['objects_modified']}")
        print(f"     Empty geometries removed: {stats['geometries_removed']}")

    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Files processed:          {len(targets)}/{len(targets)}")
    print(f"Objects modified:         {totals['objects_modified']}")
    print(f"Empty geometries removed: {totals['geometries_removed']}")
    print("=" * 70)
    print(f"Check: {output_dir}")
    print("Done.\n")


if __name__ == "__main__":
    main()
