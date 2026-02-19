# 05_json_fix.py
"""
LOD2 CityJSON Geometry Fixer (PIPELINE-FRIENDLY)

Fixes (report-driven):
- 902 EMPTY_PRIMITIVE: removes geometry entries with empty boundaries
- 102 CONSECUTIVE_POINTS_SAME: targeted cleanup only on val3dity-flagged faces
- 204 NON_PLANAR_POLYGON_NORMALS_DEVIATION: targeted face projection to a fitted plane
- 307 POLYGON_WRONG_ORIENTATION: targeted orientation flip on val3dity-flagged faces

Behavior:
- Default INPUT dir:  <project_root>/data/03_json_model
- Default OUTPUT dir: <project_root>/outputs/04_LOD2_json
- Default REPORT dir: <project_root>/outputs/03_val3dity
- Lists JSON files in INPUT dir and asks user which one to process (or ALL)
- Writes: <stem>_FIXED.json into OUTPUT dir

CLI mode:
  python 05_json_fix.py <input_json> <output_json> [--report <report_json>] [--tol 0.001]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io_helpers import choose_index, list_json_files
from utils.val3dity_102 import apply_102_fix_from_report
from utils.val3dity_104 import apply_104_fix_from_report
from utils.val3dity_204 import apply_204_fix_from_report
from utils.val3dity_307 import apply_307_fix_from_report
from utils.val3dity_report import extract_error_codes, load_report_error_codes, load_report_json


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "03_json_model"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "04_LOD2_json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs" / "03_val3dity"
DEFAULT_TOL = 0.001


def _is_empty_boundaries(boundaries) -> bool:
    if boundaries is None:
        return True
    if not isinstance(boundaries, (list, tuple)):
        return True
    return len(boundaries) == 0


def _is_effectively_empty(node) -> bool:
    if node is None:
        return True
    if isinstance(node, list):
        if len(node) == 0:
            return True
        return all(_is_effectively_empty(child) for child in node)
    return False


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


def _report_planarity_d2p_tol(report_json: dict):
    try:
        tol = report_json.get("parameters", {}).get("planarity_d2p_tol", None)
        if tol is None:
            return None
        t = float(tol)
        if t > 0:
            return t
    except Exception:
        pass
    return None


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

        # 902: remove empty primitive (including nested empty after pruning)
        if _is_empty_boundaries(boundaries) or _is_effectively_empty(boundaries):
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


def fix_cityjson_file(input_path: Path, output_path: Path, report_json: dict, tol_override=None, target_code=None):
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    city_objects = data.get("CityObjects", {})
    if not isinstance(city_objects, dict):
        raise ValueError("Invalid CityJSON: CityObjects is not a dict")

    codes = extract_error_codes(report_json)
    fix_902_enabled = 902 in codes
    fix_102_enabled = 102 in codes
    fix_104_enabled = 104 in codes
    fix_204_enabled = 204 in codes
    fix_307_enabled = 307 in codes

    if target_code is not None:
        fix_902_enabled = fix_902_enabled and target_code == 902
        fix_102_enabled = fix_102_enabled and target_code == 102
        fix_104_enabled = fix_104_enabled and target_code == 104
        fix_204_enabled = fix_204_enabled and target_code == 204
        fix_307_enabled = fix_307_enabled and target_code == 307

    selected_code = target_code if target_code is not None else None

    objects_modified = 0
    geometries_removed = 0
    fix102_stats = {
        "targets_total": 0,
        "targets_resolved": 0,
        "targets_missing": 0,
        "targets_unresolved": 0,
        "objects_modified": 0,
        "consecutive_removed": 0,
        "rings_nudged": 0,
        "new_vertices_added": 0,
        "rings_dropped": 0,
        "faces_dropped": 0,
    }
    fix104_stats = {
        "targets_total": 0,
        "targets_resolved": 0,
        "targets_missing": 0,
        "targets_unresolved": 0,
        "objects_modified": 0,
        "rings_untangled": 0,
    }
    fix204_stats = {
        "targets_total": 0,
        "targets_resolved": 0,
        "targets_missing": 0,
        "targets_unresolved": 0,
        "objects_modified": 0,
        "faces_projected": 0,
        "face_vertices_cloned": 0,
        "vertices_moved": 0,
        "max_displacement": 0.0,
        "faces_skipped_large_move": 0,
    }
    fix307_stats = {
        "targets_total": 0,
        "targets_resolved": 0,
        "targets_missing": 0,
        "targets_unresolved": 0,
        "objects_modified": 0,
        "faces_flipped": 0,
    }

    tol = tol_override if tol_override is not None else _report_snap_tol(report_json)
    if tol is None:
        tol = DEFAULT_TOL

    if fix_102_enabled:
        fix102_stats = apply_102_fix_from_report(data, report_json, tol=tol)
    if fix_104_enabled:
        fix104_stats = apply_104_fix_from_report(data, report_json, tol=tol)

    max_move_204 = _report_planarity_d2p_tol(report_json)
    if max_move_204 is None:
        max_move_204 = 0.01
    if fix_204_enabled:
        fix204_stats = apply_204_fix_from_report(data, report_json, max_move=max_move_204)
    if fix_307_enabled:
        fix307_stats = apply_307_fix_from_report(data, report_json)

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
        "fix_102_enabled": fix_102_enabled,
        "fix_104_enabled": fix_104_enabled,
        "fix_204_enabled": fix_204_enabled,
        "fix_307_enabled": fix_307_enabled,
        "selected_code": selected_code,
        "tol_used": tol,
        "max_move_204": max_move_204,
        "fix102": fix102_stats,
        "fix104": fix104_stats,
        "fix204": fix204_stats,
        "fix307": fix307_stats,
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
            if target_code not in (102, 104, 204, 307, 902):
                raise ValueError("--target-code must be one of: 102, 104, 204, 307, 902")
            i += 2
            continue
        raise ValueError(f"Unknown argument: {arg}")

    return report_path, tol_override, target_code


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

        if not in_path.exists():
            print(f"[ERROR] Input file not found: {in_path}")
            sys.exit(1)

        try:
            report_json = load_report_json(report_path)
            codes = load_report_error_codes(report_path)
        except Exception as e:
            print(f"[ERROR] Could not load val3dity report: {e}")
            sys.exit(1)

        try:
            stats = fix_cityjson_file(
                in_path,
                out_path,
                report_json=report_json,
                tol_override=tol_override,
                target_code=target_code,
            )
        except Exception as e:
            print(f"[ERROR] Fix failed: {e}")
            sys.exit(1)

        applied = []
        if stats["fix_902_enabled"]:
            applied.append("902")
        if stats["fix_102_enabled"]:
            applied.append("102")
        if stats["fix_104_enabled"]:
            applied.append("104")
        if stats["fix_204_enabled"]:
            applied.append("204")
        if stats["fix_307_enabled"]:
            applied.append("307")
        applied_text = ", ".join(applied) if applied else "none"
        print(f"Output: {out_path}")
        print(f"Applied fix: {applied_text}")
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

    files = [Path(p) for p in list_json_files(input_dir)]
    if not files:
        print(f"[ERROR] No JSON files found in: {input_dir}")
        sys.exit(1)

    print(f"\nFound {len(files)} JSON file(s):\n")
    for i, p in enumerate(files):
        size_kb = p.stat().st_size / 1024.0
        print(f"  [{i}] {p.name:<35} ({size_kb:,.1f} KB)")
    print(f"  [{len(files)}] Process ALL files")
    print("  [99] Exit\n")

    idx = choose_index(
        len(files),
        f"Select file to fix [0-{len(files)}] (or 99): ",
        max_index=len(files),
        allowed_values={99},
    )
    if idx is None:
        print("[ERROR] Invalid selection.")
        sys.exit(1)

    if idx == 99:
        return

    targets = files if idx == len(files) else [files[idx]]

    totals = {
        "objects_modified": 0,
        "geometries_removed": 0,
        "targets_total": 0,
        "targets_resolved": 0,
        "targets_unresolved": 0,
        "consecutive_removed": 0,
        "rings_nudged": 0,
        "new_vertices_added": 0,
        "faces_dropped": 0,
        "targets204_total": 0,
        "targets204_resolved": 0,
        "targets204_unresolved": 0,
        "faces204_projected": 0,
        "face204_vertices_cloned": 0,
        "vertices204_moved": 0,
        "max204_displacement": 0.0,
        "faces204_skipped_large_move": 0,
        "targets307_total": 0,
        "targets307_resolved": 0,
        "targets307_unresolved": 0,
        "faces307_flipped": 0,
    }

    for p in targets:
        out_name = f"{p.stem}_FIXED.json"
        out_path = output_dir / out_name
        report_path = _default_report_path_for_input(p)

        print(f"\nProcessing: {p.name}")
        print(f"  Using report: {report_path.name}")

        try:
            report_json = load_report_json(report_path)
            codes = load_report_error_codes(report_path)
        except Exception as e:
            print(f"[ERROR] Missing/invalid report for {p.name}: {e}")
            continue

        try:
            stats = fix_cityjson_file(p, out_path, report_json=report_json, tol_override=None)
        except Exception as e:
            print(f"[ERROR] Failed on {p.name}: {e}")
            continue

        totals["objects_modified"] += stats["objects_modified"]
        totals["geometries_removed"] += stats["geometries_removed"]
        totals["targets_total"] += stats["fix102"]["targets_total"]
        totals["targets_resolved"] += stats["fix102"]["targets_resolved"]
        totals["targets_unresolved"] += stats["fix102"]["targets_unresolved"]
        totals["consecutive_removed"] += stats["fix102"]["consecutive_removed"]
        totals["rings_nudged"] += stats["fix102"]["rings_nudged"]
        totals["new_vertices_added"] += stats["fix102"]["new_vertices_added"]
        totals["faces_dropped"] += stats["fix102"]["faces_dropped"]
        totals["targets204_total"] += stats["fix204"]["targets_total"]
        totals["targets204_resolved"] += stats["fix204"]["targets_resolved"]
        totals["targets204_unresolved"] += stats["fix204"]["targets_unresolved"]
        totals["faces204_projected"] += stats["fix204"]["faces_projected"]
        totals["face204_vertices_cloned"] += stats["fix204"]["face_vertices_cloned"]
        totals["vertices204_moved"] += stats["fix204"]["vertices_moved"]
        totals["faces204_skipped_large_move"] += stats["fix204"]["faces_skipped_large_move"]
        if stats["fix204"]["max_displacement"] > totals["max204_displacement"]:
            totals["max204_displacement"] = stats["fix204"]["max_displacement"]
        totals["targets307_total"] += stats["fix307"]["targets_total"]
        totals["targets307_resolved"] += stats["fix307"]["targets_resolved"]
        totals["targets307_unresolved"] += stats["fix307"]["targets_unresolved"]
        totals["faces307_flipped"] += stats["fix307"]["faces_flipped"]

        print(f"  Saved to: {out_path}")
        print("  Summary:")
        codes_txt = ", ".join(str(c) for c in sorted(codes)) if codes else "none"
        print(f"     val3dity codes found:      {codes_txt}")
        print(f"     Applied fix 902:           {'yes' if stats['fix_902_enabled'] else 'no'}")
        print(f"     Applied fix 102:           {'yes' if stats['fix_102_enabled'] else 'no'}")
        print(f"     Applied fix 104:           {'yes' if stats['fix_104_enabled'] else 'no'}")
        print(f"     Applied fix 204:           {'yes' if stats['fix_204_enabled'] else 'no'}")
        print(f"     Applied fix 307:           {'yes' if stats['fix_307_enabled'] else 'no'}")
        print(f"     Snap tol used for 102:     {stats['tol_used']}")
        print(f"     Max move used for 204:     {stats['max_move_204']}")
        print(f"     Objects modified:         {stats['objects_modified']}")
        print(f"     Empty geometries removed: {stats['geometries_removed']}")
        print(
            f"     102 targets resolved:      "
            f"{stats['fix102']['targets_resolved']}/{stats['fix102']['targets_total']}"
        )
        print(f"     102 targets unresolved:    {stats['fix102']['targets_unresolved']}")
        print(f"     102 consecutive removed:   {stats['fix102']['consecutive_removed']}")
        print(f"     102 rings nudged:          {stats['fix102']['rings_nudged']}")
        print(f"     102 new vertices added:    {stats['fix102']['new_vertices_added']}")
        print(f"     102 faces dropped:         {stats['fix102']['faces_dropped']} (expected 0)")
        print(
            f"     204 targets resolved:      "
            f"{stats['fix204']['targets_resolved']}/{stats['fix204']['targets_total']}"
        )
        print(f"     204 targets unresolved:    {stats['fix204']['targets_unresolved']}")
        print(f"     204 faces projected:       {stats['fix204']['faces_projected']}")
        print(f"     204 face vertices cloned:  {stats['fix204']['face_vertices_cloned']}")
        print(f"     204 vertices moved:        {stats['fix204']['vertices_moved']}")
        print(f"     204 max displacement:      {stats['fix204']['max_displacement']}")
        print(f"     204 faces skipped (move):  {stats['fix204']['faces_skipped_large_move']}")
        print(
            f"     307 targets resolved:      "
            f"{stats['fix307']['targets_resolved']}/{stats['fix307']['targets_total']}"
        )
        print(f"     307 targets unresolved:    {stats['fix307']['targets_unresolved']}")
        print(f"     307 faces flipped:         {stats['fix307']['faces_flipped']}")

    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Files processed:          {len(targets)}/{len(targets)}")
    print(f"Objects modified:         {totals['objects_modified']}")
    print(f"Empty geometries removed: {totals['geometries_removed']}")
    print(f"102 targets resolved:      {totals['targets_resolved']}/{totals['targets_total']}")
    print(f"102 targets unresolved:    {totals['targets_unresolved']}")
    print(f"102 consecutive removed:   {totals['consecutive_removed']}")
    print(f"102 rings nudged:          {totals['rings_nudged']}")
    print(f"102 new vertices added:    {totals['new_vertices_added']}")
    print(f"102 faces dropped:         {totals['faces_dropped']} (expected 0)")
    print(f"204 targets resolved:      {totals['targets204_resolved']}/{totals['targets204_total']}")
    print(f"204 targets unresolved:    {totals['targets204_unresolved']}")
    print(f"204 faces projected:       {totals['faces204_projected']}")
    print(f"204 face vertices cloned:  {totals['face204_vertices_cloned']}")
    print(f"204 vertices moved:        {totals['vertices204_moved']}")
    print(f"204 max displacement:      {totals['max204_displacement']}")
    print(f"204 faces skipped (move):  {totals['faces204_skipped_large_move']}")
    print(f"307 targets resolved:      {totals['targets307_resolved']}/{totals['targets307_total']}")
    print(f"307 targets unresolved:    {totals['targets307_unresolved']}")
    print(f"307 faces flipped:         {totals['faces307_flipped']}")
    print("=" * 70)
    print(f"Check: {output_dir}")
    print("Done.\n")


if __name__ == "__main__":
    main()
