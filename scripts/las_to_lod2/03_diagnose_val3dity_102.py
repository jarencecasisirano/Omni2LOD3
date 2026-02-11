"""
03_diagnose_val3dity_102.py

Purpose:
- Diagnose where val3dity error 102 (CONSECUTIVE_POINTS_SAME) is flagged.
- Map each reported id (coid|geom|shell|face...) back to CityJSON boundaries.
- Print consecutive vertex pairs per ring and whether they collapse under snap tolerance.

Defaults:
- Input:  <project_root>/data/03_json_model/nec_021126.json
- Report: <project_root>/outputs/03_val3dity/<input_stem>_val3dity.json
- Tolerance: report.parameters.snap_tol if available, otherwise 0.001

Usage:
  python 03_diagnose_val3dity_102.py
  python 03_diagnose_val3dity_102.py --input <cityjson> [--report <report_json>] [--tol 0.001]
"""

import json
import math
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_INPUT = PROJECT_ROOT / "data" / "03_json_model" / "nec_021126.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs" / "03_val3dity"
DEFAULT_TOL = 0.001


def _parse_args(argv):
    input_path = DEFAULT_INPUT
    report_path = None
    tol = None

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--input":
            if i + 1 >= len(argv):
                raise ValueError("Missing value for --input")
            input_path = Path(argv[i + 1])
            i += 2
            continue
        if arg == "--report":
            if i + 1 >= len(argv):
                raise ValueError("Missing value for --report")
            report_path = Path(argv[i + 1])
            i += 2
            continue
        if arg == "--tol":
            if i + 1 >= len(argv):
                raise ValueError("Missing value for --tol")
            tol = float(argv[i + 1])
            i += 2
            continue
        raise ValueError(f"Unknown argument: {arg}")

    if report_path is None:
        report_path = DEFAULT_REPORT_DIR / f"{input_path.stem}_val3dity.json"

    return input_path, report_path, tol


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _kv_id_parse(id_text: str):
    out = {}
    for part in id_text.split("|"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _as_int_or_none(v):
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _world_xyz(vertices, vidx, transform):
    v = vertices[vidx]
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    if not transform:
        return (x, y, z)
    scale = transform.get("scale", [1.0, 1.0, 1.0])
    trans = transform.get("translate", [0.0, 0.0, 0.0])
    return (
        x * float(scale[0]) + float(trans[0]),
        y * float(scale[1]) + float(trans[1]),
        z * float(scale[2]) + float(trans[2]),
    )


def _snap_key(p, tol):
    if tol <= 0:
        return p
    return (
        int(round(p[0] / tol)),
        int(round(p[1] / tol)),
        int(round(p[2] / tol)),
    )


def _distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz), abs(dx), abs(dy), abs(dz)


def _normalize_face_to_rings(face_node):
    # CityJSON face can be [ring] or [[ring], [hole]...], depending on exporter style.
    if not isinstance(face_node, list):
        return []
    if len(face_node) == 0:
        return []
    if all(isinstance(v, int) for v in face_node):
        return [face_node]
    return [r for r in face_node if isinstance(r, list)]


def _get_rings_for_error(cityobj, geom_idx, shell_idx, face_idx):
    geoms = cityobj.get("geometry", [])
    if geom_idx is None or geom_idx < 0 or geom_idx >= len(geoms):
        return None, "geom index out of range"
    geom = geoms[geom_idx]
    boundaries = geom.get("boundaries")
    gtype = geom.get("type", "")

    try:
        if gtype in ("Solid", "CompositeSolid"):
            if shell_idx is None:
                return None, "missing shell index for Solid-like geometry"
            face_node = boundaries[shell_idx][face_idx]
            return _normalize_face_to_rings(face_node), None

        if gtype in ("MultiSurface", "CompositeSurface"):
            face_node = boundaries[face_idx]
            return _normalize_face_to_rings(face_node), None

        # Fallback for unknown geometry styles: try Solid-like first, then MultiSurface-like.
        if shell_idx is not None:
            face_node = boundaries[shell_idx][face_idx]
            return _normalize_face_to_rings(face_node), None
        face_node = boundaries[face_idx]
        return _normalize_face_to_rings(face_node), None
    except Exception as e:
        return None, f"could not access boundaries: {e}"


def _extract_102_errors(report_json):
    features = report_json.get("features", [])
    out = []
    for feature in features:
        for err in feature.get("errors", []):
            if err.get("code") == 102:
                out.append(err)
    return out


def main():
    try:
        input_path, report_path, tol_arg = _parse_args(sys.argv[1:])
    except Exception as e:
        print(f"[ERROR] {e}")
        print("Usage: python 03_diagnose_val3dity_102.py --input <cityjson> [--report <report_json>] [--tol 0.001]")
        sys.exit(1)

    if not input_path.exists():
        print(f"[ERROR] Input CityJSON not found: {input_path}")
        sys.exit(1)
    if not report_path.exists():
        print(f"[ERROR] val3dity report not found: {report_path}")
        sys.exit(1)

    city = _load_json(input_path)
    report = _load_json(report_path)

    report_snap_tol = report.get("parameters", {}).get("snap_tol")
    tol = float(tol_arg) if tol_arg is not None else float(report_snap_tol if report_snap_tol is not None else DEFAULT_TOL)
    if tol <= 0:
        print("[ERROR] Tolerance must be > 0")
        sys.exit(1)

    vertices = city.get("vertices", [])
    transform = city.get("transform")
    city_objects = city.get("CityObjects", {})
    errors_102 = _extract_102_errors(report)

    print("=" * 70)
    print("VAL3DITY 102 DIAGNOSTIC")
    print("=" * 70)
    print(f"Input JSON:    {input_path}")
    print(f"Report JSON:   {report_path}")
    print(f"Snap tol used: {tol}")
    print(f"Error 102 count in report: {len(errors_102)}")

    if not errors_102:
        print("No error 102 entries found.")
        return

    total_pairs_flagged = 0

    for idx, err in enumerate(errors_102, 1):
        err_id = err.get("id", "")
        kv = _kv_id_parse(err_id)
        coid = kv.get("coid")
        geom_idx = _as_int_or_none(kv.get("geom"))
        shell_idx = _as_int_or_none(kv.get("shell"))
        face_idx = _as_int_or_none(kv.get("face"))

        print("\n" + "-" * 70)
        print(f"[{idx}] {err.get('description')}  id={err_id}")

        if coid not in city_objects:
            print(f"  [ERROR] CityObject '{coid}' not found in input JSON.")
            continue
        if face_idx is None:
            print("  [ERROR] face index missing in error id.")
            continue

        cityobj = city_objects[coid]
        rings, err_msg = _get_rings_for_error(cityobj, geom_idx, shell_idx, face_idx)
        if rings is None:
            print(f"  [ERROR] {err_msg}")
            continue
        if len(rings) == 0:
            print("  [WARN] Face resolved but has no rings.")
            continue

        print(f"  Resolved: CityObject={coid}, geom={geom_idx}, shell={shell_idx}, face={face_idx}, rings={len(rings)}")
        face_pairs = 0

        for ri, ring in enumerate(rings):
            if not isinstance(ring, list) or len(ring) < 2:
                continue

            ring_hits = []
            for i in range(1, len(ring)):
                a_idx = ring[i - 1]
                b_idx = ring[i]
                if not isinstance(a_idx, int) or not isinstance(b_idx, int):
                    continue
                if a_idx < 0 or b_idx < 0 or a_idx >= len(vertices) or b_idx >= len(vertices):
                    continue

                a = _world_xyz(vertices, a_idx, transform)
                b = _world_xyz(vertices, b_idx, transform)
                d3, dx, dy, dz = _distance(a, b)

                same_index = (a_idx == b_idx)
                same_xyz_exact = (a[0] == b[0] and a[1] == b[1] and a[2] == b[2])
                same_xyz_axis_tol = (dx <= tol and dy <= tol and dz <= tol)
                same_xyz_dist_tol = (d3 <= tol)
                same_snap_key = (_snap_key(a, tol) == _snap_key(b, tol))

                if same_index or same_xyz_exact or same_xyz_axis_tol or same_xyz_dist_tol or same_snap_key:
                    ring_hits.append(
                        {
                            "pair": (i - 1, i),
                            "vidx": (a_idx, b_idx),
                            "d3": d3,
                            "dx": dx,
                            "dy": dy,
                            "dz": dz,
                            "same_index": same_index,
                            "same_xyz_exact": same_xyz_exact,
                            "same_xyz_axis_tol": same_xyz_axis_tol,
                            "same_xyz_dist_tol": same_xyz_dist_tol,
                            "same_snap_key": same_snap_key,
                            "a": a,
                            "b": b,
                        }
                    )

            if ring_hits:
                print(f"  Ring {ri}: {len(ring_hits)} consecutive pair(s) collapse at tol={tol}")
                for h in ring_hits:
                    face_pairs += 1
                    pa, pb = h["pair"]
                    va, vb = h["vidx"]
                    print(
                        "    pair "
                        f"{pa}->{pb} (v{va}->v{vb}) "
                        f"d3={h['d3']:.9f}, dx={h['dx']:.9f}, dy={h['dy']:.9f}, dz={h['dz']:.9f} "
                        f"| idx={h['same_index']} exact={h['same_xyz_exact']} "
                        f"axis_tol={h['same_xyz_axis_tol']} dist_tol={h['same_xyz_dist_tol']} snap={h['same_snap_key']}"
                    )
                    print(f"      A={h['a']}")
                    print(f"      B={h['b']}")
            else:
                print(f"  Ring {ri}: no collapsing consecutive pair found with current checks.")

        total_pairs_flagged += face_pairs

    print("\n" + "=" * 70)
    print(f"Done. Candidate collapsing consecutive pairs found: {total_pairs_flagged}")
    print("=" * 70)


if __name__ == "__main__":
    main()
