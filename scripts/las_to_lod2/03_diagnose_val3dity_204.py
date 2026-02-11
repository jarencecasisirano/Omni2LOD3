"""
03_diagnose_val3dity_204.py

Purpose:
- Diagnose val3dity error 204 (NON_PLANAR_POLYGON_NORMALS_DEVIATION).
- Map each reported id (coid|geom|shell|face...) back to CityJSON boundaries.
- Print per-face geometric diagnostics to decide whether to:
  1) relax val3dity normal tolerance, or
  2) apply a targeted geometry repair.

Defaults:
- Input:  <project_root>/outputs/04_LOD2_json/nec_021126_FIXED.json
- Report: <project_root>/outputs/03_val3dity/<input_stem>_val3dity.json

Usage:
  python 03_diagnose_val3dity_204.py
  python 03_diagnose_val3dity_204.py --input <cityjson> [--report <report_json>]
"""

import json
import math
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "04_LOD2_json" / "nec_021126_FIXED.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs" / "03_val3dity"


def _parse_args(argv):
    input_path = DEFAULT_INPUT
    report_path = None

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
        raise ValueError(f"Unknown argument: {arg}")

    if report_path is None:
        report_path = DEFAULT_REPORT_DIR / f"{input_path.stem}_val3dity.json"

    return input_path, report_path


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _kv_id_parse(id_text: str):
    out = {}
    for part in str(id_text).split("|"):
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


def _normalize_face_to_rings(face_node):
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

        if shell_idx is not None:
            face_node = boundaries[shell_idx][face_idx]
            return _normalize_face_to_rings(face_node), None
        face_node = boundaries[face_idx]
        return _normalize_face_to_rings(face_node), None
    except Exception as e:
        return None, f"could not access boundaries: {e}"


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v):
    return math.sqrt(_dot(v, v))


def _unit(v):
    n = _norm(v)
    if n == 0.0:
        return None
    return (v[0] / n, v[1] / n, v[2] / n)


def _angle_deg(a, b):
    ua = _unit(a)
    ub = _unit(b)
    if ua is None or ub is None:
        return None
    c = max(-1.0, min(1.0, _dot(ua, ub)))
    return math.degrees(math.acos(c))


def _edge_lengths(pts):
    if len(pts) < 2:
        return []
    out = []
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        d = _norm(_sub(a, b))
        out.append(d)
    return out


def _fan_triangle_normals(pts):
    if len(pts) < 3:
        return []
    normals = []
    p0 = pts[0]
    for i in range(1, len(pts) - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        n = _cross(_sub(p1, p0), _sub(p2, p0))
        if _norm(n) > 0.0:
            normals.append(n)
    return normals


def _parse_report_deviation(info_text):
    m = re.search(r"deviation normals:\s*([0-9.+-eE]+).*tolerance=([0-9.+-eE]+)", str(info_text))
    if not m:
        return None, None
    try:
        return float(m.group(1)), float(m.group(2))
    except Exception:
        return None, None


def main():
    try:
        input_path, report_path = _parse_args(sys.argv[1:])
    except Exception as e:
        print(f"[ERROR] {e}")
        print("Usage: python 03_diagnose_val3dity_204.py --input <cityjson> [--report <report_json>]")
        sys.exit(1)

    if not input_path.exists():
        print(f"[ERROR] Input CityJSON not found: {input_path}")
        sys.exit(1)
    if not report_path.exists():
        print(f"[ERROR] val3dity report not found: {report_path}")
        sys.exit(1)

    city = _load_json(input_path)
    report = _load_json(report_path)
    vertices = city.get("vertices", [])
    transform = city.get("transform")
    city_objects = city.get("CityObjects", {})

    errors_204 = []
    for feature in report.get("features", []):
        for err in feature.get("errors", []):
            if err.get("code") == 204:
                errors_204.append(err)

    print("=" * 70)
    print("VAL3DITY 204 DIAGNOSTIC")
    print("=" * 70)
    print(f"Input JSON:    {input_path}")
    print(f"Report JSON:   {report_path}")
    print(f"Error 204 count in report: {len(errors_204)}")
    print(f"Report parameters: {report.get('parameters', {})}")

    if not errors_204:
        print("No error 204 entries found.")
        return

    for i, err in enumerate(errors_204, 1):
        err_id = err.get("id", "")
        info = _kv_id_parse(err_id)
        coid = info.get("coid")
        geom_i = _as_int_or_none(info.get("geom"))
        shell_i = _as_int_or_none(info.get("shell"))
        face_i = _as_int_or_none(info.get("face"))
        dev_deg, tol_deg = _parse_report_deviation(err.get("info", ""))

        print("\n" + "-" * 70)
        print(f"[{i}] {err.get('description')}  id={err_id}")
        if dev_deg is not None:
            print(f"  Reported normals deviation: {dev_deg:.6f} deg (tolerance={tol_deg})")
        else:
            print(f"  Report info: {err.get('info', '')}")

        if coid not in city_objects:
            print(f"  [ERROR] CityObject '{coid}' not found.")
            continue
        if face_i is None:
            print("  [ERROR] Face index missing in error id.")
            continue

        cityobj = city_objects[coid]
        rings, msg = _get_rings_for_error(cityobj, geom_i, shell_i, face_i)
        if rings is None:
            print(f"  [ERROR] {msg}")
            continue
        if len(rings) == 0:
            print("  [WARN] Face resolved but has no rings.")
            continue

        print(f"  Resolved: CityObject={coid}, geom={geom_i}, shell={shell_i}, face={face_i}, rings={len(rings)}")

        # 204 is about surface planarity, so outer ring is most informative.
        outer = rings[0]
        pts = []
        bad_vidx = 0
        for vidx in outer:
            if not isinstance(vidx, int) or vidx < 0 or vidx >= len(vertices):
                bad_vidx += 1
                continue
            pts.append(_world_xyz(vertices, vidx, transform))

        print(f"  Outer ring vertices: {len(outer)} (valid coordinate refs: {len(pts)}, bad refs: {bad_vidx})")
        if len(pts) < 3:
            print("  [WARN] Not enough valid points to analyze.")
            continue

        edges = _edge_lengths(pts)
        if edges:
            print(
                "  Edge length stats (m): "
                f"min={min(edges):.6f}, median={sorted(edges)[len(edges)//2]:.6f}, max={max(edges):.6f}"
            )

        normals = _fan_triangle_normals(pts)
        if len(normals) < 2:
            print("  [WARN] Could not compute enough triangle normals (degenerate triangulation).")
            continue

        base = normals[0]
        devs = []
        for n in normals[1:]:
            a = _angle_deg(base, n)
            if a is not None:
                # Orientation flips can show ~180; use smallest equivalent angle.
                devs.append(min(a, 180.0 - a))

        if not devs:
            print("  [WARN] Normals deviation list empty after filtering.")
            continue

        max_dev = max(devs)
        mean_dev = sum(devs) / len(devs)
        print(f"  Triangulation normals deviation (deg): mean={mean_dev:.6f}, max={max_dev:.6f}")

        tiny_edges = sum(1 for d in edges if d < 0.005)  # 5 mm heuristic
        if tiny_edges > 0:
            print(f"  Heuristic: {tiny_edges} very short edge(s) (<5mm) -> likely numerically unstable/sliver zone.")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
