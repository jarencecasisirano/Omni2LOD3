import argparse
import json
from pathlib import Path


def _parse_error_id(id_text: str):
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


def _world_xy(vertices, vidx, transform):
    v = vertices[vidx]
    x, y = float(v[0]), float(v[1])
    if not transform:
        return (x, y)
    scale = transform.get("scale", [1.0, 1.0, 1.0])
    trans = transform.get("translate", [0.0, 0.0, 0.0])
    return (
        x * float(scale[0]) + float(trans[0]),
        y * float(scale[1]) + float(trans[1]),
    )


def _orientation(a, b, c):
    v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(v) < 1e-12:
        return 0
    return 1 if v > 0 else -1


def _on_segment(a, b, c):
    return (
        min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12
        and min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12
    )


def _segments_intersect(p1, p2, q1, q2):
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, q1):
        return True
    if o2 == 0 and _on_segment(p1, p2, q2):
        return True
    if o3 == 0 and _on_segment(q1, q2, p1):
        return True
    if o4 == 0 and _on_segment(q1, q2, p2):
        return True
    return False


def _polygon_area_xy(pts):
    if len(pts) < 3:
        return 0.0
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _normalize_face_to_rings(face_node):
    if not isinstance(face_node, list):
        return []
    if len(face_node) == 0:
        return []
    if all(isinstance(v, int) for v in face_node):
        return [face_node]
    return [r for r in face_node if isinstance(r, list)]


def _resolve_face_node(cityobj, geom_i, shell_i, face_i):
    geoms = cityobj.get("geometry", [])
    if not isinstance(geoms, list) or geom_i is None or face_i is None:
        return None
    if geom_i < 0 or geom_i >= len(geoms):
        return None

    geom = geoms[geom_i]
    boundaries = geom.get("boundaries")
    gtype = geom.get("type", "")

    try:
        if gtype in ("Solid", "CompositeSolid"):
            if shell_i is None:
                return None
            return boundaries[shell_i][face_i]
        if gtype in ("MultiSurface", "CompositeSurface"):
            return boundaries[face_i]
        if shell_i is not None:
            return boundaries[shell_i][face_i]
        return boundaries[face_i]
    except Exception:
        return None


def _iter_error_targets(report_json, target_code):
    for feat in report_json.get("features", []):
        fid = feat.get("id")
        ftype = feat.get("type")
        for err in feat.get("errors", []):
            if err.get("code") != target_code:
                continue
            info = _parse_error_id(err.get("id", ""))
            yield {
                "feature_id": fid,
                "feature_type": ftype,
                "code": err.get("code"),
                "description": err.get("description"),
                "info": err.get("info"),
                "coid": info.get("coid"),
                "geom": _as_int_or_none(info.get("geom")),
                "shell": _as_int_or_none(info.get("shell")),
                "face": _as_int_or_none(info.get("face")),
            }


def _analyze_ring(ring, vertices, transform):
    pts = []
    for vidx in ring:
        if not isinstance(vidx, int) or vidx < 0 or vidx >= len(vertices):
            continue
        pts.append(_world_xy(vertices, vidx, transform))

    if len(pts) < 3:
        return {"valid_points": len(pts), "distinct_xy": len(set(pts)), "area": 0.0, "intersections": []}

    intersections = []
    n = len(pts)
    for i in range(n):
        i2 = (i + 1) % n
        for j in range(i + 1, n):
            j2 = (j + 1) % n
            if len({i, i2, j, j2}) < 4:
                continue
            if _segments_intersect(pts[i], pts[i2], pts[j], pts[j2]):
                intersections.append((i, i2, j, j2))

    return {
        "valid_points": len(pts),
        "distinct_xy": len(set(pts)),
        "area": _polygon_area_xy(pts),
        "intersections": intersections,
        "points": pts,
    }


def _write_debug_geojson(out_path: Path, feature_title: str, rings_analysis):
    features = []
    for idx, ra in enumerate(rings_analysis):
        pts = ra.get("points", [])
        if not pts:
            continue
        line_coords = [[p[0], p[1]] for p in pts + [pts[0]]]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "title": feature_title,
                    "ring_index": idx,
                    "distinct_xy": ra["distinct_xy"],
                    "area": ra["area"],
                    "intersection_count": len(ra["intersections"]),
                },
                "geometry": {"type": "LineString", "coordinates": line_coords},
            }
        )
    out = {"type": "FeatureCollection", "features": features}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Diagnose val3dity-reported problematic faces.")
    parser.add_argument("--cityjson", required=True, help="Path to CityJSON file.")
    parser.add_argument("--report", required=True, help="Path to val3dity JSON report.")
    parser.add_argument("--code", type=int, default=104, help="Error code to inspect (default: 104).")
    parser.add_argument(
        "--out",
        default="outputs/03_val3dity/debug_face.geojson",
        help="Output debug GeoJSON path for ring visualization.",
    )
    args = parser.parse_args()

    cityjson_path = Path(args.cityjson)
    report_path = Path(args.report)
    out_path = Path(args.out)

    data = json.loads(cityjson_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    city_objects = data.get("CityObjects", {})
    vertices = data.get("vertices", [])
    transform = data.get("transform", None)

    targets = list(_iter_error_targets(report, args.code))
    if not targets:
        print(f"No code {args.code} targets found in report.")
        return

    print(f"Found {len(targets)} target(s) for code {args.code}.")
    for t in targets:
        print(
            f"\nTarget: feature={t['feature_id']} coid={t['coid']} "
            f"geom={t['geom']} shell={t['shell']} face={t['face']}"
        )
        print(f"Description: {t['description']}")
        print(f"Info: {t['info']}")

        coid = t["coid"]
        if coid not in city_objects:
            print("  [WARN] coid not found in CityObjects.")
            continue

        face_node = _resolve_face_node(city_objects[coid], t["geom"], t["shell"], t["face"])
        if face_node is None:
            print("  [WARN] Could not resolve target face.")
            continue

        rings = _normalize_face_to_rings(face_node)
        if not rings:
            print("  [WARN] Face has no rings.")
            continue

        rings_analysis = []
        for i, ring in enumerate(rings):
            ra = _analyze_ring(ring, vertices, transform)
            rings_analysis.append(ra)
            print(f"  Ring {i}:")
            print(f"    valid_points: {ra['valid_points']}")
            print(f"    distinct_xy:  {ra['distinct_xy']}")
            print(f"    area:         {ra['area']:.6f}")
            print(f"    intersections:{len(ra['intersections'])}")
            if ra["intersections"]:
                first = ra["intersections"][0]
                print(f"    first_cross:  edges {first[0]}-{first[1]} x {first[2]}-{first[3]}")

        title = (
            f"code={args.code}|coid={t['coid']}|geom={t['geom']}|"
            f"shell={t['shell']}|face={t['face']}"
        )
        _write_debug_geojson(out_path, title, rings_analysis)
        print(f"  Debug geometry written to: {out_path}")


if __name__ == "__main__":
    main()
