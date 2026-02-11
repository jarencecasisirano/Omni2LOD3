import math


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


def _set_world_xyz(vertices, vidx, xyz_world, transform):
    xw, yw, zw = xyz_world
    if not transform:
        vertices[vidx] = [xw, yw, zw]
        return

    scale = transform.get("scale", [1.0, 1.0, 1.0])
    trans = transform.get("translate", [0.0, 0.0, 0.0])

    sx = float(scale[0]) if len(scale) > 0 else 1.0
    sy = float(scale[1]) if len(scale) > 1 else 1.0
    sz = float(scale[2]) if len(scale) > 2 else 1.0
    tx = float(trans[0]) if len(trans) > 0 else 0.0
    ty = float(trans[1]) if len(trans) > 1 else 0.0
    tz = float(trans[2]) if len(trans) > 2 else 0.0

    if sx == 0.0 or sy == 0.0 or sz == 0.0:
        vertices[vidx] = [xw, yw, zw]
        return

    vertices[vidx] = [(xw - tx) / sx, (yw - ty) / sy, (zw - tz) / sz]


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


def _newell_normal(points):
    if len(points) < 3:
        return None
    nx, ny, nz = 0.0, 0.0, 0.0
    n = len(points)
    for i in range(n):
        x1, y1, z1 = points[i]
        x2, y2, z2 = points[(i + 1) % n]
        nx += (y1 - y2) * (z1 + z2)
        ny += (z1 - z2) * (x1 + x2)
        nz += (x1 - x2) * (y1 + y2)
    return _unit((nx, ny, nz))


def _centroid(points):
    if not points:
        return (0.0, 0.0, 0.0)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sz = sum(p[2] for p in points)
    n = float(len(points))
    return (sx / n, sy / n, sz / n)


def _normalize_face_to_rings(face_node):
    if not isinstance(face_node, list):
        return []
    if len(face_node) == 0:
        return []
    if all(isinstance(v, int) for v in face_node):
        return [face_node]
    return [r for r in face_node if isinstance(r, list)]


def _iter_204_targets(report_json):
    features = report_json.get("features", [])
    for feature in features:
        for err in feature.get("errors", []):
            if err.get("code") != 204:
                continue
            info = _parse_error_id(err.get("id", ""))
            yield {
                "coid": info.get("coid"),
                "geom": _as_int_or_none(info.get("geom")),
                "shell": _as_int_or_none(info.get("shell")),
                "face": _as_int_or_none(info.get("face")),
                "id": err.get("id", ""),
            }


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


def _collect_valid_face_vidx(face_rings, vertices):
    unique = []
    seen = set()
    for ring in face_rings:
        for vidx in ring:
            if not isinstance(vidx, int):
                continue
            if vidx < 0 or vidx >= len(vertices):
                continue
            if vidx in seen:
                continue
            seen.add(vidx)
            unique.append(vidx)
    return unique


def apply_204_fix_from_report(cityjson_data: dict, report_json: dict, max_move: float = 0.01):
    """
    Targeted 204 fix:
    - only touches faces flagged by val3dity code 204
    - projects all unique face vertices to a best-fit (Newell) plane
    - skips face if required movement exceeds max_move
    """
    stats = {
        "targets_total": 0,
        "targets_resolved": 0,
        "targets_missing": 0,
        "targets_unresolved": 0,
        "objects_modified": 0,
        "faces_projected": 0,
        "vertices_moved": 0,
        "max_displacement": 0.0,
        "faces_skipped_large_move": 0,
    }

    city_objects = cityjson_data.get("CityObjects", {})
    vertices = cityjson_data.get("vertices", [])
    transform = cityjson_data.get("transform", None)

    changed_coids = set()
    seen = set()

    for t in _iter_204_targets(report_json):
        stats["targets_total"] += 1
        coid = t["coid"]
        geom_i = t["geom"]
        shell_i = t["shell"]
        face_i = t["face"]

        key = (coid, geom_i, shell_i, face_i)
        if key in seen:
            continue
        seen.add(key)

        if coid not in city_objects:
            stats["targets_missing"] += 1
            continue

        cityobj = city_objects[coid]
        face_node = _resolve_face_node(cityobj, geom_i, shell_i, face_i)
        if face_node is None:
            stats["targets_missing"] += 1
            continue

        rings = _normalize_face_to_rings(face_node)
        if len(rings) == 0:
            stats["targets_missing"] += 1
            continue

        outer = rings[0]
        outer_pts = []
        for vidx in outer:
            if isinstance(vidx, int) and 0 <= vidx < len(vertices):
                outer_pts.append(_world_xyz(vertices, vidx, transform))

        if len(outer_pts) < 3:
            stats["targets_unresolved"] += 1
            continue

        n = _newell_normal(outer_pts)
        if n is None:
            stats["targets_unresolved"] += 1
            continue

        c = _centroid(outer_pts)
        face_vidx = _collect_valid_face_vidx(rings, vertices)
        if not face_vidx:
            stats["targets_unresolved"] += 1
            continue

        projected = {}
        worst = 0.0
        for vidx in face_vidx:
            p = _world_xyz(vertices, vidx, transform)
            d = _dot(_sub(p, c), n)
            p_proj = (p[0] - d * n[0], p[1] - d * n[1], p[2] - d * n[2])
            ad = abs(d)
            projected[vidx] = (p_proj, ad)
            if ad > worst:
                worst = ad

        if worst > max_move:
            stats["faces_skipped_large_move"] += 1
            stats["targets_unresolved"] += 1
            continue

        moved_here = 0
        for vidx, (p_proj, ad) in projected.items():
            _set_world_xyz(vertices, vidx, p_proj, transform)
            if ad > 0.0:
                moved_here += 1
            if ad > stats["max_displacement"]:
                stats["max_displacement"] = ad

        stats["vertices_moved"] += moved_here
        stats["faces_projected"] += 1
        stats["targets_resolved"] += 1
        changed_coids.add(coid)

    stats["objects_modified"] = len(changed_coids)
    return stats
