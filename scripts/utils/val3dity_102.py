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
    translate = transform.get("translate", [0.0, 0.0, 0.0])
    return (
        x * float(scale[0]) + float(translate[0]),
        y * float(scale[1]) + float(translate[1]),
        z * float(scale[2]) + float(translate[2]),
    )


def _xy_dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _orientation(a, b, c):
    v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if math.isclose(v, 0.0, abs_tol=1e-12):
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


def _polygon_area_xy(points):
    if len(points) < 3:
        return 0.0
    s = 0.0
    for i in range(len(points)):
        x1, y1 = points[i][0], points[i][1]
        x2, y2 = points[(i + 1) % len(points)][0], points[(i + 1) % len(points)][1]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _ring_has_self_intersection(vertices, transform, ring):
    n = len(ring)
    if n < 4:
        return False
    pts = [_world_xyz(vertices, vidx, transform) for vidx in ring]
    for i in range(n):
        i2 = (i + 1) % n
        for j in range(i + 1, n):
            j2 = (j + 1) % n
            if len({i, i2, j, j2}) < 4:
                continue
            if _segments_intersect(pts[i], pts[i2], pts[j], pts[j2]):
                return True
    return False


def _ring_world_points(vertices, transform, ring):
    pts = []
    for vidx in ring:
        if not isinstance(vidx, int):
            return []
        if vidx < 0 or vidx >= len(vertices):
            return []
        pts.append(_world_xyz(vertices, vidx, transform))
    return pts


def _ring_is_104_safe(vertices, transform, ring, tol):
    # Mirror val3dity's snap-driven robustness: reject tiny/collapsed rings early.
    pts = _ring_world_points(vertices, transform, ring)
    if len(pts) < 3:
        return False
    distinct = len({_snap_xy_key(p, tol) for p in pts})
    if distinct < 3:
        return False
    if _ring_has_self_intersection(vertices, transform, ring):
        return False
    area = abs(_polygon_area_xy(pts))
    if area < float(tol) * float(tol):
        return False
    return True


def _snap_xy_key(pt, tol):
    if tol <= 0:
        return (pt[0], pt[1])
    return (int(round(pt[0] / tol)), int(round(pt[1] / tol)))


def _are_consecutive_duplicates(vertices, transform, a_idx, b_idx, tol):
    if a_idx == b_idx:
        return True

    a = _world_xyz(vertices, a_idx, transform)
    b = _world_xyz(vertices, b_idx, transform)

    # val3dity 102 can happen after 2D projection/snap; XY proximity is key.
    return _xy_dist(a, b) <= tol or _snap_xy_key(a, tol) == _snap_xy_key(b, tol)


def _ring_distinct_xy_count(vertices, transform, ring, tol):
    keys = set()
    for vidx in ring:
        if not isinstance(vidx, int):
            continue
        if vidx < 0 or vidx >= len(vertices):
            continue
        xyz = _world_xyz(vertices, vidx, transform)
        keys.add(_snap_xy_key(xyz, tol))
    return len(keys)


def _clean_ring_consecutive_duplicates(vertices, transform, ring, tol):
    if not isinstance(ring, list) or len(ring) == 0:
        return ring, 0, False

    cleaned = [ring[0]]
    removed = 0

    for vidx in ring[1:]:
        prev = cleaned[-1]
        if isinstance(prev, int) and isinstance(vidx, int):
            if (
                0 <= prev < len(vertices)
                and 0 <= vidx < len(vertices)
                and _are_consecutive_duplicates(vertices, transform, prev, vidx, tol)
            ):
                removed += 1
                continue
        cleaned.append(vidx)

    valid = _ring_distinct_xy_count(vertices, transform, cleaned, tol) >= 3
    return cleaned, removed, valid


def _clone_vertex_with_xy_nudge(vertices, transform, src_vidx, tol, axis="x", sign=1.0):
    src = vertices[src_vidx]
    x = float(src[0])
    y = float(src[1])
    z = float(src[2])

    delta_world = float(tol) * 1.01
    delta_local_x = delta_world
    delta_local_y = delta_world

    if transform:
        scale = transform.get("scale", [1.0, 1.0, 1.0])
        sx = float(scale[0]) if len(scale) > 0 else 1.0
        sy = float(scale[1]) if len(scale) > 1 else 1.0
        if sx != 0.0:
            delta_local_x = delta_world / sx
        if sy != 0.0:
            delta_local_y = delta_world / sy

    if axis == "x":
        new_v = [x + float(sign) * delta_local_x, y, z]
    else:
        new_v = [x, y + float(sign) * delta_local_y, z]

    vertices.append(new_v)
    return len(vertices) - 1


def _repair_ring_preserve_face(vertices, transform, ring, tol):
    """
    Strategy:
    1) Try normal consecutive removal.
    2) If ring would collapse, keep face and nudge cloned vertices on duplicate pairs.
    """
    cleaned, removed, valid = _clean_ring_consecutive_duplicates(vertices, transform, ring, tol)
    if valid and _ring_is_104_safe(vertices, transform, cleaned, tol):
        return cleaned, removed, 0, False, True

    if not isinstance(ring, list) or len(ring) < 3:
        return ring, removed, 0, False, False

    # Special case for A,A,B,B-like rings: build a stable triangle from the 2-point backbone.
    if len(cleaned) == 2 and all(isinstance(v, int) for v in cleaned):
        base_a, base_b = cleaned[0], cleaned[1]
        pa = _world_xyz(vertices, base_a, transform)
        pb = _world_xyz(vertices, base_b, transform)
        dx = pb[0] - pa[0]
        dy = pb[1] - pa[1]
        axis = "y" if abs(dx) >= abs(dy) else "x"
        for sign in (1.0, -1.0):
            tri_vidx = _clone_vertex_with_xy_nudge(vertices, transform, base_b, tol * 2.0, axis=axis, sign=sign)
            tri = [base_a, base_b, tri_vidx]
            if _ring_is_104_safe(vertices, transform, tri, tol):
                return tri, removed, 1, True, True

    candidate = list(ring)
    new_vertices_added = 0

    # Nudge the second vertex of each duplicate consecutive pair.
    # Alternate x/y axis for stability if multiple pairs exist.
    nudge_axis = "x"
    for i in range(1, len(candidate)):
        a_idx = candidate[i - 1]
        b_idx = candidate[i]
        if not isinstance(a_idx, int) or not isinstance(b_idx, int):
            continue
        if (
            0 <= a_idx < len(vertices)
            and 0 <= b_idx < len(vertices)
            and _are_consecutive_duplicates(vertices, transform, a_idx, b_idx, tol)
        ):
            new_vidx = _clone_vertex_with_xy_nudge(vertices, transform, b_idx, tol, axis=nudge_axis, sign=1.0)
            candidate[i] = new_vidx
            new_vertices_added += 1
            nudge_axis = "y" if nudge_axis == "x" else "x"

    candidate_cleaned, removed2, valid2 = _clean_ring_consecutive_duplicates(vertices, transform, candidate, tol)
    if valid2 and _ring_is_104_safe(vertices, transform, candidate_cleaned, tol):
        return candidate_cleaned, removed + removed2, new_vertices_added, True, True

    # Safety fallback: leave ring unchanged if we still cannot make it valid.
    return ring, 0, 0, False, False


def _normalize_face_to_rings(face_node):
    if not isinstance(face_node, list):
        return [], "unknown"
    if len(face_node) == 0:
        return [], "unknown"
    if all(isinstance(v, int) for v in face_node):
        return [face_node], "flat"
    return [r for r in face_node if isinstance(r, list)], "nested"


def _encode_rings_to_face(rings, style):
    if style == "flat":
        if len(rings) == 1:
            return rings[0]
        if len(rings) > 1:
            return rings
        return []
    return rings


def _iter_102_targets(report_json):
    features = report_json.get("features", [])
    for feature in features:
        for err in feature.get("errors", []):
            if err.get("code") != 102:
                continue
            info = _parse_error_id(err.get("id", ""))
            yield {
                "coid": info.get("coid"),
                "geom": _as_int_or_none(info.get("geom")),
                "shell": _as_int_or_none(info.get("shell")),
                "face": _as_int_or_none(info.get("face")),
                "id": err.get("id", ""),
            }


def _fix_face_node_rings(vertices, transform, face_node, tol):
    rings, style = _normalize_face_to_rings(face_node)
    if len(rings) == 0:
        return None, {
            "consecutive_removed": 0,
            "new_vertices_added": 0,
            "rings_nudged": 0,
            "resolved": False,
        }

    out_rings = []
    s = {
        "consecutive_removed": 0,
        "new_vertices_added": 0,
        "rings_nudged": 0,
        "resolved": True,
    }

    for ring in rings:
        fixed_ring, removed, added, nudged, resolved = _repair_ring_preserve_face(vertices, transform, ring, tol)
        s["consecutive_removed"] += removed
        s["new_vertices_added"] += added
        if nudged:
            s["rings_nudged"] += 1
        if not resolved:
            s["resolved"] = False
        out_rings.append(fixed_ring)

    return _encode_rings_to_face(out_rings, style), s


def apply_102_fix_from_report(cityjson_data: dict, report_json: dict, tol: float = 0.001):
    """
    Applies a targeted 102 fix only to faces reported by val3dity.
    Returns stats dict.
    """
    stats = {
        "targets_total": 0,
        "targets_resolved": 0,
        "targets_missing": 0,
        "targets_unresolved": 0,
        "objects_modified": 0,
        "consecutive_removed": 0,
        "rings_nudged": 0,
        "new_vertices_added": 0,
        # Kept for backward compatibility with existing callers/log lines.
        "rings_dropped": 0,
        "faces_dropped": 0,
    }

    city_objects = cityjson_data.get("CityObjects", {})
    vertices = cityjson_data.get("vertices", [])
    transform = cityjson_data.get("transform", None)
    changed_coids = set()

    raw_targets = list(_iter_102_targets(report_json))
    stats["targets_total"] = len(raw_targets)

    targets = sorted(
        raw_targets,
        key=lambda t: (
            str(t.get("coid")),
            -1 if t.get("geom") is None else int(t.get("geom")),
            -1 if t.get("shell") is None else int(t.get("shell")),
            10**12 if t.get("face") is None else -int(t.get("face")),
        ),
    )

    seen = set()

    for t in targets:
        coid = t["coid"]
        geom_i = t["geom"]
        shell_i = t["shell"]
        face_i = t["face"]

        # Ignore duplicate report entries for the same target
        key = (coid, geom_i, shell_i, face_i)
        if key in seen:
            continue
        seen.add(key)

        if coid not in city_objects or geom_i is None or face_i is None:
            stats["targets_missing"] += 1
            continue

        cityobj = city_objects[coid]
        geoms = cityobj.get("geometry", [])
        if not isinstance(geoms, list) or geom_i < 0 or geom_i >= len(geoms):
            stats["targets_missing"] += 1
            continue

        geom = geoms[geom_i]
        boundaries = geom.get("boundaries")
        gtype = geom.get("type", "")

        try:
            if gtype in ("Solid", "CompositeSolid"):
                if shell_i is None:
                    stats["targets_missing"] += 1
                    continue
                shell = boundaries[shell_i]
                face_node = shell[face_i]
                new_face, s = _fix_face_node_rings(vertices, transform, face_node, tol)
                if new_face is None:
                    stats["targets_missing"] += 1
                    continue
                shell[face_i] = new_face
                stats["consecutive_removed"] += s["consecutive_removed"]
                stats["new_vertices_added"] += s["new_vertices_added"]
                stats["rings_nudged"] += s["rings_nudged"]
                if not s["resolved"]:
                    stats["targets_unresolved"] += 1

            elif gtype in ("MultiSurface", "CompositeSurface"):
                face_node = boundaries[face_i]
                new_face, s = _fix_face_node_rings(vertices, transform, face_node, tol)
                if new_face is None:
                    stats["targets_missing"] += 1
                    continue
                boundaries[face_i] = new_face
                stats["consecutive_removed"] += s["consecutive_removed"]
                stats["new_vertices_added"] += s["new_vertices_added"]
                stats["rings_nudged"] += s["rings_nudged"]
                if not s["resolved"]:
                    stats["targets_unresolved"] += 1

            else:
                # Fallback: try Solid-style when shell exists, else MultiSurface-style.
                if shell_i is not None:
                    shell = boundaries[shell_i]
                    face_node = shell[face_i]
                    new_face, s = _fix_face_node_rings(vertices, transform, face_node, tol)
                    if new_face is None:
                        stats["targets_missing"] += 1
                        continue
                    shell[face_i] = new_face
                    stats["consecutive_removed"] += s["consecutive_removed"]
                    stats["new_vertices_added"] += s["new_vertices_added"]
                    stats["rings_nudged"] += s["rings_nudged"]
                    if not s["resolved"]:
                        stats["targets_unresolved"] += 1
                else:
                    face_node = boundaries[face_i]
                    new_face, s = _fix_face_node_rings(vertices, transform, face_node, tol)
                    if new_face is None:
                        stats["targets_missing"] += 1
                        continue
                    boundaries[face_i] = new_face
                    stats["consecutive_removed"] += s["consecutive_removed"]
                    stats["new_vertices_added"] += s["new_vertices_added"]
                    stats["rings_nudged"] += s["rings_nudged"]
                    if not s["resolved"]:
                        stats["targets_unresolved"] += 1

            changed_coids.add(coid)
            stats["targets_resolved"] += 1

        except Exception:
            stats["targets_missing"] += 1
            continue

    stats["objects_modified"] = len(changed_coids)
    return stats
