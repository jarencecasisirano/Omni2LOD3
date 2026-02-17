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


def _world_xy(vertices, vidx, transform):
    v = vertices[vidx]
    x, y = float(v[0]), float(v[1])
    if not transform:
        return (x, y)

    scale = transform.get("scale", [1.0, 1.0, 1.0])
    translate = transform.get("translate", [0.0, 0.0, 0.0])
    return (
        x * float(scale[0]) + float(translate[0]),
        y * float(scale[1]) + float(translate[1]),
    )


def _clone_vertex_with_xy_nudge(vertices, transform, src_vidx, delta_world=0.001, axis="y"):
    src = vertices[src_vidx]
    x = float(src[0])
    y = float(src[1])
    z = float(src[2]) if len(src) > 2 else 0.0

    delta_local_x = float(delta_world)
    delta_local_y = float(delta_world)
    if transform:
        scale = transform.get("scale", [1.0, 1.0, 1.0])
        sx = float(scale[0]) if len(scale) > 0 else 1.0
        sy = float(scale[1]) if len(scale) > 1 else 1.0
        if sx != 0.0:
            delta_local_x = float(delta_world) / sx
        if sy != 0.0:
            delta_local_y = float(delta_world) / sy

    if axis == "x":
        new_v = [x + delta_local_x, y, z]
    else:
        new_v = [x, y + delta_local_y, z]

    vertices.append(new_v)
    return len(vertices) - 1


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


def _polygon_area_xy(pts):
    if len(pts) < 3:
        return 0.0
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


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


def _first_crossing_edge_pair(ring, vertices, transform):
    n = len(ring)
    if n < 4:
        return None

    pts = [_world_xy(vertices, vidx, transform) for vidx in ring]
    for i in range(n):
        i2 = (i + 1) % n
        for j in range(i + 1, n):
            j2 = (j + 1) % n
            if len({i, i2, j, j2}) < 4:
                continue
            if _segments_intersect(pts[i], pts[i2], pts[j], pts[j2]):
                return i, j
    return None


def _uncross_ring_2opt(ring, vertices, transform):
    if not isinstance(ring, list) or len(ring) < 4:
        return ring, False

    work = list(ring)
    max_iter = len(work) * len(work)
    changed = False

    for _ in range(max_iter):
        pair = _first_crossing_edge_pair(work, vertices, transform)
        if pair is None:
            break
        i, j = pair
        work[i + 1 : j + 1] = reversed(work[i + 1 : j + 1])
        changed = True

    return work, _first_crossing_edge_pair(work, vertices, transform) is None and changed


def _expand_collapsed_ring(ring, vertices, transform, tol):
    if not isinstance(ring, list) or len(ring) < 3:
        return ring, False

    analysis = _analyze_ring(ring, vertices, transform)
    area_tol = max(float(tol) * float(tol), 1e-8)
    pts = analysis.get("points", [])
    tiny_edge_tol = max(float(tol) * 1.25, 1e-6)
    tiny_edges = []
    for i in range(len(pts)):
        p1 = pts[i]
        p2 = pts[(i + 1) % len(pts)]
        d = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if d <= tiny_edge_tol:
            tiny_edges.append(i)

    if abs(analysis.get("area", 0.0)) >= area_tol and not tiny_edges:
        return ring, False

    out = list(ring)
    nudge_axis = "y"

    # Nudge endpoints of very short edges first.
    for edge_i in tiny_edges:
        j = (edge_i + 1) % len(out)
        vidx = out[j]
        if not isinstance(vidx, int) or vidx < 0 or vidx >= len(vertices):
            continue
        new_vidx = _clone_vertex_with_xy_nudge(
            vertices,
            transform,
            vidx,
            delta_world=tol * 2.0,
            axis=nudge_axis,
        )
        out[j] = new_vidx
        nudge_axis = "x" if nudge_axis == "y" else "y"

    # If not tiny-edge-driven, still nudge one vertex for near-zero-area rings.
    if not tiny_edges:
        vidx = out[1]
        if not isinstance(vidx, int) or vidx < 0 or vidx >= len(vertices):
            return ring, False
        new_vidx = _clone_vertex_with_xy_nudge(
            vertices,
            transform,
            vidx,
            delta_world=tol * 2.0,
            axis="y",
        )
        out[1] = new_vidx

    post = _analyze_ring(out, vertices, transform)
    post_pts = post.get("points", [])
    post_tiny = False
    for i in range(len(post_pts)):
        p1 = post_pts[i]
        p2 = post_pts[(i + 1) % len(post_pts)]
        d = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if d <= tiny_edge_tol:
            post_tiny = True
            break
    if abs(post.get("area", 0.0)) >= area_tol and not post.get("intersections"):
        return out, True

    # Fallback: simplify by removing one problematic vertex (typically from tiny edges).
    # Only accept if the result is still a valid non-self-intersecting ring.
    if len(out) > 3:
        candidate_indices = []
        if tiny_edges:
            candidate_indices = [((e + 1) % len(out)) for e in tiny_edges]
        if not candidate_indices:
            candidate_indices = list(range(len(out)))

        seen_idx = set()
        for drop_idx in candidate_indices:
            if drop_idx in seen_idx:
                continue
            seen_idx.add(drop_idx)
            cand = [v for i, v in enumerate(out) if i != drop_idx]
            if len(cand) < 3:
                continue
            ca = _analyze_ring(cand, vertices, transform)
            if ca.get("distinct_xy", 0) < 3:
                continue
            if abs(ca.get("area", 0.0)) < area_tol:
                continue
            if ca.get("intersections"):
                continue
            return cand, True

    if abs(post.get("area", 0.0)) < area_tol and post_tiny:
        return ring, False
    if post.get("intersections"):
        return ring, False
    return out, True


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


def _iter_104_targets(report_json):
    features = report_json.get("features", [])
    for feature in features:
        for err in feature.get("errors", []):
            if err.get("code") != 104:
                continue
            info = _parse_error_id(err.get("id", ""))
            yield {
                "coid": info.get("coid"),
                "geom": _as_int_or_none(info.get("geom")),
                "shell": _as_int_or_none(info.get("shell")),
                "face": _as_int_or_none(info.get("face")),
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


def apply_104_fix_from_report(cityjson_data: dict, report_json: dict, tol: float = 0.001):
    stats = {
        "targets_total": 0,
        "targets_resolved": 0,
        "targets_missing": 0,
        "targets_unresolved": 0,
        "objects_modified": 0,
        "rings_untangled": 0,
        "rings_expanded": 0,
    }

    city_objects = cityjson_data.get("CityObjects", {})
    vertices = cityjson_data.get("vertices", [])
    transform = cityjson_data.get("transform", None)
    changed_coids = set()
    seen = set()

    for t in _iter_104_targets(report_json):
        stats["targets_total"] += 1
        key = (t["coid"], t["geom"], t["shell"], t["face"])
        if key in seen:
            continue
        seen.add(key)

        coid = t["coid"]
        if coid not in city_objects:
            stats["targets_missing"] += 1
            continue

        face_node = _resolve_face_node(city_objects[coid], t["geom"], t["shell"], t["face"])
        if face_node is None:
            stats["targets_missing"] += 1
            continue

        rings, style = _normalize_face_to_rings(face_node)
        if len(rings) == 0:
            stats["targets_missing"] += 1
            continue

        out_rings = []
        resolved = True
        touched = False
        for ring in rings:
            if not isinstance(ring, list) or len(ring) < 4:
                out_rings.append(ring)
                continue
            fixed_ring, ok = _uncross_ring_2opt(ring, vertices, transform)
            if ok:
                touched = True
                stats["rings_untangled"] += 1
                out_rings.append(fixed_ring)
                continue

            expanded_ring, expanded = _expand_collapsed_ring(ring, vertices, transform, tol)
            if expanded:
                touched = True
                stats["rings_expanded"] += 1
                out_rings.append(expanded_ring)
            else:
                resolved = False
                out_rings.append(ring)

        new_face = _encode_rings_to_face(out_rings, style)
        if isinstance(face_node, list):
            face_node[:] = new_face if isinstance(new_face, list) else [new_face]

        if touched:
            changed_coids.add(coid)
            stats["targets_resolved"] += 1
        elif resolved:
            stats["targets_resolved"] += 1
        else:
            stats["targets_unresolved"] += 1

    stats["objects_modified"] = len(changed_coids)
    return stats
