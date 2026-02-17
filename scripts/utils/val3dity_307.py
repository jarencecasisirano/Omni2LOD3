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


def _iter_307_targets(report_json):
    features = report_json.get("features", [])
    for feature in features:
        for err in feature.get("errors", []):
            if err.get("code") != 307:
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


def _flip_face_orientation(face_node):
    if not isinstance(face_node, list) or len(face_node) == 0:
        return False

    # Flat ring style: [i,j,k,...]
    if all(isinstance(v, int) for v in face_node):
        face_node.reverse()
        return True

    # Nested style: [[outer...], [hole...], ...]
    changed = False
    for ring in face_node:
        if isinstance(ring, list):
            ring.reverse()
            changed = True
    return changed


def apply_307_fix_from_report(cityjson_data: dict, report_json: dict):
    """
    Targeted 307 fix:
    - only flip orientation for faces flagged by val3dity 307.
    """
    stats = {
        "targets_total": 0,
        "targets_resolved": 0,
        "targets_missing": 0,
        "targets_unresolved": 0,
        "objects_modified": 0,
        "faces_flipped": 0,
    }

    city_objects = cityjson_data.get("CityObjects", {})
    changed_coids = set()
    seen = set()

    for t in _iter_307_targets(report_json):
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

        changed = _flip_face_orientation(face_node)
        if changed:
            stats["targets_resolved"] += 1
            stats["faces_flipped"] += 1
            changed_coids.add(coid)
        else:
            stats["targets_unresolved"] += 1

    stats["objects_modified"] = len(changed_coids)
    return stats
