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


def apply_902_fix(cityjson_data: dict):
    city_objects = cityjson_data.get("CityObjects", {})
    if not isinstance(city_objects, dict):
        return {
            "objects_modified": 0,
            "geometries_removed": 0,
        }

    objects_modified = 0
    geometries_removed = 0

    for _, obj in city_objects.items():
        if not isinstance(obj, dict):
            continue

        stats = _fix_cityobject_geometries(obj)
        changed = stats["geometries_removed"] > 0 or stats["geometries_touched"] > 0
        if changed:
            objects_modified += 1
            geometries_removed += stats["geometries_removed"]

    return {
        "objects_modified": objects_modified,
        "geometries_removed": geometries_removed,
    }
