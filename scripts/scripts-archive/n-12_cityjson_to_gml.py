#!/usr/bin/env python3
"""
CityJSON → CityGML LOD3 Converter.

Reads a CityJSON file from outputs/14_extrusions_json, converts every
CityObject to CityGML, and saves the result to outputs/15_flat_gml.

Wall-polygon holes (interior rings stored in CityJSON as additional ring
lists inside a polygon) are converted to opening elements:
  • Window  – if the hole's bottom edge is ≥ DOOR_HEIGHT_THRESHOLD above
              the lowest ground vertex of the building.
  • Door    – if the hole's bottom edge is below that threshold (i.e.
              the opening reaches the ground level).

Usage:
    conda activate las-env
    python scripts/LOD2toLOD3/15_cityjson_to_gml.py [-o OUTPUT]

Options:
    --door_threshold   Height above ground for Window vs Door classification
                       (default 0.4 m).  Holes whose bottom z is less than
                       ground_z + threshold are treated as Doors.
    --output, -o       Output GML filename (in outputs/15_flat_gml/).
"""

import os
import sys
import glob
import json
import uuid
import argparse

import numpy as np


# ─── Directories ──────────────────────────────────────────────────────────────
JSON_DIR   = "outputs/12_curve_json"
OUTPUT_DIR = "outputs/15_flat_gml"

# ─── CityGML namespace / header constants ─────────────────────────────────────
GML_HEADER = """\
<?xml version="1.0" encoding="UTF-8"?>
<CityModel
  xmlns="http://www.opengis.net/citygml/2.0"
  xmlns:bldg="http://www.opengis.net/citygml/building/2.0"
  xmlns:gml="http://www.opengis.net/gml"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://www.opengis.net/citygml/2.0
    http://schemas.opengis.net/citygml/2.0/cityGMLBase.xsd
    http://www.opengis.net/citygml/building/2.0
    http://schemas.opengis.net/citygml/building/2.0/building.xsd">"""

GML_FOOTER = "</CityModel>"

# Opening hole classification: holes whose lower z is within this distance
# above the building ground are treated as Doors; higher ones are Windows.
DOOR_HEIGHT_THRESHOLD_DEFAULT = 0.3   # metres


# =====================================================================
# Interactive file selector
# =====================================================================
def select_file(directory, pattern="*.json"):
    """List files in *directory* matching *pattern* and let the user pick one."""
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        print(f"  No {pattern} files found in {directory}")
        sys.exit(1)
    print(f"\n{'='*60}")
    print(f"  Files in: {directory}")
    print(f"{'='*60}")
    for i, f in enumerate(files):
        mb = os.path.getsize(f) / 1_048_576
        print(f"  [{i+1}] {os.path.basename(f):50s} ({mb:.1f} MB)")
    print()
    while True:
        try:
            c = int(input(f"  Select file [1-{len(files)}]: ").strip())
            if 1 <= c <= len(files):
                return files[c - 1]
        except (ValueError, KeyboardInterrupt):
            pass
        print("  Invalid choice, try again.")


# =====================================================================
# CityJSON vertex helpers
# =====================================================================
def decode_vertices(cm):
    raw       = np.array(cm["vertices"], dtype=np.float64)
    t         = cm.get("transform", {})
    scale     = np.array(t.get("scale",     [1, 1, 1]), dtype=np.float64)
    translate = np.array(t.get("translate", [0, 0, 0]), dtype=np.float64)
    return raw * scale + translate


def ring_coords(polygon_ring, world_verts):
    """
    Decode a CityJSON polygon ring (list of vertex indices) to a list of
    (x, y, z) tuples.  The ring is NOT closed (last point ≠ first).
    """
    return [tuple(world_verts[vi].tolist()) for vi in polygon_ring]


# =====================================================================
# GML string builders
# =====================================================================
def poslist_str(pts, indent=""):
    """
    Convert a list of (x, y, z) tuples to a space-separated gml:posList
    string, closing the ring if it is not already closed.
    """
    if pts and pts[0] != pts[-1]:
        pts = list(pts) + [pts[0]]   # ensure closed
    return " ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in pts)


def exterior_ring_gml(pts, poly_id, indent="            "):
    pl = poslist_str(pts)
    return (
        f'{indent}<gml:exterior>\n'
        f'{indent}  <gml:LinearRing>\n'
        f'{indent}    <gml:posList srsDimension="3">{pl}</gml:posList>\n'
        f'{indent}  </gml:LinearRing>\n'
        f'{indent}</gml:exterior>'
    )


def interior_ring_gml(pts, indent="            "):
    pl = poslist_str(pts)
    return (
        f'{indent}<gml:interior>\n'
        f'{indent}  <gml:LinearRing>\n'
        f'{indent}    <gml:posList srsDimension="3">{pl}</gml:posList>\n'
        f'{indent}  </gml:LinearRing>\n'
        f'{indent}</gml:interior>'
    )


def polygon_gml(ext_pts, int_rings, poly_id, indent="          "):
    """Build a gml:Polygon with optional interior rings."""
    lines = [f'{indent}<gml:Polygon gml:id="{poly_id}">',
             exterior_ring_gml(ext_pts, poly_id, indent + "  ")]
    for hole_pts in int_rings:
        lines.append(interior_ring_gml(hole_pts, indent + "  "))
    lines.append(f'{indent}</gml:Polygon>')
    return "\n".join(lines)


def multi_surface_gml(surfaces_with_holes, ms_id, indent="        "):
    """
    Build a gml:MultiSurface from a list of
    (exterior_pts, [hole_pts, ...]) tuples.
    """
    members = []
    for i, (ext_pts, int_rings) in enumerate(surfaces_with_holes):
        pid = f"{ms_id}_p{i}"
        members.append(
            f'{indent}  <gml:surfaceMember>\n'
            f'{polygon_gml(ext_pts, int_rings, pid, indent + "    ")}\n'
            f'{indent}  </gml:surfaceMember>'
        )
    body = "\n".join(members)
    return (
        f'{indent}<gml:MultiSurface gml:id="{ms_id}">\n'
        f'{body}\n'
        f'{indent}</gml:MultiSurface>'
    )


def opening_gml(ftype, gml_id, pts, indent="        "):
    """Build a bldg:opening element (Window or Door) with a single polygon."""
    tag   = "bldg:Window" if ftype == "window" else "bldg:Door"
    pl    = poslist_str(pts)
    return (
        f'{indent}<bldg:opening>\n'
        f'{indent}  <{tag} gml:id="{gml_id}">\n'
        f'{indent}    <bldg:lod3MultiSurface>\n'
        f'{indent}      <gml:MultiSurface>\n'
        f'{indent}        <gml:surfaceMember>\n'
        f'{indent}          <gml:Polygon gml:id="{gml_id}_face">\n'
        f'{indent}            <gml:exterior>\n'
        f'{indent}              <gml:LinearRing>\n'
        f'{indent}                <gml:posList srsDimension="3">{pl}</gml:posList>\n'
        f'{indent}              </gml:LinearRing>\n'
        f'{indent}            </gml:exterior>\n'
        f'{indent}          </gml:Polygon>\n'
        f'{indent}        </gml:surfaceMember>\n'
        f'{indent}      </gml:MultiSurface>\n'
        f'{indent}    </bldg:lod3MultiSurface>\n'
        f'{indent}  </{tag}>\n'
        f'{indent}</bldg:opening>'
    )


# =====================================================================
# Classify a hole ring as Window or Door
# =====================================================================
def classify_hole(hole_pts, ground_z, door_threshold):
    """
    Return 'window' or 'door' based on the vertical position of the hole.

    door_threshold: if the hole's bottom z is at most
                    ground_z + door_threshold, and its height is less than 4m,
                    it is a Door; otherwise a Window.
    """
    z_vals  = [p[2] for p in hole_pts]
    hole_z0 = min(z_vals)   # bottom of the hole
    hole_z1 = max(z_vals)   # top of the hole
    hole_height = hole_z1 - hole_z0

    if hole_z0 <= ground_z + door_threshold and hole_height < 4.0:
        return "door"
    return "window"


# =====================================================================
# Determine ground_z from the CityJSON model
# =====================================================================
_GROUND_TYPES = {"Building", "BuildingPart"}
_SKIP_TYPES   = {"Window", "Door", "BuildingInstallation", "OtherConstruction"}

def _flatten_indices(x):
    if isinstance(x, list):
        for y in x:
            yield from _flatten_indices(y)
    elif isinstance(x, int):
        yield x


def find_ground_z(cm, world_verts):
    """
    Return the minimum Z coordinate over all non-child, non-opening CityObjects,
    used as the reference ground elevation for Door/Window classification.
    """
    z_min = float("inf")
    for obj_id, obj in cm.get("CityObjects", {}).items():
        if obj.get("type", "") in _SKIP_TYPES:
            continue
        for geom in obj.get("geometry", []):
            for vi in _flatten_indices(geom.get("boundaries", [])):
                if 0 <= vi < len(world_verts):
                    z = float(world_verts[vi, 2])
                    if z < z_min:
                        z_min = z
    return z_min if z_min < float("inf") else 0.0


# =====================================================================
# Semantic surface type helpers
# =====================================================================
def _sem_type(geom, shell_idx, poly_idx_in_shell, flat_idx):
    """
    Return the CityJSON semantic type string for a polygon, handling both
    Solid geometry (values is a nested list: values[shell][poly]) and
    MultiSurface geometry (values is a flat list: values[poly]).

    Parameters
    ----------
    geom              : CityJSON geometry dict
    shell_idx         : index of the shell (0 for MultiSurface)
    poly_idx_in_shell : index of the polygon within its shell
    flat_idx          : sequential flat index across all shells
                        (used as fallback for MultiSurface flat values)
    """
    sem      = geom.get("semantics", {})
    surfaces = sem.get("surfaces", [])
    values   = sem.get("values",   [])
    if not surfaces or not values:
        return ""
    try:
        # Solid: values = [[s0, s1, ...], [s0, ...], ...]  (nested per shell)
        inner = values[shell_idx]
        if isinstance(inner, list):
            idx = inner[poly_idx_in_shell]
        else:
            # MultiSurface: values = [s0, s1, ...] (flat)
            idx = values[flat_idx]
        if idx is None:
            return ""
        return surfaces[idx].get("type", "")
    except (IndexError, KeyError, TypeError):
        return ""


# =====================================================================
# Main conversion: CityJSON → CityGML string
# =====================================================================
def cityjson_to_gml_str(cm, world_verts, door_threshold):
    """
    Convert a CityJSON model dict to a CityGML string.

    For each CityObject the function emits the appropriate GML element.
    Interior rings in wall polygons are extracted as openings (Window/Door).

    Returns the complete GML document as a string.
    """
    ground_z = find_ground_z(cm, world_verts)
    print(f"  Ground Z reference: {ground_z:.3f} m")
    print(f"  Door/Window threshold: ground + {door_threshold:.2f} m = "
          f"{ground_z + door_threshold:.3f} m")

    city_members = []   # list of top-level <cityObjectMember> strings

    # ── Build a lookup: obj_id → gml_id safe string ───────────────────
    def safe_id(s):
        return s.replace(" ", "_").replace(":", "_").replace("/", "_")

    # ── Iterate CityObjects ───────────────────────────────────────────
    for obj_id, obj in cm.get("CityObjects", {}).items():
        obj_type  = obj.get("type", "")
        gml_id    = safe_id(obj_id)

        # ── Building / BuildingPart ─────────────────────────────────
        if obj_type in ("Building", "BuildingPart"):
            bldg_lines = []
            bldg_tag  = "bldg:Building" if obj_type == "Building" else "bldg:BuildingPart"

            for geom in obj.get("geometry", []):
                boundaries = geom.get("boundaries", [])
                geom_type  = geom.get("type", "")

                # For Solid: boundaries = [shell, ...]; each shell is [polygon, ...]
                # For MultiSurface: boundaries = [polygon, ...]
                # poly_list_iter yields (shell_idx, poly_idx_in_shell, polygon)
                if geom_type == "Solid":
                    poly_list_iter = [
                        (s_idx, p_idx, polygon)
                        for s_idx, shell in enumerate(boundaries)
                        for p_idx, polygon in enumerate(shell)
                    ]
                elif geom_type == "MultiSurface":
                    poly_list_iter = [
                        (0, p_idx, polygon)
                        for p_idx, polygon in enumerate(boundaries)
                    ]
                else:
                    continue

                for flat_idx, (s_idx, p_idx, polygon) in enumerate(poly_list_iter):
                    sem = _sem_type(geom, s_idx, p_idx, flat_idx)
                    ext_ring    = polygon[0]
                    inner_rings = polygon[1:]
                    ext_pts     = ring_coords(ext_ring, world_verts)

                    surf_gml_id = f"{gml_id}_s{flat_idx}"
                    surf_uid    = str(uuid.uuid4()).replace("-", "")[:8]

                    # Holes → openings
                    openings_strs = []
                    hole_pts_list = []
                    for hole_ring in inner_rings:
                        h_pts  = ring_coords(hole_ring, world_verts)
                        ftype  = classify_hole(h_pts, ground_z, door_threshold)
                        h_id   = f"op_{surf_uid}_{uuid.uuid4().hex[:6]}"
                        openings_strs.append(opening_gml(ftype, h_id, h_pts,
                                                         indent="        "))
                        hole_pts_list.append(h_pts)
                        print(f"    Hole in polygon {flat_idx} "
                              f"[{sem or 'unknown'}]: "
                              f"z_bot={min(p[2] for p in h_pts):.2f}  "
                              f"→ {ftype.upper()}")

                    # Build wall polygon GML (ext + holes as interior rings)
                    poly_gml = polygon_gml(ext_pts, hole_pts_list,
                                           f"{surf_gml_id}_poly",
                                           indent="            ")
                    ms_gml = (
                        f'        <gml:MultiSurface>\n'
                        f'          <gml:surfaceMember>\n'
                        f'{poly_gml}\n'
                        f'          </gml:surfaceMember>\n'
                        f'        </gml:MultiSurface>'
                    )

                    # Choose the right bldg surface wrapper
                    if sem == "WallSurface":
                        surf_tag = "bldg:WallSurface"
                        lod_tag  = "bldg:lod3MultiSurface"
                    elif sem == "RoofSurface":
                        surf_tag = "bldg:RoofSurface"
                        lod_tag  = "bldg:lod3MultiSurface"
                    elif sem == "GroundSurface":
                        surf_tag = "bldg:GroundSurface"
                        lod_tag  = "bldg:lod3MultiSurface"
                    else:
                        surf_tag = "bldg:WallSurface"
                        lod_tag  = "bldg:lod3MultiSurface"

                    op_block = ""
                    if openings_strs:
                        op_block = "\n" + "\n".join(openings_strs)

                    bldg_lines.append(
                        f'    <bldg:boundedBy>\n'
                        f'      <{surf_tag} gml:id="{surf_gml_id}">\n'
                        f'        <{lod_tag}>\n'
                        f'{ms_gml}\n'
                        f'        </{lod_tag}>'
                        f'{op_block}\n'
                        f'      </{surf_tag}>\n'
                        f'    </bldg:boundedBy>'
                    )

            body = "\n".join(bldg_lines)
            city_members.append(
                f'  <cityObjectMember>\n'
                f'    <{bldg_tag} gml:id="{gml_id}">\n'
                f'{body}\n'
                f'    </{bldg_tag}>\n'
                f'  </cityObjectMember>'
            )

        # ── BuildingInstallation ────────────────────────────────────
        elif obj_type == "BuildingInstallation":
            surfaces_wh = []   # (ext_pts, []) list for this installation

            for geom in obj.get("geometry", []):
                boundaries = geom.get("boundaries", [])
                geom_type  = geom.get("type", "")
                polys = (
                    [polygon for shell in boundaries for polygon in shell]
                    if geom_type == "Solid"
                    else boundaries
                )
                for polygon in polys:
                    ext_pts = ring_coords(polygon[0], world_verts)
                    surfaces_wh.append((ext_pts, []))   # installations have no holes

            if not surfaces_wh:
                continue

            ms_id  = f"{gml_id}_ms"
            ms_str = multi_surface_gml(surfaces_wh, ms_id, indent="          ")

            city_members.append(
                f'  <cityObjectMember>\n'
                f'    <bldg:BuildingInstallation gml:id="{gml_id}">\n'
                f'      <bldg:lod3Geometry>\n'
                f'{ms_str}\n'
                f'      </bldg:lod3Geometry>\n'
                f'    </bldg:BuildingInstallation>\n'
                f'  </cityObjectMember>'
            )

        # ── Window (standalone) ─────────────────────────────────────
        elif obj_type == "Window":
            surfaces_wh = []
            for geom in obj.get("geometry", []):
                for polygon in geom.get("boundaries", []):
                    ext_pts = ring_coords(polygon[0], world_verts)
                    surfaces_wh.append((ext_pts, []))
            if not surfaces_wh:
                continue
            ms_id  = f"{gml_id}_ms"
            ms_str = multi_surface_gml(surfaces_wh, ms_id, indent="          ")
            city_members.append(
                f'  <cityObjectMember>\n'
                f'    <bldg:Window gml:id="{gml_id}">\n'
                f'      <bldg:lod3MultiSurface>\n'
                f'{ms_str}\n'
                f'      </bldg:lod3MultiSurface>\n'
                f'    </bldg:Window>\n'
                f'  </cityObjectMember>'
            )

        # ── Door (standalone) ───────────────────────────────────────
        elif obj_type == "Door":
            surfaces_wh = []
            for geom in obj.get("geometry", []):
                for polygon in geom.get("boundaries", []):
                    ext_pts = ring_coords(polygon[0], world_verts)
                    surfaces_wh.append((ext_pts, []))
            if not surfaces_wh:
                continue
            ms_id  = f"{gml_id}_ms"
            ms_str = multi_surface_gml(surfaces_wh, ms_id, indent="          ")
            city_members.append(
                f'  <cityObjectMember>\n'
                f'    <bldg:Door gml:id="{gml_id}">\n'
                f'      <bldg:lod3MultiSurface>\n'
                f'{ms_str}\n'
                f'      </bldg:lod3MultiSurface>\n'
                f'    </bldg:Door>\n'
                f'  </cityObjectMember>'
            )

        # ── Unknown / other — emit as GenericCityObject ────────────
        else:
            surfaces_wh = []
            for geom in obj.get("geometry", []):
                for polygon in geom.get("boundaries", []):
                    if isinstance(polygon, list) and polygon:
                        ext_pts = ring_coords(polygon[0], world_verts)
                        surfaces_wh.append((ext_pts, []))
            if not surfaces_wh:
                continue
            ms_id  = f"{gml_id}_ms"
            ms_str = multi_surface_gml(surfaces_wh, ms_id, indent="        ")
            city_members.append(
                f'  <cityObjectMember>\n'
                f'    <bldg:BuildingInstallation gml:id="{gml_id}">\n'
                f'      <bldg:lod3Geometry>\n'
                f'{ms_str}\n'
                f'      </bldg:lod3Geometry>\n'
                f'    </bldg:BuildingInstallation>\n'
                f'  </cityObjectMember>'
            )

    body = "\n".join(city_members)
    return f"{GML_HEADER}\n{body}\n{GML_FOOTER}\n"


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="CityJSON → CityGML LOD3 Converter with hole-to-opening support")
    parser.add_argument("--door_threshold", type=float,
                        default=DOOR_HEIGHT_THRESHOLD_DEFAULT,
                        help="Height above ground below which a wall hole is "
                             "classified as a Door (default: 0.3 m)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output GML filename (in outputs/15_flat_gml/)")
    args = parser.parse_args()

    print("=" * 60)
    print("  CityJSON → CityGML LOD3 Converter")
    print("=" * 60)

    # ── 1. Select CityJSON input ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SELECT CITYJSON FILE")
    print(f"{'='*60}")
    json_path = select_file(JSON_DIR, "*.json")

    print(f"\n  Loading {os.path.basename(json_path)} ...")
    with open(json_path, "r", encoding="utf-8") as fh:
        cm = json.load(fh)

    world_verts = decode_vertices(cm)
    print(f"  {len(cm.get('vertices', []))} vertices decoded.")
    print(f"  {len(cm.get('CityObjects', {}))} CityObjects found.")

    # ── 2. Count holes ─────────────────────────────────────────────────
    n_holes = 0
    for obj in cm.get("CityObjects", {}).values():
        for geom in obj.get("geometry", []):
            for poly in _flatten_polys(geom):
                n_holes += len(poly) - 1   # rings beyond exterior = holes
    print(f"  Wall holes (interior rings) detected: {n_holes}")

    # ── 3. Convert ─────────────────────────────────────────────────────
    print(f"\n  Converting to CityGML (door_threshold={args.door_threshold} m) ...")
    gml_str = cityjson_to_gml_str(cm, world_verts, args.door_threshold)

    # ── 4. Save ────────────────────────────────────────────────────────
    if args.output:
        out_name = (args.output if args.output.endswith(".gml")
                    else args.output + ".gml")
    else:
        base     = os.path.splitext(os.path.basename(json_path))[0]
        out_name = f"{base}.gml"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, out_name)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(gml_str)

    mb = os.path.getsize(out_path) / 1_048_576
    print(f"\n  Saved → {out_path}  ({mb:.2f} MB)")
    print(f"\n{'='*60}")
    print("  Done!")
    print(f"{'='*60}\n")


# ─── Helper: flatten all polygon lists from a geometry ───────────────────────
def _flatten_polys(geom):
    """Yield every polygon from a CityJSON geometry regardless of type."""
    boundaries = geom.get("boundaries", [])
    gtype      = geom.get("type", "")
    if gtype == "Solid":
        for shell in boundaries:
            yield from shell
    elif gtype in ("MultiSurface", "CompositeSurface"):
        yield from boundaries
    elif gtype == "MultiPolygon":
        yield from boundaries


if __name__ == "__main__":
    main()
