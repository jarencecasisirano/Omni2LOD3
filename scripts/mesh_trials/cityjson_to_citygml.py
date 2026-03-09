"""
cityjson_to_citygml.py

Converts a CityJSON 1.1 file with one Building CityObject (MultiSurface LOD2)
into a CityGML 2.0 file.

Usage:
    python scripts/mesh_trials/cityjson_to_citygml.py

Input :  outputs/trials/NIMBB-simplified-Poisson.city.json
Output:  outputs/final/NIMBB.gml
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_JSON  = Path("outputs/trials/NIMBB-simplified-Poisson.city.json")
OUTPUT_GML  = Path("outputs/final/NIMBB.gml")

# CRS — adjust EPSG if the point cloud uses a different projection.
# The coordinates are in the ~292000, 1620000 range which matches EPSG:32651
# (WGS 84 / UTM zone 51N) or a Philippine reference.  Change as needed.
SRSNAME = "urn:ogc:def:crs:EPSG::32651"

# ---------------------------------------------------------------------------
# Step 1 – Load CityJSON
# ---------------------------------------------------------------------------
print(f"[1/4] Reading {INPUT_JSON} ...")

with open(INPUT_JSON, "r") as f:
    cj = json.load(f)

scale     = cj["transform"]["scale"]        # [sx, sy, sz]
translate = cj["transform"]["translate"]    # [tx, ty, tz]
cj_verts  = cj["vertices"]                 # [[ix, iy, iz], ...]

sx, sy, sz = scale
tx, ty, tz = translate

def decode_vertex(iv):
    """Convert integer CityJSON vertex back to real-world coordinates."""
    return (iv[0] * sx + tx,
            iv[1] * sy + ty,
            iv[2] * sz + tz)

print(f"    {len(cj_verts)} vertices, {len(cj['CityObjects'])} CityObjects")

# ---------------------------------------------------------------------------
# Step 2 – Decode all vertices up front for fast lookup
# ---------------------------------------------------------------------------
print("[2/4] Decoding vertices ...")

real_verts = [decode_vertex(iv) for iv in cj_verts]

# ---------------------------------------------------------------------------
# Step 3 – Compute bounding envelope
# ---------------------------------------------------------------------------
xs = [v[0] for v in real_verts]
ys = [v[1] for v in real_verts]
zs = [v[2] for v in real_verts]
env_lower = f"{min(xs):.6f} {min(ys):.6f} {min(zs):.6f}"
env_upper = f"{max(xs):.6f} {max(ys):.6f} {max(zs):.6f}"

# ---------------------------------------------------------------------------
# Step 4 – Stream-write CityGML XML
# ---------------------------------------------------------------------------
print(f"[3/4] Writing {OUTPUT_GML} ...")

OUTPUT_GML.parent.mkdir(parents=True, exist_ok=True)

# We stream-write the XML line by line to keep memory usage low.
def coords_str(ring_indices):
    """Turn a ring (list of vertex indices) into a GML posList string.
    CityGML LinearRings must be closed (last point == first point).

    The CityJSON boundaries from obj_to_cityjson store each triangle as
    [[[v0, v1, v2]]] so when iterating surface->ring, 'ring_indices' may
    itself be a list-of-lists (e.g. [[v0, v1, v2]]).  Flatten if needed.
    """
    # Flatten one extra nesting level if the first element is a list
    if ring_indices and isinstance(ring_indices[0], list):
        flat = []
        for subring in ring_indices:
            flat.extend(subring)
        ring_indices = flat

    pts = [real_verts[i] for i in ring_indices]
    pts_closed = pts + [pts[0]]          # close the ring
    return " ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in pts_closed)

with open(OUTPUT_GML, "w", encoding="utf-8") as out:
    # -----------------------------------------------------------------------
    # XML header & root element
    # -----------------------------------------------------------------------
    out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    out.write(
        '<CityModel\n'
        '  xmlns="http://www.opengis.net/citygml/2.0"\n'
        '  xmlns:bldg="http://www.opengis.net/citygml/building/2.0"\n'
        '  xmlns:gml="http://www.opengis.net/gml"\n'
        '  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
        '  xsi:schemaLocation="\n'
        '    http://www.opengis.net/citygml/2.0\n'
        '    http://schemas.opengis.net/citygml/2.0/cityGMLBase.xsd\n'
        '    http://www.opengis.net/citygml/building/2.0\n'
        '    http://schemas.opengis.net/citygml/building/2.0/building.xsd">\n\n'
    )

    # -----------------------------------------------------------------------
    # Bounding envelope
    # -----------------------------------------------------------------------
    out.write('  <gml:boundedBy>\n')
    out.write(f'    <gml:Envelope srsName="{SRSNAME}" srsDimension="3">\n')
    out.write(f'      <gml:lowerCorner>{env_lower}</gml:lowerCorner>\n')
    out.write(f'      <gml:upperCorner>{env_upper}</gml:upperCorner>\n')
    out.write('    </gml:Envelope>\n')
    out.write('  </gml:boundedBy>\n\n')

    # -----------------------------------------------------------------------
    # Iterate over CityObjects
    # -----------------------------------------------------------------------
    for obj_id, obj in cj["CityObjects"].items():
        obj_type = obj.get("type", "Building")

        # Map CityJSON type → CityGML element
        gml_tag = "bldg:Building"   # default; extend if needed

        out.write('  <cityObjectMember>\n')
        safe_id = obj_id.replace(" ", "_").replace("/", "_")
        out.write(f'    <{gml_tag} gml:id="{safe_id}">\n')

        # -- Attributes (optional metadata) ----------------------------------
        attrs = obj.get("attributes", {})
        if attrs.get("description"):
            out.write(f'      <gml:description>{attrs["description"]}</gml:description>\n')
        if attrs.get("sourceFile"):
            out.write(f'      <gml:name>{attrs["sourceFile"]}</gml:name>\n')

        # -- Geometry --------------------------------------------------------
        for geom in obj.get("geometry", []):
            geom_type = geom.get("type")
            lod       = str(geom.get("lod", "2"))

            if geom_type in ("MultiSurface", "CompositeSurface"):
                out.write(f'      <bldg:lod{lod}MultiSurface>\n')
                out.write(f'        <gml:MultiSurface srsName="{SRSNAME}" srsDimension="3">\n')

                boundaries = geom["boundaries"]
                total      = len(boundaries)
                for idx, surface in enumerate(boundaries):
                    if (idx + 1) % 10000 == 0:
                        print(f"        writing surface {idx+1}/{total} ...")

                    out.write('          <gml:surfaceMember>\n')
                    out.write('            <gml:Polygon>\n')

                    for ring_idx, ring in enumerate(surface):
                        ring_tag = "gml:exterior" if ring_idx == 0 else "gml:interior"
                        out.write(f'              <{ring_tag}>\n')
                        out.write('                <gml:LinearRing>\n')
                        out.write(f'                  <gml:posList>{coords_str(ring)}</gml:posList>\n')
                        out.write('                </gml:LinearRing>\n')
                        out.write(f'              </{ring_tag}>\n')

                    out.write('            </gml:Polygon>\n')
                    out.write('          </gml:surfaceMember>\n')

                out.write('        </gml:MultiSurface>\n')
                out.write(f'      </bldg:lod{lod}MultiSurface>\n')

            elif geom_type == "Solid":
                # Solid: boundaries[0] = outer shell, boundaries[1..] = holes
                out.write(f'      <bldg:lod{lod}Solid>\n')
                out.write(f'        <gml:Solid srsName="{SRSNAME}" srsDimension="3">\n')
                for shell_idx, shell in enumerate(geom["boundaries"]):
                    shell_tag = "gml:exterior" if shell_idx == 0 else "gml:interior"
                    out.write(f'          <{shell_tag}>\n')
                    out.write('            <gml:CompositeSurface>\n')
                    for surface in shell:
                        out.write('              <gml:surfaceMember>\n')
                        out.write('                <gml:Polygon>\n')
                        for ring_idx, ring in enumerate(surface):
                            ring_tag = "gml:exterior" if ring_idx == 0 else "gml:interior"
                            out.write(f'                  <{ring_tag}>\n')
                            out.write('                    <gml:LinearRing>\n')
                            out.write(f'                      <gml:posList>{coords_str(ring)}</gml:posList>\n')
                            out.write('                    </gml:LinearRing>\n')
                            out.write(f'                  </{ring_tag}>\n')
                        out.write('                </gml:Polygon>\n')
                        out.write('              </gml:surfaceMember>\n')
                    out.write('            </gml:CompositeSurface>\n')
                    out.write(f'          </{shell_tag}>\n')
                out.write('        </gml:Solid>\n')
                out.write(f'      </bldg:lod{lod}Solid>\n')

        out.write(f'    </{gml_tag}>\n')
        out.write('  </cityObjectMember>\n\n')

    out.write('</CityModel>\n')

size_mb = OUTPUT_GML.stat().st_size / 1e6
print(f"[4/4] Done. {size_mb:.2f} MB -> {OUTPUT_GML}")
