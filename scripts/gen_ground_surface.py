#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gen_ground_surface.py

Add a single ground-surface (convex hull of all building footprints)
to an existing LOD-2 CityGML file – even if it has NO GroundSurface.

Input:  D:\Projects\Thesis\data\LOD2_NIMBB_tracedfootprint.gml\LOD2_NIMBB_tracedfootprint.gml
Output: D:\Projects\Thesis\data\LOD2_NIMBB_tracedfootprint_fixed.gml
"""

import sys
from pathlib import Path
from lxml import etree
from shapely.geometry import MultiPoint
from shapely.ops import unary_union

# ----------------------------------------------------------------------
# CONFIG - KEEP YOUR ORIGINAL FOLDER LOGIC
# ----------------------------------------------------------------------
DATA_FOLDER = Path(r"D:\Projects\Thesis\data")
folder_name = "LOD2_NIMBB_tracedfootprint.gml"
input_folder = DATA_FOLDER / folder_name

if not input_folder.is_dir():
    sys.exit(f"Folder not found: {input_folder}")

gml_files = list(input_folder.glob("*.gml"))
if not gml_files:
    sys.exit(f"No .gml file inside {input_folder}")
if len(gml_files) > 1:
    print("Multiple .gml files found:")
    for f in gml_files:
        print(f"  - {f.name}")
    sys.exit("Keep only one .gml file.")

INPUT_GML  = gml_files[0]
OUTPUT_GML = DATA_FOLDER / "LOD2_NIMBB_tracedfootprint_fixed.gml"

print(f"Found input: {INPUT_GML}")

# ----------------------------------------------------------------------
# CityGML namespaces
# ----------------------------------------------------------------------
NS = {
    "gml":  "http://www.opengis.net/gml",
    "bldg": "http://www.opengis.net/citygml/building/2.0",
    "core": "http://www.opengis.net/citygml/2.0",
}

def qname(prefix, local):
    return etree.QName(NS[prefix], local)

# ----------------------------------------------------------------------
# 1. Parse the CityGML file
# ----------------------------------------------------------------------
try:
    tree = etree.parse(str(INPUT_GML))
    root = tree.getroot()
except Exception as e:
    sys.exit(f"Failed to parse GML file: {e}")

# ----------------------------------------------------------------------
# 2. Collect ALL exterior LinearRings from building footprints
# ----------------------------------------------------------------------
footprint_points = []

# Look for ANY <gml:posList> inside a building's polygon (Ground, Wall, Roof, Solid, etc.)
poslists = root.xpath(
    "//bldg:Building//gml:posList | "
    "//bldg:Building//gml:pos",  # fallback for <gml:pos> instead of posList
    namespaces=NS
)

if not poslists:
    sys.exit("No geometry found in any building – check CityGML structure.")

for elem in poslists:
    text = elem.text
    if not text:
        continue
    coords = [float(v) for v in text.strip().split()]
    for i in range(0, len(coords), 3):
        if i + 2 < len(coords):
            x, y, z = coords[i], coords[i+1], coords[i+2]
            footprint_points.append((x, y, z))

if not footprint_points:
    sys.exit("No valid 3D coordinates found in posList.")

print(f"Collected {len(footprint_points)} points from building footprints.")

# ----------------------------------------------------------------------
# 3. Compute convex hull in XY, use lowest Z
# ----------------------------------------------------------------------
xy_points = [(x, y) for x, y, _ in footprint_points]
z_values = [z for _, _, z in footprint_points]

hull_2d = MultiPoint(xy_points).convex_hull
if hull_2d.geom_type != "Polygon":
    sys.exit("Convex hull is not a polygon (degenerate case).")

ground_z = min(z_values)
exterior = list(hull_2d.exterior.coords)
poslist_flat = [coord for pt in exterior for coord in (pt[0], pt[1], ground_z)]

# ----------------------------------------------------------------------
# 4. Create new GroundSurface
# ----------------------------------------------------------------------
new_gs = etree.Element(qname("bldg", "GroundSurface"),
                       attrib={qname("gml", "id"): "GroundSurface-Gen-0001"})

poly = etree.SubElement(new_gs, qname("gml", "Polygon"))
ext = etree.SubElement(poly, qname("gml", "exterior"))
ring = etree.SubElement(ext, qname("gml", "LinearRing"))
pos = etree.SubElement(ring, qname("gml", "posList"))
pos.text = " ".join(f"{v:.6f}" for v in poslist_flat)

# ----------------------------------------------------------------------
# 5. Insert into first building
# ----------------------------------------------------------------------
building = root.find(".//bldg:Building", namespaces=NS)
if building is not None:
    building.append(new_gs)
else:
    # Fallback: append to core:CityModel
    citymodel = root.find(".//core:CityModel", namespaces=NS)
    if citymodel is not None:
        citymodel.append(new_gs)
    else:
        root.append(new_gs)

# ----------------------------------------------------------------------
# 6. Write output
# ----------------------------------------------------------------------
tree.write(
    str(OUTPUT_GML),
    xml_declaration=True,
    encoding="UTF-8",
    pretty_print=True,
)

print(f"\nSUCCESS! Fixed file written to:")
print(f"   {OUTPUT_GML}")
print(f"   New ground surface with {len(exterior)} vertices at Z = {ground_z:.3f}")