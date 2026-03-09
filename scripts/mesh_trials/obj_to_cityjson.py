"""
obj_to_cityjson.py

Converts the plane-grouped OBJ produced by merge_planes.py into a
CityJSON 1.1 file containing one Building CityObject.

Each 'g plane_N' group from the OBJ becomes one MultiSurface polygon
(a collection of triangles) under a single Building geometry.

Usage:
    python scripts/mesh_trials/obj_to_cityjson.py

Output:
    outputs/trials/NIMBB-simplified-Poisson.city.json
"""

import json
import math
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_OBJ   = Path("outputs/trials/NIMBB-simplified-Poisson.obj")
OUTPUT_JSON = Path("outputs/trials/NIMBB-simplified-Poisson.city.json")

# Precision for the integer-coordinate transform (metres)
SCALE = 0.001   # 1 mm precision

# ---------------------------------------------------------------------------
# Step 1 – Parse the OBJ
# ---------------------------------------------------------------------------
print(f"[1/4] Reading {INPUT_OBJ} ...")

vertices_raw = []        # list of [x, y, z]  (float, 0-indexed)
groups       = {}        # group_name -> list of face tuples
current_group = "__default__"

with open(INPUT_OBJ, "r") as fh:
    for line in fh:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tok = parts[0]

        if tok == "v":
            vertices_raw.append([float(parts[1]),
                                  float(parts[2]),
                                  float(parts[3])])
        elif tok == "g":
            current_group = parts[1] if len(parts) > 1 else "__default__"
            if current_group not in groups:
                groups[current_group] = []
        elif tok == "f":
            face_v = []
            for p in parts[1:]:
                tokens = p.split("/")
                face_v.append(int(tokens[0]) - 1)   # 0-indexed vertex
            if current_group not in groups:
                groups[current_group] = []
            groups[current_group].append(face_v)

print(f"    {len(vertices_raw)} vertices, {len(groups)} plane groups")

# ---------------------------------------------------------------------------
# Step 2 – Build compact CityJSON vertex list with integer transform
# ---------------------------------------------------------------------------
print("[2/4] Building CityJSON vertex table ...")

import numpy as np

verts = np.array(vertices_raw, dtype=np.float64)

# Translation: use minimum bounding box corner so all offsets are positive
tx = float(verts[:, 0].min())
ty = float(verts[:, 1].min())
tz = float(verts[:, 2].min())
translate = [tx, ty, tz]

# Convert to integer-coded coordinates
# CityJSON vertex = round((coord - translate) / scale)
int_verts = np.round((verts - np.array([tx, ty, tz])) / SCALE).astype(np.int64)

# CityJSON wants each vertex as [ix, iy, iz]
cj_vertices = int_verts.tolist()

print(f"    Transform: scale={SCALE}, translate=[{tx:.3f}, {ty:.3f}, {tz:.3f}]")

# ---------------------------------------------------------------------------
# Step 3 – Build MultiSurface geometry boundaries
# ---------------------------------------------------------------------------
print("[3/4] Building geometry boundaries ...")

# CityJSON MultiSurface boundaries:
#   [ surface1, surface2, ... ]
#   surface  = [ outer_ring, *inner_rings ]   (usually just outer ring)
#   ring     = [ vi0, vi1, vi2, ... ]   (vertex indices, not repeated last)

boundaries  = []   # one entry per plane group
surface_map = {}   # group_name -> index in boundaries

group_names = sorted(groups.keys(),
                     key=lambda n: int(n.split("_")[1]) if "_" in n and n.split("_")[1].isdigit() else 0)

for gname in group_names:
    faces = groups[gname]
    for face_verts in faces:
        # Each triangle → one polygon surface [[v0, v1, v2]]
        boundaries.append([[face_verts]])

print(f"    {len(boundaries)} surface polygons built")

# ---------------------------------------------------------------------------
# Step 4 – Assemble CityJSON and write
# ---------------------------------------------------------------------------
print(f"[4/4] Writing {OUTPUT_JSON} ...")

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

cityjson = {
    "type": "CityJSON",
    "version": "1.1",
    "transform": {
        "scale": [SCALE, SCALE, SCALE],
        "translate": translate
    },
    "CityObjects": {
        "NIMBB-Building": {
            "type": "Building",
            "attributes": {
                "description": "Plane-merged Poisson mesh",
                "sourceFile": str(INPUT_OBJ)
            },
            "geometry": [
                {
                    "type": "MultiSurface",
                    "lod": "2",
                    "boundaries": boundaries
                }
            ]
        }
    },
    "vertices": cj_vertices
}

with open(OUTPUT_JSON, "w") as out:
    json.dump(cityjson, out, separators=(",", ":"))   # compact JSON

size_mb = OUTPUT_JSON.stat().st_size / 1e6
print(f"Done. {size_mb:.2f} MB -> {OUTPUT_JSON}")
print(f"  CityObjects : 1 Building")
print(f"  Surfaces    : {len(boundaries)}")
print(f"  Vertices    : {len(cj_vertices)}")
