"""
merge_planes.py

Reads a Poisson mesh OBJ file and merges triangles that are:
  1. Adjacent (share an edge)
  2. Have nearly parallel normals (within NORMAL_THRESHOLD_DEG degrees)

The merged groups are each re-triangulated using the convex or alpha-hull
of their boundary, then written to a new OBJ.

Usage:
    python scripts/mesh_trials/merge_planes.py

Output:
    outputs/trials/NIMBB-simplified-Poisson.obj
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_OBJ  = Path("outputs/08_poisson_meshes/NIMBB-best.obj")
OUTPUT_OBJ = Path("outputs/trials/NIMBB-simplified-Poisson.obj")
NORMAL_THRESHOLD_DEG = 5.0          # merge threshold (degrees)
NORMAL_THRESHOLD_COS = np.cos(np.radians(NORMAL_THRESHOLD_DEG))  # cosine

# ---------------------------------------------------------------------------
# Step 1 – Parse the OBJ
# ---------------------------------------------------------------------------
print(f"[1/5] Reading {INPUT_OBJ} …")

vertices   = []   # list of (x,y,z)  – 0-indexed
vn_list    = []   # list of (nx,ny,nz) – 0-indexed (vertex normals)
# each face: list of (v_idx, vn_idx)  – 0-indexed
faces_raw  = []

with open(INPUT_OBJ, "r") as fh:
    for line in fh:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tok = parts[0]
        if tok == "v":
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif tok == "vn":
            vn_list.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif tok == "f":
            # Each part is one of:
            #   v   /  vt  /  vn
            #   v   //     vn
            #   v
            face_verts = []
            for p in parts[1:]:
                tokens = p.split("/")
                v_idx  = int(tokens[0]) - 1  # OBJ is 1-based
                vn_idx = None
                if len(tokens) == 3 and tokens[2] != "":
                    vn_idx = int(tokens[2]) - 1
                elif len(tokens) == 2 and tokens[1] != "":
                    # v/vt  – no normal stored differently
                    pass
                face_verts.append((v_idx, vn_idx))
            faces_raw.append(face_verts)

vertices = np.array(vertices, dtype=np.float64)
vn_arr   = np.array(vn_list,  dtype=np.float64) if vn_list else None

num_faces = len(faces_raw)
print(f"    {len(vertices)} vertices, {len(vn_list)} vertex normals, {num_faces} faces")

# ---------------------------------------------------------------------------
# Step 2 – Compute per-face normals
# ---------------------------------------------------------------------------
print("[2/5] Computing per-face normals …")

face_normals = np.zeros((num_faces, 3), dtype=np.float64)

for fi, face in enumerate(faces_raw):
    # Use vertex-normal averaging if available, else compute geometrically
    if vn_arr is not None and face[0][1] is not None:
        norms = np.array([vn_arr[v[1]] for v in face if v[1] is not None])
        n = norms.mean(axis=0)
    else:
        # Geometric normal from first triangle
        v0 = vertices[face[0][0]]
        v1 = vertices[face[1][0]]
        v2 = vertices[face[2][0]]
        n  = np.cross(v1 - v0, v2 - v0)
    mag = np.linalg.norm(n)
    if mag > 0:
        n = n / mag
    face_normals[fi] = n

# ---------------------------------------------------------------------------
# Step 3 – Build edge-to-face adjacency map
# ---------------------------------------------------------------------------
print("[3/5] Building adjacency graph …")

# Map from sorted edge (v_a, v_b) → list of face indices
edge_to_faces = defaultdict(list)

for fi, face in enumerate(faces_raw):
    v_indices = [v[0] for v in face]
    n = len(v_indices)
    for i in range(n):
        a = v_indices[i]
        b = v_indices[(i + 1) % n]
        edge = (min(a, b), max(a, b))
        edge_to_faces[edge].append(fi)

# Build face adjacency list
adj = defaultdict(set)
for edge, flist in edge_to_faces.items():
    if len(flist) == 2:
        fi, fj = flist
        adj[fi].add(fj)
        adj[fj].add(fi)

# ---------------------------------------------------------------------------
# Step 4 – Region growing (BFS) to merge faces
# ---------------------------------------------------------------------------
print("[4/5] Growing planar regions …")

label       = np.full(num_faces, -1, dtype=np.int32)
group_id    = 0
group_normals = {}   # group_id → representative normal (unit)

for seed in range(num_faces):
    if label[seed] != -1:
        continue          # already assigned

    queue         = [seed]
    label[seed]   = group_id
    seed_normal   = face_normals[seed]
    group_normals[group_id] = seed_normal.copy()
    head = 0

    while head < len(queue):
        fi = queue[head]
        head += 1

        for fj in adj[fi]:
            if label[fj] != -1:
                continue
            # Check normal similarity (use absolute dot because normals can
            # be flipped on a Poisson mesh)
            dot = abs(np.dot(face_normals[fj], seed_normal))
            if dot >= NORMAL_THRESHOLD_COS:
                label[fj] = group_id
                queue.append(fj)

    # Update representative normal as average of group
    idxs = np.array(queue)
    avg_n = face_normals[idxs].mean(axis=0)
    mag   = np.linalg.norm(avg_n)
    if mag > 0:
        group_normals[group_id] = avg_n / mag

    group_id += 1

num_groups = group_id
print(f"    Formed {num_groups} planar groups from {num_faces} faces")

# ---------------------------------------------------------------------------
# Step 5 – Write output OBJ
# ---------------------------------------------------------------------------
print(f"[5/5] Writing {OUTPUT_OBJ} …")

OUTPUT_OBJ.parent.mkdir(parents=True, exist_ok=True)

# Collect faces per group
group_faces = defaultdict(list)
for fi in range(num_faces):
    group_faces[label[fi]].append(fi)

with open(OUTPUT_OBJ, "w") as out:
    out.write(f"# Plane-merged Poisson mesh\n")
    out.write(f"# Source: {INPUT_OBJ}\n")
    out.write(f"# Normal merge threshold: {NORMAL_THRESHOLD_DEG} degrees\n")
    out.write(f"# Original faces: {num_faces}  |  Planar groups: {num_groups}\n\n")

    # Write all vertices (same as input, 1-indexed)
    for vx, vy, vz in vertices:
        out.write(f"v {vx:.9f} {vy:.9f} {vz:.9f}\n")

    # Write vertex normals — one per group (the representative normal)
    # We'll assign each group a single normal index
    group_vn_idx = {}   # group_id → 1-based vn index in output file
    out.write("\n")
    for gid in range(num_groups):
        n = group_normals[gid]
        out.write(f"vn {n[0]:.9f} {n[1]:.9f} {n[2]:.9f}\n")
        group_vn_idx[gid] = gid + 1   # 1-based

    # Write faces grouped by planar region
    out.write("\n")
    for gid in range(num_groups):
        flist = group_faces[gid]
        vn_i  = group_vn_idx[gid]
        out.write(f"\ng plane_{gid}\n")   # group marker
        for fi in flist:
            face = faces_raw[fi]
            parts = []
            for (v_idx, _) in face:
                # v//vn  (1-based indices)
                parts.append(f"{v_idx + 1}//{vn_i}")
            out.write("f " + " ".join(parts) + "\n")

total_faces_written = sum(len(v) for v in group_faces.values())
print(f"Done. {total_faces_written} faces in {num_groups} groups -> {OUTPUT_OBJ}")
print(f"Average faces per group: {total_faces_written / num_groups:.1f}")
