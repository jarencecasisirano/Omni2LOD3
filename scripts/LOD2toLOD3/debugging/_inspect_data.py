"""Inspect the LAS files and CityJSON to diagnose the ghost building installations."""
import json
import numpy as np
import laspy
import os

# ── 1. Inspect point clouds ──────────────────────────────────────────
for name in ["windows_curve-2.las", "doors_curve-2.las"]:
    path = os.path.join("outputs/11A_facade_curve", name)
    if not os.path.exists(path):
        print(f"NOT FOUND: {path}")
        continue
    las = laspy.read(path)
    pts = np.vstack((las.x, las.y, las.z)).T
    print(f"\n{'='*60}")
    print(f"  {name}: {len(pts)} points")
    print(f"{'='*60}")
    print(f"  X range: {pts[:,0].min():.3f}  to  {pts[:,0].max():.3f}")
    print(f"  Y range: {pts[:,1].min():.3f}  to  {pts[:,1].max():.3f}")
    print(f"  Z range: {pts[:,2].min():.3f}  to  {pts[:,2].max():.3f}")
    print(f"  Centroid: {pts.mean(axis=0)}")

# ── 2. Inspect CityJSON building surfaces ─────────────────────────────
json_dir = "outputs/14_extrusions_json"
json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
for jf in json_files:
    jp = os.path.join(json_dir, jf)
    with open(jp) as fh:
        cm = json.load(fh)
    t = cm.get("transform", {})
    s = np.array(t.get("scale", [1,1,1]), dtype=np.float64)
    tr = np.array(t.get("translate", [0,0,0]), dtype=np.float64)
    verts = np.array(cm["vertices"], dtype=np.float64) * s + tr
    
    print(f"\n{'='*60}")
    print(f"  CityJSON: {jf}")
    print(f"{'='*60}")
    print(f"  Vertices: {len(verts)}")
    print(f"  X range: {verts[:,0].min():.3f}  to  {verts[:,0].max():.3f}")
    print(f"  Y range: {verts[:,1].min():.3f}  to  {verts[:,1].max():.3f}")
    print(f"  Z range: {verts[:,2].min():.3f}  to  {verts[:,2].max():.3f}")
    
    for obj_id, obj in cm.get("CityObjects", {}).items():
        obj_type = obj.get("type", "")
        print(f"  Object '{obj_id}': type={obj_type}")

# ── 3. Inspect the output file ────────────────────────────────────────
out_dir = "outputs/12_curve_json"
out_files = [f for f in os.listdir(out_dir) if "curved" in f and f.endswith(".json")]
for of in out_files:
    op = os.path.join(out_dir, of)
    with open(op) as fh:
        cm_out = json.load(fh)
    
    print(f"\n{'='*60}")
    print(f"  OUTPUT: {of}")
    print(f"{'='*60}")
    
    t = cm_out.get("transform", {})
    s = np.array(t.get("scale", [1,1,1]), dtype=np.float64)
    tr = np.array(t.get("translate", [0,0,0]), dtype=np.float64)
    verts_out = np.array(cm_out["vertices"], dtype=np.float64) * s + tr
    
    for obj_id, obj in cm_out.get("CityObjects", {}).items():
        obj_type = obj.get("type", "")
        if obj_type in ("Window", "Door", "BuildingInstallation"):
            # Get the geometry bounds
            geoms = obj.get("geometry", [])
            for g in geoms:
                all_idxs = []
                for boundary in g.get("boundaries", []):
                    for ring in boundary:
                        if isinstance(ring[0], list):
                            for sub_ring in ring:
                                all_idxs.extend(sub_ring)
                        else:
                            all_idxs.extend(ring)
                if all_idxs:
                    coords = verts_out[np.array(all_idxs)]
                    cx, cy, cz = coords.mean(axis=0)
                    print(f"  {obj_type:25s} '{obj_id}'  centroid=({cx:.3f}, {cy:.3f}, {cz:.3f})")
