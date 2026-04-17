#!/usr/bin/env python3
"""
CityJSON LOD3: Facade Extrusions — Convex Hull (BuildingInstallations).

Takes a CityJSON LOD3 file and a user-selected .las point cloud from
outputs/11B_flat.

Pipeline
--------
1. Extract the building 2-D XY footprint (GroundSurface → convex hull).
2. Parse all near-vertical wall surfaces for normal/tangent alignment.
3. DBSCAN-cluster the entire selected point cloud.
4. For each cluster:
      a. Find the nearest wall surface for axis alignment.
      b. Project ALL cluster points onto the wall outward-normal axis.
      c. Filter to exterior points (d ≥ d_wall) — interior portion discarded.
      d. If none protrude outward, skip the cluster.
      e. Compute the 3-D convex hull of the exterior points.
      f. Add a planar back-wall cap polygon at d_wall that closes the hull
         against the building facade.
      g. Append a BuildingInstallation MultiSurface with the hull faces.
5. Save the enriched CityJSON.

Usage
-----
    conda activate las-env
    python scripts/LOD2toLOD3/11B_extrusions_CH.py [-o OUTPUT]

Options
-------
    --eps              DBSCAN neighbourhood radius   (default 0.3 m)
    --min_samples      DBSCAN minimum cluster size   (default 30)
    --min_protrusion   Min outward protrusion to keep (default 0.01 m)
    --min_hull_pts     Min exterior points for hull   (default 4)
    --output, -o       Output filename (in outputs/15_extrusions_ch_json/).
"""

import os
import sys
import glob
import json
import uuid
import argparse

import numpy as np
import laspy
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

try:
    from shapely.geometry import MultiPoint, Polygon as SPolygon, Point
    from shapely.prepared import prep
    from shapely.validation import make_valid
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    print("WARNING: shapely not installed — exterior filtering is disabled.\n"
          "  Install: conda install -c conda-forge shapely")


# ─── Directories ──────────────────────────────────────────────────────────────
LAS_DIR    = "outputs/11B_flat"
JSON_DIR   = "outputs/13_intrusions_json"
OUTPUT_DIR = "outputs/14_extrusions_json"

# ─── Tuning constants ─────────────────────────────────────────────────────────
VERTICAL_TOL = 0.3   # |n_z / |n|| > this → near-horizontal surface, skip


# =====================================================================
# Interactive file selector
# =====================================================================
def select_file(directory, pattern="*.las"):
    """Interactive single-file selector."""
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


def encode_vertex(pt, scale, translate):
    return [round((pt[i] - translate[i]) / scale[i]) for i in range(3)]


# =====================================================================
# Building footprint extraction
# =====================================================================
_SKIP_TYPES = {"Window", "Door", "BuildingInstallation", "OtherConstruction"}


def _flatten_indices(x):
    if isinstance(x, list):
        for y in x:
            yield from _flatten_indices(y)
    elif isinstance(x, int):
        yield x


def extract_building_footprint(cm, world_verts):
    """
    Return a shapely Polygon for the building XY footprint.
    Priority: GroundSurface semantic → XY convex hull fallback.
    """
    if not HAS_SHAPELY:
        return None

    for obj_id, obj in cm.get("CityObjects", {}).items():
        if obj.get("type", "") in _SKIP_TYPES:
            continue
        for geom in obj.get("geometry", []):
            sem          = geom.get("semantics", {})
            sem_surfaces = sem.get("surfaces", [])
            sem_values   = sem.get("values",   [])
            boundaries   = geom.get("boundaries", [])
            geom_type    = geom.get("type", "")

            if geom_type not in ("Solid", "MultiSurface"):
                continue

            flat = []
            if geom_type == "Solid":
                offset = 0
                for shell in boundaries:
                    for p_idx, polygon in enumerate(shell):
                        flat.append((offset + p_idx, polygon))
                    offset += len(shell)
            else:
                for p_idx, polygon in enumerate(boundaries):
                    flat.append((p_idx, polygon))

            for global_p, polygon in flat:
                try:
                    sem_type = sem_surfaces[sem_values[global_p]].get("type", "")
                except (IndexError, KeyError, TypeError):
                    sem_type = ""
                if sem_type != "GroundSurface":
                    continue
                try:
                    coords = world_verts[np.array(polygon[0])]
                    xy     = [(float(c[0]), float(c[1])) for c in coords]
                    poly   = make_valid(SPolygon(xy))
                    if not poly.is_empty and poly.area > 0:
                        print(f"  Footprint from GroundSurface: {poly.area:.1f} m²")
                        return poly
                except Exception:
                    pass

    # Fallback: convex hull of all building vertices
    all_xy = []
    for obj_id, obj in cm.get("CityObjects", {}).items():
        if obj.get("type", "") in _SKIP_TYPES:
            continue
        for geom in obj.get("geometry", []):
            for vi in set(_flatten_indices(geom.get("boundaries", []))):
                if 0 <= vi < len(world_verts):
                    all_xy.append((float(world_verts[vi, 0]),
                                   float(world_verts[vi, 1])))
    if len(all_xy) >= 3:
        hull = make_valid(MultiPoint(all_xy).convex_hull)
        if not hull.is_empty:
            print(f"  Footprint from convex hull: {hull.area:.1f} m²")
            return hull

    print("  WARNING: could not determine building footprint.")
    return None


# =====================================================================
# Parse vertical wall surfaces
# =====================================================================
def parse_vertical_surfaces(cm, world_verts):
    """
    Collect all near-vertical polygons from the highest-LOD building
    geometry, annotated with outward 2-D normal and wall-plane depth.
    """
    surfaces = []

    for obj_id, obj in cm.get("CityObjects", {}).items():
        if obj.get("type", "") in _SKIP_TYPES:
            continue
        geoms = obj.get("geometry", [])
        if not geoms:
            continue

        def _lod(g):
            try:
                return float(g.get("lod") or 0)
            except (ValueError, TypeError):
                return 0.0

        best_idx = max(range(len(geoms)), key=lambda i: _lod(geoms[i]))
        g_idx    = best_idx
        geom     = geoms[best_idx]

        boundaries = geom.get("boundaries", [])
        geom_type  = geom.get("type", "")

        shells = (enumerate(boundaries) if geom_type == "Solid"
                  else [(0, boundaries)])

        for s_idx, shell in shells:
            for p_idx, polygon in enumerate(shell):
                try:
                    coords = world_verts[np.array(polygon[0])]
                except (IndexError, KeyError, TypeError):
                    continue
                if len(coords) < 3:
                    continue

                # Newell normal
                n = np.zeros(3)
                for i in range(len(coords)):
                    c  = coords[i]
                    nx = coords[(i + 1) % len(coords)]
                    n[0] += (c[1] - nx[1]) * (c[2] + nx[2])
                    n[1] += (c[2] - nx[2]) * (c[0] + nx[0])
                    n[2] += (c[0] - nx[0]) * (c[1] + nx[1])

                n_len = np.linalg.norm(n)
                if n_len < 1e-9:
                    continue
                n_unit = n / n_len
                if abs(n_unit[2]) > VERTICAL_TOL:
                    continue   # near-horizontal → skip

                nh  = n[:2]
                mag = np.linalg.norm(nh)
                if mag < 1e-6:
                    continue

                normal_2d = nh / mag
                centroid  = coords.mean(axis=0)
                wall_d    = float(np.dot(centroid[:2], normal_2d))

                surfaces.append({
                    "idx":       len(surfaces),
                    "obj_id":    obj_id,
                    "geom_idx":  g_idx,
                    "shell_idx": s_idx,
                    "poly_idx":  p_idx,
                    "coords":    coords,
                    "normal_2d": normal_2d,
                    "origin_2d": centroid[:2].copy(),
                    "wall_d":    wall_d,
                    "z_min":     float(coords[:, 2].min()),
                    "z_max":     float(coords[:, 2].max()),
                    "xy_min":    coords[:, :2].min(axis=0),
                    "xy_max":    coords[:, :2].max(axis=0),
                })

    print(f"  Parsed {len(surfaces)} vertical surface(s).")
    return surfaces


# =====================================================================
# 3-stage wall surface matching
# =====================================================================
def _pca_normal_2d(points):
    if len(points) < 3:
        return None
    xy  = points[:, :2]
    cxy = xy - xy.mean(axis=0)
    cov = cxy.T @ cxy
    _, vecs = np.linalg.eigh(cov)
    tang = vecs[:, 1]
    return np.array([-tang[1], tang[0]])


def find_nearest_surface(points, vert_surfaces, dist_tol=2.0, z_expand=0.5):
    """
    Match a cluster to a wall surface via 3 stages:
      1. Spatial intersection (Z overlap + signed distance + XY footprint).
      2. PCA normal alignment fallback.
      3. Nearest centroid unconditional fallback.
    """
    if not vert_surfaces:
        return None

    z_lo = float(points[:, 2].min())
    z_hi = float(points[:, 2].max())
    cxy  = points.mean(axis=0)[:2]

    best, best_d = None, float("inf")

    # Stage 1
    for vs in vert_surfaces:
        if z_hi < vs["z_min"] - z_expand or z_lo > vs["z_max"] + z_expand:
            continue
        n2d = vs["normal_2d"]
        d   = float(np.dot(cxy - vs["origin_2d"], n2d))
        if abs(d) >= dist_tol:
            continue
        proj = cxy - d * n2d
        pad  = dist_tol
        if (proj[0] < vs["xy_min"][0] - pad or proj[0] > vs["xy_max"][0] + pad or
                proj[1] < vs["xy_min"][1] - pad or proj[1] > vs["xy_max"][1] + pad):
            continue
        if abs(d) < best_d:
            best_d, best = abs(d), vs

    if best:
        return best

    # Stage 2 – PCA normal alignment
    pca_n = _pca_normal_2d(points)
    if pca_n is not None:
        best_dot = -1.0
        for vs in vert_surfaces:
            dot = abs(float(np.dot(pca_n, vs["normal_2d"])))
            if dot > best_dot:
                best_dot, best = dot, vs
        if best:
            return best

    # Stage 3 – nearest centroid
    for vs in vert_surfaces:
        d = abs(float(np.dot(cxy - vs["origin_2d"], vs["normal_2d"])))
        if d < best_d:
            best_d, best = d, vs

    return best


# =====================================================================
# Wall-plane clip: discard the portion of the bbox inside the building
# =====================================================================
def clip_bbox_at_wall(cluster_pts, wall_surface):
    """
    Project all cluster points onto the wall outward-normal axis and
    return the extents needed to build a **clipped** bounding box:

      - Near face  → fixed at d_wall  (wall plane; interior portion
                     of the cluster is simply omitted)
      - Far face   → max projection   (outermost point of the cluster)
      - Width      → tangent extents  of ALL cluster points
      - Height     → Z extents        of ALL cluster points

    Returns a dict with projection arrays and scalar extents, or None
    if no cluster point protrudes outward from the wall plane.
    """
    n2d    = wall_surface["normal_2d"]
    d_wall = wall_surface["wall_d"]

    # Tangent axis (horizontal, perpendicular to wall normal)
    z_up    = np.array([0.0, 0.0, 1.0])
    n3      = np.array([n2d[0], n2d[1], 0.0])
    n3     /= np.linalg.norm(n3)
    tangent = np.cross(z_up, n3)
    tn      = np.linalg.norm(tangent)
    tangent = tangent / tn if tn > 1e-9 else np.array([1.0, 0.0, 0.0])
    txy     = tangent[:2]

    # Project every point
    pt_d = cluster_pts[:, :2] @ n2d   # depth along outward normal
    pt_t = cluster_pts[:, :2] @ txy   # along tangent
    pt_z = cluster_pts[:, 2]

    d_out = float(pt_d.max())

    # Nothing protrudes outward — cluster is entirely inside the building
    if d_out <= d_wall:
        return None

    return {
        "n2d":    n2d,
        "txy":    txy,
        "d_wall": d_wall,
        "d_out":  d_out,
        "t_lo":   float(pt_t.min()),
        "t_hi":   float(pt_t.max()),
        "z_lo":   float(pt_z.min()),
        "z_hi":   float(pt_z.max()),
    }


# =====================================================================
# Extrusion 3-D bounding box (near face snapped to wall, protrudes out)
# =====================================================================
def make_extrusion_bbox(clip_info):
    """
    Build a wall-aligned bounding box from pre-clipped projection data.

    *clip_info* is the dict returned by clip_bbox_at_wall().

    Local axes:
      n-axis  = wall outward normal  (positive = outward from building)
      t-axis  = cross(Z, n)          (along-wall tangent)
      Z-axis  = vertical

    The near face is fixed at d_wall (the building wall plane); any
    portion of the cluster that was inside the building is effectively
    clipped away.  The far face reaches the outermost point.

    Corners  (0–3 near ring at wall, 4–7 far ring outward):
      0 = (d_wall, t_lo, z_lo)   3 = (d_wall, t_hi, z_lo)
      1 = (d_out,  t_lo, z_lo)   2 = (d_out,  t_hi, z_lo)
      4 = (d_wall, t_lo, z_hi)   7 = (d_wall, t_hi, z_hi)
      5 = (d_out,  t_lo, z_hi)   6 = (d_out,  t_hi, z_hi)

    Returns a dict with corners and axis info.
    """
    n2d    = clip_info["n2d"]
    txy    = clip_info["txy"]
    d_wall = clip_info["d_wall"]
    d_out  = clip_info["d_out"]
    t_lo   = clip_info["t_lo"]
    t_hi   = clip_info["t_hi"]
    z_lo   = clip_info["z_lo"]
    z_hi   = clip_info["z_hi"]

    def pt3(d, t, z):
        xy = d * n2d + t * txy
        return np.array([float(xy[0]), float(xy[1]), float(z)])

    # 8 corners:
    # 0–3: near ring (at wall plane)  4–7: far ring (outward)
    corners = np.array([
        pt3(d_wall, t_lo, z_lo),   # 0 near bottom-left
        pt3(d_out,  t_lo, z_lo),   # 1 far  bottom-left
        pt3(d_out,  t_hi, z_lo),   # 2 far  bottom-right
        pt3(d_wall, t_hi, z_lo),   # 3 near bottom-right
        pt3(d_wall, t_lo, z_hi),   # 4 near top-left
        pt3(d_out,  t_lo, z_hi),   # 5 far  top-left
        pt3(d_out,  t_hi, z_hi),   # 6 far  top-right
        pt3(d_wall, t_hi, z_hi),   # 7 near top-right
    ])

    protrusion_m = d_out - d_wall
    width_m      = t_hi - t_lo
    height_m     = z_hi - z_lo

    return {
        "corners":      corners,
        "n2d":          n2d,
        "txy":          txy,
        "d_wall":       d_wall,
        "d_out":        d_out,
        "protrusion_m": protrusion_m,
        "t_lo":         t_lo,
        "t_hi":         t_hi,
        "z_lo":         z_lo,
        "z_hi":         z_hi,
        "width_m":      width_m,
        "height_m":     height_m,
    }


# =====================================================================
# Convex-hull geometry builder
# =====================================================================
def get_exterior_points(cluster_pts, wall_surface):
    """
    Return (exterior_pts, n2d, txy, d_wall, d_out, protrusion_m) for the
    subset of cluster points that protrude outward from the wall plane,
    or None if no such points exist.
    """
    n2d    = wall_surface["normal_2d"]
    d_wall = wall_surface["wall_d"]

    # Build tangent axis
    z_up    = np.array([0.0, 0.0, 1.0])
    n3      = np.array([n2d[0], n2d[1], 0.0])
    n3     /= np.linalg.norm(n3)
    tangent = np.cross(z_up, n3)
    tn      = np.linalg.norm(tangent)
    tangent = tangent / tn if tn > 1e-9 else np.array([1.0, 0.0, 0.0])
    txy     = tangent[:2]

    pt_d  = cluster_pts[:, :2] @ n2d
    d_out = float(pt_d.max())

    if d_out <= d_wall:
        return None   # entirely inside

    ext_mask = pt_d >= d_wall
    ext_pts  = cluster_pts[ext_mask]

    return ext_pts, n2d, txy, d_wall, d_out, d_out - d_wall


def build_convex_hull_geometry(ext_pts, n2d, d_wall, int_verts, scale, translate):
    """
    Build CityJSON polygon boundaries for a BuildingInstallation using the
    3-D convex hull of the exterior cluster points, with a planar back-wall
    cap polygon added at d_wall to seal the hull against the building facade.

    Returns (boundaries, face_semantics) where:
      - boundaries     : list of CityJSON polygon specs [[[ vi, ... ]]]
      - face_semantics : list of semantic type strings (one per polygon)

    Returns (None, None) if the convex hull cannot be computed.
    """
    if len(ext_pts) < 4:
        return None, None

    try:
        hull = ConvexHull(ext_pts)
    except Exception:
        return None, None

    # ── Hull faces (triangles) ───────────────────────────────────────
    # Discard any simplex whose outward normal points INWARD (toward the
    # building, i.e. dot(normal, n2d) < 0).  These are the back face(s)
    # that will be replaced by our planar cap.
    boundaries     = []
    face_semantics = []

    for simplex in hull.simplices:
        verts_tri = ext_pts[simplex]   # 3 × 3
        # Newell normal for winding consistency
        e1 = verts_tri[1] - verts_tri[0]
        e2 = verts_tri[2] - verts_tri[0]
        tri_n = np.cross(e1, e2)
        tri_n_len = np.linalg.norm(tri_n)
        if tri_n_len < 1e-12:
            continue
        tri_n /= tri_n_len

        # Skip back-facing triangles (they face into the building wall)
        if float(np.dot(tri_n[:2], n2d)) < -0.1:
            continue

        # Encode triangle vertices
        start = len(int_verts)
        for v in verts_tri:
            int_verts.append(encode_vertex(v, scale, translate))
        boundaries.append([[start, start + 1, start + 2]])
        face_semantics.append("WallSurface")

    # ── Back-wall cap at d = d_wall ───────────────────────────────────
    # Project exterior points onto the wall plane (fix depth = d_wall)
    # and compute their 2-D convex hull to form a planar polygon cap.
    wall_pts_xy = []
    for p in ext_pts:
        # Keep only points very close to the wall plane
        d_p = float(p[:2] @ n2d)
        if d_p - d_wall < 0.2:   # within 20 cm of wall plane
            wall_pts_xy.append((d_p, p[2]))  # (depth, z) for 2-D hull

    # Use ALL exterior points projected onto the wall plane
    cap_2d = []
    for p in ext_pts:
        # Tangent component (horizontal along wall)
        z_up    = np.array([0.0, 0.0, 1.0])
        n3      = np.array([n2d[0], n2d[1], 0.0])
        n3     /= np.linalg.norm(n3)
        tang    = np.cross(z_up, n3)
        tn      = np.linalg.norm(tang)
        tang    = tang / tn if tn > 1e-9 else np.array([1.0, 0.0, 0.0])
        txy     = tang[:2]
        t_coord = float(p[:2] @ txy)
        cap_2d.append((t_coord, float(p[2])))

    cap_2d_arr = np.array(cap_2d)
    if len(cap_2d_arr) >= 3:
        try:
            from shapely.geometry import MultiPoint as SMP
            cap_hull_2d = SMP(cap_2d_arr.tolist()).convex_hull
            cap_coords_2d = list(cap_hull_2d.exterior.coords)[:-1]  # drop repeated last

            # Reconstruct 3-D wall-plane points from (t, z)
            n3   = np.array([n2d[0], n2d[1], 0.0])
            n3  /= np.linalg.norm(n3)
            tang = np.cross(np.array([0.0, 0.0, 1.0]), n3)
            tn   = np.linalg.norm(tang)
            tang = tang / tn if tn > 1e-9 else np.array([1.0, 0.0, 0.0])
            txy  = tang[:2]

            cap_3d = []
            for t_c, z_c in cap_coords_2d:
                xy = d_wall * n2d + t_c * txy
                cap_3d.append(np.array([float(xy[0]), float(xy[1]), float(z_c)]))

            if len(cap_3d) >= 3:
                # Reverse winding so normal points inward (toward building)
                cap_3d = cap_3d[::-1]
                start = len(int_verts)
                for v in cap_3d:
                    int_verts.append(encode_vertex(v, scale, translate))
                idx = list(range(start, start + len(cap_3d)))
                boundaries.append([idx])
                face_semantics.append("WallSurface")
        except Exception:
            pass   # if shapely fails, omit the cap

    if not boundaries:
        return None, None

    return boundaries, face_semantics


# =====================================================================
# Add BuildingInstallation CityObject
# =====================================================================
def add_installation(cm, install_id, parent_id, poly_boundaries, face_semantics):
    """
    Create a BuildingInstallation CityObject and link it to its parent.
    face_semantics is a list of semantic type strings, one per polygon.
    """
    n = len(poly_boundaries)
    cm["CityObjects"][install_id] = {
        "type":    "BuildingInstallation",
        "parents": [parent_id],
        "geometry": [{
            "type":       "MultiSurface",
            "lod":        "3",
            "boundaries": poly_boundaries,
            "semantics": {
                "surfaces": [{"type": face_semantics[i]} for i in range(n)],
                "values":   list(range(n)),
            },
        }],
    }
    cm["CityObjects"][parent_id].setdefault("children", []).append(install_id)


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="CityJSON LOD3 — Facade Extrusions, Convex Hull (BuildingInstallations)")
    parser.add_argument("--eps",            type=float, default=0.3,
                        help="DBSCAN eps radius (default 0.3 m)")
    parser.add_argument("--min_samples",    type=int,   default=30,
                        help="DBSCAN min_samples (default 30)")
    parser.add_argument("--min_protrusion", type=float, default=0.01,
                        help="Minimum outward protrusion (m) to keep a cluster "
                             "(default 0.01 m)")
    parser.add_argument("--min_hull_pts",   type=int,   default=4,
                        help="Minimum exterior points needed to build a convex hull "
                             "(default 4)")
    parser.add_argument("--output", "-o",  type=str,   default=None,
                        help="Output filename (in outputs/15_extrusions_ch_json/).")
    args = parser.parse_args()

    print("=" * 60)
    print("  CityJSON LOD3: Facade Extrusions (Convex Hull)")
    print("=" * 60)

    # ── 1. Select CityJSON ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SELECT CITYJSON MODEL")
    print(f"{'='*60}")
    json_path = select_file(JSON_DIR, "*.json")

    print(f"\n  Loading {os.path.basename(json_path)} ...")
    with open(json_path, "r", encoding="utf-8") as fh:
        cm = json.load(fh)

    world_verts = decode_vertices(cm)
    transform   = cm.get("transform", {})
    scale       = np.array(transform.get("scale",     [1, 1, 1]), dtype=np.float64)
    translate   = np.array(transform.get("translate", [0, 0, 0]), dtype=np.float64)
    int_verts   = [list(v) for v in cm["vertices"]]

    # ── 2. Building footprint ─────────────────────────────────────────
    print("\n  Extracting building footprint ...")
    footprint = extract_building_footprint(cm, world_verts)
    if footprint is None:
        print("  WARNING: no footprint — bbox clipping will use wall-plane only.")

    # ── 3. Parse wall surfaces ────────────────────────────────────────
    print("\n  Parsing vertical surfaces ...")
    vert_surfaces = parse_vertical_surfaces(cm, world_verts)

    # ── Find parent building object ───────────────────────────────────
    parent_id = None
    for obj_id, obj in cm.get("CityObjects", {}).items():
        if obj.get("type", "") not in _SKIP_TYPES:
            parent_id = obj_id
            break
    if parent_id is None:
        print("  ERROR: no building CityObject found.")
        sys.exit(1)
    print(f"  Parent building object: '{parent_id}'")

    # ── 4. Select point cloud ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SELECT POINT CLOUD  (outputs/11B_flat)")
    print(f"{'='*60}")
    las_path = select_file(LAS_DIR, "*.las")

    print(f"\n  Loading {os.path.basename(las_path)} ...")
    las    = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    print(f"  {len(points):,} points loaded.")

    # ── 5. DBSCAN ─────────────────────────────────────────────────────
    print(f"\n  Running DBSCAN "
          f"(eps={args.eps}, min_samples={args.min_samples}) ...")
    labels  = DBSCAN(eps=args.eps,
                     min_samples=args.min_samples).fit_predict(points)
    unique  = sorted(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"  {len(unique)} cluster(s) found, {n_noise:,} noise points discarded.")

    if not unique:
        print("  No clusters — nothing to do.")
        return

    # ── 6. Process each cluster ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PROCESSING CLUSTERS  (convex hull extrusions)")
    print(f"{'='*60}")

    n_extruded = 0
    n_skipped  = 0

    for label in unique:
        cluster_pts = points[labels == label]
        cluster_idx = int(label) + 1
        centroid    = cluster_pts.mean(axis=0)

        print(f"\n  Cluster {cluster_idx}  ({len(cluster_pts):,} pts)  "
              f"cen=({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")

        # ── a. Match to nearest wall surface ──────────────────────────
        matched = find_nearest_surface(cluster_pts, vert_surfaces)

        if matched is None:
            if not vert_surfaces:
                cxy = centroid[:2]
                matched = {
                    "idx":       -1,
                    "obj_id":    parent_id,
                    "geom_idx":  0,
                    "shell_idx": 0,
                    "poly_idx":  0,
                    "coords":    cluster_pts,
                    "normal_2d": np.array([1.0, 0.0]),
                    "origin_2d": cxy,
                    "wall_d":    float(cxy[0]),
                    "z_min":     float(cluster_pts[:, 2].min()),
                    "z_max":     float(cluster_pts[:, 2].max()),
                    "xy_min":    cluster_pts[:, :2].min(axis=0),
                    "xy_max":    cluster_pts[:, :2].max(axis=0),
                }
            else:
                print("  → SKIP: no matching wall surface found.")
                n_skipped += 1
                continue

        wall_angle = np.degrees(np.arctan2(
            matched["normal_2d"][1], matched["normal_2d"][0]))
        print(f"  Wall surface: angle={wall_angle:+.1f}°  "
              f"z=[{matched['z_min']:.2f}, {matched['z_max']:.2f}]")

        # ── b. Extract exterior points (clip at wall plane) ─────────────
        ext_result = get_exterior_points(cluster_pts, matched)

        if ext_result is None:
            print("  → SKIP: cluster lies entirely inside the building "
                  "(no outward protrusion).")
            n_skipped += 1
            continue

        ext_pts, n2d_m, txy_m, d_wall_m, d_out_m, protrusion_m = ext_result

        if protrusion_m < args.min_protrusion:
            print(f"  → SKIP: protrusion {protrusion_m:.3f}m "
                  f"< {args.min_protrusion:.3f}m threshold.")
            n_skipped += 1
            continue

        if len(ext_pts) < args.min_hull_pts:
            print(f"  → SKIP: only {len(ext_pts)} exterior point(s), "
                  f"need ≥ {args.min_hull_pts} for convex hull.")
            n_skipped += 1
            continue

        # ── c. Build convex hull geometry ──────────────────────────────
        boundaries, face_semantics = build_convex_hull_geometry(
            ext_pts, matched["normal_2d"], matched["wall_d"],
            int_verts, scale, translate
        )

        if boundaries is None:
            print("  → SKIP: convex hull failed (degenerate point set).")
            n_skipped += 1
            continue

        print(f"  Convex hull: protrusion={protrusion_m:.3f}m  "
              f"{len(ext_pts):,} exterior pts  → {len(boundaries)} face(s)")

        # ── d. Create BuildingInstallation ───────────────────────────────
        feat_id = f"extrusion_{cluster_idx}_{uuid.uuid4().hex[:8]}"
        add_installation(cm, feat_id, parent_id, boundaries, face_semantics)

        print(f"  → BuildingInstallation '{feat_id}' "
              f"({len(boundaries)} face(s))")
        n_extruded += 1

    # ── 7. Summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Extrusions created:          {n_extruded}")
    print(f"  Skipped (insufficient pts):  {n_skipped}")

    # ── 8. Save ───────────────────────────────────────────────────────
    cm["vertices"] = int_verts

    if args.output:
        out_name = (args.output if args.output.endswith(".json")
                    else args.output + ".json")
    else:
        base     = os.path.splitext(os.path.basename(json_path))[0]
        out_name = f"{base}_extrusions.json"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, out_name)

    print(f"\n  Saving → {out_path} ...")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(cm, fh, separators=(",", ":"))

    mb = os.path.getsize(out_path) / 1_048_576
    print(f"  Saved: {out_path}  ({mb:.1f} MB)")
    print(f"\n{'='*60}")
    print("  Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
