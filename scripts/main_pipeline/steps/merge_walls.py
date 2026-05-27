#!/usr/bin/env python3
"""
Merge fragmented WallSurfaces in CityJSON files.

This preprocessing script:
1. Reads a CityJSON file from data/lod_2
2. Merges adjacent WallSurfaces with similar normals
3. Saves the merged CityJSON to outputs/00_json_wall_merged
"""

import os
import argparse
import json
import numpy as np
from typing import List, Tuple, Optional
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

DEFAULT_INPUT_DIR = r'C:\Projects\LOD3Test\Data\cityforge'
DEFAULT_OUTPUT_DIR = r'C:\Projects\LOD3Test\Outputs\merged_copy'

def get_poly_normal(coords: np.ndarray) -> np.ndarray:
    """Calculate polygon normal using Newell's method.

    Newell's method accumulates cross-product contributions from every edge,
    so it works correctly for all convex and concave polygons regardless of
    vertex ordering.  Unlike the first-3-vertex approach it never degenerates
    on near-collinear leading vertices (common in photogrammetry meshes).
    """
    n = np.zeros(3)
    count = len(coords)
    if count < 3:
        return np.array([0.0, 0.0, 1.0])
    for i in range(count):
        j = (i + 1) % count
        n[0] += (coords[i][1] - coords[j][1]) * (coords[i][2] + coords[j][2])
        n[1] += (coords[i][2] - coords[j][2]) * (coords[i][0] + coords[j][0])
        n[2] += (coords[i][0] - coords[j][0]) * (coords[i][1] + coords[j][1])
    length = np.linalg.norm(n)
    if length < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    return n / length


def _ring_is_valid_for_export(ring: np.ndarray, normal_hint: Optional[np.ndarray] = None,
                              max_planar_dev: float = 0.05) -> bool:
    """Validate ring geometry to avoid holes from downstream polygon drops.

    A ring is export-safe when it has:
    - at least 3 unique vertices,
    - non-zero area (Newell norm),
    - near-planarity within max_planar_dev metres.
    """
    if ring is None or len(ring) < 3:
        return False

    # Remove trailing closure if present.
    pts = ring[:-1] if len(ring) > 1 and np.allclose(ring[0], ring[-1]) else ring
    if len(pts) < 3:
        return False

    # Remove consecutive duplicates.
    cleaned = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - cleaned[-1]) > 1e-8:
            cleaned.append(p)
    if len(cleaned) < 3:
        return False

    # Require at least 3 unique coordinates.
    uniq = {tuple(np.round(p, 8)) for p in cleaned}
    if len(uniq) < 3:
        return False

    arr = np.array(cleaned, dtype=np.float64)
    normal = get_poly_normal(arr)
    if np.linalg.norm(normal) < 1e-6:
        return False

    # Planarity check against ring plane.
    # We allow up to 0.5m deviation to preserve curved walls without flattening them.
    ref_n = normal_hint if normal_hint is not None and np.linalg.norm(normal_hint) > 1e-6 else normal
    ref_n = ref_n / np.linalg.norm(ref_n)
    p0 = arr[0]
    distances = [abs(float(np.dot(ref_n, p - p0))) for p in arr[1:]]
    if distances and max(distances) > max_planar_dev:
        return False

    return True


def _surface_is_valid_for_export(surface: 'WallSurface',
                                 max_planar_dev: float = 0.5) -> bool:
    """Validate all rings of a wall surface for robust CityGML conversion."""
    if not surface.rings_coords:
        return False

    ext = surface.rings_coords[0]
    ext_normal = get_poly_normal(ext)
    if not _ring_is_valid_for_export(ext, max_planar_dev=max_planar_dev):
        return False

    for hole in surface.rings_coords[1:]:
        # Hole rings should lie on the same wall plane.
        if not _ring_is_valid_for_export(hole, normal_hint=ext_normal,
                                         max_planar_dev=max_planar_dev):
            return False
    return True


class WallSurface:
    """Represents a WallSurface from CityJSON with its geometry."""
    
    def __init__(self, rings_coords: List[np.ndarray], semantic_val: int = None, original_polygon=None):
        self.rings_coords = rings_coords
        self.coordinates = rings_coords[0]  # Nx3 array of exterior vertices
        self.bbox_min = np.min(self.coordinates, axis=0)
        self.bbox_max = np.max(self.coordinates, axis=0)
        self._normal = None
        self.semantic_val = semantic_val
        self.original_polygon = original_polygon  # Store original polygon (list of rings)
    
    def get_center(self) -> np.ndarray:
        """Get the center point of the wall surface."""
        return np.mean(self.coordinates, axis=0)
    
    def get_normal(self) -> np.ndarray:
        """Calculate and return the surface normal vector."""
        if self._normal is not None:
            return self._normal
        self._normal = get_poly_normal(self.coordinates)
        return self._normal
    
    def is_adjacent(self, other: 'WallSurface', distance_threshold: float = 1.0) -> bool:
        """Check if another wall surface is spatially adjacent."""
        center1 = self.get_center()
        center2 = other.get_center()
        distance = np.linalg.norm(center1 - center2)
        
        if distance > distance_threshold * 10:  # Quick rejection
            return False
        
        for v1 in self.coordinates:
            for v2 in other.coordinates:
                if np.linalg.norm(v1 - v2) < 0.01:  # Same vertex
                    return True
        
        return False
    
    def is_coplanar(self, other: 'WallSurface', plane_distance_threshold: float = 0.5) -> bool:
        """
        Check if another wall surface is coplanar (on the same plane).
        This allows merging surfaces at different heights on the same facade.
        """
        normal1 = self.get_normal()
        point_on_plane1 = self.coordinates[0]
        
        distances = []
        for point in other.coordinates:
            d = abs(np.dot(normal1, point - point_on_plane1))
            distances.append(d)
        
        max_distance = max(distances)
        return max_distance < plane_distance_threshold


def decode_vertices(cm):
    """Decode integer vertices to real-world coordinates."""
    raw       = np.array(cm["vertices"], dtype=np.float64)
    t         = cm.get("transform", {})
    scale     = np.array(t.get("scale",     [1, 1, 1]), dtype=np.float64)
    translate = np.array(t.get("translate", [0, 0, 0]), dtype=np.float64)
    return raw * scale + translate


def encode_vertex(pt, scale, translate):
    """Encode real-world coordinates back to integer vertices."""
    return [int(round((pt[i] - translate[i]) / scale[i])) for i in range(3)]


def _walls_coplanar(wall_a: 'WallSurface', wall_b: 'WallSurface',
                    cos_threshold: float, plane_tol: float = 0.25) -> bool:
    """Return True if wall_b lies on the same plane as wall_a.

    Two wall fragments belong to the same facade plane when:
      1. Their normals are parallel (within angle threshold).
      2. Every vertex of wall_b is within plane_tol metres of wall_a's plane.
    This is the criterion that lets the merger follow the roofline: once all
    coplanar fragments are unioned the top boundary is the roofline itself.
    """
    na = wall_a.get_normal()
    nb = wall_b.get_normal()
    # Require normals to point in the SAME direction (positive dot product).
    # Using abs() would incorrectly group inner-courtyard faces with outer
    # facade faces — they're on the same plane but face opposite ways and
    # must NOT be merged together.
    if np.dot(na, nb) < cos_threshold:
        return False
    da = float(np.dot(na, wall_a.coordinates[0]))
    return all(abs(float(np.dot(na, pt)) - da) < plane_tol for pt in wall_b.coordinates)


def select_json_file(json_dir: str = DEFAULT_INPUT_DIR) -> Optional[str]:
    """Interactive selection of JSON file from directory."""
    if not os.path.exists(json_dir):
        print(f"ERROR: JSON directory not found: {json_dir}")
        return None
    
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json') or f.endswith('.cityjson')])
    
    if not json_files:
        print(f"ERROR: No .json/.cityjson files found in {json_dir}")
        return None
    
    if len(json_files) == 1:
        selected = os.path.join(json_dir, json_files[0])
        print(f"\n✓ Auto-selected (only one JSON file): {json_files[0]}")
        return selected
    
    print("\n" + "=" * 80)
    print("SELECT JSON FILE TO MERGE")
    print("=" * 80)
    print(f"\nAvailable JSON files in {json_dir}:")
    
    for i, filename in enumerate(json_files):
        filepath = os.path.join(json_dir, filename)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  [{i}] {filename} ({size_mb:.2f} MB)")
    
    while True:
        try:
            response = input(f"\nSelect JSON file (0-{len(json_files)-1}, or 'q' to quit): ")
            if response.lower() == 'q':
                return None
            idx = int(response)
            if 0 <= idx < len(json_files):
                selected = os.path.join(json_dir, json_files[idx])
                print(f"✓ Selected: {json_files[idx]}")
                return selected
            else:
                print(f"Invalid index. Please enter 0-{len(json_files)-1}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")


def get_vertex_id(enc, vertices, vertex_map):
    """Safely append or reuse vertex IDs to stop coordinate list bloat."""
    t_enc = tuple(enc)
    if t_enc in vertex_map:
        return vertex_map[t_enc]
    idx = len(vertices)
    vertices.append(enc)
    vertex_map[t_enc] = idx
    return idx


def merge_wall_surfaces(wall_surfaces: List[WallSurface],
                        normal_angle_threshold: float = 5.0,
                        distance_threshold: float = 2.0,
                        obj_id: str = "") -> Tuple[List[WallSurface], int]:
    """Merge coplanar WallSurfaces so the result follows the roofline.

    Groups wall fragments by coplanarity (same normal direction + same plane
    offset) rather than by shared vertices.  This means all fragments of a
    single facade — even ones separated by small gaps from the tessellation —
    are unioned into one polygon whose top boundary is the roofline.

    MultiPolygon results (disconnected facade sections on the same plane) are
    kept as separate WallSurface outputs so no geometry is ever discarded.

    Args:
        normal_angle_threshold: max angle (degrees) between normals to be
            considered parallel — controls facade direction tolerance.
        distance_threshold: max distance (metres) a vertex may be from the
            seed wall's plane and still be considered coplanar.  Default 2.0 m
            is intentionally generous to handle meshing noise; tighten to ~0.1
            for clean CAD-derived files.
    """
    if len(wall_surfaces) == 0:
        return [], 0

    cos_threshold = np.cos(np.radians(normal_angle_threshold))
    # plane_tol: how far off-plane a vertex may be and still count as coplanar.
    # We re-use distance_threshold so existing CLI calls keep working; cap at
    # 0.5 m to avoid swallowing unrelated surfaces on nearly-parallel planes.
    plane_tol = min(distance_threshold, 0.5)

    merged_flags = [False] * len(wall_surfaces)
    merged_surfaces = []
    merge_count = 0

    for i, wall in enumerate(wall_surfaces):
        if merged_flags[i]:
            continue

        group = [i]
        merged_flags[i] = True

        # Grow group: add every wall that lies on the same plane as the seed.
        changed = True
        while changed:
            changed = False
            for j in range(len(wall_surfaces)):
                if merged_flags[j]:
                    continue
                if _walls_coplanar(wall_surfaces[group[0]], wall_surfaces[j],
                                   cos_threshold, plane_tol):
                    is_adjacent_to_group = False
                    for group_idx in group:
                        if wall_surfaces[group_idx].is_adjacent(wall_surfaces[j], distance_threshold):
                            is_adjacent_to_group = True
                            break
                    
                    if is_adjacent_to_group:
                        group.append(j)
                        merged_flags[j] = True
                        changed = True

        if len(group) == 1:
            merged_surfaces.append(wall_surfaces[group[0]])
            continue

        all_vertices = []
        for idx in group:
            for r in wall_surfaces[idx].rings_coords:
                all_vertices.append(r)
        combined_vertices = np.vstack(all_vertices)

        try:
            # ── Step 1: compute a robust reference normal from ALL group members.
            # Sign-align each member normal to the first before averaging so that
            # anti-parallel normals (flipped input faces) don't cancel out.
            raw_normals = np.array([wall_surfaces[idx].get_normal() for idx in group])
            ref_n = raw_normals[0]
            for k in range(1, len(raw_normals)):
                if np.dot(raw_normals[k], ref_n) < 0:
                    raw_normals[k] = -raw_normals[k]
            group_normal = raw_normals.mean(axis=0)
            gn_len = np.linalg.norm(group_normal)
            group_normal = group_normal / gn_len if gn_len > 1e-6 else ref_n

            centroid = np.mean(combined_vertices, axis=0)
            centered = combined_vertices - centroid

            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            idx_sort = eigenvalues.argsort()[::-1]
            eigenvectors = eigenvectors[:, idx_sort].real

            # ── Step 2: anchor eigenvector handedness.
            # Shapely always produces CCW exterior rings in 2D.  For that CCW
            # winding to produce an outward-facing normal in 3D the cross product
            # (e0 × e1) must point in the SAME direction as group_normal.
            # If it doesn't, flip e1 to correct the handedness before projecting.
            e0 = eigenvectors[:, 0]
            e1 = eigenvectors[:, 1]
            if np.dot(np.cross(e0, e1), group_normal) < 0:
                eigenvectors[:, 1] = -e1

            shapely_polys = []
            for idx in group:
                rings_2d = []
                for r in wall_surfaces[idx].rings_coords:
                    coords_2d = (r - centroid) @ eigenvectors[:, :2]
                    rings_2d.append(coords_2d)
                if not rings_2d:
                    continue
                poly = ShapelyPolygon(shell=rings_2d[0], holes=rings_2d[1:])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty:
                    shapely_polys.append(poly)

            if not shapely_polys:
                raise ValueError("All projected polygons are empty")

            eps = 1e-3
            union_result = unary_union([p.buffer(eps) for p in shapely_polys]).buffer(-eps)

            # Keep ALL pieces of a MultiPolygon — discarding any piece would
            # silently lose geometry and break downstream scripts.
            if union_result.geom_type == 'MultiPolygon':
                poly_pieces = list(union_result.geoms)
            elif union_result.geom_type == 'Polygon':
                poly_pieces = [union_result]
            else:
                raise ValueError(f"Unexpected union geometry type: {union_result.geom_type}")

            def raw_proj_to_3d(coords_2d_arr):
                """Back-project 2D coords to 3D WITHOUT vertex snapping.
                Used only for winding-order checks so snapping noise can't
                corrupt the normal direction test."""
                return coords_2d_arr @ eigenvectors[:, :2].T + centroid

            def snap_ring(pts_3d):
                """Snap 3D points to nearest original vertex (within 0.1 m)."""
                snapped = []
                for pt in pts_3d:
                    dists = np.linalg.norm(combined_vertices - pt, axis=1)
                    min_idx = np.argmin(dists)
                    snapped.append(
                        combined_vertices[min_idx] if dists[min_idx] < 0.1 else pt
                    )
                return np.array(snapped)

            def restore_collinear_vertices(simplified_ring_3d, original_vertices_3d, tol=1e-3):
                """Re-inject original vertices that lie exactly on the simplified segments
                to prevent T-junction cracks with unmerged adjacent surfaces."""
                new_ring = []
                num_pts = len(simplified_ring_3d)
                unique_orig = np.unique(original_vertices_3d, axis=0)
                for i in range(num_pts):
                    p1 = simplified_ring_3d[i]
                    p2 = simplified_ring_3d[(i + 1) % num_pts]
                    new_ring.append(p1)
                    segment_length = np.linalg.norm(p2 - p1)
                    if segment_length < 1e-5:
                        continue
                    collinear_pts = []
                    for v in unique_orig:
                        if np.linalg.norm(v - p1) < tol or np.linalg.norm(v - p2) < tol:
                            continue
                        d1 = np.linalg.norm(v - p1)
                        d2 = np.linalg.norm(v - p2)
                        if abs((d1 + d2) - segment_length) < tol:
                            collinear_pts.append((d1, v))
                    collinear_pts.sort(key=lambda x: x[0])
                    for _, v in collinear_pts:
                        new_ring.append(v)
                return np.array(new_ring)

            def project_ring_to_plane(ring_3d, plane_pt, plane_normal):
                """Force a ring onto the reference wall plane to avoid non-planar artifacts."""
                n = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
                projected = []
                for p in ring_3d:
                    dist = np.dot(n, p - plane_pt)
                    projected.append(p - dist * n)
                return np.array(projected)

            # Keep a local orientation reference from original input surfaces.
            # For each merged piece, we pick the nearest original wall and use
            # that wall's normal as the winding reference for the piece.
            source_centers = np.array([wall_surfaces[idx].get_center() for idx in group])
            source_normals = np.array([wall_surfaces[idx].get_normal() for idx in group])

            produced = 0
            piece_surfaces = []
            for upoly in poly_pieces:
                exterior_2d = np.array(upoly.exterior.coords[:-1])

                piece_center_2d = np.array(upoly.representative_point().coords[0])
                piece_center_3d = raw_proj_to_3d(np.array([piece_center_2d]))[0]
                nearest_idx = int(np.argmin(np.linalg.norm(source_centers - piece_center_3d, axis=1)))
                piece_ref_normal = source_normals[nearest_idx]

                # ── Winding check on clean projected coords (no snapping yet).
                # Match each merged piece to the nearest original wall normal.
                # This preserves the raw JSON face direction even in courtyard
                # layouts where a global group average can be ambiguous.
                raw_ext_3d = raw_proj_to_3d(exterior_2d)
                raw_normal = get_poly_normal(raw_ext_3d)
                if np.dot(raw_normal, piece_ref_normal) < 0:
                    exterior_2d = exterior_2d[::-1]
                    raw_ext_3d = raw_ext_3d[::-1]

                # Primary build: snap to source vertices for roofline continuity,
                # removing the re-projection step so we don't flatten the roofline bulges.
                exterior_3d = snap_ring(raw_ext_3d)
                exterior_3d = restore_collinear_vertices(exterior_3d, combined_vertices)
                exterior_normal = get_poly_normal(exterior_3d)
                if np.dot(exterior_normal, piece_ref_normal) < 0:
                    exterior_3d = exterior_3d[::-1]
                    exterior_normal = -exterior_normal

                final_rings_3d = [exterior_3d]

                for interior in upoly.interiors:
                    interior_2d = np.array(interior.coords[:-1])
                    if ShapelyPolygon(interior_2d).area < 0.05:
                        continue
                    # Holes must wind OPPOSITE to the exterior (inward normal).
                    raw_int_3d = raw_proj_to_3d(interior_2d)
                    raw_int_normal = get_poly_normal(raw_int_3d)
                    if np.dot(raw_int_normal, exterior_normal) > 0:
                        interior_2d = interior_2d[::-1]
                        raw_int_3d = raw_int_3d[::-1]
                    int_3d = snap_ring(raw_int_3d)
                    int_3d = restore_collinear_vertices(int_3d, combined_vertices)
                    final_rings_3d.append(int_3d)

                candidate = WallSurface(
                    final_rings_3d,
                    semantic_val=wall_surfaces[group[0]].semantic_val
                )

                # If snapped+planar candidate is still invalid, keep merged wall
                # by rebuilding this piece from unsnapped projected geometry.
                if not _surface_is_valid_for_export(candidate):
                    fallback_rings = [restore_collinear_vertices(raw_ext_3d, combined_vertices, tol=2e-3)]
                    for interior in upoly.interiors:
                        interior_2d = np.array(interior.coords[:-1])
                        if ShapelyPolygon(interior_2d).area < 0.05:
                            continue
                        raw_int_3d = raw_proj_to_3d(interior_2d)
                        if np.dot(get_poly_normal(raw_int_3d), get_poly_normal(raw_ext_3d)) > 0:
                            raw_int_3d = raw_int_3d[::-1]
                        raw_int_3d = restore_collinear_vertices(raw_int_3d, combined_vertices, tol=2e-3)
                        fallback_rings.append(raw_int_3d)

                    fallback_candidate = WallSurface(
                        fallback_rings,
                        semantic_val=wall_surfaces[group[0]].semantic_val
                    )

                    if _surface_is_valid_for_export(fallback_candidate):
                        print(f"  ⚠ Rebuilt merged piece without snapping in {obj_id or 'object'}")
                        candidate = fallback_candidate
                    else:
                        # Final fallback: keep merged exterior and drop problematic
                        # holes for this piece so we retain facade continuity.
                        exterior_only = WallSurface(
                            [raw_ext_3d],
                            semantic_val=wall_surfaces[group[0]].semantic_val
                        )
                        if _surface_is_valid_for_export(exterior_only):
                            print(f"  ⚠ Rebuilt merged piece as exterior-only in {obj_id or 'object'}")
                            candidate = exterior_only
                        else:
                            # Last resort patch: keep merged intent by filling with
                            # a convex-hull face in the same wall plane.
                            patch_poly = upoly.convex_hull
                            if patch_poly.is_empty or patch_poly.geom_type != 'Polygon':
                                print(f"  ⚠ Skipped unrepairable merged piece in {obj_id or 'object'}")
                                continue

                            patch_2d = np.array(patch_poly.exterior.coords[:-1])
                            patch_3d = project_ring_to_plane(
                                raw_proj_to_3d(patch_2d),
                                raw_ext_3d[0],
                                piece_ref_normal,
                            )
                            if np.dot(get_poly_normal(patch_3d), piece_ref_normal) < 0:
                                patch_3d = patch_3d[::-1]

                            patch_candidate = WallSurface(
                                [patch_3d],
                                semantic_val=wall_surfaces[group[0]].semantic_val
                            )
                            if _surface_is_valid_for_export(patch_candidate):
                                print(f"  ⚠ Patched merged piece via convex hull in {obj_id or 'object'}")
                                candidate = patch_candidate
                            else:
                                # Never fall back to original faces; skip only if
                                # no merged repair could be made.
                                print(f"  ⚠ Skipped unrepairable merged piece in {obj_id or 'object'}")
                                continue

                piece_surfaces.append(candidate)
                produced += 1

            merged_surfaces.extend(piece_surfaces)

            merge_count += len(group) - produced
            label = obj_id if obj_id else "object"
            print(f"  ✓ Merged {len(group)} → {produced} surface(s) in {label}")

        except Exception as e:
            print(f"  ⚠ Warning: Could not merge group of {len(group)} surfaces: {e}")
            for idx in group:
                merged_surfaces.append(wall_surfaces[idx])

    return merged_surfaces, merge_count


def process_cityjson(cm: dict, normal_threshold: float, distance_threshold: float, ground_tolerance: float = 0.5):
    """Process CityJSON dict in-place by merging wall surfaces attached to the ground.

    Correctly handles:
    - MultiSurface / CompositeSurface (flat values list, 1-to-1 with boundaries)
    - Solid (nested values list, one list per shell)
    - Preservation of material and texture parallel arrays
    - Preservation of semantic parent/children index relationships after merging
    """
    world_verts = decode_vertices(cm)
    t = cm.get("transform", {})
    scale = np.array(t.get("scale", [1, 1, 1]), dtype=np.float64)
    translate = np.array(t.get("translate", [0, 0, 0]), dtype=np.float64)

    total_original = 0
    total_merged = 0

    vertices = cm.get("vertices", [])
    vertex_map = {tuple(v): i for i, v in enumerate(vertices)}

    def get_indices(b):
        if isinstance(b, list):
            for item in b:
                yield from get_indices(item)
        else:
            yield b

    def get_material_theme_values(geom):
        """Return a dict of {theme_name: values_list} for all material themes."""
        result = {}
        for theme, mat in geom.get("material", {}).items():
            result[theme] = mat.get("values", [])
        return result

    def get_texture_theme_values(geom):
        """Return a dict of {theme_name: values_list} for all texture themes."""
        result = {}
        for theme, tex in geom.get("texture", {}).items():
            result[theme] = tex.get("values", [])
        return result

    for obj_id, obj in cm.get("CityObjects", {}).items():
        # Compute minimum Z across all geometry of this object
        obj_z_min = float('inf')
        for geom in obj.get("geometry", []):
            indices = list(get_indices(geom.get("boundaries", [])))
            if indices:
                z_vals = [world_verts[i][2] for i in set(indices)]
                if z_vals:
                    obj_z_min = min(obj_z_min, min(z_vals))

        for geom in obj.get("geometry", []):
            geom_type = geom.get("type")
            if geom_type not in ["Solid", "MultiSurface", "CompositeSurface"]:
                continue

            boundaries = geom.get("boundaries", [])
            semantics = geom.get("semantics", {})
            surfaces = semantics.get("surfaces", [])
            values = semantics.get("values", [])

            if not boundaries or not surfaces or not values:
                continue

            # MultiSurface / CompositeSurface: boundaries is a flat list of polygons,
            # values is a flat list of semantic indices — one entry per polygon.
            # Solid: boundaries is a list of shells, values is a list of lists.
            is_solid = (geom_type == "Solid")
            shells = boundaries if is_solid else [boundaries]
            shell_values = values if is_solid else [values]

            # Collect parallel arrays (material / texture) per shell
            mat_themes = get_material_theme_values(geom)
            tex_themes = get_texture_theme_values(geom)

            # For Solid the parallel arrays are also nested per shell; for
            # MultiSurface they are flat — wrap them to match the shell loop.
            def split_parallel_by_shell(flat_or_nested, shells_list):
                """Return a list-of-lists aligned with shells_list."""
                if is_solid:
                    return flat_or_nested  # already nested
                return [flat_or_nested]    # wrap flat list as single shell

            mat_shell_values = {th: split_parallel_by_shell(vals, shells)
                                for th, vals in mat_themes.items()}
            tex_shell_values = {th: split_parallel_by_shell(vals, shells)
                                for th, vals in tex_themes.items()}

            # Identify WallSurface semantic indices
            wall_semantic_indices = {i for i, srf in enumerate(surfaces)
                                     if srf.get("type") == "WallSurface"}
            if not wall_semantic_indices:
                continue

            # Build a map: old_surface_index → set of child surface indices
            children_map: dict = {}   # parent_idx -> [child_idx, ...]
            parent_map: dict  = {}    # child_idx  -> parent_idx
            for i, srf in enumerate(surfaces):
                if "children" in srf:
                    children_map[i] = list(srf["children"])
                if "parent" in srf:
                    parent_map[i] = srf["parent"]

            new_shells = []
            new_shell_values = []
            new_mat_shell_values = {th: [] for th in mat_themes}
            new_tex_shell_values = {th: [] for th in tex_themes}

            for shell_idx, (shell, s_vals) in enumerate(zip(shells, shell_values)):
                # Gather parallel arrays for this shell
                s_mat = {th: mat_shell_values[th][shell_idx] for th in mat_themes}
                s_tex = {th: tex_shell_values[th][shell_idx] for th in tex_themes}

                # Separate wall polygons from non-wall polygons.
                # A WallSurface polygon is eligible for merging only if it is
                # attached to the ground (z_min ≤ obj_z_min + ground_tolerance).
                wall_surfaces: List[WallSurface] = []
                wall_poly_indices: List[int] = []   # position in original shell

                non_wall_polygons = []
                non_wall_vals = []
                non_wall_mat = {th: [] for th in mat_themes}
                non_wall_tex = {th: [] for th in tex_themes}

                for pi, (polygon, p_val) in enumerate(zip(shell, s_vals)):
                    if p_val in wall_semantic_indices:
                        try:
                            rings_coords = [world_verts[np.array(ring)] for ring in polygon]
                            ext_coords = rings_coords[0]
                            poly_z_min = np.min(ext_coords[:, 2])

                            if poly_z_min <= obj_z_min + ground_tolerance:
                                ws = WallSurface(rings_coords, semantic_val=p_val, original_polygon=polygon)
                                wall_surfaces.append(ws)
                                wall_poly_indices.append(pi)
                                total_original += 1
                                continue  # handled separately below
                        except (IndexError, TypeError, KeyError):
                            pass  # fall through to non-wall

                    non_wall_polygons.append(polygon)
                    non_wall_vals.append(p_val)
                    for th in mat_themes:
                        non_wall_mat[th].append(s_mat[th][pi] if pi < len(s_mat[th]) else None)
                    for th in tex_themes:
                        non_wall_tex[th].append(s_tex[th][pi] if pi < len(s_tex[th]) else None)

                # Merge the collected wall surfaces
                if wall_surfaces:
                    merged, _ = merge_wall_surfaces(
                        wall_surfaces, normal_threshold, distance_threshold, obj_id=obj_id
                    )
                    total_merged += len(merged)

                    for ms in merged:
                        if ms.original_polygon is not None:
                            # Surface was not merged — keep original polygon & parallel values.
                            # Use object identity (id()) to find it in wall_surfaces list.
                            match_pi = next(
                                (wall_poly_indices[wi] for wi, ws in enumerate(wall_surfaces) if ws is ms),
                                None
                            )
                            non_wall_polygons.append(ms.original_polygon)
                            non_wall_vals.append(ms.semantic_val)
                            for th in mat_themes:
                                non_wall_mat[th].append(s_mat[th][match_pi] if match_pi is not None and match_pi < len(s_mat[th]) else None)
                            for th in tex_themes:
                                non_wall_tex[th].append(s_tex[th][match_pi] if match_pi is not None and match_pi < len(s_tex[th]) else None)
                        else:
                            # Surface was truly merged — encode new vertices
                            new_polygon = []
                            for ring_3d in ms.rings_coords:
                                new_ring = []
                                for pt in ring_3d:
                                    enc = encode_vertex(pt, scale, translate)
                                    new_ring.append(get_vertex_id(enc, vertices, vertex_map))
                                new_polygon.append(new_ring)

                            non_wall_polygons.append(new_polygon)
                            non_wall_vals.append(ms.semantic_val)
                            # Merged surface: use None for material/texture (no single source)
                            for th in mat_themes:
                                non_wall_mat[th].append(None)
                            for th in tex_themes:
                                non_wall_tex[th].append(None)

                new_shells.append(non_wall_polygons)
                new_shell_values.append(non_wall_vals)
                for th in mat_themes:
                    new_mat_shell_values[th].append(non_wall_mat[th])
                for th in tex_themes:
                    new_tex_shell_values[th].append(non_wall_tex[th])

            # --- Rebuild semantic surfaces with corrected parent/children indices ---
            # After merging, the semantic_val integers in new_shell_values still
            # reference the original surfaces list, so the surfaces list and the
            # parent/children cross-references remain valid as-is — no reindexing
            # needed unless surfaces were removed.  The merge only removes polygon
            # rows; it never removes surface *definitions* from the surfaces list.
            # We do need to remove definitions for surfaces that no longer appear
            # in any values list, and update children/parent refs accordingly.
            all_new_vals_flat = []
            for sv in new_shell_values:
                if isinstance(sv, list):
                    for v in sv:
                        if isinstance(v, list):
                            all_new_vals_flat.extend(v)
                        else:
                            all_new_vals_flat.append(v)

            used_surface_indices = set(v for v in all_new_vals_flat if v is not None)

            # Expand used set to include parents of used children and children of used parents
            expanded = set(used_surface_indices)
            for idx in list(used_surface_indices):
                if idx in parent_map:
                    expanded.add(parent_map[idx])
                if idx in children_map:
                    expanded.update(children_map[idx])
            used_surface_indices = expanded

            # Build old→new surface index mapping (keep only used surfaces, in original order)
            old_to_new = {}
            new_surfaces = []
            for old_idx, srf in enumerate(surfaces):
                if old_idx in used_surface_indices:
                    new_idx = len(new_surfaces)
                    old_to_new[old_idx] = new_idx
                    new_surfaces.append(dict(srf))  # shallow copy

            # Update parent/children references in new_surfaces
            for new_idx, srf in enumerate(new_surfaces):
                if "parent" in srf and srf["parent"] in old_to_new:
                    srf["parent"] = old_to_new[srf["parent"]]
                elif "parent" in srf:
                    del srf["parent"]

                if "children" in srf:
                    new_children = [old_to_new[c] for c in srf["children"] if c in old_to_new]
                    if new_children:
                        srf["children"] = new_children
                    else:
                        del srf["children"]

            # Remap value arrays to new surface indices
            def remap_vals(vals):
                if isinstance(vals, list):
                    return [remap_vals(v) for v in vals]
                return old_to_new.get(vals, vals) if vals is not None else None

            new_shell_values_remapped = remap_vals(new_shell_values)

            # Write back to geom
            if is_solid:
                geom["boundaries"] = new_shells
                geom["semantics"]["surfaces"] = new_surfaces
                geom["semantics"]["values"] = new_shell_values_remapped
                for th in mat_themes:
                    geom["material"][th]["values"] = new_mat_shell_values[th]
                for th in tex_themes:
                    geom["texture"][th]["values"] = new_tex_shell_values[th]
            else:
                geom["boundaries"] = new_shells[0]
                geom["semantics"]["surfaces"] = new_surfaces
                geom["semantics"]["values"] = new_shell_values_remapped[0]
                for th in mat_themes:
                    geom["material"][th]["values"] = new_mat_shell_values[th][0]
                for th in tex_themes:
                    geom["texture"][th]["values"] = new_tex_shell_values[th][0]

    cm["vertices"] = list(vertices)
    return total_original, total_merged

def run(input_json, output_json, normal_threshold=5.0,
        distance_threshold=2.0, ground_tolerance=0.5):
    """
    Pipeline-safe wrapper for wall surface merging.
    """

    if input_json is None:
        raise ValueError("input_json is required")

    if output_json is None:
        raise ValueError("output_json is required")

    if not os.path.exists(input_json):
        raise FileNotFoundError(f"File not found: {input_json}")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    print("STEP 03: Wall Surface Merging")

    with open(input_json, "r", encoding="utf-8") as f:
        cm = json.load(f)

    total_original, total_merged = process_cityjson(
        cm,
        normal_threshold=normal_threshold,
        distance_threshold=distance_threshold,
        ground_tolerance=ground_tolerance
    )

    # write output
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(cm, f, separators=(",", ":"))

    print(f"Wall merge done: {total_original} → {total_merged}")

    return {
        "output": output_json,
        "original_surfaces": total_original,
        "merged_surfaces": total_merged
    }