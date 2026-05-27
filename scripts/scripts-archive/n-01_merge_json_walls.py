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

DEFAULT_INPUT_DIR = 'data/lod_2'
DEFAULT_OUTPUT_DIR = 'outputs/00_json_wall_merged'

def newell_normal(coords: np.ndarray) -> np.ndarray:
    """Compute polygon normal using Newell's method (robust for non-convex rings)."""
    n = np.zeros(3)
    num = len(coords)
    for i in range(num):
        c  = coords[i]
        nx = coords[(i + 1) % num]
        n[0] += (c[1] - nx[1]) * (c[2] + nx[2])
        n[1] += (c[2] - nx[2]) * (c[0] + nx[0])
        n[2] += (c[0] - nx[0]) * (c[1] + nx[1])
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


def get_poly_normal(coords: np.ndarray) -> np.ndarray:
    """Alias kept for backward compatibility; delegates to newell_normal."""
    return newell_normal(coords)


def ensure_outward_winding(coords: np.ndarray, building_centroid: np.ndarray) -> np.ndarray:
    """
    Return *coords* (exterior ring vertices) with CCW winding so its Newell
    normal points AWAY from *building_centroid* (outward for a closed shell).
    If the normal already points outward the array is returned unchanged;
    otherwise a reversed copy is returned.
    """
    normal = newell_normal(coords)
    wall_centroid = coords.mean(axis=0)
    # Vector from wall centroid toward building centroid is the inward direction.
    # The outward normal should have dot < 0 with that inward vector.
    to_bldg = building_centroid - wall_centroid
    if np.dot(normal, to_bldg) > 0:
        return coords[::-1]
    return coords


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
                        obj_id: str = "",
                        building_centroid: np.ndarray = None) -> Tuple[List[WallSurface], int]:
    """Merge adjacent wall surfaces with similar normals while preserving interior holes."""
    if len(wall_surfaces) == 0:
        return [], 0
    
    angle_threshold_rad = np.radians(normal_angle_threshold)
    cos_threshold = np.cos(angle_threshold_rad)
    
    merged_flags = [False] * len(wall_surfaces)
    merged_surfaces = []
    merge_count = 0
    
    for i, wall in enumerate(wall_surfaces):
        if merged_flags[i]:
            continue
        
        group = [i]
        group_normal = wall.get_normal()
        merged_flags[i] = True
        
        changed = True
        while changed:
            changed = False
            for j in range(len(wall_surfaces)):
                if merged_flags[j]:
                    continue
                
                other_wall = wall_surfaces[j]
                other_normal = other_wall.get_normal()
                
                dot_product = np.dot(group_normal, other_normal)
                normals_similar = abs(dot_product) >= cos_threshold
                
                if not normals_similar:
                    continue
                
                is_adjacent_to_group = False
                for group_idx in group:
                    group_surface = wall_surfaces[group_idx]
                    if group_surface.is_adjacent(other_wall, distance_threshold):
                        is_adjacent_to_group = True
                        break
                
                if is_adjacent_to_group:
                    group.append(j)
                    merged_flags[j] = True
                    changed = True
        
        if len(group) == 1:
            merged_surfaces.append(wall_surfaces[group[0]])
        else:
            all_vertices = []
            for idx in group:
                for r in wall_surfaces[idx].rings_coords:
                    all_vertices.append(r)
            combined_vertices = np.vstack(all_vertices)
            
            try:
                centroid = np.mean(combined_vertices, axis=0)
                centered = combined_vertices - centroid
                
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                idx_sort = eigenvalues.argsort()[::-1]
                eigenvectors = eigenvectors[:, idx_sort].real
                
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
                buffered_polys = [p.buffer(eps) for p in shapely_polys]
                union_result = unary_union(buffered_polys).buffer(-eps)
                
                if union_result.geom_type == 'MultiPolygon':
                    union_polys_list = list(union_result.geoms)
                else:
                    union_polys_list = [union_result]
                
                def proj_to_3d(coords_2d_arr):
                    pts_3d = coords_2d_arr @ eigenvectors[:, :2].T + centroid
                    snapped = []
                    for pt in pts_3d:
                        dists = np.linalg.norm(combined_vertices - pt, axis=1)
                        if len(dists) > 0:
                            min_idx = np.argmin(dists)
                            if dists[min_idx] < 0.05:
                                snapped.append(combined_vertices[min_idx])
                                continue
                        snapped.append(pt)
                    return np.array(snapped)

                def restore_collinear_vertices(simplified_ring_3d, original_vertices_3d, tol=1e-3):
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

                for union_poly in union_polys_list:
                    if union_poly.is_empty:
                        continue
                    
                    exterior_2d = np.array(union_poly.exterior.coords[:-1])
                    exterior_3d = proj_to_3d(exterior_2d)
                    exterior_3d = restore_collinear_vertices(exterior_3d, combined_vertices)

                    # Orient exterior ring: prefer centroid-based outward check,
                    # fall back to aligning with the pre-oriented group_normal.
                    if building_centroid is not None:
                        exterior_3d = ensure_outward_winding(exterior_3d, building_centroid)
                    else:
                        new_normal = newell_normal(exterior_3d)
                        if np.dot(new_normal, group_normal) < 0:
                            exterior_3d = exterior_3d[::-1]

                    outward_normal = newell_normal(exterior_3d)
                    final_rings_3d = [exterior_3d]

                    # Recreate interior holes — must wind OPPOSITE to exterior ring.
                    for interior in union_poly.interiors:
                        interior_2d = np.array(interior.coords[:-1])
                        int_poly = ShapelyPolygon(interior_2d)
                        if int_poly.area < 0.05:
                            continue

                        interior_3d = proj_to_3d(interior_2d)
                        interior_3d = restore_collinear_vertices(interior_3d, combined_vertices)
                        
                        int_normal = newell_normal(interior_3d)
                        # Interior ring normal must point inward (opposite to outward_normal).
                        if np.dot(int_normal, outward_normal) > 0:
                            interior_3d = interior_3d[::-1]

                        final_rings_3d.append(interior_3d)
                    
                    merged_surface = WallSurface(final_rings_3d, semantic_val=wall_surfaces[group[0]].semantic_val)
                    merged_surfaces.append(merged_surface)
                
                merge_count += len(group) - 1
                if obj_id:
                    print(f"  ✓ Merged {len(group)} surfaces in {obj_id}")
                else:
                    print(f"  ✓ Merged {len(group)} surfaces")
                    
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
        # Compute minimum Z and XYZ centroid across all geometry of this object.
        obj_z_min = float('inf')
        bldg_verts_all = []
        for geom in obj.get("geometry", []):
            indices = list(get_indices(geom.get("boundaries", [])))
            if indices:
                unique_idx = list(set(indices))
                z_vals = [world_verts[i][2] for i in unique_idx]
                if z_vals:
                    obj_z_min = min(obj_z_min, min(z_vals))
                bldg_verts_all.extend(world_verts[i] for i in unique_idx)

        bldg_centroid = (np.mean(np.array(bldg_verts_all), axis=0)
                         if bldg_verts_all else None)

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
                                # ── Correct exterior ring winding (outward normal) ──
                                corrected_polygon = [list(r) for r in polygon]
                                if bldg_centroid is not None:
                                    oriented = ensure_outward_winding(ext_coords, bldg_centroid)
                                    if not np.array_equal(oriented, ext_coords):
                                        ext_coords = oriented
                                        corrected_polygon[0] = list(reversed(polygon[0]))
                                    rings_coords[0] = ext_coords

                                # ── Correct interior ring winding (opposite to exterior) ──
                                ext_normal = newell_normal(rings_coords[0])
                                for ri in range(1, len(rings_coords)):
                                    int_coords = rings_coords[ri]
                                    if np.dot(newell_normal(int_coords), ext_normal) > 0:
                                        rings_coords[ri] = int_coords[::-1]
                                        corrected_polygon[ri] = list(reversed(polygon[ri]))

                                ws = WallSurface(rings_coords, semantic_val=p_val,
                                                 original_polygon=corrected_polygon)
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
                        wall_surfaces, normal_threshold, distance_threshold,
                        obj_id=obj_id, building_centroid=bldg_centroid
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

            # Always keep ALL WallSurface definitions so that non-ground walls
            # (which were passed through unchanged with their original semantic_val)
            # never reference a pruned index after compaction.
            used_surface_indices.update(wall_semantic_indices)

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


def main():
    parser = argparse.ArgumentParser(
        description="Merge fragmented WallSurfaces in CityJSON files"
    )
    parser.add_argument('--input_file', type=str,
                       help='Path to input CityJSON file (if not provided, will prompt for selection)')
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR,
                       help=f'Directory containing input JSON files (default: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory for merged JSON files (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--normal_threshold', type=float, default=5.0,
                       help='Normal angle threshold in degrees (default: 5.0)')
    parser.add_argument('--distance_threshold', type=float, default=2.0,
                       help='Distance threshold in meters for adjacency (default: 2.0)')
    parser.add_argument('--ground_tolerance', type=float, default=0.5,
                       help='Max height above building minimum Z to be considered attached to ground (default: 0.5)')
    
    args = parser.parse_args()
    
    if args.input_file:
        json_file = args.input_file
        if not os.path.exists(json_file):
            print(f"ERROR: JSON file not found: {json_file}")
            return
    else:
        json_file = select_json_file(args.input_dir)
        if not json_file:
            print("Cancelled by user.")
            return
            
    print(f"\nParsing JSON file: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        cm = json.load(f)
        
    print(f"\nMerging wall surfaces...")
    print(f"  Normal angle threshold: {args.normal_threshold}°")
    print(f"  Distance threshold: {args.distance_threshold}m")
    print(f"  Ground tolerance: {args.ground_tolerance}m")
    
    total_original, total_merged = process_cityjson(
        cm, 
        normal_threshold=args.normal_threshold, 
        distance_threshold=args.distance_threshold,
        ground_tolerance=args.ground_tolerance
    )
    
    if total_original == 0:
        print("ERROR: No WallSurfaces found in JSON file")
        return
        
    input_basename = os.path.basename(json_file)
    if input_basename.endswith('.cityjson'):
        output_filename = input_basename.replace('.cityjson', '_merged.cityjson')
    else:
        output_filename = input_basename.replace('.json', '_merged.json')
        
    output_path = os.path.join(args.output_dir, output_filename)
    
    print(f"\nWriting merged JSON file...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Enforce CityJSON structural key order matching the target format:
    # type → version → CityObjects → transform → vertices → metadata → appearance → (rest)
    priority_keys = ["type", "version", "CityObjects", "transform", "vertices", "metadata", "appearance"]
    out_cm = {}
    for k in priority_keys:
        if k in cm:
            out_cm[k] = cm[k]
    for k, v in cm.items():
        if k not in out_cm:
            out_cm[k] = v
            
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out_cm, f, separators=(",", ":"))
        
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✓ Saved to: {output_path} ({size_mb:.2f} MB)")
    
    print("\n" + "=" * 80)
    print("Wall surface merging complete!")
    print(f"  Input: {json_file}")
    print(f"  Output: {output_path}")
    print(f"  Ground WallSurfaces processed: {total_original}")
    print(f"  Merged WallSurfaces: {total_merged}")
    print(f"  Reduction: {total_original - total_merged} surfaces merged")
    print("=" * 80)


if __name__ == '__main__':
    main()