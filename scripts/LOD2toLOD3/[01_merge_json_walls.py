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


class WallSurface:
    """Represents a WallSurface from CityJSON with its geometry."""
    
    def __init__(self, coordinates: np.ndarray, semantic_val: int = None, original_polygon=None):
        self.coordinates = coordinates  # Nx3 array of vertices
        self.bbox_min = np.min(coordinates, axis=0)
        self.bbox_max = np.max(coordinates, axis=0)
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
        
        if len(self.coordinates) < 3:
            return np.array([0, 0, 1])
        
        # Use first three vertices to compute normal via cross product
        v1 = self.coordinates[1] - self.coordinates[0]
        v2 = self.coordinates[2] - self.coordinates[0]
        normal = np.cross(v1, v2)
        
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
        else:
            normal = np.array([0, 0, 1])
        
        self._normal = normal
        return normal
    
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
    
    # Auto-select if only one file
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


def merge_wall_surfaces(wall_surfaces: List[WallSurface], 
                        normal_angle_threshold: float = 5.0,
                        distance_threshold: float = 2.0,
                        obj_id: str = "") -> Tuple[List[WallSurface], int]:
    """Merge adjacent wall surfaces with similar normals."""
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
        
        # Start a new group with this surface
        group = [i]
        group_normal = wall.get_normal()
        merged_flags[i] = True
        
        # Find all adjacent/coplanar surfaces with similar normals
        changed = True
        while changed:
            changed = False
            for j in range(len(wall_surfaces)):
                if merged_flags[j]:
                    continue
                
                other_wall = wall_surfaces[j]
                other_normal = other_wall.get_normal()
                
                # Check normal similarity
                dot_product = np.dot(group_normal, other_normal)
                normals_similar = abs(dot_product) >= cos_threshold
                
                if not normals_similar:
                    continue
                
                # Check if adjacent (sharing vertices) to any surface in the group
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
        
        # Create merged surface from group
        if len(group) == 1:
            merged_surfaces.append(wall_surfaces[group[0]])
        else:
            all_vertices = []
            for idx in group:
                all_vertices.append(wall_surfaces[idx].coordinates)
            
            combined_vertices = np.vstack(all_vertices)
            
            try:
                # PCA to find the best 2D projection plane for the wall
                centroid = np.mean(combined_vertices, axis=0)
                centered = combined_vertices - centroid
                
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                idx_sort = eigenvalues.argsort()[::-1]
                eigenvectors = eigenvectors[:, idx_sort].real
                
                # Build a Shapely polygon for each wall surface in 2D
                shapely_polys = []
                for idx in group:
                    coords_3d = wall_surfaces[idx].coordinates
                    coords_2d = (coords_3d - centroid) @ eigenvectors[:, :2]
                    poly = ShapelyPolygon(coords_2d)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    if not poly.is_empty:
                        shapely_polys.append(poly)
                
                if not shapely_polys:
                    raise ValueError("All projected polygons are empty")
                
                # Buffer slightly to close micro-gaps between adjacent polygons
                eps = 1e-3
                buffered_polys = [p.buffer(eps) for p in shapely_polys]
                
                # Spatial union – preserves true outline (no diagonals)
                union_poly = unary_union(buffered_polys).buffer(-eps)
                
                # If the union is a MultiPolygon, take the largest piece
                if union_poly.geom_type == 'MultiPolygon':
                    union_poly = max(union_poly.geoms, key=lambda p: p.area)
                
                # Strip any interior holes (artifacts from imprecise overlaps)
                if union_poly.geom_type == 'Polygon' and list(union_poly.interiors):
                    union_poly = ShapelyPolygon(union_poly.exterior)
                
                # Extract exterior ring (drop the duplicate closing vertex)
                exterior_2d = np.array(union_poly.exterior.coords[:-1])
                
                # Reconstruct 3D coordinates from the 2D projection
                exterior_3d = exterior_2d @ eigenvectors[:, :2].T + centroid
                
                # Snap each vertex to nearest original vertex to prevent gaps
                # with neighboring non-merged surfaces
                snapped = []
                for pt in exterior_3d:
                    dists = np.linalg.norm(combined_vertices - pt, axis=1)
                    min_idx = np.argmin(dists)
                    if dists[min_idx] < 0.05:
                        snapped.append(combined_vertices[min_idx])
                    else:
                        snapped.append(pt)
                exterior_3d = np.array(snapped)
                
                # Check winding order w.r.t the original outward normal
                v1 = exterior_3d[1] - exterior_3d[0]
                v2 = exterior_3d[2] - exterior_3d[0]
                new_normal = np.cross(v1, v2)
                
                if np.dot(new_normal, group_normal) < 0:
                    exterior_3d = exterior_3d[::-1]
                
                merged_surface = WallSurface(exterior_3d, semantic_val=wall_surfaces[group[0]].semantic_val)
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
    """Process CityJSON dict in-place by merging wall surfaces attached to the ground."""
    world_verts = decode_vertices(cm)
    t = cm.get("transform", {})
    scale = np.array(t.get("scale", [1, 1, 1]), dtype=np.float64)
    translate = np.array(t.get("translate", [0, 0, 0]), dtype=np.float64)
    
    total_original = 0
    total_merged = 0
    
    vertices = cm.get("vertices", [])
    
    def get_indices(b):
        if isinstance(b, list):
            for item in b:
                yield from get_indices(item)
        else:
            yield b
    
    for obj_id, obj in cm.get("CityObjects", {}).items():
        # Typically we only merge WallSurfaces on buildings, but can do it for any valid object
        if obj.get("type") not in ["Building", "BuildingPart"]:
            pass 
            
        # Find the absolute minimum Z of this object
        obj_z_min = float('inf')
        for g_idx, geom in enumerate(obj.get("geometry", [])):
            indices = list(get_indices(geom.get("boundaries", [])))
            if indices:
                z_vals = [world_verts[i][2] for i in set(indices)]
                if z_vals:
                    obj_z_min = min(obj_z_min, min(z_vals))
                    
        for g_idx, geom in enumerate(obj.get("geometry", [])):
            geom_type = geom.get("type")
            if geom_type not in ["Solid", "MultiSurface", "CompositeSurface"]:
                continue
                
            boundaries = geom.get("boundaries", [])
            semantics = geom.get("semantics", {})
            surfaces = semantics.get("surfaces", [])
            values = semantics.get("values", [])
            
            if not boundaries or not surfaces or not values:
                continue
                
            wall_semantic_indices = set()
            for i, srf in enumerate(surfaces):
                if srf.get("type") == "WallSurface":
                    wall_semantic_indices.add(i)
                    
            if not wall_semantic_indices:
                continue
                
            is_solid = (geom_type == "Solid")
            shells = boundaries if is_solid else [boundaries]
            shell_values = values if is_solid else [values]
            
            new_shells = []
            new_shell_values = []
            
            for shell, s_vals in zip(shells, shell_values):
                wall_surfaces = []
                non_wall_polygons = []
                non_wall_vals = []
                
                # Extract walls
                for polygon, p_val in zip(shell, s_vals):
                    if p_val in wall_semantic_indices:
                        try:
                            ext_ring = polygon[0]
                            coords = world_verts[np.array(ext_ring)]
                            
                            # Only merge walls attached to the ground
                            poly_z_min = np.min(coords[:, 2])
                            if poly_z_min <= obj_z_min + ground_tolerance:
                                ws = WallSurface(coords, semantic_val=p_val, original_polygon=polygon)
                                wall_surfaces.append(ws)
                                total_original += 1
                            else:
                                non_wall_polygons.append(polygon)
                                non_wall_vals.append(p_val)
                        except (IndexError, TypeError, KeyError):
                            non_wall_polygons.append(polygon)
                            non_wall_vals.append(p_val)
                    else:
                        non_wall_polygons.append(polygon)
                        non_wall_vals.append(p_val)
                
                # Merge current shell walls
                if len(wall_surfaces) > 0:
                    merged, reduced = merge_wall_surfaces(
                        wall_surfaces, 
                        normal_threshold, 
                        distance_threshold, 
                        obj_id=obj_id
                    )
                    total_merged += len(merged)
                    
                    for ms in merged:
                        if ms.original_polygon is not None:
                            non_wall_polygons.append(ms.original_polygon)
                            non_wall_vals.append(ms.semantic_val)
                        else:
                            # Re-encode back into global vertex array
                            new_ring = []
                            for pt in ms.coordinates:
                                enc = encode_vertex(pt, scale, translate)
                                vertices.append(enc)
                                new_ring.append(len(vertices) - 1)
                            
                            non_wall_polygons.append([new_ring])
                            non_wall_vals.append(ms.semantic_val)
                            
                new_shells.append(non_wall_polygons)
                new_shell_values.append(non_wall_vals)
                
            # Replace bounds and semantics back into the tree
            if is_solid:
                geom["boundaries"] = new_shells
                geom["semantics"]["values"] = new_shell_values
            else:
                geom["boundaries"] = new_shells[0]
                geom["semantics"]["values"] = new_shell_values[0]
                
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
    
    # Select input JSON file
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
            
    # Parse JSON
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
        
    # Generate output filename
    input_basename = os.path.basename(json_file)
    if input_basename.endswith('.cityjson'):
        output_filename = input_basename.replace('.cityjson', '_merged.cityjson')
    else:
        output_filename = input_basename.replace('.json', '_merged.json')
        
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Write merged JSON
    print(f"\nWriting merged JSON file...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cm, f, separators=(",", ":"))
        
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
