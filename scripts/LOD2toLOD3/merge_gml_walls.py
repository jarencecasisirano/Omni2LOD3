#!/usr/bin/env python3
"""
Merge fragmented WallSurfaces in CityGML files.

This preprocessing script:
1. Reads a CityGML file from data/lod_2
2. Merges adjacent WallSurfaces with similar normals
3. Saves the merged GML to outputs/00_gml_wall_merged
"""

import os
import argparse
import numpy as np
from lxml import etree
from pathlib import Path
from typing import List, Tuple, Optional
from scipy.spatial import ConvexHull


# XML Namespaces for CityGML
NAMESPACES = {
    'gml': 'http://www.opengis.net/gml',
    'bldg': 'http://www.opengis.net/citygml/building/2.0',
    'core': 'http://www.opengis.net/citygml/2.0'
}

DEFAULT_INPUT_DIR = 'data/lod_2'
DEFAULT_OUTPUT_DIR = 'outputs/00_gml_wall_merged'


class WallSurface:
    """Represents a WallSurface from CityGML with its geometry."""
    
    def __init__(self, surface_id: str, coordinates: np.ndarray, original_element=None):
        self.id = surface_id
        self.coordinates = coordinates  # Nx3 array of vertices
        self.bbox_min = np.min(coordinates, axis=0)
        self.bbox_max = np.max(coordinates, axis=0)
        self._normal = None
        self.original_element = original_element  # Store original XML element
    
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
        # Check if bounding boxes are close
        center1 = self.get_center()
        center2 = other.get_center()
        distance = np.linalg.norm(center1 - center2)
        
        if distance > distance_threshold * 10:  # Quick rejection
            return False
        
        # Check for any shared vertices
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
        # Get normals
        normal1 = self.get_normal()
        
        # Define plane using this surface: normal · (point - point_on_plane) = 0
        point_on_plane1 = self.coordinates[0]
        
        # Check distances of all points in other surface to this plane
        distances = []
        for point in other.coordinates:
            # Distance from point to plane
            d = abs(np.dot(normal1, point - point_on_plane1))
            distances.append(d)
        
        max_distance = max(distances)
        
        # If all points are close to the plane, surfaces are coplanar
        return max_distance < plane_distance_threshold


def select_gml_file(gml_dir: str = DEFAULT_INPUT_DIR) -> Optional[str]:
    """Interactive selection of GML file from directory."""
    if not os.path.exists(gml_dir):
        print(f"ERROR: GML directory not found: {gml_dir}")
        return None
    
    gml_files = sorted([f for f in os.listdir(gml_dir) if f.endswith('.gml')])
    
    if not gml_files:
        print(f"ERROR: No .gml files found in {gml_dir}")
        return None
    
    # Auto-select if only one file
    if len(gml_files) == 1:
        selected = os.path.join(gml_dir, gml_files[0])
        print(f"\n✓ Auto-selected (only one GML file): {gml_files[0]}")
        return selected
    
    print("\n" + "=" * 80)
    print("SELECT GML FILE TO MERGE")
    print("=" * 80)
    print(f"\nAvailable GML files in {gml_dir}:")
    
    for i, filename in enumerate(gml_files):
        filepath = os.path.join(gml_dir, filename)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  [{i}] {filename} ({size_mb:.2f} MB)")
    
    while True:
        try:
            response = input(f"\nSelect GML file (0-{len(gml_files)-1}, or 'q' to quit): ")
            
            if response.lower() == 'q':
                return None
            
            idx = int(response)
            if 0 <= idx < len(gml_files):
                selected = os.path.join(gml_dir, gml_files[idx])
                print(f"✓ Selected: {gml_files[idx]}")
                return selected
            else:
                print(f"Invalid index. Please enter 0-{len(gml_files)-1}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")


def parse_gml_wallsurfaces(gml_file: str) -> Tuple[List[WallSurface], etree._ElementTree]:
    """
    Parse CityGML file and extract all WallSurface geometries.
    Returns both the WallSurface objects and the original tree.
    """
    print(f"\nParsing GML file: {gml_file}")
    tree = etree.parse(gml_file)
    root = tree.getroot()
    
    wall_surfaces = []
    
    # Find all WallSurface elements
    wall_elements = root.xpath('//bldg:WallSurface', namespaces=NAMESPACES)
    print(f"Found {len(wall_elements)} WallSurface elements")
    
    for wall_elem in wall_elements:
        # Find the polygon within this wall surface
        polygons = wall_elem.xpath('.//gml:Polygon', namespaces=NAMESPACES)
        
        for polygon in polygons:
            # Get the polygon ID
            polygon_id = polygon.get('{http://www.opengis.net/gml}id', 'unknown')
            
            # Extract coordinates from posList
            pos_lists = polygon.xpath('.//gml:posList', namespaces=NAMESPACES)
            
            for pos_list in pos_lists:
                text = pos_list.text.strip()
                if not text:
                    continue
                
                # Parse space-separated coordinates
                coords = list(map(float, text.split()))
                
                # Group into (x, y, z) triplets
                coords_array = np.array(coords).reshape(-1, 3)
                
                wall_surface = WallSurface(polygon_id, coords_array, wall_elem)
                wall_surfaces.append(wall_surface)
    
    print(f"Loaded {len(wall_surfaces)} wall surfaces")
    return wall_surfaces, tree


def merge_wall_surfaces(wall_surfaces: List[WallSurface], 
                        normal_angle_threshold: float = 5.0,
                        distance_threshold: float = 2.0) -> List[WallSurface]:
    """Merge adjacent wall surfaces with similar normals."""
    print(f"\nMerging wall surfaces...")
    print(f"  Initial count: {len(wall_surfaces)}")
    print(f"  Normal angle threshold: {normal_angle_threshold}°")
    print(f"  Distance threshold: {distance_threshold}m")
    
    if len(wall_surfaces) == 0:
        return []
    
    # Convert angle threshold to radians
    angle_threshold_rad = np.radians(normal_angle_threshold)
    cos_threshold = np.cos(angle_threshold_rad)
    
    # Track which surfaces have been merged
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
                
                # Check if adjacent or coplanar to any surface in the group
                is_adjacent_to_group = False
                for group_idx in group:
                    group_surface = wall_surfaces[group_idx]
                    # Check traditional adjacency (shared vertices)
                    if group_surface.is_adjacent(other_wall, distance_threshold):
                        is_adjacent_to_group = True
                        break
                    # Also check if coplanar (allows merging at different heights)
                    if group_surface.is_coplanar(other_wall, plane_distance_threshold=0.5):
                        is_adjacent_to_group = True
                        break
                
                if is_adjacent_to_group:
                    group.append(j)
                    merged_flags[j] = True
                    changed = True
        
        # Create merged surface from group
        if len(group) == 1:
            # No merging needed
            merged_surfaces.append(wall_surfaces[group[0]])
        else:
            # Merge multiple surfaces
            all_vertices = []
            merged_ids = []
            
            for idx in group:
                all_vertices.append(wall_surfaces[idx].coordinates)
                merged_ids.append(wall_surfaces[idx].id)
            
            # Combine all vertices
            combined_vertices = np.vstack(all_vertices)
            
            # Compute convex hull to get outer boundary
            try:
                # Project to 2D for convex hull
                centroid = np.mean(combined_vertices, axis=0)
                centered = combined_vertices - centroid
                
                # Use PCA to find best 2D projection plane
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                # Sort by eigenvalue
                idx_sort = eigenvalues.argsort()[::-1]
                eigenvectors = eigenvectors[:, idx_sort]
                
                # Project to 2D using first two eigenvectors
                projected_2d = centered @ eigenvectors[:, :2]
                
                # Compute 2D convex hull
                hull_2d = ConvexHull(projected_2d)
                hull_indices = hull_2d.vertices
                
                # Get 3D coordinates of hull vertices
                hull_vertices_3d = combined_vertices[hull_indices]
                
                # Create merged surface
                merged_id = f"MERGED_{len(group)}_surfaces"
                merged_surface = WallSurface(merged_id, hull_vertices_3d, wall_surfaces[group[0]].original_element)
                merged_surfaces.append(merged_surface)
                
                merge_count += len(group) - 1
                print(f"  ✓ Merged {len(group)} surfaces into {merged_id}")
                
            except Exception as e:
                print(f"  ⚠ Warning: Could not merge group of {len(group)} surfaces: {e}")
                # Fall back to keeping original surfaces
                for idx in group:
                    merged_surfaces.append(wall_surfaces[idx])
    
    print(f"\n  Final count: {len(merged_surfaces)}")
    print(f"  Merged: {merge_count} surfaces")
    return merged_surfaces


def write_merged_gml(merged_surfaces: List[WallSurface], original_tree: etree._ElementTree, 
                     output_path: str):
    """Write merged wall surfaces back to a GML file."""
    print(f"\nWriting merged GML file...")
    
    # Parse the original tree
    root = original_tree.getroot()
    
    # Find all WallSurface elements and remove them
    wall_elements = root.xpath('//bldg:WallSurface', namespaces=NAMESPACES)
    
    for wall_elem in wall_elements:
        parent = wall_elem.getparent()
        if parent is not None:
            parent.remove(wall_elem)
    
    # Find the building element to add merged surfaces
    building_elem = root.xpath('//bldg:Building', namespaces=NAMESPACES)[0]
    
    # Add merged surfaces
    for i, merged_surface in enumerate(merged_surfaces):
        # Create WallSurface element
        wall_surface = etree.SubElement(
            building_elem,
            f"{{{NAMESPACES['bldg']}}}boundedBy"
        )
        
        wall_surface_elem = etree.SubElement(
            wall_surface,
            f"{{{NAMESPACES['bldg']}}}WallSurface"
        )
        
        # Create MultiSurface
        lod2_multi = etree.SubElement(
            wall_surface_elem,
            f"{{{NAMESPACES['bldg']}}}lod2MultiSurface"
        )
        
        multi_surface = etree.SubElement(
            lod2_multi,
            f"{{{NAMESPACES['gml']}}}MultiSurface"
        )
        
        surface_member = etree.SubElement(
            multi_surface,
            f"{{{NAMESPACES['gml']}}}surfaceMember"
        )
        
        # Create Polygon
        polygon = etree.SubElement(
            surface_member,
            f"{{{NAMESPACES['gml']}}}Polygon",
            {f"{{{NAMESPACES['gml']}}}id": merged_surface.id}
        )
        
        exterior = etree.SubElement(
            polygon,
            f"{{{NAMESPACES['gml']}}}exterior"
        )
        
        linear_ring = etree.SubElement(
            exterior,
            f"{{{NAMESPACES['gml']}}}LinearRing"
        )
        
        pos_list = etree.SubElement(
            linear_ring,
            f"{{{NAMESPACES['gml']}}}posList"
        )
        
        # Convert coordinates to string
        coords_flat = merged_surface.coordinates.flatten()
        pos_list.text = ' '.join(map(str, coords_flat))
    
    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    original_tree.write(output_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')
    
    print(f"  ✓ Saved to: {output_path}")
    print(f"  ✓ Wall surfaces: {len(merged_surfaces)}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge fragmented WallSurfaces in CityGML files"
    )
    parser.add_argument('--input_file', type=str,
                       help='Path to input CityGML file (if not provided, will prompt for selection)')
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR,
                       help=f'Directory containing input GML files (default: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory for merged GML files (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--normal_threshold', type=float, default=5.0,
                       help='Normal angle threshold in degrees (default: 5.0)')
    parser.add_argument('--distance_threshold', type=float, default=2.0,
                       help='Distance threshold in meters for adjacency (default: 2.0)')
    
    args = parser.parse_args()
    
    # Select input GML file
    if args.input_file:
        gml_file = args.input_file
        if not os.path.exists(gml_file):
            print(f"ERROR: GML file not found: {gml_file}")
            return
    else:
        gml_file = select_gml_file(args.input_dir)
        if not gml_file:
            print("Cancelled by user.")
            return
    
    # Parse GML
    wall_surfaces, original_tree = parse_gml_wallsurfaces(gml_file)
    
    if not wall_surfaces:
        print("ERROR: No WallSurfaces found in GML file")
        return
    
    # Merge surfaces
    merged_surfaces = merge_wall_surfaces(
        wall_surfaces,
        normal_angle_threshold=args.normal_threshold,
        distance_threshold=args.distance_threshold
    )
    
    # Generate output filename
    input_basename = os.path.basename(gml_file)
    output_filename = input_basename.replace('.gml', '_merged.gml')
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Write merged GML
    write_merged_gml(merged_surfaces, original_tree, output_path)
    
    print("\n" + "=" * 80)
    print("Wall surface merging complete!")
    print(f"  Input: {gml_file}")
    print(f"  Output: {output_path}")
    print(f"  Original surfaces: {len(wall_surfaces)}")
    print(f"  Merged surfaces: {len(merged_surfaces)}")
    print(f"  Reduction: {len(wall_surfaces) - len(merged_surfaces)} surfaces merged")
    print("=" * 80)


if __name__ == '__main__':
    main()
