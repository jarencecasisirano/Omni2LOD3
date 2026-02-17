#!/usr/bin/env python3
"""
Align point clouds to CityGML LOD2 building model WallSurfaces.

This script:
1. Parses a CityGML file and extracts WallSurface geometries
2. Provides an interactive tool to map point clouds to wall surfaces
3. Performs ICP registration to align point clouds to walls
4. Applies scaling adjustments based on bounding box comparison
5. Saves aligned point clouds to output directory
"""

import os
import argparse
import json
import numpy as np
import laspy
import open3d as o3d
from lxml import etree
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# XML Namespaces for CityGML parsing
NAMESPACES = {
    'gml': 'http://www.opengis.net/gml',
    'bldg': 'http://www.opengis.net/citygml/building/2.0',
    'core': 'http://www.opengis.net/citygml/2.0'
}

# Default directories
DEFAULT_GML_DIR = 'data/lod_2'
DEFAULT_POINTCLOUD_BASE_DIR = 'outputs/04_manual_cleaned_point_clouds'


def select_gml_file(gml_dir: str = DEFAULT_GML_DIR) -> Optional[str]:
    """
    Interactive selection of GML file from directory.
    
    Args:
        gml_dir: Directory containing GML files
        
    Returns:
        Path to selected GML file, or None if cancelled
    """
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
    print("SELECT GML MODEL")
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


def select_pointcloud_directory(base_dir: str = DEFAULT_POINTCLOUD_BASE_DIR) -> Optional[str]:
    """
    Interactive selection of point cloud subdirectory.
    
    Args:
        base_dir: Base directory containing point cloud subdirectories
        
    Returns:
        Path to selected subdirectory, or None if cancelled
    """
    if not os.path.exists(base_dir):
        print(f"ERROR: Point cloud base directory not found: {base_dir}")
        return None
    
    # Get all subdirectories
    subdirs = sorted([d for d in os.listdir(base_dir) 
                     if os.path.isdir(os.path.join(base_dir, d))])
    
    if not subdirs:
        print(f"ERROR: No subdirectories found in {base_dir}")
        return None
    
    # Auto-select if only one directory
    if len(subdirs) == 1:
        selected = os.path.join(base_dir, subdirs[0])
        las_count = len(list(Path(selected).glob('*.las')))
        print(f"\n✓ Auto-selected (only one directory): {subdirs[0]} ({las_count} .las files)")
        return selected
    
    print("\n" + "=" * 80)
    print("SELECT POINT CLOUD DIRECTORY")
    print("=" * 80)
    print(f"\nAvailable point cloud directories in {base_dir}:")
    
    for i, dirname in enumerate(subdirs):
        dirpath = os.path.join(base_dir, dirname)
        las_files = list(Path(dirpath).glob('*.las'))
        print(f"  [{i}] {dirname} ({len(las_files)} .las files)")
    
    while True:
        try:
            response = input(f"\nSelect directory (0-{len(subdirs)-1}, or 'q' to quit): ")
            
            if response.lower() == 'q':
                return None
            
            idx = int(response)
            if 0 <= idx < len(subdirs):
                selected = os.path.join(base_dir, subdirs[idx])
                las_count = len(list(Path(selected).glob('*.las')))
                print(f"✓ Selected: {subdirs[idx]} ({las_count} .las files)")
                return selected
            else:
                print(f"Invalid index. Please enter 0-{len(subdirs)-1}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")


class WallSurface:
    """Represents a WallSurface from CityGML with its geometry."""
    
    def __init__(self, surface_id: str, coordinates: np.ndarray):
        self.id = surface_id
        self.coordinates = coordinates  # Nx3 array of vertices
        self.bbox_min = np.min(coordinates, axis=0)
        self.bbox_max = np.max(coordinates, axis=0)
        
    def to_pointcloud(self, density: float = 0.1) -> o3d.geometry.PointCloud:
        """Convert wall surface to point cloud by sampling the polygon surface."""
        if len(self.coordinates) < 3:
            return o3d.geometry.PointCloud()
        
        # Create triangulated mesh from polygon
        vertices = self.coordinates
        n_vertices = len(vertices)
        
        # Simple fan triangulation from first vertex
        triangles = []
        for i in range(1, n_vertices - 1):
            triangles.append([0, i, i + 1])
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
        
        # Sample points from mesh surface
        num_points = max(100, int(mesh.get_surface_area() / (density ** 2)))
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        
        return pcd
    
    def get_center(self) -> np.ndarray:
        """Get the center point of the wall surface."""
        return np.mean(self.coordinates, axis=0)
    
    def get_dimensions(self) -> Tuple[float, float, float]:
        """Get the dimensions (width, height, depth) of the wall surface."""
        dims = self.bbox_max - self.bbox_min
        return tuple(dims)


def parse_gml_wallsurfaces(gml_file: str) -> List[WallSurface]:
    """
    Parse CityGML file and extract all WallSurface geometries.
    
    Args:
        gml_file: Path to CityGML file
        
    Returns:
        List of WallSurface objects
    """
    print(f"Parsing GML file: {gml_file}")
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
                
                wall_surface = WallSurface(polygon_id, coords_array)
                wall_surfaces.append(wall_surface)
                print(f"  Loaded WallSurface {polygon_id}: {len(coords_array)} vertices, "
                      f"dims={wall_surface.get_dimensions()}")
    
    return wall_surfaces


def load_las_pointcloud(las_file: str) -> o3d.geometry.PointCloud:
    """
    Load a LAS point cloud file and convert to Open3D format.
    
    Args:
        las_file: Path to LAS file
        
    Returns:
        Open3D PointCloud object
    """
    print(f"Loading point cloud: {las_file}")
    las = laspy.read(las_file)
    
    # Extract coordinates
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Load colors if available
    if hasattr(las, 'red'):
        colors = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"  Loaded {len(points):,} points")
    print(f"  Bounding box: min={np.min(points, axis=0)}, max={np.max(points, axis=0)}")
    
    return pcd


def interactive_wall_selection(wall_surfaces: List[WallSurface], 
                               pointcloud_files: List[str]) -> Dict[str, int]:
    """
    Interactive tool for user to select which wall corresponds to each point cloud.
    
    Args:
        wall_surfaces: List of WallSurface objects
        pointcloud_files: List of point cloud file paths
        
    Returns:
        Dictionary mapping point cloud filename to wall surface index
    """
    print("\n" + "=" * 80)
    print("INTERACTIVE WALL SURFACE SELECTION")
    print("=" * 80)
    
    print("\nAvailable WallSurfaces:")
    for i, wall in enumerate(wall_surfaces):
        center = wall.get_center()
        dims = wall.get_dimensions()
        print(f"  [{i}] {wall.id}")
        print(f"      Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        print(f"      Dimensions: W={dims[0]:.2f}, H={dims[1]:.2f}, D={dims[2]:.2f}")
    
    mapping = {}
    
    print("\nPoint clouds to align:")
    for pc_file in pointcloud_files:
        filename = os.path.basename(pc_file)
        print(f"\n  File: {filename}")
        
        while True:
            try:
                response = input(f"    Select wall index for {filename} (0-{len(wall_surfaces)-1}, or 's' to skip): ")
                
                if response.lower() == 's':
                    print(f"    Skipping {filename}")
                    break
                
                wall_idx = int(response)
                if 0 <= wall_idx < len(wall_surfaces):
                    mapping[filename] = wall_idx
                    print(f"    ✓ Mapped {filename} → WallSurface [{wall_idx}] {wall_surfaces[wall_idx].id}")
                    break
                else:
                    print(f"    Invalid index. Please enter 0-{len(wall_surfaces)-1}")
            except ValueError:
                print("    Invalid input. Please enter a number or 's' to skip.")
    
    print("\n" + "=" * 80)
    print(f"Mapping complete: {len(mapping)} point clouds mapped")
    print("=" * 80 + "\n")
    
    return mapping


def compute_scale_factor(source_pcd: o3d.geometry.PointCloud,
                         target_pcd: o3d.geometry.PointCloud,
                         mode: str = 'uniform') -> Tuple[np.ndarray, float]:
    """
    Compute scale factor to match source to target bounding boxes.
    
    Args:
        source_pcd: Source point cloud to be scaled
        target_pcd: Target point cloud (reference)
        mode: 'uniform', 'non-uniform', or 'none'
        
    Returns:
        Tuple of (scale_vector, uniform_scale_factor)
    """
    source_bbox = source_pcd.get_axis_aligned_bounding_box()
    target_bbox = target_pcd.get_axis_aligned_bounding_box()
    
    source_extent = source_bbox.get_extent()
    target_extent = target_bbox.get_extent()
    
    # Avoid division by zero
    source_extent = np.maximum(source_extent, 1e-6)
    
    scale_per_axis = target_extent / source_extent
    
    if mode == 'uniform':
        # Use median scale factor
        uniform_scale = np.median(scale_per_axis)
        return np.array([uniform_scale] * 3), uniform_scale
    elif mode == 'non-uniform':
        return scale_per_axis, np.mean(scale_per_axis)
    else:  # 'none'
        return np.array([1.0, 1.0, 1.0]), 1.0


def align_pointcloud_to_wall(source_pcd: o3d.geometry.PointCloud,
                             wall_surface: WallSurface,
                             scale_mode: str = 'uniform',
                             icp_threshold: float = 1.0,
                             visualize: bool = False) -> Tuple[o3d.geometry.PointCloud, np.ndarray, float]:
    """
    Align a point cloud to a wall surface using ICP registration.
    
    Args:
        source_pcd: Point cloud to align
        wall_surface: Target wall surface
        scale_mode: Scaling mode ('uniform', 'non-uniform', 'none')
        icp_threshold: Distance threshold for ICP
        visualize: Whether to visualize alignment
        
    Returns:
        Tuple of (aligned_pointcloud, transformation_matrix, fitness_score)
    """
    print(f"\nAligning to WallSurface {wall_surface.id}")
    
    # Convert wall surface to point cloud
    target_pcd = wall_surface.to_pointcloud(density=0.1)
    print(f"  Wall surface sampled to {len(target_pcd.points):,} points")
    
    # Step 1: Compute and apply scaling
    scale_vector, uniform_scale = compute_scale_factor(source_pcd, target_pcd, scale_mode)
    print(f"  Scale factors: {scale_vector} (uniform={uniform_scale:.4f})")
    
    source_scaled = o3d.geometry.PointCloud(source_pcd)
    source_scaled.scale(uniform_scale if scale_mode == 'uniform' else 1.0, 
                       center=source_scaled.get_center())
    
    # Step 2: Initial alignment via bounding box centers
    source_center = source_scaled.get_center()
    target_center = target_pcd.get_center()
    translation = target_center - source_center
    
    init_transform = np.eye(4)
    init_transform[:3, 3] = translation
    
    source_aligned = o3d.geometry.PointCloud(source_scaled)
    source_aligned.transform(init_transform)
    
    print(f"  Initial translation: {translation}")
    
    # Step 3: Compute normals for point-to-plane ICP
    target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    source_aligned.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    
    # Step 4: ICP refinement
    print(f"  Running ICP with threshold={icp_threshold}")
    reg_result = o3d.pipelines.registration.registration_icp(
        source_aligned, target_pcd, icp_threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    
    print(f"  ICP fitness: {reg_result.fitness:.4f}, RMSE: {reg_result.inlier_rmse:.4f}")
    
    # Apply ICP transformation
    final_transform = reg_result.transformation @ init_transform
    result_pcd = o3d.geometry.PointCloud(source_scaled)
    result_pcd.transform(final_transform)
    
    # Visualize if requested
    if visualize:
        print("  Visualizing alignment...")
        # Color source as red, target as blue
        source_vis = o3d.geometry.PointCloud(source_pcd)
        source_vis.paint_uniform_color([1, 0, 0])  # Red - original
        
        result_vis = o3d.geometry.PointCloud(result_pcd)
        result_vis.paint_uniform_color([0, 1, 0])  # Green - aligned
        
        target_vis = o3d.geometry.PointCloud(target_pcd)
        target_vis.paint_uniform_color([0, 0, 1])  # Blue - target wall
        
        o3d.visualization.draw_geometries([source_vis, result_vis, target_vis],
                                         window_name=f"Alignment: {wall_surface.id}",
                                         width=1280, height=720)
    
    return result_pcd, final_transform, reg_result.fitness


def save_aligned_pointcloud(pcd: o3d.geometry.PointCloud, output_path: str, original_las_path: str):
    """
    Save aligned point cloud to LAS file, preserving original attributes.
    
    Args:
        pcd: Aligned Open3D point cloud
        output_path: Output LAS file path
        original_las_path: Original LAS file to copy attributes from
    """
    print(f"  Saving to {output_path}")
    
    # Load original LAS for header info
    original_las = laspy.read(original_las_path)
    
    # Create new LAS with same header
    header = laspy.LasHeader(point_format=original_las.header.point_format, 
                            version=original_las.header.version)
    header.scales = original_las.header.scales
    header.offsets = original_las.header.offsets
    
    new_las = laspy.LasData(header)
    
    # Set coordinates from aligned point cloud
    points = np.asarray(pcd.points)
    new_las.x = points[:, 0]
    new_las.y = points[:, 1]
    new_las.z = points[:, 2]
    
    # Copy colors if available
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        new_las.red = (colors[:, 0] * 65535).astype(np.uint16)
        new_las.green = (colors[:, 1] * 65535).astype(np.uint16)
        new_las.blue = (colors[:, 2] * 65535).astype(np.uint16)
    elif hasattr(original_las, 'red'):
        # Use original colors
        new_las.red = original_las.red
        new_las.green = original_las.green
        new_las.blue = original_las.blue
    
    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_las.write(output_path)
    print(f"  ✓ Saved {len(points):,} points")


def main():
    parser = argparse.ArgumentParser(
        description="Align point clouds to CityGML building model WallSurfaces"
    )
    parser.add_argument('--gml_file', type=str,
                       help='Path to CityGML file (if not provided, will prompt for selection)')
    parser.add_argument('--gml_dir', type=str, default=DEFAULT_GML_DIR,
                       help=f'Directory containing GML files (default: {DEFAULT_GML_DIR})')
    parser.add_argument('--pointcloud_dir', type=str,
                       help='Directory containing LAS point cloud files (if not provided, will prompt for selection)')
    parser.add_argument('--pointcloud_base_dir', type=str, default=DEFAULT_POINTCLOUD_BASE_DIR,
                       help=f'Base directory for point cloud folders (default: {DEFAULT_POINTCLOUD_BASE_DIR})')
    parser.add_argument('--output_dir', type=str, default='outputs/07_aligned',
                       help='Output directory for aligned point clouds')
    parser.add_argument('--mapping_file', type=str,
                       help='JSON file with pre-defined mappings (optional)')
    parser.add_argument('--scale_mode', type=str, default='uniform',
                       choices=['uniform', 'non-uniform', 'none'],
                       help='Scaling mode for alignment')
    parser.add_argument('--icp_threshold', type=float, default=1.0,
                       help='Distance threshold for ICP registration')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize each alignment')
    parser.add_argument('--visualize_walls', action='store_true',
                       help='Only visualize wall surfaces and exit')
    
    args = parser.parse_args()
    
    # Select GML file if not provided
    if args.gml_file:
        gml_file = args.gml_file
        if not os.path.exists(gml_file):
            print(f"ERROR: GML file not found: {gml_file}")
            return
    else:
        gml_file = select_gml_file(args.gml_dir)
        if not gml_file:
            print("Cancelled by user.")
            return
    
    # Parse GML and extract wall surfaces
    wall_surfaces = parse_gml_wallsurfaces(gml_file)
    
    if not wall_surfaces:
        print("ERROR: No WallSurfaces found in GML file")
        return
    
    # Visualize walls if requested
    if args.visualize_walls:
        print("\nPreparing wall surface visualization with index markers...")
        geometries = []
        
        # Create a color map for better distinction
        import colorsys
        
        for i, wall in enumerate(wall_surfaces):
            # Create wall point cloud
            pcd = wall.to_pointcloud(density=0.05)
            
            # Assign color based on index using HSV color wheel
            hue = (i * 0.618033988749895) % 1.0  # Golden ratio for good distribution
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            pcd.paint_uniform_color(rgb)
            geometries.append(pcd)
            
            # Add a sphere marker at the center with index number
            center = wall.get_center()
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
            marker.translate(center)
            marker.paint_uniform_color([1, 0, 0])  # Red markers
            geometries.append(marker)
            
            # Print legend
            if i % 10 == 0 or i < 20:  # Print first 20 and every 10th
                print(f"  [{i}] {wall.id[:30]}... - Center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
        
        print(f"\nDisplaying {len(wall_surfaces)} wall surfaces with RED sphere markers at centers")
        print("Each wall has a unique color. Sphere markers show wall centers.")
        print("Refer to the printed list or terminal output to match index numbers to wall IDs.")
        
        o3d.visualization.draw_geometries(geometries, 
                                         window_name=f"GML WallSurfaces ({len(wall_surfaces)} walls with index markers)",
                                         width=1280, height=720)
        return
    
    # Select point cloud directory if not provided
    if args.pointcloud_dir:
        pc_dir_path = args.pointcloud_dir
        if not os.path.exists(pc_dir_path):
            print(f"ERROR: Point cloud directory not found: {pc_dir_path}")
            return
    else:
        pc_dir_path = select_pointcloud_directory(args.pointcloud_base_dir)
        if not pc_dir_path:
            print("Cancelled by user.")
            return
    
    # Find point cloud files
    pc_dir = Path(pc_dir_path)
    pc_files = sorted(pc_dir.glob('*.las'))
    
    if not pc_files:
        print(f"ERROR: No .las files found in {args.pointcloud_dir}")
        return
    
    print(f"\nFound {len(pc_files)} point cloud files")
    
    # Load or create mapping
    if args.mapping_file and os.path.exists(args.mapping_file):
        print(f"Loading mapping from {args.mapping_file}")
        with open(args.mapping_file, 'r') as f:
            mapping_data = json.load(f)
        
        # Convert wall IDs to indices
        mapping = {}
        for filename, wall_id in mapping_data.items():
            if isinstance(wall_id, int):
                mapping[filename] = wall_id
            else:
                # Find wall index by ID
                for i, wall in enumerate(wall_surfaces):
                    if wall.id == wall_id:
                        mapping[filename] = i
                        break
    else:
        # Interactive selection
        mapping = interactive_wall_selection(wall_surfaces, [str(f) for f in pc_files])
    
    # Process each point cloud
    alignment_report = {
        'gml_file': gml_file,
        'pointcloud_dir': pc_dir_path,
        'scale_mode': args.scale_mode,
        'icp_threshold': args.icp_threshold,
        'alignments': []
    }
    
    for pc_file in pc_files:
        filename = pc_file.name
        
        if filename not in mapping:
            print(f"\nSkipping {filename} (not in mapping)")
            continue
        
        wall_idx = mapping[filename]
        wall_surface = wall_surfaces[wall_idx]
        
        print(f"\n{'=' * 80}")
        print(f"Processing: {filename} → WallSurface [{wall_idx}] {wall_surface.id}")
        print('=' * 80)
        
        # Load point cloud
        source_pcd = load_las_pointcloud(str(pc_file))
        
        # Align to wall
        aligned_pcd, transform, fitness = align_pointcloud_to_wall(
            source_pcd, wall_surface,
            scale_mode=args.scale_mode,
            icp_threshold=args.icp_threshold,
            visualize=args.visualize
        )
        
        # Save aligned point cloud
        output_path = os.path.join(args.output_dir, filename)
        save_aligned_pointcloud(aligned_pcd, output_path, str(pc_file))
        
        # Record in report
        alignment_report['alignments'].append({
            'pointcloud': filename,
            'wall_surface': wall_surface.id,
            'wall_index': wall_idx,
            'fitness': fitness,
            'transformation': transform.tolist()
        })
    
    # Save alignment report
    report_path = os.path.join(args.output_dir, 'alignment_report.json')
    with open(report_path, 'w') as f:
        json.dump(alignment_report, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"Alignment complete!")
    print(f"  Processed: {len(alignment_report['alignments'])} point clouds")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Report: {report_path}")
    print('=' * 80)


if __name__ == '__main__':
    main()
