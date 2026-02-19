#!/usr/bin/env python3
"""
Align manually-oriented point clouds to GML building WallSurfaces.

Each point cloud is scaled individually (since they may differ in size),
but they share a common axis mapping and rotation.

Pipeline:
  1. Build a building envelope from GML WallSurface polygons
  2. Merge all source clouds to determine global axis mapping + rotation
  3. For each cloud individually:
     a. Apply axis mapping + rotation
     b. Compute per-cloud scale (based on height = Z-extent)
     c. Translate to nearest envelope section
     d. ICP refinement against the full envelope
  4. Save aligned results

Usage:
  conda activate lidar-test
  python scripts/LOD2toLOD3/align_oriented_pointclouds.py
  python scripts/LOD2toLOD3/align_oriented_pointclouds.py --visualize
"""

import os
import sys
import argparse
import numpy as np
import laspy
import open3d as o3d
from lxml import etree
from pathlib import Path
from typing import List, Tuple

# XML Namespaces for CityGML parsing
NAMESPACES = {
    'gml': 'http://www.opengis.net/gml',
    'bldg': 'http://www.opengis.net/citygml/building/2.0',
    'core': 'http://www.opengis.net/citygml/2.0'
}

# Default paths
DEFAULT_GML_FILE = 'data/lod_2/nimbb_021126_FIXED.gml'
DEFAULT_POINTCLOUD_DIR = 'outputs/05_manual_orient_point_cloud'
DEFAULT_OUTPUT_DIR = 'outputs/06_aligned_to_gml'


def parse_wall_polygons(gml_file: str) -> List[np.ndarray]:
    """Parse CityGML and extract all WallSurface polygon coordinate arrays."""
    print(f"Parsing GML: {gml_file}")
    tree = etree.parse(gml_file)
    root = tree.getroot()

    wall_elements = root.xpath('//bldg:WallSurface', namespaces=NAMESPACES)
    print(f"  Found {len(wall_elements)} WallSurface elements")

    polygons = []
    for wall_elem in wall_elements:
        poly_elems = wall_elem.xpath('.//gml:Polygon', namespaces=NAMESPACES)
        for poly in poly_elems:
            pos_lists = poly.xpath('.//gml:posList', namespaces=NAMESPACES)
            for pos_list in pos_lists:
                text = pos_list.text.strip()
                if not text:
                    continue
                coords = list(map(float, text.split()))
                polygons.append(np.array(coords).reshape(-1, 3))

    print(f"  Extracted {len(polygons)} wall polygons")
    return polygons


def build_envelope_pointcloud(polygons: List[np.ndarray],
                               density: float = 0.05) -> o3d.geometry.PointCloud:
    """Sample all wall polygons into a single dense building envelope point cloud."""
    print("Building envelope point cloud...")
    all_points = []

    for coords in polygons:
        if len(coords) < 3:
            continue
        triangles = [[0, i, i + 1] for i in range(1, len(coords) - 1)]
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(coords)
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
        area = mesh.get_surface_area()
        n_pts = max(50, int(area / (density ** 2)))
        sampled = mesh.sample_points_uniformly(number_of_points=n_pts)
        all_points.append(np.asarray(sampled.points))

    combined = np.vstack(all_points)
    envelope = o3d.geometry.PointCloud()
    envelope.points = o3d.utility.Vector3dVector(combined)

    mn, mx = combined.min(0), combined.max(0)
    print(f"  {len(combined):,} points, dims=({mx[0]-mn[0]:.1f}, {mx[1]-mn[1]:.1f}, {mx[2]-mn[2]:.1f})")
    return envelope


def load_las_pointcloud(las_file: str) -> o3d.geometry.PointCloud:
    """Load a LAS file into an Open3D PointCloud."""
    las = laspy.read(las_file)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if hasattr(las, 'red'):
        colors = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def try_axis_mappings(src_pts: np.ndarray, tgt_pts: np.ndarray):
    """
    Try all 24 proper axis permutations and pick the one where
    sorted dimension ratios are most uniform.

    Returns the best 3x3 matrix M.
    """
    import itertools

    tgt_dims = tgt_pts.max(0) - tgt_pts.min(0)
    src_dims = src_pts.max(0) - src_pts.min(0)

    best_score = float('inf')
    best_M = np.eye(3)

    for perm in itertools.permutations([0, 1, 2]):
        for signs in itertools.product([-1, 1], repeat=3):
            M = np.zeros((3, 3))
            for i in range(3):
                M[i, perm[i]] = signs[i]
            if np.linalg.det(M) < 0:
                continue  # Only proper rotations

            mapped_dims = np.abs(M @ src_dims)
            ratios = tgt_dims / np.maximum(mapped_dims, 1e-6)
            score = np.std(ratios) / np.mean(ratios)

            if score < best_score:
                best_score = score
                best_M = M

    mapped_dims = np.abs(best_M @ src_dims)
    print(f"  Best axis mapping (ratio CV={best_score:.4f}):")
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        for j in range(3):
            if abs(best_M[i, j]) > 0.5:
                sign = '+' if best_M[i, j] > 0 else '-'
                print(f"    Target {labels[i]} ← {sign}Source {labels[j]}")
    print(f"  Mapped dims: ({mapped_dims[0]:.1f}, {mapped_dims[1]:.1f}, {mapped_dims[2]:.1f})")
    print(f"  Target dims: ({tgt_dims[0]:.1f}, {tgt_dims[1]:.1f}, {tgt_dims[2]:.1f})")

    return best_M


def find_best_z_rotation(src_pts: np.ndarray, tgt_tree, threshold: float = 5.0,
                          num_angles: int = 72) -> float:
    """
    Try rotations around Z axis. Both inputs should be centered at origin.
    Returns the best angle in radians.
    """
    step = max(1, len(src_pts) // 2000)
    sample = src_pts[::step]

    best_count = -1
    best_angle = 0.0

    for i in range(num_angles):
        angle = 2 * np.pi * i / num_angles
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        rotated = (R @ sample.T).T

        count = 0
        for pt in rotated:
            [_, _, dist] = tgt_tree.search_knn_vector_3d(pt, 1)
            if dist[0] < threshold ** 2:
                count += 1

        if count > best_count:
            best_count = count
            best_angle = angle

    print(f"  Best Z rotation: {np.degrees(best_angle):.1f}° ({best_count}/{len(sample)} inliers)")
    return best_angle


def apply_axis_rotation(pts: np.ndarray, M: np.ndarray, angle: float,
                         src_center: np.ndarray) -> np.ndarray:
    """Apply axis mapping + Z rotation, centered at src_center. No scaling or translation."""
    p = pts - src_center
    p = (M @ p.T).T
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    p = (R @ p.T).T
    return p


def align_all_to_envelope(source_pcds: dict,
                           envelope_pcd: o3d.geometry.PointCloud,
                           icp_thresholds: List[float] = None
                           ) -> Tuple[dict, dict]:
    """
    Align all source point clouds to the building envelope.

    Global: axis mapping + Z rotation (from merged cloud)
    Per-cloud: individual scale (Z-height based) + translation + ICP

    Returns:
        (aligned_pcds_dict, info_dict)
    """
    if icp_thresholds is None:
        icp_thresholds = [5.0, 2.0, 1.0, 0.5]

    # --- Merge all sources for global axis mapping + rotation ---
    print("\n" + "=" * 60)
    print("STEP 1: Global axis mapping + rotation")
    print("=" * 60)

    all_pts_list = []
    cloud_pts = {}
    for name, pcd in source_pcds.items():
        pts = np.asarray(pcd.points).copy()
        all_pts_list.append(pts)
        cloud_pts[name] = pts
        print(f"  {name}: {len(pts):,} pts")

    merged_pts = np.vstack(all_pts_list)
    src_center = merged_pts.mean(axis=0)
    src_dims = merged_pts.max(0) - merged_pts.min(0)

    tgt_pts = np.asarray(envelope_pcd.points)
    tgt_center = tgt_pts.mean(axis=0)
    tgt_dims = tgt_pts.max(0) - tgt_pts.min(0)
    tgt_z_extent = tgt_dims[2]  # Building height

    print(f"\n  Source merged: center=({src_center[0]:.2f}, {src_center[1]:.2f}, {src_center[2]:.2f}), "
          f"dims=({src_dims[0]:.1f}, {src_dims[1]:.1f}, {src_dims[2]:.1f})")
    print(f"  Target envelope: center=({tgt_center[0]:.1f}, {tgt_center[1]:.1f}, {tgt_center[2]:.1f}), "
          f"dims=({tgt_dims[0]:.1f}, {tgt_dims[1]:.1f}, {tgt_dims[2]:.1f})")

    # --- Axis mapping ---
    print("\nFinding best axis mapping...")
    M = try_axis_mappings(merged_pts, tgt_pts)

    # --- Rough global scale for rotation search only ---
    mapped_dims = np.abs(M @ src_dims)
    rough_scale = float(np.median(tgt_dims / np.maximum(mapped_dims, 1e-6)))
    print(f"\n  Rough scale for rotation search: {rough_scale:.4f}")

    # --- Z rotation search ---
    print("\nSearching for best Z rotation...")
    # Transform merged points: center → axis_map → rough_scale → (no translate)
    pre_rot = apply_axis_rotation(merged_pts, M, 0.0, src_center) * rough_scale

    # Build target tree centered at origin
    tgt_centered = tgt_pts - tgt_center
    tgt_centered_pcd = o3d.geometry.PointCloud()
    tgt_centered_pcd.points = o3d.utility.Vector3dVector(tgt_centered)
    tgt_tree = o3d.geometry.KDTreeFlann(tgt_centered_pcd)

    best_angle = find_best_z_rotation(pre_rot, tgt_tree, threshold=5.0, num_angles=72)

    # --- Per-cloud alignment ---
    print("\n" + "=" * 60)
    print("STEP 2: Per-cloud scaling + alignment")
    print("=" * 60)

    # Prepare envelope normals for ICP
    envelope_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
    )

    # First pass: apply axis mapping + rotation + per-cloud scaling
    # Scale each cloud around its own center to preserve relative positions
    pre_translated = {}
    for name, pts in cloud_pts.items():
        print(f"\n--- {name} ---")

        # Step A: Apply global axis mapping + rotation (no scale, no translate)
        rotated = apply_axis_rotation(pts, M, best_angle, src_center)
        rotated_dims = rotated.max(0) - rotated.min(0)

        # Step B: Compute per-cloud scale based on Z (height)
        cloud_z_extent = rotated_dims[2]
        scale = tgt_z_extent / max(cloud_z_extent, 1e-6)
        print(f"  Z-extent: cloud={cloud_z_extent:.1f}, target={tgt_z_extent:.1f} → scale={scale:.4f}")

        # Scale around the cloud's own center (preserves relative positions)
        cloud_center = rotated.mean(0)
        scaled = (rotated - cloud_center) * scale + cloud_center
        scaled_dims = scaled.max(0) - scaled.min(0)
        print(f"  Scaled dims: ({scaled_dims[0]:.1f}, {scaled_dims[1]:.1f}, {scaled_dims[2]:.1f})")

        pre_translated[name] = {'pts': scaled, 'scale': scale}

    # Compute group center (mean of all cloud centers) and translate to envelope
    all_scaled = np.vstack([info['pts'] for info in pre_translated.values()])
    group_center = all_scaled.mean(0)
    global_translation = tgt_center - group_center
    print(f"\n  Group center: ({group_center[0]:.1f}, {group_center[1]:.1f}, {group_center[2]:.1f})")
    print(f"  Target center: ({tgt_center[0]:.1f}, {tgt_center[1]:.1f}, {tgt_center[2]:.1f})")
    print(f"  Translation: ({global_translation[0]:.1f}, {global_translation[1]:.1f}, {global_translation[2]:.1f})")

    # Second pass: translate + ICP (origin-shifted for precision)
    # Open3D ICP loses precision with large UTM coordinates (~292000),
    # so we shift everything to near-origin for ICP, then shift back.
    env_offset = tgt_center.copy()
    env_shifted = o3d.geometry.PointCloud()
    env_shifted.points = o3d.utility.Vector3dVector(tgt_pts - env_offset)
    env_shifted.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
    )

    aligned_pcds = {}
    infos = {}

    for name in cloud_pts:
        print(f"\n--- {name} (ICP) ---")
        has_colors = source_pcds[name].has_colors()
        colors = np.asarray(source_pcds[name].colors) if has_colors else None
        scale = pre_translated[name]['scale']

        # Apply group translation
        translated = pre_translated[name]['pts'] + global_translation
        c_pre = translated.mean(0)
        print(f"  Pre-ICP center: ({c_pre[0]:.1f}, {c_pre[1]:.1f}, {c_pre[2]:.1f})")

        # ICP in origin-shifted space
        shifted_pts = translated - env_offset
        working = o3d.geometry.PointCloud()
        working.points = o3d.utility.Vector3dVector(shifted_pts)
        working.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
        )

        best_fitness = 0.0
        for threshold in icp_thresholds:
            reg = o3d.pipelines.registration.registration_icp(
                working, env_shifted, threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
            )
            if reg.fitness > 0.01:
                working.transform(reg.transformation)
                best_fitness = reg.fitness
                print(f"  ICP (threshold={threshold:.1f}): fitness={reg.fitness:.4f}, RMSE={reg.inlier_rmse:.4f} ✓")
            else:
                print(f"  ICP (threshold={threshold:.1f}): fitness={reg.fitness:.4f} — skipped")

        # Shift back to UTM space
        final_pts = np.asarray(working.points).copy() + env_offset
        c = final_pts.mean(0)
        print(f"  Final center: ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}), fitness={best_fitness:.4f}")

        aligned = o3d.geometry.PointCloud()
        aligned.points = o3d.utility.Vector3dVector(final_pts)
        if has_colors:
            aligned.colors = o3d.utility.Vector3dVector(colors)
        aligned_pcds[name] = aligned

        infos[name] = {'scale': scale, 'fitness': best_fitness}

    return aligned_pcds, infos


def save_aligned_pointcloud(pcd: o3d.geometry.PointCloud,
                             output_path: str,
                             original_las_path: str):
    """Save aligned point cloud to LAS, preserving original colors."""
    original_las = laspy.read(original_las_path)
    header = laspy.LasHeader(point_format=original_las.header.point_format,
                              version=original_las.header.version)
    points = np.asarray(pcd.points)
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])

    new_las = laspy.LasData(header)
    new_las.x = points[:, 0]
    new_las.y = points[:, 1]
    new_las.z = points[:, 2]

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        new_las.red = (colors[:, 0] * 65535).astype(np.uint16)
        new_las.green = (colors[:, 1] * 65535).astype(np.uint16)
        new_las.blue = (colors[:, 2] * 65535).astype(np.uint16)
    elif hasattr(original_las, 'red') and len(original_las.points) == len(points):
        new_las.red = original_las.red
        new_las.green = original_las.green
        new_las.blue = original_las.blue

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_las.write(output_path)
    print(f"  ✓ {output_path} ({len(points):,} points)")


def visualize_results(envelope_pcd, aligned_pcds):
    """Visualize building envelope + aligned point clouds."""
    geometries = []

    env_vis = o3d.geometry.PointCloud(envelope_pcd)
    env_vis.paint_uniform_color([0.5, 0.5, 0.5])
    geometries.append(env_vis)

    colors = [[1, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.4, 1], [1, 0.8, 0]]
    names = ['Red', 'Green', 'Blue', 'Yellow']

    print("\nVisualization legend:")
    print("  Gray   = Building envelope (GML walls)")
    for i, (name, pcd) in enumerate(aligned_pcds.items()):
        vis = o3d.geometry.PointCloud(pcd)
        vis.paint_uniform_color(colors[i % 4])
        geometries.append(vis)
        print(f"  {names[i%4]:6s} = {name}")

    o3d.visualization.draw_geometries(geometries,
        window_name="Aligned Point Clouds + Building Envelope",
        width=1400, height=900)


def main():
    parser = argparse.ArgumentParser(
        description="Align manually-oriented point clouds to GML WallSurfaces"
    )
    parser.add_argument('--gml_file', type=str, default=DEFAULT_GML_FILE)
    parser.add_argument('--pointcloud_dir', type=str, default=DEFAULT_POINTCLOUD_DIR)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--icp_thresholds', type=float, nargs='+',
                        default=[5.0, 2.0, 1.0, 0.5])
    parser.add_argument('--density', type=float, default=0.05)
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    if not os.path.exists(args.gml_file):
        print(f"ERROR: GML file not found: {args.gml_file}"); sys.exit(1)
    if not os.path.exists(args.pointcloud_dir):
        print(f"ERROR: Dir not found: {args.pointcloud_dir}"); sys.exit(1)

    las_files = sorted(Path(args.pointcloud_dir).glob('*.las'))
    if not las_files:
        print(f"ERROR: No .las files in {args.pointcloud_dir}"); sys.exit(1)

    print("=" * 80)
    print("ALIGN MANUALLY-ORIENTED POINT CLOUDS TO GML WALLSURFACES")
    print("=" * 80)
    print(f"  GML:    {args.gml_file}")
    print(f"  Input:  {args.pointcloud_dir} ({len(las_files)} files)")
    print(f"  Output: {args.output_dir}")
    print(f"  ICP:    {args.icp_thresholds}")
    print()

    # Parse GML
    polygons = parse_wall_polygons(args.gml_file)
    if not polygons:
        print("ERROR: No wall polygons found"); sys.exit(1)
    envelope = build_envelope_pointcloud(polygons, density=args.density)

    # Load point clouds
    source_pcds = {}
    for p in las_files:
        print(f"\nLoading: {p.name}")
        pcd = load_las_pointcloud(str(p))
        source_pcds[p.name] = pcd
        pts = np.asarray(pcd.points)
        dims = pts.max(0) - pts.min(0)
        print(f"  {len(pts):,} pts, center=({pts.mean(0)[0]:.1f}, {pts.mean(0)[1]:.1f}, {pts.mean(0)[2]:.1f}), "
              f"dims=({dims[0]:.1f}, {dims[1]:.1f}, {dims[2]:.1f})")

    # Align
    aligned_pcds, infos = align_all_to_envelope(
        source_pcds, envelope,
        icp_thresholds=args.icp_thresholds
    )

    # Save
    print(f"\n{'='*60}")
    print("SAVING")
    print(f"{'='*60}")
    os.makedirs(args.output_dir, exist_ok=True)
    for p in las_files:
        out = os.path.join(args.output_dir, p.name)
        save_aligned_pointcloud(aligned_pcds[p.name], out, str(p))

    # Summary
    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")
    for name, info in infos.items():
        pts = np.asarray(aligned_pcds[name].points)
        print(f"  {name}: {len(pts):,} pts, scale={info['scale']:.4f}, fitness={info['fitness']:.4f}")

    if args.visualize:
        visualize_results(envelope, aligned_pcds)


if __name__ == '__main__':
    main()
