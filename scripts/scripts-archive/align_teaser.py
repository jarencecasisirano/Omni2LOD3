#!/usr/bin/env python3
"""
Align manually-oriented point clouds to GML building WallSurfaces using TEASER++.

Each point cloud is individually registered (with optional scale estimation)
using FPFH feature matching + TEASER++ robust global registration + ICP refinement.

Pipeline:
  1. Build a building envelope from GML WallSurface polygons
     - Curved walls tessellated into finer triangles
     - Optionally exclude curved walls via --exclude_curved_walls
  2. Remove outlier points from source clouds (statistical + radius)
  3. For each cloud individually:
     a. Voxel downsample + compute FPFH features
     b. Find mutual nearest-neighbour correspondences
     c. TEASER++ robust registration (rotation + translation + optional scale)
     d. Multi-pass ICP refinement with fitness guard
  4. Save aligned results

Usage:
  conda activate lidar-test
  python scripts/LOD2toLOD3/align_teaser.py
  python scripts/LOD2toLOD3/align_teaser.py --visualize
  python scripts/LOD2toLOD3/align_teaser.py --voxel_size 0.3 --no_scaling
"""

import os
import sys
import argparse
import numpy as np
import laspy
import open3d as o3d
import teaserpp_python
from lxml import etree
from pathlib import Path
from typing import List, Tuple, Optional
from scipy.spatial import cKDTree

# XML Namespaces for CityGML parsing
NAMESPACES = {
    'gml': 'http://www.opengis.net/gml',
    'bldg': 'http://www.opengis.net/citygml/building/2.0',
    'core': 'http://www.opengis.net/citygml/2.0'
}

# Default paths
DEFAULT_POINTCLOUD_DIR = 'outputs/05_manual_orient_point_cloud'
DEFAULT_OUTPUT_DIR = 'outputs/06_aligned_to_gml'

# Planarity tolerance
CURVED_PLANARITY_TOL = 0.05


# ---------------------------------------------------------------------------
# GML / envelope (reused from align_oriented_pointclouds.py)
# ---------------------------------------------------------------------------

def is_polygon_curved(coords: np.ndarray, tol: float = CURVED_PLANARITY_TOL) -> bool:
    """Return True if polygon vertices are not coplanar within `tol`."""
    unique = np.unique(coords, axis=0)
    if len(unique) < 4:
        return False
    if len(unique) > 4:
        return True
    v0 = unique[1] - unique[0]
    v1 = unique[2] - unique[0]
    normal = np.cross(v0, v1)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-10:
        return False
    normal /= norm_len
    distances = np.abs((unique - unique[0]) @ normal)
    return float(distances.max()) > tol


def tessellate_polygon(coords: np.ndarray, max_edge: float = 0.5) -> List[np.ndarray]:
    """Subdivide a polygon into smaller triangles."""
    triangles = []
    for i in range(1, len(coords) - 1):
        tri = coords[[0, i, i + 1]]
        triangles.append(tri)
    result = []
    for tri in triangles:
        result.extend(_subdivide_triangle(tri, max_edge))
    return result


def _subdivide_triangle(tri: np.ndarray, max_edge: float) -> List[np.ndarray]:
    """Recursively midpoint-subdivide a triangle until all edges <= max_edge."""
    a, b, c = tri
    edges = [np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)]
    if max(edges) <= max_edge:
        return [tri]
    idx = int(np.argmax(edges))
    if idx == 0:
        mid = (a + b) / 2.0
        return (_subdivide_triangle(np.array([a, mid, c]), max_edge) +
                _subdivide_triangle(np.array([mid, b, c]), max_edge))
    elif idx == 1:
        mid = (b + c) / 2.0
        return (_subdivide_triangle(np.array([a, b, mid]), max_edge) +
                _subdivide_triangle(np.array([a, mid, c]), max_edge))
    else:
        mid = (a + c) / 2.0
        return (_subdivide_triangle(np.array([a, b, mid]), max_edge) +
                _subdivide_triangle(np.array([mid, b, c]), max_edge))


def parse_wall_polygons(gml_file: str) -> Tuple[List[np.ndarray], List[bool]]:
    """Parse CityGML and extract all WallSurface polygon coordinate arrays."""
    print(f"Parsing GML: {gml_file}")
    tree = etree.parse(gml_file)
    root = tree.getroot()

    wall_elements = root.xpath('//bldg:WallSurface', namespaces=NAMESPACES)
    print(f"  Found {len(wall_elements)} WallSurface elements")

    polygons: List[np.ndarray] = []
    curved_flags: List[bool] = []

    for wall_elem in wall_elements:
        poly_elems = wall_elem.xpath('.//gml:Polygon', namespaces=NAMESPACES)
        for poly in poly_elems:
            pos_lists = poly.xpath('.//gml:posList', namespaces=NAMESPACES)
            for pos_list in pos_lists:
                text = pos_list.text.strip()
                if not text:
                    continue
                coords = list(map(float, text.split()))
                arr = np.array(coords).reshape(-1, 3)
                polygons.append(arr)
                curved_flags.append(is_polygon_curved(arr))

    n_curved = sum(curved_flags)
    print(f"  Extracted {len(polygons)} wall polygons "
          f"({n_curved} flagged as curved/complex)")
    return polygons, curved_flags


def build_envelope_pointcloud(
        polygons: List[np.ndarray],
        curved_flags: List[bool],
        density: float = 0.05,
        exclude_curved: bool = False,
        curved_tessellation_max_edge: float = 0.3,
) -> o3d.geometry.PointCloud:
    """Sample all wall polygons into a single dense building envelope point cloud."""
    print("Building envelope point cloud...")
    if exclude_curved:
        print("  NOTE: curved/complex wall polygons will be excluded from envelope")

    all_points = []
    for coords, curved in zip(polygons, curved_flags):
        if len(coords) < 3:
            continue
        if curved and exclude_curved:
            continue

        if curved:
            tris = tessellate_polygon(coords, max_edge=curved_tessellation_max_edge)
        else:
            tris = [coords[[0, i, i + 1]] for i in range(1, len(coords) - 1)]
            tris = [np.array(t) if not isinstance(t, np.ndarray) else t for t in tris]

        for tri in tris:
            if len(tri) < 3:
                continue
            triangles_idx = [[0, 1, 2]]
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(tri)
            mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles_idx))
            area = mesh.get_surface_area()
            n_pts = max(10, int(area / (density ** 2)))
            sampled = mesh.sample_points_uniformly(number_of_points=n_pts)
            all_points.append(np.asarray(sampled.points))

    if not all_points:
        raise RuntimeError("No envelope points generated. "
                           "Check GML file and --exclude_curved_walls flag.")

    combined = np.vstack(all_points)
    envelope = o3d.geometry.PointCloud()
    envelope.points = o3d.utility.Vector3dVector(combined)

    mn, mx = combined.min(0), combined.max(0)
    print(f"  {len(combined):,} points, "
          f"dims=({mx[0]-mn[0]:.1f}, {mx[1]-mn[1]:.1f}, {mx[2]-mn[2]:.1f})")
    return envelope


# ---------------------------------------------------------------------------
# Point cloud I/O + cleaning
# ---------------------------------------------------------------------------

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


def remove_outliers(
        pcd: o3d.geometry.PointCloud,
        stat_nb_neighbors: int = 20,
        stat_std_ratio: float = 2.0,
        radius_nb_points: int = 6,
        radius: float = 0.3,
) -> o3d.geometry.PointCloud:
    """Two-pass outlier removal: statistical + radius."""
    n_before = len(pcd.points)
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=stat_nb_neighbors, std_ratio=stat_std_ratio)
    pcd, _ = pcd.remove_radius_outlier(
        nb_points=radius_nb_points, radius=radius)
    n_after = len(pcd.points)
    removed = n_before - n_after
    print(f"    Outlier removal: {n_before:,} → {n_after:,} pts "
          f"(removed {removed:,} = {100*removed/max(n_before,1):.1f}%)")
    return pcd


# ---------------------------------------------------------------------------
# FPFH + TEASER++ helpers (based on official TEASER++ example)
# ---------------------------------------------------------------------------

def extract_fpfh(pcd: o3d.geometry.PointCloud, voxel_size: float):
    """
    Downsample, estimate normals, and compute FPFH features.
    Returns (downsampled_pcd, fpfh_features_as_NxD_array).
    """
    down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return down, np.array(fpfh.data).T  # NxD


def find_correspondences(feats0: np.ndarray, feats1: np.ndarray,
                          mutual_filter: bool = True):
    """
    Find correspondences between two sets of FPFH features
    using mutual nearest-neighbour filtering.
    Returns (idx0, idx1) arrays.
    """
    tree1 = cKDTree(feats1)
    _, nns01 = tree1.query(feats0, k=1, workers=-1)
    corres01_idx0 = np.arange(len(nns01))
    corres01_idx1 = nns01

    if not mutual_filter:
        return corres01_idx0, corres01_idx1

    tree0 = cKDTree(feats0)
    _, nns10 = tree0.query(feats1, k=1, workers=-1)
    corres10_idx0 = nns10

    mutual_mask = (corres10_idx0[corres01_idx1] == corres01_idx0)
    return corres01_idx0[mutual_mask], corres01_idx1[mutual_mask]


def get_teaser_solver(noise_bound: float, estimate_scaling: bool = True):
    """Create a TEASER++ solver with recommended parameters."""
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = estimate_scaling
    solver_params.inlier_selection_mode = \
        teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    solver_params.rotation_tim_graph = \
        teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    solver_params.rotation_estimation_algorithm = \
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 10000
    solver_params.rotation_cost_threshold = 1e-16
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    return solver


def teaser_registration(
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        voxel_size: float = 0.5,
        estimate_scaling: bool = True,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Perform FPFH + TEASER++ global registration.

    Returns:
        (4x4_transform, scale) or None if registration failed.
    """
    print("    Extracting FPFH features...")
    src_down, src_feats = extract_fpfh(source, voxel_size)
    tgt_down, tgt_feats = extract_fpfh(target, voxel_size)

    print(f"    Source downsampled: {len(src_down.points):,} pts, "
          f"Target downsampled: {len(tgt_down.points):,} pts")

    # Find mutual correspondences
    print("    Finding mutual correspondences...")
    corres_idx0, corres_idx1 = find_correspondences(src_feats, tgt_feats,
                                                      mutual_filter=True)
    n_corres = len(corres_idx0)
    print(f"    Found {n_corres} mutual correspondences")

    if n_corres < 10:
        print("    WARNING: Too few correspondences for TEASER++, skipping")
        return None

    # Extract corresponding 3D points (TEASER++ expects 3xN arrays)
    src_pts = np.asarray(src_down.points)
    tgt_pts = np.asarray(tgt_down.points)
    src_corr = src_pts[corres_idx0].T  # 3xN
    tgt_corr = tgt_pts[corres_idx1].T  # 3xN

    # Run TEASER++
    noise_bound = voxel_size * 0.05
    print(f"    Running TEASER++ (noise_bound={noise_bound:.4f}, "
          f"estimate_scaling={estimate_scaling})...")
    solver = get_teaser_solver(noise_bound, estimate_scaling)
    solver.solve(src_corr, tgt_corr)
    solution = solver.getSolution()

    R = solution.rotation
    t = solution.translation
    s = solution.scale

    # Sanity check: rotation should be a proper rotation (det ≈ 1) and scale > 0
    det_R = np.linalg.det(R)
    if abs(det_R - 1.0) > 0.1 or s <= 0:
        print(f"    WARNING: TEASER++ solution looks invalid "
              f"(det(R)={det_R:.4f}, scale={s:.4f})")
        return None

    print(f"    TEASER++ solution:")
    print(f"      Scale:       {s:.6f}")
    print(f"      Translation: ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")
    print(f"      Rotation matrix:")
    for row in R:
        print(f"        [{row[0]:+.6f}, {row[1]:+.6f}, {row[2]:+.6f}]")

    # Build 4x4 transform: T = [s*R, t; 0, 1]
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t

    return T, s


# ---------------------------------------------------------------------------
# Main alignment pipeline
# ---------------------------------------------------------------------------

def align_all_to_envelope(
        source_pcds: dict,
        envelope_pcd: o3d.geometry.PointCloud,
        voxel_size: float = 0.5,
        estimate_scaling: bool = True,
        icp_thresholds: List[float] = None,
        stat_nb_neighbors: int = 20,
        stat_std_ratio: float = 2.0,
        radius_nb_points: int = 6,
        radius: float = 0.3,
) -> Tuple[dict, dict]:
    """
    Align all source point clouds to the building envelope using TEASER++ + ICP.

    Returns:
        (aligned_pcds_dict, info_dict)
    """
    if icp_thresholds is None:
        icp_thresholds = [5.0, 2.0, 1.0, 0.5]

    # -------------------------------------------------------------------
    # Step 0: Outlier removal
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 0: Outlier removal")
    print("=" * 60)

    cleaned_pcds = {}
    for name, pcd in source_pcds.items():
        print(f"\n  {name}:")
        cleaned = remove_outliers(
            pcd,
            stat_nb_neighbors=stat_nb_neighbors,
            stat_std_ratio=stat_std_ratio,
            radius_nb_points=radius_nb_points,
            radius=radius,
        )
        cleaned_pcds[name] = cleaned

    # -------------------------------------------------------------------
    # Prepare target (envelope) — shift near origin for precision
    # -------------------------------------------------------------------
    tgt_pts = np.asarray(envelope_pcd.points)
    tgt_center = tgt_pts.mean(axis=0)
    tgt_dims = tgt_pts.max(0) - tgt_pts.min(0)

    print(f"\n  Target envelope: center=({tgt_center[0]:.1f}, {tgt_center[1]:.1f}, "
          f"{tgt_center[2]:.1f}), dims=({tgt_dims[0]:.1f}, {tgt_dims[1]:.1f}, "
          f"{tgt_dims[2]:.1f})")

    # Work in shifted space to avoid float32 precision issues with large UTM coords
    env_offset = tgt_center.copy()
    env_shifted = o3d.geometry.PointCloud()
    env_shifted.points = o3d.utility.Vector3dVector(tgt_pts - env_offset)
    env_shifted.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))

    # -------------------------------------------------------------------
    # Step 1: Per-cloud TEASER++ registration + ICP
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: TEASER++ registration + ICP refinement")
    print("=" * 60)

    aligned_pcds = {}
    infos = {}

    for name, pcd in cleaned_pcds.items():
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")

        pts = np.asarray(pcd.points).copy()
        cloud_colors = np.asarray(pcd.colors).copy() if pcd.has_colors() else None

        src_center = pts.mean(axis=0)
        src_dims = pts.max(0) - pts.min(0)
        print(f"  Source: center=({src_center[0]:.1f}, {src_center[1]:.1f}, "
              f"{src_center[2]:.1f}), dims=({src_dims[0]:.1f}, {src_dims[1]:.1f}, "
              f"{src_dims[2]:.1f})")

        # Shift source near origin (independent of target shift)
        src_shifted = o3d.geometry.PointCloud()
        src_shifted.points = o3d.utility.Vector3dVector(pts - src_center)

        # Also shift target relative to its own center for TEASER++
        tgt_for_teaser = o3d.geometry.PointCloud()
        tgt_for_teaser.points = o3d.utility.Vector3dVector(tgt_pts - tgt_center)

        # --- TEASER++ global registration ---
        print("\n  Running TEASER++ global registration...")
        teaser_result = teaser_registration(
            src_shifted, tgt_for_teaser,
            voxel_size=voxel_size,
            estimate_scaling=estimate_scaling,
        )

        scale = 1.0
        if teaser_result is not None:
            T_teaser, scale = teaser_result
            # Apply TEASER++ transform: maps (pts - src_center) -> near (tgt_pts - tgt_center)
            # So final = T_teaser @ (pts - src_center) + tgt_center
            pts_shifted = pts - src_center
            pts_homogeneous = np.hstack([pts_shifted, np.ones((len(pts_shifted), 1))])
            transformed = (T_teaser @ pts_homogeneous.T).T[:, :3]
            # Now shift to global frame (near env_offset for ICP)
            # transformed is near tgt_center, we need it near 0 for env_shifted
            shifted_for_icp = transformed  # already centered at tgt_center ≈ env_offset
        else:
            print("  TEASER++ failed — falling back to centroid alignment")
            # Rough centroid alignment: just translate source center to target center
            pts_centered = pts - src_center
            shifted_for_icp = pts_centered  # centered at origin, like env_shifted

        # Build working PCD for ICP (in shifted coord system)
        working = o3d.geometry.PointCloud()
        working.points = o3d.utility.Vector3dVector(shifted_for_icp)
        working.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))

        # --- Guarded multi-pass ICP ---
        print("\n  Running multi-pass ICP refinement...")
        best_fitness = 0.0
        best_rmse = float('inf')
        best_pts_snapshot = np.asarray(working.points).copy()

        for threshold in icp_thresholds:
            reg = o3d.pipelines.registration.registration_icp(
                working, env_shifted, threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
            )

            improved = (reg.fitness > best_fitness + 0.005 or
                        (reg.fitness >= best_fitness and reg.inlier_rmse < best_rmse * 0.95))
            acceptable = (reg.fitness > 0.01 and reg.inlier_rmse < threshold * 0.5)

            if improved and acceptable:
                working.transform(reg.transformation)
                best_fitness = reg.fitness
                best_rmse = reg.inlier_rmse
                best_pts_snapshot = np.asarray(working.points).copy()
                print(f"    ICP (threshold={threshold:.1f}): "
                      f"fitness={reg.fitness:.4f}, RMSE={reg.inlier_rmse:.4f} ✓")
            else:
                reason = ("low fitness" if reg.fitness <= 0.01 else
                          "high RMSE" if reg.inlier_rmse >= threshold * 0.5 else
                          "no improvement")
                print(f"    ICP (threshold={threshold:.1f}): "
                      f"fitness={reg.fitness:.4f}, RMSE={reg.inlier_rmse:.4f} — skipped ({reason})")

        # Shift back to UTM space
        final_pts = best_pts_snapshot + env_offset
        c = final_pts.mean(0)
        print(f"\n  Final center: ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}), "
              f"fitness={best_fitness:.4f}, RMSE={best_rmse:.4f}")

        aligned = o3d.geometry.PointCloud()
        aligned.points = o3d.utility.Vector3dVector(final_pts)
        if cloud_colors is not None and len(cloud_colors) == len(final_pts):
            aligned.colors = o3d.utility.Vector3dVector(cloud_colors)

        aligned_pcds[name] = aligned
        infos[name] = {
            'scale': scale,
            'fitness': best_fitness,
            'rmse': best_rmse,
        }

    return aligned_pcds, infos


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_aligned_pointcloud(pcd: o3d.geometry.PointCloud,
                             output_path: str,
                             original_las_path: str):
    """Save aligned point cloud to LAS, preserving original metadata."""
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
        new_las.red   = (colors[:, 0] * 65535).astype(np.uint16)
        new_las.green = (colors[:, 1] * 65535).astype(np.uint16)
        new_las.blue  = (colors[:, 2] * 65535).astype(np.uint16)
    elif hasattr(original_las, 'red') and len(original_las.points) == len(points):
        new_las.red   = original_las.red
        new_las.green = original_las.green
        new_las.blue  = original_las.blue

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_las.write(output_path)
    print(f"  ✓ {output_path} ({len(points):,} points)")


# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------

def visualize_results(envelope_pcd: o3d.geometry.PointCloud,
                      aligned_pcds: dict):
    """Visualize building envelope + aligned point clouds."""
    geometries = []

    env_vis = o3d.geometry.PointCloud(envelope_pcd)
    env_vis.paint_uniform_color([0.5, 0.5, 0.5])
    geometries.append(env_vis)

    colors = [[1, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.4, 1], [1, 0.8, 0]]
    names  = ['Red', 'Green', 'Blue', 'Yellow']

    print("\nVisualization legend:")
    print("  Gray   = Building envelope (GML walls)")
    for i, (name, pcd) in enumerate(aligned_pcds.items()):
        vis = o3d.geometry.PointCloud(pcd)
        vis.paint_uniform_color(colors[i % 4])
        geometries.append(vis)
        print(f"  {names[i%4]:6s} = {name}")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="TEASER++ Aligned Point Clouds + Building Envelope",
        width=1400, height=900)


# ---------------------------------------------------------------------------
# Interactive selection helpers
# ---------------------------------------------------------------------------

def list_subfolders(base_dir: str) -> List[str]:
    """Return immediate child directory names under base_dir."""
    p = Path(base_dir)
    return sorted([d.name for d in p.iterdir() if d.is_dir()])


def list_gml_files(gml_dir: str) -> List[Path]:
    """Return all .gml files found directly under gml_dir."""
    return sorted(Path(gml_dir).glob('*.gml'))


def prompt_choice(prompt_label: str, options: List[str]) -> str:
    """Print a numbered menu of options and return the chosen item."""
    print(f"\n{prompt_label}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    while True:
        raw = input(f"Enter number (1-{len(options)}): ").strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        print(f"  Please enter a number between 1 and {len(options)}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Align point clouds to GML WallSurfaces using TEASER++",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/LOD2toLOD3/align_teaser.py
  python scripts/LOD2toLOD3/align_teaser.py --visualize
  python scripts/LOD2toLOD3/align_teaser.py --voxel_size 0.3
  python scripts/LOD2toLOD3/align_teaser.py --no_scaling
"""
    )

    # --- Directories ---
    parser.add_argument('--gml_dir', type=str, default='data/lod_2',
                        help="Directory to search for .gml files (default: data/lod_2)")
    parser.add_argument('--pointcloud_dir', type=str, default=DEFAULT_POINTCLOUD_DIR,
                        help=f"Base dir containing subfolders of .las files "
                             f"(default: {DEFAULT_POINTCLOUD_DIR})")
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Base output dir; results go to <output_dir>/<subfolder>/ "
                             f"(default: {DEFAULT_OUTPUT_DIR})")

    # --- TEASER++ ---
    parser.add_argument('--voxel_size', type=float, default=0.5,
                        help="Voxel size for FPFH feature extraction (default: 0.5 m)")
    parser.add_argument('--no_scaling', action='store_true',
                        help="Disable TEASER++ scale estimation (assume scale=1)")

    # --- ICP ---
    parser.add_argument('--icp_thresholds', type=float, nargs='+',
                        default=[5.0, 2.0, 1.0, 0.5],
                        help="ICP correspondence distance thresholds (coarse to fine)")

    # --- Envelope ---
    parser.add_argument('--density', type=float, default=0.05,
                        help="Envelope sampling density in metres (default: 0.05)")
    parser.add_argument('--exclude_curved_walls', action='store_true',
                        help="Exclude curved/complex GML polygons from target envelope")

    # --- Outlier removal ---
    parser.add_argument('--stat_nb_neighbors', type=int, default=20,
                        help="Statistical outlier: neighbour count (default: 20)")
    parser.add_argument('--stat_std_ratio', type=float, default=2.0,
                        help="Statistical outlier: std ratio threshold (default: 2.0)")
    parser.add_argument('--radius_nb_points', type=int, default=6,
                        help="Radius outlier: min neighbours within radius (default: 6)")
    parser.add_argument('--radius', type=float, default=0.3,
                        help="Radius outlier: search radius in metres (default: 0.3)")

    # --- Misc ---
    parser.add_argument('--visualize', action='store_true',
                        help="Open Open3D viewer after alignment")

    args = parser.parse_args()

    print("=" * 80)
    print("ALIGN POINT CLOUDS TO GML WALLSURFACES — TEASER++ EDITION")
    print("=" * 80)

    # -------------------------------------------------------------------
    # Interactive: select GML file
    # -------------------------------------------------------------------
    gml_dir = args.gml_dir
    if not os.path.isdir(gml_dir):
        print(f"ERROR: GML directory not found: {gml_dir}")
        sys.exit(1)

    gml_files = list_gml_files(gml_dir)
    if not gml_files:
        print(f"ERROR: No .gml files found in {gml_dir}")
        sys.exit(1)

    if len(gml_files) == 1:
        chosen_gml = gml_files[0].name
        print(f"\nAuto-selected GML: {chosen_gml}")
    else:
        chosen_gml = prompt_choice(
            f"Select a GML file from '{gml_dir}':",
            [f.name for f in gml_files]
        )
    gml_file = str(Path(gml_dir) / chosen_gml)

    # -------------------------------------------------------------------
    # Interactive: select point cloud subfolder
    # -------------------------------------------------------------------
    if not os.path.isdir(args.pointcloud_dir):
        print(f"ERROR: pointcloud_dir not found: {args.pointcloud_dir}")
        sys.exit(1)

    # Check if .las files exist directly in pointcloud_dir (no subfolders needed)
    direct_las = sorted(Path(args.pointcloud_dir).glob('*.las'))
    available = list_subfolders(args.pointcloud_dir)

    if direct_las and not available:
        # LAS files are directly in the pointcloud_dir, no subfolder structure
        input_dir = args.pointcloud_dir
        output_dir = args.output_dir
        las_files = direct_las
        chosen_subfolder = ""
    elif available:
        subfolder_labels = [
            f"{s}  ({len(list(Path(args.pointcloud_dir, s).glob('*.las')))} .las files)"
            for s in available
        ]
        if len(available) == 1:
            chosen_subfolder = available[0]
            print(f"\nAuto-selected subfolder: {chosen_subfolder}")
        else:
            chosen_label = prompt_choice(
                f"Select a point cloud subfolder from '{args.pointcloud_dir}':",
                subfolder_labels
            )
            chosen_subfolder = available[subfolder_labels.index(chosen_label)]

        input_dir = os.path.join(args.pointcloud_dir, chosen_subfolder)
        output_dir = os.path.join(args.output_dir, chosen_subfolder)
        las_files = sorted(Path(input_dir).glob('*.las'))
    else:
        print(f"ERROR: No .las files or subfolders found in {args.pointcloud_dir}")
        sys.exit(1)

    if not las_files:
        print(f"ERROR: No .las files found in {input_dir}")
        sys.exit(1)

    # -------------------------------------------------------------------
    # Banner
    # -------------------------------------------------------------------
    print()
    print(f"  GML:             {gml_file}")
    print(f"  Input:           {input_dir} ({len(las_files)} files)")
    print(f"  Output:          {output_dir}")
    print(f"  TEASER++:        voxel_size={args.voxel_size}, "
          f"scaling={'disabled' if args.no_scaling else 'enabled'}")
    print(f"  ICP thresholds:  {args.icp_thresholds}")
    print(f"  Curved walls:    "
          f"{'excluded' if args.exclude_curved_walls else 'tessellated + included'}")
    print(f"  Outlier removal: stat(nb={args.stat_nb_neighbors}, "
          f"ratio={args.stat_std_ratio}) + radius(nb={args.radius_nb_points}, "
          f"r={args.radius})")
    print()

    # -------------------------------------------------------------------
    # GML envelope
    # -------------------------------------------------------------------
    polygons, curved_flags = parse_wall_polygons(gml_file)
    if not polygons:
        print("ERROR: No wall polygons found")
        sys.exit(1)
    envelope = build_envelope_pointcloud(
        polygons, curved_flags,
        density=args.density,
        exclude_curved=args.exclude_curved_walls)

    # -------------------------------------------------------------------
    # Load source clouds
    # -------------------------------------------------------------------
    source_pcds = {}
    for p in las_files:
        print(f"\nLoading: {p.name}")
        pcd = load_las_pointcloud(str(p))
        source_pcds[p.name] = pcd
        pts = np.asarray(pcd.points)
        dims = pts.max(0) - pts.min(0)
        print(f"  {len(pts):,} pts, "
              f"center=({pts.mean(0)[0]:.1f}, {pts.mean(0)[1]:.1f}, {pts.mean(0)[2]:.1f}), "
              f"dims=({dims[0]:.1f}, {dims[1]:.1f}, {dims[2]:.1f})")

    # -------------------------------------------------------------------
    # Align
    # -------------------------------------------------------------------
    aligned_pcds, infos = align_all_to_envelope(
        source_pcds, envelope,
        voxel_size=args.voxel_size,
        estimate_scaling=not args.no_scaling,
        icp_thresholds=args.icp_thresholds,
        stat_nb_neighbors=args.stat_nb_neighbors,
        stat_std_ratio=args.stat_std_ratio,
        radius_nb_points=args.radius_nb_points,
        radius=args.radius,
    )

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SAVING")
    print(f"{'='*60}")
    os.makedirs(output_dir, exist_ok=True)
    for p in las_files:
        out = os.path.join(output_dir, p.name)
        save_aligned_pointcloud(aligned_pcds[p.name], out, str(p))

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("DONE — TEASER++ ALIGNMENT COMPLETE")
    print(f"{'='*80}")
    for name, info in infos.items():
        pts = np.asarray(aligned_pcds[name].points)
        print(f"  {name}: {len(pts):,} pts, "
              f"scale={info['scale']:.4f}, "
              f"fitness={info['fitness']:.4f}, "
              f"rmse={info['rmse']:.4f}")

    if args.visualize:
        visualize_results(envelope, aligned_pcds)


if __name__ == '__main__':
    main()
