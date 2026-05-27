#!/usr/bin/env python3
"""
Align point clouds to CityGML LOD2 building model WallSurfaces — PER-FILE ICP TWEAK.

Workflow:
  1. Select a subfolder under outputs/06_aligned_p2p
     (each subfolder contains .las files that form one building's facades,
      already properly scaled and more or less aligned to the GML)
  2. Select a .gml file from outputs/00_gml_wall_merged
  3. Convert every WallSurface in the GML to a dense point cloud and merge
     them all into one combined target cloud
  4. ICP-align each .las file independently against that target
  5. Save refined results to outputs/08_icp_refined/<subfolder_name>/
"""

import os
import json
import argparse
import numpy as np
import laspy
import open3d as o3d
from lxml import etree
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# XML Namespaces for CityGML parsing
# ---------------------------------------------------------------------------
NAMESPACES = {
    'gml':  'http://www.opengis.net/gml',
    'bldg': 'http://www.opengis.net/citygml/building/2.0',
    'core': 'http://www.opengis.net/citygml/2.0',
}

# ---------------------------------------------------------------------------
# Default directories
# ---------------------------------------------------------------------------
DEFAULT_PC_DIR     = 'outputs/06_aligned_p2p'     # subfolders with .las files
DEFAULT_GML_DIR    = 'outputs/00_gml_wall_merged'  # .gml files
DEFAULT_OUTPUT_DIR = 'outputs/08_icp_refined'      # refined .las output

# ---------------------------------------------------------------------------
# ICP parameters (fine tweak — clouds are already close)
# ---------------------------------------------------------------------------
ICP_THRESHOLD    = 0.5    # metres — max correspondence distance
ICP_MAX_ITER     = 200    # iterations
ICP_NORMAL_RADIUS = 0.5   # metres — normal estimation radius


# ===========================================================================
# Interactive selection helpers
# ===========================================================================

def select_subfolder(base_dir: str = DEFAULT_PC_DIR) -> Optional[str]:
    """
    List immediate subdirectories of *base_dir* and let the user choose one.
    Returns the full path to the chosen subfolder, or None on cancellation.
    """
    if not os.path.isdir(base_dir):
        print(f"ERROR: Directory not found: {base_dir}")
        return None

    subfolders = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])

    if not subfolders:
        print(f"ERROR: No subfolders found in {base_dir}")
        return None

    if len(subfolders) == 1:
        chosen = os.path.join(base_dir, subfolders[0])
        print(f"\n✓ Auto-selected (only one subfolder): {subfolders[0]}")
        return chosen

    print("\n" + "=" * 80)
    print("SELECT POINT CLOUD SUBFOLDER")
    print("=" * 80)
    print(f"\nAvailable subfolders in {base_dir}:")
    for i, name in enumerate(subfolders):
        path = os.path.join(base_dir, name)
        las_count = len([f for f in os.listdir(path) if f.lower().endswith('.las')])
        print(f"  [{i}] {name}  ({las_count} .las files)")

    while True:
        try:
            r = input(f"\nSelect subfolder (0-{len(subfolders)-1}, or 'q' to quit): ").strip()
            if r.lower() == 'q':
                return None
            idx = int(r)
            if 0 <= idx < len(subfolders):
                chosen = os.path.join(base_dir, subfolders[idx])
                print(f"✓ Selected: {subfolders[idx]}")
                return chosen
            print(f"  Invalid index. Enter 0-{len(subfolders)-1}.")
        except ValueError:
            print("  Invalid input.")


def select_gml_file(gml_dir: str = DEFAULT_GML_DIR) -> Optional[str]:
    """
    List .gml files in *gml_dir* and let the user pick one.
    Returns the full path, or None on cancellation.
    """
    if not os.path.isdir(gml_dir):
        print(f"ERROR: GML directory not found: {gml_dir}")
        return None

    gml_files = sorted([f for f in os.listdir(gml_dir) if f.lower().endswith('.gml')])

    if not gml_files:
        print(f"ERROR: No .gml files found in {gml_dir}")
        return None

    if len(gml_files) == 1:
        chosen = os.path.join(gml_dir, gml_files[0])
        print(f"\n✓ Auto-selected (only one GML file): {gml_files[0]}")
        return chosen

    print("\n" + "=" * 80)
    print("SELECT GML MODEL")
    print("=" * 80)
    print(f"\nAvailable GML files in {gml_dir}:")
    for i, name in enumerate(gml_files):
        size_mb = os.path.getsize(os.path.join(gml_dir, name)) / (1024 * 1024)
        print(f"  [{i}] {name}  ({size_mb:.2f} MB)")

    while True:
        try:
            r = input(f"\nSelect GML file (0-{len(gml_files)-1}, or 'q' to quit): ").strip()
            if r.lower() == 'q':
                return None
            idx = int(r)
            if 0 <= idx < len(gml_files):
                chosen = os.path.join(gml_dir, gml_files[idx])
                print(f"✓ Selected: {gml_files[idx]}")
                return chosen
            print(f"  Invalid index. Enter 0-{len(gml_files)-1}.")
        except ValueError:
            print("  Invalid input.")


# ===========================================================================
# CityGML wall surface representation
# ===========================================================================

class WallSurface:
    """One polygon face from a CityGML WallSurface element."""

    def __init__(self, surface_id: str, coordinates: np.ndarray):
        self.id          = surface_id
        self.coordinates = coordinates          # Nx3
        self.bbox_min    = np.min(coordinates, axis=0)
        self.bbox_max    = np.max(coordinates, axis=0)
        self._normal     = None

    # ------------------------------------------------------------------
    def to_pointcloud(self, density: float = 0.05) -> o3d.geometry.PointCloud:
        """Densely sample the polygon and return an Open3D PointCloud."""
        if len(self.coordinates) < 3:
            return o3d.geometry.PointCloud()

        n   = len(self.coordinates)
        tri = [[0, i, i + 1] for i in range(1, n - 1)]

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(self.coordinates)
        mesh.triangles = o3d.utility.Vector3iVector(np.array(tri, dtype=np.int32))

        area       = mesh.get_surface_area()
        n_pts      = max(100, int(area / (density ** 2)))
        return mesh.sample_points_uniformly(number_of_points=n_pts)

    def get_center(self) -> np.ndarray:
        return np.mean(self.coordinates, axis=0)

    def get_dimensions(self) -> Tuple[float, float, float]:
        return tuple(self.bbox_max - self.bbox_min)

    def get_normal(self) -> np.ndarray:
        if self._normal is not None:
            return self._normal
        if len(self.coordinates) < 3:
            return np.array([0.0, 0.0, 1.0])
        v1 = self.coordinates[1] - self.coordinates[0]
        v2 = self.coordinates[2] - self.coordinates[0]
        n  = np.cross(v1, v2)
        nm = np.linalg.norm(n)
        self._normal = n / nm if nm > 1e-6 else np.array([0.0, 0.0, 1.0])
        return self._normal


# ===========================================================================
# GML parsing & target-cloud construction
# ===========================================================================

def parse_gml_wallsurfaces(gml_file: str) -> List[WallSurface]:
    """Parse CityGML and extract all WallSurface polygons."""
    print(f"\nParsing GML: {gml_file}")
    root = etree.parse(gml_file).getroot()

    walls = []
    for wall_elem in root.xpath('//bldg:WallSurface', namespaces=NAMESPACES):
        for polygon in wall_elem.xpath('.//gml:Polygon', namespaces=NAMESPACES):
            poly_id   = polygon.get('{http://www.opengis.net/gml}id', 'unknown')
            for pos_list in polygon.xpath('.//gml:posList', namespaces=NAMESPACES):
                text = (pos_list.text or '').strip()
                if not text:
                    continue
                coords = np.array(list(map(float, text.split()))).reshape(-1, 3)
                walls.append(WallSurface(poly_id, coords))

    print(f"  Found {len(walls)} WallSurface polygon(s)")
    return walls


def gml_walls_to_pointcloud(
        walls: List[WallSurface],
        density: float = 0.05,
) -> o3d.geometry.PointCloud:
    """
    Sample every WallSurface polygon into points and concatenate them all
    into one combined target point cloud.
    """
    print(f"\nSampling GML walls into target point cloud (density={density} m) ...")
    combined = o3d.geometry.PointCloud()
    for i, w in enumerate(walls):
        pcd = w.to_pointcloud(density=density)
        combined += pcd
        print(f"  Wall [{i:>3}] {w.id[:40]:<40}  → {len(pcd.points):>7,} pts  "
              f"  (ctr {w.get_center().round(2)})")

    pts_total = len(combined.points)
    print(f"\n  Combined target: {pts_total:,} points from {len(walls)} wall(s)")

    if pts_total == 0:
        print("  WARNING: Target cloud is empty — check GML coordinate parsing.")
    return combined


def visualize_target(
        target_pcd: o3d.geometry.PointCloud,
        walls: List[WallSurface],
) -> None:
    """Open an Open3D window showing the colour-coded GML target cloud."""
    import colorsys
    print("\n  Opening GML target viewer (close window to continue)...")
    geoms = []
    for i, w in enumerate(walls):
        hue = (i * 0.618033988749895) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        pcd = w.to_pointcloud(density=0.05)
        pcd.paint_uniform_color(list(rgb))
        geoms.append(pcd)

        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.4)
        sph.translate(w.get_center())
        sph.paint_uniform_color([1.0, 0.0, 0.0])
        geoms.append(sph)

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"GML WallSurfaces — {len(walls)} walls (close to continue)",
        width=1400, height=800,
    )


# ===========================================================================
# Point cloud I/O
# ===========================================================================

def load_las_as_pcd(las_path: str) -> o3d.geometry.PointCloud:
    """Load a LAS file and return an Open3D PointCloud."""
    las    = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).T
    pcd    = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if hasattr(las, 'red'):
        colors = np.vstack((las.red, las.green, las.blue)).T / 65535.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def save_las_with_transform(
        original_las_path: str,
        transform_4x4: np.ndarray,
        output_path: str,
) -> None:
    """
    Read the original LAS, apply *transform_4x4* to its XYZ coordinates,
    and write the result to *output_path*, preserving all other attributes.
    """
    las    = laspy.read(original_las_path)
    pts    = np.vstack((las.x, las.y, las.z)).T          # Nx3
    # Homogeneous transform
    ones   = np.ones((len(pts), 1))
    pts_h  = np.hstack((pts, ones))                       # Nx4
    pts_t  = (transform_4x4 @ pts_h.T).T[:, :3]          # Nx3

    header          = laspy.LasHeader(point_format=las.header.point_format,
                                      version=las.header.version)
    header.offsets  = np.min(pts_t, axis=0)
    header.scales   = np.array([0.001, 0.001, 0.001])

    out_las   = laspy.LasData(header)
    out_las.x = pts_t[:, 0]
    out_las.y = pts_t[:, 1]
    out_las.z = pts_t[:, 2]

    # Preserve colour if present
    for ch in ('red', 'green', 'blue'):
        if hasattr(las, ch):
            setattr(out_las, ch, getattr(las, ch))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_las.write(output_path)
    print(f"    ✓ Saved {len(pts_t):,} pts → {output_path}")


# ===========================================================================
# ICP alignment (source → pre-built target cloud)
# ===========================================================================

def icp_align(
        source_pcd:    o3d.geometry.PointCloud,
        target_pcd:    o3d.geometry.PointCloud,
        threshold:     float = ICP_THRESHOLD,
        max_iter:      int   = ICP_MAX_ITER,
        normal_radius: float = ICP_NORMAL_RADIUS,
        visualize:     bool  = False,
        label:         str   = '',
) -> Tuple[np.ndarray, float, float]:
    """
    Fine point-to-plane ICP.  Source is assumed already close to target.

    Returns:
        (4x4 transform, fitness, inlier_rmse)
    """
    print(f"\n  ICP: {label}  threshold={threshold} m  max_iter={max_iter}")

    kd = o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    target_copy = o3d.geometry.PointCloud(target_pcd)
    target_copy.estimate_normals(search_param=kd)

    source_copy = o3d.geometry.PointCloud(source_pcd)
    source_copy.estimate_normals(search_param=kd)

    reg = o3d.pipelines.registration.registration_icp(
        source_copy,
        target_copy,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )

    print(f"    fitness={reg.fitness:.4f}  RMSE={reg.inlier_rmse:.4f} m")

    if visualize:
        print("    Opening before/after viewer (close to continue)...")
        before = o3d.geometry.PointCloud(source_pcd)
        before.paint_uniform_color([1.0, 0.4, 0.0])   # orange

        after = o3d.geometry.PointCloud(source_pcd)
        after.transform(reg.transformation)
        after.paint_uniform_color([0.0, 0.9, 0.3])    # green

        tgt_vis = o3d.geometry.PointCloud(target_pcd)
        tgt_vis.paint_uniform_color([0.2, 0.4, 1.0])  # blue

        o3d.visualization.draw_geometries(
            [before, after, tgt_vis],
            window_name=f"ICP {label} — orange=before  green=after  blue=GML",
            width=1400, height=800,
        )

    return reg.transformation, reg.fitness, reg.inlier_rmse


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Per-file fine ICP: align point clouds from 06_aligned_p2p to GML WallSurfaces"
    )
    parser.add_argument('--pc_dir',    default=DEFAULT_PC_DIR,
                        help=f'Base directory containing subfolder(s) of .las files '
                             f'(default: {DEFAULT_PC_DIR})')
    parser.add_argument('--subfolder', default=None,
                        help='Name of subfolder to process (skips interactive selection)')
    parser.add_argument('--gml_dir',   default=DEFAULT_GML_DIR,
                        help=f'Directory with .gml files (default: {DEFAULT_GML_DIR})')
    parser.add_argument('--gml_file',  default=None,
                        help='Path to GML file (skips interactive selection)')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR,
                        help=f'Output base directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--density',   type=float, default=0.05,
                        help='GML wall sampling density in metres (default: 0.05)')
    parser.add_argument('--threshold', type=float, default=ICP_THRESHOLD,
                        help=f'ICP max correspondence distance in metres (default: {ICP_THRESHOLD})')
    parser.add_argument('--max_iter',  type=int,   default=ICP_MAX_ITER,
                        help=f'ICP max iterations (default: {ICP_MAX_ITER})')
    parser.add_argument('--normal_radius', type=float, default=ICP_NORMAL_RADIUS,
                        help=f'Normal estimation radius in metres (default: {ICP_NORMAL_RADIUS})')
    parser.add_argument('--visualize', action='store_true',
                        help='Show Open3D before/after viewer for each file')
    parser.add_argument('--visualize_gml', action='store_true',
                        help='Show GML target cloud in viewer before ICP')

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Choose subfolder
    # ------------------------------------------------------------------
    if args.subfolder:
        subfolder_path = os.path.join(args.pc_dir, args.subfolder)
        if not os.path.isdir(subfolder_path):
            print(f"ERROR: Subfolder not found: {subfolder_path}")
            return
        print(f"\n✓ Using subfolder: {subfolder_path}")
    else:
        subfolder_path = select_subfolder(args.pc_dir)
        if not subfolder_path:
            print("Cancelled.")
            return

    subfolder_name = os.path.basename(subfolder_path)

    las_files = sorted([
        os.path.join(subfolder_path, f)
        for f in os.listdir(subfolder_path)
        if f.lower().endswith('.las')
    ])
    if not las_files:
        print(f"ERROR: No .las files found in {subfolder_path}")
        return

    print(f"\n  Found {len(las_files)} .las file(s) in '{subfolder_name}':")
    for p in las_files:
        size_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"    {os.path.basename(p)}  ({size_mb:.1f} MB)")

    # ------------------------------------------------------------------
    # 2. Choose GML file
    # ------------------------------------------------------------------
    if args.gml_file:
        gml_path = args.gml_file
        if not os.path.isfile(gml_path):
            print(f"ERROR: GML file not found: {gml_path}")
            return
        print(f"\n✓ Using GML: {gml_path}")
    else:
        gml_path = select_gml_file(args.gml_dir)
        if not gml_path:
            print("Cancelled.")
            return

    # ------------------------------------------------------------------
    # 3. Parse GML → build combined target point cloud
    # ------------------------------------------------------------------
    walls = parse_gml_wallsurfaces(gml_path)
    if not walls:
        print("ERROR: No WallSurfaces found in GML file.")
        return

    target_pcd = gml_walls_to_pointcloud(walls, density=args.density)
    if len(target_pcd.points) == 0:
        print("ERROR: GML target cloud is empty — cannot run ICP.")
        return

    if args.visualize_gml:
        visualize_target(target_pcd, walls)

    # ------------------------------------------------------------------
    # 4. Per-file ICP
    # ------------------------------------------------------------------
    output_subdir = os.path.join(args.output_dir, subfolder_name)
    reports       = []

    print(f"\n{'='*80}")
    print(f"Running ICP for {len(las_files)} file(s) → output: {output_subdir}")
    print(f"  threshold={args.threshold} m  max_iter={args.max_iter}  "
          f"normal_radius={args.normal_radius} m")
    print('=' * 80)

    for las_path in las_files:
        fname = os.path.basename(las_path)
        print(f"\n[{fname}]")
        print(f"  Loading ...")
        source_pcd = load_las_as_pcd(las_path)
        n_pts      = len(source_pcd.points)
        print(f"  {n_pts:,} points  bbox_min={np.asarray(source_pcd.get_min_bound()).round(2)}"
              f"  bbox_max={np.asarray(source_pcd.get_max_bound()).round(2)}")

        transform, fitness, rmse = icp_align(
            source_pcd,
            target_pcd,
            threshold=args.threshold,
            max_iter=args.max_iter,
            normal_radius=args.normal_radius,
            visualize=args.visualize,
            label=fname,
        )

        out_path = os.path.join(output_subdir, fname)
        save_las_with_transform(las_path, transform, out_path)

        reports.append({
            'file':         fname,
            'input':        las_path,
            'output':       out_path,
            'fitness':      round(fitness, 6),
            'inlier_rmse':  round(rmse, 6),
            'transform':    transform.tolist(),
        })

    # ------------------------------------------------------------------
    # 5. Summary report
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('=' * 80)
    for r in reports:
        status = "✓" if r['fitness'] > 0.1 else "⚠"
        print(f"  {status}  {r['file']:<30}  fitness={r['fitness']:.4f}  "
              f"RMSE={r['inlier_rmse']:.4f} m")

    os.makedirs(output_subdir, exist_ok=True)
    report_path = os.path.join(output_subdir, 'icp_report.json')
    with open(report_path, 'w') as fh:
        json.dump({
            'gml_file':    gml_path,
            'subfolder':   subfolder_path,
            'output_dir':  output_subdir,
            'density':     args.density,
            'threshold':   args.threshold,
            'max_iter':    args.max_iter,
            'normal_radius': args.normal_radius,
            'files':       reports,
        }, fh, indent=2)
    print(f"\n  Report saved → {report_path}")
    print('=' * 80)


if __name__ == '__main__':
    main()
