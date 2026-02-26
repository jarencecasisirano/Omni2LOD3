#!/usr/bin/env python3
"""
07_mesh_uv_plane.py — Planar meshing via UV-plane projection

For each .las file in outputs/07_merged_las/:
  1. Iteratively segment dominant planes with RANSAC
  2. Project each plane's inliers onto a 2D coordinate frame
  3. Delaunay-triangulate in 2D
  4. Build an Open3D TriangleMesh from the original 3D coords + 2D faces
  5. Merge all plane meshes and save as .obj

Usage
-----
    conda activate lidar-test
    python scripts/LOD2toLOD3/07_mesh_uv_plane.py [options]

Options
-------
    --distance-threshold  RANSAC inlier distance   (default 0.05)
    --min-points          Minimum inliers per plane (default 500)
    --max-planes          Maximum planes to extract (default 50)
"""

import sys
import argparse
import numpy as np
import open3d as o3d
import laspy
from pathlib import Path
from scipy.spatial import Delaunay

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]
INPUT_DIR  = BASE_DIR / "outputs" / "07_merged_las"
OUTPUT_DIR = BASE_DIR / "outputs" / "08_planar_meshes"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_las_as_o3d(las_path: Path) -> o3d.geometry.PointCloud:
    """Load a .las file into an Open3D PointCloud (with colour if present)."""
    las = laspy.read(str(las_path))
    points = np.vstack((las.x, las.y, las.z)).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if hasattr(las, "red"):
        colors = np.vstack((las.red, las.green, las.blue)).T / 65535.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def rotation_align_normal_to_z(normal: np.ndarray) -> np.ndarray:
    """Return a 3×3 rotation matrix that aligns `normal` with [0, 0, 1]."""
    n = normal / np.linalg.norm(normal)
    z = np.array([0.0, 0.0, 1.0])

    # If already aligned (or anti-aligned), return identity (or flip)
    dot = np.dot(n, z)
    if abs(dot) > 1.0 - 1e-8:
        R = np.eye(3)
        if dot < 0:
            R[2, 2] = -1.0  # flip Z
        return R

    v = np.cross(n, z)
    s = np.linalg.norm(v)
    c = dot
    vx = np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ],
    ])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    return R


def mesh_plane_delaunay(
    points_3d: np.ndarray,
    colors_3d: np.ndarray | None,
    plane_model: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    """
    Given 3D inlier points on a plane, project to 2D, triangulate,
    and return a TriangleMesh with original 3D coordinates.
    """
    a, b, c, _d = plane_model
    normal = np.array([a, b, c])
    R = rotation_align_normal_to_z(normal)

    centroid = points_3d.mean(axis=0)
    pts_centered = points_3d - centroid
    pts_rotated = (R @ pts_centered.T).T          # now roughly on XY plane
    pts_2d = pts_rotated[:, :2]                    # drop Z

    # Delaunay triangulation in 2D
    tri = Delaunay(pts_2d)
    faces = tri.simplices                          # (N_tri, 3)

    # Build mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points_3d)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    if colors_3d is not None and len(colors_3d) == len(points_3d):
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors_3d)

    mesh.compute_vertex_normals()
    return mesh


def extract_planes(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float,
    min_points: int,
    max_planes: int,
) -> o3d.geometry.TriangleMesh:
    """
    Iteratively segment planes from `pcd` and return a merged mesh.
    """
    remaining = pcd
    meshes: list[o3d.geometry.TriangleMesh] = []

    has_color = len(pcd.colors) > 0

    for i in range(max_planes):
        n_pts = len(remaining.points)
        if n_pts < min_points:
            print(f"    [STOP] Only {n_pts} points left (< {min_points})")
            break

        # RANSAC plane segmentation
        plane_model, inlier_idx = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000,
        )

        if len(inlier_idx) < min_points:
            print(f"    [STOP] Plane {i}: {len(inlier_idx)} inliers (< {min_points})")
            break

        inlier_pts = np.asarray(remaining.points)[inlier_idx]
        inlier_clr = np.asarray(remaining.colors)[inlier_idx] if has_color else None

        print(f"    Plane {i:3d}: {len(inlier_idx):>7,} inliers  "
              f"(eq: {plane_model[0]:+.3f}x {plane_model[1]:+.3f}y "
              f"{plane_model[2]:+.3f}z {plane_model[3]:+.3f} = 0)")

        mesh = mesh_plane_delaunay(inlier_pts, inlier_clr, plane_model)
        meshes.append(mesh)

        # Remove inliers from the remaining cloud
        remaining = remaining.select_by_index(inlier_idx, invert=True)

    if not meshes:
        print("    [WARN] No planes found — returning empty mesh")
        return o3d.geometry.TriangleMesh()

    # Merge all plane meshes
    combined = meshes[0]
    for m in meshes[1:]:
        combined += m

    combined.compute_vertex_normals()
    print(f"    → {len(meshes)} plane(s), "
          f"{len(combined.vertices):,} verts, "
          f"{len(combined.triangles):,} tris")
    return combined


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate planar meshes from point clouds (UV-plane method)")
    parser.add_argument("--distance-threshold", type=float, default=0.05,
                        help="RANSAC inlier distance threshold (default: 0.05)")
    parser.add_argument("--min-points", type=int, default=500,
                        help="Minimum inlier count per plane (default: 500)")
    parser.add_argument("--max-planes", type=int, default=None,
                        help="Maximum planes to extract per cloud (skips prompt)")
    parser.add_argument("--file", type=str, default=None,
                        help="Filename to process (skips prompt)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Planar Mesh from Point Clouds (UV-plane method)")
    print("=" * 60)
    print(f"\n  Input : {INPUT_DIR}")
    print(f"  Output: {OUTPUT_DIR}")

    if not INPUT_DIR.exists():
        sys.exit(f"\nERROR: Input directory not found:\n  {INPUT_DIR}")

    las_files = sorted(INPUT_DIR.glob("*.las"))
    if not las_files:
        sys.exit(f"\nERROR: No .las files found in:\n  {INPUT_DIR}")

    # ── File selection ───────────────────────────────────────────────────────
    if args.file:
        # CLI override — find matching file
        match = [f for f in las_files if f.stem == args.file or f.name == args.file]
        if not match:
            sys.exit(f"\nERROR: '{args.file}' not found in {INPUT_DIR}\n"
                     f"  Available: {', '.join(f.stem for f in las_files)}")
        selected_files = match
    else:
        # Interactive selection
        print(f"\n  Available files:")
        print(f"    0) ALL")
        for idx, f in enumerate(las_files, start=1):
            print(f"    {idx}) {f.name}")

        choice = input(f"\n  Select file to process [0-{len(las_files)}]: ").strip()
        try:
            choice_idx = int(choice)
        except ValueError:
            sys.exit(f"\nERROR: Invalid selection: '{choice}'")

        if choice_idx == 0:
            selected_files = las_files
        elif 1 <= choice_idx <= len(las_files):
            selected_files = [las_files[choice_idx - 1]]
        else:
            sys.exit(f"\nERROR: Selection out of range: {choice_idx}")

    # ── Max planes ───────────────────────────────────────────────────────────
    if args.max_planes is not None:
        max_planes = args.max_planes
    else:
        mp_input = input("  Max planes to extract per cloud [50]: ").strip()
        max_planes = int(mp_input) if mp_input else 50

    print(f"\n  Params: dist_thresh={args.distance_threshold}  "
          f"min_pts={args.min_points}  max_planes={max_planes}")
    print(f"  Files:  {', '.join(f.name for f in selected_files)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for las_path in selected_files:
        print(f"\n{'─' * 60}")
        print(f"  Processing: {las_path.name}")
        print(f"{'─' * 60}")

        pcd = load_las_as_o3d(las_path)
        print(f"    Loaded {len(pcd.points):,} points"
              f"{'  (with colour)' if len(pcd.colors) > 0 else ''}")

        mesh = extract_planes(
            pcd,
            distance_threshold=args.distance_threshold,
            min_points=args.min_points,
            max_planes=max_planes,
        )

        out_path = OUTPUT_DIR / f"{las_path.stem}.obj"
        o3d.io.write_triangle_mesh(str(out_path), mesh, write_vertex_colors=True)
        print(f"    ✓ Saved: {out_path}")

    print(f"\n{'=' * 60}")
    print(f"  Done — {len(selected_files)} mesh(es) written to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

