#!/usr/bin/env python3
"""
07_mesh_uv_plane.py — Planar (and curved) meshing via UV-plane projection

Pipeline for each .las file in outputs/07_merged_las/:
  1. Estimate normals and compute per-point curvature from local normal variation.
  2. Pre-split the cloud into FLAT (low curvature) and CURVED (high curvature)
     points BEFORE any RANSAC — this is the critical step that prevents RANSAC
     from fragmenting the curved wall.
  3. Flat points  → iterative RANSAC plane segmentation
                  → 2-D Delaunay + alpha-shape filtering (removes ghost triangles)
                  → TriangleMesh
  4. Curved points → ball-pivoting reconstruction (Open3D built-in, handles
                     arbitrary curved surfaces; normals already computed in step 1)
  5. Merge and save as .obj

Usage
-----
    conda activate las-env
    python scripts/LOD2toLOD3/07_mesh_uv_plane.py [options]

Options
-------
    --distance-threshold  RANSAC inlier distance   (default 0.05 m)
    --min-points          Minimum inliers per plane (default 500)
    --max-planes          Maximum planes to extract per cloud
    --alpha               Alpha-shape circumradius  (default 0.5 m)
    --curvature-knn       Neighbours used for curvature (default 20)
    --curvature-threshold Curvature above which a point is "curved" (default 0.15)
    --no-curve            Skip the curved-surface reconstruction pass
    --file                Process a specific file (skips interactive prompt)
"""

import sys
import argparse
import numpy as np
import open3d as o3d
import laspy
from pathlib import Path
from scipy.spatial import Delaunay, cKDTree

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]
INPUT_DIR  = BASE_DIR / "outputs" / "07_merged_las"
OUTPUT_DIR = BASE_DIR / "outputs" / "08_planar_meshes"


# ── I/O ──────────────────────────────────────────────────────────────────────

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


# ── curvature-based pre-segmentation ─────────────────────────────────────────

def split_flat_curved(
    pcd: o3d.geometry.PointCloud,
    k_nn: int,
    curvature_threshold: float,
    normal_radius: float = 0.3,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    Estimate normals and classify every point as flat or curved.

    Curvature metric: 1 - ‖mean(neighbour normals)‖
      → 0.0 means all neighbours point the same way  (flat)
      → → 1.0 means normals scatter in all directions (highly curved)

    The KNN matrix is built in one call so the inner loop is just
    a single numpy indexing + mean operation — no per-point Python loop.

    Returns (flat_pcd, curved_pcd) — both carry normals and colours.
    """
    print(f"    Estimating normals (radius={normal_radius} m, max_nn=30) …")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=10)

    points  = np.asarray(pcd.points)   # (N, 3)
    normals = np.asarray(pcd.normals)  # (N, 3)
    N = len(points)

    print(f"    Computing curvature (k={k_nn} neighbours) …")
    tree = cKDTree(points)
    # query returns shape (N, k_nn+1); first column is self → skip
    k_query = min(k_nn + 1, N)
    _, indices = tree.query(points, k=k_query)   # (N, k_query)
    nb_idx = indices[:, 1:]                       # (N, k_nn)  — drop self

    # Gather neighbour normals: (N, k_nn, 3)
    nb_normals = normals[nb_idx]

    # Mean normal magnitude: close to 1 ⇒ flat, close to 0 ⇒ curved
    mean_n = nb_normals.mean(axis=1)              # (N, 3)
    mean_mag = np.linalg.norm(mean_n, axis=1)     # (N,)
    curvature = 1.0 - mean_mag                    # (N,)

    flat_mask   = curvature < curvature_threshold
    curved_mask = ~flat_mask

    flat_pcd   = pcd.select_by_index(np.where(flat_mask)[0])
    curved_pcd = pcd.select_by_index(np.where(curved_mask)[0])

    print(f"    → {flat_mask.sum():,} flat pts   "
          f"({flat_mask.mean()*100:.1f}%)")
    print(f"    → {curved_mask.sum():,} curved pts "
          f"({curved_mask.mean()*100:.1f}%)  [threshold={curvature_threshold}]")

    return flat_pcd, curved_pcd


# ── flat-plane helpers ────────────────────────────────────────────────────────

def rotation_align_normal_to_z(normal: np.ndarray) -> np.ndarray:
    """Return a 3×3 rotation matrix that rotates `normal` onto [0, 0, 1]."""
    n = normal / np.linalg.norm(normal)
    z = np.array([0.0, 0.0, 1.0])
    dot = np.dot(n, z)
    if abs(dot) > 1.0 - 1e-8:
        R = np.eye(3)
        if dot < 0:
            R[2, 2] = -1.0
        return R
    v  = np.cross(n, z)
    s  = np.linalg.norm(v)
    c  = dot
    vx = np.array([[ 0,    -v[2],  v[1]],
                   [ v[2],  0,    -v[0]],
                   [-v[1],  v[0],  0   ]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def alpha_shape_filter(pts_2d: np.ndarray, simplices: np.ndarray, alpha: float) -> np.ndarray:
    """
    Remove Delaunay triangles whose circumradius > alpha.

    This eliminates 'ghost' triangles that bridge across empty space —
    the main cause of imaginary planes reported by the user.
    """
    if len(simplices) == 0:
        return np.empty((0, 3), dtype=np.int32)

    p = pts_2d[simplices]          # (M, 3, 2)
    a_ = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    b_ = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    c_ = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    s_ = (a_ + b_ + c_) / 2.0
    area = s_ * (s_ - a_) * (s_ - b_) * (s_ - c_)

    valid = area > 1e-14
    R = np.full(len(simplices), np.inf)
    R[valid] = (a_[valid] * b_[valid] * c_[valid]) / (4.0 * np.sqrt(area[valid]))

    kept = simplices[R <= alpha]
    return kept.astype(np.int32)


def mesh_plane_delaunay(
    points_3d: np.ndarray,
    colors_3d: np.ndarray | None,
    plane_model: np.ndarray,
    alpha: float,
) -> o3d.geometry.TriangleMesh:
    """
    Project 3-D plane inliers to 2-D, Delaunay-triangulate,
    alpha-shape filter, rebuild in 3-D.
    """
    a, b, c, _d = plane_model
    R = rotation_align_normal_to_z(np.array([a, b, c]))

    centroid = points_3d.mean(axis=0)
    pts_2d   = (R @ (points_3d - centroid).T).T[:, :2]

    tri   = Delaunay(pts_2d)
    faces = alpha_shape_filter(pts_2d, tri.simplices, alpha)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(points_3d)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if colors_3d is not None and len(colors_3d) == len(points_3d):
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors_3d)
    mesh.compute_vertex_normals()
    return mesh


def mesh_flat_planes(
    flat_pcd: o3d.geometry.PointCloud,
    distance_threshold: float,
    min_points: int,
    max_planes: int,
    alpha: float,
) -> list[o3d.geometry.TriangleMesh]:
    """Iterative RANSAC plane segmentation on the flat-point subset."""
    remaining  = flat_pcd
    has_color  = len(flat_pcd.colors) > 0
    meshes: list[o3d.geometry.TriangleMesh] = []

    for i in range(max_planes):
        n_pts = len(remaining.points)
        if n_pts < min_points:
            print(f"      [STOP] {n_pts} pts left (< {min_points})")
            break

        plane_model, inlier_idx = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000,
        )

        if len(inlier_idx) < min_points:
            print(f"      [STOP] plane {i}: only {len(inlier_idx)} inliers")
            break

        inlier_pts = np.asarray(remaining.points)[inlier_idx]
        inlier_clr = np.asarray(remaining.colors)[inlier_idx] if has_color else None

        print(f"      Plane {i:3d}: {len(inlier_idx):>7,} inliers  "
              f"(eq: {plane_model[0]:+.3f}x {plane_model[1]:+.3f}y "
              f"{plane_model[2]:+.3f}z {plane_model[3]:+.3f} = 0)")

        mesh = mesh_plane_delaunay(inlier_pts, inlier_clr, plane_model, alpha)
        n_tris = len(mesh.triangles)
        print(f"             → {n_tris:,} tris after α-filter")
        if n_tris > 0:
            meshes.append(mesh)

        remaining = remaining.select_by_index(inlier_idx, invert=True)

    return meshes


# ── curved-surface helper ─────────────────────────────────────────────────────

def mesh_curved_ballpivoting(
    curved_pcd: o3d.geometry.PointCloud,
    min_points: int,
) -> o3d.geometry.TriangleMesh | None:
    """
    Reconstruct a curved surface using Open3D's ball-pivoting algorithm.

    Ball-pivoting rolls a sphere of varying radius over the point cloud and
    records the triangles it forms — it naturally handles arbitrary curves
    without any plane-projection assumption.

    The ball radii are derived automatically from median nearest-neighbour
    spacing so the algorithm adapts to the density of the scan.

    Requires normals to be pre-computed on `curved_pcd`.
    """
    n = len(curved_pcd.points)
    if n < min_points:
        print(f"      [CURVE] Only {n} points — skipped")
        return None

    pts  = np.asarray(curved_pcd.points)
    tree = cKDTree(pts)
    # Sample up to 5000 points to estimate spacing (faster for large clouds)
    sample_idx = np.random.choice(n, min(5000, n), replace=False)
    dists, _ = tree.query(pts[sample_idx], k=2)
    avg_spacing = float(np.median(dists[:, 1]))
    if avg_spacing < 1e-6:
        avg_spacing = 0.01

    radii = [avg_spacing * r for r in [1.5, 2.5, 4.0, 6.0, 10.0]]
    print(f"      [CURVE] Ball-pivoting: {n:,} pts, "
          f"spacing≈{avg_spacing:.4f} m, "
          f"radii=[{', '.join(f'{r:.4f}' for r in radii)}]")

    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            curved_pcd, o3d.utility.DoubleVector(radii)
        )
    except Exception as e:
        print(f"      [CURVE] Ball-pivoting failed: {e}")
        return None

    n_tris = len(mesh.triangles)
    if n_tris == 0:
        print("      [CURVE] Ball-pivoting produced 0 triangles")
        return None

    print(f"      [CURVE] → {n_tris:,} triangles")
    mesh.compute_vertex_normals()
    return mesh


# ── main extraction ───────────────────────────────────────────────────────────

def extract_surfaces(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float,
    min_points: int,
    max_planes: int,
    alpha: float,
    curvature_knn: int,
    curvature_threshold: float,
    use_curve: bool,
) -> o3d.geometry.TriangleMesh:
    """Full pipeline: pre-split → flat RANSAC → curved ball-pivot → merge."""

    meshes: list[o3d.geometry.TriangleMesh] = []

    # ── 1. Pre-segment flat vs curved ────────────────────────────────────────
    if use_curve:
        print("\n  [1/3] Curvature pre-segmentation")
        flat_pcd, curved_pcd = split_flat_curved(
            pcd,
            k_nn=curvature_knn,
            curvature_threshold=curvature_threshold,
        )
    else:
        flat_pcd   = pcd
        curved_pcd = o3d.geometry.PointCloud()
        print("\n  [1/3] Curvature pre-segmentation — skipped (--no-curve)")

    # ── 2. Flat planes via RANSAC ─────────────────────────────────────────────
    print(f"\n  [2/3] Flat-plane RANSAC on {len(flat_pcd.points):,} points")
    flat_meshes = mesh_flat_planes(
        flat_pcd, distance_threshold, min_points, max_planes, alpha
    )
    meshes.extend(flat_meshes)
    print(f"      → {len(flat_meshes)} plane mesh(es)")

    # ── 3. Curved surface via ball-pivoting ───────────────────────────────────
    if use_curve and len(curved_pcd.points) >= min_points:
        print(f"\n  [3/3] Curved surface ball-pivoting on "
              f"{len(curved_pcd.points):,} points")
        cyl_mesh = mesh_curved_ballpivoting(curved_pcd, min_points)
        if cyl_mesh is not None:
            meshes.append(cyl_mesh)
    else:
        n_left = len(curved_pcd.points) if use_curve else 0
        print(f"\n  [3/3] Curved surface — "
              f"{'skipped (--no-curve)' if not use_curve else f'not enough points ({n_left})'}")

    # ── combine ───────────────────────────────────────────────────────────────
    if not meshes:
        print("  [WARN] No surfaces produced — returning empty mesh")
        return o3d.geometry.TriangleMesh()

    combined = meshes[0]
    for m in meshes[1:]:
        combined += m
    combined.compute_vertex_normals()

    print(f"\n  → {len(meshes)} surface(s) total, "
          f"{len(combined.vertices):,} verts, "
          f"{len(combined.triangles):,} tris")
    return combined


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate planar/curved meshes from point clouds")
    parser.add_argument("--distance-threshold", type=float, default=0.05)
    parser.add_argument("--min-points",         type=int,   default=500)
    parser.add_argument("--max-planes",         type=int,   default=None)
    parser.add_argument("--alpha",              type=float, default=None,
                        help="Alpha-shape radius in metres (default 0.5)")
    parser.add_argument("--curvature-knn",      type=int,   default=20,
                        help="Neighbours for curvature estimation (default 20)")
    parser.add_argument("--curvature-threshold", type=float, default=None,
                        help="Curvature cutoff 0–1 (default 0.15). "
                             "Lower = classify more points as curved.")
    parser.add_argument("--no-curve",  action="store_true",
                        help="Skip curved-surface reconstruction")
    parser.add_argument("--file",      type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  Surface Mesh from Point Clouds")
    print("=" * 60)
    print(f"\n  Input : {INPUT_DIR}")
    print(f"  Output: {OUTPUT_DIR}")

    if not INPUT_DIR.exists():
        sys.exit(f"\nERROR: Input directory not found:\n  {INPUT_DIR}")
    las_files = sorted(INPUT_DIR.glob("*.las"))
    if not las_files:
        sys.exit(f"\nERROR: No .las files found in:\n  {INPUT_DIR}")

    # ── file selection ────────────────────────────────────────────────────────
    if args.file:
        match = [f for f in las_files if f.stem == args.file or f.name == args.file]
        if not match:
            sys.exit(f"\nERROR: '{args.file}' not found.\n"
                     f"  Available: {', '.join(f.stem for f in las_files)}")
        selected_files = match
    else:
        print("\n  Available files:")
        print("    0) ALL")
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

    # ── per-run parameters ────────────────────────────────────────────────────
    if args.max_planes is not None:
        max_planes = args.max_planes
    else:
        v = input("  Max planes to extract per cloud [50]: ").strip()
        max_planes = int(v) if v else 50

    if args.alpha is not None:
        alpha = args.alpha
    else:
        v = input("  Alpha-shape radius in metres [0.5]: ").strip()
        alpha = float(v) if v else 0.5

    if args.curvature_threshold is not None:
        curv_thresh = args.curvature_threshold
    else:
        v = input("  Curvature threshold 0–1 [0.15]  "
                  "(lower → more points treated as curved): ").strip()
        curv_thresh = float(v) if v else 0.15

    use_curve = not args.no_curve

    print(f"\n  Params:")
    print(f"    dist_thresh        = {args.distance_threshold}")
    print(f"    min_points         = {args.min_points}")
    print(f"    max_planes         = {max_planes}")
    print(f"    alpha              = {alpha}")
    print(f"    curvature_knn      = {args.curvature_knn}")
    print(f"    curvature_threshold= {curv_thresh}")
    print(f"    curved pass        = {'yes (ball-pivoting)' if use_curve else 'no'}")
    print(f"  Files: {', '.join(f.name for f in selected_files)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for las_path in selected_files:
        print(f"\n{'─' * 60}")
        print(f"  Processing: {las_path.name}")
        print(f"{'─' * 60}")

        pcd = load_las_as_o3d(las_path)
        print(f"  Loaded {len(pcd.points):,} points"
              f"{'  (with colour)' if len(pcd.colors) > 0 else ''}")

        mesh = extract_surfaces(
            pcd,
            distance_threshold=args.distance_threshold,
            min_points=args.min_points,
            max_planes=max_planes,
            alpha=alpha,
            curvature_knn=args.curvature_knn,
            curvature_threshold=curv_thresh,
            use_curve=use_curve,
        )

        out_path = OUTPUT_DIR / f"{las_path.stem}.obj"
        o3d.io.write_triangle_mesh(str(out_path), mesh, write_vertex_colors=True)
        print(f"\n  ✓ Saved: {out_path}")

    print(f"\n{'=' * 60}")
    print(f"  Done — {len(selected_files)} mesh(es) written to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
