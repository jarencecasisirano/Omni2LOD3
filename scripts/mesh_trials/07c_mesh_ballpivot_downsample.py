#!/usr/bin/env python3
"""
07c_mesh_ballpivot_downsample.py — Ball-pivoting with hardcoded paths & 0.3 m voxel

Processes all .las files in INPUT_DIR, voxel-downsamples at 0.3 m,
and writes BPA meshes to OUTPUT_DIR.

Usage
-----
    conda activate las-env
    python scripts/LOD2toLOD3/07c_mesh_ballpivot_downsample.py
"""


import sys
import numpy as np
import open3d as o3d
import laspy
from pathlib import Path
from scipy.spatial import cKDTree



# ── hardcoded config ──────────────────────────────────────────────────────────
INPUT_FILE  = Path("/home/khalil.torneros/07_merged_las/NIMBB-2-cleaned.las")
OUTPUT_DIR  = Path("/home/khalil.torneros/07_mesh_ballpivot")
VOXEL_SIZE  = 0.3          # metres
NORMAL_NN   = 30


# ── helpers ───────────────────────────────────────────────────────────────────

def load_las_as_o3d(las_path: Path) -> o3d.geometry.PointCloud:
    """Load a .las file into an Open3D PointCloud (colour if available)."""
    las    = laspy.read(str(las_path))
    points = np.vstack((las.x, las.y, las.z)).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if hasattr(las, "red"):
        colors = np.vstack((las.red, las.green, las.blue)).T / 65535.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def voxel_downsample(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> o3d.geometry.PointCloud:
    """
    Downsample the cloud to one point per voxel cell.

    This is the single most effective way to speed up BPA — halving the
    point count roughly halves the runtime.  Quality loss is negligible
    as long as voxel_size < the smallest feature you want to resolve.
    """
    n_before = len(pcd.points)
    t0 = time.perf_counter()
    ds = pcd.voxel_down_sample(voxel_size)
    elapsed = time.perf_counter() - t0
    n_after = len(ds.points)
    ratio = n_before / max(n_after, 1)
    print(f"    Downsampled {n_before:,} → {n_after:,} pts  "
          f"({ratio:.1f}× reduction)  [{elapsed:.1f}s]")
    return ds


def estimate_normals(
    pcd: o3d.geometry.PointCloud,
    radius: float,
    max_nn: int,
) -> None:
    """
    Estimate normals and orient them toward the cloud centroid.

    Using orient_normals_towards_camera_location with the centroid means
    normals consistently point outward (as the scanner was placed outside
    the building), without the graph-propagation failures of
    orient_normals_consistent_tangent_plane.
    """
    t0 = time.perf_counter()
    print(f"    Estimating normals  (radius={radius} m, max_nn={max_nn}) …")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn
        )
    )
    centroid = np.asarray(pcd.points).mean(axis=0).astype(np.float64)
    pcd.orient_normals_towards_camera_location(camera_location=centroid)
    print(f"    Normals ready  [{time.perf_counter()-t0:.1f}s]")


def compute_spacing(
    pcd: o3d.geometry.PointCloud,
    n_sample: int = 5000,
) -> float:
    """Return median nearest-neighbour distance from a sample of the cloud."""
    pts = np.asarray(pcd.points)
    idx = np.random.choice(len(pts), min(n_sample, len(pts)), replace=False)
    tree = cKDTree(pts[idx])
    dists, _ = tree.query(pts[idx], k=2)
    spacing = float(np.median(dists[:, 1]))
    return max(spacing, 1e-6)


def compute_auto_voxel(pcd: o3d.geometry.PointCloud) -> float:
    """
    Choose a voxel size so the downsampled cloud has at most ~100 K points,
    with a floor of 5× the raw point spacing.
    """
    spacing = compute_spacing(pcd)
    target  = 100_000
    n       = len(pcd.points)
    if n <= target:
        # Already small enough — set voxel to spacing so BPA radii are sensible
        return spacing * 2.0
    # Estimate voxel size from cloud volume and target point count
    pts    = np.asarray(pcd.points)
    extent = pts.max(axis=0) - pts.min(axis=0)
    volume = float(np.prod(np.maximum(extent, 1e-3)))
    voxel  = (volume / target) ** (1 / 3)
    voxel  = max(voxel, spacing * 5.0)   # never finer than 5× raw spacing
    return voxel


def compute_auto_radii(spacing: float) -> list[float]:
    """Return five BPA radii spanning 1× – 8× `spacing`."""
    radii = [spacing * m for m in (1.0, 1.5, 2.5, 4.0, 8.0)]
    print(f"    Auto radii (spacing≈{spacing:.4f} m): "
          f"{', '.join(f'{r:.4f}' for r in radii)}")
    return radii


def run_ball_pivoting(
    pcd: o3d.geometry.PointCloud,
    radii: list[float],
) -> o3d.geometry.TriangleMesh:
    """
    Run Open3D's ball-pivoting algorithm.

    Passing multiple radii lets BPA fill gaps at different scales — small
    radii resolve fine detail, larger radii bridge across sparser regions
    (e.g. a curved wall that was scanned at a more oblique angle).
    """
    print(f"\n    Running ball-pivoting  ({len(pcd.points):,} pts, "
          f"{len(radii)} radius/radii) …")
    t0 = time.perf_counter()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    elapsed = time.perf_counter() - t0
    print(f"    BPA done in {elapsed:.1f}s  →  "
          f"{len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")
    if len(mesh.triangles) == 0:
        print("    [WARN] No triangles — try larger --radii")
    mesh.compute_vertex_normals()
    return mesh


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Ball-Pivoting Surface Reconstruction  (voxel = 0.3 m)")
    print("=" * 60)
    print(f"\n  Input : {INPUT_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Voxel : {VOXEL_SIZE} m")

    if not INPUT_DIR.exists():
        sys.exit(f"\nERROR: Input directory not found:\n  {INPUT_DIR}")

    las_files = sorted(INPUT_DIR.glob("*.las"))
    if not las_files:
        sys.exit(f"\nERROR: No .las files found in:\n  {INPUT_DIR}")

    print(f"  Files : {', '.join(f.name for f in las_files)}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for las_path in las_files:
        print(f"\n{'─' * 60}")
        print(f"  Processing: {las_path.name}")
        print(f"{'─' * 60}")

        t_total = time.perf_counter()

        pcd = load_las_as_o3d(las_path)
        print(f"    Loaded {len(pcd.points):,} points"
              f"{'  (with colour)' if len(pcd.colors) > 0 else ''}")

        # ── voxel downsample ──────────────────────────────────────────────────
        print(f"    Voxel downsampling  (voxel={VOXEL_SIZE:.4f} m) …")
        ds_pcd = voxel_downsample(pcd, VOXEL_SIZE)

        # ── normals ───────────────────────────────────────────────────────────
        normal_radius = VOXEL_SIZE * 3.0
        estimate_normals(ds_pcd, radius=normal_radius, max_nn=NORMAL_NN)

        # ── ball radii ────────────────────────────────────────────────────────
        spacing = compute_spacing(ds_pcd)
        radii   = compute_auto_radii(spacing)

        mesh = run_ball_pivoting(ds_pcd, radii)

        out_path = OUTPUT_DIR / f"{las_path.stem}_bpa.obj"
        o3d.io.write_triangle_mesh(str(out_path), mesh, write_vertex_colors=True)
        elapsed = time.perf_counter() - t_total
        print(f"\n  ✓ Saved: {out_path}  [{elapsed:.1f}s total]")

    print(f"\n{'=' * 60}")
    print(f"  Done — {len(las_files)} mesh(es) written to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
