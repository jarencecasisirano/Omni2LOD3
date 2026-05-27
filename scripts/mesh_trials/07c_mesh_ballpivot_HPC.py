#!/usr/bin/env python3
"""
07c_mesh_ballpivot.py — Ball-pivoting surface reconstruction from a point cloud

Reads a .las file from outputs/07_merged_las, estimates oriented normals,
and runs Open3D's ball-pivoting algorithm (BPA) to create a 3D mesh.

Ball-pivoting rolls spheres of one or more radii over the point cloud and
records each triangle the rolling sphere touches.  Unlike Poisson it does
NOT add extra geometry beyond the real data, so curved and flat surfaces
are both reconstructed accurately without any plane-segmentation step.

Usage
-----
    conda activate las-env
    python scripts/LOD2toLOD3/07c_mesh_ballpivot.py [options]

Options
-------
    --radii            Comma-separated ball radii in metres, e.g. 0.05,0.1,0.2
                       Default: auto-computed from median point spacing.
    --normal-radius    Radius for normal estimation in metres (default 0.3).
    --normal-nn        Max neighbours for normal estimation (default 30).
    --file             Filename to process, skips interactive prompt.

Notes
-----
- Ball-pivoting requires normals. Orientation is done via
  orient_normals_towards_camera_location using the cloud's centroid as the
  reference point (= the scanner was outside the building).
- The ball radii must bracket the local point spacing. Too small → holes,
  too large → triangles bridging across gaps. Auto-mode tries a spread of
  radii from 1× to 8× the median nearest-neighbour distance.
"""
from pathlib import Path
import numpy as np
import open3d as o3d
import laspy
from scipy.spatial import cKDTree

# ── paths (hardcoded for HPC) ─────────────────────────────────────────────────
INPUT_FILE  = Path("/home/khalil.torneros/07_merged_las/NIMBB-2-cleaned.las")
OUTPUT_DIR  = Path("/home/khalil.torneros/07_mesh_ballpivot")


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
    print(f"    Estimating normals  (radius={radius} m, max_nn={max_nn}) …")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn
        )
    )
    centroid = np.asarray(pcd.points).mean(axis=0).astype(np.float64)
    pcd.orient_normals_towards_camera_location(camera_location=centroid)
    print(f"    Normals ready for {len(pcd.points):,} points")


def compute_auto_radii(pcd: o3d.geometry.PointCloud, n_sample: int = 5000) -> list[float]:
    """
    Estimate good ball radii from the median nearest-neighbour spacing.

    Samples up to `n_sample` points for speed.  Returns five radii that
    span 1× – 8× the median spacing so the BPA can fill both fine detail
    and slightly larger gaps.
    """
    pts = np.asarray(pcd.points)
    idx = np.random.choice(len(pts), min(n_sample, len(pts)), replace=False)
    tree = cKDTree(pts[idx])
    dists, _ = tree.query(pts[idx], k=2)          # k=2: self + 1 neighbour
    spacing = float(np.median(dists[:, 1]))
    if spacing < 1e-6:
        spacing = 0.01
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
    print(f"\n    Running ball-pivoting with {len(radii)} radius/radii …")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    print(f"    Result: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")
    if len(mesh.triangles) == 0:
        print("    [WARN] No triangles — consider adjusting --radii")
    mesh.compute_vertex_normals()
    return mesh


# ── hardcoded parameters ──────────────────────────────────────────────────────
NORMAL_RADIUS = 0.3   # metres
NORMAL_NN     = 30


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Ball-Pivoting Surface Reconstruction (HPC)")
    print("=" * 60)
    print(f"\n  Input : {INPUT_FILE}")
    print(f"  Output: {OUTPUT_DIR}")

    if not INPUT_FILE.exists():
        sys.exit(f"\nERROR: Input file not found:\n  {INPUT_FILE}")

    print(f"\n{'─' * 60}")
    print(f"  Processing: {INPUT_FILE.name}")
    print(f"{'─' * 60}")

    pcd = load_las_as_o3d(INPUT_FILE)
    print(f"    Loaded {len(pcd.points):,} points"
          f"{'  (with colour)' if len(pcd.colors) > 0 else ''}")

    estimate_normals(pcd, radius=NORMAL_RADIUS, max_nn=NORMAL_NN)

    radii = compute_auto_radii(pcd)

    mesh = run_ball_pivoting(pcd, radii)

    out_path = OUTPUT_DIR / f"{INPUT_FILE.stem}_bpa.obj"
    o3d.io.write_triangle_mesh(str(out_path), mesh, write_vertex_colors=True)
    print(f"\n  ✓ Saved: {out_path}")

    print(f"\n{'=' * 60}")
    print("  Done")
    print("=" * 60)


if __name__ == "__main__":
    main()
