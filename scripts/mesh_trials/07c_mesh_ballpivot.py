#!/usr/bin/env python3
"""
07c_mesh_ballpivot.py — Ball-pivoting surface reconstruction from a point cloud

Reads a .las file from outputs/07_merged_las, estimates oriented normals,
and runs Open3D's ball-pivoting algorithm (BPA) to create a 3D mesh.

Ball-pivoting speed is dominated by point count.  The script therefore
voxel-downsamples the cloud first (--voxel-size, default auto ~0.05 m),
which typically reduces millions of raw scan points to tens-of-thousands
and cuts runtime from hours to under a minute.

Usage
-----
    conda activate las-env
    python scripts/LOD2toLOD3/07c_mesh_ballpivot.py [options]

Options
-------
    --voxel-size       Voxel side length for downsampling in metres.
                       Default: auto (~5× median point spacing).
                       Set to 0 to skip downsampling (WARNING: very slow).
    --radii            Comma-separated ball radii in metres, e.g. 0.05,0.1,0.2
                       Default: auto-computed from voxel size / point spacing.
    --normal-radius    Radius for normal estimation in metres (default: 3× voxel).
    --normal-nn        Max neighbours for normal estimation (default 30).
    --file             Filename to process, skips interactive prompt.
"""


import argparse
import numpy as np
import open3d as o3d
import laspy
from pathlib import Path
from scipy.spatial import cKDTree

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]
INPUT_DIR  = BASE_DIR / "outputs" / "07_merged_las"
OUTPUT_DIR = BASE_DIR / "outputs" / "08_ballpivot_meshes"


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
    parser = argparse.ArgumentParser(
        description="Ball-pivoting 3D mesh from a LAS point cloud")
    parser.add_argument("--voxel-size",    type=float, default=None,
                        help="Voxel size for downsampling in metres (default auto). "
                             "Set to 0 to skip (WARNING: very slow on large clouds).")
    parser.add_argument("--radii",         type=str,   default=None,
                        help="Comma-separated ball radii in metres, e.g. 0.05,0.1,0.2 "
                             "(default: auto from voxel size)")
    parser.add_argument("--normal-radius", type=float, default=None,
                        help="Radius for normal estimation (default: 3× voxel size)")
    parser.add_argument("--normal-nn",     type=int,   default=30,
                        help="Max neighbours for normal estimation (default 30)")
    parser.add_argument("--file",          type=str,   default=None,
                        help="Filename to process, skips interactive prompt")
    args = parser.parse_args()

    print("=" * 60)
    print("  Ball-Pivoting Surface Reconstruction")
    print("=" * 60)
    print(f"\n  Input : {INPUT_DIR}")
    print(f"  Output: {OUTPUT_DIR}")

    if not INPUT_DIR.exists():
        sys.exit(f"\nERROR: Input directory not found:\n  {INPUT_DIR}")

    las_files = sorted(INPUT_DIR.glob("*.las"))
    if not las_files:
        sys.exit(f"\nERROR: No .las files found in:\n  {INPUT_DIR}")

    # ── file selection ─────────────────────────────────────────────────────────
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

    # ── extra params ───────────────────────────────────────────────────────────
    manual_radii: list[float] | None = None
    if args.radii:
        manual_radii = [float(r.strip()) for r in args.radii.split(",") if r.strip()]

    print(f"\n  Params:")
    print(f"    Voxel size     = {args.voxel_size if args.voxel_size is not None else 'auto'}")
    print(f"    Normal radius  = {args.normal_radius if args.normal_radius else 'auto (3× voxel)'}")
    print(f"    Normal max_nn  = {args.normal_nn}")
    print(f"    Ball radii     = {manual_radii if manual_radii else 'auto'}")
    print(f"  Files: {', '.join(f.name for f in selected_files)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for las_path in selected_files:
        print(f"\n{'─' * 60}")
        print(f"  Processing: {las_path.name}")
        print(f"{'─' * 60}")

        t_total = time.perf_counter()

        pcd = load_las_as_o3d(las_path)
        print(f"    Loaded {len(pcd.points):,} points"
              f"{'  (with colour)' if len(pcd.colors) > 0 else ''}")

        # ── voxel downsample ──────────────────────────────────────────────────
        if args.voxel_size == 0.0:
            print("    Downsampling skipped (--voxel-size 0)  — this may be slow")
            ds_pcd = pcd
            voxel  = compute_spacing(pcd) * 2.0
        else:
            voxel = args.voxel_size if args.voxel_size else compute_auto_voxel(pcd)
            print(f"    Voxel downsampling  (voxel={voxel:.4f} m) …")
            ds_pcd = voxel_downsample(pcd, voxel)

        # ── normals ───────────────────────────────────────────────────────────
        normal_radius = args.normal_radius if args.normal_radius else voxel * 3.0
        estimate_normals(ds_pcd, radius=normal_radius, max_nn=args.normal_nn)

        # ── ball radii ────────────────────────────────────────────────────────
        if manual_radii:
            radii = manual_radii
        else:
            spacing = compute_spacing(ds_pcd)
            radii   = compute_auto_radii(spacing)

        mesh = run_ball_pivoting(ds_pcd, radii)

        out_path = OUTPUT_DIR / f"{las_path.stem}_bpa.obj"
        o3d.io.write_triangle_mesh(str(out_path), mesh, write_vertex_colors=True)
        print(f"\n  ✓ Saved: {out_path}")

    print(f"\n{'=' * 60}")
    print(f"  Done — {len(selected_files)} mesh(es) written to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
