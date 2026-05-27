#!/usr/bin/env python3
"""
07b_mesh_poisson.py — Poisson surface reconstruction from a point cloud

Reads a .las file from outputs/07_merged_las, estimates oriented normals,
runs Open3D's Poisson surface reconstruction, trims low-density boundary
artefacts, and saves the result as an .obj to outputs/08_poisson_meshes.

Poisson reconstruction is watertight and handles curved surfaces naturally
without any plane-detection step.

Usage
-----
    conda activate las-env
    python scripts/LOD2toLOD3/07b_mesh_poisson.py [options]

Options
-------
    --depth            Poisson octree depth (default 9).
                       Higher = more detail, slower. Typical range 8–11.
    --normal-radius    Radius for normal estimation in metres (default 0.3).
    --normal-nn        Max neighbours for normal estimation (default 30).
    --density-quantile Remove vertices whose Poisson density falls below
                       this quantile (default 0.0 = keep everything).
                       Raise to 0.05–0.10 only if you see floating artefacts
                       at the boundary *after* confirming the curved wall is
                       already present.
    --scale            Poisson scale factor for the bounding cube (default 1.1).
    --file             Filename to process, skips interactive prompt.

Note on normals
---------------
Poisson is very sensitive to normal consistency.  This script orients
normals using the centroid of the point cloud as the outward-facing
reference (i.e. "the scanner was roughly here").  If your data was
scanned from an unusual position you can override with --viewpoint.
"""

import sys
import argparse
import numpy as np
import open3d as o3d
import laspy
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]
INPUT_DIR  = BASE_DIR / "outputs" / "07_merged_las"
OUTPUT_DIR = BASE_DIR / "outputs" / "08_poisson_meshes"


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
    viewpoint: np.ndarray | None = None,
) -> None:
    """
    Estimate normals and orient them consistently.

    Strategy: orient_normals_towards_camera_location() is used with a
    'camera' placed at the centroid of the cloud (or a user-supplied
    viewpoint).  Because the scanner was outside the building, normals
    will naturally point outward — toward the scanner — which is what
    Poisson needs for a correct watertight surface.

    Why not orient_normals_consistent_tangent_plane()?
    That function propagates orientations through a Riemannian graph.
    On buildings with both flat walls and curved sections it frequently
    flips the normals on the curved region, causing Poisson to silently
    omit or invert it.
    """
    print(f"    Estimating normals  (radius={radius} m, max_nn={max_nn}) …")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )

    if viewpoint is None:
        # Use the centroid of the cloud as the reference 'camera' position.
        # Since the scanner was outside the building this makes normals
        # point towards the exterior (= towards the scanner).
        viewpoint = np.asarray(pcd.points).mean(axis=0)

    pcd.orient_normals_towards_camera_location(
        camera_location=viewpoint.astype(np.float64)
    )
    print(f"    Normals estimated and oriented for {len(pcd.points):,} points")
    print(f"    Viewpoint used for orientation: "
          f"[{viewpoint[0]:.2f}, {viewpoint[1]:.2f}, {viewpoint[2]:.2f}]")


def poisson_reconstruct(
    pcd: o3d.geometry.PointCloud,
    depth: int,
    scale: float,
    density_quantile: float,
) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """
    Run Poisson surface reconstruction, optionally trim low-density boundary
    vertices, and return (mesh, densities).

    density_quantile=0.0 keeps every vertex — recommended as the first run
    so nothing is accidentally removed before you verify the curved wall is
    present.  Raise to 0.05–0.10 only to clean up floating caps afterwards.
    """
    print(f"\n    Running Poisson reconstruction (depth={depth}, scale={scale}) …")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale, linear_fit=False
    )
    densities = np.asarray(densities)
    print(f"    Raw mesh: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")
    print(f"    Density range: [{densities.min():.4f}, {densities.max():.4f}]")

    if density_quantile > 0.0:
        threshold = np.quantile(densities, density_quantile)
        print(f"    Trimming vertices below quantile "
              f"{density_quantile:.2f} (threshold={threshold:.4f}) …")
        remove_idx = np.where(densities < threshold)[0].tolist()
        mesh.remove_vertices_by_index(remove_idx)
        print(f"    After trim: {len(mesh.vertices):,} verts, "
              f"{len(mesh.triangles):,} tris")
    else:
        print("    Density trimming skipped (quantile=0.0)")

    mesh.compute_vertex_normals()
    return mesh, densities


def transfer_colors(
    mesh: o3d.geometry.TriangleMesh,
    pcd: o3d.geometry.PointCloud,
) -> None:
    """
    Paint mesh vertices with the nearest point-cloud colour using a KD-tree.
    Needed because Poisson does not carry colours from the input cloud.
    """
    if len(pcd.colors) == 0:
        return

    print("    Transferring colours from point cloud to mesh …")
    pcd_tree   = o3d.geometry.KDTreeFlann(pcd)
    mesh_verts = np.asarray(mesh.vertices)
    pcd_colors = np.asarray(pcd.colors)
    vert_colors = np.zeros((len(mesh_verts), 3), dtype=np.float64)

    for i, v in enumerate(mesh_verts):
        _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)
        vert_colors[i] = pcd_colors[idx[0]]

    mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)
    print("    Colours transferred")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Poisson surface reconstruction from a LAS point cloud")
    parser.add_argument("--depth",            type=int,   default=None,
                        help="Poisson octree depth (default 10)")
    parser.add_argument("--normal-radius",    type=float, default=0.3,
                        help="Normal estimation radius in metres (default 0.3)")
    parser.add_argument("--normal-nn",        type=int,   default=30,
                        help="Max neighbours for normals (default 30)")
    parser.add_argument("--density-quantile", type=float, default=None,
                        help="Trim vertices below this density quantile "
                             "0–1 (default 0.0 = keep all)")
    parser.add_argument("--scale",            type=float, default=1.1,
                        help="Poisson bounding-cube scale (default 1.1)")
    parser.add_argument("--viewpoint",        type=float, nargs=3,
                        metavar=("X", "Y", "Z"), default=None,
                        help="Override the normal-orientation viewpoint "
                             "(3 floats X Y Z). Defaults to point cloud centroid.")
    parser.add_argument("--file",             type=str,   default=None,
                        help="Specific filename to process (skips prompt)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Poisson Surface Reconstruction")
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

    # ── per-run parameters (interactive if not supplied via CLI) ───────────────
    if args.depth is not None:
        depth = args.depth
    else:
        v = input("  Poisson depth [10]  (higher = more detail, slower): ").strip()
        depth = int(v) if v else 10

    if args.density_quantile is not None:
        density_quantile = args.density_quantile
    else:
        v = input(
            "  Density trim quantile [0.0]  "
            "(0=keep all — recommended first run; raise to 0.05 to trim caps): "
        ).strip()
        density_quantile = float(v) if v else 0.0

    viewpoint = np.array(args.viewpoint) if args.viewpoint else None

    print(f"\n  Params:")
    print(f"    Poisson depth      = {depth}")
    print(f"    Normal radius      = {args.normal_radius} m")
    print(f"    Normal max_nn      = {args.normal_nn}")
    print(f"    Density quantile   = {density_quantile}")
    print(f"    Scale              = {args.scale}")
    print(f"    Viewpoint override = {args.viewpoint if args.viewpoint else 'auto (cloud centroid)'}")
    print(f"  Files: {', '.join(f.name for f in selected_files)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for las_path in selected_files:
        print(f"\n{'─' * 60}")
        print(f"  Processing: {las_path.name}")
        print(f"{'─' * 60}")

        pcd = load_las_as_o3d(las_path)
        print(f"    Loaded {len(pcd.points):,} points"
              f"{'  (with colour)' if len(pcd.colors) > 0 else ''}")

        estimate_normals(
            pcd,
            radius=args.normal_radius,
            max_nn=args.normal_nn,
            viewpoint=viewpoint,
        )

        mesh, _ = poisson_reconstruct(
            pcd,
            depth=depth,
            scale=args.scale,
            density_quantile=density_quantile,
        )

        # Paint mesh vertices with original point-cloud colours
        transfer_colors(mesh, pcd)

        out_path = OUTPUT_DIR / f"{las_path.stem}_poisson.obj"
        o3d.io.write_triangle_mesh(str(out_path), mesh, write_vertex_colors=True)
        print(f"\n    ✓ Saved: {out_path}")

    print(f"\n{'=' * 60}")
    print(f"  Done — {len(selected_files)} mesh(es) written to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
