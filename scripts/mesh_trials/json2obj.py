#!/usr/bin/env python
"""Create a planar LOD3 facade approximation from a CityJSON LOD2 model and LAS point cloud.

This script does two things:
1. Exports semantic roof and wall surfaces from the LOD2 CityJSON model to OBJ.
2. Builds planar facade detail patches by measuring point-cloud offsets from each wall plane.

The facade-detail output is intentionally planar. It avoids Poisson-style blobby meshing by:
- using the LOD2 wall planes as the base reference,
- projecting nearby LAS points into wall-local 2D coordinates,
- estimating signed depth per grid cell,
- extracting connected recessed / protruding regions,
- emitting rectangular planar patches for each region.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import laspy
except ImportError:  # pragma: no cover - handled by runtime message
    laspy = None


EPS = 1e-9


@dataclass
class Surface:
    semantic_type: str
    ring: np.ndarray
    normal: np.ndarray
    plane_offset: float
    u_axis: np.ndarray
    v_axis: np.ndarray
    origin: np.ndarray
    uv: np.ndarray
    min_uv: np.ndarray
    max_uv: np.ndarray
    wall_id: str


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    default_cityjson = project_dir / "Data" / "nimbb_021726_fixed_json.json"
    default_las = project_dir / "Data" / "NIMBB-2-cleaned.las"
    default_outdir = project_dir / "Outputs" / "lod3_run_default"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cityjson",
        default=str(default_cityjson),
        help="Path to the LOD2 CityJSON file.",
    )
    parser.add_argument(
        "--las",
        default=str(default_las),
        help="Path to the facade point cloud LAS file.",
    )
    parser.add_argument(
        "--outdir",
        default=str(default_outdir),
        help="Output directory.",
    )
    parser.add_argument(
        "--building-id",
        default=None,
        help="CityObject id to use. Defaults to the first object with geometry.",
    )
    parser.add_argument(
        "--grid",
        type=float,
        default=0.35,
        help="Wall-local grid size in meters for facade detail detection.",
    )
    parser.add_argument(
        "--wall-buffer",
        type=float,
        default=0.75,
        help="Extra envelope around each wall while collecting LAS points.",
    )
    parser.add_argument(
        "--plane-tol",
        type=float,
        default=0.35,
        help="Maximum distance from the wall plane for LAS points to be considered.",
    )
    parser.add_argument(
        "--detail-depth",
        type=float,
        default=0.12,
        help="Minimum signed offset in meters to accept a recess / overhang region.",
    )
    parser.add_argument(
        "--min-region-cells",
        type=int,
        default=4,
        help="Minimum number of occupied grid cells for a facade-detail region.",
    )
    return parser.parse_args()


def load_cityjson(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def apply_transform(vertices: np.ndarray, transform: dict | None) -> np.ndarray:
    if not transform:
        return vertices.astype(float)
    scale = np.asarray(transform.get("scale", [1.0, 1.0, 1.0]), dtype=float)
    translate = np.asarray(transform.get("translate", [0.0, 0.0, 0.0]), dtype=float)
    return vertices.astype(float) * scale + translate


def pick_geometry(cityjson: dict, building_id: str | None) -> tuple[str, dict]:
    city_objects = cityjson.get("CityObjects", {})
    if building_id:
        obj = city_objects[building_id]
        if not obj.get("geometry"):
            raise ValueError(f"CityObject '{building_id}' has no geometry.")
        return building_id, obj["geometry"][0]
    for object_id, obj in city_objects.items():
        if obj.get("geometry"):
            return object_id, obj["geometry"][0]
    raise ValueError("No CityObject with geometry found.")


def iter_surfaces(geometry: dict) -> Iterable[tuple[list[int], str]]:
    semantics = geometry.get("semantics", {})
    semantic_surfaces = semantics.get("surfaces", [])
    values = semantics.get("values", [])
    shells = geometry["boundaries"]
    if geometry.get("type") != "Solid":
        raise ValueError("Only Solid CityJSON geometry is supported.")
    shell0 = shells[0]
    for face_index, face in enumerate(shell0):
        semantic_index = values[0][face_index] if values and values[0] else None
        semantic_type = (
            semantic_surfaces[semantic_index]["type"] if semantic_index is not None else "UnknownSurface"
        )
        outer_ring = face[0]
        yield outer_ring, semantic_type


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < EPS:
        raise ValueError("Degenerate vector encountered.")
    return vector / norm


def polygon_normal(ring: np.ndarray) -> np.ndarray:
    centered = ring - ring.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return normalize(vh[-1])


def make_surface(ring: np.ndarray, semantic_type: str, wall_id: str) -> Surface:
    normal = polygon_normal(ring)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(normal, z_axis)) > 0.98:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        ref = z_axis
    u_axis = normalize(np.cross(ref, normal))
    v_axis = normalize(np.cross(normal, u_axis))
    origin = ring[0]
    uv = np.column_stack(((ring - origin) @ u_axis, (ring - origin) @ v_axis))
    min_uv = uv.min(axis=0)
    max_uv = uv.max(axis=0)
    plane_offset = -float(np.dot(normal, origin))
    return Surface(
        semantic_type=semantic_type,
        ring=ring,
        normal=normal,
        plane_offset=plane_offset,
        u_axis=u_axis,
        v_axis=v_axis,
        origin=origin,
        uv=uv,
        min_uv=min_uv,
        max_uv=max_uv,
        wall_id=wall_id,
    )


def triangulate_ring(ring: Sequence[np.ndarray]) -> list[tuple[int, int, int]]:
    if len(ring) < 3:
        return []
    triangles = []
    for i in range(1, len(ring) - 1):
        triangles.append((0, i, i + 1))
    return triangles


def write_obj(path: Path, objects: list[tuple[str, np.ndarray, list[tuple[int, int, int]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        vertex_offset = 1
        for name, vertices, triangles in objects:
            fh.write(f"o {name}\n")
            for vx, vy, vz in vertices:
                fh.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
            for a, b, c in triangles:
                fh.write(f"f {a + vertex_offset} {b + vertex_offset} {c + vertex_offset}\n")
            vertex_offset += len(vertices)


def export_base_surfaces(cityjson: dict, geometry: dict, outdir: Path) -> tuple[list[Surface], list[Surface]]:
    vertices = apply_transform(np.asarray(cityjson["vertices"], dtype=float), cityjson.get("transform"))
    roofs: list[Surface] = []
    walls: list[Surface] = []
    roof_objs = []
    wall_objs = []
    wall_counter = 0
    roof_counter = 0
    for ring_ix, (ring_indices, semantic_type) in enumerate(iter_surfaces(geometry)):
        ring = vertices[np.asarray(ring_indices, dtype=int)]
        name = f"{semantic_type.lower()}_{ring_ix:03d}"
        tris = triangulate_ring(ring)
        surface = make_surface(ring, semantic_type, wall_id=name)
        if semantic_type == "RoofSurface":
            roof_counter += 1
            roofs.append(surface)
            roof_objs.append((f"roof_{roof_counter:03d}", ring, tris))
        elif semantic_type == "WallSurface":
            wall_counter += 1
            surface.wall_id = f"wall_{wall_counter:03d}"
            walls.append(surface)
            wall_objs.append((surface.wall_id, ring, tris))
    if roof_objs:
        write_obj(outdir / "roof.obj", roof_objs)
    if wall_objs:
        write_obj(outdir / "walls.obj", wall_objs)
    return roofs, walls


def load_las_points(path: Path) -> np.ndarray:
    if laspy is None:
        raise RuntimeError("laspy is required. Install it with: python -m pip install laspy")
    las = laspy.read(path)
    return np.column_stack((las.x, las.y, las.z)).astype(float)


def point_to_surface_frame(surface: Surface, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rel = points - surface.origin
    u = rel @ surface.u_axis
    v = rel @ surface.v_axis
    d = rel @ surface.normal
    return np.column_stack((u, v)), d


def select_wall_points(points: np.ndarray, surface: Surface, wall_buffer: float, plane_tol: float) -> tuple[np.ndarray, np.ndarray]:
    uv, depth = point_to_surface_frame(surface, points)
    inside = (
        (uv[:, 0] >= surface.min_uv[0] - wall_buffer)
        & (uv[:, 0] <= surface.max_uv[0] + wall_buffer)
        & (uv[:, 1] >= surface.min_uv[1] - wall_buffer)
        & (uv[:, 1] <= surface.max_uv[1] + wall_buffer)
        & (np.abs(depth) <= plane_tol)
    )
    return uv[inside], depth[inside]


def build_depth_grid(uv: np.ndarray, depth: np.ndarray, min_uv: np.ndarray, max_uv: np.ndarray, grid: float) -> tuple[np.ndarray, np.ndarray]:
    nx = max(1, int(math.ceil((max_uv[0] - min_uv[0]) / grid)))
    ny = max(1, int(math.ceil((max_uv[1] - min_uv[1]) / grid)))
    cell_values: dict[tuple[int, int], list[float]] = defaultdict(list)
    ij = np.floor((uv - min_uv) / grid).astype(int)
    for (i, j), dep in zip(ij, depth, strict=False):
        if 0 <= i < nx and 0 <= j < ny:
            cell_values[(i, j)].append(float(dep))
    median_grid = np.full((nx, ny), np.nan, dtype=float)
    count_grid = np.zeros((nx, ny), dtype=int)
    for (i, j), values in cell_values.items():
        median_grid[i, j] = float(np.median(values))
        count_grid[i, j] = len(values)
    return median_grid, count_grid


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    regions: list[list[tuple[int, int]]] = []
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if not mask[i, j] or seen[i, j]:
                continue
            region = []
            queue = deque([(i, j)])
            seen[i, j] = True
            while queue:
                ci, cj = queue.popleft()
                region.append((ci, cj))
                for di, dj in neighbors:
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1] and mask[ni, nj] and not seen[ni, nj]:
                        seen[ni, nj] = True
                        queue.append((ni, nj))
            regions.append(region)
    return regions


def rectangular_patch(surface: Surface, min_corner: np.ndarray, max_corner: np.ndarray, depth: float) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    # A planar quad for the detail face plus four side quads tying it back to the base wall plane.
    p00 = surface.origin + surface.u_axis * min_corner[0] + surface.v_axis * min_corner[1]
    p10 = surface.origin + surface.u_axis * max_corner[0] + surface.v_axis * min_corner[1]
    p11 = surface.origin + surface.u_axis * max_corner[0] + surface.v_axis * max_corner[1]
    p01 = surface.origin + surface.u_axis * min_corner[0] + surface.v_axis * max_corner[1]
    offset = surface.normal * depth

    front = np.vstack([p00 + offset, p10 + offset, p11 + offset, p01 + offset])
    back = np.vstack([p00, p10, p11, p01])
    vertices = np.vstack([front, back])
    faces = [
        (0, 1, 2), (0, 2, 3),  # front
        (4, 7, 6), (4, 6, 5),  # back
        (0, 4, 5), (0, 5, 1),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 4), (3, 4, 0),
    ]
    return vertices, faces


def reconstruct_facade_details(
    points: np.ndarray,
    walls: list[Surface],
    outdir: Path,
    grid: float,
    wall_buffer: float,
    plane_tol: float,
    detail_depth: float,
    min_region_cells: int,
) -> int:
    detail_objects = []
    detail_count = 0
    for wall in walls:
        wall_uv, wall_depth = select_wall_points(points, wall, wall_buffer=wall_buffer, plane_tol=plane_tol)
        if len(wall_depth) == 0:
            continue
        depth_grid, count_grid = build_depth_grid(wall_uv, wall_depth, wall.min_uv, wall.max_uv, grid)
        if np.all(np.isnan(depth_grid)):
            continue
        valid_depths = depth_grid[~np.isnan(depth_grid)]
        baseline = float(np.median(valid_depths))
        signed = depth_grid - baseline
        occupied = count_grid > 0
        recess_mask = occupied & (signed <= -detail_depth)
        overhang_mask = occupied & (signed >= detail_depth)
        for region_kind, mask in (("recess", recess_mask), ("overhang", overhang_mask)):
            for region in connected_components(mask):
                if len(region) < min_region_cells:
                    continue
                depths = np.array([signed[i, j] for i, j in region], dtype=float)
                region_depth = float(np.median(depths))
                ii = np.array([i for i, _ in region], dtype=int)
                jj = np.array([j for _, j in region], dtype=int)
                min_corner = wall.min_uv + np.array([ii.min() * grid, jj.min() * grid], dtype=float)
                max_corner = wall.min_uv + np.array([(ii.max() + 1) * grid, (jj.max() + 1) * grid], dtype=float)
                vertices, faces = rectangular_patch(wall, min_corner, max_corner, region_depth)
                detail_count += 1
                detail_objects.append((f"{wall.wall_id}_{region_kind}_{detail_count:03d}", vertices, faces))
    if detail_objects:
        write_obj(outdir / "facade_details.obj", detail_objects)
    return detail_count


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cityjson_path = Path(args.cityjson)
    las_path = Path(args.las)

    cityjson = load_cityjson(cityjson_path)
    building_id, geometry = pick_geometry(cityjson, args.building_id)
    roofs, walls = export_base_surfaces(cityjson, geometry, outdir)
    points = load_las_points(las_path)
    detail_count = reconstruct_facade_details(
        points=points,
        walls=walls,
        outdir=outdir,
        grid=args.grid,
        wall_buffer=args.wall_buffer,
        plane_tol=args.plane_tol,
        detail_depth=args.detail_depth,
        min_region_cells=args.min_region_cells,
    )

    summary = {
        "building_id": building_id,
        "roof_surface_count": len(roofs),
        "wall_surface_count": len(walls),
        "detail_patch_count": detail_count,
        "outputs": {
            "roof_obj": str((outdir / "roof.obj").resolve()),
            "walls_obj": str((outdir / "walls.obj").resolve()),
            "facade_details_obj": str((outdir / "facade_details.obj").resolve()),
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
