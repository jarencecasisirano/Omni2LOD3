#!/usr/bin/env python3
"""
09_visibility_conflict_map.py
==============================
Visibility analysis and conflict probability mapping for LOD2 CityGML buildings.

Implements the ray-casting / voxel-based visibility branch from:
  "Scan2LoD3: Reconstructing semantic 3D building models at LoD3 using ray
   casting and Bayesian networks" (CVPR Workshops 2023, Wysocki et al.)
  https://github.com/OloOcki/scan2lod3

Algorithm
---------
For every laser point p_i with sensor origin s_i:
  1. Trace the ray s_i → p_i through a voxel grid using the 3-D DDA algorithm.
  2. Mark traversed voxels as EMPTY (free space).
  3. Mark the terminal voxel (where p_i lives) as OCCUPIED.

For every GML WallSurface polygon:
  4. Find all voxels whose centres lie on / near the polygon plane and inside
     the polygon's footprint.
  5. Each such voxel is:
       CONFIRMED  – if it is OCCUPIED  (point cloud confirms the wall exists)
       CONFLICTED – if it is EMPTY     (laser passed through → likely opening)
  6. conflict_probability = conflicted / (confirmed + conflicted)

Outputs
-------
  outputs/09_visibility/nimbb_voxels.las        – voxel cloud coloured by state
  outputs/09_visibility/nimbb_wall_conflicts.csv – per-wall table
  Interactive Open3D window (close to finish)

Usage
-----
  conda run -n las-env python scripts/LOD2toLOD3/09_visibility_conflict_map.py
"""

import os
import csv
import time
import numpy as np
import laspy
import open3d as o3d
from lxml import etree

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

GML_FILE   = r'outputs/00_gml_wall_merged/nimbb_021126_FIXED_merged.gml'
LAS_FILE   = r'outputs/07_merged_las/NIMBB-2-super-cleaned.las'
OUTPUT_DIR = r'outputs/09_visibility'

VOXEL_SIZE  = 0.30    # metres — voxel edge length
PLANE_TOL   = 0.25    # metres — how close a voxel centre must be to a wall plane
MAX_POINTS  = 5_000_000  # sub-sample if cloud is larger
SENSOR_JUMP = 5.0     # metres — minimum jump in XY to detect a new scan position

# Sensor-origin estimation strategy: 'gps_time' | 'point_source' | 'centroid'
ORIGIN_STRATEGY = 'gps_time'

NAMESPACES = {
    'gml':  'http://www.opengis.net/gml',
    'bldg': 'http://www.opengis.net/citygml/building/2.0',
}

# Voxel state codes
FREE = 0
EMPTY = 1       # ray passed through
OCCUPIED = 2    # hit point

# ─────────────────────────────────────────────────────────────────────────────
# GML parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_wall_surfaces(gml_path: str):
    """Return list of (wall_id, Nx3 coords) for every WallSurface polygon."""
    root = etree.parse(gml_path).getroot()
    walls = []
    for ws in root.xpath('//bldg:WallSurface', namespaces=NAMESPACES):
        for poly in ws.xpath('.//gml:Polygon', namespaces=NAMESPACES):
            pid = poly.get('{http://www.opengis.net/gml}id', 'unknown')
            for pos in poly.xpath('.//gml:posList', namespaces=NAMESPACES):
                text = (pos.text or '').strip()
                if not text:
                    continue
                arr = np.array(text.split(), dtype=np.float64).reshape(-1, 3)
                if len(arr) >= 3:
                    walls.append((pid, arr))
    print(f'  Parsed {len(walls)} WallSurface polygon(s)')
    return walls


# ─────────────────────────────────────────────────────────────────────────────
# LAS loading + sensor-origin estimation
# ─────────────────────────────────────────────────────────────────────────────

def load_las(las_path: str, max_pts: int = MAX_POINTS):
    """Return (xyz array float64, sensor_origins array float64 same shape)."""
    print(f'  Loading {las_path} …')
    las = laspy.read(las_path)
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    N = len(xyz)
    print(f'  {N:,} points loaded')

    # Sub-sample if necessary
    if N > max_pts:
        idx = np.random.choice(N, max_pts, replace=False)
        idx.sort()
        xyz = xyz[idx]
        print(f'  Sub-sampled to {len(xyz):,} points')
    else:
        idx = np.arange(N)

    # ── Sensor origin estimation ──────────────────────────────────────────────
    origins = _estimate_origins(las, idx, xyz)
    return xyz, origins


def _estimate_origins(las, idx, xyz):
    """
    Estimate a per-point sensor origin.

    Strategy 1 (gps_time): group points by GPS-time bin → centroid of each
      group shifted UP (scanner was above ground, so we shift +2 m in Z).
    Strategy 2 (point_source): group by point-source-id.
    Strategy 3 (centroid): single global origin at cloud centroid + 2 m Z.
    """
    N = len(xyz)
    origins = np.empty_like(xyz)

    # Strategy: GPS time grouping ─────────────────────────────────────────────
    if ORIGIN_STRATEGY == 'gps_time' and hasattr(las, 'gps_time'):
        raw_gps = np.asarray(las.gps_time, dtype=np.float64)[idx]
        # Bin into 0.5-second windows → each bin ≈ one scan pass segment
        BIN = 0.5
        bins = (raw_gps // BIN).astype(np.int64)
        unique_bins = np.unique(bins)
        print(f'  GPS-time strategy: {len(unique_bins)} scan-position bins')
        for b in unique_bins:
            mask = bins == b
            # Sensor origin: XY centroid of the bin, Z = min(Z in bin) - 1.5 m
            # (scanner sits above road, points hit building walls)
            grp = xyz[mask]
            cx, cy = grp[:, 0].mean(), grp[:, 1].mean()
            cz = grp[:, 2].min() - 1.5   # scanner below lowest point in bin
            origins[mask] = [cx, cy, cz]
        return origins

    # Strategy: point source id ───────────────────────────────────────────────
    if ORIGIN_STRATEGY == 'point_source' and hasattr(las, 'point_source_id'):
        psi = np.asarray(las.point_source_id, dtype=np.int32)[idx]
        unique_ids = np.unique(psi)
        print(f'  Point-source-id strategy: {len(unique_ids)} unique source IDs')
        for sid in unique_ids:
            mask = psi == sid
            grp = xyz[mask]
            cx, cy = grp[:, 0].mean(), grp[:, 1].mean()
            cz = grp[:, 2].min() - 1.5
            origins[mask] = [cx, cy, cz]
        return origins

    # Fallback: single centroid ───────────────────────────────────────────────
    print('  Centroid fallback: single sensor origin for all points')
    cx, cy = xyz[:, 0].mean(), xyz[:, 1].mean()
    cz = xyz[:, 2].min() - 1.5
    origins[:] = [cx, cy, cz]
    return origins


# ─────────────────────────────────────────────────────────────────────────────
# Voxel grid
# ─────────────────────────────────────────────────────────────────────────────

class VoxelGrid:
    """
    3-D voxel grid backed by a numpy uint8 array.
    Voxel state: FREE(0), EMPTY(1), OCCUPIED(2).
    OCCUPIED overwrites EMPTY but not vice-versa.
    """

    def __init__(self, xyz_all: np.ndarray, walls: list, voxel_size: float):
        # Scene bounding box: union of cloud + wall vertices
        all_pts = [xyz_all]
        for _, coords in walls:
            all_pts.append(coords)
        all_pts = np.vstack(all_pts)

        pad = voxel_size * 2
        self.origin = all_pts.min(axis=0) - pad
        upper = all_pts.max(axis=0) + pad
        self.size = voxel_size
        dims = np.ceil((upper - self.origin) / voxel_size).astype(int) + 1
        self.dims = dims
        self.grid = np.zeros(dims, dtype=np.uint8)  # FREE everywhere
        print(f'  Voxel grid: {dims[0]}×{dims[1]}×{dims[2]} = {dims.prod():,} voxels  '
              f'({dims.prod() * 1e-6:.1f}M)  at {voxel_size} m resolution')

    def _to_vox(self, pts: np.ndarray) -> np.ndarray:
        """Convert world XYZ(s) to integer voxel indices (Nx3)."""
        return np.floor((pts - self.origin) / self.size).astype(np.int32)

    def _vox_centre(self, ijk: np.ndarray) -> np.ndarray:
        """Voxel-index → world centre (Nx3)."""
        return self.origin + (ijk + 0.5) * self.size

    def _in_bounds(self, ijk: np.ndarray) -> np.ndarray:
        return (np.all(ijk >= 0, axis=-1) &
                np.all(ijk < self.dims, axis=-1))

    # -- Ray insertion --------------------------------------------------------

    def insert_rays(self, xyz: np.ndarray, origins: np.ndarray,
                    batch: int = 50_000):
        """
        Vectorised DDA ray casting.
        For each point [p] with sensor origin [o]:
          - Walk the ray o→p and mark traversed voxels as EMPTY.
          - Mark the terminal voxel as OCCUPIED.
        """
        N = len(xyz)
        print(f'  Ray casting {N:,} rays …')
        t0 = time.time()
        n_batches = (N + batch - 1) // batch

        for b in range(n_batches):
            sl = slice(b * batch, (b + 1) * batch)
            _pts = xyz[sl]
            _ori = origins[sl]
            self._cast_batch(_pts, _ori)
            if b % 10 == 0 or b == n_batches - 1:
                elapsed = time.time() - t0
                pct = 100 * (b + 1) / n_batches
                print(f'    {pct:5.1f}%  ({elapsed:.1f}s)', end='\r')

        print(f'\n  Ray casting done in {time.time()-t0:.1f}s')

    def _cast_batch(self, pts: np.ndarray, origins: np.ndarray):
        """DDA walk for a batch of rays (fully numpy, no Python loop per ray)."""
        # Voxel indices of start (origin) and end (hit point)
        i_start = self._to_vox(origins)   # (B,3)
        i_end   = self._to_vox(pts)       # (B,3)

        # Mark occupied (hit) voxels
        valid_end = self._in_bounds(i_end)
        if valid_end.any():
            ie = i_end[valid_end]
            # Only upgrade FREE/EMPTY → OCCUPIED
            self.grid[ie[:, 0], ie[:, 1], ie[:, 2]] = OCCUPIED

        # DDA traversal — step towards end voxel
        # We iterate STEP times along each dimension in discrete jumps.
        # For memory efficiency we do this dimension by dimension using
        # Bresenham's approach.
        B = len(pts)
        diff = i_end - i_start   # (B,3)
        steps = np.abs(diff).max(axis=1)  # (B,) — number of steps per ray

        # Avoid division by zero for zero-length rays
        nonzero = steps > 0
        if not nonzero.any():
            return

        diff_f  = diff[nonzero].astype(np.float32)
        start_f = i_start[nonzero].astype(np.float32)
        smax    = steps[nonzero].astype(np.float32)[:, None]

        # Parametric: for step k = 0..S-1 (exclude endpoint → already OCCUPIED)
        # Generate all intermediate voxel positions at once (can be memory heavy
        # for long rays; we cap at 200 steps per ray, then skip beyond)
        MAX_STEPS = 200
        S = int(smax.max())
        if S > MAX_STEPS:
            S = MAX_STEPS

        for k in range(1, S):  # skip k=0 (origin itself) and k=S (endpoint)
            t = k / smax  # (B',1) — fraction along each ray
            t = np.clip(t, 0, 1)
            # Only process rays that still have k < their own step count
            active = (k < smax[:, 0])
            if not active.any():
                break
            vox = np.floor(start_f[active] + t[active] * diff_f[active]
                           ).astype(np.int32)
            # Filter in-bounds
            ib = self._in_bounds(vox)
            if not ib.any():
                continue
            vox = vox[ib]
            # Mark as EMPTY only if currently FREE (don't overwrite OCCUPIED)
            cur = self.grid[vox[:, 0], vox[:, 1], vox[:, 2]]
            mark = cur == FREE
            if mark.any():
                v = vox[mark]
                self.grid[v[:, 0], v[:, 1], v[:, 2]] = EMPTY

    # -- Wall intersection ----------------------------------------------------

    def analyse_wall(self, wall_id: str, coords: np.ndarray):
        """
        Find voxels intersecting this wall polygon, classify as
        CONFIRMED or CONFLICTED, return (confirmed_count, conflicted_count).
        """
        # Compute plane normal & d from first 3 non-collinear vertices
        v0, v1, v2 = coords[0], coords[1], coords[2]
        n = np.cross(v1 - v0, v2 - v0)
        n_len = np.linalg.norm(n)
        if n_len < 1e-9:
            return 0, 0
        n = n / n_len
        d = -np.dot(n, v0)   # plane: n·x + d = 0

        # Bounding box of polygon with a buffer
        buf = self.size
        bb_min = coords.min(axis=0) - buf
        bb_max = coords.max(axis=0) + buf

        i_min = np.clip(self._to_vox(bb_min), 0, self.dims - 1)
        i_max = np.clip(self._to_vox(bb_max), 0, self.dims - 1)

        # Enumerate candidate voxels in bounding box
        ix = np.arange(i_min[0], i_max[0] + 1, dtype=np.int32)
        iy = np.arange(i_min[1], i_max[1] + 1, dtype=np.int32)
        iz = np.arange(i_min[2], i_max[2] + 1, dtype=np.int32)
        gi, gj, gk = np.meshgrid(ix, iy, iz, indexing='ij')
        all_ijk = np.column_stack([gi.ravel(), gj.ravel(), gk.ravel()])

        centres = self._vox_centre(all_ijk)   # world positions

        # Filter 1: distance to plane < PLANE_TOL
        dist = np.abs(centres @ n + d)
        close = dist < PLANE_TOL
        if not close.any():
            return 0, 0

        centres_c = centres[close]
        ijk_c     = all_ijk[close]

        # Filter 2: inside polygon footprint (2-D point-in-polygon test)
        # Project onto the plane's 2-D coordinate system
        inside = _points_in_polygon(centres_c, coords, n)
        if not inside.any():
            return 0, 0

        ijk_in = ijk_c[inside]

        states = self.grid[ijk_in[:, 0], ijk_in[:, 1], ijk_in[:, 2]]
        confirmed  = int((states == OCCUPIED).sum())
        conflicted = int((states == EMPTY).sum())
        return confirmed, conflicted

    # -- Export ---------------------------------------------------------------

    def export_voxel_cloud(self, out_path: str):
        """Save non-FREE voxels as a coloured LAS point cloud."""
        ijk = np.argwhere(self.grid > FREE)
        if len(ijk) == 0:
            print('  No non-free voxels to export.')
            return
        centres = self._vox_centre(ijk)
        states  = self.grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]

        # Colour map: EMPTY=blue, OCCUPIED=green
        r = np.where(states == OCCUPIED, 0,   0  ).astype(np.uint16)
        g = np.where(states == OCCUPIED, 65535, 0 ).astype(np.uint16)
        b = np.where(states == EMPTY,   65535, 0 ).astype(np.uint16)

        header = laspy.LasHeader(point_format=2, version='1.4')
        header.scales  = [0.001, 0.001, 0.001]
        header.offsets = [np.floor(centres[:, 0].min()),
                          np.floor(centres[:, 1].min()),
                          np.floor(centres[:, 2].min())]
        las = laspy.LasData(header)
        las.x = centres[:, 0]
        las.y = centres[:, 1]
        las.z = centres[:, 2]
        las.red   = r
        las.green = g
        las.blue  = b
        las.write(out_path)
        print(f'  ✓ Voxel cloud saved: {out_path}  ({len(centres):,} voxels)')

    def get_conflict_voxel_colours(self, wall_results: list):
        """
        Return (centres, colours) for confirmed/conflicted voxels
        coloured by wall conflict probability (for Open3D visualisation).
        Yellow = confirmed, Red = conflicted.
        """
        all_c, all_col = [], []
        for wall_id, coords, confirmed, conflicted in wall_results:
            if confirmed + conflicted == 0:
                continue
            # Repeat plane-intersection logic to get the specific voxel ijk list
            v0, v1, v2 = coords[0], coords[1], coords[2]
            n = np.cross(v1 - v0, v2 - v0)
            n_len = np.linalg.norm(n)
            if n_len < 1e-9:
                continue
            n = n / n_len
            d = -np.dot(n, v0)
            buf = self.size
            bb_min = coords.min(axis=0) - buf
            bb_max = coords.max(axis=0) + buf
            i_min = np.clip(self._to_vox(bb_min), 0, self.dims - 1)
            i_max = np.clip(self._to_vox(bb_max), 0, self.dims - 1)
            ix = np.arange(i_min[0], i_max[0] + 1, dtype=np.int32)
            iy = np.arange(i_min[1], i_max[1] + 1, dtype=np.int32)
            iz = np.arange(i_min[2], i_max[2] + 1, dtype=np.int32)
            gi, gj, gk = np.meshgrid(ix, iy, iz, indexing='ij')
            all_ijk = np.column_stack([gi.ravel(), gj.ravel(), gk.ravel()])
            centres = self._vox_centre(all_ijk)
            dist = np.abs(centres @ n + d)
            close = dist < PLANE_TOL
            if not close.any():
                continue
            centres_c = centres[close]
            ijk_c = all_ijk[close]
            inside = _points_in_polygon(centres_c, coords, n)
            if not inside.any():
                continue
            ijk_in = ijk_c[inside]
            centres_in = centres_c[inside]
            states = self.grid[ijk_in[:, 0], ijk_in[:, 1], ijk_in[:, 2]]
            for i, s in enumerate(states):
                if s == OCCUPIED:
                    all_c.append(centres_in[i])
                    all_col.append([1.0, 0.8, 0.0])  # yellow = confirmed
                elif s == EMPTY:
                    all_c.append(centres_in[i])
                    all_col.append([1.0, 0.1, 0.1])  # red = conflicted
        if not all_c:
            return None, None
        return np.array(all_c), np.array(all_col)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _points_in_polygon(pts: np.ndarray, poly: np.ndarray,
                       normal: np.ndarray) -> np.ndarray:
    """
    2-D point-in-polygon test projected onto the polygon's plane.
    Returns boolean mask of shape (len(pts),).
    """
    # Build local 2-D axes on the plane
    v0 = poly[0]
    ax_u = poly[1] - poly[0]
    ax_u_len = np.linalg.norm(ax_u)
    if ax_u_len < 1e-9:
        return np.zeros(len(pts), dtype=bool)
    ax_u = ax_u / ax_u_len
    ax_v = np.cross(normal, ax_u)

    # Project polygon vertices to 2-D
    poly_2d = np.column_stack([
        np.dot(poly - v0, ax_u),
        np.dot(poly - v0, ax_v),
    ])
    # Project query points to 2-D
    pts_2d = np.column_stack([
        np.dot(pts - v0, ax_u),
        np.dot(pts - v0, ax_v),
    ])

    return _pnpoly_batch(pts_2d, poly_2d)


def _pnpoly_batch(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """
    Ray-casting point-in-polygon for multiple points simultaneously.
    pts  : (N,2)
    poly : (M,2)
    Returns boolean mask (N,).
    """
    N = len(pts)
    M = len(poly)
    inside = np.zeros(N, dtype=bool)
    j = M - 1
    for i in range(M):
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]
        # Test crossing for all query points at once
        cond = ((yi > pts[:, 1]) != (yj > pts[:, 1])) & \
               (pts[:, 0] < (xj - xi) * (pts[:, 1] - yi) / ((yj - yi) + 1e-12) + xi)
        inside ^= cond
        j = i
    return inside


def wall_to_mesh(coords: np.ndarray):
    """Convert a polygon to an Open3D TriangleMesh (fan triangulation)."""
    n = len(coords)
    if n < 3:
        return None
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(coords.astype(np.float64))
    tris = [[0, i, i + 1] for i in range(1, n - 1)]
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tris, dtype=np.int32))
    mesh.compute_vertex_normals()
    return mesh


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise(wall_polys, vgrid, wall_results, xyz_sample):
    """Interactive Open3D visualisation."""
    geoms = []

    # 1. Point cloud (original, grey, thin)
    pcd_orig = o3d.geometry.PointCloud()
    # Use at most 500k points for display
    n_disp = min(len(xyz_sample), 500_000)
    idx = np.random.choice(len(xyz_sample), n_disp, replace=False)
    pcd_orig.points = o3d.utility.Vector3dVector(xyz_sample[idx])
    pcd_orig.colors = o3d.utility.Vector3dVector(
        np.tile([0.55, 0.55, 0.55], (n_disp, 1)))
    geoms.append(pcd_orig)

    # 2. GML wall meshes (semi-transparent light blue)
    for wall_id, coords in wall_polys:
        m = wall_to_mesh(coords)
        if m is None:
            continue
        m.paint_uniform_color([0.4, 0.6, 0.9])
        geoms.append(m)

    # 3. Conflict-classified voxels on wall surfaces
    centres, colours = vgrid.get_conflict_voxel_colours(wall_results)
    if centres is not None:
        cpcd = o3d.geometry.PointCloud()
        cpcd.points = o3d.utility.Vector3dVector(centres)
        cpcd.colors = o3d.utility.Vector3dVector(colours)
        geoms.append(cpcd)

    # 4. Coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    bb_min = xyz_sample.min(axis=0)
    frame.translate(bb_min)
    geoms.append(frame)

    print('\nOpening Open3D visualiser …')
    print('  Grey points  = original point cloud')
    print('  Blue mesh    = GML wall surfaces')
    print('  Yellow dots  = CONFIRMED voxels (wall backed by points)')
    print('  Red dots     = CONFLICTED voxels (ray passed through → opening)')
    print('  Close the window to exit.\n')
    o3d.visualization.draw_geometries(
        geoms,
        window_name='Scan2LoD3 – Visibility Analysis',
        width=1280, height=800,
        point_show_normal=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=' * 65)
    print('  Scan2LoD3 – Visibility & Conflict Probability Mapping')
    print('=' * 65)

    # 1. Parse GML ────────────────────────────────────────────────────────────
    print(f'\n[1/5] Parsing GML: {GML_FILE}')
    wall_polys = parse_wall_surfaces(GML_FILE)
    if not wall_polys:
        print('  ERROR: No WallSurface polygons found. Exiting.')
        return

    # 2. Load LAS ─────────────────────────────────────────────────────────────
    print(f'\n[2/5] Loading LAS: {LAS_FILE}')
    xyz, origins = load_las(LAS_FILE)

    # 3. Build voxel grid & ray cast ─────────────────────────────────────────
    print(f'\n[3/5] Building voxel grid (resolution = {VOXEL_SIZE} m) …')
    vgrid = VoxelGrid(xyz, wall_polys, VOXEL_SIZE)
    vgrid.insert_rays(xyz, origins)

    occupied = int((vgrid.grid == OCCUPIED).sum())
    empty    = int((vgrid.grid == EMPTY   ).sum())
    print(f'  Occupied voxels : {occupied:,}')
    print(f'  Empty voxels    : {empty:,}')

    # 4. Analyse each wall ────────────────────────────────────────────────────
    print(f'\n[4/5] Analysing {len(wall_polys)} wall polygon(s) …')
    wall_results = []
    for wall_id, coords in wall_polys:
        conf, conf_empty = vgrid.analyse_wall(wall_id, coords)
        total = conf + conf_empty
        prob  = conf_empty / total if total > 0 else 0.0
        wall_results.append((wall_id, coords, conf, conf_empty))
        print(f'  {wall_id[:40]:40s}  '
              f'conf={conf:4d}  confl={conf_empty:4d}  '
              f'conflict_prob={prob:.3f}')

    # 5. Export ───────────────────────────────────────────────────────────────
    print(f'\n[5/5] Exporting results to {OUTPUT_DIR}')

    # CSV
    csv_path = os.path.join(OUTPUT_DIR, 'nimbb_wall_conflicts.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['wall_id', 'confirmed_voxels',
                    'conflicted_voxels', 'conflict_probability'])
        for wall_id, coords, conf, conf_empty in wall_results:
            total = conf + conf_empty
            prob  = conf_empty / total if total > 0 else 0.0
            w.writerow([wall_id, conf, conf_empty, f'{prob:.4f}'])
    print(f'  ✓ CSV saved: {csv_path}')

    # Voxel cloud LAS
    las_out = os.path.join(OUTPUT_DIR, 'nimbb_voxels.las')
    vgrid.export_voxel_cloud(las_out)

    # Summary
    print('\n' + '─' * 65)
    print('  SUMMARY – Wall Conflict Probabilities')
    print('─' * 65)
    print(f'  {"Wall ID":<42} {"Conf":>5} {"Confl":>5} {"Prob":>6}')
    print('─' * 65)
    for wall_id, coords, conf, conf_empty in wall_results:
        total = conf + conf_empty
        prob = conf_empty / total if total > 0 else 0.0
        flag = ' ◄ HIGH' if prob > 0.5 else ''
        print(f'  {wall_id[:42]:<42} {conf:>5} {conf_empty:>5} {prob:>6.3f}{flag}')
    print('─' * 65)
    high = [(wid, conf_empty/(conf+conf_empty))
            for wid, _, conf, conf_empty in wall_results
            if conf + conf_empty > 0 and conf_empty/(conf+conf_empty) > 0.5]
    print(f'\n  Walls with conflict_prob > 0.5 (likely openings): {len(high)}')

    # 6. Visualise ────────────────────────────────────────────────────────────
    visualise(wall_polys, vgrid, wall_results, xyz)


if __name__ == '__main__':
    main()
