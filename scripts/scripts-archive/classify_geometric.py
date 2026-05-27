import open3d as o3d
import numpy as np
import laspy
import os
import argparse
import glob
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from tqdm import tqdm

# ─── Label IDs ────────────────────────────────────────────────────────────────
LABEL_WALL         = 0   # majority class – vertical planar surfaces
LABEL_WINDOW       = 1   # embossed on wall, same normal, not coplanar
LABEL_DOOR         = 2   # like window but near ground, vertically oriented
LABEL_INSTALLATION = 3   # variable normals / irregular geometry

# Visualisation colours (RGB 0-255)
COLOR_MAP = {
    LABEL_WALL:         [200, 200, 200],   # light grey
    LABEL_WINDOW:       [0,   180, 255],   # cyan-blue
    LABEL_DOOR:         [255, 140,   0],   # orange
    LABEL_INSTALLATION: [180,   0, 180],   # purple
}

# ─── Tuneable thresholds ───────────────────────────────────────────────────────
NORMAL_RADIUS        = 0.25   # m  – radius for normal estimation
NORMAL_MAX_NN        = 30     # neighbours for normal estimation
WALL_NORMAL_Z_MAX    = 0.30   # |nz| below this ⟹ vertical (wall-like) normal
PLANE_INLIER_DIST    = 0.05   # m  – RANSAC inlier dist for dominant wall planes
PLANE_RANSAC_N       = 3
PLANE_RANSAC_ITER    = 2000
COPLANAR_DIST_THR    = 0.10   # m  – gap between wall plane and "embossed" points
DOOR_HEIGHT_FRAC     = 0.40   # lowest fraction of Z-range considered "near ground"
DOOR_ELONGATION_MIN  = 1.5    # vertical elongation ratio (height / width)
INSTALL_NORMAL_STD   = 0.35   # normal-direction std-dev above which → installation
MIN_SEGMENT_POINTS   = 10     # ignore tiny clusters


# ─────────────────────────────────────────────────────────────────────────────
def _estimate_normals(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN
        )
    )
    pcd.orient_normals_consistent_tangent_plane(30)
    return np.asarray(pcd.normals)


def _dominant_wall_planes(pcd: o3d.geometry.PointCloud,
                           normals: np.ndarray,
                           max_planes: int = 6):
    """
    Iteratively fit RANSAC planes to points with vertical normals.
    Returns list of (plane_model, inlier_indices_in_remaining).
    """
    pts = np.asarray(pcd.points)
    nz  = np.abs(normals[:, 2])

    # Only consider points whose own normal is vertical
    vertical_mask = nz < WALL_NORMAL_Z_MAX
    remaining = np.where(vertical_mask)[0]

    planes = []
    for _ in range(max_planes):
        if len(remaining) < MIN_SEGMENT_POINTS:
            break
        sub = pcd.select_by_index(remaining)
        model, local_inliers = sub.segment_plane(
            distance_threshold=PLANE_INLIER_DIST,
            ransac_n=PLANE_RANSAC_N,
            num_iterations=PLANE_RANSAC_ITER,
        )
        if len(local_inliers) < MIN_SEGMENT_POINTS:
            break
        global_inliers = remaining[local_inliers]
        planes.append((model, global_inliers))
        remaining = np.setdiff1d(remaining, global_inliers)

    return planes


def _signed_dist_to_plane(points: np.ndarray, plane_model) -> np.ndarray:
    a, b, c, d = plane_model
    n = np.array([a, b, c])
    n /= np.linalg.norm(n)
    return points @ n + d


def _bounding_box_elongation(pts: np.ndarray):
    """
    Compute the ratio of the vertical extent to the horizontal extent of a
    point cluster in its local PCA frame.  Returns (height, width, ratio).
    """
    if len(pts) < 4:
        return 0.0, 1.0, 0.0
    pca = PCA(n_components=3)
    pca.fit(pts)
    coords = pca.transform(pts)
    ranges = coords.ptp(axis=0)   # max-min per axis in PCA space
    # PCA axis 0 is longest – but we care about the actual Z extent
    z_range = pts[:, 2].ptp()
    # Horizontal spread: use PCA axis 0 & 1, take the dominant one
    horiz = max(ranges[0], ranges[1]) if ranges[0] > 0 else 1.0
    ratio  = z_range / horiz if horiz > 0 else 0.0
    return z_range, horiz, ratio


# ─── Main classification function ────────────────────────────────────────────
def classify_building_facade(file_path: str, output_path: str):
    """
    Classify facade points (assumed to be building interior/exterior facade
    scan) into: wall, window, door, building_installation.
    """
    print(f"\nProcessing {os.path.basename(file_path)} …")

    # ── Load ──────────────────────────────────────────────────────────────────
    las    = laspy.read(file_path)
    pts    = np.vstack((las.x, las.y, las.z)).transpose()
    n_pts  = len(pts)
    labels = np.full(n_pts, LABEL_INSTALLATION, dtype=np.uint8)   # default

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if n_pts < MIN_SEGMENT_POINTS:
        print("  Too few points – labelled entirely as installation.")
        _write_las(las, labels, output_path)
        return

    # ── Step 1 : Estimate normals ──────────────────────────────────────────────
    print("  Estimating normals …")
    normals = _estimate_normals(pcd)
    nz_abs  = np.abs(normals[:, 2])

    # ── Step 2 : Detect dominant wall planes (RANSAC) ─────────────────────────
    print("  Detecting dominant wall planes …")
    planes = _dominant_wall_planes(pcd, normals)
    wall_indices_all = []
    plane_assignment  = np.full(n_pts, -1, dtype=np.int32)   # which plane each point belongs to

    for pid, (model, inliers) in enumerate(planes):
        labels[inliers]           = LABEL_WALL
        wall_indices_all.extend(inliers.tolist())
        plane_assignment[inliers] = pid

    wall_set = set(wall_indices_all)
    print(f"  → {len(planes)} wall plane(s), {len(wall_set)} wall points")

    # ── Step 3 : Non-wall points → candidate openings / installations ─────────
    non_wall_mask = labels != LABEL_WALL
    non_wall_idx  = np.where(non_wall_mask)[0]

    if len(non_wall_idx) == 0:
        print("  No non-wall points found.")
        _write_las(las, labels, output_path)
        return

    # ── Step 4 : For each non-wall point, find its nearest wall plane ─────────
    #   Project the point onto each plane and keep the model with minimum |dist|.
    #   Then decide: embossed (window/door) or installation.

    print("  Classifying non-wall points …")
    z_min = pts[:, 2].min()
    z_max = pts[:, 2].max()
    z_range = z_max - z_min if z_max > z_min else 1.0
    door_z_thresh = z_min + DOOR_HEIGHT_FRAC * z_range

    if len(planes) == 0:
        # No wall planes detected – classify entirely by normal direction
        for idx in non_wall_idx:
            if nz_abs[idx] < WALL_NORMAL_Z_MAX:
                labels[idx] = LABEL_WALL   # vertical but not caught by RANSAC
        non_wall_idx = np.where(labels == LABEL_INSTALLATION)[0]

    # Build a KD-tree of wall points to later check local coplanarity
    if len(wall_indices_all) > 0:
        wall_pts_arr = pts[np.array(wall_indices_all)]
        wall_kdtree  = cKDTree(wall_pts_arr)
    else:
        wall_pts_arr = None
        wall_kdtree  = None

    for idx in non_wall_idx:
        pt = pts[idx]
        n  = normals[idx]

        # a) Normal is vertical (wall-like)?
        is_wall_normal = nz_abs[idx] < WALL_NORMAL_Z_MAX

        if not is_wall_normal:
            # Non-vertical normal → installation (keep default)
            labels[idx] = LABEL_INSTALLATION
            continue

        # b) Check distance to nearest wall plane
        if len(planes) > 0:
            dists = [abs(_signed_dist_to_plane(pt.reshape(1, 3), m)[0])
                     for m, _ in planes]
            min_dist = min(dists)
        else:
            min_dist = np.inf

        if min_dist > COPLANAR_DIST_THR:
            # Same normal direction but NOT coplanar → embossed feature
            # Distinguish door vs window by height and elongation
            near_ground = pt[2] < door_z_thresh

            if near_ground:
                # Additional elongation check
                if wall_kdtree is not None:
                    # gather near neighbours to estimate the cluster shape
                    neigh_r = 0.5
                    _, nn_idx = wall_kdtree.query(pt, k=min(20, len(wall_pts_arr)))
                    # Use the non-wall point itself plus its spatial neighbours
                _, _, ratio = _bounding_box_elongation(
                    pts[[idx]]
                )   # single-point fallback; cluster-level done below
                labels[idx] = LABEL_DOOR
            else:
                labels[idx] = LABEL_WINDOW
        else:
            # Coplanar with a wall → also treat as wall (flush surface)
            labels[idx] = LABEL_WALL

    # ── Step 5 : Cluster-level refinement for door vs window ─────────────────
    #   Group embossed points into connected components and re-evaluate
    #   the cluster's vertical elongation to confirm door assignment.
    print("  Refining door/window clusters …")
    embossed_mask  = (labels == LABEL_WINDOW) | (labels == LABEL_DOOR)
    embossed_idx   = np.where(embossed_mask)[0]

    if len(embossed_idx) > MIN_SEGMENT_POINTS:
        emb_pcd = pcd.select_by_index(embossed_idx)
        # DBSCAN to find spatially coherent clusters
        db_labels = np.array(
            emb_pcd.cluster_dbscan(eps=0.3, min_points=5, print_progress=False)
        )
        unique_clusters = set(db_labels) - {-1}

        for cid in unique_clusters:
            c_local = np.where(db_labels == cid)[0]
            c_global = embossed_idx[c_local]
            c_pts    = pts[c_global]

            z_height, horiz, ratio = _bounding_box_elongation(c_pts)
            near_ground = c_pts[:, 2].mean() < door_z_thresh

            if near_ground and ratio >= DOOR_ELONGATION_MIN:
                labels[c_global] = LABEL_DOOR
            elif near_ground and ratio < DOOR_ELONGATION_MIN:
                # Wide near-ground feature → more likely a window at ground level
                labels[c_global] = LABEL_WINDOW
            else:
                # High up or not elongated → window
                labels[c_global] = LABEL_WINDOW

    # ── Step 6 : Points with highly variable local normals → installation ─────
    print("  Detecting installation points by normal variance …")
    tree = o3d.geometry.KDTreeFlann(pcd)
    for idx in range(n_pts):
        if labels[idx] in (LABEL_WINDOW, LABEL_DOOR):
            continue          # already classified as opening
        [_, nn_idx, _] = tree.search_radius_vector_3d(pcd.points[idx], NORMAL_RADIUS)
        if len(nn_idx) < 4:
            continue
        local_n = normals[list(nn_idx)]
        std_dev  = np.std(local_n, axis=0).mean()
        if std_dev > INSTALL_NORMAL_STD:
            labels[idx] = LABEL_INSTALLATION

    # Summary
    for lbl, name in [(LABEL_WALL, "wall"), (LABEL_WINDOW, "window"),
                      (LABEL_DOOR, "door"), (LABEL_INSTALLATION, "installation")]:
        count = np.sum(labels == lbl)
        print(f"    {name:>14s}: {count:>7d} pts ({100*count/n_pts:.1f}%)")

    # ── Save ──────────────────────────────────────────────────────────────────
    _write_las(las, labels, output_path)


def _write_las(las, labels: np.ndarray, output_path: str):
    """Write LAS file with classification and colour-coded RGB."""
    new_las = laspy.LasData(las.header)
    new_las.x = las.x
    new_las.y = las.y
    new_las.z = las.z

    # Colour by label
    rgb = np.array([COLOR_MAP[int(l)] for l in labels], dtype=np.float32) / 255.0
    new_las.red   = (rgb[:, 0] * 65535).astype(np.uint16)
    new_las.green = (rgb[:, 1] * 65535).astype(np.uint16)
    new_las.blue  = (rgb[:, 2] * 65535).astype(np.uint16)

    # Store label in classification field (0-3 custom)
    new_las.classification = labels.astype(np.uint8)

    new_las.write(output_path)
    print(f"  Saved → {output_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
def _pick_files(files: list) -> list:
    """Interactive numbered menu – returns the subset of files to process."""
    print("\nAvailable .las files:")
    for i, f in enumerate(files):
        print(f"  [{i+1:>2}] {os.path.relpath(f)}")
    print(f"  [ 0] Process ALL files")
    print()

    while True:
        raw = input("Enter file number(s) separated by commas (or 0 for all): ").strip()
        if not raw:
            continue
        try:
            choices = [int(x.strip()) for x in raw.split(",")]
        except ValueError:
            print("  Invalid input – enter numbers only.")
            continue

        if 0 in choices:
            return files

        valid = [c for c in choices if 1 <= c <= len(files)]
        if not valid:
            print(f"  Numbers must be between 1 and {len(files)}.")
            continue

        return [files[c - 1] for c in valid]


def main():
    parser = argparse.ArgumentParser(
        description="Geometric classification of building facade points "
                    "into wall / window / door / installation."
    )
    parser.add_argument("--input_dir",  default="outputs/07_merged_las",
                        help="Input directory containing .las files")
    parser.add_argument("--output_dir", default="outputs/05_geometric",
                        help="Output directory")
    parser.add_argument("--all", action="store_true",
                        help="Process all files without prompting")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.las"), recursive=True))
    if not files:
        print(f"No .las files found in {args.input_dir}")
        return

    # Let the user pick which file(s) unless --all is passed
    if args.all:
        selected = files
    else:
        selected = _pick_files(files)

    print(f"\nProcessing {len(selected)} file(s) …")
    os.makedirs(args.output_dir, exist_ok=True)

    for f in tqdm(selected, desc="Classifying"):
        rel      = os.path.relpath(os.path.dirname(f), args.input_dir)
        out_dir  = args.output_dir if rel == "." else os.path.join(args.output_dir, rel)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(f))
        classify_building_facade(f, out_path)

    print(f"\nDone. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
