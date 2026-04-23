#!/usr/bin/env python3
"""
Unsupervised Point Cloud Segmentation using HSV Color Values.

Segments point clouds by clustering in HSV color space using KMeans.
Each cluster is assigned a distinct color in the output LAS file.

Usage:
    conda activate lidar-test
    python scripts/LOD2toLOD3/06_segment_hsv_color.py \
        --input outputs/07_merged_las/NIMBB-2-super-cleaned.las \
        --output_dir outputs/08_segmented \
        --n_clusters 5

Output: outputs/08_segmented/color_segmented_<input_filename>.las
"""

import os
import argparse
import colorsys

import numpy as np
import laspy
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


# =====================================================================
# Distinct colors for cluster visualization (up to 20 clusters)
# =====================================================================
def generate_distinct_colors(n):
    """Generate n visually distinct RGB colors using HSV spacing."""
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        colors.append([int(c * 255) for c in rgb])
    return colors


# =====================================================================
# RGB to HSV conversion
# =====================================================================
def rgb_to_hsv_array(rgb):
    """
    Convert RGB array (N, 3) with values [0, 1] to HSV array (N, 3).

    H in [0, 360], S in [0, 1], V in [0, 1].
    """
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    s = np.where(maxc > 0, (maxc - minc) / (maxc + 1e-10), 0.0)
    delta = maxc - minc + 1e-10

    # Hue calculation
    h = np.zeros_like(r)
    mask_r = (maxc == r) & (delta > 1e-9)
    mask_g = (maxc == g) & (delta > 1e-9) & ~mask_r
    mask_b = (maxc == b) & (delta > 1e-9) & ~mask_r & ~mask_g

    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    return np.column_stack([h, s, v]).astype(np.float32)


# =====================================================================
# Main segmentation pipeline
# =====================================================================
def segment_by_hsv(file_path, output_dir, n_clusters=5, hsv_weights=(2.0, 2.0, 0.3),
                   use_minibatch=True, random_state=42):
    """
    Segment a point cloud by clustering HSV color values.

    Args:
        file_path: Path to input LAS file.
        output_dir: Output directory.
        n_clusters: Number of clusters for KMeans.
        hsv_weights: Relative weights for (H, S, V) channels.
                     Higher weight = more influence on clustering.
        use_minibatch: Use MiniBatchKMeans for large point clouds.
        random_state: Random seed for reproducibility.
    """
    filename = os.path.basename(file_path)
    name_no_ext = os.path.splitext(filename)[0]
    print(f"\nProcessing: {filename}")

    # Load LAS
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    n_points = len(points)
    print(f"  Points: {n_points:,}")

    # Extract RGB
    if not (hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue')):
        print("  ERROR: No RGB color data found in LAS file. Cannot perform HSV segmentation.")
        return

    red = np.array(las.red, dtype=np.float32)
    green = np.array(las.green, dtype=np.float32)
    blue = np.array(las.blue, dtype=np.float32)

    # Normalize to [0, 1]
    max_val = max(red.max(), green.max(), blue.max(), 1.0)
    if max_val > 255:
        red /= 65535.0
        green /= 65535.0
        blue /= 65535.0
    else:
        red /= 255.0
        green /= 255.0
        blue /= 255.0

    rgb = np.column_stack([red, green, blue]).astype(np.float32)
    print(f"  RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")

    # Convert to HSV
    print(f"  Converting RGB to HSV...")
    hsv = rgb_to_hsv_array(rgb)
    print(f"  H range: [{hsv[:, 0].min():.1f}, {hsv[:, 0].max():.1f}]")
    print(f"  S range: [{hsv[:, 1].min():.3f}, {hsv[:, 1].max():.3f}]")
    print(f"  V range: [{hsv[:, 2].min():.3f}, {hsv[:, 2].max():.3f}]")

    # Normalize H to [0, 1] range for clustering
    hsv_norm = hsv.copy()
    hsv_norm[:, 0] /= 360.0

    # Standardize features FIRST
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(hsv_norm)

    # THEN apply channel weights
    w_h, w_s, w_v = hsv_weights
    features = features_scaled.copy()
    features[:, 0] *= w_h
    features[:, 1] *= w_s
    features[:, 2] *= w_v

    # KMeans clustering
    print(f"  Clustering into {n_clusters} segments (HSV weights: H={w_h}, S={w_s}, V={w_v})...")
    if use_minibatch and n_points > 100000:
        print(f"  Using MiniBatchKMeans (large point cloud)")
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=min(10000, n_points),
            n_init=3,
            max_iter=300)
    else:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300)

    labels = kmeans.fit_predict(features)
    print(f"  Clustering complete.")

    # Print cluster statistics
    print(f"\n  Cluster statistics:")
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Sort clusters by size (largest first)
    sort_idx = np.argsort(-counts)
    cluster_colors = generate_distinct_colors(n_clusters)

    # Compute mean HSV per cluster for labeling
    for rank, idx in enumerate(sort_idx):
        label = unique_labels[idx]
        count = counts[idx]
        pct = 100.0 * count / n_points
        mask = labels == label
        mean_h = hsv[mask, 0].mean()
        mean_s = hsv[mask, 1].mean()
        mean_v = hsv[mask, 2].mean()
        color = cluster_colors[label]
        print(f"    Cluster {label:2d}: {count:>10,} pts ({pct:5.1f}%)  "
              f"mean HSV=({mean_h:5.1f}°, {mean_s:.2f}, {mean_v:.2f})  "
              f"color=RGB({color[0]},{color[1]},{color[2]})")

    # Assign cluster colors
    colors_out = np.zeros((n_points, 3), dtype=np.uint16)
    for label in unique_labels:
        mask = labels == label
        rgb_color = cluster_colors[label]
        # Scale to 16-bit for LAS
        colors_out[mask, 0] = int(rgb_color[0]) * 256
        colors_out[mask, 1] = int(rgb_color[1]) * 256
        colors_out[mask, 2] = int(rgb_color[2]) * 256

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    out_name = f"color_segmented_{name_no_ext}.las"
    output_path = os.path.join(output_dir, out_name)

    header = laspy.LasHeader(point_format=2, version="1.2")
    header.scales = las.header.scales
    header.offsets = las.header.offsets

    new_las = laspy.LasData(header)
    new_las.x = las.x
    new_las.y = las.y
    new_las.z = las.z
    new_las.red = colors_out[:, 0]
    new_las.green = colors_out[:, 1]
    new_las.blue = colors_out[:, 2]
    new_las.classification = labels.astype(np.uint8)

    new_las.write(output_path)
    print(f"\n  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Unsupervised point cloud segmentation using HSV color clustering")
    parser.add_argument("--input", type=str,
                        default="outputs/07_merged_las/ICHEM-2-GOOD.las",
                        help="Input LAS file path")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Input directory of LAS files (overrides --input)")
    parser.add_argument("--output_dir", type=str, default="outputs/08_segmented",
                        help="Output directory")
    parser.add_argument("--n_clusters", type=int, default=5,
                        help="Number of color clusters")
    parser.add_argument("--weight_h", type=float, default=2.0,
                        help="Weight for Hue channel in clustering (default: 2.0)")
    parser.add_argument("--weight_s", type=float, default=3.0,
                        help="Weight for Saturation channel (default: 3.0, higher = more emphasis on color purity)")
    parser.add_argument("--weight_v", type=float, default=0.3,
                        help="Weight for Value/brightness channel (default: 0.3, lower = less sensitive to shadows)")
    args = parser.parse_args()

    hsv_weights = (args.weight_h, args.weight_s, args.weight_v)

    # Find input files
    if args.input_dir:
        import glob
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.las")))
    else:
        files = [args.input]

    if not files:
        print("No .las files found")
        return

    print(f"Found {len(files)} file(s) to process")
    print(f"Clusters: {args.n_clusters}, HSV weights: H={args.weight_h}, S={args.weight_s}, V={args.weight_v}")
    print("=" * 60)

    for file_path in files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        segment_by_hsv(file_path, args.output_dir, n_clusters=args.n_clusters,
                       hsv_weights=hsv_weights)

    print("\n" + "=" * 60)
    print("All files processed!")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
