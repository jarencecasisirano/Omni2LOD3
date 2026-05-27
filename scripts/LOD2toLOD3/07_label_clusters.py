#!/usr/bin/env python3
"""
Interactive Cluster Labeling Tool.

Opens a color-segmented point cloud (from 06_segment_hsv_color.py),
displays it in Open3D, and lets the user assign semantic labels to each
color cluster via the terminal.

Usage:
    conda activate lidar-test
    python scripts/LOD2toLOD3/07_label_clusters.py

Workflow:
    1. Select a point cloud from outputs/07_segmented
    2. The cloud opens in an Open3D window for visual reference
    3. For each color cluster the terminal shows:
         - A clear human-readable color name (e.g. "Bright Red", "Dark Green")
         - Its percentage of total points
       You type a label (Wall / Door / Window / Roof / Other)
    4. The labelled cloud is saved to outputs/08_labelled
"""

import os
import sys
import glob
import colorsys

import numpy as np
import laspy

try:
    import open3d as o3d
except ImportError:
    print("ERROR: open3d is required.  Install with: pip install open3d")
    sys.exit(1)


# =====================================================================
# Classification codes (matching CityGML / LAS conventions)
# =====================================================================
LABEL_MAP = {
    "wall":    2,
    "door":    3,
    "window":  4,
    "roof":    5,
    "ground":  6,
    "other":   1,
}

LABEL_NAMES = {v: k.capitalize() for k, v in LABEL_MAP.items()}


# =====================================================================
# Human-readable color naming
# =====================================================================
def _hue_name(h):
    """Return a hue-sector name for a hue angle in [0, 360)."""
    if h < 15:
        return "Red"
    elif h < 40:
        return "Orange"
    elif h < 70:
        return "Yellow"
    elif h < 160:
        return "Green"
    elif h < 200:
        return "Cyan"
    elif h < 260:
        return "Blue"
    elif h < 290:
        return "Purple"
    elif h < 335:
        return "Pink"
    else:
        return "Red"


def rgb_to_readable_name(r, g, b):
    """
    Convert an 8-bit RGB tuple to a human-readable color name like
    'Bright Red', 'Dark Green', 'Light Pink', 'Gray', etc.
    """
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)
    h_deg = h * 360.0

    # Achromatic
    if s < 0.12:
        if v < 0.25:
            return "Black"
        elif v < 0.55:
            return "Dark Gray"
        elif v < 0.80:
            return "Light Gray"
        else:
            return "White"

    # Chromatic
    hue = _hue_name(h_deg)

    if v < 0.35:
        brightness = "Dark "
    elif v > 0.80 and s < 0.50:
        brightness = "Light "
    elif v > 0.80:
        brightness = "Bright "
    else:
        brightness = ""

    return f"{brightness}{hue}"


def _ansi_color_block(r, g, b, width=3):
    """Return an ANSI-escaped colored block string for terminal display."""
    block = "█" * width
    return f"\033[38;2;{r};{g};{b}m{block}\033[0m"


# =====================================================================
# File selection
# =====================================================================
def select_file(directory):
    """List LAS files in directory and let user pick one."""
    pattern = os.path.join(directory, "*.las")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No .las files found in {directory}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Point clouds in: {directory}")
    print(f"{'='*60}")
    for i, f in enumerate(files):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  [{i+1}] {os.path.basename(f):40s}  ({size_mb:.1f} MB)")
    print()

    while True:
        try:
            choice = input(f"Select file [1-{len(files)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]
        except (ValueError, KeyboardInterrupt):
            pass
        print("  Invalid choice, try again.")


# =====================================================================
# Discover color clusters in the segmented point cloud
# =====================================================================
def discover_clusters(las):
    """
    Find unique RGB colors in the LAS and return cluster info.

    Returns list of dicts:
        {
            'rgb_16bit': (R16, G16, B16),
            'rgb_8bit':  (R8, G8, B8),
            'name':      'Bright Red',
            'mask':      np.array[bool],
            'count':     int,
        }
    sorted by point count descending.
    """
    red = np.array(las.red, dtype=np.uint16)
    green = np.array(las.green, dtype=np.uint16)
    blue = np.array(las.blue, dtype=np.uint16)
    n = len(red)

    # Pack into single int for fast unique finding
    packed = red.astype(np.uint64) << 32 | green.astype(np.uint64) << 16 | blue.astype(np.uint64)
    unique_packed, inverse = np.unique(packed, return_inverse=True)

    clusters = []
    for uid_idx, uid in enumerate(unique_packed):
        uid = int(uid)  # convert numpy scalar to Python int
        r16 = (uid >> 32) & 0xFFFF
        g16 = (uid >> 16) & 0xFFFF
        b16 = uid & 0xFFFF
        r8 = min(r16 // 256, 255)
        g8 = min(g16 // 256, 255)
        b8 = min(b16 // 256, 255)

        mask = inverse == uid_idx
        count = int(mask.sum())

        clusters.append({
            'rgb_16bit': (r16, g16, b16),
            'rgb_8bit': (r8, g8, b8),
            'name': rgb_to_readable_name(r8, g8, b8),
            'mask': mask,
            'count': count,
        })

    # Sort by count descending
    clusters.sort(key=lambda c: c['count'], reverse=True)

    # Disambiguate duplicate names
    name_counts = {}
    for c in clusters:
        base = c['name']
        name_counts[base] = name_counts.get(base, 0) + 1
    # If duplicates, add a numeric suffix
    seen = {}
    for c in clusters:
        base = c['name']
        if name_counts[base] > 1:
            seen[base] = seen.get(base, 0) + 1
            c['name'] = f"{base} #{seen[base]}"

    return clusters


# =====================================================================
# Visualization
# =====================================================================
def show_pointcloud(las):
    """Open the point cloud in an Open3D visualization window."""
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)

    red = np.array(las.red, dtype=np.float64)
    green = np.array(las.green, dtype=np.float64)
    blue = np.array(las.blue, dtype=np.float64)

    max_val = max(red.max(), green.max(), blue.max(), 1.0)
    if max_val > 255:
        red /= 65535.0
        green /= 65535.0
        blue /= 65535.0
    else:
        red /= 255.0
        green /= 255.0
        blue /= 255.0

    colors = np.column_stack([red, green, blue])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("\n  Open3D window opened. Use it as visual reference.")
    print("  Keep it open and return to this terminal to assign labels.\n")

    # Non-blocking draw so the user can interact with terminal
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Segmented Point Cloud — Label Reference", width=1280, height=800)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])

    vis.poll_events()
    vis.update_renderer()

    return vis


# =====================================================================
# Label assignment
# =====================================================================
def assign_labels(clusters, n_points):
    """
    Interactively assign a semantic label to each color cluster.

    Returns an array of classification codes, one per point.
    """
    labels = np.ones(n_points, dtype=np.uint8) * LABEL_MAP["other"]

    valid_keys = list(LABEL_MAP.keys())
    valid_display = " / ".join(f"{k.capitalize()} ({v})" for k, v in LABEL_MAP.items())

    print(f"{'='*60}")
    print("  LABEL ASSIGNMENT")
    print(f"{'='*60}")
    print(f"  Available labels: {valid_display}")
    print(f"  (type the label name, e.g. 'wall', 'door', 'window')")
    print(f"  (press Enter to skip → defaults to 'Other')")
    print(f"{'='*60}\n")

    for i, cluster in enumerate(clusters):
        r, g, b = cluster['rgb_8bit']
        pct = 100.0 * cluster['count'] / n_points
        color_block = _ansi_color_block(r, g, b, width=5)

        print(f"  Cluster {i+1}/{len(clusters)}:  "
              f"{color_block}  {cluster['name']:20s}  "
              f"RGB({r:3d},{g:3d},{b:3d})  "
              f"{cluster['count']:>10,} pts ({pct:5.1f}%)")

        while True:
            answer = input(f"    Label → ").strip().lower()
            if answer == "":
                answer = "other"
            if answer in valid_keys:
                code = LABEL_MAP[answer]
                labels[cluster['mask']] = code
                print(f"    ✓ Assigned: {answer.capitalize()} (code {code})\n")
                break
            else:
                print(f"    Invalid. Choose from: {', '.join(valid_keys)}")

    return labels


# =====================================================================
# Save
# =====================================================================
def save_labelled(las, labels, output_path):
    """Save the point cloud with updated classification codes."""
    header = laspy.LasHeader(point_format=2, version="1.2")
    header.scales = las.header.scales
    header.offsets = las.header.offsets

    new_las = laspy.LasData(header)
    new_las.x = las.x
    new_las.y = las.y
    new_las.z = las.z
    new_las.red = las.red
    new_las.green = las.green
    new_las.blue = las.blue
    new_las.classification = labels

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_las.write(output_path)
    print(f"\n  Saved: {output_path}")


# =====================================================================
# Main
# =====================================================================
def main():
    input_dir = "outputs/07_segmented"
    output_dir = "outputs/08_labelled"

    # 1. Select file
    file_path = select_file(input_dir)
    filename = os.path.basename(file_path)
    print(f"\n  Loading: {filename}")

    # 2. Load
    las = laspy.read(file_path)
    n_points = len(las.x)
    print(f"  Points: {n_points:,}")

    # 3. Discover clusters
    clusters = discover_clusters(las)
    print(f"  Found {len(clusters)} color clusters\n")

    # 4. Show in Open3D
    vis = show_pointcloud(las)

    # 5. Assign labels via terminal
    labels = assign_labels(clusters, n_points)

    # 6. Close Open3D
    vis.destroy_window()

    # Summary
    print(f"\n{'='*60}")
    print("  LABEL SUMMARY")
    print(f"{'='*60}")
    unique, counts = np.unique(labels, return_counts=True)
    for code, count in zip(unique, counts):
        name = LABEL_NAMES.get(code, f"Unknown({code})")
        pct = 100.0 * count / n_points
        print(f"    {name:12s}: {count:>10,} pts ({pct:5.1f}%)")

    # 7. Save
    out_name = filename.replace("color_segmented_", "labelled_")
    if out_name == filename:
        out_name = f"labelled_{filename}"
    output_path = os.path.join(output_dir, out_name)
    save_labelled(las, labels, output_path)

    print(f"\n{'='*60}")
    print("  Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
