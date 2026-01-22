import os
import sys
import open3d as o3d
import numpy as np
import time 

start_time = time.time()

def classify_planes_by_normal(pcd, roof_threshold=0.8, wall_threshold=0.5):
    if not pcd.has_normals():
        print(
            "[ERROR] Input point cloud has no normals. Ensure segment_planes.py assigns plane normals."
        )
        return None, None

    normals = np.asarray(pcd.normals)
    # Check for zero normals (unsegmented points from segment_planes.py)
    valid_normals = np.linalg.norm(normals, axis=1) > 1e-6  # Avoid divide-by-zero
    if not np.any(valid_normals):
        print("[ERROR] All normals are zero or invalid.")
        return None, None

    # Normalize only valid normals
    normals_valid = normals[valid_normals].copy()
    normals_valid = normals_valid / np.linalg.norm(normals_valid, axis=1)[:, np.newaxis]
    normals[valid_normals] = normals_valid
    pcd.normals = o3d.utility.Vector3dVector(normals)

    z_values = np.abs(normals[:, 2])
    print(
        "[INFO] Z-component stats for valid normals: min =",
        np.min(z_values[valid_normals]),
        "max =",
        np.max(z_values[valid_normals]),
        "mean =",
        np.mean(z_values[valid_normals]),
    )
    print("[INFO] Valid normals (non-zero):", np.sum(valid_normals))
    print("[INFO] Invalid normals (zero):", len(normals) - np.sum(valid_normals))
    # Log distribution of Z-components
    if np.sum(valid_normals) > 0:
        z_hist, bins = np.histogram(z_values[valid_normals], bins=10, range=(0, 1))
        print("[INFO] Z-component histogram (valid normals):")
        for i in range(len(bins) - 1):
            print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}]: {z_hist[i]} points")

    roof_mask = z_values > roof_threshold
    wall_mask = z_values < wall_threshold
    # Only classify points with valid normals
    roof_mask = roof_mask & valid_normals
    wall_mask = wall_mask & valid_normals

    print(f"[INFO] Number of roof points: {np.sum(roof_mask)}")
    print(f"[INFO] Number of wall points: {np.sum(wall_mask)}")
    print(f"[INFO] Total points: {len(pcd.points)}")
    print(
        f"[INFO] Unclassified points: {len(pcd.points) - np.sum(roof_mask) - np.sum(wall_mask)}"
    )

    roofs = pcd.select_by_index(np.where(roof_mask)[0])
    walls = pcd.select_by_index(np.where(wall_mask)[0])

    # Check for valid points
    roofs_points = np.asarray(roofs.points)
    walls_points = np.asarray(walls.points)
    print(f"[DEBUG] Roofs point cloud: {len(roofs.points)} points")
    print(f"[DEBUG] Walls point cloud: {len(walls.points)} points")
    if len(roofs.points) > 0:
        print(f"[DEBUG] Roofs points valid: {not np.any(np.isnan(roofs_points))}")
    if len(walls.points) > 0:
        print(f"[DEBUG] Walls points valid: {not np.any(np.isnan(walls_points))}")

    # Reset colors to avoid interference
    roofs.colors = o3d.utility.Vector3dVector(np.zeros((len(roofs.points), 3)))
    walls.colors = o3d.utility.Vector3dVector(np.zeros((len(walls.points), 3)))

    roofs.paint_uniform_color([1.0, 0.0, 0.0])  # Red
    walls.paint_uniform_color([0.0, 0.0, 1.0])  # Blue

    return roofs, walls


def main():
    if len(sys.argv) != 4:
        print(
            "[USAGE] python plane_classifier.py <input_ply> <output_roofs_ply> <output_walls_ply>"
        )
        return

    input_path = sys.argv[1]
    roofs_path = sys.argv[2]
    walls_path = sys.argv[3]

    if not os.path.exists(input_path):
        print(f"[ERROR] Input PLY not found: {input_path}")
        return

    print(f"[INFO] Reading: {input_path}")
    pcd = o3d.io.read_point_cloud(input_path)

    # Reset any existing colors
    pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(pcd.points), 3)))

    roofs, walls = classify_planes_by_normal(pcd)
    if roofs is None or walls is None:
        print("[ERROR] Classification failed.")
        return

    o3d.io.write_point_cloud(roofs_path, roofs)
    o3d.io.write_point_cloud(walls_path, walls)

    print("[SUCCESS] Saved:", roofs_path)
    print("[SUCCESS] Saved:", walls_path)

    # === DONE ===
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"=== Done! Plane classification finished in {elapsed:.2f} seconds ===")


    # Visualize both roofs and walls in one window
    print("[INFO] Launching Open3D visualizer for roofs (red) and walls (blue)...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    if len(roofs.points) > 0:
        vis.add_geometry(roofs)
        print("[DEBUG] Added roofs to visualizer")
    else:
        print("[WARNING] No roof points to visualize")
    if len(walls.points) > 0:
        vis.add_geometry(walls)
        print("[DEBUG] Added walls to visualizer")
    else:
        print("[WARNING] No wall points to visualize")

    opt = vis.get_render_option()
    opt.point_show_normal = False
    opt.light_on = False
    opt.point_size = 5.0  # Increase point size for visibility
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
