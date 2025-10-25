import os
import time
import numpy as np
import laspy
import open3d as o3d
import matplotlib.pyplot as plt
import itertools
import sys
from progress.bar import ChargingBar

def las_to_points(las_file):
    las = laspy.read(las_file)
    building_mask = las.classification == 6  # Use only class 6 (building) points
    return np.vstack(
        (las.x[building_mask], las.y[building_mask], las.z[building_mask])
    ).T

def color_segments_by_normals(clusters, normals):
    unique_labels = np.unique(clusters)
    cmap = plt.get_cmap("hsv", len(unique_labels))
    colors = np.zeros((len(clusters), 3))
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise
        colors[clusters == label] = cmap(label)[:3]
    print("Assigned colors (sample):", colors[:5])
    print("Unique labels:", unique_labels)
    return colors

def merge_similar_planes(
    segments_dict, normal_dict, distance_threshold=1.0, angle_deg_threshold=5.0
):
    merged_segments = []
    used = set()
    keys = list(segments_dict.keys())

    for i, j in itertools.combinations(keys, 2):
        if i in used or j in used:
            continue

        n1 = normal_dict[i]
        n2 = normal_dict[j]

        angle_rad = np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        if angle_deg < angle_deg_threshold:
            seg1 = np.asarray(segments_dict[i].points)
            seg2 = np.asarray(segments_dict[j].points)

            centroid1 = np.mean(seg1, axis=0)
            centroid2 = np.mean(seg2, axis=0)
            dist = np.linalg.norm(centroid1 - centroid2)

            if dist < distance_threshold:
                merged_points = np.vstack((seg1, seg2))
                merged_pcd = o3d.geometry.PointCloud()
                merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
                # Assign normal of larger segment
                if len(seg1) > len(seg2):
                    merged_normal = normal_dict[i]
                else:
                    merged_normal = normal_dict[j]
                merged_normals = np.tile(merged_normal, (len(merged_points), 1))
                merged_pcd.normals = o3d.utility.Vector3dVector(merged_normals)
                merged_pcd.paint_uniform_color([1.0, 0.0, 1.0])
                merged_segments.append(merged_pcd)
                used.add(i)
                used.add(j)

    for k in keys:
        if k not in used:
            merged_segments.append(segments_dict[k])

    return merged_segments

def segment_planes(point_sets):
    segmented_geometries = []
    all_points = []
    all_colors = []
    all_normals = []  # Store plane normals

    for idx, pts in enumerate(point_sets):
        print(f"[INFO] Segmenting building {idx} with {len(pts)} points...")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30),
            fast_normal_computation=True,
        )

        segments = {}
        segment_models = {}
        rest = pcd
        max_plane_idx = 30
        d_threshold = 1.0  # Increased to detect more planes

        for i in range(max_plane_idx):
            if len(rest.points) < 50:
                break

            try:
                model, inliers = rest.segment_plane(
                    distance_threshold=d_threshold, ransac_n=3, num_iterations=800
                )
            except Exception as e:
                print(f"[ERROR] Plane segmentation failed at iteration {i}: {e}")
                continue

            if len(inliers) < 30:
                continue

            colors = plt.get_cmap("tab20")(i)
            segment_models[i] = model
            segment = rest.select_by_index(inliers)

            labels = np.array(
                segment.cluster_dbscan(eps=d_threshold * 10, min_points=10)
            )
            if labels.max() < 0:
                continue

            candidates = [np.sum(labels == j) for j in np.unique(labels)]
            best_label = int(np.unique(labels)[np.argmax(candidates)])
            best_cluster = segment.select_by_index(np.where(labels == best_label)[0])
            best_cluster.paint_uniform_color(colors[:3])

            # Assign RANSAC plane normal to all points in the cluster
            plane_normal = np.array(model[:3]) / np.linalg.norm(model[:3])
            segment_normals = np.tile(plane_normal, (len(best_cluster.points), 1))
            best_cluster.normals = o3d.utility.Vector3dVector(segment_normals)

            segments[i] = best_cluster

            all_points.append(np.asarray(best_cluster.points))
            all_colors.append(np.asarray(best_cluster.colors))
            all_normals.append(np.asarray(best_cluster.normals))

            rest = rest.select_by_index(inliers, invert=True) + segment.select_by_index(
                np.where(labels != best_label)[0]
            )
            print(f"[INFO] -> Plane {i + 1}: {len(best_cluster.points)} points")

        print(f"[INFO] -> Remaining unsegmented points: {len(rest.points)}")

        if segment_models:
            normal_dict = {
                k: np.array(m[:3]) / np.linalg.norm(m[:3])
                for k, m in segment_models.items()
            }
            merged_segments = merge_similar_planes(segments, normal_dict)
        else:
            merged_segments = list(segments.values())

        if len(rest.points) > 50:
            labels = np.array(rest.cluster_dbscan(eps=0.5, min_points=5))
            max_label = labels.max()
            colors = plt.get_cmap("tab10")(
                labels / (max_label if max_label > 0 else 1)
            )[:, :3]
            colors[labels < 0] = [0, 0, 0]
            rest.colors = o3d.utility.Vector3dVector(colors)
            # Assign zero normals to unsegmented points
            rest.normals = o3d.utility.Vector3dVector(np.zeros((len(rest.points), 3)))
            merged_segments.extend([rest])
        else:
            merged_segments.extend([])

        segmented_geometries.extend(merged_segments)

    if all_points:
        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)
        merged_normals = np.vstack(all_normals)

        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(merged_points)
        combined_pcd.colors = o3d.utility.Vector3dVector(merged_colors)
        combined_pcd.normals = o3d.utility.Vector3dVector(merged_normals)

        # Log sample normals for roof-like planes (Z-normal near 1)
        roof_mask = np.abs(merged_normals[:, 2]) > 0.9
        roof_normals = merged_normals[roof_mask]
        print(f"[INFO] Sample roof normals (Z > 0.9): {roof_normals[:5]}")

        # Add visualization of the combined point cloud
        print("[INFO] Launching visualization of combined point cloud...")
        vis_combined = o3d.visualization.Visualizer()
        vis_combined.create_window(window_name="Combined Point Cloud Visualization", width=800, height=600)
        vis_combined.add_geometry(combined_pcd)
        opt = vis_combined.get_render_option()
        opt.point_size = 2.0
        opt.light_on = False  # Disable lighting to show raw colors
        vis_combined.run()
        vis_combined.destroy_window()

        return segmented_geometries, combined_pcd
    else:
        return segmented_geometries, None

def main():
    if len(sys.argv) != 3:  # Expect script name, input LAS, and output PLY
        print("[USAGE] python segment_planes.py <input_las_file> <output_ply_file>")
        sys.exit(1)

    las_path = sys.argv[1]
    ply_path = sys.argv[2]

    if not os.path.exists(las_path):
        print(f"[ERROR] LAS file not found: {las_path}")
        sys.exit(1)

    start_time = time.time()

    print("[INFO] Reading LAS file...")
    points = las_to_points(las_path)

    # No GeoJSON filtering, process all building points directly
    print("[INFO] Running region growing segmentation...")
    segmented, combined_pcd = segment_planes([points])  # Single point set

    # Ensure output directory exists
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

    if combined_pcd:
        try:
            o3d.io.write_point_cloud(ply_path, combined_pcd)
            print(f"[SUCCESS] Saved segmented planes PLY: {ply_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save PLY file: {e}")
            sys.exit(1)

    elapsed = time.time() - start_time
    print(f"=== Done! Segmentation finished in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()