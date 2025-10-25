import laspy
import numpy as np
import open3d as o3d
import os

# Path to the input LAS file
INPUT_LAS = r"D:\Projects\Thesis\outputs\building_classification\mbb_bldg.las"

# Ensure input file exists
if not os.path.exists(INPUT_LAS):
    print(f"Error: Input file not found: {INPUT_LAS}")
    exit(1)

# Step 1: Read the LAS file
print("Reading LAS file...")
las = laspy.read(INPUT_LAS)
points = np.vstack((las.x, las.y, las.z)).T
classification = las.classification

# Step 2: Create point cloud and assign colors
print("Creating point cloud with colors...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
colors = np.zeros((len(points), 3))

for i, cls in enumerate(classification):
    if cls == 6:  # Building points
        colors[i] = [1.0, 0.0, 0.0]  # Red
    elif cls == 2:  # Ground points
        colors[i] = [0.59, 0.29, 0.0]  # Brown
    else:  # Other classes
        colors[i] = [0.5, 0.5, 0.5]  # Gray
pcd.colors = o3d.utility.Vector3dVector(colors)

# Step 3: Visualize the point cloud (NADIR view)
print("Visualizing point cloud (NADIR view)...")
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Building Visualization (NADIR View)", width=800, height=600)
vis.add_geometry(pcd)
view_control = vis.get_view_control()
view_control.set_front([0, 0, -1])  # NADIR view (looking down Z-axis)
view_control.set_up([0, 1, 0])  # Ensure vertical alignment
view_control.set_lookat([np.mean(points[:, 0]), np.mean(points[:, 1]), np.mean(points[:, 2])])
view_control.set_zoom(0.7)
vis.get_render_option().point_size = 1.0
vis.run()
vis.destroy_window()