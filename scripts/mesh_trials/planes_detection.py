import numpy as np
import open3d as o3d
import laspy
from pathlib import Path
import argparse
import pyransac3d as pyrsc

radius = 0.1
max_nn = 30

print('Loading...')
las_path = Path("outputs/trials/nimbb-best-normals.las")
las    = laspy.read(str(las_path))
points = np.vstack((las.x, las.y, las.z)).T

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

labels =np.array(pcd.cluster_dbscan(eps=1, min_points=10))
print(labels)

# print('Estimating normals...')
# pcd.estimate_normals(
#     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn)
# )

# print('Detecting planes...')
# oboxes = pcd.detect_planar_patches(
#     normal_variance_threshold_deg=30,
#     coplanarity_deg=60,
#     outlier_ratio=0.75,
#     min_plane_edge_length=0,
#     min_num_points=0,
#     search_param=o3d.geometry.KDTreeSearchParamKNN(knn=5))

# print("Detected {} patches".format(len(oboxes)))

# geometries = []
# for obox in oboxes:
#     mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 1])
#     mesh.paint_uniform_color(obox.color)
#     geometries.append(mesh)

# merged_vertices = []
# merged_triangles = []
    
# current_vertex_count = 0
    
# for mesh in geometries:
#     v = np.asarray(mesh.vertices)
#     t = np.asarray(mesh.triangles)
        
#     merged_vertices.append(v)
#     merged_triangles.append(t + current_vertex_count)
        
#     current_vertex_count += v.shape[0]
#     print(current_vertex_count)
        
# final_vertices = np.vstack(merged_vertices)
# final_triangles = np.vstack(merged_triangles)
    
# merged_mesh = o3d.geometry.TriangleMesh()
# merged_mesh.vertices = o3d.utility.Vector3dVector(final_vertices)
# merged_mesh.triangles = o3d.utility.Vector3iVector(final_triangles)

# merged_mesh.remove_unreferenced_vertices()

# print("Saving mesh...")
# out_path = Path("outputs/trials/NIMBB-planes.ply")
# o3d.io.write_triangle_mesh(str(out_path), merged_mesh, write_vertex_colors=True)

#o3d.visualization.draw_geometries(geometries,
#                                  zoom=0.62,
#                                  front=[0.4361, -0.2632, -0.8605],
#                                  lookat=[2.4947, 1.7728, 1.5541],
#                                  up=[-0.1726, -0.9630, 0.2071])