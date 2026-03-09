import numpy as np
import open3d as o3d
import laspy
from pathlib import Path

print("Loading point cloud...")

radius = 0.5
max_nn = 30

mesh_path = Path("outputs/09_Poisson_surface/NIMM-cleaned-mesh.obj")
mesh_1   = o3d.io.read_triangle_mesh(str(mesh_path))

print("Computing vertex normals...")
mesh_1.compute_vertex_normals()

print("Sampling points...")
pcd = mesh_1.sample_points_poisson_disk(1000000)

# pcd.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(
#             radius=radius, max_nn=max_nn
#         )
#     )

print("Computing ball-pivoting...")
radii = [0.5, 0.4, 0.3]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
mesh.compute_vertex_normals()

print("Saving mesh...")
out_path = Path("outputs/10_alpha_from_Poisson/NIMBB_ballpivoting.ply")
o3d.io.write_triangle_mesh(str(out_path), mesh, write_vertex_colors=True)

print("Successful reconstruction")