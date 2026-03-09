import numpy as np
import open3d as o3d
import laspy
from pathlib import Path

print("Loading mesh...")
alpha = 10.0
mesh_path = Path("outputs/09_Poisson_surface/NIMM-cleaned-mesh.obj")
mesh_1   = o3d.io.read_triangle_mesh(str(mesh_path))

print("Computing vertex normals...")
mesh_1.compute_vertex_normals()

print("Sampling points...")
pcd = mesh_1.sample_points_poisson_disk(100000)

print("Computing alpha shape...")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
mesh.compute_triangle_normals()

print("Saving mesh...")
out_path = Path("outputs/10_alpha_from_Poisson/NIMBB_alpha.ply")
o3d.io.write_triangle_mesh(str(out_path), mesh, write_vertex_colors=True)
