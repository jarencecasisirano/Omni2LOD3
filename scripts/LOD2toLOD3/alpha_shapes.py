import numpy as np
import open3d as o3d
import laspy
from pathlib import Path

print("Loading point cloud...")
alpha = 10.0
las_path = Path("outputs/07_merged_las/NIMBB-2-curve.las")
las    = laspy.read(str(las_path))
points = np.vstack((las.x, las.y, las.z)).T

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

print("Computing alpha shape...")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
mesh.compute_vertex_normals()
print("Saving mesh...")
out_path = Path("outputs/08_alpha_shapes/NIMBB-2-curve.ply")
o3d.io.write_triangle_mesh(str(out_path), mesh, write_vertex_colors=True)
