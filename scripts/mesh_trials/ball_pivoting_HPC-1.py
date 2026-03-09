import numpy as np
import open3d as o3d
import laspy
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Ball Pivoting Mesh Reconstruction")
parser.add_argument("--input", type=str, required=True, help="Path to input LAS file")

print("Loading point cloud...")

radius = 0.1
max_nn = 30

las_path = Path("/home/khalil.torneros/NIMBB-2-curve.las")
las    = laspy.read(str(las_path))
points = np.vstack((las.x, las.y, las.z)).T

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn
        )
    )

print("Computing ball-pivoting...")
radii = [0.5, 0.4, 0.3]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
mesh.compute_vertex_normals()

print("Saving mesh...")
out_path = Path("/home/khalil.torneros/NIMBB-2-curve_bpa1.ply")
o3d.io.write_triangle_mesh(str(out_path), mesh, write_vertex_colors=True)

print("Successful reconstruction")