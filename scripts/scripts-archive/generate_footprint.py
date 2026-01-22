#!/usr/bin/env python3
import sys, os, uuid
import laspy, numpy as np, open3d as o3d
from sklearn.cluster import DBSCAN
import alphashape
import geopandas as gpd

from shapely.geometry import Polygon

def main(input_las, output_shp, alpha=0.0, eps=1.0, min_samples=10):
    # ----------  your ORIGINAL code  ----------
    las = laspy.read(input_las)
    building_mask = las.classification == 6
    if np.sum(building_mask) == 0:
        print("No building points found (class 6)."); sys.exit(1)

    xyz = np.vstack((las.x[building_mask], las.y[building_mask], las.z[building_mask])).T
    xy  = xyz[:, :2]; min_z, max_z = xyz[:, 2].min(), xyz[:, 2].max()

    db = DBSCAN(eps=eps, min_samples=int(min_samples)).fit(xy)
    labels = db.labels_
    largest = max(set(labels[labels >= 0]), key=labels.tolist().count, default=None)
    if largest is None: print("No valid cluster."); sys.exit(1)
    clean_xy = xy[labels == largest]
    if len(clean_xy) < 3: print("Insufficient points."); sys.exit(1)

    # ORIGINAL HULL BLOCK (unchanged)
    hull = alphashape.alphashape(clean_xy, alpha=0.5)
    if isinstance(hull, Polygon):
        hull = Polygon(hull.exterior)
        hull = hull.simplify(0.25, preserve_topology=True)
    else: print("Hull is not a single Polygon."); sys.exit(1)

    # ----------  POST-CLEAN (new)  ----------
    # 1. keep only exterior ring   (already done above, but repeat for safety)
    clean_poly = Polygon(hull.exterior)
    # 2. smooth & down-sample vertices (5 cm tolerance)
    clean_poly = clean_poly.simplify(0.05, preserve_topology=True)
    # 3. optional micro-spike snap
    clean_poly = clean_poly.buffer(0.01).buffer(-0.01)
    # use the cleaned polygon from here on
    hull = clean_poly

    # ----------  save (your original logic)  ----------
    counter = 0; base, ext = os.path.splitext(output_shp)
    while os.path.exists(f"{base}_{counter}{ext}"): counter += 1
    final_path = f"{base}_{counter}{ext}"
    gdf = gpd.GeoDataFrame(geometry=[hull], crs="EPSG:32651")
    gdf.to_file(final_path, driver='GeoJSON')
    print(f"Footprint saved to: {final_path}")

    # ----------  visualise (your original)  ----------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (len(xyz), 1)))
    hull_coords = np.array(hull.exterior.coords)[:-1]
    z_base = min_z - 0.5; z_top = z_base + 1.0
    pts_3d, lines = [], []
    for i, (x, y) in enumerate(hull_coords):
        pts_3d.append([x, y, z_base]); pts_3d.append([x, y, z_top])
        lines.append([2*i, 2*i+1])
        lines.append([2*i, 2*(i+1)%len(hull_coords)])
        lines.append([2*i+1, 2*(i+1)%len(hull_coords)+1])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts_3d)
    line_set.lines  = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile([1, 1, 0], (len(lines), 1)))
    print("Visualising… (close window to continue)")
    o3d.visualization.draw_geometries([pcd, line_set], window_name="Building & Footprint")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_footprint.py <input.las> <output.shp> [alpha] [eps] [min_samples]")
        sys.exit(1)
    main(*sys.argv[1:6])