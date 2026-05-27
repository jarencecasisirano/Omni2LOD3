import laspy, numpy as np, open3d as o3d
from sklearn.cluster import DBSCAN
import alphashape, geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.cm as cm

def main(input_las, output_geojson, eps=1.0, min_samples=20, alpha=0.5):

    las = laspy.read(input_las)
    mask = las.classification == 6
    xyz = np.vstack((las.x[mask], las.y[mask], las.z[mask])).T
    xy = xyz[:, :2]

    # --- cluster ---
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(xy)
    labels = db.labels_

    valid_labels = sorted(set(labels) - {-1})
    if not valid_labels:
        raise RuntimeError("No building clusters found")

    print("Detected clusters:")
    for lbl in valid_labels:
        print(f"  Cluster {lbl}: {(labels == lbl).sum()} points")

    # --- visualize clusters ---
    cmap = cm.get_cmap("tab20", len(valid_labels))
    colors = np.zeros((len(xy), 3))
    for i, lbl in enumerate(valid_labels):
        colors[labels == lbl] = cmap(i)[:3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("Close window, then select cluster ID in terminal")
    o3d.visualization.draw_geometries([pcd])

    # --- user selects cluster ---
    chosen = int(input(f"Select cluster ID {valid_labels}: "))
    if chosen not in valid_labels:
        raise ValueError("Invalid cluster ID")

    clean_xy = xy[labels == chosen]

    # --- footprint ---
    hull = alphashape.alphashape(clean_xy, alpha)
    if not isinstance(hull, Polygon):
        raise RuntimeError("Alpha shape did not return polygon")

    hull = hull.simplify(0.1, preserve_topology=True)

    gdf = gpd.GeoDataFrame(geometry=[hull], crs="EPSG:32651")
    gdf.to_file(output_geojson, driver="GeoJSON")

    print(f"Footprint written: {output_geojson}")

if __name__ == "__main__":
    main(
        r"C:\Projects\Thesis\outputs\normalized\nimmb_hag.las",
        r"C:\Projects\Thesis\outputs\footprint\nimmb_auto.geojson",
        eps=1.0,
        alpha=0.5
    )
