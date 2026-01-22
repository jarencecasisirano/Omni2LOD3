import os
import sys
import numpy as np
import laspy
import open3d as o3d

from sklearn.cluster import DBSCAN
import alphashape
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.cm as cm

# ------------------ paths ------------------
ROOT = r"C:\Projects\Thesis"
CLIPPED_DIR = os.path.join(ROOT, "outputs", "clipped")
FOOTPRINT_DIR = os.path.join(ROOT, "outputs", "footprint")
os.makedirs(FOOTPRINT_DIR, exist_ok=True)

# ------------------ helpers ------------------
def select_las_file(folder):
    las_files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".las"))
    if not las_files:
        print("❌ No .las files found in outputs/clipped/")
        sys.exit(1)

    print("\nAvailable LAS files:")
    for i, f in enumerate(las_files):
        print(f"[{i}] {f}")

    choice = input("Select file index: ").strip()
    if not choice.isdigit() or int(choice) not in range(len(las_files)):
        print("❌ Invalid selection")
        sys.exit(1)

    return os.path.join(folder, las_files[int(choice)]), las_files[int(choice)]

# ------------------ main ------------------
def main(eps=1.0, min_samples=20, alpha=0.5):

    las_path, las_name = select_las_file(CLIPPED_DIR)
    print(f"\nProcessing: {las_name}")

    las = laspy.read(las_path)

    # Building class only
    mask = las.classification == 6
    if mask.sum() == 0:
        print("❌ No building points (class 6) found.")
        sys.exit(1)

    xyz = np.vstack((las.x[mask], las.y[mask], las.z[mask])).T
    xy = xyz[:, :2]

    # ------------------ clustering ------------------
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(xy)
    labels = db.labels_

    clusters = sorted(set(labels) - {-1})
    if not clusters:
        print("❌ No valid clusters detected.")
        sys.exit(1)

    print("\nDetected building clusters:")
    for c in clusters:
        print(f"  Cluster {c}: {(labels == c).sum()} points")

    # ------------------ visualization ------------------
    cmap = cm.get_cmap("tab20", len(clusters))
    colors = np.zeros((len(xyz), 3))

    for i, c in enumerate(clusters):
        colors[labels == c] = cmap(i)[:3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("\nClose the viewer.")
    o3d.visualization.draw_geometries([pcd], window_name="Building Clusters")

    # Automatically select largest cluster (ignore noise = -1)
    cluster_sizes = {
        c: np.sum(labels == c)
        for c in clusters
    }

    selected_cluster_id = max(cluster_sizes, key=cluster_sizes.get)

    print(f"\nAuto-selected cluster {selected_cluster_id} "
        f"({cluster_sizes[selected_cluster_id]} points)")
    
    clean_xy = xy[labels == selected_cluster_id]


    # ------------------ footprint extraction ------------------
    hull = alphashape.alphashape(clean_xy, alpha)

    if not isinstance(hull, Polygon):
        print("❌ Alpha shape did not return a single polygon.")
        sys.exit(1)

    hull = hull.simplify(0.1, preserve_topology=True)

    # ------------------ save ------------------
    out_name = os.path.splitext(las_name)[0] + f"_cluster{selected_cluster_id}.geojson"
    out_path = os.path.join(FOOTPRINT_DIR, out_name)

    gdf = gpd.GeoDataFrame(geometry=[hull], crs="EPSG:32651")
    gdf.to_file(out_path, driver="GeoJSON")

    print(f"\n✅ Footprint saved to:\n{out_path}")

# ------------------ run ------------------
if __name__ == "__main__":
    main(
        eps=1.0,      # tweak if clusters merge/split
        min_samples=20,
        alpha=0.5     # ≈ 1.5 × voxel size
    )
