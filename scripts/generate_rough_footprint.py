import os
import sys
import numpy as np
import laspy
import open3d as o3d

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import alphashape
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.cm as cm
import matplotlib.pyplot as plt

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


def compute_normals_fast(xyz, k=30):
    """Compute normals for point cloud"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )
    normals = np.asarray(pcd.normals)
    return normals


def extract_roof_by_normals(xyz, normal_z_threshold=0.7, visualize=True):
    """
    Extract roof points based on surface normal direction.
    Roof points have normals pointing upward (high Z component).
    
    Parameters:
    - normal_z_threshold: 0.0-1.0, how vertical the normal must be
      - 0.7 = ~45° from vertical (includes sloped roofs)
      - 0.85 = ~30° from vertical (stricter)
      - 0.5 = ~60° from vertical (very inclusive)
    """
    
    print(f"\n🔍 Computing surface normals...")
    normals = compute_normals_fast(xyz, k=30)
    
    # Ensure normals point upward
    normals[:, 2] = np.abs(normals[:, 2])
    
    # Roof points have normals pointing up (high Z component)
    roof_mask = normals[:, 2] > normal_z_threshold
    
    print(f"\n🏠 Roof extraction (normal-based method):")
    print(f"   Normal Z threshold: {normal_z_threshold:.2f}")
    print(f"   Roof points: {roof_mask.sum()} / {len(xyz)} ({roof_mask.sum()/len(xyz)*100:.1f}%)")
    
    if visualize:
        plt.figure(figsize=(10, 4))
        
        # Normal Z distribution
        plt.subplot(1, 2, 1)
        plt.hist(normals[:, 2], bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(normal_z_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold = {normal_z_threshold}')
        plt.xlabel('Normal Z component')
        plt.ylabel('Number of points')
        plt.title('Surface Normal Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Height distribution
        plt.subplot(1, 2, 2)
        z_roof = xyz[roof_mask, 2]
        z_all = xyz[:, 2]
        plt.hist(z_all, bins=50, alpha=0.5, label='All points', edgecolor='black')
        plt.hist(z_roof, bins=50, alpha=0.7, label='Roof points', edgecolor='black')
        plt.xlabel('Height (m)')
        plt.ylabel('Number of points')
        plt.title('Height Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return xyz[roof_mask], roof_mask


def extract_roof_adaptive(xyz, height_percentile=0.3, normal_z_threshold=0.6, visualize=True):
    """
    ADAPTIVE method: Combine height and normal filtering.
    
    Strategy:
    1. First, keep only upper portion of building (e.g., top 70%)
    2. Then, filter by normals to get roof surfaces
    
    This prevents capturing vertical walls in tall buildings.
    
    Parameters:
    - height_percentile: Keep points above this height percentile (0.3 = top 70%)
    - normal_z_threshold: Then filter by normal direction
    """
    
    z = xyz[:, 2]
    z_threshold = np.percentile(z, height_percentile * 100)
    
    # Step 1: Filter by height
    upper_mask = z >= z_threshold
    upper_xyz = xyz[upper_mask]
    
    print(f"\n📏 Height filtering:")
    print(f"   Keeping top {(1-height_percentile)*100:.0f}% by height")
    print(f"   Height threshold: {z_threshold:.2f}m")
    print(f"   Points retained: {upper_mask.sum()} / {len(xyz)}")
    
    # Step 2: Compute normals on upper portion
    print(f"\n🔍 Computing normals on upper portion...")
    normals = compute_normals_fast(upper_xyz, k=30)
    normals[:, 2] = np.abs(normals[:, 2])
    
    # Step 3: Filter by normals
    roof_mask_upper = normals[:, 2] > normal_z_threshold
    roof_xyz = upper_xyz[roof_mask_upper]
    
    print(f"\n🏠 Normal filtering:")
    print(f"   Normal Z threshold: {normal_z_threshold:.2f}")
    print(f"   Roof points: {roof_mask_upper.sum()} / {len(upper_xyz)} ({roof_mask_upper.sum()/len(upper_xyz)*100:.1f}%)")
    print(f"\n✅ Final roof points: {len(roof_xyz)} / {len(xyz)} ({len(roof_xyz)/len(xyz)*100:.1f}%)")
    
    if visualize:
        plt.figure(figsize=(12, 4))
        
        # Height distribution
        plt.subplot(1, 3, 1)
        plt.hist(z, bins=50, alpha=0.5, label='All points', edgecolor='black')
        plt.axvline(z_threshold, color='red', linestyle='--', linewidth=2, label='Height cutoff')
        plt.hist(upper_xyz[:, 2], bins=30, alpha=0.7, label='Upper portion', edgecolor='black')
        plt.xlabel('Height (m)')
        plt.ylabel('Count')
        plt.title('Step 1: Height Filtering')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Normal distribution
        plt.subplot(1, 3, 2)
        plt.hist(normals[:, 2], bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(normal_z_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Normal threshold')
        plt.xlabel('Normal Z component')
        plt.ylabel('Count')
        plt.title('Step 2: Normal Filtering')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Final result
        plt.subplot(1, 3, 3)
        plt.hist(xyz[:, 2], bins=50, alpha=0.4, label='All', edgecolor='black')
        plt.hist(roof_xyz[:, 2], bins=30, alpha=0.8, label='Final roof', edgecolor='black', color='green')
        plt.xlabel('Height (m)')
        plt.ylabel('Count')
        plt.title('Final Result')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Create full mask
    full_roof_mask = np.zeros(len(xyz), dtype=bool)
    upper_indices = np.where(upper_mask)[0]
    full_roof_mask[upper_indices[roof_mask_upper]] = True
    
    return roof_xyz, full_roof_mask


# ------------------ main ------------------
def main(eps=1.0, min_samples=20, alpha=0.5, 
         method='adaptive',
         height_percentile=0.3,
         normal_z_threshold=0.6,
         visualize=True):
    """
    Extract building footprint from ROOF points.
    
    METHODS:
    --------
    1. 'adaptive' (RECOMMENDED): Height + Normal filtering
       - height_percentile: 0.3 = top 70%, 0.2 = top 80%, 0.4 = top 60%
       - normal_z_threshold: 0.6 (inclusive), 0.7 (balanced), 0.8 (strict)
       
    2. 'normals': Pure normal-based filtering
       - normal_z_threshold: 0.6-0.8
    
    OTHER PARAMS:
    -------------
    - eps: DBSCAN clustering distance
    - min_samples: DBSCAN minimum points
    - alpha: alphashape concavity
    """

    las_path, las_name = select_las_file(CLIPPED_DIR)
    print(f"\n📂 Processing: {las_name}")

    las = laspy.read(las_path)

    # Building class only
    mask = las.classification == 6
    if mask.sum() == 0:
        print("❌ No building points (class 6) found.")
        sys.exit(1)

    xyz = np.vstack((las.x[mask], las.y[mask], las.z[mask])).T
    
    print(f"📊 Total building points: {len(xyz)}")
    print(f"   Height range: {xyz[:, 2].min():.2f}m - {xyz[:, 2].max():.2f}m")

    # ------------------ ROOF EXTRACTION ------------------
    if method == 'adaptive':
        roof_xyz, roof_mask = extract_roof_adaptive(
            xyz,
            height_percentile=height_percentile,
            normal_z_threshold=normal_z_threshold,
            visualize=visualize
        )
    elif method == 'normals':
        roof_xyz, roof_mask = extract_roof_by_normals(
            xyz,
            normal_z_threshold=normal_z_threshold,
            visualize=visualize
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if len(roof_xyz) < 100:
        print("⚠️  Very few roof points detected! Try:")
        print("   - LOWER normal_z_threshold (e.g., 0.5)")
        print("   - LOWER height_percentile (e.g., 0.2 for top 80%)")
        print("\nContinuing anyway...")
    
    xy = roof_xyz[:, :2]

    # ------------------ clustering ------------------
    print(f"\n🔍 Clustering roof points...")
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(xy)
    labels = db.labels_

    clusters = sorted(set(labels) - {-1})
    if not clusters:
        print("❌ No valid clusters detected.")
        print("   Try DECREASING 'eps' parameter")
        sys.exit(1)

    print(f"\n   Detected {len(clusters)} roof cluster(s):")
    for c in clusters:
        print(f"   - Cluster {c}: {(labels == c).sum()} points")

    # ------------------ visualization ------------------
    if len(clusters) > 1:
        cmap = cm.get_cmap("tab20", len(clusters))
        colors = np.zeros((len(roof_xyz), 3))

        for i, c in enumerate(clusters):
            colors[labels == c] = cmap(i)[:3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(roof_xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        print("\n👁️  Showing roof clusters (close viewer to continue)...")
        o3d.visualization.draw_geometries([pcd], window_name="Roof Clusters")

    # Select largest cluster
    cluster_sizes = {c: np.sum(labels == c) for c in clusters}
    selected_cluster_id = max(cluster_sizes, key=cluster_sizes.get)

    print(f"\n✅ Auto-selected cluster {selected_cluster_id} "
          f"({cluster_sizes[selected_cluster_id]} points)")
    
    clean_xy = xy[labels == selected_cluster_id]

    # ------------------ footprint extraction ------------------
    print(f"\n🔨 Extracting footprint with alpha={alpha}...")
    hull = alphashape.alphashape(clean_xy, alpha)

    if not isinstance(hull, Polygon):
        print("❌ Alpha shape did not return a single polygon.")
        print("   Try adjusting 'alpha' parameter:")
        print("   - LOWER alpha (e.g., 0.3) for more detail")
        print("   - HIGHER alpha (e.g., 0.8) for simpler shape")
        sys.exit(1)

    print(f"   Initial vertices: {len(hull.exterior.coords)-1}")

    # Simplify slightly
    hull = hull.simplify(0.1, preserve_topology=True)
    
    print(f"   After simplification: {len(hull.exterior.coords)-1}")
    print(f"   Area: {hull.area:.2f} m²")

    # ------------------ save ------------------
    out_name = os.path.splitext(las_name)[0] + f"_roof_cluster{selected_cluster_id}.geojson"
    out_path = os.path.join(FOOTPRINT_DIR, out_name)

    gdf = gpd.GeoDataFrame(geometry=[hull], crs="EPSG:32651")
    gdf.to_file(out_path, driver="GeoJSON")

    print(f"\n✅ Footprint saved to:\n   {out_path}")


# ------------------ run ------------------
if __name__ == "__main__":
    # 🎛️ TUNING GUIDE FOR ADAPTIVE METHOD:
    #
    # If roof capture is TOO SMALL (missing roof parts):
    #   → LOWER height_percentile (0.2 = top 80%, 0.1 = top 90%)
    #   → LOWER normal_z_threshold (0.5 = more inclusive)
    #
    # If capturing too much (walls included):
    #   → HIGHER height_percentile (0.4 = top 60%)
    #   → HIGHER normal_z_threshold (0.75 = stricter)
    #
    # For buildings with:
    #   - Flat roofs: normal_z_threshold=0.8 (strict)
    #   - Sloped roofs: normal_z_threshold=0.6 (inclusive)
    #   - Complex roofs: method='normals', normal_z_threshold=0.5
    
    main(
        eps=1.0,                      # DBSCAN clustering
        min_samples=20,               # Min points per cluster
        alpha=0.5,                    # Alpha shape parameter
        method='adaptive',            # 'adaptive' or 'normals'
        height_percentile=0.3,        # Top 70% by height
        normal_z_threshold=0.6,       # Normal pointing up threshold
        visualize=True                # Show plots
    )