# compare_working_vs_failing.py
import laspy
import numpy as np
import sys, os

def analyze_for_cityforge(las_path):
    """Extract CityForge-critical metrics"""
    las = laspy.read(las_path)
    
    # Get building points only
    building_mask = las.classification == 6
    bldg_pts = np.vstack((las.x[building_mask], las.y[building_mask], las.z[building_mask])).T
    
    # Key metrics that matter to Geoflow
    metrics = {
        "file": os.path.basename(las_path),
        "total_points": len(las.points),
        "building_points": len(bldg_pts),
        "building_percentage": np.mean(building_mask) * 100,
        "xy_spread": np.std(bldg_pts[:, :2]),  # How "spread out" XY is
        "z_spread": np.std(bldg_pts[:, 2]),     # How varied Z is
        "z_skew": np.abs(np.mean(bldg_pts[:, 2]) - np.median(bldg_pts[:, 2])),  # Natural vs synthetic
        "xy_entropy": -np.sum(np.histogram2d(bldg_pts[:, 0], bldg_pts[:, 1], bins=20)[0] * 
                              np.log2(np.histogram2d(bldg_pts[:, 0], bldg_pts[:, 1], bins=20)[0] + 1e-10)),
        "is_single_cluster": check_single_cluster(bldg_pts)
    }
    
    return metrics

def check_single_cluster(points):
    """Check if points form one connected cluster vs multiple islands"""
    from sklearn.cluster import DBSCAN
    if len(points) < 100:
        return False
    
    # DBSCAN to find clusters
    db = DBSCAN(eps=2.0, min_samples=10).fit(points[:, :2])
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    
    return n_clusters == 1

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare.py working.las failing.las")
        sys.exit(1)
    
    working = analyze_for_cityforge(sys.argv[1])
    failing = analyze_for_cityforge(sys.argv[2])
    
    print("\n" + "="*70)
    print("CITYFORGE CRITICAL COMPARISON")
    print("="*70)
    
    for key in working:
        if key == "file":
            print(f"\n{working[key]:<40} {failing[key]}")
            continue
        
        w_val = working[key]
        f_val = failing[key]
        
        # Highlight key differences
        marker = "✓" if abs(w_val - f_val) < 0.1 * max(abs(w_val), 1) else "⚠️"
        if key in ["building_points", "xy_spread", "z_spread"] and abs(w_val - f_val) > 50:
            marker = "❌"
        
        print(f"{marker} {key:<25}: {w_val:.3f}  vs  {f_val:.3f}")