# diagnose_cityforge_failure.py (FIXED)
import numpy as np
import laspy
import os, sys
import open3d as o3d

def run_comprehensive_diagnostic(las_path):
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC REPORT: {os.path.basename(las_path)}")
    print(f"{'='*60}\n")
    
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).T
    
    # 1. FACADE HOLE ANALYSIS
    print("### 1. FACADE HOLE ANALYSIS ###")
    building_pts = points[las.classification == 6]
    
    if len(building_pts) < 100:
        print("❌ CRITICAL: Not enough building points for analysis.")
        return ["Insufficient building points"]
    
    # Downsample for speed
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(building_pts)
    pcd = pcd.voxel_down_sample(voxel_size=0.5)
    
    # Compute nearest neighbor distances
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    for pt in pcd.points:
        k, idx, dist = kdtree.search_knn_vector_3d(pt, 2)  # 1 neighbor + self
        if k > 1:  # FIX: Check if we found at least 2 points
            neighbor_dist = np.sqrt(dist[1])  # dist[0] is self, dist[1] is neighbor
            dists.append(neighbor_dist)
    
    if not dists:
        print("❌ CRITICAL: Could not compute distances - points may be too sparse.")
        return ["Invalid point distribution"]
    
    mean_gap = np.mean(dists)
    max_gap = np.max(dists)
    
    print(f"Mean gap between facade points: {mean_gap:.3f}m")
    print(f"Max gap between facade points: {max_gap:.3f}m")
    
    if max_gap > 2.0:
        print("❌ CRITICAL: Facade has holes >2m. Geoflow will fail to close surfaces.")
        facade_issue = "Large facade holes"
    elif mean_gap > 0.8:
        print("⚠️  WARNING: Facade spacing is too sparse. Increase point density.")
        facade_issue = "Sparse facade points"
    else:
        print("✓ Facade point density is acceptable.")
        facade_issue = None
    
    # 2. SYNTHETIC POINT PATTERN DETECTION (MOST IMPORTANT FOR YOUR CASE)
    print("\n### 2. SYNTHETIC POINT UNIFORMITY CHECK ###")
    # Check for unnatural grid patterns in XY
    xy_rounded = np.round(building_pts[:, :2] * 10) / 10  # Round to 10cm
    
    # Count duplicates at 10cm precision
    unique_xy = len(np.unique(xy_rounded, axis=0))
    total_points = len(building_pts)
    duplicate_ratio = 1 - (unique_xy / total_points)
    
    print(f"Total building points: {total_points}")
    print(f"Unique XY positions (10cm precision): {unique_xy}")
    print(f"Duplicate ratio: {duplicate_ratio:.1%}")
    
    if duplicate_ratio > 0.25:
        print("❌ CRITICAL: Points are too uniform (synthetic grid detected).")
        print("   Geoflow's Poisson reconstruction interprets this as noise.")
        synthetic_issue = "Synthetic grid pattern"
    else:
        print("✓ Point distribution appears natural enough.")
        synthetic_issue = None
    
    # 3. BOUNDARY COMPLETENESS
    print("\n### 3. BOUNDARY COMPLETENESS ###")
    x_min, x_max = np.min(building_pts[:, 0]), np.max(building_pts[:, 0])
    y_min, y_max = np.min(building_pts[:, 1]), np.max(building_pts[:, 1])
    
    # Count points within 0.5m of boundary
    edge_threshold = 0.5
    edge_mask = (
        (building_pts[:, 0] < x_min + edge_threshold) |
        (building_pts[:, 0] > x_max - edge_threshold) |
        (building_pts[:, 1] < y_min + edge_threshold) |
        (building_pts[:, 1] > y_max - edge_threshold)
    )
    edge_point_ratio = np.sum(edge_mask) / len(building_pts)
    
    print(f"Points on building edges: {edge_point_ratio:.1%}")
    
    if edge_point_ratio < 0.10:
        print("❌ CRITICAL: Too few points on building perimeter.")
        print("   Geoflow cannot reconstruct walls without edge points.")
        edge_issue = "Missing edge points"
    else:
        print("✓ Edge points are present.")
        edge_issue = None
    
    # 4. ROOF/WALL SEPARATION (Height distribution)
    print("\n### 4. ROOF/WALL SEPARATION ###")
    z_values = building_pts[:, 2]
    z_hist, z_edges = np.histogram(z_values, bins=50)
    
    # Find significant peaks in height distribution
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(z_hist, height=np.max(z_hist)*0.1)
    
    if len(peaks) < 2:
        print("⚠️  WARNING: Only one height peak detected.")
        print("   Geoflow may struggle to separate roof from walls.")
        separation_issue = "Unclear roof/wall separation"
    else:
        print(f"✓ Found {len(peaks)} height peaks (good for roof/wall separation).")
        separation_issue = None
    
    # 5. SUMMARY
    print("\n" + "="*60)
    print("GEOFLOW READINESS SUMMARY")
    print("="*60)
    
    all_issues = [facade_issue, synthetic_issue, edge_issue, separation_issue]
    critical_issues = [i for i in all_issues if i is not None]
    
    if critical_issues:
        print(f"❌ GEOFLOW WILL FAIL due to:")
        for i, issue in enumerate(critical_issues, 1):
            print(f"   {i}. {issue}")
        print("\n🔧 FIX PRIORITY:")
        print("   1. Add random jitter to facade points (±5cm)")
        print("   2. Fill large holes in facade generation")
        print("   3. Ensure points cover footprint edges")
    else:
        print("✓ All checks passed!")
        print("  If Geoflow still fails, check:")
        print("  - CRS/units match (UTM 51N, meters)")
        print("  - GeoJSON footprint is valid & closed")
        print("  - Run Geoflow with --verbose flag")
    
    return critical_issues

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1].strip('"').strip("'")
    else:
        path = input("Drag and drop your LAS file here: ").strip('"').strip("'")
    
    issues = run_comprehensive_diagnostic(path)
    
    if issues:
        print(f"\nDiagnostic found {len(issues)} critical issues.")
        sys.exit(1)
    else:
        print("\n🎉 Diagnostic passed. Data is Geoflow-ready!")
        sys.exit(0)