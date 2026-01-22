import laspy
import numpy as np
import sys, os

def check_facade_completeness(las_path):
    print(f"\n{'='*60}")
    print("FACADE COMPLETENESS ANALYSIS")
    print(f"{'='*60}")
    
    las = laspy.read(las_path)
    building_pts = np.vstack((las.x, las.y, las.z)).T[las.classification == 6]
    
    # Create a 3D occupancy grid
    # Resolution: 0.5m voxels
    res = 0.5
    
    # Compute bounds
    x_min, x_max = np.min(building_pts[:, 0]), np.max(building_pts[:, 0])
    y_min, y_max = np.min(building_pts[:, 1]), np.max(building_pts[:, 1])
    z_min, z_max = np.min(building_pts[:, 2]), np.max(building_pts[:, 2])
    
    # Discretize points
    x_bins = np.arange(x_min, x_max + res, res)
    y_bins = np.arange(y_min, y_max + res, res)
    z_bins = np.arange(z_min, z_max + res, res)
    
    # Count occupied voxels
    x_indices = np.digitize(building_pts[:, 0], x_bins) - 1
    y_indices = np.digitize(building_pts[:, 1], y_bins) - 1
    z_indices = np.digitize(building_pts[:, 2], z_bins) - 1
    
    occupied_voxels = set(zip(x_indices, y_indices, z_indices))
    
    # Check facade continuity by looking at vertical columns
    facade_columns = {}
    for x_idx, y_idx, z_idx in occupied_voxels:
        key = (x_idx, y_idx)
        facade_columns.setdefault(key, []).append(z_idx)
    
    # Count columns with gaps
    columns_with_gaps = 0
    total_columns = len(facade_columns)
    
    for col, z_levels in facade_columns.items():
        if max(z_levels) - min(z_levels) > 5:  # Tall column
            # Check for gaps >2m
            z_sorted = sorted(z_levels)
            gaps = np.diff(z_sorted) * res
            if np.any(gaps > 2.0):
                columns_with_gaps += 1
    
    print(f"\n📊 Facade Coverage:")
    print(f"   Total vertical columns: {total_columns}")
    print(f"   Columns with gaps >2m: {columns_with_gaps} ({columns_with_gaps/total_columns*100:.1f}%)")
    
    if columns_with_gaps > total_columns * 0.3:
        print("❌ CRITICAL: Too many facade gaps")
        print("   Geoflow cannot create closed walls")
        facade_ok = False
    elif columns_with_gaps > total_columns * 0.1:
        print("⚠️  WARNING: Some facade gaps present")
        facade_ok = True
    else:
        print("✓ Facade coverage is good")
        facade_ok = True
    
    # Also check horizontal coverage
    xy_occupied = set((x_idx, y_idx) for x_idx, y_idx, _ in occupied_voxels)
    footprint_area = (x_max - x_min) * (y_max - y_min)
    voxel_area = len(xy_occupied) * res * res
    coverage_ratio = voxel_area / footprint_area
    
    print(f"\n📐 Horizontal Coverage:")
    print(f"   Footprint area: {footprint_area:.1f} m²")
    print(f"   Point-covered area: {voxel_area:.1f} m²")
    print(f"   Coverage ratio: {coverage_ratio:.1%}")
    
    if coverage_ratio < 0.5:
        print("⚠️  WARNING: Sparse horizontal coverage")
        facade_ok = False
    
    return facade_ok

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_facade_completeness.py \"file.las\"")
        sys.exit(1)
    
    check_facade_completeness(sys.argv[1])