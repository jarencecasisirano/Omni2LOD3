#!/usr/bin/env python3
"""
Check if ground points (class 2) are actually present and readable
"""
import laspy
import numpy as np

def check_ground_points(filepath, label):
    print(f"\n{'='*60}")
    print(f"CHECKING: {label}")
    print(f"File: {filepath}")
    print(f"{'='*60}")
    
    las = laspy.read(filepath)
    
    # Get ground points (classification = 2)
    ground_mask = las.classification == 2
    ground_count = np.sum(ground_mask)
    
    print(f"\nTotal points: {len(las.points):,}")
    print(f"Ground points (class 2): {ground_count:,}")
    
    if ground_count > 0:
        ground_z = las.z[ground_mask]
        print(f"\nGround point Z statistics:")
        print(f"  Min Z: {ground_z.min():.3f}")
        print(f"  Max Z: {ground_z.max():.3f}")
        print(f"  Mean Z: {ground_z.mean():.3f}")
        print(f"  Median Z: {np.median(ground_z):.3f}")
        
        # Check if ground points are at reasonable elevations
        if ground_z.min() < 0 or ground_z.max() > 300:
            print(f"\n⚠️  WARNING: Unusual ground elevations detected!")
        
        # Sample a few ground points
        print(f"\nFirst 5 ground points:")
        ground_indices = np.where(ground_mask)[0][:5]
        for idx in ground_indices:
            print(f"  Point {idx}: X={las.x[idx]:.3f}, Y={las.y[idx]:.3f}, Z={las.z[idx]:.3f}, Class={las.classification[idx]}")
    else:
        print(f"\n❌ ERROR: NO GROUND POINTS FOUND!")
        print(f"This is why CityForge shows h_ground = FLT_MAX")
    
    # Check all classifications
    print(f"\nAll classifications present:")
    unique_classes, counts = np.unique(las.classification, return_counts=True)
    for cls, count in zip(unique_classes, counts):
        print(f"  Class {cls}: {count:,} points")
    
    return las

if __name__ == "__main__":
    print("="*60)
    print("GROUND POINTS DIAGNOSTIC")
    print("="*60)
    
    file_b_works = r"C:\Projects\Omni2LOD3\outputs\00_archive\02_test_B_process_clipped.las"
    file_a_broken = r"C:\Projects\Omni2LOD3\outputs\02_clipped\NIMBB 112025_05_clipped.las"
    file_b_original = r"C:\Projects\Omni2LOD3\outputs\00_archive\01_test_clipped.las"
    
    print("\nThis will check if ground points are present and accessible in each file.")
    
    try:
        # Check the working file from exact B process
        las_b_works = check_ground_points(file_b_works, "TEST B PROCESS (WORKS)")
        
        # Check the broken Pipeline A file
        las_a_broken = check_ground_points(file_a_broken, "PIPELINE A (BROKEN)")
        
        # Check original Pipeline B file
        las_b_original = check_ground_points(file_b_original, "ORIGINAL PIPELINE B (WORKS)")
        
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        
        # Compare ground point counts
        ground_b = np.sum(las_b_works.classification == 2)
        ground_a = np.sum(las_a_broken.classification == 2)
        
        if ground_b == ground_a:
            print(f"\n✓ Both have same number of ground points: {ground_b:,}")
        else:
            print(f"\n⚠️  DIFFERENCE in ground point count!")
            print(f"  B works: {ground_b:,}")
            print(f"  A broken: {ground_a:,}")
        
        # Compare exact values
        print(f"\nComparing first ground point in each file:")
        ground_idx_b = np.where(las_b_works.classification == 2)[0][0]
        ground_idx_a = np.where(las_a_broken.classification == 2)[0][0]
        
        print(f"  B: X={las_b_works.x[ground_idx_b]:.6f}, Y={las_b_works.y[ground_idx_b]:.6f}, Z={las_b_works.z[ground_idx_b]:.6f}")
        print(f"  A: X={las_a_broken.x[ground_idx_a]:.6f}, Y={las_a_broken.y[ground_idx_a]:.6f}, Z={las_a_broken.z[ground_idx_a]:.6f}")
        
        if (las_b_works.x[ground_idx_b] == las_a_broken.x[ground_idx_a] and
            las_b_works.y[ground_idx_b] == las_a_broken.y[ground_idx_a] and
            las_b_works.z[ground_idx_b] == las_a_broken.z[ground_idx_a]):
            print(f"\n✓ First ground point is identical!")
        else:
            print(f"\n⚠️  First ground point is DIFFERENT!")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()