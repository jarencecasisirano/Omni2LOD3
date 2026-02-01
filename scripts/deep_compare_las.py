#!/usr/bin/env python3
"""
Deep comparison of two LAS files to find hidden differences
"""
import laspy
import numpy as np

def analyze_las_deep(filepath, label):
    """Perform deep analysis of LAS file"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {label}")
    print(f"File: {filepath}")
    print(f"{'='*60}")
    
    las = laspy.read(filepath)
    
    # Basic info
    print(f"\n[BASIC INFO]")
    print(f"Points: {len(las.points):,}")
    print(f"Version: {las.header.version}")
    print(f"Point Format: {las.header.point_format.id}")
    
    # Header details
    print(f"\n[HEADER DETAILS]")
    print(f"Scales: {las.header.scales}")
    print(f"Offsets: {las.header.offsets}")
    print(f"Mins: {las.header.mins}")
    print(f"Maxs: {las.header.maxs}")
    
    # VLRs (Variable Length Records - includes CRS)
    print(f"\n[VLRs - Variable Length Records]")
    print(f"Number of VLRs: {len(las.header.vlrs)}")
    for i, vlr in enumerate(las.header.vlrs):
        print(f"  VLR {i}: user_id='{vlr.user_id}', record_id={vlr.record_id}, description='{vlr.description}'")
        if vlr.user_id == "LASF_Projection":
            print(f"    -> This is CRS/Projection data")
    
    # Point format dimensions
    print(f"\n[POINT FORMAT DIMENSIONS]")
    print(f"Available dimensions: {las.point_format.dimension_names}")
    
    # Check for extra bytes / extra dimensions
    print(f"\n[EXTRA DIMENSIONS]")
    if hasattr(las.header, 'vlrs'):
        extra_dims = [vlr for vlr in las.header.vlrs if vlr.record_id == 4]
        if extra_dims:
            print(f"Found {len(extra_dims)} extra dimension VLR(s)")
            for vlr in extra_dims:
                print(f"  {vlr}")
        else:
            print("No extra dimensions")
    
    # Point data statistics
    print(f"\n[POINT DATA]")
    print(f"X range: {las.x.min():.3f} to {las.x.max():.3f}")
    print(f"Y range: {las.y.min():.3f} to {las.y.max():.3f}")
    print(f"Z range: {las.z.min():.3f} to {las.z.max():.3f}")
    
    # Classifications
    print(f"\n[CLASSIFICATIONS]")
    unique_classes = np.unique(las.classification)
    for c in sorted(unique_classes):
        count = np.sum(las.classification == c)
        print(f"  Class {c}: {count:,} points")
    
    # Check for additional fields
    print(f"\n[ADDITIONAL FIELDS]")
    for dim in las.point_format.dimension_names:
        if dim not in ['X', 'Y', 'Z', 'classification']:
            data = getattr(las, dim.lower())
            print(f"  {dim}: min={data.min()}, max={data.max()}, unique_values={len(np.unique(data))}")
    
    # Check if RGB exists
    if 'red' in [d.lower() for d in las.point_format.dimension_names]:
        print(f"\n[RGB DATA]")
        print(f"  Red: {las.red.min()} to {las.red.max()}")
        print(f"  Green: {las.green.min()} to {las.green.max()}")
        print(f"  Blue: {las.blue.min()} to {las.blue.max()}")
    
    # Check for GPS time
    if 'gps_time' in [d.lower() for d in las.point_format.dimension_names]:
        print(f"\n[GPS TIME]")
        print(f"  Range: {las.gps_time.min()} to {las.gps_time.max()}")
    else:
        print(f"\n[GPS TIME] Not present in this format")
    
    # Global encoding
    print(f"\n[GLOBAL ENCODING]")
    print(f"  Value: {las.header.global_encoding}")
    
    # System identifier and generating software
    print(f"\n[METADATA]")
    print(f"  System ID: '{las.header.system_identifier}'")
    print(f"  Generating Software: '{las.header.generating_software}'")
    
    return las


if __name__ == "__main__":
    print("\n" + "="*60)
    print("LAS FILE DEEP COMPARISON TOOL")
    print("="*60)
    
    # Update these paths to your actual files
    file_b = r"C:\Projects\Omni2LOD3\outputs\00_archive\01_test_clipped.las"
    file_a = r"C:\Projects\Omni2LOD3\outputs\02_clipped\NIMBB 112025_05_clipped.las"
    
    print("\n[INSTRUCTIONS]")
    print("This script will analyze both LAS files in detail.")
    print("Update the file paths in the script before running.")
    print(f"\nFile B (WORKS): {file_b}")
    print(f"File A (DOESN'T WORK): {file_a}")
    
    try:
        las_b = analyze_las_deep(file_b, "PIPELINE B (WORKS)")
        las_a = analyze_las_deep(file_a, "PIPELINE A (DOESN'T WORK)")
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        # Compare VLRs
        print(f"\nVLR Count: B={len(las_b.header.vlrs)}, A={len(las_a.header.vlrs)}")
        if len(las_b.header.vlrs) != len(las_a.header.vlrs):
            print("  ⚠️  DIFFERENCE in VLR count!")
        
        # Compare dimensions
        dims_b = set(las_b.point_format.dimension_names)
        dims_a = set(las_a.point_format.dimension_names)
        if dims_b != dims_a:
            print(f"\n⚠️  DIFFERENCE in dimensions!")
            print(f"  Only in B: {dims_b - dims_a}")
            print(f"  Only in A: {dims_a - dims_b}")
        else:
            print(f"\n✓ Dimensions match: {dims_a}")
        
        # Compare generating software
        if las_b.header.generating_software != las_a.header.generating_software:
            print(f"\n⚠️  DIFFERENCE in generating software!")
            print(f"  B: '{las_b.header.generating_software}'")
            print(f"  A: '{las_a.header.generating_software}'")
        
        # Compare system identifier
        if las_b.header.system_identifier != las_a.header.system_identifier:
            print(f"\n⚠️  DIFFERENCE in system identifier!")
            print(f"  B: '{las_b.header.system_identifier}'")
            print(f"  A: '{las_a.header.system_identifier}'")
        
        print("\n" + "="*60)
        print("Analysis complete. Check for ⚠️  markers above.")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease update the file paths in this script to match your system.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()