# verify_crs_alignment_fixed.py
import os
import sys
import json
import laspy
import numpy as np

def verify_alignment(las_path, footprint_geojson_path):
    print(f"\n{'='*70}")
    print("CRS & COORDINATE ALIGNMENT CHECK")
    print(f"{'='*70}")
    
    # Load LAS
    las = laspy.read(las_path)
    print(f"\n📍 LAS File: {os.path.basename(las_path)}")
    
    # Get CRS info (works with all laspy versions)
    crs_info = "Not found in header"
    try:
        # Check for CRS in VLRs
        if hasattr(las.header, 'vlrs'):
            for vlr in las.header.vlrs:
                if hasattr(vlr, 'record_id'):
                    if vlr.record_id == 2112:  # WKT CRS
                        crs_info = f"WKT CRS present"
                        break
                    elif vlr.record_id == 34735:  # GeoTIFF
                        crs_info = "GeoTIFF CRS info present"
                        break
        
        # Check if pyproj CRS is available
        if hasattr(las, 'crs') and las.crs is not None:
            crs_info = str(las.crs)
        
        # Fallback: check global encoding for WKT
        if las.header.global_encoding.wkt:
            crs_info = "WKT CRS flag enabled"
            
    except Exception as e:
        crs_info = f"Error reading CRS: {e}"
    
    print(f"   CRS: {crs_info}")
    print(f"   X range: {las.header.x_min:.3f} → {las.header.x_max:.3f}")
    print(f"   Y range: {las.header.y_min:.3f} → {las.header.y_max:.3f}")
    
    # Load GeoJSON
    try:
        with open(footprint_geojson_path) as f:
            geojson = json.load(f)
        
        # Handle FeatureCollection or single Feature
        if geojson['type'] == 'FeatureCollection':
            coords = geojson['features'][0]['geometry']['coordinates'][0]
        else:
            coords = geojson['geometry']['coordinates'][0]
            
        xs, ys = zip(*coords)
        
        print(f"\n📐 Footprint: {os.path.basename(footprint_geojson_path)}")
        print(f"   X range: {min(xs):.3f} → {max(xs):.3f}")
        print(f"   Y range: {min(ys):.3f} → {max(ys):.3f}")
    except Exception as e:
        print(f"[ERROR] Could not parse GeoJSON: {e}")
        sys.exit(1)
    
    # Check overlap
    x_overlap = max(0, min(max(xs), las.header.x_max) - max(min(xs), las.header.x_min))
    y_overlap = max(0, min(max(ys), las.header.y_max) - max(min(ys), las.header.y_min))
    
    print(f"\n🔍 Overlap Area: {x_overlap:.2f}m × {y_overlap:.2f}m")
    
    if x_overlap > 1 and y_overlap > 1:
        print("✓ LAS and footprint overlap - CRS alignment OK")
        alignment_ok = True
    else:
        print("❌ NO OVERLAP! CRS mismatch or wrong footprint")
        print("\nPossible fixes:")
        print("  1. Regenerate footprint from THIS LAS file")
        print("  2. Check if LAS is in degrees (long/lat) vs meters (UTM)")
        print("  3. Verify GeoJSON CRS matches LAS CRS")
        alignment_ok = False
    
    # Check for synthetic grid pattern
    xy_rounded = np.round(np.vstack((las.x, las.y)).T * 100) / 100  # 1cm precision
    unique_xy = len(np.unique(xy_rounded, axis=0))
    unique_ratio = unique_xy / len(las.points)
    
    print(f"\n🔍 Grid Pattern Check:")
    print(f"   Unique XY positions (1cm): {unique_xy:,}")
    print(f"   Total points: {len(las.points):,}")
    print(f"   Unique ratio: {unique_ratio:.1%}")
    
    if unique_ratio < 0.3:
        print("⚠️  CRITICAL: Severe synthetic grid detected")
        print("   Geoflow's Poisson fails on perfectly uniform points")
    elif unique_ratio < 0.6:
        print("⚠️  WARNING: High grid uniformity")
    else:
        print("✓ XY distribution is natural enough")
    
    return alignment_ok, unique_ratio

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python verify_crs_alignment.py \"file.las\" \"footprint.geojson\"")
        sys.exit(1)
    
    alignment_ok, grid_ratio = verify_alignment(sys.argv[1], sys.argv[2])
    
    if not alignment_ok:
        print("\n❌ CRS/Alignment issue confirmed!")
        sys.exit(1)
    
    if grid_ratio < 0.3:
        print("\n❌ Grid pattern is too synthetic for CityForge!")
        sys.exit(1)
        
    print("\n✅ Alignment and pattern checks PASSED")