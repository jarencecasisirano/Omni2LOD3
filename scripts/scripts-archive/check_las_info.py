#!/usr/bin/env python3
import os, warnings, laspy
import numpy as np

def print_las_info(path: str) -> None:
    warnings.filterwarnings("ignore", module="pyproj")
    try:
        with laspy.open(path) as f:
            # Read header (fast)
            h = f.header
            print("LAS File Summary")
            print("=" * 50)
            print(f"Version        : {h.version}")
            print(f"Point Format   : {h.point_format.id}")
            print(f"Point Count    : {h.point_count:,}")
            print(f"Scale  (X,Y,Z) : {h.scales}")
            print(f"Offset (X,Y,Z) : {h.offsets}")
            for ax in "XYZ":
                i = "XYZ".index(ax)
                print(f"{ax} Range       : {h.mins[i]:.3f} → {h.maxs[i]:.3f}")
            crs = h.parse_crs()
            print(f"CRS            : {crs.name if crs else 'Not specified'}")
            print(f"Dimensions     : {', '.join(h.point_format.dimension_names)}")
            
            # Read points (slower, but needed for classification)
            print("\nClassification Statistics")
            print("=" * 50)
            las = f.read()
            
            # Check if classification dimension exists
            if 'classification' not in las.point_format.dimension_names:
                print("No classification data found in this file.")
                return
                
            classifications = las.classification
            unique, counts = np.unique(classifications, return_counts=True)
            
            # ASPRS Standard Class Names
            class_names = {
                0: "Never Classified", 1: "Unclassified", 2: "Ground",
                3: "Low Vegetation", 4: "Medium Vegetation", 5: "High Vegetation",
                6: "Building", 7: "Low Point (Noise)", 8: "Model Key-point",
                9: "Water", 10: "Rail", 11: "Road Surface", 12: "Overlap",
                14: "Transmission Tower", 15: "Wire-Guard", 16: "Wire-Conductor",
                17: "Bridge Deck", 18: "High Noise"
            }
            
            total_points = h.point_count
            print(f"{'Class':<6} {'Name':<25} {'Count':>12} {'Percentage':>10}")
            print("-" * 55)
            
            building_count = 0
            for cls, cnt in sorted(zip(unique, counts)):
                name = class_names.get(cls, f"Unknown ({cls})")
                percentage = (cnt / total_points) * 100
                print(f"{cls:<6} {name:<25} {cnt:>12,} {percentage:>9.1f}%")
                
                if cls == 6:  # ASPRS class 6 = Building
                    building_count = cnt
            
            # Highlight building points
            print("\n" + "=" * 50)
            print(f"🏢 BUILDING POINTS (Class 6): {building_count:,}")
            print("=" * 50)

            # Enhanced check_las_info.py snippet
            print("\n=== CITYFORGE READINESS CHECK ===")
            ground = las[las.classification == 2]
            buildings = las[las.classification == 6]

            if len(ground) > 0 and len(buildings) > 0:
                clearance = buildings.z.min() - ground.z.max()
                print(f"✓ Vertical clearance: {clearance:.2f}m (need >3m)")
                print(f"✓ Ground Z-range: {ground.z.min():.2f} to {ground.z.max():.2f}m")
                print(f"✓ Building Z-range: {buildings.z.min():.2f} to {buildings.z.max():.2f}m")
            else:
                print("✗ Missing ground or building class!")

            density = len(buildings) / ((las.header.x_max - las.header.x_min) * 
                                        (las.header.y_max - las.header.y_min))
            print(f"✓ Building density: {density:.1f} pts/m² (aim 5-8)")
            
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    print_las_info(os.path.expanduser(input("LAS file: ").strip().strip('"')))