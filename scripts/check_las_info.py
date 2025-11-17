#!/usr/bin/env python3
import os, warnings, laspy

def print_las_info(path: str) -> None:
    warnings.filterwarnings("ignore", module="pyproj")
    try:
        with laspy.open(path) as f:
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
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    print_las_info(os.path.expanduser(input("LAS file: ").strip().strip('"')))