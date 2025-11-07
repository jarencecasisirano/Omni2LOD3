import laspy
import numpy as np

def print_las_info(file_path):
    try:
        with laspy.open(file_path) as las_file:
            header = las_file.header
            las = las_file.read()

            print("LAS File Summary")
            print("=" * 50)

            # Version & Point Format
            print(f"Version        : {header.version}")
            print(f"Point Format   : {header.point_format.id}")
            print(f"Point Count    : {header.point_count:,}")

            # Scale & Offset
            print(f"Scale (X,Y,Z)  : {header.scales}")
            print(f"Offset (X,Y,Z) : {header.offsets}")

            # Bounds
            print(f"X Range        : {header.mins[0]:.3f} → {header.maxs[0]:.3f}")
            print(f"Y Range        : {header.mins[1]:.3f} → {header.maxs[1]:.3f}")
            print(f"Z Range        : {header.mins[2]:.3f} → {header.maxs[2]:.3f}")

            # CRS
            crs = header.parse_crs()
            if crs:
                print(f"CRS            : {crs.name}")
                wkt = crs.to_wkt(pretty=True)
                if len(wkt) > 200:
                    print(f"CRS WKT        : (long WKT string, {len(wkt)} chars)")
                else:
                    print(f"CRS WKT        : {wkt}")
            else:
                print("CRS            : Not specified")

            # Available dimensions (just names)
            dims = list(las.point_format.dimension_names)
            print(f"Dimensions     : {', '.join(dims)}")

    except Exception as e:
        print(f"Error reading LAS file: {e}")

if __name__ == "__main__":
    file_path = input("Enter the full path to your LAS file: ").strip().strip('"')
    print_las_info(file_path)