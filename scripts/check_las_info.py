import laspy
import numpy as np

def print_las_info(file_path):
    try:
        with laspy.open(file_path) as las_file:
            header = las_file.header
            points = las_file.read()

            # Basic header information
            print("LAS File Information:")
            print(f"- File Version: {header.version}")
            print(f"- Point Format: {header.point_format.id}")
            print(f"- Point Count: {header.point_count}")
            print(f"- Scale Factors: {header.scales}")
            print(f"- Offsets: {header.offsets}")

            # Bounds
            bounds = header.mins, header.maxs
            print(f"- Bounds (min, max): {bounds}")

            # CRS (Coordinate Reference System)
            if header.parse_crs() is not None:
                crs = header.parse_crs()
                print(f"- CRS: {crs.name if crs else 'Not specified or unparseable'}")
                print(f"- CRS WKT: {crs.to_wkt() if crs else 'Not available'}")
            else:
                print("- CRS: Not found or not parseable")

            # Classification information
            if "classification" in points.point_format.dimension_names:
                classifications = points.classification
                if classifications is not None and len(classifications) > 0:
                    unique_classes = np.unique(classifications)
                    class_counts = dict(zip(unique_classes, [np.sum(classifications == c) for c in unique_classes]))
                    print("- Classifications Present:")
                    for class_id, count in class_counts.items():
                        print(f"  - Class {int(class_id)}: {count} points")
                    # Common LAS classification codes for reference
                    print("\n- Common LAS Classification Codes:")
                    print("  - 0: Never Classified")
                    print("  - 1: Unclassified")
                    print("  - 2: Ground")
                    print("  - 6: Building")
                    print("  - 9: Water")
                    print("  - 12: Overlap")
                else:
                    print("- Classifications: No valid classification data found")
            else:
                print("- Classifications: Not available in point format")

            # Additional dimensions
            available_dims = points.point_format.dimension_names
            print(f"- Available Dimensions: {available_dims}")

            # Global Encoding
            global_encoding = header.global_encoding
            encoding_bits = []
            if global_encoding.wkt:
                encoding_bits.append("WKT")
            if global_encoding.geocentric:
                encoding_bits.append("Geocentric")
            print(f"- Global Encoding Flags: {', '.join(encoding_bits) if encoding_bits else 'None'}")

    except Exception as e:
        print(f"Error reading LAS file: {e}")

if __name__ == "__main__":
    file_path = input("Enter the full path to your LAS file: ")
    print_las_info(file_path)