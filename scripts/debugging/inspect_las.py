import laspy
import numpy as np
import sys
from pathlib import Path

def inspect_las(las_path: Path):
    if not las_path.exists():
        print(f"\n❌ File not found: {las_path}")
        return

    print(f"\nInspecting: {las_path}")
    las = laspy.read(las_path)

    print("Points:", len(las.points))
    print("CRS:", las.header.parse_crs())
    print("Scales:", las.header.scales)
    print("Offsets:", las.header.offsets)

    if hasattr(las, "classification"):
        classes, counts = np.unique(las.classification, return_counts=True)
        print("Classes:", dict(zip(classes, counts)))
    else:
        print("No classification field")

    print("Z min/max:", las.z.min(), las.z.max())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python inspect_las.py path/to/file.las [path/to/file2.laz]")
        sys.exit(1)

    for arg in sys.argv[1:]:
        inspect_las(Path(arg))
