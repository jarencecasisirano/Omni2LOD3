# 02_reclassify.py
"""
Reclassify point cloud by footprint.

Behavior:
- Reads full LAS (typically downsampled output)
- Finds points covered by the given footprint polygon(s)
- Forces those points to LAS class 6 (Building)
- Keeps all points (inside + outside footprint) in the output
"""

import os
import sys
import numpy as np
import geopandas as gpd
import laspy
from shapely.geometry import Point
from shapely.prepared import prep


BUILDING_CLASS = 6


def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print("  python 02_reclassify.py <input_las> <footprint_shp> <output_las>")
        sys.exit(1)

    las_input = sys.argv[1]
    footprint_shp = sys.argv[2]
    las_output = sys.argv[3]

    os.makedirs(os.path.dirname(las_output) or ".", exist_ok=True)

    print("\n" + "=" * 70)
    print("POINT CLOUD RECLASSIFICATION BY FOOTPRINT")
    print("=" * 70)
    print(f"Input LAS: {las_input}")
    print(f"Footprint: {footprint_shp}")
    print(f"Output:    {las_output}")

    las = laspy.read(las_input)
    X = np.asarray(las.x)
    Y = np.asarray(las.y)

    if hasattr(las, "classification"):
        cls = np.asarray(las.classification).copy()
    else:
        cls = np.zeros(len(X), dtype=np.uint8)

    gdf = gpd.read_file(footprint_shp)
    if gdf.empty:
        print("[ERROR] Footprint shapefile is empty.")
        sys.exit(1)

    las_crs = las.header.parse_crs()
    if las_crs is not None and gdf.crs is not None and gdf.crs != las_crs:
        print("-> Reprojecting footprint to LAS CRS...")
        gdf = gdf.to_crs(las_crs)

    footprints = [geom.buffer(0) for geom in gdf.geometry.values if geom is not None and not geom.is_empty]
    if not footprints:
        print("[ERROR] No valid footprint geometry found.")
        sys.exit(1)

    n_total = len(X)
    forced_count = 0
    inside_any = np.zeros(n_total, dtype=bool)

    print(f"-> Total points: {n_total:,}")
    print(f"-> Footprints:   {len(footprints)}")
    print("-> Reclassifying points inside footprint(s)...")

    for poly_idx, poly in enumerate(footprints):
        prep_poly = prep(poly)
        minx, miny, maxx, maxy = poly.bounds

        bbox_mask = (
            (X >= minx) & (X <= maxx) &
            (Y >= miny) & (Y <= maxy)
        )
        idx_bbox = np.where(bbox_mask)[0]
        if len(idx_bbox) == 0:
            continue

        inside_idx = []
        for i in idx_bbox:
            if prep_poly.covers(Point(X[i], Y[i])):
                inside_idx.append(i)

        if not inside_idx:
            continue

        inside_idx = np.asarray(inside_idx, dtype=int)
        inside_any[inside_idx] = True

        not_building = inside_idx[cls[inside_idx] != BUILDING_CLASS]
        if len(not_building) > 0:
            cls[not_building] = BUILDING_CLASS
            forced_count += len(not_building)

        if (poly_idx + 1) % 5 == 0:
            print(f"   Processed footprint {poly_idx + 1}/{len(footprints)}")

    las.classification = cls
    las.write(las_output)

    inside_total = int(inside_any.sum())
    print("\n" + "-" * 60)
    print(f"Points inside footprint(s):      {inside_total:,}")
    print(f"Points forced to class 6:        {forced_count:,}")
    print(f"Points outside footprint(s):     {n_total - inside_total:,}")
    print("-" * 60)
    print(f"-> Saved: {las_output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
