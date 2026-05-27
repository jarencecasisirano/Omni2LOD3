import os
import numpy as np
import geopandas as gpd
import laspy
from shapely.geometry import Point
from shapely.prepared import prep

BUILDING_CLASS = 6


def run(input_las, footprint_shp, output_las, progress_callback=None):
    """
    Pipeline-safe LAS reclassification using building footprints.
    """

    las = laspy.read(input_las)

    X = np.asarray(las.x)
    Y = np.asarray(las.y)

    cls = np.asarray(las.classification).copy() if hasattr(las, "classification") else np.zeros(len(X), dtype=np.uint8)

    gdf = gpd.read_file(footprint_shp)

    if gdf.empty:
        raise ValueError("Footprint shapefile is empty")

    las_crs = las.header.parse_crs()

    if las_crs is not None and gdf.crs is not None and gdf.crs != las_crs:
        print("-> Reprojecting footprint to LAS CRS...")
        gdf = gdf.to_crs(las_crs)

    footprints = [
        geom.buffer(0)
        for geom in gdf.geometry.values
        if geom is not None and not geom.is_empty
    ]

    if not footprints:
        raise ValueError("No valid footprint geometry found")

    inside_any = np.zeros(len(X), dtype=bool)
    forced_count = 0

    total = len(footprints)

    print(f"-> Points: {len(X):,}")
    print(f"-> Footprints: {total}")

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

        # enforce classification
        not_building = inside_idx[cls[inside_idx] != BUILDING_CLASS]

        if len(not_building) > 0:
            cls[not_building] = BUILDING_CLASS
            forced_count += len(not_building)

        # progress hook (GUI-ready)
        if progress_callback:
            progress_callback(poly_idx / total)

        if (poly_idx + 1) % 5 == 0:
            print(f"-> Processed {poly_idx + 1}/{total}")

    inside_total = int(inside_any.sum())

    if inside_total == 0:
        print("[INFO] No points inside footprints. No output generated.")
        return {
            "status": "no_points_found",
            "output": None
        }

    os.makedirs(os.path.dirname(output_las) or ".", exist_ok=True)

    las.classification = cls
    las.write(output_las)

    print(f"-> Saved: {output_las}")

    return {
        "status": "success",
        "output": str(output_las),
        "inside_points": inside_total,
        "reclassified_points": forced_count
    }


# CLI wrapper (kept minimal)
def _cli():
    import sys

    if len(sys.argv) < 4:
        print("Usage: python 02_reclassify.py <input_las> <footprint_shp> <output_las>")
        sys.exit(1)

    run(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    _cli()