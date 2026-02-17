from pathlib import Path
import argparse
import re

import geopandas as gpd
import laspy
import numpy as np
from shapely import contains_xy


def _normalize_feature_id(raw_id: object) -> str | None:
    if raw_id is None:
        return None

    if isinstance(raw_id, (float, np.floating)):
        if np.isnan(raw_id):
            return None
        if float(raw_id).is_integer():
            return str(int(raw_id))
        return str(raw_id).replace(".", "_")

    if isinstance(raw_id, (int, np.integer)):
        return str(int(raw_id))

    text = str(raw_id).strip()
    if not text or text.lower() == "nan":
        return None

    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
        text = text.replace(".", "_")
    except ValueError:
        pass

    return re.sub(r"[^A-Za-z0-9_-]+", "_", text)


def clip_las_by_polygons(input_las: Path, polygon_shp: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(polygon_shp)
    if gdf.empty:
        raise ValueError(f"No polygons found in: {polygon_shp}")

    polygons = list(gdf.geometry)
    print(f"Found {len(polygons)} polygon(s) in {polygon_shp.name}")

    id_col = next((col for col in gdf.columns if str(col).lower() == "id"), None)

    with laspy.open(input_las) as reader:
        las = reader.read()

    x = las.x
    y = las.y

    counts = []
    used_names = set()

    for idx, polygon in enumerate(polygons, start=1):
        if polygon is None or polygon.is_empty:
            print(f"Skipping empty geometry at polygon #{idx}")
            continue

        mask = np.asarray(contains_xy(polygon, x, y), dtype=bool)

        out_las = laspy.LasData(las.header)
        out_las.points = las.points[mask]

        feature_id = None
        if id_col is not None:
            feature_id = _normalize_feature_id(gdf.iloc[idx - 1][id_col])

        base_name = f"clip_id_{feature_id}" if feature_id is not None else f"clipped_{idx}"
        out_name = f"{base_name}.las"

        # Guard against duplicate IDs in the shapefile.
        if out_name in used_names:
            out_name = f"{base_name}_{idx}.las"

        used_names.add(out_name)
        out_path = output_dir / out_name
        out_las.write(out_path)

        point_count = int(mask.sum())
        counts.append((out_path.name, point_count))

    print("\nClipping complete. Point count per clipped LAS:")
    for name, point_count in counts:
        print(f"- {name}: {point_count} points")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clip a LAS file by polygons from a shapefile and print point counts."
    )
    parser.add_argument(
        "--input-las",
        type=Path,
        default=Path(r"C:\Projects\Omni2LOD3\data\01_point_cloud\NIMBB_LIDAR_aligned.las"),
        help="Input LAS file",
    )
    parser.add_argument(
        "--polygon-shp",
        type=Path,
        default=Path(r"C:\Projects\Omni2LOD3\data\02_footprint\lidar_clipping\lidarclip.shp"),
        help="Polygon shapefile for clipping",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Projects\Omni2LOD3\data\01_point_cloud\clipped"),
        help="Output directory for clipped LAS files",
    )

    args = parser.parse_args()

    if not args.input_las.exists():
        raise FileNotFoundError(f"Input LAS not found: {args.input_las}")
    if not args.polygon_shp.exists():
        raise FileNotFoundError(f"Polygon shapefile not found: {args.polygon_shp}")

    clip_las_by_polygons(args.input_las, args.polygon_shp, args.output_dir)


if __name__ == "__main__":
    main()
