import os
import argparse

from steps.downsampling import run as step01
from steps.reclassify import run as step02
from steps.merge_walls import run as step03
from steps.process_cubemaps import process_images as step04


def run_pipeline(
    input_las,
    shp,
    output_dir,
    voxel_size=0.1,
    cityjson_input=None,
    image_input_dir=None,
    yaw=0.0,
    pitch=0.0,
    roll=0.0
):

    # -------------------------
    # 1. VALIDATION
    # -------------------------
    if not input_las:
        raise ValueError("input_las is required")

    if not shp:
        raise ValueError("shapefile is required")

    if not output_dir:
        raise ValueError("output_dir is required")

    os.makedirs(output_dir, exist_ok=True)

    print("\n==============================")
    print("OMNI2LOD3 PIPELINE START")
    print("==============================\n")

    results = {}

    # =========================================================
    # STEP 01: DOWNSAMPLING (LAS)
    # =========================================================
    print("STEP 01: Downsampling")

    ds_output = os.path.join(output_dir, "01_downsample.las")

    result1 = step01(
        input_las=input_las,
        output_las=ds_output,
        voxel_size=voxel_size
    )

    if result1 is None or result1.get("output") is None:
        raise RuntimeError("Step 01 failed: no output generated")

    results["step01"] = result1
    las_after_step1 = result1["output"]

    # =========================================================
    # STEP 02: RECLASSIFICATION (LAS)
    # =========================================================
    print("STEP 02: Reclassification")

    rec_output = os.path.join(output_dir, "02_reclassified.las")

    result2 = step02(
        input_las=las_after_step1,
        footprint_shp=shp,
        output_las=rec_output
    )

    results["step02"] = result2
    las_after_step2 = result2["output"]

    # =========================================================
    # STEP 03: CITYJSON WALL MERGING (OPTIONAL)
    # =========================================================
    result3 = None
    cityjson_output = None

    if cityjson_input:
        print("STEP 03: Merge Wall Surfaces (CityJSON)")

        cityjson_output = os.path.join(output_dir, "03_merged.cityjson")

        result3 = step03(
            input_json=cityjson_input,
            output_json=cityjson_output,
            normal_threshold=5.0,
            distance_threshold=2.0,
            ground_tolerance=0.5
        )

    results["step03"] = result3

    # =========================================================
    # STEP 04: CUBEMAP PROCESSING (OPTIONAL, IMAGES)
    # =========================================================
    result4 = None
    cubemap_output = None

    if image_input_dir:
        print("STEP 04: Cubemap Processing")

        cubemap_output = os.path.join(output_dir, "04_cubemaps")

        result4 = step04(
            input_dir=image_input_dir,
            output_dir=cubemap_output,
            yaw=yaw,
            pitch=pitch,
            roll=roll
        )

    results["step04"] = result4

    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print("\n==============================")
    print("PIPELINE DONE")
    print("==============================\n")

    results["final_las"] = las_after_step2
    results["final_cityjson"] = cityjson_output
    results["final_cubemaps"] = cubemap_output

    return results


# =========================================================
# CLI ENTRY POINT
# =========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Omni2LOD3 Pipeline")

    # LAS pipeline
    parser.add_argument("--input", required=True, help="Input LAS file")
    parser.add_argument("--shp", required=True, help="Footprint shapefile")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--voxel", type=float, default=0.05)

    # CityJSON (optional)
    parser.add_argument("--cityjson", default=None)

    # Images (optional)
    parser.add_argument("--images", default=None)

    # Cubemap params
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--roll", type=float, default=0.0)

    args = parser.parse_args()

    run_pipeline(
        input_las=args.input,
        shp=args.shp,
        output_dir=args.output_dir,
        voxel_size=args.voxel,
        cityjson_input=args.cityjson,
        image_input_dir=args.images,
        yaw=args.yaw,
        pitch=args.pitch,
        roll=args.roll
    )