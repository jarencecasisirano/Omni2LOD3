import os
import argparse

from steps.downsampling import run as step01
from steps.reclassify import run as step02

from steps.merge_walls import run as step03
from steps.process_cubemaps import process_images as step04

from steps.glb_to_las import run as step05


def run_pipeline(
    input_las=None,
    shp=None,
    cityjson=None,
    images=None,
    glb_dir=None,
    output_dir="outputs",
    voxel_size=0.05,
    normal_threshold=5.0,
    distance_threshold=2.0,
    ground_tolerance=0.5,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    glb_samples=1000000,
):

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # ==================================================
    # SECTION A — LiDAR PREPROCESSING
    # ==================================================

    run_section_a = input_las is not None or shp is not None

    if run_section_a:

        print("\n" + "=" * 80)
        print("SECTION A — LiDAR Preprocessing")
        print("=" * 80)

        # --------------------------------------------------
        # STEP 01 + STEP 02 REQUIRE BOTH INPUTS
        # --------------------------------------------------

        if input_las is None:
            raise ValueError("SECTION A requires --input")

        if shp is None:
            raise ValueError("SECTION A requires --shp")

        # --------------------------------------------------
        # STEP 01 — DOWNSAMPLING
        # --------------------------------------------------

        print("\nSTEP 01: Downsampling")

        ds_output = os.path.join(output_dir, "01_downsample.las")

        result1 = step01(
            input_las=input_las, output_las=ds_output, voxel_size=voxel_size
        )

        results["step01"] = result1

        # --------------------------------------------------
        # STEP 02 — RECLASSIFICATION
        # --------------------------------------------------

        print("\nSTEP 02: Reclassification")

        rec_output = os.path.join(output_dir, "02_reclassified.las")

        result2 = step02(
            input_las=result1["output"], footprint_shp=shp, output_las=rec_output
        )

        results["step02"] = result2

    else:

        print("\nSkipping SECTION A")

    # ==================================================
    # SECTION B — CITYJSON
    # ==================================================

    run_section_b = cityjson is not None

    if run_section_b:

        print("\n" + "=" * 80)
        print("SECTION B — CityJSON Processing")
        print("=" * 80)

        # --------------------------------------------------
        # STEP 03 — WALL MERGING
        # --------------------------------------------------

        print("\nSTEP 03: Merge Wall Surfaces")

        merged_output = os.path.join(output_dir, "03_merged.cityjson")

        result3 = step03(
            input_json=cityjson,
            output_json=merged_output,
            normal_threshold=normal_threshold,
            distance_threshold=distance_threshold,
            ground_tolerance=ground_tolerance,
        )

        results["step03"] = result3

    else:

        print("\nSkipping SECTION B")

    # ==================================================
    # SECTION C — OMNI IMAGE PROCESSING
    # ==================================================

    run_section_c = images is not None

    if run_section_c:

        print("\n" + "=" * 80)
        print("SECTION C — Cubemap Generation")
        print("=" * 80)

        # --------------------------------------------------
        # STEP 04 — CUBEMAPS
        # --------------------------------------------------

        print("\nSTEP 04: Cubemap Processing")

        cubemap_output = os.path.join(output_dir, "04_cubemaps")

        os.makedirs(cubemap_output, exist_ok=True)

        step04(
            input_dir=images, output_dir=cubemap_output, yaw=yaw, pitch=pitch, roll=roll
        )

        results["step04"] = {"output_dir": cubemap_output}

    else:

        print("\nSkipping SECTION C")

    # ==================================================
    # SECTION D — GLB PROCESSING
    # ==================================================

    run_section_d = glb_dir is not None

    if run_section_d:

        print("\n" + "=" * 80)
        print("SECTION D — GLB to LAS Conversion")
        print("=" * 80)

        # --------------------------------------------------
        # STEP 05 — GLB TO LAS
        # --------------------------------------------------

        print("\nSTEP 05: GLB → LAS Conversion")

        glb_output = os.path.join(output_dir, "05_glb_las")

        result5 = step05(input_dir=glb_dir, output_dir=glb_output, samples=glb_samples)

        results["step05"] = result5

    else:

        print("\nSkipping SECTION D")

    # ==================================================
    # FINAL SUMMARY
    # ==================================================

    if not results:

        print("\nNo sections were executed.")

    else:

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)

        print("\nExecuted Steps:")

        for key in results.keys():
            print(f"  ✓ {key}")

    return results


# ======================================================
# CLI ENTRY
# ======================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # --------------------------------------------------
    # SECTION A — LIDAR
    # --------------------------------------------------

    parser.add_argument("--input")
    parser.add_argument("--shp")

    parser.add_argument("--voxel", type=float, default=0.05)

    # --------------------------------------------------
    # SECTION B — CITYJSON
    # --------------------------------------------------

    parser.add_argument("--cityjson")

    parser.add_argument("--normal_threshold", type=float, default=5.0)

    parser.add_argument("--distance_threshold", type=float, default=2.0)

    parser.add_argument("--ground_tolerance", type=float, default=0.5)

    # --------------------------------------------------
    # SECTION C — IMAGERY
    # --------------------------------------------------

    parser.add_argument("--images")

    parser.add_argument("--yaw", type=float, default=0.0)

    parser.add_argument("--pitch", type=float, default=0.0)

    parser.add_argument("--roll", type=float, default=0.0)

    # --------------------------------------------------
    # SECTION D — GLB
    # --------------------------------------------------

    parser.add_argument("--glb_dir")

    parser.add_argument("--glb_samples", type=int, default=1000000)

    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------

    parser.add_argument("--output_dir", default="outputs")

    args = parser.parse_args()

    run_pipeline(
        input_las=args.input,
        shp=args.shp,
        cityjson=args.cityjson,
        images=args.images,
        glb_dir=args.glb_dir,
        output_dir=args.output_dir,
        voxel_size=args.voxel,
        normal_threshold=args.normal_threshold,
        distance_threshold=args.distance_threshold,
        ground_tolerance=args.ground_tolerance,
        yaw=args.yaw,
        pitch=args.pitch,
        roll=args.roll,
        glb_samples=args.glb_samples,
    )
