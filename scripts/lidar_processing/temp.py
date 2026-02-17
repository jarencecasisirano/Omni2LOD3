from pathlib import Path
import subprocess
import sys

import laspy


PROJECT_ROOT = Path(r"C:\Projects\Omni2LOD3")
INPUT_DIR = PROJECT_ROOT / r"data\01_point_cloud\clipped"
OUTPUT_DIR = PROJECT_ROOT / r"data\01_point_cloud\down"
DOWNSAMPLER = PROJECT_ROOT / r"scripts\las_to_lod2\01_downsampling.py"
VOXEL_SIZE = 0.04


def main() -> None:
    if not DOWNSAMPLER.exists():
        raise FileNotFoundError(f"Downsampling script not found: {DOWNSAMPLER}")
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = sorted(INPUT_DIR.glob("*.las"))
    if not input_files:
        raise FileNotFoundError(f"No LAS files found in: {INPUT_DIR}")

    results = []

    for input_las in input_files:
        output_las = OUTPUT_DIR / input_las.name
        print(f"\n=== Downsampling: {input_las.name} (voxel={VOXEL_SIZE}) ===")

        cmd = [
            sys.executable,
            str(DOWNSAMPLER),
            str(input_las),
            str(output_las),
            str(VOXEL_SIZE),
        ]

        subprocess.run(cmd, check=True)

        down_las = laspy.read(output_las)
        point_count = len(down_las.points)
        results.append((output_las.name, point_count))

    print("\nDownsampling complete. Point count per downsampled LAS:")
    for name, point_count in results:
        print(f"- {name}: {point_count} points")


if __name__ == "__main__":
    main()
