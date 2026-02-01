import laspy
import numpy as np
from pathlib import Path

def clip_z_outliers(
    input_las,
    output_las,
    z_min=-50,
    z_max=200
):
    print(f"Reading: {input_las}")
    las = laspy.read(input_las)

    z = las.z
    mask = (z >= z_min) & (z <= z_max)

    print(f"Original points: {len(z)}")
    print(f"Kept points: {np.sum(mask)}")
    print(f"Removed points: {len(z) - np.sum(mask)}")

    clipped_las = laspy.create(
        point_format=las.header.point_format,
        file_version=las.header.version
    )

    clipped_las.header = las.header

    for dim in las.point_format.dimension_names:
        setattr(
            clipped_las,
            dim,
            getattr(las, dim)[mask]
        )

    clipped_las.write(output_las)
    print(f"Saved: {output_las}")

if __name__ == "__main__":
    input_las = Path(
        r"C:\Projects\Omni2LOD3\outputs\00_archive\01_test_downsampled.las"
    )

    output_las = Path(
        r"C:\Projects\Omni2LOD3\outputs\00_archive\01_test_clipped.las"
    )
    output_las.parent.mkdir(parents=True, exist_ok=True)

    clip_z_outliers(input_las, output_las)