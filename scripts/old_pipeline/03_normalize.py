# normalize_height.py
# Height normalization using PDAL (NO external JSON)

import subprocess
from pathlib import Path

def normalize_hag(input_las: Path, output_las: Path):
    print(f"\nNormalizing height:")
    print(f"  {input_las.name} → {output_las.name}")

    pipeline = r"""
    {
      "pipeline": [
        { "type": "readers.las" },
        { "type": "filters.hag_delaunay" },
        {
          "type": "filters.ferry",
          "dimensions": "HeightAboveGround=>Z"
        },
        {
          "type": "writers.las",
          "extra_dims": "all"
        }
      ]
    }
    """

    cmd = [
        "pdal", "pipeline",
        "--stdin",
        "--readers.las.filename=" + str(input_las),
        "--writers.las.filename=" + str(output_las)
    ]

    subprocess.run(
        cmd,
        input=pipeline,
        text=True,
        check=True
    )

    print("✔ Height normalization complete")

if __name__ == "__main__":
    normalize_hag(
        Path(r"C:\Projects\Thesis\outputs\clipped\nimmb_zclean.las"),
        Path(r"C:\Projects\Thesis\outputs\normalized\nimmb_hag.las")
    )
