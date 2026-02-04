# 03_normalize.py
"""
03_normalize.py
Height normalization using PDAL (HAG via Delaunay)
Converts Z to HeightAboveGround (normalized heights)
"""

import sys
import os
import subprocess
from pathlib import Path
import time

start_time = time.time()

# ------------------------------------------------------------------
# 1. Arguments (supplied by main.py)
# ------------------------------------------------------------------
if len(sys.argv) < 3:
    print("Usage: python 03_normalize.py <input_las> <output_las>")
    sys.exit(1)

INPUT_LAS  = Path(sys.argv[1])
OUTPUT_LAS = Path(sys.argv[2])

# ------------------------------------------------------------------
# 2. Ensure output folder exists
# ------------------------------------------------------------------
os.makedirs(OUTPUT_LAS.parent, exist_ok=True)

# ------------------------------------------------------------------
# 3. Run PDAL HAG normalization
# ------------------------------------------------------------------
print("\n=== Running height normalization (HAG) ===")
print(f"Input:  {INPUT_LAS}")
print(f"Output: {OUTPUT_LAS}")

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
    "--readers.las.filename=" + str(INPUT_LAS),
    "--writers.las.filename=" + str(OUTPUT_LAS)
]

try:
    subprocess.run(
        cmd,
        input=pipeline,
        text=True,
        check=True
    )
except subprocess.CalledProcessError as e:
    print("[ERROR] PDAL normalization failed.")
    sys.exit(1)

print("✔ Height normalization complete")

end_time = time.time()
print(f"=== Done! Normalization finished in {end_time - start_time:.2f} seconds ===")
