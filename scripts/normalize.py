# normalize.py
import laspy
import numpy as np
from pathlib import Path
import argparse

# === FIXED OUTPUT DIRECTORY ===
OUTPUT_DIR = Path(r"D:\Projects\Thesis\outputs\normalized")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_z(input_path, output_dir=None):
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}")
        return

    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{input_path.stem}_normalized{input_path.suffix}"

    print(f"Reading: {input_path.name}")

    # --- READ FULL FILE ---
    with laspy.open(input_path) as f:
        las = f.read()
        if len(las.points) == 0:
            print("[ERROR] Empty point cloud.")
            return

    scale_z = las.header.scale[2]
    offset_z = las.header.offset[2]

    # --- COMPUTE REAL Z VALUES ---
    z_real = las.z * scale_z + offset_z
    min_z_real = z_real.min()
    max_z_real = z_real.max()

    print(f"Original Z (real): {min_z_real: .6f} → {max_z_real: .6f} m")

    # If already >= 0, just copy
    if min_z_real >= 0:
        print("Already normalized (min Z >= 0). Copying...")
        las.write(output_path)
        print(f"Saved unchanged: {output_path.name}")
        return

    # --- SHIFT POINTS UP IN SCALED SPACE ---
    shift_scaled = -min_z_real / scale_z  # e.g., 1151.06 / 0.001 = 1,151,060
    las.z = (las.z + shift_scaled).astype(las.z.dtype)

    # --- UPDATE OFFSET: ADD THE POSITIVE SHIFT ---
    # We shifted points UP by |min_z|, so offset must increase by |min_z|
    las.header.offset[2] = offset_z - min_z_real  # 0 - (-1151.06) = +1151.06

    # --- RECOMPUTE HEADER BOUNDS FROM ACTUAL POINTS ---
    # X and Y
    for i, dim in enumerate(['x', 'y']):
        vals = getattr(las, dim)
        real_vals = vals * las.header.scale[i] + las.header.offset[i]
        las.header.min[i] = real_vals.min()
        las.header.max[i] = real_vals.max()

    # Z
    new_z_real = las.z * scale_z + las.header.offset[2]
    las.header.min[2] = new_z_real.min()
    las.header.max[2] = new_z_real.max()

    print(f"New Z (real)     : {las.header.min[2]: .6f} → {las.header.max[2]: .6f} m")
    print(f"Saving to        : {output_path.name}")

    # --- WRITE FILE ---
    las.write(output_path)

    # --- FINAL VERIFICATION ---
    with laspy.open(output_path) as f:
        chk = f.read()
        chk_z = chk.z * chk.header.scale[2] + chk.header.offset[2]
        print(f"Verified min Z   : {chk_z.min(): .6f}")
    print("Normalization complete!\n")


# === CLI / INTERACTIVE ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize Z → min = 0")
    parser.add_argument("input", nargs="?", help="Input .las/.laz file")
    parser.add_argument("-o", "--output_dir", help="Output directory")
    args = parser.parse_args()

    if args.input:
        normalize_z(args.input, args.output_dir)
    else:
        print("Drag & drop your LAS/LAZ file or paste path:")
        path = input("> ").strip().strip('"').strip("'")
        if path:
            normalize_z(path, args.output_dir)
        else:
            print("No file provided.")