import os
import glob
import argparse
import numpy as np

# Use OpenEXR for reading (same lib used in 05_las_to_exr.py)
import OpenEXR
import Imath

# Pillow for PNG writing
from PIL import Image


def read_exr_channel(exr_file, channel_name, width, height):
    """Read a single FLOAT channel from an open EXR file and return as (H,W) ndarray."""
    raw = exr_file.channel(channel_name, Imath.PixelType(Imath.PixelType.FLOAT))
    arr = np.frombuffer(raw, dtype=np.float32).reshape((height, width))
    return arr


def exr_to_png(exr_path, out_path):
    """
    Convert an EXR file that contains R, G, B (and X, Y, Z) channels into a
    standard 8-bit PNG.  Pixels that have no point data (stored as NaN) are
    rendered as black.
    """
    exr = OpenEXR.InputFile(exr_path)
    header = exr.header()

    dw = header['dataWindow']
    width  = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    r = read_exr_channel(exr, 'R', width, height)
    g = read_exr_channel(exr, 'G', width, height)
    b = read_exr_channel(exr, 'B', width, height)
    exr.close()

    # Replace NaN (empty pixels) with 0
    r = np.nan_to_num(r, nan=0.0)
    g = np.nan_to_num(g, nan=0.0)
    b = np.nan_to_num(b, nan=0.0)

    # Clamp to [0, 1] and scale to uint8
    r = np.clip(r, 0.0, 1.0)
    g = np.clip(g, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)

    rgb = np.stack([r, g, b], axis=-1)
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    img = Image.fromarray(rgb_uint8, mode='RGB')
    img.save(out_path)
    print(f"  Saved: {out_path}  ({width}x{height} px)")


def process_folder(input_folder, output_base):
    """
    Convert all EXR files inside *input_folder* (including sub-folders) to PNG,
    mirroring the directory structure under *output_base*.
    """
    exr_files = glob.glob(os.path.join(input_folder, "**", "*.exr"), recursive=True)

    if not exr_files:
        print(f"No .exr files found under: {input_folder}")
        return

    print(f"Found {len(exr_files)} EXR file(s). Converting...")

    for exr_path in sorted(exr_files):
        # Mirror sub-folder structure
        rel_path = os.path.relpath(exr_path, input_folder)
        png_rel   = os.path.splitext(rel_path)[0] + ".png"
        out_path  = os.path.join(output_base, png_rel)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        print(f"  Processing: {rel_path}")
        try:
            exr_to_png(exr_path, out_path)
        except Exception as e:
            print(f"  ERROR converting {exr_path}: {e}")

    print("\nAll conversions complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert EXR images (from outputs/07_exr_image) to PNG (outputs/08_png_image)."
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        default="outputs/07_exr_image",
        help="Root folder containing .exr files (default: outputs/07_exr_image)."
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default="outputs/08_png_image",
        help="Root folder for output .png files (default: outputs/08_png_image)."
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.")
        return

    os.makedirs(args.output, exist_ok=True)
    process_folder(args.input, args.output)


if __name__ == "__main__":
    main()
