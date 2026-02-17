import argparse
from pathlib import Path
import sys

import laspy


DATA_POINT_CLOUD_DIR = Path(r"C:\Projects\Omni2LOD3\data\01_point_cloud")
INPUT_DOWN_DIR = DATA_POINT_CLOUD_DIR / "down"
OUTPUTS_DIR = Path(r"C:\Projects\Omni2LOD3\outputs")
OUTPUT_XYZ_DIR = DATA_POINT_CLOUD_DIR / "xyz"

LOW_RGB = (48, 4, 4)        # deeper dark red for low elevations
HIGH_RGB = (210, 95, 20)    # dark orange


def _lerp_color(t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    r = round(LOW_RGB[0] + (HIGH_RGB[0] - LOW_RGB[0]) * t)
    g = round(LOW_RGB[1] + (HIGH_RGB[1] - LOW_RGB[1]) * t)
    b = round(LOW_RGB[2] + (HIGH_RGB[2] - LOW_RGB[2]) * t)
    return r, g, b


def _normalize(value: float, minimum: float, maximum: float) -> float:
    span = maximum - minimum
    if span <= 0:
        return 0.0
    return (value - minimum) / span


def _find_las_files(path: Path, recursive: bool = False) -> list[Path]:
    patterns = ("*.las", "*.laz")
    files: list[Path] = []
    for pattern in patterns:
        if recursive:
            files.extend(path.rglob(pattern))
        else:
            files.extend(path.glob(pattern))
    return sorted(p for p in files if p.is_file())


def _choose_index(prompt: str, count: int) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            idx = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if 0 <= idx < count:
            return idx
        print(f"Choose a number from 0 to {count - 1}.")


def _choose_data_or_outputs() -> str:
    print("Which folder do you want to load LAS/LAZ from?")
    print("[0] data")
    print("[1] outputs")
    idx = _choose_index("Select source [0-1]: ", 2)
    return "data" if idx == 0 else "outputs"


def _choose_input_las_interactive() -> Path:
    source = _choose_data_or_outputs()

    if source == "data":
        files = _find_las_files(DATA_POINT_CLOUD_DIR, recursive=False)
        if not files:
            raise FileNotFoundError(f"No LAS/LAZ files found in {DATA_POINT_CLOUD_DIR}")
        print("\nLAS/LAZ files in data/01_point_cloud:")
        for i, file in enumerate(files):
            print(f"[{i}] {file.name}")
        selected = _choose_index("Select LAS/LAZ file: ", len(files))
        return files[selected]

    subfolders = sorted(
        p
        for p in OUTPUTS_DIR.iterdir()
        if p.is_dir() and _find_las_files(p, recursive=True)
    )
    if not subfolders:
        raise FileNotFoundError(
            f"No subfolders with LAS/LAZ files found in {OUTPUTS_DIR}"
        )

    print("\nSubfolders in outputs with LAS/LAZ files:")
    for i, folder in enumerate(subfolders):
        print(f"[{i}] {folder.name}")
    folder_idx = _choose_index("Select subfolder: ", len(subfolders))
    chosen_folder = subfolders[folder_idx]

    files = _find_las_files(chosen_folder, recursive=True)
    print(f"\nLAS/LAZ files in {chosen_folder.name}:")
    for i, file in enumerate(files):
        rel = file.relative_to(OUTPUTS_DIR)
        print(f"[{i}] {rel}")
    file_idx = _choose_index("Select LAS/LAZ file: ", len(files))
    return files[file_idx]


def _choose_color_mode_interactive() -> str:
    print("\nColor mode:")
    print("[0] none (XYZ only)")
    print("[1] elevation (XYZRGB gradient)")
    idx = _choose_index("Select color mode [0-1]: ", 2)
    return "none" if idx == 0 else "elevation"


def _default_output_path(input_las: Path) -> Path:
    return OUTPUT_XYZ_DIR / f"{input_las.stem}.xyz"


def las_to_xyz(
    input_las: Path,
    output_xyz: Path | None = None,
    chunk_size: int = 1_000_000,
    color_by: str = "none",
) -> int:
    if not input_las.exists():
        raise FileNotFoundError(f"Input LAS not found: {input_las}")

    if output_xyz is None:
        output_xyz = _default_output_path(input_las)
    output_xyz.parent.mkdir(parents=True, exist_ok=True)

    if color_by == "elevation":
        with laspy.open(input_las) as reader:
            value_min = float(reader.header.mins[2])
            value_max = float(reader.header.maxs[2])
    else:
        value_min, value_max = 0.0, 1.0

    total_written = 0
    with laspy.open(input_las) as reader, output_xyz.open("w", encoding="utf-8") as out:
        for points in reader.chunk_iterator(chunk_size):
            x = points.x
            y = points.y
            z = points.z
            if color_by == "none":
                for xi, yi, zi in zip(x, y, z):
                    out.write(f"{xi:.6f} {yi:.6f} {zi:.6f}\n")
            else:
                values = z
                for xi, yi, zi, vi in zip(x, y, z, values):
                    t = _normalize(float(vi), value_min, value_max)
                    r, g, b = _lerp_color(t)
                    out.write(f"{xi:.6f} {yi:.6f} {zi:.6f} {r} {g} {b}\n")
            total_written += len(points)

    print(f"Input:  {input_las}")
    print(f"Output: {output_xyz}")
    print(f"Color mode: {color_by}")
    print(f"Points written: {total_written:,}")
    return total_written


def convert_all_down_to_xyz(
    input_dir: Path = INPUT_DOWN_DIR,
    output_dir: Path = OUTPUT_XYZ_DIR,
    chunk_size: int = 1_000_000,
) -> None:
    files = _find_las_files(input_dir, recursive=False)
    if not files:
        raise FileNotFoundError(f"No LAS/LAZ files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(files)} LAS/LAZ file(s) in: {input_dir}")
    print(f"Output folder: {output_dir}\n")

    results: list[tuple[str, int]] = []
    for input_las in files:
        output_xyz = output_dir / f"{input_las.stem}.xyz"
        written = las_to_xyz(
            input_las=input_las,
            output_xyz=output_xyz,
            chunk_size=chunk_size,
            color_by="elevation",
        )
        results.append((output_xyz.name, written))
        print("")

    print("Conversion complete. Point count per XYZ file:")
    for name, count in results:
        print(f"- {name}: {count:,} points")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LAS/LAZ point cloud(s) to XYZ/XYZRGB text format."
    )
    parser.add_argument(
        "--batch-down",
        action="store_true",
        help="Convert all LAS/LAZ from data/01_point_cloud/down to data/01_point_cloud/xyz using elevation color",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input LAS/LAZ file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output XYZ file path (default: data/01_point_cloud/xyz/<input_stem>.xyz)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of points read per chunk",
    )
    parser.add_argument(
        "--color-by",
        choices=["none", "elevation"],
        default=None,
        help="Write plain XYZ or XYZRGB colorized by elevation",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Do not prompt; requires --input and optional --color-by/--output",
    )
    args = parser.parse_args()

    if args.batch_down or len(sys.argv) == 1:
        convert_all_down_to_xyz(chunk_size=args.chunk_size)
        return

    if args.non_interactive:
        if args.input is None:
            raise ValueError("--non-interactive requires --input")
        color_by = args.color_by or "none"
        las_to_xyz(args.input, args.output, args.chunk_size, color_by)
        return

    input_las = args.input if args.input is not None else _choose_input_las_interactive()
    color_by = args.color_by if args.color_by is not None else _choose_color_mode_interactive()
    las_to_xyz(input_las, args.output, args.chunk_size, color_by)


if __name__ == "__main__":
    main()
