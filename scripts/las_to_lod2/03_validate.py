# 03_validate.py
"""
CLI mode:
  python 03_validate.py <input_json>
"""
import re
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io_helpers import choose_index, list_json_files
from utils.las_helpers import extract_prefix
from utils.paths import DATA_JSON_DIR, OUT_VAL3DITY, VAL3DITY_EXE as VAL3DITY_EXE_PATH

DEFAULT_INPUT_DIR = Path(DATA_JSON_DIR)
REPORT_DIR = Path(OUT_VAL3DITY)
VAL3DITY_EXE = Path(VAL3DITY_EXE_PATH)

EXIT_VALID = 0
EXIT_INVALID = 2
EXIT_FAILURE = 1

def _print_failure_help():
    print("\t[INFO] Validation could not be completed (tool/runtime/parsing issue),")
    print("\tnot a confirmed geometry INVALID result.")


def _print_failure_help_tool():
    print(f"\t[INFO] Expected val3dity executable: {VAL3DITY_EXE}")
    print("\t[INFO] Try reinstalling val3dity if needed:")
    print("\thttps://github.com/tudelft3d/val3dity")

def _parse_validity(output_text: str):
    for line in output_text.splitlines():
        if re.search(r"^\s*INVALID\b", line, re.IGNORECASE):
            return EXIT_INVALID
        if re.search(r"^\s*VALID\b", line, re.IGNORECASE):
            return EXIT_VALID
    return EXIT_FAILURE


def _extract_errors_present_block(output_text: str):
    lines = output_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Errors present:"):
            start = i
            break

    if start is None:
        return []

    block = [lines[start]]
    for line in lines[start + 1:]:
        if not line.strip():
            break
        if line.startswith(" ") or line.startswith("\t"):
            block.append(line)
            continue
        break
    return block


def _print_validation_summary(in_path: Path, report_txt_path: Path, status: int):
    print(f"Input:  {in_path}")
    print(f"Report saved to folder: {report_txt_path.parent}")
    if status == EXIT_VALID:
        print("\tResult: VALID")
        return
    if status == EXIT_INVALID:
        print("\tResult: INVALID")
        output_text = report_txt_path.read_text(encoding="utf-8", errors="replace")
        errors_block = _extract_errors_present_block(output_text)
        if errors_block:
            print("\tErrors present:")
            for line in errors_block[1:]:
                print(f"\t{line}")
        return
    print("\tResult: FAILURE")
    _print_failure_help()


def run_val3dity(input_path: Path):
    if not VAL3DITY_EXE.exists():
        print(f"[ERROR] val3dity.exe not found: {VAL3DITY_EXE}")
        _print_failure_help()
        _print_failure_help_tool()
        return EXIT_FAILURE, None, None

    if not input_path.exists():
        print(f"[ERROR] Input JSON not found: {input_path}")
        _print_failure_help()
        return EXIT_FAILURE, None, None

    prefix_dir = REPORT_DIR / extract_prefix(str(input_path)).upper()
    prefix_dir.mkdir(parents=True, exist_ok=True)
    report_txt_path = prefix_dir / f"{input_path.stem}_val3dity.txt"
    report_json_path = prefix_dir / f"{input_path.stem}_val3dity.json"

    cmd = [str(VAL3DITY_EXE), str(input_path), "--report", str(report_json_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    combined = stdout if not stderr else (stdout + "\n" + stderr)

    report_txt_path.write_text(combined, encoding="utf-8")

    if result.returncode != 0:
        print(f"[ERROR] val3dity failed with exit code {result.returncode}")
        _print_failure_help()
        _print_failure_help_tool()
        return EXIT_FAILURE, report_txt_path, report_json_path

    status = _parse_validity(combined)
    if status == EXIT_FAILURE:
        print("[ERROR] Could not parse VALID/INVALID status from val3dity output.")
        _print_failure_help()
    return status, report_txt_path, report_json_path


def main():
    # CLI mode
    if len(sys.argv) >= 2:
        in_path = Path(sys.argv[1])
        status, report_txt_path, report_json_path = run_val3dity(in_path)
        if report_txt_path:
            _print_validation_summary(in_path, report_txt_path, status)
        else:
            print(f"Input:  {in_path}")
            print("Result: FAILURE")
            _print_failure_help()
        sys.exit(status)

    # Interactive mode
    input_dir = DEFAULT_INPUT_DIR
    print("\n=== val3dity validation (INTERACTIVE MODE) ===")
    print(f"Input directory: {input_dir}")

    files = [Path(p) for p in list_json_files(input_dir)]
    if not files:
        print(f"[ERROR] No JSON files found in: {input_dir}")
        sys.exit(EXIT_FAILURE)

    print("\nFound JSON file(s):")
    for i, p in enumerate(files):
        size_kb = p.stat().st_size / 1024.0
        print(f"  [{i}] {p.name:<35} ({size_kb:,.1f} KB)")

    idx = choose_index(len(files), f"Select file to validate [0-{len(files)-1}]: ")
    if idx is None:
        print("[ERROR] Invalid selection.")
        sys.exit(EXIT_FAILURE)

    in_path = files[idx]
    status, report_txt_path, report_json_path = run_val3dity(in_path)
    if report_txt_path:
        _print_validation_summary(in_path, report_txt_path, status)
    else:
        print(f"Input:  {in_path}")
        print("Result: FAILURE")
        _print_failure_help()
    sys.exit(status)


if __name__ == "__main__":
    main()
