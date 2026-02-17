# 03_validate.py
import re
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io_helpers import choose_index, list_json_files

# -------------------------
# Path defaults (relative to repo)
# -------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "03_json_model"
REPORT_DIR = PROJECT_ROOT / "outputs" / "03_val3dity"
VAL3DITY_EXE = PROJECT_ROOT / "tools" / "val3dity" / "val3dity-win64" / "val3dity.exe"

# Exit codes
EXIT_VALID = 0
EXIT_INVALID = 2
EXIT_FAILURE = 1


def _parse_validity(output_text: str):
    """
    Parse val3dity summary output.
    Returns EXIT_VALID / EXIT_INVALID / EXIT_FAILURE.
    """
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


def run_val3dity(input_path: Path):
    if not VAL3DITY_EXE.exists():
        print(f"[ERROR] val3dity.exe not found: {VAL3DITY_EXE}")
        return EXIT_FAILURE, None, None

    if not input_path.exists():
        print(f"[ERROR] Input JSON not found: {input_path}")
        return EXIT_FAILURE, None, None

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_txt_path = REPORT_DIR / f"{input_path.stem}_val3dity.txt"
    report_json_path = REPORT_DIR / f"{input_path.stem}_val3dity.json"

    cmd = [str(VAL3DITY_EXE), str(input_path), "--report", str(report_json_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    combined = stdout if not stderr else (stdout + "\n" + stderr)

    report_txt_path.write_text(combined, encoding="utf-8")

    if result.returncode != 0:
        print(f"[ERROR] val3dity failed with exit code {result.returncode}")
        return EXIT_FAILURE, report_txt_path, report_json_path

    status = _parse_validity(combined)
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
    sys.exit(status)


if __name__ == "__main__":
    main()
