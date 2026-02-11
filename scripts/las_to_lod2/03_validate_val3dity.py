import json
import re
import sys
import subprocess
from pathlib import Path

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


def list_json_files(folder: Path):
    if not folder.exists():
        return []
    return sorted(folder.glob("*.json"))


def choose_index(n: int, prompt: str):
    choice = input(prompt).strip()
    if not choice.isdigit():
        return None
    idx = int(choice)
    if idx < 0 or idx >= n:
        return None
    return idx


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


def _summarize_report_json(report_path: Path):
    """
    Best-effort summary from val3dity JSON report.
    Returns dict with 'codes' list (may be empty) and 'note'.
    """
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"codes": [], "note": f"[WARN] Could not parse JSON report: {e}"}

    codes = {}

    def walk(node, key_hint=""):
        if isinstance(node, dict):
            for k, v in node.items():
                kh = f"{key_hint}.{k}" if key_hint else k
                if isinstance(v, int) and ("error" in k.lower() or "code" in k.lower()):
                    codes[v] = codes.get(v, 0) + 1
                else:
                    walk(v, kh)
        elif isinstance(node, list):
            for item in node:
                walk(item, key_hint)

    walk(data)

    return {"codes": sorted(codes.items()), "note": ""}


def run_val3dity(input_path: Path):
    if not VAL3DITY_EXE.exists():
        print(f"[ERROR] val3dity.exe not found: {VAL3DITY_EXE}")
        return EXIT_FAILURE, None

    if not input_path.exists():
        print(f"[ERROR] Input JSON not found: {input_path}")
        return EXIT_FAILURE, None

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
        print("\n=== val3dity validation (CLI MODE) ===")
        print(f"Input:  {in_path}")

        status, report_txt_path, report_json_path = run_val3dity(in_path)
        if report_txt_path:
            print(f"Report (text): {report_txt_path}")
        if report_json_path and report_json_path.exists():
            print(f"Report (json): {report_json_path}")
            summary = _summarize_report_json(report_json_path)
            if summary["note"]:
                print(summary["note"])
            if summary["codes"]:
                codes = ", ".join([f"{c} (x{n})" for c, n in summary["codes"]])
                print(f"JSON codes: {codes}")

        if status == EXIT_VALID:
            print("Result: VALID")
        elif status == EXIT_INVALID:
            print("Result: INVALID")
        else:
            print("Result: FAILURE")
        sys.exit(status)

    # Interactive mode
    input_dir = DEFAULT_INPUT_DIR
    print("\n=== val3dity validation (INTERACTIVE MODE) ===")
    print(f"Input directory: {input_dir}")

    files = list_json_files(input_dir)
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
        print(f"Report (text): {report_txt_path}")
    if report_json_path and report_json_path.exists():
        print(f"Report (json): {report_json_path}")
        summary = _summarize_report_json(report_json_path)
        if summary["note"]:
            print(summary["note"])
        if summary["codes"]:
            codes = ", ".join([f"{c} (x{n})" for c, n in summary["codes"]])
            print(f"JSON codes: {codes}")

    if status == EXIT_VALID:
        print("Result: VALID")
    elif status == EXIT_INVALID:
        print("Result: INVALID")
    else:
        print("Result: FAILURE")
    sys.exit(status)


if __name__ == "__main__":
    main()
