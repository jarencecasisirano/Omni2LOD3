import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.schema_profile import resolve_schema_defaults


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def _rel(pathlike):
    p = Path(pathlike)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(p)


def _run_cjio_validate(input_path: Path):
    try:
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        return subprocess.run(
            ["cjio", str(input_path), "validate"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except FileNotFoundError:
        return None


def _combined_text(run_result):
    if run_result is None:
        return ""
    return (run_result.stdout or "") + "\n" + (run_result.stderr or "")


def _parse_cjio_sections(run_result):
    """
    Parse cjio validate sections like:
      === json_syntax ===
      ok
      === schema ===
      ...
    """
    text = _combined_text(run_result)
    lines = text.splitlines()
    sections = {}
    current = None
    for raw in lines:
        line = raw.strip()
        if line.startswith("===") and line.endswith("==="):
            name = line.strip("=").strip().lower().replace(" ", "_")
            current = name
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)
    return sections


def _section_status(lines):
    vals = [x for x in lines if x]
    if not vals:
        return "VALID"
    if len(vals) == 1 and vals[0].lower() == "ok":
        return "VALID"
    return "INVALID"


def _core_schema_status(run_result):
    if run_result is None:
        return "UNAVAILABLE", "UNAVAILABLE"
    text = _combined_text(run_result).lower()
    if "error:" in text and "can't encode character" in text:
        return "CHECK_REPORT", "CHECK_REPORT"

    sections = _parse_cjio_sections(run_result)
    json_status = _section_status(sections.get("json_syntax", []))
    schema_status = _section_status(sections.get("schema", []))
    return json_status, schema_status


def _apply_metadata_schema_fixes(data, defaults):
    changed = False
    applied = []

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        data["metadata"] = metadata
        changed = True
        applied.append("metadata")

    poc = metadata.get("pointOfContact")
    if not isinstance(poc, dict):
        poc = {}
        metadata["pointOfContact"] = poc
        changed = True
        applied.append("metadata.pointOfContact")

    if not isinstance(poc.get("contactName"), str) or not poc.get("contactName", "").strip():
        poc["contactName"] = defaults["contactName"]
        changed = True
        applied.append("metadata.pointOfContact.contactName")

    if not isinstance(poc.get("emailAddress"), str) or not poc.get("emailAddress", "").strip():
        poc["emailAddress"] = defaults["emailAddress"]
        changed = True
        applied.append("metadata.pointOfContact.emailAddress")

    ref_date = metadata.get("referenceDate")
    if ref_date != defaults["referenceDate"]:
        metadata["referenceDate"] = defaults["referenceDate"]
        changed = True
        applied.append("metadata.referenceDate")

    return changed, applied


def main():
    if len(sys.argv) < 3:
        print("Usage: python 04_schema_fix.py <input_json> <output_json>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    if len(sys.argv) > 3:
        print("Usage: python 04_schema_fix.py <input_json> <output_json>")
        sys.exit(1)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    defaults = resolve_schema_defaults()
    changed, applied = _apply_metadata_schema_fixes(data, defaults)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    check = _run_cjio_validate(output_path)
    json_status, schema_status = _core_schema_status(check)

    print("\n=== RUNNING CJIO SCHEMA VALIDATION ===")
    if changed:
        print(
            f"Applied schema metadata fix for input \"{Path(input_path).name}\": "
            f"yes ({', '.join(applied)})"
        )
    else:
        print(f"Applied schema metadata fix for input \"{Path(input_path).name}\": no")
    print(f"JSON syntax:      {json_status}")
    print(f"CityJSON schema:  {schema_status}")
    print(f"Output:           {_rel(output_path)}")
    if check is None:
        print("[WARN] cjio command is unavailable in this Python environment.")
        print("[WARN] Activate the environment where cjio/cjvalpy are installed.")


if __name__ == "__main__":
    main()

