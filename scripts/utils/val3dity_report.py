import json
from pathlib import Path


def extract_error_codes(node) -> set[int]:
    codes = set()

    def walk(n):
        if isinstance(n, dict):
            for key, value in n.items():
                key_lower = str(key).lower()
                if "error" in key_lower or "code" in key_lower:
                    if isinstance(value, int):
                        codes.add(value)
                    elif isinstance(value, str) and value.strip().isdigit():
                        codes.add(int(value.strip()))
                walk(value)
            return

        if isinstance(n, list):
            for item in n:
                walk(item)

    walk(node)
    return codes


def load_report_json(report_json_path: Path | str) -> dict:
    path = Path(report_json_path)
    if not path.exists():
        raise FileNotFoundError(f"val3dity report not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not parse val3dity report JSON: {exc}") from exc


def load_report_error_codes(report_json_path: Path | str) -> set[int]:
    return extract_error_codes(load_report_json(report_json_path))
