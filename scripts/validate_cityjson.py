# validate_cityjson.py
import json
from pathlib import Path

# -------------------------------------------------
# CONFIG: folder where your CityJSON files are
# -------------------------------------------------
DATA_FOLDER = Path(r"D:\Projects\Thesis\data")

# -------------------------------------------------
# Tiny validator
# -------------------------------------------------
def validate_cityjson(file_path: Path) -> bool:
    try:
        with file_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        return False

    required = {"CityObjects", "vertices", "type", "version"}
    if not required.issubset(data.keys()):
        print(f"Missing keys: {required - data.keys()}")
        return False

    if data["type"] != "CityJSON":
        print(f'"type" must be "CityJSON", got: {data["type"]}')
        return False

    if data["version"] not in ("1.0", "1.1", "2.0"):
        print(f'Unsupported version: {data["version"]}')
        return False

    if not isinstance(data["vertices"], list) or len(data["vertices"]) == 0:
        print("vertices must be non-empty list")
        return False

    print(f"VALID -> {file_path.name}")
    return True


# -------------------------------------------------
# List .json files and let user pick
# -------------------------------------------------
def main():
    if not DATA_FOLDER.exists():
        print(f"Folder not found: {DATA_FOLDER}")
        return

    json_files = sorted([p for p in DATA_FOLDER.iterdir() if p.suffix.lower() == ".json"])
    if not json_files:
        print("No .json files found in:", DATA_FOLDER)
        return

    print("Select CityJSON file to validate:")
    for i, p in enumerate(json_files):
        print(f"[{i}] {p.name}")

    while True:
        try:
            idx = int(input("Enter index of file: ").strip())
            if 0 <= idx < len(json_files):
                selected = json_files[idx]
                validate_cityjson(selected)
                break
            else:
                print("Invalid index. Try again.")
        except ValueError:
            print("Please enter a number.")


if __name__ == "__main__":
    main()