# cityjson_to_gml2.py
# Convert CityJSON → CityGML 2.0 using citygml-tools (OFFICIAL FLAG)

import subprocess
from pathlib import Path

DATA_FOLDER = Path(r"D:\Projects\Thesis\data")
CITYGML_TOOLS_BAT = Path(r"D:\Projects\Thesis\tools\citygml-tools-2.4.0\citygml-tools-2.4.0\citygml-tools.bat")

def list_json_files():
    return sorted([p for p in DATA_FOLDER.iterdir() if p.suffix.lower() == ".json"])

def convert_to_citygml2(json_path: Path):
    output_gml = DATA_FOLDER / f"{json_path.stem}_v2.gml"

    print(f"\nConverting to CityGML 2.0:")
    print(f"   {json_path.name}")
    print(f"→ {output_gml.name}")

    cmd = [
        str(CITYGML_TOOLS_BAT),
        "from-cityjson",
        str(json_path),
        "-v", "2.0",           # THIS IS THE CORRECT FLAG
        "-o", str(output_gml)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("SUCCESS! CityGML 2.0 saved:")
        print(f"   {output_gml}")
        print("\nSEND THIS TO YOUR PARTNER. DONE.")
    except subprocess.CalledProcessError as e:
        print("ERROR:")
        print(e.stderr)
    except FileNotFoundError:
        print(f"Tool not found: {CITYGML_TOOLS_BAT}")

def main():
    if not DATA_FOLDER.exists():
        print(f"Folder not found: {DATA_FOLDER}")
        return
    json_files = list_json_files()
    if not json_files:
        print("No .json files.")
        return
    print("Select file:")
    for i, p in enumerate(json_files):
        print(f"[{i}] {p.name}")
    while True:
        try:
            idx = int(input("Enter index: ").strip())
            if 0 <= idx < len(json_files):
                convert_to_citygml2(json_files[idx])
                break
        except ValueError:
            print("Number.")

if __name__ == "__main__":
    main()