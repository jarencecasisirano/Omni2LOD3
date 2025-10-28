# convert_to_gml.py
import subprocess
from pathlib import Path

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_FOLDER = Path(r"D:\Projects\Thesis\data")
CITYGML_TOOLS = "citygml-tools"  # global command

# -------------------------------------------------
# List .json files
# -------------------------------------------------
def list_json_files():
    return sorted([p for p in DATA_FOLDER.iterdir() if p.suffix.lower() == ".json"])

# -------------------------------------------------
# Convert selected file
# -------------------------------------------------
def convert_to_citygml(json_path: Path):
    output_gml = json_path.with_suffix(".gml")
    
    print(f"\nConverting:")
    print(f"   {json_path.name}")
    print(f"→ {output_gml.name}")

    cmd = [
        CITYGML_TOOLS,
        "cityjson2gml",
        str(json_path),
        str(output_gml)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("SUCCESS! CityGML saved:")
        print(f"   {output_gml}")
        print("\nYou can now open this in QGIS, FZK Viewer, or Blender!")
    except subprocess.CalledProcessError as e:
        print("ERROR during conversion:")
        print(e.stderr)
    except FileNotFoundError:
        print("citygml-tools not found. Run: npm install -g citygml-tools")

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    if not DATA_FOLDER.exists():
        print(f"Folder not found: {DATA_FOLDER}")
        return

    json_files = list_json_files()
    if not json_files:
        print("No .json files found.")
        return

    print("Select CityJSON file to convert to CityGML:")
    for i, p in enumerate(json_files):
        print(f"[{i}] {p.name}")

    while True:
        try:
            idx = int(input("Enter index of file: ").strip())
            if 0 <= idx < len(json_files):
                selected = json_files[idx]
                convert_to_citygml(selected)
                break
            else:
                print("Invalid index.")
        except ValueError:
            print("Enter a number.")

if __name__ == "__main__":
    main()