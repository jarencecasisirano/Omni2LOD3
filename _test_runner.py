import sys
import os
from unittest.mock import patch

# Add directory to sys.path so we can import the script
sys.path.append(os.path.abspath('scripts/LOD2toLOD3'))
import importlib

extrusions = importlib.import_module("11B_extrusions")

def mock_select_file(directory, pattern="*.las"):
    if "13_openings_json" in directory:
        return "outputs/13_openings_json/ICHEM-final-6-best.json"
    elif "11B_flat" in directory:
        return "outputs/11B_flat/ICHEM-extrusion.las"
    return None

with patch.object(extrusions, 'select_file', mock_select_file):
    extrusions.main()
