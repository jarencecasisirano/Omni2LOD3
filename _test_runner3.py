import sys
import io

with open("scripts/LOD2toLOD3/11B_extrusions.py", "r", encoding="utf-8") as f:
    code = f.read()

mock_func = """
def select_file(directory, pattern="*.las"):
    if "13_openings_json" in directory:
        return "outputs/13_openings_json/ICHEM-final-6-best.json"
    elif "11B_flat" in directory:
        return "outputs/11B_flat/ICHEM-extrusion.las"
    return None
"""

# Replace the select_file definition block
import re
code = re.sub(r'def select_file.*?try again\."\n', mock_func, code, flags=re.DOTALL)

# Let's run it
try:
    exec(code, {'__name__': '__main__', '__file__': 'scripts/LOD2toLOD3/11B_extrusions.py'})
except Exception as e:
    import traceback
    traceback.print_exc()
