import sys
import io

sys.stdin = io.StringIO("1\n8\n")

with open("scripts/LOD2toLOD3/11B_extrusions.py", "r", encoding="utf-8") as f:
    code = f.read()

exec(code, {'__name__': '__main__', '__file__': 'scripts/LOD2toLOD3/11B_extrusions.py'})
