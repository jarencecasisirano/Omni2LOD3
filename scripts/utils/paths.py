import os

# ======================= PATHS =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_ROOT, ".."))

DATA_PC_DIR = os.path.join(PROJECT_ROOT, "data", "01_point_cloud")
DATA_SHP_DIR = os.path.join(PROJECT_ROOT, "data", "02_footprint")
DATA_JSON_DIR = os.path.join(PROJECT_ROOT, "data", "03_json_model")

OUT_INFO = os.path.join(PROJECT_ROOT, "outputs", "00_las_info")
OUT_DOWNSAMPLED = os.path.join(PROJECT_ROOT, "outputs", "01_downsampled")
OUT_RECLASSIFIED = os.path.join(PROJECT_ROOT, "outputs", "02_reclassified")

OUTPUT_DIRS = (
    OUT_INFO,
    OUT_DOWNSAMPLED,
    OUT_RECLASSIFIED,
)

SCRIPT_INSPECT = os.path.join(SCRIPTS_ROOT, "las_to_lod2", "inspect_las.py")
SCRIPT_DOWN = os.path.join(SCRIPTS_ROOT, "las_to_lod2", "01_downsampling.py")
SCRIPT_ASSIGN = os.path.join(SCRIPTS_ROOT, "las_to_lod2", "02_reclassify.py")
SCRIPT_VISUALIZE = os.path.join(SCRIPTS_ROOT, "las_to_lod2", "visualize.py")
