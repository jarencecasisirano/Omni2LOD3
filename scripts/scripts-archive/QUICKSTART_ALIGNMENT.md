# Quick Start: Point Cloud to GML Alignment

Complete workflow for aligning point clouds to CityGML building models.

---

## Prerequisites

**Environment**: `lidar-test` conda environment

---

## Two-Step Workflow

### Step 1: Merge Wall Surfaces (Preprocessing)

Merge fragmented wall surfaces in GML files to create continuous facades.

```bash
conda activate lidar-test
python scripts/LOD2toLOD3/merge_gml_walls.py
```

**What it does**:
- Reads GML from `data/lod_2/`
- Auto-selects if only one file available
- Merges surfaces with similar normals and coplanarity
- Saves to `outputs/00_gml_wall_merged/`

**Result**: 206 surfaces → 57 surfaces (72% reduction!)

**Options**:
```bash
# Custom thresholds
python scripts/LOD2toLOD3/merge_gml_walls.py \
  --normal_threshold 10.0 \
  --distance_threshold 3.0

# Specify input file
python scripts/LOD2toLOD3/merge_gml_walls.py \
  --input_file data/lod_2/my_building.gml
```

---

### Step 2: Align Point Clouds

Align point clouds to the merged wall surfaces.

```bash
# Visualize merged walls (verify merging)
python scripts/LOD2toLOD3/align_pointclouds_to_gml.py --visualize_walls

# Run alignment (interactive)
python scripts/LOD2toLOD3/align_pointclouds_to_gml.py

# With per-alignment visualization
python scripts/LOD2toLOD3/align_pointclouds_to_gml.py --visualize
```

**What it does**:
- Auto-selects merged GML from `outputs/00_gml_wall_merged/`
- Auto-selects point cloud directory if only one available
- Shows wall surfaces with colored legend and index markers
- Interactively map point clouds to walls
- Performs ICP alignment with scaling
- Saves to `outputs/07_aligned/`

**Visualization Features**:
- 🎨 **Color-coded legend** in terminal showing each wall's index and color
- 🌈 **Unique colors** for each wall surface in 3D view
- 🔴 **Red spheres** marking wall centers
- Match colors from terminal legend to 3D view to identify walls

**Options**:
```bash
# Non-interactive (specify all paths)
python scripts/LOD2toLOD3/align_pointclouds_to_gml.py \
  --gml_file outputs/00_gml_wall_merged/building_merged.gml \
  --pointcloud_dir outputs/04_manual_cleaned_point_clouds/NIMBB \
  --scale_mode uniform \
  --icp_threshold 1.0
```

---

## Quick Reference

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `merge_gml_walls.py` | Merge wall surfaces | `data/lod_2/*.gml` | `outputs/00_gml_wall_merged/*_merged.gml` |
| `align_pointclouds_to_gml.py` | Align point clouds | Merged GML + point clouds | `outputs/07_aligned/*.las` |

---

## Tips

- **Run merging once**: Merged GML files can be reused for multiple alignments
- **Adjust thresholds**: Use `--normal_threshold` to control how aggressively surfaces merge
- **Visualize first**: Always run `--visualize_walls` to verify merging before alignment
- **Use color legend**: Match the colored blocks in terminal to wall surfaces in 3D view
- **Interactive mode**: Use `conda activate` for interactive prompts (not `conda run`)


---

## Why "conda activate" instead of "conda run"?

**`conda activate`**: Starts an interactive shell with the environment - **use this for interactive selection**  
**`conda run`**: Runs a single command non-interactively - **use this only when all options are specified via flags**

---

## Options

| Option | Description |
|--------|-------------|
| `--gml_file` | Specific GML file path (skips GML selection) |
| `--gml_dir` | Directory to search for GML files (default: `data/lod_2`) |
| `--pointcloud_dir` | Specific point cloud directory (skips directory selection) |
| `--pointcloud_base_dir` | Base directory for point clouds (default: `outputs/04_manual_cleaned_point_clouds`) |
| `--output_dir` | Where to save aligned point clouds (default: `outputs/07_aligned`) |
| `--scale_mode` | `uniform`, `non-uniform`, or `none` (default: `uniform`) |
| `--icp_threshold` | ICP distance threshold (default: `1.0`) |
| `--visualize` | Show 3D visualization for each alignment |
| `--visualize_walls` | Only show walls and exit |
| `--mapping_file` | JSON file with predefined wall mappings |

---

## Output

**Aligned point clouds**: `outputs/07_aligned/` (or your `--output_dir`)  
**Alignment report**: `outputs/07_aligned/alignment_report.json`
