# Quick Start Guide - Point Cloud Alignment

## Simple Interactive Usage (Recommended)

**IMPORTANT**: For interactive mode, activate the conda environment first:

```bash
conda activate lidar-test
python scripts/align_pointclouds_to_gml.py
```

The script will prompt you to:
1. **Select a GML model** from `data/lod_2/` (auto-selects if only one)
2. **Select a point cloud folder** from `outputs/04_manual_cleaned_point_clouds/` (auto-selects if only one)
3. **Map each point cloud** to a wall surface interactively

---

## Quick Commands

### Visualize walls only
```bash
conda activate lidar-test
python scripts/align_pointclouds_to_gml.py --visualize_walls
```

### Interactive alignment with visualization
```bash
conda activate lidar-test
python scripts/align_pointclouds_to_gml.py --visualize
```

### Non-interactive (specify all paths directly)
```bash
conda run -n lidar-test python scripts/align_pointclouds_to_gml.py \
  --gml_file data/lod_2/nimbb_021126_FIXED.gml \
  --pointcloud_dir outputs/04_manual_cleaned_point_clouds/NIMBB
```

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
