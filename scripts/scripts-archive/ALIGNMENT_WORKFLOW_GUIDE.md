# Point Cloud Alignment Workflow - Complete Guide

## Overview

This project provides a complete workflow for aligning point clouds to CityGML building models:

1. **Preprocessing**: Merge fragmented wall surfaces in GML files
2. **Alignment**: Align point clouds to merged wall surfaces

---

## Step 1: Wall Surface Merging (Preprocessing)

### Purpose
Merge fragmented `WallSurfaces` in CityGML files into continuous facades, reducing the number of surfaces you need to align.

### Script
[`scripts/LOD2toLOD3/merge_gml_walls.py`](file:///home/demi/Omni2LOD3/scripts/LOD2toLOD3/merge_gml_walls.py)

### Usage

```bash
# Interactive mode (select GML file)
conda activate lidar-test
python scripts/LOD2toLOD3/merge_gml_walls.py

# Specify input file
python scripts/LOD2toLOD3/merge_gml_walls.py \
  --input_file data/lod_2/my_building.gml

# Custom thresholds
python scripts/LOD2toLOD3/merge_gml_walls.py \
  --normal_threshold 10.0 \
  --distance_threshold 3.0
```

### Input/Output
- **Input**: `data/lod_2/*.gml` (original GML files)
- **Output**: `outputs/00_gml_wall_merged/*_merged.gml` (merged GML files)

### Results (NIMBB Building)
- Original: **206 wall surfaces**
- Merged: **57 wall surfaces**
- Reduction: **72% fewer surfaces** to align!

### Features
- **Normal similarity**: Groups surfaces with parallel normals (default: 5° threshold)
- **Coplanar detection**: Merges surfaces at different heights on same facade
- **Convex hull merging**: Creates continuous surfaces from fragments

---

##  Step 2: Point Cloud Alignment

### Purpose
Align `.las` point clouds to the `WallSurfaces` in merged GML files using ICP registration.

### Script
[`scripts/LOD2toLOD3/align_pointclouds_to_gml.py`](file:///home/demi/Omni2LOD3/scripts/LOD2toLOD3/align_pointclouds_to_gml.py)

### Usage

```bash
# Visualize merged wall surfaces (verify merging worked)
conda activate lidar-test
python scripts/LOD2toLOD3/align_pointclouds_to_gml.py --visualize_walls

# Run alignment (interactive)
python scripts/LOD2toLOD3/align_pointclouds_to_gml.py

# With visualization
python scripts/LOD2toLOD3/align_pointclouds_to_gml.py --visualize
```

### Input/Output
- **GML Input**: `outputs/00_gml_wall_merged/*.gml` (merged GML files)
- **Point Cloud Input**: `outputs/04_manual_cleaned_point_clouds/*/*.las`
- **Output**: `outputs/07_aligned/*.las` (aligned point clouds)

### Features
- **Interactive selection**: Choose GML file and point cloud directory
- **Wall visualization**: Red sphere markers show wall surface indices
- **Wall mapping**: Map each point cloud to its corresponding wall
- **ICP registration**: Point-to-plane ICP for accurate alignment
- **Scaling options**: Uniform, non-uniform, or no scaling

---

## Complete Workflow Example

```bash
# 1. Activate environment
conda activate lidar-test

# 2. Merge wall surfaces (preprocessing)
python scripts/LOD2toLOD3/merge_gml_walls.py
# Output: outputs/00_gml_wall_merged/nimbb_021126_FIXED_merged.gml
# Result: 206 → 57 surfaces

# 3. Visualize merged surfaces
python scripts/LOD2toLOD3/align_pointclouds_to_gml.py --visualize_walls
# Shows 57 wall surfaces with red markers

# 4. Align point clouds
python scripts/LOD2toLOD3/align_pointclouds_to_gml.py
# Interactive: map each point cloud to a wall surface
# Output: outputs/07_aligned/*.las
```

---

## Benefits of Two-Step Workflow

### Preprocessing (Merging)
✅ **One-time operation**: Merge GML files once, reuse for all alignments  
✅ **Cleaner models**: Continuous facades instead of fragments  
✅ **Faster alignment**: Fewer surfaces = fewer mapping decisions  

### Alignment
✅ **Simpler mapping**: Map to 57 surfaces instead of 206  
✅ **Better results**: Align to complete facades, not fragments  
✅ **Reusable merged files**: Use same merged GML for multiple point cloud sets

---

## File Structure

```
Omni2LOD3/
├── data/lod_2/                          # Original GML files
│   └── nimbb_021126_FIXED.gml           (206 wall surfaces)
├── outputs/
│   ├── 00_gml_wall_merged/              # Merged GML files (NEW)
│   │   └── nimbb_021126_FIXED_merged.gml  (57 surfaces)
│   ├── 04_manual_cleaned_point_clouds/  # Point clouds to align
│   └── 07_aligned/                      # Aligned point clouds
└── scripts/LOD2toLOD3/
    ├── merge_gml_walls.py              # Step 1: Preprocessing
    └── align_pointclouds_to_gml.py     # Step 2: Alignment
```
