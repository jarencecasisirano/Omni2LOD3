# Semi-Automated 3D Model Generation from LiDAR Point Clouds

## Overview

This project provides a **semi-automated Python-based workflow** for generating 3D building models from airborne LiDAR point clouds. It enables scalable, reproducible, and fully customizable modeling through a series of structured Python scripts.

The workflow supports both CityJSON LoD2 model generation and rapid mesh-based visualizations. This allows users to generate vector-based outputs for GIS platforms or textured models for visual presentation.

---

## 🚀 Key Features

* 🔹 Voxel-based downsampling to handle large-scale LiDAR efficiently
* 🔹 SMRF ground classification and building extraction using PDAL
* 🔹 DBSCAN-based clustering to isolate individual buildings
* 🔹 2D footprint extraction and planar segmentation via Open3D
* 🔹 CityJSON-compatible LoD1 and LoD2 model generation
* 🔹 Auto-cleaned outputs and organized structure per processing stage
* 🔹 Support for Cesium visualization using mesh outputs

---

## 📦 Folder Structure

```
3d-modeling/
├── data/                     # Raw input .las files
├── outputs/
│   ├── 3d-model/             # Final CityJSON or Mesh outputs
│   ├── classification/       # SMRF-labeled LAS files
│   ├── dbscan/               # Clustered outputs per building
│   ├── downsampled/          # Voxel-filtered LAS files
│   ├── footprint/            # Extracted building footprints (GeoJSON)
│   ├── location/             # Geo-referenced metadata
│   └── segmentation/         # Segmented planes as .ply and .json
├── scripts/                  # All Python scripts
│   ├── classify_points.py
│   ├── extract_footprint.py
│   ├── filter.py
│   ├── lod1_model_generation.py
│   ├── lod2_model_generation.py
│   ├── main.py
│   ├── segment_planes.py
│   └── voxel_downsampling.py
├── environment.yml           # Conda environment
└── README.md                 # This file
```

---

## 🛠 Installation

### Prerequisites

* Python 3.8+
* [Conda or Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Setup Instructions

```bash
git clone https://github.com/Lungsod/3d-modeling.git
cd 3d-modeling
conda env create -f environment.yml -n lidar-modeling
conda activate lidar-modeling
```

> 💡 *Install Miniconda if you don’t have Conda already.*

---

## ▶️ Usage

### Run the Main Pipeline

```bash
python main.py
```

This will automatically:

1. Downsample the LAS file
2. Run ground/building classification
3. Filter by classification labels
4. Cluster buildings with DBSCAN
5. Extract building footprints
6. Segment roof and wall planes
7. Generate CityJSON LoD2 models

### Run Individual Scripts

You can also run each stage manually if needed:

```bash
python voxel_downsampling.py
python classify_points.py
python filter.py
python extract_footprint.py
python segment_planes.py
python lod2_model_generation.py
```

---

## 🔁 Pipeline Breakdown

### 1. **Voxel Downsampling** (`voxel_downsampling.py`)

Reduces point cloud density for faster processing without losing building structure.

### 2. **Classification** (`classify_points.py`)

Uses PDAL's SMRF algorithm to label ground and building points.

### 3. **Filtering** (`filter.py`)

Removes vegetation and noise. Retains building-classified and ground points.

### 4. **DBSCAN Clustering** (`extract_footprint.py`)

Groups building points by density to isolate individual buildings. Extracts a 2D footprint for each.

### 5. **Plane Segmentation** (`segment_planes.py`)

Performs RANSAC plane segmentation for roofs and walls within each footprint.

### 6. **LoD2 Model Generation** (`lod2_model_generation.py`)

Generates watertight CityJSON models with vertical walls and segmented roofs.

---

## 🗂 Outputs

* `.las` classified and filtered files
* `.geojson` for extracted 2D building footprints
* `.ply` and `.json` for roof/wall planes
* `CityJSON` files containing full LoD2 building models
* (Optional) `.glb`/`.obj` mesh outputs for Cesium visualization (via marching cubes)

---

## 📡 Data Sources

* **Plaza Roma Point Cloud**: [https://github.com/maning/plaza-roma-imagery](https://github.com/maning/plaza-roma-imagery)
* **Pavia LiDAR**: From PhilLiDAR-1 (courtesy of Sir Dom)

---

## 👤 Contact

**Demi Gentiles**
📧 [demigentiles@gmail.com](mailto:demigentiles@gmail.com)

---

> Built for reproducibility, open data, and smart city innovation 🌐
