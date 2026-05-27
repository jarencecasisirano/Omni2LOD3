import os
import argparse
import glob
import numpy as np
import laspy
import torch
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from tqdm import tqdm

# ParisLille3D Class Mapping (10 classes)
# 0: 'unclassified', 1: 'ground', 2: 'building', 3: 'pole...', 4: 'bollard...', 
# 5: 'trash_can', 6: 'barrier', 7: 'pedestrian', 8: 'car', 9: 'natural-vegetation'

# Desired Color Mapping (R, G, B)
COLOR_MAP = {
    'vegetation': [0, 255, 0],       # Light Green
    'ground': [165, 42, 42],         # Brown
    # 'road': [128, 128, 128],       # Grey - Merged into ground
    'building': [255, 0, 0],         # Red
    'other': [0, 0, 0]               # Black (default)
}

# Map ParisLille3D labels to our categories
LABEL_TO_CATEGORY = {
    9: 'vegetation',  # natural-vegetation -> vegetation
    
    1: 'ground',      # ground -> ground
    
    2: 'building',    # building -> building
    
    # All others default to 'other' (Black)
    # 0, 3, 4, 5, 6, 7, 8
}

def get_color(label):
    category = LABEL_TO_CATEGORY.get(label, 'other')
    return COLOR_MAP[category]

def setup_model():
    print("Setting up RandLaNet model (ParisLille3D)...")
    # Using ParisLille3D configuration
    cfg_file = "ml3d/configs/randlanet_parislille3d.yml"
    
    # We need to find where Open3D-ML is installed or use the cloned repo if we are running from there.
    # Since we are in the workspace, we can look for the config in the cloned repo.
    repo_path = "/home/demi/Omni2LOD3/Open3D-ML"
    full_cfg_path = os.path.join(repo_path, cfg_file)
    
    if not os.path.exists(full_cfg_path):
        # Fallback to loading from package if available, or error out
        print(f"Config file not found at {full_cfg_path}. implementation requires the Open3D-ML repo.")
        # Try to use the installed package's config if possible, but the path is clearer with the repo.
        return None, None

    cfg = _ml3d.utils.Config.load_from_file(full_cfg_path)
    
    model = ml3d.models.RandLANet(**cfg.model)
    
    # Setup dataset path (dummy, just to initialize pipeline)
    cfg.dataset['dataset_path'] = "/tmp/dataset_dummy"
    dataset = ml3d.datasets.ParisLille3D(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
    
    pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)
    
    # Download and load weights
    ckpt_folder = "./logs/"
    os.makedirs(ckpt_folder, exist_ok=True)
    ckpt_path = os.path.join(ckpt_folder, "randlanet_parislille3d_202201071330utc.pth")
    randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_parislille3d_202201071330utc.pth"
    
    if not os.path.exists(ckpt_path):
        print(f"Downloading weights to {ckpt_path}...")
        cmd = f"wget {randlanet_url} -O {ckpt_path}"
        os.system(cmd)
        
    print("Loading weights...")
    pipeline.load_ckpt(ckpt_path=ckpt_path)
    
    return pipeline, cfg

def process_file(pipeline, file_path, output_dir):
    try:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        # Load LAS file
        las = laspy.read(file_path)
        points = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float32)
        
        # ParisLille3D RandLaNet config usually uses in_channels=3 (XYZ only)
        # So we pass feat=None.
        feat = None
        
        # Prepare data for inference
        # RandLaNet expects a dictionary with 'point' key
        data = {'point': points, 'feat': feat, 'label': np.zeros(len(points), dtype=np.int32)}
        
        # Run inference
        token = pipeline.run_inference(data)
        pred_labels = token['predict_labels']
        
        # Colorize
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        for i, label in enumerate(pred_labels):
            colors[i] = get_color(label)
            
        # Update LAS file with new colors and classifications if possible
        # We will create a new LAS file to write
        # Assuming we want to keep original data but verify color updates
        
        output_path = os.path.join(output_dir, filename)
        
        # Create new LAS
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = las.header.scales
        header.offsets = las.header.offsets
        
        new_las = laspy.LasData(header)
        new_las.x = las.x
        new_las.y = las.y
        new_las.z = las.z
        
        # Set colors
        new_las.red = colors[:, 0].astype(np.uint16) * 256
        new_las.green = colors[:, 1].astype(np.uint16) * 256
        new_las.blue = colors[:, 2].astype(np.uint16) * 256
        
        # Ideally we also save the classification ID, but minimal requirement is visual colors
        # Standard LAS classification is 0-31, our labels are 0-19 so it fits.
        new_las.classification = pred_labels.astype(np.uint8)
        
        new_las.write(output_path)
        print(f"Saved to: {output_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Classify point clouds using RandLaNet.")
    parser.add_argument("--input_dir", type=str, default="outputs/03_pointclouds", help="Input directory containing LAS files")
    parser.add_argument("--output_dir", type=str, default="outputs/04_classified", help="Output directory for classified files")
    
    args = parser.parse_args()
    
    # Setup Output Dir
    # We need to maintain subdirectory structure, so we'll do that in loop
    
    # Find all LAS files recursively
    files = glob.glob(os.path.join(args.input_dir, "**", "*.las"), recursive=True)
    
    if not files:
        print(f"No .las files found in {args.input_dir}")
        return
        
    print(f"Found {len(files)} files.")
    
    # Setup Model
    pipeline, cfg = setup_model()
    if pipeline is None:
        print("Failed to initialize model.")
        return
        
    for file_path in tqdm(files):
        # Determine output path maintaining structure
        rel_path = os.path.relpath(os.path.dirname(file_path), args.input_dir)
        if rel_path == ".":
            current_output_dir = args.output_dir
        else:
            current_output_dir = os.path.join(args.output_dir, rel_path)
            
        os.makedirs(current_output_dir, exist_ok=True)
        
        process_file(pipeline, file_path, current_output_dir)

if __name__ == "__main__":
    main()
