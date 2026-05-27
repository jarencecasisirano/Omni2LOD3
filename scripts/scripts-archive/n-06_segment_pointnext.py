#!/usr/bin/env python3
"""
Point Cloud Semantic Segmentation using RandLA-Net (S3DIS pretrained).

Segments point clouds from outputs/07_merged_las into:
  - Walls, Doors, Windows (+ Other)

Uses the Open3D-ML RandLA-Net architecture with S3DIS pretrained weights.
The model code is self-contained to avoid Open3D-ML import issues.

Usage:
    conda activate lidar-test
    python scripts/LOD2toLOD3/06_segment_pointnext.py \
        --input_dir outputs/07_merged_las \
        --output_dir outputs/08_segmented

S3DIS classes (13):
    0: ceiling, 1: floor, 2: wall, 3: beam, 4: column,
    5: window, 6: door, 7: table, 8: chair, 9: sofa,
    10: bookcase, 11: board, 12: clutter
"""

import os
import sys
import argparse
import glob
import urllib.request

import numpy as np
import laspy
import torch
import torch.nn as nn
from sklearn.neighbors import KDTree
from tqdm import tqdm


# =====================================================================
# S3DIS Class Definitions
# =====================================================================
S3DIS_CLASSES = [
    'ceiling', 'floor', 'wall', 'beam', 'column',
    'window', 'door', 'table', 'chair', 'sofa',
    'bookcase', 'board', 'clutter'
]

# Target class mapping: S3DIS label -> our category
TARGET_CLASSES = {
    2: 'Wall',
    5: 'Window',
    6: 'Door',
}

# Color map for output (RGB 0-255)
COLOR_MAP = {
    'Wall':   [255, 255, 255],  # White
    'Door':   [255, 0,   0],    # Red
    'Window': [0,   0,   255],  # Blue
    'Other':  [0,   0,   0],    # Black
}

# S3DIS RandLANet model config
S3DIS_CONFIG = {
    'num_neighbors': 16,
    'num_layers': 5,
    'num_points': 40960,
    'num_classes': 13,
    'ignored_label_inds': [],
    'sub_sampling_ratio': [4, 4, 4, 4, 2],
    'in_channels': 6,
    'dim_features': 8,
    'dim_output': [16, 64, 128, 256, 512],
    'grid_size': 0.04,
}

CKPT_URL = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_s3dis_202201071330utc.pth"
CKPT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
CKPT_FILENAME = "randlanet_s3dis_202201071330utc.pth"


# =====================================================================
# RandLA-Net Model (standalone, from Open3D-ML source)
# =====================================================================

class SharedMLP(nn.Module):
    """Conv2d + BN + activation."""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 transpose=False, bn=True, activation_fn=None):
        super().__init__()
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=kernel_size, stride=stride,
                                           padding=(kernel_size - 1) // 2)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=(kernel_size - 1) // 2)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.01) if bn else None
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    """Encodes local spatial features for k neighbours."""
    def __init__(self, dim_in, dim_out, num_neighbors, encode_pos=False):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(dim_in, dim_out, activation_fn=nn.LeakyReLU(0.2))
        self.encode_pos = encode_pos

    def gather_neighbor(self, coords, neighbor_indices):
        B, N, K = neighbor_indices.size()
        dim = coords.shape[2]
        extended_indices = neighbor_indices.unsqueeze(1).expand(B, dim, N, K)
        extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, dim, N, K)
        neighbor_coords = torch.gather(extended_coords, 2, extended_indices)
        return neighbor_coords

    def forward(self, coords, features, neighbor_indices, relative_features=None):
        B, N, K = neighbor_indices.size()
        if self.encode_pos:
            neighbor_coords = self.gather_neighbor(coords, neighbor_indices)
            extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)
            relative_pos = extended_coords - neighbor_coords
            relative_dist = torch.sqrt(
                torch.sum(torch.square(relative_pos), dim=1, keepdim=True))
            relative_features = torch.cat(
                [relative_dist, relative_pos, extended_coords, neighbor_coords], dim=1)
        else:
            if relative_features is None:
                raise ValueError("Require relative_features for second pass.")

        relative_features = self.mlp(relative_features)
        neighbor_features = self.gather_neighbor(
            features.transpose(1, 2).squeeze(3), neighbor_indices)
        return torch.cat([neighbor_features, relative_features], dim=1), relative_features


class AttentivePooling(nn.Module):
    """Attention-based pooling for k neighbours."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.score_fn = nn.Sequential(nn.Linear(in_channels, in_channels),
                                      nn.Softmax(dim=-2))
        self.mlp = SharedMLP(in_channels, out_channels, activation_fn=nn.LeakyReLU(0.2))

    def forward(self, x):
        scores = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        features = torch.sum(scores * x, dim=-1, keepdim=True)
        return self.mlp(features)


class LocalFeatureAggregation(nn.Module):
    """Two-pass local feature aggregation with attentive pooling."""
    def __init__(self, d_in, d_out, num_neighbors):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.mlp1 = SharedMLP(d_in, d_out // 2, activation_fn=nn.LeakyReLU(0.2))
        self.lse1 = LocalSpatialEncoding(10, d_out // 2, num_neighbors, encode_pos=True)
        self.pool1 = AttentivePooling(d_out, d_out // 2)
        self.lse2 = LocalSpatialEncoding(d_out // 2, d_out // 2, num_neighbors)
        self.pool2 = AttentivePooling(d_out, d_out)
        self.mlp2 = SharedMLP(d_out, 2 * d_out)
        self.shortcut = SharedMLP(d_in, 2 * d_out)
        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, feat, neighbor_indices):
        x = self.mlp1(feat)
        x, neighbor_features = self.lse1(coords, x, neighbor_indices)
        x = self.pool1(x)
        x, _ = self.lse2(coords, x, neighbor_indices, relative_features=neighbor_features)
        x = self.pool2(x)
        return self.lrelu(self.mlp2(x) + self.shortcut(feat))


class RandLANet(nn.Module):
    """RandLA-Net for semantic segmentation (standalone)."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fc0 = nn.Linear(cfg['in_channels'], cfg['dim_features'])
        self.bn0 = nn.BatchNorm2d(cfg['dim_features'], eps=1e-6, momentum=0.01)

        # Encoder
        self.encoder = nn.ModuleList()
        encoder_dim_list = []
        dim_feature = cfg['dim_features']
        for i in range(cfg['num_layers']):
            self.encoder.append(
                LocalFeatureAggregation(dim_feature, cfg['dim_output'][i], cfg['num_neighbors']))
            dim_feature = 2 * cfg['dim_output'][i]
            if i == 0:
                encoder_dim_list.append(dim_feature)
            encoder_dim_list.append(dim_feature)

        self.mlp = SharedMLP(dim_feature, dim_feature, activation_fn=nn.LeakyReLU(0.2))

        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(cfg['num_layers']):
            self.decoder.append(
                SharedMLP(encoder_dim_list[-i - 2] + dim_feature,
                          encoder_dim_list[-i - 2],
                          transpose=True, activation_fn=nn.LeakyReLU(0.2)))
            dim_feature = encoder_dim_list[-i - 2]

        self.fc1 = nn.Sequential(
            SharedMLP(dim_feature, 64, activation_fn=nn.LeakyReLU(0.2)),
            SharedMLP(64, 32, activation_fn=nn.LeakyReLU(0.2)),
            nn.Dropout(0.5),
            SharedMLP(32, cfg['num_classes'], bn=False))

    def forward(self, inputs):
        cfg = self.cfg
        device = next(self.parameters()).device
        feat = inputs['features'].to(device)
        coords_list = [arr.to(device) for arr in inputs['coords']]
        neighbor_indices_list = [arr.to(device) for arr in inputs['neighbor_indices']]
        subsample_indices_list = [arr.to(device) for arr in inputs['sub_idx']]
        interpolation_indices_list = [arr.to(device) for arr in inputs['interp_idx']]

        feat = self.fc0(feat).transpose(-2, -1).unsqueeze(-1)
        feat = self.bn0(feat)
        feat = nn.LeakyReLU(0.2)(feat)

        encoder_feat_list = []
        for i in range(cfg['num_layers']):
            feat_enc = self.encoder[i](coords_list[i], feat, neighbor_indices_list[i])
            feat_sampled = self._random_sample(feat_enc, subsample_indices_list[i])
            if i == 0:
                encoder_feat_list.append(feat_enc.clone())
            encoder_feat_list.append(feat_sampled.clone())
            feat = feat_sampled

        feat = self.mlp(feat)

        for i in range(cfg['num_layers']):
            feat_interp = self._nearest_interpolation(feat, interpolation_indices_list[-i - 1])
            feat = torch.cat([encoder_feat_list[-i - 2], feat_interp], dim=1)
            feat = self.decoder[i](feat)

        scores = self.fc1(feat)
        return scores.squeeze(3).transpose(1, 2)

    @staticmethod
    def _random_sample(feature, pool_idx):
        feature = feature.squeeze(3)
        num_neigh = pool_idx.size()[2]
        batch_size = feature.size()[0]
        d = feature.size()[1]
        pool_idx = torch.reshape(pool_idx, (batch_size, -1))
        pool_idx = pool_idx.unsqueeze(2).expand(batch_size, -1, d)
        feature = feature.transpose(1, 2)
        pool_features = torch.gather(feature, 1, pool_idx)
        pool_features = torch.reshape(pool_features, (batch_size, -1, num_neigh, d))
        pool_features, _ = torch.max(pool_features, 2, keepdim=True)
        pool_features = pool_features.permute(0, 3, 1, 2)
        return pool_features

    @staticmethod
    def _nearest_interpolation(feature, interp_idx):
        feature = feature.squeeze(3)
        d = feature.size(1)
        batch_size = interp_idx.size()[0]
        up_num_points = interp_idx.size()[1]
        interp_idx = torch.reshape(interp_idx, (batch_size, up_num_points))
        interp_idx = interp_idx.unsqueeze(1).expand(batch_size, d, -1)
        interpolated = torch.gather(feature, 2, interp_idx)
        interpolated = interpolated.unsqueeze(3)
        return interpolated


# =====================================================================
# Data Processing Utilities
# =====================================================================

def knn_search(support_pts, query_pts, k):
    """KNN search using sklearn KDTree."""
    tree = KDTree(support_pts)
    _, indices = tree.query(query_pts, k=k)
    return indices.astype(np.int64)


def grid_subsampling(points, features=None, labels=None, grid_size=0.04):
    """Voxel grid subsampling."""
    # Compute voxel indices
    voxel_indices = np.floor(points / grid_size).astype(np.int32)

    # Unique voxels
    _, unique_idx, inverse = np.unique(
        voxel_indices, axis=0, return_index=True, return_inverse=True)

    # Average points per voxel
    sub_points = np.zeros((len(unique_idx), 3), dtype=np.float32)
    counts = np.zeros(len(unique_idx), dtype=np.float32)

    for i, inv in enumerate(inverse):
        sub_points[inv] += points[i]
        counts[inv] += 1

    sub_points /= counts[:, np.newaxis]

    sub_labels = None
    sub_features = None

    if labels is not None:
        sub_labels = np.zeros(len(unique_idx), dtype=np.int32)
        for i, inv in enumerate(inverse):
            sub_labels[inv] = labels[i]  # majority vote simplified to last-write

    if features is not None:
        sub_features = np.zeros((len(unique_idx), features.shape[1]), dtype=np.float32)
        for i, inv in enumerate(inverse):
            sub_features[inv] += features[i]
        sub_features /= counts[:, np.newaxis]

    if features is not None:
        return sub_points, sub_features, sub_labels
    return sub_points, sub_labels


def prepare_input(points, features, cfg):
    """Prepare input data for the RandLA-Net model (single sample)."""
    num_points = cfg['num_points']
    pc = points.copy()

    # Combine XYZ + features if available
    if features is not None:
        feat = np.concatenate([pc, features], axis=1).astype(np.float32)
    else:
        feat = pc.copy().astype(np.float32)

    # Randomly sample or pad to num_points
    if len(pc) >= num_points:
        choice = np.random.choice(len(pc), num_points, replace=False)
    else:
        choice = np.random.choice(len(pc), num_points, replace=True)

    pc_sampled = pc[choice]
    feat_sampled = feat[choice]

    input_points = []
    input_neighbors = []
    input_pools = []
    input_up_samples = []

    for i in range(cfg['num_layers']):
        neighbour_idx = knn_search(pc_sampled, pc_sampled, cfg['num_neighbors'])
        sub_points = pc_sampled[:pc_sampled.shape[0] // cfg['sub_sampling_ratio'][i], :]
        pool_i = neighbour_idx[:pc_sampled.shape[0] // cfg['sub_sampling_ratio'][i], :]
        up_i = knn_search(sub_points, pc_sampled, 1)
        input_points.append(pc_sampled)
        input_neighbors.append(neighbour_idx)
        input_pools.append(pool_i)
        input_up_samples.append(up_i)
        pc_sampled = sub_points

    inputs = {
        'coords': [torch.from_numpy(p).float().unsqueeze(0) for p in input_points],
        'neighbor_indices': [torch.from_numpy(n).long().unsqueeze(0) for n in input_neighbors],
        'sub_idx': [torch.from_numpy(p).long().unsqueeze(0) for p in input_pools],
        'interp_idx': [torch.from_numpy(u).long().unsqueeze(0) for u in input_up_samples],
        'features': torch.from_numpy(feat_sampled).float().unsqueeze(0),
        'point_inds': choice,
    }
    return inputs


# =====================================================================
# Inference Pipeline
# =====================================================================

def download_weights(ckpt_dir, ckpt_filename, url):
    """Download pretrained weights if not present."""
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
    if not os.path.exists(ckpt_path):
        print(f"Downloading pretrained weights to {ckpt_path}...")
        print(f"  URL: {url}")
        urllib.request.urlretrieve(url, ckpt_path)
        print("  Download complete.")
    return ckpt_path


def load_model(ckpt_path, cfg, device):
    """Load pretrained RandLA-Net model."""
    model = RandLANet(cfg)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model


def run_inference(model, points, features, cfg, device, num_votes=3):
    """
    Run inference on a point cloud by tiling through ALL points systematically.

    Uses overlapping chunks to ensure every point gets predictions. Each chunk
    of num_points points is processed through the model, and probabilities are
    accumulated point-by-point.

    Args:
        model: The RandLA-Net model.
        points: (N, 3) numpy array of point coordinates.
        features: (N, 3) numpy array of RGB features scaled to [0, 1], or None.
        cfg: Model configuration dict.
        device: torch device.
        num_votes: Number of full sweeps over the point cloud.

    Returns:
        pred_labels: (N,) numpy array of predicted class indices.
    """
    num_classes = cfg['num_classes']
    num_points_total = len(points)
    chunk_size = cfg['num_points']

    # Combine XYZ + features
    if features is not None:
        feat_all = np.concatenate([points, features], axis=1).astype(np.float32)
    else:
        feat_all = points.copy().astype(np.float32)

    # Accumulate predictions
    all_probs = np.zeros((num_points_total, num_classes), dtype=np.float64)
    vote_counts = np.zeros(num_points_total, dtype=np.float64)

    # Calculate number of chunks needed per sweep
    num_chunks = max(1, (num_points_total + chunk_size - 1) // chunk_size)
    total_iters = num_chunks * num_votes

    print(f"  Points: {num_points_total:,}, chunk_size: {chunk_size:,}, "
          f"chunks/sweep: {num_chunks}, sweeps: {num_votes}, total chunks: {total_iters}")

    pbar = tqdm(total=total_iters, desc="  Inference", unit="chunk")

    for vote in range(num_votes):
        # Shuffle indices for this sweep
        all_indices = np.random.permutation(num_points_total)

        for chunk_i in range(num_chunks):
            start = chunk_i * chunk_size
            end = min(start + chunk_size, num_points_total)
            chunk_indices = all_indices[start:end]

            # Pad if chunk is smaller than required
            if len(chunk_indices) < chunk_size:
                pad = np.random.choice(all_indices, chunk_size - len(chunk_indices), replace=True)
                padded_indices = np.concatenate([chunk_indices, pad])
            else:
                padded_indices = chunk_indices

            pc_chunk = points[padded_indices]
            feat_chunk = feat_all[padded_indices]

            # Build input
            inputs = prepare_input_from_arrays(pc_chunk, feat_chunk, cfg)

            with torch.no_grad():
                scores = model(inputs)  # (1, chunk_size, C)

            probs = torch.softmax(scores, dim=-1).cpu().numpy()[0]  # (chunk_size, C)

            # Only accumulate for the actual (non-padded) points
            actual_count = len(chunk_indices)
            all_probs[chunk_indices] += probs[:actual_count]
            vote_counts[chunk_indices] += 1

            pbar.update(1)

    pbar.close()

    # Average probabilities
    valid = vote_counts > 0
    all_probs[valid] /= vote_counts[valid, np.newaxis]

    # For points never covered (shouldn't happen with systematic sweep), use NN
    if not np.all(valid):
        from scipy.spatial import cKDTree
        valid_pts = points[valid]
        invalid_pts = points[~valid]
        tree = cKDTree(valid_pts)
        _, nn_idx = tree.query(invalid_pts, k=1)
        valid_indices = np.where(valid)[0]
        all_probs[~valid] = all_probs[valid_indices[nn_idx]]

    pred_labels = np.argmax(all_probs, axis=1)
    return pred_labels


def prepare_input_from_arrays(pc, feat, cfg):
    """Prepare model input directly from point and feature arrays (already sized)."""
    input_points = []
    input_neighbors = []
    input_pools = []
    input_up_samples = []

    current_pc = pc.copy()
    for i in range(cfg['num_layers']):
        neighbour_idx = knn_search(current_pc, current_pc, cfg['num_neighbors'])
        n_sub = current_pc.shape[0] // cfg['sub_sampling_ratio'][i]
        sub_points = current_pc[:n_sub, :]
        pool_i = neighbour_idx[:n_sub, :]
        up_i = knn_search(sub_points, current_pc, 1)
        input_points.append(current_pc)
        input_neighbors.append(neighbour_idx)
        input_pools.append(pool_i)
        input_up_samples.append(up_i)
        current_pc = sub_points

    inputs = {
        'coords': [torch.from_numpy(p).float().unsqueeze(0) for p in input_points],
        'neighbor_indices': [torch.from_numpy(n).long().unsqueeze(0) for n in input_neighbors],
        'sub_idx': [torch.from_numpy(p).long().unsqueeze(0) for p in input_pools],
        'interp_idx': [torch.from_numpy(u).long().unsqueeze(0) for u in input_up_samples],
        'features': torch.from_numpy(feat).float().unsqueeze(0),
    }
    return inputs


def process_file(model, file_path, output_dir, cfg, device, num_votes=5):
    """Process a single LAS file."""
    filename = os.path.basename(file_path)
    print(f"\nProcessing: {filename}")

    # Load LAS
    las = laspy.read(file_path)
    points_raw = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    print(f"  Points: {len(points_raw):,}")

    # Center points (subtract min) to avoid large coordinate offsets
    p_min = points_raw.min(axis=0)
    p_max = points_raw.max(axis=0)
    p_range = p_max - p_min
    print(f"  Coordinate mins: {p_min}")
    print(f"  Coordinate maxs: {p_max}")
    print(f"  Coordinate range: {p_range}")
    
    points = points_raw - p_min
    print(f"  Points centered. New mins: {points.min(axis=0)}")

    # Extract RGB features if available
    features = None
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        red = np.array(las.red, dtype=np.float32)
        green = np.array(las.green, dtype=np.float32)
        blue = np.array(las.blue, dtype=np.float32)

        # Normalize to [0, 1] — LAS stores colors as 16-bit
        # Some LAS files use 8-bit even if structured as 16-bit
        max_val = max(red.max(), green.max(), blue.max(), 1.0)
        if max_val > 255:
            red /= 65535.0
            green /= 65535.0
            blue /= 65535.0
        else:
            red /= 255.0
            green /= 255.0
            blue /= 255.0

        features = np.column_stack([red, green, blue]).astype(np.float32)
        print(f"  RGB features: normalized and available")
    else:
        print(f"  RGB features: not available (using XYZ only)")

    # Grid subsampling for preprocessing
    print(f"  Grid subsampling (grid_size={cfg['grid_size']})...")
    if features is not None:
        sub_points, sub_features, _ = grid_subsampling(
            points, features=features, labels=np.zeros(len(points), dtype=np.int32),
            grid_size=cfg['grid_size'])
    else:
        sub_points, _ = grid_subsampling(
            points, labels=np.zeros(len(points), dtype=np.int32),
            grid_size=cfg['grid_size'])
        sub_features = None

    print(f"  Subsampled points: {len(sub_points):,}")

    # Run inference on subsampled points
    sub_pred = run_inference(model, sub_points, sub_features, cfg, device, num_votes=num_votes)

    # Project back to original point cloud
    print(f"  Projecting labels to original points...")
    tree = KDTree(sub_points)
    proj_inds = tree.query(points, return_distance=False).squeeze()
    pred_labels = sub_pred[proj_inds]

    # Map to target classes
    category_labels = []
    for label in pred_labels:
        category_labels.append(TARGET_CLASSES.get(int(label), 'Other'))

    # Assign colors
    colors = np.zeros((len(points), 3), dtype=np.uint16)
    for i, cat in enumerate(category_labels):
        rgb = COLOR_MAP[cat]
        colors[i] = [int(c * 256) for c in rgb]  # Scale to 16-bit for LAS

    # Print statistics
    unique_cats, cat_counts = np.unique(category_labels, return_counts=True)
    print(f"\n  Classification results:")
    for cat, count in sorted(zip(unique_cats, cat_counts), key=lambda x: -x[1]):
        pct = 100.0 * count / len(points)
        print(f"    {cat:10s}: {count:>10,} points ({pct:5.1f}%)")

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    header = laspy.LasHeader(point_format=2, version="1.2")
    header.scales = las.header.scales
    header.offsets = las.header.offsets

    new_las = laspy.LasData(header)
    new_las.x = las.x
    new_las.y = las.y
    new_las.z = las.z
    new_las.red = colors[:, 0]
    new_las.green = colors[:, 1]
    new_las.blue = colors[:, 2]
    new_las.classification = pred_labels.astype(np.uint8)

    new_las.write(output_path)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Segment point clouds into Walls/Doors/Windows using RandLA-Net (S3DIS)")
    parser.add_argument("--input_dir", type=str, default="outputs/07_merged_las",
                        help="Input directory containing LAS files")
    parser.add_argument("--output_dir", type=str, default="outputs/08_segmented",
                        help="Output directory for segmented files")
    parser.add_argument("--num_votes", type=int, default=10,
                        help="Number of inference passes for voting (more = slower but better)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto'")
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Configuration
    cfg = S3DIS_CONFIG.copy()

    # Download weights
    ckpt_path = download_weights(CKPT_DIR, CKPT_FILENAME, CKPT_URL)

    # Load model
    model = load_model(ckpt_path, cfg, device)

    # Find input files
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.las")))
    if not files:
        print(f"No .las files found in {args.input_dir}")
        return

    print(f"\nFound {len(files)} file(s) to process")
    print(f"Target classes: Wall (white), Door (red), Window (blue), Other (black)")
    print("=" * 60)

    for file_path in files:
        process_file(model, file_path, args.output_dir, cfg, device,
                     num_votes=args.num_votes)

    print("\n" + "=" * 60)
    print("All files processed!")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
