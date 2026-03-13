#!/usr/bin/env python3
"""
Point Cloud Semantic Segmentation using Point Transformer V1 (S3DIS pretrained).

Similar to the Scan2LoD3 pipeline (https://github.com/OloOcki/scan2lod3),
this script uses a Point Transformer network to segment building facades into:
  - Walls, Doors, Windows (+ Other)

Architecture ported from Open3D-ML PointTransformer (standalone, no Open3D dependency).
Uses S3DIS pretrained weights from Open3D model zoo.

Usage:
    conda activate lidar-test
    python scripts/LOD2toLOD3/06_segment_point_transformer.py \
        --input outputs/07_merged_las/NIMBB-2-cleaned-super.las \
        --output_dir outputs/08_segmented

S3DIS classes (13):
    0: ceiling, 1: floor, 2: wall, 3: beam, 4: column,
    5: window, 6: door, 7: table, 8: chair, 9: sofa,
    10: bookcase, 11: board, 12: clutter
"""

import os
import sys
import argparse
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

# Point Transformer S3DIS config (matches Open3D-ML config)
PT_CONFIG = {
    'blocks': [2, 3, 4, 6, 3],
    'in_channels': 6,        # XYZ + RGB
    'num_classes': 13,
    'voxel_size': 0.04,
    'max_voxels': 50000,
    'planes': [32, 64, 128, 256, 512],
    'strides': [1, 4, 4, 4, 4],
    'nsamples': [8, 16, 16, 16, 16],
    'share_planes': 8,
}

CKPT_URL = "https://storage.googleapis.com/open3d-releases/model-zoo/pointtransformer_s3dis_202109241350utc.pth"
CKPT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
CKPT_FILENAME = "pointtransformer_s3dis_202109241350utc.pth"


# =====================================================================
# Standalone KNN and Grouping Utilities (replacing Open3D ops)
# =====================================================================

def knn_search_sklearn(support_pts, query_pts, k):
    """KNN search using sklearn KDTree. Returns indices (m, k)."""
    tree = KDTree(support_pts)
    distances, indices = tree.query(query_pts, k=k)
    return indices.astype(np.int64), distances.astype(np.float32)


def queryandgroup(nsample, points, queries, feat, row_splits_p, row_splits_q, use_xyz=True):
    """
    Find nearest neighbours and return grouped features.

    Standalone replacement for Open3D-ML queryandgroup.
    Processes batch-by-batch using row_splits.

    Args:
        nsample: Number of neighbours (k).
        points: (n, 3) tensor - support points.
        queries: (m, 3) tensor - query points.
        feat: (n, c) tensor - features on support points.
        row_splits_p: (b+1,) tensor - row splits for points.
        row_splits_q: (b+1,) tensor - row splits for queries.
        use_xyz: Whether to concatenate relative XYZ with features.

    Returns:
        (m, nsample, 3+c) or (m, nsample, c) tensor.
    """
    device = points.device
    points_np = points.detach().cpu().numpy()
    queries_np = queries.detach().cpu().numpy()
    feat_np = feat.detach().cpu().numpy()
    row_splits_p_np = row_splits_p.cpu().numpy()
    row_splits_q_np = row_splits_q.cpu().numpy()

    m = queries_np.shape[0]
    c = feat_np.shape[1]
    out_dim = (3 + c) if use_xyz else c
    result = np.zeros((m, nsample, out_dim), dtype=np.float32)

    batch_size = len(row_splits_p_np) - 1
    for b in range(batch_size):
        p_start, p_end = int(row_splits_p_np[b]), int(row_splits_p_np[b + 1])
        q_start, q_end = int(row_splits_q_np[b]), int(row_splits_q_np[b + 1])

        pts_b = points_np[p_start:p_end]
        qry_b = queries_np[q_start:q_end]
        feat_b = feat_np[p_start:p_end]

        if len(pts_b) == 0 or len(qry_b) == 0:
            continue

        k = min(nsample, len(pts_b))
        tree = KDTree(pts_b)
        _, idx_b = tree.query(qry_b, k=k)  # (m_b, k)

        # Pad if needed
        if k < nsample:
            pad_idx = np.tile(idx_b[:, -1:], (1, nsample - k))
            idx_b = np.concatenate([idx_b, pad_idx], axis=1)

        grouped_xyz = pts_b[idx_b] - qry_b[:, np.newaxis, :]  # (m_b, nsample, 3)
        grouped_feat = feat_b[idx_b]  # (m_b, nsample, c)

        if use_xyz:
            result[q_start:q_end] = np.concatenate([grouped_xyz, grouped_feat], axis=-1)
        else:
            result[q_start:q_end] = grouped_feat

    return torch.from_numpy(result).to(device)


def interpolation_op(points, queries, feat, row_splits_p, row_splits_q, k=3):
    """
    Interpolation of features with nearest neighbours.

    Standalone replacement for Open3D-ML interpolation.

    Args:
        points: (m, 3) tensor.
        queries: (n, 3) tensor.
        feat: (m, c) tensor.
        row_splits_p: row_splits for points.
        row_splits_q: row_splits for queries.
        k: Number of neighbours for interpolation.

    Returns:
        Interpolated features (n, c).
    """
    device = points.device
    points_np = points.detach().cpu().numpy()
    queries_np = queries.detach().cpu().numpy()
    feat_np = feat.detach().cpu().numpy()
    row_splits_p_np = row_splits_p.cpu().numpy()
    row_splits_q_np = row_splits_q.cpu().numpy()

    n = queries_np.shape[0]
    c = feat_np.shape[1]
    result = np.zeros((n, c), dtype=np.float32)

    batch_size = len(row_splits_p_np) - 1
    for b in range(batch_size):
        p_start, p_end = int(row_splits_p_np[b]), int(row_splits_p_np[b + 1])
        q_start, q_end = int(row_splits_q_np[b]), int(row_splits_q_np[b + 1])

        pts_b = points_np[p_start:p_end]
        qry_b = queries_np[q_start:q_end]
        feat_b = feat_np[p_start:p_end]

        if len(pts_b) == 0 or len(qry_b) == 0:
            continue

        k_actual = min(k, len(pts_b))
        tree = KDTree(pts_b)
        dist_b, idx_b = tree.query(qry_b, k=k_actual)

        dist_recip = 1.0 / (dist_b + 1e-8)
        norm = np.sum(dist_recip, axis=1, keepdims=True)
        weight = dist_recip / norm

        interp = np.zeros((len(qry_b), c), dtype=np.float32)
        for i in range(k_actual):
            interp += feat_b[idx_b[:, i]] * weight[:, i:i+1]

        result[q_start:q_end] = interp

    return torch.from_numpy(result).to(device)


def stride_subsample(points, row_splits, stride):
    """
    Subsample points by stride (every stride-th point).

    Standalone replacement for furthest_point_sample_v2.
    Uses random subsampling to approximate FPS.

    Args:
        points: (n, 3) tensor.
        row_splits: (b+1,) tensor.
        stride: Subsampling factor.

    Returns:
        idx: Indices of selected points.
        new_row_splits: Updated row splits.
    """
    row_splits_np = row_splits.cpu().numpy()
    batch_size = len(row_splits_np) - 1

    indices = []
    new_splits = [0]
    count = 0

    for b in range(batch_size):
        start, end = int(row_splits_np[b]), int(row_splits_np[b + 1])
        n_b = end - start
        n_sub = max(n_b // stride, 2)  # At least 2 for BatchNorm

        # Random subsample (approximating FPS)
        perm = np.random.permutation(n_b)[:n_sub]
        perm.sort()
        idx_b = perm + start
        indices.append(idx_b)
        count += n_sub
        new_splits.append(count)

    indices = np.concatenate(indices)
    new_row_splits = torch.LongTensor(new_splits).to(row_splits.device)
    return torch.from_numpy(indices).long().to(points.device), new_row_splits


# =====================================================================
# Point Transformer Model (standalone, from Open3D-ML architecture)
# =====================================================================

class Transformer(nn.Module):
    """Transformer self-attention layer from Point Transformer V1."""

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample

        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes),
        )
        self.linear_w = nn.Sequential(
            nn.BatchNorm1d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, mid_planes // share_planes),
            nn.BatchNorm1d(mid_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo):
        """
        Forward call for Transformer.

        Args:
            pxo: [point, feat, row_splits] with shapes (n,3), (n,c), (b+1,)

        Returns:
            Transformer features (n, c).
        """
        point, feat, row_splits = pxo
        feat_q = self.linear_q(feat)
        feat_k = self.linear_k(feat)
        feat_v = self.linear_v(feat)

        # Group key features with relative positions
        feat_k_grouped = queryandgroup(
            self.nsample, point, point, feat_k, row_splits, row_splits, use_xyz=True)
        # Group value features without positions
        feat_v_grouped = queryandgroup(
            self.nsample, point, point, feat_v, row_splits, row_splits, use_xyz=False)

        point_r = feat_k_grouped[:, :, 0:3]   # (n, nsample, 3)
        feat_k_grouped = feat_k_grouped[:, :, 3:]  # (n, nsample, c)

        # Position encoding
        for i, layer in enumerate(self.linear_p):
            if i == 1:  # BatchNorm1d needs (N, C) input
                point_r = layer(point_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            else:
                point_r = layer(point_r)

        # Attention weights
        w = feat_k_grouped - feat_q.unsqueeze(1) + point_r.view(
            point_r.shape[0], point_r.shape[1],
            self.out_planes // self.mid_planes, self.mid_planes).sum(2)

        for i, layer in enumerate(self.linear_w):
            if i % 3 == 0:  # BatchNorm1d
                w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            else:
                w = layer(w)

        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = feat_v_grouped.shape
        s = self.share_planes
        feat = ((feat_v_grouped + point_r).view(n, nsample, s, c // s) *
                w.unsqueeze(2)).sum(1).view(n, c)

        return feat


class TransitionDown(nn.Module):
    """TransitionDown layer: subsamples points and increases receptive field."""

    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        point, feat, row_splits = pxo

        if self.stride != 1:
            idx, new_row_splits = stride_subsample(point, row_splits, self.stride)
            new_point = point[idx.long(), :]

            feat_grouped = queryandgroup(
                self.nsample, point, new_point, feat,
                row_splits, new_row_splits, use_xyz=True)  # (m, nsample, 3+c)

            feat_grouped = self.relu(
                self.bn(self.linear(feat_grouped).transpose(1, 2).contiguous()))
            feat_grouped = self.pool(feat_grouped).squeeze(-1)

            point, feat, row_splits = new_point, feat_grouped, new_row_splits
        else:
            feat = self.relu(self.bn(self.linear(feat)))

        return [point, feat, row_splits]


class TransitionUp(nn.Module):
    """TransitionUp layer: interpolates features back to higher resolution."""

    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(
                nn.Linear(2 * in_planes, in_planes),
                nn.BatchNorm1d(in_planes),
                nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, in_planes),
                nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(out_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True))

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, feat, row_splits = pxo1
            feat_tmp = []
            row_splits_np = row_splits.cpu().numpy()
            for i in range(len(row_splits_np) - 1):
                start_i = int(row_splits_np[i])
                end_i = int(row_splits_np[i + 1])
                count = end_i - start_i
                feat_b = feat[start_i:end_i, :]
                feat_b = torch.cat(
                    (feat_b, self.linear2(feat_b.sum(0, True) / count).repeat(count, 1)), 1)
                feat_tmp.append(feat_b)
            feat = torch.cat(feat_tmp, 0)
            feat = self.linear1(feat)
        else:
            point_1, feat_1, row_splits_1 = pxo1
            point_2, feat_2, row_splits_2 = pxo2
            feat = self.linear1(feat_1) + interpolation_op(
                point_2, point_1, self.linear2(feat_2),
                row_splits_2, row_splits_1)
        return feat


class Bottleneck(nn.Module):
    """Bottleneck block with Transformer self-attention."""
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = Transformer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        point, feat, row_splits = pxo
        identity = feat
        feat = self.relu(self.bn1(self.linear1(feat)))
        feat = self.relu(self.bn2(self.transformer2([point, feat, row_splits])))
        feat = self.bn3(self.linear3(feat))
        feat += identity
        feat = self.relu(feat)
        return [point, feat, row_splits]


class PointTransformerSeg(nn.Module):
    """
    Point Transformer V1 for Semantic Segmentation (standalone).

    Encoder-Decoder architecture with 5 levels.
    Ported from Open3D-ML PointTransformer.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        blocks_list = cfg['blocks']
        in_channels = cfg['in_channels']
        num_classes = cfg['num_classes']
        planes = cfg['planes']
        strides = cfg['strides']
        nsamples = cfg['nsamples']
        share_planes = cfg['share_planes']

        self.in_channels = in_channels
        self.in_planes = in_channels
        block = Bottleneck

        # Build encoder
        self.encoders = nn.ModuleList()
        for i in range(5):
            self.encoders.append(
                self._make_enc(block, planes[i], blocks_list[i],
                               share_planes, stride=strides[i],
                               nsample=nsamples[i]))

        # Build decoder
        self.decoders = nn.ModuleList()
        for i in range(4, -1, -1):
            self.decoders.append(
                self._make_dec(block, planes[i], 2, share_planes,
                               nsample=nsamples[i],
                               is_head=(i == 4)))

        # Classification head
        self.cls = nn.Sequential(
            nn.Linear(planes[0], planes[0]),
            nn.BatchNorm1d(planes[0]),
            nn.ReLU(inplace=True),
            nn.Linear(planes[0], num_classes))

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(
            TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(
            TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, points_input, feat_input):
        """
        Forward pass.

        Args:
            points_input: (n, 3) tensor of point coordinates.
            feat_input:   (n, c) tensor of input features.

        Returns:
            (n, num_classes) tensor of logits.
        """
        # Single-sample batch: row_splits = [0, n]
        n = points_input.shape[0]
        row_splits = torch.LongTensor([0, n]).to(points_input.device)

        points_list = [points_input]
        feats_list = [feat_input]
        row_splits_list = [row_splits]

        # Concatenate XYZ with features if in_channels > 3
        if self.in_channels == 3:
            feats_list[0] = points_list[0]
        else:
            feats_list[0] = torch.cat((points_list[0], feats_list[0]), 1)

        # Encoder
        for i in range(5):
            p, f, r = self.encoders[i]([points_list[i], feats_list[i], row_splits_list[i]])
            points_list.append(p)
            feats_list.append(f)
            row_splits_list.append(r)

        # Decoder
        for i in range(4, -1, -1):
            if i == 4:
                feats_list[i + 1] = self.decoders[4 - i][1:]([
                    points_list[i + 1],
                    self.decoders[4 - i][0](
                        [points_list[i + 1], feats_list[i + 1], row_splits_list[i + 1]]),
                    row_splits_list[i + 1]
                ])[1]
            else:
                feats_list[i + 1] = self.decoders[4 - i][1:]([
                    points_list[i + 1],
                    self.decoders[4 - i][0](
                        [points_list[i + 1], feats_list[i + 1], row_splits_list[i + 1]],
                        [points_list[i + 2], feats_list[i + 2], row_splits_list[i + 2]]),
                    row_splits_list[i + 1]
                ])[1]

        feat = self.cls(feats_list[1])
        return feat


# =====================================================================
# Data Processing Utilities
# =====================================================================

def grid_subsampling(points, features=None, labels=None, grid_size=0.04):
    """Voxel grid subsampling."""
    voxel_indices = np.floor(points / grid_size).astype(np.int32)

    _, unique_idx, inverse = np.unique(
        voxel_indices, axis=0, return_index=True, return_inverse=True)

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
            sub_labels[inv] = labels[i]

    if features is not None:
        sub_features = np.zeros((len(unique_idx), features.shape[1]), dtype=np.float32)
        for i, inv in enumerate(inverse):
            sub_features[inv] += features[i]
        sub_features /= counts[:, np.newaxis]

    if features is not None:
        return sub_points, sub_features, sub_labels
    return sub_points, sub_labels


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
    """Load pretrained Point Transformer model."""
    model = PointTransformerSeg(cfg)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # Remove 'module.' prefix if from DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    # Try to load, allowing for key mismatches (report them)
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")

    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model


def run_inference(model, points, features, cfg, device, max_voxels=50000, num_votes=3):
    """
    Run inference on a point cloud using tiled chunks.

    Processes the point cloud in chunks of max_voxels points,
    accumulates softmax probabilities, and uses voting.

    Args:
        model: The Point Transformer model.
        points: (N, 3) numpy array of centered point coordinates.
        features: (N, 3) numpy array of RGB features [0,1], or None.
        cfg: Model configuration dict.
        device: torch device.
        max_voxels: Max points per chunk.
        num_votes: Number of voting passes.

    Returns:
        pred_labels: (N,) numpy array of predicted class indices.
    """
    num_classes = cfg['num_classes']
    num_points = len(points)

    all_probs = np.zeros((num_points, num_classes), dtype=np.float64)
    vote_counts = np.zeros(num_points, dtype=np.float64)

    num_chunks = max(1, (num_points + max_voxels - 1) // max_voxels)
    total_iters = num_chunks * num_votes

    print(f"  Points: {num_points:,}, chunk_size: {max_voxels:,}, "
          f"chunks/sweep: {num_chunks}, sweeps: {num_votes}, total chunks: {total_iters}")

    pbar = tqdm(total=total_iters, desc="  Inference", unit="chunk")

    for vote in range(num_votes):
        all_indices = np.random.permutation(num_points)

        for chunk_i in range(num_chunks):
            start = chunk_i * max_voxels
            end = min(start + max_voxels, num_points)
            chunk_indices = all_indices[start:end]

            pc_chunk = points[chunk_indices].astype(np.float32)
            # Center the chunk
            chunk_center = (pc_chunk.min(0) + pc_chunk.max(0)) / 2.0
            pc_centered = pc_chunk - chunk_center

            # Build features
            if features is not None:
                feat_chunk = features[chunk_indices].astype(np.float32)
            else:
                feat_chunk = np.zeros((len(chunk_indices), 3), dtype=np.float32)

            pts_t = torch.from_numpy(pc_centered).float().to(device)
            feat_t = torch.from_numpy(feat_chunk).float().to(device)

            with torch.no_grad():
                logits = model(pts_t, feat_t)  # (n, C)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            all_probs[chunk_indices] += probs
            vote_counts[chunk_indices] += 1

            pbar.update(1)

    pbar.close()

    # Average probabilities
    valid = vote_counts > 0
    all_probs[valid] /= vote_counts[valid, np.newaxis]

    # Handle uncovered points
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


def process_file(model, file_path, output_dir, cfg, device, num_votes=3):
    """Process a single LAS file."""
    filename = os.path.basename(file_path)
    print(f"\nProcessing: {filename}")

    # Load LAS
    las = laspy.read(file_path)
    points_raw = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    print(f"  Points: {len(points_raw):,}")

    # Center points
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
        print(f"  RGB features: not available (using zeros)")

    # Grid subsampling
    voxel_size = cfg['voxel_size']
    print(f"  Grid subsampling (grid_size={voxel_size})...")
    if features is not None:
        sub_points, sub_features, _ = grid_subsampling(
            points, features=features,
            labels=np.zeros(len(points), dtype=np.int32),
            grid_size=voxel_size)
    else:
        sub_points, _ = grid_subsampling(
            points, labels=np.zeros(len(points), dtype=np.int32),
            grid_size=voxel_size)
        sub_features = None

    print(f"  Subsampled points: {len(sub_points):,}")

    # Run inference on subsampled points
    max_voxels = cfg.get('max_voxels', 50000)
    sub_pred = run_inference(model, sub_points, sub_features, cfg, device,
                            max_voxels=max_voxels, num_votes=num_votes)

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
        colors[i] = [int(c * 256) for c in rgb]

    # Print statistics
    unique_cats, cat_counts = np.unique(category_labels, return_counts=True)
    print(f"\n  Classification results:")
    for cat, count in sorted(zip(unique_cats, cat_counts), key=lambda x: -x[1]):
        pct = 100.0 * count / len(points)
        print(f"    {cat:10s}: {count:>10,} points ({pct:5.1f}%)")

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    out_name = os.path.splitext(filename)[0] + "_pt.las"
    output_path = os.path.join(output_dir, out_name)

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
        description="Segment point clouds using Point Transformer V1 (S3DIS pretrained) "
                    "— similar to Scan2LoD3 pipeline")
    parser.add_argument("--input", type=str,
                        default="outputs/07_merged_las/NIMBB-2-cleaned-super.las",
                        help="Input LAS file path")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Input directory containing LAS files (overrides --input)")
    parser.add_argument("--output_dir", type=str, default="outputs/08_segmented",
                        help="Output directory for segmented files")
    parser.add_argument("--num_votes", type=int, default=5,
                        help="Number of inference passes for voting (more = slower but better)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--max_voxels", type=int, default=50000,
                        help="Maximum voxels per inference chunk")
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Configuration
    cfg = PT_CONFIG.copy()
    cfg['max_voxels'] = args.max_voxels

    # Download weights
    ckpt_path = download_weights(CKPT_DIR, CKPT_FILENAME, CKPT_URL)

    # Load model
    model = load_model(ckpt_path, cfg, device)

    # Find input files
    if args.input_dir:
        import glob
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.las")))
    else:
        files = [args.input]

    if not files:
        print(f"No .las files found")
        return

    print(f"\nFound {len(files)} file(s) to process")
    print(f"Model: Point Transformer V1 (S3DIS pretrained, mIoU 69.2)")
    print(f"Target classes: Wall (white), Door (red), Window (blue), Other (black)")
    print("=" * 60)

    for file_path in files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        process_file(model, file_path, args.output_dir, cfg, device,
                     num_votes=args.num_votes)

    print("\n" + "=" * 60)
    print("All files processed!")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
