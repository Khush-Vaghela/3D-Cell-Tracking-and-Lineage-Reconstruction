"""
Phase 1.3 -- a single trainable step from a real sample, end to end:

  data_io.load_geff (GT) + data_io.open_zarr_volume (image)
    -> preprocessing (normalize, downsample image + GT coords -- official
       (1,4,4) convention, confirmed isotropic)
    -> targets.build_heatmap_target / build_loss_weight_map (this
       project's own sparse-GT scheme, kept over the official repo's
       simpler single-voxel scheme per earlier decision)
    -> Detector3D forward
    -> targets.weighted_heatmap_mse_loss
    -> backward

This is deliberately NOT the official repo's joint detector+linker
training loop (train_unet_transformer.py trains TemporalUNet3D and
SimpleNodeTransformer together) -- this project is doing detection and
linking as separate phases (see workflow), so this file only ever trains
the detector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import data_io as bio_io
import preprocessing as prep
import targets
from detector_model import Detector3D
from pipeline_config import DOWNSAMPLE, SIGMA_UM

import torch
import torch.nn.functional as F

def focal_heatmap_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    weight: torch.Tensor = None, 
    alpha: float = 2.0, 
    beta: float = 4.0
) -> torch.Tensor:
    """
    CenterNet-style Modified Focal Loss for 3D Gaussian heatmaps.
    
    Args:
        pred: Predicted raw logits or heatmaps (1, T, Z, Y, X).
        target: Ground truth Gaussian heatmaps [0, 1].
        weight: (Optional) Custom loss weight map provided by the pipeline.
        alpha: Reduces the penalty for easy, well-classified examples.
        beta: Reduces the penalty for near-misses around the Gaussian peak.
    """
    # 1. Clamp predictions to prevent log(0) NaN errors
    # (Removed torch.sigmoid because Detector3D already applies it)
    pred = torch.clamp(pred, min=1e-5, max=1.0 - 1e-5)
    
    # 2. Identify the absolute peaks vs the background/tails
    # Assuming targets.build_heatmap_target normalizes the peak to 1.0
    # Look for the exact peak
    pos_mask = (target == 1.0).float()
    neg_mask = (target < 1.0).float()

    # Calculate Focal Loss for positive peaks
    pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos_mask

    # Boost the positive gradient
    pos_weight_boost = 50.0  
    pos_loss = pos_loss * pos_weight_boost
    
    # 4. Calculate Focal Loss for background and Gaussian tails
    # The (1 - target)^beta term ensures we don't heavily penalize predictions 
    # that are very close to the true node center.
    neg_loss = -torch.log(1.0 - pred) * torch.pow(pred, alpha) * torch.pow(1.0 - target, beta) * neg_mask
    
    # 5. Apply your project's custom background weights (if provided)
    if weight is not None:
        pos_loss = pos_loss * weight
        neg_loss = neg_loss * weight
        
    # 6. Sum the loss and normalize by the number of positive peaks
    num_pos = pos_mask.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_pos == 0:
        return neg_loss
        
    return (pos_loss + neg_loss) / num_pos


def build_training_window(
    sample: bio_io.Sample,
    t_start: int,
    window_size: int,
    downsample: tuple[int, int, int] = DOWNSAMPLE,
    sigma_um: float = SIGMA_UM,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build one (window, target, weight) training triple for frames
    [t_start, t_start + window_size).

    Returns
    -------
    window : (1, T, Z, Y, X) float32 -- batch dim of 1, ready for Detector3D
    target : (1, T, Z, Y, X) float32 -- Gaussian heatmap targets, one per frame
    weight : (1, T, Z, Y, X) float32 -- per-voxel loss weights, one per frame

    All three are moved to `device` before returning -- callers (train_step,
    and any validation loop) don't need to remember to do this themselves.
    """
    if sample.geff_path is None:
        raise ValueError(f"sample {sample.name} has no ground truth (.geff) -- can't train on it")

    vol = bio_io.open_zarr_volume(sample.zarr_path)
    geff = bio_io.load_geff(sample.geff_path)

    raw_window = np.stack([np.asarray(vol[t]) for t in range(t_start, t_start + window_size)])
    proc_window = prep.preprocess_window(raw_window, downsample)  # (T, Zd, Yd, Xd)
    frame_shape_down = proc_window.shape[1:]
    scale_down = prep.downsampled_scale(bio_io.NATIVE_SCALE, downsample)

    background_weights = targets.background_weight_per_frame(
        geff.nodes, geff.estimated_number_of_nodes, vol.shape[0]
    )

    heatmaps, weights = [], []
    for i, t in enumerate(range(t_start, t_start + window_size)):
        nodes_t_native = geff.nodes[geff.nodes["t"] == t]
        nodes_t_down = prep.downsample_node_coords(nodes_t_native, downsample)
        heatmaps.append(
            targets.build_heatmap_target(frame_shape_down, nodes_t_down, scale_down, sigma_um=sigma_um)
        )
        weights.append(
            targets.build_loss_weight_map(
                frame_shape_down, nodes_t_down, scale_down, background_weight=background_weights[t]
            )
        )

    window_t = torch.from_numpy(proc_window).unsqueeze(0).to(device)  # (1, T, Zd, Yd, Xd)
    target_t = torch.from_numpy(np.stack(heatmaps)).unsqueeze(0).to(device)
    weight_t = torch.from_numpy(np.stack(weights)).unsqueeze(0).to(device)
    return window_t, target_t, weight_t


def train_step(
    model: Detector3D,
    optimizer: torch.optim.Optimizer,
    sample: bio_io.Sample,
    t_start: int,
    window_size: int,
    downsample: tuple[int, int, int] = DOWNSAMPLE,
    sigma_um: float = SIGMA_UM,
    device: str | torch.device = "cpu",
) -> float:
    """One optimizer step on one training window. Returns the scalar loss."""
    window, target, weight = build_training_window(sample, t_start, window_size, downsample, sigma_um, device)

    model.train()
    optimizer.zero_grad()
    pred = model(window)
    
    # --- REPLACED: targets.weighted_heatmap_mse_loss ---
    loss = focal_heatmap_loss(pred, target, weight)
    # ---------------------------------------------------
    
    loss.backward()

    # --- NEW: Clip gradients to prevent explosions ---
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # -----------------------------------------------

    optimizer.step()
    return loss.item()