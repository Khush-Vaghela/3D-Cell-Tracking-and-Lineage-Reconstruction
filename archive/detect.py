"""
Phase 1.4 -- turn a detector heatmap into a set of (t, z, y, x, score)
node detections in NATIVE voxel coordinates.

Restored original peak_local_max implementation to accurately detect 
touching daughter cells during division events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
import torch

from img_proc import nms_3d
import preprocessing as prep


def extract_peaks_single_frame(
    heatmap: np.ndarray,
    threshold: float,
    min_distance_um: float,
    scale_down: tuple[float, float, float],
) -> pd.DataFrame:
    if heatmap.ndim != 3:
        raise ValueError(f"expected a single (Z, Y, X) frame, got shape {heatmap.shape}")

    # Use peak_local_max to find multiple centers even within merged probability blobs
    coords = peak_local_max(heatmap, threshold_abs=threshold, exclude_border=False)
    if len(coords) == 0:
        return pd.DataFrame({"z": [], "y": [], "x": [], "score": []})

    scores = heatmap[coords[:, 0], coords[:, 1], coords[:, 2]]
    
    # Use physical-space NMS to deduplicate
    keep_idx = nms_3d(coords, scores, min_distance=min_distance_um, scale=scale_down)

    kept = coords[keep_idx]
    kept_scores = scores[keep_idx]
    
    # Cast to float to satisfy downstream upsampling compatibility
    return pd.DataFrame({
        "z": kept[:, 0].astype(float),
        "y": kept[:, 1].astype(float),
        "x": kept[:, 2].astype(float),
        "score": kept_scores,
    })


def peaks_to_native_coords(
    peaks_down: pd.DataFrame,
    downsample: tuple[int, int, int],
) -> pd.DataFrame:
    """Convert a downsampled-grid peaks DataFrame to native voxel-grid coordinates."""
    if len(peaks_down) == 0:
        return peaks_down.copy()
    return prep.upsample_node_coords(peaks_down, downsample)


def compute_heatmap_single_frame(model, frame_processed):
    """Runs the model on a single preprocessed frame to get the raw heatmap."""
    device = next(model.parameters()).device
    model.eval()
    
    tensor_frame = torch.from_numpy(frame_processed).unsqueeze(0).to(device)
    
    with torch.no_grad():
        heatmap_tensor = model.detect_single_frame(tensor_frame)
        
    return heatmap_tensor.squeeze(0).cpu().numpy()

def detect_frame(
    model,
    frame_processed: np.ndarray,
    downsample: tuple[int, int, int],
    native_scale: tuple[float, float, float],
    threshold: float = 0.3,
    min_distance_um: float = 5.0,
) -> pd.DataFrame:
    """
    End-to-end single-frame detection: preprocessed frame -> model ->
    heatmap -> peaks (downsampled grid) -> native voxel coordinates.

    frame_processed : (Z, Y, X) already normalized + downsampled (i.e.
                       already run through preprocessing.preprocess_window
                       for a single frame).
    Returns DataFrame[z, y, x, score] in NATIVE voxel coordinates.
    """
    import torch

    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        frame_t = torch.from_numpy(frame_processed).unsqueeze(0).to(device)  # (1, Z, Y, X)
        heatmap = model.detect_single_frame(frame_t)[0].cpu().numpy()  # (Z, Y, X)

    scale_down = prep.downsampled_scale(native_scale, downsample)
    peaks_down = extract_peaks_single_frame(heatmap, threshold, min_distance_um, scale_down)
    return peaks_to_native_coords(peaks_down, downsample)