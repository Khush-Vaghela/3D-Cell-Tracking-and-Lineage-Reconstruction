"""
Phase 1.1 -- sparse-GT-aware detection targets.

The GT gives point centroids only (no masks), and is *sparse*: not every
real cell in a frame is labeled. Two things this module has to get right,
in order:

1. Turn sparse points into a heatmap regression target (a Gaussian bump
   per labeled cell, anisotropic to match the physical voxel scale).
2. Produce a *loss weight map* alongside it, so training doesn't punish
   the model for lighting up on real-but-unlabeled cells. Standard
   heatmap-regression loss (dense MSE/focal against the full frame)
   implicitly treats "unlabeled" as "confirmed empty", which is false here
   -- get this wrong and you reproduce exactly the recall ceiling the
   reference notebook's own U-Nets hit (0.57-0.66).

Key assumption, flagged because the dataset card doesn't fully pin this
down: `estimated_number_of_nodes` is described as a single per-*sample*
estimate, but a sample spans ~100 timepoints during which cell count
grows (divisions). There's no stated per-frame breakdown. This module
allocates that single estimate across frames proportionally to how many
GT nodes are actually labeled at each frame -- a reasonable heuristic,
not a fact from the spec. Treat `estimate_true_count_per_frame` as an
assumption to sanity-check against real data (e.g. does labeled count
grow roughly monotonically the way you'd expect from divisions?), not as
ground truth.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-frame true-count estimation (the sparse-GT heuristic)
# ---------------------------------------------------------------------------

def per_frame_labeled_counts(nodes_df: pd.DataFrame, n_timepoints: int) -> np.ndarray:
    """Number of GT-labeled nodes at each timepoint, shape (n_timepoints,)."""
    counts = np.zeros(n_timepoints, dtype=np.int64)
    vc = nodes_df["t"].value_counts()
    for t, c in vc.items():
        if 0 <= t < n_timepoints:
            counts[int(t)] = c
    return counts

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


def estimate_true_count_per_frame(
    nodes_df: pd.DataFrame,
    estimated_number_of_nodes: int,
    n_timepoints: int,
) -> np.ndarray:
    """
    Allocate the single per-sample `estimated_number_of_nodes` estimate
    across frames proportionally to labeled count per frame. Documented
    assumption, not a spec fact -- see module docstring.

    Returns float array, shape (n_timepoints,), each entry >= the actual
    labeled count at that frame (clamped so the ratio in
    `background_weight_per_frame` never exceeds 1.0 even if the
    proportional allocation undershoots at some frame).
    """
    labeled = per_frame_labeled_counts(nodes_df, n_timepoints)
    total_labeled = labeled.sum()
    if total_labeled == 0:
        # no labels at all -- nothing to allocate against; fall back to a
        # uniform split so callers get a defined (if uninformative) value
        return np.full(n_timepoints, estimated_number_of_nodes / n_timepoints, dtype=np.float64)

    estimated = estimated_number_of_nodes * (labeled / total_labeled)
    return np.maximum(estimated, labeled)  # never estimate fewer true cells than we've labeled


def background_weight_per_frame(
    nodes_df: pd.DataFrame,
    estimated_number_of_nodes: int,
    n_timepoints: int,
) -> np.ndarray:
    """
    Per-frame scalar weight (in [0, 1]) to apply to *background* loss --
    i.e. how much to trust "unlabeled implies empty" at that frame. Low
    where labeling is sparse relative to the estimated true count, close
    to 1.0 where labeling is close to complete.
    """
    labeled = per_frame_labeled_counts(nodes_df, n_timepoints)
    estimated = estimate_true_count_per_frame(nodes_df, estimated_number_of_nodes, n_timepoints)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(estimated > 0, labeled / estimated, 0.0)
    return np.clip(ratio, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Heatmap target generation
# ---------------------------------------------------------------------------

def sigma_voxels(sigma_um: float, scale: tuple[float, float, float]) -> np.ndarray:
    """Convert an isotropic physical sigma (µm) to per-axis (z, y, x) voxel
    sigma, given the anisotropic voxel scale. z is coarser (1.625 µm/voxel
    vs 0.406 for y/x), so the same physical spread is a *smaller* sigma in
    z voxels than in y/x voxels."""
    return sigma_um / np.asarray(scale, dtype=np.float64)


def build_heatmap_target(
    frame_shape: tuple[int, int, int],
    nodes_t: pd.DataFrame,
    scale: tuple[float, float, float],
    sigma_um: float = 3.0,
    cutoff_sigmas: float = 3.0,
) -> np.ndarray:
    """
    Build a single frame's heatmap target from the GT nodes labeled at
    that timepoint.

    frame_shape : (Z, Y, X)
    nodes_t     : DataFrame already filtered to one timepoint, columns z,y,x
    scale       : (z, y, x) µm/voxel, e.g. (1.625, 0.40625, 0.40625)
    sigma_um    : physical Gaussian spread (µm) -- pick relative to actual
                  cell radius in this dataset once you've measured it in
                  EDA; 3.0 is a placeholder, not a measured value.

    Overlapping Gaussians combine via elementwise max, not sum -- summing
    would inflate intensity in dense/crowded regions and blur separate
    peaks into one, which is exactly wrong for a detector meant to find
    individual cells in dense tissue.
    """
    Z, Y, X = frame_shape
    heatmap = np.zeros(frame_shape, dtype=np.float32)
    sig_z, sig_y, sig_x = sigma_voxels(sigma_um, scale)

    for row in nodes_t.itertuples(index=False):
        z0, y0, x0 = float(row.z), float(row.y), float(row.x)

        z_lo, z_hi = max(0, int(z0 - cutoff_sigmas * sig_z)), min(Z, int(z0 + cutoff_sigmas * sig_z) + 1)
        y_lo, y_hi = max(0, int(y0 - cutoff_sigmas * sig_y)), min(Y, int(y0 + cutoff_sigmas * sig_y) + 1)
        x_lo, x_hi = max(0, int(x0 - cutoff_sigmas * sig_x)), min(X, int(x0 + cutoff_sigmas * sig_x) + 1)
        if z_lo >= z_hi or y_lo >= y_hi or x_lo >= x_hi:
            continue  # node sits entirely outside the frame -- skip

        zz, yy, xx = np.meshgrid(
            np.arange(z_lo, z_hi), np.arange(y_lo, y_hi), np.arange(x_lo, x_hi), indexing="ij"
        )
        bump = np.exp(
            -0.5 * (
                ((zz - z0) / sig_z) ** 2
                + ((yy - y0) / sig_y) ** 2
                + ((xx - x0) / sig_x) ** 2
            )
        ).astype(np.float32)

        # --- ADD THIS LINE ---
        bump = bump / bump.max()
        # ---------------------

        region = heatmap[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
        np.maximum(region, bump, out=region)

    return heatmap


def build_loss_weight_map(
    frame_shape: tuple[int, int, int],
    nodes_t: pd.DataFrame,
    scale: tuple[float, float, float],
    background_weight: float,
    positive_radius_um: float = 2.0,
) -> np.ndarray:
    """
    Per-voxel loss weight for one frame: 1.0 within `positive_radius_um`
    of any labeled cell (these are confirmed real, always fully
    supervised), `background_weight` everywhere else (down-weighted by
    how incomplete labeling is estimated to be at this frame, from
    `background_weight_per_frame`).
    """
    Z, Y, X = frame_shape
    weight = np.full(frame_shape, background_weight, dtype=np.float32)
    r_z, r_y, r_x = sigma_voxels(positive_radius_um, scale)  # reuse: same unit conversion

    for row in nodes_t.itertuples(index=False):
        z0, y0, x0 = float(row.z), float(row.y), float(row.x)
        z_lo, z_hi = max(0, int(z0 - r_z)), min(Z, int(z0 + r_z) + 1)
        y_lo, y_hi = max(0, int(y0 - r_y)), min(Y, int(y0 + r_y) + 1)
        x_lo, x_hi = max(0, int(x0 - r_x)), min(X, int(x0 + r_x) + 1)
        if z_lo >= z_hi or y_lo >= y_hi or x_lo >= x_hi:
            continue

        zz, yy, xx = np.meshgrid(
            np.arange(z_lo, z_hi), np.arange(y_lo, y_hi), np.arange(x_lo, x_hi), indexing="ij"
        )
        inside = (
            ((zz - z0) / r_z) ** 2 + ((yy - y0) / r_y) ** 2 + ((xx - x0) / r_x) ** 2
        ) <= 1.0
        region = weight[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
        region[inside] = 1.0

    return weight

