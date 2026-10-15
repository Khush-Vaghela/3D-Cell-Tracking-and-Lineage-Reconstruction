"""
Phase 1.2 -- detector architecture.

Backbone: TemporalUNet3D from the official competition baseline repo
(temporal_unet.py, copied verbatim -- see that file's header). Takes a
window of consecutive frames and produces per-frame feature maps, using
per-voxel attention *across time* at each encoder stage (skipped at full
resolution for speed, per the original implementation).

On top of that backbone this module adds a detection head (a single
Conv3d, mirroring the official repo's `detect_head`) but trains it
against this project's own target scheme from targets.py -- anisotropic
Gaussian heatmaps with sparse-GT-aware per-voxel loss weighting -- rather
than the official repo's single-voxel + flat neg_weight scheme. That was
a deliberate choice (see conversation), not an oversight: the two target
schemes are incompatible loss targets, so this file commits to one.

Output activation is sigmoid, not raw logits: targets.build_heatmap_target
produces continuous values in [0, 1] (Gaussian bumps), not the official
repo's hard 0/1 single-voxel target, so plain BCE-with-logits (which
assumes a hard binary target) isn't the right loss here -- this file
pairs a sigmoid output with targets.weighted_heatmap_mse_loss instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from temporal_unet import TemporalUNet3D


@dataclass
class DetectorConfig:
    unet_out_channels: int = 32
    unet_layers: tuple[int, ...] = (32, 64, 128)
    gradient_checkpointing: bool = True
    skip_fullres_temporal: bool = True


class Detector3D(nn.Module):
    """TemporalUNet3D backbone + a 1x1x1 conv detection head.

    forward(window) -> heatmap predictions, shape (B, T, Z, Y, X), values
    in [0, 1] via sigmoid -- directly comparable to targets.py's
    build_heatmap_target output with targets.weighted_heatmap_mse_loss.
    """

    def __init__(self, config: DetectorConfig = DetectorConfig()):
        super().__init__()
        self.config = config
        self.unet = TemporalUNet3D(
            in_channels=1,
            out_channels=config.unet_out_channels,
            layers=list(config.unet_layers),
            gradient_checkpointing=config.gradient_checkpointing,
            skip_fullres_temporal=config.skip_fullres_temporal,
        )
        self.detect_head = nn.Conv3d(config.unet_out_channels, 1, kernel_size=1)

        # --- NEW: Focal Loss Bias Initialization ---
        # Force the network to initially predict a 1% probability for all voxels
        # to prevent gradient explosions from the massive empty background.
        import math
        pi = 0.01 
        bias_value = -math.log((1 - pi) / pi)
        torch.nn.init.constant_(self.detect_head.bias, bias_value)
        # -------------------------------------------

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """
        window : (B, T, Z, Y, X) -- a window of consecutive raw/normalized
                  frames, single channel (no explicit channel dim; added
                  here to match TemporalUNet3D's (B, T, C_in, Z, Y, X)).
        returns: (B, T, Z, Y, X) heatmap predictions in [0, 1].
        """
        if window.ndim != 5:
            raise ValueError(f"expected (B, T, Z, Y, X), got shape {tuple(window.shape)}")
        x = window.unsqueeze(2)  # (B, T, 1, Z, Y, X)
        feat = self.unet(x)  # (B, T, C_feat, Z, Y, X)
        B, T = feat.shape[:2]
        feat_flat = feat.reshape(B * T, *feat.shape[2:])
        logits = self.detect_head(feat_flat)  # (B*T, 1, Z, Y, X)
        logits = logits.reshape(B, T, *logits.shape[1:])  # (B, T, 1, Z, Y, X)
        logits = logits[:, :, 0]  # drop channel dim -> (B, T, Z, Y, X)
        return torch.sigmoid(logits)

    def detect_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Convenience path for standalone single-frame detection, mirroring
        the official repo's detect() -- duplicates the frame into a
        length-2 window so the temporal-attention U-Net has something to
        run on, then returns only the first frame's prediction.

        frame  : (B, Z, Y, X)
        returns: (B, Z, Y, X)
        """
        window = torch.stack([frame, frame], dim=1)  # (B, 2, Z, Y, X)
        return self.forward(window)[:, 0]
