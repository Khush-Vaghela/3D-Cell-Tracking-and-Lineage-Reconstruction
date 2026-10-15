"""
Preprocessing layer tying data_io.py + img_proc.py + targets.py together,
matching the official baseline's *actual* default convention -- confirmed
by reading the real repo rather than assumed:

- Default downsample is (1, 4, 4): strided indexing, Z untouched, Y/X
  strided by 4. This is NOT img_proc.py's resample_image_to_isotropic
  (GPU trilinear interpolation) -- that function exists but is only used
  when an explicit target_scale is passed to open_dataset, which the
  default training config does not do.
- The reason (1, 4, 4) is the default and not some other stride: it's
  exactly isotropic. Native scale is (1.625, 0.40625, 0.40625) um/voxel;
  0.40625 * 4 == 1.625 exactly. So the downsampled grid has voxel size
  1.625um in all three axes with zero interpolation error -- confirmed by
  direct computation, not assumed.
- Normalization (quantile_normalize) happens BEFORE downsampling, on the
  full-resolution array, per the official _process_on_gpu docstring
  ("Processing order: 1. Normalize ... 2. Transfer to GPU 3. Resample").
  This module follows the same order for the striding path.

Because a non-default downsample tuple need not be exactly isotropic,
targets.py's build_heatmap_target still takes an explicit scale argument
rather than assuming (1,1,1) or hardcoding isotropy -- this module computes
the correct downsampled scale for whatever downsample tuple is used, and
that's what gets passed through.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from img_proc import quantile_normalize, nms_3d


def downsampled_scale(
    native_scale: tuple[float, float, float],
    downsample: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Physical voxel size (um) per axis after strided downsampling.
    Matches the official predict script's `voxel_size = scale * downsample`."""
    return tuple(s * d for s, d in zip(native_scale, downsample))


def downsample_volume(volume: np.ndarray, downsample: tuple[int, int, int]) -> np.ndarray:
    """Strided spatial downsample. volume: (..., Z, Y, X) -- works on a
    single frame (Z,Y,X) or a window (T,Z,Y,X), downsample applies to the
    last 3 axes either way."""
    dz, dy, dx = downsample
    return volume[..., ::dz, ::dy, ::dx]


def downsample_node_coords(nodes: pd.DataFrame, downsample: tuple[int, int, int]) -> pd.DataFrame:
    """Rescale GT node z/y/x (native voxel units) into the downsampled
    grid. Matches the official repo's "coordinates are returned in the
    downsampled grid" convention for its own training targets."""
    dz, dy, dx = downsample
    out = nodes.copy()
    out["z"] = nodes["z"] / dz
    out["y"] = nodes["y"] / dy
    out["x"] = nodes["x"] / dx
    return out


def upsample_node_coords(nodes: pd.DataFrame, downsample: tuple[int, int, int]) -> pd.DataFrame:
    """Inverse of downsample_node_coords -- convert predicted coordinates
    (downsampled grid) back to native voxel-grid integers for the
    submission CSV, which requires native-resolution integer coordinates."""
    dz, dy, dx = downsample
    out = nodes.copy()
    out["z"] = np.round(nodes["z"] * dz).astype(int)
    out["y"] = np.round(nodes["y"] * dy).astype(int)
    out["x"] = np.round(nodes["x"] * dx).astype(int)
    return out


def preprocess_window(
    raw_window: np.ndarray,
    downsample: tuple[int, int, int],
    gamma: float = 1.0,
) -> np.ndarray:
    """
    raw_window : (T, Z, Y, X) raw uint16 (or any numeric) volume window.
    Returns    : (T, Z, Y, X) float32, normalized then downsampled -- in
                 that order, matching the official pipeline (see module
                 docstring).
    """
    normalized = quantile_normalize(raw_window, gamma=gamma).astype(np.float32)
    # quantile_normalize's `** gamma` op upcasts float32 -> float64 in numpy
    # even for gamma=1.0 -- confirmed by testing against the actual copied
    # function, not assumed. Cast back explicitly so downstream torch
    # tensors are float32 as expected.
    return downsample_volume(normalized, downsample)
