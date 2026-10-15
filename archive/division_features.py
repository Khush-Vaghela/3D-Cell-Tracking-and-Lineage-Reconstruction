"""
Phase 4.5 feature extraction.

compute_local_intensity computes a raw-intensity integral in a small box 
around each node directly from a lazy Zarr volume, preventing out-of-memory errors.
"""

from __future__ import annotations

import numpy as np

FEATURE_ORDER = [
    "distance_to_daughter1", "distance_to_daughter2",
    "intensity_ratio", "symmetry_ratio", "divergence_angle_deg",
    "parent_confidence", "daughter1_confidence", "daughter2_confidence",
    "association_cost",
]


def compute_local_intensity(
    vol, # Pass the lazy Zarr volume directly!
    t: int,
    position_native: np.ndarray,
    radius_um: float,
    native_scale: tuple[float, float, float],
) -> float:
    """
    Sum of raw intensity in a physical box around position_native.
    Lazily queries the exact bounding box from the Zarr volume without 
    loading the entire frame into memory.
    """
    radius_voxels = np.array([radius_um / s for s in native_scale])
    z0, y0, x0 = np.round(position_native).astype(int)
    rz, ry, rx = np.round(radius_voxels).astype(int)

    # Note: vol.shape is (T, Z, Y, X)
    _, Z, Y, X = vol.shape 
    
    z_lo, z_hi = max(0, z0 - rz), min(Z, z0 + rz + 1)
    y_lo, y_hi = max(0, y0 - ry), min(Y, y0 + ry + 1)
    x_lo, x_hi = max(0, x0 - rx), min(X, x0 + rx + 1)

    if z_lo >= z_hi or y_lo >= y_hi or x_lo >= x_hi:
        return 0.0
        
    # Magic happens here: This 4D slice tells Zarr to only fetch this exact 
    # tiny chunk of data from the hard drive, bypassing the need for a custom cache.
    local_box = vol[t, z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
    
    return float(local_box.sum())


def build_division_features(
    parent_pos: np.ndarray, daughter1_pos: np.ndarray, daughter2_pos: np.ndarray,
    parent_conf: float, daughter1_conf: float, daughter2_conf: float,
    parent_intensity: float, daughter1_intensity: float, daughter2_intensity: float,
    association_cost: float,
    native_scale: tuple[float, float, float],
    eps: float = 1e-6,
) -> dict[str, float]:
    scale_arr = np.array(native_scale)
    parent_physical = parent_pos * scale_arr
    d1_physical = daughter1_pos * scale_arr
    d2_physical = daughter2_pos * scale_arr

    distance_to_daughter1 = float(np.linalg.norm(d1_physical - parent_physical))
    distance_to_daughter2 = float(np.linalg.norm(d2_physical - parent_physical))

    intensity_ratio = float((daughter1_intensity + daughter2_intensity) / (parent_intensity + eps))
    symmetry_ratio = float(daughter1_intensity / (daughter2_intensity + eps))

    v1 = d1_physical - parent_physical
    v2 = d2_physical - parent_physical
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        divergence_angle_deg = 0.0  
    else:
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        divergence_angle_deg = float(np.degrees(np.arccos(cos_angle)))

    return {
        "distance_to_daughter1": distance_to_daughter1,
        "distance_to_daughter2": distance_to_daughter2,
        "intensity_ratio": intensity_ratio,
        "symmetry_ratio": symmetry_ratio,
        "divergence_angle_deg": divergence_angle_deg,
        "parent_confidence": float(parent_conf),
        "daughter1_confidence": float(daughter1_conf),
        "daughter2_confidence": float(daughter2_conf),
        "association_cost": float(association_cost),
    }


def features_dict_to_vector(features: dict) -> np.ndarray:
    return np.array([features[k] for k in FEATURE_ORDER], dtype=np.float32)