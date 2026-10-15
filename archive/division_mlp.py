"""
Phase 4.5 -- learned division gate.

DivisionScorerMLP is NOT the same model as learned_association.py's
EdgeScorerMLP (Phase 2): different feature set (division_features.py's 9
division-specific features vs Phase 2's 6 generic continuation features),
different training data (division candidates only), different place in
the pipeline (gates Phase 4's rule-based output, not every continuation
link).

Labeling methodology mirrors Phase 2's learned_association.py exactly,
for the same reason: a division candidate (parent, daughter2) is a
POSITIVE only if both endpoints match real GT nodes (node_validation.py)
AND those GT nodes have a real GT edge between them; SKIPPED (not
negative) if either endpoint doesn't match any GT node, since sparse GT
means an unmatched prediction could be a real-but-unlabeled cell, not
necessarily a false positive.

Training-data generation runs the full Phase 1->2->3->4 pipeline over
real training samples -- this genuinely cannot be built or tested without
that full pipeline existing first (same chicken-and-egg dependency the
original MLP proposal flagged), so this module's generate_training_data
is only exercisable end-to-end once a trained Phase 1 checkpoint exists.
Its pieces (feature extraction, labeling, the MLP itself, training loop)
are each independently tested against synthetic/hand-crafted inputs here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import data_io as bio_io
import preprocessing as prep
import detect
import association as assoc
import tracker
import gap_closing as gc
import division_candidates as dc
import division_features as dfeat
import node_validation as nv
from pipeline_config import DOWNSAMPLE


class DivisionScorerMLP(nn.Module):
    """9 hand-crafted division features -> scalar logit."""

    def __init__(self, n_features: int = len(dfeat.FEATURE_ORDER)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits, shape (N,)

    def predict_proba(self, features: dict) -> float:
        vec = torch.from_numpy(dfeat.features_dict_to_vector(features)).unsqueeze(0)
        
        # --- THE FIX: Send the input tensor to the model's device ---
        device = next(self.parameters()).device
        vec = vec.to(device)
        # ------------------------------------------------------------
        
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(vec)).item()


@dataclass
class DivisionTrainingConfig:
    intensity_radius_um: float = 5.0
    match_max_distance_um: float = 7.0
    division_config: dc.DivisionConfig = field(default_factory=dc.DivisionConfig)
    detect_threshold: float = 0.3
    nms_min_distance_um: float = 5.0


def _run_phase1_to_4(
    model,
    sample: bio_io.Sample,
    config: DivisionTrainingConfig,
    downsample: tuple[int, int, int],
) -> tuple[tracker.TrackingResult, list[dc.DivisionCandidate], dict[int, np.ndarray]]:
    """Runs Phase 1 (inference) -> 2 -> 3 -> 4(rule-based) for one sample.
    Returns the pre-division tracking result, the rule-gated division
    candidates, and the raw per-frame native volumes (needed for the
    intensity features -- kept in memory here rather than re-read later)."""
    vol = bio_io.open_zarr_volume(sample.zarr_path)
    n_frames = vol.shape[0]

    detections_per_frame = {}
    heatmaps_per_frame = {}
    raw_frames = {}
    for t in range(n_frames):
        raw_frame = np.asarray(vol[t])
        raw_frames[t] = raw_frame
        frame_processed = prep.preprocess_window(raw_frame[None], downsample)[0]
        heatmap = detect.compute_heatmap_single_frame(model, frame_processed)
        heatmaps_per_frame[t] = heatmap
        scale_down = prep.downsampled_scale(bio_io.NATIVE_SCALE, downsample)
        peaks_down = detect.extract_peaks_single_frame(heatmap, config.detect_threshold, config.nms_min_distance_um, scale_down)
        detections_per_frame[t] = detect.peaks_to_native_coords(peaks_down, downsample)

    tracking_result = tracker.track_video(detections_per_frame, bio_io.NATIVE_SCALE, config.division_config.assoc_config)
    tracking_result = gc.close_gaps(tracking_result, heatmaps_per_frame, downsample, bio_io.NATIVE_SCALE, n_frames)
    candidates = dc.generate_division_candidates(tracking_result, bio_io.NATIVE_SCALE, config.division_config)

    return tracking_result, candidates, raw_frames


def _match_predicted_nodes_to_gt(
    tracking_result: tracker.TrackingResult,
    geff,
    match_max_distance_um: float,
) -> dict[int, int]:
    """node_id -> gt_node_id (only for matched nodes); unmatched predicted
    nodes are simply absent from the returned dict."""
    node_to_gt: dict[int, int] = {}
    for t, group in tracking_result.nodes.groupby("t"):
        gt_nodes_t = geff.nodes[geff.nodes["t"] == t][["node_id", "z", "y", "x"]].reset_index(drop=True)
        if len(gt_nodes_t) == 0:
            continue
        pred = group.assign(t=t)[["t", "z", "y", "x"]].reset_index(drop=True)
        row_matches = nv.match_nodes(pred, gt_nodes_t.assign(t=t)[["t", "z", "y", "x"]], bio_io.NATIVE_SCALE, match_max_distance_um)
        for pred_node_id, match_idx in zip(group["node_id"], row_matches):
            if match_idx != -1:
                node_to_gt[pred_node_id] = int(gt_nodes_t.iloc[match_idx]["node_id"])
    return node_to_gt


def build_features_for_candidate(
    candidate: dc.DivisionCandidate,
    tracking_result: tracker.TrackingResult,
    raw_frames_by_t: dict[int, np.ndarray],
    native_scale: tuple[float, float, float],
    intensity_radius_um: float = 5.0,
) -> dict:
    """
    Builds the real 9-feature division_features.py dict for one
    candidate. Deliberately separate from DivisionCandidate.features
    (division_candidates.py) -- that field holds Phase 2's generic
    association._pair_features dict (used for the rule-based cost gate),
    NOT these division-specific features. Conflating the two was a real
    bug caught while wiring up threshold tuning; this function is the
    single place both generate_training_data and inference-time MLP
    filtering should build the 9-feature vector from, so they can't
    silently diverge.
    """
    node_info = tracking_result.nodes.set_index("node_id")[["t", "z", "y", "x", "score"]].to_dict("index")
    parent_info = node_info[candidate.parent_id]
    d1_info = node_info[candidate.daughter1_id]
    d2_info = node_info[candidate.daughter2_id]

    parent_pos = np.array([parent_info["z"], parent_info["y"], parent_info["x"]])
    d1_pos = np.array([d1_info["z"], d1_info["y"], d1_info["x"]])
    d2_pos = np.array([d2_info["z"], d2_info["y"], d2_info["x"]])

    parent_intensity = dfeat.compute_local_intensity(raw_frames_by_t[parent_info["t"]], parent_pos, intensity_radius_um, native_scale)
    d1_intensity = dfeat.compute_local_intensity(raw_frames_by_t[d1_info["t"]], d1_pos, intensity_radius_um, native_scale)
    d2_intensity = dfeat.compute_local_intensity(raw_frames_by_t[d2_info["t"]], d2_pos, intensity_radius_um, native_scale)

    return dfeat.build_division_features(
        parent_pos, d1_pos, d2_pos,
        parent_info["score"], d1_info["score"], d2_info["score"],
        parent_intensity, d1_intensity, d2_intensity,
        candidate.cost, native_scale,
    )


def generate_training_data(
    model,
    samples: list[bio_io.Sample],
    config: DivisionTrainingConfig = DivisionTrainingConfig(),
    downsample: tuple[int, int, int] = DOWNSAMPLE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (X, y): X shape (N, 9), y shape (N,) binary labels (1 = real
    GT division edge, 0 = hard negative -- passed Phase 4's rule-based
    gates but isn't a real GT edge).
    """
    all_X, all_y = [], []

    for sample in samples:
        if sample.geff_path is None:
            continue
        geff = bio_io.load_geff(sample.geff_path)
        gt_edge_set = set(zip(geff.edges["source_id"], geff.edges["target_id"]))

        tracking_result, candidates, raw_frames = _run_phase1_to_4(model, sample, config, downsample)
        if not candidates:
            continue
        node_to_gt = _match_predicted_nodes_to_gt(tracking_result, geff, config.match_max_distance_um)

        for cand in candidates:
            parent_gt = node_to_gt.get(cand.parent_id)
            d2_gt = node_to_gt.get(cand.daughter2_id)
            if parent_gt is None or d2_gt is None:
                continue  # can't determine ground truth -- skip, don't guess

            label = 1 if (parent_gt, d2_gt) in gt_edge_set else 0
            features = build_features_for_candidate(cand, tracking_result, raw_frames, bio_io.NATIVE_SCALE, config.intensity_radius_um)
            all_X.append(dfeat.features_dict_to_vector(features))
            all_y.append(label)

    if not all_X:
        return np.empty((0, len(dfeat.FEATURE_ORDER)), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(all_X), np.array(all_y, dtype=np.float32)


def train_division_scorer(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 150, # <-- Updated default to match main() script
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    seed: int = 0,
    pos_weight_multiplier: float = 2.0, # <-- NEW: Extra penalty for missing divisions
) -> tuple[DivisionScorerMLP, dict]:
    """Class-balanced BCE, same pattern as Phase 2's train_edge_scorer."""
    if len(X) == 0:
        raise ValueError("no training pairs -- generate_training_data returned nothing")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training DivisionScorerMLP on device: {device}")

    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    # --- Type casting and device pushing ---
    X_train = torch.from_numpy(X[train_idx]).float().to(device)
    y_train = torch.from_numpy(y[train_idx]).float().to(device)
    
    if n_val > 0:
        X_val = torch.from_numpy(X[val_idx]).float().to(device)
        y_val = torch.from_numpy(y[val_idx]).float().to(device)
    else:
        X_val, y_val = None, None

    # --- Enforce strict (N, 1) shape to prevent BCE broadcasting errors ---
    y_train = y_train.view(-1, 1)
    if y_val is not None:
        y_val = y_val.view(-1, 1)

    # --- Aggressive Positive Weighting ---
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    
    # Calculate base weight and apply the aggressive multiplier
    base_weight = n_neg / max(n_pos, 1.0)
    final_weight = base_weight * pos_weight_multiplier
    pos_weight = torch.tensor([final_weight], dtype=torch.float32).to(device)

    model = DivisionScorerMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = {"train_loss": [], "val_loss": []}
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # --- THE FIX: Force both to be flat 1D tensors ---
        preds = model(X_train).view(-1)
        targets = y_train.view(-1)
        loss = loss_fn(preds, targets)
        # -------------------------------------------------
        
        loss.backward()
        optimizer.step()
        history["train_loss"].append(loss.item())

        if X_val is not None and len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val).view(-1)
                val_targets = y_val.view(-1)
                history["val_loss"].append(loss_fn(val_preds, val_targets).item())

    return model, history