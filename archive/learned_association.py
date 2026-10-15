"""
learned_prob_ij -- the Phase 2 association learned edge scorer.

NOT the same thing as Phase 4's division-classifier MLP: this one scores
every normal candidate track-to-detection continuation link (generic
motion/confidence features), the division MLP only scores already-gated
division candidates (parent + 2 daughters, volume/symmetry/divergence
features). Different feature sets, different training data, different
place in the pipeline -- see the conversation this was designed in.

Training data generation (the "hard part", same shape as the Phase 4 MLP
design): run the real detector across real training samples, enumerate
every geometrically-gated candidate pair per frame (association.py's
enumerate_gated_candidates -- the exact same candidate set the production
cost path considers, not a separately-imagined one), and label each
candidate using GT-node matching:
  - POSITIVE: both endpoints match real GT nodes (node_validation.py),
    AND those GT nodes have a real GT edge between them.
  - NEGATIVE (hard negative): both endpoints match real GT nodes, but
    there's no GT edge between them -- this candidate passed the
    geometric gate but is wrong, exactly the kind of mistake a learned
    scorer should learn to downweight.
  - SKIPPED: either endpoint doesn't match any GT node. We can't tell
    whether an unmatched prediction is a real-but-unlabeled cell (GT is
    sparse) or a genuine false detection -- labeling it either way would
    inject noise, so it's excluded rather than guessed at, consistent
    with how sparsity is handled everywhere else in this project.
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
import node_validation as nv
from pipeline_config import DOWNSAMPLE


FEATURE_ORDER = [
    "predicted_position_error", "raw_distance", "velocity_consistency",
    "track_confidence", "detection_confidence", "temporal_gap",
]


def features_dict_to_vector(features: dict) -> np.ndarray:
    return np.array([features[k] for k in FEATURE_ORDER], dtype=np.float32)


class EdgeScorerMLP(nn.Module):
    """Small MLP: 6 hand-crafted features -> scalar logit. Sigmoid applied
    separately (predict_proba) so training can use BCEWithLogitsLoss,
    which is numerically more stable than sigmoid+BCE split across two ops."""

    def __init__(self, n_features: int = len(FEATURE_ORDER)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits, shape (N,)

    def predict_proba(self, features: dict) -> float:
        """Convenience for a single candidate's feature dict -- this is
        what association.py's cost function would call at inference time
        once a trained scorer is wired in (not yet done automatically;
        see the module docstring in association.py)."""
        vec = torch.from_numpy(features_dict_to_vector(features)).unsqueeze(0)
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(vec)).item()


@dataclass
class TrainingDataConfig:
    candidate_gate_um: float = 15.0     # broad net when enumerating candidates -- wider than production's base_gate_um, to harvest more hard negatives
    match_max_distance_um: float = 7.0  # GT node-matching radius, matches the competition's own edge tolerance
    assoc_config: assoc.AssociationConfig = field(default_factory=assoc.AssociationConfig)  # only used for track state advancement between frames, not for gating candidates (see candidate_gate_um)


def _detect_all_frames(model, sample: bio_io.Sample, downsample) -> dict[int, pd.DataFrame]:
    vol = bio_io.open_zarr_volume(sample.zarr_path)
    detections = {}
    for t in range(vol.shape[0]):
        raw_frame = np.asarray(vol[t])[None]
        frame_processed = prep.preprocess_window(raw_frame, downsample)[0]
        detections[t] = detect.detect_frame(model, frame_processed, downsample, bio_io.NATIVE_SCALE)
    return detections


def _generate_pairs_from_detections(
    detections_per_frame: dict[int, pd.DataFrame],
    geff: "bio_io.GeffData",
    config: TrainingDataConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The actual labeling logic, factored out from generate_training_pairs
    so it's directly unit-testable against hand-crafted detections and GT
    -- without needing a real trained detector to produce detections_per_frame.
    """
    all_features = []
    all_labels = []

    gt_edge_set = set(zip(geff.edges["source_id"], geff.edges["target_id"]))

    gt_match_per_frame: dict[int, np.ndarray] = {}
    for t, dets in detections_per_frame.items():
        gt_nodes_t = geff.nodes[geff.nodes["t"] == t][["node_id", "z", "y", "x"]].reset_index(drop=True)
        if len(dets) == 0 or len(gt_nodes_t) == 0:
            gt_match_per_frame[t] = np.full(len(dets), -1, dtype=np.int64)
            continue
        row_matches = nv.match_nodes(
            dets.assign(t=t)[["t", "z", "y", "x"]], gt_nodes_t.assign(t=t)[["t", "z", "y", "x"]],
            bio_io.NATIVE_SCALE, config.match_max_distance_um,
        )
        gt_match_per_frame[t] = np.array([
            gt_nodes_t.iloc[i]["node_id"] if i != -1 else -1 for i in row_matches
        ])

    active_tracks: list[assoc.Track] = []
    active_track_gt_ids: dict[int, int] = {}  # track_id -> gt_node_id of its most recent detection (-1 if unmatched)
    next_track_id = 1

    for t in sorted(detections_per_frame.keys()):
        detections = detections_per_frame[t].reset_index(drop=True)
        gt_ids_this_frame = gt_match_per_frame[t]

        if active_tracks and len(detections) > 0:
            # crowding_scale=None deliberately -- unlike production
            # association, training-data generation wants the broadest
            # candidate net regardless of local density, specifically to
            # harvest hard negatives from dense regions (exactly where a
            # learned scorer earns its keep). Applying the same crowding
            # gate used at inference would silently exclude most
            # candidates in exactly the areas most worth learning from.
            candidates = assoc.enumerate_gated_candidates(
                active_tracks, detections, bio_io.NATIVE_SCALE,
                config.candidate_gate_um, None, config.assoc_config,
            )
            for cand in candidates:
                track = active_tracks[cand["track_index"]]
                track_gt_id = active_track_gt_ids.get(track.track_id, -1)
                det_gt_id = gt_ids_this_frame[cand["detection_index"]]
                if track_gt_id == -1 or det_gt_id == -1:
                    continue  # can't determine ground truth -- skip, don't guess
                label = 1 if (track_gt_id, det_gt_id) in gt_edge_set else 0
                all_features.append(features_dict_to_vector(cand["features"]))
                all_labels.append(label)

        # advance track state using the real (pure-geometric) association,
        # so the NEXT frame's candidates come from realistic track state
        if active_tracks:
            result = assoc.associate_frame(active_tracks, detections, bio_io.NATIVE_SCALE, config.assoc_config)
        else:
            result = assoc.AssociationResult([], [], list(range(len(detections))))

        new_active_tracks = []
        for track_idx, det_idx in result.matched_pairs:
            track = active_tracks[track_idx]
            det_row = detections.iloc[det_idx]
            track.positions.append((det_row["z"], det_row["y"], det_row["x"]))
            track.confidences.append(float(det_row["score"]) if "score" in detections.columns else 1.0)
            track.timepoints.append(t)
            active_track_gt_ids[track.track_id] = gt_ids_this_frame[det_idx]
            new_active_tracks.append(track)
        for det_idx in result.unmatched_detection_indices:
            det_row = detections.iloc[det_idx]
            new_track = assoc.Track(
                track_id=next_track_id,
                positions=[(det_row["z"], det_row["y"], det_row["x"])],
                confidences=[float(det_row["score"]) if "score" in detections.columns else 1.0],
                timepoints=[t],
            )
            next_track_id += 1
            active_track_gt_ids[new_track.track_id] = gt_ids_this_frame[det_idx]
            new_active_tracks.append(new_track)
        active_tracks = new_active_tracks

    if not all_features:
        return np.empty((0, len(FEATURE_ORDER)), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(all_features), np.array(all_labels, dtype=np.float32)


def generate_training_pairs(
    model,
    samples: list[bio_io.Sample],
    config: TrainingDataConfig = TrainingDataConfig(),
    downsample: tuple[int, int, int] = DOWNSAMPLE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (X, y): X shape (N, 6) feature vectors, y shape (N,) binary
    labels (1 = real GT edge, 0 = hard negative -- geometrically
    plausible but not a real edge).
    """
    all_X, all_y = [], []
    for sample in samples:
        if sample.geff_path is None:
            continue
        geff = bio_io.load_geff(sample.geff_path)
        detections_per_frame = _detect_all_frames(model, sample, downsample)
        X, y = _generate_pairs_from_detections(detections_per_frame, geff, config)
        if len(X):
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        return np.empty((0, len(FEATURE_ORDER)), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.concatenate(all_X), np.concatenate(all_y)


def train_edge_scorer(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[EdgeScorerMLP, dict]:
    """
    Class-balanced BCE (pos_weight = n_neg/n_pos, standard for imbalanced
    binary classification -- true edges are the minority class among all
    geometrically-plausible candidates). Returns the trained model and a
    small history dict for sanity-checking convergence.
    """
    if len(X) == 0:
        raise ValueError("no training pairs -- generate_training_pairs returned nothing (check that samples have GT and detections)")

    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    X_val = torch.from_numpy(X[val_idx]) if n_val > 0 else None
    y_val = torch.from_numpy(y[val_idx]) if n_val > 0 else None

    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0))

    model = EdgeScorerMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        history["train_loss"].append(loss.item())

        if X_val is not None and len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_loss = loss_fn(val_logits, y_val).item()
            history["val_loss"].append(val_loss)

    return model, history
