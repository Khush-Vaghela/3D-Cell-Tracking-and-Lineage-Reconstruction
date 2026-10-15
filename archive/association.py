"""
Phase 2 -- association (frame t -> frame t+1 linking).

cost = predicted_position_error + w_raw*raw_distance
       - w_conf*min(conf_i, conf_j) - w_learned*learned_prob_ij

Two-pass Hungarian:
  1. Tight pass: crowding-aware gate, min(base_gate, crowding_scale *
     local_nearest_neighbor_distance) -- tightens in dense clusters,
     loosens in sparse regions, instead of one fixed threshold.
  2. Fallback pass on leftovers: looser fixed gate, and an ambiguity-
     margin check -- a candidate match is only accepted if its cost beats
     the next-best alternative for either endpoint by at least
     margin_ratio, otherwise it's left unmatched rather than forced.

learned_prob_ij is a placeholder (always 0) -- no learned edge scorer
exists yet. If one gets built later it slots in here as an additional
per-pair score, not a replacement for the geometric cost.

Scope: this module only produces frame-to-frame matches. It does NOT
attempt gap-closing across missed detections (Phase 3) or division
resolution (Phase 4) -- unmatched tracks/detections here are exactly the
input those later phases consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


@dataclass
class AssociationConfig:
    base_gate_um: float = 10.0        # tight-pass gate ceiling
    crowding_scale: float = 0.8       # gate = min(base_gate_um, crowding_scale * local_nn_distance)
    fallback_gate_um: float = 15.0    # looser gate for the second pass
    margin_ratio: float = 0.15        # fallback match rejected unless it beats runner-up cost by this fraction
    w_raw: float = 0.3
    w_conf: float = 2.0
    w_learned: float = 1.0            # no-op while learned_prob_ij stays 0


@dataclass
class Track:
    track_id: int
    positions: list[tuple[float, float, float]] = field(default_factory=list)  # (z, y, x), physical or voxel -- caller's choice, consistent throughout
    confidences: list[float] = field(default_factory=list)
    timepoints: list[int] = field(default_factory=list)

    def predict_next_position(self) -> np.ndarray:
        """Constant-velocity prediction. Zero-velocity (last position) if
        fewer than 2 observations exist yet."""
        pos = np.array(self.positions[-1])
        if len(self.positions) < 2:
            return pos
        prev = np.array(self.positions[-2])
        velocity = pos - prev
        return pos + velocity

    @property
    def last_position(self) -> np.ndarray:
        return np.array(self.positions[-1])

    @property
    def last_confidence(self) -> float:
        return self.confidences[-1]


@dataclass
class AssociationResult:
    matched_pairs: list[tuple[int, int]]       # (track_index, detection_index) into the input lists/DataFrame
    unmatched_track_indices: list[int]
    unmatched_detection_indices: list[int]


def _local_nn_distance_um(coords_physical: np.ndarray) -> np.ndarray:
    """Nearest-neighbor physical distance for each point to any other
    point in the same array. len==1 -> np.inf (no neighbor to be crowded by)."""
    n = len(coords_physical)
    if n < 2:
        return np.full(n, np.inf)
    tree = cKDTree(coords_physical)
    dists, _ = tree.query(coords_physical, k=2)
    return dists[:, 1]


def _pair_features(
    predicted_pos: np.ndarray,
    track_conf: float,
    det_pos: np.ndarray,
    det_conf: float,
    last_pos: np.ndarray,
    track_velocity: np.ndarray,
    temporal_gap: int = 1,
) -> dict[str, float]:
    """
    Shared feature computation used both by the (pure-geometric) cost
    function and by learned_association.py's training-data generation --
    computing these once here means the MLP's features and the inference
    cost function's geometric terms can never silently drift apart.
    """
    predicted_position_error = float(np.linalg.norm(predicted_pos - det_pos))
    raw_distance = float(np.linalg.norm(last_pos - det_pos))

    displacement = det_pos - last_pos
    velocity_norm = np.linalg.norm(track_velocity)
    displacement_norm = np.linalg.norm(displacement)
    if velocity_norm < 1e-6 or displacement_norm < 1e-6:
        velocity_consistency = 0.0  # neutral -- no velocity history, or zero displacement, to compare against
    else:
        velocity_consistency = float(np.dot(track_velocity, displacement) / (velocity_norm * displacement_norm))

    return {
        "predicted_position_error": predicted_position_error,
        "raw_distance": raw_distance,
        "velocity_consistency": velocity_consistency,
        "track_confidence": float(track_conf),
        "detection_confidence": float(det_conf),
        "temporal_gap": float(temporal_gap),
    }


def _pair_cost(
    predicted_pos: np.ndarray,
    track_conf: float,
    det_pos: np.ndarray,
    det_conf: float,
    last_pos: np.ndarray,
    config: AssociationConfig,
    track_velocity: np.ndarray | None = None,
    learned_prob: float = 0.0,
) -> float:
    features = _pair_features(
        predicted_pos, track_conf, det_pos, det_conf, last_pos,
        track_velocity if track_velocity is not None else np.zeros(3),
    )
    return (
        features["predicted_position_error"]
        + config.w_raw * features["raw_distance"]
        - config.w_conf * min(track_conf, det_conf)
        - config.w_learned * learned_prob
    )


def _build_cost_and_gate_matrix(
    tracks: list[Track],
    detections: pd.DataFrame,
    scale: tuple[float, float, float],
    gate_um: float,
    crowding_scale: float | None,
    config: AssociationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (cost, allowed) both shape (n_tracks, n_detections). `allowed`
    is a boolean gate mask -- True where the pair is within the (possibly
    crowding-adjusted) gate distance.
    """
    scale_arr = np.array(scale)
    n_tracks, n_dets = len(tracks), len(detections)
    cost = np.full((n_tracks, n_dets), np.inf)
    allowed = np.zeros((n_tracks, n_dets), dtype=bool)

    if n_tracks == 0 or n_dets == 0:
        return cost, allowed

    det_positions = detections[["z", "y", "x"]].to_numpy()
    det_positions_physical = det_positions * scale_arr
    det_confidences = detections["score"].to_numpy() if "score" in detections.columns else np.ones(n_dets)

    if crowding_scale is not None:
        local_nn = _local_nn_distance_um(det_positions_physical)
        gates = np.minimum(gate_um, crowding_scale * local_nn)
    else:
        gates = np.full(n_dets, gate_um)

    for i, track in enumerate(tracks):
        predicted_physical = track.predict_next_position() * scale_arr
        last_physical = track.last_position * scale_arr
        
        # --- NEW LOGIC: Calculate velocity in physical space ---
        velocity_physical = predicted_physical - last_physical
        # -------------------------------------------------------

        diffs = det_positions_physical - predicted_physical
        dists_to_predicted = np.linalg.norm(diffs, axis=1)
        within_gate = dists_to_predicted <= gates
        
        for j in np.where(within_gate)[0]:
            cost[i, j] = _pair_cost(
                predicted_physical, track.last_confidence,
                det_positions_physical[j], det_confidences[j],
                last_physical, config,
                track_velocity=velocity_physical, # <-- Pass the velocity here
            )
            allowed[i, j] = True

    return cost, allowed


def enumerate_gated_candidates(
    tracks: list[Track],
    detections: pd.DataFrame,
    scale: tuple[float, float, float],
    gate_um: float,
    crowding_scale: float | None,
    config: AssociationConfig = AssociationConfig(),
) -> list[dict]:
    """
    Same gating logic as _build_cost_and_gate_matrix, but returns every
    allowed (track, detection) candidate pair with its raw feature dict
    instead of a cost matrix -- used by learned_association.py to build
    MLP training examples. Kept as a thin wrapper around the same
    internals the production cost path uses, so the candidate set the
    MLP trains on can never silently diverge from what association.py
    actually considers at inference time.

    Returns list of {"track_index": int, "detection_index": int,
    "features": dict} (see _pair_features for the feature dict's keys).
    """
    scale_arr = np.array(scale)
    n_dets = len(detections)
    if len(tracks) == 0 or n_dets == 0:
        return []

    det_positions = detections[["z", "y", "x"]].to_numpy()
    det_positions_physical = det_positions * scale_arr
    det_confidences = detections["score"].to_numpy() if "score" in detections.columns else np.ones(n_dets)

    if crowding_scale is not None:
        local_nn = _local_nn_distance_um(det_positions_physical)
        gates = np.minimum(gate_um, crowding_scale * local_nn)
    else:
        gates = np.full(n_dets, gate_um)

    results = []
    for i, track in enumerate(tracks):
        predicted_physical = track.predict_next_position() * scale_arr
        last_physical = track.last_position * scale_arr
        velocity_physical = (predicted_physical - last_physical)  # already accounts for constant-velocity extrapolation
        diffs = det_positions_physical - predicted_physical
        dists_to_predicted = np.linalg.norm(diffs, axis=1)
        within_gate = dists_to_predicted <= gates
        for j in np.where(within_gate)[0]:
            features = _pair_features(
                predicted_physical, track.last_confidence,
                det_positions_physical[j], det_confidences[j],
                last_physical, velocity_physical,
            )
            results.append({"track_index": i, "detection_index": int(j), "features": features})

    return results


def _solve_gated_hungarian(cost: np.ndarray, allowed: np.ndarray) -> list[tuple[int, int]]:
    """Hungarian solve with disallowed pairs sentineled to a large cost,
    then any assignment that landed on a disallowed pair is discarded --
    standard workaround for scipy's assignment solver not natively
    supporting optional/forbidden pairs."""
    if cost.size == 0:
        return []
    SENTINEL = 1e6
    padded_cost = np.where(allowed, cost, SENTINEL)
    row_idx, col_idx = linear_sum_assignment(padded_cost)
    return [(int(r), int(c)) for r, c in zip(row_idx, col_idx) if allowed[r, c]]


def associate_frame(
    tracks: list[Track],
    detections: pd.DataFrame,
    scale: tuple[float, float, float],
    config: AssociationConfig = AssociationConfig(),
) -> AssociationResult:
    """
    detections : DataFrame[z, y, x] (+ optional 'score' column, defaults
        to confidence 1.0 per row if absent). Index positions (0..N-1)
        are what unmatched_detection_indices / matched_pairs refer to --
        reset_index on the caller's side if needed.
    scale      : physical (Z, Y, X) scale matching whatever coordinate
        space `tracks` and `detections` positions are in (voxel or
        physical -- this function converts to physical internally either
        way via `scale`, so pass native voxel coordinates + NATIVE_SCALE
        consistently).
    """
    n_tracks, n_dets = len(tracks), len(detections)

    # --- pass 1: tight, crowding-aware gate ---
    cost1, allowed1 = _build_cost_and_gate_matrix(
        tracks, detections, scale, config.base_gate_um, config.crowding_scale, config
    )
    matched = _solve_gated_hungarian(cost1, allowed1)
    matched_track_idx = {m[0] for m in matched}
    matched_det_idx = {m[1] for m in matched}

    leftover_track_idx = [i for i in range(n_tracks) if i not in matched_track_idx]
    leftover_det_idx = [j for j in range(n_dets) if j not in matched_det_idx]

    if not leftover_track_idx or not leftover_det_idx:
        return AssociationResult(matched, leftover_track_idx, leftover_det_idx)

    # --- pass 2: fallback, looser fixed gate, margin-gated acceptance ---
    leftover_tracks = [tracks[i] for i in leftover_track_idx]
    leftover_dets = detections.iloc[leftover_det_idx].reset_index(drop=True)

    cost2, allowed2 = _build_cost_and_gate_matrix(
        leftover_tracks, leftover_dets, scale, config.fallback_gate_um, None, config
    )
    fallback_matched_local = _solve_gated_hungarian(cost2, allowed2)

    accepted = []
    for local_i, local_j in fallback_matched_local:
        chosen_cost = cost2[local_i, local_j]
        row_costs = cost2[local_i, :]
        col_costs = cost2[:, local_j]
        # runner-up cost for this track (excluding the chosen detection) and
        # this detection (excluding the chosen track), among gated-allowed pairs
        row_alt = row_costs[np.arange(len(row_costs)) != local_j]
        col_alt = col_costs[np.arange(len(col_costs)) != local_i]
        row_runner_up = row_alt[np.isfinite(row_alt)].min() if np.isfinite(row_alt).any() else np.inf
        col_runner_up = col_alt[np.isfinite(col_alt)].min() if np.isfinite(col_alt).any() else np.inf
        runner_up = min(row_runner_up, col_runner_up)

        if not np.isfinite(runner_up):
            accepted.append((local_i, local_j))  # no competing alternative at all -- unambiguous
            continue
        # require the chosen cost to beat the runner-up by at least margin_ratio
        # (handles negative costs too, since confidence bonus can make cost negative)
        margin = runner_up - chosen_cost
        threshold = config.margin_ratio * (abs(runner_up) + 1e-6)
        if margin >= threshold:
            accepted.append((local_i, local_j))

    final_matched = list(matched)
    final_leftover_tracks = set(leftover_track_idx)
    final_leftover_dets = set(leftover_det_idx)
    for local_i, local_j in accepted:
        global_i = leftover_track_idx[local_i]
        global_j = leftover_det_idx[local_j]
        final_matched.append((global_i, global_j))
        final_leftover_tracks.discard(global_i)
        final_leftover_dets.discard(global_j)

    return AssociationResult(
        matched_pairs=final_matched,
        unmatched_track_indices=sorted(final_leftover_tracks),
        unmatched_detection_indices=sorted(final_leftover_dets),
    )
