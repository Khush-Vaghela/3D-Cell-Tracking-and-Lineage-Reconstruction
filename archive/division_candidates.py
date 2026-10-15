"""
Phase 4 -- division resolution, rule-based candidate generation.

Runs on the output of Phase 2 (association) + Phase 3 (gap closing): for
every node that already has exactly one outgoing edge (a normal
continuation to "daughter1", decided by Phase 2/3), look for a second
daughter among detections still orphaned (no incoming edge) at the same
target frame. A second edge is only added when all three gates pass:

  1. Parent track age gate: the parent's track must have existed for at
     least min_parent_age frames -- a track that just started and
     immediately "divides" is weak evidence.
  2. Daughter confidence gate: the candidate second daughter's detector
     confidence must clear min_daughter_confidence.
  3. Association-cost gate: the (parent, candidate) pair is scored with
     the SAME cost function used for normal continuation links
     (association._pair_features, reused directly -- not reimplemented),
     gated by a distance threshold. This is deliberately the same
     machinery as Phase 2, not a separate division-specific geometric
     model.

No new nodes are created here -- divisions only ever claim EXISTING
orphaned detections. This candidate set (including the false positives
that pass these gates but aren't real GT divisions) is exactly what
Phase 4.5's learned gate is trained and run on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tracker import TrackingResult
import association as assoc


@dataclass
class DivisionConfig:
    min_parent_age: int = 2
    min_daughter_confidence: float = 0.3
    division_gate_um: float = 15.0
    assoc_config: assoc.AssociationConfig = field(default_factory=assoc.AssociationConfig)


@dataclass
class DivisionCandidate:
    parent_id: int
    daughter1_id: int    # the existing continuation, already in the graph
    daughter2_id: int    # the candidate second daughter
    cost: float
    features: dict        # from association._pair_features -- reused by Phase 4.5


def _build_graph_maps(nodes: pd.DataFrame, edges: pd.DataFrame):
    incoming = {}
    outgoing = {}
    for _, row in edges.iterrows():
        outgoing.setdefault(row["source_id"], []).append(row["target_id"])
        incoming[row["target_id"]] = row["source_id"]
    node_info = nodes.set_index("node_id")[["t", "z", "y", "x", "score"]].to_dict("index")
    return incoming, outgoing, node_info


def _track_age(node_id: int, incoming: dict) -> int:
    age = 1
    current = node_id
    while current in incoming:
        current = incoming[current]
        age += 1
    return age


def _estimate_velocity(node_id: int, incoming: dict, node_info: dict) -> np.ndarray:
    pos = np.array([node_info[node_id]["z"], node_info[node_id]["y"], node_info[node_id]["x"]])
    if node_id not in incoming:
        return np.zeros(3)
    parent_id = incoming[node_id]
    parent_pos = np.array([node_info[parent_id]["z"], node_info[parent_id]["y"], node_info[parent_id]["x"]])
    return pos - parent_pos


def generate_division_candidates(
    tracking_result: TrackingResult,
    native_scale: tuple[float, float, float],
    config: DivisionConfig = DivisionConfig(),
) -> list[DivisionCandidate]:
    nodes = tracking_result.nodes
    edges = tracking_result.edges
    incoming, outgoing, node_info = _build_graph_maps(nodes, edges)
    scale_arr = np.array(native_scale)

    orphan_ids_by_t: dict[int, list[int]] = {}
    for nid in nodes["node_id"]:
        if nid not in incoming:
            orphan_ids_by_t.setdefault(node_info[nid]["t"], []).append(nid)

    candidates = []
    for parent_id, children in outgoing.items():
        if len(children) != 1:
            continue  # already a division, or (shouldn't happen) more than 2 -- not a rule-based candidate target
        daughter1_id = children[0]

        age = _track_age(parent_id, incoming)
        if age < config.min_parent_age:
            continue

        target_t = node_info[daughter1_id]["t"]
        pool = [nid for nid in orphan_ids_by_t.get(target_t, []) if nid != daughter1_id]
        if not pool:
            continue

        parent_pos = np.array([node_info[parent_id]["z"], node_info[parent_id]["y"], node_info[parent_id]["x"]])
        parent_conf = node_info[parent_id]["score"]
        velocity = _estimate_velocity(parent_id, incoming, node_info)
        predicted_pos = parent_pos + velocity  # constant-velocity prediction, same convention as association.Track

        for cand_id in pool:
            cand_conf = node_info[cand_id]["score"]
            if cand_conf < config.min_daughter_confidence:
                continue
            cand_pos = np.array([node_info[cand_id]["z"], node_info[cand_id]["y"], node_info[cand_id]["x"]])

            predicted_physical = predicted_pos * scale_arr
            parent_physical = parent_pos * scale_arr
            cand_physical = cand_pos * scale_arr
            velocity_physical = predicted_physical - parent_physical

            predicted_position_error = float(np.linalg.norm(predicted_physical - cand_physical))
            if predicted_position_error > config.division_gate_um:
                continue

            features = assoc._pair_features(
                predicted_physical, parent_conf, cand_physical, cand_conf, parent_physical, velocity_physical,
            )
            cost = (
                features["predicted_position_error"]
                + config.assoc_config.w_raw * features["raw_distance"]
                - config.assoc_config.w_conf * min(parent_conf, cand_conf)
            )
            candidates.append(DivisionCandidate(
                parent_id=parent_id, daughter1_id=daughter1_id, daughter2_id=cand_id,
                cost=cost, features=features,
            ))

    return candidates


def apply_division_candidates(
    tracking_result: TrackingResult,
    candidates: list[DivisionCandidate],
) -> TrackingResult:
    """Add the given candidates' edges to the graph -- one call site so
    Phase 4 (accept all rule-gated candidates) and Phase 4.5 (accept only
    MLP-scored survivors) both produce output the same way."""
    if not candidates:
        return tracking_result
    new_edges = pd.DataFrame(
        [(c.parent_id, c.daughter2_id) for c in candidates], columns=["source_id", "target_id"]
    )
    edges = pd.concat([tracking_result.edges, new_edges], ignore_index=True)
    return TrackingResult(nodes=tracking_result.nodes, edges=edges)
