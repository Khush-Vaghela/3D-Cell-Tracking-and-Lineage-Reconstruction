"""
Phase 4.5, re-wired for the tracksdata-native pipeline (tracksdata_pipeline.py).

Old role (division_mlp.py + division_candidates.py): filter RULE-BASED
candidates before they were ever added as edges.

New role: the solver (NearestNeighborsSolver/ILPSolver) now decides
divisions on its own, natively, via max_children=2 / division_weight --
there's no separate candidate-generation step left to filter. So this
module VALIDATES the solver's own decisions after the fact: find every
node with out-degree 2 in the solved graph, score it with the (unchanged)
DivisionScorerMLP, and reject (remove) the weaker of the two edges for
any division scoring below the tuned threshold.

Reuses division_mlp.DivisionScorerMLP, division_mlp.train_division_scorer,
and division_features.py entirely unchanged -- the model architecture and
what a "division" looks like as a 9-feature vector didn't change, only
where the candidates come from and what happens to rejected ones.

One feature's source changed: the old association_cost feature came from
division_candidates.py's rule-based DivisionCandidate.cost (association.py's
_pair_cost). That object no longer exists in this pipeline. Here,
association_cost is the solver's own recorded edge_dist for the "weaker"
(farther) daughter's edge instead -- genuinely different information from
distance_to_daughter2 only once a learned edge weight (e.g. learned_prob_ij
via GenericFuncEdgeAttrs, not wired in yet) replaces raw distance as
DistanceEdges' output; today the two are numerically close, since both are
ultimately physical distance. Documented here rather than silently reusing
a stale docstring claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tracker import TrackingResult
import division_features as dfeat
import node_validation as nv
from division_mlp import DivisionScorerMLP  # unchanged, reused as-is


@dataclass
class Division:
    parent_id: int
    daughter1_id: int   # closer (smaller edge_dist) -- kept unconditionally
    daughter2_id: int   # farther -- the one being validated


def find_divisions(tracking_result: TrackingResult) -> list[Division]:
    """Every node with out-degree 2 in the solved graph. daughter1/daughter2
    ordering is by edge_dist ascending (closer = daughter1), a convention
    for which edge gets dropped on rejection, not a claim about which one
    is "more real" -- both are equally the solver's own decision."""
    divisions = []
    if "edge_dist" not in tracking_result.edges.columns or len(tracking_result.edges) == 0:
        return divisions
    for parent_id, group in tracking_result.edges.groupby("source_id"):
        if len(group) != 2:
            continue
        sorted_group = group.sort_values("edge_dist")
        d1, d2 = sorted_group.iloc[0]["target_id"], sorted_group.iloc[1]["target_id"]
        divisions.append(Division(parent_id=int(parent_id), daughter1_id=int(d1), daughter2_id=int(d2)))
    return divisions


def build_features_for_division(
    division: Division,
    tracking_result: TrackingResult,
    vol, 
    native_scale: tuple[float, float, float],
    intensity_radius_um: float = 5.0,
    node_info: dict = None,   # <-- ADDED
    edge_costs: dict = None,  # <-- ADDED
) -> dict:
    # Fallback just in case they aren't passed
    if node_info is None:
        node_info = tracking_result.nodes.set_index("node_id")[["t", "z", "y", "x", "score"]].to_dict("index")
    if edge_costs is None:
        edge_costs = tracking_result.edges.set_index(["source_id", "target_id"])["edge_dist"].to_dict()

    parent_info = node_info[division.parent_id]
    d1_info = node_info[division.daughter1_id]
    d2_info = node_info[division.daughter2_id]

    parent_pos = np.array([parent_info["z"], parent_info["y"], parent_info["x"]])
    d1_pos = np.array([d1_info["z"], d1_info["y"], d1_info["x"]])
    d2_pos = np.array([d2_info["z"], d2_info["y"], d2_info["x"]])

    parent_intensity = dfeat.compute_local_intensity(vol, int(parent_info["t"]), parent_pos, intensity_radius_um, native_scale)
    d1_intensity = dfeat.compute_local_intensity(vol, int(d1_info["t"]), d1_pos, intensity_radius_um, native_scale)
    d2_intensity = dfeat.compute_local_intensity(vol, int(d2_info["t"]), d2_pos, intensity_radius_um, native_scale)

    # Instant dictionary lookup instead of a slow Pandas scan!
    association_cost = float(edge_costs[(division.parent_id, division.daughter2_id)])

    return dfeat.build_division_features(
        parent_pos, d1_pos, d2_pos,
        parent_info["score"], d1_info["score"], d2_info["score"],
        parent_intensity, d1_intensity, d2_intensity,
        association_cost, native_scale,
    )

def validate_divisions(
    tracking_result: TrackingResult,
    model: DivisionScorerMLP, # (ensure this matches your import name)
    vol,  
    native_scale: tuple[float, float, float],
    threshold: float,
    intensity_radius_um: float = 5.0,
) -> TrackingResult:
    divisions = find_divisions(tracking_result)
    if not divisions:
        return tracking_result

    # --- THE FIX: Pre-compute ONCE for the entire inference step ---
    print("Pre-computing dictionaries for ultra-fast validation filtering...")
    node_info_dict = tracking_result.nodes.set_index("node_id")[["t", "z", "y", "x", "score"]].to_dict("index")
    edge_costs_dict = tracking_result.edges.set_index(["source_id", "target_id"])["edge_dist"].to_dict()
    # --------------------------------------------------------------

    edges_to_drop = set()
    for division in divisions:
        # Pass the pre-computed dicts to bypass the O(N^2) trap!
        features = build_features_for_division(
            division, tracking_result, vol, native_scale, intensity_radius_um,
            node_info=node_info_dict, edge_costs=edge_costs_dict
        )
        proba = model.predict_proba(features)
        if proba < threshold:
            edges_to_drop.add((division.parent_id, division.daughter2_id))

    if not edges_to_drop:
        return tracking_result

    keep_mask = ~tracking_result.edges.apply(
        lambda r: (r["source_id"], r["target_id"]) in edges_to_drop, axis=1
    )
    new_edges = tracking_result.edges[keep_mask].reset_index(drop=True)
    return TrackingResult(nodes=tracking_result.nodes, edges=new_edges)

def generate_validation_training_data(
    tracking_result: TrackingResult,
    geff,
    vol,  # <-- Changed
    native_scale: tuple[float, float, float],
    match_max_distance_um: float = 15.0, # <-- INCREASED: Gives daughter cells room to move
    intensity_radius_um: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    divisions = find_divisions(tracking_result)
    if not divisions:
        return np.empty((0, len(dfeat.FEATURE_ORDER)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    gt_edge_set = set(zip(geff.edges["source_id"], geff.edges["target_id"]))
    node_to_gt: dict[int, int] = {}
    
    for t, group in tracking_result.nodes.groupby("t"):
        gt_nodes_t = geff.nodes[geff.nodes["t"] == t][["node_id", "z", "y", "x"]].reset_index(drop=True)
        if len(gt_nodes_t) == 0:
            continue
        pred = group.assign(t=t)[["t", "z", "y", "x"]].reset_index(drop=True)
        row_matches = nv.match_nodes(pred, gt_nodes_t.assign(t=t)[["t", "z", "y", "x"]], native_scale, match_max_distance_um)
        for pred_node_id, match_idx in zip(group["node_id"], row_matches):
            if match_idx != -1:
                node_to_gt[pred_node_id] = int(gt_nodes_t.iloc[match_idx]["node_id"])

    # --- THE FIX: Calculate these massive dictionaries ONCE! ---
    print("Pre-computing dictionaries for ultra-fast feature extraction...")
    node_info_dict = tracking_result.nodes.set_index("node_id")[["t", "z", "y", "x", "score"]].to_dict("index")
    edge_costs_dict = tracking_result.edges.set_index(["source_id", "target_id"])["edge_dist"].to_dict()
    # -----------------------------------------------------------

    all_X, all_y = [], []
    for division in divisions:
        parent_gt = node_to_gt.get(division.parent_id)
        d1_gt = node_to_gt.get(division.daughter1_id) # <-- ADDED D1
        d2_gt = node_to_gt.get(division.daughter2_id)
        
        # --- STRICT DIVISION HARVESTING ---
        label = 0
        if parent_gt is not None and d1_gt is not None and d2_gt is not None:
            if d1_gt != d2_gt:  # Ensures the daughters are unique cells
                if (parent_gt, d1_gt) in gt_edge_set and (parent_gt, d2_gt) in gt_edge_set:
                    label = 1
        # ----------------------------------

        # Pass the pre-calculated dicts down to the feature builder
        features = build_features_for_division(
            division, tracking_result, vol, native_scale, intensity_radius_um,
            node_info=node_info_dict, edge_costs=edge_costs_dict
        )
        all_X.append(dfeat.features_dict_to_vector(features))
        all_y.append(label)

    if not all_X:
        return np.empty((0, len(dfeat.FEATURE_ORDER)), dtype=np.float32), np.empty((0,), dtype=np.float32)
        
    print(f"Total True Positives found: {sum(all_y)}")
    return np.stack(all_X), np.array(all_y, dtype=np.float32)
