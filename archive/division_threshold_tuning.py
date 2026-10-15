"""
Phase 4.5, step 5 -- tune the division-acceptance threshold against the
REAL official division Jaccard.

Rewired for the tracksdata-native pipeline: validates/prunes the solver's
existing out-degree-2 nodes instead of applying rule-based candidates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import tracksdata as td

from tracker import TrackingResult
import division_validation as dv
import division_mlp as dmlp


def build_full_graph(tracking_result: TrackingResult) -> td.graph.IndexedRXGraph:
    """Includes edges for tracking_cellmot.metrics.evaluate()."""
    import polars as pl
    import tracksdata as td
    
    # 1. Switch to IndexedRXGraph to support custom IDs natively
    g = td.graph.IndexedRXGraph()
    
    for key in ("z", "y", "x"):
        g.add_node_attr_key(key, pl.Float64, 0.0)

    for row in tracking_result.nodes.itertuples(index=False):
        # 2. Pass the ID to the 'index' argument, NOT inside the dictionary!
        g.add_node(
            {
                "t": int(row.t), 
                "z": float(row.z), 
                "y": float(row.y), 
                "x": float(row.x)
            }, 
            index=int(row.node_id)
        )
        
    for row in tracking_result.edges.itertuples(index=False):
        # 3. Because the IDs match exactly now, we just add the edges directly
        g.add_edge(int(row.source_id), int(row.target_id), {})
        
    return g


def score_division_jaccard(
    tracking_result: TrackingResult,
    gt_graph, 
    scale: tuple[float, float, float],
    max_distance: float = 7.0,
) -> dict:
    """One sample's per_sample_metrics row, using the REAL official metric."""
    from tracking_cellmot.metrics import evaluate as compute_metric, per_sample_metrics

    pred_graph = build_full_graph(tracking_result)
    er = compute_metric(pred_graph, gt_graph, scale=scale, max_distance=max_distance)
    return per_sample_metrics(er, n_total=float("nan"), node_recall=0.0)


@dataclass
class ValidationSample:
    """
    One held-out sample's data for fast threshold tuning.
    Pre-computes the MLP probabilities so the sweep loop is fast.
    """
    tracking_result: TrackingResult
    gt_graph: object  # td.graph.IndexedRXGraph
    division_probas: list[tuple[dv.Division, float]]  # (Division object, predicted probability)


def prepare_validation_sample(
    tracking_result: TrackingResult,
    gt_graph: object,
    model: dmlp.DivisionScorerMLP,
    vol,  
    scale: tuple[float, float, float],
) -> ValidationSample:
    divisions = dv.find_divisions(tracking_result)
    division_probas = []
    
    # --- THE FIX: Pre-compute ONCE ---
    node_info_dict = tracking_result.nodes.set_index("node_id")[["t", "z", "y", "x", "score"]].to_dict("index")
    edge_costs_dict = tracking_result.edges.set_index(["source_id", "target_id"])["edge_dist"].to_dict()
    
    for div in divisions:
        feats = dv.build_features_for_division(
            div, tracking_result, vol, scale, 
            node_info=node_info_dict, edge_costs=edge_costs_dict
        )
        proba = model.predict_proba(feats)
        division_probas.append((div, proba))
        
    return ValidationSample(
        tracking_result=tracking_result, 
        gt_graph=gt_graph, 
        division_probas=division_probas
    )


@dataclass
class ThresholdSweepResult:
    threshold: float
    division_jaccard: float
    score: float


def tune_division_threshold(
    validation_samples: list[ValidationSample],
    scale: tuple[float, float, float],
    thresholds: np.ndarray = np.arange(0.1, 1.0, 0.05),
) -> tuple[float, list[ThresholdSweepResult]]:
    """
    For each candidate threshold: drops the weaker daughter edge for any 
    division scoring below the threshold, scores against real GT, and 
    micro-averages across samples to find the maximum division Jaccard.
    """
    from tracking_cellmot.metrics import summarise

    sweep = []
    for threshold in thresholds:
        rows = []
        for vs in validation_samples:
            # 1. Identify which edges need to be dropped at this specific threshold
            edges_to_drop = set()
            for div, proba in vs.division_probas:
                if proba < threshold:
                    edges_to_drop.add((div.parent_id, div.daughter2_id))
            
            # 2. Apply the rejections to the tracking result
            if not edges_to_drop:
                pruned_result = vs.tracking_result
            else:
                keep_mask = ~vs.tracking_result.edges.apply(
                    lambda r: (r["source_id"], r["target_id"]) in edges_to_drop, axis=1
                )
                new_edges = vs.tracking_result.edges[keep_mask].reset_index(drop=True)
                pruned_result = TrackingResult(nodes=vs.tracking_result.nodes, edges=new_edges)
                
            # 3. Score the pruned graph
            rows.append(score_division_jaccard(pruned_result, vs.gt_graph, scale))
            
        # 4. Aggregate results across all validation samples
        summary = summarise(rows)
        sweep.append(ThresholdSweepResult(
            threshold=float(threshold),
            division_jaccard=summary["division_jaccard"],
            score=summary["score"],
        ))

    best = max(sweep, key=lambda r: r.division_jaccard)
    return best.threshold, sweep