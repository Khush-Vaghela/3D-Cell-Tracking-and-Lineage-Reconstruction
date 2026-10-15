"""
Real validation signal for the Phase 1 DETECTOR ONLY, using tracksdata's
actual centroid-distance matching -- not the full competition metric.

Why not the full edge/division Jaccard: that metric is only meaningful once
edges exist, and Phase 1 has no linker yet (that's Phase 2, not built). The
official evaluate() wrapper in the competition repo's metrics.py explicitly
early-returns score=0.0 with a warning when the predicted graph has zero
edges (confirmed by reading tracking_cellmot/metrics.py directly) -- so
calling it as-is on detector-only output wouldn't produce a real number.

What this module does instead: call tracksdata's graph.match() directly
(the same DistanceMatching centroid-matching primitive the official metric
uses internally) on a NODE-ONLY predicted graph, bypassing the edge-gated
evaluate() wrapper -- confirmed this works cleanly (tested interactively;
match() only warns "No matching edges found", which is expected and
harmless when there are no edges to match). This gives a genuine,
non-proxy number: NODE RECALL -- the fraction of GT-labeled cells the
detector actually found -- which is exactly the thing Phase 1 should be
judged on, without overclaiming it's the leaderboard score.

This intentionally does NOT reimplement edge_jaccard/division_jaccard --
those require Phase 2 (linking) to mean anything, and faking them with a
throwaway linker just to produce a number would be more misleading, not
less.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
import tracksdata as td
from tracksdata.metrics import DistanceMatching
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass
class NodeValidationResult:
    n_gt: int
    n_pred: int
    n_matched: int
    recall: float       # fraction of GT nodes matched by a predicted node
    precision: float     # fraction of predicted nodes that matched a GT node


def _build_node_graph(nodes: pd.DataFrame) -> td.graph.InMemoryGraph:
    """nodes: DataFrame with columns t, z, y, x (any extra columns ignored)."""
    g = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        g.add_node_attr_key(key, pl.Float64, 0.0)
    for row in nodes.itertuples(index=False):
        g.add_node({"t": int(row.t), "z": float(row.z), "y": float(row.y), "x": float(row.x)})
    return g


def compute_node_recall(
    pred_nodes: pd.DataFrame,
    gt_nodes: pd.DataFrame,
    scale: tuple[float, float, float],
    max_distance: float = 7.0,
) -> NodeValidationResult:
    """
    pred_nodes, gt_nodes : DataFrame[t, z, y, x] in the SAME coordinate
        space (native voxel coordinates -- convert predictions back via
        preprocessing.upsample_node_coords before calling this, same as
        the submission format requires).
    scale       : physical (Z, Y, X) scale for centroid-distance matching
        (data_io.NATIVE_SCALE for native-space coordinates).
    max_distance: matching radius in the SAME physical units as `scale`
        produces (um) -- 7.0 matches the competition's own edge-matching
        tolerance, reused here for node matching as a reasonable default.
    """
    n_gt = len(gt_nodes)
    n_pred = len(pred_nodes)

    if n_gt == 0:
        return NodeValidationResult(n_gt=0, n_pred=n_pred, n_matched=0, recall=float("nan"), precision=float("nan"))
    if n_pred == 0:
        return NodeValidationResult(n_gt=n_gt, n_pred=0, n_matched=0, recall=0.0, precision=float("nan"))

    pred_graph = _build_node_graph(pred_nodes)
    gt_graph = _build_node_graph(gt_nodes)

    matching = DistanceMatching(max_distance=max_distance, scale=scale)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No matching edges found")
        pred_graph.match(gt_graph, matching=matching)

    attrs = pred_graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    matched_gt_ids = attrs.filter(pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID) != -1)[
        td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
    ]
    n_matched_pred = len(matched_gt_ids)
    n_matched_gt = matched_gt_ids.n_unique()  # a GT node can only be claimed once in a correct matching, but count uniquely to be safe

    return NodeValidationResult(
        n_gt=n_gt,
        n_pred=n_pred,
        n_matched=n_matched_gt,
        recall=n_matched_gt / n_gt,
        precision=n_matched_pred / n_pred,
    )

def match_nodes(
    pred_df: pd.DataFrame, 
    gt_df: pd.DataFrame, 
    scale: tuple[float, float, float], 
    max_distance_um: float
) -> np.ndarray:
    """
    Matches predicted nodes to GT nodes based on physical distance within max_distance_um.
    Returns an array of length len(pred_df) with indices into gt_df (0..M-1) or -1 if unmatched.
    """
    n_pred = len(pred_df)
    n_gt = len(gt_df)
    matches = np.full(n_pred, -1, dtype=np.int64)
    
    if n_pred == 0 or n_gt == 0:
        return matches

    scale_arr = np.array(scale)
    pred_coords = pred_df[["z", "y", "x"]].to_numpy() * scale_arr
    gt_coords = gt_df[["z", "y", "x"]].to_numpy() * scale_arr

    # Pairwise physical distance matrix (N_pred, M_gt)
    dists = cdist(pred_coords, gt_coords)

    # 1-to-1 Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(dists)

    for r, c in zip(row_ind, col_ind):
        if dists[r, c] <= max_distance_um:
            matches[r] = c

    return matches
