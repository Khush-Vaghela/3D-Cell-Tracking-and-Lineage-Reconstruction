"""
Real tracksdata-native replacement for the hand-rolled Phase 2/3/4-rule-based
pipeline, built after getting visibility into tracksdata's actual internals
(edges/, solvers/, functional/_motile.py).

What this replaces and why:

- association.py (Phase 2, frame-by-frame Hungarian + crowding gate) and
  gap_closing.py (Phase 3, separate evidence-gated repair pass) are both
  subsumed by tracksdata.edges.DistanceEdges with delta_t > 1: it builds
  candidate edges across MULTIPLE frames in one pass, natively. A track
  "gap" isn't a separate repair problem here -- it's just a longer-range
  candidate edge that was always in the pool, competing on equal footing
  with delta_t=1 edges during the solve.

- division_candidates.py's RULE-BASED candidate generation (age/confidence/
  distance gates before scoring) is subsumed by the solver itself:
  NearestNeighborsSolver(max_children=2) and ILPSolver(division_weight=...)
  both natively allow a node two outgoing edges as part of the constrained
  optimization, not as a separate hand-written rule engine.

What this does NOT replace (yet -- see followups below):
- division_mlp.py / division_features.py / division_threshold_tuning.py
  (Phase 4.5) still have a real role -- just a DIFFERENT one than before.
  Previously they filtered rule-based candidates before acceptance; now
  the natural role is VALIDATING the solver's own division decisions
  after the fact (score every out-degree-2 node the solver produced, flag
  or reject low-confidence ones).
- learned_association.py's EdgeScorerMLP isn't wired into the edge weight
  yet either -- GenericFuncEdgeAttrs is confirmed as the right mechanism,
  but the MLP's velocity_consistency feature needs a node's own
  predecessor, which GenericFuncEdgeAttrs's pure source/target-attribute
  function signature can't see directly. That needs a node-level
  "predicted position" attribute computed BEFORE attaching the learned
  weight (e.g. from a first NearestNeighborsSolver pass).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
import tracksdata as td
from tracksdata.edges import DistanceEdges
from tracksdata.solvers import NearestNeighborsSolver

from tracker import TrackingResult


@dataclass
class CandidateGraphConfig:
    distance_threshold_um: float = 15.0
    n_neighbors: int = 3
    delta_t: int = 3          # allows gaps up to (delta_t - 1) missed frames -- subsumes Phase 3
    neighbors_per_frame: bool = False


def build_candidate_graph(
    detections_per_frame: dict[int, pd.DataFrame],
    native_scale: tuple[float, float, float],
    config: CandidateGraphConfig = CandidateGraphConfig(),
):
    """
    Builds the full candidate graph: every detection across all frames as
    a node (with a `confidence` attr), then DistanceEdges adds candidate
    edges -- including multi-frame-gap ones via delta_t -- in one native
    pass rather than the separate association+gap-closing passes this
    replaces.

    Coordinates are stored in PHYSICAL units (native voxel * native_scale)
    so DistanceEdges' Euclidean distance is directly in um, matching this
    project's gate/threshold conventions elsewhere -- not left in raw
    voxel units, which would silently misweight the anisotropic axes.
    """
    g = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        g.add_node_attr_key(key, pl.Float64, 0.0)
    g.add_node_attr_key("confidence", pl.Float64, 1.0)

    scale_arr = np.array(native_scale)
    for t, detections in detections_per_frame.items():
        for row in detections.itertuples(index=False):
            pos_physical = np.array([row.z, row.y, row.x]) * scale_arr
            confidence = float(row.score) if hasattr(row, "score") else 1.0
            g.add_node({
                "t": int(t),
                "z": float(pos_physical[0]), "y": float(pos_physical[1]), "x": float(pos_physical[2]),
                "confidence": confidence,
            })

    edge_op = DistanceEdges(
        distance_threshold=config.distance_threshold_um,
        n_neighbors=config.n_neighbors,
        delta_t=config.delta_t,
        neighbors_per_frame=config.neighbors_per_frame,
    )
    edge_op.add_edges(g)

    return g


def solve_nearest_neighbors(graph, max_children: int = 2):
    """Fast, greedy solve -- no extra dependencies. Divisions (out-degree 2)
    fall out of max_children=2 natively, no separate rule engine."""
    solver = NearestNeighborsSolver(max_children=max_children)
    return solver.solve(graph)


def solve_ilp(
    graph,
    division_weight: float = 1.0,
    appearance_weight: float = 1.0,
    disappearance_weight: float = 1.0,
    node_weight: float = -10.0,
    timeout: float | None = None,
    gap: float = 0.0,  # <-- ADDED GAP TO THE SIGNATURE
):
    """
        Globally optimal solve via the real ILP formulation (confirmed
        working with the free SCIP backend, no Gurobi license needed).
        division_weight/appearance_weight/disappearance_weight are native
        per-node costs in the flow-conservation constraints -- this is the
        actual mechanism the competition's own domain (network-flow tracking)
        is built on, not a hand-rolled approximation of it.
    
        node_weight defaults to a NEGATIVE value deliberately: ILPSolver's own
        default is 0.0, meaning there's no baseline reward for including a
        node at all -- with any non-negative edge/appearance/disappearance
        costs, "include nothing" is then trivially the global optimum
        (confirmed directly: without this, the solver returns an empty
        solution and logs "Trivial solution found with all variables set to
        0!"). A negative node_weight is the standard flow-ILP convention for
        rewarding keeping real detections in the solution.
        """
    from tracksdata.solvers import ILPSolver

    solver = ILPSolver(
        division_weight=division_weight,
        appearance_weight=appearance_weight,
        disappearance_weight=disappearance_weight,
        node_weight=node_weight,
        timeout=timeout,
        gap=gap,       # <-- PASSED GAP TO THE UNDERLYING SOLVER
    )
    return solver.solve(graph)


def graph_view_to_tracking_result(solution) -> TrackingResult:
    """Converts a solved GraphView back into this project's TrackingResult
    format (node_id/t/z/y/x/score, source_id/target_id) for compatibility
    with everything downstream (submission writer, Phase 4.5 MLP, etc)."""
    nodes_pl = solution.node_attrs(attr_keys=[
        td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.T,
        td.DEFAULT_ATTR_KEYS.Z, td.DEFAULT_ATTR_KEYS.Y, td.DEFAULT_ATTR_KEYS.X, "confidence",
    ])
    nodes_df = nodes_pl.to_pandas().rename(columns={
        td.DEFAULT_ATTR_KEYS.NODE_ID: "node_id", td.DEFAULT_ATTR_KEYS.T: "t",
        td.DEFAULT_ATTR_KEYS.Z: "z", td.DEFAULT_ATTR_KEYS.Y: "y", td.DEFAULT_ATTR_KEYS.X: "x",
        "confidence": "score",
    })

    if solution.num_edges() == 0:
        edges_df = pd.DataFrame(columns=["source_id", "target_id", "edge_dist"])
    else:
        edges_pl = solution.edge_attrs(attr_keys=[
            td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET, td.DEFAULT_ATTR_KEYS.EDGE_DIST,
        ])
        edges_df = edges_pl.to_pandas().rename(columns={
            td.DEFAULT_ATTR_KEYS.EDGE_SOURCE: "source_id", td.DEFAULT_ATTR_KEYS.EDGE_TARGET: "target_id",
            td.DEFAULT_ATTR_KEYS.EDGE_DIST: "edge_dist",
        })

    return TrackingResult(nodes=nodes_df, edges=edges_df)
