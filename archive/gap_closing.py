"""
Phase 3 -- gap closing.

Operates on tracker.TrackingResult (nodes/edges already produced by Phase
2's frame-to-frame association) plus the raw per-frame heatmaps computed
during Phase 1 detection. For every track that dead-ends before the
video's final frame:

  1. Require the track to have existed for at least MIN_TRACK_AGE frames
     before repair is attempted at all -- a track that just barely started
     and immediately died is weak evidence to begin with; bridging it
     across a gap compounds that weakness rather than fixing anything.
  2. Search forward, one gap length at a time (closest first), up to
     MAX_GAP_FRAMES:
       a. REATTACH (free, no node budget spent): is there an existing
          orphaned detection (a node with no incoming edge -- i.e. Phase 2
          couldn't link it to anything) within evidence_radius_um of the
          constant-velocity predicted position? If so, link directly to
          it -- the detection was real, Phase 2 just couldn't bridge the
          gap to reach it.
       b. INSERT (budget-limited): if no such node exists, check the raw
          detector heatmap at that frame, at the predicted location
          (refined to the local peak within a small neighborhood). If it
          exceeds DETECT_THRESHOLD_REPAIR, insert exactly one new node
          there and link to it -- but only while the per-video node
          budget remains.
  3. First successful gap length wins -- stop searching once a track is
     bridged, don't skip a closer real reappearance for a farther one.
  4. No evidence found within MAX_GAP_FRAMES -> the track simply ends. No
     node is fabricated without either kind of evidence.

Node budget: min(node_budget_fraction * original_node_count,
node_budget_absolute_cap) -- caps runaway gap-node insertion even on a
very large or very noisy video, not just a percentage of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tracker import TrackingResult
import preprocessing as prep


@dataclass
class GapClosingConfig:
    min_track_age: int = 2               # frames a track must have existed before repair is attempted
    max_gap_frames: int = 3              # how many frames ahead to search for a reappearance
    evidence_radius_um: float = 12.0     # search radius for an existing orphaned detection
    detect_threshold_repair: float = 0.2 # heatmap value required to insert a new gap node (below the main detection threshold -- this is specifically for weak-but-real evidence)
    heatmap_search_radius_voxels: int = 2  # local-peak refinement window (downsampled-grid voxels) around the predicted location
    node_budget_fraction: float = 0.006
    node_budget_absolute_cap: int = 120
    # Note: round(node_budget_fraction * original_node_count) is 0 for
    # small graphs (e.g. <84 nodes at the default fraction) -- the insert
    # path is then never available, only reattach. That's expected on
    # real videos (thousands of nodes), not a bug; tests exercising the
    # insert path need to pass a larger node_budget_fraction explicitly.


def _build_graph_maps(nodes: pd.DataFrame, edges: pd.DataFrame):
    incoming = {}   # node_id -> parent node_id
    outgoing = {}   # node_id -> list of child node_ids
    for _, row in edges.iterrows():
        outgoing.setdefault(row["source_id"], []).append(row["target_id"])
        incoming[row["target_id"]] = row["source_id"]
    node_info = nodes.set_index("node_id")[["t", "z", "y", "x"]].to_dict("index")
    return incoming, outgoing, node_info


def _track_age(node_id: int, incoming: dict) -> int:
    """Number of nodes in the backward chain ending at node_id (inclusive)."""
    age = 1
    current = node_id
    while current in incoming:
        current = incoming[current]
        age += 1
    return age


def _estimate_velocity(node_id: int, incoming: dict, node_info: dict) -> np.ndarray:
    """Constant-velocity estimate from the last two positions in the
    track ending at node_id. Zero velocity if the track has only one
    observation (matches association.Track.predict_next_position's
    fallback for consistency)."""
    pos = np.array([node_info[node_id]["z"], node_info[node_id]["y"], node_info[node_id]["x"]])
    if node_id not in incoming:
        return np.zeros(3)
    parent_id = incoming[node_id]
    parent_pos = np.array([node_info[parent_id]["z"], node_info[parent_id]["y"], node_info[parent_id]["x"]])
    return pos - parent_pos


def _refine_to_local_peak(
    heatmap: np.ndarray,
    predicted_down: np.ndarray,
    search_radius: int,
) -> tuple[np.ndarray, float] | None:
    """Look for the strongest voxel within `search_radius` of the
    predicted downsampled-grid location. Returns (location, value) or
    None if the predicted location is entirely outside the heatmap."""
    Z, Y, X = heatmap.shape
    z0, y0, x0 = np.round(predicted_down).astype(int)
    if not (0 <= z0 < Z and 0 <= y0 < Y and 0 <= x0 < X):
        return None

    z_lo, z_hi = max(0, z0 - search_radius), min(Z, z0 + search_radius + 1)
    y_lo, y_hi = max(0, y0 - search_radius), min(Y, y0 + search_radius + 1)
    x_lo, x_hi = max(0, x0 - search_radius), min(X, x0 + search_radius + 1)

    region = heatmap[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
    local_max_idx = np.unravel_index(np.argmax(region), region.shape)
    value = region[local_max_idx]
    location = np.array([z_lo, y_lo, x_lo]) + np.array(local_max_idx)
    return location, float(value)


def close_gaps(
    tracking_result: TrackingResult,
    heatmaps_per_frame: dict[int, np.ndarray],
    downsample: tuple[int, int, int],
    native_scale: tuple[float, float, float],
    n_video_frames: int,
    config: GapClosingConfig = GapClosingConfig(),
) -> TrackingResult:
    """
    heatmaps_per_frame : {t: (Zd, Yd, Xd) sigmoid heatmap}, downsampled
        grid -- one per frame, from detect.compute_heatmap_single_frame,
        computed during the same Phase 1 pass that produced the
        detections used to build `tracking_result`.
    n_video_frames : total timepoints in the video -- needed to know
        which track-ends are genuinely at the video's last frame (not a
        gap, nothing to repair) vs. dead-ending early (a real gap
        candidate).
    """
    nodes = tracking_result.nodes.copy()
    edges = tracking_result.edges.copy()
    incoming, outgoing, node_info = _build_graph_maps(nodes, edges)

    original_node_count = len(nodes)
    node_budget = min(
        round(config.node_budget_fraction * original_node_count),
        config.node_budget_absolute_cap,
    )
    budget_used = 0

    track_end_ids = [
        nid for nid in nodes["node_id"]
        if nid not in outgoing and node_info[nid]["t"] < n_video_frames - 1
    ]
    # orphaned detections: nodes with no incoming edge, not at t=0 (t=0
    # orphans are just normal track starts, not gap-closing targets)
    orphan_ids_by_t: dict[int, list[int]] = {}
    for nid in nodes["node_id"]:
        if nid not in incoming and node_info[nid]["t"] > 0:
            orphan_ids_by_t.setdefault(node_info[nid]["t"], []).append(nid)

    new_nodes = []
    new_edges = []
    next_node_id = int(nodes["node_id"].max()) + 1 if len(nodes) else 1
    scale_arr = np.array(native_scale)
    scale_down = np.array(prep.downsampled_scale(native_scale, downsample))

    for end_id in track_end_ids:
        age = _track_age(end_id, incoming)
        if age < config.min_track_age:
            continue

        end_t = node_info[end_id]["t"]
        end_pos = np.array([node_info[end_id]["z"], node_info[end_id]["y"], node_info[end_id]["x"]])
        velocity = _estimate_velocity(end_id, incoming, node_info)

        bridged = False
        for g in range(1, config.max_gap_frames + 1):
            target_t = end_t + g
            if target_t >= n_video_frames:
                break
            predicted_native = end_pos + velocity * g

            # (a) reattach to an existing orphaned detection, if one is close enough
            candidates = orphan_ids_by_t.get(target_t, [])
            if candidates:
                cand_positions = np.array([
                    [node_info[c]["z"], node_info[c]["y"], node_info[c]["x"]] for c in candidates
                ])
                dists_physical = np.linalg.norm((cand_positions - predicted_native) * scale_arr, axis=1)
                best_idx = np.argmin(dists_physical)
                if dists_physical[best_idx] <= config.evidence_radius_um:
                    chosen_id = candidates[best_idx]
                    new_edges.append((end_id, chosen_id))
                    del orphan_ids_by_t[target_t][best_idx]  # each orphan can only be reattached once
                    bridged = True
                    break

            # (b) fresh-evidence path: check the detector's own heatmap
            if budget_used >= node_budget:
                continue  # no budget left for new nodes, but a later (closer-in-time) reattach could still work at a smaller g -- keep searching
            if target_t not in heatmaps_per_frame:
                continue
            predicted_down = predicted_native / np.array(downsample)
            refined = _refine_to_local_peak(
                heatmaps_per_frame[target_t], predicted_down, config.heatmap_search_radius_voxels
            )
            if refined is None:
                continue
            location_down, value = refined
            if value < config.detect_threshold_repair:
                continue

            location_native = location_down * np.array(downsample)
            gap_node_id = next_node_id
            next_node_id += 1
            new_nodes.append({
                "node_id": gap_node_id, "t": target_t,
                "z": location_native[0], "y": location_native[1], "x": location_native[2],
                "score": value,
            })
            new_edges.append((end_id, gap_node_id))
            node_info[gap_node_id] = {"t": target_t, "z": location_native[0], "y": location_native[1], "x": location_native[2]}
            budget_used += 1
            bridged = True
            break

        # bridged or not, move on -- an unbridged track just ends, no fabricated node

    if new_nodes:
        nodes = pd.concat([nodes, pd.DataFrame(new_nodes)], ignore_index=True)
    if new_edges:
        edges = pd.concat([edges, pd.DataFrame(new_edges, columns=["source_id", "target_id"])], ignore_index=True)

    return TrackingResult(nodes=nodes, edges=edges)