"""
Chains association.associate_frame across a full video's detections,
producing plain node/edge output. Deliberately minimal: no gap-closing
(Phase 3) and no division creation (Phase 4) -- unmatched tracks simply
end, unmatched detections simply start new tracks. This is what Phase 3
and 4 will each build on top of, not a replacement for them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import association as assoc


@dataclass
class TrackedNode:
    node_id: int
    t: int
    z: float
    y: float
    x: float
    score: float = 1.0


@dataclass
class TrackingResult:
    nodes: pd.DataFrame   # node_id, t, z, y, x
    edges: pd.DataFrame   # source_id, target_id


def _detection_score(row, has_score_column: bool) -> float:
    return float(row.score) if has_score_column else 1.0


def track_video(
    detections_per_frame: dict[int, pd.DataFrame],
    scale: tuple[float, float, float],
    config: assoc.AssociationConfig = assoc.AssociationConfig(),
) -> TrackingResult:
    """
    detections_per_frame : {t: DataFrame[z, y, x, score]}, one entry per
        timepoint that has detections (gaps in the key sequence are fine --
        this function does NOT bridge them, a track simply ends the frame
        before a gap and no track claims the detections after it; that
        bridging is Phase 3's job).
    scale : physical (Z, Y, X) scale for the coordinate space detections
        are in (pass NATIVE_SCALE for native voxel coordinates).

    Returns node_id-labeled nodes and source_id/target_id edges, ready to
    be combined with division edges (Phase 4) and written to the
    submission format.
    """
    node_rows: list[TrackedNode] = []
    edge_rows: list[tuple[int, int]] = []
    next_node_id = 1
    next_track_id = 1

    active_tracks: list[assoc.Track] = []
    last_node_id_of_track: dict[int, int] = {}  # track_id -> node_id of its most recent node

    for t in sorted(detections_per_frame.keys()):
        detections = detections_per_frame[t].reset_index(drop=True)
        has_score = "score" in detections.columns

        # every detection at this frame becomes a node, whether or not it
        # ends up matched to an existing track
        det_node_ids = []
        for row in detections.itertuples(index=False):
            node_id = next_node_id
            next_node_id += 1
            score = float(row.score) if has_score else 1.0
            node_rows.append(TrackedNode(node_id=node_id, t=t, z=row.z, y=row.y, x=row.x, score=score))
            det_node_ids.append(node_id)

        if active_tracks:
            result = assoc.associate_frame(active_tracks, detections, scale, config)
        else:
            result = assoc.AssociationResult([], [], list(range(len(detections))))

        new_active_tracks: list[assoc.Track] = []

        for track_idx, det_idx in result.matched_pairs:
            track = active_tracks[track_idx]
            det_row = detections.iloc[det_idx]
            track.positions.append((det_row["z"], det_row["y"], det_row["x"]))
            track.confidences.append(_detection_score(det_row, has_score))
            track.timepoints.append(t)

            parent_node_id = last_node_id_of_track[track.track_id]
            child_node_id = det_node_ids[det_idx]
            edge_rows.append((parent_node_id, child_node_id))
            last_node_id_of_track[track.track_id] = child_node_id
            new_active_tracks.append(track)

        # unmatched tracks simply end here (no gap-closing at this phase);
        # unmatched detections start new tracks
        for det_idx in result.unmatched_detection_indices:
            det_row = detections.iloc[det_idx]
            new_track = assoc.Track(
                track_id=next_track_id,
                positions=[(det_row["z"], det_row["y"], det_row["x"])],
                confidences=[_detection_score(det_row, has_score)],
                timepoints=[t],
            )
            next_track_id += 1
            last_node_id_of_track[new_track.track_id] = det_node_ids[det_idx]
            new_active_tracks.append(new_track)

        active_tracks = new_active_tracks

    nodes_df = pd.DataFrame([{"node_id": n.node_id, "t": n.t, "z": n.z, "y": n.y, "x": n.x, "score": n.score} for n in node_rows])
    edges_df = pd.DataFrame(edge_rows, columns=["source_id", "target_id"])
    return TrackingResult(nodes=nodes_df, edges=edges_df)