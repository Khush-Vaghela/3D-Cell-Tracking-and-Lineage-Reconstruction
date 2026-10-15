"""
End-to-End Pipeline (Tracksdata Native).

Executes Phase 1 (Detection) -> Phase 2/3/4 (Native Graph + ILP Solve) -> 
Phase 4.5 (Division Validation) -> Submission CSV formatting.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import torch

import data_io as bio_io
import preprocessing as prep
import detect
import tracksdata_pipeline as tdp
import division_validation as dv

# Import model architectures
from detector_model import Detector3D, DetectorConfig
from learned_association import EdgeScorerMLP
from division_mlp import DivisionScorerMLP
from pipeline_config import DOWNSAMPLE

# Standard Kaggle submission columns[cite: 10]
COLUMNS: tuple[str, ...] = (
    "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id",
)


def load_models(
    detector_path: str, 
    division_scorer_path: str, 
    edge_scorer_path: str | None = None,
    device: str | torch.device = "cpu"
) -> tuple[Detector3D, DivisionScorerMLP, EdgeScorerMLP | None]:
    """Loads all trained models from their Kaggle dataset checkpoint paths."""
    
    # 1. Load Phase 1 Detector
    torch.serialization.add_safe_globals([DetectorConfig])
    det_checkpoint = torch.load(detector_path, map_location=device, weights_only=False)
    detector = Detector3D(det_checkpoint["config"]).to(device)
    detector.load_state_dict(det_checkpoint["model_state_dict"])
    detector.eval()

    # 2. Load Phase 4.5 Division Scorer[cite: 9]
    div_scorer = DivisionScorerMLP()
    div_scorer.load_state_dict(torch.load(division_scorer_path, map_location=device, weights_only=True))
    div_scorer.eval()

    # 3. Load Phase 2 Edge Scorer (Optional)
    edge_scorer = None
    if edge_scorer_path is not None and os.path.exists(edge_scorer_path):
        edge_scorer = EdgeScorerMLP()
        edge_scorer.load_state_dict(torch.load(edge_scorer_path, map_location=device, weights_only=True))
        edge_scorer.eval()

    return detector, div_scorer, edge_scorer


def run_sample_end_to_end(
    detector: Detector3D,
    div_scorer: DivisionScorerMLP,
    sample: bio_io.Sample,
    edge_scorer: EdgeScorerMLP | None = None,
    downsample: tuple[int, int, int] = DOWNSAMPLE,
    detect_threshold: float = 0.3,
    nms_min_distance_um: float = 5.0,
    division_threshold: float = 0.5,
    use_ilp: bool = True,
) -> dv.TrackingResult:
    """Runs the complete tracksdata native pipeline on a single sample."""
    
    device = next(detector.parameters()).device
    vol = bio_io.open_zarr_volume(sample.zarr_path)
    n_frames = vol.shape[0]

    detections_per_frame: dict[int, pd.DataFrame] = {}
    raw_frames_by_t: dict[int, np.ndarray] = {}

    print(f"--- Phase 1: Detecting cells across {n_frames} frames ---")
    for t in range(n_frames):
        raw_frame = np.asarray(vol[t])[None]
        raw_frames_by_t[t] = raw_frame
        
        frame_processed = prep.preprocess_window(raw_frame, downsample)[0]
        
        heatmap = detect.compute_heatmap_single_frame(detector, frame_processed)
        scale_down = prep.downsampled_scale(bio_io.NATIVE_SCALE, downsample)
        peaks_down = detect.extract_peaks_single_frame(heatmap, detect_threshold, nms_min_distance_um, scale_down)
        detections_per_frame[t] = detect.peaks_to_native_coords(peaks_down, downsample)

    print("--- Phase 2 & 3: Building multi-frame Candidate Graph ---")
    candidate_graph = tdp.build_candidate_graph(
        detections_per_frame, 
        bio_io.NATIVE_SCALE,
        config=tdp.CandidateGraphConfig(delta_t=3) 
    )

    if edge_scorer is not None:
        print("Note: Phase 2 EdgeScorerMLP loaded but GenericFuncEdgeAttrs injection is pending.")

    print("--- Phase 4: Solving tracking network flow ---")
    if use_ilp:
        solution_graph = tdp.solve_ilp(candidate_graph, node_weight=-10.0)
    else:
        solution_graph = tdp.solve_nearest_neighbors(candidate_graph, max_children=2)

    tracking_result = tdp.graph_view_to_tracking_result(solution_graph)

    print("--- Phase 4.5: Validating solver divisions ---")
    final_result = dv.validate_divisions(
        tracking_result,
        div_scorer,
        raw_frames_by_t,
        bio_io.NATIVE_SCALE,
        threshold=division_threshold
    )
    
    return final_result


def tracking_result_to_submission_rows(tracking_result: dv.TrackingResult, dataset_name: str) -> pl.DataFrame:
    """
    Converts a TrackingResult into the exact Kaggle submission schema[cite: 10].
    Unused per-row-type fields are set to -1, and coordinates are rounded to integers[cite: 10].
    """
    nodes_df = tracking_result.nodes
    edges_df = tracking_result.edges

    # Format Nodes
    nodes_pl = pl.from_pandas(nodes_df).select(
        pl.lit(dataset_name).alias("dataset"),
        pl.lit("node").alias("row_type"),
        pl.col("node_id").cast(pl.Int64),
        pl.col("t").cast(pl.Int64),
        pl.col("z").cast(pl.Float64).round(0).cast(pl.Int64),
        pl.col("y").cast(pl.Float64).round(0).cast(pl.Int64),
        pl.col("x").cast(pl.Float64).round(0).cast(pl.Int64),
        pl.lit(-1, dtype=pl.Int64).alias("source_id"),
        pl.lit(-1, dtype=pl.Int64).alias("target_id"),
    )

    # Format Edges
    if len(edges_df) > 0:
        edges_pl = pl.from_pandas(edges_df).select(
            pl.lit(dataset_name).alias("dataset"),
            pl.lit("edge").alias("row_type"),
            pl.lit(-1, dtype=pl.Int64).alias("node_id"),
            pl.lit(-1, dtype=pl.Int64).alias("t"),
            pl.lit(-1, dtype=pl.Int64).alias("z"),
            pl.lit(-1, dtype=pl.Int64).alias("y"),
            pl.lit(-1, dtype=pl.Int64).alias("x"),
            pl.col("source_id").cast(pl.Int64),
            pl.col("target_id").cast(pl.Int64),
        )
    else:
        edges_pl = pl.DataFrame(schema=nodes_pl.schema)

    return pl.concat([nodes_pl, edges_pl])


def generate_final_submission(
    detector_path: str,
    division_scorer_path: str,
    test_dir: str,
    output_csv: str,
    device: str = "cuda",
) -> None:
    """Runs inference on all test samples and writes the final submission.csv."""
    
    print("Loading models...")
    detector, div_scorer, _ = load_models(detector_path, division_scorer_path, device=device)
    
    # Load all test samples (no ground truth needed)
    test_samples = bio_io.list_samples(test_dir)
    
    all_frames = []
    
    for sample in test_samples:
        print(f"\nProcessing sample: {sample.name}")
        tracking_result = run_sample_end_to_end(
            detector=detector,
            div_scorer=div_scorer,
            sample=sample,
            use_ilp=True
        )
        
        # Flatten and format for this specific sample[cite: 10]
        sample_rows = tracking_result_to_submission_rows(tracking_result, sample.name)
        all_frames.append(sample_rows)
        print(f"Produced {sample_rows.height} submission rows for {sample.name}.")

    # Concatenate everything and add the 'id' index column[cite: 10]
    if all_frames:
        final_table = pl.concat(all_frames)
    else:
        final_table = pl.DataFrame(schema={c: (pl.String if c in ("dataset", "row_type") else pl.Int64) for c in COLUMNS})
        
    final_table = final_table.with_row_index("id")
    
    # Write to disk[cite: 10]
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_table.write_csv(output_path)
    print(f"\nSuccess! Wrote {final_table.height} total rows to {output_path}")


if __name__ == "__main__":
    # Example Kaggle Execution
    DETECTOR_WEIGHTS = "/kaggle/input/your-weights-dataset/detector_weights_best.pt"
    DIVISION_WEIGHTS = "/kaggle/input/your-weights-dataset/division_scorer_weights.pt"
    TEST_DIRECTORY = bio_io.TEST_DIR
    OUTPUT_FILE = "/kaggle/working/submission.csv"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    generate_final_submission(
        detector_path=DETECTOR_WEIGHTS,
        division_scorer_path=DIVISION_WEIGHTS,
        test_dir=TEST_DIRECTORY,
        output_csv=OUTPUT_FILE,
        device=device
    )