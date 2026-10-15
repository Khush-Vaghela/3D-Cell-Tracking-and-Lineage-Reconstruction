"""
Combined Phase 2 (association) + Phase 3 (gap closing) for one sample.

Run on Kaggle as a library import (see run_sample below), using a Phase 1
checkpoint -- typically detector_weights_best.pt (selected by real
val_node_recall, not the proxy loss -- see train_phase1_full.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import data_io as bio_io
import preprocessing as prep
import detect
import association as assoc
import tracker
import gap_closing as gc
from detector_model import Detector3D, DetectorConfig
from pipeline_config import DOWNSAMPLE


def load_detector(checkpoint_path: str, device: str | torch.device = "cpu") -> Detector3D:
    # torch>=2.6 defaults torch.load to weights_only=True, which rejects
    # any non-tensor object in the checkpoint -- including our own
    # DetectorConfig dataclass. Allowlisting it explicitly (rather than
    # blanket weights_only=False) since this checkpoint format is ours,
    # not an untrusted third-party file, but still only allowlists the
    # one class we actually need.
    torch.serialization.add_safe_globals([DetectorConfig])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = Detector3D(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_sample(
    model: Detector3D,
    sample: bio_io.Sample,
    downsample: tuple[int, int, int] = DOWNSAMPLE,
    detect_threshold: float = 0.3,
    nms_min_distance_um: float = 5.0,
    assoc_config: assoc.AssociationConfig = assoc.AssociationConfig(),
    gap_config: gc.GapClosingConfig = gc.GapClosingConfig(),
) -> tracker.TrackingResult:
    """
    Full Phase 1(inference) -> 2 -> 3 pipeline for one sample:
      1. Detect every frame, keeping both the extracted peaks (for
         Phase 2) and the raw heatmap (for Phase 3's evidence check).
      2. Associate frame-to-frame (Phase 2).
      3. Close gaps using the same heatmaps already computed in step 1
         -- no re-inference needed (detect.compute_heatmap_single_frame
         was already called once per frame; Phase 3 reuses it directly).
    """
    device = next(model.parameters()).device
    vol = bio_io.open_zarr_volume(sample.zarr_path)
    n_frames = vol.shape[0]

    detections_per_frame: dict[int, pd.DataFrame] = {}
    heatmaps_per_frame: dict[int, np.ndarray] = {}

    for t in range(n_frames):
        raw_frame = np.asarray(vol[t])[None]
        frame_processed = prep.preprocess_window(raw_frame, downsample)[0]

        heatmap = detect.compute_heatmap_single_frame(model, frame_processed)
        heatmaps_per_frame[t] = heatmap

        scale_down = prep.downsampled_scale(bio_io.NATIVE_SCALE, downsample)
        peaks_down = detect.extract_peaks_single_frame(heatmap, detect_threshold, nms_min_distance_um, scale_down)
        detections_per_frame[t] = detect.peaks_to_native_coords(peaks_down, downsample)

    # Phase 2
    tracking_result = tracker.track_video(detections_per_frame, bio_io.NATIVE_SCALE, assoc_config)

    # Phase 3
    repaired_result = gc.close_gaps(
        tracking_result, heatmaps_per_frame, downsample, bio_io.NATIVE_SCALE, n_frames, gap_config
    )
    return repaired_result


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python pipeline_phase2_3.py <checkpoint_path>")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_detector(sys.argv[1], device)

    samples = [s for s in bio_io.list_samples(bio_io.TRAIN_DIR) if s.geff_path is not None]
    sample = samples[0]
    print(f"running phase 2+3 on {sample.name}")
    result = run_sample(model, sample)
    print(f"nodes={len(result.nodes)} edges={len(result.edges)}")
