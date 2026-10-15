"""
Phase 0.3 -- EDA against real training data, to replace placeholder
parameters with numbers measured from the actual dataset instead of
guesses.

Run on Kaggle as:
    uv run python eda_phase0.py

Measures three things, each mapping to specific parameters you should
then update in pipeline_config.py / train_phase1_full.py:

1. Physical spacing between GT-labeled cells (nearest-neighbor distance,
   per frame, in um) -> informs SIGMA_UM (targets.py's Gaussian spread)
   and NMS_MIN_DISTANCE_UM (detect.py's peak-suppression radius).
2. Labeled node count over time, per sample -> sanity-checks the
   sparse-GT density assumptions baked into targets.py's
   background_weight_per_frame heuristic (does labeled count grow
   roughly monotonically, consistent with cell divisions, or is it
   erratic in a way that heuristic doesn't handle well).
3. Wall-clock time for one real forward+backward step at the actual
   DETECTOR_CONFIG size -> informs whether unet_layers/out_channels and
   N_EPOCHS/WINDOWS_PER_SAMPLE_PER_EPOCH in train_phase1_full.py are
   sized sensibly for your time budget, before committing to a long run.

This does NOT touch tuning of DETECT_THRESHOLD -- that one can only be
tuned meaningfully against val_node_recall/precision from an actually
*trained* model (see the printed guidance at the end), not from raw data
alone.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import data_io as bio_io
from detector_model import Detector3D, DetectorConfig
from train_detector import build_training_window, train_step
from pipeline_config import DOWNSAMPLE, WINDOW_SIZE


N_SAMPLES_TO_INSPECT = 5     # how many train samples to pull GT spacing stats from
N_TIMING_STEPS = 5           # how many real train_step() calls to time


# ---------------------------------------------------------------------------
# 1. Physical spacing between GT-labeled cells
# ---------------------------------------------------------------------------

def nearest_neighbor_distances_per_frame(nodes: pd.DataFrame, scale: tuple[float, float, float]) -> np.ndarray:
    """For each timepoint with >=2 labeled nodes, physical-distance nearest
    neighbor for every node. Returns a flat array across all timepoints."""
    scale_arr = np.array(scale)
    all_dists = []
    for t, group in nodes.groupby("t"):
        if len(group) < 2:
            continue
        coords_physical = group[["z", "y", "x"]].to_numpy() * scale_arr
        tree = cKDTree(coords_physical)
        dists, _ = tree.query(coords_physical, k=2)  # k=1 is the point itself (distance 0)
        all_dists.append(dists[:, 1])
    return np.concatenate(all_dists) if all_dists else np.array([])


def run_spacing_eda(samples: list[bio_io.Sample]) -> None:
    print("=" * 70)
    print("1. GT cell spacing (nearest-neighbor distance, physical um)")
    print("=" * 70)

    all_dists = []
    for sample in samples[:N_SAMPLES_TO_INSPECT]:
        geff = bio_io.load_geff(sample.geff_path)
        dists = nearest_neighbor_distances_per_frame(geff.nodes, bio_io.NATIVE_SCALE)
        if len(dists) == 0:
            print(f"  {sample.name}: not enough co-labeled nodes per frame to measure spacing")
            continue
        all_dists.append(dists)
        print(f"  {sample.name}: n={len(dists)}  median={np.median(dists):.2f}um  "
              f"p10={np.percentile(dists, 10):.2f}um  p90={np.percentile(dists, 90):.2f}um")

    if not all_dists:
        print("  No spacing data collected -- can't recommend sigma_um/NMS radius from this. "
              "Check that your samples actually have >=2 GT nodes in at least some shared frames.")
        return

    combined = np.concatenate(all_dists)
    median_spacing = np.median(combined)
    p10_spacing = np.percentile(combined, 10)

    print(f"\n  Combined across samples: median={median_spacing:.2f}um  p10={p10_spacing:.2f}um")
    print("\n  Recommendation:")
    print(f"    SIGMA_UM ~= median_spacing / 4  ~= {median_spacing / 4:.2f}  "
          f"(keeps Gaussian bumps from adjacent cells mostly non-overlapping; "
          f"current placeholder is 3.0)")
    print(f"    NMS_MIN_DISTANCE_UM ~= p10_spacing * 0.7  ~= {p10_spacing * 0.7:.2f}  "
          f"(suppresses duplicate detections of the same cell without merging "
          f"genuinely close-but-distinct cells at the tight end of the spacing "
          f"distribution; current placeholder is 5.0)")
    print(f"    Sanity check: downsampled voxel size is {bio_io.NATIVE_SCALE[0] * DOWNSAMPLE[0]:.3f}um "
          f"isotropic -- if median_spacing is only a few multiples of that, cells are "
          f"barely resolved at this downsample factor and DOWNSAMPLE may need revisiting.")


# ---------------------------------------------------------------------------
# 2. Labeled density over time
# ---------------------------------------------------------------------------

def run_density_eda(samples: list[bio_io.Sample]) -> None:
    print("\n" + "=" * 70)
    print("2. Labeled node count over time (sanity check for targets.py's density heuristic)")
    print("=" * 70)

    for sample in samples[:N_SAMPLES_TO_INSPECT]:
        geff = bio_io.load_geff(sample.geff_path)
        vol = bio_io.open_zarr_volume(sample.zarr_path)
        counts = geff.nodes["t"].value_counts().sort_index()
        n_t = vol.shape[0]
        first_count = counts.iloc[0] if len(counts) else 0
        last_count = counts.iloc[-1] if len(counts) else 0
        print(f"  {sample.name}: n_timepoints={n_t}  labeled_at_t0={first_count}  "
              f"labeled_at_tN={last_count}  estimated_number_of_nodes={geff.estimated_number_of_nodes}")

    print("\n  Interpretation: labeled count should generally trend up over time (divisions "
          "increase cell count) for the density-growth assumption in "
          "targets.background_weight_per_frame to hold reasonably. If a sample's count is flat "
          "or erratic, that sample's per-frame background weight estimate may be less reliable.")


# ---------------------------------------------------------------------------
# 3. Real timing budget
# ---------------------------------------------------------------------------

def run_timing_eda(samples: list[bio_io.Sample], config: DetectorConfig, device: str) -> None:
    print("\n" + "=" * 70)
    print(f"3. Real train_step() timing at DETECTOR_CONFIG (layers={config.unet_layers}, "
          f"out_channels={config.unet_out_channels}) on device={device}")
    print("=" * 70)

    import torch
    model = Detector3D(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sample = samples[0]

    # one warmup step (excluded from timing -- first CUDA call pays init cost)
    train_step(model, optimizer, sample, t_start=0, window_size=WINDOW_SIZE, device=device)

    times = []
    for _ in range(N_TIMING_STEPS):
        t0 = time.time()
        train_step(model, optimizer, sample, t_start=0, window_size=WINDOW_SIZE, device=device)
        times.append(time.time() - t0)

    mean_step = np.mean(times)
    print(f"  mean step time: {mean_step:.3f}s over {N_TIMING_STEPS} steps (std={np.std(times):.3f}s)")
    print("\n  Recommendation:")
    print(f"    For N_EPOCHS x WINDOWS_PER_SAMPLE_PER_EPOCH x len(train_samples) total steps, "
          f"total wall clock ~= that count x {mean_step:.3f}s.")
    print(f"    E.g. current defaults (20 epochs x 4 windows/sample) on {len(samples)} samples "
          f"~= {20 * 4 * len(samples) * mean_step / 60:.1f} minutes -- compare against how long "
          f"you're actually willing to let a Kaggle session run before adjusting N_EPOCHS, "
          f"unet_layers, or out_channels.")


def main():
    print(f"loading samples from {bio_io.TRAIN_DIR}")
    all_samples = [s for s in bio_io.list_samples(bio_io.TRAIN_DIR) if s.geff_path is not None]
    if not all_samples:
        raise RuntimeError(f"no samples with ground truth found under {bio_io.TRAIN_DIR}")
    print(f"found {len(all_samples)} samples, inspecting the first {N_SAMPLES_TO_INSPECT}\n")

    run_spacing_eda(all_samples)
    run_density_eda(all_samples)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_config = DetectorConfig(unet_out_channels=32, unet_layers=(32, 64, 128),
                                  gradient_checkpointing=True, skip_fullres_temporal=True)
    run_timing_eda(all_samples, real_config, device)

    print("\n" + "=" * 70)
    print("Not covered here -- needs a trained model first:")
    print("  DETECT_THRESHOLD and further NMS tuning: sweep against val_node_recall/precision "
          "(node_validation.py) once you have a trained checkpoint, not from raw data alone.")
    print("=" * 70)


if __name__ == "__main__":
    main()
