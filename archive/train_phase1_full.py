"""
Phase 1, full pipeline: raw zarr -> trained detector weights.

Run on Kaggle as:
    uv run python train_phase1_full.py

Two datasets must be attached:
  1. The competition dataset -- mounts at data_io.TRAIN_DIR /
     data_io.TEST_DIR automatically (both under
     /kaggle/input/competitions/biohub-cell-tracking-during-development/).
  2. Your own uploaded code dataset (CODE_DATASET_DIR below) containing
     every file this script imports.

What "validation" means here -- read this before trusting the numbers:
  - `val_loss` is the same weighted heatmap MSE used for training,
    evaluated on held-out embryos. It's a proxy for "is the detector
    learning at all", NOT a leaderboard-predictive number -- heatmap MSE
    can improve while actual detection quality stays flat or gets worse.
  - `node_recall` (node_validation.py) is a REAL signal: genuine
    tracksdata centroid-distance matching between predicted peaks and GT
    nodes, the same primitive the official competition metric itself uses
    internally. This is the number to actually watch.
  - Neither of these is the full competition score (edge/division
    Jaccard) -- that requires Phase 2 (linking), which doesn't exist yet.
    Building a throwaway linker just to produce an edge-Jaccard-shaped
    number here would look more complete than it is; better to be
    explicit that this script only validates detection quality.
  - sigma_um, detection threshold, and NMS radius below are still the
    same placeholders flagged throughout this project -- not measured
    against real data yet.
"""

from __future__ import annotations

import os
import sys
import time

CODE_DATASET_DIR = "/kaggle/input/datasets/khushvaghela/train-u-net/Train U-NET"
if not os.path.isdir(CODE_DATASET_DIR):
    raise FileNotFoundError(
        f"Code dataset directory does not exist: {CODE_DATASET_DIR}\n"
        f"Check the actual mount name with `ls /kaggle/input` in a cell first -- "
        f"Kaggle slugifies your dataset title, it may not match what you expect."
    )
if CODE_DATASET_DIR not in sys.path:
    sys.path.insert(0, CODE_DATASET_DIR)

import numpy as np
import torch

import data_io as bio_io
import preprocessing as prep
import targets
import detect
import node_validation as nv
from detector_model import Detector3D, DetectorConfig
from train_detector import build_training_window, train_step
from pipeline_config import DOWNSAMPLE, SIGMA_UM, WINDOW_SIZE


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VAL_FRACTION = 0.2
SPLIT_SEED = 0
VAL_WINDOWS_PER_SAMPLE = 4    # FIXED once at startup -- not redrawn each epoch (see select_fixed_val_windows)

WINDOWS_PER_SAMPLE_PER_EPOCH = 4   # random training (sample, t_start) draws per sample per epoch
N_EPOCHS = 20
LEARNING_RATE = 1e-4
CHECKPOINT_EVERY_N_EPOCHS = 5

DETECT_THRESHOLD = 0.3     # placeholder -- not tuned against real data yet
NMS_MIN_DISTANCE_UM = 9.54  # placeholder -- same
NODE_MATCH_MAX_DISTANCE_UM = 7.0  # matches the competition's own edge-matching tolerance

DETECTOR_CONFIG = DetectorConfig(
    unet_out_channels=32,
    unet_layers=(32, 64, 128),
    gradient_checkpointing=True,
    skip_fullres_temporal=True,
)

OUTPUT_DIR = "/kaggle/working"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "detector_weights.pt")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def sample_random_window_start(sample: bio_io.Sample, window_size: int, rng: np.random.Generator) -> int:
    vol = bio_io.open_zarr_volume(sample.zarr_path)
    n_t = vol.shape[0]
    if n_t < window_size:
        raise ValueError(f"sample {sample.name} has only {n_t} timepoints, needs >= {window_size}")
    return int(rng.integers(0, n_t - window_size + 1))


def select_fixed_val_windows(
    val_samples: list[bio_io.Sample],
    windows_per_sample: int,
    seed: int,
) -> list[tuple[bio_io.Sample, int]]:
    """
    Pick validation (sample, t_start) pairs ONCE, before training starts,
    and reuse the exact same set every epoch. Sharing one rng between
    training and validation (drawing fresh validation windows each epoch)
    makes val_loss/val_recall trend noise indistinguishable from real
    improvement -- this fixes that by making validation deterministic
    across epochs, using its own rng seeded independently of training's.
    """
    val_rng = np.random.default_rng(seed)
    windows = []
    for sample in val_samples:
        for _ in range(windows_per_sample):
            t_start = sample_random_window_start(sample, WINDOW_SIZE, val_rng)
            windows.append((sample, t_start))
    return windows


def run_training_epoch(model, optimizer, train_samples, rng, windows_per_sample, device) -> float:
    losses = []
    for sample in train_samples:
        for _ in range(windows_per_sample):
            t_start = sample_random_window_start(sample, WINDOW_SIZE, rng)
            loss = train_step(model, optimizer, sample, t_start, WINDOW_SIZE, DOWNSAMPLE, SIGMA_UM, device)
            losses.append(loss)
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_val_loss(model, val_windows, device) -> float:
    """Same weighted heatmap MSE as training -- a proxy signal, see module docstring."""
    if not val_windows:
        return float("nan")
    losses = []
    model.eval()
    with torch.no_grad():
        for sample, t_start in val_windows:
            window, target, weight = build_training_window(
                sample, t_start, WINDOW_SIZE, DOWNSAMPLE, SIGMA_UM, device
            )
            pred = model(window)
            loss = targets.focal_heatmap_loss(pred, target, weight).item()
            losses.append(loss)
    return float(np.mean(losses))


def evaluate_node_recall(model, val_windows, device) -> float:
    """
    REAL signal: run the actual detection path (preprocess -> model ->
    peak extraction -> native coords) on each validation window's first
    frame, then match against real GT nodes via tracksdata. This is what
    the printed "recall" number in the training loop actually measures --
    not a proxy.
    """
    if not val_windows:
        return float("nan")
    recalls = []
    for sample, t_start in val_windows:
        vol = bio_io.open_zarr_volume(sample.zarr_path)
        geff = bio_io.load_geff(sample.geff_path)

        raw_frame = np.asarray(vol[t_start])[None]
        frame_processed = prep.preprocess_window(raw_frame, DOWNSAMPLE)[0]

        pred_native = detect.detect_frame(
            model, frame_processed, DOWNSAMPLE, bio_io.NATIVE_SCALE,
            threshold=DETECT_THRESHOLD, min_distance_um=NMS_MIN_DISTANCE_UM,
        )
        pred_native["t"] = t_start  # detect_frame doesn't attach t -- single frame, add it back for node_validation

        gt_nodes = geff.nodes[geff.nodes["t"] == t_start][["t", "z", "y", "x"]]
        result = nv.compute_node_recall(
            pred_native[["t", "z", "y", "x"]], gt_nodes, bio_io.NATIVE_SCALE, NODE_MATCH_MAX_DISTANCE_UM,
        )
        if result.recall == result.recall:  # skip NaN (no GT nodes in this window's frame)
            recalls.append(result.recall)
    return float(np.mean(recalls)) if recalls else float("nan")


def sanity_check_peaks(model, sample: bio_io.Sample, device):
    vol = bio_io.open_zarr_volume(sample.zarr_path)
    geff = bio_io.load_geff(sample.geff_path)
    t = 0
    raw_frame = np.asarray(vol[t])[None]
    frame_processed = prep.preprocess_window(raw_frame, DOWNSAMPLE)[0]
    result = detect.detect_frame(
        model, frame_processed, DOWNSAMPLE, bio_io.NATIVE_SCALE,
        threshold=DETECT_THRESHOLD, min_distance_um=NMS_MIN_DISTANCE_UM,
    )
    n_gt = (geff.nodes["t"] == t).sum()
    print(f"  sanity check on {sample.name} t={t}: {len(result)} predicted peaks vs {n_gt} GT-labeled nodes")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_rng = np.random.default_rng(SPLIT_SEED)

    print(f"loading samples from {bio_io.TRAIN_DIR}")
    all_samples = bio_io.list_samples(bio_io.TRAIN_DIR)
    all_samples = [s for s in all_samples if s.geff_path is not None]
    if not all_samples:
        raise RuntimeError(
            f"no samples with ground truth found under {bio_io.TRAIN_DIR} -- "
            f"is the competition dataset actually attached to this notebook?"
        )
    train_samples, val_samples = bio_io.embryo_grouped_split(all_samples, VAL_FRACTION, SPLIT_SEED)
    print(f"{len(all_samples)} samples total, {len({s.embryo_id for s in all_samples})} embryos -> "
          f"{len(train_samples)} train / {len(val_samples)} val (split by embryo)")

    val_windows = select_fixed_val_windows(val_samples, VAL_WINDOWS_PER_SAMPLE, seed=SPLIT_SEED + 1)
    print(f"{len(val_windows)} fixed validation windows selected once, reused every epoch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = Detector3D(DETECTOR_CONFIG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_recall = -1.0
    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        train_loss = run_training_epoch(model, optimizer, train_samples, train_rng, WINDOWS_PER_SAMPLE_PER_EPOCH, device)
        val_loss = evaluate_val_loss(model, val_windows, device)
        val_recall = evaluate_node_recall(model, val_windows, device)
        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{N_EPOCHS}  train_loss={train_loss:.5f}  "
              f"val_loss(proxy)={val_loss:.5f}  val_node_recall(real)={val_recall:.4f}  ({dt:.1f}s)")

        if val_recall == val_recall and val_recall > best_recall:  # skip NaN
            best_recall = val_recall
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": DETECTOR_CONFIG,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_node_recall": val_recall,
            }, CHECKPOINT_PATH.replace(".pt", "_best.pt"))

        if epoch % CHECKPOINT_EVERY_N_EPOCHS == 0 or epoch == N_EPOCHS:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": DETECTOR_CONFIG,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_node_recall": val_recall,
            }, CHECKPOINT_PATH)
            print(f"  checkpoint saved to {CHECKPOINT_PATH}")

    if val_samples:
        print("\nfinal sanity check:")
        sanity_check_peaks(model, val_samples[0], device)

    print(f"\ndone. best val_node_recall={best_recall:.4f}, "
          f"weights at {CHECKPOINT_PATH} and {CHECKPOINT_PATH.replace('.pt', '_best.pt')}")


if __name__ == "__main__":
    main()
