"""
End-to-End Training Script for Phase 4.5 (DivisionScorerMLP).

1. Runs Phase 1 -> 4 to generate solver-native division candidates.
2. Extracts (X, y) features and labels using GT graphs.
3. Trains the DivisionScorerMLP.
4. Sweeps for the optimal threshold on a held-out validation set.
"""

import numpy as np
import pandas as pd
import torch
import random
import os
import zipfile

import data_io as bio_io
import preprocessing as prep
import detect
import tracksdata_pipeline as tdp
import division_validation as dv
import division_mlp as dmlp
import division_threshold_tuning as dtt
from detector_model import Detector3D, DetectorConfig
from pipeline_config import DOWNSAMPLE
import json
from pathlib import Path
import numpy as np
import blosc2

class FrameCache:
    """Smart cache using the blazing-fast Blosc2 direct-read trick!"""
    def __init__(self, zarr_path):
        self.zarr_path = Path(zarr_path)
        self.cache = {}
        
        # Read the metadata just once
        with open(self.zarr_path / "0" / "zarr.json") as f: 
            m = json.load(f)
        self.meta = dict(shape=tuple(m["shape"]), dtype=np.dtype(m["data_type"]))
        self.shape = self.meta["shape"] 
        
    def __getitem__(self, key):
        # 1. Separate the time index 't' from the 3D spatial slices
        if isinstance(key, tuple):
            t = key[0]
            spatial_slices = key[1:]
        else:
            t = key
            spatial_slices = tuple() # No spatial slices provided
            
        # 2. Load the frame into RAM if it isn't already there
        if t not in self.cache:
            chunk_path = self.zarr_path / "0" / "c" / str(t) / "0" / "0" / "0"
            with open(chunk_path, "rb") as f:
                buf = blosc2.decompress(f.read())
            frame = np.frombuffer(buf, dtype=self.meta["dtype"]).reshape(self.meta["shape"][1:])
            self.cache[t] = frame
            
        # 3. Apply the spatial crop (if any) and return the pixels!
        return self.cache[t][spatial_slices]

def generate_sample_data(
    detector: Detector3D, 
    sample: bio_io.Sample, 
    downsample: tuple[int, int, int] = DOWNSAMPLE,
    detect_threshold: float = 0.30,  
) -> tuple[dv.TrackingResult, object]:
    vol = bio_io.open_zarr_volume(sample.zarr_path)
    n_frames = vol.shape[0]

    detections_per_frame: dict[int, pd.DataFrame] = {}

    for t in range(n_frames):
        raw_frame = np.asarray(vol[t])[None]
        frame_processed = prep.preprocess_window(raw_frame, downsample)[0]
        heatmap = detect.compute_heatmap_single_frame(detector, frame_processed)
        scale_down = prep.downsampled_scale(bio_io.NATIVE_SCALE, downsample)
        
        # NMS 2.5 perfectly preserves the close daughters
        peaks_down = detect.extract_peaks_single_frame(heatmap, detect_threshold, 2.5, scale_down)
        detections_per_frame[t] = detect.peaks_to_native_coords(peaks_down, downsample)

    # WIDE NET, STRICT TIMING
    candidate_graph = tdp.build_candidate_graph(
        detections_per_frame, 
        bio_io.NATIVE_SCALE,
        config=tdp.CandidateGraphConfig(
            distance_threshold_um=25.0,  
            delta_t=1,                   # CRITICAL: Forces exact frame-to-frame matching for the harvester!
            n_neighbors=10, 
            neighbors_per_frame=True     
        ) 
    )
    
    print("Solving graph via Robust Nearest Neighbors...")
    
    # THE PROVEN NN SOLVER
    solution_graph = tdp.solve_nearest_neighbors(candidate_graph, max_children=2)
    
    print("Graph Solved!")
    tracking_result = tdp.graph_view_to_tracking_result(solution_graph)
    
    # Coordinate reversal for the harvester
    scale_arr = np.array(bio_io.NATIVE_SCALE)
    tracking_result.nodes["z"] /= scale_arr[0]
    tracking_result.nodes["y"] /= scale_arr[1]
    tracking_result.nodes["x"] /= scale_arr[2]

    return tracking_result, vol

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    unzipped_dir_path = "/kaggle/input/datasets/khushvaghela/best-weights/detector_weights_best"

    # 2. Where we will save the reconstructed file
    repacked_pt_path = "/kaggle/working/detector_weights_best.pt"

    print("Repacking unzipped PyTorch weights with correct 'archive/' structure...")

    # 3. Create the ZIP file manually, forcing the 'archive/' parent folder
    with zipfile.ZipFile(repacked_pt_path, 'w', zipfile.ZIP_STORED) as zipf:
        for root, dirs, files in os.walk(unzipped_dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate path relative to the extracted folder
                rel_path = os.path.relpath(file_path, unzipped_dir_path)
                
                # CRITICAL FIX: Prepend 'archive/' to the internal ZIP path
                zip_path = os.path.join('archive', rel_path)
                zipf.write(file_path, zip_path)

    print(f"Successfully reconstructed weights at: {repacked_pt_path}")

    DETECTOR_WEIGHTS = repacked_pt_path
    
    # 1. Load Phase 1 Model
    print("Loading Phase 1 Detector...")
    torch.serialization.add_safe_globals([DetectorConfig])
    checkpoint = torch.load(DETECTOR_WEIGHTS, map_location=device, weights_only=False)
    detector = Detector3D(checkpoint["config"]).to(device)
    detector.load_state_dict(checkpoint["model_state_dict"])
    detector.eval()

    # 2. Get samples and split Train/Val
    # 2. Get samples and split Train/Val
    samples = [s for s in bio_io.list_samples(bio_io.TRAIN_DIR) if s.geff_path is not None]
    random.seed(42)
    random.shuffle(samples)
    
    # --- ADD THIS SINGLE LINE FOR YOUR QUICK TEST ---
    # samples = samples[:30] 
    # ------------------------------------------------
    
    # The math below remains exactly the same, but it will now operate on just 30 samples.
    # 80% of 30 means: 24 samples for MLP training, 6 samples for threshold tuning.
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    print(f"Using {len(train_samples)} samples for MLP training, {len(val_samples)} for threshold tuning.")

    # 3. Harvest Training Data (X, y)
    all_X, all_y = [], []
    for sample in train_samples:
        print(f"Harvesting training divisions from {sample.name}...")
        geff = bio_io.load_geff(sample.geff_path)
        tracking_result, raw_frames = generate_sample_data(detector, sample)
        
        # --- NEW UPDATE: ON-DEMAND BLOSC2 CACHE ---
        print("Initializing blosc2-powered frame cache for lightning-fast feature extraction...")
        # We pass the PATH (sample.zarr_path) to our new custom cache class!
        vol_ram = FrameCache(sample.zarr_path)
        
        X, y = dv.generate_validation_training_data(
            tracking_result, 
            geff, 
            vol_ram, 
            bio_io.NATIVE_SCALE,
            match_max_distance_um=12.0  # <-- WIDENED: Safe because Hungarian prevents double-mapping
        )
        
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            
    # --- ADD THIS SAFETY CHECK ---
    if not all_X:
        print("\n❌ ERROR: Zero divisions were found in these samples!")
        print("Try increasing your sample slice (e.g., samples[:100]) or run on the full dataset.")
        return
    # -----------------------------

    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)
    print(f"Harvested {len(X_train)} division candidates (Positives: {int(y_train.sum())}).")

    # --- THE FIX: BALANCE THE DATASET ---
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    # Keep all positives, but aggressively limit negatives to 2x the positives
    n_neg = min(len(neg_idx), len(pos_idx) * 2) 
    
    np.random.seed(42)
    neg_idx_sampled = np.random.choice(neg_idx, n_neg, replace=False)

    balanced_idx = np.concatenate([pos_idx, neg_idx_sampled])
    np.random.shuffle(balanced_idx) # Shuffle them together

    X_train = X_train[balanced_idx]
    y_train = y_train[balanced_idx]
    print(f"Balanced training set: {len(X_train)} candidates (Positives: {int(y_train.sum())}).")
    # ------------------------------------

    # >>> INSERT THE AWGN AUGMENTATION CODE HERE <<<
    # --- NEW: SYNTHETIC AUGMENTATION (AWGN) ---
    pos_features = X_train[y_train == 1]
    
    if len(pos_features) > 0:
        # Generate Additive White Gaussian Noise
        noise_std = 0.05 # Tune this based on your feature scaling
        noise = np.random.normal(0, noise_std, pos_features.shape)
        
        synthetic_pos_features = pos_features + noise
        synthetic_pos_labels = np.ones(len(synthetic_pos_features))

        # Append back to the training data
        X_train = np.vstack([X_train, synthetic_pos_features])
        y_train = np.concatenate([y_train, synthetic_pos_labels])
        
        # Shuffle one last time
        shuffle_idx = np.random.permutation(len(X_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
        print(f"After AWGN augmentation: {len(X_train)} candidates (Positives: {int(y_train.sum())}).")
    # ------------------------------------------

    # 4. Train the MLP
    print("\nTraining DivisionScorerMLP...")
    
    # --- ADD THIS SAFETY CHECK ---
    if len(X_train) == 0 or int(y_train.sum()) == 0:
        print("🚨 ERROR: 0 True Positives were harvested!")
        print("Skipping MLP training because a neural network cannot learn without at least some real divisions.")
        print("Please run on the full dataset (remove the samples slice) to gather enough examples.")
        return  # Gracefully exits the main() function without crashing
    # -----------------------------
    
    mlp, history = dmlp.train_division_scorer(X_train, y_train, epochs=150, lr=1e-3)
    
    # Save the weights
    torch.save(mlp.state_dict(), "/kaggle/working/division_scorer_weights.pt")
    print("Saved division_scorer_weights.pt")

    # 5. Threshold Tuning on Validation Set
    print("\nPreparing Validation set for Threshold Tuning...")
    val_prepared = []
    
    for sample in val_samples: 
        print(f"Scoring {sample.name}...")
        geff = bio_io.load_geff(sample.geff_path)
        
        import tracksdata as td
        gt_graph = td.graph.IndexedRXGraph.from_geff(sample.geff_path)
        if isinstance(gt_graph, tuple): gt_graph = gt_graph[0]

        tracking_result, raw_frames = generate_sample_data(detector, sample)
        
        vol_ram = FrameCache(sample.zarr_path)
        
        vs = dtt.prepare_validation_sample(
            tracking_result, gt_graph, mlp, vol_ram, bio_io.NATIVE_SCALE
        )
        val_prepared.append(vs)

    print("\nRunning Jaccard Threshold Sweep on Validation Set...")
    best_threshold, sweep_results = dtt.tune_division_threshold(
        val_prepared, 
        bio_io.NATIVE_SCALE, 
        thresholds=np.arange(0.0, 1.0, 0.05)
    )
    
    print(f"\n--- Best Division Threshold: {best_threshold:.2f} ---")
    for res in sweep_results:
        marker = "<-- BEST" if np.isclose(res.threshold, best_threshold) else ""
        print(f"Threshold: {res.threshold:.2f} | Jaccard: {res.division_jaccard:.4f} | Overall Score: {res.score:.4f} {marker}")

if __name__ == "__main__":
    main()