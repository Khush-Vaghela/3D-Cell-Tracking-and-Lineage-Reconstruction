# Biohub – Cell Tracking During Development

> **Kaggle Competition** | 3-D zebrafish cell detection, tracking & lineage reconstruction

| Metric | Score |
|--------|-------|
| 🏆 **Private Score** | **0.83411** |
| 📊 **Public Score** | **0.84102** |

### Kaggle Notebooks
| Notebook | Link |
|----------|------|
| 🧠 Training – Detector (Phase 1) | [training-detector](https://www.kaggle.com/code/khushvaghela/training-detector/notebook) |
| 🔬 Training – Division MLP (Phase 4.5) | [training-division-mlp](https://www.kaggle.com/code/khushvaghela/training-division-mlp) |
| 📤 Submission Generation | [submission-generation](https://www.kaggle.com/code/khushvaghela/submission-generation/notebook) |

---

## Competition Overview

The goal is to detect and track zebrafish cells through **3-D space and time** in fluorescence microscopy data, identifying cell divisions to reconstruct full developmental lineages. The evaluation metric is a combined score of:
- **Edge Jaccard** – how well cell-to-cell links are predicted across time
- **Division Jaccard** – how well cell division events are identified

A key challenge is that the ground truth is **sparse** (not every real cell is labeled in every frame), requiring special handling to avoid penalising the model for detecting unlabeled-but-real cells.

---

## Solution Overview

The solution is a **four-phase end-to-end pipeline**:

```
Raw 3-D Volume
     │
     ▼
┌─────────────────────────────────┐
│ PHASE 1: Detection              │  TemporalUNet3D + Detection Head
│  Gaussian heatmap per frame     │  Focal Loss with sparse-GT weighting
└────────────┬────────────────────┘
             │ peak detections (z, y, x, confidence)
             ▼
┌─────────────────────────────────┐
│ PHASE 2 & 3: Graph Building     │  tracksdata DistanceEdges
│  Multi-frame candidate graph    │  distance_threshold=25 µm, delta_t=1
└────────────┬────────────────────┘
             │ candidate graph
             ▼
┌─────────────────────────────────┐
│ PHASE 4: Tracking Solve         │  NearestNeighborsSolver (max_children=2)
│  Assign cells across time       │  Allows divisions natively
└────────────┬────────────────────┘
             │ solved tracks + division candidates
             ▼
┌─────────────────────────────────┐
│ PHASE 4.5: Division Validation  │  DivisionScorerMLP (9 features)
│  Filter false division events   │  threshold = 0.85
└────────────┬────────────────────┘
             │
             ▼
      submission.csv
```

---

## Phase 1 – Cell Detection

### Architecture: `Detector3D`

The detector is built on the **`TemporalUNet3D`** backbone from the official competition baseline repository, with a custom detection head added on top.

**Backbone – `TemporalUNet3D`:**
- Input shape: `(B, T, C_in=1, Z, Y, X)` — a sliding window of `T=2` consecutive frames
- Encoder channel widths: **`[32, 64, 128]`** (3 stages)
- Per-voxel **multi-head temporal attention** (4 heads) at each encoder stage, applied across the time dimension, so the model can see motion context while detecting
- Full-resolution temporal attention is **skipped** (`skip_fullres_temporal=True`) for ~3× speedup and ~30% memory saving with negligible quality loss
- **Gradient checkpointing** enabled to reduce activation memory during training
- MaxPool3d downsampling + trilinear upsampling with skip connections

**Detection Head:**
- A single `Conv3d(32 → 1, kernel=1)` + **Sigmoid** output
- Outputs a heatmap in `[0, 1]` per voxel per frame
- **Focal-bias initialization**: head bias set to `log((1-π)/π)` with `π=0.01`, forcing initial predictions towards 1% probability — prevents gradient explosions from the vast empty background

### Targets: Sparse-GT-Aware Gaussian Heatmaps

A critical design decision is the **sparse-GT-aware target scheme** (`targets.py`):

- GT labels are **point centroids only** and are **sparse** (not all real cells are labeled per frame). A naive dense-MSE loss would penalise the model for detecting unlabeled-but-real cells.
- Each labeled cell is rendered as an **anisotropic Gaussian bump** (σ = 2.0 µm, converted to per-axis voxel sigma given the native scale), combined via **elementwise max** (not sum) to avoid inflating dense regions.
- A **per-voxel loss weight map** is built alongside the heatmap: weight = 1.0 within 2 µm of any labeled cell centroid; weight = `background_weight` elsewhere.
- `background_weight` per frame is estimated as `(labeled_count / estimated_true_count)` — low where labeling is sparse (down-weights "unlabeled = empty" assumption), approaching 1.0 where labeling is dense.

### Loss: CenterNet-Style Modified Focal Loss

```
L = Σ [-log(p) · (1-p)^α · 50 · pos_mask]        ← positive peak loss
  + Σ [-log(1-p) · p^α · (1-target)^β · neg_mask]  ← background loss
```

- `α = 2.0`, `β = 4.0` (CenterNet defaults)
- Positive weight boost factor: **50×** — counteracts the extreme foreground/background imbalance
- Gradient clipping: `max_norm = 1.0`
- Normalised by the number of positive (peak) voxels

### Volume Preprocessing

- **Downsampling**: `DOWNSAMPLE = (1, 4, 4)` — no Z downsampling (already coarser at 1.625 µm/voxel vs 0.406 µm in Y/X), 4× in Y and X for speed
- Native voxel scale: `(1.625, 0.40625, 0.40625) µm/voxel`
- **Sliding window**: `WINDOW_SIZE = 2` consecutive frames fed together

### Inference

- Detection threshold: **0.30**
- NMS minimum distance: **2.5 µm** (tuned for σ = 2.0 µm U-Net)
- Peaks extracted and back-projected from downsampled space to native voxel coordinates

---

## Phase 2 & 3 – Multi-Frame Candidate Graph

**Module:** `tracksdata_pipeline.py`

Previous hand-rolled approaches (frame-by-frame Hungarian matching in Phase 2 + separate gap-closing repair in Phase 3) are replaced by a single **tracksdata `DistanceEdges`** pass:

```python
CandidateGraphConfig(
    distance_threshold_um = 25.0,
    delta_t = 1,          # candidate edges between adjacent frames only
    n_neighbors = 10,
    neighbors_per_frame = True
)
```

- Every detection across all frames is a graph node (with `confidence` attribute)
- Coordinates stored in **physical units (µm)** so distance thresholds are directly interpretable
- `DistanceEdges` adds candidate edges including multi-frame-gap ones in one native pass — gap closing is no longer a separate repair problem

---

## Phase 4 – Tracking Solve

**Module:** `tracksdata_pipeline.py`

Two solvers were evaluated:

| Solver | Description | Notes |
|--------|-------------|-------|
| `NearestNeighborsSolver(max_children=2)` | Fast greedy solve | **Champion — used in final submission** |
| `ILPSolver(node_weight=-10.0)` | Globally optimal flow ILP via SCIP | More expensive, ILP fallback |

The **Nearest-Neighbour solver** proved more robust empirically. Setting `max_children=2` naturally allows cell divisions (a parent can have two outgoing links), removing the need for a separate rule-based division candidate engine.

---

## Phase 4.5 – Division Validation

**Module:** `division_mlp.py`, `division_features.py`

After the solver runs, every node with out-degree 2 (a candidate division) is validated by the `DivisionScorerMLP`:

### Architecture: `DivisionScorerMLP`

```
9 hand-crafted features → Linear(9→32) → ReLU → Linear(32→16) → ReLU → Linear(16→1) → Sigmoid
```

### The 9 Division Features (`division_features.py`)

| Feature | Description |
|---------|-------------|
| `distance_to_daughter1` | Physical distance parent → daughter 1 (µm) |
| `distance_to_daughter2` | Physical distance parent → daughter 2 (µm) |
| `intensity_ratio` | (d1 + d2 intensity) / parent intensity |
| `symmetry_ratio` | d1 intensity / d2 intensity |
| `divergence_angle_deg` | Angle between the two daughter displacement vectors |
| `parent_confidence` | Detector heatmap score at parent |
| `daughter1_confidence` | Detector heatmap score at daughter 1 |
| `daughter2_confidence` | Detector heatmap score at daughter 2 |
| `association_cost` | Solver edge cost for the candidate division |

Local intensities are read **lazily from the Zarr volume** using a tiny bounding-box slice — no full-frame load needed.

### Training the Division MLP

- **Data generation**: runs the full Phase 1→4 pipeline over training samples; for each candidate division, both endpoints are matched to GT nodes. A candidate is labelled **positive** only if both match AND a GT edge exists between them. Unmatched predictions are **skipped** (not negatives) — sparse GT means an unmatched prediction could be a real-but-unlabeled cell.
- **Loss**: `BCEWithLogitsLoss` with `pos_weight = (n_neg/n_pos) × 2.0` (aggressive positive weighting to penalise missed real divisions)
- **Epochs**: 150, `lr = 1e-3` (Adam), 80/20 train/val split
- **Inference threshold**: **0.85** (tuned to maximise Division Jaccard)

---

## Repository Structure

```
archive/
├── README.md
├── training-detector.ipynb        ← Detector training (Phase 1)
├── training-division-mlp.ipynb    ← Division MLP training (Phase 4.5)
├── submission-generation.ipynb    ← Full inference pipeline + CSV generation
└── archive/
    ├── detector_model.py          ← Detector3D + DetectorConfig
    ├── temporal_unet.py           ← TemporalUNet3D backbone (from baseline)
    ├── targets.py                 ← Sparse-GT heatmap targets + Focal Loss
    ├── train_detector.py          ← Detector training step
    ├── train_phase1_full.py       ← Full Phase 1 training loop
    ├── preprocessing.py           ← Volume normalization & downsampling
    ├── detect.py                  ← Heatmap → NMS peaks
    ├── data_io.py                 ← Zarr volume & GEFF ground truth loading
    ├── pipeline_config.py         ← Global constants (DOWNSAMPLE, SIGMA_UM, WINDOW_SIZE)
    ├── tracksdata_pipeline.py     ← Phase 2/3/4 graph + solver (tracksdata-native)
    ├── division_mlp.py            ← DivisionScorerMLP + training loop
    ├── division_features.py       ← 9-feature division feature extraction
    ├── division_validation.py     ← Phase 4.5 post-solver validation
    ├── division_candidates.py     ← Division candidate generation (legacy rule-based)
    ├── learned_association.py     ← EdgeScorerMLP (Phase 2 learned weights, optional)
    ├── association.py             ← Frame-by-frame Hungarian matching (legacy)
    ├── gap_closing.py             ← Gap closing (legacy, superseded by tracksdata)
    ├── tracker.py                 ← TrackingResult dataclass
    ├── node_validation.py         ← GT node matching utilities
    ├── division_threshold_tuning.py← Threshold search utilities
    └── test_*.py                  ← Unit tests for each module
```

---

## Key Design Decisions

1. **Sparse-GT loss weighting**: The single most important change over a naive baseline. Down-weighting the background loss proportionally to estimated label completeness prevents the model from being punished for detecting real-but-unlabeled cells — avoiding the 0.57–0.66 recall ceiling observed with standard dense-MSE approaches.

2. **CenterNet Focal Loss with 50× positive boost**: Addresses the extreme foreground/background imbalance (cells occupy <1% of voxels). Combined with focal-bias initialization on the detection head.

3. **tracksdata-native graph solver**: Replaces three separate custom modules (association + gap_closing + division_candidates) with a single `DistanceEdges` + `NearestNeighborsSolver` pass. Gap closing falls out naturally as longer-range edges in the candidate graph.

4. **Division MLP post-filtering**: Runs *after* the solver (not before) so the solver's own greedy decisions can be audited. The 9 biologically-motivated features capture division geometry and intensity signatures more reliably than raw detector scores alone.

5. **Nearest-Neighbour over ILP**: Despite ILP being globally optimal in theory, the NN solver proved more robust and faster on this dataset, avoiding the ILP's tendency to return trivially empty solutions without careful weight tuning.

---

## Submission Generation

**Notebook:** `submission-generation.ipynb`

The full pipeline runs on the Kaggle GPU (NVIDIA Tesla T4). The submission notebook:
1. Reconstructs `.pt` weight files from unzipped Kaggle datasets (PyTorch ZIP format fix)
2. Loads `Detector3D` (from `detector_weights_best.pt`) and `DivisionScorerMLP` (from `division_scorer_weights.pt`)
3. Runs the 4-phase pipeline on all test samples
4. Formats output as the required Kaggle schema: `dataset, row_type, node_id, t, z, y, x, source_id, target_id`

### Submission Rows by Sample

| Sample | Frames | Submission Rows |
|--------|--------|-----------------|
| `44b6_0113de3b` | 100 | 50,195 |
| `44b6_0b24845f` | 100 | 64,694 |
| `6bba_05b6850b` | 100 | 12,006 |
| `6bba_05db0fb1` | 100 | 136,285 |
| **Total** | — | **263,180** |

Total notebook runtime: **~14 minutes** on T4 GPU.

---

## Dependencies

- Python ≥ 3.12
- PyTorch
- `tracksdata` (graph-based cell tracking library)
- `zarr` (lazy volumetric data loading)
- `polars` (submission formatting)
- `numpy`, `pandas`, `scipy`

---

## Results

| Leaderboard | Score |
|-------------|-------|
| **Private** | **0.83411** |
| **Public** | **0.84102** |

The close agreement between public and private scores suggests the solution generalises well and is not over-fitted to the public leaderboard samples.
