# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **orthogonal uncertainty decomposition** for video object detection and tracking. The project decomposes total uncertainty into **aleatoric** (data-inherent) and **epistemic** (model-inherent) components, with **conformal prediction** for distribution-free coverage guarantees.

**Key Innovation**: Triple-S Framework (Spectral, Spatial, Statistical) achieving orthogonal epistemic uncertainty across all 7 MOT17 sequences with mean correlation |r| = 0.048.

## Environment Setup

**Python Version**: 3.8.19
**Conda Environment**: `env_py311` (located at `/home/divake/miniconda3/envs/env_py311/`)

### Activate Environment
```bash
conda activate env_py311
# Or use direct Python path:
/home/divake/miniconda3/envs/env_py311/bin/python
```

### Core Dependencies
```bash
pip install numpy>=1.20.0 scipy>=1.7.0 matplotlib>=3.4.0 scikit-learn>=0.24.0 pyyaml>=5.4.0
pip install ultralytics  # For YOLO models
```

## Repository Structure

```
temporal_uncertainty/
├── conformal_tracking/          # Main uncertainty framework (COMPLETE)
│   ├── src/uncertainty/         # Core uncertainty modules
│   │   ├── mahalanobis.py      # Aleatoric uncertainty (KNN + Mahalanobis)
│   │   ├── epistemic_spectral.py    # Method 1: Spectral collapse
│   │   ├── epistemic_repulsive.py   # Method 2: Repulsive void detection
│   │   ├── epistemic_gradient.py    # Method 3: Inter-layer divergence
│   │   ├── epistemic_combined.py    # Triple-S combined framework
│   │   └── conformal_calibration.py # Conformal prediction
│   ├── data_loaders/
│   │   └── mot17_loader.py     # MOT17 cache loader (multi-layer features)
│   ├── experiments/            # Experiment runners and visualization
│   │   ├── run_epistemic_mot17.py        # Single sequence epistemic
│   │   ├── run_conformal_mot17.py        # Single sequence conformal
│   │   ├── run_conformal_all_sequences.py # Aggregate conformal
│   │   └── paper_*.py                     # Paper figure generation
│   ├── results/                # Experiment outputs (77 plots across 7 sequences)
│   └── config/                 # YAML configuration files
├── data/
│   ├── MOT17/                  # 7 sequences (5,316 frames) - COMPLETE
│   ├── MOT20/                  # 4 sequences (crowded scenes) - EXTRACTED
│   └── DanceTrack/             # 65 sequences (uniform appearance) - EXTRACTED
├── models/
│   ├── yolov8*.pt             # 5 YOLO variants (n, s, m, l, x)
│   ├── rtdetr/                # RT-DETR transformer model
│   ├── yolo-world/            # YOLO-World open-vocabulary
│   └── dino/                  # DINO state-of-the-art transformer
└── github_repos/              # 18 reference implementations for uncertainty methods
```

## Common Commands

### Run Epistemic Uncertainty Experiments

**Single sequence** (recommended for testing):
```bash
cd /ssd_4TB/divake/temporal_uncertainty/conformal_tracking
/home/divake/miniconda3/envs/env_py311/bin/python experiments/run_epistemic_mot17.py MOT17-11-FRCNN
```

**All 7 MOT17 sequences** (parallel execution):
```bash
cd /ssd_4TB/divake/temporal_uncertainty/conformal_tracking
for seq in MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN; do
    /home/divake/miniconda3/envs/env_py311/bin/python experiments/run_epistemic_mot17.py $seq &
done
```

### Run Conformal Prediction Experiments

**Single sequence**:
```bash
/home/divake/miniconda3/envs/env_py311/bin/python experiments/run_conformal_mot17.py MOT17-11-FRCNN
```

**All sequences with aggregation**:
```bash
/home/divake/miniconda3/envs/env_py311/bin/python experiments/run_conformal_all_sequences.py
```

**Aggregate existing results**:
```bash
/home/divake/miniconda3/envs/env_py311/bin/python experiments/aggregate_conformal_results.py
```

### View Results

**Check experiment results**:
```bash
cat results/epistemic_mot17_11/results.json
cat results/conformal_mot17_11/conformal_results.json
cat results/conformal_summary/aggregated_results.json
```

**View plots**:
```bash
ls results/epistemic_mot17_11/plots/
ls results/conformal_summary/plots/
```

### Generate Paper Figures

```bash
/home/divake/miniconda3/envs/env_py311/bin/python experiments/paper_figure1_decomposition.py
/home/divake/miniconda3/envs/env_py311/bin/python experiments/paper_figure2_iou_analysis.py
/home/divake/miniconda3/envs/env_py311/bin/python experiments/paper_figure3_conformal.py
```

## Architecture Deep Dive

### Uncertainty Decomposition Pipeline

The system decomposes total uncertainty into orthogonal components:

**Total Uncertainty** = Aleatoric + Epistemic
```
σ²_total(x) = σ²_aleatoric(x) + σ²_epistemic(x)
```

**Key Constraint**: Aleatoric ⊥ Epistemic (|correlation| < 0.3)

### 1. Aleatoric Uncertainty (Data Uncertainty)

**Module**: `src/uncertainty/mahalanobis.py`

**Method**: Mahalanobis distance-based KNN with Gaussian modeling

**Algorithm**:
1. Fit multivariate Gaussian to calibration features: μ, Σ
2. For test sample with features x:
   - Compute Mahalanobis distance: M(x) = √((x-μ)ᵀ Σ⁻¹ (x-μ))
   - Find K nearest neighbors in calibration set
   - Weight by distance: w_i = exp(-d_i)
   - Aggregate: σ²_aleatoric = Σ w_i · error_i²

**Key Parameters**:
- `k_neighbors`: 15 (default)
- `reg_lambda`: 1e-4 (covariance regularization)

### 2. Epistemic Uncertainty (Model Uncertainty)

**Module**: `src/uncertainty/epistemic_combined.py`

**Triple-S Framework** combines three complementary methods:

#### Method 1: Spectral Collapse Detection
**Module**: `src/uncertainty/epistemic_spectral.py`

**Principle**: Detects feature manifold degeneracy via eigenspectrum analysis

**Algorithm**:
1. Find k=50 nearest neighbors
2. Compute local covariance matrix
3. Eigendecomposition → λ₁, λ₂, ..., λ_D
4. Compute effective rank: R_eff = exp(entropy(λ_normalized))
5. Epistemic = 1 - (R_eff / D)

**Key Finding**: YOLO uses only 4-7% of 256D feature space (R_eff: 10-17)

#### Method 2: Repulsive Void Detection
**Module**: `src/uncertainty/epistemic_repulsive.py`

**Principle**: Physics-inspired Coulomb-like forces detect voids in feature space

**Algorithm**:
1. Find k=100 nearest neighbors
2. For each neighbor i: F_i = exp(-d_i/T) / (d_i² + ε)
3. Compute net force vector: F_net = Σ F_i × direction_i
4. Epistemic = ||F_net||

**Parameters**: T=1.0 (temperature), ε=1e-6 (cutoff)

#### Method 3: Inter-Layer Gradient Divergence
**Module**: `src/uncertainty/epistemic_gradient.py`

**Principle**: Measures feature evolution instability across YOLO layers

**Algorithm**:
1. Extract features from layers [4, 9, 15, 21]
2. For each consecutive pair: compute cosine similarity
3. Divergence = 1 - cosine_similarity
4. Epistemic = mean(divergences)

#### Adaptive Weight Optimization

**Objective**: Minimize correlation with aleatoric uncertainty

```python
minimize: |correlation(w₁·Spectral + w₂·Repulsive + w₃·Gradient, aleatoric)|
subject to: w₁ + w₂ + w₃ = 1, wᵢ ≥ 0
```

**Method**: SLSQP (Sequential Least Squares Programming)

**Result**: Sequence-specific weight selection achieving |r| < 0.3 for all sequences

### 3. Conformal Prediction

**Module**: `src/uncertainty/conformal_calibration.py`

**Novel Approach**: Combines uncertainties BEFORE calibration (not after)

**Algorithm**:
1. **Stage 1**: Combine uncertainties
   - σ_combined = √(σ²_aleatoric + σ²_epistemic)

2. **Stage 2**: Compute nonconformity scores
   - S_i = |y_i - ŷ_i| / σ_combined(x_i)
   - Where y = IoU, ŷ = confidence (proxy)

3. **Stage 3**: Global quantile
   - q̂ = Quantile_{(1-α)}(S₁, ..., S_n)

4. **Stage 4**: Local adaptation (optional)
   - Decision tree partitions feature space
   - Separate quantile q̂_k per stratum

**Coverage Guarantee**: P(Y ∈ I(X)) ≥ 1-α

**Results**: 90.3% coverage (target: 90%) with mean width 0.377

### Data Loading Architecture

**Module**: `data_loaders/mot17_loader.py`

**Cache System**: Pre-computed features from YOLO layers stored in NPZ format

**Cache Location**: `/ssd_4TB/divake/temporal_uncertainty/cache/mot17/yolov8n/`

**Cache Structure**:
```python
{
    'features_layer_4': [N, 64],    # Early features
    'features_layer_9': [N, 256],   # Mid-level features (primary)
    'features_layer_15': [N, 64],   # High-level features
    'features_layer_21': [N, 256],  # Pre-classification features
    'boxes': [N, 4],                # Bounding boxes (x1,y1,x2,y2)
    'confidence': [N],              # YOLO confidence scores
    'ious': [N],                    # IoU with ground truth
    'center_errors': [N],           # Center distance errors (pixels)
    'frame_ids': [N]                # Frame indices
}
```

**Load Data**:
```python
from mot17_loader import MOT17DataLoader

loader = MOT17DataLoader(
    cache_dir='/ssd_4TB/divake/temporal_uncertainty/cache/mot17/yolov8n',
    sequence='MOT17-11-FRCNN',
    feature_layer=9  # Primary layer
)

data = loader.load_sequence()
# Returns: features, ious, center_errors, confidence, frame_ids
```

### Experiment Configuration

**Location**: `conformal_tracking/config/`

**Configuration Files**:
- `epistemic_mot17.yaml`: Epistemic uncertainty parameters
- `conformal_mot17.yaml`: Conformal prediction parameters

**Key Settings**:
```yaml
# Aleatoric
k_neighbors: 15
reg_lambda: 0.0001

# Epistemic
k_neighbors_spectral: 50
k_neighbors_repulsive: 100
temperature: 1.0
weights: 'optimize'  # or 'equal' or [w1, w2, w3]

# Conformal
alpha: 0.1  # 90% coverage
method: 'combined_local'  # 'vanilla', 'combined_global', 'combined_local'
```

## Model Architecture

### Detection Models Available

**YOLO Family** (CNN-based):
- `yolov8n.pt`: 6.3 MB (3.2M params) - Nano
- `yolov8s.pt`: 22 MB (11.2M params) - Small
- `yolov8m.pt`: 50 MB (25.9M params) - Medium
- `yolov8l.pt`: 84 MB (43.7M params) - Large
- `yolov8x.pt`: 131 MB (68.2M params) - Extra Large

**Transformer Models**:
- RT-DETR (228 MB): Real-time transformer detector
- DINO (9.7 GB): State-of-the-art transformer detector

**Open-Vocabulary**:
- YOLO-World (82 MB): CLIP-based zero-shot detector

### Loading Models

```python
from ultralytics import YOLO

# Load YOLO
model = YOLO('models/yolov8n.pt')

# Extract features
results = model(image, verbose=False)
features = model.model.model[9].forward(x)  # Layer 9 features
```

## Key Results and Status

### Epistemic Uncertainty (COMPLETE ✅)

**All 7 MOT17 Sequences - 100% Success**:

| Sequence | Weights (S/R/G) | Orthogonality | Status |
|----------|-----------------|---------------|--------|
| MOT17-02 | 0.00/0.01/0.99 | 0.036 | ✅ |
| MOT17-04 | 0.84/0.00/0.16 | 0.049 | ✅ |
| MOT17-05 | 0.50/0.27/0.23 | 0.031 | ✅ |
| MOT17-09 | 0.11/0.00/0.89 | 0.081 | ✅ |
| MOT17-10 | 0.00/0.05/0.95 | 0.053 | ✅ |
| MOT17-11 | 0.00/0.00/1.00 | 0.007 | ✅ |
| MOT17-13 | 0.73/0.00/0.27 | 0.025 | ✅ |

**Mean Orthogonality**: 0.048 (Target: <0.3)

### Conformal Prediction (COMPLETE ✅)

**21,324 test samples across 7 sequences**:

| Method | Coverage | Mean Width |
|--------|----------|------------|
| Vanilla | 89.8% | 0.336 |
| Combined (Global) | 91.2% | 0.404 |
| Combined (Local) | 90.3% | 0.377 |

### Three Distinct Uncertainty Strategies

**Gradient-Dominant** (MOT17-02, 09, 10, 11): 89-100% gradient weight
- Uncertainty from layer-wise feature instability

**Spectral-Dominant** (MOT17-04, 13): 73-84% spectral weight
- Uncertainty from feature manifold collapse

**Balanced** (MOT17-05): 50/27/23 (S/R/G)
- Multiple uncertainty sources present

## Datasets

### MOT17 (Primary Dataset - COMPLETE)
- **Sequences**: 7 training sequences
- **Frames**: 5,316 total frames
- **Detections**: 26,756 matched detections
- **Location**: `/ssd_4TB/divake/temporal_uncertainty/data/MOT17/train/`
- **Status**: Fully processed with cached features

### MOT20 (Expansion Dataset - EXTRACTED)
- **Sequences**: 4 training sequences (MOT20-01, 02, 03, 05)
- **Characteristics**: High-density crowded scenes
- **Location**: `/ssd_4TB/divake/temporal_uncertainty/data/MOT20/MOT20/train/`
- **Status**: Extracted, ready for processing

### DanceTrack (Expansion Dataset - EXTRACTED)
- **Sequences**: 65 total (40 train + 25 val)
- **Characteristics**: Uniform appearance, diverse motion
- **Location**: `/ssd_4TB/divake/temporal_uncertainty/data/DanceTrack/`
- **Status**: Extracted, ready for processing

## Important Implementation Details

### Feature Extraction Layers

YOLO model has multiple layers. We extract features from:
- **Layer 4**: Early features (64D) - edges, textures
- **Layer 9**: Mid-level features (256D) - PRIMARY LAYER for uncertainty
- **Layer 15**: High-level features (64D) - object parts
- **Layer 21**: Pre-classification features (256D) - semantic features

**Default**: Use Layer 9 for single-layer experiments

### Calibration/Test Split

- **Split Ratio**: 50/50 calibration/test
- **Method**: Random split with fixed seed (seed=42)
- **Per-sequence**: Each sequence split independently
- **No temporal leakage**: Detections from same frame can be in both sets (they're independent samples)

### Why No Training/Test Split for New Datasets?

**Key Insight**: Since models are **pre-trained** and **not fine-tuned** on these datasets:
- Traditional train/val/test split is unnecessary
- All sequences can be treated as calibration/validation for uncertainty
- We split detections into calibration (fit uncertainty) and validation (test coverage)
- This applies to MOT20 and DanceTrack equally

### Multi-Layer Feature Loading

For epistemic gradient divergence, load all layers:

```python
loader = MOT17DataLoader(
    cache_dir=cache_dir,
    sequence=sequence,
    feature_layer=9  # Primary layer
)

# Load all layers for gradient method
features_dict = loader.load_all_layers()
# Returns: {4: array, 9: array, 15: array, 21: array}
```

### Handling Missing Ground Truth

Some detections lack ground truth matches (IoU = 0). These are typically:
- False positives
- Occluded objects
- Out-of-frame detections

**Current approach**: Filter by `iou_threshold=0.3` and `conf_threshold=0.3`

## Debugging and Common Issues

### Issue: "Cache file not found"
**Solution**: Verify cache location and sequence name:
```bash
ls /ssd_4TB/divake/temporal_uncertainty/cache/mot17/yolov8n/
# Should show: MOT17-02-FRCNN.npz, MOT17-04-FRCNN.npz, etc.
```

### Issue: "Singular covariance matrix"
**Solution**: Increase regularization parameter:
```python
mahalanobis = MahalanobisUncertainty(reg_lambda=1e-3)  # Increase from 1e-4
```

### Issue: "Orthogonality constraint violated"
**Solution**: This shouldn't happen with the optimizer, but check:
- Sufficient calibration samples (N > 1000)
- Feature variance is not too low
- All three epistemic methods are enabled

### Issue: "Conformal coverage too low/high"
**Solution**: Adjust alpha parameter:
```python
# For 90% coverage: alpha=0.1
# For 95% coverage: alpha=0.05
calibrator = CombinedConformalCalibrator(alpha=0.1)
```

## Code Style and Conventions

- **Docstrings**: All modules have comprehensive docstrings
- **Type hints**: Used throughout core modules
- **Naming**:
  - `snake_case` for functions/variables
  - `PascalCase` for classes
  - `_private` for internal methods
- **Imports**: Grouped by stdlib, third-party, local
- **Constants**: UPPERCASE with underscores

## Publication Status

**Framework**: Complete and validated
**Documentation**: Extensive (77 plots + technical reports)
**Status**: Ready for CVPR submission

**Key Contributions**:
1. First orthogonal uncertainty decomposition in object detection
2. Novel Triple-S framework with adaptive weights
3. Combined conformal prediction with uncertainty fusion
4. Validation on 26,756 detections across 7 diverse sequences

## References

**Documentation Files** (in `conformal_tracking/`):
- `README.md`: Quick start and results summary
- `PROJECT_GOAL_AND_METHODS.md`: Comprehensive methodology explanation
- `EPISTEMIC_FINDINGS.md`: Technical results and mathematics
- `EPISTEMIC_FINAL_SUMMARY.md`: Executive summary
- `CACHE_ANALYSIS_SUMMARY.md`: Cache structure and statistics

**External Resources**:
- MOT Challenge: https://motchallenge.net/
- YOLO: https://github.com/ultralytics/ultralytics
- Conformal Prediction: https://arxiv.org/abs/2107.07511
