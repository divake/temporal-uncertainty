# Conformal Uncertainty for MOT17 Tracking
## V1 Implementation Plan (Week 1)

**Goal**: Implement Enhanced CACD V1 on MOT17 for CVPR 2025

**Timeline**: 5-7 days for V1 core implementation

**Environment**: Use conda environment `env_py311` for all experiments and installations

---

## What We Have (Already Available)

### ✅ Data & Infrastructure
- **MOT17 Dataset**: `/ssd_4TB/divake/temporal_uncertainty/data/MOT17/train/`
  - 7 sequences (02, 04, 05, 09, 10, 11, 13)
  - Ground truth with visibility field (occlusion labels!)

- **Comprehensive Metadata**: `/ssd_4TB/divake/temporal_uncertainty/metadata/`
  - Per-sequence statistics (track lifetimes, occlusions, crowding)
  - Hero tracks identified (long stable, occlusion-heavy, high motion)
  - Recommended test frames
  - Visualizations (Gantt charts, heatmaps, crowding plots)

- **YOLO Models**: `/ssd_4TB/divake/temporal_uncertainty/models/`
  - yolov8n.pt (6.3M) - Nano (fastest) - **Use this for V1**
  - yolov8s.pt (22M) - Small
  - yolov8m.pt (50M) - Medium
  - yolov8l.pt (84M) - Large
  - yolov8x.pt (131M) - Extra Large

### ✅ Prior Work (Proven on UCI)
- **Mahalanobis KNN** for aleatoric (~10% improvement)
- **Multi-source epistemic ensemble** (35% improvement in OOD detection)
- Code in: `/ssd_4TB/divake/temporal_uncertainty/cacd/enhanced_cacd/src/`

### ✅ Dependencies
- All required Python packages already installed in `env_py311`
- If anything new is needed, install via: `conda activate env_py311 && pip install <package>`

---

## V1 Architecture

### Philosophy
> "Prove the core conformal calibration idea works with minimum complexity"

### What We're Doing
**Adding calibrated uncertainty intervals to tracker predictions**

```python
# Standard tracker output:
bbox_predicted = [x, y, w, h]

# Our V1 enhancement adds:
uncertainty_interval = [
    x ± q̂ × ξ × σ_total,
    y ± q̂ × ξ × σ_total,
    w ± q̂ × ξ × σ_total,  # optional
    h ± q̂ × ξ × σ_total   # optional
]

# Interpretation: "90% confident the true position is within this interval"
```

### Ground Truth Target (y)
**2D position error in pixels:**

```python
# For each detection/track:
y = ||center_predicted - center_gt||_2  # Euclidean distance in pixels

# Where:
center_predicted = [x_pred + w_pred/2, y_pred + h_pred/2]
center_gt = [x_gt + w_gt/2, y_gt + h_gt/2]
```

### Components
1. **Features**: YOLO backbone (256-dim from YOLOv8n)
2. **Aleatoric**: Euclidean KNN (local variance)
3. **Epistemic**: Inverse density via KDE
4. **Calibration**: Combined score conformal (NOVEL - main contribution!)
5. **Local Scaling**: Decision tree partitioning
6. **Temporal**: Kalman filter with structured process noise (NOVEL!)

### Integration Strategy (Post-processing)
```python
# Step 1: Run ByteTrack normally
tracks = ByteTrack(video_frames)

# Step 2: For each track bbox, add uncertainty
for track in tracks:
    features = extract_yolo_features(track.bbox, frame)
    σ_alea = compute_aleatoric(features)
    σ_epis = compute_epistemic(features)
    σ_total = √(σ²_alea + σ²_epis)

    # Add calibrated uncertainty
    track.uncertainty = q̂ × ξ_region × σ_total
    track.interval = track.bbox ± track.uncertainty
```

---

## Data Split Strategy (Temporal Splits)

**IMPORTANT: Keep temporal order! Don't randomly sample frames.**

### Per-Sequence Split (Example: MOT17-09 with 525 frames)
```python
frames_1_175 = "calibration"  # 33% for calibration
frames_176_350 = "validation"  # 33% for hyperparam tuning
frames_351_525 = "test"        # 34% for evaluation
```

### Sequence-Level Split
**Calibration Sequences** (cleaner, for fitting models):
- MOT17-02-FRCNN
- MOT17-09-FRCNN

**Test Sequences** (more challenging):
- MOT17-04-FRCNN
- MOT17-11-FRCNN
- MOT17-13-FRCNN

**Validation Sequences** (hyperparameter tuning):
- MOT17-05-FRCNN
- MOT17-10-FRCNN

---

## Day-by-Day Implementation

### Day 1: ByteTrack Setup + Ground Truth Matching
**Goal**: Get tracker predictions and match them to ground truth

**Tasks**:
- [ ] Install/setup ByteTrack (or use simple IoU tracker)
- [ ] Run tracker on calibration sequences (MOT17-02, MOT17-09)
- [ ] Match tracker predictions to ground truth boxes
- [ ] Compute position errors: `y = ||center_pred - center_gt||`
- [ ] Save matched data: `[frame_id, track_id, bbox_pred, bbox_gt, error, features]`

**Key Code**:
```python
def match_tracker_to_gt(tracker_boxes, gt_boxes, iou_threshold=0.5):
    """
    Match tracker predictions to ground truth via Hungarian matching.

    Returns:
        matched_data: List of (bbox_pred, bbox_gt, center_error)
    """
    # Use IoU or center distance for matching
    # Compute center error for matched pairs

def compute_center_error(bbox_pred, bbox_gt):
    """
    Returns: ||center_pred - center_gt||_2 in pixels
    """
    cx_pred = bbox_pred[0] + bbox_pred[2]/2
    cy_pred = bbox_pred[1] + bbox_pred[3]/2
    cx_gt = bbox_gt[0] + bbox_gt[2]/2
    cy_gt = bbox_gt[1] + bbox_gt[3]/2
    return np.sqrt((cx_pred - cx_gt)**2 + (cy_pred - cy_gt)**2)
```

**Expected Output**:
```
data/
├── cal_matched_tracks.pkl  # Calibration data
└── test_matched_tracks.pkl # Test data
```

---

### Day 2: YOLO Feature Extraction
**Goal**: Extract YOLO features for each matched bbox

**Tasks**:
- [ ] Load YOLOv8n model
- [ ] For each bbox in matched data, extract features from last backbone layer
- [ ] Save features alongside matched data
- [ ] Validate: feature dim = 256, no NaN/Inf

**Key Code**:
```python
from ultralytics import YOLO

def extract_yolo_features(bbox, frame, model):
    """
    Extract 256-dim features from YOLOv8n backbone.

    Args:
        bbox: [x, y, w, h]
        frame: RGB image [H, W, 3]
        model: YOLOv8n model

    Returns:
        features: [256] feature vector
    """
    # Crop bbox region
    crop = frame[y:y+h, x:x+w]

    # Resize to 640x640 (YOLO input)
    crop_resized = resize(crop, (640, 640))

    # Forward through backbone (first 10 layers)
    with torch.no_grad():
        features = model.model.model[:10](crop_resized)
        pooled = adaptive_avg_pool2d(features, (1, 1))

    return pooled.squeeze().cpu().numpy()
```

**Expected Output**:
```
results/v1/features/
├── cal_features.npz   # [N_cal, 256]
└── test_features.npz  # [N_test, 256]
```

**Validation**:
- Feature shape: [N, 256]
- No NaN/Inf values
- Feature distribution looks reasonable (mean ≈ 0.5, std ≈ 0.2)

---

### Day 3: Uncertainty Estimation + Calibration
**Goal**: Compute aleatoric + epistemic uncertainties and calibrate

**Tasks**:
- [ ] Implement Euclidean KNN for aleatoric uncertainty
- [ ] Implement KDE for epistemic uncertainty
- [ ] Compute combined uncertainty: σ_total = √(σ²_alea + σ²_epis)
- [ ] Compute conformalized scores: α̃ = |y - ŷ| / σ_total
- [ ] Find quantile: q̂ = Quantile_{0.9}(α̃_cal)
- [ ] Validate coverage on test set

**Key Equations**:
```python
# 1. Aleatoric (KNN local variance)
knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
knn.fit(features_cal)
distances, indices = knn.kneighbors(features_test)
sigma_alea = np.std(errors_cal[indices], axis=1)

# 2. Epistemic (Inverse density)
from scipy.stats import gaussian_kde
kde = gaussian_kde(features_cal.T)
density = kde.evaluate(features_test.T)
sigma_epis = (np.max(density) - density) / (density + 1e-6)

# 3. Combined uncertainty
sigma_total = np.sqrt(sigma_alea**2 + sigma_epis**2)

# 4. Conformal calibration
scores_cal = errors_cal / sigma_total_cal
q_hat = np.quantile(scores_cal, 0.9 * (1 + 1/len(scores_cal)))

# 5. Prediction intervals on test set
error_bound_test = q_hat * sigma_total_test
coverage = np.mean(errors_test <= error_bound_test)
```

**Success Criteria**:
- Coverage ≥ 88% on test set
- Orthogonality: |corr(σ_alea, σ_epis)| < 0.3
- Interval width: Mean(error_bound) < vanilla baseline

**Expected Output**:
```
results/v1/calibration/
├── coverage_plot.png
├── uncertainty_vs_error.png
├── orthogonality_plot.png
└── calibration_stats.json
```

---

### Day 4: Local Scaling
**Goal**: Improve calibration with region-specific scaling

**Tasks**:
- [ ] Build state features: [x, y, w, h, visibility, crowding, frame_num]
- [ ] Fit decision tree to predict scores
- [ ] Compute per-leaf scaling factors: ξ_k = std(errors_k) / mean(σ_total_k)
- [ ] Apply two-stage calibration: error_bound = q̂ × ξ_k × σ_total
- [ ] Validate coverage maintained ≥ 88%

**Key Code**:
```python
from sklearn.tree import DecisionTreeRegressor

# Build state features (can use metadata for crowding/visibility)
state_features_cal = np.column_stack([
    bboxes_cal[:, :4],          # [x, y, w, h]
    visibility_cal,              # From GT
    crowding_cal,                # From metadata
    frame_nums_cal / max_frame   # Normalized frame number
])

# Fit decision tree
tree = DecisionTreeRegressor(
    max_depth=5,              # Shallow to prevent overfitting
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
tree.fit(state_features_cal, scores_cal)

# Get leaf IDs
leaf_ids_cal = tree.apply(state_features_cal)
leaf_ids_test = tree.apply(state_features_test)

# Compute scaling factors per leaf
xi = {}
for leaf_id in np.unique(leaf_ids_cal):
    mask = (leaf_ids_cal == leaf_id)
    xi[leaf_id] = np.std(errors_cal[mask]) / np.mean(sigma_total_cal[mask])

# Apply on test set
error_bound_scaled = []
for i, leaf_id in enumerate(leaf_ids_test):
    error_bound_scaled.append(q_hat * xi[leaf_id] * sigma_total_test[i])
```

**Success Criteria**:
- Overall coverage ≥ 88% maintained
- Interval width reduced in "easy" regions (low crowding, high visibility)
- Interval width increased in "hard" regions (high crowding, occlusions)

---

### Day 5-7: Temporal Propagation + Full Integration
**Goal**: Propagate uncertainty across frames using Kalman filter

**Tasks**:
- [ ] Implement Kalman filter with state [x, y, w, h, vx, vy, vw, vh]
- [ ] Use structured process noise: Q = diag(σ²_alea I₄, σ²_epis I₄)
- [ ] Propagate uncertainty when detections missing (occlusions)
- [ ] Update uncertainty when detections reappear
- [ ] Run full pipeline on all test sequences
- [ ] Compute MOTA, IDF1, ID switches

**Key Code**:
```python
from filterpy.kalman import KalmanFilter

# Initialize Kalman filter for each track
kf = KalmanFilter(dim_x=8, dim_z=4)

# State transition (constant velocity)
dt = 1.0
kf.F = np.eye(8)
kf.F[0:4, 4:8] = np.eye(4) * dt

# Measurement matrix (observe position only)
kf.H = np.zeros((4, 8))
kf.H[0:4, 0:4] = np.eye(4)

# Process noise (structured)
Q_alea = sigma_alea**2 * np.eye(4)
Q_epis = sigma_epis**2 * np.eye(4)
kf.Q = scipy.linalg.block_diag(Q_alea, Q_epis)

# Measurement noise (from calibrated uncertainty)
kf.R = (q_hat * sigma_total)**2 * np.eye(4)

# Predict step (when detection missing)
kf.predict()

# Update step (when detection appears)
kf.update(detection)

# Track uncertainty = trace(P[:4, :4])
track_uncertainty = np.trace(kf.P[:4, :4])
```

**Success Criteria**:
- Uncertainty increases during occlusions
- Uncertainty decreases when track reacquired
- MOTA ≥ baseline ByteTrack
- Fewer ID switches than baseline

---

## Evaluation Metrics

### Uncertainty Quality
1. **Coverage**: P(error ∈ [0, error_bound]) ≥ 90% (should be ~90%)
2. **Interval Width**: Mean(error_bound) (narrower is better, given coverage)
3. **Orthogonality**: |corr(σ_alea, σ_epis)| (should be < 0.3)
4. **Aleatoric-Error Correlation**: corr(σ_alea, error) (higher is better)
5. **Epistemic-OOD Correlation**: corr(σ_epis, ood_proxy) (higher is better)

### Tracking Quality (if integrated with ByteTrack)
1. **MOTA** (Multi-Object Tracking Accuracy)
2. **IDF1** (ID F1 Score)
3. **ID Switches** (lower is better)
4. **FPS** (frames per second)

### Baseline Comparisons
- **Vanilla conformal**: Constant width intervals (no uncertainty decomposition)
- **No calibration**: Use raw σ_total without conformal quantile
- **No local scaling**: Use global q̂ only

---

## Expected Results (V1)

| Metric | Baseline | V1 Expected | Improvement |
|--------|----------|-------------|-------------|
| Coverage | 90.0% | 90.5% | Maintained ✓ |
| Interval Width (pixels) | 25.0 | 21.0 | **-16%** ✓ |
| Alea-Error Corr | 0.30 | 0.35 | +17% |
| Epis-OOD Corr | 0.05 | 0.15 | **+3×** ✓ |
| Orthogonality | - | 0.25 | Good (< 0.3) ✓ |

**Key Wins**:
- 16% narrower intervals while maintaining coverage
- 3× better epistemic OOD detection
- Interpretable position uncertainty in pixels

---

## Success Criteria for V1

### Must-Have (Week 1 End)
- [ ] Coverage ≥ 88% on test set
- [ ] Intervals narrower than vanilla conformal
- [ ] |ρ(alea, epis)| < 0.3
- [ ] Code runs end-to-end on at least 1 sequence
- [ ] Visualization plots showing uncertainty vs error

### Nice-to-Have
- [ ] All 7 sequences processed
- [ ] Full ByteTrack integration with MOTA results
- [ ] Comprehensive ablation (with/without local scaling)

**Decision Point**:
- **IF** Must-Haves met → Proceed to V2 (Week 2)
- **IF** Must-Haves not met → Debug V1, skip V2/V3

---

## File Structure

```
conformal_tracking/
├── PROJECT_PLAN.md                 # This file
├── README.md                       # User guide
│
├── src/
│   ├── __init__.py
│   ├── tracker_utils.py            # ByteTrack wrapper + GT matching
│   ├── feature_extraction.py      # YOLO feature extraction
│   ├── uncertainty.py              # Aleatoric + Epistemic
│   ├── calibration.py              # Combined score conformal
│   ├── local_scaling.py            # Decision tree scaling
│   ├── temporal_propagation.py    # Kalman filter
│   └── evaluation.py               # Metrics computation
│
├── experiments/
│   ├── day1_tracker_matching.py   # Day 1
│   ├── day2_feature_extraction.py # Day 2
│   ├── day3_calibration.py        # Day 3
│   ├── day4_local_scaling.py      # Day 4
│   ├── day5_temporal.py           # Day 5-7
│   └── run_full_pipeline.py       # End-to-end
│
├── results/
│   └── v1/
│       ├── features/              # Extracted features
│       ├── calibration/           # Coverage plots
│       ├── tracking/              # MOTA/IDF1 results
│       └── visualizations/        # Uncertainty plots
│
├── data/
│   ├── cal_matched_tracks.pkl     # Calibration data
│   └── test_matched_tracks.pkl    # Test data
│
├── data_splits/
│   ├── cal_sequences.txt
│   ├── val_sequences.txt
│   └── test_sequences.txt
│
└── docs/
    ├── V1_RESULTS.md              # Results summary
    └── TROUBLESHOOTING.md         # Common issues
```

---

## Next Steps After V1

### Week 2: V2 (Multi-Source Epistemic)
- Add distance and entropy epistemic sources
- Learn weights via SLSQP
- OOD validation experiments
- Expected: 21% narrower intervals, 5× better OOD detection

### Week 3-4: Ablations + Paper
- Critical ablations (aleatoric only, epistemic only, no local scaling)
- Comparison with baselines
- Paper writing
- CVPR submission

### Optional: V3 (SAM Features)
- Only if time permits
- Extract SAM features
- Use Mahalanobis distance
- Compare YOLO vs SAM

---

**END OF V1 PLAN**
