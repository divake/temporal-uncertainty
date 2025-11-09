# YOLO Cache Design Document

**Purpose**: Pre-compute and cache all YOLO outputs to enable rapid experimentation without re-running inference.

**Created**: 2025-01-09
**Author**: Temporal Uncertainty Project
**Target**: MOT17 Tracking with Enhanced CACD V1

---

## Why Cache?

### The Problem
Running YOLO inference in the experiment loop is **prohibitively slow**:
- 7 MOT17 sequences × ~750 frames × ~20 detections = **~105,000 forward passes**
- At 50ms per pass = **87 minutes per experiment**
- With iterations/ablations = **20-30 hours for V1**

### The Solution
**Pre-compute once, use forever**:
- Run YOLO once per frame (7 × 750 = 5,250 forward passes)
- Cache all outputs with high precision
- Experiment time drops to **< 60 seconds**
- **400× speedup!**

---

## What We Cache

### ✅ TIER 1: Raw YOLO Outputs (MUST HAVE)

#### 1. Detections (Per-Detection Data)
```python
'detections/frame_ids': [N_dets] (int32)          # Which frame each detection is from
'detections/bboxes': [N_dets, 4] (float64)        # [x, y, w, h] - HIGH PRECISION!
'detections/confidences': [N_dets] (float32)      # Detection confidence [0, 1]
'detections/class_ids': [N_dets] (int32)          # COCO class ID (0 = person)
```

**Why High Precision for Bboxes?**
- We compute center errors: `||center_pred - center_gt||`
- Small numerical errors compound in downstream metrics
- float64 ensures sub-pixel accuracy

#### 2. Multi-Layer Features (CRITICAL!)
```python
'features/layer4': [N_dets, 128] (float32)        # Early features
'features/layer9': [N_dets, 256] (float32)        # Mid features (V1 default)
'features/layer12': [N_dets, 512] (float32)       # Late features
'features/layer21': [N_dets, 1024] (float32)      # Pre-detection head
```

**Why Multiple Layers?**
- Different layers capture different abstraction levels
- Layer 9 (256-dim) is V1 default
- Enables ablation: "Which layer gives best uncertainty?"
- Minimal extra cost (storage is cheap)

#### 3. Class Predictions (All 80 COCO Classes)
```python
'classes/logits': [N_dets, 80] (float32)          # Raw scores (before softmax)
'classes/probs': [N_dets, 80] (float32)           # After softmax
```

**Why All 80 Classes?**
- Even though MOT17 is just "person" (class 0)
- Epistemic uncertainty: How confused is YOLO?
- If `P(person)=0.6, P(bicycle)=0.3` → high uncertainty
- If `P(person)=0.99, P(all_else)<0.01` → low uncertainty

---

### ✅ TIER 2: NMS Analysis (For Epistemic Uncertainty)

#### 4. Pre-NMS Information
```python
'nms/total_raw_detections': [N_frames] (int32)    # Detections before NMS
'nms/suppressed_count': [N_dets] (int32)          # How many boxes merged into this one
'nms/max_iou_suppressed': [N_dets] (float32)      # Max IoU with suppressed boxes
'nms/confidence_histogram': [N_frames, 20] (int32) # Distribution per frame
```

**Why This Matters?**
- High NMS overlap = Many competing hypotheses = **High aleatoric uncertainty**
- If 5 boxes with conf [0.7, 0.65, 0.6, 0.55, 0.5] get merged → uncertain detection
- If 1 box with conf [0.95] alone → certain detection

---

### ✅ TIER 3: Multi-Threshold Detections

#### 5. Confidence Threshold Variants
```python
'thresholds/conf_0.3/indices': [N_0.3] (int32)    # Indices passing conf > 0.3
'thresholds/conf_0.5/indices': [N_0.5] (int32)    # Standard threshold
'thresholds/conf_0.7/indices': [N_0.7] (int32)    # High confidence only
```

**Why Multiple Thresholds?**
- Number of low-confidence detections = epistemic uncertainty proxy
- Ablation: "Does uncertainty change with detection threshold?"
- Storage efficient: Just indices, not full data

---

### ✅ TIER 4: Ground Truth Matching (Saves HOURS!)

#### 6. GT Matching Results
```python
'gt/det_indices': [N_matched] (int32)             # Index into detections
'gt/gt_indices': [N_matched] (int32)              # Index into GT annotations
'gt/iou': [N_matched] (float32)                   # IoU between det and GT
'gt/center_error': [N_matched] (float64)          # ||center_det - center_gt|| (OUR y!)
'gt/visibility': [N_matched] (float32)            # GT visibility [0, 1]
'gt/is_occluded': [N_matched] (bool)              # visibility < 0.3
'gt/crowding': [N_matched] (int32)                # Num objects within 50px
'gt/bbox_area': [N_matched] (float32)             # w × h (for size analysis)
```

**Why Cache Matching?**
- Hungarian matching is **O(n³)** - expensive!
- We'll run experiments 100+ times
- Matching once and caching saves ~10 hours total

**Why High Precision for center_error?**
- This is our target variable `y`
- Conformal calibration is sensitive to precision
- float64 ensures accurate coverage metrics

#### 7. Failure Cases (For Error Analysis)
```python
'gt/false_positive_indices': [N_fp] (int32)       # Detections with no GT match
'gt/false_negative_gt_ids': [N_fn] (int32)        # GT boxes with no detection
```

---

### ✅ TIER 5: Temporal Information (For Days 5-7)

#### 8. Track Associations
```python
'temporal/track_ids': [N_matched] (int32)         # Track ID (from GT or tracker)
'temporal/prev_det_idx': [N_matched] (int32)      # Index in previous frame (-1 if new)
'temporal/next_det_idx': [N_matched] (int32)      # Index in next frame (-1 if lost)
'temporal/track_age': [N_matched] (int32)         # Frames since track started
'temporal/track_lifetime': [N_matched] (int32)    # Total frames track exists
'temporal/frames_since_update': [N_matched] (int32) # For occlusion handling
'temporal/gap_count': [N_matched] (int32)         # Num occlusion gaps in track
```

**Why This Matters?**
- Kalman filter experiments (Day 5-7)
- Uncertainty should decrease as track ages (more observations)
- Uncertainty should increase during gaps (occlusions)

---

### ✅ TIER 6: Frame-Level Metadata (For Local Scaling)

#### 9. Frame Statistics
```python
'frames/frame_ids': [N_frames] (int32)
'frames/num_detections': [N_frames] (int32)       # Total detections per frame
'frames/avg_confidence': [N_frames] (float32)     # Mean detection confidence
'frames/crowding_score': [N_frames] (float32)     # Spatial density
'frames/brightness': [N_frames] (float32)         # Mean pixel intensity [0, 255]
'frames/blur_score': [N_frames] (float32)         # Laplacian variance
```

**Why Frame-Level Data?**
- Local scaling (Day 4): Some **frames** are harder than others
- Blurry frames → higher uncertainty
- Crowded frames → higher uncertainty
- Can partition state space by frame difficulty

---

### ✅ TIER 7: Data Splits (CRITICAL!)

#### 10. Split Assignments
```python
'splits/det_split': [N_matched] (str)             # 'cal', 'val', or 'test'
'splits/frame_split': [N_frames] (str)            # Split per frame
'splits/sequence_split': str                      # Overall sequence assignment
```

**Why Pre-Assign Splits?**
- **Never accidentally use test data for calibration!**
- Reproducibility: Same splits across all experiments
- Temporal splits: Early frames → cal, middle → val, late → test

---

### ✅ TIER 8: Reproducibility Metadata

#### 11. Computational Metadata
```python
'meta/yolo_version': str                          # e.g., '8.0.232'
'meta/model_name': str                            # e.g., 'yolov8n.pt'
'meta/model_path': str                            # Absolute path to .pt file
'meta/image_size': [2] (int32)                    # [H, W] e.g., [1920, 1080]
'meta/preprocessing': str                         # e.g., 'letterbox_640'
'meta/date_computed': str                         # ISO format
'meta/gpu': str                                   # e.g., 'A100'
'meta/avg_inference_ms': float                    # Average time per frame
'meta/total_frames': int
'meta/total_detections': int
'meta/total_matched': int

# Random seeds
'meta/numpy_seed': int                            # 42
'meta/torch_seed': int                            # 42
'meta/cuda_deterministic': bool                   # True
```

---

## What We DON'T Cache

### ❌ NOT Caching (Can Add Later if Needed)

1. **Augmented Versions** ❌
   - Features from brightness/contrast/blur augmented images
   - **Why Skip**: Overkill for V1, can add in V2 if needed
   - **How to Add Later**: Run preprocessing with `--augment` flag

2. **Attention Maps** ❌
   - Transformer attention weights (if using YOLOv8 transformer variant)
   - **Why Skip**: Not easily accessible in standard YOLOv8 API
   - **How to Add Later**: Requires custom hooks into model

3. **Multiple YOLO Models** ❌
   - YOLOv8s, YOLOv8m, YOLOv8x, YOLOv5, etc.
   - **Why Skip**: V1 uses only YOLOv8n
   - **How to Add Later**: Run `precompute_mot17.py --model yolov8s`

4. **Spatial Feature Maps** ❌
   - Full spatial features before global pooling
   - **Why Skip**: Massive storage (100× larger), not needed for V1
   - **How to Add Later**: Modify `feature_extractor.py` to save raw maps

5. **Derived Statistics** ❌
   - Pre-computed histograms, correlations, etc.
   - **Why Skip**: Can compute on-the-fly from cached data
   - **How to Add Later**: Not needed - compute in analysis scripts

---

## File Format: NPZ with High Precision

### Why NPZ?
- **Simple**: Native numpy, no extra dependencies
- **Fast**: ~2 seconds to load 500MB file
- **Compressed**: `np.savez_compressed()` gives 3-5× reduction
- **Inspectable**: `cache = np.load('file.npz'); print(cache.files)`
- **Precision Control**: Can specify dtype per array

### High Precision Strategy
```python
# Critical data: float64 (double precision)
- Bounding boxes (for sub-pixel accuracy)
- Center errors (our target variable y)

# Regular data: float32 (single precision)
- Features (256-1024 dim, precision loss negligible)
- Confidences, IoUs, frame stats

# Categorical data: Smallest int possible
- Frame IDs, track IDs: int32
- Boolean flags: bool (1 byte)
- Class IDs: int8 (0-79 for COCO)
```

### Storage Estimate (Per Sequence)
```
~750 frames, ~15,000 detections, ~12,000 matched:

Bboxes (float64):          15k × 4 × 8 = 480 KB
Features 4 layers:         15k × 1920 × 4 = 461 MB
Class logits:              15k × 80 × 4 = 4.8 MB
GT matching (float64):     12k × 8 = 96 KB
Temporal:                  12k × 7 × 4 = 336 KB
Frame metadata:            750 × 6 × 4 = 18 KB
------------------------------------------------
Total per sequence:        ~470 MB
With compression:          ~180 MB

All 7 sequences:           ~1.3 GB (totally manageable!)
```

---

## Validation Strategy

### Mandatory Post-Cache Validation
After caching each sequence, we **MUST** verify cache integrity:

```python
# For 10 random samples:
1. Load cached bbox and features
2. Re-run YOLO on same frame
3. Find matching detection
4. Compare cached vs fresh:
   - Bbox error < 0.1 pixels
   - Feature error < 1e-6 (relative)
5. Save validation report
```

### Validation Report Format
```
MOT17-02-FRCNN Validation Report
================================
Date: 2025-01-09 14:23:45
Samples Tested: 10

Sample 1:
  Frame: 42
  Bbox Error: 0.03 pixels ✓
  Feature Error: 2.3e-7 ✓

Sample 2:
  Frame: 156
  Bbox Error: 0.01 pixels ✓
  Feature Error: 1.1e-7 ✓

...

Overall: PASS ✓
Max Bbox Error: 0.08 pixels
Max Feature Error: 8.9e-7
```

If validation fails → **DO NOT USE CACHE** → Re-run preprocessing

---

## Extensibility: Adding New Datasets/Models

### Adding a New Dataset (e.g., KITTI)
```bash
# 1. Create directory
mkdir -p data/kitti/yolov8n

# 2. Run preprocessing
python scripts/precompute_kitti.py \
    --model yolov8n.pt \
    --dataset-path /path/to/KITTI \
    --output data/kitti/yolov8n

# 3. Code automatically adapts (same cache structure)
```

### Adding a New Model (e.g., YOLOv8s)
```bash
# 1. Create directory
mkdir -p data/mot17/yolov8s

# 2. Run preprocessing
python scripts/precompute_mot17.py \
    --model yolov8s.pt \
    --output data/mot17/yolov8s

# 3. Cache structure identical, just different features/bboxes
```

### Code Flexibility
All code in `src/` is **dataset-agnostic** and **model-agnostic**:
- `yolo_runner.py`: Works with any YOLO model (v5/v8/v10)
- `gt_matcher.py`: Works with any GT format (MOT/KITTI/etc.)
- `cache_writer.py`: Always saves same structure

Only dataset-specific code in `scripts/precompute_<dataset>.py`

---

## Usage After Caching

### Load and Use (Lightning Fast!)
```python
import numpy as np

# Load cache (2 seconds)
cache = np.load('data/mot17/yolov8n/MOT17-02-FRCNN_cache.npz')

# Get features (instant!)
features = cache['features/layer9']  # [N, 256]

# Get matched data (instant!)
errors = cache['gt/center_error']    # Our target y!

# Get calibration split (instant!)
cal_mask = (cache['splits/det_split'] == 'cal')
features_cal = features[cal_mask]
errors_cal = errors[cal_mask]

# Run experiment (< 60 seconds total!)
uncertainties = compute_aleatoric_epistemic(features_cal)
q_hat = calibrate(errors_cal, uncertainties)
coverage = evaluate(errors_test, uncertainties_test, q_hat)
```

### No More YOLO Loading!
```python
# OLD WAY (87 minutes):
model = YOLO('yolov8n.pt')  # Load model
for frame in frames:
    results = model(frame)   # Inference (SLOW!)

# NEW WAY (2 seconds):
cache = np.load('cache.npz')  # Load cache
features = cache['features']  # Instant!
```

---

## Implementation Checklist

- [ ] Create directory structure (`yolo_cache/`)
- [ ] Write modular code in `src/` (8 files)
- [ ] Write preprocessing script (`scripts/precompute_mot17.py`)
- [ ] Write validation script (`scripts/validate_cache.py`)
- [ ] Run preprocessing on all 7 MOT17 sequences
- [ ] Validate each cache file (10 samples per sequence)
- [ ] Document results in `validation_reports/`
- [ ] Update conformal_tracking experiments to use cache

**Total Time**: 3-4 hours preprocessing + validation
**Time Saved**: 20-30 hours over Week 1
**ROI**: 6-8× return on investment!

---

## Future Enhancements (V2/V3)

### Possible Additions
1. **SAM Features** (V3):
   - Cache SAM embeddings (256-dim)
   - Same structure, just different feature extractor

2. **Multi-Model Ensemble**:
   - Cache YOLOv8n + YOLOv8s + YOLOv8m
   - Epistemic uncertainty from model disagreement

3. **Test-Time Augmentation**:
   - Cache features from 5× augmented versions
   - Use variance as aleatoric uncertainty

4. **Temporal Smoothing**:
   - Cache optical flow between frames
   - Temporal consistency losses

All additions follow **same cache structure** - just add new fields to NPZ!

---

**END OF CACHE DESIGN DOCUMENT**
