# Complete Data Understanding: From Raw Dataset to Cached Features

**Created**: November 9, 2025
**Purpose**: Build complete understanding of what data we have and how it flows through our system

---

## Overview: Two Data Layers

We have **TWO** separate but related data sources:

1. **Raw MOT17 Dataset** - Original video sequences with ground truth annotations
2. **YOLO Cache Files** - Pre-computed YOLO detections, features, and GT matching

**Why two layers?**
- **Raw dataset**: Ground truth for evaluation
- **Cache**: Pre-computed YOLO outputs for fast experimentation (don't need to re-run YOLO every time!)

---

## Layer 1: Raw MOT17 Dataset

### Location
```
/ssd_4TB/divake/temporal_uncertainty/data/MOT17/train/
```

### MOT17-05-FRCNN Example Structure
```
MOT17-05-FRCNN/
├── seqinfo.ini          # Sequence metadata
├── gt/
│   └── gt.txt           # Ground truth annotations (8,013 lines)
├── det/
│   └── det.txt          # Pre-computed detections (not used by us)
└── img1/
    ├── 000001.jpg       # Frame 1
    ├── 000002.jpg       # Frame 2
    ...
    └── 000837.jpg       # Frame 837 (last frame)
```

### Sequence Info (seqinfo.ini)
```ini
[Sequence]
name=MOT17-05-FRCNN
imDir=img1               # Image directory
frameRate=14             # 14 FPS
seqLength=837            # 837 frames total
imWidth=640              # 640 pixels wide
imHeight=480             # 480 pixels tall
imExt=.jpg               # JPEG format
```

**What this tells us:**
- MOT17-05 is 837 frames at 14 FPS = **59.8 seconds of video**
- Resolution: 640×480 (VGA resolution)
- Frame rate is low (14 FPS) compared to modern videos (30-60 FPS)

### Ground Truth Format (gt/gt.txt)

**Format**: CSV with no header
```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
```

**Example lines:**
```
1,1,17,150,77,191,1,1,1
2,1,20,148,78,195,1,1,1
3,1,23,147,79,199,1,1,1
```

**Decoded:**
- Frame 1: Person ID 1 at bbox [17, 150, 77, 191], confidence=1, class=1 (pedestrian), visibility=1 (fully visible)
- Frame 2: Same person ID 1, moved to [20, 148, 78, 195]
- Frame 3: Same person, continues moving

**Key Fields:**
- `<frame>`: Frame number (1-indexed)
- `<id>`: Track ID (unique person identifier across frames)
- `<bb_left>, <bb_top>`: Top-left corner (x, y)
- `<bb_width>, <bb_height>`: Bounding box size
- `<conf>`: Always 1 for GT (ground truth is certain)
- `<class>`: 1 = pedestrian (MOT17 only has pedestrians)
- `<visibility>`: Occlusion level (0=fully occluded, 1=fully visible)

**Statistics:**
- MOT17-05 has **8,013 GT annotations** across 837 frames
- Average: 8,013 / 837 = **~9.6 annotations per frame**
- This means ~9-10 people are visible in each frame on average

---

## Layer 2: YOLO Cache Files

### Location
```
/ssd_4TB/divake/temporal_uncertainty/yolo_cache/data/mot17/yolov8n/
```

### File Structure
Each sequence has TWO files:
1. **MOT17-05-FRCNN.npz** - Binary NumPy archive (43 MB)
2. **MOT17-05-FRCNN.json** - Metadata (254 bytes)

### What's Inside the Cache?

The cache contains **everything** we need for experiments, pre-computed:

#### 1. YOLO Detections (28,404 total)
```
detections/frame_ids:    [28404]      # Which frame each detection belongs to
detections/bboxes:       [28404, 4]   # [x1, y1, x2, y2] format
detections/confidences:  [28404]      # YOLO confidence scores (0-1)
detections/class_ids:    [28404]      # Class ID (0 = person in YOLO)
```

**Key Insight**: YOLO detected **28,404 objects** across 837 frames
- Average: 28,404 / 837 = **~34 detections per frame**
- But GT only has ~9.6 per frame!
- This means **YOLO has many false positives** (detects things that aren't people)

#### 2. YOLO Features (Multiple Layers)
```
features/layer_4:   [28404, 64]    # Early layer: 64 dimensions
features/layer_9:   [28404, 256]   # Mid layer: 256 dimensions
features/layer_15:  [28404, 64]    # Late layer: 64 dimensions
features/layer_21:  [28404, 256]   # Final layer: 256 dimensions
```

**What are these?**
- These are **intermediate feature representations** from YOLO's backbone
- Layer 4: Early features (edges, textures)
- Layer 9: Mid-level features (parts, shapes)
- Layer 15: High-level features (object-like patterns)
- Layer 21: **Final layer** before classification (most semantic)

**Why do we save them?**
- For uncertainty quantification! We use feature space distances to measure uncertainty
- Different layers capture different levels of abstraction
- Our experiments test which layer is best for uncertainty

#### 3. Ground Truth Matching (5,304 matched)
```
gt_matching/det_indices:   [5304]      # Which YOLO detections matched GT
gt_matching/gt_indices:    [5304]      # Which GT annotations they matched
gt_matching/iou:           [5304]      # IoU score (overlap quality)
gt_matching/center_error:  [5304]      # Distance between centers (pixels)
gt_matching/gt_bboxes:     [5304, 4]   # The GT boxes that matched
gt_matching/gt_track_ids:  [5304]      # GT track IDs
gt_matching/visibility:    [5304]      # GT visibility (occlusion level)
```

**Critical Understanding:**
- Out of **28,404 YOLO detections**, only **5,304 matched GT** (18.7%)
- This means:
  - **5,304** = True Positives (TP) - correct detections
  - **23,100** = False Positives (FP) - YOLO detected something that isn't there
  - **?** = False Negatives (FN) - GT objects YOLO missed

#### 4. Unmatched Detections
```
unmatched/fp_det_indices:  [23100]      # False positive detections
unmatched/fn_gt_indices:   [1613]       # False negative GT (YOLO missed these)
unmatched/fn_gt_bboxes:    [1613, 4]    # The GT boxes YOLO missed
```

**Now we can compute recall:**
- Total GT objects: 5,304 (matched) + 1,613 (missed) = **6,917**
- YOLO detected: 5,304 / 6,917 = **76.7% recall**
- YOLO missed: 1,613 / 6,917 = **23.3%**

Wait, but GT file has 8,013 annotations, not 6,917!
- The difference (8,013 - 6,917 = 1,096) might be:
  - Heavily occluded objects (visibility < threshold)
  - Objects outside YOLO's detection threshold
  - Annotations for non-person classes (ignored)

#### 5. Confidence Thresholds (Pre-filtered)
```
thresholds/conf_0.01:  [28404]    # All detections (conf ≥ 0.01)
thresholds/conf_0.3:   [5226]     # Medium confidence
thresholds/conf_0.5:   [4204]     # High confidence
thresholds/conf_0.7:   [3332]     # Very high confidence
```

**What this means:**
- At conf ≥ 0.3: 5,226 detections (18.4% of all)
- At conf ≥ 0.5: 4,204 detections (14.8% of all)
- At conf ≥ 0.7: 3,332 detections (11.7% of all)

**Compare to matched:**
- We have 5,304 matched detections (TP)
- At conf ≥ 0.5, we only keep 4,204 detections
- This means we're **throwing away some true positives** to reduce false positives!

#### 6. Per-Frame Info
```
frames/frame_ids:      [837]    # Frame numbers (1-837)
frames/num_detections: [837]    # How many detections per frame
```

#### 7. Metadata
```
meta/version:                  Version string
meta/sequence:                 'MOT17-05-FRCNN'
meta/dataset_type:             'MOT17'
meta/yolo_model:               'yolov8n'
meta/cached_layers:            [4, 9, 15, 21]
meta/cached_thresholds:        [0.01, 0.3, 0.5, 0.7]
meta/iou_threshold:            IoU threshold for matching (likely 0.5)
meta/date_cached:              When this was created
meta/total_frames:             837
meta/total_detections:         28404
meta/total_matched:            5304
```

---

## Data Flow: From Raw Video to Experiments

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: RAW MOT17 DATASET                                       │
│ - 837 frames (000001.jpg - 000837.jpg)                          │
│ - Ground truth annotations (gt.txt)                             │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: YOLO INFERENCE (Pre-computed)                           │
│ - Run YOLOv8n on all 837 frames                                 │
│ - Extract detections: bboxes, confidences, class IDs            │
│ - Extract features from layers: 4, 9, 15, 21                    │
│ - Result: 28,404 detections with features                       │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: GROUND TRUTH MATCHING (Pre-computed)                    │
│ - Match YOLO detections to GT using IoU                         │
│ - Compute center_error for each match                           │
│ - Result: 5,304 matched pairs (TP)                              │
│          23,100 false positives (FP)                             │
│           1,613 false negatives (FN)                             │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: SAVE TO CACHE (.npz file)                               │
│ - All detections, features, matching stored                     │
│ - Size: 43 MB (compressed)                                      │
│ - Loading time: ~0.1 seconds (vs minutes for YOLO inference!)   │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: V1 ENHANCED CACD EXPERIMENTS                            │
│ - Load cache (fast!)                                            │
│ - Filter by confidence threshold (e.g., ≥0.5)                   │
│ - Extract features from specific layer (e.g., layer 9)          │
│ - Use only MATCHED detections (5,304)                           │
│ - Split into calibration/test (50/50)                           │
│ - Compute uncertainty using KNN + KDE                           │
│ - Evaluate correlation with center_error                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Critical Numbers for MOT17-05

| Metric | Value | Notes |
|--------|-------|-------|
| **Raw Dataset** | | |
| Total frames | 837 | At 14 FPS = 59.8 seconds |
| Resolution | 640×480 | VGA quality |
| GT annotations | 8,013 | ~9.6 per frame |
| | | |
| **YOLO Detections** | | |
| Total detections | 28,404 | ~34 per frame |
| True positives (matched) | 5,304 | 18.7% of detections |
| False positives | 23,100 | 81.3% of detections! |
| False negatives (missed GT) | 1,613 | 23.3% of GT missed |
| | | |
| **Confidence Filtering** | | |
| At conf ≥ 0.3 | 5,226 | 18.4% kept |
| At conf ≥ 0.5 | 4,204 | 14.8% kept |
| At conf ≥ 0.7 | 3,332 | 11.7% kept |
| | | |
| **Features** | | |
| Layer 4 dimensions | 64 | Early features |
| Layer 9 dimensions | 256 | Mid features |
| Layer 15 dimensions | 64 | Late features |
| Layer 21 dimensions | 256 | Final features |

---

## What We Actually Use in V1 Experiments

When we run V1 Enhanced CACD on MOT17-05:

```python
# Load cache
loader = YOLOCacheLoader('MOT17-05-FRCNN.npz')

# Get features from layer 9 (256-dim)
features = loader.get_features(layer_id=9)  # [28404, 256]

# Get matched detections only
matched_mask = loader.get_matched_mask()    # [28404] boolean
features_matched = features[matched_mask]    # [5304, 256]

# Get center errors (our target variable)
center_errors = loader.get_center_errors()   # [5304]

# Get confidences for filtering
confidences = loader.get_confidences()       # [28404]
conf_matched = confidences[matched_mask]     # [5304]

# Filter by confidence ≥ 0.5
mask = conf_matched >= 0.5                   # [5304] boolean
features_final = features_matched[mask]      # [3609, 256]
errors_final = center_errors[mask]           # [3609]
```

**Final dataset for experiments:**
- **3,609 detections** (after conf ≥ 0.5 filtering)
- **256-dimensional features** (from layer 9)
- **Center errors** as ground truth (what we try to predict uncertainty for)

---

## Why MOT17-05 Might Be Different

Looking at the data, here are potential reasons MOT17-05 fails:

### Hypothesis 1: Low Frame Rate
- 14 FPS is quite low (modern videos are 30-60 FPS)
- Large jumps between frames → more motion blur?
- Objects move more between frames → harder to track?

### Hypothesis 2: High False Positive Rate
- 81.3% of YOLO detections are false positives!
- This is unusually high - need to check other sequences
- If YOLO is confused, its features might not be reliable

### Hypothesis 3: Resolution
- 640×480 is low resolution
- Small objects might be harder to detect accurately
- Features might be noisier

### Hypothesis 4: Scene Characteristics
- Sequence name: "street with crowds"
- Heavy crowding → occlusions → YOLO confusion?
- Need to visualize frames to understand

---

## Next Steps: What We Need to Understand

### Immediate Questions to Answer:

1. **Compare MOT17-05 to other sequences**
   - Are FP rates similar across sequences?
   - Are frame rates different?
   - Are resolutions different?

2. **Visualize MOT17-05 frames**
   - What does the scene actually look like?
   - Are there heavy occlusions?
   - Is lighting poor?
   - Are objects small/far away?

3. **Verify cache correctness**
   - Load a few frames manually
   - Run YOLO ourselves
   - Compare to cached values
   - Ensure cache is correct!

4. **Understand feature extraction**
   - Which layers in YOLOv8 are 4, 9, 15, 21?
   - What do these layers represent?
   - Why these specific layers?

---

## Summary: The Complete Picture

**What we have:**
1. **Raw dataset**: Video frames + ground truth tracking annotations
2. **YOLO cache**: Pre-computed detections, features, and GT matching

**Why cache?**
- Running YOLO on all sequences is slow (~minutes per sequence)
- Cache loads in ~0.1 seconds
- Enables fast iteration on uncertainty algorithms

**What's in the cache:**
- All YOLO detections (including false positives)
- Features from 4 different layers
- GT matching results with errors
- Pre-filtered by multiple confidence thresholds

**What we use:**
- **Only matched detections** (true positives)
- **One layer's features** (we test which is best)
- **Center error** as ground truth
- **Confidence filtering** to reduce noise

**MOT17-05 challenge:**
- 81.3% false positive rate
- Positive correlation between confidence and error (+0.237)
- Compressed feature space (low variance)
- Needs investigation to understand why it's different

---

**Created by**: Deep dive analysis
**Date**: November 9, 2025
**Status**: Foundation for understanding - ready to build on this
