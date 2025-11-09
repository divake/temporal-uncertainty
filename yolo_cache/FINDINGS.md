# Evidence-Based Findings from Dataset Investigation

**Date**: 2025-01-09
**Investigation**: Pre-implementation dataset analysis to avoid assumptions

---

## 🔍 **What We Discovered**

### ✅ **MOT17 Dataset Structure (CONFIRMED)**

```
/ssd_4TB/divake/temporal_uncertainty/data/MOT17/train/
├── MOT17-02-FRCNN/  (600 frames, 62 tracks, 18,581 detections)
├── MOT17-04-FRCNN/
├── MOT17-05-FRCNN/
├── MOT17-09-FRCNN/
├── MOT17-10-FRCNN/
├── MOT17-11-FRCNN/
└── MOT17-13-FRCNN/

Each sequence contains:
├── img1/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── gt/
│   └── gt.txt
└── seqinfo.ini
```

**Evidence**: `ls /ssd_4TB/divake/temporal_uncertainty/data/MOT17/train/`

---

### ✅ **Ground Truth Format (CONFIRMED)**

```csv
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
```

**Example from MOT17-02-FRCNN/gt/gt.txt:**
```
1,1,912,484,97,109,0,7,1
2,1,912,484,97,109,0,7,1
```

**Fields**:
- `frame`: 1-indexed frame number
- `id`: Track ID (unique within sequence)
- `bb_left`, `bb_top`, `bb_width`, `bb_height`: Bounding box
- `conf`: 0 = ignore, 1 = consider (use for evaluation)
- `class`: Object class ID
- `visibility`: [0.0, 1.0] - Ratio of visible bbox (0=fully occluded, 1=fully visible)

**Evidence**: `head -5 /ssd_4TB/divake/temporal_uncertainty/data/MOT17/train/MOT17-02-FRCNN/gt/gt.txt`

---

### ✅ **Class Distribution (CONFIRMED)**

**MOT17-02 Ground Truth Classes** (when conf==1):
```
Class 1: 18,581 detections (pedestrian) ← ONLY THIS USED FOR EVALUATION
Class 2: 0 detections (person on vehicle)
Others: Ignored (conf==0)
```

**Critical Finding**:
- Only class 1 (pedestrian) is used when conf==1
- All other classes have conf==0 (ignored)
- We should filter GT to: `conf==1 AND class==1`

**Evidence**:
```bash
awk -F',' '$7==1' MOT17-02-FRCNN/gt/gt.txt | awk -F',' '{print $8}' | sort | uniq -c
# Output: 18581 class_1
```

---

### ✅ **YOLO Model (CONFIRMED)**

**Model**: YOLOv8n (`/ssd_4TB/divake/temporal_uncertainty/models/yolov8n.pt`)

**Architecture**:
- Total layers: 23
- COCO class 0 = "person" (matches MOT17 pedestrian)
- Feature dimensions vary by layer (128, 256, 512, 1024)

**Evidence**:
```python
from ultralytics import YOLO
m = YOLO('models/yolov8n.pt')
print(m.names[0])  # Output: 'person'
print(len(m.model.model))  # Output: 23
```

**Class Mapping**:
```python
YOLO class 0 (person) ↔ MOT17 class 1 (pedestrian) ✓
```

---

### ✅ **Existing Metadata (GOLD MINE!)**

**Location**: `/ssd_4TB/divake/temporal_uncertainty/metadata/`

**What's Already Computed**:
```
metadata/
├── raw_outputs/
│   ├── seq02_metadata.json  (comprehensive per-sequence analysis)
│   ├── seq04_metadata.json
│   ├── seq05_metadata.json
│   ├── seq09_metadata.json
│   ├── seq10_metadata.json
│   ├── seq11_metadata.json
│   ├── seq13_metadata.json
│   ├── summary_all_sequences.json  (cross-sequence comparison)
│   └── hero_tracks_all_sequences.json  (selected tracks for analysis)
├── visualizations/  (Gantt charts, heatmaps, crowding plots)
└── README.md  (detailed documentation)
```

**Per-Sequence Metadata Includes**:
- Track lifetimes and durations
- Occlusion events (start, end, duration, min visibility)
- Crowding analysis (sparse/medium/crowded frames)
- Motion patterns (static vs moving tracks)
- Hero tracks (long_stable, occlusion_heavy, high_motion)
- Recommended test frames
- Entry/exit points
- Class labels ("pedestrian")

**Example Usage**:
```python
import json
with open('metadata/raw_outputs/seq02_metadata.json') as f:
    meta = json.load(f)

# Get track info
track_2 = meta['tracks']['2']
print(f"Lifetime: {track_2['lifetime_frames']}")
print(f"Duration: {track_2['duration']}")
print(f"Avg visibility: {track_2['avg_visibility']}")
print(f"Has occlusion: {track_2['has_occlusion']}")
print(f"Class: {track_2['class']}")  # 'pedestrian'
```

**Evidence**: `ls /ssd_4TB/divake/temporal_uncertainty/metadata/raw_outputs/`

---

## 📋 **Final Design Decisions (Evidence-Based)**

### **1. Data Splits**
**Decision**: Cache everything WITHOUT splits
**Rationale**: Splits are algorithmic decisions that belong in experiment code, not cache
**Implementation**: Save all 7 sequences completely, let algorithm code decide splits later

---

### **2. Ground Truth Matching**
**Decision**: IoU threshold = 0.5 (standard)
**Rationale**: Standard MOT/COCO evaluation metric
**Implementation**:
```python
# Filter GT to valid pedestrians
valid_gt = gt_data[(gt_data[:, 6] == 1) & (gt_data[:, 7] == 1)]

# Match YOLO (class 0) to MOT17 (class 1)
if yolo_class_id == 0:  # person
    match_to_gt_class_1()  # pedestrian
```

---

### **3. YOLO Feature Layers**
**Decision**: Cache layers 4, 9, 15, 21
**Rationale**: Multiple abstraction levels for ablation studies
**Implementation**:
```python
layers_to_cache = {
    4: 128,    # Early features
    9: 256,    # Mid features (V1 default)
    15: 512,   # Late features
    21: 1024   # Pre-detection head
}
```

---

### **4. Feature Extraction Method**
**Decision**: ROI pooling (forward full image once)
**Rationale**: Efficiency - one forward pass per frame vs hundreds
**Implementation**:
```python
# Forward full image once
features = model.model.model[:layer_idx](image)

# ROI pool for each bbox
for bbox in bboxes:
    bbox_features = roi_pool(features, bbox_coords)
```

---

### **5. Confidence Thresholds**
**Decision**: Run at conf=0.01, save indices for [0.3, 0.5, 0.7]
**Rationale**: Maximum flexibility without re-caching
**Implementation**:
```python
# Run YOLO at very low threshold
results = model(image, conf=0.01)

# Save all detections
all_detections = results.boxes

# Save indices for different thresholds
cache['thresholds'] = {
    '0.3': np.where(all_confs >= 0.3)[0],
    '0.5': np.where(all_confs >= 0.5)[0],
    '0.7': np.where(all_confs >= 0.7)[0],
}
```

---

### **6. Storage Path**
**Decision**:
```
/ssd_4TB/divake/temporal_uncertainty/yolo_cache/data/mot17/yolov8n/
├── MOT17-02-FRCNN.npz
├── MOT17-04-FRCNN.npz
├── MOT17-05-FRCNN.npz
├── MOT17-09-FRCNN.npz
├── MOT17-10-FRCNN.npz
├── MOT17-11-FRCNN.npz
└── MOT17-13-FRCNN.npz
```

**Rationale**: Extensible structure for future datasets/models

---

## 🎯 **What We're Caching (FINAL)**

### **Core Data (V1 Essentials)**

```python
cache_structure = {
    # All YOLO detections at conf >= 0.01
    'detections': {
        'frame_ids': [N] int32,           # Which frame
        'bboxes': [N, 4] float64,         # [x, y, w, h] - HIGH PRECISION
        'confidences': [N] float32,        # Detection confidence
        'class_ids': [N] int8,             # Should all be 0 (person)
    },

    # Multi-layer features (from 4 layers)
    'features': {
        'layer_4': [N, 128] float32,      # Early
        'layer_9': [N, 256] float32,      # Mid (V1 default)
        'layer_15': [N, 512] float32,     # Late
        'layer_21': [N, 1024] float32,    # Pre-head
    },

    # Threshold indices (saves storage!)
    'thresholds': {
        'conf_0.3': [N_0.3] int32,        # Indices where conf >= 0.3
        'conf_0.5': [N_0.5] int32,        # Standard
        'conf_0.7': [N_0.7] int32,        # High confidence
    },

    # GT matching (IoU >= 0.5)
    'gt_matching': {
        'det_indices': [M] int32,          # Index into detections
        'gt_indices': [M] int32,           # Index into GT
        'gt_bboxes': [M, 4] float64,       # GT bbox (for verification)
        'gt_track_ids': [M] int32,         # GT track ID
        'iou': [M] float32,                # IoU value
        'center_error': [M] float64,       # ||center_det - center_gt|| (OUR y!)
        'visibility': [M] float32,         # From GT visibility field
    },

    # False positives/negatives
    'unmatched': {
        'fp_det_indices': [K] int32,       # YOLO dets with no GT match
        'fn_gt_indices': [L] int32,        # GT boxes with no YOLO match
        'fn_gt_bboxes': [L, 4] float64,    # Undetected GT boxes
    },

    # Frame-level info
    'frames': {
        'frame_ids': [F] int32,
        'num_detections': [F] int32,       # Per frame
        'avg_confidence': [F] float32,     # Mean detection conf
        'image_paths': list[str],          # For verification
    },

    # Link to existing metadata
    'existing_metadata': {
        'metadata_path': str,              # Path to seq{XX}_metadata.json
        'has_occlusion_info': True,
        'has_crowding_info': True,
        'has_hero_tracks': True,
    },

    # Extension placeholders (for V2/V3)
    'v2_extensions': {},  # Empty - for multi-source epistemic
    'v3_extensions': {},  # Empty - for SAM features

    # Metadata
    'meta': {
        'version': '1.0',
        'sequence': 'MOT17-02-FRCNN',
        'yolo_model': 'yolov8n',
        'yolo_version': str,
        'cached_layers': [4, 9, 15, 21],
        'cached_thresholds': [0.01, 0.3, 0.5, 0.7],
        'iou_threshold': 0.5,
        'date_cached': str,
        'total_frames': 600,
        'total_detections': int,
        'total_matched': int,
        'class_mapping': {
            'yolo_0': 'person',
            'mot17_1': 'pedestrian',
            'match': True
        }
    }
}
```

---

## ❌ **What We're NOT Caching**

### **Skipped for V1 (Can Add Later)**

1. **Pre-NMS Detections** ❌
   - Reason: Not easily accessible in ultralytics API
   - Alternative: Run at low conf threshold to get more boxes
   - Can add in V2 if needed via custom hooks

2. **Multiple YOLO Models** ❌
   - Reason: V1 uses only YOLOv8n
   - How to add later: Run preprocessing with different model

3. **Augmented Versions** ❌
   - Reason: Overkill for V1
   - How to add later: Add to v2_extensions

4. **Attention Maps** ❌
   - Reason: Not accessible in standard YOLOv8 API
   - How to add later: Requires custom model hooks

5. **Data Splits** ❌
   - Reason: Splits are algorithmic choices, not cache data
   - How to use: Define splits in experiment code

6. **Derived Statistics** ❌
   - Reason: Can compute on-the-fly from cached data
   - Examples: Histograms, correlations, etc.

---

## 🔧 **Precision Strategy**

### **High Precision (float64)**
- Bounding boxes (sub-pixel accuracy)
- Center errors (our target y - critical!)
- GT bboxes (verification)

### **Standard Precision (float32)**
- Features (256-1024 dim, negligible loss)
- Confidences, IoU values
- Visibility scores

### **Smallest Int Possible**
- Frame IDs: int32
- Track IDs: int32
- Class IDs: int8 (0-79 for COCO)
- Indices: int32

---

## 📦 **Storage Estimate**

Per sequence (~750 frames, ~15,000 detections):

```
Bboxes (float64):          15k × 4 × 8 = 480 KB
Features (4 layers):       15k × 1920 × 4 = 461 MB
GT matching (float64):     12k × 8 = 96 KB
Threshold indices:         3 × 15k × 4 = 180 KB
Frames:                    750 × 8 = 6 KB
Metadata:                  < 1 KB
------------------------------------------------
Total per sequence:        ~462 MB
With compression (3-5×):   ~150 MB

All 7 sequences:           ~1 GB (very manageable!)
```

---

## ✅ **Validation Strategy**

After caching each sequence, verify with 10 random samples:

```python
for i in random_sample(10):
    # Load cached
    cached_bbox = cache['detections/bboxes'][i]
    cached_features = cache['features/layer_9'][i]

    # Recompute
    frame = load_image(cache['detections/frame_ids'][i])
    fresh_result = model(frame)
    fresh_bbox = find_matching_detection(fresh_result, cached_bbox)
    fresh_features = extract_layer_9_features(fresh_bbox)

    # Verify
    bbox_error = np.abs(cached_bbox - fresh_bbox).max()
    feature_error = np.abs(cached_features - fresh_features).max()

    assert bbox_error < 0.1, f"Bbox error: {bbox_error}"
    assert feature_error < 1e-6, f"Feature error: {feature_error}"
```

**Tolerances**:
- Bbox error: < 0.1 pixels
- Feature error: < 1e-6 (relative)

---

## 🚀 **Usage After Caching**

### **Load Cache (Lightning Fast)**
```python
cache = np.load('yolo_cache/data/mot17/yolov8n/MOT17-02-FRCNN.npz')

# Get features (instant!)
features = cache['features/layer_9']  # [N, 256]

# Get matched detections with errors
errors = cache['gt_matching/center_error']  # Our y!

# Filter by confidence threshold
conf_0_5_indices = cache['thresholds/conf_0.5']
features_0_5 = features[conf_0_5_indices]

# Load existing metadata
import json
with open(cache['existing_metadata/metadata_path'].item()) as f:
    metadata = json.load(f)

# Get occlusion info from metadata (don't recompute!)
hero_tracks = metadata['hero_tracks']['occlusion_heavy']
```

---

**END OF FINDINGS DOCUMENT**
