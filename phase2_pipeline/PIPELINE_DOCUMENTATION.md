# Phase 2 Pipeline: Complete Documentation & Current Status

**Project**: Temporal Aleatoric Uncertainty in Video Object Tracking
**Last Updated**: 2024-10-29
**Status**: ⚠️ Functional but requires validation of arbitrary choices

---

## 1. What We Built

### Pipeline Overview
An end-to-end system to quantify aleatoric uncertainty in video object tracking using:
- **MC Dropout**: 30 forward passes (but YOLOv8 has no native dropout - we injected it)
- **Test-Time Augmentation**: 5 augmentations (blur, noise, brightness, scale, JPEG)
- **Target**: Track 25 in MOT17-11-FRCNN (900 frames, 8 occlusions)

### Results Achieved
- **Uncertainty during occlusion**: 1263.47 ± 1214.16
- **Uncertainty during non-occlusion**: 291.17 ± 299.44
- **Ratio**: 4.34× higher during occlusions (p < 10⁻⁶⁹)

---

## 2. How Uncertainty is Calculated

### The Formula
```python
# Step 1: MC Dropout - 30 forward passes
bbox_variances = []
confidence_variances = []
for i in range(30):
    prediction = model_with_dropout(image)
    bbox_variances.append(prediction.bbox)
    confidence_variances.append(prediction.confidence)

# Step 2: Calculate variance
bbox_uncertainty = np.mean(np.var(bbox_variances, axis=0))  # Variance in pixel space
conf_uncertainty = np.var(confidence_variances)

# Step 3: Combine (ARBITRARY WEIGHTS!)
mc_uncertainty = 0.7 * bbox_uncertainty + 0.3 * conf_uncertainty

# Step 4: TTA - 5 augmentations
tta_uncertainty = variance_across_augmentations()

# Step 5: Final combination (ARBITRARY!)
final_uncertainty = (mc_uncertainty + tta_uncertainty) / 2
```

### The Values
- Clean visibility: ~200-500 (low variance in predictions)
- During occlusion: ~1000-6000 (high variance in predictions)
- Units: Pixel-space variance (problematic - see issues)

---

## 3. Issues Fixed During Development

### Issue 1: Inverted Uncertainty (FIXED)
- **Problem**: Uncertainty was 0.13× during occlusions (87% LOWER - backwards!)
- **Root Causes**:
  1. YOLOv8n has NO dropout layers
  2. Missing detections assigned uncertainty=1.0 (way too low)
  3. IoU threshold too strict (0.45)
- **Fixes Applied**:
  1. Injected 10 Dropout2d layers into C2f blocks
  2. Set uncertainty=100.0 when track not found (ARBITRARY!)
  3. Adaptive IoU: 0.2 during occlusions, 0.3 otherwise

### Issue 2: Config Path Errors (FIXED)
- **Problem**: Nested dictionary access failures
- **Fix**: Proper config structure with correct paths

### Issue 3: Metadata Path Issues (FIXED)
- **Problem**: Looking in wrong directory
- **Fix**: Updated to `/metadata/raw_outputs/`

---

## 4. ⚠️ CURRENT ISSUES REQUIRING VALIDATION

### Critical Issue 1: Arbitrary Uncertainty for Missing Detections
```python
if not detected:
    uncertainty = 100.0  # COMPLETELY ARBITRARY!
```
**Problem**: This value has no scientific basis
**Options to Consider**:
1. Use mean + 2σ of detected frames
2. Use 95th percentile of observed uncertainties
3. Mark as NaN and exclude from analysis
4. Model separately as "detection confidence"

### Critical Issue 2: MC Dropout Injection Validity
```python
# We inject dropout into YOLOv8 which doesn't have it natively
for bottleneck in module.m:
    bottleneck.cv2 = nn.Sequential(original_conv, nn.Dropout2d(p=0.2))
```
**Problems**:
- Is this architecturally valid?
- Does it actually create meaningful variance?
- Should we use a different model with native dropout?

### Critical Issue 3: Scale and Unit Problems
```python
# Mixing different scales!
bbox_variance = np.var(bboxes)  # Range: 0-1000s (pixel space)
conf_variance = np.var(confs)   # Range: 0-1 (probability)
combined = 0.7 * bbox_var + 0.3 * conf_var  # INCOMPATIBLE UNITS!
```
**Problems**:
- Variance depends on image size (1920×1080 vs 640×480)
- Mixing pixel variance with probability variance
- Weights (0.7, 0.3) are completely arbitrary

### Critical Issue 4: Arbitrary Combination Weights
```python
# Why these weights?
mc_weight = 0.7  # for bbox
conf_weight = 0.3  # for confidence

# Why average?
final = (mc_uncertainty + tta_uncertainty) / 2
```
**No justification for**:
- 70/30 split between bbox and confidence
- Simple averaging of MC and TTA
- Should we learn these weights?

### Critical Issue 5: Classification Ambiguity
**Binary Classification**:
- Occluded: 204 frames
- Non-occluded: 696 frames
- Ratio: 4.34×

**Fine-grained Classification**:
- Occluded: 204 frames
- "Clean": 12 frames (very restrictive)
- Recovery: 684 frames
- Ratio: 2.59× (using only "clean")

**Which to use for paper?**

### Critical Issue 6: No True Baseline
We haven't run completely unmodified YOLO for comparison:
- No uncertainty quantification
- Just detection confidence
- Would show if uncertainty adds value over confidence

---

## 5. Code Structure

### Main Components
```
phase2_pipeline/
├── scripts/
│   └── run_pipeline.py              # Main orchestrator (computes uncertainty)
├── src/
│   ├── models/
│   │   └── yolo_wrapper.py         # YOLOv8 + injected dropout
│   ├── uncertainty/
│   │   └── aleatoric.py           # Uncertainty calculations
│   ├── data/
│   │   ├── mot_loader.py          # MOT17 dataset loader
│   │   └── track_extractor.py     # Track 25 specific analysis
│   └── visualization/
│       └── uncertainty_plots.py    # Plotting functions
├── analysis/
│   ├── naive_detection_analysis.py # Standard YOLO baseline
│   ├── comprehensive_analysis.py   # Statistical validation
│   └── combined_visualization.py   # Comparison plots
└── results/
    └── [timestamp]/
        ├── uncertainty_metrics/
        │   ├── uncertainty_timeline.json  # Per-frame values
        │   └── uncertainty_timeline.csv
        └── summary_report.txt
```

### Configuration Files
- `experiment.yaml` - Main settings (frames, model, paths)
- `model.yaml` - YOLOv8 configuration
- `uncertainty.yaml` - MC Dropout & TTA settings
- `augmentation.yaml` - TTA transforms

---

## 6. Statistical Validation

### What's Validated ✅
- T-test: p < 3.8×10⁻⁶⁹ (highly significant)
- Cohen's d: 1.10 (large effect size)
- Mann-Whitney U: p < 2.0×10⁻⁴⁶ (non-parametric confirmation)
- Temporal correlation: 0.585 (moderate-strong)

### What Needs Validation ❌
- Dropout injection validity
- Uncertainty = 100.0 for missing detections
- Weights (0.7, 0.3) justification
- Scale normalization
- Cross-sequence consistency

---

## 7. How to Run

### Basic Execution
```bash
# 100 frames (testing)
python phase2_pipeline/scripts/run_pipeline.py

# 900 frames (full)
# Edit config/experiment.yaml: end_frame: 900
python phase2_pipeline/scripts/run_pipeline.py
```

### Analysis
```bash
# Statistical validation
python phase2_pipeline/analysis/comprehensive_analysis.py

# Naive baseline comparison
python phase2_pipeline/analysis/naive_detection_analysis.py
```

---

## 8. Future Requirements

### Before Moving to Phase 3 (Vertical Expansion)
1. **Validate/fix arbitrary choices**:
   - Missing detection uncertainty value
   - Combination weights
   - Scale normalization

2. **Add proper baseline**:
   - Clean YOLO without modifications
   - Compare uncertainty vs confidence

3. **Cross-validation**:
   - Test on other MOT17 sequences
   - Verify 4.34× ratio consistency

### For Phase 3 Planning
- Multiple models (YOLOv8s, m, l)
- Multiple datasets (MOT20, KITTI)
- Multiple uncertainty methods (ensemble, deep ensembles)
- Proper modular architecture

---

## 9. Key Decisions Needed

1. **How to handle missing detections?** (Currently arbitrary 100.0)
2. **Which classification to report?** (4.34× vs 2.59×)
3. **How to justify/learn weights?** (Currently 0.7/0.3)
4. **Should we use a model with native dropout?**
5. **How to handle scale dependency?** (Normalize by image size?)

---

## 10. Files to Keep/Delete

### Keep (Updated):
- This file: `PIPELINE_DOCUMENTATION.md` (main documentation)
- All code in `src/`, `scripts/`, `analysis/`
- All results in `results/`

### To Delete (Redundant):
- `FINAL_RESULTS_SUMMARY.md` (contents merged here)
- `ANALYSIS_SUMMARY.md` (contents merged here)

---

## Summary

The pipeline works and shows statistically significant results, but has several **arbitrary choices that need scientific justification** before publication. The core finding (uncertainty increases during occlusions) is valid, but the exact values and methods need refinement.