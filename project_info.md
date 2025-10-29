# Temporal Aleatoric Uncertainty in Video Object Tracking: Complete Implementation Guide

## Project Overview

### What We Have
- **Dataset**: MOT17 train set with 7 sequences (525-1050 frames each)
  - Located at: `/ssd_4TB/divake/temporal_uncertainty/MOT17/train/`
  - MOT17-02-FRCNN: 600 frames (street scene with occlusions)
  - MOT17-04-FRCNN: 1050 frames (shopping mall - longest sequence)
  - MOT17-05-FRCNN: 837 frames (street with crowds)
  - MOT17-09-FRCNN: 525 frames (pedestrian crossing - shortest sequence)
  - MOT17-10-FRCNN: 654 frames (night scene)
  - MOT17-11-FRCNN: 900 frames (crowded scene)
  - MOT17-13-FRCNN: 750 frames (street scene)
  - **Note**: Using FRCNN (Faster R-CNN) detections, which are more accurate than DPM
  - **Videos**: Pre-rendered MP4 videos available at `/ssd_4TB/divake/temporal_uncertainty/MOT17/video/`

- **Models**: 5 YOLOv8 variants (pre-trained on COCO)
  - YOLOv8n (nano): 3.2M parameters
  - YOLOv8s (small): 11.2M parameters
  - YOLOv8m (medium): 25.9M parameters
  - YOLOv8l (large): 43.7M parameters
  - YOLOv8x (xlarge): 68.2M parameters

- **GitHub Resources** at `/ssd_4TB/divake/temporal_uncertainty/github_repos/`:
  - Core uncertainty implementations (5 repos)
  - Tracking algorithms (4 repos)
  - Augmentation libraries (3 repos)
  - TTA-specific implementations (2 repos)
  - Papers with code (2 repos)
  - Evaluation metrics (2 repos)
  - **Total: 18 successfully cloned repositories**

### Project Directory Structure
```
/ssd_4TB/divake/temporal_uncertainty/
├── MOT17/
│   ├── train/
│   │   ├── MOT17-02-FRCNN/
│   │   ├── MOT17-04-FRCNN/
│   │   ├── MOT17-05-FRCNN/
│   │   ├── MOT17-09-FRCNN/
│   │   ├── MOT17-10-FRCNN/
│   │   ├── MOT17-11-FRCNN/
│   │   └── MOT17-13-FRCNN/
│   └── video/          # Pre-rendered MP4 videos
├── github_repos/
│   ├── core_uncertainty/
│   ├── tracking_implementations/
│   ├── augmentation_libs/
│   ├── tta_specific/
│   ├── papers_with_code/
│   └── evaluation_metrics/
├── models/             # YOLOv8 models will be downloaded here
└── project_info.md     # This file
```

### What We Are Achieving

**Core Hypothesis**: Aleatoric uncertainty in video object tracking is an inherent property of the data (not the model) that exhibits temporal consistency and can be quantified through test-time augmentation without any training.

**Key Contributions**:
1. Demonstrating that aleatoric uncertainty patterns remain consistent across models of vastly different capacities (3.2M to 68.2M parameters)
2. Showing temporal propagation of uncertainty through video sequences
3. Proving this can be done purely post-hoc without any training
4. Establishing correlation between uncertainty and real-world phenomena (occlusions, motion blur, crowd density)

## Mathematical Framework

### Core Aleatoric Uncertainty Formulation

The fundamental equation for aleatoric uncertainty through Test-Time Augmentation (TTA):

```
σ²_aleatoric(x,t) = Var_T[f(T(x_t))]
```

Where:
- `x_t` is the frame at time t
- `T` is a set of augmentation functions
- `f` is the pre-trained detector
- `Var` computes variance across augmentations

**Extract this from**: `Bayesian-Neural-Networks/experiments/regression/heteroscedastic_dropout.py`

### Temporal Propagation Model

For video sequences, we introduce temporal smoothing:

```
σ²_temporal(t) = α · σ²_current(t) + (1-α) · σ²_temporal(t-1)
```

Where α ∈ [0,1] is the temporal smoothing factor (typically 0.3)

**Implementation reference**: Combine concepts from `boxmot/trackers/bytetrack/byte_tracker.py` with uncertainty propagation

### Bounding Box Uncertainty

For object detection, we decompose uncertainty into:

```
Uncertainty_bbox = {
    'center': Var[cx, cy],
    'size': Var[w, h],
    'total': Var[IoU]
}
```

**Extract IoU variance computation from**: `uncertainty-toolbox/uncertainty_toolbox/metrics.py`

## Augmentation Strategy

### Primary Augmentations (Must Implement)

From `albumentations/albumentations/augmentations/transforms.py`, use these specific functions:

1. **Gaussian Blur**
   ```python
   GaussianBlur(blur_limit=(3,7), p=1.0)
   ```
   Simulates focus/motion issues

2. **Gaussian Noise**
   ```python
   GaussNoise(var_limit=(10.0, 50.0), p=1.0)
   ```
   Simulates sensor noise

3. **Brightness/Contrast**
   ```python
   RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)
   ```
   Simulates lighting variations

4. **Scale Transform**
   ```python
   RandomScale(scale_limit=0.1, p=1.0)
   ```
   Simulates distance variations

5. **JPEG Compression**
   ```python
   ImageCompression(quality_lower=40, quality_upper=100, p=1.0)
   ```
   Simulates compression artifacts

**Implementation Pattern from `ttach/ttach/augmentations.py`**:
- Apply each augmentation
- Get predictions
- Compute variance
- This variance IS your aleatoric uncertainty

## Metrics to Compute

### Primary Metrics

1. **Predictive Variance** (from `uncertainty-toolbox/uncertainty_toolbox/metrics.py`):
   ```python
   def predictive_variance(predictions_list):
       return np.var(predictions_list, axis=0)
   ```

2. **Expected Calibration Error (ECE)**:
   ```python
   def expected_calibration_error(confidences, accuracies, n_bins=10)
   ```
   Shows if uncertainty correlates with actual errors

3. **Mutual Information**:
   ```python
   MI = entropy_of_mean - mean_of_entropy
   ```

4. **Temporal Consistency**:
   ```python
   temporal_correlation = np.corrcoef(uncertainty[:-1], uncertainty[1:])[0,1]
   ```

### Cross-Model Correlation

**Critical for proving aleatoric nature**:
```python
correlation_matrix[i,j] = np.corrcoef(
    uncertainty_model_i,
    uncertainty_model_j
)[0,1]
```

Target: correlation > 0.85 between all model pairs

## Implementation Pipeline

### Step 1: Load Models and Data
```python
# Use ultralytics YOLO
models = {
    'nano': YOLO('yolov8n.pt'),
    'small': YOLO('yolov8s.pt'),
    'medium': YOLO('yolov8m.pt'),
    'large': YOLO('yolov8l.pt'),
    'xlarge': YOLO('yolov8x.pt')
}
```

### Step 2: Augmentation Pipeline
Use `ttach` wrapper pattern but adapt for detection:
```python
# From ttach/ttach/base.py - adapt merge strategy for boxes
transforms = Compose([augmentations...])
predictions = [model(transform(image)) for transform in transforms]
uncertainty = compute_variance(predictions)
```

### Step 3: Tracking Integration
From `boxmot/examples/track.py`:
- Use ByteTrack for association
- Add uncertainty as additional track attribute
- Propagate uncertainty temporally

### Step 4: Analysis Pipeline

Generate these specific analyses:

1. **Uncertainty Heatmaps Over Time**
   - X-axis: Frame number
   - Y-axis: Track ID
   - Color: Uncertainty magnitude

2. **Cross-Model Correlation Matrix**
   - 5x5 matrix showing correlations
   - Should show >0.85 correlation

3. **Uncertainty vs Scene Properties**
   - Plot uncertainty against:
     - Number of objects in frame
     - Average motion magnitude
     - Occlusion events

4. **Temporal Propagation Visualization**
   - Show decay of uncertainty after occlusion
   - Compare different α values

## Key Code Extraction Points

### From `Bayesian-Neural-Networks`:
- File: `experiments/regression/heteroscedastic_dropout.ipynb`
- Extract: TTA implementation loop
- Look for: How they aggregate multiple forward passes

### From `uncertainty-toolbox`:
- File: `uncertainty_toolbox/metrics.py`
- Extract: All uncertainty metrics
- File: `uncertainty_toolbox/viz.py`
- Extract: Visualization functions

### From `ttach`:
- File: `ttach/base.py`
- Extract: Merger classes (how to combine predictions)
- File: `ttach/augmentations.py`
- Extract: Augmentation parameter ranges

### From `albumentations`:
- File: `albumentations/augmentations/transforms.py`
- Extract: Specific augmentation implementations
- Look for: Realistic parameter ranges

### From `boxmot`:
- File: `boxmot/trackers/bytetrack/byte_tracker.py`
- Extract: Tracking logic
- Modify: Add uncertainty as track attribute

## Expected Results

### Primary Findings

1. **Model Consistency**:
   - All 5 models should show uncertainty spikes at same frames
   - Correlation matrix should show >0.85 between all pairs

2. **Temporal Patterns**:
   - Uncertainty should spike during occlusions
   - Gradual decay after reappearance (not instant drop)
   - Persistence during continued occlusion

3. **Scene Correlation**:
   - High uncertainty in crowded scenes
   - Correlation with motion magnitude
   - Higher at frame edges

### Visualization Requirements

1. **Figure 1**: Multi-model uncertainty comparison
   - 5 subplots (one per model)
   - Same sequence showing synchronized spikes

2. **Figure 2**: Temporal propagation
   - Before/during/after occlusion
   - Show decay parameter effect

3. **Figure 3**: Correlation analysis
   - Heatmap of model correlations
   - Scatter plots of uncertainty pairs

4. **Figure 4**: Scene property correlations
   - Uncertainty vs crowd density
   - Uncertainty vs motion
   - Uncertainty vs distance from camera center

## Instructions for Terminal AI

### Freedom to Explore

The terminal AI should feel free to:

1. **Add Additional Augmentations**: If you find interesting augmentations in the cloned repos, try them

2. **Vary Parameters**: Experiment with:
   - Number of augmentations (5-10)
   - Augmentation intensities
   - Temporal smoothing α (0.1-0.9)

3. **Create New Visualizations**: Beyond the required plots, create any visualization that reveals patterns

4. **Statistical Tests**: Run additional tests like:
   - Kolmogorov-Smirnov test for distribution similarity
   - Granger causality for temporal relationships
   - PCA on uncertainty patterns

5. **Ablation Studies**:
   - Try single augmentations vs combinations
   - Test different merge strategies (mean, median, max)
   - Compare different IoU computation methods

### Analysis Extensions

Feel free to compute:

1. **Per-Object Class Analysis**: Does uncertainty vary by object type?

2. **Sequence-Specific Patterns**: Do different MOT17 sequences show different uncertainty characteristics?

3. **Frame Rate Analysis**: Subsample frames to simulate lower FPS - how does uncertainty propagate?

4. **Robustness Tests**: How does uncertainty correlate with:
   - Tracking failures
   - ID switches
   - False positives/negatives

5. **Uncertainty Calibration**: Are high-uncertainty predictions actually less accurate?

## Critical Implementation Notes

### No Training Required
- We are NOT training any networks
- We are NOT modifying model architectures
- We are NOT computing gradients
- This is pure inference-time analysis

### Computational Efficiency
- Use batch processing where possible
- Cache augmented images
- Parallelize across models
- Save intermediate results

### Reproducibility
- Set random seeds
- Document all parameters
- Save configuration files
- Version control results

## Success Criteria

1. **Cross-model correlation > 0.85** proves aleatoric nature
2. **Temporal consistency** shows uncertainty persistence
3. **Scene correlation** validates real-world relevance
4. **Zero training** demonstrates practical applicability
5. **15-30 minute runtime** ensures reproducibility

## Final Notes for Terminal AI

You have access to production-quality code in the cloned repositories. Mix and match the best practices:
- Uncertainty computation from `Bayesian-Neural-Networks`
- Augmentations from `albumentations`
- Metrics from `uncertainty-toolbox`
- Tracking from `boxmot`

The goal is to prove that uncertainty in video object tracking is primarily aleatoric (data-inherent) rather than epistemic (model-dependent), and that this uncertainty exhibits temporal structure that can improve tracking performance.

Be creative in your analysis - the more patterns you discover, the stronger the paper becomes. Focus especially on the temporal aspect, as this is what makes our work novel compared to single-image uncertainty estimation.
