# Implementation Decisions Log

**Project**: Enhanced CACD for Video Object Detection
**Purpose**: Single source of truth for all design decisions
**Last Updated**: 2025-11-10

---

## Design Decisions

### ✅ DECISION 1: Use Only Layer 21 (Final Layer)
**Date**: 2025-11-09
**Decision**: Use ONLY Layer 21 features (256-dim), ignore intermediate layers (4, 9, 15)

**Rationale**:
- YOLO is a pre-trained black box - we don't care about intermediate representations
- Layer 21 (final layer before classification) has best performance: 0.883 correlation
- Simpler, faster, less storage
- Intermediate layers were only for exploration - Layer 21 is best

**Impact**:
- V1/V2/V3 implementations use only Layer 21
- Ignore cached layers 4, 9, 15
- Simplifies code and reduces complexity

---

### ✅ DECISION 2: Within-Sequence Split (Not Cross-Sequence)
**Date**: 2025-11-09
**Decision**: Use within-sequence split for calibration/test. Each sequence is evaluated independently.

**Approach**:
- Each sequence (e.g., MOT17-05): Split 50/50 into calibration and test
- Calibration: 50% of matched detections from that sequence
- Test: 50% of matched detections from same sequence
- Evaluate each of the 7 sequences separately

**Rationale**:
- Simpler: Calibration and test from same distribution
- No normalization needed (same feature scale within sequence)
- Fix basics first before testing cross-sequence generalization
- Current V1 already does this - keep it consistent

**What we're NOT doing (for now)**:
- ❌ Cross-sequence: Train on MOT17-02,04,09,10 → Test on MOT17-05,11,13
- ❌ This requires normalization (feature scales differ between sequences)
- ⏳ Phase 2 (future work)

**Impact**:
- Each sequence gets its own calibration set
- Results are per-sequence (7 separate evaluations)
- Can't test generalization yet, but that's Phase 2

---

### ✅ DECISION 3: 50/50 Calibration/Test Split
**Date**: 2025-11-09
**Decision**: Use 50/50 split ratio for calibration and test sets

**Rationale**:
- Balanced: Enough calibration data for good covariance estimate
- Balanced: Enough test data for reliable evaluation
- V1 already uses this - proven to work
- Can experiment with 70/30 or 60/40 later if needed

**Impact**:
- Example: MOT17-05 with 3,609 detections (conf≥0.5)
  - Calibration: 1,804 detections
  - Test: 1,805 detections

---

### ✅ DECISION 4: No Normalization (Within-Sequence)
**Date**: 2025-11-09
**Decision**: Do NOT normalize features for within-sequence split

**Rationale**:
- Calibration and test are from same sequence → Same distribution
- Feature scale is consistent within sequence
- Normalization not needed, adds unnecessary complexity
- Mahalanobis distance already handles feature correlations

**Note**: If we do cross-sequence (Phase 2), we'll need z-score normalization

**Impact**:
- Simpler code
- Fewer preprocessing steps
- Direct use of raw Layer 21 features

---

### ✅ DECISION 5: Two Splits Only (Calibration + Test)
**Date**: 2025-11-09
**Decision**: Only use calibration and test splits. No training or validation sets.

**Rationale**:
- We're NOT training any model (YOLO is pre-trained)
- Calibration = Learn covariance matrix, fit KDE
- Test = Evaluate uncertainty predictions
- No hyperparameter tuning requiring validation set (using fixed K=10, conf=0.5)

**Traditional ML vs Our Approach**:
```
Traditional ML:
  - Training: Train model weights
  - Validation: Tune hyperparameters
  - Test: Final evaluation

Our Approach:
  - Calibration: Fit covariance, KDE (no gradient descent!)
  - Test: Evaluate uncertainty
```

**Impact**:
- Only 2 data splits needed
- Simpler mental model
- All detections used (50% cal, 50% test)

---

### ✅ DECISION 6: DROP KNN Approach - Use Mahalanobis Distance Method
**Date**: 2025-11-09
**Decision**: Use Mahalanobis distance-based multivariate Gaussian method for aleatoric uncertainty. DROP the KNN-based approach (K=10 neighbors, weighted errors).

**Rationale**:
- ✅ **Theoretically rigorous**: Based on statistical theory (Gaussian distribution)
- ✅ **No arbitrary hyperparameters**: No need to choose K (was K=10 arbitrarily)
- ✅ **Conference-ready**: Well-established method, easy to justify in paper
- ✅ **Matches human perception**: Paper shows uncertainty scores align with visual perception
- ✅ **Simpler**: One global distribution vs local neighborhoods

**What we're dropping**:
- ❌ KNN-based approach (find K=10 nearest neighbors)
- ❌ Weighted average of neighbor errors
- ❌ Arbitrary choice of K value

---

### ✅ DECISION 7: Mahalanobis Distance-Based Aleatoric Uncertainty (Full Method)
**Date**: 2025-11-09
**Decision**: Implement aleatoric uncertainty using Mahalanobis distance from multivariate Gaussian distribution

**Based on paper**: "Mahalanobis Distance-based Multivariate Gaussian Distribution-based Aleatoric Uncertainty"

---

#### **Method Overview**

**Core Idea**:
- Fit a multivariate Gaussian distribution to calibration features
- Measure how far each test sample is from the "typical" distribution
- Distance = Aleatoric uncertainty (far from typical = high uncertainty)

---

#### **Mathematical Formulation**

**Step 1: Fit Multivariate Gaussian (Calibration Phase)**

Given calibration features V(z₁), V(z₂), ..., V(zₙ) ∈ ℝᴰ (where D=256 for Layer 21):

**Equation 1 - Mean vector:**
```
μ = (1/N) Σᵢ V(zᵢ)                                    [D-dimensional vector]
```

**Equation 2 - Covariance matrix:**
```
Σ = (1/N) Σᵢ (V(zᵢ) - μ)(V(zᵢ) - μ)ᵀ                [D×D matrix]
```

**Regularization** (to prevent singular matrix):
```
Σ_reg = Σ + λ·(tr(Σ)/D)·I                            [λ = 1e-4]
```
Where:
- λ = regularization parameter
- tr(Σ) = trace of Σ (sum of diagonal elements)
- I = identity matrix
- This adds small values to diagonal to ensure invertibility

---

**Step 2: Compute Mahalanobis Distance (Test Phase)**

For each test sample V(x):

**Equation 3 - Mahalanobis distance:**
```
M(x) = √[(V(x) - μ)ᵀ Σ⁻¹ (V(x) - μ)]
```

**Interpretation:**
- M(x) = 0: Sample is exactly at mean (perfectly typical)
- Small M(x): Sample is close to mean (typical, low uncertainty)
- Large M(x): Sample is far from mean (unusual, high uncertainty)

**Why Mahalanobis (not Euclidean)?**
- Accounts for feature correlations via Σ⁻¹
- Scales each dimension by its variance
- More robust to correlated features

---

**Step 3: Normalize to [0, 1] Range**

Raw Mahalanobis distances can vary widely (0 to 1000+). We normalize for interpretability.

**Equation 4 - Log transform + Min-Max normalization:**
```
d(x) = [log(M(x)) - min_j{log M(z_j)}] / [max_j{log M(z_j)} - min_j{log M(z_j)}]
```

Where the min/max are computed over the calibration set.

**Why log transform?**
- Raw M can have very large outliers
- log(M) compresses the range
- Makes normalization more stable
- Similar to log-likelihood in probabilistic interpretation

**Result**: d(x) ∈ [0, 1]
- 0 = Most typical (lowest uncertainty)
- 1 = Most unusual (highest uncertainty)

---

#### **Two Uncertainty Outputs**

**1. Raw Mahalanobis Distance (for evaluation)**
```python
uncertainty_raw = M(x)²  # or just M(x)
```
- Use this for correlation with ground truth errors
- More direct relationship to actual errors
- Better for evaluation metrics

**2. Normalized Score (for interpretation)**
```python
uncertainty_normalized = d(x)  # from Equation 4, in [0, 1]
```
- Use this for visualization and interpretation
- Easy to understand: 0=easy, 1=hard
- Can categorize: low (0-0.3), medium (0.3-0.7), high (0.7-1.0)

---

#### **Uncertainty Interpretation (from Paper)**

| Score Range | Category | Meaning | Characteristics |
|-------------|----------|---------|-----------------|
| **0.0 - 0.3** | Low uncertainty | Easy sample | Clear, unambiguous, typical features. Readily recognized by humans/models |
| **0.3 - 0.7** | Medium uncertainty | Challenging sample | Distant objects, partially occluded, harder to detect but valid |
| **0.7 - 1.0** | High uncertainty | Noisy/Ambiguous | Low quality, unrecognizable, misleading annotations, data noise |

**From paper's MS-COCO analysis:**
- ~5-10% of samples have high uncertainty (0.7-1.0)
- Majority in medium range (0.3-0.7)
- Uncertainty scores match human visual perception

---

#### **Single Class vs Multi-Class**

**Paper's Approach (Multi-Class):**
- For each class c (person, car, dog, etc.):
  - Fit separate μ_c and Σ_c
  - Normalize within each class separately
  - Class-conditional Gaussian

**Our Approach (Single Class):**
- MOT17 only has ONE class: "person" (pedestrian)
- Fit single μ and Σ for all detections
- Normalize across all detections
- **Simpler and appropriate for our use case**

**Equation 3 simplifies to:**
```
M(x) = √[(V(x) - μ)ᵀ Σ⁻¹ (V(x) - μ)]
```
(No class-conditional notation needed)

---

#### **Complete Algorithm**

```
Algorithm: Mahalanobis Distance-Based Aleatoric Uncertainty

INPUT:
  - Calibration features: X_cal = [N_cal, D] from Layer 21
  - Test features: X_test = [N_test, D] from Layer 21
  - Regularization: λ = 1e-4

CALIBRATION PHASE:
  1. Compute mean vector:
     μ = mean(X_cal)                           [D-dim vector]

  2. Compute covariance matrix:
     Σ = cov(X_cal)                            [D×D matrix]

  3. Regularize covariance:
     reg_val = λ · trace(Σ) / D
     Σ_reg = Σ + reg_val · I                   [Prevent singular matrix]

  4. Compute inverse:
     Σ_inv = inverse(Σ_reg)                    [D×D matrix]

  5. Compute calibration distances (for normalization):
     For each x_i in X_cal:
       M_cal[i] = √[(x_i - μ)ᵀ Σ_inv (x_i - μ)]

     log_M_cal = log(M_cal + ε)                [ε=1e-10 to avoid log(0)]
     log_M_min = min(log_M_cal)
     log_M_max = max(log_M_cal)

     Store: μ, Σ_inv, log_M_min, log_M_max

TEST PHASE:
  For each test sample x in X_test:

    1. Compute Mahalanobis distance (Equation 3):
       M(x) = √[(x - μ)ᵀ Σ_inv (x - μ)]

    2a. Raw uncertainty (for evaluation):
        uncertainty_raw = M(x)²

    2b. Normalized uncertainty (Equation 4, for interpretation):
        log_M = log(M(x) + ε)
        d_norm = (log_M - log_M_min) / (log_M_max - log_M_min)
        d_norm = clip(d_norm, 0, 1)            [Ensure [0,1] range]
        uncertainty_normalized = d_norm

OUTPUT:
  - uncertainty_raw: [N_test] - for correlation with errors
  - uncertainty_normalized: [N_test] in [0,1] - for interpretation
```

---

#### **What We Need**

**Inputs:**
1. Layer 21 features (256-dim) from YOLO
2. Ground truth errors (center_error in pixels) - for evaluation only
3. Confidences - for filtering (conf ≥ 0.5)

**Outputs:**
1. Raw uncertainty (M²) - correlate with errors²
2. Normalized uncertainty (d ∈ [0,1]) - interpret as low/medium/high

**No need for:**
- ❌ K value (no KNN)
- ❌ Bandwidth (no KDE yet - that's for epistemic)
- ❌ Neighbor search algorithms

---

#### **Success Criteria**

**Quantitative:**
- Correlation(uncertainty_raw, errors²) > 0.7 on most sequences
- Fix MOT17-05 failure (currently -0.473 with old KNN method)

**Qualitative:**
- Low uncertainty (0-0.3): Clear, well-detected objects
- Medium uncertainty (0.3-0.7): Partially occluded, distant objects
- High uncertainty (0.7-1.0): Ambiguous, noisy detections

**Distribution check:**
- ~5-10% should have high uncertainty (like paper's MS-COCO)
- Should match human perception

---

#### **Implementation Notes**

**Computational Complexity:**
- Calibration: O(N·D²) for covariance + O(D³) for matrix inversion
  - One-time cost per sequence
  - For D=256, N=1804: ~1 second

- Test: O(D²) per sample for Mahalanobis distance
  - Very fast: ~0.1ms per detection

**Memory:**
- Store μ: [256] floats = 1 KB
- Store Σ_inv: [256, 256] floats = 256 KB
- Store log_M_min, log_M_max: 2 floats
- Total: ~260 KB per sequence (minimal!)

**Numerical Stability:**
- Add regularization to prevent singular Σ
- Add ε=1e-10 before log to avoid log(0)
- Clip normalized scores to [0, 1]
- Use np.maximum(M_squared, 0) before sqrt

---

#### **Advantages Over KNN Approach**

| Aspect | Mahalanobis (Our Choice) | KNN (Dropped) |
|--------|-------------------------|---------------|
| **Theory** | Rigorous (Gaussian assumption) | Heuristic |
| **Hyperparameters** | Only λ (standard 1e-4) | K=10 (arbitrary) |
| **Computation** | O(D²) per test sample | O(N·D) per test sample |
| **Interpretability** | Distance from typical distribution | Average of neighbor errors |
| **Conference Appeal** | ✅ High (well-established) | ⚠️ Medium (ad-hoc) |
| **Paper Support** | ✅ Published method | ❌ Our invention |
| **Scalability** | ✅ Constant time per sample | ⚠️ Linear in N |

---

### ✅ DECISION 8: Detection-Level First, Then Track-Level
**Date**: 2025-11-09
**Decision**: Start with detection-level uncertainty (frame-by-frame), then add track-level temporal analysis

**Two Phases:**

**Phase 1: Detection-Level Uncertainty (NOW)**
- Analyze EACH detection in EACH frame independently
- No temporal component yet
- Question: "How uncertain is THIS detection in THIS frame?"

**Data:**
- All matched detections across all sequences
- Example MOT17-05: 3,609 detections (after conf≥0.5 filter)
- Mixed from different people, different frames

**Purpose:**
- Validate the Mahalanobis method works
- Get correlation with conformity scores
- Larger sample size → stronger validation

---

**Phase 2: Track-Level Temporal Analysis (LATER)**
- Analyze uncertainty evolution for SPECIFIC tracks over time
- Use long tracks from metadata: `/ssd_4TB/divake/temporal_uncertainty/metadata`
- Question: "How does uncertainty evolve for this person across frames?"

**Data:**
- Focus on long-lived tracks (e.g., Track 25 in MOT17-11: 900 frames)
- Track detections temporally

**Purpose:**
- Temporal consistency: Does uncertainty spike during occlusions?
- Predictive power: Can uncertainty predict track loss?
- Smooth evolution: Is uncertainty temporally coherent?

**Rationale:**
- Detection-level validates the method works at all
- More data (all detections) → robust correlation
- Simpler to implement first
- Track-level adds temporal dimension after basics work

---

### ✅ DECISION 9: Use 1-IoU as Conformity Score (Not Center Distance)
**Date**: 2025-11-09
**Decision**: Use IoU-based conformity score as target variable, NOT center distance

**Conformity Score Definition:**
```
conformity_score = 1 - IoU
```

Where:
- **IoU** ∈ [0, 1]: Intersection over Union between predicted bbox and ground truth bbox
- **conformity_score** ∈ [0, 1]: Non-conformity score
  - 0 = Perfect conformity (IoU = 1.0, boxes match perfectly)
  - 0.5 = Poor conformity (IoU = 0.5, at match threshold)
  - 1 = No conformity (IoU = 0.0, no overlap)

**Why 1-IoU?**

1. **Conformal Prediction Framework:**
   - Standard non-conformity score for object detection
   - Used in recent research (2024-2025 papers)
   - Higher score = worse conformity (correct orientation)

2. **Captures Full Detection Quality:**
   - ✅ Position errors (bbox center offset)
   - ✅ Size errors (bbox too large/small)
   - ✅ Shape errors (aspect ratio mismatch)
   - ✅ Handles case where center matches but box size is wrong

3. **Already Available:**
   - IoU stored in cache: `gt_matching/iou`
   - Just compute `1 - iou`
   - No additional computation needed

**What We're NOT Using:**

**Center Distance** (previously considered):
```
center_error = √[(cx_pred - cx_gt)² + (cy_pred - cy_gt)²]
```

**Why NOT center distance:**
- ❌ Ignores box size (can be 0 even if box is wrong size)
- ❌ Doesn't capture shape mismatch
- ❌ Not standard in conformal prediction literature
- ❌ Weaker measure of detection quality

**Example illustrating the problem:**
```
Ground Truth:  [100, 100, 200, 200]  (100×100 box, center at 150,150)
Prediction:    [140, 140, 160, 160]  (20×20 box, center at 150,150)

Center distance: 0 pixels  ← Looks perfect!
IoU: 0.04                  ← Correctly shows poor match
1 - IoU: 0.96             ← High non-conformity (correct!)
```

**Implementation:**
```python
# Load IoU from cache
ious = loader.get_ious()  # [N] in [0, 1]

# Compute conformity scores (target variable)
conformity_scores = 1 - ious  # [N] in [0, 1]

# Predicted uncertainty from Mahalanobis
predicted_uncertainty = M(x)²  # or normalized version

# Evaluate correlation
correlation = pearsonr(predicted_uncertainty, conformity_scores)[0]
```

**Success Criterion:**
```
correlation(predicted_uncertainty, conformity_scores) > 0.7
```

High predicted uncertainty should correlate with high actual non-conformity (low IoU).

**Terminology:**
- Use "conformity score" or "non-conformity score" (not "error")
- Aligns with conformal prediction literature
- More precise than generic "error"

---

### ✅ DECISION 10: Confidence Threshold = 0.5 (Start Here)
**Date**: 2025-11-09
**Decision**: Use confidence threshold ≥ 0.5 for initial experiments

**Filter pipeline:**
```
1. Matched detections: IoU ≥ 0.5 with ground truth (already in cache)
2. Confidence filter: conf ≥ 0.5
3. Result: High-quality true positive detections
```

**Rationale:**
- ✅ Standard threshold in detection literature
- ✅ Reasonable balance: quality vs quantity
- ✅ Removes low-confidence false positives
- ✅ Still sufficient data (see table below)

**Data Availability:**

| Sequence | At conf≥0.5 | At conf≥0.3 | At conf≥0.7 |
|----------|-------------|-------------|-------------|
| MOT17-02 | 3,024 (30%) | 3,715 (37%) | 2,087 (21%) |
| MOT17-05 | 3,609 (68%) | 3,929 (74%) | 3,037 (57%) |
| MOT17-11 | 5,187 (77%) | 5,527 (82%) | 4,542 (67%) |
| MOT17-13 | 1,963 (26%) | 2,766 (36%) | 697 (9%) |

**Future experiments:**
- If 0.5 works well: Try 0.3 to get more data
- If 0.5 is too strict: Lower to 0.3
- If 0.5 is too loose: Raise to 0.7

**Why NOT 0.3 initially:**
- More data but might include marginal detections
- Test 0.5 first, then relax if needed

**Why NOT 0.7 initially:**
- Too restrictive (MOT17-13: only 697 samples!)
- Might lose informative medium-confidence detections

---

### ✅ DECISION 11: Start with MOT17-11 Sequence
**Date**: 2025-11-10
**Decision**: Use MOT17-11 as our initial validation sequence

**Rationale** (from cache analysis):
- **Highest IoU quality**: 0.8157 mean IoU (best among all sequences)
- **82.4% excellent detections**: IoU ≥ 0.7
- **Strongest confidence-IoU correlation**: 0.6561 (confidence is most reliable predictor)
- **Sufficient samples**: 5,187 detections at conf≥0.5
- **Low conformity scores**: Mean 1-IoU = 0.184 (good matches, clear signal)
- **Moderate feature norms**: 4.08 ± 1.34 (not extreme like MOT17-05: 3.76)

**Why NOT MOT17-05:**
- Lowest feature norms (3.76) - different distribution
- Previous V1 showed catastrophic failure (-0.473 correlation)
- Need clean baseline first

**Why NOT MOT17-13:**
- Worst IoU quality (0.7054) - only 51% excellent detections
- Most challenging sequence
- Test after validating method on MOT17-11

**Impact:**
- Single sequence focus allows methodical development
- High-quality data validates Mahalanobis method works
- Can expand to other sequences after success
- Use MOT17-11 cache: `/ssd_4TB/divake/temporal_uncertainty/yolo_cache/MOT17-11.npz`

---

### ✅ DECISION 12: Clean Modular Architecture
**Date**: 2025-11-10
**Decision**: Organize code with clear separation of concerns

**Directory Structure:**
```
conformal_tracking/
├── src/                          # Core algorithms (dataset-agnostic)
│   ├── uncertainty/
│   │   └── mahalanobis.py       # Mahalanobis uncertainty class
│   └── data_loader.py           # Generic cache loader (kept from before)
│
├── data_loaders/                 # Dataset-specific I/O
│   └── mot17_loader.py          # MOT17-specific logic
│
├── config/                       # YAML configuration files
│   ├── datasets.yaml            # Dataset paths, splits
│   ├── models.yaml              # Model hyperparameters
│   └── experiment.yaml          # Experiment settings
│
├── experiments/                  # Experiment scripts (glue code)
│   └── run_aleatoric_mot17_11.py
│
├── results/                      # Outputs
│   ├── cache_analysis/
│   └── plots/
│
├── *.md                          # Documentation
└── analyze_cache_before_implementation.py  # Analysis script
```

**Principles:**
1. **Core code never changes**: `src/uncertainty/` is generic, reusable
2. **Dataset-specific in data_loaders/**: Easy to add new datasets
3. **Config-driven**: Hyperparameters in YAML, not hardcoded
4. **Experiments as glue**: Thin scripts that connect components

**Separation of Concerns:**
- `src/uncertainty/mahalanobis.py`: Pure algorithm (fit, predict, normalize)
- `data_loaders/mot17_loader.py`: Load MOT17 cache, apply filters, split data
- `config/`: All hyperparameters and paths
- `experiments/`: Run full pipeline, save results

**Benefits:**
- Easy to test components in isolation
- Easy to swap datasets (add `data_loaders/kitti_loader.py`)
- Easy to version control configs
- Professional, conference-ready code

---

## Open Questions

*(None currently)*

---

## Next Decisions to Make

1. When to add epistemic uncertainty (KDE-based)?
2. How to combine aleatoric + epistemic for total uncertainty?
3. Temporal smoothing: Kalman filter parameters?

---

## Summary of All Decisions

1. ✅ Layer 21 only (no intermediate layers)
2. ✅ Within-sequence split (not cross-sequence)
3. ✅ 50/50 calibration/test split
4. ✅ No feature normalization (within-sequence)
5. ✅ Two splits only (calibration + test)
6. ✅ DROP KNN approach (use Mahalanobis instead)
7. ✅ Mahalanobis distance-based aleatoric uncertainty
8. ✅ Detection-level first, then track-level
9. ✅ Use 1-IoU as conformity score (not center distance)
10. ✅ Confidence threshold = 0.5
11. ✅ Start with MOT17-11 sequence (best quality)
12. ✅ Clean modular architecture (src/, data_loaders/, config/, experiments/)

**Status: Directory cleaned, architecture defined, ready to implement!** 🚀

---

**Instructions**: Add new decisions here as we discuss. Keep it concise and actionable.
