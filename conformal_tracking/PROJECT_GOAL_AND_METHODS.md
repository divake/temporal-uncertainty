# Project Goal and Methods: Complete Understanding

**Date**: November 9, 2025
**Purpose**: Define what we're building, why, and how

---

## 🎯 The Core Goal

### What We're Trying to Achieve

**Build a system that can answer this question:**

> **"How uncertain is this detection, and why?"**

For **every single detection** in a video tracking sequence, we want to know:
1. **How uncertain** is this detection? (magnitude)
2. **Why** is it uncertain? (decomposition into sources)
   - **Aleatoric**: Uncertainty from the **data itself** (noise, occlusion, blur)
   - **Epistemic**: Uncertainty from the **model** (lack of knowledge, OOD)

### Real-World Use Case

**Scenario**: Autonomous vehicle detecting pedestrians in a video stream

**Frame 1**: Car detects a person crossing the street
- **Question**: How confident should the car be in this detection?
- **Our system answers**:
  - Total uncertainty: **High** (0.8/1.0)
  - Aleatoric: **High** (0.7) ← Person is partially occluded by a pole
  - Epistemic: **Low** (0.1) ← Model knows what pedestrians look like
  - **Interpretation**: "The person is there, but I can't see them fully due to occlusion"

**Frame 2**: Car detects a unusual object (someone in costume)
- **Question**: How confident should the car be?
- **Our system answers**:
  - Total uncertainty: **High** (0.9/1.0)
  - Aleatoric: **Low** (0.2) ← Object is clearly visible, no occlusion
  - Epistemic: **High** (0.7) ← Model hasn't seen this type of object before (OOD)
  - **Interpretation**: "I can see it clearly, but I don't know what it is"

**The Value**: By **decomposing uncertainty**, we know **why** the system is uncertain, which helps us:
- Make better decisions (slow down for occlusions, stop for unknown objects)
- Identify when to request human intervention
- Understand failure modes

---

## 🧠 Understanding Uncertainty Decomposition

### What is Aleatoric vs Epistemic Uncertainty?

Think of it like this:

**Aleatoric Uncertainty (Data Uncertainty)**
- **Definition**: Uncertainty that comes from the **inherent randomness** in the data
- **Cannot be reduced** by collecting more training data or using a better model
- **Examples**:
  - Sensor noise (camera grain, motion blur)
  - Occlusions (person behind a pole)
  - Ambiguity (is that a dog or a cat from far away?)
  - Lighting variations (person in shadow)

**Epistemic Uncertainty (Model Uncertainty)**
- **Definition**: Uncertainty that comes from the **model's lack of knowledge**
- **Can be reduced** by training on more data or using an ensemble
- **Examples**:
  - Out-of-distribution (OOD) objects (never seen this before)
  - Sparse training data (few examples of this class)
  - Model capacity (too small to capture patterns)

### Mathematical Formulation

**Total Uncertainty** = Aleatoric + Epistemic

```
σ²_total(x) = σ²_aleatoric(x) + σ²_epistemic(x)
```

**Where:**
- `x` = input (detection, features)
- `σ²_aleatoric` = variance due to data noise
- `σ²_epistemic` = variance due to model uncertainty

### Why Do We Care About Decomposition?

**Without decomposition:**
- System says: "Uncertainty = 0.8" → What does this mean? What should we do?

**With decomposition:**
- System says: "Aleatoric = 0.7, Epistemic = 0.1"
  - **Action**: Slow down and track carefully (data is noisy but object is known)
- System says: "Aleatoric = 0.1, Epistemic = 0.7"
  - **Action**: Stop and request human review (unknown object type!)

---

## 📊 Our Application: Video Object Detection & Tracking

### The Setup

**Input**: Video sequence (e.g., MOT17-05 with 837 frames)

**Process**:
1. Run YOLO detector on each frame
2. For each detection (bounding box):
   - Extract features from YOLO's internal layers
   - Compute uncertainty (total, aleatoric, epistemic)
3. Track detections across frames (temporal consistency)

**Output**: For each detection:
```python
{
  'frame_id': 42,
  'bbox': [x1, y1, x2, y2],
  'confidence': 0.85,  # YOLO's native confidence
  'total_uncertainty': 0.65,
  'aleatoric_uncertainty': 0.50,
  'epistemic_uncertainty': 0.15,
  'predicted_error': 8.5  # pixels
}
```

### Frame-by-Frame Analysis

**Goal**: Understand complexity of each frame in the sequence

**Questions we can answer:**
1. **Which frames are hardest?**
   - High aleatoric → Occlusions, crowding, motion blur
   - High epistemic → Unusual objects, lighting, camera angle

2. **How does uncertainty evolve over time?**
   - Track a person across frames
   - Uncertainty should be **temporally consistent** (not random!)
   - Spikes in uncertainty → Occlusion events, interactions

3. **What makes a sequence difficult?**
   - Compare MOT17-05 vs MOT17-13
   - Is it data complexity (aleatoric) or distribution shift (epistemic)?

### The Challenge: Post-Hoc Uncertainty (No Training!)

**Key Constraint**: We use **pre-trained YOLO** without retraining

**Why?**
- Real-world deployment: Can't retrain models for every scenario
- Zero-shot uncertainty: Works on any detector
- Computationally efficient: Just feature extraction

**How?**
- Use YOLO's **internal features** (layers 4, 9, 15, 21)
- Apply **conformal prediction** and **statistical methods**
- No gradient descent, no backprop!

---

## 🔧 The Three Methods: V1, V2, V3

### Overview

| Method | Focus | Complexity | Status |
|--------|-------|------------|--------|
| **V1** | Basic decomposition (KNN + KDE) | Simple | ✅ Complete |
| **V2** | Better epistemic (multi-source) | Medium | 🔄 Next |
| **V3** | Robustness (local scaling, temporal) | High | ⏳ Future |

---

## 🔬 V1 Enhanced CACD: The Foundation

### What V1 Does

**Goal**: Decompose uncertainty into aleatoric and epistemic using simple distance-based methods

**Method Overview**:
1. **Calibration Phase**: Learn from "calibration set" (half of detections)
   - Extract features from YOLO layer (e.g., layer 9: 256 dimensions)
   - Store features and their **true errors** (distance from GT)

2. **Test Phase**: For each new detection
   - Extract features
   - Compute **aleatoric uncertainty** using K-nearest neighbors
   - Compute **epistemic uncertainty** using kernel density estimation
   - Predict expected error

### V1 Components Explained

#### 1. Features (from YOLO layers)

**What are features?**
- YOLO processes images through multiple layers (backbone → neck → head)
- Each layer produces **feature maps** (numerical representations)
- We extract features at 4 specific layers:
  - **Layer 4** (early): Low-level (edges, textures) - 64 dim
  - **Layer 9** (mid): Mid-level (parts, shapes) - 256 dim
  - **Layer 15** (late): High-level (objects) - 64 dim
  - **Layer 21** (final): Pre-classification (semantic) - 256 dim

**Why use features?**
- Features capture **what YOLO sees** before making a decision
- Similar features → Similar objects → Similar errors (hopefully!)
- Different layers capture different levels of abstraction

**Example**:
- Two detections with similar Layer 9 features:
  - Both are "person-shaped"
  - If one has high error, the other likely does too
  - This is the basis of KNN!

#### 2. K-Nearest Neighbors (KNN)

**What it does**: Find the K most similar past examples

**How it works**:
```
1. Calibration: Store all calibration features + errors
2. Test: For new detection with features x_test:
   a. Compute distance to all calibration points
   b. Find K closest neighbors
   c. Weight neighbors by distance (closer = more weight)
   d. Average their errors → Aleatoric uncertainty
```

**Why KNN for aleatoric?**
- **Local errors**: Objects with similar features have similar noise
- **Data-driven**: Captures actual error patterns, not assumptions
- **Interpretable**: "This looks like these K past examples"

**Example**:
```
Test detection: Person partially occluded
K=5 nearest neighbors in calibration:
  1. Person occluded by pole: error=12.3 pixels (distance=0.2)
  2. Person occluded by car: error=10.8 pixels (distance=0.3)
  3. Person partially visible: error=9.5 pixels (distance=0.4)
  4. Clear person: error=3.2 pixels (distance=0.5)
  5. Clear person: error=2.8 pixels (distance=0.6)

Weighted average error (aleatoric): ~8.5 pixels
```

#### 3. Mahalanobis Distance (Not Euclidean)

**What it is**: Distance metric that accounts for **feature correlations**

**Euclidean distance** (naive):
```
d = √[(x1-y1)² + (x2-y2)² + ... + (xD-yD)²]
```
- Treats all dimensions equally
- Ignores correlations between features

**Mahalanobis distance** (smarter):
```
d = √[(x-y)ᵀ Σ⁻¹ (x-y)]
```
- Where Σ = covariance matrix of features
- Accounts for **how features vary together**
- More robust to correlated dimensions

**Why Mahalanobis?**
- YOLO features are **highly correlated** (not independent)
- Euclidean distance would be mislead by redundant dimensions
- Mahalanobis "stretches" space to decorrelate features

**Example**:
```
Two features: [brightness, contrast]
- These are correlated (bright images tend to have high contrast)
- Euclidean: Would count them as separate dimensions
- Mahalanobis: Recognizes they're related, adjusts distance accordingly
```

#### 4. Kernel Density Estimation (KDE)

**What it does**: Estimate how "common" a feature vector is

**How it works**:
```
1. Calibration: Fit a density model to all calibration features
2. Test: For new detection with features x_test:
   a. Compute density p(x_test)
   b. Low density → Rare/unusual → High epistemic uncertainty
   c. High density → Common/typical → Low epistemic uncertainty
```

**Why KDE for epistemic?**
- **Novelty detection**: Rare features → Model hasn't seen many examples
- **OOD detection**: Very low density → Out of distribution
- **Non-parametric**: No assumptions about distribution shape

**Example**:
```
Calibration features: Mostly frontal person views
Test detection 1: Frontal person view
  → High density (p=0.85) → Common → Low epistemic (ε=0.1)

Test detection 2: Person from unusual angle (top-down)
  → Low density (p=0.05) → Rare → High epistemic (ε=0.7)
```

### V1 Algorithm Step-by-Step

**Calibration (offline, once per sequence)**:
```python
Input:
  - X_cal: [N_cal, D] calibration features
  - y_cal: [N_cal] ground truth errors

Steps:
1. Compute covariance matrix Σ from X_cal
2. Regularize: Σ_reg = Σ + λ·I (avoid singular matrix)
3. Compute inverse Cholesky: L⁻¹ (for fast Mahalanobis)
4. Fit KDE on X_cal
5. Store X_cal, y_cal, L⁻¹, KDE model
```

**Prediction (online, for each test detection)**:
```python
Input:
  - x_test: [D] test features

Steps:
1. Compute Mahalanobis distances to all calibration points:
   d_i = ||L⁻¹(x_test - x_cal_i)||₂

2. Find K nearest neighbors (smallest d_i)

3. Compute softmax weights:
   w_i = exp(-d_i) / Σ exp(-d_k)  (for k in K neighbors)

4. Aleatoric uncertainty:
   σ²_aleatoric = Σ w_i · (y_cal_i)²

5. Epistemic uncertainty:
   density = KDE.score(x_test)
   σ²_epistemic = 1 / density  (inverse density)

6. Total uncertainty:
   σ²_total = σ²_aleatoric + σ²_epistemic

Output:
  - σ²_aleatoric, σ²_epistemic, σ²_total
```

### V1 Strengths and Weaknesses

**Strengths** ✅:
- Simple and interpretable
- No training required
- Fast inference (~0.1ms per detection)
- Works well on 6/7 sequences (mean corr = 0.723)

**Weaknesses** ❌:
- Fails on MOT17-05 (-0.473 correlation)
- Weak epistemic signal (100% aleatoric fraction)
- Sensitive to feature scale and distribution
- KNN can be mislead by outliers
- Single-source epistemic (only inverse density)

---

## 🔬 V2 Enhanced CACD: Multi-Source Epistemic

### What V2 Adds

**Goal**: Fix V1's weak epistemic signal by combining multiple sources

**Key Insight**: Epistemic uncertainty should capture **multiple aspects** of model uncertainty:
1. **Rarity** (inverse density) ← V1 already has this
2. **Distance to training data** (min distance)
3. **Neighborhood diversity** (entropy)

**Why multi-source?**
- Single metric (density) might miss some OOD cases
- Combining sources → More robust epistemic signal
- Learned weights → Optimal combination

### V2 Components

#### 1. Source 1: Inverse Density (from V1)
```python
ε₁(x) = 1 / KDE.score(x)
```
- Low density → High epistemic
- Kept from V1

#### 2. Source 2: Minimum Mahalanobis Distance
```python
ε₂(x) = min(d_mahalanobis(x, x_cal_i))  for all i
```
- Large distance to nearest calibration point → OOD → High epistemic
- Complements density (density might be deceived by outliers)

#### 3. Source 3: Neighborhood Entropy
```python
# For K nearest neighbors:
ε₃(x) = -Σ p_k · log(p_k)
```
- Where p_k = weight of k-th neighbor
- High entropy → Neighbors are diverse → Uncertain region → High epistemic
- Low entropy → Neighbors are similar → Confident region → Low epistemic

#### 4. Learned Combination
```python
σ²_epistemic = w₁·ε₁ + w₂·ε₂ + w₃·ε₃
```
- Where w₁, w₂, w₃ ≥ 0 and w₁ + w₂ + w₃ = 1
- Optimize weights to maximize correlation with errors
- Use SLSQP (constrained optimization)

### V2 Expected Benefits

1. **Stronger epistemic signal**: Multiple sources → Higher variance
2. **Better orthogonality**: Aleatoric and epistemic more independent
3. **Robustness**: If one source fails, others compensate
4. **OOD detection**: Catches cases V1 misses (e.g., MOT17-05?)

---

## 🔬 V3 Enhanced CACD: Robustness + Temporal

### What V3 Adds

**Goal**: Make the system **robust** to outliers and use **temporal information**

**Key Additions**:
1. **Robust statistics** (median instead of mean)
2. **Local scaling** (adapt to different scene regions)
3. **Temporal propagation** (smooth uncertainty over time)

### V3 Components

#### 1. Robust Statistics

**Problem with V1/V2**: Mean and covariance are sensitive to outliers

**V3 Solution**:
- Use **median** instead of mean for KNN aggregation
- Use **MinCovDet** for robust covariance estimation
- Use **Huber loss** for distance weighting (downweight extremes)

**Why?**
- MOT17-05 might have outliers in calibration
- Robust methods prevent outliers from dominating

#### 2. Local Scaling (Decision Tree Partitioning)

**Problem**: Scenes have different regions with different characteristics
- Example: People near camera vs far from camera have different feature scales

**V3 Solution**:
```
1. Build decision tree on calibration features
2. Each leaf = local region with scaling factor ξ_k
3. Test point falls into leaf k → Apply scaling ξ_k
```

**Why?**
- Adapts to local feature distributions
- Different scenes (MOT17-05 vs MOT17-13) get different scalings

#### 3. Temporal Propagation (Kalman Filter)

**Problem**: Frame-by-frame uncertainty can be noisy

**V3 Solution**:
```python
# Kalman filter for uncertainty:
σ²_temporal(t) = α·σ²_current(t) + (1-α)·σ²_temporal(t-1)

# With process noise:
Q = structured_noise_matrix  # Models natural uncertainty changes
```

**Why?**
- Uncertainty should be **temporally consistent**
- Sudden spikes → Occlusion events (expected)
- Random noise → Smoothed out

### V3 Expected Benefits

1. **Fix MOT17-05**: Robust to distribution shift
2. **Smooth temporal evolution**: Easier to interpret trends
3. **Adapt to scenes**: Local scaling handles diversity

---

## 🎯 Putting It All Together

### The Pipeline (Complete System)

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Video Sequence (837 frames)                              │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Run YOLO on all frames                                  │
│ - Extract bounding boxes                                        │
│ - Extract features from layers 4, 9, 15, 21                     │
│ - (Cached in our setup!)                                        │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Match detections to ground truth                        │
│ - Compute IoU with GT                                           │
│ - Compute center errors (pixels)                                │
│ - (Cached in our setup!)                                        │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Split into calibration and test                         │
│ - Calibration: 50% of matched detections                        │
│ - Test: 50% of matched detections                               │
│ - (Random split with fixed seed)                                │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Calibration (V1/V2/V3)                                  │
│ - Fit Mahalanobis distance metric                               │
│ - Fit KDE for density estimation                                │
│ - (V2: Fit multi-source weights)                                │
│ - (V3: Fit decision tree for local scaling)                     │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Prediction on test set                                  │
│ - For each test detection:                                      │
│   • Compute aleatoric uncertainty (KNN)                         │
│   • Compute epistemic uncertainty (KDE + more in V2)            │
│   • (V3: Apply temporal smoothing)                              │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: Evaluation                                              │
│ - Correlate uncertainty with true errors                        │
│ - Check orthogonality (aleatoric ⊥ epistemic)                   │
│ - Analyze per-frame complexity                                  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: Per-detection uncertainty decomposition                 │
│ {                                                                │
│   frame_id, bbox,                                               │
│   σ²_aleatoric, σ²_epistemic, σ²_total                          │
│ }                                                                │
└─────────────────────────────────────────────────────────────────┘
```

### What We Can Learn

**From a single sequence (e.g., MOT17-05)**:

1. **Hardest frames**:
   - Frames with highest aleatoric → Crowding, occlusions
   - Frames with highest epistemic → Unusual scenes, lighting

2. **Tracking difficulty**:
   - Track uncertainty over time for specific objects
   - Occlusion events → Uncertainty spikes

3. **Method validation**:
   - Does uncertainty correlate with actual errors?
   - Are aleatoric and epistemic independent?

**From cross-sequence comparison**:

1. **Sequence characteristics**:
   - MOT17-13: High feature variance → Easy for V1
   - MOT17-05: Low feature variance → Hard for V1

2. **Generalization**:
   - Train on MOT17-13, test on MOT17-05
   - Does epistemic uncertainty flag OOD?

3. **Method robustness**:
   - V1: Works on 6/7 sequences
   - V2: Should handle all 7 (better epistemic)
   - V3: Should be robust to distribution shift

---

## 📊 Success Criteria

### How Do We Know If It's Working?

**Metric 1: Correlation with Errors**
```
corr(σ²_total, true_errors²) > 0.7
```
- Total uncertainty should correlate with actual errors
- Higher uncertainty → Higher errors

**Metric 2: Orthogonality**
```
|corr(σ²_aleatoric, σ²_epistemic)| < 0.2
```
- Aleatoric and epistemic should be independent
- Not just splitting the same signal

**Metric 3: Aleatoric Fraction**
```
30% < (σ²_aleatoric / σ²_total) < 70%
```
- Both sources should contribute meaningfully
- Not 100% aleatoric (V1's problem)

**Metric 4: Temporal Consistency**
```
corr(uncertainty[t], uncertainty[t+1]) > 0.5
```
- Uncertainty should be smooth across frames (for same object)
- Not random noise

---

## 🎓 Key Takeaways

### Why This Project Matters

1. **Post-hoc uncertainty**: Works on any detector, no retraining
2. **Interpretable**: We know *why* system is uncertain
3. **Actionable**: Different responses for aleatoric vs epistemic
4. **Temporal**: Tracks uncertainty evolution over time

### The Three-Stage Plan

| Stage | Goal | Status |
|-------|------|--------|
| **V1** | Prove concept works (simple KNN + KDE) | ✅ Done (6/7 sequences work) |
| **V2** | Fix epistemic signal (multi-source) | 🔄 Next (should fix weak epistemic) |
| **V3** | Production-ready (robust + temporal) | ⏳ Future (handles all edge cases) |

### What We've Learned So Far

1. **V1 works** on most sequences (mean corr = 0.723)
2. **MOT17-05 is challenging** due to low feature magnitudes
3. **Epistemic signal is weak** (100% aleatoric) → Need V2
4. **YOLO confidence is unreliable** (corr(C,E) ≈ 0) → Can't use it
5. **Layer 21 has highest variance** → Might help

---

**Status**: Complete understanding of goal and methods ✅
**Next**: Ready to build with clear purpose and plan
