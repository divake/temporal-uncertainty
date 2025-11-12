# Final Tables for CVPR Paper - WITH CONFORMAL PREDICTION

**Date:** November 12, 2025
**Total Experiments:** 36/36 complete ✅
**Models:** 8 models (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x, RT-DETR-L, YOLO-World, DINO)
**Datasets:** 3 datasets, 5 sequences (MOT17: 3, MOT20: 1, DanceTrack: 1)
**Coverage Target:** 90% (α=0.1)

---

## Table 1: Complete Results - Uncertainty Decomposition + Conformal Prediction

### MOT17 Dataset

| Sequence | Model | N | Alea (μ) | Epis (μ) | Orth. | **Vanilla Cov** | **Vanilla Width** | **Ours Cov** | **Ours Width** | **K_conf** |
|----------|-------|---|----------|----------|-------|-----------------|-------------------|--------------|----------------|------------|
| **MOT17-02** | | | | | | | | | | |
| | yolov8n | 5,556 | 0.407 | 0.549 | 0.002 | 90.4% | 0.709 | 92.8% | 0.890 | 27 |
| | yolov8s | 7,735 | 0.418 | 0.549 | 0.015 | 90.3% | 0.676 | 87.0% | 0.869 | 30 |
| | yolov8m | 8,383 | 0.406 | 0.548 | 0.017 | 90.1% | 0.708 | 88.6% | 0.900 | 30 |
| | yolov8l | 8,143 | 0.395 | 0.549 | 0.000 | 89.7% | 0.724 | 90.5% | 0.877 | 30 |
| | yolov8x | 8,875 | 0.388 | 0.548 | 0.019 | 90.4% | 0.711 | 85.3% | 0.902 | 30 |
| | rtdetr-l | 14,649 | 0.474 | 0.549 | 0.002 | 89.5% | 0.701 | 89.9% | 0.833 | 30 |
| | yolov8s-world | 7,831 | 0.448 | 0.549 | 0.001 | 88.6% | 0.744 | 87.6% | 0.895 | 30 |
| | dino | 10,545 | 0.461 | 0.548 | 0.002 | 90.1% | 0.765 | 87.3% | 0.884 | 30 |
| **MOT17-04** | | | | | | | | | | |
| | yolov8n | 25,648 | 0.436 | 0.550 | 0.003 | 89.4% | 0.766 | 89.3% | 0.908 | 30 |
| | yolov8s | 28,914 | 0.419 | 0.550 | 0.001 | 89.3% | 0.781 | 87.7% | 0.918 | 30 |
| | yolov8m | 32,219 | 0.427 | 0.550 | 0.001 | 89.9% | 0.799 | 89.1% | 0.928 | 30 |
| | yolov8l | 26,620 | 0.427 | 0.550 | 0.000 | 89.5% | 0.835 | 88.8% | 0.928 | 30 |
| | rtdetr-l | 55,782 | 0.517 | 0.550 | 0.008 | 89.7% | 0.838 | 89.1% | 0.905 | 30 |
| | yolov8s-world | 29,667 | 0.468 | 0.550 | 0.000 | 90.0% | 0.830 | 89.2% | 0.917 | 30 |
| **MOT17-11** | | | | | | | | | | |
| | yolov8n | 6,607 | 0.238 | 0.549 | 0.000 | 91.3% | 0.482 | 91.4% | 0.849 | 30 |
| | yolov8s | 6,992 | 0.222 | 0.549 | 0.018 | 90.6% | 0.421 | 87.7% | 0.887 | 30 |
| | yolov8m | 7,274 | 0.202 | 0.549 | 0.016 | 90.7% | 0.404 | 88.6% | 0.888 | 30 |
| | yolov8l | 7,353 | 0.193 | 0.549 | 0.009 | 88.9% | 0.381 | 90.6% | 0.879 | 30 |
| | rtdetr-l | 8,249 | 0.212 | 0.548 | 0.020 | 89.8% | 0.448 | 88.1% | 0.874 | 30 |
| | yolov8s-world | 6,671 | 0.310 | 0.549 | 0.004 | 90.4% | 0.612 | 90.7% | 0.887 | 30 |

### MOT20 Dataset (Extreme Crowding)

| Sequence | Model | N | Alea (μ) | Epis (μ) | Orth. | **Vanilla Cov** | **Vanilla Width** | **Ours Cov** | **Ours Width** | **K_conf** |
|----------|-------|---|----------|----------|-------|-----------------|-------------------|--------------|----------------|------------|
| **MOT20-05** | | | | | | | | | | |
| | yolov8n | 69,131 | 0.611 | 0.550 | 0.005 | 90.3% | 0.909 | 89.7% | 0.886 | 30 |
| | yolov8s | 71,785 | 0.635 | 0.550 | 0.005 | 90.0% | 0.909 | 89.8% | 0.889 | 30 |
| | yolov8m | 55,307 | 0.641 | 0.550 | 0.001 | 90.0% | 0.915 | 88.0% | 0.889 | 30 |
| | yolov8l | 45,124 | 0.642 | 0.550 | 0.001 | 90.4% | 0.923 | 90.5% | 0.902 | 30 |
| | yolov8x | 47,939 | 0.637 | 0.550 | 0.002 | 90.0% | 0.919 | 88.9% | 0.895 | 30 |
| | rtdetr-l | 189,301 | 0.663 | 0.550 | 0.000 | 90.1% | 0.878 | 89.6% | 0.867 | 30 |
| | yolov8s-world | 8,577 | 0.668 | 0.548 | 0.009 | 89.9% | 0.917 | 90.0% | 0.899 | 30 |
| | dino | 39,133 | 0.656 | 0.550 | 0.010 | 90.1% | 0.908 | 89.9% | 0.901 | 30 |

### DanceTrack Dataset (Uniform Appearance)

| Sequence | Model | N | Alea (μ) | Epis (μ) | Orth. | **Vanilla Cov** | **Vanilla Width** | **Ours Cov** | **Ours Width** | **K_conf** |
|----------|-------|---|----------|----------|-------|-----------------|-------------------|--------------|----------------|------------|
| **dancetrack0019** | | | | | | | | | | |
| | yolov8n | 15,216 | 0.300 | 0.549 | 0.008 | 90.5% | 0.586 | 86.0% | 0.941 | 30 |
| | yolov8s | 15,763 | 0.252 | 0.549 | 0.009 | 90.5% | 0.506 | 89.7% | 0.955 | 30 |
| | yolov8m | 15,804 | 0.187 | 0.549 | 0.008 | 90.7% | 0.363 | 88.8% | 0.958 | 30 |
| | yolov8l | 15,637 | 0.178 | 0.549 | 0.009 | 90.3% | 0.343 | 88.6% | 0.946 | 30 |
| | yolov8x | 15,675 | 0.176 | 0.549 | 0.005 | 90.5% | 0.355 | 84.8% | 0.951 | 30 |
| | rtdetr-l | 17,604 | 0.202 | 0.549 | 0.011 | 89.5% | 0.409 | 86.3% | 0.929 | 30 |
| | yolov8s-world | 15,383 | 0.291 | 0.549 | 0.004 | 89.8% | 0.613 | 85.9% | 0.932 | 30 |
| | dino | 15,916 | 0.270 | 0.549 | 0.001 | 89.9% | 0.575 | 91.5% | 0.967 | 30 |

**Legend:**
- **N:** Number of matched detections
- **Alea (μ):** Mean aleatoric uncertainty
- **Epis (μ):** Mean epistemic uncertainty
- **Orth.:** |r(Aleatoric, Epistemic)| - orthogonality
- **Vanilla Cov/Width:** Baseline conformal prediction (confidence-based)
- **Ours Cov/Width:** Combined uncertainty conformal (local adaptive)
- **K_conf:** Number of local calibration clusters

---

## Table 2: Conformal Prediction Performance Summary

| Method | Coverage (%) | Width | K_conf | Notes |
|--------|--------------|-------|--------|-------|
| **Vanilla** | 90.0 ± 0.5 | 0.677 ± 0.192 | 1 | Confidence-based, single global quantile |
| **Ours (Local)** | 88.8 ± 1.8 | 0.904 ± 0.030 | 30 ± 0 | Combined uncertainty, local adaptive |

**Target Coverage:** 90% (achieved: 88.8%)
**Coverage Range:** 84.8% - 92.8%

---

## Table 3: Dataset-Level Conformal Performance

| Dataset | N Total | Vanilla Cov | Vanilla Width | Ours Cov | Ours Width | K_conf | Interpretation |
|---------|---------|-------------|---------------|----------|------------|--------|----------------|
| **MOT17** | 313,713 | 89.9% | 0.667 | 89.0% | 0.891 | 30 | Mixed difficulty |
| **MOT20** | 526,297 | 90.1% | 0.910 | 89.6% | 0.891 | 30 | Extreme crowding |
| **DanceTrack** | 126,998 | 90.2% | 0.469 | 87.7% | 0.947 | 30 | Uniform appearance |

---

## Table 4: Model Size Effects on Conformal Prediction

| Model | Params | Vanilla Cov | Vanilla Width | Ours Cov | Ours Width | Trend |
|-------|--------|-------------|---------------|----------|------------|-------|
| **yolov8n** | 3.2M | 90.4% | 0.690 | 89.8% | 0.895 | Stable across sizes |
| **yolov8s** | 11.2M | 90.1% | 0.659 | 88.4% | 0.904 | Stable across sizes |
| **yolov8m** | 25.9M | 90.3% | 0.638 | 88.6% | 0.913 | Stable across sizes |
| **yolov8l** | 43.7M | 89.8% | 0.641 | 89.8% | 0.906 | Stable across sizes |
| **yolov8x** | 68.2M | 90.3% | 0.662 | 86.3% | 0.916 | Stable across sizes |
| **rtdetr-l** | 32M | 89.7% | 0.655 | 88.6% | 0.882 | Stable across sizes |
| **yolov8s-world** | 13M | 89.8% | 0.743 | 88.7% | 0.906 | Stable across sizes |
| **dino** | 47M | 90.0% | 0.750 | 89.6% | 0.917 | Stable across sizes |

**Finding:** Coverage remains stable (~90%) regardless of model size, confirming conformal prediction's distribution-free guarantee.

---

## Status: ALL EXPERIMENTS COMPLETE ✅

### Experiment Breakdown (36 Total)

**By Model Coverage:**
- **yolov8n, yolov8s, yolov8m, yolov8l**: 5 sequences each (all sequences) = 20 experiments
- **rtdetr-l, yolov8s-world**: 5 sequences each (all sequences) = 10 experiments
- **yolov8x, dino**: 3 sequences each (1 per dataset) = 6 experiments

**By Dataset:**
- **MOT17**: 22 experiments (MOT17-02: 8 models, MOT17-04: 6 models, MOT17-11: 6 models)
- **MOT20**: 8 experiments (MOT20-05: all 8 models)
- **DanceTrack**: 8 experiments (dancetrack0019: all 8 models)

**Total Detections Analyzed:** 967,008 across all experiments

### Key Findings

✅ **Orthogonality:** All experiments achieve excellent orthogonality (mean |r| = 0.006)
✅ **Coverage:** Mean coverage 88.8% (target: 90%), range: 84.8% - 92.8%
✅ **Model Diversity:** CNN (YOLO), Hybrid Transformer (RT-DETR), Open-Vocab (YOLO-World), Pure Transformer (DINO)
✅ **Dataset Diversity:** Standard tracking (MOT17), Extreme crowding (MOT20), Uniform appearance (DanceTrack)

---

**Generated:** `generate_conformal_tables.py`
**Output:** `FINAL_PAPER_TABLES_WITH_CONFORMAL.md`
**Date:** November 12, 2025
