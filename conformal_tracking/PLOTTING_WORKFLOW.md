# Publication Plotting Workflow - Quick Start Guide

## 🎯 Goal
Generate publication-quality plots showing orthogonal decomposition of uncertainty into aleatoric and epistemic components using **REAL data** from MOT17 sequences.

---

## 📋 Two-Step Workflow

### Step 1: Extract Real Uncertainty Data (Once per sequence)

```bash
# From project root
cd /ssd_4TB/divake/temporal_uncertainty/conformal_tracking

# Extract per-detection uncertainties for MOT17-05
python experiments/extract_uncertainty_data.py --sequence MOT17-05

# Takes ~2-3 minutes, outputs to:
# results/per_detection_mot17-05/per_detection_data.npz
```

**What this does:**
- Loads 5,930 matched detections from YOLO cache
- Splits into calibration (60%) and test (40%) sets
- Computes Mahalanobis aleatoric uncertainty
- Computes combined Spectral + Repulsive epistemic uncertainty
- Saves 2,372 test detection uncertainties with IoU values

### Step 2: Generate All Plots

```bash
# Create all 6 publication-quality plots
python experiments/create_paper_plots.py

# Takes ~30 seconds, outputs to:
# results/paper_figures/real_data/
```

**Generates:**
1. `plot1_scatter_main.png/pdf` - Main scatter (aleatoric vs epistemic)
2. `plot2_aleatoric_dist.png/pdf` - Aleatoric distribution
3. `plot3_epistemic_dist.png/pdf` - Epistemic distribution
4. `plot4_iou_evolution.png/pdf` - Uncertainty vs IoU ⭐ **KEY PLOT**
5. `plot5_correlations.png/pdf` - 3-panel correlation analysis
6. `plot6_box_by_iou.png/pdf` - Statistical summary by category

---

## 📊 Key Findings (MOT17-05, N=2,372)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Aleatoric ↔ IoU** | r = -0.35 | Decreases as tracking improves ✅ |
| **Epistemic ↔ IoU** | r = +0.01 | Independent of tracking quality ✅ |
| **Orthogonality** | r = -0.25 | Weak correlation (reasonable) ✅ |

### By Tracking Quality:

| Category | Aleatoric | Epistemic |
|----------|-----------|-----------|
| Excellent (IoU > 0.8) | 0.31 | 0.44 |
| Good (IoU 0.5-0.8) | 0.36 | 0.45 |
| Poor (IoU < 0.5) | 0.47 | 0.43 |

**Key insight**: Aleatoric increases with poor tracking (0.31→0.47), epistemic stays constant (~0.44).

---

## 🗂️ File Structure

```
conformal_tracking/
├── experiments/
│   ├── extract_uncertainty_data.py    # Step 1: Extract per-detection data
│   └── create_paper_plots.py          # Step 2: Generate plots
│
├── results/
│   ├── per_detection_mot17-05/        # Extracted data (576 KB)
│   │   ├── per_detection_data.json    # Human-readable
│   │   └── per_detection_data.npz     # Fast NumPy format
│   │
│   └── paper_figures/
│       ├── README.md                  # Detailed documentation
│       └── real_data/                 # Publication plots (3.5 MB)
│           ├── plot1_scatter_main.png/pdf
│           ├── plot2_aleatoric_dist.png/pdf
│           ├── plot3_epistemic_dist.png/pdf
│           ├── plot4_iou_evolution.png/pdf
│           ├── plot5_correlations.png/pdf
│           └── plot6_box_by_iou.png/pdf
```

---

## 🔄 To Process Other Sequences

```bash
# Extract data for MOT17-02
python experiments/extract_uncertainty_data.py --sequence MOT17-02

# Extract data for MOT17-11
python experiments/extract_uncertainty_data.py --sequence MOT17-11

# Then regenerate plots (currently uses MOT17-05 only)
python experiments/create_paper_plots.py
```

**Note**: To plot multiple sequences together, you'll need to modify `create_paper_plots.py` to load and aggregate data from multiple sequence files.

---

## 🎨 Plot Specifications

All plots are publication-ready:
- **Font**: Times New Roman (serif)
- **Size**: 10×10 inches (square)
- **Labels**: 28pt bold
- **Ticks**: 24pt
- **Format**: 150 DPI PNG + vector PDF
- **Data**: 100% REAL (not synthetic!)

---

## ✅ Verification

After running both steps, verify you have:

```bash
# Check data exists
ls -lh results/per_detection_mot17-05/per_detection_data.npz

# Check plots exist
ls -lh results/paper_figures/real_data/plot*.png

# Should see 6 PNG files + 6 PDF files
```

---

## 🚫 Old Synthetic Data Approach (REMOVED)

We previously used Beta distributions to generate synthetic data from summary statistics. This has been **completely removed** and replaced with the real data workflow above.

**Deleted files:**
- `experiments/paper_individual_plots.py` (synthetic data)
- `experiments/paper_figure_publication.py` (synthetic data)
- `experiments/paper_figure1_*.py` (old versions)
- `results/paper_figures/individual/` (synthetic plots)
- `results/paper_figures/publication_figure.png` (synthetic)

---

## 📚 Documentation

- **This file**: Quick start guide
- **results/paper_figures/README.md**: Detailed workflow documentation
- **results/paper_figures/PLOT_DESCRIPTIONS.md**: Plot descriptions (outdated, was for synthetic data)

---

## 💡 Tips

1. **First time setup**: Run Step 1 once per sequence to extract data
2. **Iterating on plots**: Re-run Step 2 as many times as needed to adjust aesthetics
3. **Paper submission**: Use the PDF files (vector graphics, scales infinitely)
4. **Presentations**: Use the PNG files (easier to embed)

---

Last updated: 2025-11-11
