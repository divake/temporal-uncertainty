# Enhanced CACD: Temporal Uncertainty in Video Object Detection

**Status**: V1 COMPLETE ✅ | V2 PENDING | V3 PENDING

---

## Quick Start

### V1 is Complete!

All V1 experiments, plots, and analysis are done. Here's what we have:

**Results**: `/ssd_4TB/divake/temporal_uncertainty/conformal_tracking/results/v1/v1_complete_20251109_135016.json`

**Plots**: `results/plots/v1/`
- `v1_k_ablation.png` - K neighbors ablation
- `v1_layer_ablation.png` - Feature layers comparison  
- `v1_conf_ablation.png` - Confidence thresholds
- `v1_cross_sequence.png` - All 7 sequences performance

**Tables**: `results/plots/v1/`
- `v1_summary_table.txt` - Human-readable
- `v1_summary_table.csv` - For Excel/analysis

**Report**: `V1_ANALYSIS_REPORT.md` - Complete 9-section analysis

---

## V1 Summary

### Performance
- **Best**: 0.840 correlation (MOT17-13)
- **Mean**: 0.559 ± 0.429 (all 7 sequences)
- **Failure**: MOT17-05 shows -0.473 (negative!)

### Optimal Configuration
```python
V1_EnhancedCACD(k_neighbors=5)  # Not K=10!
feature_layer = 21               # Not Layer 9!
conf_threshold = 0.3             # Not 0.5!
```

### Key Findings
1. **Layer 21 is best** (0.883 corr) - 27% better than Layer 9
2. **K=5 is optimal** (0.698 corr) - slightly better than K=10
3. **Conf≥0.3 works best** (0.807 corr) - best sample/quality trade-off
4. **MOT17-05 fails catastrophically** - need V2/V3 for robustness

---

## Directory Structure

```
conformal_tracking/
├── src/
│   ├── v1_enhanced_cacd.py      # V1 implementation
│   ├── data_loader.py           # YOLO cache loader
│   └── plot_v1_results.py       # Plotting script
│
├── experiments/
│   └── run_v1_complete.py       # V1 full experiment suite
│
├── results/
│   ├── v1/
│   │   └── v1_complete_*.json   # JSON results
│   └── plots/
│       └── v1/                  # All plots (PNG + PDF)
│
├── V1_ANALYSIS_REPORT.md        # Complete analysis
└── README.md                     # This file
```

---

## Next Steps

### For V2 (Multi-Source Epistemic)
1. Implement 3-source epistemic:
   - Inverse density (from V1)
   - Min Mahalanobis distance
   - Entropy (neighbor distribution)
2. Learn optimal weights via SLSQP
3. Expect: Stronger epistemic signal, better orthogonality

### For V3 (Local Scaling + Temporal)
1. Add local scaling (decision tree partitioning)
2. Add temporal propagation (Kalman filter)
3. Implement robust statistics (median, MinCovDet)
4. Expect: Fix MOT17-05 failure, smooth uncertainty

---

## Files You Need

### Run V1 Experiments
```bash
cd /ssd_4TB/divake/temporal_uncertainty/conformal_tracking
python experiments/run_v1_complete.py
```

### Generate Plots
```bash
python src/plot_v1_results.py results/v1/v1_complete_*.json
```

### View Results
```bash
cat results/plots/v1/v1_summary_table.txt
open results/plots/v1/v1_cross_sequence.png
```

---

## What Went Right

✅ Clean, modular code (v1_enhanced_cacd.py)
✅ Comprehensive experiments (K, layers, conf, cross-seq)
✅ Beautiful plots (4 figures, PNG + PDF)
✅ Detailed analysis report (9 sections)
✅ Works well on 6/7 sequences (mean 0.723)

## What Needs Work

❌ MOT17-05 failure (-0.473 correlation)
❌ Weak epistemic signal (100% aleatoric)
❌ Poor orthogonality (5/7 sequences >0.2)
❌ Overnight Claude Chat made wrong claims (K=20, Layer 9)

---

## Citation

```bibtex
@misc{enhanced_cacd_2025,
  title={Enhanced CACD: Temporal Uncertainty Decomposition for Video Object Detection},
  author={Your Name},
  year={2025},
  note={Implementation of V1, V2, V3 methods for uncertainty quantification}
}
```

---

**Status**: V1 ✅ COMPLETE | Ready for V2
**Date**: November 9, 2025
**Next**: Implement V2 with multi-source epistemic ensemble

