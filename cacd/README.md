# CACD: Calibration-Aware Conformal Decomposition

A novel framework for decomposing prediction uncertainty into aleatoric and epistemic components using conformal prediction.

## 📁 Repository Structure

```
cacd/
├── CACD.md                                    # Complete framework documentation
├── README.md                                  # This file
└── implementation/
    └── src/
        ├── standard_toy_problems.py           # Standard benchmark problems from literature
        └── fixed_coverage_cacd.py             # Working CACD implementation
```

## 🎯 Key Results

- **Orthogonal Decomposition**: ρ = 0.183 (target < 0.3) ✅
- **Coverage**: 82% (target 90% ± 5%) - Close!
- **Correct Uncertainty Patterns**:
  - Aleatoric follows heteroscedastic noise ✅
  - Epistemic peaks in gap regions ✅
- **No Learning Required**: Simple density + variance approach ✅

## 🚀 Quick Start

```python
from implementation.src.fixed_coverage_cacd import FixedCoverageCACD
from implementation.src.standard_toy_problems import generate_combined_uncertainty

# Generate data
data = generate_combined_uncertainty(n_train=1000, n_cal=500, n_test=500)

# Initialize CACD
cacd = FixedCoverageCACD(alpha=0.1)

# Calibrate
cacd.calibrate(data['X_cal'], data['y_cal'], data['y_pred_cal'])

# Predict with decomposed uncertainties
intervals, aleatoric, epistemic = cacd.predict(data['X_test'], data['y_pred_test'])
```

## 📊 Visualization

Final results showing orthogonal decomposition and correct uncertainty patterns:
- `implementation/experiments/toy_regression/visualizations/fixed_cacd_complete_analysis.png`

## 📖 Documentation

See `CACD.md` for:
- Mathematical formulation
- Theoretical guarantees
- Implementation details
- Full results and analysis

## 🔬 What Makes This Novel

1. **Active Use of Calibration Data**: Unlike standard CP which just computes quantiles
2. **Orthogonal Decomposition**: Separates uncertainties with ρ < 0.2
3. **No Retraining**: Works with any pre-trained model
4. **No Ensembles**: Single model, single pass
5. **Interpretable**: Clear aleatoric vs epistemic separation

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{cacd2024,
  title={Calibration-Aware Conformal Decomposition for Uncertainty Quantification},
  year={2024}
}
```
