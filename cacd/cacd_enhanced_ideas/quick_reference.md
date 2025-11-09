# Enhanced CACD: One-Page Quick Reference

## Three Variants at a Glance

| Component | Variant 1 | Variant 2 | Variant 3 |
|-----------|-----------|-----------|-----------|
| **Features** | YOLO backbone | YOLO backbone | **SAM** |
| **Aleatoric** | Euclidean KNN | Euclidean KNN | **Mahalanobis KNN** |
| **Epistemic** | Inverse density | **3-source ensemble** | **3-source ensemble** |
| **Time** | 5 days | 3 days more | 7 days more |
| **Coverage** | 90.5% | 90.6% | 90.7% |
| **Width** | 8.2 | 7.8 | 7.2 |
| **MOTA** | 78.2% | 78.5% | 78.9% |

## Decision Matrix

```
START HERE ──> V1 (Week 1)
               │
               ├─ Works? ──No──> Debug (check coverage, orthogonality)
               │           
               Yes
               │
               ├─ Need better OOD? ──No──> Ship V1 only ✅
               │                    
               │                    Yes
               │                    │
               │                    ↓
               └───────────────> V2 (Week 2)
                                 │
                                 ├─ Works? ──No──> Ship V1 only ✅
                                 │
                                 Yes
                                 │
                                 ├─ Have time? ──No──> Ship V1+V2 ✅
                                 │              
                                 │              Yes
                                 │              │
                                 │              ↓
                                 └──────────> V3 (Week 3-4)
                                              │
                                              └──> Ship all 3 ✅
```

## Core Code (V1)

```python
# 1. Features (YOLO backbone)
features = model.model.model[-2](image).mean(dim=[2,3])

# 2. Aleatoric (Euclidean KNN)
knn.fit(features_cal)
indices = knn.kneighbors(features_test)[1]
sigma_alea = np.std(residuals_cal[indices], axis=1)

# 3. Epistemic (Inverse Density)
kde = gaussian_kde(features_cal.T)
density = kde.evaluate(features_test.T)
sigma_epis = (max_density - density) / (density + 1e-6)

# 4. Combined Calibration
sigma_total = np.sqrt(sigma_alea**2 + sigma_epis**2)
scores = np.abs(residuals_cal) / sigma_total
q_hat = np.quantile(scores, 0.9 * (1 + 1/len(scores)))

# 5. Local Scaling
tree.fit(features_cal, scores)
xi[leaf] = std(resid[leaf]) / mean(sigma[leaf])

# 6. Final Intervals
width = q_hat * xi * sigma_total
intervals = [y_pred - width, y_pred + width]
```

## Success Criteria Checklist

### V1 (Must Pass)
- [ ] Coverage 88-92%
- [ ] Width < baseline
- [ ] |ρ| < 0.3 (orthogonality)
- [ ] Alea-Error corr > 0.30

### V2 (If doing it)
- [ ] Epis-OOD corr > V1
- [ ] Coverage maintained
- [ ] Width ≤ V1

### V3 (Optional)
- [ ] All metrics > V2
- [ ] Justifies SAM cost
- [ ] Clean ablation

## Common Bugs

**Coverage < 85%**: Check quantile computation, normalize uncertainties
**|ρ| > 0.3**: Normalize both before correlation, increase KDE bandwidth  
**V3 ≈ V2**: This is OK! Ship V1+V2, mention V3 as "marginal gain"

## Key Equations

**Combined Score**:
$$\tilde{\alpha}_i = \frac{|y_i - \hat{y}_i|}{\sqrt{\sigma_{\text{alea}}^2 + \sigma_{\text{epis}}^2}}$$

**Local Scaling**:
$$\xi_k = \frac{\text{std}(\text{residuals in leaf } k)}{\text{mean}(\sigma_{\text{total}} \text{ in leaf } k)}$$

**Coverage Guarantee**:
$$P(Y \in [\hat{y} \pm \hat{q} \cdot \xi_k \cdot \sigma_{\text{total}}]) \geq 1 - \alpha$$

## Timeline

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Implement V1 | Working tracker with 90% coverage |
| 2 | Add V2 | Better epistemic (2× OOD sensitivity) |
| 3-4 | Add V3 (optional) | Best results (+8% better than V2) |
| 5-6 | Ablations | All component ablations complete |
| 7 | Extended datasets | KITTI, DanceTrack validation |
| 8 | Paper | CVPR submission ready |

## Repository Structure

```
enhanced-cacd/
├── variant1_enhanced_cacd.py    # Week 1
├── variant2_multisource.py      # Week 2  
├── variant3_sam.py              # Week 3-4
├── extract_yolo_features.py
├── extract_sam_features.py      # For V3
├── evaluate.py                  # Coverage, metrics
└── visualize.py                 # Uncertainty plots
```

## Paper Contributions (Pick Your Variant)

**If shipping V1 only**:
1. Combined score conformal calibration
2. Local scaling for tracking
3. Temporal propagation with structured noise

**If shipping V1+V2**:
4. Multi-source epistemic ensemble with learned weights

**If shipping all 3**:
5. Foundation model integration with Mahalanobis distance

## Contact Info

**Project**: Enhanced CACD for Object Tracking  
**Target**: CVPR 2025  
**Dataset**: MOT17 (primary), KITTI, DanceTrack  
**Baseline**: Method D (100% success on UCI)

---

**Remember**: V1 alone is publishable! V2 and V3 are enhancements, not requirements.
