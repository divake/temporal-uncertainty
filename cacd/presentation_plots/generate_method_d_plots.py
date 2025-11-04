"""
Generate Clean Presentation Plots for Method D
Dataset: Energy Heating (Real UCI Dataset)

Creates 9 plots, one for each step of Method D
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors, KernelDensity
from pathlib import Path
import sys

sys.path.append('/ssd_4TB/divake/temporal_uncertainty/cacd/implementation/src')

# Set style
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 18

OUTPUT_DIR = Path('/ssd_4TB/divake/temporal_uncertainty/cacd/presentation_plots/method_D')

print("="*80)
print("GENERATING METHOD D PRESENTATION PLOTS")
print("Dataset: Energy Heating")
print("="*80)

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("\nLoading Energy Heating dataset...")
df = pd.read_csv('/ssd_4TB/divake/temporal_uncertainty/cacd/datasets/energy_heating.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(f"Dataset shape: {X.shape}")
print(f"Features: {df.columns[:-1].tolist()}")
print(f"Target: {df.columns[-1]}")

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train, X_cal, y_train, y_cal = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

print(f"\nData split:")
print(f"  Training: {len(X_train)} samples")
print(f"  Calibration: {len(X_cal)} samples")
print(f"  Test: {len(X_test)} samples")

# ============================================================================
# STEP 1: TRAIN BASE MODEL
# ============================================================================

print("\n" + "="*80)
print("STEP 1: Training Base Model")
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cal_scaled = scaler.transform(X_cal)
X_test_scaled = scaler.transform(X_test)

# Train with tracking
model = MLPRegressor(hidden_layer_sizes=(50, 30), max_iter=500, random_state=42,
                     early_stopping=True, validation_fraction=0.15, verbose=False)
model.fit(X_train_scaled, y_train)

# Get predictions for all sets
y_pred_train = model.predict(X_train_scaled)
y_pred_cal = model.predict(X_cal_scaled)
y_pred_test = model.predict(X_test_scaled)

# Compute R² scores
from sklearn.metrics import r2_score
r2_train = r2_score(y_train, y_pred_train)
r2_cal = r2_score(y_cal, y_pred_cal)
r2_test = r2_score(y_test, y_pred_test)

print(f"Model R² scores:")
print(f"  Training: {r2_train:.4f}")
print(f"  Calibration: {r2_cal:.4f}")
print(f"  Test: {r2_test:.4f}")

# PLOT 1: Model Predictions vs True Values
fig, ax = plt.subplots(figsize=(10, 8))

# Plot all three sets
ax.scatter(y_train, y_pred_train, alpha=0.5, s=50, label=f'Train (R²={r2_train:.3f})', color='blue')
ax.scatter(y_cal, y_pred_cal, alpha=0.6, s=50, label=f'Calibration (R²={r2_cal:.3f})', color='green')
ax.scatter(y_test, y_pred_test, alpha=0.6, s=50, label=f'Test (R²={r2_test:.3f})', color='red')

# Perfect prediction line
min_val = min(y.min(), y_pred_train.min())
max_val = max(y.max(), y_pred_train.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')

ax.set_xlabel('True Heating Load')
ax.set_ylabel('Predicted Heating Load')
ax.set_title('Step 1: Base Model Predictions (MLP Neural Network)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'step1_model_predictions.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: step1_model_predictions.png")

# ============================================================================
# STEP 2: CALIBRATION - COMPUTE CONFORMAL SCORES
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Computing Conformal Scores")
print("="*80)

cal_scores = np.abs(y_cal - y_pred_cal)

print(f"Conformal scores statistics:")
print(f"  Mean: {cal_scores.mean():.4f}")
print(f"  Std: {cal_scores.std():.4f}")
print(f"  Min: {cal_scores.min():.4f}")
print(f"  Max: {cal_scores.max():.4f}")
print(f"  Median: {np.median(cal_scores):.4f}")

# PLOT 2: Conformal Scores Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Histogram
ax = axes[0]
n, bins, patches = ax.hist(cal_scores, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(cal_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {cal_scores.mean():.2f}')
ax.axvline(np.median(cal_scores), color='orange', linestyle='--', linewidth=2, label=f'Median = {np.median(cal_scores):.2f}')
ax.set_xlabel('Conformal Score |y - ŷ|')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Calibration Scores')
ax.legend()
ax.grid(True, alpha=0.3)

# Right: Scores vs Predictions
ax = axes[1]
ax.scatter(y_pred_cal, cal_scores, alpha=0.6, s=50, color='green', edgecolors='black', linewidth=0.5)
ax.set_xlabel('Predicted Value')
ax.set_ylabel('Conformal Score |y - ŷ|')
ax.set_title('Scores vs Predictions (Heteroscedasticity Check)')
ax.grid(True, alpha=0.3)
ax.axhline(cal_scores.mean(), color='red', linestyle='--', linewidth=1, alpha=0.5)

plt.suptitle('Step 2: Conformal Scores on Calibration Set', fontsize=18, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'step2_conformal_scores.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: step2_conformal_scores.png")

# ============================================================================
# STEP 3: COMPUTE VANILLA CONFORMAL QUANTILE
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Computing Vanilla Conformal Quantile")
print("="*80)

alpha = 0.1
vanilla_quantile = np.quantile(cal_scores, 1 - alpha)

print(f"Alpha (coverage level): {alpha} (targeting {(1-alpha)*100}% coverage)")
print(f"Vanilla quantile (90th percentile): {vanilla_quantile:.4f}")

# PLOT 3: Quantile Visualization
fig, ax = plt.subplots(figsize=(12, 7))

# Sort scores for plotting
sorted_scores = np.sort(cal_scores)
cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)

ax.plot(sorted_scores, cumulative * 100, linewidth=2, color='steelblue')
ax.axvline(vanilla_quantile, color='red', linestyle='--', linewidth=3,
           label=f'90th Percentile = {vanilla_quantile:.3f}')
ax.axhline(90, color='red', linestyle='--', linewidth=2, alpha=0.5)

# Shade area
ax.fill_betweenx([0, 90], 0, vanilla_quantile, alpha=0.2, color='green',
                  label='90% of scores ≤ quantile')
ax.fill_betweenx([90, 100], vanilla_quantile, sorted_scores.max(), alpha=0.2, color='red',
                  label='10% of scores > quantile')

ax.set_xlabel('Conformal Score')
ax.set_ylabel('Cumulative Percentage (%)')
ax.set_title('Step 3: Vanilla Conformal Quantile (Coverage Guarantee)')
ax.legend(loc='lower right', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, sorted_scores.max() * 1.05])
ax.set_ylim([0, 105])

# Add text annotation
ax.text(vanilla_quantile * 0.5, 50,
        f'{(1-alpha)*100:.0f}% Coverage\nGuaranteed!',
        fontsize=14, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'step3_vanilla_quantile.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: step3_vanilla_quantile.png")

# ============================================================================
# STEP 4: FIT KNN FOR ALEATORIC UNCERTAINTY
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Fitting KNN for Aleatoric Uncertainty")
print("="*80)

k_neighbors = 10
knn = NearestNeighbors(n_neighbors=k_neighbors)
knn.fit(X_cal_scaled)

residuals_cal = y_cal - y_pred_cal

# Compute aleatoric for test set
aleatoric_raw = []
for x_test in X_test_scaled:
    distances, indices = knn.kneighbors([x_test])
    neighbor_residuals = residuals_cal[indices[0]]
    local_std = np.std(neighbor_residuals)
    aleatoric_raw.append(local_std)

aleatoric_raw = np.array(aleatoric_raw)

print(f"K-neighbors: {k_neighbors}")
print(f"Aleatoric uncertainty (raw) statistics:")
print(f"  Mean: {aleatoric_raw.mean():.4f}")
print(f"  Std: {aleatoric_raw.std():.4f}")
print(f"  Min: {aleatoric_raw.min():.4f}")
print(f"  Max: {aleatoric_raw.max():.4f}")

# PLOT 4: KNN Aleatoric Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Top-left: Example of KNN for one point
ax = axes[0, 0]
example_idx = 50  # Choose a test point
x_example = X_test_scaled[example_idx:example_idx+1]
distances, indices = knn.kneighbors(x_example)

neighbor_residuals = residuals_cal[indices[0]]
ax.bar(range(k_neighbors), neighbor_residuals, color='steelblue', edgecolor='black', alpha=0.7)
ax.axhline(0, color='black', linewidth=1)
ax.axhline(neighbor_residuals.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {neighbor_residuals.mean():.2f}')
ax.axhline(neighbor_residuals.mean() + np.std(neighbor_residuals), color='orange',
           linestyle='--', linewidth=1, alpha=0.7, label=f'±σ = {np.std(neighbor_residuals):.2f}')
ax.axhline(neighbor_residuals.mean() - np.std(neighbor_residuals), color='orange',
           linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlabel('Neighbor Index')
ax.set_ylabel('Residual (y - ŷ)')
ax.set_title(f'Example: Residuals of 10 Nearest Neighbors (Test Point #{example_idx})')
ax.legend()
ax.grid(True, alpha=0.3)

# Top-right: Distribution of aleatoric
ax = axes[0, 1]
ax.hist(aleatoric_raw, bins=25, edgecolor='black', alpha=0.7, color='coral')
ax.axvline(aleatoric_raw.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {aleatoric_raw.mean():.2f}')
ax.set_xlabel('Aleatoric Uncertainty (Local Std Dev)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Aleatoric Uncertainty')
ax.legend()
ax.grid(True, alpha=0.3)

# Bottom-left: Aleatoric vs True Error
ax = axes[1, 0]
true_errors = np.abs(y_test - y_pred_test)
ax.scatter(aleatoric_raw, true_errors, alpha=0.6, s=50, color='coral', edgecolors='black', linewidth=0.5)
corr_alea = np.corrcoef(aleatoric_raw, true_errors)[0, 1]
ax.set_xlabel('Aleatoric Uncertainty')
ax.set_ylabel('True Error |y - ŷ|')
ax.set_title(f'Aleatoric vs True Error (Correlation = {corr_alea:.3f})')
ax.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(aleatoric_raw, true_errors, 1)
p = np.poly1d(z)
x_trend = np.linspace(aleatoric_raw.min(), aleatoric_raw.max(), 100)
ax.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.7, label='Trend')
ax.legend()

# Bottom-right: Spatial distribution
ax = axes[1, 1]
scatter = ax.scatter(y_pred_test, y_test, c=aleatoric_raw, s=50, cmap='YlOrRd',
                     edgecolors='black', linewidth=0.5, alpha=0.7)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax.set_xlabel('Predicted Value')
ax.set_ylabel('True Value')
ax.set_title('Predictions Colored by Aleatoric Uncertainty')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Aleatoric', rotation=270, labelpad=20)
ax.grid(True, alpha=0.3)

plt.suptitle('Step 4: KNN-Based Aleatoric Uncertainty (Local Variance)', fontsize=18, y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'step4_knn_aleatoric.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: step4_knn_aleatoric.png")

# ============================================================================
# STEP 5: FIT KDE FOR EPISTEMIC UNCERTAINTY
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Fitting KDE for Epistemic Uncertainty")
print("="*80)

kde = KernelDensity(bandwidth='scott', kernel='gaussian')
kde.fit(X_cal_scaled)

# Compute epistemic for test set
log_densities_test = kde.score_samples(X_test_scaled)
densities_test = np.exp(log_densities_test)
max_density = densities_test.max()
epistemic_raw = (max_density / (densities_test + 1e-10)) - 1.0

print(f"Epistemic uncertainty (raw) statistics:")
print(f"  Mean: {epistemic_raw.mean():.4f}")
print(f"  Std: {epistemic_raw.std():.4f}")
print(f"  Min: {epistemic_raw.min():.4f}")
print(f"  Max: {epistemic_raw.max():.4f}")

# PLOT 5: KDE Epistemic Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Top-left: Density distribution
ax = axes[0, 0]
ax.hist(densities_test, bins=30, edgecolor='black', alpha=0.7, color='purple')
ax.set_xlabel('Density p(x)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Estimated Densities')
ax.axvline(densities_test.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {densities_test.mean():.2e}')
ax.legend()
ax.grid(True, alpha=0.3)

# Top-right: Epistemic distribution
ax = axes[0, 1]
ax.hist(epistemic_raw, bins=30, edgecolor='black', alpha=0.7, color='mediumorchid')
ax.axvline(epistemic_raw.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {epistemic_raw.mean():.2f}')
ax.set_xlabel('Epistemic Uncertainty (Inverse Density)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Epistemic Uncertainty')
ax.legend()
ax.grid(True, alpha=0.3)

# Bottom-left: Epistemic vs True Error
ax = axes[1, 0]
ax.scatter(epistemic_raw, true_errors, alpha=0.6, s=50, color='mediumorchid',
           edgecolors='black', linewidth=0.5)
corr_epis = np.corrcoef(epistemic_raw, true_errors)[0, 1]
ax.set_xlabel('Epistemic Uncertainty')
ax.set_ylabel('True Error |y - ŷ|')
ax.set_title(f'Epistemic vs True Error (Correlation = {corr_epis:.3f})')
ax.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(epistemic_raw, true_errors, 1)
p = np.poly1d(z)
x_trend = np.linspace(epistemic_raw.min(), epistemic_raw.max(), 100)
ax.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.7, label='Trend')
ax.legend()

# Bottom-right: Spatial distribution
ax = axes[1, 1]
scatter = ax.scatter(y_pred_test, y_test, c=epistemic_raw, s=50, cmap='Purples',
                     edgecolors='black', linewidth=0.5, alpha=0.7)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax.set_xlabel('Predicted Value')
ax.set_ylabel('True Value')
ax.set_title('Predictions Colored by Epistemic Uncertainty')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Epistemic', rotation=270, labelpad=20)
ax.grid(True, alpha=0.3)

plt.suptitle('Step 5: KDE-Based Epistemic Uncertainty (Inverse Density)', fontsize=18, y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'step5_kde_epistemic.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: step5_kde_epistemic.png")

# ============================================================================
# STEP 6: NORMALIZE AND SCALE UNCERTAINTIES
# ============================================================================

print("\n" + "="*80)
print("STEP 6: Normalizing and Scaling Uncertainties")
print("="*80)

# Normalize to [0, 1]
aleatoric_norm = (aleatoric_raw - aleatoric_raw.min()) / (aleatoric_raw.max() - aleatoric_raw.min() + 1e-10)
epistemic_norm = (epistemic_raw - epistemic_raw.min()) / (epistemic_raw.max() - epistemic_raw.min() + 1e-10)

# Scale by vanilla quantile
aleatoric = aleatoric_norm * vanilla_quantile
epistemic = epistemic_norm * vanilla_quantile

print(f"Normalized aleatoric [0,1]:")
print(f"  Mean: {aleatoric_norm.mean():.4f}")
print(f"  Min: {aleatoric_norm.min():.4f}")
print(f"  Max: {aleatoric_norm.max():.4f}")

print(f"\nNormalized epistemic [0,1]:")
print(f"  Mean: {epistemic_norm.mean():.4f}")
print(f"  Min: {epistemic_norm.min():.4f}")
print(f"  Max: {epistemic_norm.max():.4f}")

print(f"\nScaled aleatoric (×{vanilla_quantile:.3f}):")
print(f"  Mean: {aleatoric.mean():.4f}")
print(f"  Min: {aleatoric.min():.4f}")
print(f"  Max: {aleatoric.max():.4f}")

print(f"\nScaled epistemic (×{vanilla_quantile:.3f}):")
print(f"  Mean: {epistemic.mean():.4f}")
print(f"  Min: {epistemic.min():.4f}")
print(f"  Max: {epistemic.max():.4f}")

# PLOT 6: Normalization Process
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 1: Aleatoric transformation
# Raw
ax = axes[0, 0]
ax.hist(aleatoric_raw, bins=25, edgecolor='black', alpha=0.7, color='coral')
ax.set_xlabel('Raw Aleatoric')
ax.set_ylabel('Frequency')
ax.set_title(f'Raw: [{aleatoric_raw.min():.2f}, {aleatoric_raw.max():.2f}]')
ax.grid(True, alpha=0.3)

# Normalized
ax = axes[0, 1]
ax.hist(aleatoric_norm, bins=25, edgecolor='black', alpha=0.7, color='coral')
ax.set_xlabel('Normalized Aleatoric')
ax.set_ylabel('Frequency')
ax.set_title('Normalized: [0.00, 1.00]')
ax.axvline(aleatoric_norm.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean={aleatoric_norm.mean():.2f}')
ax.legend()
ax.grid(True, alpha=0.3)

# Scaled
ax = axes[0, 2]
ax.hist(aleatoric, bins=25, edgecolor='black', alpha=0.7, color='coral')
ax.set_xlabel('Scaled Aleatoric')
ax.set_ylabel('Frequency')
ax.set_title(f'Scaled: [0.00, {vanilla_quantile:.2f}]')
ax.axvline(aleatoric.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean={aleatoric.mean():.2f}')
ax.legend()
ax.grid(True, alpha=0.3)

# Row 2: Epistemic transformation
# Raw
ax = axes[1, 0]
ax.hist(epistemic_raw, bins=25, edgecolor='black', alpha=0.7, color='mediumorchid')
ax.set_xlabel('Raw Epistemic')
ax.set_ylabel('Frequency')
ax.set_title(f'Raw: [{epistemic_raw.min():.2f}, {epistemic_raw.max():.2f}]')
ax.grid(True, alpha=0.3)

# Normalized
ax = axes[1, 1]
ax.hist(epistemic_norm, bins=25, edgecolor='black', alpha=0.7, color='mediumorchid')
ax.set_xlabel('Normalized Epistemic')
ax.set_ylabel('Frequency')
ax.set_title('Normalized: [0.00, 1.00]')
ax.axvline(epistemic_norm.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean={epistemic_norm.mean():.2f}')
ax.legend()
ax.grid(True, alpha=0.3)

# Scaled
ax = axes[1, 2]
ax.hist(epistemic, bins=25, edgecolor='black', alpha=0.7, color='mediumorchid')
ax.set_xlabel('Scaled Epistemic')
ax.set_ylabel('Frequency')
ax.set_title(f'Scaled: [0.00, {vanilla_quantile:.2f}]')
ax.axvline(epistemic.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean={epistemic.mean():.2f}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Step 6: Normalization and Scaling Process', fontsize=18, y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'step6_normalize_scale.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: step6_normalize_scale.png")

# ============================================================================
# STEP 7: PREDICTION INTERVALS
# ============================================================================

print("\n" + "="*80)
print("STEP 7: Computing Prediction Intervals")
print("="*80)

lower = y_pred_test - vanilla_quantile
upper = y_pred_test + vanilla_quantile
width = upper - lower

print(f"Interval statistics:")
print(f"  Width (constant): {vanilla_quantile * 2:.4f}")
print(f"  Mean lower: {lower.mean():.4f}")
print(f"  Mean upper: {upper.mean():.4f}")

# Check coverage
coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"\nCoverage: {coverage:.1%} ({int(coverage*len(y_test))}/{len(y_test)} points)")

# PLOT 7: Prediction Intervals
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Top-left: Intervals visualization (sorted by prediction)
ax = axes[0, 0]
sorted_idx = np.argsort(y_pred_test)
x_plot = np.arange(len(y_test))

ax.fill_between(x_plot, lower[sorted_idx], upper[sorted_idx], alpha=0.3, color='lightblue', label='90% Prediction Interval')
ax.plot(x_plot, y_pred_test[sorted_idx], 'b-', linewidth=2, label='Prediction')
ax.scatter(x_plot, y_test[sorted_idx], s=20, color='red', alpha=0.6, zorder=5, label='True Value')

# Mark points outside interval
outside = ~((y_test >= lower) & (y_test <= upper))
outside_idx = sorted_idx[outside[sorted_idx]]
if len(outside_idx) > 0:
    outside_x = np.where(np.isin(sorted_idx, outside_idx))[0]
    ax.scatter(outside_x, y_test[outside_idx], s=100, facecolors='none',
               edgecolors='red', linewidths=2, label='Outside Interval', zorder=6)

ax.set_xlabel('Test Sample (sorted by prediction)')
ax.set_ylabel('Heating Load')
ax.set_title(f'Prediction Intervals (Coverage: {coverage:.1%})')
ax.legend()
ax.grid(True, alpha=0.3)

# Top-right: Interval width across predictions
ax = axes[0, 1]
ax.scatter(y_pred_test, width, s=50, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
ax.axhline(width.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean Width = {width.mean():.2f}')
ax.set_xlabel('Predicted Value')
ax.set_ylabel('Interval Width')
ax.set_title('Interval Width (Constant for Vanilla CP)')
ax.legend()
ax.grid(True, alpha=0.3)

# Bottom-left: Coverage by prediction value
ax = axes[1, 0]
n_bins = 10
bins = np.linspace(y_pred_test.min(), y_pred_test.max(), n_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2
coverage_by_bin = []

for i in range(n_bins):
    mask = (y_pred_test >= bins[i]) & (y_pred_test < bins[i+1])
    if mask.sum() > 0:
        cov = np.mean((y_test[mask] >= lower[mask]) & (y_test[mask] <= upper[mask]))
        coverage_by_bin.append(cov * 100)
    else:
        coverage_by_bin.append(0)

ax.bar(bin_centers, coverage_by_bin, width=(bins[1]-bins[0])*0.8,
       edgecolor='black', alpha=0.7, color='lightgreen')
ax.axhline(90, color='red', linestyle='--', linewidth=2, label='Target: 90%')
ax.set_xlabel('Predicted Value (binned)')
ax.set_ylabel('Coverage (%)')
ax.set_title('Conditional Coverage by Prediction Range')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 105])

# Bottom-right: Error analysis
ax = axes[1, 1]
errors = y_test - y_pred_test
ax.scatter(y_pred_test, errors, s=50, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
ax.axhline(0, color='black', linewidth=1)
ax.fill_between([y_pred_test.min(), y_pred_test.max()], -vanilla_quantile, vanilla_quantile,
                alpha=0.2, color='lightblue', label=f'±{vanilla_quantile:.2f} (quantile)')
ax.set_xlabel('Predicted Value')
ax.set_ylabel('Error (y - ŷ)')
ax.set_title('Residuals with Quantile Bounds')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Step 7: Prediction Intervals (Vanilla Conformal Quantile)', fontsize=18, y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'step7_prediction_intervals.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: step7_prediction_intervals.png")

# ============================================================================
# STEP 8: FINAL OUTPUT - UNCERTAINTY DECOMPOSITION
# ============================================================================

print("\n" + "="*80)
print("STEP 8: Final Output - Uncertainty Decomposition")
print("="*80)

total_uncertainty = np.full_like(y_pred_test, vanilla_quantile)

print(f"Final uncertainties:")
print(f"  Total (constant): {vanilla_quantile:.4f}")
print(f"  Aleatoric mean: {aleatoric.mean():.4f}")
print(f"  Epistemic mean: {epistemic.mean():.4f}")
print(f"  Sum of means: {aleatoric.mean() + epistemic.mean():.4f}")

# PLOT 8: Uncertainty Decomposition
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Top-left: Stacked uncertainties
ax = axes[0, 0]
sorted_idx = np.argsort(y_pred_test)
x_plot = np.arange(len(y_test))

# Stack aleatoric and epistemic
ax.fill_between(x_plot, 0, aleatoric[sorted_idx], alpha=0.6, color='coral', label='Aleatoric')
ax.fill_between(x_plot, aleatoric[sorted_idx], aleatoric[sorted_idx] + epistemic[sorted_idx],
                alpha=0.6, color='mediumorchid', label='Epistemic')
ax.plot(x_plot, total_uncertainty[sorted_idx], 'r--', linewidth=2, label='Total (Vanilla Quantile)')

ax.set_xlabel('Test Sample (sorted by prediction)')
ax.set_ylabel('Uncertainty')
ax.set_title('Uncertainty Decomposition: Aleatoric + Epistemic')
ax.legend()
ax.grid(True, alpha=0.3)

# Top-right: Joint distribution
ax = axes[0, 1]
scatter = ax.scatter(aleatoric, epistemic, c=true_errors, s=60, cmap='YlOrRd',
                     edgecolors='black', linewidth=0.5, alpha=0.7)
ax.set_xlabel('Aleatoric Uncertainty')
ax.set_ylabel('Epistemic Uncertainty')
ax.set_title('Joint Distribution (colored by True Error)')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('True Error', rotation=270, labelpad=20)
ax.grid(True, alpha=0.3)

# Add diagonal line
max_val = max(aleatoric.max(), epistemic.max())
ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.3, label='Equal Line')
ax.legend()

# Bottom-left: Uncertainty contributions
ax = axes[1, 0]
contributions = pd.DataFrame({
    'Sample': np.arange(len(y_test)),
    'Aleatoric': aleatoric,
    'Epistemic': epistemic
})
# Show first 30 samples for clarity
n_show = 30
contributions_subset = contributions.iloc[:n_show]

x_bar = np.arange(n_show)
ax.bar(x_bar, contributions_subset['Aleatoric'], label='Aleatoric', color='coral', alpha=0.7)
ax.bar(x_bar, contributions_subset['Epistemic'], bottom=contributions_subset['Aleatoric'],
       label='Epistemic', color='mediumorchid', alpha=0.7)
ax.axhline(vanilla_quantile, color='red', linestyle='--', linewidth=2, label='Vanilla Quantile')

ax.set_xlabel('Test Sample')
ax.set_ylabel('Uncertainty')
ax.set_title(f'Uncertainty Breakdown (First {n_show} Samples)')
ax.legend()
ax.grid(True, alpha=0.3)

# Bottom-right: Summary statistics
ax = axes[1, 1]
ax.axis('off')

summary_stats = f"""
UNCERTAINTY DECOMPOSITION SUMMARY
{'='*45}

Total Test Samples: {len(y_test)}

ALEATORIC (Local Noise):
  Mean: {aleatoric.mean():.4f}
  Std:  {aleatoric.std():.4f}
  Min:  {aleatoric.min():.4f}
  Max:  {aleatoric.max():.4f}

EPISTEMIC (Model Uncertainty):
  Mean: {epistemic.mean():.4f}
  Std:  {epistemic.std():.4f}
  Min:  {epistemic.min():.4f}
  Max:  {epistemic.max():.4f}

TOTAL:
  Vanilla Quantile: {vanilla_quantile:.4f}
  Interval Width:   {vanilla_quantile * 2:.4f}

CORRELATIONS:
  Corr(Aleatoric, Epistemic): {np.corrcoef(aleatoric, epistemic)[0,1]:.4f}
  Corr(Aleatoric, Error):     {np.corrcoef(aleatoric, true_errors)[0,1]:.4f}
  Corr(Epistemic, Error):     {np.corrcoef(epistemic, true_errors)[0,1]:.4f}
"""

ax.text(0.1, 0.9, summary_stats, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Step 8: Final Uncertainty Decomposition', fontsize=18, y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'step8_final_output.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: step8_final_output.png")

# ============================================================================
# STEP 9: EVALUATION METRICS
# ============================================================================

print("\n" + "="*80)
print("STEP 9: Evaluation Metrics")
print("="*80)

# Coverage
coverage = np.mean((y_test >= lower) & (y_test <= upper))

# Orthogonality
correlation = np.corrcoef(aleatoric, epistemic)[0, 1]

# Width
avg_width = np.mean(upper - lower)

# Quality metrics
alea_quality = np.corrcoef(aleatoric, true_errors)[0, 1]
epis_quality = np.corrcoef(epistemic, true_errors)[0, 1]
total_quality = np.corrcoef(aleatoric + epistemic, true_errors)[0, 1]

print(f"EVALUATION RESULTS:")
print(f"  Coverage: {coverage:.1%} {'✅' if abs(coverage - 0.9) < 0.05 else '❌'}")
print(f"  Orthogonality (ρ): {correlation:.4f} {'✅' if abs(correlation) < 0.3 else '❌'}")
print(f"  Average Width: {avg_width:.4f}")
print(f"  Aleatoric Quality: {alea_quality:.4f}")
print(f"  Epistemic Quality: {epis_quality:.4f}")
print(f"  Total Quality: {total_quality:.4f}")

# PLOT 9: Evaluation Metrics Dashboard
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35, top=0.95, bottom=0.05, left=0.08, right=0.98)

# 1. Coverage gauge
ax = fig.add_subplot(gs[0, 0])
coverage_pct = coverage * 100
colors = ['red' if coverage_pct < 85 else 'orange' if coverage_pct < 88 else 'green']
ax.bar(['Coverage'], [coverage_pct], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.axhline(90, color='blue', linestyle='--', linewidth=2, label='Target: 90%')
ax.set_ylabel('Coverage (%)', fontsize=11)
ax.set_title('Coverage Metric', fontsize=12, pad=10)
ax.set_ylim([0, 105])
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
# Add text
status = '✅ PASS' if abs(coverage - 0.9) < 0.05 else '❌ FAIL'
ax.text(0, coverage_pct + 5, f'{coverage_pct:.1f}%\n{status}', ha='center', fontsize=11, fontweight='bold')

# 2. Orthogonality gauge
ax = fig.add_subplot(gs[0, 1])
colors = ['green' if abs(correlation) < 0.3 else 'red']
ax.bar(['Orthogonality'], [abs(correlation)], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.axhline(0.3, color='blue', linestyle='--', linewidth=2, label='Threshold: 0.3')
ax.set_ylabel('|Correlation|', fontsize=11)
ax.set_title('Orthogonality Metric', fontsize=12, pad=10)
ax.set_ylim([0, 1.05])
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
status = '✅ PASS' if abs(correlation) < 0.3 else '❌ FAIL'
ax.text(0, abs(correlation) + 0.08, f'{correlation:.3f}\n{status}', ha='center', fontsize=11, fontweight='bold')

# 3. Quality metrics
ax = fig.add_subplot(gs[0, 2])
metrics = ['Aleatoric', 'Epistemic', 'Total']
values = [alea_quality, epis_quality, total_quality]
colors_qual = ['coral', 'mediumorchid', 'steelblue']
bars = ax.bar(metrics, values, color=colors_qual, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Correlation with True Error', fontsize=11)
ax.set_title('Uncertainty Quality', fontsize=12, pad=10)
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3, axis='y')
ax.tick_params(axis='x', labelsize=10)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.3f}',
            ha='center', fontsize=10, fontweight='bold')

# 4. Calibration plot
ax = fig.add_subplot(gs[1, :])
sorted_idx = np.argsort(y_pred_test)
n_bins = 20
bins = np.array_split(sorted_idx, n_bins)
bin_coverage = []
bin_expected = []

for bin_idx in bins:
    cov = np.mean((y_test[bin_idx] >= lower[bin_idx]) & (y_test[bin_idx] <= upper[bin_idx]))
    bin_coverage.append(cov * 100)
    bin_expected.append(90)  # Expected 90%

x_bins = np.arange(len(bin_coverage))
ax.bar(x_bins, bin_coverage, alpha=0.7, color='lightblue', edgecolor='black', label='Actual Coverage')
ax.axhline(90, color='red', linestyle='--', linewidth=2, label='Expected: 90%')
ax.fill_between(x_bins, 85, 95, alpha=0.2, color='green', label='Acceptable Range (85-95%)')
ax.set_xlabel('Prediction Quantile (sorted)', fontsize=11)
ax.set_ylabel('Coverage (%)', fontsize=11)
ax.set_title('Calibration Plot: Coverage Across Prediction Range', fontsize=12, pad=10)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 105])

# 5. Confusion matrix style - coverage breakdown
ax = fig.add_subplot(gs[2, 0])
covered = (y_test >= lower) & (y_test <= upper)
n_covered = covered.sum()
n_uncovered = (~covered).sum()

coverage_breakdown = [n_covered, n_uncovered]
labels = [f'Covered\n{n_covered}/{len(y_test)}', f'Not Covered\n{n_uncovered}/{len(y_test)}']
colors_pie = ['lightgreen', 'lightcoral']
wedges, texts, autotexts = ax.pie(coverage_breakdown, labels=labels, colors=colors_pie,
                                    autopct='%1.1f%%', startangle=90,
                                    textprops={'fontsize': 10, 'fontweight': 'bold'})
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(11)
ax.set_title('Coverage Breakdown', fontsize=12, pad=15)

# 6. Orthogonality scatter
ax = fig.add_subplot(gs[2, 1])
ax.scatter(aleatoric, epistemic, s=50, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
ax.set_xlabel('Aleatoric', fontsize=11)
ax.set_ylabel('Epistemic', fontsize=11)
ax.set_title(f'Orthogonality: ρ = {correlation:.3f}', fontsize=12, pad=10)
ax.grid(True, alpha=0.3)

# Add correlation info
ax.text(0.05, 0.95, f'Correlation: {correlation:.4f}\nTarget: < 0.3',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, pad=0.5))

# 7. Summary table
ax = fig.add_subplot(gs[2, 2])
ax.axis('off')

summary_text = f"""FINAL EVALUATION SUMMARY
{'='*32}

SUCCESS CRITERIA:
  Coverage ≈ 90%:     {'✅ PASS' if abs(coverage - 0.9) < 0.05 else '❌ FAIL'}
  Orthogonality < 0.3: {'✅ PASS' if abs(correlation) < 0.3 else '❌ FAIL'}

DETAILED METRICS:
  Coverage:     {coverage:.1%}
  Correlation:  {correlation:.4f}
  Avg Width:    {avg_width:.4f}

  Alea Quality: {alea_quality:.4f}
  Epis Quality: {epis_quality:.4f}
  Total Quality: {total_quality:.4f}

DATASET:
  Test Samples: {len(y_test)}
  Model R²:     {r2_test:.4f}
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', pad=0.8,
                  facecolor='lightgreen' if (abs(coverage - 0.9) < 0.05 and abs(correlation) < 0.3) else 'lightyellow',
                  alpha=0.7))

plt.suptitle('Step 9: Evaluation Metrics Dashboard', fontsize=18, y=0.98)
plt.savefig(OUTPUT_DIR / 'step9_evaluation_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: step9_evaluation_metrics.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("COMPLETE! All 9 plots generated successfully!")
print("="*80)
print(f"\nPlots saved to: {OUTPUT_DIR}/")
print("\nGenerated files:")
for i in range(1, 10):
    files = list(OUTPUT_DIR.glob(f'step{i}_*.png'))
    if files:
        print(f"  {i}. {files[0].name}")

print("\n" + "="*80)
print("METHOD D RESULTS SUMMARY")
print("="*80)
print(f"Dataset: Energy Heating (Real UCI Data)")
print(f"  Training samples: {len(X_train)}")
print(f"  Calibration samples: {len(X_cal)}")
print(f"  Test samples: {len(X_test)}")
print(f"\nModel Performance:")
print(f"  R² (test): {r2_test:.4f}")
print(f"\nMethod D Results:")
print(f"  Coverage: {coverage:.1%} {'✅' if abs(coverage - 0.9) < 0.05 else '❌'}")
print(f"  Orthogonality: {correlation:.4f} {'✅' if abs(correlation) < 0.3 else '❌'}")
print(f"  Status: {'SUCCESS - Both criteria met!' if (abs(coverage - 0.9) < 0.05 and abs(correlation) < 0.3) else 'PARTIAL - See metrics'}")
print("="*80)
