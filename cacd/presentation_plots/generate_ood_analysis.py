"""
OOD Analysis: Compare In-Distribution vs Out-of-Distribution Uncertainty

Creates visualizations showing how aleatoric and epistemic uncertainties
behave differently on ID vs OOD data.

Strategy:
1. Train on middle range of a feature (e.g., compactness 0.62-0.82)
2. Calibrate on middle range
3. Test on:
   - ID: Middle range (same distribution)
   - OOD: Extreme ranges (low: <0.62, high: >0.82)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors, KernelDensity
from pathlib import Path
import sys

sys.path.append('/ssd_4TB/divake/temporal_uncertainty/cacd/implementation/src')

BASE_PATH = Path('/ssd_4TB/divake/temporal_uncertainty/cacd/datasets')
OUTPUT_DIR = Path('/ssd_4TB/divake/temporal_uncertainty/cacd/presentation_plots/ood_analysis')
OUTPUT_DIR.mkdir(exist_ok=True)


def load_and_split_ood(dataset_name='energy_heating', feature_idx=0):
    """
    Load dataset and create ID/OOD splits based on a feature

    Args:
        dataset_name: UCI dataset name
        feature_idx: Index of feature to use for OOD split (0 = compactness for energy)

    Returns:
        Dictionary with train, cal, test_id, test_ood data
    """
    path = BASE_PATH / f"{dataset_name}.csv"
    df = pd.read_csv(path)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Get the feature for splitting
    split_feature = X[:, feature_idx]

    # Determine percentiles for ID/OOD split
    p25 = np.percentile(split_feature, 25)
    p75 = np.percentile(split_feature, 75)

    print(f"\nFeature {feature_idx} statistics:")
    print(f"  Min: {split_feature.min():.3f}")
    print(f"  25th percentile: {p25:.3f}")
    print(f"  Median: {np.median(split_feature):.3f}")
    print(f"  75th percentile: {p75:.3f}")
    print(f"  Max: {split_feature.max():.3f}")

    # Split data into ID and OOD regions
    id_mask = (split_feature >= p25) & (split_feature <= p75)
    ood_mask = ~id_mask

    # In-distribution data (middle 50%)
    X_id = X[id_mask]
    y_id = y[id_mask]

    # Out-of-distribution data (lower 25% + upper 25%)
    X_ood = X[ood_mask]
    y_ood = y[ood_mask]

    print(f"\nData split:")
    print(f"  In-Distribution (ID): {len(X_id)} samples ({p25:.3f} <= feature <= {p75:.3f})")
    print(f"  Out-of-Distribution (OOD): {len(X_ood)} samples (feature < {p25:.3f} or > {p75:.3f})")

    # Split ID data into train/cal/test
    n_id = len(X_id)
    n_train = int(0.6 * n_id)
    n_cal = int(0.2 * n_id)

    indices = np.random.RandomState(42).permutation(n_id)

    train_idx = indices[:n_train]
    cal_idx = indices[n_train:n_train+n_cal]
    test_id_idx = indices[n_train+n_cal:]

    # Use ALL OOD data for testing
    test_ood_idx = np.arange(len(X_ood))

    return {
        'X_train': X_id[train_idx],
        'y_train': y_id[train_idx],
        'X_cal': X_id[cal_idx],
        'y_cal': y_id[cal_idx],
        'X_test_id': X_id[test_id_idx],
        'y_test_id': y_id[test_id_idx],
        'X_test_ood': X_ood[test_ood_idx],
        'y_test_ood': y_ood[test_ood_idx],
        'feature_range': (p25, p75)
    }


def train_model(X_train, y_train):
    """Train MLP model"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = MLPRegressor(
        hidden_layer_sizes=(50, 30),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        verbose=False
    )
    model.fit(X_train_scaled, y_train)

    class ScaledModel:
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler

        def predict(self, X):
            return self.model.predict(self.scaler.transform(X))

    return ScaledModel(model, scaler)


def compute_uncertainties(X_cal, y_cal, y_pred_cal, X_test, y_pred_test, alpha=0.1, k=10):
    """
    Compute aleatoric and epistemic uncertainties using Method D
    """
    # Scale data
    scaler = StandardScaler()
    X_cal_scaled = scaler.fit_transform(X_cal)
    X_test_scaled = scaler.transform(X_test)

    # Residuals
    residuals_cal = np.abs(y_cal - y_pred_cal)

    # Vanilla quantile
    vanilla_quantile = np.quantile(residuals_cal, 1 - alpha)

    # KNN for aleatoric
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X_cal_scaled)
    distances, indices = knn.kneighbors(X_test_scaled)

    aleatoric = []
    for neighbor_idx in indices:
        neighbor_residuals = residuals_cal[neighbor_idx]
        local_std = np.std(neighbor_residuals)
        aleatoric.append(local_std)
    aleatoric = np.array(aleatoric)

    # KDE for epistemic
    kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
    kde.fit(X_cal_scaled)
    log_densities = kde.score_samples(X_test_scaled)
    densities = np.exp(log_densities)

    max_density = densities.max()
    epistemic = (max_density / (densities + 1e-6)) - 1.0

    # Normalize using Min-Max (consistent with original Method D)
    # This ensures values are always positive and in [0, 1] range
    aleatoric_norm = (aleatoric - aleatoric.min()) / (aleatoric.max() - aleatoric.min() + 1e-10)
    epistemic_norm = (epistemic - epistemic.min()) / (epistemic.max() - epistemic.min() + 1e-10)

    # Scale to vanilla quantile range [0, vanilla_quantile]
    aleatoric_scaled = aleatoric_norm * vanilla_quantile
    epistemic_scaled = epistemic_norm * vanilla_quantile

    # Prediction intervals
    lower = y_pred_test - vanilla_quantile
    upper = y_pred_test + vanilla_quantile

    return {
        'aleatoric': aleatoric_scaled,
        'epistemic': epistemic_scaled,
        'vanilla_quantile': vanilla_quantile,
        'lower': lower,
        'upper': upper,
        'aleatoric_raw': aleatoric,
        'epistemic_raw': epistemic
    }


def create_ood_comparison_plot(data, model, output_dir):
    """
    Create comprehensive OOD comparison plot similar to step8_final_output.png
    """
    # Get predictions
    y_pred_cal = model.predict(data['X_cal'])
    y_pred_test_id = model.predict(data['X_test_id'])
    y_pred_test_ood = model.predict(data['X_test_ood'])

    # Compute uncertainties for both ID and OOD
    print("\nComputing uncertainties for ID test set...")
    uncert_id = compute_uncertainties(
        data['X_cal'], data['y_cal'], y_pred_cal,
        data['X_test_id'], y_pred_test_id
    )

    print("Computing uncertainties for OOD test set...")
    uncert_ood = compute_uncertainties(
        data['X_cal'], data['y_cal'], y_pred_cal,
        data['X_test_ood'], y_pred_test_ood
    )

    # Compute errors
    errors_id = np.abs(data['y_test_id'] - y_pred_test_id)
    errors_ood = np.abs(data['y_test_ood'] - y_pred_test_ood)

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, top=0.95, bottom=0.05)

    # ========================================================================
    # Plot 1: ID - Stacked Uncertainty Decomposition
    # ========================================================================
    ax = fig.add_subplot(gs[0, 0])

    n_samples = len(uncert_id['aleatoric'])
    x_range = np.arange(n_samples)

    ax.fill_between(x_range, 0, uncert_id['aleatoric'],
                     alpha=0.7, color='coral', label='Aleatoric', edgecolor='black', linewidth=0.5)
    ax.fill_between(x_range, uncert_id['aleatoric'],
                     uncert_id['aleatoric'] + uncert_id['epistemic'],
                     alpha=0.7, color='mediumorchid', label='Epistemic', edgecolor='black', linewidth=0.5)
    ax.axhline(uncert_id['vanilla_quantile'], color='red', linestyle='--',
               linewidth=2, label='Vanilla Quantile')

    ax.set_xlabel('Test Sample (ID)', fontsize=11)
    ax.set_ylabel('Uncertainty', fontsize=11)
    ax.set_title('In-Distribution: Uncertainty Decomposition', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # ========================================================================
    # Plot 2: OOD - Stacked Uncertainty Decomposition
    # ========================================================================
    ax = fig.add_subplot(gs[0, 1])

    n_samples_ood = len(uncert_ood['aleatoric'])
    x_range_ood = np.arange(n_samples_ood)

    ax.fill_between(x_range_ood, 0, uncert_ood['aleatoric'],
                     alpha=0.7, color='coral', label='Aleatoric', edgecolor='black', linewidth=0.5)
    ax.fill_between(x_range_ood, uncert_ood['aleatoric'],
                     uncert_ood['aleatoric'] + uncert_ood['epistemic'],
                     alpha=0.7, color='mediumorchid', label='Epistemic', edgecolor='black', linewidth=0.5)
    ax.axhline(uncert_ood['vanilla_quantile'], color='red', linestyle='--',
               linewidth=2, label='Vanilla Quantile')

    ax.set_xlabel('Test Sample (OOD)', fontsize=11)
    ax.set_ylabel('Uncertainty', fontsize=11)
    ax.set_title('Out-of-Distribution: Uncertainty Decomposition', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # ========================================================================
    # Plot 3: Distribution Comparison
    # ========================================================================
    ax = fig.add_subplot(gs[0, 2])

    metrics_data = {
        'Aleatoric': [uncert_id['aleatoric'].mean(), uncert_ood['aleatoric'].mean()],
        'Epistemic': [uncert_id['epistemic'].mean(), uncert_ood['epistemic'].mean()],
        'Error': [errors_id.mean(), errors_ood.mean()]
    }

    x_pos = np.arange(2)
    width = 0.25

    for i, (name, values) in enumerate(metrics_data.items()):
        offset = (i - 1) * width
        colors = {'Aleatoric': 'coral', 'Epistemic': 'mediumorchid', 'Error': 'steelblue'}
        ax.bar(x_pos + offset, values, width, label=name, alpha=0.7,
               color=colors[name], edgecolor='black')

    ax.set_ylabel('Mean Value', fontsize=11)
    ax.set_title('ID vs OOD: Mean Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['ID', 'OOD'], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # ========================================================================
    # Plot 4: ID - Joint Distribution (Alea vs Epis, colored by error)
    # ========================================================================
    ax = fig.add_subplot(gs[1, 0])

    scatter = ax.scatter(uncert_id['aleatoric'], uncert_id['epistemic'],
                        c=errors_id, s=50, alpha=0.7, cmap='YlOrRd',
                        edgecolors='black', linewidth=0.5)
    ax.plot([uncert_id['aleatoric'].min(), uncert_id['aleatoric'].max()],
            [uncert_id['aleatoric'].min(), uncert_id['aleatoric'].max()],
            'k--', alpha=0.3, label='Equal Line')

    ax.set_xlabel('Aleatoric Uncertainty', fontsize=11)
    ax.set_ylabel('Epistemic Uncertainty', fontsize=11)
    ax.set_title('ID: Aleatoric vs Epistemic (colored by error)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('True Error', fontsize=10)

    # Add correlation
    corr_id = np.corrcoef(uncert_id['aleatoric'], uncert_id['epistemic'])[0, 1]
    ax.text(0.05, 0.95, f'ρ = {corr_id:.3f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # ========================================================================
    # Plot 5: OOD - Joint Distribution (Alea vs Epis, colored by error)
    # ========================================================================
    ax = fig.add_subplot(gs[1, 1])

    scatter = ax.scatter(uncert_ood['aleatoric'], uncert_ood['epistemic'],
                        c=errors_ood, s=50, alpha=0.7, cmap='YlOrRd',
                        edgecolors='black', linewidth=0.5)
    ax.plot([uncert_ood['aleatoric'].min(), uncert_ood['aleatoric'].max()],
            [uncert_ood['aleatoric'].min(), uncert_ood['aleatoric'].max()],
            'k--', alpha=0.3, label='Equal Line')

    ax.set_xlabel('Aleatoric Uncertainty', fontsize=11)
    ax.set_ylabel('Epistemic Uncertainty', fontsize=11)
    ax.set_title('OOD: Aleatoric vs Epistemic (colored by error)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('True Error', fontsize=10)

    # Add correlation
    corr_ood = np.corrcoef(uncert_ood['aleatoric'], uncert_ood['epistemic'])[0, 1]
    ax.text(0.05, 0.95, f'ρ = {corr_ood:.3f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # ========================================================================
    # Plot 6: Correlations Comparison
    # ========================================================================
    ax = fig.add_subplot(gs[1, 2])

    # Compute correlations
    corr_id_alea_err = np.corrcoef(uncert_id['aleatoric'], errors_id)[0, 1]
    corr_id_epis_err = np.corrcoef(uncert_id['epistemic'], errors_id)[0, 1]
    corr_ood_alea_err = np.corrcoef(uncert_ood['aleatoric'], errors_ood)[0, 1]
    corr_ood_epis_err = np.corrcoef(uncert_ood['epistemic'], errors_ood)[0, 1]

    correlations = {
        'ID': [corr_id_alea_err, corr_id_epis_err, corr_id],
        'OOD': [corr_ood_alea_err, corr_ood_epis_err, corr_ood]
    }

    x_pos = np.arange(3)
    width = 0.35

    labels = ['Alea-Error', 'Epis-Error', 'Alea-Epis']

    ax.bar(x_pos - width/2, correlations['ID'], width, label='ID',
           alpha=0.7, color='steelblue', edgecolor='black')
    ax.bar(x_pos + width/2, correlations['OOD'], width, label='OOD',
           alpha=0.7, color='orange', edgecolor='black')

    ax.set_ylabel('Correlation', fontsize=11)
    ax.set_title('Correlation Comparison: ID vs OOD', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.axhline(0, color='gray', linestyle='-', linewidth=1)
    ax.axhline(0.3, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # ========================================================================
    # Plot 7: ID - Error vs Aleatoric
    # ========================================================================
    ax = fig.add_subplot(gs[2, 0])
    ax.scatter(uncert_id['aleatoric'], errors_id, s=50, alpha=0.6,
               color='coral', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Aleatoric Uncertainty', fontsize=11)
    ax.set_ylabel('True Error', fontsize=11)
    ax.set_title(f'ID: Error vs Aleatoric (ρ={corr_id_alea_err:.3f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 8: OOD - Error vs Epistemic
    # ========================================================================
    ax = fig.add_subplot(gs[2, 1])
    ax.scatter(uncert_ood['epistemic'], errors_ood, s=50, alpha=0.6,
               color='mediumorchid', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Epistemic Uncertainty', fontsize=11)
    ax.set_ylabel('True Error', fontsize=11)
    ax.set_title(f'OOD: Error vs Epistemic (ρ={corr_ood_epis_err:.3f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 9: Summary Statistics
    # ========================================================================
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    summary_text = f"""OOD ANALYSIS SUMMARY
{'='*40}

IN-DISTRIBUTION (ID):
  Samples: {len(errors_id)}
  Mean Error: {errors_id.mean():.3f}
  Aleatoric Mean: {uncert_id['aleatoric'].mean():.3f}
  Epistemic Mean: {uncert_id['epistemic'].mean():.3f}

  Alea-Error Corr: {corr_id_alea_err:.3f}
  Epis-Error Corr: {corr_id_epis_err:.3f}
  Alea-Epis Corr:  {corr_id:.3f}

OUT-OF-DISTRIBUTION (OOD):
  Samples: {len(errors_ood)}
  Mean Error: {errors_ood.mean():.3f}
  Epistemic Mean: {uncert_ood['epistemic'].mean():.3f}
  Aleatoric Mean: {uncert_ood['aleatoric'].mean():.3f}

  Alea-Error Corr: {corr_ood_alea_err:.3f}
  Epis-Error Corr: {corr_ood_epis_err:.3f}
  Alea-Epis Corr:  {corr_ood:.3f}

CHANGE (OOD vs ID):
  Error: {(errors_ood.mean() - errors_id.mean()):.3f}
  Epistemic: {(uncert_ood['epistemic'].mean() - uncert_id['epistemic'].mean()):.3f}
  Epis-Error Corr: {(corr_ood_epis_err - corr_id_epis_err):.3f}
"""

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=0.8))

    plt.suptitle('OOD Analysis: In-Distribution vs Out-of-Distribution Uncertainty',
                 fontsize=16, y=0.98)

    plt.savefig(output_dir / 'ood_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved: {output_dir / 'ood_comparison.png'}")

    # Print summary
    print("\n" + "="*80)
    print("OOD ANALYSIS RESULTS")
    print("="*80)
    print(f"\nIN-DISTRIBUTION:")
    print(f"  Mean Error: {errors_id.mean():.3f}")
    print(f"  Aleatoric-Error Corr: {corr_id_alea_err:.3f}")
    print(f"  Epistemic-Error Corr: {corr_id_epis_err:.3f}")
    print(f"\nOUT-OF-DISTRIBUTION:")
    print(f"  Mean Error: {errors_ood.mean():.3f} ({((errors_ood.mean()/errors_id.mean()-1)*100):+.1f}%)")
    print(f"  Aleatoric-Error Corr: {corr_ood_alea_err:.3f}")
    print(f"  Epistemic-Error Corr: {corr_ood_epis_err:.3f} (INCREASE: {corr_ood_epis_err - corr_id_epis_err:+.3f})")
    print(f"\nKEY FINDING:")
    if corr_ood_epis_err > corr_id_epis_err + 0.1:
        print(f"  ✅ Epistemic-Error correlation INCREASED on OOD!")
        print(f"  ✅ Epistemic successfully detects unfamiliar regions!")
    else:
        print(f"  ⚠️  Epistemic-Error correlation did not increase significantly")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OOD ANALYSIS: In-Distribution vs Out-of-Distribution")
    print("="*80)

    # Load data with OOD split
    print("\nLoading Energy Heating dataset with OOD split...")
    data = load_and_split_ood(dataset_name='energy_heating', feature_idx=0)

    # Train model
    print("\nTraining model on ID data only...")
    model = train_model(data['X_train'], data['y_train'])

    # Evaluate
    from sklearn.metrics import r2_score
    r2_id = r2_score(data['y_test_id'], model.predict(data['X_test_id']))
    r2_ood = r2_score(data['y_test_ood'], model.predict(data['X_test_ood']))
    print(f"\nModel Performance:")
    print(f"  R² on ID test: {r2_id:.3f}")
    print(f"  R² on OOD test: {r2_ood:.3f}")

    # Create comparison plot
    print("\nGenerating OOD comparison plots...")
    create_ood_comparison_plot(data, model, OUTPUT_DIR)

    print("\n" + "="*80)
    print("OOD ANALYSIS COMPLETE!")
    print("="*80)
