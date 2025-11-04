"""
Deep Analysis of UCI Datasets for Uncertainty Quantification

Goals:
1. Understand data distribution and properties
2. Identify sources of aleatoric uncertainty (noise patterns)
3. Identify sources of epistemic uncertainty (sparse regions, outliers)
4. Visualize to convince these are suitable for CACD
5. Recommend which datasets to use

This analysis will help us understand if UCI datasets have:
- Heteroscedastic noise (aleatoric)
- Non-uniform density (epistemic)
- Good separation between uncertainty types
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KernelDensity, NearestNeighbors
from scipy import stats
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

BASE_PATH = Path('/ssd_4TB/divake/temporal_uncertainty/cacd/datasets')
VIZ_PATH = Path('/ssd_4TB/divake/temporal_uncertainty/cacd/implementation/experiments/toy_regression/visualizations')
VIZ_PATH.mkdir(exist_ok=True, parents=True)

# ============================================================
# Load all datasets
# ============================================================
def load_dataset(name):
    """Load and prepare dataset"""
    path = BASE_PATH / f"{name}.csv"
    df = pd.read_csv(path)

    # Last column is usually target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return X, y, df

# ============================================================
# Analysis Functions
# ============================================================

def analyze_heteroscedasticity(X, y, name):
    """
    Check if data has heteroscedastic noise (aleatoric uncertainty varies with x).
    Strong heteroscedasticity is good for CACD!
    """
    print(f"\n{'='*70}")
    print(f"HETEROSCEDASTICITY ANALYSIS: {name}")
    print(f"{'='*70}")

    # Train a model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPRegressor(hidden_layer_sizes=(50, 30), max_iter=500,
                         random_state=42, early_stopping=True)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    residuals = np.abs(y_test - y_pred)

    # Test 1: Breusch-Pagan test for heteroscedasticity
    # Group by predicted value and check if variance changes
    n_bins = 10
    pred_bins = pd.qcut(y_pred, q=n_bins, duplicates='drop')
    grouped_var = []
    bin_centers = []

    for bin_val in pred_bins.categories:
        mask = pred_bins == bin_val
        if mask.sum() > 1:
            grouped_var.append(residuals[mask].var())
            bin_centers.append(y_pred[mask].mean())

    grouped_var = np.array(grouped_var)
    bin_centers = np.array(bin_centers)

    # Compute coefficient of variation of variances
    het_score = grouped_var.std() / (grouped_var.mean() + 1e-10)

    print(f"\nModel Performance:")
    print(f"  R² Score: {model.score(X_test_scaled, y_test):.3f}")
    print(f"  Mean Absolute Error: {np.mean(residuals):.3f}")

    print(f"\nHeteroscedasticity Analysis:")
    print(f"  Variance across bins:")
    print(f"    Min: {grouped_var.min():.4f}")
    print(f"    Max: {grouped_var.max():.4f}")
    print(f"    Ratio (max/min): {grouped_var.max() / grouped_var.min():.2f}")
    print(f"  Heteroscedasticity Score: {het_score:.3f}")
    print(f"    > 0.5: Strong (Good for CACD!) ✅")
    print(f"    0.2-0.5: Moderate")
    print(f"    < 0.2: Weak (Homoscedastic)")

    # Test 2: Correlation between prediction and residual magnitude
    corr = np.corrcoef(y_pred, residuals)[0, 1]
    print(f"  Correlation(prediction, |residual|): {corr:.3f}")

    return {
        'residuals': residuals,
        'predictions': y_pred,
        'y_test': y_test,
        'het_score': het_score,
        'grouped_var': grouped_var,
        'bin_centers': bin_centers,
        'model': model,
        'scaler': scaler,
        'X_test': X_test
    }


def analyze_density_gaps(X, name):
    """
    Check if data has non-uniform density (epistemic uncertainty varies spatially).
    Gaps and sparse regions are good for CACD!
    """
    print(f"\n{'='*70}")
    print(f"DENSITY & EPISTEMIC ANALYSIS: {name}")
    print(f"{'='*70}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use KDE to estimate density
    kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
    kde.fit(X_scaled)

    log_densities = kde.score_samples(X_scaled)
    densities = np.exp(log_densities)

    # Analyze density distribution
    print(f"\nDensity Distribution:")
    print(f"  Min density: {densities.min():.6f}")
    print(f"  Max density: {densities.max():.6f}")
    print(f"  Ratio (max/min): {densities.max() / densities.min():.1f}")
    print(f"  Std of log-density: {log_densities.std():.3f}")

    # Find sparse regions (low density)
    threshold = np.percentile(densities, 10)  # Bottom 10%
    sparse_mask = densities < threshold

    print(f"\nSparse Regions (bottom 10% density):")
    print(f"  Number of sparse points: {sparse_mask.sum()} / {len(X)}")
    print(f"  Percentage: {sparse_mask.mean()*100:.1f}%")

    # Analyze NN distances (another epistemic indicator)
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(X_scaled)
    distances, _ = knn.kneighbors(X_scaled)
    mean_distances = distances.mean(axis=1)

    print(f"\nNearest Neighbor Distances:")
    print(f"  Mean: {mean_distances.mean():.3f}")
    print(f"  Std: {mean_distances.std():.3f}")
    print(f"  Max: {mean_distances.max():.3f}")
    print(f"  Ratio (max/mean): {mean_distances.max() / mean_distances.mean():.2f}")

    # Outlier detection score
    outlier_threshold = mean_distances.mean() + 2*mean_distances.std()
    outliers = mean_distances > outlier_threshold

    print(f"\nPotential Outliers (distance > mean + 2std):")
    print(f"  Count: {outliers.sum()} / {len(X)} ({outliers.mean()*100:.1f}%)")

    return {
        'densities': densities,
        'log_densities': log_densities,
        'sparse_mask': sparse_mask,
        'mean_distances': mean_distances,
        'outliers': outliers,
        'X_scaled': X_scaled
    }


def analyze_separability(X, y, name):
    """
    Check if aleatoric and epistemic uncertainty sources are separable.
    This is crucial for CACD!
    """
    print(f"\n{'='*70}")
    print(f"UNCERTAINTY SEPARABILITY ANALYSIS: {name}")
    print(f"{'='*70}")

    # Get heteroscedastic analysis
    het_result = analyze_heteroscedasticity(X, y, f"{name}_temp")
    density_result = analyze_density_gaps(X, name)

    # For test set, check correlation between density and residuals
    # Low correlation = good separability!

    # We need to get density for test points
    X_test_scaled = het_result['scaler'].transform(het_result['X_test'])

    kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
    X_train_scaled = het_result['scaler'].fit_transform(X)
    kde.fit(X_train_scaled)

    test_log_densities = kde.score_samples(X_test_scaled)
    test_densities = np.exp(test_log_densities)

    # Correlation between density (epistemic indicator) and residuals (total uncertainty)
    corr_density_residual = np.corrcoef(test_densities, het_result['residuals'])[0, 1]

    print(f"\nSeparability Metrics:")
    print(f"  Corr(density, residuals): {corr_density_residual:.3f}")
    print(f"    Close to 0 = Good separability ✅")
    print(f"    Far from 0 = Coupled uncertainties ❌")

    # Check if low-density regions have different error patterns
    low_density = test_densities < np.percentile(test_densities, 25)
    high_density = test_densities > np.percentile(test_densities, 75)

    print(f"\nError in Different Density Regions:")
    print(f"  Low density (bottom 25%): {het_result['residuals'][low_density].mean():.3f}")
    print(f"  High density (top 25%): {het_result['residuals'][high_density].mean():.3f}")
    print(f"  Ratio: {het_result['residuals'][low_density].mean() / (het_result['residuals'][high_density].mean() + 1e-10):.2f}")

    return {
        'het_result': het_result,
        'density_result': density_result,
        'corr_density_residual': corr_density_residual,
        'test_densities': test_densities
    }


def create_comprehensive_visualization(X, y, name, results):
    """Create comprehensive visualization for dataset analysis"""

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    het = results['het_result']
    dens = results['density_result']

    # Plot 1: Data distribution (if low-dimensional, use PCA for high-d)
    ax = fig.add_subplot(gs[0, 0])
    if X.shape[1] <= 2:
        if X.shape[1] == 1:
            ax.scatter(X[:, 0], y, alpha=0.5, s=10)
            ax.set_xlabel('Feature')
        else:
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.5, s=10)
            plt.colorbar(scatter, ax=ax, label='Target')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
    else:
        # Use first 2 principal components
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5, s=10)
        plt.colorbar(scatter, ax=ax, label='Target')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title(f'Data Distribution\n(n={len(X)}, d={X.shape[1]})', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 2: Target distribution
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(y, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(y.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {y.mean():.2f}')
    ax.axvline(np.median(y), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(y):.2f}')
    ax.set_xlabel('Target Value')
    ax.set_ylabel('Count')
    ax.set_title('Target Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Residuals vs Predictions (Heteroscedasticity Check)
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(het['predictions'], het['residuals'], alpha=0.5, s=10)
    ax.axhline(het['residuals'].mean(), color='r', linestyle='--', label='Mean residual')

    # Add variance trend
    if len(het['bin_centers']) > 0:
        ax.plot(het['bin_centers'], np.sqrt(het['grouped_var']), 'r-', linewidth=3,
               label='Std by bin', alpha=0.7)

    ax.set_xlabel('Predicted Value')
    ax.set_ylabel('Absolute Residual')
    ax.set_title(f'Heteroscedasticity Check\n(Score: {het["het_score"]:.3f})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Residual distribution
    ax = fig.add_subplot(gs[0, 3])
    ax.hist(het['residuals'], bins=30, alpha=0.7, edgecolor='black', density=True)

    # Fit and overlay normal distribution
    mu, std = het['residuals'].mean(), het['residuals'].std()
    x = np.linspace(het['residuals'].min(), het['residuals'].max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal fit')

    ax.set_xlabel('Residual')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Density map (epistemic indicator)
    ax = fig.add_subplot(gs[1, 0])
    if X.shape[1] <= 2:
        if X.shape[1] == 1:
            ax.scatter(X[:, 0], dens['densities'], alpha=0.5, s=10, c=dens['densities'],
                      cmap='YlOrRd')
            ax.set_xlabel('Feature')
            ax.set_ylabel('Density')
        else:
            scatter = ax.scatter(X[:, 0], X[:, 1], c=dens['densities'],
                               cmap='YlOrRd', alpha=0.6, s=20)
            plt.colorbar(scatter, ax=ax, label='Density')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=dens['densities'],
                           cmap='YlOrRd', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label='Density')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

    # Mark sparse regions
    if X.shape[1] <= 2 and X.shape[1] > 1:
        sparse_points = X[dens['sparse_mask']]
        ax.scatter(sparse_points[:, 0], sparse_points[:, 1], c='red', s=100,
                  marker='x', linewidths=2, label='Sparse (bottom 10%)')
        ax.legend()

    ax.set_title('Data Density (Epistemic Indicator)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 6: NN Distance distribution
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(dens['mean_distances'], bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(dens['mean_distances'].mean(), color='r', linestyle='--',
              linewidth=2, label='Mean')
    ax.axvline(dens['mean_distances'].mean() + 2*dens['mean_distances'].std(),
              color='orange', linestyle='--', linewidth=2, label='Mean+2std (outliers)')
    ax.set_xlabel('Mean Distance to 5-NN')
    ax.set_ylabel('Count')
    ax.set_title(f'NN Distance Distribution\n({dens["outliers"].sum()} outliers)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 7: Correlation scatter (Density vs Residuals)
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(results['test_densities'], het['residuals'], alpha=0.5, s=10)
    z = np.polyfit(results['test_densities'], het['residuals'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(results['test_densities'].min(), results['test_densities'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2,
           label=f'r={results["corr_density_residual"]:.3f}')
    ax.set_xlabel('Density (Epistemic)')
    ax.set_ylabel('Residual (Total Uncertainty)')
    ax.set_title('Uncertainty Separability Check', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 8: Feature correlation heatmap
    ax = fig.add_subplot(gs[1, 3])
    if X.shape[1] <= 15:  # Only if not too many features
        corr_matrix = np.corrcoef(X.T)
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax)
        ax.set_title('Feature Correlation', fontweight='bold')
    else:
        ax.text(0.5, 0.5, f'Too many features\n({X.shape[1]})',
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Feature Correlation', fontweight='bold')
    ax.axis('off' if X.shape[1] > 15 else 'on')

    # Plot 9: Summary statistics table
    ax = fig.add_subplot(gs[2, :2])
    ax.axis('off')

    summary_text = f"""
    DATASET SUMMARY: {name}
    {'='*50}

    Basic Statistics:
      • Samples: {len(X)}
      • Features: {X.shape[1]}
      • Target range: [{y.min():.2f}, {y.max():.2f}]
      • Target mean ± std: {y.mean():.2f} ± {y.std():.2f}

    Model Performance:
      • R² Score: {het['het_score']:.3f}
      • Mean Absolute Error: {het['residuals'].mean():.3f}

    Aleatoric Uncertainty (Heteroscedasticity):
      • Het. Score: {het['het_score']:.3f} {'✅ STRONG' if het['het_score'] > 0.5 else '⚠️  MODERATE' if het['het_score'] > 0.2 else '❌ WEAK'}
      • Variance ratio (max/min): {het['grouped_var'].max() / het['grouped_var'].min():.2f}x
      • Interpretation: {'High noise variability - Good for CACD!' if het['het_score'] > 0.3 else 'Low noise variability'}

    Epistemic Uncertainty (Density variation):
      • Density ratio (max/min): {dens['densities'].max() / dens['densities'].min():.1f}x
      • Sparse points (bottom 10%): {dens['sparse_mask'].sum()} ({dens['sparse_mask'].mean()*100:.1f}%)
      • Outliers (distance): {dens['outliers'].sum()} ({dens['outliers'].mean()*100:.1f}%)
      • Interpretation: {'Good density variation - Good for CACD!' if dens['log_densities'].std() > 1.0 else 'Moderate density variation'}

    Separability (Key for CACD!):
      • Corr(density, residual): {results['corr_density_residual']:.3f}
      • Status: {'✅ GOOD (independent sources)' if abs(results['corr_density_residual']) < 0.3 else '⚠️  MODERATE' if abs(results['corr_density_residual']) < 0.5 else '❌ POOR (coupled)'}
      • Interpretation: {'Aleatoric and epistemic are separable!' if abs(results['corr_density_residual']) < 0.3 else 'Some coupling between uncertainties'}
    """

    ax.text(0.05, 0.5, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='center', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Plot 10: Suitability score
    ax = fig.add_subplot(gs[2, 2:])
    ax.axis('off')

    # Compute overall suitability score
    het_score = 1.0 if het['het_score'] > 0.5 else (0.5 if het['het_score'] > 0.2 else 0.0)
    dens_score = 1.0 if dens['log_densities'].std() > 1.0 else 0.5
    sep_score = 1.0 if abs(results['corr_density_residual']) < 0.3 else (0.5 if abs(results['corr_density_residual']) < 0.5 else 0.0)

    overall_score = (het_score + dens_score + sep_score) / 3.0

    criteria = ['Heteroscedasticity\n(Aleatoric)', 'Density Variation\n(Epistemic)', 'Separability']
    scores = [het_score, dens_score, sep_score]
    colors = ['green' if s >= 0.8 else 'orange' if s >= 0.4 else 'red' for s in scores]

    bars = ax.barh(criteria, scores, color=colors, alpha=0.7)
    ax.set_xlim([0, 1.1])
    ax.set_xlabel('Suitability Score', fontsize=12, fontweight='bold')
    ax.set_title(f'CACD Suitability: {overall_score:.0%}\n{"✅ HIGHLY SUITABLE" if overall_score > 0.7 else "⚠️  MODERATELY SUITABLE" if overall_score > 0.4 else "❌ NOT SUITABLE"}',
                fontsize=14, fontweight='bold')

    # Add score labels
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{score:.0%}', ha='left', va='center', fontsize=11, fontweight='bold')

    ax.grid(True, alpha=0.3, axis='x')

    # Main title
    fig.suptitle(f'UCI Dataset Analysis: {name.upper()}',
                fontsize=18, fontweight='bold', y=0.98)

    # Save
    save_path = VIZ_PATH / f'uci_analysis_{name}.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved visualization: {save_path}")

    return overall_score


# ============================================================
# Main Analysis
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("UCI DATASETS DEEP ANALYSIS FOR UNCERTAINTY QUANTIFICATION")
    print("="*70)

    datasets = [
        'concrete',
        'energy_heating',
        'energy_cooling',
        'wine_quality_red',
        'yacht',
        'power_plant'
    ]

    suitability_scores = {}

    for dataset_name in datasets:
        try:
            print(f"\n\n{'#'*70}")
            print(f"# ANALYZING: {dataset_name.upper()}")
            print(f"{'#'*70}")

            X, y, df = load_dataset(dataset_name)

            print(f"\nDataset loaded:")
            print(f"  Shape: {X.shape}")
            print(f"  Target: {y.shape}")
            print(f"  Missing values: {pd.DataFrame(X).isnull().sum().sum()}")

            # Run analyses
            results = analyze_separability(X, y, dataset_name)

            # Create visualization
            score = create_comprehensive_visualization(X, y, dataset_name, results)
            suitability_scores[dataset_name] = score

        except Exception as e:
            print(f"\n❌ Error analyzing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            suitability_scores[dataset_name] = 0.0

    # ============================================================
    # Final Recommendations
    # ============================================================
    print("\n\n" + "="*70)
    print("FINAL RECOMMENDATIONS")
    print("="*70)

    print("\nSuitability Scores:")
    sorted_datasets = sorted(suitability_scores.items(), key=lambda x: x[1], reverse=True)

    for name, score in sorted_datasets:
        status = "✅ HIGHLY SUITABLE" if score > 0.7 else "⚠️  MODERATELY SUITABLE" if score > 0.4 else "❌ NOT SUITABLE"
        print(f"  {name:25s}: {score:.0%} {status}")

    print("\n📌 RECOMMENDED DATASETS FOR CACD:")
    recommended = [name for name, score in sorted_datasets if score > 0.5]
    for i, name in enumerate(recommended[:3], 1):
        print(f"  {i}. {name} (score: {suitability_scores[name]:.0%})")

    print(f"\n📁 All visualizations saved to: {VIZ_PATH}")
    print("="*70)
