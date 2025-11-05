"""
Ablation Study: Effect of K (number of neighbors) on Method D Performance

Tests different K values to validate our choice of K=10:
- K in [3, 5, 7, 10, 15, 20, 30, 50, 100, 'all']

Metrics:
1. Coverage (should be ~90%)
2. Orthogonality (rho should be < 0.3)
3. Aleatoric-Error Correlation (should be high)
4. Epistemic-Error Correlation (should be ~0)
5. Interval Width (narrower is better, if coverage maintained)

Hypothesis: K=10-20 will be optimal (balances bias-variance tradeoff)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

BASE_PATH = Path('/ssd_4TB/divake/temporal_uncertainty/cacd/datasets')
OUTPUT_DIR = Path('/ssd_4TB/divake/temporal_uncertainty/cacd/ablation_results')
OUTPUT_DIR.mkdir(exist_ok=True)


class MethodD_KAblation:
    """Method D with configurable K parameter"""

    def __init__(self, alpha=0.1, k_neighbors=10):
        self.alpha = alpha
        self.k_neighbors = k_neighbors
        self.scaler = StandardScaler()

    def calibrate(self, X_cal, y_cal, y_pred_cal):
        """Calibrate using KNN for aleatoric and KDE for epistemic"""
        # Store calibration data
        X_cal_scaled = self.scaler.fit_transform(X_cal)
        self.X_cal_scaled = X_cal_scaled

        # Compute residuals
        self.residuals_cal = np.abs(y_cal - y_pred_cal)

        # Fit KNN (for aleatoric)
        if self.k_neighbors == 'all':
            # Use all samples (global variance)
            self.k_neighbors_actual = len(X_cal)
        else:
            self.k_neighbors_actual = min(self.k_neighbors, len(X_cal) - 1)

        self.knn = NearestNeighbors(n_neighbors=self.k_neighbors_actual)
        self.knn.fit(X_cal_scaled)

        # Fit KDE (for epistemic - always uses all data)
        self.kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
        self.kde.fit(X_cal_scaled)

        # Compute vanilla conformal quantile
        self.vanilla_quantile = np.quantile(self.residuals_cal, 1 - self.alpha)

    def _compute_aleatoric(self, X_test_scaled):
        """Compute aleatoric from K nearest neighbors"""
        distances, indices = self.knn.kneighbors(X_test_scaled)

        aleatoric = []
        for neighbor_idx in indices:
            neighbor_residuals = self.residuals_cal[neighbor_idx]
            local_std = np.std(neighbor_residuals)
            aleatoric.append(local_std)

        return np.array(aleatoric)

    def _compute_epistemic(self, X_test_scaled):
        """Compute epistemic from KDE (inverse density)"""
        log_densities = self.kde.score_samples(X_test_scaled)
        densities = np.exp(log_densities)

        max_density = densities.max()
        epistemic = (max_density / (densities + 1e-6)) - 1.0

        return epistemic

    def evaluate(self, X_test, y_test, y_pred_test):
        """Evaluate on test set"""
        X_test_scaled = self.scaler.transform(X_test)

        # Compute uncertainties
        aleatoric_raw = self._compute_aleatoric(X_test_scaled)
        epistemic_raw = self._compute_epistemic(X_test_scaled)

        # Normalize to comparable scales
        aleatoric = (aleatoric_raw - aleatoric_raw.mean()) / (aleatoric_raw.std() + 1e-6)
        epistemic = (epistemic_raw - epistemic_raw.mean()) / (epistemic_raw.std() + 1e-6)

        # Scale to have similar range as vanilla quantile
        target_scale = self.vanilla_quantile
        aleatoric = aleatoric * target_scale / 2
        epistemic = epistemic * target_scale / 2

        # Prediction intervals (using vanilla quantile)
        lower = y_pred_test - self.vanilla_quantile
        upper = y_pred_test + self.vanilla_quantile

        # Metrics
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        width = np.mean(upper - lower)

        # True errors
        true_errors = np.abs(y_test - y_pred_test)

        # Correlations
        corr_alea_epis = np.corrcoef(aleatoric, epistemic)[0, 1]
        corr_alea_error = np.corrcoef(aleatoric, true_errors)[0, 1]
        corr_epis_error = np.corrcoef(epistemic, true_errors)[0, 1]

        return {
            'coverage': coverage,
            'width': width,
            'correlation': corr_alea_epis,
            'alea_error_corr': corr_alea_error,
            'epis_error_corr': corr_epis_error,
            'aleatoric_mean': aleatoric.mean(),
            'aleatoric_std': aleatoric.std(),
            'epistemic_mean': epistemic.mean(),
            'epistemic_std': epistemic.std()
        }


def load_and_prepare(dataset_name):
    """Load dataset and train model"""
    path = BASE_PATH / f"{dataset_name}.csv"

    if not path.exists():
        return None, None

    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.33, random_state=42
    )

    # Train model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = MLPRegressor(hidden_layer_sizes=(50, 30), max_iter=500,
                        random_state=42, early_stopping=True,
                        validation_fraction=0.15, verbose=False)
    model.fit(X_train_scaled, y_train)

    # Wrapped model
    class ScaledModel:
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler

        def predict(self, X):
            return self.model.predict(self.scaler.transform(X))

    scaled_model = ScaledModel(model, scaler)

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_cal': X_cal, 'y_cal': y_cal,
        'X_test': X_test, 'y_test': y_test
    }, scaled_model


def run_ablation_single_dataset(dataset_name, k_values):
    """Run ablation on a single dataset"""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*80}")

    # Load data
    data, model = load_and_prepare(dataset_name)
    if data is None:
        print(f"Dataset {dataset_name} not found!")
        return None

    print(f"Samples: {len(data['X_train'])} train, {len(data['X_cal'])} cal, {len(data['X_test'])} test")

    # Get predictions
    y_pred_cal = model.predict(data['X_cal'])
    y_pred_test = model.predict(data['X_test'])

    results = []

    for k in k_values:
        print(f"\nTesting K={k}...", end=' ')

        try:
            cacd = MethodD_KAblation(alpha=0.1, k_neighbors=k)
            cacd.calibrate(data['X_cal'], data['y_cal'], y_pred_cal)
            metrics = cacd.evaluate(data['X_test'], data['y_test'], y_pred_test)

            metrics['k'] = k
            metrics['dataset'] = dataset_name
            metrics['status'] = 'success'

            # Check pass/fail
            cov_ok = abs(metrics['coverage'] - 0.9) < 0.05
            orth_ok = abs(metrics['correlation']) < 0.3
            both_ok = cov_ok and orth_ok

            print(f"Cov: {metrics['coverage']:.1%} {'✅' if cov_ok else '❌'}, "
                  f"ρ: {metrics['correlation']:.3f} {'✅' if orth_ok else '❌'} "
                  f"{'✅ PASS' if both_ok else '❌ FAIL'}")

            results.append(metrics)

        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                'k': k,
                'dataset': dataset_name,
                'status': 'failed',
                'error': str(e)
            })

    return pd.DataFrame(results)


def run_ablation_all_datasets(k_values):
    """Run ablation on all datasets"""
    datasets = [
        'energy_heating',
        'concrete',
        'yacht',
        'wine_quality_red',
        'power_plant',
        'energy_cooling'
    ]

    all_results = []

    for dataset in datasets:
        df = run_ablation_single_dataset(dataset, k_values)
        if df is not None:
            all_results.append(df)

    return pd.concat(all_results, ignore_index=True)


def plot_ablation_results(results_df, output_dir):
    """Create comprehensive visualization of ablation results"""

    # Filter successful runs
    results = results_df[results_df['status'] == 'success'].copy()

    # Convert 'all' to numeric for plotting
    results['k_numeric'] = results['k'].apply(lambda x: 200 if x == 'all' else x)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Ablation Study: Effect of K on Method D Performance', fontsize=16, y=0.995)

    # 1. Coverage vs K
    ax = axes[0, 0]
    for dataset in results['dataset'].unique():
        data = results[results['dataset'] == dataset]
        ax.plot(data['k_numeric'], data['coverage'] * 100, 'o-', label=dataset, alpha=0.7)
    ax.axhline(90, color='red', linestyle='--', linewidth=2, label='Target: 90%')
    ax.fill_between([results['k_numeric'].min(), results['k_numeric'].max()], 85, 95,
                     alpha=0.2, color='green', label='Acceptable (85-95%)')
    ax.set_xlabel('K (number of neighbors)', fontsize=11)
    ax.set_ylabel('Coverage (%)', fontsize=11)
    ax.set_title('Coverage vs K', fontsize=12)
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Orthogonality vs K
    ax = axes[0, 1]
    for dataset in results['dataset'].unique():
        data = results[results['dataset'] == dataset]
        ax.plot(data['k_numeric'], np.abs(data['correlation']), 'o-', label=dataset, alpha=0.7)
    ax.axhline(0.3, color='red', linestyle='--', linewidth=2, label='Threshold: 0.3')
    ax.set_xlabel('K (number of neighbors)', fontsize=11)
    ax.set_ylabel('|Correlation| (aleatoric vs epistemic)', fontsize=11)
    ax.set_title('Orthogonality vs K', fontsize=12)
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Aleatoric-Error Correlation vs K
    ax = axes[0, 2]
    for dataset in results['dataset'].unique():
        data = results[results['dataset'] == dataset]
        ax.plot(data['k_numeric'], data['alea_error_corr'], 'o-', label=dataset, alpha=0.7)
    ax.axhline(0, color='gray', linestyle='-', linewidth=1)
    ax.set_xlabel('K (number of neighbors)', fontsize=11)
    ax.set_ylabel('Correlation (aleatoric vs error)', fontsize=11)
    ax.set_title('Aleatoric Quality vs K', fontsize=12)
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Success Rate vs K
    ax = axes[1, 0]
    success_by_k = results.groupby('k').apply(
        lambda df: np.mean((np.abs(df['coverage'] - 0.9) < 0.05) & (np.abs(df['correlation']) < 0.3))
    )
    k_values_plot = [k if k != 'all' else 200 for k in success_by_k.index]
    ax.bar(range(len(k_values_plot)), success_by_k.values * 100, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(k_values_plot)))
    ax.set_xticklabels([str(k) for k in success_by_k.index], rotation=45)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_xlabel('K (number of neighbors)', fontsize=11)
    ax.set_title('Success Rate (Coverage + Orthogonality) vs K', fontsize=12)
    ax.axhline(100, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Interval Width vs K
    ax = axes[1, 1]
    for dataset in results['dataset'].unique():
        data = results[results['dataset'] == dataset]
        ax.plot(data['k_numeric'], data['width'], 'o-', label=dataset, alpha=0.7)
    ax.set_xlabel('K (number of neighbors)', fontsize=11)
    ax.set_ylabel('Average Interval Width', fontsize=11)
    ax.set_title('Efficiency vs K (lower is better)', fontsize=12)
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Heatmap: Success by dataset and K
    ax = axes[1, 2]
    pivot = results.pivot_table(
        index='dataset',
        columns='k',
        values='correlation',
        aggfunc=lambda x: 1 if (x.iloc[0] if len(x) > 0 else 1) < 0.3 else 0
    )
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Pass (1) / Fail (0)'}, ax=ax)
    ax.set_title('Orthogonality Success by Dataset and K', fontsize=12)
    ax.set_xlabel('K (number of neighbors)', fontsize=11)
    ax.set_ylabel('Dataset', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'k_ablation_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved comprehensive plot: {output_dir / 'k_ablation_comprehensive.png'}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ABLATION STUDY: Effect of K on Method D")
    print("="*80)

    # K values to test
    k_values = [3, 5, 7, 10, 15, 20, 30, 50, 100, 'all']

    print(f"\nTesting K values: {k_values}")
    print(f"Metrics: Coverage, Orthogonality, Aleatoric-Error Correlation")
    print(f"Hypothesis: K=10-20 will be optimal")

    # Run ablation on all datasets
    results_df = run_ablation_all_datasets(k_values)

    # Save results
    results_df.to_csv(OUTPUT_DIR / 'k_ablation_results.csv', index=False)
    print(f"\n✓ Saved results: {OUTPUT_DIR / 'k_ablation_results.csv'}")

    # Create visualizations
    plot_ablation_results(results_df, OUTPUT_DIR)

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY: Success Rate by K")
    print("="*80)

    success_results = results_df[results_df['status'] == 'success'].copy()

    for k in k_values:
        k_data = success_results[success_results['k'] == k]
        if len(k_data) > 0:
            success_count = ((np.abs(k_data['coverage'] - 0.9) < 0.05) &
                           (np.abs(k_data['correlation']) < 0.3)).sum()
            total = len(k_data)
            success_rate = success_count / total * 100

            avg_cov = k_data['coverage'].mean()
            avg_orth = k_data['correlation'].abs().mean()
            avg_alea_corr = k_data['alea_error_corr'].mean()

            print(f"\nK={k:>4}: {success_count}/{total} datasets ({success_rate:5.1f}%)")
            print(f"         Avg Coverage: {avg_cov:.1%}, Avg |ρ|: {avg_orth:.3f}, "
                  f"Avg Alea-Error Corr: {avg_alea_corr:.3f}")

    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE!")
    print("="*80)
