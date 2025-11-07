"""
Step 1: Comprehensive comparison of aleatoric uncertainty methods.
Compares Euclidean KNN (baseline) vs Mahalanobis KNN (enhanced).
"""

import sys
sys.path.append('/ssd_4TB/divake/temporal_uncertainty/cacd/enhanced_cacd/src')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from data_utils import prepare_dataset
from aleatoric import BaselineAleatoric, EnhancedAleatoric, HybridAleatoric

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Results directory
RESULTS_DIR = Path('/ssd_4TB/divake/temporal_uncertainty/cacd/enhanced_cacd/results/plots/step1')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(aleatoric, errors):
    """Compute evaluation metrics for aleatoric uncertainty."""
    # Correlation with errors (should be high for good aleatoric)
    pearson_corr, pearson_p = pearsonr(aleatoric, errors)
    spearman_corr, spearman_p = spearmanr(aleatoric, errors)

    # Normalized metrics
    norm_aleatoric = (aleatoric - aleatoric.min()) / (aleatoric.max() - aleatoric.min() + 1e-10)
    norm_errors = (errors - errors.min()) / (errors.max() - errors.min() + 1e-10)

    # Mean squared difference (should be low)
    mse = np.mean((norm_aleatoric - norm_errors)**2)

    return {
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'mse_normalized': mse,
        'mean_aleatoric': np.mean(aleatoric),
        'std_aleatoric': np.std(aleatoric),
        'min_aleatoric': np.min(aleatoric),
        'max_aleatoric': np.max(aleatoric)
    }


def plot_comparison_grid(data, results_dict, dataset_name):
    """Create comprehensive comparison plots."""
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

    errors_test = np.abs(data['y_test'] - data['y_pred_test'])

    # Row 1: Aleatoric distributions
    ax1 = fig.add_subplot(gs[0, :2])
    for method, results in results_dict.items():
        ax1.hist(results['aleatoric'], alpha=0.5, bins=30, label=method, density=True)
    ax1.set_xlabel('Aleatoric Uncertainty')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Aleatoric Uncertainty')
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 2:])
    positions = np.arange(len(results_dict))
    violin_data = [results['aleatoric'] for results in results_dict.values()]
    parts = ax2.violinplot(violin_data, positions=positions, showmeans=True)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(list(results_dict.keys()), rotation=45)
    ax2.set_ylabel('Aleatoric Uncertainty')
    ax2.set_title('Violin Plot Comparison')

    # Row 2: Scatter plots vs errors
    for i, (method, results) in enumerate(results_dict.items()):
        ax = fig.add_subplot(gs[1, i])
        aleatoric = results['aleatoric']
        ax.scatter(errors_test, aleatoric, alpha=0.5, s=10)

        # Fit line
        z = np.polyfit(errors_test, aleatoric, 1)
        p = np.poly1d(z)
        x_line = np.linspace(errors_test.min(), errors_test.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', alpha=0.7, linewidth=2)

        corr = results['metrics']['pearson_corr']
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Aleatoric')
        ax.set_title(f'{method}\nCorr: {corr:.3f}')

    # Row 3: Weight analysis (for enhanced method)
    if 'Enhanced' in results_dict:
        details = results_dict['Enhanced']['details']

        # Weight distribution
        ax = fig.add_subplot(gs[2, 0])
        weights = details['weights']
        mean_weights = np.mean(weights, axis=0)
        ax.bar(range(len(mean_weights)), mean_weights)
        ax.set_xlabel('Neighbor Index')
        ax.set_ylabel('Mean Weight')
        ax.set_title('Average Softmax Weights by Position')

        # Weight vs distance relationship
        ax = fig.add_subplot(gs[2, 1])
        flat_weights = weights.flatten()
        flat_distances = details['distances'].flatten()
        ax.scatter(flat_distances, flat_weights, alpha=0.3, s=5)
        ax.set_xlabel('Mahalanobis Distance')
        ax.set_ylabel('Softmax Weight')
        ax.set_title('Weight vs Distance Relationship')

        # Bandwidth effect
        ax = fig.add_subplot(gs[2, 2])
        bandwidth = details['bandwidth']
        d_range = np.linspace(0, np.max(flat_distances), 100)
        weight_curve = np.exp(-d_range**2 / (2 * bandwidth**2))
        ax.plot(d_range, weight_curve, 'b-', linewidth=2)
        ax.axvline(bandwidth, color='r', linestyle='--', label=f'Bandwidth={bandwidth:.3f}')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Weight (unnormalized)')
        ax.set_title('Weight Decay Function')
        ax.legend()

    # Distance comparison (if both baseline and enhanced exist)
    if 'Baseline' in results_dict and 'Enhanced' in results_dict:
        ax = fig.add_subplot(gs[2, 3])
        eucl_dist = results_dict['Baseline']['details']['distances'].mean(axis=1)
        maha_dist = results_dict['Enhanced']['details']['distances'].mean(axis=1)
        ax.scatter(eucl_dist, maha_dist, alpha=0.5, s=10)
        ax.set_xlabel('Mean Euclidean Distance')
        ax.set_ylabel('Mean Mahalanobis Distance')
        ax.set_title('Distance Metric Comparison')

        # Add diagonal line
        max_val = max(eucl_dist.max(), maha_dist.max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)

    # Row 4: Correlation comparison and metrics table
    ax = fig.add_subplot(gs[3, :2])
    methods = list(results_dict.keys())
    correlations = [results['metrics']['pearson_corr'] for results in results_dict.values()]
    colors = ['red' if c < 0.3 else 'yellow' if c < 0.35 else 'green' for c in correlations]

    bars = ax.bar(methods, correlations, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Pearson Correlation with Errors')
    ax.set_title('Aleatoric-Error Correlation Comparison')
    ax.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Poor threshold')
    ax.axhline(y=0.35, color='y', linestyle='--', alpha=0.5, label='Good threshold')
    ax.set_ylim([0, max(correlations) * 1.1])

    # Add value labels
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
    ax.legend()

    # Metrics table
    ax = fig.add_subplot(gs[3, 2:])
    ax.axis('tight')
    ax.axis('off')

    # Create metrics table
    metrics_data = []
    for method, results in results_dict.items():
        m = results['metrics']
        metrics_data.append([
            method,
            f"{m['pearson_corr']:.3f}",
            f"{m['spearman_corr']:.3f}",
            f"{m['mean_aleatoric']:.3f}",
            f"{m['std_aleatoric']:.3f}"
        ])

    table = ax.table(cellText=metrics_data,
                     colLabels=['Method', 'Pearson', 'Spearman', 'Mean', 'Std'],
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color code correlation columns
    for i in range(1, len(metrics_data) + 1):
        pearson_val = float(metrics_data[i-1][1])
        if pearson_val >= 0.35:
            table[(i, 1)].set_facecolor('#90EE90')  # Light green
        elif pearson_val >= 0.3:
            table[(i, 1)].set_facecolor('#FFFFE0')  # Light yellow
        else:
            table[(i, 1)].set_facecolor('#FFB6C1')  # Light red

    plt.suptitle(f'Aleatoric Uncertainty Comparison - {dataset_name}', fontsize=16, y=1.02)
    plt.tight_layout()

    return fig


def plot_ablation_k_values(data, dataset_name, k_values=[3, 5, 7, 10, 15, 20, 30]):
    """Test different K values for both methods."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    errors_test = np.abs(data['y_test'] - data['y_pred_test'])

    baseline_corrs = []
    enhanced_corrs = []
    hybrid_corrs = []

    for k in k_values:
        # Baseline
        baseline = BaselineAleatoric(k_neighbors=k)
        baseline.fit(data['X_cal_scaled'], data['y_cal'], data['y_pred_cal'])
        alea_baseline = baseline.predict(data['X_test_scaled'])
        corr_baseline = pearsonr(alea_baseline, errors_test)[0]
        baseline_corrs.append(corr_baseline)

        # Enhanced
        enhanced = EnhancedAleatoric(k_neighbors=k)
        enhanced.fit(data['X_cal_scaled'], data['y_cal'], data['y_pred_cal'])
        alea_enhanced = enhanced.predict(data['X_test_scaled'])
        corr_enhanced = pearsonr(alea_enhanced, errors_test)[0]
        enhanced_corrs.append(corr_enhanced)

        # Hybrid
        hybrid = HybridAleatoric(k_neighbors=k)
        hybrid.fit(data['X_cal_scaled'], data['y_cal'], data['y_pred_cal'])
        alea_hybrid = hybrid.predict(data['X_test_scaled'])
        corr_hybrid = pearsonr(alea_hybrid, errors_test)[0]
        hybrid_corrs.append(corr_hybrid)

    # Plot correlation vs K
    ax = axes[0]
    ax.plot(k_values, baseline_corrs, 'o-', label='Baseline (Euclidean)', linewidth=2, markersize=8)
    ax.plot(k_values, enhanced_corrs, 's-', label='Enhanced (Mahalanobis + Softmax)', linewidth=2, markersize=8)
    ax.plot(k_values, hybrid_corrs, '^-', label='Hybrid (Mahalanobis + Uniform)', linewidth=2, markersize=8)
    ax.set_xlabel('K (number of neighbors)')
    ax.set_ylabel('Pearson Correlation with Errors')
    ax.set_title('Ablation: Effect of K on Aleatoric Quality')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Highlight K=10
    ax.axvline(x=10, color='red', linestyle='--', alpha=0.5)
    ax.text(10.5, ax.get_ylim()[0] + 0.01, 'K=10\n(default)', fontsize=10)

    # Plot improvement over baseline
    ax = axes[1]
    enhanced_improvement = [(e - b) / b * 100 for e, b in zip(enhanced_corrs, baseline_corrs)]
    hybrid_improvement = [(h - b) / b * 100 for h, b in zip(hybrid_corrs, baseline_corrs)]

    x = np.arange(len(k_values))
    width = 0.35

    bars1 = ax.bar(x - width/2, enhanced_improvement, width, label='Enhanced', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, hybrid_improvement, width, label='Hybrid', color='orange', alpha=0.7)

    ax.set_xlabel('K value')
    ax.set_ylabel('Improvement over Baseline (%)')
    ax.set_title('Relative Improvement in Correlation')
    ax.set_xticks(x)
    ax.set_xticklabels(k_values)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

    # Detailed analysis for K=10
    k_default = 10
    for i, (method_name, method_class) in enumerate([
        ('Baseline', BaselineAleatoric),
        ('Enhanced', EnhancedAleatoric),
        ('Hybrid', HybridAleatoric)
    ]):
        ax = axes[i + 2]

        method = method_class(k_neighbors=k_default)
        method.fit(data['X_cal_scaled'], data['y_cal'], data['y_pred_cal'])

        if hasattr(method, 'predict_with_details'):
            details = method.predict_with_details(data['X_test_scaled'])
            aleatoric = details['aleatoric']
        elif hasattr(method, 'enhanced'):
            details = method.enhanced.predict_with_details(data['X_test_scaled'])
            aleatoric = details['aleatoric']
        else:
            details = method.predict_with_details(data['X_test_scaled'])
            aleatoric = details['aleatoric']

        # Scatter plot with marginal distributions
        from scipy.stats import gaussian_kde

        # Main scatter
        ax.scatter(errors_test, aleatoric, alpha=0.5, s=10)

        # Fit line
        z = np.polyfit(errors_test, aleatoric, 1)
        p = np.poly1d(z)
        x_line = np.linspace(errors_test.min(), errors_test.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', alpha=0.7, linewidth=2)

        corr = pearsonr(aleatoric, errors_test)[0]
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Aleatoric Uncertainty')
        ax.set_title(f'{method_name} (K={k_default})\nCorr: {corr:.3f}')

    # Remove unused subplots
    for i in range(5, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(f'K-Value Ablation Study - {dataset_name}', fontsize=14, y=1.02)
    plt.tight_layout()

    return fig, {
        'k_values': k_values,
        'baseline_corrs': baseline_corrs,
        'enhanced_corrs': enhanced_corrs,
        'hybrid_corrs': hybrid_corrs
    }


def run_experiment(dataset_name='energy_heating'):
    """Run complete Step 1 experiment for a dataset."""
    print(f"\n{'='*80}")
    print(f"Step 1: Aleatoric Uncertainty Comparison - {dataset_name}")
    print(f"{'='*80}")

    # Load and prepare data
    print("\n1. Loading dataset...")
    data = prepare_dataset(dataset_name)
    print(f"   Dataset: {dataset_name}")
    print(f"   Features: {data['n_features']}")
    print(f"   Train: {len(data['y_train'])}, Cal: {len(data['y_cal'])}, Test: {len(data['y_test'])}")

    # Compute test errors
    errors_test = np.abs(data['y_test'] - data['y_pred_test'])

    # Initialize methods
    print("\n2. Training aleatoric models...")
    results = {}

    # Baseline (Euclidean + Uniform)
    print("   - Baseline (Euclidean KNN)...")
    baseline = BaselineAleatoric(k_neighbors=10)
    baseline.fit(data['X_cal_scaled'], data['y_cal'], data['y_pred_cal'])
    baseline_details = baseline.predict_with_details(data['X_test_scaled'])
    results['Baseline'] = {
        'aleatoric': baseline_details['aleatoric'],
        'details': baseline_details,
        'metrics': compute_metrics(baseline_details['aleatoric'], errors_test)
    }

    # Enhanced (Mahalanobis + Softmax)
    print("   - Enhanced (Mahalanobis + Softmax)...")
    enhanced = EnhancedAleatoric(k_neighbors=10)
    enhanced.fit(data['X_cal_scaled'], data['y_cal'], data['y_pred_cal'])
    enhanced_details = enhanced.predict_with_details(data['X_test_scaled'])
    results['Enhanced'] = {
        'aleatoric': enhanced_details['aleatoric'],
        'details': enhanced_details,
        'metrics': compute_metrics(enhanced_details['aleatoric'], errors_test)
    }

    # Hybrid (Mahalanobis + Uniform)
    print("   - Hybrid (Mahalanobis + Uniform)...")
    hybrid = HybridAleatoric(k_neighbors=10)
    hybrid.fit(data['X_cal_scaled'], data['y_cal'], data['y_pred_cal'])
    hybrid_aleatoric = hybrid.predict(data['X_test_scaled'])
    results['Hybrid'] = {
        'aleatoric': hybrid_aleatoric,
        'details': {'aleatoric': hybrid_aleatoric},  # Limited details
        'metrics': compute_metrics(hybrid_aleatoric, errors_test)
    }

    # Print results summary
    print("\n3. Results Summary:")
    print(f"   {'Method':<20} {'Pearson Corr':<15} {'Improvement':<15} {'Mean Alea':<15}")
    print(f"   {'-'*65}")

    baseline_corr = results['Baseline']['metrics']['pearson_corr']
    for method, res in results.items():
        corr = res['metrics']['pearson_corr']
        improvement = (corr - baseline_corr) / baseline_corr * 100 if method != 'Baseline' else 0
        mean_alea = res['metrics']['mean_aleatoric']

        print(f"   {method:<20} {corr:<15.4f} {improvement:<15.2f}% {mean_alea:<15.4f}")

    # Generate plots
    print("\n4. Generating plots...")

    # Main comparison plot
    fig1 = plot_comparison_grid(data, results, dataset_name)
    fig1.savefig(RESULTS_DIR / f'{dataset_name}_comparison.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {dataset_name}_comparison.png")

    # K-value ablation
    fig2, k_results = plot_ablation_k_values(data, dataset_name)
    fig2.savefig(RESULTS_DIR / f'{dataset_name}_k_ablation.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {dataset_name}_k_ablation.png")

    # Save numerical results
    results_df = pd.DataFrame({
        'Method': list(results.keys()),
        'Pearson_Corr': [r['metrics']['pearson_corr'] for r in results.values()],
        'Spearman_Corr': [r['metrics']['spearman_corr'] for r in results.values()],
        'Mean_Aleatoric': [r['metrics']['mean_aleatoric'] for r in results.values()],
        'Std_Aleatoric': [r['metrics']['std_aleatoric'] for r in results.values()]
    })
    results_df.to_csv(RESULTS_DIR / f'{dataset_name}_results.csv', index=False)
    print(f"   Saved: {dataset_name}_results.csv")

    return results, k_results


def main():
    """Run experiments on multiple datasets."""
    datasets = [
        'energy_heating',
        'energy_cooling',
        'power_plant',
        # Add more as needed
    ]

    all_results = {}
    for dataset in datasets:
        try:
            results, k_results = run_experiment(dataset)
            all_results[dataset] = {
                'results': results,
                'k_results': k_results
            }
        except Exception as e:
            print(f"\nError processing {dataset}: {e}")
            continue

    # Create summary plot across all datasets
    if all_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Correlation comparison
        ax = axes[0]
        dataset_names = list(all_results.keys())
        methods = ['Baseline', 'Enhanced', 'Hybrid']
        x = np.arange(len(dataset_names))
        width = 0.25

        for i, method in enumerate(methods):
            corrs = [all_results[ds]['results'][method]['metrics']['pearson_corr']
                     for ds in dataset_names if method in all_results[ds]['results']]
            ax.bar(x + i * width, corrs, width, label=method, alpha=0.8)

        ax.set_xlabel('Dataset')
        ax.set_ylabel('Pearson Correlation')
        ax.set_title('Aleatoric-Error Correlation Across Datasets')
        ax.set_xticks(x + width)
        ax.set_xticklabels(dataset_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Improvement summary
        ax = axes[1]
        improvements = []
        for ds in dataset_names:
            baseline = all_results[ds]['results']['Baseline']['metrics']['pearson_corr']
            enhanced = all_results[ds]['results']['Enhanced']['metrics']['pearson_corr']
            improvements.append((enhanced - baseline) / baseline * 100)

        bars = ax.bar(dataset_names, improvements, color='green', alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Enhanced vs Baseline Improvement')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:.1f}%', ha='center', va='bottom' if height > 0 else 'top')

        plt.suptitle('Step 1: Aleatoric Uncertainty - Summary Across Datasets', fontsize=14)
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'all_datasets_summary.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved summary plot: all_datasets_summary.png")

    print("\n" + "="*80)
    print("Step 1 Experiment Complete!")
    print("="*80)


if __name__ == '__main__':
    main()