"""
Step 2: Multi-source Epistemic Uncertainty Comparison (Fixed Version)
======================================================================

This script implements and compares multiple epistemic uncertainty sources,
then creates an optimized ensemble. Excluding Power Plant dataset due to
low dimensionality issues.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json
from typing import Dict, List, Tuple
from scipy import stats

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_utils import prepare_dataset
from src.aleatoric import EnhancedAleatoric
from src.epistemic import (
    InverseDensityEpistemic,
    MinDistanceEpistemic,
    EntropyEpistemic,
    MultiSourceEpistemic
)


def analyze_epistemic_methods(dataset_name: str) -> Dict:
    """
    Analyze different epistemic uncertainty methods for a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing epistemic methods for {dataset_name}")
    print(f"{'='*60}")

    # Load dataset
    dataset = prepare_dataset(dataset_name)

    # Get aleatoric uncertainty for orthogonality check
    aleatoric_model = EnhancedAleatoric(k_neighbors=10)
    aleatoric_model.fit(
        dataset['X_train_scaled'],
        dataset['y_train'],
        dataset['y_pred_train']
    )
    aleatoric_test = aleatoric_model.predict(dataset['X_test_scaled'])

    # Initialize epistemic models
    models = {
        'Density': InverseDensityEpistemic(),
        'Distance': MinDistanceEpistemic(),
        'Entropy': EntropyEpistemic(),
        'Multi-Source': MultiSourceEpistemic()
    }

    results = {
        'dataset': dataset_name,
        'aleatoric': aleatoric_test,
        'methods': {}
    }

    # Fit and evaluate each method
    for name, model in models.items():
        print(f"\nTesting {name} epistemic...")

        # Fit on training data
        model.fit(
            dataset['X_train_scaled'],
            dataset['y_train'],
            dataset['y_pred_train']
        )

        # Predict on test set
        epistemic_test = model.predict(dataset['X_test_scaled'])

        # Calculate metrics
        test_errors = np.abs(dataset['y_test'] - dataset['y_pred_test'])

        # OOD detection capability (correlation with absolute error)
        ood_corr, ood_p = stats.spearmanr(test_errors, epistemic_test)

        # High error detection (80th percentile)
        high_error_threshold = np.percentile(test_errors, 80)
        high_error_mask = test_errors > high_error_threshold
        if np.sum(high_error_mask) > 0:
            high_error_corr, _ = stats.pointbiserialr(high_error_mask, epistemic_test)
        else:
            high_error_corr = 0

        # Orthogonality with aleatoric
        ortho_corr, ortho_p = stats.spearmanr(aleatoric_test, epistemic_test)

        results['methods'][name] = {
            'predictions': epistemic_test,
            'ood_correlation': ood_corr,
            'ood_p_value': ood_p,
            'high_error_correlation': high_error_corr,
            'orthogonality': ortho_corr,
            'mean': np.mean(epistemic_test),
            'std': np.std(epistemic_test),
            'errors': test_errors
        }

        print(f"  OOD Correlation: {ood_corr:.3f}")
        print(f"  High Error Correlation: {high_error_corr:.3f}")
        print(f"  Orthogonality with aleatoric: {ortho_corr:.3f}")

        # For multi-source, also get the learned weights
        if name == 'Multi-Source' and hasattr(model, 'weights'):
            results['methods'][name]['weights'] = model.weights
            print(f"  Learned weights: {model.weights}")

    return results


def create_comprehensive_plots(all_results: Dict[str, Dict], output_dir: Path):
    """
    Create comprehensive visualization of epistemic uncertainty results.

    Args:
        all_results: Results for all datasets
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Color palette
    colors = {
        'Density': '#FF6B6B',
        'Distance': '#4ECDC4',
        'Entropy': '#45B7D1',
        'Multi-Source': '#FFA07A'
    }

    for dataset_name, results in all_results.items():
        print(f"\nCreating plots for {dataset_name}...")

        # Create comprehensive comparison plot
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'Epistemic Uncertainty Analysis - {dataset_name}', fontsize=16, fontweight='bold')

        # 1. Distribution comparison
        ax1 = plt.subplot(3, 4, 1)
        for method_name in ['Density', 'Distance', 'Entropy']:
            if method_name in results['methods']:
                epistemic = results['methods'][method_name]['predictions']
                ax1.hist(epistemic, bins=30, alpha=0.5, label=method_name, color=colors[method_name])
        ax1.set_xlabel('Epistemic Uncertainty')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Single-Source Epistemic')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Multi-source distribution
        ax2 = plt.subplot(3, 4, 2)
        if 'Multi-Source' in results['methods']:
            epistemic = results['methods']['Multi-Source']['predictions']
            ax2.hist(epistemic, bins=30, color=colors['Multi-Source'], alpha=0.7)
            ax2.set_xlabel('Epistemic Uncertainty')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Multi-Source Ensemble Epistemic')
            ax2.grid(True, alpha=0.3)

        # 3. Scatter plots vs error for each method
        for idx, method_name in enumerate(['Density', 'Distance', 'Entropy', 'Multi-Source']):
            ax = plt.subplot(3, 4, 3 + idx)
            if method_name in results['methods']:
                method_data = results['methods'][method_name]
                errors = method_data['errors']
                epistemic = method_data['predictions']

                # Color by high error
                high_error_threshold = np.percentile(errors, 80)
                high_error_mask = errors > high_error_threshold

                ax.scatter(errors[~high_error_mask], epistemic[~high_error_mask],
                          alpha=0.3, s=10, color=colors[method_name], label='Normal')
                ax.scatter(errors[high_error_mask], epistemic[high_error_mask],
                          alpha=0.7, s=20, color='red', label='High Error (>80%)')

                ax.set_xlabel('Absolute Error')
                ax.set_ylabel('Epistemic')
                ax.set_title(f'{method_name}\nOOD Corr: {method_data["ood_correlation"]:.3f}')
                ax.grid(True, alpha=0.3)
                if idx == 0:
                    ax.legend(fontsize=8)

        # 4. Learned weights (if multi-source)
        ax7 = plt.subplot(3, 4, 7)
        if 'Multi-Source' in results['methods'] and 'weights' in results['methods']['Multi-Source']:
            weights_dict = results['methods']['Multi-Source']['weights']
            # Convert dict to list if needed
            if isinstance(weights_dict, dict):
                weights = [weights_dict.get('density', 0),
                          weights_dict.get('distance', 0),
                          weights_dict.get('entropy', 0)]
            else:
                weights = weights_dict
            methods = ['Density', 'Distance', 'Entropy']
            bars = ax7.bar(methods, weights, color=[colors[m] for m in methods])
            ax7.set_ylabel('Weight')
            ax7.set_title('Learned Ensemble Weights')
            ax7.set_ylim([0, 1])
            for bar, w in zip(bars, weights):
                ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{w:.3f}', ha='center', va='bottom')
            ax7.grid(True, alpha=0.3, axis='y')

        # 5. Sources vs Combined
        ax8 = plt.subplot(3, 4, 8)
        if 'Multi-Source' in results['methods']:
            errors = results['methods']['Multi-Source']['errors']
            for method_name in ['Density', 'Distance', 'Entropy']:
                if method_name in results['methods']:
                    epistemic = results['methods'][method_name]['predictions']
                    # Normalize for comparison
                    epistemic_norm = (epistemic - np.min(epistemic)) / (np.max(epistemic) - np.min(epistemic) + 1e-8)
                    ax8.plot(errors, epistemic_norm, alpha=0.3, label=method_name, color=colors[method_name])

            # Add combined
            epistemic = results['methods']['Multi-Source']['predictions']
            epistemic_norm = (epistemic - np.min(epistemic)) / (np.max(epistemic) - np.min(epistemic) + 1e-8)
            ax8.plot(errors, epistemic_norm, alpha=0.8, label='Combined', color=colors['Multi-Source'], linewidth=2)

            ax8.set_xlabel('Absolute Error')
            ax8.set_ylabel('Normalized Epistemic')
            ax8.set_title('Sources vs Combined')
            ax8.legend(fontsize=8)
            ax8.grid(True, alpha=0.3)

        # 6. Correlation matrix between sources
        ax9 = plt.subplot(3, 4, 9)
        method_names = ['Density', 'Distance', 'Entropy']
        n_methods = len(method_names)
        corr_matrix = np.zeros((n_methods, n_methods))

        for i, m1 in enumerate(method_names):
            for j, m2 in enumerate(method_names):
                if m1 in results['methods'] and m2 in results['methods']:
                    corr, _ = stats.spearmanr(
                        results['methods'][m1]['predictions'],
                        results['methods'][m2]['predictions']
                    )
                    corr_matrix[i, j] = corr

        im = ax9.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax9.set_xticks(range(n_methods))
        ax9.set_yticks(range(n_methods))
        ax9.set_xticklabels(method_names, rotation=45)
        ax9.set_yticklabels(method_names)
        ax9.set_title('Source Correlations')

        # Add correlation values
        for i in range(n_methods):
            for j in range(n_methods):
                text = ax9.text(j, i, f'{corr_matrix[i, j]:.2f}',
                               ha="center", va="center", color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")

        plt.colorbar(im, ax=ax9)

        # 7. OOD Detection comparison
        ax10 = plt.subplot(3, 4, 10)
        methods = []
        ood_corrs = []
        for method_name in ['Density', 'Distance', 'Entropy', 'Multi-Source']:
            if method_name in results['methods']:
                methods.append(method_name)
                ood_corrs.append(results['methods'][method_name]['ood_correlation'])

        bars = ax10.bar(methods, ood_corrs, color=[colors[m] for m in methods])
        ax10.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Poor')
        ax10.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Good')
        ax10.set_ylabel('OOD Correlation')
        ax10.set_title('OOD Detection Capability')
        ax10.set_ylim([0, max(ood_corrs) * 1.2] if ood_corrs else [0, 0.5])
        ax10.legend(fontsize=8)
        ax10.grid(True, alpha=0.3, axis='y')

        for bar, corr in zip(bars, ood_corrs):
            ax10.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{corr:.3f}', ha='center', va='bottom')

        # 8. Summary statistics table
        ax11 = plt.subplot(3, 4, 11)
        ax11.axis('tight')
        ax11.axis('off')

        table_data = []
        table_data.append(['Method', 'OOD Corr', 'High Err Corr', 'Mean', 'Std'])

        for method_name in ['Density', 'Distance', 'Entropy', 'Multi-Source']:
            if method_name in results['methods']:
                m = results['methods'][method_name]
                table_data.append([
                    method_name,
                    f"{m['ood_correlation']:.3f}",
                    f"{m['high_error_correlation']:.3f}",
                    f"{m['mean']:.3f}",
                    f"{m['std']:.3f}"
                ])

        table = ax11.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Color header
        for i in range(5):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax11.set_title('Summary Statistics', y=0.8)

        # 9. Orthogonality check with aleatoric
        ax12 = plt.subplot(3, 4, 12)
        methods = []
        ortho_corrs = []
        for method_name in ['Density', 'Distance', 'Entropy', 'Multi-Source']:
            if method_name in results['methods']:
                methods.append(method_name)
                ortho_corrs.append(abs(results['methods'][method_name]['orthogonality']))

        bars = ax12.bar(methods, ortho_corrs, color=[colors[m] for m in methods])
        ax12.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax12.set_ylabel('|Correlation with Aleatoric|')
        ax12.set_title('Orthogonality Check (Lower is Better)')
        ax12.legend(fontsize=8)
        ax12.grid(True, alpha=0.3, axis='y')

        for bar, corr in zip(bars, ortho_corrs):
            ax12.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{corr:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset_name}_epistemic_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Create separate orthogonality scatter plots
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f'Aleatoric vs Epistemic Orthogonality - {dataset_name}', fontsize=14, fontweight='bold')

        for idx, method_name in enumerate(['Density', 'Distance', 'Entropy', 'Multi-Source']):
            ax = axes[idx]
            if method_name in results['methods']:
                aleatoric = results['aleatoric']
                epistemic = results['methods'][method_name]['predictions']
                errors = results['methods'][method_name]['errors']

                # Create scatter plot colored by error
                scatter = ax.scatter(aleatoric, epistemic, c=errors, cmap='viridis',
                                   alpha=0.5, s=10)
                ax.set_xlabel('Aleatoric')
                ax.set_ylabel('Epistemic')

                # Add correlation info
                corr = results['methods'][method_name]['orthogonality']
                p_val = stats.spearmanr(aleatoric, epistemic)[1]
                ax.set_title(f'{method_name}\nCorr: {corr:.3f} (p={p_val:.2e})')

                # Add diagonal line
                ax.plot([0, max(aleatoric)], [0, max(aleatoric)], 'k--', alpha=0.3)

                # Add colorbar for the last subplot
                if idx == 3:
                    plt.colorbar(scatter, ax=ax, label='Error')

                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset_name}_orthogonality_detail.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Create summary plot across datasets
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Step 2: Epistemic Uncertainty - Summary Across Datasets', fontsize=14, fontweight='bold')

    # OOD Detection capability
    ax1 = axes[0]
    dataset_names = list(all_results.keys())
    x = np.arange(len(dataset_names))
    width = 0.2

    for i, method_name in enumerate(['Density', 'Distance', 'Entropy', 'Multi-Source']):
        ood_corrs = []
        for dataset_name in dataset_names:
            if method_name in all_results[dataset_name]['methods']:
                ood_corrs.append(all_results[dataset_name]['methods'][method_name]['ood_correlation'])
            else:
                ood_corrs.append(0)

        offset = (i - 1.5) * width
        bars = ax1.bar(x + offset, ood_corrs, width, label=method_name, color=colors[method_name])

    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('OOD Correlation')
    ax1.set_title('OOD Detection Capability Across Datasets')
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.replace('_', ' ').title() for d in dataset_names])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Orthogonality with aleatoric
    ax2 = axes[1]
    ortho_data = []
    labels = []

    for dataset_name in dataset_names:
        if 'Multi-Source' in all_results[dataset_name]['methods']:
            ortho = abs(all_results[dataset_name]['methods']['Multi-Source']['orthogonality'])
            ortho_data.append(ortho)
            labels.append(dataset_name.replace('_', ' ').title())

    bars = ax2.bar(labels, ortho_data, color=['red' if o > 0.3 else 'green' for o in ortho_data])
    ax2.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax2.set_ylabel('|Correlation(Aleatoric, Epistemic)|')
    ax2.set_title('Orthogonality Check (Lower is Better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, ortho_data):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'all_datasets_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nAll plots saved to {output_dir}")


def save_results(all_results: Dict[str, Dict], output_dir: Path):
    """
    Save numerical results to CSV and JSON.

    Args:
        all_results: Results for all datasets
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary CSV
    summary_data = []
    for dataset_name, results in all_results.items():
        for method_name, method_data in results['methods'].items():
            summary_data.append({
                'Dataset': dataset_name,
                'Method': method_name,
                'OOD_Correlation': method_data['ood_correlation'],
                'High_Error_Correlation': method_data['high_error_correlation'],
                'Orthogonality': method_data['orthogonality'],
                'Mean': method_data['mean'],
                'Std': method_data['std']
            })

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_dir / 'epistemic_summary.csv', index=False)

    # Save detailed JSON (without predictions to keep file size small)
    json_results = {}
    for dataset_name, results in all_results.items():
        json_results[dataset_name] = {
            'methods': {}
        }
        for method_name, method_data in results['methods'].items():
            json_results[dataset_name]['methods'][method_name] = {
                'ood_correlation': float(method_data['ood_correlation']),
                'high_error_correlation': float(method_data['high_error_correlation']),
                'orthogonality': float(method_data['orthogonality']),
                'mean': float(method_data['mean']),
                'std': float(method_data['std'])
            }
            if 'weights' in method_data:
                weights_data = method_data['weights']
                if isinstance(weights_data, dict):
                    json_results[dataset_name]['methods'][method_name]['weights'] = {
                        k: float(v) for k, v in weights_data.items()
                    }
                else:
                    json_results[dataset_name]['methods'][method_name]['weights'] = [
                        float(w) for w in weights_data
                    ]

    with open(output_dir / 'epistemic_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


def main():
    """Main experiment runner."""
    # Setup paths
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "results" / "step2_fixed"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Datasets to analyze (excluding power_plant)
    datasets = ['energy_heating', 'energy_cooling']

    # Run analysis for each dataset
    all_results = {}
    for dataset_name in datasets:
        results = analyze_epistemic_methods(dataset_name)
        all_results[dataset_name] = results

    # Create comprehensive plots
    plot_dir = results_dir / "plots"
    create_comprehensive_plots(all_results, plot_dir)

    # Save results
    save_results(all_results, results_dir)

    print("\n" + "="*60)
    print("Step 2 Analysis Complete!")
    print("="*60)

    # Print summary
    print("\nSummary of Results:")
    for dataset_name in datasets:
        print(f"\n{dataset_name.upper()}:")
        best_ood = 0
        best_method = None

        for method_name, method_data in all_results[dataset_name]['methods'].items():
            ood_corr = method_data['ood_correlation']
            if ood_corr > best_ood:
                best_ood = ood_corr
                best_method = method_name

        print(f"  Best OOD Detection: {best_method} ({best_ood:.3f})")

        if 'Multi-Source' in all_results[dataset_name]['methods']:
            multi_ood = all_results[dataset_name]['methods']['Multi-Source']['ood_correlation']
            multi_ortho = all_results[dataset_name]['methods']['Multi-Source']['orthogonality']
            print(f"  Multi-Source OOD: {multi_ood:.3f}")
            print(f"  Multi-Source Orthogonality: {multi_ortho:.3f}")

            if 'weights' in all_results[dataset_name]['methods']['Multi-Source']:
                weights_data = all_results[dataset_name]['methods']['Multi-Source']['weights']
                if isinstance(weights_data, dict):
                    print(f"  Learned Weights: Density={weights_data['density']:.3f}, Distance={weights_data['distance']:.3f}, Entropy={weights_data['entropy']:.3f}")
                else:
                    print(f"  Learned Weights: Density={weights_data[0]:.3f}, Distance={weights_data[1]:.3f}, Entropy={weights_data[2]:.3f}")


if __name__ == "__main__":
    main()