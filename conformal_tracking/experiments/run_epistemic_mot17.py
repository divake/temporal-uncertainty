"""
Run Epistemic Uncertainty Experiment on MOT17

This script runs the complete epistemic uncertainty pipeline:
1. Load MOT17 data (reuse data loader)
2. Fit Mahalanobis model for aleatoric (reuse)
3. Fit combined epistemic model (spectral + repulsive)
4. Evaluate orthogonality and correlation
5. Generate comprehensive visualizations

Author: Enhanced CACD Team
Date: 2025-11-10
"""

import numpy as np
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import json
from datetime import datetime
import sys

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / 'src' / 'uncertainty'))
sys.path.append(str(PROJECT_ROOT / 'src'))
sys.path.append(str(PROJECT_ROOT / 'data_loaders'))

# Import aleatoric components (reuse)
from mahalanobis import MahalanobisUncertainty
from mot17_loader import MOT17DataLoader

# Import epistemic components (new)
from epistemic_combined import EpistemicUncertainty


def load_config(config_name: str) -> dict:
    """Load YAML configuration file."""
    config_path = PROJECT_ROOT / 'config' / f'{config_name}.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def plot_uncertainty_comparison(aleatoric: np.ndarray,
                                epistemic: np.ndarray,
                                total: np.ndarray,
                                conformity_scores: np.ndarray,
                                save_dir: Path,
                                prefix: str = ""):
    """
    Compare aleatoric, epistemic, and total uncertainties
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Subsample for scatter plots
    n_plot = min(3000, len(conformity_scores))
    idx_plot = np.random.choice(len(conformity_scores), n_plot, replace=False)

    # Plot 1: Aleatoric vs Conformity
    axes[0, 0].scatter(aleatoric[idx_plot], conformity_scores[idx_plot],
                      alpha=0.3, s=15, c='blue')
    axes[0, 0].set_xlabel('Aleatoric Uncertainty', fontsize=12)
    axes[0, 0].set_ylabel('Conformity Score (1 - IoU)', fontsize=12)
    axes[0, 0].set_title('Aleatoric vs Conformity', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    corr_a, pval_a = pearsonr(aleatoric, conformity_scores)
    axes[0, 0].text(0.05, 0.95, f'Pearson: {corr_a:.4f}\np-value: {pval_a:.2e}',
                   transform=axes[0, 0].transAxes, fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Plot 2: Epistemic vs Conformity
    axes[0, 1].scatter(epistemic[idx_plot], conformity_scores[idx_plot],
                      alpha=0.3, s=15, c='red')
    axes[0, 1].set_xlabel('Epistemic Uncertainty', fontsize=12)
    axes[0, 1].set_ylabel('Conformity Score (1 - IoU)', fontsize=12)
    axes[0, 1].set_title('Epistemic vs Conformity', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    corr_e, pval_e = pearsonr(epistemic, conformity_scores)
    axes[0, 1].text(0.05, 0.95, f'Pearson: {corr_e:.4f}\np-value: {pval_e:.2e}',
                   transform=axes[0, 1].transAxes, fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    # Plot 3: Total vs Conformity
    axes[0, 2].scatter(total[idx_plot], conformity_scores[idx_plot],
                      alpha=0.3, s=15, c='purple')
    axes[0, 2].set_xlabel('Total Uncertainty', fontsize=12)
    axes[0, 2].set_ylabel('Conformity Score (1 - IoU)', fontsize=12)
    axes[0, 2].set_title('Total vs Conformity', fontsize=13, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)

    corr_t, pval_t = pearsonr(total, conformity_scores)
    axes[0, 2].text(0.05, 0.95, f'Pearson: {corr_t:.4f}\np-value: {pval_t:.2e}',
                   transform=axes[0, 2].transAxes, fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))

    # Plot 4: Orthogonality check - Aleatoric vs Epistemic
    axes[1, 0].scatter(aleatoric[idx_plot], epistemic[idx_plot],
                      alpha=0.3, s=15, c=conformity_scores[idx_plot], cmap='RdYlBu_r')
    axes[1, 0].set_xlabel('Aleatoric Uncertainty', fontsize=12)
    axes[1, 0].set_ylabel('Epistemic Uncertainty', fontsize=12)
    axes[1, 0].set_title('Orthogonality: Aleatoric vs Epistemic', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    corr_ae = np.corrcoef(aleatoric, epistemic)[0, 1]
    color_box = 'green' if abs(corr_ae) < 0.3 else 'red'
    axes[1, 0].text(0.05, 0.95, f'Correlation: {corr_ae:.4f}',
                   transform=axes[1, 0].transAxes, fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=color_box, alpha=0.5))

    # Plot 5: Uncertainty distributions
    axes[1, 1].hist(aleatoric, bins=30, alpha=0.5, label='Aleatoric',
                   color='blue', edgecolor='black')
    axes[1, 1].hist(epistemic, bins=30, alpha=0.5, label='Epistemic',
                   color='red', edgecolor='black')
    axes[1, 1].hist(total, bins=30, alpha=0.5, label='Total',
                   color='purple', edgecolor='black')
    axes[1, 1].set_xlabel('Uncertainty', fontsize=12)
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].set_title('Uncertainty Distributions', fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Fraction analysis
    fractions = epistemic / (total + 1e-10)
    axes[1, 2].hist(fractions, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 2].set_xlabel('Epistemic Fraction (Epistemic/Total)', fontsize=12)
    axes[1, 2].set_ylabel('Count', fontsize=12)
    axes[1, 2].set_title('Epistemic Contribution to Total', fontsize=13, fontweight='bold')
    axes[1, 2].axvline(fractions.mean(), color='red', linestyle='--',
                      label=f'Mean: {fractions.mean():.3f}')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle('Uncertainty Decomposition Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()

    save_path = save_dir / f"{prefix}uncertainty_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")

    plt.close()


def plot_uncertainty_by_iou_categories(aleatoric: np.ndarray,
                                       epistemic: np.ndarray,
                                       total: np.ndarray,
                                       ious: np.ndarray,
                                       save_dir: Path,
                                       prefix: str = ""):
    """
    Analyze uncertainty decomposition by IoU quality categories
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Define IoU categories
    excellent_mask = ious >= 0.7
    good_mask = (ious >= 0.5) & (ious < 0.7)
    poor_mask = ious < 0.5

    categories = ['Excellent\n(IoU≥0.7)', 'Good\n(0.5≤IoU<0.7)', 'Poor\n(IoU<0.5)']

    # Plot 1: Stacked bar chart of mean uncertainties
    aleatoric_means = [
        aleatoric[excellent_mask].mean(),
        aleatoric[good_mask].mean(),
        aleatoric[poor_mask].mean()
    ]
    epistemic_means = [
        epistemic[excellent_mask].mean(),
        epistemic[good_mask].mean(),
        epistemic[poor_mask].mean()
    ]

    x_pos = np.arange(len(categories))
    width = 0.6

    axes[0, 0].bar(x_pos, aleatoric_means, width, label='Aleatoric',
                  color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].bar(x_pos, epistemic_means, width, bottom=aleatoric_means,
                  label='Epistemic', color='red', alpha=0.7, edgecolor='black')

    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(categories)
    axes[0, 0].set_ylabel('Mean Uncertainty', fontsize=12)
    axes[0, 0].set_title('Uncertainty Decomposition by IoU Quality', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Add total values on top
    for i, (a, e) in enumerate(zip(aleatoric_means, epistemic_means)):
        axes[0, 0].text(i, a + e + 0.01, f'{a+e:.3f}',
                       ha='center', fontweight='bold')

    # Plot 2: Epistemic fraction by category
    epistemic_fractions = [
        epistemic[excellent_mask].mean() / (total[excellent_mask].mean() + 1e-10),
        epistemic[good_mask].mean() / (total[good_mask].mean() + 1e-10),
        epistemic[poor_mask].mean() / (total[poor_mask].mean() + 1e-10)
    ]

    colors = ['green', 'orange', 'red']
    axes[0, 1].bar(x_pos, epistemic_fractions, width,
                  color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(categories)
    axes[0, 1].set_ylabel('Epistemic Fraction', fontsize=12)
    axes[0, 1].set_title('Epistemic Contribution by IoU Quality', fontsize=13, fontweight='bold')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for i, frac in enumerate(epistemic_fractions):
        axes[0, 1].text(i, frac + 0.01, f'{frac:.2%}',
                       ha='center', fontweight='bold')

    # Plot 3: Box plots for aleatoric
    aleatoric_data = [
        aleatoric[excellent_mask],
        aleatoric[good_mask],
        aleatoric[poor_mask]
    ]

    bp1 = axes[1, 0].boxplot(aleatoric_data, positions=x_pos, widths=0.6,
                             patch_artist=True, showmeans=True,
                             meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

    for patch, color in zip(bp1['boxes'], ['green', 'orange', 'red']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(categories)
    axes[1, 0].set_ylabel('Aleatoric Uncertainty', fontsize=12)
    axes[1, 0].set_title('Aleatoric Distribution by IoU Quality', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: Box plots for epistemic
    epistemic_data = [
        epistemic[excellent_mask],
        epistemic[good_mask],
        epistemic[poor_mask]
    ]

    bp2 = axes[1, 1].boxplot(epistemic_data, positions=x_pos, widths=0.6,
                             patch_artist=True, showmeans=True,
                             meanprops=dict(marker='D', markerfacecolor='blue', markersize=6))

    for patch, color in zip(bp2['boxes'], ['green', 'orange', 'red']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(categories)
    axes[1, 1].set_ylabel('Epistemic Uncertainty', fontsize=12)
    axes[1, 1].set_title('Epistemic Distribution by IoU Quality', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Uncertainty Analysis by IoU Categories', fontsize=15, fontweight='bold')
    plt.tight_layout()

    save_path = save_dir / f"{prefix}uncertainty_by_iou.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")

    plt.close()


def evaluate_and_save_results(aleatoric: np.ndarray,
                              epistemic: np.ndarray,
                              epistemic_components: dict,
                              total: np.ndarray,
                              conformity_scores: np.ndarray,
                              ious: np.ndarray,
                              confidences: np.ndarray,
                              save_dir: Path,
                              sequence_name: str):
    """
    Evaluate metrics and save comprehensive results
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Compute correlations
    pearson_a, pval_a = pearsonr(aleatoric, conformity_scores)
    pearson_e, pval_e = pearsonr(epistemic, conformity_scores)
    pearson_t, pval_t = pearsonr(total, conformity_scores)

    spearman_a, pval_sa = spearmanr(aleatoric, conformity_scores)
    spearman_e, pval_se = spearmanr(epistemic, conformity_scores)
    spearman_t, pval_st = spearmanr(total, conformity_scores)

    # Orthogonality
    corr_ae = np.corrcoef(aleatoric, epistemic)[0, 1]

    # IoU categories
    excellent_mask = ious >= 0.7
    good_mask = (ious >= 0.5) & (ious < 0.7)
    poor_mask = ious < 0.5

    results = {
        'experiment': {
            'name': f'epistemic_{sequence_name.lower()}',
            'timestamp': datetime.now().isoformat(),
            'sequence': sequence_name
        },
        'data': {
            'n_samples': len(conformity_scores),
            'iou_distribution': {
                'excellent': int(np.sum(excellent_mask)),
                'good': int(np.sum(good_mask)),
                'poor': int(np.sum(poor_mask))
            }
        },
        'correlations': {
            'aleatoric': {
                'pearson': {'r': float(pearson_a), 'p_value': float(pval_a)},
                'spearman': {'rho': float(spearman_a), 'p_value': float(pval_sa)}
            },
            'epistemic': {
                'pearson': {'r': float(pearson_e), 'p_value': float(pval_e)},
                'spearman': {'rho': float(spearman_e), 'p_value': float(pval_se)}
            },
            'total': {
                'pearson': {'r': float(pearson_t), 'p_value': float(pval_t)},
                'spearman': {'rho': float(spearman_t), 'p_value': float(pval_st)}
            },
            'orthogonality': {
                'aleatoric_epistemic_corr': float(corr_ae)
            }
        },
        'uncertainty_by_iou': {
            'excellent': {
                'aleatoric': {'mean': float(aleatoric[excellent_mask].mean()),
                             'std': float(aleatoric[excellent_mask].std())},
                'epistemic': {'mean': float(epistemic[excellent_mask].mean()),
                             'std': float(epistemic[excellent_mask].std())},
                'total': {'mean': float(total[excellent_mask].mean()),
                          'std': float(total[excellent_mask].std())}
            },
            'good': {
                'aleatoric': {'mean': float(aleatoric[good_mask].mean()),
                             'std': float(aleatoric[good_mask].std())},
                'epistemic': {'mean': float(epistemic[good_mask].mean()),
                             'std': float(epistemic[good_mask].std())},
                'total': {'mean': float(total[good_mask].mean()),
                          'std': float(total[good_mask].std())}
            },
            'poor': {
                'aleatoric': {'mean': float(aleatoric[poor_mask].mean() if np.sum(poor_mask) > 0 else 0),
                             'std': float(aleatoric[poor_mask].std() if np.sum(poor_mask) > 0 else 0)},
                'epistemic': {'mean': float(epistemic[poor_mask].mean() if np.sum(poor_mask) > 0 else 0),
                             'std': float(epistemic[poor_mask].std() if np.sum(poor_mask) > 0 else 0)},
                'total': {'mean': float(total[poor_mask].mean() if np.sum(poor_mask) > 0 else 0),
                          'std': float(total[poor_mask].std() if np.sum(poor_mask) > 0 else 0)}
            }
        },
        'epistemic_components': {
            'spectral': {
                'mean': float(epistemic_components['spectral'].mean()),
                'std': float(epistemic_components['spectral'].std())
            },
            'repulsive': {
                'mean': float(epistemic_components['repulsive'].mean()),
                'std': float(epistemic_components['repulsive'].std())
            }
        },
        'epistemic_fraction': {
            'mean': float((epistemic / (total + 1e-10)).mean()),
            'std': float((epistemic / (total + 1e-10)).std()),
            'min': float((epistemic / (total + 1e-10)).min()),
            'max': float((epistemic / (total + 1e-10)).max())
        }
    }

    # Save to JSON
    save_path = save_dir / 'results.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved: {save_path}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS - {sequence_name}")
    print(f"{'='*80}")

    print(f"\nCorrelations with Conformity Score:")
    print(f"  Aleatoric:  r = {pearson_a:7.4f} (p = {pval_a:.2e})")
    print(f"  Epistemic:  r = {pearson_e:7.4f} (p = {pval_e:.2e})")
    print(f"  Total:      r = {pearson_t:7.4f} (p = {pval_t:.2e})")

    print(f"\nOrthogonality:")
    print(f"  Correlation(Aleatoric, Epistemic) = {corr_ae:.4f}")
    if abs(corr_ae) < 0.2:
        print("  ✅ EXCELLENT: Very low correlation (< 0.2)")
    elif abs(corr_ae) < 0.3:
        print("  ✅ GOOD: Low correlation (< 0.3)")
    else:
        print("  ⚠️ WARNING: Moderate correlation (>= 0.3)")

    print(f"\nEpistemic Fraction:")
    print(f"  Mean: {(epistemic / (total + 1e-10)).mean():.2%}")
    print(f"  Std:  {(epistemic / (total + 1e-10)).std():.2%}")

    print(f"\nUncertainty by IoU Quality:")
    print(f"  {'Category':<15} {'Aleatoric':<15} {'Epistemic':<15} {'Total':<15}")
    print(f"  {'-'*60}")
    print(f"  {'Excellent':<15} {aleatoric[excellent_mask].mean():.3f} ± {aleatoric[excellent_mask].std():.3f}   "
          f"{epistemic[excellent_mask].mean():.3f} ± {epistemic[excellent_mask].std():.3f}   "
          f"{total[excellent_mask].mean():.3f} ± {total[excellent_mask].std():.3f}")
    print(f"  {'Good':<15} {aleatoric[good_mask].mean():.3f} ± {aleatoric[good_mask].std():.3f}   "
          f"{epistemic[good_mask].mean():.3f} ± {epistemic[good_mask].std():.3f}   "
          f"{total[good_mask].mean():.3f} ± {total[good_mask].std():.3f}")
    if np.sum(poor_mask) > 0:
        print(f"  {'Poor':<15} {aleatoric[poor_mask].mean():.3f} ± {aleatoric[poor_mask].std():.3f}   "
              f"{epistemic[poor_mask].mean():.3f} ± {epistemic[poor_mask].std():.3f}   "
              f"{total[poor_mask].mean():.3f} ± {total[poor_mask].std():.3f}")

    print(f"\n{'='*80}\n")

    return results


def main():
    """Main experiment pipeline."""
    print("\n" + "="*80)
    print("EPISTEMIC UNCERTAINTY EXPERIMENT - MOT17")
    print("="*80)

    # Load configurations
    print("\n[1/7] Loading configurations...")
    dataset_cfg = load_config('datasets')
    model_cfg = load_config('models')
    exp_cfg = load_config('experiment')

    # Setup paths
    cache_dir = Path(dataset_cfg['mot17']['cache_dir'])

    # Allow sequence override via command line
    import sys
    if len(sys.argv) > 1:
        sequence_name = sys.argv[1]
        print(f"  Using sequence from command line: {sequence_name}")
    else:
        sequence_name = dataset_cfg['mot17']['selected_sequence']

    cache_path = cache_dir / f"{sequence_name}.npz"

    # Extract sequence number for output directory
    seq_num = sequence_name.split('-')[1]  # e.g., "11" from "MOT17-11-FRCNN"

    results_dir = PROJECT_ROOT / 'results' / f'epistemic_mot17_{seq_num}'
    plots_dir = results_dir / 'plots'
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Results will be saved to: {results_dir}")
    print(f"  Plots will be saved to: {plots_dir}")

    # Load data (reuse data loader)
    print("\n[2/7] Loading and preparing data...")
    loader = MOT17DataLoader(
        cache_path=cache_path,
        layer_id=dataset_cfg['mot17']['filters']['layer_id'],
        conf_threshold=dataset_cfg['mot17']['filters']['confidence_threshold'],
        split_ratio=dataset_cfg['mot17']['split']['ratio'],
        random_seed=dataset_cfg['mot17']['split']['random_seed']
    )

    # Plot data distributions
    print("\n  Plotting data distributions...")
    loader.plot_data_distributions(save_dir=plots_dir, prefix="01_")

    # Get data
    cal_data = loader.get_calibration_data()
    test_data = loader.get_test_data()

    # Fit Mahalanobis model for aleatoric (reuse)
    print("\n[3/7] Fitting Mahalanobis model for aleatoric...")
    mahalanobis_model = MahalanobisUncertainty(
        reg_lambda=model_cfg['mahalanobis']['reg_lambda'],
        eps=model_cfg['mahalanobis']['eps']
    )
    mahalanobis_model.fit(cal_data['features'], verbose=True)

    # Compute aleatoric uncertainty
    print("\n[4/7] Computing aleatoric uncertainty...")
    aleatoric_cal_results = mahalanobis_model.predict(cal_data['features'], verbose=False)
    aleatoric_test_results = mahalanobis_model.predict(test_data['features'], verbose=False)

    aleatoric_cal = aleatoric_cal_results['normalized']
    aleatoric_test = aleatoric_test_results['normalized']

    # Fit epistemic model
    print("\n[5/7] Fitting epistemic uncertainty model...")
    epistemic_model = EpistemicUncertainty(
        k_neighbors_spectral=model_cfg['epistemic']['spectral']['k_neighbors'],
        k_neighbors_repulsive=model_cfg['epistemic']['repulsive']['k_neighbors'],
        temperature=model_cfg['epistemic']['repulsive']['temperature'],
        weights=model_cfg['epistemic']['weights']['mode'],
        verbose=True
    )

    epistemic_model.fit(
        cal_data['features'],
        mahalanobis_model=mahalanobis_model,
        aleatoric_cal=aleatoric_cal,
        conformity_cal=cal_data['conformity_scores'],
        plot_diagnostics=True,
        save_dir=plots_dir / 'calibration'
    )

    # Predict epistemic uncertainty
    print("\n[6/7] Predicting epistemic uncertainty on test set...")
    epistemic_results = epistemic_model.predict(
        test_data['features'],
        return_components=True,
        plot_diagnostics=True,
        save_dir=plots_dir / 'test'
    )

    epistemic_test = epistemic_results['combined']

    # Compute total uncertainty
    total_test = aleatoric_test + epistemic_test

    # Evaluation and visualization
    print("\n[7/7] Evaluating and visualizing results...")

    print("\n  Plotting uncertainty comparisons...")
    plot_uncertainty_comparison(
        aleatoric=aleatoric_test,
        epistemic=epistemic_test,
        total=total_test,
        conformity_scores=test_data['conformity_scores'],
        save_dir=plots_dir,
        prefix="02_"
    )

    print("\n  Plotting uncertainty by IoU categories...")
    plot_uncertainty_by_iou_categories(
        aleatoric=aleatoric_test,
        epistemic=epistemic_test,
        total=total_test,
        ious=test_data['ious'],
        save_dir=plots_dir,
        prefix="03_"
    )

    # Save results
    print("\n  Saving evaluation results...")
    results = evaluate_and_save_results(
        aleatoric=aleatoric_test,
        epistemic=epistemic_test,
        epistemic_components=epistemic_results,
        total=total_test,
        conformity_scores=test_data['conformity_scores'],
        ious=test_data['ious'],
        confidences=test_data['confidences'],
        save_dir=results_dir,
        sequence_name=sequence_name.replace('-FRCNN', '')
    )

    # Save model parameters
    epistemic_model.save_model(results_dir / 'epistemic_model.json')

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE ✓")
    print("="*80)
    print(f"\nResults saved to: {results_dir}")
    print(f"Plots saved to: {plots_dir}")
    print(f"\nKey Results:")
    print(f"  Aleatoric correlation:  r = {results['correlations']['aleatoric']['pearson']['r']:.4f}")
    print(f"  Epistemic correlation:  r = {results['correlations']['epistemic']['pearson']['r']:.4f}")
    print(f"  Total correlation:      r = {results['correlations']['total']['pearson']['r']:.4f}")
    print(f"  Orthogonality:          |r| = {abs(results['correlations']['orthogonality']['aleatoric_epistemic_corr']):.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()