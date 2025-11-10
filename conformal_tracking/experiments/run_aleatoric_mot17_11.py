"""
Run Aleatoric Uncertainty Experiment on MOT17-11

This script runs the complete pipeline:
1. Load MOT17-11 data
2. Fit Mahalanobis uncertainty model
3. Predict uncertainty on test set
4. Evaluate correlation with conformity scores
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
sys.path.append(str(PROJECT_ROOT / 'data_loaders'))

from mahalanobis import MahalanobisUncertainty
from mot17_loader import MOT17DataLoader


def load_config(config_name: str) -> dict:
    """Load YAML configuration file."""
    config_path = PROJECT_ROOT / 'config' / f'{config_name}.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def plot_uncertainty_vs_conformity(uncertainty_raw: np.ndarray,
                                   uncertainty_norm: np.ndarray,
                                   conformity_scores: np.ndarray,
                                   save_dir: Path,
                                   prefix: str = ""):
    """
    Plot uncertainty vs conformity scores with correlation analysis.

    Args:
        uncertainty_raw: Raw Mahalanobis distances
        uncertainty_norm: Normalized uncertainty [0, 1]
        conformity_scores: Ground truth conformity scores (1 - IoU)
        save_dir: Directory to save plots
        prefix: Prefix for saved filenames
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Subsample for scatter plots (max 3000 points)
    n_plot = min(3000, len(conformity_scores))
    idx_plot = np.random.choice(len(conformity_scores), n_plot, replace=False)

    # Plot 1: Raw uncertainty vs conformity
    axes[0, 0].scatter(uncertainty_raw[idx_plot], conformity_scores[idx_plot],
                      alpha=0.3, s=15, c='blue')
    axes[0, 0].set_xlabel('Raw Mahalanobis Distance', fontsize=12)
    axes[0, 0].set_ylabel('Conformity Score (1 - IoU)', fontsize=12)
    axes[0, 0].set_title('Raw Uncertainty vs Conformity Score', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Add correlation
    corr_raw, pval_raw = pearsonr(uncertainty_raw, conformity_scores)
    axes[0, 0].text(0.05, 0.95, f'Pearson: {corr_raw:.4f}\np-value: {pval_raw:.2e}',
                   transform=axes[0, 0].transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Plot 2: Normalized uncertainty vs conformity
    axes[0, 1].scatter(uncertainty_norm[idx_plot], conformity_scores[idx_plot],
                      alpha=0.3, s=15, c='green')
    axes[0, 1].set_xlabel('Normalized Uncertainty [0, 1]', fontsize=12)
    axes[0, 1].set_ylabel('Conformity Score (1 - IoU)', fontsize=12)
    axes[0, 1].set_title('Normalized Uncertainty vs Conformity Score', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Add correlation
    corr_norm, pval_norm = pearsonr(uncertainty_norm, conformity_scores)
    axes[0, 1].text(0.05, 0.95, f'Pearson: {corr_norm:.4f}\np-value: {pval_norm:.2e}',
                   transform=axes[0, 1].transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Plot 3: Binned analysis - Raw uncertainty
    n_bins = 10
    bins = np.linspace(np.percentile(uncertainty_raw, 5), np.percentile(uncertainty_raw, 95), n_bins + 1)
    bin_indices = np.digitize(uncertainty_raw, bins)

    bin_centers = []
    mean_conformity = []
    std_conformity = []

    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_centers.append((bins[i-1] + bins[i]) / 2)
            mean_conformity.append(np.mean(conformity_scores[mask]))
            std_conformity.append(np.std(conformity_scores[mask]))

    axes[1, 0].errorbar(bin_centers, mean_conformity, yerr=std_conformity,
                       marker='o', markersize=8, linewidth=2, capsize=5,
                       color='blue', ecolor='lightblue')
    axes[1, 0].set_xlabel('Raw Uncertainty (binned)', fontsize=12)
    axes[1, 0].set_ylabel('Mean Conformity Score', fontsize=12)
    axes[1, 0].set_title('Binned Analysis: Raw Uncertainty', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Binned analysis - Normalized uncertainty
    bins_norm = np.linspace(0, 1, n_bins + 1)
    bin_indices_norm = np.digitize(uncertainty_norm, bins_norm)

    bin_centers_norm = []
    mean_conformity_norm = []
    std_conformity_norm = []

    for i in range(1, n_bins + 1):
        mask = bin_indices_norm == i
        if np.sum(mask) > 0:
            bin_centers_norm.append((bins_norm[i-1] + bins_norm[i]) / 2)
            mean_conformity_norm.append(np.mean(conformity_scores[mask]))
            std_conformity_norm.append(np.std(conformity_scores[mask]))

    axes[1, 1].errorbar(bin_centers_norm, mean_conformity_norm, yerr=std_conformity_norm,
                       marker='o', markersize=8, linewidth=2, capsize=5,
                       color='green', ecolor='lightgreen')
    axes[1, 1].set_xlabel('Normalized Uncertainty (binned)', fontsize=12)
    axes[1, 1].set_ylabel('Mean Conformity Score', fontsize=12)
    axes[1, 1].set_title('Binned Analysis: Normalized Uncertainty', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(-0.05, 1.05)

    plt.tight_layout()

    save_path = save_dir / f"{prefix}uncertainty_vs_conformity.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")

    plt.close()


def plot_uncertainty_by_categories(uncertainty_norm: np.ndarray,
                                   conformity_scores: np.ndarray,
                                   ious: np.ndarray,
                                   save_dir: Path,
                                   prefix: str = ""):
    """
    Plot uncertainty distributions by IoU quality categories.

    Args:
        uncertainty_norm: Normalized uncertainty [0, 1]
        conformity_scores: Ground truth conformity scores
        ious: IoU values
        save_dir: Directory to save plots
        prefix: Prefix for saved filenames
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Define IoU categories
    excellent_mask = ious >= 0.7
    good_mask = (ious >= 0.5) & (ious < 0.7)
    poor_mask = ious < 0.5

    # Plot 1: Uncertainty distribution by IoU category
    axes[0, 0].hist(uncertainty_norm[excellent_mask], bins=30, alpha=0.6,
                   label=f'Excellent (IoU≥0.7, n={np.sum(excellent_mask)})',
                   color='green', edgecolor='black')
    axes[0, 0].hist(uncertainty_norm[good_mask], bins=30, alpha=0.6,
                   label=f'Good (0.5≤IoU<0.7, n={np.sum(good_mask)})',
                   color='orange', edgecolor='black')
    axes[0, 0].hist(uncertainty_norm[poor_mask], bins=30, alpha=0.6,
                   label=f'Poor (IoU<0.5, n={np.sum(poor_mask)})',
                   color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Normalized Uncertainty', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].set_title('Uncertainty Distribution by IoU Quality', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Box plots by IoU category
    data_to_plot = [
        uncertainty_norm[excellent_mask],
        uncertainty_norm[good_mask],
        uncertainty_norm[poor_mask]
    ]
    positions = [1, 2, 3]
    labels = ['Excellent\n(IoU≥0.7)', 'Good\n(0.5≤IoU<0.7)', 'Poor\n(IoU<0.5)']
    colors = ['green', 'orange', 'red']

    bp = axes[0, 1].boxplot(data_to_plot, positions=positions, widths=0.6,
                            patch_artist=True, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    axes[0, 1].set_xticks(positions)
    axes[0, 1].set_xticklabels(labels)
    axes[0, 1].set_ylabel('Normalized Uncertainty', fontsize=12)
    axes[0, 1].set_title('Uncertainty by IoU Quality (Box Plot)', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Mean uncertainty by IoU category
    means = [
        np.mean(uncertainty_norm[excellent_mask]),
        np.mean(uncertainty_norm[good_mask]),
        np.mean(uncertainty_norm[poor_mask])
    ]
    stds = [
        np.std(uncertainty_norm[excellent_mask]),
        np.std(uncertainty_norm[good_mask]),
        np.std(uncertainty_norm[poor_mask])
    ]

    axes[1, 0].bar(positions, means, yerr=stds, color=colors, alpha=0.7,
                  capsize=10, edgecolor='black', linewidth=2)
    axes[1, 0].set_xticks(positions)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_ylabel('Mean Normalized Uncertainty', fontsize=12)
    axes[1, 0].set_title('Mean Uncertainty by IoU Quality', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for i, (pos, mean, std) in enumerate(zip(positions, means, stds)):
        axes[1, 0].text(pos, mean + std + 0.02, f'{mean:.3f}\n±{std:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 4: Uncertainty categories breakdown
    # Define uncertainty categories
    low_unc = uncertainty_norm < 0.3
    med_unc = (uncertainty_norm >= 0.3) & (uncertainty_norm < 0.7)
    high_unc = uncertainty_norm >= 0.7

    # Count for each combination
    excellent_low = np.sum(excellent_mask & low_unc)
    excellent_med = np.sum(excellent_mask & med_unc)
    excellent_high = np.sum(excellent_mask & high_unc)

    good_low = np.sum(good_mask & low_unc)
    good_med = np.sum(good_mask & med_unc)
    good_high = np.sum(good_mask & high_unc)

    poor_low = np.sum(poor_mask & low_unc)
    poor_med = np.sum(poor_mask & med_unc)
    poor_high = np.sum(poor_mask & high_unc)

    # Stacked bar chart
    width = 0.6
    x = np.arange(3)

    p1 = axes[1, 1].bar(x, [excellent_low, good_low, poor_low],
                       width, label='Low Unc (0-0.3)', color='lightgreen', edgecolor='black')
    p2 = axes[1, 1].bar(x, [excellent_med, good_med, poor_med],
                       width, bottom=[excellent_low, good_low, poor_low],
                       label='Med Unc (0.3-0.7)', color='yellow', edgecolor='black')
    p3 = axes[1, 1].bar(x, [excellent_high, good_high, poor_high],
                       width, bottom=[excellent_low + excellent_med,
                                     good_low + good_med,
                                     poor_low + poor_med],
                       label='High Unc (0.7-1.0)', color='red', edgecolor='black')

    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['Excellent\nIoU', 'Good\nIoU', 'Poor\nIoU'])
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].set_title('Uncertainty Categories by IoU Quality', fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    save_path = save_dir / f"{prefix}uncertainty_by_categories.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")

    plt.close()


def evaluate_and_save_results(uncertainty_raw: np.ndarray,
                              uncertainty_norm: np.ndarray,
                              conformity_scores: np.ndarray,
                              ious: np.ndarray,
                              confidences: np.ndarray,
                              save_dir: Path):
    """
    Evaluate metrics and save results to JSON.

    Args:
        uncertainty_raw: Raw Mahalanobis distances
        uncertainty_norm: Normalized uncertainty
        conformity_scores: Ground truth conformity scores
        ious: IoU values
        confidences: Confidence scores
        save_dir: Directory to save results
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Compute correlations
    pearson_raw, pval_pearson_raw = pearsonr(uncertainty_raw, conformity_scores)
    spearman_raw, pval_spearman_raw = spearmanr(uncertainty_raw, conformity_scores)

    pearson_norm, pval_pearson_norm = pearsonr(uncertainty_norm, conformity_scores)
    spearman_norm, pval_spearman_norm = spearmanr(uncertainty_norm, conformity_scores)

    # Compute statistics by IoU category
    excellent_mask = ious >= 0.7
    good_mask = (ious >= 0.5) & (ious < 0.7)
    poor_mask = ious < 0.5

    # Uncertainty categories
    low_unc = uncertainty_norm < 0.3
    med_unc = (uncertainty_norm >= 0.3) & (uncertainty_norm < 0.7)
    high_unc = uncertainty_norm >= 0.7

    results = {
        'experiment': {
            'name': 'aleatoric_mot17_11',
            'timestamp': datetime.now().isoformat(),
            'sequence': 'MOT17-11'
        },
        'data': {
            'n_samples': len(conformity_scores),
            'conformity_score': {
                'mean': float(np.mean(conformity_scores)),
                'std': float(np.std(conformity_scores)),
                'min': float(np.min(conformity_scores)),
                'max': float(np.max(conformity_scores))
            },
            'iou': {
                'mean': float(np.mean(ious)),
                'std': float(np.std(ious)),
                'min': float(np.min(ious)),
                'max': float(np.max(ious))
            }
        },
        'correlations': {
            'raw_uncertainty': {
                'pearson': {
                    'r': float(pearson_raw),
                    'p_value': float(pval_pearson_raw)
                },
                'spearman': {
                    'rho': float(spearman_raw),
                    'p_value': float(pval_spearman_raw)
                }
            },
            'normalized_uncertainty': {
                'pearson': {
                    'r': float(pearson_norm),
                    'p_value': float(pval_pearson_norm)
                },
                'spearman': {
                    'rho': float(spearman_norm),
                    'p_value': float(pval_spearman_norm)
                }
            }
        },
        'uncertainty_by_iou_quality': {
            'excellent_iou_0.7': {
                'count': int(np.sum(excellent_mask)),
                'mean_uncertainty': float(np.mean(uncertainty_norm[excellent_mask])),
                'std_uncertainty': float(np.std(uncertainty_norm[excellent_mask]))
            },
            'good_iou_0.5_0.7': {
                'count': int(np.sum(good_mask)),
                'mean_uncertainty': float(np.mean(uncertainty_norm[good_mask])),
                'std_uncertainty': float(np.std(uncertainty_norm[good_mask]))
            },
            'poor_iou_below_0.5': {
                'count': int(np.sum(poor_mask)),
                'mean_uncertainty': float(np.mean(uncertainty_norm[poor_mask])),
                'std_uncertainty': float(np.std(uncertainty_norm[poor_mask]))
            }
        },
        'uncertainty_categories': {
            'low_0_0.3': {
                'count': int(np.sum(low_unc)),
                'percentage': float(np.sum(low_unc) / len(uncertainty_norm) * 100),
                'mean_conformity': float(np.mean(conformity_scores[low_unc]))
            },
            'medium_0.3_0.7': {
                'count': int(np.sum(med_unc)),
                'percentage': float(np.sum(med_unc) / len(uncertainty_norm) * 100),
                'mean_conformity': float(np.mean(conformity_scores[med_unc]))
            },
            'high_0.7_1.0': {
                'count': int(np.sum(high_unc)),
                'percentage': float(np.sum(high_unc) / len(uncertainty_norm) * 100),
                'mean_conformity': float(np.mean(conformity_scores[high_unc]))
            }
        }
    }

    # Save to JSON
    save_path = save_dir / 'results.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved: {save_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\nCorrelations with Conformity Score (1 - IoU):")
    print(f"  Raw Uncertainty:")
    print(f"    Pearson:  r = {pearson_raw:7.4f}, p = {pval_pearson_raw:.2e}")
    print(f"    Spearman: ρ = {spearman_raw:7.4f}, p = {pval_spearman_raw:.2e}")
    print(f"\n  Normalized Uncertainty:")
    print(f"    Pearson:  r = {pearson_norm:7.4f}, p = {pval_pearson_norm:.2e}")
    print(f"    Spearman: ρ = {spearman_norm:7.4f}, p = {pval_spearman_norm:.2e}")

    print(f"\nUncertainty by IoU Quality:")
    print(f"  Excellent (IoU≥0.7): {np.mean(uncertainty_norm[excellent_mask]):.3f} ± {np.std(uncertainty_norm[excellent_mask]):.3f}")
    print(f"  Good (0.5≤IoU<0.7):  {np.mean(uncertainty_norm[good_mask]):.3f} ± {np.std(uncertainty_norm[good_mask]):.3f}")
    print(f"  Poor (IoU<0.5):      {np.mean(uncertainty_norm[poor_mask]):.3f} ± {np.std(uncertainty_norm[poor_mask]):.3f}")

    print(f"\nUncertainty Category Distribution:")
    print(f"  Low (0-0.3):     {np.sum(low_unc)/len(uncertainty_norm)*100:5.1f}%")
    print(f"  Medium (0.3-0.7): {np.sum(med_unc)/len(uncertainty_norm)*100:5.1f}%")
    print(f"  High (0.7-1.0):   {np.sum(high_unc)/len(uncertainty_norm)*100:5.1f}%")

    print(f"\n{'='*60}\n")

    return results


def main():
    """Main experiment pipeline."""
    print("\n" + "="*80)
    print("MAHALANOBIS ALEATORIC UNCERTAINTY - MOT17-11")
    print("="*80)

    # Load configurations
    print("\n[1/6] Loading configurations...")
    dataset_cfg = load_config('datasets')
    model_cfg = load_config('models')
    exp_cfg = load_config('experiment')

    # Setup paths
    cache_dir = Path(dataset_cfg['mot17']['cache_dir'])
    sequence_name = dataset_cfg['mot17']['selected_sequence']
    cache_path = cache_dir / f"{sequence_name}.npz"

    results_dir = PROJECT_ROOT / exp_cfg['experiment']['output']['results_dir']
    plots_dir = PROJECT_ROOT / exp_cfg['experiment']['output']['plots_dir']
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Results will be saved to: {results_dir}")
    print(f"  Plots will be saved to: {plots_dir}")

    # Load data
    print("\n[2/6] Loading and preparing data...")
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

    # Fit Mahalanobis model
    print("\n[3/6] Fitting Mahalanobis uncertainty model...")
    model = MahalanobisUncertainty(
        reg_lambda=model_cfg['mahalanobis']['reg_lambda'],
        eps=model_cfg['mahalanobis']['eps']
    )

    model.fit(cal_data['features'], verbose=True)

    # Plot diagnostic plots
    print("\n  Plotting Mahalanobis diagnostics...")
    model.plot_diagnostics(save_dir=plots_dir, prefix="02_")

    # Predict uncertainty
    print("\n[4/6] Predicting uncertainty on test set...")
    predictions = model.predict(test_data['features'], verbose=True)

    uncertainty_raw = predictions['raw']
    uncertainty_norm = predictions['normalized']

    # Evaluation and plotting
    print("\n[5/6] Evaluating and visualizing results...")

    print("\n  Plotting uncertainty vs conformity...")
    plot_uncertainty_vs_conformity(
        uncertainty_raw=uncertainty_raw,
        uncertainty_norm=uncertainty_norm,
        conformity_scores=test_data['conformity_scores'],
        save_dir=plots_dir,
        prefix="03_"
    )

    print("\n  Plotting uncertainty by categories...")
    plot_uncertainty_by_categories(
        uncertainty_norm=uncertainty_norm,
        conformity_scores=test_data['conformity_scores'],
        ious=test_data['ious'],
        save_dir=plots_dir,
        prefix="04_"
    )

    # Save results
    print("\n[6/6] Saving evaluation results...")
    results = evaluate_and_save_results(
        uncertainty_raw=uncertainty_raw,
        uncertainty_norm=uncertainty_norm,
        conformity_scores=test_data['conformity_scores'],
        ious=test_data['ious'],
        confidences=test_data['confidences'],
        save_dir=results_dir
    )

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE ✓")
    print("="*80)
    print(f"\nResults saved to: {results_dir}")
    print(f"Plots saved to: {plots_dir}")
    print(f"\nKey Result: Pearson r = {results['correlations']['raw_uncertainty']['pearson']['r']:.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
